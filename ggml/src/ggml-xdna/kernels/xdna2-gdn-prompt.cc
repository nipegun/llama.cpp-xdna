// llama.cpp-xdna: the recurrence of a gated delta net (GATED_DELTA_NET of Qwen3.5) over a batch of tokens, by
// chunks of 16 tokens as products of small matrices, for the cores of the overlay of the prompts (AIE2P).
//
//   xdna2_gdn_state(pObject, pSums, pRowAt, pRow)      the states of a round: keeps the one of its row
//   xdna2_gdn_kq(pIn0, pSums, pWork0, pWork1)          a chunk of 16 tokens: its first object (k, q and Gamma)
//   xdna2_gdn_rest(pIn1, pSums, pWork1, pOut, pBlockJ)  and its second (v and the factors)
//   xdna2_gdn_state_out(pSums, pOut)                    the states at the end of a round
//
// A core takes the two value heads of a head of keys (s and s + 16 in Qwen3.5, which share k and q) and 8 value
// indices of each, j = 8 pBlockJ .. 8 pBlockJ + 7 (the core program gives the block). Its state, S^T restricted to
// those rows (S^T[j][d] = the state of ggml, whose row j is column j of S), stays in its sums for the whole round. Per
// head and chunk, with
// cs the cumulative sum of the gate g in the chunk, Gamma[t][t'] = e^(cs_t - cs_t') for t' <= t, E_t = e^cs_t and
// G_t = e^(cs_last - cs_t), all from the host:
//
//   M    = the part under the diagonal of Gamma (K K^T)          T = (I + M D_beta)^-1
//   R    = V - D_E (K S)              new = D_beta T R
//   o    = D_E (Q S) + (Gamma (Q K^T)) new
//   S   <- e^cs_last S + K^T (D_G new)
//
// which is build_delta_net_chunking of src/models/delta-net-base.cpp with W S written as T D_beta D_E (K S), and with
// beta moved out of the inverse: (I + M D_beta) X = R and new = D_beta X give (I + D_beta M) new = D_beta R, without
// dividing by beta. q comes already scaled by 1 / sqrt(128). The scalings by beta, E and G are products by diagonal
// matrices, so the matrix unit does them; only Gamma goes lane by lane. T goes by blocks of 8: every diagonal block D
// by doubling, (I - D)(I + D^2)(I + D^4), and X0 = T00 R0, X1 = T11 (R1 - A10 X0), with A = M D_beta.
//
// Everything works on groups of 8 x 8, and the products are A B^T (mac_8x8_8x8T): to multiply by B, B^T has to be at
// hand, which is why the rows of the state, (K S)^T and (Q S)^T keep j as their rows, and why the doubling carries
// every power of D with its transpose.
//
// Precision: every product of matrices takes both operands split into two parts of block floating point (bfp16), the
// high part and what is left of the 32-bit float, and adds hi hi + hi lo + lo hi: some 14 bits, three products of the
// matrix unit (the products of bf16 of AIE2P are 8 vector instructions each: 16 times slower than one of bfp16).
// Gamma by a group of floats goes as three exact products of its three parts of bf16, and the decay of the state as six
// (some 24 bits). Against the recurrence token by token in 64-bit floats this is about as close as that recurrence
// with the operands of its products in bf16 (_/xdna2-lab/gdn_chunk_check.py).
//
// The objects (bf16 is 2 bytes, f32 4, a group of 8 x 8 in bfp16 72: 8 blocks of 9 bytes, one per row):
//
//   pIn0   K [2 blocks of t][16 blocks of d][8 t][8 d] bf16, then Q alike, then at byte 8192 Gamma [2 heads][the
//          groups 00, 10, 11][8][8] bf16, zero above the diagonal
//   pIn1   V^T [2 heads][16 blocks of j][2 blocks of t][8 j][8 t] bf16, then at byte 8192 [2 heads][16] f32 each of
//          beta, E and G (at 8192, 8320 and 8448), and at 8576 e^cs_last [2 heads] f32
//   pOut   o [2 heads][8 chunks][16 t][8 j] f32: the outputs of 8 chunks, then the object goes
//   state  [2 heads][8 j][128 d] f32, in pObject (the band of the column holds one per row) and in pOut at the end
//
// and the memory of the core, which belongs to the multiplication of the prompts and is used here as it is:
//
//   pSums    S^T [2 heads][16 blocks of d][8 j][8 d] f32 (8 KB), then the work of the chunk, then the counter of the
//            chunks of the object of C
//   pWork0   k and q split, [16 blocks of d][2 blocks of t][hi, lo] (rows t, blocks along d)        9216 bytes
//   pWork1   k^T split [16][2][hi, lo] (rows d, blocks along t), then S^T split [16][2 heads][hi, lo]
//
// Built for the cores with Peano through mlir-aie, into the overlay of the prompts (_/xdna2-lab/
// bfpmm_resident.py --gdn 1), one object per function: -DXDNA2_GDN_STATE, -DXDNA2_GDN_KQ, -DXDNA2_GDN_REST or
// -DXDNA2_GDN_STATE_OUT (-DXDNA2_GDN_ALL gives the four, to read their sizes). A chunk comes in two objects of B, one
// call each: for a core to hold both at once mlir-aie would give the fifo a third buffer, and there is no room. The
// multiplier is the one of xdna2-mm-bfp16-s3.cc (aie_kernels/linalg/mm_bfp.cc of mlir-aie, Copyright (C) 2025 Advanced
// Micro Devices, Inc., Apache-2.0 WITH LLVM-exception).

#include <aie_api/aie.hpp>

#include <stdint.h>

namespace {

constexpr unsigned cChunk = 16;            // tokens of a chunk
constexpr unsigned cDim = 128;             // values of a head of k, q, v and of a row of the state
constexpr unsigned cRowsJ = 8;             // value indices of a head in this core
constexpr unsigned cHeads = 2;             // value heads of a core
constexpr unsigned cLanes = 64;            // values of a group of 8 x 8
constexpr unsigned cGroupBytes = 72;       // a group in bfp16
constexpr unsigned cPairBytes = 2 * cGroupBytes;
constexpr unsigned cBlocksT = cChunk / 8;  // blocks of 8 tokens
constexpr unsigned cBlocksD = cDim / 8;    // blocks of 8 values of a head
constexpr unsigned cChunksOut = 8;         // chunks an object of C holds

// pIn0 (k, q and Gamma) and pIn1 (v^T and the factors), in bytes
constexpr unsigned cAtQ = cChunk * cDim * 2;
constexpr unsigned cAtGamma = 2 * cAtQ;
constexpr unsigned cGammaHead = 3 * cLanes * 2;
constexpr unsigned cAtBeta = cHeads * cDim * cChunk * 2;
constexpr unsigned cAtE = cAtBeta + cHeads * cChunk * 4;
constexpr unsigned cAtG = cAtE + cHeads * cChunk * 4;
constexpr unsigned cAtDecay = cAtG + cHeads * cChunk * 4;

// pWork0 and pWork1, in bytes
constexpr unsigned cAtK = 0;
constexpr unsigned cAtQPairs = cBlocksD * cBlocksT * cPairBytes;
constexpr unsigned cAtKT = 0;
constexpr unsigned cAtS = cBlocksD * cBlocksT * cPairBytes;

// pSums: the state, then the work of a chunk; in floats up to cAtPairs, in bytes from there
constexpr unsigned cAtState = 0;
constexpr unsigned cAtKK = cHeads * cBlocksD * cLanes;      // k k^T, groups 00, 10, 11
constexpr unsigned cAtQK = cAtKK + 3 * cLanes;               // q k^T, alike
constexpr unsigned cAtKS = cAtQK + 3 * cLanes;               // (K S)^T [head][block of t]: rows j, columns t
constexpr unsigned cAtQS = cAtKS + cHeads * cBlocksT * cLanes;  // (Q S)^T alike
constexpr unsigned cAtM = cAtQS + cHeads * cBlocksT * cLanes;   // M [head][groups 00, 10, 11]
constexpr unsigned cAtR = cAtM + cHeads * 3 * cLanes;            // R, then Y, of a head
constexpr unsigned cAtPairs = cAtR + cBlocksT * cLanes;
constexpr unsigned cBytesPairs = cAtPairs * 4;
constexpr unsigned cAtGN = cBytesPairs;                       // (D_G new)^T [head][block of t], for the state
constexpr unsigned cAtA10 = cAtGN + cHeads * cBlocksT * cPairBytes;
constexpr unsigned cAtT = cAtA10 + cPairBytes;                // T00, T11
constexpr unsigned cAtKq = cAtT + 2 * cPairBytes;             // Gamma (Q K^T) [head][00, 10, 11]
constexpr unsigned cAtN = cAtKq + cHeads * 3 * cPairBytes;    // new^T [block of t]
constexpr unsigned cAtCounter = cAtN + cBlocksT * cPairBytes;
constexpr unsigned cSumsBytes = 16384;
static_assert(cAtCounter + 4 <= cSumsBytes, "the work of a chunk does not fit in the sums");

using Xdna2Sums = aie::accum<accfloat, cLanes>;
using Xdna2Bf16 = aie::vector<bfloat16, cLanes>;
using Xdna2Floats = aie::vector<float, cLanes>;
using Xdna2Block = aie::block_vector<bfp16ebs8, cLanes>;
using Xdna2Reader = aie::block_vector_input_buffer_stream<bfp16ebs8, cLanes>;
using Xdna2Writer = aie::block_vector_output_buffer_stream<bfp16ebs8, cLanes>;

// the two parts of a group: the high one, and what is left of the 32-bit floats
struct Xdna2Pair {
  Xdna2Block vHi;
  Xdna2Block vLo;
};

// the lanes of a group are 8 rows of 8 columns
constexpr uint64_t cDiagonal = 0x8040201008040201ull;

inline bfp16ebs8 *fGroups(uint8_t *pBase, unsigned pBytes) {
  return reinterpret_cast<bfp16ebs8 *>(pBase + pBytes);
}

inline Xdna2Sums fFromVector(const Xdna2Floats &pFloats) {
  Xdna2Sums vSum;
  vSum.from_vector(pFloats);
  return vSum;
}

inline Xdna2Sums fFromFloats(const float *pFloats) {
  return fFromVector(aie::load_v<cLanes>(pFloats));
}

// p8 on every row of a group
inline Xdna2Floats fColumns(const float *p8) {
  const aie::vector<float, 8> v = aie::load_v<8>(p8);
  return aie::concat(v, v, v, v, v, v, v, v);
}

// the diagonal of a group, as floats: p8 there, or pValue
inline Xdna2Floats fDiagonalFloats(const Xdna2Floats &pOn) {
  return aie::select(aie::zeros<float, cLanes>(), pOn, aie::mask<cLanes>::from_uint64(cDiagonal));
}

// minus the identity in bfp16, built in registers: every function that splits builds its own, because a block of
// bfp16 that a function reads from memory (a reference) comes in byte by byte
inline Xdna2Block fMinusIdentity() {
  return fFromVector(fDiagonalFloats(aie::broadcast<float, cLanes>(-1.0f))).template to_vector<bfp16ebs8>();
}

inline Xdna2Pair fSplit(Xdna2Sums pSum, const Xdna2Block &pMinusI) {
  Xdna2Pair vPair;
  vPair.vHi = pSum.template to_vector<bfp16ebs8>();
  pSum = mac_8x8_8x8T(vPair.vHi, pMinusI, pSum);
  vPair.vLo = pSum.template to_vector<bfp16ebs8>();
  return vPair;
}

// diag(p8), split
inline Xdna2Pair fDiagonal(const float *p8, const Xdna2Block &pMinusI) {
  return fSplit(fFromVector(fDiagonalFloats(fColumns(p8))), pMinusI);
}

inline void fPut(Xdna2Writer &pOut, const Xdna2Pair &pPair) {
  pOut.push(pPair.vHi);
  pOut.push(pPair.vLo);
}

inline Xdna2Pair fGet(Xdna2Reader &pIn) {
  Xdna2Pair vPair;
  vPair.vHi = pIn.pop();
  vPair.vLo = pIn.pop();
  return vPair;
}

inline Xdna2Pair fPairAt(uint8_t *pBase, unsigned pBytes) {
  Xdna2Reader vIn(fGroups(pBase, pBytes));
  return fGet(vIn);
}

inline void fPairTo(uint8_t *pBase, unsigned pBytes, const Xdna2Pair &pPair) {
  Xdna2Writer vOut(fGroups(pBase, pBytes));
  fPut(vOut, pPair);
}

// pSum + A B^T with both split
inline Xdna2Sums fMac3(Xdna2Sums pSum, const Xdna2Pair &pA, const Xdna2Pair &pB) {
  pSum = mac_8x8_8x8T(pA.vHi, pB.vHi, pSum);
  pSum = mac_8x8_8x8T(pA.vHi, pB.vLo, pSum);
  return mac_8x8_8x8T(pA.vLo, pB.vHi, pSum);
}

inline Xdna2Sums fProduct(const Xdna2Pair &pA, const Xdna2Pair &pB) {
  return fMac3(aie::zeros<accfloat, cLanes>(), pA, pB);
}

inline Xdna2Floats fFloats(const Xdna2Sums &pSum) {
  return pSum.template to_vector<float>();
}

// the three parts of bf16 of a group of 32-bit floats
inline void fParts3(Xdna2Sums pSum, Xdna2Bf16 &pP1, Xdna2Bf16 &pP2, Xdna2Bf16 &pP3) {
  const Xdna2Bf16 vOne = aie::broadcast<bfloat16, cLanes>(bfloat16(1.0f));
  pP1 = pSum.template to_vector<bfloat16>();
  pSum = aie::msc(pSum, pP1, vOne);
  pP2 = pSum.template to_vector<bfloat16>();
  pSum = aie::msc(pSum, pP2, vOne);
  pP3 = pSum.template to_vector<bfloat16>();
}

// a group of 32-bit floats by bf16, lane by lane: three exact products
inline Xdna2Sums fTimes(const Xdna2Sums &pSum, const Xdna2Bf16 &pFactor) {
  Xdna2Bf16 vP1;
  Xdna2Bf16 vP2;
  Xdna2Bf16 vP3;
  fParts3(pSum, vP1, vP2, vP3);
  Xdna2Sums vOut = aie::mul(vP1, pFactor);
  vOut = aie::mac(vOut, vP2, pFactor);
  return aie::mac(vOut, vP3, pFactor);
}

// The groups of bf16 at pIn (pTransposed: each one transposed), split into pOut in the order [block of d][block of
// t]: one writer per loop, because the core has one register of stream for writing, and with three writers in one
// loop the compiler kept it in the stack and emptied it at every push
// loop (48 cycles for three groups). The transposition is a parameter of the template: a branch in the loop kept the
// compiler from overlapping its steps.
template <bool pTransposed>
__attribute__((noinline)) void fSplitBf16(const bfloat16 *__restrict pIn, uint8_t *__restrict pOut) {
  const Xdna2Block vMinusI = fMinusIdentity();
  Xdna2Writer vOut(fGroups(pOut, 0));
#pragma clang loop min_iteration_count(4)
  for (unsigned vAt = 0; vAt < cBlocksD * cBlocksT; ++vAt) {
    const unsigned vD = vAt / cBlocksT;
    const unsigned vT = vAt % cBlocksT;
    const Xdna2Bf16 vGroup = aie::load_v<cLanes>(pIn + (vT * cBlocksD + vD) * cLanes);
    if constexpr (pTransposed) {
      fPut(vOut, fSplit(Xdna2Sums(aie::transpose(vGroup, 8, 8)), vMinusI));
    } else {
      fPut(vOut, fSplit(Xdna2Sums(vGroup), vMinusI));
    }
  }
}

// the state, split in the order [block of d][head]
__attribute__((noinline)) void fSplitState(const float *__restrict pIn, uint8_t *__restrict pOut) {
  const Xdna2Block vMinusI = fMinusIdentity();
  Xdna2Writer vOut(fGroups(pOut, 0));
#pragma clang loop min_iteration_count(4)
  for (unsigned vAt = 0; vAt < cBlocksD * cHeads; ++vAt) {
    const unsigned vD = vAt / cHeads;
    const unsigned vH = vAt % cHeads;
    fPut(vOut, fSplit(fFromFloats(pIn + (vH * cBlocksD + vD) * cLanes), vMinusI));
  }
}

// k, q and k^T split, and the state
__attribute__((noinline)) void fSplitInputs(const bfloat16 *pK, const bfloat16 *pQ, const float *pSums,
                                            uint8_t *pWork0, uint8_t *pWork1) {
  fSplitBf16<false>(pK, pWork0 + cAtK);
  fSplitBf16<false>(pQ, pWork0 + cAtQPairs);
  fSplitBf16<true>(pK, pWork1 + cAtKT);
  fSplitState(pSums + cAtState, pWork1 + cAtS);
}

// pA B^T for the four groups of two readers of pairs along d, 2 pairs each (blocks of t or heads): [a][b] into pOut
__attribute__((noinline)) void fFour(uint8_t *pA, uint8_t *pB, float *pOut) {
  Xdna2Reader vA(fGroups(pA, 0));
  Xdna2Reader vB(fGroups(pB, 0));
  Xdna2Sums v00 = aie::zeros<accfloat, cLanes>();
  Xdna2Sums v01 = aie::zeros<accfloat, cLanes>();
  Xdna2Sums v10 = aie::zeros<accfloat, cLanes>();
  Xdna2Sums v11 = aie::zeros<accfloat, cLanes>();
  for (unsigned vD = 0; vD < cBlocksD; ++vD) {
    const Xdna2Pair vA0 = fGet(vA);
    const Xdna2Pair vA1 = fGet(vA);
    const Xdna2Pair vB0 = fGet(vB);
    const Xdna2Pair vB1 = fGet(vB);
    v00 = fMac3(v00, vA0, vB0);
    v01 = fMac3(v01, vA0, vB1);
    v10 = fMac3(v10, vA1, vB0);
    v11 = fMac3(v11, vA1, vB1);
  }
  aie::store_v(pOut, fFloats(v00));
  aie::store_v(pOut + cLanes, fFloats(v01));
  aie::store_v(pOut + 2 * cLanes, fFloats(v10));
  aie::store_v(pOut + 3 * cLanes, fFloats(v11));
}

// k k^T and q k^T (groups 00, 10, 11), and (K S)^T and (Q S)^T [head][block of t], along d
__attribute__((noinline)) void fProducts(uint8_t *pWork0, uint8_t *pWork1, float *pSums) {
  for (unsigned vWhich = 0; vWhich < 2; ++vWhich) {
    Xdna2Reader vX(fGroups(pWork0, vWhich == 0 ? cAtK : cAtQPairs));
    Xdna2Reader vK(fGroups(pWork0, cAtK));
    Xdna2Sums v00 = aie::zeros<accfloat, cLanes>();
    Xdna2Sums v10 = aie::zeros<accfloat, cLanes>();
    Xdna2Sums v11 = aie::zeros<accfloat, cLanes>();
    for (unsigned vD = 0; vD < cBlocksD; ++vD) {
      const Xdna2Pair vX0 = fGet(vX);
      const Xdna2Pair vX1 = fGet(vX);
      const Xdna2Pair vK0 = fGet(vK);
      const Xdna2Pair vK1 = fGet(vK);
      v00 = fMac3(v00, vX0, vK0);
      v10 = fMac3(v10, vX1, vK0);
      v11 = fMac3(v11, vX1, vK1);
    }
    float *vOut = pSums + (vWhich == 0 ? cAtKK : cAtQK);
    aie::store_v(vOut, fFloats(v00));
    aie::store_v(vOut + cLanes, fFloats(v10));
    aie::store_v(vOut + 2 * cLanes, fFloats(v11));
  }
  fFour(pWork1 + cAtS, pWork0 + cAtK, pSums + cAtKS);
  fFour(pWork1 + cAtS, pWork0 + cAtQPairs, pSums + cAtQS);
}

// (I + D)^-1 for D = M_dd D_beta, the diagonal block of M at pM, beta of its tokens at pBeta: by doubling,
// (I - D)(I + D^2)(I + D^4), with every power of D and its transpose (D^T = D_beta M_dd^T), into pPairs at pTo
__attribute__((noinline)) void fInverse(const float *pM, const float *pBeta, uint8_t *pPairs, unsigned pTo) {
  const Xdna2Block vMinusI = fMinusIdentity();
  const Xdna2Floats vI = fDiagonalFloats(aie::broadcast<float, cLanes>(1.0f));
  const Xdna2Pair vM = fSplit(fFromFloats(pM), vMinusI);
  const Xdna2Pair vB = fDiagonal(pBeta, vMinusI);
  const Xdna2Sums vD = fProduct(vM, vB);
  const Xdna2Pair vDp = fSplit(vD, vMinusI);
  const Xdna2Pair vDt = fSplit(fProduct(vB, vM), vMinusI);
  const Xdna2Pair vD2 = fSplit(fProduct(vDp, vDt), vMinusI);                    // D D
  const Xdna2Sums vD2t = fProduct(vDt, vDp);                                     // D^T D^T
  const Xdna2Floats vD4t = fFloats(fProduct(fSplit(vD2t, vMinusI), vD2));        // D^2^T D^2^T
  const Xdna2Pair vMinus = fSplit(fFromVector(aie::sub(vI, fFloats(vD))), vMinusI);         // I - D
  const Xdna2Pair vPlus2 = fSplit(fFromVector(aie::add(vI, fFloats(vD2t))), vMinusI);       // (I + D^2)^T
  const Xdna2Pair vP = fSplit(fProduct(vMinus, vPlus2), vMinusI);                // (I - D)(I + D^2)
  const Xdna2Pair vPlus4 = fSplit(fFromVector(aie::add(vI, vD4t)), vMinusI);     // (I + D^4)^T
  fPairTo(pPairs, pTo, fSplit(fProduct(vP, vPlus4), vMinusI));
}

// M = Gamma (K K^T) without its diagonal, and Kq = Gamma (Q K^T) with it, groups 00, 10, 11 of both heads
__attribute__((noinline)) void fGammaTimes(const uint8_t *pIn0, float *pSums) {
  const Xdna2Block vMinusI = fMinusIdentity();
  uint8_t *vPairs = reinterpret_cast<uint8_t *>(pSums);
  const bfloat16 *vGamma = reinterpret_cast<const bfloat16 *>(pIn0 + cAtGamma);
  for (unsigned vAt = 0; vAt < cHeads * 3; ++vAt) {
    const unsigned vGroup = vAt % 3;
    const Xdna2Bf16 vGam = aie::load_v<cLanes>(vGamma + vAt * cLanes);
    const Xdna2Bf16 vUnder =
        vGroup == 1 ? vGam
                    : aie::select(vGam, aie::zeros<bfloat16, cLanes>(), aie::mask<cLanes>::from_uint64(cDiagonal));
    aie::store_v(pSums + cAtM + vAt * cLanes, fFloats(fTimes(fFromFloats(pSums + cAtKK + vGroup * cLanes), vUnder)));
    fPairTo(vPairs, cAtKq + vAt * cPairBytes,
            fSplit(fTimes(fFromFloats(pSums + cAtQK + vGroup * cLanes), vGam), vMinusI));
  }
}

// the inverse by blocks, new and the outputs of one head; (D_G new)^T is left for the state
__attribute__((noinline)) void fHead(unsigned pHead, const uint8_t *pIn1, float *pSums, float *pOut, unsigned pBlockJ) {
  const Xdna2Block vMinusI = fMinusIdentity();
  uint8_t *vPairs = reinterpret_cast<uint8_t *>(pSums);
  const float *vBeta = reinterpret_cast<const float *>(pIn1 + cAtBeta) + pHead * cChunk;
  const float *vE = reinterpret_cast<const float *>(pIn1 + cAtE) + pHead * cChunk;
  const float *vG = reinterpret_cast<const float *>(pIn1 + cAtG) + pHead * cChunk;
  const bfloat16 *vV = reinterpret_cast<const bfloat16 *>(pIn1);
  const float *vM = pSums + cAtM + pHead * 3 * cLanes;
  const unsigned vKq = cAtKq + pHead * 3 * cPairBytes;
  // T00 and T11, and A10 = M10 D_beta of the first block
  fInverse(vM, vBeta, vPairs, cAtT);
  fInverse(vM + 2 * cLanes, vBeta + 8, vPairs, cAtT + cPairBytes);
  fPairTo(vPairs, cAtA10,
          fSplit(fProduct(fSplit(fFromFloats(vM + cLanes), vMinusI), fDiagonal(vBeta, vMinusI)), vMinusI));
  // R^T = V^T - (K S)^T D_E, rows j and columns t
  for (unsigned vT = 0; vT < cBlocksT; ++vT) {
    const Xdna2Sums vVt(aie::load_v<cLanes>(vV + ((pHead * cBlocksD + pBlockJ) * cBlocksT + vT) * cLanes));
    const Xdna2Pair vKS = fSplit(fFromFloats(pSums + cAtKS + (pHead * cBlocksT + vT) * cLanes), vMinusI);
    aie::store_v(pSums + cAtR + vT * cLanes,
                 aie::sub(fFloats(vVt), fFloats(fProduct(vKS, fDiagonal(vE + 8 * vT, vMinusI)))));
  }
  // X^T = R^T T^T by blocks, X0 = T00 R0 and X1 = T11 (R1 - A10 X0), and new^T = X^T D_beta
  const Xdna2Pair vX0 =
      fSplit(fProduct(fSplit(fFromFloats(pSums + cAtR), vMinusI), fPairAt(vPairs, cAtT)), vMinusI);
  const Xdna2Floats vY1 =
      aie::sub(aie::load_v<cLanes>(pSums + cAtR + cLanes), fFloats(fProduct(vX0, fPairAt(vPairs, cAtA10))));
  const Xdna2Pair vX1 =
      fSplit(fProduct(fSplit(fFromVector(vY1), vMinusI), fPairAt(vPairs, cAtT + cPairBytes)), vMinusI);
  fPairTo(vPairs, cAtN, fSplit(fProduct(vX0, fDiagonal(vBeta, vMinusI)), vMinusI));
  fPairTo(vPairs, cAtN + cPairBytes, fSplit(fProduct(vX1, fDiagonal(vBeta + 8, vMinusI)), vMinusI));
  // the outputs, o = D_E (Q S) + Kq new, rows t and columns j, into this chunk's place of the object of C; and
  // (D_G new)^T for the state
  for (unsigned vT = 0; vT < cBlocksT; ++vT) {
    const Xdna2Pair vN = fPairAt(vPairs, cAtN + vT * cPairBytes);
    const Xdna2Pair vQS = fSplit(fFromFloats(pSums + cAtQS + (pHead * cBlocksT + vT) * cLanes), vMinusI);
    Xdna2Sums vSum = fProduct(fDiagonal(vE + 8 * vT, vMinusI), vQS);
    vSum = fMac3(vSum, fPairAt(vPairs, vKq + vT * cPairBytes), fPairAt(vPairs, cAtN));
    if (vT == 1) {
      vSum = fMac3(vSum, fPairAt(vPairs, vKq + 2 * cPairBytes), vN);
    }
    aie::store_v(pOut + vT * cLanes, fFloats(vSum));
    fPairTo(vPairs, cAtGN + (pHead * cBlocksT + vT) * cPairBytes,
            fSplit(fProduct(vN, fDiagonal(vG + 8 * vT, vMinusI)), vMinusI));
  }
}

// S <- e^cs_last S + K^T (D_G new), in two passes with few values alive each (in one, the three parts of the decay,
// the two pairs of (D_G new)^T and those of k^T went in and out of the stack: 50 cycles a group). The decay goes
// through the matrix unit too: the state in three parts of bfp16 (some 21 bits of the largest of every 8 values of a
// row, which are alike: k has norm 1) by e^cs_last on the diagonal in three parts, the six products whose weight is
// 2^-16 or more, in two chains. Lane by lane, as three parts of bf16 by three, it took five registers of sums and
// the compiler did not overlap the steps (30 to 37 cycles a group). pIn and pOut are the same state: two pointers
// that the compiler takes for different memory, so that it overlaps the steps of the loop (every step reads and
// writes its own group).
__attribute__((noinline)) void fDecay(const float *pDecay, const float *__restrict pIn, float *__restrict pOut) {
  const Xdna2Block vMinusI = fMinusIdentity();
  for (unsigned vH = 0; vH < cHeads; ++vH) {
    Xdna2Sums vDiagonal = fFromVector(fDiagonalFloats(aie::broadcast<float, cLanes>(pDecay[vH])));
    const Xdna2Block vD1 = vDiagonal.template to_vector<bfp16ebs8>();
    vDiagonal = mac_8x8_8x8T(vD1, vMinusI, vDiagonal);
    const Xdna2Block vD2 = vDiagonal.template to_vector<bfp16ebs8>();
    vDiagonal = mac_8x8_8x8T(vD2, vMinusI, vDiagonal);
    const Xdna2Block vD3 = vDiagonal.template to_vector<bfp16ebs8>();
    const float *__restrict vIn = pIn + vH * cBlocksD * cLanes;
    float *__restrict vOut = pOut + vH * cBlocksD * cLanes;
#pragma clang loop min_iteration_count(4)
    for (unsigned vD = 0; vD < cBlocksD; ++vD) {
      Xdna2Sums vS = fFromFloats(vIn + vD * cLanes);
      const Xdna2Block vS1 = vS.template to_vector<bfp16ebs8>();
      vS = mac_8x8_8x8T(vS1, vMinusI, vS);
      const Xdna2Block vS2 = vS.template to_vector<bfp16ebs8>();
      vS = mac_8x8_8x8T(vS2, vMinusI, vS);
      const Xdna2Block vS3 = vS.template to_vector<bfp16ebs8>();
      Xdna2Sums vSumA = mul_8x8_8x8T(vS1, vD1);
      Xdna2Sums vSumB = mul_8x8_8x8T(vS1, vD2);
      vSumA = mac_8x8_8x8T(vS2, vD1, vSumA);
      vSumB = mac_8x8_8x8T(vS1, vD3, vSumB);
      vSumA = mac_8x8_8x8T(vS2, vD2, vSumA);
      vSumB = mac_8x8_8x8T(vS3, vD1, vSumB);
      aie::store_v(vOut + vD * cLanes, fFloats(aie::add(vSumA, fFloats(vSumB))));
    }
  }
}

__attribute__((noinline)) void fUpdate(uint8_t *pWork1, uint8_t *pPairs, const float *__restrict pIn,
                                       float *__restrict pOut) {
  for (unsigned vH = 0; vH < cHeads; ++vH) {
    const Xdna2Pair vG0 = fPairAt(pPairs, cAtGN + (vH * cBlocksT) * cPairBytes);
    const Xdna2Pair vG1 = fPairAt(pPairs, cAtGN + (vH * cBlocksT + 1) * cPairBytes);
    Xdna2Reader vKT(fGroups(pWork1, cAtKT));
    const float *__restrict vIn = pIn + vH * cBlocksD * cLanes;
    float *__restrict vOut = pOut + vH * cBlocksD * cLanes;
#pragma clang loop min_iteration_count(4)
    for (unsigned vD = 0; vD < cBlocksD; ++vD) {
      const Xdna2Pair vKT0 = fGet(vKT);
      const Xdna2Pair vKT1 = fGet(vKT);
      Xdna2Sums vSumC = mac_8x8_8x8T(vG0.vHi, vKT0.vHi, fFromFloats(vIn + vD * cLanes));
      Xdna2Sums vSumD = mul_8x8_8x8T(vG1.vHi, vKT1.vHi);
      vSumC = mac_8x8_8x8T(vG0.vHi, vKT0.vLo, vSumC);
      vSumD = mac_8x8_8x8T(vG1.vHi, vKT1.vLo, vSumD);
      vSumC = mac_8x8_8x8T(vG0.vLo, vKT0.vHi, vSumC);
      vSumD = mac_8x8_8x8T(vG1.vLo, vKT1.vHi, vSumD);
      aie::store_v(vOut + vD * cLanes, fFloats(aie::add(vSumC, fFloats(vSumD))));
    }
  }
}

}  // namespace

extern "C" {

#if defined(XDNA2_GDN_STATE) || defined(XDNA2_GDN_ALL)
void xdna2_gdn_state(uint8_t *pObject, float *pSums, int32_t pRowAt, int32_t pRow) {
  int32_t *vCounter = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(pSums) + cAtCounter);
  if (pRowAt == 0) {
    *vCounter = 0;
  }
  if (pRowAt != pRow) {
    return;
  }
  // [head][j][d] into groups [head][block of d][8 j][8 d]
  const float *vIn = reinterpret_cast<const float *>(pObject);
  for (unsigned vH = 0; vH < cHeads; ++vH) {
    for (unsigned vD = 0; vD < cBlocksD; ++vD) {
      Xdna2Floats vGroup;
      for (unsigned vJ = 0; vJ < cRowsJ; ++vJ) {
        vGroup.insert(vJ, aie::load_v<8>(vIn + (vH * cRowsJ + vJ) * cDim + 8 * vD));
      }
      aie::store_v(pSums + cAtState + (vH * cBlocksD + vD) * cLanes, vGroup);
    }
  }
}

#endif

#if defined(XDNA2_GDN_KQ) || defined(XDNA2_GDN_ALL)
// the first half of a chunk, with its first object: k and q split, the products along d, M and Kq
void xdna2_gdn_kq(uint8_t *pIn0, float *pSums, uint8_t *pWork0, uint8_t *pWork1) {
#if defined(XDNA2_GDN_SKIP_KQ)
  return;  // to measure: the data move without this half
#endif
  // the core starts in the rounding mode floor; the splits and the products of bf16 want the nearest
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  // XDNA2_GDN_SKIP_<piece>: to measure what every piece takes
#if !defined(XDNA2_GDN_SKIP_SPLIT)
  fSplitInputs(reinterpret_cast<const bfloat16 *>(pIn0), reinterpret_cast<const bfloat16 *>(pIn0 + cAtQ), pSums, pWork0,
               pWork1);
#endif
#if !defined(XDNA2_GDN_SKIP_PRODUCTS)
  fProducts(pWork0, pWork1, pSums);
#endif
#if !defined(XDNA2_GDN_SKIP_GAMMA)
  fGammaTimes(pIn0, pSums);
#endif
  aie::set_rounding(vSaved);
}

#endif

#if defined(XDNA2_GDN_REST) || defined(XDNA2_GDN_ALL)
// the second half, with its second object: the inverses, new, the outputs and the state
void xdna2_gdn_rest(uint8_t *pIn1, float *pSums, uint8_t *pWork1, float *pOut, int32_t pBlockJ) {
#if defined(XDNA2_GDN_SKIP_REST)
  return;  // to measure: the data move without this half
#endif
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  int32_t *vCounter = reinterpret_cast<int32_t *>(reinterpret_cast<uint8_t *>(pSums) + cAtCounter);
  const unsigned vChunk = *vCounter;
#if !defined(XDNA2_GDN_SKIP_HEAD)
  for (unsigned vH = 0; vH < cHeads; ++vH) {
    fHead(vH, pIn1, pSums, pOut + (vH * cChunksOut + vChunk) * cChunk * cRowsJ, pBlockJ);
  }
#endif
#if !defined(XDNA2_GDN_SKIP_DECAY)
  fDecay(reinterpret_cast<const float *>(pIn1 + cAtDecay), pSums + cAtState, pSums + cAtState);
#endif
#if !defined(XDNA2_GDN_SKIP_UPDATE)
  fUpdate(pWork1, reinterpret_cast<uint8_t *>(pSums), pSums + cAtState, pSums + cAtState);
#endif
  *vCounter = (vChunk + 1) % cChunksOut;
  aie::set_rounding(vSaved);
}

#endif

#if defined(XDNA2_GDN_STATE_OUT) || defined(XDNA2_GDN_ALL)
void xdna2_gdn_state_out(float *pSums, float *pOut) {
  for (unsigned vH = 0; vH < cHeads; ++vH) {
    for (unsigned vD = 0; vD < cBlocksD; ++vD) {
      const Xdna2Floats vGroup = aie::load_v<cLanes>(pSums + cAtState + (vH * cBlocksD + vD) * cLanes);
      for (unsigned vJ = 0; vJ < cRowsJ; ++vJ) {
        aie::store_v(pOut + (vH * cRowsJ + vJ) * cDim + 8 * vD, vGroup.template extract<8>(vJ));
      }
    }
  }
}

#endif

}  // extern "C"
