// llama.cpp-xdna: the kernel of the chained overlay for generating tokens (xdna2-gemv 9), for the core tiles of the
// XDNA2 NPU (AIE2P). Like xdna2-gemv-bfp16.cc it multiplies a few rows by a matrix of weights with 32-bit float sums,
// with the same pieces of weights (bfp16, Q4_K, Q5_K and Q6_K, decoded here), but the two cores fed by a channel
// share the work along K instead of along the outputs, and a command can run several phases one after the other,
// each taking its X from the host or from the Y of the phase before it (see gemv_chain.py of the lab):
//
//   - The core of an even row (core 0 of the pair) takes the first half of the tiles of 256 values of K of every
//     block of 32 outputs, the core above it (core 1) the second half; core 1 sends its sums of every block to core
//     0 over the cascade, which adds them, finishes them (the epilogue of the phase) and sends Y. Every core keeps
//     only its half of X, and nothing is duplicated in the channel: every object of the channel is [4608 bytes of
//     core 0 | 4608 bytes of core 1].
//   - The pieces of the blocks of a segment follow each other in the objects of a core with no padding between
//     blocks (version 6; before, every block started an object of its own, and the padding of the last object of a
//     block was some 9 % of the bytes of Q6_K and 14 % of those of Q5_K with K = 2560). So a block can end inside an
//     object: the call of the gemv stops there and says so, the block is finished or sent, and the next call goes on
//     in the same object. Only the last object of a segment has padding.
//   - The parameters come at the start of every command, in an object of their own (the same in both halves), and
//     a phase begins with the objects of its prologue: the residual of the epilogue and X, in bfp16 from the host or
//     in 32-bit floats (one row), which the core rounds to bfp16, or the rows of an RMS norm, h and its weights.
//
//   xdna2_chain_params(pObject, pParams, pStage)                          built with -DXDNA2_CHAIN_PARAMS
//     copies the parameters of the command from its first object
//   xdna2_chain_phase(pStage, pParams)                                    -DXDNA2_CHAIN_PHASE
//     starts the next phase
//   xdna2_chain_prologue(pObject, pX, pStage, pChain, pParams, pCore)    -DXDNA2_CHAIN_PROLOGUE
//     the next object of the prologue of the phase
//   xdna2_chain_gemv(pX, pW, pSums, pStage, pParams, pRows, pFormat, pTiles, pBlocks)   -DXDNA2_CHAIN_GEMV
//     as xdna2_gemv_bfp16, for 4 groups of 8 outputs and objects of 4608 bytes, into the sums of the block that runs
//     (pSums); it stops where a block ends, with pParams[4][1] set to 1 (0 otherwise), and the next call goes on in
//     the same object. pBlocks, the blocks of the segment: the call after the last one ends the segment (what is left
//     of its object is padding) and does nothing else. With -DXDNA2_CHAIN_NO_COMPUTE it only counts the pieces.
//   xdna2_chain_send(pSums, pStage)                                       -DXDNA2_CHAIN_SEND
//     core 1: its sums of a block, over the cascade
//   xdna2_chain_finish(pSums, pStage, pChain, pParams)                    -DXDNA2_CHAIN_FINISH
//     core 0: the sums of core 1 for the block added to its own, the scale of the norm and the epilogue of the phase,
//     in its sums
//   xdna2_chain_emit(pY, pSums, pStage)                                   -DXDNA2_CHAIN_EMIT
//     core 0: the next row of the block (0, 1...) into an object of Y: its 32 outputs, [4 groups][8 outputs] (version
//     7; before, an object of Y was the whole block, [4 groups][8 rows][8 outputs], with room for 8 rows, and with one
//     row 7 of every 8 floats were sent for nothing)
//   xdna2_chain_gdn0(pW, pX, pSums, pStage, pParams)                      -DXDNA2_CHAIN_GDN0
//   xdna2_chain_gdn1(pW, pX, pSums, pStage)                               -DXDNA2_CHAIN_GDN1
//     version 8, a phase of the recurrence of a layer of linear attention (see fGdnRow): core 0 and core 1, the next
//     row of the state of the head of the core in its object of W; core 0 writes in pParams[4][1] the objects of Y it
//     sends for the row
//   xdna2_chain_gdn_emit(pY, pW, pSums, pStage)                           -DXDNA2_CHAIN_GDN_EMIT
//     core 0: the next object of Y of a row of the recurrence: its row, that of core 1 (over the cascade), and after the
//     last row of a head the outputs of both
//   xdna2_chain_conv(pX, pParams)                                         -DXDNA2_CHAIN_CONV
//   xdna2_chain_conv_recv(pX, pParams)                                    -DXDNA2_CHAIN_CONV_RECV
//     version 9, a phase of a layer of linear attention (pParams[phase][26] = 1, see cLaItem): core 1, the convolution
//     of the 512 channels of the pair, their SiLU and the L2 norms of q and k, and q, k and v of the head of core 0
//     over the cascade; core 0, those; both, the inputs of the recurrence of their head, which then ends with the gated
//     norm of the head
//
// The parameters, 5 rows of 32 32-bit integers (eps a float): row p (0 to 3) for phase p, row 4 [0] the phases (1 to 4)
// and [1] whether the last call of the gemv ended a block, or the objects of Y of a row of the recurrence (written by
// the kernel, read by the program of the core):
//   [0] prologue: 0 X in bfp16 from the host, 1 X in floats (1 row), 2 RMS norm (1 row), 3 (version 9) the objects
//       of a phase of linear attention, copied as they come from the second object of X on
//   [1] objects of the prologue                     [2] rows       [3] tiles of K of a core    [4] epilogue
//   [5] objects of the residual (the first ones of the prologue)    [6] eps    [7] values of K of a core
//   [8 + 4 s] segment s (0 and 1; the overlay runs 2): blocks, objects of the segment (of a core), format, 0
//   [24] version 8: objects of the states of the recurrence of a core (0: none)
//   [25] version 8: 1 / K, for the mean of the squares of the RMS norm (a float)
//   [26] version 9: 1 in a phase of a layer of linear attention (prologue mode 3, then the convolution and the
//        recurrence with the gated norm), 0 otherwise
// Epilogues (core 0, one row): 0 none; 1 Y + r, the residual of the stream (r[32 b + 8 g + i]), kept as h; 2 Y + h;
// 3 SwiGLU: segment 0 (the gate) is kept, and segment 1 (the up) becomes silu(gate) * up. With the prologue of the
// norm, the sums of a block are first multiplied by the scale of the norm (the norm is applied to X without it).
//
// pChain, 1728 32-bit floats of a core: the residual of the stream [0, 320), h [320, 640), the gate [640, 1728) (320
// outputs a stream: 5120 outputs; 1088: 17408). It was 2048 floats before version 6, which made X fit in 7 objects.

#include <aie_api/aie.hpp>

#include <stdint.h>

namespace {

constexpr unsigned cBlock = 8;                        // values of a block, and blocks of a group
constexpr unsigned cBlockBytes = 9;                   // bytes of a block
constexpr unsigned cLanes = cBlock * cBlock;          // values of a group, and sums of a group of Y
constexpr unsigned cGroupBytes = cBlock * cBlockBytes;            // 72
constexpr unsigned cGroupsOut = 4;                    // groups of 8 outputs of a block: pieces of a tile
constexpr unsigned cObjectBytes = 4608;               // bytes of an object of a core: half an object of a channel
constexpr unsigned cTileGroups = 32;                  // groups of 8 values of K of a tile
constexpr unsigned cStageBytes = cTileGroups * cGroupBytes;       // 2304
constexpr unsigned cPieceCounter = cStageBytes;       // offsets of the counters in pStage: the next piece of the block,
constexpr unsigned cCarryCounter = cStageBytes + 4;   // the bytes of a piece carried from the last object, and where
constexpr unsigned cObjectCounter = cStageBytes + 8;  // the next call goes on in the object (0: a new object)
constexpr unsigned cFactors = cStageBytes + 64;       // the factors of the sub-blocks of a piece (aligned to 64)
constexpr unsigned cBases = cFactors + 16 * cLanes * 2;           // and the bases of those of Q4_K and Q5_K
constexpr unsigned cCarry = cBases + 8 * cLanes * 4;  // a piece that goes on into the next object
constexpr unsigned cCarryBytes = 1792;

// the formats, and the bytes of a piece of each
constexpr int32_t cFormatBfp16 = 0;
constexpr int32_t cFormatQ4K = 1;
constexpr int32_t cFormatQ5K = 2;
constexpr int32_t cFormatQ6K = 3;
constexpr int32_t cFormatQ8_0 = 4;
constexpr unsigned cPieceBfp16 = 2304;
constexpr unsigned cPieceQ4K = 1152;
constexpr unsigned cPieceQ5K = 1408;
constexpr unsigned cPieceQ6K = 1696;
constexpr unsigned cPieceQ8_0 = 2176;

// where the parts of a piece start
constexpr unsigned cScalesK = 32;                     // Q4_K, Q5_K: the 12 bytes of scales and minimums
constexpr unsigned cValuesK = 128;                    // Q4_K, Q5_K: the 4-bit values
constexpr unsigned cFifthQ5K = 1152;                  // Q5_K: the fifth bits
constexpr unsigned cScalesQ6K = 32;                   // Q6_K: the 8-bit scales
constexpr unsigned cValuesQ6K = 160;                  // Q6_K: the low 4 bits
constexpr unsigned cTopQ6K = 1184;                    // Q6_K: the top 2 bits
constexpr unsigned cValuesQ8_0 = 128;                 // Q8_0: the 8-bit signed values, after [8][8] bf16 scales

static_assert(cObjectBytes % 32 == 0 && cPieceQ5K % 32 == 0 && cPieceQ6K % 32 == 0 && cPieceQ8_0 % 32 == 0,
              "pieces aligned to 32 bytes");
static_assert(cCarryBytes >= cPieceQ6K && cBases + cPieceQ8_0 <= cCarry + cCarryBytes, "the carry holds a piece");

using Xdna2Sums = aie::accum<accfloat, cLanes>;
using Xdna2Groups = aie::block_vector_input_buffer_stream<bfp16ebs8, cLanes>;

inline int32_t &fCounter(uint8_t *pStage, unsigned pOffset) {
  return *reinterpret_cast<int32_t *>(pStage + pOffset);
}

// bytes of a group of X: its rows, padded to a multiple of 4
inline unsigned fGroupBytesX(unsigned pRows) {
  return (pRows * cBlockBytes + 3) & ~3u;
}

inline unsigned fPieceBytes(int32_t pFormat) {
  return pFormat == cFormatQ4K  ? cPieceQ4K :
         pFormat == cFormatQ5K  ? cPieceQ5K :
         pFormat == cFormatQ6K  ? cPieceQ6K :
         pFormat == cFormatQ8_0 ? cPieceQ8_0 :
                                  cPieceBfp16;
}

// The groups of a tile, pStride 32-bit words apart, into the slots of the stage, pCopy words each (pCopy >= pStride),
// with the copy of a group unrolled.
template <unsigned pCopy>
inline void fStageWords(const uint32_t *__restrict pFrom, uint32_t *__restrict pTo, unsigned pStride) {
  for (unsigned vSlot = 0; vSlot < cTileGroups; ++vSlot) {
#pragma clang loop unroll(full)
    for (unsigned i = 0; i < pCopy; ++i) {
      pTo[i] = pFrom[i];
    }
    pFrom += pStride;
    pTo += cGroupBytes / 4;
  }
}

// Copies the groups of X of a tile into the stage, from group pFirst of X on, in 32-bit words, with a copy unrolled for
// 1, 2, 4 and 8 rows, since with a loop over the words of a group several rows took about twice as long as one. With
// 3, 5, 6 or 7 rows the copy for the next of those takes a few words more of every group, the first rows of the next
// group (or, after the last group, of what follows X), which go to rows the phase does not have. Version 6 has 4 copies
// instead of one for every number of rows: with 8, the program of core 0 did not fit in its 16 KB.
inline void fStage(const uint8_t *__restrict pX, uint8_t *__restrict pStage, unsigned pRows, unsigned pFirst) {
  const unsigned vWords = fGroupBytesX(pRows) / 4;
  const uint32_t *__restrict vFrom = reinterpret_cast<const uint32_t *>(pX + pFirst * vWords * 4);
  uint32_t *__restrict vTo = reinterpret_cast<uint32_t *>(pStage);
  if (vWords <= 3) {
    fStageWords<3>(vFrom, vTo, vWords);
    return;
  }
  for (unsigned vSlot = 0; vSlot < cTileGroups; ++vSlot) {
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < vWords; ++i) {
      vTo[i] = vFrom[i];
    }
    vFrom += vWords;
    vTo += cGroupBytes / 4;
  }
}

// copies pBytes (a multiple of 32) from pFrom to pTo, both aligned to 32
inline void fCopy(const uint8_t *__restrict pFrom, uint8_t *__restrict pTo, unsigned pBytes) {
  for (unsigned i = 0; i < pBytes; i += 32) {
    aie::store_v(pTo + i, aie::load_v<32>(pFrom + i));
  }
}

inline Xdna2Sums fLoadSums(const float *__restrict pSums, bool pFirst) {
  Xdna2Sums vSum;
  if (pFirst) {
    vSum = aie::zeros<accfloat, cLanes>();
  } else {
    vSum.from_vector(aie::load_v<cLanes>(pSums));
  }
  return vSum;
}

void fPieceBfp16(const uint8_t *__restrict pPiece, float *__restrict pSums, uint8_t *__restrict pStage, bool pFirst) {
  Xdna2Groups vStreamW(reinterpret_cast<bfp16ebs8 *>(const_cast<uint8_t *>(pPiece)));
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(pStage));
  Xdna2Sums vSum = fLoadSums(pSums, pFirst);
#pragma clang loop unroll_count(4)
  for (unsigned vGroup = 0; vGroup < cTileGroups; ++vGroup) {
    // rows of X by outputs of W, along 8 values of K
    vSum = mac_8x8_8x8T(vStreamX.pop(), vStreamW.pop(), vSum);
  }
  aie::store_v(pSums, vSum.template to_vector<float>());
}

// each of the first 8 lanes 8 times: lane 8j + t gets lane j
template <typename T> inline aie::vector<T, 64> fRepeat8(const aie::vector<T, 64> &pValues) {
  const auto [vTwo, vTwoHigh] = aie::interleave_zip(pValues, pValues, 1);
  const auto [vFour, vFourHigh] = aie::interleave_zip(vTwo, vTwo, 2);
  const auto [vEight, vEightHigh] = aie::interleave_zip(vFour, vFour, 4);
  (void) vTwoHigh;
  (void) vFourHigh;
  (void) vEightHigh;
  return vEight;
}

// 128 + q in bf16 for 64 integers q from 0 to 127, one per byte: every byte next to 0x43, the high byte of 128 in
// bf16, gives bf16 0x4300 | q
inline aie::vector<bfloat16, 64> fPlus128(const aie::vector<uint8, 64> &pQ) {
  const aie::vector<uint8, 64> v43 = aie::broadcast<uint8, 64>(uint8(0x43));
  const auto [vLow, vHigh] = aie::interleave_zip(pQ, v43, 1);
  return aie::concat(vLow, vHigh).template cast_to<bfloat16>();
}

// The first 8 lanes of pPair (32-lane uint16), each 8 times in a row as bf16: lane 8j + t gets lane j.
inline aie::vector<bfloat16, 64> fRepeat8From16(const aie::vector<uint16, 32> &pPair) {
  const auto [v2, v2High] = aie::interleave_zip(pPair, pPair, 1);
  const auto [v4, v4High] = aie::interleave_zip(v2, v2, 2);
  const auto [v8, v8High] = aie::interleave_zip(v4, v4, 4);
  (void) v2High;
  (void) v4High;
  return aie::concat(v8, v8High).template cast_to<bfloat16>();
}

// The 8 values of 16 bits at pFrom (bf16, 32-byte aligned), each 8 times in a row: lane 8j + t gets value j.
inline aie::vector<bfloat16, 64> fEightTimes(const uint8_t *__restrict pFrom) {
  return fRepeat8From16(
      aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pFrom)), aie::zeros<uint16, 16>()));
}

// The factors and the bases of the 8 sub-blocks of a piece of Q4_K or Q5_K, into pStage: factor s is d * scale of the
// sub-block s rounded to bf16, base s is - 128 * factor - dmin * minimum in 32-bit floats, both in the 8 lanes of
// every output (lane 8j + t), so that a value is base + (128 + q) * factor. The products of two bf16 come out exact
// in 32 bits: d * (128 + scale) - 128 * d is d * scale, and dmin * (128 + minimum) - 128 * dmin is dmin * minimum.
void fPrepareK(const uint8_t *__restrict pPiece, uint8_t *__restrict pStage) {
  const aie::vector<bfloat16, 64> v128 = aie::broadcast<bfloat16, 64>(bfloat16(128.0f));
  const aie::vector<uint8, 64> vLow4 = aie::broadcast<uint8, 64>(uint8(0x0f));
  const aie::vector<uint8, 64> vLow6 = aie::broadcast<uint8, 64>(uint8(0x3f));
  // the scales and minimums of the 8 sub-blocks of the 8 outputs, lane 8s + j, as ggml's get_scale_min_k4:
  // sub-blocks 0 to 3 from bytes 0 to 3 (scales) and 4 to 7 (minimums), 4 to 7 from bytes 8 to 11 and the top
  // bits of bytes 0 to 7
  const aie::vector<uint8, 32> vBytes0 = aie::load_v<32>(pPiece + cScalesK);       // bytes 0 to 3 of the 8 outputs
  const aie::vector<uint8, 32> vBytes4 = aie::load_v<32>(pPiece + cScalesK + 32);  // bytes 4 to 7
  const aie::vector<uint8, 32> vBytes8 = aie::load_v<32>(pPiece + cScalesK + 64);  // bytes 8 to 11
  const aie::vector<uint8, 64> vLowHalf = aie::concat(vBytes0, vBytes4);  // scales 0-3, minimums 0-3
  const aie::vector<uint8, 64> vTop = aie::downshift(vLowHalf, 6);        // their top two bits
  const aie::vector<uint8, 64> vHighHalf =
      aie::concat(aie::bit_and(vBytes8, vLow4.extract<32>(0)), aie::downshift(vBytes8, 4));
  const aie::vector<uint8, 64> vSixLow = aie::bit_and(vLowHalf, vLow6);
  const aie::vector<uint8, 64> vSixHigh = aie::bit_or(vHighHalf, aie::upshift(vTop, 4));
  // lanes 8s + j: scales of the sub-blocks 0 to 7, and minimums of 0 to 7
  const aie::vector<uint8, 64> vScales = aie::concat(vSixLow.extract<32>(0), vSixHigh.extract<32>(0));
  const aie::vector<uint8, 64> vMins = aie::concat(vSixLow.extract<32>(1), vSixHigh.extract<32>(1));
  // d and dmin of the 8 outputs, from one load: dmin, 16 bytes after d, is taken from the same 32 bytes, since a
  // load from pPiece + 16 would load from pPiece
  const aie::vector<bfloat16, 64> vD = fEightTimes(pPiece);
  const aie::vector<uint16, 32> vPairD =
      aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece)), aie::zeros<uint16, 16>());
  const aie::vector<bfloat16, 64> vDmin = fRepeat8From16(aie::shuffle_down(vPairD, 8));

  bfloat16 *__restrict vFactors = reinterpret_cast<bfloat16 *>(pStage + cFactors);
  float *__restrict vBases = reinterpret_cast<float *>(pStage + cBases);
  for (unsigned vSub = 0; vSub < 8; ++vSub) {
    // the scale and the minimum of the sub-block for every output, 8 times each, as 128 + them in bf16
    const aie::vector<bfloat16, 64> vScale = fPlus128(fRepeat8(aie::shuffle_down(vScales, vSub * 8)));
    const aie::vector<bfloat16, 64> vMinimum = fPlus128(fRepeat8(aie::shuffle_down(vMins, vSub * 8)));
    const aie::vector<bfloat16, 64> vFactor = aie::msc(aie::mul(vD, vScale), vD, v128).template to_vector<bfloat16>();
    const Xdna2Sums vBase = aie::mac(aie::msc(aie::negmul(vFactor, v128), vDmin, vMinimum), vDmin, v128);
    aie::store_v(vFactors + vSub * cLanes, vFactor);
    aie::store_v(vBases + vSub * cLanes, vBase.template to_vector<float>());
  }
}

// A piece of Q4_K (pFifth null) or of Q5_K (pFifth its fifth bits).
void fPieceK(const uint8_t *__restrict pPiece, const uint8_t *__restrict pFifth, float *__restrict pSums,
             uint8_t *__restrict pStage, bool pFirst) {
  fPrepareK(pPiece, pStage);
  const bfloat16 *__restrict vFactors = reinterpret_cast<const bfloat16 *>(pStage + cFactors);
  const float *__restrict vBases = reinterpret_cast<const float *>(pStage + cBases);
  const uint8_t *__restrict vValues = pPiece + cValuesK;
  const aie::vector<uint8, 64> v16 = aie::broadcast<uint8, 64>(uint8(16));
  // One set of sums: decoding a group takes longer than a product needs before the next can use its sums, and the
  // core has 5 registers for sets of 64 sums in floats
  Xdna2Sums vSum = fLoadSums(pSums, pFirst);
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(pStage));
  for (unsigned vSub = 0; vSub < 8; ++vSub) {
    const aie::vector<bfloat16, 64> vFactor = aie::load_v<cLanes>(vFactors + vSub * cLanes);
    Xdna2Sums vBase;
    vBase.from_vector(aie::load_v<cLanes>(vBases + vSub * cLanes));
    if (pFifth == nullptr) {
#pragma clang loop unroll(full)
      for (unsigned vGroup = 0; vGroup < 4; ++vGroup) {
        // a group is 8 outputs by 8 values of K of 4 bits: 32 bytes
        const aie::vector<uint8, 64> vQ =
            aie::load_v<64>(reinterpret_cast<const uint4 *>(vValues + (vSub * 4 + vGroup) * 32)).unpack();
        vSum = mac_8x8_8x8T(vStreamX.pop(), aie::mac(vBase, fPlus128(vQ), vFactor).template to_vector<bfp16ebs8>(),
                            vSum);
      }
    } else {
      // the fifth bits of the 4 groups of the sub-block, 8 bytes each: bit t of byte j is the one of lane 8j + t, so
      // the two words of a group are a mask of its 64 lanes
      const uint32_t *__restrict vFifth = reinterpret_cast<const uint32_t *>(pFifth + vSub * 32);
#pragma clang loop unroll(full)
      for (unsigned vGroup = 0; vGroup < 4; ++vGroup) {
        const aie::vector<uint8, 64> vLow =
            aie::load_v<64>(reinterpret_cast<const uint4 *>(vValues + (vSub * 4 + vGroup) * 32)).unpack();
        const aie::mask<64> vFive = aie::mask<64>::from_uint32(vFifth[2 * vGroup], vFifth[2 * vGroup + 1]);
        const aie::vector<uint8, 64> vQ = aie::select(vLow, aie::bit_or(vLow, v16), vFive);
        vSum = mac_8x8_8x8T(vStreamX.pop(), aie::mac(vBase, fPlus128(vQ), vFactor).template to_vector<bfp16ebs8>(),
                            vSum);
      }
    }
  }
  aie::store_v(pSums, vSum.template to_vector<float>());
}

// The factors of the 16 sub-blocks of a piece of Q6_K, into pStage: factor s is d * scale of the sub-block s, rounded
// to bf16, in the 8 lanes of every output. The scale is a signed byte: with its low 7 bits as l, d * scale is
// d * (128 + l) - 128 * d when it is positive and d * (128 + l) - 256 * d when it is negative, products of two bf16
// that are exact in 32 bits.
void fPrepareQ6K(const uint8_t *__restrict pPiece, uint8_t *__restrict pStage) {
  const aie::vector<bfloat16, 64> v128 = aie::broadcast<bfloat16, 64>(bfloat16(128.0f));
  const aie::vector<bfloat16, 64> v256 = aie::broadcast<bfloat16, 64>(bfloat16(256.0f));
  const aie::vector<uint8, 64> vLow7 = aie::broadcast<uint8, 64>(uint8(0x7f));
  // d of the 8 outputs in lane 8s + j, for 8 sub-blocks at a time
  const aie::vector<uint16, 16> vD8 = aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece));
  const aie::vector<uint16, 16> vD8Twice = aie::concat(vD8.extract<8>(0), vD8.extract<8>(0));
  const aie::vector<uint16, 32> vD32 = aie::concat(vD8Twice, vD8Twice);
  const aie::vector<bfloat16, 64> vD = aie::concat(vD32, vD32).template cast_to<bfloat16>();
  bfloat16 *__restrict vFactors = reinterpret_cast<bfloat16 *>(pStage + cFactors);
  for (unsigned vHalf = 0; vHalf < 2; ++vHalf) {
    // the scales of 8 sub-blocks of the 8 outputs, lane 8s + j
    const aie::vector<uint8, 64> vScales = aie::concat(aie::load_v<32>(pPiece + cScalesQ6K + vHalf * 64),
                                                       aie::load_v<32>(pPiece + cScalesQ6K + vHalf * 64 + 32));
    const aie::vector<bfloat16, 64> vC = aie::select(v128, v256, aie::ge(vScales, uint8(128)));
    const aie::vector<bfloat16, 64> vAll =
        aie::msc(aie::mul(vD, fPlus128(aie::bit_and(vScales, vLow7))), vD, vC).template to_vector<bfloat16>();
    for (unsigned vSub = 0; vSub < 8; ++vSub) {
      aie::store_v(vFactors + (vHalf * 8 + vSub) * cLanes, fRepeat8(aie::shuffle_down(vAll, vSub * 8)));
    }
  }
}

void fPieceQ6K(const uint8_t *__restrict pPiece, float *__restrict pSums, uint8_t *__restrict pStage, bool pFirst) {
  fPrepareQ6K(pPiece, pStage);
  const bfloat16 *__restrict vFactors = reinterpret_cast<const bfloat16 *>(pStage + cFactors);
  const aie::vector<bfloat16, 64> vMinus160 = aie::broadcast<bfloat16, 64>(bfloat16(-160.0f));
  const aie::vector<uint8, 64> vLow2 = aie::broadcast<uint8, 64>(uint8(3));
  Xdna2Sums vSum = fLoadSums(pSums, pFirst);
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(pStage));
  for (unsigned vSub = 0; vSub < 16; ++vSub) {
    const aie::vector<bfloat16, 64> vFactor = aie::load_v<cLanes>(vFactors + vSub * cLanes);
    // (q - 32) * factor is (128 + q) * factor - 160 * factor
    const Xdna2Sums vBase = aie::mul(vFactor, vMinus160);
    // the top 2 bits of the two groups: lanes 0 to 31 the fields of the first group, 32 to 63 those of the second
    const aie::vector<uint8, 64> vTop =
        aie::load_v<64>(reinterpret_cast<const uint4 *>(pPiece + cTopQ6K + vSub * 32)).unpack();
    const aie::vector<uint8, 64> vTopLow = aie::bit_and(vTop, vLow2);
    const aie::vector<uint8, 64> vTopHigh = aie::downshift(vTop, 2);
#pragma clang loop unroll(full)
    for (unsigned vGroup = 0; vGroup < 2; ++vGroup) {
      const aie::vector<uint8, 64> vLow =
          aie::load_v<64>(reinterpret_cast<const uint4 *>(pPiece + cValuesQ6K + (vSub * 2 + vGroup) * 32)).unpack();
      const aie::vector<uint8, 64> vHigh = aie::concat(vTopLow.extract<32>(vGroup), vTopHigh.extract<32>(vGroup));
      const aie::vector<uint8, 64> vQ = aie::bit_or(vLow, aie::upshift(vHigh, 4));
      vSum = mac_8x8_8x8T(vStreamX.pop(), aie::mac(vBase, fPlus128(vQ), vFactor).template to_vector<bfp16ebs8>(),
                          vSum);
    }
  }
  aie::store_v(pSums, vSum.template to_vector<float>());
}

void fPieceQ8_0(const uint8_t *__restrict pPiece, float *__restrict pSums, uint8_t *__restrict pStage, bool pFirst) {
  const aie::vector<bfloat16, 64> v128 = aie::broadcast<bfloat16, 64>(bfloat16(128.0f));
  const aie::vector<bfloat16, 64> v256 = aie::broadcast<bfloat16, 64>(bfloat16(256.0f));
  const aie::vector<uint8, 64> vLow7 = aie::broadcast<uint8, 64>(uint8(0x7f));
  Xdna2Sums vSum = fLoadSums(pSums, pFirst);
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(pStage));
  for (unsigned vSub = 0; vSub < 8; ++vSub) {
    const aie::vector<uint16, 32> vPairD =
        aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece + (vSub / 2) * 32)),
                    aie::zeros<uint16, 16>());
    const aie::vector<bfloat16, 64> vFactor =
        fRepeat8From16(vSub % 2 == 0 ? vPairD : aie::shuffle_down(vPairD, 8));
    const uint8_t *__restrict vSubValues = pPiece + cValuesQ8_0 + vSub * 256;
#pragma clang loop unroll(full)
    for (unsigned vGroup = 0; vGroup < 4; ++vGroup) {
      const uint8_t *__restrict vAt = vSubValues + vGroup * 64;
      const aie::vector<uint8, 64> vU = aie::concat(aie::load_v<32>(vAt), aie::load_v<32>(vAt + 32));
      const aie::vector<bfloat16, 64> vC = aie::select(v128, v256, aie::ge(vU, uint8(128)));
      const auto vW =
          aie::msc(aie::mul(fPlus128(aie::bit_and(vU, vLow7)), vFactor), vC, vFactor).template to_vector<bfp16ebs8>();
      vSum = mac_8x8_8x8T(vStreamX.pop(), vW, vSum);
    }
  }
  aie::store_v(pSums, vSum.template to_vector<float>());
}

// The piece pIndex of the block, at pPiece: tile pIndex / G, group of outputs pIndex % G. A function of its own, as
// version 5 had it (two places called it): inline in one place it took 500 bytes more.
__attribute__((noinline)) void fPiece(const uint8_t *__restrict pX, const uint8_t *__restrict pPiece,
                                      float *__restrict pY, uint8_t *__restrict pStage, unsigned pRows,
                                      int32_t pFormat, unsigned pIndex) {
#if defined(XDNA2_CHAIN_NO_COMPUTE)
  // only the path of the data: the pieces are counted, not computed
  (void) pX;
  (void) pPiece;
  (void) pY;
  (void) pStage;
  (void) pRows;
  (void) pFormat;
  (void) pIndex;
  return;
#endif
  const unsigned vTile = pIndex / cGroupsOut;
  const unsigned vGroup = pIndex % cGroupsOut;
  if (vGroup == 0) {
    fStage(pX, pStage, pRows, vTile * cTileGroups);
  }
  float *__restrict vSums = pY + vGroup * cLanes;
  const bool vFirst = vTile == 0;
  if (pFormat == cFormatQ4K) {
    fPieceK(pPiece, nullptr, vSums, pStage, vFirst);
  } else if (pFormat == cFormatQ5K) {
    fPieceK(pPiece, pPiece + cFifthQ5K, vSums, pStage, vFirst);
  } else if (pFormat == cFormatQ6K) {
    fPieceQ6K(pPiece, vSums, pStage, vFirst);
  } else if (pFormat == cFormatQ8_0) {
    fPieceQ8_0(pPiece, vSums, pStage, vFirst);
  } else {
    fPieceBfp16(pPiece, vSums, pStage, vFirst);
  }
}

// the counters of the prologue and the epilogue, after those of the pieces, and the floats of the norm
constexpr unsigned cPrologueCounter = cStageBytes + 12;  // the next object of the prologue
constexpr unsigned cBlockCounter = cStageBytes + 16;     // the next block of the phase, for the epilogue
constexpr unsigned cSquares = cStageBytes + 20;          // the sum of the squares of h of the core (float)
constexpr unsigned cScale = cStageBytes + 24;            // the scale of the norm (float)
constexpr unsigned cNormed = cStageBytes + 28;           // whether the sums of the phase take that scale
constexpr unsigned cPhaseCounter = cStageBytes + 32;     // the phase that runs
constexpr unsigned cSegmentCounter = cStageBytes + 36;   // the blocks of the segment that have ended (gemv)
constexpr unsigned cRowCounter = cStageBytes + 40;       // the next row of the block to emit (finish sets it to 0)
constexpr unsigned cChunk = 1152;                        // floats of an object of a core

// The recurrence of a layer of linear attention (version 8): the gated delta net of ggml (ggml_gated_delta_net with
// heads of 128 and one gate per head, as Qwen3.5 has), one head of values per core. The state of a head is 128 rows of
// 128 floats (row j is column j of the state of the model: ggml keeps it transposed), which come 9 to an object of a
// core (4608 bytes), 15 objects a head, the last one with 2 rows and padding; every row evolves alone.
constexpr unsigned cGdnDim = 128;                                                     // values of a head
constexpr unsigned cGdnRowBytes = cGdnDim * 4;                                        // bytes of a row of the state
constexpr unsigned cGdnRowsObject = cObjectBytes / cGdnRowBytes;                      // 9
constexpr unsigned cGdnCallsHead = (cGdnDim + cGdnRowsObject - 1) / cGdnRowsObject * cGdnRowsObject;  // 135
// The inputs of a head in X, in floats: q, k, beta v, the decay e^g of the gate g and beta e^g (the host computes them:
// an exponential of 32-bit floats took 1.1 KB of program, as the core has no scalar floating point). They come in an
// object of the prologue of the phase for every head of the core, which the prologue of X in bfp16 (mode 0) copies as
// it is. After them, the parts of k, q and e^g in bf16 that the first row of a head leaves for the others, and the
// parts of the row that runs.
constexpr unsigned cGdnQ = 0;
constexpr unsigned cGdnK = 128;
constexpr unsigned cGdnBetaV = 256;
constexpr unsigned cGdnDecay = 384;
constexpr unsigned cGdnBetaDecay = 385;
constexpr unsigned cGdnPartsK = 400;     // k in three parts of bf16, 128 each (192 floats)
constexpr unsigned cGdnPartsQ = 592;     // q in two (128 floats)
constexpr unsigned cGdnPartsDecay = 720; // e^g in three, 16 lanes each (24 floats)
constexpr unsigned cGdnPartsRow = 752;   // the row in three (192 floats)
constexpr unsigned cGdnInputs = cObjectBytes / 4;        // floats of X of a head: an object of a core
// its counters in pStage
constexpr unsigned cGdnCallCounter = cStageBytes + 44;   // the next row of the head (0 to 134; from 128 on, padding)
constexpr unsigned cGdnHeadCounter = cStageBytes + 48;   // the head of the core that runs
constexpr unsigned cGdnPartCounter = cStageBytes + 52;   // core 0: the next object of Y of a row
constexpr unsigned cGdnRowAt = cStageBytes + 56;         // core 0: where that row is in its object (bytes)
constexpr unsigned cGdnNextAt = cStageBytes + 60;        // where the next row is in its object (bytes)

// Version 9: a phase of a layer of linear attention (pParams[phase][26] = 1), as qwen35.cpp computes it when
// generating (build_layer_attn_linear): the convolution of the channels of q, k and v with the 3 old states of every
// channel (ggml_ssm_conv), its SiLU, the L2 norms of q and k (build_gdn_l2_norm: x / sqrt(sum of x^2 + eps)) before
// the recurrence, and after it the gated norm of every head (build_norm_gated: rms_norm(o) times the weights times
// silu(z)). The pair of stream s takes the head of keys s and the value heads s (core 0) and s + 16 (core 1), which
// share it. Every product is of bf16 into 32-bit float sums: one of 32-bit floats takes some 25 bundles of
// instructions to emulate, and with them the program of neither core fitted. What they feed is bf16 at most anyway:
// the recurrence of a prompt takes q, k and v in bf16 and the model does not notice, and the gated norm goes to a
// product in bfp16. Core 0 has no room for a SiLU either, so core 1 does all of them: it convolves the 512 channels of
// the pair, takes the SiLU and the L2 norms of q and k, computes the gate of both heads (the weights of the norm times
// silu(z)), and sends q, k and v of head s and its gate to core 0 over the cascade; both then put the inputs of the
// recurrence of their head in X[0] (q, k, beta v, e^g and beta e^g), where fGdnRow reads them, and fGdnRow ends the
// head with o sqrt(128) / sqrt(sum of o^2 + E) times the gate (with the outputs o of fGdnRow, which lack the
// 1 / sqrt(128) of ggml, that is rms_norm(o) with eps if E is 16384 eps, and the weights carry the sqrt(128)).
// The objects of the prologue (mode 3) go to the objects 1 to 3 of X of every core (offsets in bytes):
//   X[1]  from the host, every token: the floats beta [0], e^g [4], beta e^g [8], E [12] and the eps of the L2 norms
//         [16] of the head of the core; core 1, in bf16, x of the 512 channels of the pair [64, 1088) (q and k of the
//         head of keys, v of head s and v of head s + 16) and their 3 old states of the convolution by tap, [3][512]
//         [1088, 4160) (the host lays them out so, and writes the new ones, [s1, s2, x], to the cache)
//   X[2]  from the host, every token: core 1, in bf16, z of head s + 16 [0, 256) and of head s [256, 512)
//   X[3]  from the host, the same every token: core 1, in bf16, the weights of the gated norm times sqrt(128) [0,
//         256) and the 4 weights of the convolution of the channels by tap, [4][512] [256, 4352)
// Core 1 keeps the convolution after the SiLU in X[4] (floats), and both cores the gate of their head after it (core 1\n// then that of head s, and the convolution before the SiLU in bf16).
constexpr unsigned cLaItem = 26;
constexpr unsigned cLaChannels = 512;
constexpr unsigned cLaToken = cObjectBytes;           // X[1], in bytes; its floats:
constexpr unsigned cLaBeta = 0;
constexpr unsigned cLaDecay = 1;
constexpr unsigned cLaBetaDecay = 2;
constexpr unsigned cLaEpsGated = 3;
constexpr unsigned cLaEpsL2 = 4;
constexpr unsigned cLaInput = cLaToken + 64;          // its bf16
constexpr unsigned cLaStates = cLaToken + 1088;
constexpr unsigned cLaZ = 2 * cObjectBytes;           // X[2]
constexpr unsigned cLaFixed = 3 * cObjectBytes;       // X[3]
constexpr unsigned cLaNorm = cLaFixed;
constexpr unsigned cLaWeights = cLaFixed + 256;
constexpr unsigned cLaConv = 4 * cChunk;              // X[4], in floats: the convolution after the SiLU
constexpr unsigned cLaGate = cLaConv + cLaChannels;   // the gate of the head of the core, then (core 1) that of head s
constexpr unsigned cLaSums = (cLaGate + 2 * cGdnDim) * 4;  // core 1: the convolution before the SiLU, in bf16 (bytes)
static_assert(cLaStates + 3 * cLaChannels * 2 <= 2 * cObjectBytes && cLaSums + cLaChannels * 2 <= 7 * cObjectBytes &&
                  cLaZ + 2 * cGdnDim * 2 <= cLaFixed && cLaWeights + 4 * cLaChannels * 2 <= 4 * cObjectBytes &&
                  cLaGate + 2 * cGdnDim <= 7 * cChunk,
              "the objects of a phase of linear attention fit in X");

// where the parameters of a phase start, and their items
constexpr unsigned cParamsPhase = 0;
constexpr unsigned cParamsPerPhase = 32;
constexpr unsigned cPrologueMode = 0;
constexpr unsigned cRowsItem = 2;
constexpr unsigned cEpilogueMode = 4;
constexpr unsigned cResidualItem = 5;
constexpr unsigned cEpsItem = 6;
constexpr unsigned cValuesItem = 7;
constexpr unsigned cInverseItem = 25;                    // 1 / K, for the mean of the RMS norm (a float)
constexpr unsigned cParamsWords = 160;
constexpr unsigned cParamsEnded = 4 * cParamsPerPhase + 1;  // whether the last call of the gemv ended a block

// the parts of pChain
constexpr unsigned cChainResidual = 0;
constexpr unsigned cChainH = 320;
constexpr unsigned cChainGate = 640;

inline float &fFloatAt(uint8_t *pStage, unsigned pOffset) {
  return *reinterpret_cast<float *>(pStage + pOffset);
}

using Xdna2Acc16 = aie::accum<accfloat, 16>;
using Xdna2Bf16x16 = aie::vector<bfloat16, 16>;

// The 32-bit floats of pAcc as three parts of bf16 that add up to them (to some 24 bits): the high part, rounded, and
// again for what is left, which the core takes out of the accumulator exactly as a product by 1.0 (as the emulation of
// the compiler does it).
inline void fSplit3(Xdna2Acc16 pAcc, Xdna2Bf16x16 &pHigh, Xdna2Bf16x16 &pMiddle, Xdna2Bf16x16 &pLow) {
  const Xdna2Bf16x16 vOne = aie::broadcast<bfloat16, 16>(bfloat16(1.0f));
  pHigh = pAcc.template to_vector<bfloat16>();
  pAcc = aie::msc(pAcc, pHigh, vOne);
  pMiddle = pAcc.template to_vector<bfloat16>();
  pAcc = aie::msc(pAcc, pMiddle, vOne);
  pLow = pAcc.template to_vector<bfloat16>();
}

// pAcc + a b for 32-bit floats a and b given as their three parts of bf16: the six products of parts whose weight is
// 2^-16 or more of the product (the other three are under 2^-24 of it)
inline Xdna2Acc16 fMac3(Xdna2Acc16 pAcc, const Xdna2Bf16x16 &pA0, const Xdna2Bf16x16 &pA1, const Xdna2Bf16x16 &pA2,
                        const Xdna2Bf16x16 &pB0, const Xdna2Bf16x16 &pB1, const Xdna2Bf16x16 &pB2) {
  pAcc = aie::mac(pAcc, pA0, pB0);
  pAcc = aie::mac(pAcc, pA0, pB1);
  pAcc = aie::mac(pAcc, pA1, pB0);
  pAcc = aie::mac(pAcc, pA0, pB2);
  pAcc = aie::mac(pAcc, pA1, pB1);
  return aie::mac(pAcc, pA2, pB0);
}

// version 9: 16 or 32 floats rounded to bf16
inline Xdna2Bf16x16 fToBf16(const aie::vector<float, 16> &pValues) {
  Xdna2Acc16 vAcc;
  vAcc.from_vector(pValues);
  return vAcc.template to_vector<bfloat16>();
}

inline aie::vector<bfloat16, 32> fToBf16(const aie::vector<float, 32> &pValues) {
  aie::accum<accfloat, 32> vAcc;
  vAcc.from_vector(pValues);
  return vAcc.template to_vector<bfloat16>();
}

// version 9: 1 / sqrt(the sum of the squares of the 128 floats at pAt (rounded to bf16) + pEps), in all the lanes of a
// vector of bf16 (the sum and pEps added in the lanes of a vector: the core has no scalar floating point)
inline Xdna2Bf16x16 fInverseNorm(const float *__restrict pAt, float pEps) {
  Xdna2Acc16 vSum = aie::zeros<accfloat, 16>();
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    const Xdna2Bf16x16 vValues = fToBf16(aie::load_v<16>(pAt + i));
    vSum = aie::mac(vSum, vValues, vValues);
  }
  const aie::vector<float, 16> vTotal =
      aie::add(aie::broadcast<float, 16>(aie::reduce_add(vSum.template to_vector<float>())), pEps);
  return fToBf16(aie::broadcast<float, 16>(aie::invsqrt(vTotal[0])));
}

// version 9: the gated norm of the head of the core after the recurrence, over its 128 outputs in pSums: o times
// 1 / sqrt(sum of o^2 + E) times the gate of the head (see cLaItem)
inline void fGatedNorm(float *__restrict pSums, const uint8_t *__restrict pX) {
  const float *__restrict vGate = reinterpret_cast<const float *>(pX) + cLaGate;
  const Xdna2Bf16x16 vScale = fInverseNorm(pSums, reinterpret_cast<const float *>(pX + cLaToken)[cLaEpsGated]);
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    const aie::vector<float, 16> vFactor = aie::mul(fToBf16(aie::load_v<16>(vGate + i)), vScale).template to_vector<float>();
    aie::store_v(pSums + i,
                 aie::mul(fToBf16(aie::load_v<16>(pSums + i)), fToBf16(vFactor)).template to_vector<float>());
  }
}

// The next row of the recurrence of the head of the core, in its object pW: what ggml_gated_delta_net does to row j of
// a head for a token (ops.cpp, ggml_gdn_head_128): the row times the decay e^g, delta = (v[j] - row . k) beta, the row
// plus delta k, and the output j = row . q, kept in pSums[j] without the 1 / sqrt(128) of ggml (whoever reads it
// applies it, or the RMS norm that follows makes it go away). pCore 0 leaves the new row in its object, from where core
// 0 sends it to Y; pCore 1 sends it to core 0 over the cascade, and its outputs after the last row. The rows of the
// padding of the last object of a head (from 128 on) are skipped. Returns the objects of Y core 0 sends for the row:
// 8 (its row and that of core 1), 16 after the last row of a head (and both outputs), 0 for padding.
//
// The core has no products of 32-bit floats: the compiler emulates one with 9 products of bf16 and the splitting of
// both operands into three parts (some 25 bundles of instructions), and a row took about 3,000 cycles. Here every row
// is split once, k, q and e^g once a head (by its first row), and a product of two of them is the 6 products of fMac3:
// the row is e^g row + delta k in one accumulator, and delta = beta v[j] - beta e^g (row . k) with the row before its
// decay (beta v and beta e^g come from the host), its one product of 32-bit floats emulated by the compiler. The output
// takes two parts of the new row by two of q (some 16 bits): it goes to an RMS norm and then, rounded to bfp16, to a
// product.
template <int pCore>
inline int32_t fGdnRow(uint8_t *__restrict pW, uint8_t *__restrict pX, float *__restrict pSums,
                       uint8_t *__restrict pStage, const int32_t *__restrict pParams) {
  int32_t &vCall = fCounter(pStage, cGdnCallCounter);
  int32_t &vNextAt = fCounter(pStage, cGdnNextAt);
  const unsigned vRow = unsigned(vCall);
  const unsigned vAt = unsigned(vNextAt);
  vCall = vRow + 1 == cGdnCallsHead ? 0 : int32_t(vRow + 1);
  vNextAt = vAt + cGdnRowBytes == cObjectBytes ? 0 : int32_t(vAt + cGdnRowBytes);
  if (vRow >= cGdnDim) {
    return 0;
  }
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  int32_t &vHead = fCounter(pStage, cGdnHeadCounter);
  float *__restrict vIn = reinterpret_cast<float *>(pX) + cGdnInputs * unsigned(vHead);
  bfloat16 *__restrict vPartsK = reinterpret_cast<bfloat16 *>(vIn + cGdnPartsK);
  bfloat16 *__restrict vPartsQ = reinterpret_cast<bfloat16 *>(vIn + cGdnPartsQ);
  bfloat16 *__restrict vPartsDecay = reinterpret_cast<bfloat16 *>(vIn + cGdnPartsDecay);
  bfloat16 *__restrict vPartsRow = reinterpret_cast<bfloat16 *>(vIn + cGdnPartsRow);
  float *__restrict vState = reinterpret_cast<float *>(pW + vAt);
  Xdna2Bf16x16 vA0, vA1, vA2;
  if (vRow == 0) {
    // the first row of the head: the parts of k, q and e^g for all its rows
    for (unsigned i = 0; i < cGdnDim; i += 16) {
      Xdna2Acc16 vValues;
      vValues.from_vector(aie::load_v<16>(vIn + cGdnK + i));
      fSplit3(vValues, vA0, vA1, vA2);
      aie::store_v(vPartsK + i, vA0);
      aie::store_v(vPartsK + cGdnDim + i, vA1);
      aie::store_v(vPartsK + 2 * cGdnDim + i, vA2);
      vValues.from_vector(aie::load_v<16>(vIn + cGdnQ + i));
      fSplit3(vValues, vA0, vA1, vA2);
      aie::store_v(vPartsQ + i, vA0);
      aie::store_v(vPartsQ + cGdnDim + i, vA1);
    }
    Xdna2Acc16 vDecay;
    vDecay.from_vector(aie::broadcast<float, 16>(vIn[cGdnDecay]));
    fSplit3(vDecay, vA0, vA1, vA2);
    aie::store_v(vPartsDecay, vA0);
    aie::store_v(vPartsDecay + 16, vA1);
    aie::store_v(vPartsDecay + 32, vA2);
  }
#if defined(XDNA2_CHAIN_NO_COMPUTE)
  // only the path of the data: core 1 sends a row of zeros, and the outputs after the last row
  (void) vIn;
  (void) vPartsK;
  (void) vPartsQ;
  (void) vPartsDecay;
  (void) vPartsRow;
  (void) vA0;
  (void) vA1;
  (void) vA2;
  if (pCore != 0) {
    for (unsigned i = 0; i < cGdnDim; i += 16) {
      put_mcd(v16float(aie::zeros<float, 16>()));
    }
  }
#else
  // The parts of the row, and its dot product with k (before the decay). (Three accumulators for its six products, so
  // that the chains overlap, were slower: the core has too few registers of accumulators, and they went to the stack.)
  Xdna2Acc16 vDot = aie::zeros<accfloat, 16>();
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    Xdna2Acc16 vValues;
    vValues.from_vector(aie::load_v<16>(vState + i));
    fSplit3(vValues, vA0, vA1, vA2);
    aie::store_v(vPartsRow + i, vA0);
    aie::store_v(vPartsRow + cGdnDim + i, vA1);
    aie::store_v(vPartsRow + 2 * cGdnDim + i, vA2);
    vDot = fMac3(vDot, vA0, vA1, vA2, aie::load_v<16>(vPartsK + i), aie::load_v<16>(vPartsK + cGdnDim + i),
                 aie::load_v<16>(vPartsK + 2 * cGdnDim + i));
  }
  // delta = beta v[j] - beta e^g (row . k), the same in every lane, in three parts
  const aie::vector<float, 16> vDelta =
      aie::sub(aie::broadcast<float, 16>(vIn[cGdnBetaV + vRow]),
               aie::mul(aie::broadcast<float, 16>(aie::reduce_add(vDot.template to_vector<float>())),
                        vIn[cGdnBetaDecay])
                   .template to_vector<float>());
  Xdna2Acc16 vDeltaAcc;
  vDeltaAcc.from_vector(vDelta);
  Xdna2Bf16x16 vD0, vD1, vD2;
  fSplit3(vDeltaAcc, vD0, vD1, vD2);
  const Xdna2Bf16x16 vE0 = aie::load_v<16>(vPartsDecay);
  const Xdna2Bf16x16 vE1 = aie::load_v<16>(vPartsDecay + 16);
  const Xdna2Bf16x16 vE2 = aie::load_v<16>(vPartsDecay + 32);
  const Xdna2Bf16x16 vOne = aie::broadcast<bfloat16, 16>(bfloat16(1.0f));
  // the new row, e^g row + delta k, and its dot product with q
  Xdna2Acc16 vOut = aie::zeros<accfloat, 16>();
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    Xdna2Acc16 vNew = fMac3(aie::zeros<accfloat, 16>(), aie::load_v<16>(vPartsRow + i),
                            aie::load_v<16>(vPartsRow + cGdnDim + i), aie::load_v<16>(vPartsRow + 2 * cGdnDim + i), vE0,
                            vE1, vE2);
    vNew = fMac3(vNew, aie::load_v<16>(vPartsK + i), aie::load_v<16>(vPartsK + cGdnDim + i),
                 aie::load_v<16>(vPartsK + 2 * cGdnDim + i), vD0, vD1, vD2);
    const aie::vector<float, 16> vS = vNew.template to_vector<float>();
    if (pCore == 0) {
      aie::store_v(vState + i, vS);
    } else {
      put_mcd(v16float(vS));
    }
    const Xdna2Bf16x16 vS0 = vNew.template to_vector<bfloat16>();
    const Xdna2Bf16x16 vS1 = aie::msc(vNew, vS0, vOne).template to_vector<bfloat16>();
    const Xdna2Bf16x16 vQ0 = aie::load_v<16>(vPartsQ + i);
    vOut = aie::mac(vOut, vS0, vQ0);
    vOut = aie::mac(vOut, vS0, aie::load_v<16>(vPartsQ + cGdnDim + i));
    vOut = aie::mac(vOut, vS1, vQ0);
  }
  pSums[vRow] = aie::reduce_add(vOut.template to_vector<float>());
#endif
  const bool vLast = vRow + 1 == cGdnDim;
  int32_t vObjects = 8;
  if (pCore == 0) {
    fCounter(pStage, cGdnPartCounter) = 0;
    fCounter(pStage, cGdnRowAt) = int32_t(vAt);
  }
  if (vLast) {
#if !defined(XDNA2_CHAIN_NO_COMPUTE)
    // version 9: in a phase of linear attention, the gated norm of the head before its outputs leave
    if (pParams[cParamsPhase + cParamsPerPhase * unsigned(fCounter(pStage, cPhaseCounter)) + cLaItem] != 0) {
      fGatedNorm(pSums, pX);
    }
#else
    (void) pParams;
#endif
    if (pCore != 0) {
      for (unsigned i = 0; i < cGdnDim; i += 16) {
        put_mcd(v16float(aie::load_v<16>(pSums + i)));
      }
    }
    vObjects = 16;
    vHead++;
  }
  aie::set_rounding(vSaved);
  return vObjects;
}

// 64 floats of one row, rounded to the nearest (even) as 8 blocks of bfp16, into the slots of 12 bytes of X for one
// row (9 bytes of a block, padded to a multiple of 4), from the slot pGroup on
__attribute__((noinline)) void fRoundRow(const float *__restrict pFrom, uint8_t *__restrict pX, unsigned pGroup,
                                         uint8_t *__restrict pTemp) {
  aie::accum<accfloat, cLanes> vAcc;
  vAcc.from_vector(aie::load_v<cLanes>(pFrom));
  aie::block_vector_output_buffer_stream<bfp16ebs8, cLanes> vOut(reinterpret_cast<bfp16ebs8 *>(pTemp));
  vOut << vAcc.template to_vector<bfp16ebs8>();
  for (unsigned j = 0; j < cBlock; ++j) {
    uint8_t *__restrict vTo = pX + (pGroup + j) * 12;
    for (unsigned b = 0; b < cBlockBytes; ++b) {
      vTo[b] = pTemp[j * cBlockBytes + b];
    }
  }
}

} // namespace

extern "C" {

#if defined(XDNA2_CHAIN_PARAMS)

void xdna2_chain_params(uint8_t *__restrict pObject, int32_t *__restrict pParams, uint8_t *__restrict pStage) {
  const int32_t *__restrict vFrom = reinterpret_cast<const int32_t *>(pObject);
  for (unsigned i = 0; i < cParamsWords; i += 8) {
    aie::store_v(pParams + i, aie::load_v<8>(vFrom + i));
  }
  fCounter(pStage, cPhaseCounter) = -1;
}

#elif defined(XDNA2_CHAIN_PHASE)

void xdna2_chain_phase(uint8_t *__restrict pStage, int32_t *__restrict pParams) {
  pParams[cParamsEnded] = 0;
  fCounter(pStage, cPhaseCounter)++;
  fCounter(pStage, cPrologueCounter) = -1;
  fCounter(pStage, cPieceCounter) = 0;
  fCounter(pStage, cCarryCounter) = 0;
  fCounter(pStage, cObjectCounter) = 0;
  fCounter(pStage, cSegmentCounter) = 0;
  fCounter(pStage, cBlockCounter) = 0;
  fFloatAt(pStage, cSquares) = 0.0f;
  fCounter(pStage, cNormed) = 0;
  fCounter(pStage, cGdnCallCounter) = 0;
  fCounter(pStage, cGdnHeadCounter) = 0;
  fCounter(pStage, cGdnNextAt) = 0;
}

#elif defined(XDNA2_CHAIN_PROLOGUE)

void xdna2_chain_prologue(uint8_t *__restrict pObject, uint8_t *__restrict pX, uint8_t *__restrict pStage,
                          float *__restrict pChain, int32_t *__restrict pParams, int32_t pCore) {
  const unsigned           vPhaseNow = unsigned(fCounter(pStage, cPhaseCounter));
  const int32_t *__restrict vPhase   = pParams + cParamsPhase + cParamsPerPhase * vPhaseNow;
  int32_t &vIndex = fCounter(pStage, cPrologueCounter);
  vIndex++;
  const int32_t vMode = vPhase[cPrologueMode];
  const unsigned vResidual = unsigned(vPhase[cResidualItem]);
  const unsigned vValues = unsigned(vPhase[cValuesItem]);  // values of K of the core
  const unsigned vAt = unsigned(vIndex);
  if (vAt < vResidual) {
    // the residual of the stream: core 0 keeps it (320 floats at most)
    if (pCore == 0 && vAt == 0) {
      const float *__restrict vFrom = reinterpret_cast<const float *>(pObject);
      for (unsigned i = 0; i < cChainH - cChainResidual; i += 16) {
        aie::store_v(pChain + cChainResidual + i, aie::load_v<16>(vFrom + i));
      }
    }
    return;
  }
  const unsigned vObject = vAt - vResidual;
  if (vMode == 0 || vMode == 3) {
    // X in bfp16 from the host, as the overlay keeps it; or (version 9) the objects of a phase of linear attention,
    // from the second object of X on (the first one takes the inputs of the recurrence)
    const int32_t *__restrict vFrom = reinterpret_cast<const int32_t *>(pObject);
    int32_t *__restrict vTo = reinterpret_cast<int32_t *>(pX + (vObject + (vMode == 3 ? 1 : 0)) * cObjectBytes);
    for (unsigned i = 0; i < cObjectBytes / 32; ++i) {
      aie::store_v(vTo + 8 * i, aie::load_v<8>(vFrom + 8 * i));
    }
    return;
  }
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  uint8_t *__restrict vTemp = pStage + cCarry;  // the carry of the pieces is free during the prologue
  const unsigned vChunks = (vValues + cChunk - 1) / cChunk;
  const float *__restrict vFloats = reinterpret_cast<const float *>(pObject);
  if (vMode == 1) {
    // X in floats, one row: every 64 values to 8 blocks of bfp16
    const unsigned vFirstValue = vObject * cChunk;
    for (unsigned i = 0; i < cChunk && vFirstValue + i < vValues; i += cLanes) {
      fRoundRow(vFloats + i, pX, (vFirstValue + i) / cBlock, vTemp);
    }
    aie::set_rounding(vSaved);
    return;
  }
  // the RMS norm: first h (the floats are kept in X, and their squares added), then its weights (X becomes
  // h * weight in bfp16, written over the floats it has read)
  if (vObject < vChunks) {
    const unsigned vFirstValue = vObject * cChunk;
    float *__restrict vKeep = reinterpret_cast<float *>(pX) + vFirstValue;
    aie::accum<accfloat, 16> vSum = aie::zeros<accfloat, 16>();
    for (unsigned i = 0; i < cChunk && vFirstValue + i < vValues; i += 16) {
      const aie::vector<float, 16> vH = aie::load_v<16>(vFloats + i);
      aie::store_v(vKeep + i, vH);
      vSum = aie::mac(vSum, vH, vH);
    }
    fFloatAt(pStage, cSquares) += aie::reduce_add(vSum.template to_vector<float>());
  } else {
    const unsigned vFirstValue = (vObject - vChunks) * cChunk;
    const float *__restrict vH = reinterpret_cast<const float *>(pX) + vFirstValue;
    alignas(64) float aProduct[cLanes];
    for (unsigned i = 0; i < cChunk && vFirstValue + i < vValues; i += cLanes) {
      aie::store_v(aProduct, aie::mul(aie::load_v<cLanes>(vH + i), aie::load_v<cLanes>(vFloats + i))
                                 .template to_vector<float>());
      fRoundRow(aProduct, pX, (vFirstValue + i) / cBlock, vTemp);
    }
    if (vObject + 1 == 2 * vChunks) {
      // the last object: the sums of squares of both cores, over the cascade from core 1 to core 0
      if (pCore == 1) {
        aie::vector<float, 16> vSend = aie::zeros<float, 16>();
        vSend[0] = fFloatAt(pStage, cSquares);
        put_mcd(v16float(vSend));
      } else {
        const aie::vector<float, 16> vGot = get_scd_v16float();
        float vEps;
        float vInverse;
        __builtin_memcpy(&vEps, &vPhase[cEpsItem], sizeof(vEps));
        __builtin_memcpy(&vInverse, &vPhase[cInverseItem], sizeof(vInverse));
        // the mean of the squares, as their sum times 1 / K from the host, in the lanes of a vector (version 8): a
        // division of scalar floats took the routines __divsf3 and __floatunsisf, 1.3 KB of the program of both cores
        const aie::vector<float, 16> vSum = aie::add(aie::broadcast<float, 16>(fFloatAt(pStage, cSquares)), vGot);
        const float vMean = aie::mul(vSum, vInverse).template to_vector<float>()[0];
        fFloatAt(pStage, cScale) = aie::invsqrt(vMean + vEps);
        fCounter(pStage, cNormed) = 1;
      }
    }
  }
  aie::set_rounding(vSaved);
}

#elif defined(XDNA2_CHAIN_GEMV)

void xdna2_chain_gemv(uint8_t *__restrict pX, uint8_t *__restrict pW, float *__restrict pSums,
                      uint8_t *__restrict pStage, int32_t *__restrict pParams, int32_t pRows, int32_t pFormat,
                      int32_t pTiles, int32_t pBlocks) {
  int32_t &vNext = fCounter(pStage, cPieceCounter);
  int32_t &vCarried = fCounter(pStage, cCarryCounter);
  int32_t &vResume = fCounter(pStage, cObjectCounter);
  int32_t &vEnded = fCounter(pStage, cSegmentCounter);
  pParams[cParamsEnded] = 0;
  if (vEnded >= pBlocks) {
    // the call after the last block of the segment: the rest of its object is padding, and the next segment starts
    // afresh
    vNext = 0;
    vCarried = 0;
    vResume = 0;
    vEnded = 0;
    return;
  }
  const unsigned vPieceBytes = fPieceBytes(pFormat);
  const unsigned vPieces = unsigned(pTiles) * cGroupsOut;
  const unsigned vRows = unsigned(pRows);
  unsigned vIndex = unsigned(vNext);
  unsigned vAt = unsigned(vResume);
  bool vBlockEnds = false;
  // the core starts in the rounding mode floor: the decoded values are rounded to bf16 and bfp16 to the nearest
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  uint8_t *__restrict vCarry = pStage + (pFormat == cFormatQ8_0 ? cBases : cCarry);
  // One place that computes a piece, so that the code of the pieces is there once (with a second one for the piece
  // carried over, the program of core 0 did not fit in its 16 KB).
  while (!vBlockEnds) {
    const uint8_t *vPiece = pW + vAt;  // not __restrict: it may point to the carry, written just before
    unsigned vNextAt = vAt + vPieceBytes;
    if (vAt == 0 && vCarried > 0) {
      // the piece that began in the last object: its first part is in the carry, the rest starts this object
      const unsigned vRest = vPieceBytes - unsigned(vCarried);
      fCopy(pW, vCarry + vCarried, vRest);
      vPiece = vCarry;
      vNextAt = vRest;
      vCarried = 0;
    } else if (vNextAt > cObjectBytes) {
      break;
    }
    fPiece(pX, vPiece, pSums, pStage, vRows, pFormat, vIndex);
    vIndex++;
    vAt = vNextAt;
    vBlockEnds = vIndex == vPieces;
  }
  if (vBlockEnds) {
    // the block is finished (core 0) or sent (core 1) before the next call goes on here, in the same object (a block
    // takes an object at least, so it cannot end at its start: vAt > 0)
    vNext = 0;
    vResume = int32_t(vAt);
    vEnded++;
    pParams[cParamsEnded] = 1;
  } else {
    // the object is over: a piece that does not fit goes on in the next one
    vNext = int32_t(vIndex);
    vResume = 0;
    if (vAt < cObjectBytes) {
      vCarried = int32_t(cObjectBytes - vAt);
      fCopy(pW + vAt, vCarry, cObjectBytes - vAt);
    }
  }
  aie::set_rounding(vSaved);
}

#elif defined(XDNA2_CHAIN_SEND)

void xdna2_chain_send(float *__restrict pSums, uint8_t *__restrict pStage) {
  (void) pStage;
  for (unsigned i = 0; i < cGroupsOut * cLanes; i += 16) {
    put_mcd(v16float(aie::load_v<16>(pSums + i)));
  }
}

#elif defined(XDNA2_CHAIN_FINISH)

void xdna2_chain_finish(float *__restrict pSums, uint8_t *__restrict pStage, float *__restrict pChain,
                        int32_t *__restrict pParams) {
  const unsigned           vPhaseNow = unsigned(fCounter(pStage, cPhaseCounter));
  const int32_t *__restrict vPhase   = pParams + cParamsPhase + cParamsPerPhase * vPhaseNow;
  const bool vNormed = fCounter(pStage, cNormed) != 0;
  const float vScale = fFloatAt(pStage, cScale);
  float *__restrict pY = pSums;  // the block is finished where it was summed, and then emitted row by row
  fCounter(pStage, cRowCounter) = 0;
  // the sums of core 0 and those of core 1, added, and the scale of the norm
  for (unsigned i = 0; i < cGroupsOut * cLanes; i += 16) {
    aie::vector<float, 16> vSum = aie::add(aie::load_v<16>(pSums + i), aie::vector<float, 16>(get_scd_v16float()));
    if (vNormed) {
      vSum = aie::mul(vSum, vScale).template to_vector<float>();
    }
    aie::store_v(pY + i, vSum);
  }
  const int32_t vMode = vPhase[cEpilogueMode];
  int32_t &vBlock = fCounter(pStage, cBlockCounter);
  if (vMode == 0) {
    vBlock++;
    return;
  }
  // one row: the 32 outputs of the block are row 0 of its 4 groups, pY[64 g + i]
  if (vMode == 1 || vMode == 2) {
    const float *__restrict vAdd = pChain + (vMode == 1 ? cChainResidual : cChainH) + 32 * unsigned(vBlock);
    float *__restrict vKeep = pChain + cChainH + 32 * unsigned(vBlock);
    for (unsigned g = 0; g < cGroupsOut; ++g) {
      const aie::vector<float, 8> vSum = aie::add(aie::load_v<8>(pY + cLanes * g), aie::load_v<8>(vAdd + 8 * g));
      aie::store_v(pY + cLanes * g, vSum);
      if (vMode == 1) {
        aie::store_v(vKeep + 8 * g, vSum);
      }
    }
    vBlock++;
    return;
  }
  // SwiGLU: the blocks of segment 0 (the gate) are kept, those of segment 1 (the up) become silu(gate) * up
  if (vBlock < vPhase[8]) {
    float *__restrict vGate = pChain + cChainGate + 32 * unsigned(vBlock);
    for (unsigned g = 0; g < cGroupsOut; ++g) {
      aie::store_v(vGate + 8 * g, aie::load_v<8>(pY + cLanes * g));
    }
    vBlock++;
    return;
  }
  const unsigned vUp = unsigned(vBlock) - unsigned(vPhase[8]);  // blocks of the gate came first
  const float *__restrict vGate = pChain + cChainGate + 32 * vUp;
  alignas(32) float aGate[32];
  alignas(32) float aUp[32];
  for (unsigned g = 0; g < cGroupsOut; ++g) {
    aie::store_v(aGate + 8 * g, aie::load_v<8>(vGate + 8 * g));
    aie::store_v(aUp + 8 * g, aie::load_v<8>(pY + cLanes * g));
  }
  // silu(x) = x / (1 + 2^(-x log2 e)), with an exponential of bf16 precision: what it feeds is rounded to bfp16
  const aie::vector<float, 32> vG = aie::load_v<32>(aGate);
  const aie::vector<float, 32> vMinusLog = aie::mul(vG, -1.4426950408889634f).template to_vector<float>();
  const aie::vector<float, 32> vExp =
      aie::mul(aie::exp2<bfloat16>(vMinusLog), bfloat16(1.0f)).template to_vector<float>();
  const aie::vector<float, 32> vDen = aie::add(vExp, 1.0f);
  const aie::vector<float, 32> vSilu = aie::mul(vG, aie::inv(vDen)).template to_vector<float>();
  aie::store_v(aUp, aie::mul(vSilu, aie::load_v<32>(aUp)).template to_vector<float>());
  for (unsigned g = 0; g < cGroupsOut; ++g) {
    aie::store_v(pY + cLanes * g, aie::load_v<8>(aUp + 8 * g));
  }
  vBlock++;
}

#elif defined(XDNA2_CHAIN_EMIT)

void xdna2_chain_emit(float *__restrict pY, float *__restrict pSums, uint8_t *__restrict pStage) {
  int32_t &vRow = fCounter(pStage, cRowCounter);
  // row r of the 4 groups of the block, [group][8 rows][8 outputs] in the sums, as [group][8 outputs]
  for (unsigned g = 0; g < cGroupsOut; ++g) {
    aie::store_v(pY + cBlock * g, aie::load_v<8>(pSums + cLanes * g + cBlock * unsigned(vRow)));
  }
  vRow++;
}

#elif defined(XDNA2_CHAIN_GDN0)

void xdna2_chain_gdn0(uint8_t *__restrict pW, uint8_t *__restrict pX, float *__restrict pSums,
                      uint8_t *__restrict pStage, int32_t *__restrict pParams) {
  pParams[cParamsEnded] = fGdnRow<0>(pW, pX, pSums, pStage, pParams);
}

#elif defined(XDNA2_CHAIN_GDN1)

void xdna2_chain_gdn1(uint8_t *__restrict pW, uint8_t *__restrict pX, float *__restrict pSums,
                      uint8_t *__restrict pStage, int32_t *__restrict pParams) {
  (void) fGdnRow<1>(pW, pX, pSums, pStage, pParams);
}

#elif defined(XDNA2_CHAIN_GDN_EMIT)

void xdna2_chain_gdn_emit(float *__restrict pY, uint8_t *__restrict pW, float *__restrict pSums,
                          uint8_t *__restrict pStage) {
  int32_t &vPart = fCounter(pStage, cGdnPartCounter);
  const unsigned vAt = unsigned(vPart);
  vPart = int32_t(vAt + 1);
  if ((vAt & 4) != 0) {
    // parts 4 to 7 (the row) and 12 to 15 (the outputs) of core 1, over the cascade
    aie::store_v(pY, aie::vector<float, 16>(get_scd_v16float()));
    aie::store_v(pY + 16, aie::vector<float, 16>(get_scd_v16float()));
    return;
  }
  // parts 0 to 3, the row of core 0 where it was updated; 8 to 11, its outputs
  const float *__restrict vFrom =
      vAt < 8 ? reinterpret_cast<const float *>(pW + fCounter(pStage, cGdnRowAt)) : pSums;
  const unsigned vOffset = 32 * (vAt & 3);
  aie::store_v(pY, aie::load_v<16>(vFrom + vOffset));
  aie::store_v(pY + 16, aie::load_v<16>(vFrom + vOffset + 16));
}

#elif defined(XDNA2_CHAIN_CONV)

}  // extern "C"

namespace {

// silu(x) = x / (1 + 2^(-x log2 e)) for pCount values of bf16 (a multiple of 32), into floats: -log2 e in two parts of
// bf16 (to some 16 bits), the exponential of bf16 precision as in the SwiGLU of xdna2_chain_finish; one place for the
// three calls of xdna2_chain_conv
__attribute__((noinline)) void fSilu(const bfloat16 *__restrict pIn, float *__restrict pOut, unsigned pCount) {
  for (unsigned i = 0; i < pCount; i += 32) {
    const aie::vector<bfloat16, 32> vX = aie::load_v<32>(pIn + i);
    aie::accum<accfloat, 32> vArg = aie::mul(vX, bfloat16(-1.4453125f));
    vArg = aie::mac(vArg, vX, bfloat16(0.0026168823f));
    const aie::vector<float, 32> vExp =
        aie::mul(aie::exp2<bfloat16>(vArg.template to_vector<float>()), bfloat16(1.0f)).template to_vector<float>();
    aie::store_v(pOut + i, aie::mul(vX, fToBf16(aie::inv(aie::add(vExp, 1.0f)))).template to_vector<float>());
  }
}

// x / sqrt(sum of the squares of x + eps) for the 128 floats at pAt, in place (build_gdn_l2_norm)
__attribute__((noinline)) void fL2Norm(float *__restrict pAt, float pEps) {
  const Xdna2Bf16x16 vScale = fInverseNorm(pAt, pEps);
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    aie::store_v(pAt + i, aie::mul(fToBf16(aie::load_v<16>(pAt + i)), vScale).template to_vector<float>());
  }
}

}  // namespace

extern "C" {

void xdna2_chain_conv(uint8_t *__restrict pX, int32_t *__restrict pParams) {
  (void) pParams;
  float *__restrict vX = reinterpret_cast<float *>(pX);
  const float *__restrict vScalars = reinterpret_cast<const float *>(pX + cLaToken);
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
#if defined(XDNA2_CHAIN_NO_COMPUTE)
  for (unsigned i = 0; i < 4 * cGdnDim; i += 16) {
    put_mcd(v16float(aie::zeros<float, 16>()));
  }
#else
  // the convolution of every channel, its 3 old states and x by its 4 weights in that order (ggml_ssm_conv), and its
  // SiLU
  const bfloat16 *__restrict vIn = reinterpret_cast<const bfloat16 *>(pX + cLaInput);
  const bfloat16 *__restrict vStates = reinterpret_cast<const bfloat16 *>(pX + cLaStates);
  const bfloat16 *__restrict vWeights = reinterpret_cast<const bfloat16 *>(pX + cLaWeights);
  bfloat16 *__restrict vSums = reinterpret_cast<bfloat16 *>(pX + cLaSums);
  for (unsigned c = 0; c < cLaChannels; c += 32) {
    aie::accum<accfloat, 32> vAcc = aie::mul(aie::load_v<32>(vStates + c), aie::load_v<32>(vWeights + c));
    vAcc = aie::mac(vAcc, aie::load_v<32>(vStates + cLaChannels + c), aie::load_v<32>(vWeights + cLaChannels + c));
    vAcc = aie::mac(vAcc, aie::load_v<32>(vStates + 2 * cLaChannels + c),
                    aie::load_v<32>(vWeights + 2 * cLaChannels + c));
    vAcc = aie::mac(vAcc, aie::load_v<32>(vIn + c), aie::load_v<32>(vWeights + 3 * cLaChannels + c));
    aie::store_v(vSums + c, vAcc.template to_vector<bfloat16>());
  }
  fSilu(vSums, vX + cLaConv, cLaChannels);
  // the L2 norms of q and k of the head of keys
  fL2Norm(vX + cLaConv, vScalars[cLaEpsL2]);
  fL2Norm(vX + cLaConv + cGdnDim, vScalars[cLaEpsL2]);
  // the gates of heads s + 16 and s: the weights of the gated norm times silu(z)
  fSilu(reinterpret_cast<const bfloat16 *>(pX + cLaZ), vX + cLaGate, 2 * cGdnDim);
  const bfloat16 *__restrict vNorm = reinterpret_cast<const bfloat16 *>(pX + cLaNorm);
  for (unsigned i = 0; i < 2 * cGdnDim; i += 16) {
    aie::store_v(vX + cLaGate + i, aie::mul(aie::load_v<16>(vNorm + (i % cGdnDim)), fToBf16(aie::load_v<16>(vX + cLaGate + i)))
                                       .template to_vector<float>());
  }
  // q, k and v of head s and its gate, to core 0 over the cascade
  for (unsigned i = 0; i < 3 * cGdnDim; i += 16) {
    put_mcd(v16float(aie::load_v<16>(vX + cLaConv + i)));
  }
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    put_mcd(v16float(aie::load_v<16>(vX + cLaGate + cGdnDim + i)));
  }
  // the inputs of the recurrence of head s + 16: q, k and beta v
  for (unsigned i = 0; i < 2 * cGdnDim; i += 16) {
    aie::store_v(vX + cGdnQ + i, aie::load_v<16>(vX + cLaConv + i));
  }
  const Xdna2Bf16x16 vBeta = fToBf16(aie::broadcast<float, 16>(vScalars[cLaBeta]));
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    aie::store_v(vX + cGdnBetaV + i,
                 aie::mul(fToBf16(aie::load_v<16>(vX + cLaConv + 3 * cGdnDim + i)), vBeta).template to_vector<float>());
  }
#endif
  // e^g and beta e^g, as the host gives them
  vX[cGdnDecay] = vScalars[cLaDecay];
  vX[cGdnBetaDecay] = vScalars[cLaBetaDecay];
  aie::set_rounding(vSaved);
}

#elif defined(XDNA2_CHAIN_CONV_RECV)

void xdna2_chain_conv_recv(uint8_t *__restrict pX, int32_t *__restrict pParams) {
  (void) pParams;
  float *__restrict vX = reinterpret_cast<float *>(pX);
  const float *__restrict vScalars = reinterpret_cast<const float *>(pX + cLaToken);
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  // q, k and v of the head of this core (s) from core 1, where the recurrence reads them, and its gate
  for (unsigned i = 0; i < 3 * cGdnDim; i += 16) {
    aie::store_v(vX + cGdnQ + i, aie::vector<float, 16>(get_scd_v16float()));
  }
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    aie::store_v(vX + cLaGate + i, aie::vector<float, 16>(get_scd_v16float()));
  }
#if !defined(XDNA2_CHAIN_NO_COMPUTE)
  const Xdna2Bf16x16 vBeta = fToBf16(aie::broadcast<float, 16>(vScalars[cLaBeta]));
  for (unsigned i = 0; i < cGdnDim; i += 16) {
    aie::store_v(vX + cGdnBetaV + i,
                 aie::mul(fToBf16(aie::load_v<16>(vX + cGdnBetaV + i)), vBeta).template to_vector<float>());
  }
#endif
  // e^g and beta e^g, as the host gives them
  vX[cGdnDecay] = vScalars[cLaDecay];
  vX[cGdnBetaDecay] = vScalars[cLaBetaDecay];
  aie::set_rounding(vSaved);
}

#endif

} // extern "C"
