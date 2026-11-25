// llama.cpp-xdna: matrix multiplication for the core tiles of the XDNA2 NPU (AIE2P), on block floating point
// operands and with 32-bit float sums, in steps of three sets of sums.
//
//   xdna2_mm_bfp16_s3(pA, pB, pSums, pFirst)       Sums = A x B when pFirst is not zero, Sums += A x B when it is zero
//   xdna2_mm_bfp16_s3_last(pA, pB, pSums, pOut)    Out = Sums + A x B rounded to bfloat16 (or to block floating
//                                                  point, see XDNA2_C_BFP16); Sums is left as it was
//
// The same as xdna2-mm-bfp16.cc (see there what the functions, the stages of K and XDNA2_C_BFP16 are), with
// another order of the work and of the tiles. That kernel multiplies 16 rows by 16 columns at a time, four sets of
// 64 sums, and the core has five registers for them: there is none left to read the sums of the next step while
// the multiplier works, and every step waits for them, about 105 cycles for 64 multiplies. This one multiplies 8
// rows by 24 columns, three sets: the two registers left let the compiler overlap the steps, 60 cycles for 48
// multiplies (80 %). A set can only take another multiply 3 cycles later, so three that alternate keep the
// multiplier busy.
//
// The 8 groups of 8 columns of a tile go in sets of 3 (and a last one of 2 when 3 does not divide them): every
// group of 8 rows of C is one step for every set of 3, and then one step of 2 sets of sums for the last set of 2.
// A and B are groups of 8 x 8 values (8 blocks of 9 bytes, one per row of A or column of B, each along 8 values
// of K), in the order the steps read them:
//
//   A     [M / 8][K / 8][8 rows][9 bytes]
//   B     [sets][K / 8][3 groups of columns, or 2 in the last set][8 columns][9 bytes]        B transposed
//   Sums  [M / 8][sets of 3][3][8 x 8], then [M / 8][2][8 x 8] for the last set of 2       Out alike
//
// every 8 x 8 as 8 rows of 8 columns (with XDNA2_C_BFP16, 8 columns of 8 rows, and Out as 8 blocks of 9 bytes).
//
// llama.cpp-xdna: with XDNA2_C_ROWS (C in bfloat16), Out is the rows of the tile instead, [M][N], every 8 x 8 as 8
// stores of a row of 8 values, so that the shim tiles can lay the tiles out as the rows of C in the buffer of the
// host (see XdnaOverlay::vRowsC); the function is then xdna2_mm_bfp16_s3_last_rows.
//
// Built for the cores with Peano (the clang of llvm-aie) through mlir-aie:
//   -DXDNA2_DIM_M=<m> -DXDNA2_DIM_K=<k> -DXDNA2_DIM_N=<n>   [-DXDNA2_C_BFP16 | -DXDNA2_C_ROWS]
// gives xdna2_mm_bfp16_s3, and the same with -DXDNA2_LAST_TILE gives xdna2_mm_bfp16_s3_last instead.
//
// The multiplier and the reading of groups through block streams are those of aie_kernels/linalg/mm_bfp.cc of
// mlir-aie (Copyright (C) 2025 Advanced Micro Devices, Inc., Apache-2.0 WITH LLVM-exception).

#include <aie_api/aie.hpp>

#include <stdint.h>

#ifndef XDNA2_DIM_M
#define XDNA2_DIM_M 64
#endif

#ifndef XDNA2_DIM_K
#define XDNA2_DIM_K 128
#endif

#ifndef XDNA2_DIM_N
#define XDNA2_DIM_N 64
#endif

namespace {

constexpr unsigned cBlock = 8;               // values of a block, and blocks of a group
constexpr unsigned cLanes = cBlock * cBlock;  // values of a group of A or of B, and sums of a group of C

constexpr unsigned cRowGroups = XDNA2_DIM_M / cBlock;
constexpr unsigned cGroupsK = XDNA2_DIM_K / cBlock;
constexpr unsigned cColumnGroups = XDNA2_DIM_N / cBlock;
constexpr unsigned cSets = cColumnGroups / 3;   // sets of 3 groups of columns
constexpr unsigned cLast = cColumnGroups % 3;   // groups of columns of the last set: 0 or 2

static_assert(XDNA2_DIM_M % cBlock == 0 && XDNA2_DIM_K % cBlock == 0 && XDNA2_DIM_N % cBlock == 0,
              "the tiles go 8 rows, columns and values of K at a time");
static_assert(cLast != 1, "a last set of one group of columns would leave the multiplier waiting");
static_assert(cRowGroups * cSets >= 4 && (cLast == 0 || cRowGroups >= 4),
              "the loops are declared with 4 steps or more");

#if defined(XDNA2_C_ROWS) && defined(XDNA2_C_BFP16)
#error "the rows of the tile are rows of bfloat16"
#endif

using Xdna2Sums = aie::accum<accfloat, cLanes>;
using Xdna2Groups = aie::block_vector_input_buffer_stream<bfp16ebs8, cLanes>;

#ifdef XDNA2_C_BFP16
using Xdna2Out = bfp16ebs8;
using Xdna2Writer = aie::block_vector_output_buffer_stream<bfp16ebs8, cLanes>;
#else
using Xdna2Out = bfloat16;
#endif

// A 0 the compiler cannot see. The groups of B of the last set are the same for every step of its loop, and with
// a position it knows the compiler reads them once before the loop and keeps them in the stack: twice as slow.
volatile unsigned gZero = 0;

// pSum + the products of 8 rows of A by 8 columns of B, along 8 values of K: as they are, or transposed for C in
// block floating point
template <typename T>
inline Xdna2Sums fMac(const T &pA, const T &pB, const Xdna2Sums &pSum) {
#ifdef XDNA2_C_BFP16
  return mac_8x8_8x8T(pB, pA, pSum);
#else
  return mac_8x8_8x8T(pA, pB, pSum);
#endif
}

// Writes a set of sums rounded into Out: into the writer of block floating point, or at pAt sets from pOut. The
// sums that stay in Sums are stored where they are read, in the loops: through a function, the compiler forgets
// that the two pointers of a loop do not meet, and stops overlapping its steps.
template <typename TOut>
inline void fWriteOut(const Xdna2Sums &pSum, TOut &pOut, unsigned pAt) {
#ifdef XDNA2_C_BFP16
  pOut.push(pSum.template to_vector<bfp16ebs8>());
#else
  aie::store_v(pOut + pAt * cLanes, pSum.template to_vector<bfloat16>());
#endif
}

#ifdef XDNA2_C_ROWS
// Writes a set of sums rounded into Out as 8 rows of 8 values: pOut is the first one, and the rows of the tile are
// XDNA2_DIM_N values apart (16-byte stores at 16-byte boundaries)
inline void fWriteRows(const Xdna2Sums &pSum, bfloat16 *pOut) {
  const aie::vector<bfloat16, cLanes> vValues = pSum.template to_vector<bfloat16>();
#pragma clang loop unroll(full)
  for (unsigned i = 0; i < cBlock; ++i) {
    aie::store_v(pOut + i * XDNA2_DIM_N, vValues.template extract<cBlock>(i));
  }
}
#endif

// The steps of the sets of 3. pIn and pSums are the same sums: two pointers that the compiler takes for different
// memory, so that it reads the sums of a step before it writes the ones of the step before (every step reads and
// writes its own).
template <bool pFromZero, bool pToOut, typename TOut>
inline void fThree(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, const float *__restrict pIn,
                   float *__restrict pSums, TOut pOut) {
  constexpr unsigned cSteps = cRowGroups * cSets;
  const float *__restrict vIn = pIn;
  float *__restrict vSums = pSums;
  // one loop over the rows and the sets: Peano only pipelines an innermost loop
#pragma clang loop min_iteration_count(4)
  for (unsigned vStep = 0; vStep < cSteps; ++vStep) {
    const unsigned vRow = vStep / cSets;
    const unsigned vSet = vStep % cSets;
    Xdna2Sums vSum0;
    Xdna2Sums vSum1;
    Xdna2Sums vSum2;
    if constexpr (pFromZero) {
      vSum0 = aie::zeros<accfloat, cLanes>();
      vSum1 = aie::zeros<accfloat, cLanes>();
      vSum2 = aie::zeros<accfloat, cLanes>();
    } else {
      vSum0.from_vector(aie::load_v<cLanes>(vIn));
      vSum1.from_vector(aie::load_v<cLanes>(vIn + cLanes));
      vSum2.from_vector(aie::load_v<cLanes>(vIn + 2 * cLanes));
    }
    Xdna2Groups vStreamA(pA);
    vStreamA.seek(vRow * cGroupsK);
    Xdna2Groups vStreamB(pB);
    vStreamB.seek(vSet * cGroupsK * 3);
#pragma clang loop unroll(full)
    for (unsigned k = 0; k < cGroupsK; ++k) {
      auto vA = vStreamA.pop();
      auto vB0 = vStreamB.pop();
      auto vB1 = vStreamB.pop();
      auto vB2 = vStreamB.pop();
      vSum0 = fMac(vA, vB0, vSum0);
      vSum1 = fMac(vA, vB1, vSum1);
      vSum2 = fMac(vA, vB2, vSum2);
    }
    if constexpr (pToOut) {
#ifdef XDNA2_C_ROWS
      bfloat16 *vRows = pOut + vRow * cBlock * XDNA2_DIM_N + vSet * 3 * cBlock;
      fWriteRows(vSum0, vRows);
      fWriteRows(vSum1, vRows + cBlock);
      fWriteRows(vSum2, vRows + 2 * cBlock);
#else
      fWriteOut(vSum0, pOut, 0);
      fWriteOut(vSum1, pOut, 1);
      fWriteOut(vSum2, pOut, 2);
#endif
    } else {
      aie::store_v(vSums, vSum0.template to_vector<float>());
      aie::store_v(vSums + cLanes, vSum1.template to_vector<float>());
      aie::store_v(vSums + 2 * cLanes, vSum2.template to_vector<float>());
    }
    vIn += 3 * cLanes;
    vSums += 3 * cLanes;
    if constexpr (pToOut) {
#if !defined(XDNA2_C_BFP16) && !defined(XDNA2_C_ROWS)
      pOut += 3 * cLanes;
#endif
    }
  }
}

// The steps of the last set, of 2 groups of columns: two sets of sums, one per group, every one taking a multiply
// every 2 cycles, which leaves the multiplier waiting one cycle in three (about 72 % of the cycles).
template <bool pFromZero, bool pToOut, typename TOut>
inline void fTwo(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, const float *__restrict pIn,
                 float *__restrict pSums, TOut pOut, unsigned pZero) {
  const float *__restrict vIn = pIn;
  float *__restrict vSums = pSums;
#pragma clang loop min_iteration_count(4)
  for (unsigned vRow = 0; vRow < cRowGroups; ++vRow) {
    Xdna2Sums vSum0;
    Xdna2Sums vSum1;
    if constexpr (pFromZero) {
      vSum0 = aie::zeros<accfloat, cLanes>();
      vSum1 = aie::zeros<accfloat, cLanes>();
    } else {
      vSum0.from_vector(aie::load_v<cLanes>(vIn));
      vSum1.from_vector(aie::load_v<cLanes>(vIn + cLanes));
    }
    Xdna2Groups vStreamA(pA);
    vStreamA.seek(vRow * cGroupsK);
    Xdna2Groups vStreamB(pB);
    vStreamB.seek(cSets * cGroupsK * 3 + (vRow & pZero));
#pragma clang loop unroll(full)
    for (unsigned k = 0; k < cGroupsK; ++k) {
      auto vA = vStreamA.pop();
      auto vB0 = vStreamB.pop();
      auto vB1 = vStreamB.pop();
      vSum0 = fMac(vA, vB0, vSum0);
      vSum1 = fMac(vA, vB1, vSum1);
    }
    if constexpr (pToOut) {
#ifdef XDNA2_C_ROWS
      bfloat16 *vRows = pOut + vRow * cBlock * XDNA2_DIM_N;
      fWriteRows(vSum0, vRows);
      fWriteRows(vSum1, vRows + cBlock);
#else
      fWriteOut(vSum0, pOut, 0);
      fWriteOut(vSum1, pOut, 1);
#endif
    } else {
      aie::store_v(vSums, vSum0.template to_vector<float>());
      aie::store_v(vSums + cLanes, vSum1.template to_vector<float>());
    }
    vIn += 2 * cLanes;
    vSums += 2 * cLanes;
    if constexpr (pToOut) {
#if !defined(XDNA2_C_BFP16) && !defined(XDNA2_C_ROWS)
      pOut += 2 * cLanes;
#endif
    }
  }
}

// One tile: the steps of the sets of 3, then the ones of the last set. Out is written in the same order as Sums.
// Its pointers carry no __restrict: pSums goes into every loop as two pointers, and with one here the compiler
// stops overlapping the steps of the loops that add up (87 cycles for 48 multiplies instead of 60).
template <bool pFromZero, bool pToOut>
inline void fMultiply(const bfp16ebs8 *pA, const bfp16ebs8 *pB, float *pSums, Xdna2Out *pOut) {
  constexpr unsigned cFirstPart = cRowGroups * cSets * 3;  // groups of 8 x 8 of the sets of 3
  const unsigned vZero = gZero;
  float *vTwo = pSums + cFirstPart * cLanes;
  if constexpr (pToOut) {
#ifdef XDNA2_C_BFP16
    // one writer per loop: the second one starts where the first one ends, 72 bytes per 8 x 8
    Xdna2Writer vOutThree(pOut);
    fThree<pFromZero, pToOut, Xdna2Writer &>(pA, pB, pSums, pSums, vOutThree);
    if constexpr (cLast == 2) {
      Xdna2Writer vOutTwo(reinterpret_cast<bfp16ebs8 *>(reinterpret_cast<uint8_t *>(pOut) + cFirstPart * 72));
      fTwo<pFromZero, pToOut, Xdna2Writer &>(pA, pB, vTwo, vTwo, vOutTwo, vZero);
    }
#elif defined(XDNA2_C_ROWS)
    // the last set of 2 starts at its first column, in every row of the tile
    fThree<pFromZero, pToOut, bfloat16 *>(pA, pB, pSums, pSums, pOut);
    if constexpr (cLast == 2) {
      fTwo<pFromZero, pToOut, bfloat16 *>(pA, pB, vTwo, vTwo, pOut + cSets * 3 * cBlock, vZero);
    }
#else
    fThree<pFromZero, pToOut, bfloat16 *>(pA, pB, pSums, pSums, pOut);
    if constexpr (cLast == 2) {
      fTwo<pFromZero, pToOut, bfloat16 *>(pA, pB, vTwo, vTwo, pOut + cFirstPart * cLanes, vZero);
    }
#endif
  } else {
    fThree<pFromZero, pToOut, Xdna2Out *>(pA, pB, pSums, pSums, pOut);
    if constexpr (cLast == 2) {
      fTwo<pFromZero, pToOut, Xdna2Out *>(pA, pB, vTwo, vTwo, pOut, vZero);
    }
  }
}

#ifdef XDNA2_MM_COMP

constexpr unsigned cQ6kColBytes = 864;
constexpr unsigned cQ6kScales   = 32;
constexpr unsigned cQ6kValues   = 96;
constexpr unsigned cQ6kTop      = 608;
constexpr unsigned cQ8ColBytes  = 1088;
constexpr unsigned cQ8Values    = 64;
constexpr unsigned cBfpColBytes = 1152;

using Xdna2BlockWriter = aie::block_vector_output_buffer_stream<bfp16ebs8, cLanes>;

__attribute__((noinline)) void fCopy32(const uint8_t *__restrict pFrom, uint8_t *__restrict pTo, unsigned pBytes) {
#pragma clang loop min_iteration_count(4)
  for (unsigned i = 0; i < pBytes; i += 32) {
    aie::store_v(pTo + i, aie::load_v<32>(pFrom + i));
  }
}

inline aie::vector<bfloat16, 64> fRepeat8Bf16(const aie::vector<bfloat16, 64> &pValues) {
  const auto [vTwo, vTwoHigh] = aie::interleave_zip(pValues, pValues, 1);
  const auto [vFour, vFourHigh] = aie::interleave_zip(vTwo, vTwo, 2);
  const auto [vEight, vEightHigh] = aie::interleave_zip(vFour, vFour, 4);
  (void) vTwoHigh;
  (void) vFourHigh;
  (void) vEightHigh;
  return vEight;
}

inline aie::vector<bfloat16, 64> fPlus128(const aie::vector<uint8, 64> &pQ) {
  const aie::vector<uint8, 64> v43 = aie::broadcast<uint8, 64>(uint8(0x43));
  const auto [vLow, vHigh] = aie::interleave_zip(pQ, v43, 1);
  return aie::concat(vLow, vHigh).template cast_to<bfloat16>();
}

inline aie::vector<bfloat16, 64> fRepeat8From16(const aie::vector<uint16, 32> &pPair) {
  return fRepeat8Bf16(aie::concat(pPair, pPair).template cast_to<bfloat16>());
}

// llama.cpp-xdna: decodes a 6912-byte Q6_K tile (pIsQ8 == false, 8 column-groups of 864 bytes) or an 8704-byte Q8_0
// tile (pIsQ8 == true, 8 column-groups of 1088 bytes) in-place into the 9216-byte bfp16 tile order of cSets
// ([set][k = 0..15][j = 0..count-1]), working from set 2 down to set 0 so unread sets are untouched.
__attribute__((noinline)) void fDecodeTileComp(uint8_t *pB, uint8_t *pScratch, bool pIsQ8) {
  alignas(32) uint8_t aSet[2 * cQ6kColBytes];
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const aie::vector<bfloat16, 64> v128 = aie::broadcast<bfloat16, 64>(bfloat16(128.0f));
  const aie::vector<bfloat16, 64> v256 = aie::broadcast<bfloat16, 64>(bfloat16(256.0f));
  const aie::vector<bfloat16, 64> vMinus160 = aie::broadcast<bfloat16, 64>(bfloat16(-160.0f));
  const aie::vector<uint8, 64> vLow7 = aie::broadcast<uint8, 64>(uint8(0x7f));
  const aie::vector<uint8, 64> vLow2 = aie::broadcast<uint8, 64>(uint8(3));

#pragma clang loop unroll(disable)
  for (int vSet = 2; vSet >= 0; --vSet) {
    const unsigned vFirstCol = unsigned(vSet) * 3;
    const unsigned vCount = vSet == 2 ? 2u : 3u;
    aie::vector<bfloat16, 64> vAll0;
    aie::vector<bfloat16, 64> vAll1;
    aie::vector<bfloat16, 64> vAll2;
    if (pIsQ8) {
      fCopy32(pB + vFirstCol * cQ8ColBytes, pScratch, vCount * cQ8ColBytes);
    } else {
      fCopy32(pB + vFirstCol * cQ6kColBytes, aSet, 2 * cQ6kColBytes);
      if (vCount == 3u) {
        fCopy32(pB + (vFirstCol + 2u) * cQ6kColBytes, pScratch, cQ6kColBytes);
      }
#pragma clang loop unroll(disable)
      for (unsigned j = 0; j < vCount; ++j) {
        const uint8_t *__restrict vCol = j < 2u ? aSet + j * cQ6kColBytes : pScratch;
        const aie::vector<uint16, 16> vD8 = aie::load_v<16>(reinterpret_cast<const uint16 *>(vCol));
        const aie::vector<uint16, 16> vD8Twice = aie::concat(vD8.extract<8>(0), vD8.extract<8>(0));
        const aie::vector<uint16, 32> vD32 = aie::concat(vD8Twice, vD8Twice);
        const aie::vector<bfloat16, 64> vD = aie::concat(vD32, vD32).template cast_to<bfloat16>();
        const aie::vector<uint8, 64> vScales =
            aie::concat(aie::load_v<32>(vCol + cQ6kScales), aie::load_v<32>(vCol + cQ6kScales + 32));
        const aie::vector<bfloat16, 64> vC = aie::select(v128, v256, aie::ge(vScales, uint8(128)));
        const aie::vector<bfloat16, 64> vAll =
            aie::msc(aie::mul(vD, fPlus128(aie::bit_and(vScales, vLow7))), vD, vC).template to_vector<bfloat16>();
        if (j == 0) {
          vAll0 = vAll;
        } else if (j == 1) {
          vAll1 = vAll;
        } else {
          vAll2 = vAll;
        }
      }
    }
    Xdna2BlockWriter vWriter(reinterpret_cast<bfp16ebs8 *>(pB + vFirstCol * cBfpColBytes));
#pragma clang loop unroll(disable)
    for (unsigned k = 0; k < 16; ++k) {
#pragma clang loop unroll(disable)
      for (unsigned j = 0; j < vCount; ++j) {
        const uint8_t *__restrict vCol =
            pIsQ8 ? pScratch + j * cQ8ColBytes : (j < 2u ? aSet + j * cQ6kColBytes : pScratch);
        Xdna2Sums vAcc;
        if (pIsQ8) {
          const unsigned vSub = k >> 2;
          const aie::vector<uint16, 32> vPairD =
              aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(vCol + (vSub >> 1) * 32)),
                          aie::zeros<uint16, 16>());
          const aie::vector<bfloat16, 64> vFactor =
              fRepeat8From16((vSub & 1u) == 0 ? vPairD : aie::shuffle_down(vPairD, 8));
          const uint8_t *__restrict vAt = vCol + cQ8Values + k * 64;
          const aie::vector<uint8, 64> vU = aie::concat(aie::load_v<32>(vAt), aie::load_v<32>(vAt + 32));
          const aie::vector<bfloat16, 64> vC = aie::select(v128, v256, aie::ge(vU, uint8(128)));
          vAcc = aie::msc(aie::mul(fPlus128(aie::bit_and(vU, vLow7)), vFactor), vC, vFactor);
        } else {
          const unsigned vSub = k >> 1;
          const aie::vector<bfloat16, 64> vAllJ = j == 0 ? vAll0 : j == 1 ? vAll1 : vAll2;
          const aie::vector<bfloat16, 64> vFactor = fRepeat8Bf16(aie::shuffle_down(vAllJ, vSub * 8));
          const Xdna2Sums vBase = aie::mul(vFactor, vMinus160);
          const aie::vector<uint8, 64> vTop =
              aie::load_v<64>(reinterpret_cast<const uint4 *>(vCol + cQ6kTop + vSub * 32)).unpack();
          const aie::vector<uint8, 64> vTopLow = aie::bit_and(vTop, vLow2);
          const aie::vector<uint8, 64> vTopHigh = aie::downshift(vTop, 2);
          const aie::vector<uint8, 64> vLow =
              aie::load_v<64>(reinterpret_cast<const uint4 *>(vCol + cQ6kValues + k * 32)).unpack();
          const aie::vector<uint8, 64> vHigh =
              (k & 1u) == 0 ? aie::concat(vTopLow.extract<32>(0), vTopHigh.extract<32>(0)) :
                              aie::concat(vTopLow.extract<32>(1), vTopHigh.extract<32>(1));
          const aie::vector<uint8, 64> vQ = aie::bit_or(vLow, aie::upshift(vHigh, 4));
          vAcc = aie::mac(vBase, fPlus128(vQ), vFactor);
        }
        vWriter.push(vAcc.template to_vector<bfp16ebs8>());
      }
    }
  }
  aie::set_rounding(vSaved);
}

#endif

} // namespace

extern "C" {

void xdna2_mm_bfp16_s3(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums, int32_t pFirst);
void xdna2_mm_bfp16_s3_last_rows(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums,
                                 Xdna2Out *__restrict pOut);

#if defined(XDNA2_MM_COMP)

__attribute__((noinline)) void fRunPlan(uint8_t *pB, uint8_t *pC, uint32_t pPlan) {
#pragma clang loop unroll(disable)
  while (pPlan != 0u) {
    const unsigned vFromSlot = pPlan & 7u;
    const unsigned vToSlot   = (pPlan >> 3u) & 7u;
    const uint8_t *vFrom     = vFromSlot < 4u ? pB + vFromSlot * 2304u : pC + 1280u + (vFromSlot - 4u) * 2304u;
    uint8_t *vTo             = vToSlot < 4u ? pB + vToSlot * 2304u : pC + 1280u + (vToSlot - 4u) * 2304u;
    fCopy32(vFrom, vTo, 2304u);
    pPlan >>= 6u;
  }
}

constexpr uint32_t fMoveOp(unsigned pFrom, unsigned pTo, unsigned pIndex) {
  return ((pFrom & 7u) | ((pTo & 7u) << 3u)) << (6u * pIndex);
}

// llama.cpp-xdna: one step of compressed B in the prompt overlay (pStep 0..3: the 4 tiles of a 512-input Q6_K quad
// packed into 3 objects of 9216 bytes; pStep 4: one 128-input Q8_0 tile in 1 object of 9216 bytes).
void xdna2_mm_comp_step(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, Xdna2Out *__restrict pScratchC,
                        float *__restrict pSums, int32_t pStep, int32_t pQuad, int32_t pQuadsK) {
  uint8_t *vB = reinterpret_cast<uint8_t *>(pB);
  uint8_t *vC = reinterpret_cast<uint8_t *>(pScratchC);
  const bool vIsQ8 = pStep >= 4;
  if (!vIsQ8) {
    constexpr uint32_t cP0A = fMoveOp(3, 4, 0);
    constexpr uint32_t cP1A = fMoveOp(2, 5, 0) | fMoveOp(3, 6, 1) | fMoveOp(1, 2, 2) | fMoveOp(0, 1, 3) |
                              fMoveOp(4, 0, 4);
    constexpr uint32_t cP2A = fMoveOp(1, 4, 0) | fMoveOp(6, 1, 1) | fMoveOp(3, 6, 2) | fMoveOp(2, 3, 3) |
                              fMoveOp(0, 2, 4);
    constexpr uint32_t cP2B = fMoveOp(5, 0, 0) | fMoveOp(3, 5, 1);
    constexpr uint32_t cP3A = fMoveOp(4, 0, 0) | fMoveOp(5, 1, 1) | fMoveOp(6, 2, 2);
    const uint32_t vPlanA = pStep == 0 ? cP0A : pStep == 1 ? cP1A : pStep == 2 ? cP2A : cP3A;
    fRunPlan(vB, vC, vPlanA);
    if (pStep == 2) {
      fRunPlan(vB, vC, cP2B);
    }
  }
  fDecodeTileComp(vB, vC, vIsQ8);
  if (pStep >= 3 && (pQuad + 1 == pQuadsK)) {
    xdna2_mm_bfp16_s3_last_rows(pA, pB, pSums, pScratchC);
  } else {
    const int32_t vFirst = ((pStep == 0 || vIsQ8) && pQuad == 0) ? 1 : 0;
    xdna2_mm_bfp16_s3(pA, pB, pSums, vFirst);
  }
}

#elif !defined(XDNA2_LAST_TILE)

void xdna2_mm_bfp16_s3(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums, int32_t pFirst) {
  if (pFirst != 0) {
    const aie::vector<float, cLanes> vZero = aie::zeros<float, cLanes>();
#pragma clang loop unroll(disable)
    for (unsigned i = 0; i < XDNA2_DIM_M * XDNA2_DIM_N; i += cLanes) {
      aie::store_v(pSums + i, vZero);
    }
  }
  fMultiply<false, false>(pA, pB, pSums, nullptr);
}

#else

#ifdef XDNA2_C_ROWS
void xdna2_mm_bfp16_s3_last_rows(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums,
                                 Xdna2Out *__restrict pOut) {
#else
void xdna2_mm_bfp16_s3_last(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums,
                            Xdna2Out *__restrict pOut) {
#endif
  // The core starts in the rounding mode floor, which would push every value of C down by half a step on average:
  // to the nearest, ties to even, for this call only.
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  fMultiply<false, true>(pA, pB, pSums, pOut);
  aie::set_rounding(vSaved);
}

#endif

} // extern "C"
