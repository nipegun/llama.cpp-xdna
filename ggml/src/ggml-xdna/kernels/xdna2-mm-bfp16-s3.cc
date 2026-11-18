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

} // namespace

extern "C" {

#ifndef XDNA2_LAST_TILE

void xdna2_mm_bfp16_s3(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums, int32_t pFirst) {
  if (pFirst != 0) {
    fMultiply<true, false>(pA, pB, pSums, nullptr);
  } else {
    fMultiply<false, false>(pA, pB, pSums, nullptr);
  }
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
