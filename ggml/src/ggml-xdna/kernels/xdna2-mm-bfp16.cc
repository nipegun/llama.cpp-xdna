// llama.cpp-xdna: matrix multiplication for the core tiles of the XDNA2 NPU (AIE2P), on block floating point
// operands and with 32-bit float sums.
//
//   xdna2_mm_bfp16(pA, pB, pSums, pFirst)       Sums = A x B when pFirst is not zero, Sums += A x B when it is zero
//   xdna2_mm_bfp16_last(pA, pB, pSums, pOut)    Out = Sums + A x B rounded to bfloat16 (or to block floating
//                                               point, see XDNA2_C_BFP16); Sums is left as it was
//
// A is XDNA2_DIM_M x XDNA2_DIM_K, B is XDNA2_DIM_K x XDNA2_DIM_N, and Sums and Out are XDNA2_DIM_M x
// XDNA2_DIM_N: one tile of each. A whole product adds up one call per tile of the K dimension into the same tile
// of sums, the first one with pFirst set, so that the sums need no zeroing. When C goes back in 32-bit floats,
// Sums is C and every tile of K goes through xdna2_mm_bfp16. When it goes back in bfloat16, Sums stays in the
// core and the last tile of K goes through xdna2_mm_bfp16_last, which writes C into Out, rounded to the nearest
// and ties to even: K has to be two tiles at least.
//
// A and B hold bfp16ebs8, the block floating point of AIE2P: 8 values with 8-bit mantissas share one 8-bit
// exponent, 9 bytes for 8 values. In both operands a block is 8 consecutive values along K. Sums holds floats and
// Out bfloat16.
//
// Built with -DXDNA2_C_BFP16 (both functions), Out holds C in bfp16ebs8 instead, 9 bytes for 8 values, and a block
// is 8 rows of one column of C: 8 tokens of one output, which have values of a similar size, where 8 outputs of
// one token can be far apart (a few outputs of a model are much larger than the rest). The multiplier then
// computes C transposed, block by block (B times A instead of A times B), so that its 8 x 8 sums come out as 8
// columns of 8 rows: Sums holds every block of 8 x 8 transposed, which only the core sees.
//
// The multiplier takes 8 x 8 values of A and 8 x 8 of B at a time (a group: 8 blocks, 72 bytes) and adds their
// 8 x 8 products to 64 sums. The kernel keeps four sets of sums in registers at a time: the ones of 16 rows of A
// by 16 columns of B. For them it walks the K of the tile 16 values at a time, and then moves on to the next 16
// columns. The tiles come laid out in that order, so that the kernel reads and writes every one of them from
// the first byte to the last, the tile of A once for every 16 columns:
//
//   A  [M / 16][K / 16][2 groups of rows][2 groups along K][8 rows][9 bytes]
//   B  [N / 16][K / 16][2 groups of columns][2 groups along K][8 columns][9 bytes]      B transposed
//   C  [M / 16][N / 16][2 groups of rows][2 groups of columns][8 rows][8 columns]       Sums and Out alike
//      with XDNA2_C_BFP16: [M / 16][N / 16][2][2][8 columns][8 rows], Out as 8 blocks of 9 bytes per 8 x 8
//
// Built for the cores with Peano (the clang of llvm-aie) through mlir-aie:
//   -DXDNA2_DIM_M=<m> -DXDNA2_DIM_K=<k> -DXDNA2_DIM_N=<n>
// gives xdna2_mm_bfp16, and the same with -DXDNA2_LAST_TILE gives xdna2_mm_bfp16_last instead: one function in
// every object, so that a core can take both objects.
//
// The scheme of four sets of sums and two block streams is the one of aie_kernels/linalg/mm_bfp.cc of mlir-aie
// (Copyright (C) 2025 Advanced Micro Devices, Inc., Apache-2.0 WITH LLVM-exception). That kernel keeps its sums
// in block floating point, rounded to 8-bit mantissas after every call, and hops between rows of groups; this
// one keeps the sums in 32-bit floats and reads in one run.

#include <aie_api/aie.hpp>

#include <stdint.h>

#ifndef XDNA2_DIM_M
#define XDNA2_DIM_M 64
#endif

#ifndef XDNA2_DIM_K
#define XDNA2_DIM_K 64
#endif

#ifndef XDNA2_DIM_N
#define XDNA2_DIM_N 64
#endif

namespace {

constexpr unsigned cBlock = 8;               // values of a block, and blocks of a group
constexpr unsigned cLanes = cBlock * cBlock;  // values of a group of A or of B, and sums of a group of C
constexpr unsigned cPair = 2 * cBlock;        // rows, columns and values of K the kernel takes at a time

using Xdna2Sums = aie::accum<accfloat, cLanes>;
using Xdna2Groups = aie::block_vector_input_buffer_stream<bfp16ebs8, cLanes>;

#ifdef XDNA2_C_BFP16
using Xdna2Out = bfp16ebs8;
#else
using Xdna2Out = bfloat16;
#endif

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

// pRowPairs, pPairsK and pColPairs count the tile in pairs of groups: M / 16, K / 16 and N / 16. pFromZero starts
// the sums instead of reading them, and pToOut writes them rounded to bfloat16 in pOut instead of back in pSums.
template <unsigned pRowPairs, unsigned pPairsK, unsigned pColPairs, bool pFromZero, bool pToOut>
inline void fMultiply(const bfp16ebs8 *__restrict pA, const bfp16ebs8 *__restrict pB, float *__restrict pSums,
                      Xdna2Out *__restrict pOut) {
  constexpr unsigned cSteps = pRowPairs * pColPairs;
  constexpr unsigned cGroupsPerPair = 4 * pPairsK;  // groups of A of 16 rows, and of B of 16 columns

  float *__restrict vSums = pSums;
#ifdef XDNA2_C_BFP16
  aie::block_vector_output_buffer_stream<bfp16ebs8, cLanes> vOut(pOut);
#else
  bfloat16 *__restrict vOut = pOut;
#endif

  // one loop over the rows and the columns: Peano only pipelines an innermost loop
#pragma clang loop min_iteration_count(4)
  for (unsigned vStep = 0; vStep < cSteps; ++vStep) {
    const unsigned vRowPair = vStep / pColPairs;
    const unsigned vColPair = vStep % pColPairs;

    // Only the four sets of sums of the step are alive at a time: the core has five registers that hold one,
    // and a set that does not fit in them goes to the stack and back.
    Xdna2Sums vSum00;
    Xdna2Sums vSum01;
    Xdna2Sums vSum10;
    Xdna2Sums vSum11;
    if constexpr (pFromZero) {
      vSum00 = aie::zeros<accfloat, cLanes>();
      vSum01 = aie::zeros<accfloat, cLanes>();
      vSum10 = aie::zeros<accfloat, cLanes>();
      vSum11 = aie::zeros<accfloat, cLanes>();
    } else {
      vSum00.from_vector(aie::load_v<cLanes>(vSums));
      vSum01.from_vector(aie::load_v<cLanes>(vSums + cLanes));
      vSum10.from_vector(aie::load_v<cLanes>(vSums + 2 * cLanes));
      vSum11.from_vector(aie::load_v<cLanes>(vSums + 3 * cLanes));
    }

    // Each stream keeps its state in one of the two registers AIE2P has for that: one stream for A, one for B.
    Xdna2Groups vStreamA(pA);
    vStreamA.seek(vRowPair * cGroupsPerPair);
    Xdna2Groups vStreamB(pB);
    vStreamB.seek(vColPair * cGroupsPerPair);

#pragma clang loop unroll(full)
    for (unsigned vPair = 0; vPair < pPairsK; ++vPair) {
      // rows 0 to 7 and 8 to 15 of the step (columns, for B), each along 8 and 8 values of K
      auto vA0a = vStreamA.pop();
      auto vA0b = vStreamA.pop();
      auto vA1a = vStreamA.pop();
      auto vA1b = vStreamA.pop();
      auto vB0a = vStreamB.pop();
      auto vB0b = vStreamB.pop();
      auto vB1a = vStreamB.pop();
      auto vB1b = vStreamB.pop();
      vSum00 = fMac(vA0a, vB0a, vSum00);
      vSum01 = fMac(vA0a, vB1a, vSum01);
      vSum10 = fMac(vA1a, vB0a, vSum10);
      vSum11 = fMac(vA1a, vB1a, vSum11);
      vSum00 = fMac(vA0b, vB0b, vSum00);
      vSum01 = fMac(vA0b, vB1b, vSum01);
      vSum10 = fMac(vA1b, vB0b, vSum10);
      vSum11 = fMac(vA1b, vB1b, vSum11);
    }

    if constexpr (pToOut) {
#ifdef XDNA2_C_BFP16
      vOut.push(vSum00.template to_vector<bfp16ebs8>());
      vOut.push(vSum01.template to_vector<bfp16ebs8>());
      vOut.push(vSum10.template to_vector<bfp16ebs8>());
      vOut.push(vSum11.template to_vector<bfp16ebs8>());
#else
      aie::store_v(vOut, vSum00.template to_vector<bfloat16>());
      aie::store_v(vOut + cLanes, vSum01.template to_vector<bfloat16>());
      aie::store_v(vOut + 2 * cLanes, vSum10.template to_vector<bfloat16>());
      aie::store_v(vOut + 3 * cLanes, vSum11.template to_vector<bfloat16>());
      vOut += 4 * cLanes;
#endif
    } else {
      aie::store_v(vSums, vSum00.template to_vector<float>());
      aie::store_v(vSums + cLanes, vSum01.template to_vector<float>());
      aie::store_v(vSums + 2 * cLanes, vSum10.template to_vector<float>());
      aie::store_v(vSums + 3 * cLanes, vSum11.template to_vector<float>());
    }
    vSums += 4 * cLanes;
  }
}

template <unsigned pRowPairs, unsigned pColPairs>
constexpr void fCheckTile() {
  static_assert(XDNA2_DIM_M % cPair == 0, "the rows of the tile of A go 16 at a time");
  static_assert(XDNA2_DIM_K % cPair == 0, "the K of the tile is walked 16 values at a time");
  static_assert(XDNA2_DIM_N % cPair == 0, "the columns of the tile of B go 16 at a time");
  static_assert(pRowPairs * pColPairs >= 4, "the loop over the tile of C is declared with 4 steps or more");
}

} // namespace

extern "C" {

#ifndef XDNA2_LAST_TILE

void xdna2_mm_bfp16(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums, int32_t pFirst) {
  constexpr unsigned cRowPairs = XDNA2_DIM_M / cPair;
  constexpr unsigned cPairsK = XDNA2_DIM_K / cPair;
  constexpr unsigned cColPairs = XDNA2_DIM_N / cPair;
  fCheckTile<cRowPairs, cColPairs>();

  if (pFirst != 0) {
    fMultiply<cRowPairs, cPairsK, cColPairs, true, false>(pA, pB, pSums, nullptr);
  } else {
    fMultiply<cRowPairs, cPairsK, cColPairs, false, false>(pA, pB, pSums, nullptr);
  }
}

#else

void xdna2_mm_bfp16_last(bfp16ebs8 *__restrict pA, bfp16ebs8 *__restrict pB, float *__restrict pSums,
                         Xdna2Out *__restrict pOut) {
  constexpr unsigned cRowPairs = XDNA2_DIM_M / cPair;
  constexpr unsigned cPairsK = XDNA2_DIM_K / cPair;
  constexpr unsigned cColPairs = XDNA2_DIM_N / cPair;
  fCheckTile<cRowPairs, cColPairs>();

  // The core starts in the rounding mode floor, which would push every value of C down by half a step on average:
  // to the nearest, ties to even, for this call only.
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  fMultiply<cRowPairs, cPairsK, cColPairs, false, true>(pA, pB, pSums, pOut);
  aie::set_rounding(vSaved);
}

#endif

} // extern "C"
