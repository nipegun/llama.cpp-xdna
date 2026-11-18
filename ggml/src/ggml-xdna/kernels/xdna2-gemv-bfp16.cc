// llama.cpp-xdna: multiplication of a few rows by a matrix of weights, for the core tiles of the XDNA2 NPU (AIE2P),
// with 32-bit float sums. It is the kernel of the overlay for generating tokens: every weight is used once per row,
// so what bounds it is how fast the weights arrive, and the fewer bytes a weight takes the better.
//
//   xdna2_gemv_bfp16(pX, pW, pY, pStage, pRows, pFirst, pFormat, pTiles)
//     Y = X x W^T for the next object of weights of a block of outputs, the first one of the block when pFirst is not
//     zero. A block is G groups of 8 outputs (-DXDNA2_GEMV_GROUPS_OUT=G: 2 in the overlay where every channel feeds
//     two cores, each taking one half of every object of the channel) by pTiles tiles of 256 values of K, and its
//     weights come as a stream of pieces, one piece for every group of outputs of every tile, tile after tile:
//
//       block  [tile of 256 values of K][group of 8 outputs][piece]
//
//     cut into objects of G * 2304 bytes. A piece that does not fit in what is left of an object goes on in the next
//     one; the rest of the last object of a block is padding. pFormat says what a piece holds (see below):
//       0  bfp16, 2304 bytes          2  Q5_K, 1408 bytes
//       1  Q4_K, 1152 bytes           3  Q6_K, 1696 bytes
//   xdna2_gemv_load_x(pObject, pX, pStage, pFirst)       built with -DXDNA2_GEMV_LOAD_X instead
//     copies an object (G * 2304 bytes) into X: the first one when pFirst is not zero, the next one when it is zero
//
// X holds the rows (pRows of them, 1 to 8) in bfp16ebs8, the block floating point of AIE2P (8 values with 8-bit
// mantissas share one 8-bit exponent: 9 bytes for 8 values), a block being 8 consecutive values along K, group of 8
// values of K after group, each group as its rows one after another, padded to a multiple of 4 bytes:
//
//   X  [K / 8][pRows * 9 bytes, rounded up to a multiple of 4]          the whole K, in the core for the whole command
//   Y  [G groups of outputs][8 rows][8 outputs]                       32-bit floats; the rows from pRows on are not
//                                                                       meaningful
//
// A piece is 8 outputs by 256 values of K. In the formats of ggml it holds the blocks of the 8 outputs, laid out for
// the vector unit; a group is 8 outputs by 8 values of K, as 64 lanes, output after output (lane 8 j + t: output j,
// value t), and 4-bit fields go two per byte, the first in the low half:
//
//   bfp16  [32 groups of 8 values of K][8 outputs][9 bytes]
//   Q4_K   d [8] bf16, dmin [8] bf16, the 12 bytes of 6-bit scales and minimums [12][8], the 4-bit values [8
//          sub-blocks of 32][4 groups][32 bytes]
//   Q5_K   as Q4_K, the low 4 bits of the values, and then their fifth bits [8 sub-blocks][4 groups][8 bytes]: bit t
//          of byte j is the one of lane 8 j + t
//   Q6_K   d [8] bf16, 16 bytes of padding, the 8-bit scales [16 sub-blocks of 16][8], the low 4 bits of the values
//          [16 sub-blocks][2 groups][32 bytes], and their top 2 bits [16 sub-blocks][2 groups][16 bytes]: field k of a
//          group (k = 0 to 31) holds the bits of lane k and, two bits up, those of lane k + 32
//
// A value of Q4_K and Q5_K is q * d * sc - dmin * m, sc and m being the 6-bit scale and minimum of its sub-block, and
// one of Q6_K is (q - 32) * d * sc, sc being the 8-bit scale of its sub-block, as ggml decodes them. The host gives d
// and dmin rounded to bf16 (ggml keeps them in fp16); the kernel rounds d * sc to bf16, computes the value in 32-bit
// floats (every product exact) and rounds every block of 8 values to bfp16, to the nearest, to multiply.
//
// The multiplier takes 8 x 8 values at a time (a group: 8 blocks, 72 bytes). For every tile the rows of X are copied
// into pStage, a group for every group of 8 values of K; the rows of a group from pRows on are whatever was there
// before, so their sums are not meaningful, which costs nothing: no row is read across rows. pStage keeps between
// calls the counters (the next piece of the block, the bytes of it already copied, the next object of X), the
// factors and bases of the sub-blocks of a piece, and a piece that goes on into the next object:
//
//   pStage  [32 groups][8 rows][9 bytes] (2304 bytes), int32 counters (64 bytes), factors [16][64] bf16 (2048
//           bytes), bases [8][64] 32-bit floats (2048 bytes), a piece (1792 bytes): 8256 bytes
//
// A vector load of 32 bytes from an address that is not a multiple of 32 loads, without any error, from the multiple
// of 32 below it: every piece and every part of it starts at a multiple of 32 bytes, and the loads from pieces take
// 32 bytes.
//
// Built for the cores with Peano (the clang of llvm-aie) through mlir-aie; one function in every object, so that a
// core can take both objects. The use of the block streams and of mac_8x8_8x8T is the one of xdna2-mm-bfp16.cc,
// whose scheme follows aie_kernels/linalg/mm_bfp.cc of mlir-aie (Copyright (C) 2025 Advanced Micro Devices, Inc.,
// Apache-2.0 WITH LLVM-exception).

#include <aie_api/aie.hpp>

#include <stdint.h>

namespace {

constexpr unsigned cBlock = 8;                        // values of a block, and blocks of a group
constexpr unsigned cBlockBytes = 9;                   // bytes of a block
constexpr unsigned cLanes = cBlock * cBlock;          // values of a group, and sums of a group of Y
constexpr unsigned cGroupBytes = cBlock * cBlockBytes;            // 72
#ifndef XDNA2_GEMV_GROUPS_OUT
#define XDNA2_GEMV_GROUPS_OUT 2
#endif
constexpr unsigned cGroupsOut = XDNA2_GEMV_GROUPS_OUT;  // groups of 8 outputs of a block: pieces of a tile
constexpr unsigned cObjectBytes = cGroupsOut * 2304;
constexpr unsigned cTileGroups = 32;                  // groups of 8 values of K of a tile
constexpr unsigned cStageBytes = cTileGroups * cGroupBytes;       // 2304
constexpr unsigned cPieceCounter = cStageBytes;       // offsets of the counters in pStage
constexpr unsigned cCarryCounter = cStageBytes + 4;
constexpr unsigned cObjectCounter = cStageBytes + 8;
constexpr unsigned cFactors = cStageBytes + 64;       // the factors of the sub-blocks of a piece (aligned to 64)
constexpr unsigned cBases = cFactors + 16 * cLanes * 2;           // and the bases of those of Q4_K and Q5_K
constexpr unsigned cCarry = cBases + 8 * cLanes * 4;  // a piece that goes on into the next object
constexpr unsigned cCarryBytes = 1792;

// the formats, and the bytes of a piece of each
constexpr int32_t cFormatBfp16 = 0;
constexpr int32_t cFormatQ4K = 1;
constexpr int32_t cFormatQ5K = 2;
constexpr int32_t cFormatQ6K = 3;
constexpr unsigned cPieceBfp16 = 2304;
constexpr unsigned cPieceQ4K = 1152;
constexpr unsigned cPieceQ5K = 1408;
constexpr unsigned cPieceQ6K = 1696;

// where the parts of a piece start
constexpr unsigned cScalesK = 32;                     // Q4_K, Q5_K: the 12 bytes of scales and minimums
constexpr unsigned cValuesK = 128;                    // Q4_K, Q5_K: the 4-bit values
constexpr unsigned cFifthQ5K = 1152;                  // Q5_K: the fifth bits
constexpr unsigned cScalesQ6K = 32;                   // Q6_K: the 8-bit scales
constexpr unsigned cValuesQ6K = 160;                  // Q6_K: the low 4 bits
constexpr unsigned cTopQ6K = 1184;                    // Q6_K: the top 2 bits

static_assert(cObjectBytes % 32 == 0 && cPieceQ5K % 32 == 0 && cPieceQ6K % 32 == 0, "pieces aligned to 32 bytes");
static_assert(cCarryBytes >= cPieceQ6K, "the carry holds a piece");

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
  return pFormat == cFormatQ4K ? cPieceQ4K : pFormat == cFormatQ5K ? cPieceQ5K : pFormat == cFormatQ6K ? cPieceQ6K
                                                                                                     : cPieceBfp16;
}

// The groups of a tile, pWords 32-bit words each, into the slots of the stage, with the copy of a group unrolled.
template <unsigned pWords>
inline void fStageWords(const uint32_t *__restrict pFrom, uint32_t *__restrict pTo) {
  for (unsigned vSlot = 0; vSlot < cTileGroups; ++vSlot) {
#pragma clang loop unroll(full)
    for (unsigned i = 0; i < pWords; ++i) {
      pTo[i] = pFrom[i];
    }
    pFrom += pWords;
    pTo += cGroupBytes / 4;
  }
}

// Copies the groups of X of a tile into the stage, from group pFirst of X on, in 32-bit words: a copy for every
// number of rows, since with a loop over the words of a group several rows took about twice as long as one.
inline void fStage(const uint8_t *__restrict pX, uint8_t *__restrict pStage, unsigned pRows, unsigned pFirst) {
  const unsigned vWords = fGroupBytesX(pRows) / 4;
  const uint32_t *__restrict vFrom = reinterpret_cast<const uint32_t *>(pX + pFirst * vWords * 4);
  uint32_t *__restrict vTo = reinterpret_cast<uint32_t *>(pStage);
  switch (vWords) {
  case 3:
    fStageWords<3>(vFrom, vTo);
    break;
  case 5:
    fStageWords<5>(vFrom, vTo);
    break;
  case 7:
    fStageWords<7>(vFrom, vTo);
    break;
  case 9:
    fStageWords<9>(vFrom, vTo);
    break;
  case 12:
    fStageWords<12>(vFrom, vTo);
    break;
  case 14:
    fStageWords<14>(vFrom, vTo);
    break;
  case 16:
    fStageWords<16>(vFrom, vTo);
    break;
  default:
    fStageWords<18>(vFrom, vTo);
    break;
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

// The 8 values of 16 bits at pFrom (bf16), each 8 times in a row: lane 8j + t gets value j.
inline aie::vector<bfloat16, 64> fEightTimes(const uint8_t *__restrict pFrom) {
  const aie::vector<uint16, 32> vPair =
      aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pFrom)), aie::zeros<uint16, 16>());
  const auto [v2, v2High] = aie::interleave_zip(vPair, vPair, 1);
  const auto [v4, v4High] = aie::interleave_zip(v2, v2, 2);
  const auto [v8, v8High] = aie::interleave_zip(v4, v4, 4);
  (void) v2High;
  (void) v4High;
  // v8: values 0 to 3, 8 times each; v8High: 4 to 7
  return aie::concat(v8, v8High).template cast_to<bfloat16>();
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
  const aie::vector<uint16, 32> vPairMin = aie::shuffle_down(vPairD, 8);
  const auto [vM2, vM2High] = aie::interleave_zip(vPairMin, vPairMin, 1);
  const auto [vM4, vM4High] = aie::interleave_zip(vM2, vM2, 2);
  const auto [vM8, vM8High] = aie::interleave_zip(vM4, vM4, 4);
  (void) vM2High;
  (void) vM4High;
  const aie::vector<bfloat16, 64> vDmin = aie::concat(vM8, vM8High).template cast_to<bfloat16>();

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

// The piece pIndex of the block, at pPiece: tile pIndex / G, group of outputs pIndex % G.
inline void fPiece(const uint8_t *__restrict pX, const uint8_t *__restrict pPiece, float *__restrict pY,
                   uint8_t *__restrict pStage, unsigned pRows, int32_t pFormat, unsigned pIndex) {
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
  } else {
    fPieceBfp16(pPiece, vSums, pStage, vFirst);
  }
}
} // namespace

extern "C" {

#ifndef XDNA2_GEMV_LOAD_X

void xdna2_gemv_bfp16(uint8_t *__restrict pX, uint8_t *__restrict pW, float *__restrict pY, uint8_t *__restrict pStage,
                      int32_t pRows, int32_t pFirst, int32_t pFormat, int32_t pTiles) {
  int32_t &vNext = fCounter(pStage, cPieceCounter);
  int32_t &vCarried = fCounter(pStage, cCarryCounter);
  if (pFirst) {
    vNext = 0;
    vCarried = 0;
  }
  const unsigned vPieceBytes = fPieceBytes(pFormat);
  const unsigned vPieces = unsigned(pTiles) * cGroupsOut;
  const unsigned vRows = unsigned(pRows);
  unsigned vIndex = unsigned(vNext);
  unsigned vAt = 0;
  // the core starts in the rounding mode floor: the decoded values are rounded to bf16 and bfp16 to the nearest
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  uint8_t *__restrict vCarry = pStage + cCarry;
  if (vCarried > 0 && vIndex < vPieces) {
    // the piece that began in the last object: its first part is in the carry, the rest starts this object
    const unsigned vRest = vPieceBytes - unsigned(vCarried);
    fCopy(pW, vCarry + vCarried, vRest);
    fPiece(pX, vCarry, pY, pStage, vRows, pFormat, vIndex);
    vIndex++;
    vAt = vRest;
  }
  while (vIndex < vPieces && vAt + vPieceBytes <= cObjectBytes) {
    fPiece(pX, pW + vAt, pY, pStage, vRows, pFormat, vIndex);
    vIndex++;
    vAt += vPieceBytes;
  }
  vCarried = 0;
  if (vIndex < vPieces && vAt < cObjectBytes) {
    vCarried = int32_t(cObjectBytes - vAt);
    fCopy(pW + vAt, vCarry, cObjectBytes - vAt);
  }
  vNext = int32_t(vIndex);
  aie::set_rounding(vSaved);
}

#else

void xdna2_gemv_load_x(uint8_t *__restrict pObject, uint8_t *__restrict pX, uint8_t *__restrict pStage, int32_t pFirst) {
  int32_t &vObject = fCounter(pStage, cObjectCounter);
  vObject = pFirst ? 0 : vObject + 1;
  const int32_t *__restrict vFrom = reinterpret_cast<const int32_t *>(pObject);
  int32_t *__restrict vTo = reinterpret_cast<int32_t *>(pX + (unsigned) vObject * cObjectBytes);
  for (unsigned i = 0; i < cObjectBytes / 32; ++i) {
    aie::store_v(vTo + 8 * i, aie::load_v<8>(vFrom + 8 * i));
  }
}

#endif

} // extern "C"
