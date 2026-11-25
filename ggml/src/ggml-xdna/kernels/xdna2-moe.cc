// llama.cpp-xdna: the kernel of the mixture-of-experts overlay (xdna2-moe 1), for the core tiles of the XDNA2 NPU
// (AIE2P). It multiplies C chunks of rows (1 to 24 rows per chunk, in 1 to 3 groups of 8 rows) by their chosen expert
// matrices of weights in one command, with 32-bit float sums and the weights in bfp16, Q4_K, Q5_K, Q6_K, Q8_0 or Q5_1
// decoded in the cores:
//
//   - As in xdna2-gemv-chain.cc, the two cores fed by each of the 16 input channels share the work along K: core 0
//     (row 2 or 4) takes the first half of the tiles of 256 values of K of every block of 32 outputs, core 1 (row 3
//     or 5) takes the second half and sends its sums over the cascade, and core 0 adds them, optionally applies SwiGLU
//     across the two segments of the expert (gate and up), and emits the rows of Y.
//   - Up to 3 groups of 8 rows (24 rows) of X are kept in pX as [tile][group of 8 values of K][row group][72 bytes],
//     so every decoded piece of weights multiplies all 1 to 3 row groups in one pass with one block-stream reader and
//     at most 3 accumulators of 64 floats, without re-reading or re-decoding the weights of the expert.
//   - The first object of the command carries the parameters (first 64 bytes) and the table of chunks (up to 1136
//     32-bit entries (objectsX << 16) | rows, stored in the 7th object slot of pX).
//
//   xdna2_moe_params(pObject, pParams, pX, pStage)                      built with -DXDNA2_MOE_PARAMS
//   xdna2_moe_chunk(pX, pStage, pParams)                                -DXDNA2_MOE_CHUNK
//   xdna2_moe_x(pObject, pX, pStage)                                    -DXDNA2_MOE_X
//   xdna2_moe_gemv(pX, pW, pSums, pStage, pParams, pFormat, pTiles, pBlocks)  -DXDNA2_MOE_GEMV
//   xdna2_moe_send(pSums, pStage)                                       -DXDNA2_MOE_SEND
//   xdna2_moe_finish(pSums, pStage, pChain, pParams)                    -DXDNA2_MOE_FINISH
//   xdna2_moe_emit(pY, pSums, pChain, pStage)                           -DXDNA2_MOE_EMIT

#include <aie_api/aie.hpp>

#include <stdint.h>

namespace {

constexpr unsigned cBlock = 8;
constexpr unsigned cBlockBytes = 9;
constexpr unsigned cLanes = cBlock * cBlock;
constexpr unsigned cGroupBytes = cBlock * cBlockBytes;  // 72
constexpr unsigned cGroupsOut = 4;                      // groups of 8 outputs of a block (32 outputs)
constexpr unsigned cObjectBytes = 4608;                 // bytes of a half-object of a core
constexpr unsigned cTileGroups = 32;                    // groups of 8 values of K in a tile of 256
constexpr unsigned cTileBytesX = cTileGroups * cGroupBytes;  // 2304 bytes per row group and tile of K
constexpr unsigned cSumsPerRowGroup = cGroupsOut * cLanes;   // 256 floats per group of 8 rows
constexpr unsigned cMaxRowGroups = 3;
constexpr unsigned cMaxRows = cMaxRowGroups * cBlock;        // 24 rows per chunk at most
constexpr unsigned cTableOffsetX = 6 * cObjectBytes;         // 7th object slot of pX holds the chunk table
constexpr unsigned cParamsHeaderBytes = 64;                  // 16 int32 parameters at the start of object 0
constexpr unsigned cTableBytes = cObjectBytes - cParamsHeaderBytes;  // 4544 bytes = 1136 uint32 entries

// Offsets inside pStage (5952 bytes: 64B counters, 2048B factors, 2048B bases, 1792B carry)
constexpr unsigned cPieceCounter = 0;
constexpr unsigned cCarryCounter = 4;
constexpr unsigned cObjectCounter = 8;
constexpr unsigned cSegmentCounter = 12;
constexpr unsigned cChunkCounter = 16;
constexpr unsigned cXObjectCounter = 20;
constexpr unsigned cRowsNow = 24;
constexpr unsigned cRowGroupsNow = 28;
constexpr unsigned cBlockCounter = 32;
constexpr unsigned cRowCounter = 36;
constexpr unsigned cEmitFromChain = 40;
constexpr unsigned cFactors = 64;
constexpr unsigned cBases = cFactors + 16 * cLanes * 2;
constexpr unsigned cCarry = cBases + 8 * cLanes * 4;
constexpr unsigned cCarryBytes = 1792;

// Items of pParams (16 int32 values)
constexpr unsigned cParamChunks = 0;
constexpr unsigned cParamTiles = 1;
constexpr unsigned cParamSwiglu = 2;
constexpr unsigned cParamBlocks0 = 3;
constexpr unsigned cParamObjects0 = 4;
constexpr unsigned cParamFormat0 = 5;
constexpr unsigned cParamBlocks1 = 6;
constexpr unsigned cParamObjects1 = 7;
constexpr unsigned cParamFormat1 = 8;
constexpr unsigned cParamTailGroups = 9;
constexpr unsigned cParamTailSubs = 10;
constexpr unsigned cParamObjectsX = 12;
constexpr unsigned cParamEnded = 13;
constexpr unsigned cParamEmitRows = 14;
constexpr unsigned cParamsWords = 16;

// Weight formats and piece sizes
constexpr int32_t cFormatBfp16 = 0;
constexpr int32_t cFormatQ4K = 1;
constexpr int32_t cFormatQ5K = 2;
constexpr int32_t cFormatQ6K = 3;
constexpr int32_t cFormatQ8_0 = 4;
constexpr int32_t cFormatQ5_1 = 5;
constexpr unsigned cPieceBfp16 = 2304;
constexpr unsigned cPieceQ4K = 1152;
constexpr unsigned cPieceQ5K = 1408;
constexpr unsigned cPieceQ6K = 1696;
constexpr unsigned cPieceQ8_0 = 2176;
constexpr unsigned cPieceQ5_1 = 1536;

constexpr unsigned cScalesK = 32;
constexpr unsigned cValuesK = 128;
constexpr unsigned cFifthQ5K = 1152;
constexpr unsigned cFifthQ5_1 = 1280;
constexpr unsigned cScalesQ6K = 32;
constexpr unsigned cValuesQ6K = 160;
constexpr unsigned cTopQ6K = 1184;
constexpr unsigned cValuesQ8_0 = 128;

static_assert(cObjectBytes % 32 == 0 && cTableBytes % 32 == 0 && cPieceQ8_0 % 32 == 0 && cPieceQ5_1 % 32 == 0,
              "aligned to 32 bytes");
static_assert(cCarryBytes >= cPieceQ6K && cCarryBytes >= cPieceQ5_1 && cBases + cPieceQ8_0 <= cCarry + cCarryBytes,
              "the carry holds a piece");

using Xdna2Sums = aie::accum<accfloat, cLanes>;
using Xdna2Groups = aie::block_vector_input_buffer_stream<bfp16ebs8, cLanes>;

inline int32_t & fCounter(uint8_t * pStage, unsigned pOffset) {
  return *reinterpret_cast<int32_t *>(pStage + pOffset);
}

inline unsigned fPieceBytes(int32_t pFormat, unsigned pSubs) {
  if (pFormat == cFormatQ8_0) {
    return (pSubs >> 1) * 32 + pSubs * 256;
  }
  if (pFormat == cFormatQ5_1) {
    return (pSubs >> 1) * 64 + pSubs * 160;
  }
  return pFormat == cFormatQ4K ? cPieceQ4K :
         pFormat == cFormatQ5K ? cPieceQ5K :
         pFormat == cFormatQ6K ? cPieceQ6K :
                                 cPieceBfp16;
}

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

template <unsigned pRowGroups>
inline void fRunBfp16(const uint8_t *__restrict pPiece, const uint8_t *__restrict pTileX, float *__restrict pSums,
                      bool pFirst) {
  Xdna2Groups vStreamW(reinterpret_cast<bfp16ebs8 *>(const_cast<uint8_t *>(pPiece)));
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(const_cast<uint8_t *>(pTileX)));
  Xdna2Sums vSum0 = fLoadSums(pSums, pFirst);
  Xdna2Sums vSum1 = pRowGroups >= 2 ? fLoadSums(pSums + cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
  Xdna2Sums vSum2 = pRowGroups >= 3 ? fLoadSums(pSums + 2 * cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
#pragma clang loop unroll_count(4)
  for (unsigned vGroup = 0; vGroup < cTileGroups; ++vGroup) {
    const auto vW = vStreamW.pop();
    vSum0 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum0);
    if constexpr (pRowGroups >= 2) {
      vSum1 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum1);
    }
    if constexpr (pRowGroups >= 3) {
      vSum2 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum2);
    }
  }
  aie::store_v(pSums, vSum0.template to_vector<float>());
  if constexpr (pRowGroups >= 2) {
    aie::store_v(pSums + cSumsPerRowGroup, vSum1.template to_vector<float>());
  }
  if constexpr (pRowGroups >= 3) {
    aie::store_v(pSums + 2 * cSumsPerRowGroup, vSum2.template to_vector<float>());
  }
}

template <typename T>
inline aie::vector<T, 64> fRepeat8(const aie::vector<T, 64> & pValues) {
  const auto [vTwo, vTwoHigh] = aie::interleave_zip(pValues, pValues, 1);
  const auto [vFour, vFourHigh] = aie::interleave_zip(vTwo, vTwo, 2);
  const auto [vEight, vEightHigh] = aie::interleave_zip(vFour, vFour, 4);
  (void) vTwoHigh;
  (void) vFourHigh;
  (void) vEightHigh;
  return vEight;
}

inline aie::vector<bfloat16, 64> fPlus128(const aie::vector<uint8, 64> & pQ) {
  const aie::vector<uint8, 64> v43 = aie::broadcast<uint8, 64>(uint8(0x43));
  const auto [vLow, vHigh] = aie::interleave_zip(pQ, v43, 1);
  return aie::concat(vLow, vHigh).template cast_to<bfloat16>();
}

inline aie::vector<bfloat16, 64> fRepeat8From16(const aie::vector<uint16, 32> & pPair) {
  const auto [v2, v2High] = aie::interleave_zip(pPair, pPair, 1);
  const auto [v4, v4High] = aie::interleave_zip(v2, v2, 2);
  const auto [v8, v8High] = aie::interleave_zip(v4, v4, 4);
  (void) v2High;
  (void) v4High;
  return aie::concat(v8, v8High).template cast_to<bfloat16>();
}

inline aie::vector<bfloat16, 64> fEightTimes(const uint8_t *__restrict pFrom) {
  return fRepeat8From16(
      aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pFrom)), aie::zeros<uint16, 16>()));
}

void fPrepareK(const uint8_t *__restrict pPiece, uint8_t *__restrict pStage) {
  const aie::vector<bfloat16, 64> v128 = aie::broadcast<bfloat16, 64>(bfloat16(128.0f));
  const aie::vector<uint8, 64> vLow4 = aie::broadcast<uint8, 64>(uint8(0x0f));
  const aie::vector<uint8, 64> vLow6 = aie::broadcast<uint8, 64>(uint8(0x3f));
  const aie::vector<uint8, 32> vBytes0 = aie::load_v<32>(pPiece + cScalesK);
  const aie::vector<uint8, 32> vBytes4 = aie::load_v<32>(pPiece + cScalesK + 32);
  const aie::vector<uint8, 32> vBytes8 = aie::load_v<32>(pPiece + cScalesK + 64);
  const aie::vector<uint8, 64> vLowHalf = aie::concat(vBytes0, vBytes4);
  const aie::vector<uint8, 64> vTop = aie::downshift(vLowHalf, 6);
  const aie::vector<uint8, 64> vHighHalf =
      aie::concat(aie::bit_and(vBytes8, vLow4.extract<32>(0)), aie::downshift(vBytes8, 4));
  const aie::vector<uint8, 64> vSixLow = aie::bit_and(vLowHalf, vLow6);
  const aie::vector<uint8, 64> vSixHigh = aie::bit_or(vHighHalf, aie::upshift(vTop, 4));
  const aie::vector<uint8, 64> vScales = aie::concat(vSixLow.extract<32>(0), vSixHigh.extract<32>(0));
  const aie::vector<uint8, 64> vMins = aie::concat(vSixLow.extract<32>(1), vSixHigh.extract<32>(1));
  const aie::vector<bfloat16, 64> vD = fEightTimes(pPiece);
  const aie::vector<uint16, 32> vPairD =
      aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece)), aie::zeros<uint16, 16>());
  const aie::vector<bfloat16, 64> vDmin = fRepeat8From16(aie::shuffle_down(vPairD, 8));

  bfloat16 *__restrict vFactors = reinterpret_cast<bfloat16 *>(pStage + cFactors);
  float *__restrict vBases = reinterpret_cast<float *>(pStage + cBases);
  for (unsigned vSub = 0; vSub < 8; ++vSub) {
    const aie::vector<bfloat16, 64> vScale = fPlus128(fRepeat8(aie::shuffle_down(vScales, vSub * 8)));
    const aie::vector<bfloat16, 64> vMinimum = fPlus128(fRepeat8(aie::shuffle_down(vMins, vSub * 8)));
    const aie::vector<bfloat16, 64> vFactor = aie::msc(aie::mul(vD, vScale), vD, v128).template to_vector<bfloat16>();
    const Xdna2Sums vBase = aie::mac(aie::msc(aie::negmul(vFactor, v128), vDmin, vMinimum), vDmin, v128);
    aie::store_v(vFactors + vSub * cLanes, vFactor);
    aie::store_v(vBases + vSub * cLanes, vBase.template to_vector<float>());
  }
}

template <unsigned pRowGroups>
inline void fRunK(const uint8_t *__restrict pValues, const uint8_t *__restrict pFifth, const uint8_t *__restrict pTileX,
                  float *__restrict pSums, const bfloat16 *__restrict pFactors, const float *__restrict pBases,
                  unsigned pSubs, bool pFirst) {
  const aie::vector<uint8, 64> v16 = aie::broadcast<uint8, 64>(uint8(16));
  Xdna2Sums vSum0 = fLoadSums(pSums, pFirst);
  Xdna2Sums vSum1 = pRowGroups >= 2 ? fLoadSums(pSums + cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
  Xdna2Sums vSum2 = pRowGroups >= 3 ? fLoadSums(pSums + 2 * cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(const_cast<uint8_t *>(pTileX)));
  for (unsigned vSub = 0; vSub < pSubs; ++vSub) {
    const aie::vector<bfloat16, 64> vFactor = aie::load_v<cLanes>(pFactors + vSub * cLanes);
    Xdna2Sums vBase;
    vBase.from_vector(aie::load_v<cLanes>(pBases + vSub * cLanes));
    if (pFifth == nullptr) {
#pragma clang loop unroll(full)
      for (unsigned vGroup = 0; vGroup < 4; ++vGroup) {
        const aie::vector<uint8, 64> vQ =
            aie::load_v<64>(reinterpret_cast<const uint4 *>(pValues + (vSub * 4 + vGroup) * 32)).unpack();
        const auto vW = aie::mac(vBase, fPlus128(vQ), vFactor).template to_vector<bfp16ebs8>();
        vSum0 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum0);
        if constexpr (pRowGroups >= 2) {
          vSum1 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum1);
        }
        if constexpr (pRowGroups >= 3) {
          vSum2 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum2);
        }
      }
    } else {
      const uint32_t *__restrict vFifth = reinterpret_cast<const uint32_t *>(pFifth + vSub * 32);
#pragma clang loop unroll(full)
      for (unsigned vGroup = 0; vGroup < 4; ++vGroup) {
        const aie::vector<uint8, 64> vLow =
            aie::load_v<64>(reinterpret_cast<const uint4 *>(pValues + (vSub * 4 + vGroup) * 32)).unpack();
        const aie::mask<64> vFive = aie::mask<64>::from_uint32(vFifth[2 * vGroup], vFifth[2 * vGroup + 1]);
        const aie::vector<uint8, 64> vQ = aie::select(vLow, aie::bit_or(vLow, v16), vFive);
        const auto vW = aie::mac(vBase, fPlus128(vQ), vFactor).template to_vector<bfp16ebs8>();
        vSum0 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum0);
        if constexpr (pRowGroups >= 2) {
          vSum1 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum1);
        }
        if constexpr (pRowGroups >= 3) {
          vSum2 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum2);
        }
      }
    }
  }
  aie::store_v(pSums, vSum0.template to_vector<float>());
  if constexpr (pRowGroups >= 2) {
    aie::store_v(pSums + cSumsPerRowGroup, vSum1.template to_vector<float>());
  }
  if constexpr (pRowGroups >= 3) {
    aie::store_v(pSums + 2 * cSumsPerRowGroup, vSum2.template to_vector<float>());
  }
}

void fPrepareQ6K(const uint8_t *__restrict pPiece, uint8_t *__restrict pStage) {
  const aie::vector<bfloat16, 64> v128 = aie::broadcast<bfloat16, 64>(bfloat16(128.0f));
  const aie::vector<bfloat16, 64> v256 = aie::broadcast<bfloat16, 64>(bfloat16(256.0f));
  const aie::vector<uint8, 64> vLow7 = aie::broadcast<uint8, 64>(uint8(0x7f));
  const aie::vector<uint16, 16> vD8 = aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece));
  const aie::vector<uint16, 16> vD8Twice = aie::concat(vD8.extract<8>(0), vD8.extract<8>(0));
  const aie::vector<uint16, 32> vD32 = aie::concat(vD8Twice, vD8Twice);
  const aie::vector<bfloat16, 64> vD = aie::concat(vD32, vD32).template cast_to<bfloat16>();
  bfloat16 *__restrict vFactors = reinterpret_cast<bfloat16 *>(pStage + cFactors);
  for (unsigned vHalf = 0; vHalf < 2; ++vHalf) {
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

template <unsigned pRowGroups>
inline void fRunQ6K(const uint8_t *__restrict pPiece, const uint8_t *__restrict pTileX, float *__restrict pSums,
                    const bfloat16 *__restrict pFactors, bool pFirst) {
  const aie::vector<bfloat16, 64> vMinus160 = aie::broadcast<bfloat16, 64>(bfloat16(-160.0f));
  const aie::vector<uint8, 64> vLow2 = aie::broadcast<uint8, 64>(uint8(3));
  Xdna2Sums vSum0 = fLoadSums(pSums, pFirst);
  Xdna2Sums vSum1 = pRowGroups >= 2 ? fLoadSums(pSums + cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
  Xdna2Sums vSum2 = pRowGroups >= 3 ? fLoadSums(pSums + 2 * cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(const_cast<uint8_t *>(pTileX)));
  for (unsigned vSub = 0; vSub < 16; ++vSub) {
    const aie::vector<bfloat16, 64> vFactor = aie::load_v<cLanes>(pFactors + vSub * cLanes);
    const Xdna2Sums vBase = aie::mul(vFactor, vMinus160);
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
      const auto vW = aie::mac(vBase, fPlus128(vQ), vFactor).template to_vector<bfp16ebs8>();
      vSum0 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum0);
      if constexpr (pRowGroups >= 2) {
        vSum1 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum1);
      }
      if constexpr (pRowGroups >= 3) {
        vSum2 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum2);
      }
    }
  }
  aie::store_v(pSums, vSum0.template to_vector<float>());
  if constexpr (pRowGroups >= 2) {
    aie::store_v(pSums + cSumsPerRowGroup, vSum1.template to_vector<float>());
  }
  if constexpr (pRowGroups >= 3) {
    aie::store_v(pSums + 2 * cSumsPerRowGroup, vSum2.template to_vector<float>());
  }
}

template <unsigned pRowGroups>
inline void fRunQ8_0(const uint8_t *__restrict pPiece, const uint8_t *__restrict pTileX, float *__restrict pSums,
                     unsigned pPairs, bool pFirst) {
  const aie::vector<bfloat16, 64> v128    = aie::broadcast<bfloat16, 64>(bfloat16(128.0f));
  const aie::vector<bfloat16, 64> v256    = aie::broadcast<bfloat16, 64>(bfloat16(256.0f));
  const aie::vector<uint8, 64>    vLow7   = aie::broadcast<uint8, 64>(uint8(0x7f));
  const uint8_t *__restrict       vValues = pPiece + pPairs * 32;
  Xdna2Sums vSum0 = fLoadSums(pSums, pFirst);
  Xdna2Sums vSum1 = pRowGroups >= 2 ? fLoadSums(pSums + cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
  Xdna2Sums vSum2 = pRowGroups >= 3 ? fLoadSums(pSums + 2 * cSumsPerRowGroup, pFirst) : aie::zeros<accfloat, cLanes>();
  Xdna2Groups vStreamX(reinterpret_cast<bfp16ebs8 *>(const_cast<uint8_t *>(pTileX)));
  for (unsigned vPair = 0; vPair < pPairs; ++vPair) {
    const aie::vector<uint16, 32> vPairD =
        aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece + vPair * 32)), aie::zeros<uint16, 16>());
    for (unsigned vHalf = 0; vHalf < 2; ++vHalf) {
      const aie::vector<bfloat16, 64> vFactor = fRepeat8From16(vHalf == 0 ? vPairD : aie::shuffle_down(vPairD, 8));
      const unsigned vSub = vPair * 2 + vHalf;
#pragma clang loop unroll(full)
      for (unsigned vGroup = 0; vGroup < 4; ++vGroup) {
        const uint8_t *__restrict vAt = vValues + (vSub * 4 + vGroup) * 64;
        const aie::vector<uint8, 64> vU = aie::concat(aie::load_v<32>(vAt), aie::load_v<32>(vAt + 32));
        const aie::vector<bfloat16, 64> vC = aie::select(v128, v256, aie::ge(vU, uint8(128)));
        const auto vW =
            aie::msc(aie::mul(fPlus128(aie::bit_and(vU, vLow7)), vFactor), vC, vFactor).template to_vector<bfp16ebs8>();
        vSum0 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum0);
        if constexpr (pRowGroups >= 2) {
          vSum1 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum1);
        }
        if constexpr (pRowGroups >= 3) {
          vSum2 = mac_8x8_8x8T(vStreamX.pop(), vW, vSum2);
        }
      }
    }
  }
  aie::store_v(pSums, vSum0.template to_vector<float>());
  if constexpr (pRowGroups >= 2) {
    aie::store_v(pSums + cSumsPerRowGroup, vSum1.template to_vector<float>());
  }
  if constexpr (pRowGroups >= 3) {
    aie::store_v(pSums + 2 * cSumsPerRowGroup, vSum2.template to_vector<float>());
  }
}

__attribute__((noinline)) void fPrepareQ5_1(const uint8_t *__restrict pPiece, uint8_t *__restrict pStage,
                                            unsigned pPairs) {
  const aie::vector<bfloat16, 64> vMinus128 = aie::broadcast<bfloat16, 64>(bfloat16(-128.0f));
  const aie::vector<bfloat16, 64> vOne      = aie::broadcast<bfloat16, 64>(bfloat16(1.0f));
  bfloat16 *__restrict            vFactors  = reinterpret_cast<bfloat16 *>(pStage + cFactors);
  float *__restrict               vBases    = reinterpret_cast<float *>(pStage + cBases);
  const unsigned                  vMinsOff  = pPairs * 32;
  for (unsigned vPair = 0; vPair < pPairs; ++vPair) {
    const aie::vector<uint16, 32> vPairD =
        aie::concat(aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece + vPair * 32)), aie::zeros<uint16, 16>());
    const aie::vector<uint16, 32> vPairM = aie::concat(
        aie::load_v<16>(reinterpret_cast<const uint16 *>(pPiece + vMinsOff + vPair * 32)), aie::zeros<uint16, 16>());
    for (unsigned vHalf = 0; vHalf < 2; ++vHalf) {
      const aie::vector<bfloat16, 64> vFactor = fRepeat8From16(vHalf == 0 ? vPairD : aie::shuffle_down(vPairD, 8));
      const aie::vector<bfloat16, 64> vMin    = fRepeat8From16(vHalf == 0 ? vPairM : aie::shuffle_down(vPairM, 8));
      const Xdna2Sums                 vBase   = aie::mac(aie::mul(vMin, vOne), vFactor, vMinus128);
      const unsigned                  vSub    = vPair * 2 + vHalf;
      aie::store_v(vFactors + vSub * cLanes, vFactor);
      aie::store_v(vBases + vSub * cLanes, vBase.template to_vector<float>());
    }
  }
}

__attribute__((noinline)) void fPiece(const uint8_t *__restrict pX, const uint8_t *__restrict pPiece,
                                      float *__restrict pSums, uint8_t *__restrict pStage, unsigned pRowGroups,
                                      unsigned pGroupsOut, unsigned pSubs, int32_t pFormat, unsigned pIndex) {
#if defined(XDNA2_MOE_NO_COMPUTE)
  (void) pX;
  (void) pPiece;
  (void) pSums;
  (void) pStage;
  (void) pRowGroups;
  (void) pGroupsOut;
  (void) pSubs;
  (void) pFormat;
  (void) pIndex;
  return;
#endif
  unsigned vTile  = 0;
  unsigned vGroup = pIndex;
  if (pGroupsOut == cGroupsOut) {
    vTile  = pIndex >> 2;
    vGroup = pIndex & 3u;
  } else {
    while (vGroup >= pGroupsOut) {
      vGroup -= pGroupsOut;
      ++vTile;
    }
  }
  const uint8_t *__restrict vTileX = pX + vTile * pRowGroups * cTileBytesX;
  float *__restrict vSums = pSums + vGroup * cLanes;
  const bool vFirst = vTile == 0;
  const bfloat16 *__restrict vFactors = reinterpret_cast<const bfloat16 *>(pStage + cFactors);
  const float *__restrict vBases = reinterpret_cast<const float *>(pStage + cBases);

  if (pFormat == cFormatQ6K) {
    fPrepareQ6K(pPiece, pStage);
    if (pRowGroups == 1) {
      fRunQ6K<1>(pPiece, vTileX, vSums, vFactors, vFirst);
    } else if (pRowGroups == 2) {
      fRunQ6K<2>(pPiece, vTileX, vSums, vFactors, vFirst);
    } else {
      fRunQ6K<3>(pPiece, vTileX, vSums, vFactors, vFirst);
    }
  } else if (pFormat == cFormatQ8_0) {
    const unsigned vPairs = pSubs >> 1;
    if (pRowGroups == 1) {
      fRunQ8_0<1>(pPiece, vTileX, vSums, vPairs, vFirst);
    } else if (pRowGroups == 2) {
      fRunQ8_0<2>(pPiece, vTileX, vSums, vPairs, vFirst);
    } else {
      fRunQ8_0<3>(pPiece, vTileX, vSums, vPairs, vFirst);
    }
  } else if (pFormat == cFormatQ4K || pFormat == cFormatQ5K || pFormat == cFormatQ5_1) {
    const unsigned            vPairs  = pSubs >> 1;
    const uint8_t *__restrict vValues = pFormat == cFormatQ5_1 ? pPiece + vPairs * 64 : pPiece + cValuesK;
    const uint8_t *__restrict vFifth  = pFormat == cFormatQ5_1 ? pPiece + vPairs * 64 + pSubs * 128 :
                                        pFormat == cFormatQ5K  ? pPiece + cFifthQ5K :
                                                                 nullptr;
    if (pFormat == cFormatQ5_1) {
      fPrepareQ5_1(pPiece, pStage, vPairs);
    } else {
      fPrepareK(pPiece, pStage);
    }
    if (pRowGroups == 1) {
      fRunK<1>(vValues, vFifth, vTileX, vSums, vFactors, vBases, pSubs, vFirst);
    } else if (pRowGroups == 2) {
      fRunK<2>(vValues, vFifth, vTileX, vSums, vFactors, vBases, pSubs, vFirst);
    } else {
      fRunK<3>(vValues, vFifth, vTileX, vSums, vFactors, vBases, pSubs, vFirst);
    }
  } else {
    if (pRowGroups == 1) {
      fRunBfp16<1>(pPiece, vTileX, vSums, vFirst);
    } else if (pRowGroups == 2) {
      fRunBfp16<2>(pPiece, vTileX, vSums, vFirst);
    } else {
      fRunBfp16<3>(pPiece, vTileX, vSums, vFirst);
    }
  }
}

}  // namespace

extern "C" {

#if defined(XDNA2_MOE_PARAMS)

void xdna2_moe_params(uint8_t *__restrict pObject, int32_t *__restrict pParams, uint8_t *__restrict pX,
                      uint8_t *__restrict pStage) {
  const int32_t *__restrict vFrom = reinterpret_cast<const int32_t *>(pObject);
  for (unsigned i = 0; i < cParamsWords; i += 8) {
    aie::store_v(pParams + i, aie::load_v<8>(vFrom + i));
  }
  fCopy(pObject + cParamsHeaderBytes, pX + cTableOffsetX, cTableBytes);
  fCounter(pStage, cChunkCounter) = 0;
}

#elif defined(XDNA2_MOE_CHUNK)

void xdna2_moe_chunk(uint8_t *__restrict pX, uint8_t *__restrict pStage, int32_t *__restrict pParams) {
  int32_t & vChunk = fCounter(pStage, cChunkCounter);
  const uint32_t vEntry = reinterpret_cast<const uint32_t *>(pX + cTableOffsetX)[unsigned(vChunk)];
  vChunk++;
  const unsigned vRows = vEntry & 0xffffu;
  const unsigned vObjectsX = vEntry >> 16;
  pParams[cParamObjectsX] = int32_t(vObjectsX);
  pParams[cParamEnded] = 0;
  pParams[cParamEmitRows] = 0;
  fCounter(pStage, cRowsNow) = int32_t(vRows);
  fCounter(pStage, cRowGroupsNow) = int32_t((vRows + cBlock - 1) / cBlock);
  fCounter(pStage, cXObjectCounter) = 0;
  fCounter(pStage, cPieceCounter) = 0;
  fCounter(pStage, cCarryCounter) = 0;
  fCounter(pStage, cObjectCounter) = 0;
  fCounter(pStage, cSegmentCounter) = 0;
  fCounter(pStage, cBlockCounter) = 0;
}

#elif defined(XDNA2_MOE_X)

void xdna2_moe_x(uint8_t *__restrict pObject, uint8_t *__restrict pX, uint8_t *__restrict pStage) {
  int32_t & vIndex = fCounter(pStage, cXObjectCounter);
  fCopy(pObject, pX + unsigned(vIndex) * cObjectBytes, cObjectBytes);
  vIndex++;
}

#elif defined(XDNA2_MOE_GEMV)

void xdna2_moe_gemv(uint8_t *__restrict pX, uint8_t *__restrict pW, float *__restrict pSums,
                    uint8_t *__restrict pStage, int32_t *__restrict pParams, int32_t pFormat, int32_t pTiles,
                    int32_t pBlocks) {
  int32_t & vNext = fCounter(pStage, cPieceCounter);
  int32_t & vCarried = fCounter(pStage, cCarryCounter);
  int32_t & vResume = fCounter(pStage, cObjectCounter);
  int32_t & vEnded = fCounter(pStage, cSegmentCounter);
  pParams[cParamEnded] = 0;
  if (vEnded >= pBlocks) {
    vNext = 0;
    vCarried = 0;
    vResume = 0;
    vEnded = 0;
    return;
  }
  const unsigned vTailRaw        = unsigned(pParams[cParamTailGroups]);
  const unsigned vGroupsNow      = (vEnded + 1 == pBlocks && vTailRaw >= 1 && vTailRaw <= cGroupsOut) ? vTailRaw : cGroupsOut;
  const unsigned vPieces         = unsigned(pTiles) * vGroupsNow;
  const unsigned vTailSubsRaw    = unsigned(pParams[cParamTailSubs]);
  const unsigned vTailSubs       = (vTailSubsRaw >= 2 && vTailSubsRaw <= 8 && (vTailSubsRaw & 1u) == 0 &&
                                    (pFormat == cFormatQ8_0 || pFormat == cFormatQ5_1)) ?
                                       vTailSubsRaw :
                                       8u;
  const unsigned vTailPieceStart = unsigned(pTiles - 1) * vGroupsNow;
  const unsigned vRowGroups      = unsigned(fCounter(pStage, cRowGroupsNow));
  unsigned vIndex = unsigned(vNext);
  unsigned vAt = unsigned(vResume);
  if (vIndex == 0 && vCarried == 0 && vGroupsNow < cGroupsOut) {
    for (unsigned i = 0; i < cMaxRowGroups * cSumsPerRowGroup; i += 16) {
      aie::store_v(pSums + i, aie::zeros<float, 16>());
    }
  }
  bool vBlockEnds = false;
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  uint8_t *__restrict vCarry = pStage + (pFormat == cFormatQ8_0 ? cBases : cCarry);
  while (!vBlockEnds) {
    const unsigned  vSubsNow    = vIndex >= vTailPieceStart ? vTailSubs : 8u;
    const unsigned  vPieceBytes = fPieceBytes(pFormat, vSubsNow);
    const uint8_t * vPiece      = pW + vAt;
    unsigned        vNextAt     = vAt + vPieceBytes;
    if (vAt == 0 && vCarried > 0) {
      const unsigned vRest = vPieceBytes - unsigned(vCarried);
      fCopy(pW, vCarry + vCarried, vRest);
      vPiece = vCarry;
      vNextAt = vRest;
      vCarried = 0;
    } else if (vNextAt > cObjectBytes) {
      break;
    }
    fPiece(pX, vPiece, pSums, pStage, vRowGroups, vGroupsNow, vSubsNow, pFormat, vIndex);
    vIndex++;
    vAt = vNextAt;
    vBlockEnds = vIndex == vPieces;
  }
  if (vBlockEnds) {
    vNext = 0;
    vResume = int32_t(vAt);
    vEnded++;
    pParams[cParamEnded] = 1;
  } else {
    vNext = int32_t(vIndex);
    vResume = 0;
    if (vAt < cObjectBytes) {
      vCarried = int32_t(cObjectBytes - vAt);
      fCopy(pW + vAt, vCarry, cObjectBytes - vAt);
    }
  }
  aie::set_rounding(vSaved);
}

#elif defined(XDNA2_MOE_SEND)

void xdna2_moe_send(float *__restrict pSums, uint8_t *__restrict pStage) {
  const unsigned vFloats = unsigned(fCounter(pStage, cRowGroupsNow)) * cSumsPerRowGroup;
  for (unsigned i = 0; i < vFloats; i += 16) {
    put_mcd(v16float(aie::load_v<16>(pSums + i)));
  }
}

#elif defined(XDNA2_MOE_FINISH)

void xdna2_moe_finish(float *__restrict pSums, uint8_t *__restrict pStage, float *__restrict pChain,
                      int32_t *__restrict pParams) {
  const unsigned vRows = unsigned(fCounter(pStage, cRowsNow));
  const unsigned vFloats = unsigned(fCounter(pStage, cRowGroupsNow)) * cSumsPerRowGroup;
  fCounter(pStage, cRowCounter) = 0;
  for (unsigned i = 0; i < vFloats; i += 16) {
    const aie::vector<float, 16> vSum = aie::add(aie::load_v<16>(pSums + i), aie::vector<float, 16>(get_scd_v16float()));
    aie::store_v(pSums + i, vSum);
  }
  if (pParams[cParamSwiglu] == 0) {
    fCounter(pStage, cEmitFromChain) = 0;
    pParams[cParamEmitRows] = int32_t(vRows);
    return;
  }
  int32_t & vBlock = fCounter(pStage, cBlockCounter);
  const unsigned vBlocks0 = unsigned(pParams[cParamBlocks0]);
  if (unsigned(vBlock) < vBlocks0) {
    float *__restrict vGateBase = pChain + unsigned(vBlock) * vRows * 32;
    for (unsigned m = 0; m < vRows; ++m) {
      const unsigned vBase = (m / cBlock) * cSumsPerRowGroup + (m % cBlock) * cBlock;
      for (unsigned o = 0; o < cGroupsOut; ++o) {
        aie::store_v(vGateBase + m * 32 + 8 * o, aie::load_v<8>(pSums + vBase + o * cLanes));
      }
    }
    vBlock++;
    pParams[cParamEmitRows] = 0;
    return;
  }
  const aie::rounding_mode vSaved = aie::swap_rounding(aie::rounding_mode::conv_even);
  const unsigned vUpBlock = unsigned(vBlock) - vBlocks0;
  const float *__restrict vGateBase = pChain + vUpBlock * vRows * 32;
  alignas(32) float aUp[32];
  for (unsigned m = 0; m < vRows; ++m) {
    const unsigned vBase = (m / cBlock) * cSumsPerRowGroup + (m % cBlock) * cBlock;
    for (unsigned o = 0; o < cGroupsOut; ++o) {
      aie::store_v(aUp + 8 * o, aie::load_v<8>(pSums + vBase + o * cLanes));
    }
    const aie::vector<float, 32> vG = aie::load_v<32>(vGateBase + m * 32);
    const aie::vector<float, 32> vMinusLog = aie::mul(vG, -1.4426950408889634f).template to_vector<float>();
    const aie::vector<float, 32> vExp =
        aie::mul(aie::exp2<bfloat16>(vMinusLog), bfloat16(1.0f)).template to_vector<float>();
    const aie::vector<float, 32> vDen = aie::add(vExp, 1.0f);
    const aie::vector<float, 32> vSilu = aie::mul(vG, aie::inv(vDen)).template to_vector<float>();
    aie::store_v(pChain + m * 32, aie::mul(vSilu, aie::load_v<32>(aUp)).template to_vector<float>());
  }
  aie::set_rounding(vSaved);
  vBlock++;
  fCounter(pStage, cEmitFromChain) = 1;
  pParams[cParamEmitRows] = int32_t(vRows);
}

#elif defined(XDNA2_MOE_EMIT)

void xdna2_moe_emit(float *__restrict pY, float *__restrict pSums, float *__restrict pChain,
                    uint8_t *__restrict pStage) {
  int32_t & vRow = fCounter(pStage, cRowCounter);
  const unsigned m = unsigned(vRow);
  vRow++;
  if (fCounter(pStage, cEmitFromChain) != 0) {
    const float *__restrict vFrom = pChain + m * 32;
    aie::store_v(pY, aie::load_v<16>(vFrom));
    aie::store_v(pY + 16, aie::load_v<16>(vFrom + 16));
    return;
  }
  const unsigned vBase = (m / cBlock) * cSumsPerRowGroup + (m % cBlock) * cBlock;
  for (unsigned o = 0; o < cGroupsOut; ++o) {
    aie::store_v(pY + cBlock * o, aie::load_v<8>(pSums + vBase + o * cLanes));
  }
}

#endif

}  // extern "C"
