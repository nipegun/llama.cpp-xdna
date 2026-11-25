#include "xdna-moe.h"

#include "xdna-bfp16.h"
#include "xdna-insts.h"
#include "xdna-workers.h"

#include <algorithm>
#include <atomic>
#include <cstring>
#include <fstream>
#include <sstream>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace {

constexpr uint32_t cManifestVersion   = 1;
constexpr uint32_t cStreamsExpected   = 16;
constexpr uint32_t cCoreBytesExpected = 4608;
constexpr uint32_t cBlock             = 8;
constexpr uint32_t cBlockBytes        = 9;
constexpr uint32_t cGroupBytes        = cBlock * cBlockBytes;  // 72 bytes (4608 = 64 * 72)
constexpr uint32_t cGroupsOut         = 4;
constexpr uint32_t cBlockOutputs      = 32;
constexpr uint32_t cGroupOutputs      = 8;
constexpr uint32_t cSuper             = 256;
constexpr uint32_t cTileBytesX        = 32 * cGroupBytes;      // 2304 bytes per row-group and tile of K
constexpr uint32_t cParamsWords       = 16;
constexpr uint32_t cParamsHeaderBytes = cParamsWords * sizeof(int32_t);
constexpr uint64_t cPageBytes         = 4096;

constexpr uint32_t cBufferIo = 0;
constexpr uint32_t cBufferW  = 1;

constexpr uint32_t cParamChunks   = 0;
constexpr uint32_t cParamTiles    = 1;
constexpr uint32_t cParamSwiglu   = 2;
constexpr uint32_t cParamBlocks0  = 3;
constexpr uint32_t cParamObjects0 = 4;
constexpr uint32_t cParamFormat0  = 5;
constexpr uint32_t cParamBlocks1    = 6;
constexpr uint32_t cParamObjects1   = 7;
constexpr uint32_t cParamFormat1    = 8;
constexpr uint32_t cParamTailGroups = 9;
constexpr uint32_t cParamTailSubs   = 10;

uint64_t fRoundUp(uint64_t pValue, uint64_t pStep) {
  return (pValue + pStep - 1) / pStep * pStep;
}

bool fSetStream(std::vector<XdnaOverlayStream> & aStreams, uint32_t pNumber, uint32_t pColumn, uint32_t pChannel) {
  if (pNumber >= cStreamsExpected || pColumn > 255 || pChannel > 255) {
    return false;
  }
  if (aStreams.size() <= pNumber) {
    aStreams.resize(pNumber + 1, XdnaOverlayStream{255, 255});
  }
  aStreams[pNumber] = {uint8_t(pColumn), uint8_t(pChannel)};
  return true;
}

bool fComplete(const std::vector<XdnaOverlayStream> & aStreams) {
  if (aStreams.size() != cStreamsExpected) {
    return false;
  }
  for (const XdnaOverlayStream & vStream : aStreams) {
    if (vStream.vColumn == 255) {
      return false;
    }
  }
  return true;
}

uint8_t * fGroupPtrX(uint8_t * pChunkX, const XdnaMoeOverlay & pOverlay, uint32_t pCore, uint32_t pRowGroups,
                     uint32_t pLocalGroupK, uint32_t pRg) {
  const uint64_t vCoreOffset = (uint64_t(pLocalGroupK) * pRowGroups + pRg) * cGroupBytes;
  const uint64_t vObject     = vCoreOffset / pOverlay.vCoreBytes;
  const uint64_t vWithin     = vCoreOffset % pOverlay.vCoreBytes;
  return pChunkX + vObject * pOverlay.vObjectBytes + uint64_t(pCore) * pOverlay.vCoreBytes + vWithin;
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

__attribute__((target("avx2,fma"))) void fScaleWriteAvx2(const float * pSrc, float * pDst, uint32_t pCount,
                                                         float pScale, bool pAccumulate) {
  const __m256 vScale = _mm256_set1_ps(pScale);
  uint32_t     i      = 0;
  if (pAccumulate) {
    for (; i + 8 <= pCount; i += 8) {
      const __m256 vPrev = _mm256_loadu_ps(pDst + i);
      _mm256_storeu_ps(pDst + i, _mm256_fmadd_ps(_mm256_loadu_ps(pSrc + i), vScale, vPrev));
    }
  } else if (pScale == 1.0f) {
    std::memcpy(pDst, pSrc, size_t(pCount) * sizeof(float));
    return;
  } else {
    for (; i + 8 <= pCount; i += 8) {
      _mm256_storeu_ps(pDst + i, _mm256_mul_ps(_mm256_loadu_ps(pSrc + i), vScale));
    }
  }
  for (; i < pCount; i++) {
    if (pAccumulate) {
      pDst[i] += pScale * pSrc[i];
    } else {
      pDst[i] = pScale * pSrc[i];
    }
  }
}

__attribute__((target("avx2,fma"))) void fAddCopyAvx2(const float * pSrc, const float * pAlso, float * pDst,
                                                        uint32_t pCount) {
  uint32_t i = 0;
  for (; i + 8 <= pCount; i += 8) {
    _mm256_storeu_ps(pDst + i, _mm256_add_ps(_mm256_loadu_ps(pSrc + i), _mm256_loadu_ps(pAlso + i)));
  }
  for (; i < pCount; i++) {
    pDst[i] = pSrc[i] + pAlso[i];
  }
}

__attribute__((target("avx2"))) void fNormRowAvx2(const float * pSrc, const float * pWeights, float * pDst,
                                                  uint32_t pCount, float pScale) {
  const __m256 vScale = _mm256_set1_ps(pScale);
  uint32_t     i      = 0;
  for (; i + 8 <= pCount; i += 8) {
    const __m256 vNormed = _mm256_mul_ps(_mm256_loadu_ps(pSrc + i), vScale);
    _mm256_storeu_ps(pDst + i, _mm256_mul_ps(vNormed, _mm256_loadu_ps(pWeights + i)));
  }
  for (; i < pCount; i++) {
    pDst[i] = (pSrc[i] * pScale) * pWeights[i];
  }
}

#endif

void fAddCopy(const float * pSrc, const float * pAlso, float * pDst, uint32_t pCount) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool cHasAvx2Fma = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  if (cHasAvx2Fma) {
    fAddCopyAvx2(pSrc, pAlso, pDst, pCount);
    return;
  }
#endif
  for (uint32_t i = 0; i < pCount; i++) {
    pDst[i] = pSrc[i] + pAlso[i];
  }
}

void fNormRow(const float * pSrc, const float * pWeights, float * pDst, uint32_t pCount, float pScale) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool cHasAvx2 = __builtin_cpu_supports("avx2") != 0;
  if (cHasAvx2) {
    fNormRowAvx2(pSrc, pWeights, pDst, pCount, pScale);
    return;
  }
#endif
  for (uint32_t i = 0; i < pCount; i++) {
    pDst[i] = (pSrc[i] * pScale) * pWeights[i];
  }
}

void fScaleWrite(const float * pSrc, float * pDst, uint32_t pCount, float pScale, bool pAccumulate) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool cHasAvx2Fma = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  if (cHasAvx2Fma) {
    fScaleWriteAvx2(pSrc, pDst, pCount, pScale, pAccumulate);
    return;
  }
#endif
  if (!pAccumulate && pScale == 1.0f) {
    std::memcpy(pDst, pSrc, size_t(pCount) * sizeof(float));
    return;
  }
  for (uint32_t i = 0; i < pCount; i++) {
    if (pAccumulate) {
      pDst[i] += pScale * pSrc[i];
    } else {
      pDst[i] = pScale * pSrc[i];
    }
  }
}

}  // namespace

bool XdnaMoeOverlay::mLoad(const std::string & pPath, std::string & pError) {
  std::ifstream vFile(pPath);
  if (!vFile) {
    pError = "cannot open the manifest " + pPath;
    return false;
  }
  *this             = XdnaMoeOverlay();
  bool        vSeen = false;
  std::string vLine;
  uint32_t    vLineNumber = 0;
  while (std::getline(vFile, vLine)) {
    vLineNumber++;
    std::istringstream vWords(vLine);
    std::string        vKey;
    if (!(vWords >> vKey) || vKey[0] == '#') {
      continue;
    }
    bool vOk = true;
    if (vKey == "xdna2-moe") {
      uint32_t vVersion = 0;
      vOk               = bool(vWords >> vVersion) && vVersion == cManifestVersion;
      vSeen             = vOk;
    } else if (vKey == "array") {
      vOk = bool(vWords >> vColumns >> vRows >> vMemoryRows);
    } else if (vKey == "kernel") {
      vOk = bool(vWords >> vKernel);
    } else if (vKey == "object") {
      vOk = bool(vWords >> vObjectBytes >> vCoreBytes);
    } else if (vKey == "x") {
      vOk = bool(vWords >> vKeptX >> vRoomX);
    } else if (vKey == "rows") {
      vOk = bool(vWords >> vRowsMax);
    } else if (vKey == "chunks") {
      vOk = bool(vWords >> vChunksMax);
    } else if (vKey == "segments") {
      vOk = bool(vWords >> vSegments) && vSegments >= 1 && vSegments <= cXdnaMoeSegmentsMax;
    } else if (vKey == "chain") {
      vOk = bool(vWords >> vChainFloats);
    } else if (vKey == "formats") {
      std::string vName;
      while (vOk && (vWords >> vName)) {
        XdnaGemvFormat vFormat = XdnaGemvFormat::cBfp16;
        vOk                     = fXdnaGemvFormatFromName(vName, vFormat);
        vFormats |= vOk ? 1u << uint32_t(vFormat) : 0u;
      }
    } else if (vKey == "stream") {
      std::string vName;
      uint32_t    vNumber  = 0;
      uint32_t    vColumn  = 0;
      uint32_t    vChannel = 0;
      vOk = bool(vWords >> vName >> vNumber >> vColumn >> vChannel) && (vName == "W" || vName == "Y") &&
            fSetStream(vName == "W" ? aStreamsW : aStreamsY, vNumber, vColumn, vChannel);
    } else {
      vOk = false;
    }
    if (!vOk) {
      pError = "line " + std::to_string(vLineNumber) + " of the manifest " + pPath + " is not understood";
      return false;
    }
  }
  if (!vSeen || vColumns == 0 || vKernel.empty() || vCoreBytes != cCoreBytesExpected ||
      vObjectBytes != 2 * cCoreBytesExpected || vRoomX < cTileBytesX || vKeptX != vRoomX + vCoreBytes ||
      vRowsMax == 0 || vRowsMax % cBlock != 0 || vChunksMax == 0 ||
      vChunksMax > (vCoreBytes - cParamsHeaderBytes) / sizeof(uint32_t) || vSegments == 0 || vChainFloats < cBlockOutputs ||
      !mTakes(XdnaGemvFormat::cBfp16) || !fComplete(aStreamsW) || !fComplete(aStreamsY)) {
    pError = "the manifest " + pPath + " does not describe a complete mixture-of-experts overlay";
    return false;
  }
  return true;
}

uint32_t fXdnaMoeMaxRows(const XdnaMoeOverlay & pOverlay, const XdnaMoeMatrix & pMatrix) {
  if (pMatrix.vTilesCore == 0 || pMatrix.vBlocksEmit == 0 || pMatrix.vPaddedK == 0) {
    return 0;
  }
  const uint64_t vRowGroupBytes = uint64_t(pMatrix.vPaddedK / (2 * cBlock)) * cGroupBytes;
  const uint32_t vMaxGroups     = uint32_t(std::min<uint64_t>(pOverlay.vRowsMax / cBlock, pOverlay.vRoomX / vRowGroupBytes));
  uint32_t       vMaxRows       = vMaxGroups * cBlock;
  if (pMatrix.vSwiglu) {
    const uint32_t vByChain = pOverlay.vChainFloats / (pMatrix.vBlocksEmit * cBlockOutputs);
    vMaxRows                = std::min(vMaxRows, vByChain);
  }
  return vMaxRows;
}

bool fXdnaMoeMatrixShape(const XdnaMoeOverlay & pOverlay, uint32_t pExperts, uint32_t pK,
                          const XdnaGemvPart * aParts, uint32_t pParts, bool pSwiglu, XdnaMoeMatrix & pMatrix,
                          std::string & pError) {
  const uint64_t vStreams = pOverlay.mStreams();
  if (pExperts == 0 || pK == 0 || vStreams == 0) {
    pError = "the expert matrix is empty";
    return false;
  }
  if ((pSwiglu && pParts != 2) || (!pSwiglu && pParts != 1) || pParts > pOverlay.vSegments) {
    pError = "the mixture-of-experts overlay takes 1 segment, or 2 segments fused with SwiGLU";
    return false;
  }
  if (pSwiglu && aParts[0].vOutputs != aParts[1].vOutputs) {
    pError = "the gate and up segments of a SwiGLU must have the same outputs";
    return false;
  }
  bool vCanSymmetricSubTiles = pK % 128 == 0 && pK % (2 * cSuper) != 0;
  for (uint32_t i = 0; vCanSymmetricSubTiles && i < pParts; i++) {
    vCanSymmetricSubTiles =
        aParts[i].vFormat == XdnaGemvFormat::cQ8_0 || aParts[i].vFormat == XdnaGemvFormat::cQ5_1;
    const uint32_t vHalfTry   = pK / 2;
    const uint32_t vTilesTry  = (vHalfTry + cSuper - 1) / cSuper;
    const uint32_t vSubsTry   = (vHalfTry - (vTilesTry - 1) * cSuper) / 32;
    const uint64_t vPiecePair = uint64_t(vTilesTry - 1) * fXdnaGemvPieceBytes(aParts[i].vFormat) +
                                fXdnaGemvSubPieceBytes(aParts[i].vFormat, vSubsTry);
    if (aParts[i].vOutputs > vStreams * cBlockOutputs && uint64_t(cGroupsOut) * vPiecePair < pOverlay.vCoreBytes) {
      vCanSymmetricSubTiles = false;
    }
  }
  const uint64_t vPaddedK = vCanSymmetricSubTiles ? uint64_t(pK) : fRoundUp(pK, 2 * cSuper);
  const uint32_t vHalfK   = uint32_t(vPaddedK / 2);
  pMatrix                 = XdnaMoeMatrix();
  pMatrix.vExperts        = pExperts;
  pMatrix.vK              = pK;
  pMatrix.vPaddedK        = uint32_t(vPaddedK);
  pMatrix.vTilesCore      = (vHalfK + cSuper - 1) / cSuper;
  pMatrix.vTailSubs       = vCanSymmetricSubTiles ? (vHalfK - (pMatrix.vTilesCore - 1) * cSuper) / 32 : 8u;
  pMatrix.vSegments       = pParts;
  pMatrix.vSwiglu         = pSwiglu;
  uint64_t vExpertBytes   = 0;
  for (uint32_t i = 0; i < pParts; i++) {
    const XdnaGemvPart & vPart = aParts[i];
    if (!pOverlay.mTakes(vPart.vFormat) || vPart.vOutputs == 0) {
      pError = "the mixture-of-experts overlay does not take this format of weights";
      return false;
    }
    const uint32_t vStepK =
        (vPart.vFormat == XdnaGemvFormat::cQ8_0 || vPart.vFormat == XdnaGemvFormat::cQ5_1) ? 32u : cSuper;
    if (vPart.vFormat != XdnaGemvFormat::cBfp16 && pK % vStepK != 0) {
      pError = "weights in the formats of ggml come in blocks of 256 (or 32 for Q8_0 and Q5_1) values of K";
      return false;
    }
    const uint32_t   vFullPiece = fXdnaGemvPieceBytes(vPart.vFormat);
    const uint32_t   vTailPiece = fXdnaGemvSubPieceBytes(vPart.vFormat, pMatrix.vTailSubs);
    const uint64_t   vGroupSpan = uint64_t(pMatrix.vTilesCore - 1) * vFullPiece + vTailPiece;
    const uint64_t   vCoreBlock = uint64_t(cGroupsOut) * vGroupSpan;
    XdnaMoeSegment & vSegment   = pMatrix.aSegments[i];
    vSegment.vFormat            = vPart.vFormat;
    vSegment.vOutputs           = vPart.vOutputs;
    uint32_t vPadded            = uint32_t(fRoundUp(vPart.vOutputs, vStreams * cGroupOutputs));
    uint32_t vPerStream         = uint32_t(vPadded / vStreams);
    uint32_t vBlocks            = (vPerStream + cBlockOutputs - 1) / cBlockOutputs;
    uint32_t vTailGroups        = (vPerStream - (vBlocks - 1) * cBlockOutputs) / cGroupOutputs;
    // when vBlocks >= 2, a tail block shorter than one core half-object (4608 B) could finish inside the same object
    // where block vBlocks - 2 ended; pad to full blocks in that small-K case so at most one block ends per object
    if (vBlocks >= 2 && uint64_t(vTailGroups) * vGroupSpan < pOverlay.vCoreBytes) {
      vPadded     = uint32_t(fRoundUp(vPart.vOutputs, vStreams * cBlockOutputs));
      vPerStream  = uint32_t(vPadded / vStreams);
      vBlocks     = vPerStream / cBlockOutputs;
      vTailGroups = cGroupsOut;
    }
    if (pSwiglu && i == 1 && vTailGroups != pMatrix.aSegments[0].vTailGroups) {
      vPadded                        = uint32_t(fRoundUp(vPart.vOutputs, vStreams * cBlockOutputs));
      vBlocks                        = uint32_t(vPadded / (vStreams * cBlockOutputs));
      vTailGroups                    = cGroupsOut;
      XdnaMoeSegment & vSeg0         = pMatrix.aSegments[0];
      const uint64_t   vOldSeg0Bytes = uint64_t(vSeg0.vObjects) * pOverlay.vObjectBytes;
      vSeg0.vPadded                  = vPadded;
      vSeg0.vBlocks                  = vBlocks;
      vSeg0.vTailGroups              = cGroupsOut;
      vSeg0.vObjects                 = uint32_t((uint64_t(vBlocks) * vSeg0.vCoreBlock + pOverlay.vCoreBytes - 1) /
                                                pOverlay.vCoreBytes);
      vExpertBytes                   = vExpertBytes - vOldSeg0Bytes + uint64_t(vSeg0.vObjects) * pOverlay.vObjectBytes;
    }
    vSegment.vPadded             = vPadded;
    vSegment.vBlocks             = vBlocks;
    vSegment.vTailGroups         = vTailGroups;
    vSegment.vCoreBlock          = vCoreBlock;
    const uint64_t vCoreTail     = uint64_t(vTailGroups) * vGroupSpan;
    const uint64_t vCoreTotal    = uint64_t(vBlocks - 1) * vCoreBlock + vCoreTail;
    const uint64_t vObjects      = (vCoreTotal + pOverlay.vCoreBytes - 1) / pOverlay.vCoreBytes;
    if (vObjects == 0 || vObjects > 0x7fffffffu) {
      pError = "an expert segment is too large";
      return false;
    }
    vSegment.vObjects      = uint32_t(vObjects);
    vSegment.vOffsetExpert = vExpertBytes;
    vExpertBytes += uint64_t(vSegment.vObjects) * pOverlay.vObjectBytes;
  }
  pMatrix.vOutputs        = pMatrix.aSegments[0].vOutputs;
  pMatrix.vPaddedOut      = pMatrix.aSegments[0].vPadded;
  pMatrix.vBlocksEmit     = pMatrix.aSegments[0].vBlocks;
  pMatrix.vTailGroupsEmit = pMatrix.aSegments[0].vTailGroups;
  pMatrix.vExpertBytes = vExpertBytes;
  pMatrix.vPartW       = uint64_t(pExperts) * vExpertBytes;
  pMatrix.vBytesW      = vStreams * pMatrix.vPartW;
  if (fXdnaMoeMaxRows(pOverlay, pMatrix) == 0) {
    pError = "a row of K = " + std::to_string(pK) + " does not fit in a core of the mixture-of-experts overlay";
    return false;
  }
  return true;
}

bool fXdnaMoePackExpertSegment(const XdnaMoeOverlay & pOverlay, const XdnaMoeMatrix & pMatrix, uint32_t pExpert,
                                uint32_t pSegment, const void * pRows, uint64_t pRowBytes, uint8_t * pBuffer) {
  if (pExpert >= pMatrix.vExperts || pSegment >= pMatrix.vSegments) {
    return false;
  }
  const XdnaMoeSegment & vSegment   = pMatrix.aSegments[pSegment];
  const uint32_t          vStreams   = pOverlay.mStreams();
  const uint32_t          vPerStream = vSegment.vPadded / vStreams;
  const uint32_t          vHalfK     = pMatrix.vPaddedK / 2;
  alignas(64) uint8_t     aPiece[2304];
  for (uint32_t vStream = 0; vStream < vStreams; vStream++) {
    uint8_t * vPart = pBuffer + uint64_t(vStream) * pMatrix.vPartW + uint64_t(pExpert) * pMatrix.vExpertBytes +
                      vSegment.vOffsetExpert;
    std::memset(vPart + uint64_t(vSegment.vObjects - 1) * pOverlay.vObjectBytes, 0, pOverlay.vObjectBytes);
    const uint64_t  vFirst = uint64_t(vStream) * vPerStream;
    const uint32_t  vGiven = vFirst >= vSegment.vOutputs ?
                                 0 :
                                 uint32_t(std::min<uint64_t>(vPerStream, vSegment.vOutputs - vFirst));
    const uint8_t * vFrom  = static_cast<const uint8_t *>(pRows) + std::min<uint64_t>(vFirst, vSegment.vOutputs) * pRowBytes;
    for (uint32_t vBlock = 0; vBlock < vSegment.vBlocks; vBlock++) {
      const uint32_t vGroupsNow = (vBlock + 1 == vSegment.vBlocks) ? vSegment.vTailGroups : cGroupsOut;
      for (uint32_t vCore = 0; vCore < 2; vCore++) {
        uint64_t vInCore = uint64_t(vBlock) * vSegment.vCoreBlock;
        for (uint32_t vTile = 0; vTile < pMatrix.vTilesCore; vTile++) {
          const uint32_t vSubsNow    = (vTile + 1 == pMatrix.vTilesCore) ? pMatrix.vTailSubs : 8u;
          const uint32_t vPieceBytes = fXdnaGemvSubPieceBytes(vSegment.vFormat, vSubsNow);
          const uint32_t vFirstK     = vCore * vHalfK + vTile * cSuper;
          for (uint32_t vGroup = 0; vGroup < vGroupsNow; vGroup++) {
            const uint32_t vRow = vBlock * cBlockOutputs + vGroup * cGroupOutputs;
            if (!fXdnaGemvSubPiece(vSegment.vFormat, vFrom, vGiven, pMatrix.vK, pRowBytes, vRow, vFirstK, vSubsNow,
                                   aPiece)) {
              return false;
            }
            for (uint32_t vDone = 0; vDone < vPieceBytes;) {
              const uint64_t vObject = vInCore / pOverlay.vCoreBytes;
              const uint64_t vWithin = vInCore % pOverlay.vCoreBytes;
              const uint64_t vBytes  = std::min<uint64_t>(vPieceBytes - vDone, pOverlay.vCoreBytes - vWithin);
              std::memcpy(vPart + vObject * pOverlay.vObjectBytes + vCore * pOverlay.vCoreBytes + vWithin,
                          aPiece + vDone, vBytes);
              vDone += uint32_t(vBytes);
              vInCore += vBytes;
            }
          }
        }
      }
    }
  }
  return true;
}

bool fXdnaMoeCommandShape(const XdnaMoeOverlay & pOverlay, const XdnaMoeMatrix & pMatrix,
                           const XdnaMoeChunk * aChunks, uint32_t pChunks, XdnaMoeCommand & pCommand,
                           std::string & pError) {
  const uint32_t vMaxRows = fXdnaMoeMaxRows(pOverlay, pMatrix);
  if (pChunks == 0 || pChunks > pOverlay.vChunksMax || vMaxRows == 0) {
    pError = "a command of the mixture-of-experts overlay runs 1 to " + std::to_string(pOverlay.vChunksMax) + " chunks";
    return false;
  }
  pCommand         = XdnaMoeCommand();
  pCommand.vMatrix = pMatrix;
  pCommand.vChunks = pChunks;
  pCommand.aChunks.assign(aChunks, aChunks + pChunks);
  pCommand.aPlaces.resize(pChunks);
  uint64_t vOffset    = pOverlay.vObjectBytes;  // object 0 holds the parameters and the chunk table
  uint64_t vTotalRows = 0;
  for (uint32_t c = 0; c < pChunks; c++) {
    const XdnaMoeChunk & vChunk = aChunks[c];
    if (vChunk.vExpert >= pMatrix.vExperts || vChunk.vRows == 0 || vChunk.vRows > vMaxRows) {
      pError = "a chunk of the mixture-of-experts command has an invalid expert or row count";
      return false;
    }
    const uint32_t vRowGroups  = (vChunk.vRows + cBlock - 1) / cBlock;
    const uint64_t vCoreBytesX = uint64_t(vRowGroups) * (pMatrix.vPaddedK / (2 * cBlock)) * cGroupBytes;
    const uint32_t vObjectsX   = uint32_t((vCoreBytesX + pOverlay.vCoreBytes - 1) / pOverlay.vCoreBytes);
    pCommand.aPlaces[c]        = {vRowGroups, vObjectsX, uint32_t(vTotalRows), vOffset};
    vOffset += uint64_t(vObjectsX) * pOverlay.vObjectBytes;
    vTotalRows += vChunk.vRows;
  }
  if (vTotalRows > 0x7fffffffu) {
    pError = "the mixture-of-experts command has too many rows";
    return false;
  }
  pCommand.vTotalRows    = uint32_t(vTotalRows);
  pCommand.vBytesIn      = vOffset;
  pCommand.vOffsetY      = fRoundUp(vOffset, cPageBytes) + cPageBytes;
  pCommand.vStreamBytesY = vTotalRows * pMatrix.vBlocksEmit * cBlockOutputs * sizeof(float);
  pCommand.vBytesY       = uint64_t(pOverlay.mStreams()) * pCommand.vStreamBytesY;
  pCommand.vBytesIo      = pCommand.vOffsetY + pCommand.vBytesY;
  return true;
}

bool fXdnaMoePlan(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand,
                   std::vector<uint32_t> & aInstructions, std::string & pError) {
  const uint32_t vStreams = pOverlay.mStreams();
  if (pCommand.vChunks == 0 || pCommand.vChunks > pOverlay.vChunksMax || vStreams != cStreamsExpected) {
    pError = "the mixture-of-experts command is empty or invalid";
    return false;
  }
  XdnaInstructions vInstructions;
  vInstructions.mReserve(1024 + size_t(pCommand.vChunks) * vStreams * 68);
  XdnaShimDma      vDma(vInstructions, pOverlay.vColumns);
  auto             fLinear = [](const XdnaOverlayStream & pStream, XdnaDirection pDirection, uint32_t pBuffer,
                    uint64_t pOffset, uint64_t pBytes, bool pToken) {
    XdnaShimTransfer vTransfer;
    vTransfer.vColumn    = pStream.vColumn;
    vTransfer.vDirection = pDirection;
    vTransfer.vChannel   = pStream.vChannel;
    vTransfer.vBuffer    = pBuffer;
    vTransfer.vOffset    = pOffset;
    vTransfer.vBytes     = pBytes;
    vTransfer.vToken     = pToken;
    return vTransfer;
  };

  // One S2MM descriptor per stream drains all emitted rows of all chunks linearly (2 descriptors per shim column).
  XdnaShimTask aTaskY[cStreamsExpected];
  for (uint32_t s = 0; s < vStreams; s++) {
    if (!vDma.mPush(fLinear(pOverlay.aStreamsY[s], XdnaDirection::cToMemory, cBufferIo,
                            pCommand.vOffsetY + uint64_t(s) * pCommand.vStreamBytesY, pCommand.vStreamBytesY, true),
                    aTaskY[s], pError)) {
      return false;
    }
  }

  // Three pairs of MM2S descriptors per stream (12 descriptors per shim column; 14 of 16 in all): chunk c prepares its
  // descriptors in slot c % 3 while chunks c - 2 and c - 1 stream, then awaits chunk c - 2, frees its descriptors and
  // starts chunk c so the 4-task hardware queue never runs dry and never overflows.
  XdnaShimTask aTaskX[3][cStreamsExpected];
  XdnaShimTask aTaskW[3][cStreamsExpected];
  for (uint32_t c = 0; c < pCommand.vChunks; c++) {
    const uint32_t             vSlot   = c % 3;
    const XdnaMoeChunk &       vChunk  = pCommand.aChunks[c];
    const XdnaMoeChunkPlace &  vPlace  = pCommand.aPlaces[c];
    const uint64_t             vOffX   = c == 0 ? 0 : vPlace.vOffsetX;
    const uint64_t             vBytesX = uint64_t(c == 0 ? 1 + vPlace.vObjectsX : vPlace.vObjectsX) * pOverlay.vObjectBytes;
    const bool                 vTokenW = c + 2 < pCommand.vChunks;
    for (uint32_t s = 0; s < vStreams; s++) {
      const uint64_t vOffW =
          uint64_t(s) * pCommand.vMatrix.vPartW + uint64_t(vChunk.vExpert) * pCommand.vMatrix.vExpertBytes;
      if (!vDma.mPrepare(fLinear(pOverlay.aStreamsW[s], XdnaDirection::cFromMemory, cBufferIo, vOffX, vBytesX, false),
                         aTaskX[vSlot][s], pError) ||
          !vDma.mPrepare(fLinear(pOverlay.aStreamsW[s], XdnaDirection::cFromMemory, cBufferW, vOffW,
                                 pCommand.vMatrix.vExpertBytes, vTokenW),
                         aTaskW[vSlot][s], pError)) {
        return false;
      }
    }
    if (c >= 2) {
      const uint32_t vOld = (c - 2) % 3;
      for (uint32_t s = 0; s < vStreams; s++) {
        vDma.mAwait(aTaskW[vOld][s]);
        vDma.mFree(aTaskX[vOld][s]);
        vDma.mFree(aTaskW[vOld][s]);
      }
    }
    for (uint32_t s = 0; s < vStreams; s++) {
      vDma.mStart(aTaskX[vSlot][s]);
      vDma.mStart(aTaskW[vSlot][s]);
    }
  }

  for (uint32_t s = 0; s < vStreams; s++) {
    vDma.mAwait(aTaskY[s]);
  }
  for (uint32_t s = 0; s < vStreams; s++) {
    vDma.mFree(aTaskY[s]);
  }
  const uint32_t vFirstTail = pCommand.vChunks > 2 ? pCommand.vChunks - 2 : 0;
  for (uint32_t c = vFirstTail; c < pCommand.vChunks; c++) {
    const uint32_t vSlot = c % 3;
    for (uint32_t s = 0; s < vStreams; s++) {
      vDma.mFree(aTaskX[vSlot][s]);
      vDma.mFree(aTaskW[vSlot][s]);
    }
  }
  for (uint32_t s = 0; s < vStreams; s++) {
    vDma.mSettle(pOverlay.aStreamsW[s].vColumn, XdnaDirection::cFromMemory, pOverlay.aStreamsW[s].vChannel);
  }
  vInstructions.mFinishInto(pOverlay.vColumns, pOverlay.vRows, pOverlay.vMemoryRows, aInstructions);
  return true;
}

void fXdnaMoeWriteParameters(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, uint8_t * pBuffer) {
  const XdnaMoeMatrix & vMatrix           = pCommand.vMatrix;
  int32_t               aParams[cParamsWords] = {};
  aParams[cParamChunks]                   = int32_t(pCommand.vChunks);
  aParams[cParamTiles]                    = int32_t(vMatrix.vTilesCore);
  aParams[cParamSwiglu]                   = vMatrix.vSwiglu ? 1 : 0;
  aParams[cParamBlocks0]                  = int32_t(vMatrix.aSegments[0].vBlocks);
  aParams[cParamObjects0]                 = int32_t(vMatrix.aSegments[0].vObjects);
  aParams[cParamFormat0]                  = int32_t(vMatrix.aSegments[0].vFormat);
  if (vMatrix.vSegments == 2) {
    aParams[cParamBlocks1]  = int32_t(vMatrix.aSegments[1].vBlocks);
    aParams[cParamObjects1] = int32_t(vMatrix.aSegments[1].vObjects);
    aParams[cParamFormat1]  = int32_t(vMatrix.aSegments[1].vFormat);
  }
  aParams[cParamTailGroups] = int32_t(vMatrix.vTailGroupsEmit);
  aParams[cParamTailSubs]   = int32_t(vMatrix.vTailSubs);
  std::memset(pBuffer, 0, pOverlay.vCoreBytes);
  std::memcpy(pBuffer, aParams, sizeof(aParams));
  uint32_t * vTable = reinterpret_cast<uint32_t *>(pBuffer + cParamsHeaderBytes);
  for (uint32_t c = 0; c < pCommand.vChunks; c++) {
    vTable[c] = (pCommand.aPlaces[c].vObjectsX << 16) | pCommand.aChunks[c].vRows;
  }
  std::memcpy(pBuffer + pOverlay.vCoreBytes, pBuffer, pOverlay.vCoreBytes);
}

uint64_t fXdnaMoeRowEncodedBytes(const XdnaMoeMatrix & pMatrix) {
  return uint64_t(pMatrix.vPaddedK / cBlock) * cBlockBytes;
}

bool fXdnaMoeEncodeRowBfp16(const XdnaMoeMatrix & pMatrix, const float * pRow, uint8_t * pEncoded,
                            const float * pNormWeights, float pNormScale) {
  const uint32_t     vTotalGroups = pMatrix.vPaddedK / cBlock;
  const uint32_t     vFullGroups  = pMatrix.vK / cBlock;
  alignas(32) float  aStackNormed[4096];
  std::vector<float> aHeapNormed;
  const float *      vSrc = pRow;
  if (pNormWeights != nullptr) {
    float * vNormBuf = pMatrix.vK <= 4096 ? aStackNormed : (aHeapNormed.resize(pMatrix.vK), aHeapNormed.data());
    fNormRow(pRow, pNormWeights, vNormBuf, pMatrix.vK, pNormScale);
    vSrc = vNormBuf;
  }
  if (vFullGroups > 0 && !fXdnaBfp16EncodeNearest(vSrc, cBlock, vFullGroups, pEncoded)) {
    return false;
  }
  uint32_t vGroup = vFullGroups;
  if (vGroup * cBlock < pMatrix.vK) {
    float aValues[cBlock] = {};
    for (uint32_t i = 0; vGroup * cBlock + i < pMatrix.vK; i++) {
      aValues[i] = vSrc[vGroup * cBlock + i];
    }
    if (!fXdnaBfp16Encode(aValues, pEncoded + uint64_t(vGroup) * cBlockBytes, XdnaRounding::cNearest)) {
      return false;
    }
    vGroup++;
  }
  if (vGroup < vTotalGroups) {
    std::memset(pEncoded + uint64_t(vGroup) * cBlockBytes, 0, uint64_t(vTotalGroups - vGroup) * cBlockBytes);
  }
  return true;
}

void fXdnaMoeWriteChunkEncodedX(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, uint32_t pChunk,
                                 const uint8_t * const * aEncodedRows, uint8_t * pBuffer) {
  const XdnaMoeChunk &      vChunk      = pCommand.aChunks[pChunk];
  const XdnaMoeChunkPlace & vPlace      = pCommand.aPlaces[pChunk];
  const uint32_t             vHalfGroups = pCommand.vMatrix.vPaddedK / (2 * cBlock);
  uint8_t *                  vChunkX     = pBuffer + vPlace.vOffsetX;
  // zero only the trailing padding of the last X object of the chunk
  std::memset(vChunkX + uint64_t(vPlace.vObjectsX - 1) * pOverlay.vObjectBytes, 0, pOverlay.vObjectBytes);
  for (uint32_t vCore = 0; vCore < 2; vCore++) {
    for (uint32_t vLocalGroup = 0; vLocalGroup < vHalfGroups; vLocalGroup++) {
      const uint64_t vRowOff = uint64_t(vCore * vHalfGroups + vLocalGroup) * cBlockBytes;
      for (uint32_t vRg = 0; vRg < vPlace.vRowGroups; vRg++) {
        uint8_t * vTo = fGroupPtrX(vChunkX, pOverlay, vCore, vPlace.vRowGroups, vLocalGroup, vRg);
        for (uint32_t r = 0; r < cBlock; r++) {
          const uint32_t vRow = vRg * cBlock + r;
          if (vRow < vChunk.vRows) {
            std::memcpy(vTo + r * cBlockBytes, aEncodedRows[vRow] + vRowOff, cBlockBytes);
          } else {
            std::memset(vTo + r * cBlockBytes, 0, cBlockBytes);
          }
        }
      }
    }
  }
}

bool fXdnaMoeWriteChunkX(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, uint32_t pChunk,
                          const float * const * aRows, uint8_t * pBuffer) {
  const XdnaMoeChunk &      vChunk      = pCommand.aChunks[pChunk];
  const XdnaMoeChunkPlace & vPlace      = pCommand.aPlaces[pChunk];
  const uint32_t             vK          = pCommand.vMatrix.vK;
  const uint32_t             vHalfGroups = pCommand.vMatrix.vPaddedK / (2 * cBlock);
  uint8_t *                  vChunkX     = pBuffer + vPlace.vOffsetX;
  std::memset(vChunkX + uint64_t(vPlace.vObjectsX - 1) * pOverlay.vObjectBytes, 0, pOverlay.vObjectBytes);
  for (uint32_t vCore = 0; vCore < 2; vCore++) {
    for (uint32_t vLocalGroup = 0; vLocalGroup < vHalfGroups; vLocalGroup++) {
      const uint32_t vColumn = (vCore * vHalfGroups + vLocalGroup) * cBlock;
      for (uint32_t vRg = 0; vRg < vPlace.vRowGroups; vRg++) {
        uint8_t * vTo = fGroupPtrX(vChunkX, pOverlay, vCore, vPlace.vRowGroups, vLocalGroup, vRg);
        if (vColumn >= vK) {
          std::memset(vTo, 0, cGroupBytes);
          continue;
        }
        const uint32_t vValidRows =
            vRg * cBlock < vChunk.vRows ? std::min<uint32_t>(cBlock, vChunk.vRows - vRg * cBlock) : 0;
        for (uint32_t r = 0; r < vValidRows; r++) {
          const float * vSrc = aRows[vRg * cBlock + r] + vColumn;
          if (vColumn + cBlock <= vK) {
            if (!fXdnaBfp16EncodeNearest(vSrc, cBlock, 1, vTo + r * cBlockBytes)) {
              return false;
            }
          } else {
            float aTail[cBlock] = {};
            for (uint32_t i = 0; vColumn + i < vK; i++) {
              aTail[i] = vSrc[i];
            }
            if (!fXdnaBfp16Encode(aTail, vTo + r * cBlockBytes, XdnaRounding::cNearest)) {
              return false;
            }
          }
        }
        if (vValidRows < cBlock) {
          std::memset(vTo + vValidRows * cBlockBytes, 0, (cBlock - vValidRows) * cBlockBytes);
        }
      }
    }
  }
  return true;
}

bool fXdnaMoeTranscodeYToX(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pPrevCommand,
                            const uint8_t * pPrevBuffer, const XdnaMoeCommand & pNextCommand, uint8_t * pNextBuffer,
                            const float * aRowScales, XdnaWorkers * pWorkers) {
  if (pPrevCommand.vChunks != pNextCommand.vChunks || pPrevCommand.vTotalRows != pNextCommand.vTotalRows ||
      pPrevCommand.vMatrix.vOutputs != pNextCommand.vMatrix.vK) {
    return false;
  }
  const uint32_t    vStreams        = pOverlay.mStreams();
  const uint32_t    vPrevBlocks     = pPrevCommand.vMatrix.vBlocksEmit;
  const uint32_t    vPrevPerStream  = pPrevCommand.vMatrix.vPaddedOut / vStreams;
  const uint32_t    vPrevTailGroups = pPrevCommand.vMatrix.vTailGroupsEmit;
  const uint32_t    vNextK          = pNextCommand.vMatrix.vK;
  const uint32_t    vHalfGroups     = pNextCommand.vMatrix.vPaddedK / (2 * cBlock);
  std::atomic<bool> vFinite(true);

  auto fChunk = [&](uint32_t pChunk) {
    const XdnaMoeChunk &      vChunk  = pNextCommand.aChunks[pChunk];
    const XdnaMoeChunkPlace & vPlace  = pNextCommand.aPlaces[pChunk];
    uint8_t *                  vChunkX = pNextBuffer + vPlace.vOffsetX;
    // zero all X objects of the chunk first so any padded groups past vNextK or past vRows are clean zeros
    std::memset(vChunkX, 0, uint64_t(vPlace.vObjectsX) * pOverlay.vObjectBytes);
    for (uint32_t s = 0; s < vStreams; s++) {
      const float * vStreamY = reinterpret_cast<const float *>(
          pPrevBuffer + pPrevCommand.vOffsetY + uint64_t(s) * pPrevCommand.vStreamBytesY +
          uint64_t(vPlace.vFirstRowY) * vPrevBlocks * cBlockOutputs * sizeof(float));
      for (uint32_t b = 0; b < vPrevBlocks; b++) {
        const float *  vBlockY    = vStreamY + uint64_t(b) * vChunk.vRows * cBlockOutputs;
        const uint32_t vGroupsNow = (b + 1 == vPrevBlocks) ? vPrevTailGroups : cGroupsOut;
        for (uint32_t o = 0; o < vGroupsNow; o++) {
          const uint32_t vColumn = s * vPrevPerStream + b * cBlockOutputs + o * cBlock;
          if (vColumn >= vNextK) {
            continue;
          }
          const uint32_t vGlobalGroup = vColumn / cBlock;
          const uint32_t vCore        = vGlobalGroup / vHalfGroups;
          const uint32_t vLocalGroup  = vGlobalGroup % vHalfGroups;
          for (uint32_t vRg = 0; vRg < vPlace.vRowGroups; vRg++) {
            const uint32_t vValidRows = std::min<uint32_t>(cBlock, vChunk.vRows - vRg * cBlock);
            uint8_t *      vTo        = fGroupPtrX(vChunkX, pOverlay, vCore, vPlace.vRowGroups, vLocalGroup, vRg);
            const float *  vFrom      = vBlockY + uint64_t(vRg * cBlock) * cBlockOutputs + o * cBlock;
            if (aRowScales == nullptr && vColumn + cBlock <= vNextK) {
              if (!fXdnaBfp16EncodeNearest(vFrom, cBlockOutputs, vValidRows, vTo)) {
                vFinite = false;
                return;
              }
            } else {
              alignas(32) float aScaled[cBlock][cBlock] = {};
              const uint32_t    vValidCols              = std::min<uint32_t>(cBlock, vNextK - vColumn);
              for (uint32_t r = 0; r < vValidRows; r++) {
                const float vScale = aRowScales != nullptr ? aRowScales[vPlace.vFirstRowY + vRg * cBlock + r] : 1.0f;
                for (uint32_t i = 0; i < vValidCols; i++) {
                  aScaled[r][i] = vFrom[r * cBlockOutputs + i] * vScale;
                }
              }
              if (!fXdnaBfp16EncodeNearest(&aScaled[0][0], cBlock, vValidRows, vTo)) {
                vFinite = false;
                return;
              }
            }
          }
        }
      }
    }
  };

  if (pWorkers != nullptr && pNextCommand.vChunks >= 8) {
    pWorkers->mForEach(pNextCommand.vChunks, fChunk);
  } else {
    for (uint32_t c = 0; c < pNextCommand.vChunks; c++) {
      fChunk(c);
    }
  }
  return vFinite;
}

void fXdnaMoeReadY(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, const uint8_t * pBuffer,
                    const XdnaMoeRowOut * aRowsOut, XdnaWorkers * pWorkers, const float * pInitSrc, float * pInitDst,
                    uint32_t pInitRows, uint64_t pInitStride) {
  const uint32_t vStreams    = pOverlay.mStreams();
  const uint32_t vBlocks     = pCommand.vMatrix.vBlocksEmit;
  const uint32_t vOutputs    = pCommand.vMatrix.vOutputs;
  const uint32_t vPerStream  = pCommand.vMatrix.vPaddedOut / vStreams;

  auto fStream = [&](uint32_t pStream) {
    const uint32_t vStreamFirst = pStream * vPerStream;
    if (vStreamFirst >= vOutputs) {
      return;
    }
    const uint32_t vStreamEnd = std::min<uint32_t>(vOutputs, vStreamFirst + vPerStream);
    if (pInitSrc != nullptr && pInitDst != nullptr && pInitSrc != pInitDst) {
      const uint32_t vCols = vStreamEnd - vStreamFirst;
      for (uint32_t r = 0; r < pInitRows; r++) {
        std::memcpy(pInitDst + uint64_t(r) * pInitStride + vStreamFirst,
                    pInitSrc + uint64_t(r) * pInitStride + vStreamFirst, size_t(vCols) * sizeof(float));
      }
    }
    const float * vStreamY =
        reinterpret_cast<const float *>(pBuffer + pCommand.vOffsetY + uint64_t(pStream) * pCommand.vStreamBytesY);
    for (uint32_t c = 0; c < pCommand.vChunks; c++) {
      const XdnaMoeChunk &      vChunk  = pCommand.aChunks[c];
      const XdnaMoeChunkPlace & vPlace  = pCommand.aPlaces[c];
      const float *             vChunkY = vStreamY + uint64_t(vPlace.vFirstRowY) * vBlocks * cBlockOutputs;
      for (uint32_t b = 0; b < vBlocks; b++) {
        const uint32_t vColFirst = vStreamFirst + b * cBlockOutputs;
        if (vColFirst >= vStreamEnd) {
          break;
        }
        const uint32_t vCount  = std::min<uint32_t>(cBlockOutputs, vStreamEnd - vColFirst);
        const float *  vBlockY = vChunkY + uint64_t(b) * vChunk.vRows * cBlockOutputs;
        for (uint32_t m = 0; m < vChunk.vRows; m++) {
          const XdnaMoeRowOut & vOut = aRowsOut[vPlace.vFirstRowY + m];
          if (vOut.vDst == nullptr) {
            continue;
          }
          fScaleWrite(vBlockY + uint64_t(m) * cBlockOutputs, vOut.vDst + vColFirst, vCount, vOut.vScale,
                      vOut.vAccumulate);
        }
      }
    }
  };

  if (pWorkers != nullptr && uint64_t(pCommand.vTotalRows) * vOutputs >= (1u << 13)) {
    pWorkers->mForEach(vStreams, fStream);
  } else {
    for (uint32_t s = 0; s < vStreams; s++) {
      fStream(s);
    }
  }
}

void fXdnaMoeReadYTargets(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, const uint8_t * pBuffer,
                           const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers) {
  const uint32_t vStreams   = pOverlay.mStreams();
  const uint32_t vOutputs   = pCommand.vMatrix.vOutputs;
  const uint32_t vPerStream = pCommand.vMatrix.vPaddedOut / vStreams;
  const uint32_t vRows      = pCommand.vTotalRows;

  auto fStream = [&](uint32_t pStream) {
    const uint32_t vStreamFirst = pStream * vPerStream;
    if (vStreamFirst >= vOutputs) {
      return;
    }
    const uint32_t vStreamEnd = std::min<uint32_t>(vOutputs, vStreamFirst + vPerStream);
    const float *  vStreamY =
        reinterpret_cast<const float *>(pBuffer + pCommand.vOffsetY + uint64_t(pStream) * pCommand.vStreamBytesY);
    for (uint32_t t = 0; t < pTargets; t++) {
      const XdnaGemmTarget & vInto = aTargets[t];
      const uint64_t         vLo   = std::max<uint64_t>(vInto.vFirst, vStreamFirst);
      const uint64_t         vHi   = std::min<uint64_t>(uint64_t(vInto.vFirst) + vInto.vCount, vStreamEnd);
      if (vLo >= vHi || vInto.vData == nullptr) {
        continue;
      }
      if (vRows == 1) {
        const float * vSrc = vStreamY + (vLo - vStreamFirst);
        float *       vDst = vInto.vData + (vLo - vInto.vFirst);
        const uint32_t vN  = uint32_t(vHi - vLo);
        if (vInto.vAlso != nullptr) {
          fAddCopy(vSrc, vInto.vAlso + (vLo - vInto.vFirst), vDst, vN);
        } else {
          std::memcpy(vDst, vSrc, size_t(vN) * sizeof(float));
        }
        continue;
      }
      for (uint64_t vOut = vLo; vOut < vHi;) {
        const uint64_t vInStream = vOut - vStreamFirst;
        const uint64_t vBlock    = vInStream / cBlockOutputs;
        const uint64_t vWithin   = vInStream % cBlockOutputs;
        const uint64_t vEnd      = std::min<uint64_t>(vHi, vStreamFirst + (vBlock + 1) * cBlockOutputs);
        const uint32_t vN        = uint32_t(vEnd - vOut);
        for (uint32_t r = 0; r < vRows; r++) {
          const float *  vSrc       = vStreamY + (vBlock * vRows + r) * cBlockOutputs + vWithin;
          const uint64_t vDstOffset = uint64_t(r) * vInto.vStride + (vOut - vInto.vFirst);
          if (vInto.vAlso != nullptr) {
            fAddCopy(vSrc, vInto.vAlso + vDstOffset, vInto.vData + vDstOffset, vN);
          } else {
            std::memcpy(vInto.vData + vDstOffset, vSrc, size_t(vN) * sizeof(float));
          }
        }
        vOut = vEnd;
      }
    }
  };

  if (pWorkers != nullptr && uint64_t(vRows) * vOutputs >= (1u << 13)) {
    pWorkers->mForEach(vStreams, fStream);
  } else {
    for (uint32_t s = 0; s < vStreams; s++) {
      fStream(s);
    }
  }
}
