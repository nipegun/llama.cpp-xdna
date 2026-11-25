#include "xdna-gemv.h"

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

const uint32_t cManifestVersion = 4;
const uint32_t cMaxStreams      = 64;
const uint32_t cMaxSplit        = 4;
const uint32_t cMaxCores        = cMaxStreams * cMaxSplit;
const uint32_t cBlock           = 8;    // values that share an exponent
const uint32_t cBlockBytes      = 9;    // and the bytes they take
const uint32_t cGroupOutputs    = 8;    // outputs of a group of sums, and of a piece
const uint32_t cSuper           = 256;  // values of a block of the formats of ggml, and of a piece

// buffers of the command
const uint32_t cBufferX = 0;
const uint32_t cBufferW = 1;
const uint32_t cBufferY = 2;

// the value of the lock number 0 of a core tile, and bytes to the next
const uint32_t cCoreLockBase = 0x1f000;
const uint32_t cCoreLockStep = 0x10;

// where the parts of a piece start (see the kernel)
const uint32_t cScalesK   = 32;    // Q4_K, Q5_K: the 12 bytes of scales and minimums, [12][8]
const uint32_t cValuesK   = 128;   // Q4_K, Q5_K: the 4-bit values
const uint32_t cFifthQ5K  = 1152;  // Q5_K: the fifth bits
const uint32_t cScalesQ6K  = 32;    // Q6_K: the 8-bit scales, [16][8]
const uint32_t cValuesQ6K  = 160;   // Q6_K: the low 4 bits
const uint32_t cTopQ6K     = 1184;  // Q6_K: the top 2 bits
const uint32_t cValuesQ8_0 = 128;   // Q8_0: the 8-bit signed values, after [8][8] bf16 scales
const uint32_t cMinsQ5_1   = 128;   // Q5_1: [8][8] bf16 minimums, after [8][8] bf16 scales
const uint32_t cValuesQ5_1 = 256;   // Q5_1: the low 4 bits
const uint32_t cFifthQ5_1  = 1280;  // Q5_1: the fifth bits

const char * const aFormatNames[cXdnaGemvFormats] = {"bfp16", "q4k", "q5k", "q6k", "q8_0", "q5_1"};
const uint32_t     aBlockBytes[cXdnaGemvFormats]  = {0, 144, 176, 210, 272, 192};
const uint32_t     aPieceBytes[cXdnaGemvFormats]  = {2304, 1152, 1408, 1696, 2176, 1536};

// puts a stream at its number, growing the list
bool fSetStream(std::vector<XdnaOverlayStream> & aStreams, uint32_t pNumber, uint32_t pColumn, uint32_t pChannel) {
  if (pNumber >= cMaxStreams || pColumn > 255 || pChannel > 255) {
    return false;
  }
  if (aStreams.size() <= pNumber) {
    aStreams.resize(pNumber + 1, XdnaOverlayStream{255, 255});
  }
  aStreams[pNumber] = {uint8_t(pColumn), uint8_t(pChannel)};
  return true;
}

bool fComplete(const std::vector<XdnaOverlayStream> & aStreams) {
  for (const XdnaOverlayStream & vStream : aStreams) {
    if (vStream.vColumn == 255) {
      return false;
    }
  }
  return !aStreams.empty();
}

uint64_t fRoundUp(uint64_t pValue, uint64_t pStep) {
  return (pValue + pStep - 1) / pStep * pStep;
}

// the groups of 8 outputs a core makes of a block
uint32_t fGroupsOfCore(const XdnaGemvOverlay & pOverlay) {
  return pOverlay.vOutputs / cGroupOutputs / pOverlay.vSplit;
}

// The 256 values of 4 bits of a block of Q4_K or Q5_K, in order: byte l of the pair c of sub-blocks holds the value
// 64c + l in its low half and 64c + 32 + l in its high half.
void fLowNibblesK(const uint8_t * pQs, uint8_t * aValues) {
  for (uint32_t c = 0; c < 4; c++) {
    for (uint32_t l = 0; l < 32; l++) {
      aValues[64 * c + l]      = pQs[32 * c + l] & 0xf;
      aValues[64 * c + 32 + l] = pQs[32 * c + l] >> 4;
    }
  }
}

// Writes 64 values of 4 bits per group, lane 8j + t, two per byte, the first in the low half: the group of the 8
// values of K from pFirst of the 8 outputs of aValues.
void fPutNibbles(const uint8_t (*aValues)[cSuper], uint32_t pFirst, uint8_t * pTo) {
  for (uint32_t n = 0; n < 64; n += 2) {
    const uint32_t vFirst  = aValues[n / 8][pFirst + n % 8];
    const uint32_t vSecond = aValues[(n + 1) / 8][pFirst + (n + 1) % 8];
    pTo[n / 2]             = uint8_t((vFirst & 0xf) | ((vSecond & 0xf) << 4));
  }
}

// d (and dmin) of a block of ggml at pFrom, fp16, as bf16 to the nearest
uint16_t fHalfAsBf16(const uint8_t * pFrom) {
  uint16_t vHalf = 0;
  std::memcpy(&vHalf, pFrom, 2);
  return fXdnaBf16FromFloat(fXdnaHalfToFloat(vHalf));
}

// A piece of Q4_K or Q5_K from the blocks of the 8 outputs (null past the last output: zeros, d = 0).
void fPieceK(const uint8_t * const * aBlocks, bool pFifth, uint8_t * pPiece) {
  uint8_t  aValues[cGroupOutputs][cSuper];  // the 4-bit values of the 8 outputs, in order
  uint8_t  aHigh[cGroupOutputs][cSuper];    // Q5_K: their fifth bits
  uint16_t aD[cGroupOutputs]    = {};
  uint16_t aDmin[cGroupOutputs] = {};
  // block_q4_K: d, dmin, scales [12], qs [128]; block_q5_K: d, dmin, scales [12], qh [32], qs [128]
  for (uint32_t j = 0; j < cGroupOutputs; j++) {
    if (aBlocks[j] == nullptr) {
      std::memset(aValues[j], 0, sizeof(aValues[j]));
      std::memset(aHigh[j], 0, sizeof(aHigh[j]));
      continue;
    }
    const uint8_t * vBlock = aBlocks[j];
    aD[j]                  = fHalfAsBf16(vBlock);
    aDmin[j]               = fHalfAsBf16(vBlock + 2);
    for (uint32_t i = 0; i < 12; i++) {
      pPiece[cScalesK + i * cGroupOutputs + j] = vBlock[4 + i];
    }
    fLowNibblesK(vBlock + (pFifth ? 48 : 16), aValues[j]);
    if (pFifth) {
      // the fifth bit of value 32s + l is bit s of qh[l]
      const uint8_t * vQh = vBlock + 16;
      for (uint32_t v = 0; v < cSuper; v++) {
        aHigh[j][v] = (vQh[v % 32] >> (v / 32)) & 1;
      }
    }
  }
  std::memcpy(pPiece, aD, sizeof(aD));
  std::memcpy(pPiece + 16, aDmin, sizeof(aDmin));
  // the values: [sub-block of 32][group of 8 values of K][64 lanes]
  for (uint32_t vGroup = 0; vGroup < 32; vGroup++) {
    fPutNibbles(aValues, vGroup * 8, pPiece + cValuesK + vGroup * 32);
  }
  if (pFifth) {
    // [sub-block][group][8 bytes]: bit t of byte j is the fifth bit of lane 8j + t
    for (uint32_t vGroup = 0; vGroup < 32; vGroup++) {
      for (uint32_t j = 0; j < cGroupOutputs; j++) {
        uint8_t vByte = 0;
        for (uint32_t t = 0; t < cBlock; t++) {
          vByte |= uint8_t(aHigh[j][vGroup * 8 + t] << t);
        }
        pPiece[cFifthQ5K + vGroup * 8 + j] = vByte;
      }
    }
  }
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))

__attribute__((target("avx2"))) void fPackSliceQ6K(__m128i pLo, __m128i pHi, uint32_t * pOut) {
  const __m128i vPack = _mm_set1_epi16(0x1001);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(pOut),
                   _mm_packus_epi16(_mm_maddubs_epi16(pLo, vPack), _mm_maddubs_epi16(pHi, vPack)));
}

__attribute__((target("avx2"))) void fPieceQ6KAvx2(const uint8_t * const * aBlocks, uint8_t * pPiece) {
  alignas(32) uint32_t aLow32[cGroupOutputs][32];
  alignas(32) uint32_t aTop32[4][32];
  uint16_t             aD[cGroupOutputs] = {};
  const __m128i        vMask4            = _mm_set1_epi8(0x0f);
  const __m256i        vMask2            = _mm256_set1_epi8(0x03);
  for (uint32_t j = 0; j < cGroupOutputs; j++) {
    if (aBlocks[j] == nullptr) {
      std::memset(aLow32[j], 0, sizeof(aLow32[j]));
      continue;
    }
    const uint8_t * vBlock = aBlocks[j];
    aD[j]                  = fHalfAsBf16(vBlock + 208);
    for (uint32_t s = 0; s < 16; s++) {
      pPiece[cScalesQ6K + s * cGroupOutputs + j] = vBlock[192 + s];
    }
    for (uint32_t h = 0; h < 2; h++) {
      const uint8_t * vQl = vBlock + 64 * h;
      const __m128i   vQ0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl));
      const __m128i   vQ1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl + 16));
      const __m128i   vQ2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl + 32));
      const __m128i   vQ3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl + 48));
      fPackSliceQ6K(_mm_and_si128(vQ0, vMask4), _mm_and_si128(vQ1, vMask4), &aLow32[j][16 * h]);
      fPackSliceQ6K(_mm_and_si128(vQ2, vMask4), _mm_and_si128(vQ3, vMask4), &aLow32[j][16 * h + 4]);
      fPackSliceQ6K(_mm_and_si128(_mm_srli_epi16(vQ0, 4), vMask4),
                    _mm_and_si128(_mm_srli_epi16(vQ1, 4), vMask4), &aLow32[j][16 * h + 8]);
      fPackSliceQ6K(_mm_and_si128(_mm_srli_epi16(vQ2, 4), vMask4),
                    _mm_and_si128(_mm_srli_epi16(vQ3, 4), vMask4), &aLow32[j][16 * h + 12]);
    }
  }
  for (uint32_t j = 0; j < 4; j++) {
    for (uint32_t h = 0; h < 2; h++) {
      const __m256i vLo = aBlocks[j] != nullptr ?
                              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(aBlocks[j] + 128 + 32 * h)) :
                              _mm256_setzero_si256();
      const __m256i vHi = aBlocks[j + 4] != nullptr ?
                              _mm256_loadu_si256(reinterpret_cast<const __m256i *>(aBlocks[j + 4] + 128 + 32 * h)) :
                              _mm256_setzero_si256();
      const __m256i aFields[4] = {
        _mm256_or_si256(_mm256_and_si256(vLo, vMask2), _mm256_slli_epi16(_mm256_and_si256(vHi, vMask2), 2)),
        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vLo, 2), vMask2),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vHi, 2), vMask2), 2)),
        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vLo, 4), vMask2),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vHi, 4), vMask2), 2)),
        _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vLo, 6), vMask2),
                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vHi, 6), vMask2), 2)),
      };
      for (uint32_t k = 0; k < 4; k++) {
        fPackSliceQ6K(_mm256_castsi256_si128(aFields[k]), _mm256_extracti128_si256(aFields[k], 1),
                      &aTop32[j][16 * h + 4 * k]);
      }
    }
  }
  std::memcpy(pPiece, aD, sizeof(aD));
  for (uint32_t vGroup = 0; vGroup < 32; vGroup++) {
    uint32_t aOutLow[cGroupOutputs];
    uint32_t aOutTop[4];
    for (uint32_t j = 0; j < cGroupOutputs; j++) {
      aOutLow[j] = aLow32[j][vGroup];
    }
    for (uint32_t j = 0; j < 4; j++) {
      aOutTop[j] = aTop32[j][vGroup];
    }
    std::memcpy(pPiece + cValuesQ6K + vGroup * 32, aOutLow, sizeof(aOutLow));
    std::memcpy(pPiece + cTopQ6K + vGroup * 16, aOutTop, sizeof(aOutTop));
  }
}

#endif

// A piece of Q6_K from the blocks of the 8 outputs (null past the last output: zeros, d = 0).
void fPieceQ6K(const uint8_t * const * aBlocks, uint8_t * pPiece) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool cHasAvx2 = __builtin_cpu_supports("avx2");
  if (cHasAvx2) {
    fPieceQ6KAvx2(aBlocks, pPiece);
    return;
  }
#endif
  uint8_t  aLow[cGroupOutputs][cSuper];   // the low 4 bits of the values of the 8 outputs, in order
  uint8_t  aTop[cGroupOutputs][cSuper];   // and their top 2 bits
  uint16_t aD[cGroupOutputs] = {};
  // block_q6_K: ql [128], qh [64], scales [16] (signed), d
  for (uint32_t j = 0; j < cGroupOutputs; j++) {
    if (aBlocks[j] == nullptr) {
      std::memset(aLow[j], 0, sizeof(aLow[j]));
      std::memset(aTop[j], 0, sizeof(aTop[j]));
      continue;
    }
    const uint8_t * vBlock = aBlocks[j];
    const uint8_t * vQl    = vBlock;
    const uint8_t * vQh    = vBlock + 128;
    aD[j]                  = fHalfAsBf16(vBlock + 208);
    for (uint32_t s = 0; s < 16; s++) {
      pPiece[cScalesQ6K + s * cGroupOutputs + j] = vBlock[192 + s];
    }
    // value 128h + 32k + l: low 4 bits from ql[64h + l] (k = 0, 2) or ql[64h + 32 + l] (k = 1, 3), in the low half for
    // k < 2 and the high half for k >= 2; top 2 bits from qh[32h + l], 2k bits up
    for (uint32_t h = 0; h < 2; h++) {
      for (uint32_t k = 0; k < 4; k++) {
        for (uint32_t l = 0; l < 32; l++) {
          const uint32_t v     = 128 * h + 32 * k + l;
          const uint8_t  vByte = vQl[64 * h + (k % 2) * 32 + l];
          aLow[j][v]           = k < 2 ? (vByte & 0xf) : (vByte >> 4);
          aTop[j][v]           = (vQh[32 * h + l] >> (2 * k)) & 3;
        }
      }
    }
  }
  std::memcpy(pPiece, aD, sizeof(aD));
  // the low 4 bits: [sub-block of 16][group of 8 values of K][64 lanes]
  for (uint32_t vGroup = 0; vGroup < 32; vGroup++) {
    fPutNibbles(aLow, vGroup * 8, pPiece + cValuesQ6K + vGroup * 32);
  }
  // the top 2 bits: [sub-block][group][16 bytes], field k of a group (two per byte, the first in the low half) the
  // bits of lane k and, two bits up, those of lane k + 32
  for (uint32_t vGroup = 0; vGroup < 32; vGroup++) {
    uint8_t * vTo = pPiece + cTopQ6K + vGroup * 16;
    for (uint32_t k = 0; k < 32; k++) {
      const uint32_t vLow   = aTop[k / 8][vGroup * 8 + k % 8];
      const uint32_t vHigh  = aTop[(k + 32) / 8][vGroup * 8 + (k + 32) % 8];
      const uint32_t vField = vLow | (vHigh << 2);
      vTo[k / 2] |= uint8_t(k % 2 == 0 ? vField : vField << 4);
    }
  }
}

// A piece of Q8_0 from up to pTotalSubs consecutive block_q8_0 structs (34 bytes = 32 values of K each, pTotalSubs
// even in 2..8) of each of the 8 outputs.
void fPieceQ8_0(const uint8_t * const * aBlocks, uint32_t pSubBlocks, uint32_t pTotalSubs, uint8_t * pPiece) {
  const uint32_t vPairs  = pTotalSubs >> 1;
  uint16_t *     vScales = reinterpret_cast<uint16_t *>(pPiece);
  uint64_t *     vValues = reinterpret_cast<uint64_t *>(pPiece + vPairs * 32);
  for (uint32_t j = 0; j < cGroupOutputs; j++) {
    const uint8_t * vRowBlocks = aBlocks[j];
    const uint32_t  vValidSubs = vRowBlocks == nullptr ? 0u : std::min<uint32_t>(pTotalSubs, pSubBlocks);
    for (uint32_t vSub = 0; vSub < vValidSubs; vSub++) {
      const uint8_t * vBlock = vRowBlocks + vSub * 34;
      vScales[vSub * 8 + j]  = fHalfAsBf16(vBlock);
      uint64_t aRuns[4];
      std::memcpy(aRuns, vBlock + 2, 32);
      vValues[(vSub * 4 + 0) * 8 + j] = aRuns[0];
      vValues[(vSub * 4 + 1) * 8 + j] = aRuns[1];
      vValues[(vSub * 4 + 2) * 8 + j] = aRuns[2];
      vValues[(vSub * 4 + 3) * 8 + j] = aRuns[3];
    }
    for (uint32_t vSub = vValidSubs; vSub < pTotalSubs; vSub++) {
      vScales[vSub * 8 + j]           = 0;
      vValues[(vSub * 4 + 0) * 8 + j] = 0;
      vValues[(vSub * 4 + 1) * 8 + j] = 0;
      vValues[(vSub * 4 + 2) * 8 + j] = 0;
      vValues[(vSub * 4 + 3) * 8 + j] = 0;
    }
  }
}

// A piece of Q5_1 from up to pTotalSubs consecutive block_q5_1 structs (24 bytes = 32 values of K each, pTotalSubs
// even in 2..8) of each of the 8 outputs.
void fPieceQ5_1(const uint8_t * const * aBlocks, uint32_t pSubBlocks, uint32_t pTotalSubs, uint8_t * pPiece) {
  const uint32_t vPairs     = pTotalSubs >> 1;
  const uint32_t vValuesOff = vPairs * 64;
  const uint32_t vFifthOff  = vValuesOff + pTotalSubs * 128;
  uint16_t *     vScales    = reinterpret_cast<uint16_t *>(pPiece);
  uint16_t *     vMins      = reinterpret_cast<uint16_t *>(pPiece + vPairs * 32);
  uint8_t        aValues[cGroupOutputs][cSuper] = {};
  for (uint32_t j = 0; j < cGroupOutputs; j++) {
    const uint8_t * vRowBlocks = aBlocks[j];
    const uint32_t  vValidSubs = vRowBlocks == nullptr ? 0u : std::min<uint32_t>(pTotalSubs, pSubBlocks);
    for (uint32_t vSub = 0; vSub < vValidSubs; vSub++) {
      const uint8_t * vBlock = vRowBlocks + vSub * 24;
      vScales[vSub * 8 + j]  = fHalfAsBf16(vBlock);
      vMins[vSub * 8 + j]    = fHalfAsBf16(vBlock + 2);
      for (uint32_t g = 0; g < 4; g++) {
        pPiece[vFifthOff + (vSub * 4 + g) * 8 + j] = vBlock[4 + g];
      }
      for (uint32_t l = 0; l < 16; l++) {
        const uint8_t vByte            = vBlock[8 + l];
        aValues[j][vSub * 32 + l]      = vByte & 0xf;
        aValues[j][vSub * 32 + 16 + l] = vByte >> 4;
      }
    }
  }
  for (uint32_t vGroup = 0; vGroup < pTotalSubs * 4; vGroup++) {
    fPutNibbles(aValues, vGroup * 8, pPiece + vValuesOff + vGroup * 32);
  }
}

}  // namespace

uint32_t fXdnaGemvBlockBytes(XdnaGemvFormat pFormat) {
  return uint32_t(pFormat) < cXdnaGemvFormats ? aBlockBytes[uint32_t(pFormat)] : 0;
}

uint32_t fXdnaGemvPieceBytes(XdnaGemvFormat pFormat) {
  return uint32_t(pFormat) < cXdnaGemvFormats ? aPieceBytes[uint32_t(pFormat)] : 0;
}

uint32_t fXdnaGemvSubPieceBytes(XdnaGemvFormat pFormat, uint32_t pSubs) {
  if (pSubs >= 8 || (pFormat != XdnaGemvFormat::cQ8_0 && pFormat != XdnaGemvFormat::cQ5_1)) {
    return fXdnaGemvPieceBytes(pFormat);
  }
  if (pFormat == XdnaGemvFormat::cQ8_0) {
    return (pSubs >> 1) * 32 + pSubs * 256;
  }
  return (pSubs >> 1) * 64 + pSubs * 160;
}

const char * fXdnaGemvFormatName(XdnaGemvFormat pFormat) {
  return uint32_t(pFormat) < cXdnaGemvFormats ? aFormatNames[uint32_t(pFormat)] : "?";
}

bool fXdnaGemvFormatFromName(const std::string & pName, XdnaGemvFormat & pFormat) {
  for (uint32_t i = 0; i < cXdnaGemvFormats; i++) {
    if (pName == aFormatNames[i]) {
      pFormat = XdnaGemvFormat(i);
      return true;
    }
  }
  return false;
}

bool XdnaGemvOverlay::mLoad(const std::string & pPath, std::string & pError) {
  std::ifstream vFile(pPath);
  if (!vFile) {
    pError = "cannot open the manifest " + pPath;
    return false;
  }
  *this = XdnaGemvOverlay();
  std::vector<bool> aCoreSeen;

  bool        vHasVersion = false;
  bool        vHasSplit   = false;
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
    if (vKey == "xdna2-gemv") {
      uint32_t vVersion = 0;
      vOk               = bool(vWords >> vVersion) && vVersion == cManifestVersion;
      vHasVersion       = vOk;
    } else if (vKey == "array") {
      vOk = bool(vWords >> vColumns >> vRows >> vMemoryRows);
    } else if (vKey == "kernel") {
      vOk = bool(vWords >> vKernel);
    } else if (vKey == "object") {
      vOk = bool(vWords >> vObjectBytes >> vOutputs >> vTileK >> vObjectsX);
    } else if (vKey == "split") {
      vOk       = bool(vWords >> vSplit) && vSplit >= 1 && vSplit <= cMaxSplit;
      vHasSplit = vOk;
    } else if (vKey == "segments") {
      vOk = bool(vWords >> vSegments) && vSegments >= 1 && vSegments <= cXdnaGemvSegmentsMax;
    } else if (vKey == "formats") {
      std::string vName;
      while (vOk && (vWords >> vName)) {
        XdnaGemvFormat vFormat = XdnaGemvFormat::cBfp16;
        vOk                     = fXdnaGemvFormatFromName(vName, vFormat);
        vFormats |= vOk ? 1u << uint32_t(vFormat) : 0u;
      }
    } else if (vKey == "core") {
      uint32_t         vNumber = 0;
      uint32_t         vTileColumn = 0;
      uint32_t         vTileRow = 0;
      XdnaOverlayCore vCore;
      vOk = bool(vWords >> vNumber >> vTileColumn >> vTileRow >> vCore.vParameters >> vCore.vLock) &&
            vNumber < cMaxCores && vTileColumn < 256 && vTileRow < 256;
      if (vOk) {
        vCore.vTileColumn = uint8_t(vTileColumn);
        vCore.vTileRow    = uint8_t(vTileRow);
        if (aCores.size() <= vNumber) {
          aCores.resize(vNumber + 1);
          aCoreSeen.resize(vNumber + 1, false);
        }
        aCores[vNumber]    = vCore;
        aCoreSeen[vNumber] = true;
      }
    } else if (vKey == "stream") {
      std::string vName;
      uint32_t    vNumber = 0;
      uint32_t    vColumn = 0;
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

  bool vCoresComplete = !aCoreSeen.empty();
  for (bool vSeen : aCoreSeen) {
    vCoresComplete = vCoresComplete && vSeen;
  }
  if (!vHasVersion || !vHasSplit || vSegments == 0 || vColumns == 0 || vKernel.empty() || vObjectBytes == 0 ||
      vOutputs == 0 || vTileK == 0 || vObjectsX == 0 || !mTakes(XdnaGemvFormat::cBfp16) || !vCoresComplete ||
      !fComplete(aStreamsW) || !fComplete(aStreamsY)) {
    pError = "the manifest " + pPath + " is not complete";
    return false;
  }
  // a block of a stream is vOutputs outputs, in groups of 8 shared out evenly between the cores of the stream, and a
  // tile of bfp16 of a block fills an object; pieces of every format are whole multiples of 32 bytes
  if (aStreamsW.size() != aStreamsY.size() || aCores.size() != aStreamsW.size() * vSplit ||
      vOutputs % (cGroupOutputs * vSplit) != 0 || vTileK != cSuper ||
      uint64_t(vOutputs) * vTileK / cBlock * cBlockBytes != vObjectBytes || vObjectBytes % (32 * vSplit) != 0) {
    pError = "the manifest " + pPath + " does not describe an overlay for generating tokens";
    return false;
  }
  return true;
}

bool fXdnaGemvShape(const XdnaGemvOverlay & pOverlay, uint32_t pRows, uint32_t pK, const XdnaGemvPart * aParts,
                     uint32_t pParts, XdnaGemvShape & pShape, std::string & pError) {
  const uint64_t vStreams = pOverlay.mStreams();
  if (pRows == 0 || pRows > cXdnaGemvRowsMax || pK == 0 || vStreams == 0) {
    pError = "the overlay for generating tokens multiplies 1 to " + std::to_string(cXdnaGemvRowsMax) + " rows";
    return false;
  }
  if (pParts == 0 || pParts > pOverlay.vSegments) {
    pError = "the overlay for generating tokens takes 1 to " + std::to_string(pOverlay.vSegments) + " segments";
    return false;
  }
  const uint64_t vPaddedK    = fRoundUp(pK, pOverlay.vTileK);
  const uint32_t vGroupX     = (pRows * cBlockBytes + 3) / 4 * 4;
  const uint64_t vObjectCore = pOverlay.vObjectBytes / pOverlay.vSplit;  // an object of a core
  const uint64_t vObjectsX   = (vPaddedK / cBlock * vGroupX + vObjectCore - 1) / vObjectCore;
  if (vObjectsX > pOverlay.vObjectsX) {
    pError = std::to_string(pRows) + " rows of " + std::to_string(pK) + " values do not fit in a core of the overlay";
    return false;
  }
  pShape           = XdnaGemvShape();
  pShape.vRows     = pRows;
  pShape.vK        = pK;
  pShape.vPaddedK  = uint32_t(vPaddedK);
  pShape.vTilesK   = uint32_t(vPaddedK / pOverlay.vTileK);
  pShape.vGroupX   = vGroupX;
  pShape.vObjectsX = uint32_t(vObjectsX);
  pShape.vSegments = pParts;
  uint64_t vFirst  = 0;
  uint64_t vBlocks = 0;
  uint64_t vPartW  = 0;
  for (uint32_t i = 0; i < pParts; i++) {
    const XdnaGemvPart & vPart = aParts[i];
    if (!pOverlay.mTakes(vPart.vFormat) || vPart.vOutputs == 0) {
      pError = "the overlay for generating tokens does not take this format of weights";
      return false;
    }
    const uint32_t vStepK =
        (vPart.vFormat == XdnaGemvFormat::cQ8_0 || vPart.vFormat == XdnaGemvFormat::cQ5_1) ? 32u : cSuper;
    if (vPart.vFormat != XdnaGemvFormat::cBfp16 && pK % vStepK != 0) {
      pError = "weights in the formats of ggml come in blocks of 256 (or 32 for Q8_0 and Q5_1) values of K";
      return false;
    }
    // a block of a core: its groups of 8 outputs by the tiles of K, a piece each, in whole objects of a core
    const uint64_t     vBlockBytes = uint64_t(fGroupsOfCore(pOverlay)) * pShape.vTilesK * fXdnaGemvPieceBytes(vPart.vFormat);
    XdnaGemvSegment & vSegment    = pShape.aSegments[i];
    vSegment.vFormat     = vPart.vFormat;
    vSegment.vOutputs    = vPart.vOutputs;
    vSegment.vPadded     = uint32_t(fRoundUp(vPart.vOutputs, vStreams * pOverlay.vOutputs));
    vSegment.vFirst      = uint32_t(vFirst);
    vSegment.vBlocks     = uint32_t(vSegment.vPadded / (vStreams * pOverlay.vOutputs));
    vSegment.vObjects    = uint32_t((vBlockBytes + vObjectCore - 1) / vObjectCore);
    vSegment.vFirstBlock = uint32_t(vBlocks);
    vSegment.vOffsetW    = vPartW;
    vFirst += vPart.vOutputs;
    vBlocks += vSegment.vBlocks;
    vPartW += uint64_t(vSegment.vBlocks) * vSegment.vObjects * pOverlay.vObjectBytes;
  }
  if (vPaddedK > 0xffffffffu || vFirst > 0xffffffffu || vBlocks * pOverlay.vOutputs * vStreams > 0xffffffffu) {
    pError = "the multiplication is too large";
    return false;
  }
  pShape.vN       = uint32_t(vFirst);
  pShape.vBlocksY = uint32_t(vBlocks);
  pShape.vPartW   = vPartW;
  // every object of X of a core goes as many times as a stream feeds cores: an object of a stream
  pShape.vBytesX  = vObjectsX * pOverlay.vObjectBytes;
  pShape.vBytesW  = vStreams * vPartW;
  pShape.vBytesY  = vStreams * vBlocks * pOverlay.vOutputs * cXdnaGemvRowsMax * sizeof(float);
  return true;
}

bool fXdnaGemvShape(const XdnaGemvOverlay & pOverlay, uint32_t pRows, uint32_t pK, uint32_t pN, XdnaGemvFormat pFormat,
                     XdnaGemvShape & pShape, std::string & pError) {
  const XdnaGemvPart vPart = {pFormat, pN};
  return fXdnaGemvShape(pOverlay, pRows, pK, &vPart, 1, pShape, pError);
}

bool fXdnaGemvPlan(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, std::vector<uint32_t> & aInstructions,
                    std::string & pError) {
  const uint32_t vStreams = pOverlay.mStreams();
  if (pShape.vSegments == 0 || pShape.vSegments > pOverlay.vSegments || pShape.vTilesK == 0 || pShape.vObjectsX == 0) {
    pError = "the shape of the multiplication is empty";
    return false;
  }
  const uint64_t vPartY = uint64_t(pShape.vBlocksY) * pOverlay.vOutputs * cXdnaGemvRowsMax * sizeof(float);

  XdnaInstructions vInstructions;
  // Every core gets its parameters and then the lock that lets it start: objects of X after the first, rows, and
  // for every segment its blocks, objects of a block after the first, tiles of K and format (zeros past the last).
  // The cores of a stream take a part of every object of it, so they all get the same parameters. They go in one
  // block write per core: one operation instead of one per word (mlir-aie writes them word by word), since every
  // operation of a stream costs the firmware time in every command.
  std::vector<uint32_t> aParameters(2 + 4 * size_t(pOverlay.vSegments), 0);
  aParameters[0] = pShape.vObjectsX - 1;
  aParameters[1] = pShape.vRows;
  for (uint32_t i = 0; i < pShape.vSegments; i++) {
    const XdnaGemvSegment & vSegment = pShape.aSegments[i];
    aParameters[2 + 4 * i]     = vSegment.vBlocks;
    aParameters[2 + 4 * i + 1] = vSegment.vObjects - 1;
    aParameters[2 + 4 * i + 2] = pShape.vTilesK;
    aParameters[2 + 4 * i + 3] = uint32_t(vSegment.vFormat);
  }
  for (const XdnaOverlayCore & vCore : pOverlay.aCores) {
    const XdnaTile vTile = {vCore.vTileColumn, vCore.vTileRow};
    vInstructions.mBlockWrite(vTile, vCore.vParameters, aParameters.data(), aParameters.size());
    vInstructions.mWrite(vTile, cCoreLockBase + vCore.vLock * cCoreLockStep, 1);
  }

  XdnaShimDma               vDma(vInstructions, pOverlay.vColumns);
  std::vector<XdnaShimTask> aTasks;
  auto fPush = [&](const XdnaOverlayStream & pStream, XdnaDirection pDirection, uint32_t pBuffer, uint64_t pOffset,
                   uint64_t pBytes, bool pToken) {
    XdnaShimTransfer vTransfer;
    vTransfer.vColumn    = pStream.vColumn;
    vTransfer.vDirection = pDirection;
    vTransfer.vChannel   = pStream.vChannel;
    vTransfer.vBuffer    = pBuffer;
    vTransfer.vOffset    = pOffset;
    vTransfer.vBytes     = pBytes;
    vTransfer.vRepeat    = 1;
    vTransfer.vToken     = pToken;
    XdnaShimTask vTask;
    if (!vDma.mPush(vTransfer, vTask, pError)) {
      return false;
    }
    aTasks.push_back(vTask);
    return true;
  };
  // the sums of every stream, which are waited for, and then X and the weights of every stream
  for (uint32_t vStream = 0; vStream < vStreams; vStream++) {
    if (!fPush(pOverlay.aStreamsY[vStream], XdnaDirection::cToMemory, cBufferY, vStream * vPartY, vPartY, true)) {
      return false;
    }
  }
  for (uint32_t vStream = 0; vStream < vStreams; vStream++) {
    if (!fPush(pOverlay.aStreamsW[vStream], XdnaDirection::cFromMemory, cBufferX, 0, pShape.vBytesX, false) ||
        !fPush(pOverlay.aStreamsW[vStream], XdnaDirection::cFromMemory, cBufferW, vStream * pShape.vPartW,
               pShape.vPartW, false)) {
      return false;
    }
  }
  for (const XdnaShimTask & vTask : aTasks) {
    if (vTask.vToken) {
      vDma.mAwait(vTask);
    }
  }
  for (const XdnaShimTask & vTask : aTasks) {
    vDma.mFree(vTask);
  }
  aInstructions = vInstructions.mFinish(pOverlay.vColumns, pOverlay.vRows, pOverlay.vMemoryRows);
  return true;
}

uint64_t fXdnaGemvRowBytes(const XdnaGemvShape & pShape, XdnaGemvFormat pFormat) {
  if (pFormat == XdnaGemvFormat::cBfp16) {
    return uint64_t(pShape.vK) * sizeof(float);
  }
  if (pFormat == XdnaGemvFormat::cQ8_0) {
    return uint64_t(pShape.vK) / 32 * 34;
  }
  if (pFormat == XdnaGemvFormat::cQ5_1) {
    return uint64_t(pShape.vK) / 32 * 24;
  }
  return uint64_t(pShape.vK) / cSuper * fXdnaGemvBlockBytes(pFormat);
}

bool fXdnaGemvPiece(XdnaGemvFormat pFormat, const void * pRows, uint32_t pGiven, uint32_t pK, uint64_t pRowBytes,
                     uint32_t pRow, uint32_t pTile, uint8_t * pPiece) {
  const uint32_t vPieceBytes = fXdnaGemvPieceBytes(pFormat);
  std::memset(pPiece, 0, vPieceBytes);
  if (uint64_t(pTile) * cSuper >= pK) {
    return true;  // a tile past the values of K: zeros, which add nothing
  }
  if (pFormat == XdnaGemvFormat::cBfp16) {
    return fXdnaPackTileRows(static_cast<const float *>(pRows), pGiven, pK, pRow, uint64_t(pTile) * cSuper,
                              cGroupOutputs, cSuper, pPiece, XdnaRounding::cNearest) != nullptr;
  }
  const uint8_t * aBlocks[cGroupOutputs];
  for (uint32_t j = 0; j < cGroupOutputs; j++) {
    aBlocks[j] = pRow + j < pGiven ? static_cast<const uint8_t *>(pRows) + uint64_t(pRow + j) * pRowBytes +
                                         uint64_t(pTile) * fXdnaGemvBlockBytes(pFormat) :
                                     nullptr;
  }
  if (pFormat == XdnaGemvFormat::cQ6K) {
    fPieceQ6K(aBlocks, pPiece);
  } else if (pFormat == XdnaGemvFormat::cQ8_0) {
    const uint32_t vSubBlocks = uint32_t(std::min<uint64_t>(8, (uint64_t(pK) - uint64_t(pTile) * cSuper) / 32));
    fPieceQ8_0(aBlocks, vSubBlocks, 8, pPiece);
  } else if (pFormat == XdnaGemvFormat::cQ5_1) {
    const uint32_t vSubBlocks = uint32_t(std::min<uint64_t>(8, (uint64_t(pK) - uint64_t(pTile) * cSuper) / 32));
    fPieceQ5_1(aBlocks, vSubBlocks, 8, pPiece);
  } else {
    fPieceK(aBlocks, pFormat == XdnaGemvFormat::cQ5K, pPiece);
  }
  return true;
}

bool fXdnaGemvSubPiece(XdnaGemvFormat pFormat, const void * pRows, uint32_t pGiven, uint32_t pK, uint64_t pRowBytes,
                       uint32_t pRow, uint32_t pFirstK, uint32_t pSubs, uint8_t * pPiece) {
  if (pSubs == 8 && pFirstK % cSuper == 0) {
    return fXdnaGemvPiece(pFormat, pRows, pGiven, pK, pRowBytes, pRow, pFirstK / cSuper, pPiece);
  }
  const uint32_t vPieceBytes = fXdnaGemvSubPieceBytes(pFormat, pSubs);
  std::memset(pPiece, 0, vPieceBytes);
  if (pFirstK >= pK || (pFormat != XdnaGemvFormat::cQ8_0 && pFormat != XdnaGemvFormat::cQ5_1)) {
    return pFormat == XdnaGemvFormat::cQ8_0 || pFormat == XdnaGemvFormat::cQ5_1;
  }
  const uint64_t  vColByteOff = uint64_t(pFirstK / 32) * (pFormat == XdnaGemvFormat::cQ8_0 ? 34u : 24u);
  const uint8_t * aBlocks[cGroupOutputs];
  for (uint32_t j = 0; j < cGroupOutputs; j++) {
    aBlocks[j] = pRow + j < pGiven ? static_cast<const uint8_t *>(pRows) + uint64_t(pRow + j) * pRowBytes + vColByteOff :
                                     nullptr;
  }
  const uint32_t vSubBlocks = uint32_t(std::min<uint64_t>(pSubs, (uint64_t(pK) - pFirstK) / 32));
  if (pFormat == XdnaGemvFormat::cQ8_0) {
    fPieceQ8_0(aBlocks, vSubBlocks, pSubs, pPiece);
  } else {
    fPieceQ5_1(aBlocks, vSubBlocks, pSubs, pPiece);
  }
  return true;
}

bool fXdnaGemvPackStream(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, uint32_t pSegment,
                          uint32_t pStream, const void * pRows, uint32_t pGiven, uint8_t * pBuffer,
                          XdnaWorkers * pWorkers) {
  const XdnaGemvSegment & vSegment    = pShape.aSegments[pSegment];
  const uint32_t           vPerStream  = fXdnaGemvOutputsPerStream(pOverlay, pShape, pSegment);
  const uint32_t           vGiven      = std::min(pGiven, vPerStream);
  const uint32_t           vGroups     = fGroupsOfCore(pOverlay);  // groups of 8 outputs of a core in a block
  const uint64_t           vObjectCore = pOverlay.vObjectBytes / pOverlay.vSplit;
  const uint32_t           vPieceBytes = fXdnaGemvPieceBytes(vSegment.vFormat);
  const uint64_t           vRowBytes   = fXdnaGemvRowBytes(pShape, vSegment.vFormat);
  const uint64_t           vBlockBytes = uint64_t(vSegment.vObjects) * pOverlay.vObjectBytes;
  uint8_t *                vPart       = pBuffer + uint64_t(pStream) * pShape.vPartW + vSegment.vOffsetW;
  std::atomic<bool>        vFinite(true);
  // one task per block of outputs of the stream; its rows count from the first output of the stream
  auto fBlock = [&](uint32_t pBlock) {
    uint8_t * vAt = vPart + uint64_t(pBlock) * vBlockBytes;
    std::memset(vAt, 0, vBlockBytes);  // the padding of the last object of every core
    alignas(64) uint8_t aPiece[2304];
    for (uint32_t vCore = 0; vCore < pOverlay.vSplit; vCore++) {
      // the stream of pieces of this core: [tile][group of 8 outputs][piece], cut into objects of a core that
      // alternate with those of the other cores of the stream
      uint64_t vInCore = 0;
      for (uint32_t vTile = 0; vTile < pShape.vTilesK; vTile++) {
        for (uint32_t vGroup = 0; vGroup < vGroups; vGroup++) {
          const uint32_t vRow = pBlock * pOverlay.vOutputs + (vCore * vGroups + vGroup) * cGroupOutputs;
          std::memset(aPiece, 0, vPieceBytes);
          if (vSegment.vFormat == XdnaGemvFormat::cBfp16) {
            if (fXdnaPackTileRows(static_cast<const float *>(pRows), vGiven, pShape.vK, vRow,
                                   uint64_t(vTile) * cSuper, cGroupOutputs, cSuper, aPiece,
                                   XdnaRounding::cNearest) == nullptr) {
              vFinite = false;
              return;
            }
          } else if (uint64_t(vTile) * cSuper < pShape.vK) {
            const uint8_t * aBlocks[cGroupOutputs];
            for (uint32_t j = 0; j < cGroupOutputs; j++) {
              aBlocks[j] = vRow + j < vGiven ? static_cast<const uint8_t *>(pRows) + uint64_t(vRow + j) * vRowBytes +
                                                   uint64_t(vTile) * fXdnaGemvBlockBytes(vSegment.vFormat) :
                                               nullptr;
            }
            if (vSegment.vFormat == XdnaGemvFormat::cQ6K) {
              fPieceQ6K(aBlocks, aPiece);
            } else if (vSegment.vFormat == XdnaGemvFormat::cQ8_0) {
              const uint32_t vSubBlocks =
                  uint32_t(std::min<uint64_t>(8, (uint64_t(pShape.vK) - uint64_t(vTile) * cSuper) / 32));
              fPieceQ8_0(aBlocks, vSubBlocks, 8, aPiece);
            } else if (vSegment.vFormat == XdnaGemvFormat::cQ5_1) {
              const uint32_t vSubBlocks =
                  uint32_t(std::min<uint64_t>(8, (uint64_t(pShape.vK) - uint64_t(vTile) * cSuper) / 32));
              fPieceQ5_1(aBlocks, vSubBlocks, 8, aPiece);
            } else {
              fPieceK(aBlocks, vSegment.vFormat == XdnaGemvFormat::cQ5K, aPiece);
            }
          }
          // the piece, across objects of the core where it does not fit in one
          for (uint32_t vDone = 0; vDone < vPieceBytes;) {
            const uint64_t vObject = vInCore / vObjectCore;
            const uint64_t vWithin = vInCore % vObjectCore;
            const uint64_t vBytes  = std::min<uint64_t>(vPieceBytes - vDone, vObjectCore - vWithin);
            std::memcpy(vAt + vObject * pOverlay.vObjectBytes + vCore * vObjectCore + vWithin, aPiece + vDone, vBytes);
            vDone += uint32_t(vBytes);
            vInCore += vBytes;
          }
        }
      }
    }
  };
  if (pWorkers != nullptr) {
    pWorkers->mForEach(vSegment.vBlocks, fBlock);
  } else {
    for (uint32_t vBlock = 0; vBlock < vSegment.vBlocks; vBlock++) {
      fBlock(vBlock);
    }
  }
  return vFinite;
}

bool fXdnaGemvPackW(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, const void * pRows,
                     uint8_t * pBuffer, XdnaWorkers * pWorkers) {
  const XdnaGemvSegment & vSegment   = pShape.aSegments[0];
  const uint32_t           vPerStream = fXdnaGemvOutputsPerStream(pOverlay, pShape, 0);
  const uint64_t           vRowBytes  = fXdnaGemvRowBytes(pShape, vSegment.vFormat);
  bool                     vFinite    = true;
  for (uint32_t vStream = 0; vStream < pOverlay.mStreams(); vStream++) {
    const uint64_t vFirst = uint64_t(vStream) * vPerStream;
    const uint32_t vGiven =
        vFirst >= vSegment.vOutputs ? 0 : uint32_t(std::min<uint64_t>(vPerStream, vSegment.vOutputs - vFirst));
    const uint8_t * vFrom = static_cast<const uint8_t *>(pRows) + std::min<uint64_t>(vFirst, vSegment.vOutputs) * vRowBytes;
    vFinite = fXdnaGemvPackStream(pOverlay, pShape, 0, vStream, vFrom, vGiven, pBuffer, pWorkers) && vFinite;
  }
  return vFinite;
}

bool fXdnaGemvPackX(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, const float * pX, size_t pStride,
                     uint8_t * pBuffer) {
  // [K / 8][rows * 9 bytes, padded to a multiple of 4]: every group of 8 values of K holds that group of every row,
  // the rows one after another; after the last group, zeros up to the whole objects of a core. Written first one
  // object after another, and then every object of a core copied to its place as many times as a stream feeds cores.
  std::memset(pBuffer, 0, pShape.vBytesX);
  const uint32_t vGroups = pShape.vPaddedK / cBlock;
  bool           vFinite = true;
  for (uint32_t vGroup = 0; vGroup < vGroups; vGroup++) {
    const uint64_t vColumn = uint64_t(vGroup) * cBlock;
    uint8_t *      vAt     = pBuffer + uint64_t(vGroup) * pShape.vGroupX;
    if (vColumn >= pShape.vK) {
      continue;  // zeros, already there
    }
    if (vColumn + cBlock <= pShape.vK) {
      vFinite = fXdnaBfp16EncodeNearest(pX + vColumn, pStride, pShape.vRows, vAt) && vFinite;
      continue;
    }
    for (uint32_t vRow = 0; vRow < pShape.vRows; vRow++) {
      float aValues[cBlock] = {};
      for (uint32_t i = 0; vColumn + i < pShape.vK; i++) {
        aValues[i] = pX[vRow * pStride + vColumn + i];
      }
      vFinite = fXdnaBfp16Encode(aValues, vAt + vRow * cBlockBytes, XdnaRounding::cNearest) && vFinite;
    }
  }
  // From the last object to the first: the copies of an object start at or past where it is written, and past every
  // object before it, so nothing is overwritten before it is copied; its own place goes last.
  if (pOverlay.vSplit > 1) {
    const uint64_t vObjectCore = pOverlay.vObjectBytes / pOverlay.vSplit;
    for (uint32_t vObject = pShape.vObjectsX; vObject-- > 0;) {
      const uint8_t * vFrom = pBuffer + uint64_t(vObject) * vObjectCore;
      for (uint32_t vCopy = pOverlay.vSplit; vCopy-- > 0;) {
        std::memmove(pBuffer + (uint64_t(vObject) * pOverlay.vSplit + vCopy) * vObjectCore, vFrom, vObjectCore);
      }
    }
  }
  return vFinite;
}

void fXdnaGemvUnpackY(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, const float * pBuffer,
                       const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers) {
  const uint32_t vStreams       = pOverlay.mStreams();
  const uint32_t vGroupsOfBlock = pOverlay.vOutputs / cGroupOutputs;
  // The buffer goes in its own order: [stream][segment][block][group of 8 outputs][8 rows][8 outputs], so every row
  // of a group is 8 consecutive outputs, which mostly fall in one target and are copied at once. One task per stream.
  auto fStream = [&](uint32_t pStream) {
    for (uint32_t vIndex = 0; vIndex < pShape.vSegments; vIndex++) {
      const XdnaGemvSegment & vSegment   = pShape.aSegments[vIndex];
      const uint32_t           vPerStream = vSegment.vPadded / vStreams;
      const uint32_t           vGroups    = vSegment.vBlocks * vGroupsOfBlock;  // groups of 8 of the stream
      uint32_t                 vTarget    = 0;
      for (uint32_t vGroup = 0; vGroup < vGroups; vGroup++) {
        const uint32_t vInSegment = pStream * vPerStream + vGroup * cGroupOutputs;
        if (vInSegment >= vSegment.vOutputs) {
          break;
        }
        const uint32_t vFirst = vSegment.vFirst + vInSegment;  // its first output in the numbering of the targets
        const uint32_t vValid = std::min<uint32_t>(cGroupOutputs, vSegment.vOutputs - vInSegment);
        const float *  vFrom  = pBuffer + (uint64_t(pStream) * pShape.vBlocksY * vGroupsOfBlock +
                                           uint64_t(vSegment.vFirstBlock) * vGroupsOfBlock + vGroup) *
                                              cXdnaGemvRowsMax * cGroupOutputs;
        // the targets are sorted: the first that may take this group
        while (vTarget < pTargets && aTargets[vTarget].vFirst + aTargets[vTarget].vCount <= vFirst) {
          vTarget++;
        }
        if (vTarget < pTargets && vValid == cGroupOutputs && vFirst >= aTargets[vTarget].vFirst &&
            vFirst + cGroupOutputs <= aTargets[vTarget].vFirst + aTargets[vTarget].vCount) {
          const XdnaGemmTarget & vInto = aTargets[vTarget];
          float *                 vTo   = vInto.vData + (vFirst - vInto.vFirst);
          for (uint32_t vRow = 0; vRow < pShape.vRows; vRow++) {
            std::memcpy(vTo + vRow * vInto.vStride, vFrom + vRow * cGroupOutputs, cGroupOutputs * sizeof(float));
          }
          continue;
        }
        // a group across two targets, or the last one of the segment: output by output
        for (uint32_t vColumn = 0; vColumn < vValid; vColumn++) {
          const uint32_t vOutput = vFirst + vColumn;
          for (uint32_t t = vTarget; t < pTargets && aTargets[t].vFirst <= vOutput; t++) {
            if (vOutput - aTargets[t].vFirst < aTargets[t].vCount) {
              for (uint32_t vRow = 0; vRow < pShape.vRows; vRow++) {
                aTargets[t].vData[vRow * aTargets[t].vStride + (vOutput - aTargets[t].vFirst)] =
                    vFrom[vRow * cGroupOutputs + vColumn];
              }
              break;
            }
          }
        }
      }
    }
  };
  // threads only where there is enough to share: the large matrices, as the one of the logits
  if (pWorkers != nullptr && uint64_t(pShape.vBlocksY) * pOverlay.vOutputs * vStreams >= (1u << 16)) {
    pWorkers->mForEach(vStreams, fStream);
  } else {
    for (uint32_t vStream = 0; vStream < vStreams; vStream++) {
      fStream(vStream);
    }
  }
}
