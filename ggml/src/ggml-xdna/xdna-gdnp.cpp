#include "xdna-gdnp.h"

#include "xdna-bfp16.h"
#include "xdna-gemm.h"
#include "xdna-glu.h"
#include "xdna-workers.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <functional>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

const uint32_t cRows    = 4;
const uint32_t cColumns = 8;

// the objects of a chunk (kernels/xdna2-gdn-prompt.cc), in bytes: the first one
const uint32_t cAtQ       = cXdnaGdnChunk * cXdnaGdnDim * 2;
const uint32_t cAtGamma   = 2 * cAtQ;
const uint32_t cGammaHead = 3 * 64 * 2;
// and the second one
const uint32_t cAtBeta  = 2 * cXdnaGdnDim * cXdnaGdnChunk * 2;
const uint32_t cAtE     = cAtBeta + 2 * cXdnaGdnChunk * 4;
const uint32_t cAtG     = cAtE + 2 * cXdnaGdnChunk * 4;
const uint32_t cAtDecay = cAtG + 2 * cXdnaGdnChunk * 4;

void fForEach(XdnaWorkers * pWorkers, uint32_t pCount, const std::function<void(uint32_t)> & pTask) {
  if (pWorkers != nullptr) {
    pWorkers->mForEach(pCount, pTask);
    return;
  }
  for (uint32_t i = 0; i < pCount; i++) {
    pTask(i);
  }
}

// the head of keys of row pRow of round pRound, and the block of 8 value indices of the core of column pColumn of it
inline uint32_t fKeyHead(uint32_t pRound, uint32_t pRow) {
  return 2 * pRound + pRow / 2;
}

inline uint32_t fBlockJ(uint32_t pRow, uint32_t pColumn) {
  return 8 * (pRow % 2) + pColumn;
}

#if defined(__x86_64__)
const bool cHasAvx2 = __builtin_cpu_supports("avx2") != 0;

// 8 floats times pScale, rounded to bf16 to the nearest (ties to even, as fXdnaBf16FromFloat for finite values)
__attribute__((target("avx2"))) inline __m128i fBf16x8(__m256 pValues) {
  __m256i       vBits = _mm256_castps_si256(pValues);
  const __m256i vOdd  = _mm256_and_si256(_mm256_srli_epi32(vBits, 16), _mm256_set1_epi32(1));
  vBits               = _mm256_srli_epi32(_mm256_add_epi32(vBits, _mm256_add_epi32(_mm256_set1_epi32(0x7fff), vOdd)), 16);
  return _mm_packus_epi32(_mm256_castsi256_si128(vBits), _mm256_extracti128_si256(vBits, 1));
}

// k and q of a chunk in groups [2 blocks of t][16 blocks of d][8 t][8 d]: every row of 8 values is a store of 16 bytes
__attribute__((target("avx2"))) void fPackKQAvx2(const XdnaGdnIo & pIo, uint32_t pKey, uint32_t pFirst,
                                                 const float * pRmsQ, const float * pRmsK, uint16_t * pK,
                                                 uint16_t * pQ) {
  const __m256 vScale = _mm256_set1_ps(pIo.vScale);
  const __m256 vNorm  = _mm256_set1_ps(pIo.vNormScale);
  for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
    const float * vRowK = pIo.vK + size_t(pFirst + vT) * pIo.vKToken + size_t(pKey) * pIo.vKHead;
    const float * vRowQ = pIo.vQ + size_t(pFirst + vT) * pIo.vQToken + size_t(pKey) * pIo.vQHead;
    if (pRmsQ != nullptr) {
      // llama.cpp-xdna: the L2 norms, (x * scale of the RMS_NORM) * the factor of the SCALE, and q then times vScale
      const __m256 vRmsK = _mm256_set1_ps(pRmsK[vT]);
      const __m256 vRmsQ = _mm256_set1_ps(pRmsQ[vT]);
      for (uint32_t vDB = 0; vDB < cXdnaGdnDim / 8; vDB++) {
        const uint32_t vAt = (((vT / 8) * 16 + vDB) * 8 + vT % 8) * 8;
        _mm_storeu_si128(reinterpret_cast<__m128i *>(pK + vAt),
                         fBf16x8(_mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(vRowK + 8 * vDB), vRmsK), vNorm)));
        _mm_storeu_si128(reinterpret_cast<__m128i *>(pQ + vAt),
                         fBf16x8(_mm256_mul_ps(
                             _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(vRowQ + 8 * vDB), vRmsQ), vNorm), vScale)));
      }
      continue;
    }
    for (uint32_t vDB = 0; vDB < cXdnaGdnDim / 8; vDB++) {
      const uint32_t vAt = (((vT / 8) * 16 + vDB) * 8 + vT % 8) * 8;
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pK + vAt), fBf16x8(_mm256_loadu_ps(vRowK + 8 * vDB)));
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pQ + vAt),
                       fBf16x8(_mm256_mul_ps(_mm256_loadu_ps(vRowQ + 8 * vDB), vScale)));
    }
  }
}

__attribute__((target("avx2"))) void fPackSingleKQAvx2(const float * pRows, const float * pRms, float pNormScale,
                                                       float pExtraScale, bool pUseExtra, uint16_t * pOut) {
  const __m256 vNorm  = _mm256_set1_ps(pNormScale);
  const __m256 vExtra = _mm256_set1_ps(pExtraScale);
  for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
    const float * vRow = pRows + size_t(vT) * cXdnaGdnDim;
    if (pRms != nullptr) {
      const __m256 vRms = _mm256_set1_ps(pRms[vT]);
      for (uint32_t vDB = 0; vDB < cXdnaGdnDim / 8; vDB++) {
        const uint32_t vAt     = (((vT / 8) * 16 + vDB) * 8 + vT % 8) * 8;
        __m256         vScaled = _mm256_mul_ps(_mm256_mul_ps(_mm256_loadu_ps(vRow + 8 * vDB), vRms), vNorm);
        if (pUseExtra) {
          vScaled = _mm256_mul_ps(vScaled, vExtra);
        }
        _mm_storeu_si128(reinterpret_cast<__m128i *>(pOut + vAt), fBf16x8(vScaled));
      }
      continue;
    }
    for (uint32_t vDB = 0; vDB < cXdnaGdnDim / 8; vDB++) {
      const uint32_t vAt     = (((vT / 8) * 16 + vDB) * 8 + vT % 8) * 8;
      __m256         vValues = _mm256_loadu_ps(vRow + 8 * vDB);
      if (pUseExtra) {
        vValues = _mm256_mul_ps(vValues, vExtra);
      }
      _mm_storeu_si128(reinterpret_cast<__m128i *>(pOut + vAt), fBf16x8(vValues));
    }
  }
}

// v^T of a chunk and a head in groups [16 blocks of j][2 blocks of t][8 j][8 t]: every group is 8 rows of 8 tokens,
// transposed in registers
__attribute__((target("avx2"))) void fPackVRowsAvx2(const float * pBase, size_t pTokenStride, uint16_t * pV) {
  for (uint32_t vTB = 0; vTB < 2; vTB++) {
    const float * aRows[8];
    for (uint32_t i = 0; i < 8; i++) {
      aRows[i] = pBase + size_t(8 * vTB + i) * pTokenStride;
    }
    for (uint32_t vJB = 0; vJB < cXdnaGdnDim / 8; vJB++) {
      // rows: tokens; columns: j
      const __m256 v0 = _mm256_loadu_ps(aRows[0] + 8 * vJB);
      const __m256 v1 = _mm256_loadu_ps(aRows[1] + 8 * vJB);
      const __m256 v2 = _mm256_loadu_ps(aRows[2] + 8 * vJB);
      const __m256 v3 = _mm256_loadu_ps(aRows[3] + 8 * vJB);
      const __m256 v4 = _mm256_loadu_ps(aRows[4] + 8 * vJB);
      const __m256 v5 = _mm256_loadu_ps(aRows[5] + 8 * vJB);
      const __m256 v6 = _mm256_loadu_ps(aRows[6] + 8 * vJB);
      const __m256 v7 = _mm256_loadu_ps(aRows[7] + 8 * vJB);
      const __m256 t0 = _mm256_unpacklo_ps(v0, v1);
      const __m256 t1 = _mm256_unpackhi_ps(v0, v1);
      const __m256 t2 = _mm256_unpacklo_ps(v2, v3);
      const __m256 t3 = _mm256_unpackhi_ps(v2, v3);
      const __m256 t4 = _mm256_unpacklo_ps(v4, v5);
      const __m256 t5 = _mm256_unpackhi_ps(v4, v5);
      const __m256 t6 = _mm256_unpacklo_ps(v6, v7);
      const __m256 t7 = _mm256_unpackhi_ps(v6, v7);
      const __m256 s0 = _mm256_shuffle_ps(t0, t2, 0x44);
      const __m256 s1 = _mm256_shuffle_ps(t0, t2, 0xee);
      const __m256 s2 = _mm256_shuffle_ps(t1, t3, 0x44);
      const __m256 s3 = _mm256_shuffle_ps(t1, t3, 0xee);
      const __m256 s4 = _mm256_shuffle_ps(t4, t6, 0x44);
      const __m256 s5 = _mm256_shuffle_ps(t4, t6, 0xee);
      const __m256 s6 = _mm256_shuffle_ps(t5, t7, 0x44);
      const __m256 s7 = _mm256_shuffle_ps(t5, t7, 0xee);
      // rows: j; columns: tokens
      const __m256 aCols[8] = {
        _mm256_permute2f128_ps(s0, s4, 0x20), _mm256_permute2f128_ps(s1, s5, 0x20),
        _mm256_permute2f128_ps(s2, s6, 0x20), _mm256_permute2f128_ps(s3, s7, 0x20),
        _mm256_permute2f128_ps(s0, s4, 0x31), _mm256_permute2f128_ps(s1, s5, 0x31),
        _mm256_permute2f128_ps(s2, s6, 0x31), _mm256_permute2f128_ps(s3, s7, 0x31),
      };
      uint16_t * vGroup = pV + (vJB * 2 + vTB) * 64;
      for (uint32_t i = 0; i < 8; i++) {
        _mm_storeu_si128(reinterpret_cast<__m128i *>(vGroup + 8 * i), fBf16x8(aCols[i]));
      }
    }
  }
}

// llama.cpp-xdna: a row of the outputs, 16 runs of 8 floats (pFrom[b] + pAt for run b) one after the other into pTo,
// aligned to 32 bytes, past the cache: a whole row is whole lines, and nothing reads it soon
__attribute__((target("avx2"))) inline void fRowOf16(float * pTo, const float * const * pFrom, size_t pAt) {
  for (uint32_t b = 0; b < 16; b++) {
    _mm256_stream_ps(pTo + 8 * b, _mm256_loadu_ps(pFrom[b] + pAt));
  }
}
#endif

// the chunk pChunk of the head of keys pKey and value-head pair pass pPass (out of pRatio value heads per key head)
void fPackChunk(const XdnaGdnIo & pIo, uint32_t pKey, uint32_t pPass, uint32_t pRatio, uint32_t pChunk,
                uint8_t * pObjects) {
  uint8_t *      vIn0   = pObjects;
  uint8_t *      vIn1   = pObjects + cXdnaGdnTile;
  const uint32_t vFirst = pChunk * cXdnaGdnChunk;
  // the tokens of the chunk that exist; the ones after them are empty (see xdna-gdnp.h)
  const uint32_t vReal = vFirst >= pIo.vTokens ? 0 : std::min<uint32_t>(cXdnaGdnChunk, pIo.vTokens - vFirst);
  if (!pIo.vPrepackedKQV) {
    // k and q in groups [2 blocks of t][16 blocks of d][8 t][8 d], q scaled
    uint16_t * vK = reinterpret_cast<uint16_t *>(vIn0);
    uint16_t * vQ = reinterpret_cast<uint16_t *>(vIn0 + cAtQ);
    float      aLocalRmsQ[cXdnaGdnChunk] = {};
    float      aLocalRmsK[cXdnaGdnChunk] = {};
    if (pIo.vL2InPack && vReal > 0) {
      fXdnaRmsScales(pIo.vQ + size_t(vFirst) * pIo.vQToken + size_t(pKey) * pIo.vQHead, vReal, cXdnaGdnDim,
                     pIo.vQToken, pIo.vL2Eps, aLocalRmsQ);
      fXdnaRmsScales(pIo.vK + size_t(vFirst) * pIo.vKToken + size_t(pKey) * pIo.vKHead, vReal, cXdnaGdnDim,
                     pIo.vKToken, pIo.vL2Eps, aLocalRmsK);
    }
    const float * vRmsQ = pIo.vL2InPack ? aLocalRmsQ :
                          pIo.vQNorms != nullptr ? pIo.vQNorms + size_t(pKey) * pIo.vTokens + vFirst : nullptr;
    const float * vRmsK = pIo.vL2InPack ? aLocalRmsK :
                          pIo.vKNorms != nullptr ? pIo.vKNorms + size_t(pKey) * pIo.vTokens + vFirst : nullptr;
#if defined(__x86_64__)
    if (cHasAvx2 && vReal == cXdnaGdnChunk) {
      fPackKQAvx2(pIo, pKey, vFirst, vRmsQ, vRmsK, vK, vQ);
    } else
#endif
    for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
      const uint32_t vTB = vT / 8;
      if (vT >= vReal) {
        for (uint32_t vDB = 0; vDB < cXdnaGdnDim / 8; vDB++) {
          const uint32_t vAt = ((vTB * 16 + vDB) * 8 + vT % 8) * 8;
          std::memset(vK + vAt, 0, 8 * sizeof(uint16_t));
          std::memset(vQ + vAt, 0, 8 * sizeof(uint16_t));
        }
        continue;
      }
      const float * vRowK  = pIo.vK + size_t(vFirst + vT) * pIo.vKToken + size_t(pKey) * pIo.vKHead;
      const float * vRowQ  = pIo.vQ + size_t(vFirst + vT) * pIo.vQToken + size_t(pKey) * pIo.vQHead;
      const float   vScaleK = vRmsK != nullptr ? vRmsK[vT] : 1.0f;
      const float   vScaleQ = vRmsQ != nullptr ? vRmsQ[vT] : 1.0f;
      for (uint32_t vD = 0; vD < cXdnaGdnDim; vD++) {
        const uint32_t vAt = ((vTB * 16 + vD / 8) * 8 + vT % 8) * 8 + vD % 8;
        if (vRmsQ != nullptr) {
          const float vK2 = (vRowK[vD] * vScaleK) * pIo.vNormScale;
          const float vQ2 = (vRowQ[vD] * vScaleQ) * pIo.vNormScale;
          vK[vAt]         = fXdnaBf16FromFloat(vK2);
          vQ[vAt]         = fXdnaBf16FromFloat(vQ2 * pIo.vScale);
          continue;
        }
        vK[vAt] = fXdnaBf16FromFloat(vRowK[vD]);
        vQ[vAt] = fXdnaBf16FromFloat(vRowQ[vD] * pIo.vScale);
      }
    }
  }
  for (uint32_t vP = 0; vP < 2; vP++) {
    const uint32_t vSlot = 2 * pPass + vP;
    if (vSlot >= pRatio) {
      float * vBeta = reinterpret_cast<float *>(vIn1 + cAtBeta) + vP * cXdnaGdnChunk;
      float * vE    = reinterpret_cast<float *>(vIn1 + cAtE) + vP * cXdnaGdnChunk;
      float * vG    = reinterpret_cast<float *>(vIn1 + cAtG) + vP * cXdnaGdnChunk;
      for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
        vBeta[vT] = 0.0f;
        vE[vT]    = 1.0f;
        vG[vT]    = 1.0f;
      }
      reinterpret_cast<float *>(vIn1 + cAtDecay)[vP] = 1.0f;
      uint16_t * vGamma = reinterpret_cast<uint16_t *>(vIn0 + cAtGamma + vP * cGammaHead);
      std::memset(vGamma, 0, cGammaHead);
      uint16_t * vV = reinterpret_cast<uint16_t *>(vIn1) + vP * cXdnaGdnDim * cXdnaGdnChunk;
      std::memset(vV, 0, size_t(cXdnaGdnDim) * cXdnaGdnChunk * sizeof(uint16_t));
      continue;
    }
    const uint32_t vHead = pKey + vSlot * pIo.vKeyHeads;
    // e^g of every token (1 for an empty one), and from them by products Gamma, E, G and the decay
    double aExp[cXdnaGdnChunk];
    for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
      aExp[vT] = vT < vReal ? std::exp(double(pIo.vG[size_t(vFirst + vT) * pIo.vValueHeads + vHead])) : 1.0;
    }
    float * vBeta  = reinterpret_cast<float *>(vIn1 + cAtBeta) + vP * cXdnaGdnChunk;
    float * vE     = reinterpret_cast<float *>(vIn1 + cAtE) + vP * cXdnaGdnChunk;
    float * vG     = reinterpret_cast<float *>(vIn1 + cAtG) + vP * cXdnaGdnChunk;
    double  vSoFar = 1.0;
    for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
      vSoFar   *= aExp[vT];
      vE[vT]    = float(vSoFar);
      vBeta[vT] = vT < vReal ? pIo.vBeta[size_t(vFirst + vT) * pIo.vValueHeads + vHead] : 0.0f;
    }
    reinterpret_cast<float *>(vIn1 + cAtDecay)[vP] = float(vSoFar);
    double vAfter = 1.0;
    for (uint32_t vT = cXdnaGdnChunk; vT-- > 0;) {
      vG[vT]  = float(vAfter);
      vAfter *= aExp[vT];
    }
    // Gamma[t][t'] = the product of e^g of the tokens t' + 1 to t, in the groups 00, 10 and 11, zero above the
    // diagonal
    uint16_t * vGamma = reinterpret_cast<uint16_t *>(vIn0 + cAtGamma + vP * cGammaHead);
    std::memset(vGamma, 0, cGammaHead);
    for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
      double vValue = 1.0;
      for (uint32_t vU = vT + 1; vU-- > 0;) {
        const uint32_t vGroup = vT < 8 ? 0 : (vU < 8 ? 1 : 2);
        vGamma[vGroup * 64 + (vT % 8) * 8 + vU % 8] = fXdnaBf16FromFloat(float(vValue));
        vValue *= aExp[vU];
      }
    }
    if (pIo.vPrepackedKQV) {
      continue;
    }
    // v^T in groups [16 blocks of j][2 blocks of t][8 j][8 t]
    uint16_t * vV = reinterpret_cast<uint16_t *>(vIn1) + vP * cXdnaGdnDim * cXdnaGdnChunk;
#if defined(__x86_64__)
    if (cHasAvx2 && vReal == cXdnaGdnChunk) {
      fPackVRowsAvx2(pIo.vV + size_t(vFirst) * pIo.vVToken + size_t(vHead) * pIo.vVHead, pIo.vVToken, vV);
      continue;
    }
#endif
    for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
      const uint32_t vTB = vT / 8;
      if (vT >= vReal) {
        for (uint32_t vJ = 0; vJ < cXdnaGdnDim; vJ++) {
          vV[((vJ / 8 * 2 + vTB) * 8 + vJ % 8) * 8 + vT % 8] = 0;
        }
        continue;
      }
      const float * vRowV = pIo.vV + size_t(vFirst + vT) * pIo.vVToken + size_t(vHead) * pIo.vVHead;
      for (uint32_t vJ = 0; vJ < cXdnaGdnDim; vJ++) {
        vV[((vJ / 8 * 2 + vTB) * 8 + vJ % 8) * 8 + vT % 8] = fXdnaBf16FromFloat(vRowV[vJ]);
      }
    }
  }
}

}  // namespace

bool fXdnaGdnShape(const XdnaGdnIo & pIo, uint32_t & pRounds, uint32_t & pGroups) {
  const uint32_t vGroupTokens = cXdnaGdnChunksOut * cXdnaGdnChunk;
  if (pIo.vTokens == 0 || pIo.vKeyHeads == 0 || pIo.vKeyHeads % 2 != 0 || pIo.vValueHeads < pIo.vKeyHeads ||
      pIo.vValueHeads % pIo.vKeyHeads != 0 || pIo.vValueHeads > 4 * pIo.vKeyHeads) {
    return false;
  }
  const uint32_t vRatio  = pIo.vValueHeads / pIo.vKeyHeads;
  const uint32_t vPasses = (vRatio + 1) / 2;
  pRounds = (pIo.vKeyHeads / 2) * vPasses;
  pGroups = (pIo.vTokens + vGroupTokens - 1) / vGroupTokens;
  return true;
}

void fXdnaGdnPackConvHeadChunk(const XdnaGdnPrepack & pPrepack, uint32_t pChannelHead, uint32_t pChunk,
                               const float * pRows, bool pZeroTail) {
  if (pPrepack.vB == nullptr || pPrepack.vParts == 0 || pPrepack.vKeyHeads == 0 || pPrepack.vTokens == 0) {
    return;
  }
  const uint32_t vKeyRounds   = pPrepack.vKeyHeads / 2;
  const uint32_t vRatio       = pPrepack.vValueHeads / pPrepack.vKeyHeads;
  const uint32_t vPasses      = (vRatio + 1) / 2;
  const uint32_t vRounds      = vKeyRounds * vPasses;
  const uint32_t vRoundsPart  = vRounds / pPrepack.vParts;
  const uint32_t vGroupTokens = cXdnaGdnChunksOut * cXdnaGdnChunk;
  const uint32_t vGroups      = (pPrepack.vTokens + vGroupTokens - 1) / vGroupTokens;
  const uint32_t vChunks      = vGroups * cXdnaGdnChunksOut;
  const uint32_t vFirst       = pChunk * cXdnaGdnChunk;
  const uint32_t vReal        = vFirst >= pPrepack.vTokens ? 0 :
                                                             std::min<uint32_t>(cXdnaGdnChunk, pPrepack.vTokens - vFirst);
  auto fObjectsOf = [&](uint32_t pAbsRound, uint32_t pKeyParity, uint32_t pChk) -> uint8_t * {
    const uint32_t vPart    = pAbsRound / vRoundsPart;
    const uint32_t vSlotKey = 2 * (pAbsRound % vRoundsPart) + pKeyParity;
    return pPrepack.vB + uint64_t(vPart) * pPrepack.vPartBytesB +
           (uint64_t(vSlotKey) * vChunks + pChk) * 2 * cXdnaGdnTile;
  };
  if (pChannelHead < 2 * pPrepack.vKeyHeads) {
    const bool     vIsQ    = pChannelHead < pPrepack.vKeyHeads;
    const uint32_t vKey    = vIsQ ? pChannelHead : pChannelHead - pPrepack.vKeyHeads;
    const uint32_t vOffset = vIsQ ? cAtQ : 0;
    float          aRms[cXdnaGdnChunk] = {};
    if (pPrepack.vHasL2 && vReal > 0) {
      fXdnaRmsScales(pRows, vReal, cXdnaGdnDim, cXdnaGdnDim, pPrepack.vL2Eps, aRms);
    }
    const float * vRms  = pPrepack.vHasL2 ? aRms : nullptr;
    uint8_t *     vObj0 = fObjectsOf(vKey / 2, vKey % 2, pChunk) + vOffset;
    uint16_t *    vOut0 = reinterpret_cast<uint16_t *>(vObj0);
#if defined(__x86_64__)
    if (cHasAvx2 && vReal == cXdnaGdnChunk) {
      fPackSingleKQAvx2(pRows, vRms, pPrepack.vNormScale, pPrepack.vScale, vIsQ, vOut0);
    } else
#endif
    for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
      const uint32_t vTB = vT / 8;
      if (vT >= vReal) {
        for (uint32_t vDB = 0; vDB < cXdnaGdnDim / 8; vDB++) {
          const uint32_t vAt = ((vTB * 16 + vDB) * 8 + vT % 8) * 8;
          std::memset(vOut0 + vAt, 0, 8 * sizeof(uint16_t));
        }
        continue;
      }
      const float * vRow   = pRows + size_t(vT) * cXdnaGdnDim;
      const float   vScale = vRms != nullptr ? vRms[vT] : 1.0f;
      for (uint32_t vD = 0; vD < cXdnaGdnDim; vD++) {
        const uint32_t vAt  = ((vTB * 16 + vD / 8) * 8 + vT % 8) * 8 + vD % 8;
        float          vVal = vRms != nullptr ? (vRow[vD] * vScale) * pPrepack.vNormScale : vRow[vD];
        if (vIsQ) {
          vVal *= pPrepack.vScale;
        }
        vOut0[vAt] = fXdnaBf16FromFloat(vVal);
      }
    }
    for (uint32_t vPass = 1; vPass < vPasses; vPass++) {
      std::memcpy(fObjectsOf(vPass * vKeyRounds + vKey / 2, vKey % 2, pChunk) + vOffset, vObj0, cAtQ);
    }
    if (pZeroTail) {
      for (uint32_t vChk = pChunk + 1; vChk < vChunks; vChk++) {
        for (uint32_t vPass = 0; vPass < vPasses; vPass++) {
          std::memset(fObjectsOf(vPass * vKeyRounds + vKey / 2, vKey % 2, vChk) + vOffset, 0, cAtQ);
        }
      }
    }
    return;
  }
  const uint32_t vHead = pChannelHead - 2 * pPrepack.vKeyHeads;
  if (vHead >= pPrepack.vValueHeads) {
    return;
  }
  const uint32_t vKey      = vHead % pPrepack.vKeyHeads;
  const uint32_t vSlot     = vHead / pPrepack.vKeyHeads;
  const uint32_t vPass     = vSlot / 2;
  const uint32_t vP        = vSlot % 2;
  const uint32_t vAbsRound = vPass * vKeyRounds + vKey / 2;
  uint16_t *     vV        = reinterpret_cast<uint16_t *>(fObjectsOf(vAbsRound, vKey % 2, pChunk) + cXdnaGdnTile) +
                             vP * cXdnaGdnDim * cXdnaGdnChunk;
#if defined(__x86_64__)
  if (cHasAvx2 && vReal == cXdnaGdnChunk) {
    fPackVRowsAvx2(pRows, cXdnaGdnDim, vV);
  } else
#endif
  for (uint32_t vT = 0; vT < cXdnaGdnChunk; vT++) {
    const uint32_t vTB = vT / 8;
    if (vT >= vReal) {
      for (uint32_t vJ = 0; vJ < cXdnaGdnDim; vJ++) {
        vV[((vJ / 8 * 2 + vTB) * 8 + vJ % 8) * 8 + vT % 8] = 0;
      }
      continue;
    }
    const float * vRowV = pRows + size_t(vT) * cXdnaGdnDim;
    for (uint32_t vJ = 0; vJ < cXdnaGdnDim; vJ++) {
      vV[((vJ / 8 * 2 + vTB) * 8 + vJ % 8) * 8 + vT % 8] = fXdnaBf16FromFloat(vRowV[vJ]);
    }
  }
  if (pZeroTail) {
    for (uint32_t vChk = pChunk + 1; vChk < vChunks; vChk++) {
      uint16_t * vVTail = reinterpret_cast<uint16_t *>(fObjectsOf(vAbsRound, vKey % 2, vChk) + cXdnaGdnTile) +
                          vP * cXdnaGdnDim * cXdnaGdnChunk;
      std::memset(vVTail, 0, size_t(cXdnaGdnDim) * cXdnaGdnChunk * sizeof(uint16_t));
    }
  }
}

void fXdnaGdnPack(const XdnaGdnIo & pIo, uint32_t pFirstRound, uint32_t pRounds, uint8_t * pA, uint8_t * pB,
                   XdnaWorkers * pWorkers, const XdnaGdnFlush & pFlush) {
  uint32_t vRounds = 0;
  uint32_t vGroups = 0;
  if (!fXdnaGdnShape(pIo, vRounds, vGroups) || pRounds == 0 || pFirstRound + pRounds > vRounds) {
    return;
  }
  const uint32_t vKeyRounds = pIo.vKeyHeads / 2;
  const uint32_t vRatio     = pIo.vValueHeads / pIo.vKeyHeads;
  const uint32_t vChunks    = vGroups * cXdnaGdnChunksOut;
  // the chunks of every head of keys of the part: [slot of round][chunk][2 objects]
  fForEach(pWorkers, 2 * pRounds * vChunks, [&](uint32_t pTask) {
    const uint32_t vSlotKey  = pTask / vChunks;
    const uint32_t vChunk    = pTask % vChunks;
    const uint32_t vAbsRound = pFirstRound + vSlotKey / 2;
    const uint32_t vPass     = vAbsRound / vKeyRounds;
    const uint32_t vKey      = 2 * (vAbsRound % vKeyRounds) + (vSlotKey % 2);
    uint8_t *      vObjects  = pB + (uint64_t(vSlotKey) * vChunks + vChunk) * 2 * cXdnaGdnTile;
    fPackChunk(pIo, vKey, vPass, vRatio, vChunk, vObjects);
    if (pFlush) {
      pFlush(vObjects, 2 * cXdnaGdnTile);
    }
  });
  // the bands: [column][round][the states of the 4 rows, then 2 objects of work], a state [2 heads][8 j][128]
  fForEach(pWorkers, cColumns * pRounds, [&](uint32_t pTask) {
    const uint32_t vColumn = pTask / pRounds;
    const uint32_t vRound  = pFirstRound + pTask % pRounds;
    const uint32_t vPass   = vRound / vKeyRounds;
    uint8_t *      vBand   = pA + uint64_t(pTask) * cXdnaGdnBand * cXdnaGdnTile;
    for (uint32_t vRow = 0; vRow < cRows; vRow++) {
      const uint32_t vKey   = fKeyHead(vRound % vKeyRounds, vRow);
      const uint32_t vBlock = fBlockJ(vRow, vColumn);
      for (uint32_t vP = 0; vP < 2; vP++) {
        const uint32_t vSlot = 2 * vPass + vP;
        if (vSlot >= vRatio) {
          std::memset(vBand + vRow * cXdnaGdnTile + vP * 4096, 0, 4096);
          continue;
        }
        const uint32_t vHead = vKey + vSlot * pIo.vKeyHeads;
        std::memcpy(vBand + vRow * cXdnaGdnTile + vP * 4096,
                    pIo.vStateIn + (size_t(vHead) * cXdnaGdnDim + 8 * vBlock) * cXdnaGdnDim, 4096);
      }
    }
    if (pFlush) {
      pFlush(vBand, cRows * cXdnaGdnTile);
    }
  });
}

bool fXdnaGdnUnpack(const XdnaGdnIo & pIo, uint32_t pFirstRound, uint32_t pRounds, const uint8_t * pC,
                     XdnaWorkers * pWorkers, const XdnaGdnFlush & pFlush) {
  uint32_t vRounds = 0;
  uint32_t vGroups = 0;
  if (!fXdnaGdnShape(pIo, vRounds, vGroups) || pRounds == 0 || pFirstRound + pRounds > vRounds) {
    return true;
  }
  const uint32_t vKeyRounds   = pIo.vKeyHeads / 2;
  const uint32_t vRatio       = pIo.vValueHeads / pIo.vKeyHeads;
  const uint32_t vGroupTokens = cXdnaGdnChunksOut * cXdnaGdnChunk;
  const uint32_t vOutTasks    = pRounds * vGroups * 4;
  const auto     fJoined      = [&](uint32_t pColumn, uint32_t pRound, uint32_t pGroup) {
    return pC + ((uint64_t(pColumn) * pRounds + (pRound - pFirstRound)) * (vGroups + 1) + pGroup) * cRows *
                    cXdnaGdnObjectC;
  };
  std::atomic<bool> vFinite(true);
  fForEach(pWorkers, vOutTasks + cColumns * pRounds, [&](uint32_t pTask) {
    if (pTask >= vOutTasks) {
      // the states of a column and a round
      const uint32_t  vColumn = (pTask - vOutTasks) / pRounds;
      const uint32_t  vRound  = pFirstRound + (pTask - vOutTasks) % pRounds;
      const uint32_t  vPass   = vRound / vKeyRounds;
      const uint8_t * vJoined = fJoined(vColumn, vRound, vGroups);
      if (pFlush) {
        pFlush(vJoined, cRows * cXdnaGdnObjectC);
      }
      for (uint32_t vRow = 0; vRow < cRows; vRow++) {
        const float *  vObject = reinterpret_cast<const float *>(vJoined + vRow * cXdnaGdnObjectC);
        const uint32_t vKey    = fKeyHead(vRound % vKeyRounds, vRow);
        const uint32_t vBlock  = fBlockJ(vRow, vColumn);
        for (uint32_t vP = 0; vP < 2; vP++) {
          const uint32_t vSlot = 2 * vPass + vP;
          if (vSlot >= vRatio) {
            continue;
          }
          const uint32_t vHead = vKey + vSlot * pIo.vKeyHeads;
          std::memcpy(pIo.vStateOut + (size_t(vHead) * cXdnaGdnDim + 8 * vBlock) * cXdnaGdnDim, vObject + vP * 1024,
                      4096);
        }
      }
      return;
    }
    const uint32_t vP     = pTask % 2;
    const uint32_t vPair  = (pTask / 2) % 2;  // rows 2 vPair and 2 vPair + 1 of C: a head of keys
    const uint32_t vGroup = (pTask / 4) % vGroups;
    const uint32_t vRound = pFirstRound + pTask / (4 * vGroups);
    const uint32_t vPass  = vRound / vKeyRounds;
    const uint32_t vSlot  = 2 * vPass + vP;
    if (vSlot >= vRatio) {
      return;
    }
    const uint32_t vHead  = fKeyHead(vRound % vKeyRounds, 2 * vPair) + vSlot * pIo.vKeyHeads;
    // the 16 blocks of 8 values of a row: block j from the object of its column and row, its half of the value head
    const float * aFrom[16];
    for (uint32_t vColumn = 0; vColumn < cColumns; vColumn++) {
      for (uint32_t vHalf = 0; vHalf < 2; vHalf++) {
        const uint32_t vRow    = 2 * vPair + vHalf;
        const float *  vObject = reinterpret_cast<const float *>(fJoined(vColumn, vRound, vGroup) +
                                                                 vRow * cXdnaGdnObjectC) +
                                size_t(vP) * vGroupTokens * 8;
        if (pFlush) {
          pFlush(vObject, size_t(vGroupTokens) * 8 * sizeof(float));
        }
        aFrom[fBlockJ(vRow, vColumn)] = vObject;
      }
    }
    if (pIo.vGated != nullptr) {
      const XdnaGdnGated &      vGated = *pIo.vGated;
      const XdnaGdnGatedPiece * vPiece = nullptr;
      for (uint32_t p = 0; p < vGated.vPieces; p++) {
        if (vHead >= vGated.aPieces[p].vFirstHead && vHead - vGated.aPieces[p].vFirstHead < vGated.aPieces[p].vHeads) {
          vPiece = &vGated.aPieces[p];
          break;
        }
      }
      if (vPiece == nullptr || vPiece->vBufferA == nullptr) {
        vFinite = false;
        return;
      }
      constexpr uint32_t cTileM     = 64;
      constexpr uint64_t cTileBytes = cXdnaGdnTile;
      const uint32_t     vTile      = vHead - vPiece->vFirstHead;
      const uint64_t     vBandBytes = uint64_t(vPiece->vHeads) * cTileBytes;
      const uint32_t     vBands     = vPiece->vPaddedM / cTileM;
      alignas(32) float  aTileO[cTileM * cXdnaGdnDim];
      alignas(32) float  aTile[cTileM * cXdnaGdnDim];
      alignas(32) float  aNormed[cXdnaGdnDim];
      float              aScales[cTileM];
      for (uint32_t vSub = 0; vSub < 2; vSub++) {
        const uint32_t vBand = 2 * vGroup + vSub;
        if (vBand >= vBands) {
          break;
        }
        uint8_t *      vDst  = vPiece->vBufferA + uint64_t(vBand) * vBandBytes + uint64_t(vTile) * cTileBytes;
        const uint64_t vRow0 = uint64_t(vBand) * cTileM;
        if (vRow0 >= pIo.vTokens) {
          std::memset(vDst, 0, cTileBytes);
          if (pFlush) {
            pFlush(vDst, cTileBytes);
          }
          continue;
        }
        const uint32_t vRows = std::min<uint32_t>(cTileM, pIo.vTokens - uint32_t(vRow0));
        for (uint32_t r = 0; r < vRows; r++) {
          const size_t vAt  = size_t(vSub * cTileM + r) * 8;
          float *      vRow = aTileO + size_t(r) * cXdnaGdnDim;
          for (uint32_t b = 0; b < 16; b++) {
            std::memcpy(vRow + 8 * b, aFrom[b] + vAt, 32);
          }
        }
        fXdnaRmsScales(aTileO, vRows, cXdnaGdnDim, cXdnaGdnDim, vGated.vEps, aScales);
        const uint64_t vCol = uint64_t(vHead) * cXdnaGdnDim;
        for (uint32_t r = 0; r < vRows; r++) {
          const float * vX     = aTileO + size_t(r) * cXdnaGdnDim;
          const float   vScale = aScales[r];
          for (uint32_t i = 0; i < cXdnaGdnDim; i++) {
            aNormed[i] = vX[i] * vScale * vGated.vWeights[i];
          }
          const float * vRowZ = vGated.vZ + (vRow0 + r) * vGated.vStrideZ + vCol;
          float *       vOut  = aTile + size_t(r) * cXdnaGdnDim;
          if (vGated.vSigmoid) {
            fXdnaSigmoidMul(vRowZ, aNormed, vOut, cXdnaGdnDim, 0, cXdnaGdnDim);
          } else {
            fXdnaSwiglu(vRowZ, aNormed, vOut, cXdnaGdnDim, 0, cXdnaGdnDim);
          }
        }
        if (fXdnaPackTileA(aTile, vRows, cTileM, cXdnaGdnDim, vDst, vGated.vRounding, vGated.vSteps3) == nullptr) {
          vFinite = false;
        }
        if (pFlush) {
          pFlush(vDst, cTileBytes);
        }
      }
      if (vGroup + 1 == vGroups) {
        for (uint32_t vBand = 2 * vGroups; vBand < vBands; vBand++) {
          uint8_t * vDst = vPiece->vBufferA + uint64_t(vBand) * vBandBytes + uint64_t(vTile) * cTileBytes;
          std::memset(vDst, 0, cTileBytes);
          if (pFlush) {
            pFlush(vDst, cTileBytes);
          }
        }
      }
      return;
    }
    for (uint32_t vT = 0; vT < vGroupTokens; vT++) {
      const size_t vToken = size_t(vGroup) * vGroupTokens + vT;
      if (vToken >= pIo.vTokens) {
        break;
      }
      float * vTo = pIo.vOut + (vToken * pIo.vValueHeads + vHead) * cXdnaGdnDim;
#if defined(__x86_64__)
      if (cHasAvx2 && reinterpret_cast<uintptr_t>(vTo) % 32 == 0) {
        fRowOf16(vTo, aFrom, size_t(vT) * 8);
        continue;
      }
#endif
      for (uint32_t b = 0; b < 16; b++) {
        std::memcpy(vTo + 8 * b, aFrom[b] + size_t(vT) * 8, 32);
      }
    }
#if defined(__x86_64__)
    // what went past the cache is in memory before anyone reads it
    _mm_sfence();
#endif
  });
  return vFinite.load();
}
