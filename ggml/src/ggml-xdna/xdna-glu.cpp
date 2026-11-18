#include "xdna-glu.h"

#include "xdna-bfp16.h"

#include <algorithm>
#include <cmath>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

float fSiluScalar(float pValue) {
  return pValue / (1.0f + expf(-pValue));
}

#if defined(__x86_64__)

enum class XdnaVector { cNone, cAvx2, cAvx512 };

XdnaVector fVector() {
  static const XdnaVector cKind =
      __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512dq") ? XdnaVector::cAvx512 :
      __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")         ? XdnaVector::cAvx2 :
                                                                                 XdnaVector::cNone;
  return cKind;
}

// ggml_v_expf of ggml-cpu/vec.h for AVX-512 (the maximum error is 1.45358 plus 0.5 ulps)
__attribute__((target("avx512f,avx512dq"))) __m512 fExp16(__m512 x) {
  const __m512 r = _mm512_set1_ps(0x1.8p23f);
  const __m512 z = _mm512_fmadd_ps(x, _mm512_set1_ps(0x1.715476p+0f), r);
  const __m512 n = _mm512_sub_ps(z, r);
  const __m512 b =
      _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.7f7d1cp-20f), _mm512_fnmadd_ps(n, _mm512_set1_ps(0x1.62e4p-1f), x));
  const __mmask16 d = _mm512_cmp_ps_mask(_mm512_abs_ps(n), _mm512_set1_ps(192), _CMP_GT_OQ);
  const __m512    u = _mm512_mul_ps(b, b);
  const __m512    j = _mm512_fmadd_ps(
      _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_set1_ps(0x1.0e4020p-7f), b, _mm512_set1_ps(0x1.573e2ep-5f)), u,
                      _mm512_fmadd_ps(_mm512_set1_ps(0x1.555e66p-3f), b, _mm512_set1_ps(0x1.fffdb6p-2f))),
      u, _mm512_fmadd_ps(_mm512_set1_ps(0x1.ffffecp-1f), b, _mm512_set1_ps(1.0F)));
  const __m512 res = _mm512_scalef_ps(j, n);
  if (_mm512_kortestz(d, d)) {
    return res;
  }
  const __m512 zero = _mm512_setzero_ps();
  const __m512 alt  = _mm512_mask_blend_ps(_mm512_cmp_ps_mask(n, zero, _CMP_LE_OQ), _mm512_set1_ps(INFINITY), zero);
  return _mm512_mask_blend_ps(d, res, alt);
}

// up to 16 elements, the others of the vector zeros
__attribute__((target("avx512f,avx512dq"))) void fSwiglu16(const float * pGate, const float * pUp, float * pOut,
                                                            uint32_t pCount) {
  const __mmask16 vMask = __mmask16((1u << pCount) - 1u);
  const __m512    x     = _mm512_maskz_loadu_ps(vMask, pGate);
  const __m512    g     = _mm512_maskz_loadu_ps(vMask, pUp);
  // ggml_v_silu: x / (1 + exp(0 - x))
  const __m512 vSilu =
      _mm512_div_ps(x, _mm512_add_ps(_mm512_set1_ps(1), fExp16(_mm512_sub_ps(_mm512_setzero_ps(), x))));
  _mm512_mask_storeu_ps(pOut, vMask, _mm512_mul_ps(vSilu, g));
}

// 16 bfloat16 as floats: every one the upper half of a float
__attribute__((target("avx512f,avx512dq"))) __m512 fBf16x16(const uint16_t * pFrom) {
  return _mm512_castsi512_ps(
      _mm512_slli_epi32(_mm512_cvtepu16_epi32(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(pFrom))), 16));
}

// an 8 x 8 of bfloat16, two rows of 8 at a time, each half of a vector of 16 to its row
__attribute__((target("avx512f,avx512dq"))) void fSwigluBlock16(const uint16_t * pGate, const uint16_t * pUp,
                                                                 float * pOut, uint64_t pStride) {
  for (uint32_t vPair = 0; vPair < 4; vPair++) {
    const __m512 x     = fBf16x16(pGate + 16 * vPair);
    const __m512 g     = fBf16x16(pUp + 16 * vPair);
    const __m512 vSilu =
        _mm512_div_ps(x, _mm512_add_ps(_mm512_set1_ps(1), fExp16(_mm512_sub_ps(_mm512_setzero_ps(), x))));
    const __m512 vResult = _mm512_mul_ps(vSilu, g);
    _mm256_storeu_ps(pOut + (2 * vPair) * pStride, _mm512_castps512_ps256(vResult));
    _mm256_storeu_ps(pOut + (2 * vPair + 1) * pStride, _mm512_extractf32x8_ps(vResult, 1));
  }
}

// ggml_v_expf of ggml-cpu/vec.h for AVX2 with FMA
__attribute__((target("avx2,fma"))) __m256 fExp8(__m256 x) {
  const __m256  r = _mm256_set1_ps(0x1.8p23f);
  const __m256  z = _mm256_fmadd_ps(x, _mm256_set1_ps(0x1.715476p+0f), r);
  const __m256  n = _mm256_sub_ps(z, r);
  const __m256  b = _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.7f7d1cp-20f),
                                     _mm256_fnmadd_ps(n, _mm256_set1_ps(0x1.62e4p-1f), x));
  const __m256i e = _mm256_slli_epi32(_mm256_castps_si256(z), 23);
  const __m256  k = _mm256_castsi256_ps(_mm256_add_epi32(e, _mm256_castps_si256(_mm256_set1_ps(1))));
  const __m256i c = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(126), _CMP_GT_OQ));
  const __m256 u = _mm256_mul_ps(b, b);
  const __m256 j = _mm256_fmadd_ps(
      _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_set1_ps(0x1.0e4020p-7f), b, _mm256_set1_ps(0x1.573e2ep-5f)), u,
                      _mm256_fmadd_ps(_mm256_set1_ps(0x1.555e66p-3f), b, _mm256_set1_ps(0x1.fffdb6p-2f))),
      u, _mm256_mul_ps(_mm256_set1_ps(0x1.ffffecp-1f), b));
  if (!_mm256_movemask_ps(_mm256_castsi256_ps(c))) {
    return _mm256_fmadd_ps(j, k, k);
  }
  const __m256i g = _mm256_and_si256(_mm256_castps_si256(_mm256_cmp_ps(n, _mm256_setzero_ps(), _CMP_LE_OQ)),
                                     _mm256_set1_epi32(0x82000000u));
  const __m256  s1 = _mm256_castsi256_ps(_mm256_add_epi32(g, _mm256_set1_epi32(0x7f000000u)));
  const __m256  s2 = _mm256_castsi256_ps(_mm256_sub_epi32(e, g));
  const __m256i d  = _mm256_castps_si256(
      _mm256_cmp_ps(_mm256_andnot_ps(_mm256_set1_ps(-0.f), n), _mm256_set1_ps(192), _CMP_GT_OQ));
  return _mm256_or_ps(
      _mm256_and_ps(_mm256_castsi256_ps(d), _mm256_mul_ps(s1, s1)),
      _mm256_andnot_ps(_mm256_castsi256_ps(d),
                       _mm256_or_ps(_mm256_and_ps(_mm256_castsi256_ps(c), _mm256_mul_ps(_mm256_fmadd_ps(s2, j, s2), s1)),
                                    _mm256_andnot_ps(_mm256_castsi256_ps(c), _mm256_fmadd_ps(k, j, k)))));
}

// up to 8 elements, the others of the vector zeros
__attribute__((target("avx2,fma"))) void fSwiglu8(const float * pGate, const float * pUp, float * pOut,
                                                   uint32_t pCount) {
  alignas(32) float aGate[8] = {};
  alignas(32) float aUp[8]   = {};
  alignas(32) float aOut[8];
  for (uint32_t i = 0; i < pCount; i++) {
    aGate[i] = pGate[i];
    aUp[i]   = pUp[i];
  }
  const __m256 x     = _mm256_load_ps(aGate);
  const __m256 vSilu = _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1), fExp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
  _mm256_store_ps(aOut, _mm256_mul_ps(vSilu, _mm256_load_ps(aUp)));
  for (uint32_t i = 0; i < pCount; i++) {
    pOut[i] = aOut[i];
  }
}

// an 8 x 8 of bfloat16, a row of 8 at a time
__attribute__((target("avx2,fma"))) void fSwigluBlock8(const uint16_t * pGate, const uint16_t * pUp, float * pOut,
                                                        uint64_t pStride) {
  for (uint32_t vLine = 0; vLine < 8; vLine++) {
    const __m256 x = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pGate + 8 * vLine))), 16));
    const __m256 g = _mm256_castsi256_ps(_mm256_slli_epi32(
        _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pUp + 8 * vLine))), 16));
    const __m256 vSilu = _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1), fExp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
    _mm256_storeu_ps(pOut + vLine * pStride, _mm256_mul_ps(vSilu, g));
  }
}

// llama.cpp-xdna: ggml_v_silu of ggml-cpu/vec.h, x / (1 + exp(0 - x)), for AVX-512 and for AVX2
__attribute__((target("avx512f,avx512dq"))) __m512 fSiluOf16(__m512 x) {
  return _mm512_div_ps(x, _mm512_add_ps(_mm512_set1_ps(1), fExp16(_mm512_sub_ps(_mm512_setzero_ps(), x))));
}

__attribute__((target("avx2,fma"))) __m256 fSiluOf8(__m256 x) {
  return _mm256_div_ps(x, _mm256_add_ps(_mm256_set1_ps(1), fExp8(_mm256_sub_ps(_mm256_setzero_ps(), x))));
}

// llama.cpp-xdna: a token of the short convolution (fXdnaConvToken) by vectors of 16, the last one masked: every value
// goes through the lane of a vector as on the CPU backend, so the same float whichever values share its vector
__attribute__((target("avx512f,avx512dq"))) void fConvToken16(const float * const aTaps[4], const float * pWeights,
                                                               uint64_t pWeightStride, float * pOut, uint32_t pCount,
                                                               bool pSilu) {
  for (uint32_t i = 0; i < pCount; i += 16) {
    const __mmask16 vMask = pCount - i >= 16 ? __mmask16(0xffff) : __mmask16((1u << (pCount - i)) - 1u);
    __m512          vAcc  = _mm512_setzero_ps();
    vAcc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(vMask, aTaps[0] + i),
                           _mm512_maskz_loadu_ps(vMask, pWeights + 0 * pWeightStride + i), vAcc);
    vAcc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(vMask, aTaps[1] + i),
                           _mm512_maskz_loadu_ps(vMask, pWeights + 1 * pWeightStride + i), vAcc);
    vAcc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(vMask, aTaps[2] + i),
                           _mm512_maskz_loadu_ps(vMask, pWeights + 2 * pWeightStride + i), vAcc);
    vAcc = _mm512_fmadd_ps(_mm512_maskz_loadu_ps(vMask, aTaps[3] + i),
                           _mm512_maskz_loadu_ps(vMask, pWeights + 3 * pWeightStride + i), vAcc);
    _mm512_mask_storeu_ps(pOut + i, vMask, pSilu ? fSiluOf16(vAcc) : vAcc);
  }
}

// llama.cpp-xdna: the same with the 4 taps as bfloat16 (rows of the C of the overlay of the prompts), converted in the
// registers, pCount a multiple of 16; with pStream the results go past the cache (pOut aligned to 64 bytes), as C is
// written when it is read
__attribute__((target("avx512f,avx512dq"))) void fConvTokenBf16x16(const uint16_t * const aRows[4],
                                                                    const float * pWeights, uint64_t pWeightStride,
                                                                    float * pOut, uint32_t pCount, bool pStream) {
  for (uint32_t i = 0; i < pCount; i += 16) {
    __m512 vAcc = _mm512_setzero_ps();
    vAcc        = _mm512_fmadd_ps(fBf16x16(aRows[0] + i), _mm512_loadu_ps(pWeights + i), vAcc);
    vAcc        = _mm512_fmadd_ps(fBf16x16(aRows[1] + i), _mm512_loadu_ps(pWeights + pWeightStride + i), vAcc);
    vAcc        = _mm512_fmadd_ps(fBf16x16(aRows[2] + i), _mm512_loadu_ps(pWeights + 2 * pWeightStride + i), vAcc);
    vAcc        = _mm512_fmadd_ps(fBf16x16(aRows[3] + i), _mm512_loadu_ps(pWeights + 3 * pWeightStride + i), vAcc);
    if (pStream) {
      _mm512_stream_ps(pOut + i, fSiluOf16(vAcc));
    } else {
      _mm512_storeu_ps(pOut + i, fSiluOf16(vAcc));
    }
  }
}

// llama.cpp-xdna: fConvTokenBf16x16 for the rows pFirstRow to pEndRow (3 or more) of a strip of C, pWidth bfloat16 a
// row, the outputs of row t at pOut + t * pOutStride
__attribute__((target("avx512f,avx512dq"))) void fConvRowsBf16x16(const uint16_t * pStrip, uint64_t pWidth,
                                                                   uint64_t pFirstRow, uint64_t pEndRow,
                                                                   const float * pWeights, uint64_t pWeightStride,
                                                                   float * pOut, uint64_t pOutStride, uint32_t pCount,
                                                                   bool pStream) {
  for (uint64_t vRow = pFirstRow; vRow < pEndRow; vRow++) {
    const uint16_t * aRows[4] = {pStrip + (vRow - 3) * pWidth, pStrip + (vRow - 2) * pWidth,
                                 pStrip + (vRow - 1) * pWidth, pStrip + vRow * pWidth};
    fConvTokenBf16x16(aRows, pWeights, pWeightStride, pOut + vRow * pOutStride, pCount, pStream);
  }
}

__attribute__((target("avx512f,avx512dq"))) void fSilu16(const float * pIn, float * pOut, uint32_t pCount) {
  for (uint32_t i = 0; i < pCount; i += 16) {
    const __mmask16 vMask = pCount - i >= 16 ? __mmask16(0xffff) : __mmask16((1u << (pCount - i)) - 1u);
    _mm512_mask_storeu_ps(pOut + i, vMask, fSiluOf16(_mm512_maskz_loadu_ps(vMask, pIn + i)));
  }
}

// the same by vectors of 8 for AVX2, the rest through a vector of its own with zeros in the other lanes
__attribute__((target("avx2,fma"))) __m256 fConvVector8(const float * const aTaps[4], const float * pWeights,
                                                        uint64_t pWeightStride, uint32_t pAt, bool pSilu) {
  __m256 vAcc = _mm256_setzero_ps();
  vAcc = _mm256_fmadd_ps(_mm256_loadu_ps(aTaps[0] + pAt), _mm256_loadu_ps(pWeights + 0 * pWeightStride + pAt), vAcc);
  vAcc = _mm256_fmadd_ps(_mm256_loadu_ps(aTaps[1] + pAt), _mm256_loadu_ps(pWeights + 1 * pWeightStride + pAt), vAcc);
  vAcc = _mm256_fmadd_ps(_mm256_loadu_ps(aTaps[2] + pAt), _mm256_loadu_ps(pWeights + 2 * pWeightStride + pAt), vAcc);
  vAcc = _mm256_fmadd_ps(_mm256_loadu_ps(aTaps[3] + pAt), _mm256_loadu_ps(pWeights + 3 * pWeightStride + pAt), vAcc);
  return pSilu ? fSiluOf8(vAcc) : vAcc;
}

__attribute__((target("avx2,fma"))) void fConvToken8(const float * const aTaps[4], const float * pWeights,
                                                     uint64_t pWeightStride, float * pOut, uint32_t pCount,
                                                     bool pSilu) {
  uint32_t i = 0;
  for (; i + 8 <= pCount; i += 8) {
    _mm256_storeu_ps(pOut + i, fConvVector8(aTaps, pWeights, pWeightStride, i, pSilu));
  }
  if (i < pCount) {
    alignas(32) float aRest[4][8]    = {};
    alignas(32) float aWeights[4][8] = {};
    alignas(32) float aOut[8];
    for (uint32_t k = 0; k < 4; k++) {
      for (uint32_t j = 0; i + j < pCount; j++) {
        aRest[k][j]    = aTaps[k][i + j];
        aWeights[k][j] = pWeights[k * pWeightStride + i + j];
      }
    }
    const float * aRestTaps[4] = {aRest[0], aRest[1], aRest[2], aRest[3]};
    _mm256_store_ps(aOut, fConvVector8(aRestTaps, &aWeights[0][0], 8, 0, pSilu));
    for (uint32_t j = 0; i + j < pCount; j++) {
      pOut[i + j] = aOut[j];
    }
  }
}

// llama.cpp-xdna: fConvTokenBf16x16 by vectors of 8 for AVX2 (pOut aligned to 32 bytes with pStream)
__attribute__((target("avx2,fma"))) __m256 fBf16x8(const uint16_t * pFrom) {
  return _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom))), 16));
}

__attribute__((target("avx2,fma"))) void fConvTokenBf16x8(const uint16_t * const aRows[4], const float * pWeights,
                                                          uint64_t pWeightStride, float * pOut, uint32_t pCount,
                                                          bool pStream) {
  for (uint32_t i = 0; i < pCount; i += 8) {
    __m256 vAcc = _mm256_setzero_ps();
    vAcc        = _mm256_fmadd_ps(fBf16x8(aRows[0] + i), _mm256_loadu_ps(pWeights + i), vAcc);
    vAcc        = _mm256_fmadd_ps(fBf16x8(aRows[1] + i), _mm256_loadu_ps(pWeights + pWeightStride + i), vAcc);
    vAcc        = _mm256_fmadd_ps(fBf16x8(aRows[2] + i), _mm256_loadu_ps(pWeights + 2 * pWeightStride + i), vAcc);
    vAcc        = _mm256_fmadd_ps(fBf16x8(aRows[3] + i), _mm256_loadu_ps(pWeights + 3 * pWeightStride + i), vAcc);
    if (pStream) {
      _mm256_stream_ps(pOut + i, fSiluOf8(vAcc));
    } else {
      _mm256_storeu_ps(pOut + i, fSiluOf8(vAcc));
    }
  }
}

__attribute__((target("avx2,fma"))) void fConvRowsBf16x8(const uint16_t * pStrip, uint64_t pWidth, uint64_t pFirstRow,
                                                         uint64_t pEndRow, const float * pWeights,
                                                         uint64_t pWeightStride, float * pOut, uint64_t pOutStride,
                                                         uint32_t pCount, bool pStream) {
  for (uint64_t vRow = pFirstRow; vRow < pEndRow; vRow++) {
    const uint16_t * aRows[4] = {pStrip + (vRow - 3) * pWidth, pStrip + (vRow - 2) * pWidth,
                                 pStrip + (vRow - 1) * pWidth, pStrip + vRow * pWidth};
    fConvTokenBf16x8(aRows, pWeights, pWeightStride, pOut + vRow * pOutStride, pCount, pStream);
  }
}

__attribute__((target("avx2,fma"))) void fSilu8(const float * pIn, float * pOut, uint32_t pCount) {
  uint32_t i = 0;
  for (; i + 8 <= pCount; i += 8) {
    _mm256_storeu_ps(pOut + i, fSiluOf8(_mm256_loadu_ps(pIn + i)));
  }
  if (i < pCount) {
    alignas(32) float aRest[8] = {};
    alignas(32) float aOut[8];
    for (uint32_t j = 0; i + j < pCount; j++) {
      aRest[j] = pIn[i + j];
    }
    _mm256_store_ps(aOut, fSiluOf8(_mm256_load_ps(aRest)));
    for (uint32_t j = 0; i + j < pCount; j++) {
      pOut[i + j] = aOut[j];
    }
  }
}

#endif

float fBf16ToFloat(uint16_t pValue) {
  const uint32_t vBits  = uint32_t(pValue) << 16;
  float          vValue = 0;
  __builtin_memcpy(&vValue, &vBits, sizeof(vValue));
  return vValue;
}

// llama.cpp-xdna: the scalar arithmetic of the attention of a batch, for what no vector takes: a score plus its mask
float fScoreScalar(const uint16_t * pScores, const uint16_t * pMask, uint32_t pIndex) {
  const float vScore = fBf16ToFloat(pScores[pIndex]);
  return pMask != nullptr ? vScore + fXdnaHalfToFloat(pMask[pIndex]) : vScore;
}

#if defined(__x86_64__)

bool fHasF16c() {
  static const bool cHas = __builtin_cpu_supports("f16c");
  return cHas;
}

// llama.cpp-xdna: 16 scores as floats, plus their mask when there is one
__attribute__((target("avx512f,avx512dq"))) __m512 fScores16(const uint16_t * pScores, const uint16_t * pMask) {
  const __m512 vScores = fBf16x16(pScores);
  if (pMask == nullptr) {
    return vScores;
  }
  return _mm512_add_ps(vScores, _mm512_cvtph_ps(_mm256_loadu_si256(reinterpret_cast<const __m256i *>(pMask))));
}

__attribute__((target("avx512f,avx512dq"))) float fScoresMax16(const uint16_t * pScores, const uint16_t * pMask,
                                                                uint32_t pCount, uint32_t & pDone) {
  __m512   vMax = _mm512_set1_ps(-INFINITY);
  uint32_t i    = 0;
  for (; i + 16 <= pCount; i += 16) {
    vMax = _mm512_max_ps(vMax, fScores16(pScores + i, pMask != nullptr ? pMask + i : nullptr));
  }
  pDone = i;
  return _mm512_reduce_max_ps(vMax);
}

__attribute__((target("avx512f,avx512dq"))) float fScoresExp16(const uint16_t * pScores, const uint16_t * pMask,
                                                                float pShift, float * pOut, uint32_t pCount,
                                                                uint32_t & pDone) {
  const __m512 vShift = _mm512_set1_ps(pShift);
  __m512       vSum   = _mm512_setzero_ps();
  uint32_t     i      = 0;
  for (; i + 16 <= pCount; i += 16) {
    // minus infinity, a masked key, gives 0 (see fExp16)
    const __m512 vExp = fExp16(_mm512_sub_ps(fScores16(pScores + i, pMask != nullptr ? pMask + i : nullptr), vShift));
    _mm512_storeu_ps(pOut + i, vExp);
    vSum = _mm512_add_ps(vSum, vExp);
  }
  pDone = i;
  return _mm512_reduce_add_ps(vSum);
}

__attribute__((target("avx2,fma,f16c"))) __m256 fScores8(const uint16_t * pScores, const uint16_t * pMask) {
  const __m256 vScores = _mm256_castsi256_ps(
      _mm256_slli_epi32(_mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pScores))), 16));
  if (pMask == nullptr) {
    return vScores;
  }
  return _mm256_add_ps(vScores, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pMask))));
}

__attribute__((target("avx2,fma,f16c"))) float fScoresMax8(const uint16_t * pScores, const uint16_t * pMask,
                                                            uint32_t pCount, uint32_t & pDone) {
  __m256   vMax = _mm256_set1_ps(-INFINITY);
  uint32_t i    = 0;
  for (; i + 8 <= pCount; i += 8) {
    vMax = _mm256_max_ps(vMax, fScores8(pScores + i, pMask != nullptr ? pMask + i : nullptr));
  }
  pDone = i;
  alignas(32) float aMax[8];
  _mm256_store_ps(aMax, vMax);
  float vResult = aMax[0];
  for (uint32_t j = 1; j < 8; j++) {
    vResult = aMax[j] > vResult ? aMax[j] : vResult;
  }
  return vResult;
}

__attribute__((target("avx2,fma,f16c"))) float fScoresExp8(const uint16_t * pScores, const uint16_t * pMask,
                                                            float pShift, float * pOut, uint32_t pCount,
                                                            uint32_t & pDone) {
  const __m256 vShift = _mm256_set1_ps(pShift);
  __m256       vSum   = _mm256_setzero_ps();
  uint32_t     i      = 0;
  for (; i + 8 <= pCount; i += 8) {
    // minus infinity, a masked key, gives 0 (see fExp8)
    const __m256 vExp = fExp8(_mm256_sub_ps(fScores8(pScores + i, pMask != nullptr ? pMask + i : nullptr), vShift));
    _mm256_storeu_ps(pOut + i, vExp);
    vSum = _mm256_add_ps(vSum, vExp);
  }
  pDone = i;
  alignas(32) float aSum[8];
  _mm256_store_ps(aSum, vSum);
  float vResult = 0;
  for (uint32_t j = 0; j < 8; j++) {
    vResult += aSum[j];
  }
  return vResult;
}

__attribute__((target("avx2,f16c"))) void fHalves8(const uint16_t * pHalves, float * pFloats, uint32_t pCount,
                                                    uint32_t & pDone) {
  uint32_t i = 0;
  for (; i + 8 <= pCount; i += 8) {
    _mm256_storeu_ps(pFloats + i, _mm256_cvtph_ps(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pHalves + i))));
  }
  pDone = i;
}

#endif

// the sum of the squares of a row, as the scalar loop of RMS_NORM of the CPU backend adds them
double fSquaresScalar(const float * pRow, uint64_t pLength) {
  double vSum = 0.0;
  for (uint64_t i = 0; i < pLength; i++) {
    vSum += double(pRow[i] * pRow[i]);
  }
  return vSum;
}

#if defined(__x86_64__)
// the same for 8 rows at a time, one in every lane, each in the same order (ggml_rms_norm_sums_8 of ggml-cpu/ops.cpp)
__attribute__((target("avx"))) void fSquares8(const float * const pRows[8], uint64_t pLength, double pSums[8]) {
  __m256d  vSums03 = _mm256_setzero_pd();
  __m256d  vSums47 = _mm256_setzero_pd();
  uint64_t i       = 0;
  for (; i + 8 <= pLength; i += 8) {
    __m256 aSquares[8];
    for (int r = 0; r < 8; r++) {
      const __m256 vValues = _mm256_loadu_ps(pRows[r] + i);
      aSquares[r]          = _mm256_mul_ps(vValues, vValues);
    }
    // u[j], j < 4: rows 0 to 3 of the column i + j (low half) and of the column i + j + 4 (high half); u[4 + j],
    // rows 4 to 7
    const __m256 t0   = _mm256_unpacklo_ps(aSquares[0], aSquares[1]);
    const __m256 t1   = _mm256_unpackhi_ps(aSquares[0], aSquares[1]);
    const __m256 t2   = _mm256_unpacklo_ps(aSquares[2], aSquares[3]);
    const __m256 t3   = _mm256_unpackhi_ps(aSquares[2], aSquares[3]);
    const __m256 t4   = _mm256_unpacklo_ps(aSquares[4], aSquares[5]);
    const __m256 t5   = _mm256_unpackhi_ps(aSquares[4], aSquares[5]);
    const __m256 t6   = _mm256_unpacklo_ps(aSquares[6], aSquares[7]);
    const __m256 t7   = _mm256_unpackhi_ps(aSquares[6], aSquares[7]);
    const __m256 u[8] = {
      _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(1, 0, 1, 0)), _mm256_shuffle_ps(t0, t2, _MM_SHUFFLE(3, 2, 3, 2)),
      _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(1, 0, 1, 0)), _mm256_shuffle_ps(t1, t3, _MM_SHUFFLE(3, 2, 3, 2)),
      _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(1, 0, 1, 0)), _mm256_shuffle_ps(t4, t6, _MM_SHUFFLE(3, 2, 3, 2)),
      _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(1, 0, 1, 0)), _mm256_shuffle_ps(t5, t7, _MM_SHUFFLE(3, 2, 3, 2)),
    };
    for (int j = 0; j < 4; j++) {
      vSums03 = _mm256_add_pd(vSums03, _mm256_cvtps_pd(_mm256_castps256_ps128(u[j])));
      vSums47 = _mm256_add_pd(vSums47, _mm256_cvtps_pd(_mm256_castps256_ps128(u[4 + j])));
    }
    for (int j = 0; j < 4; j++) {
      vSums03 = _mm256_add_pd(vSums03, _mm256_cvtps_pd(_mm256_extractf128_ps(u[j], 1)));
      vSums47 = _mm256_add_pd(vSums47, _mm256_cvtps_pd(_mm256_extractf128_ps(u[4 + j], 1)));
    }
  }
  _mm256_storeu_pd(pSums + 0, vSums03);
  _mm256_storeu_pd(pSums + 4, vSums47);
  for (; i < pLength; i++) {
    for (int r = 0; r < 8; r++) {
      pSums[r] += double(pRows[r][i] * pRows[r][i]);
    }
  }
}
#endif

float fRmsScaleOf(double pSum, uint64_t pLength, float pEps) {
  const float vMean = float(pSum / double(pLength));
  return 1.0f / sqrtf(vMean + pEps);
}

}  // namespace

void fXdnaRmsScales(const float * pRows, uint64_t pCount, uint64_t pLength, uint64_t pStride, float pEps,
                     float * pScales) {
  uint64_t vRow = 0;
#if defined(__x86_64__)
  if (pLength >= 8 && __builtin_cpu_supports("avx")) {
    for (; vRow + 8 <= pCount; vRow += 8) {
      const float * aRows[8];
      for (int r = 0; r < 8; r++) {
        aRows[r] = pRows + (vRow + uint64_t(r)) * pStride;
      }
      double aSums[8];
      fSquares8(aRows, pLength, aSums);
      for (int r = 0; r < 8; r++) {
        pScales[vRow + uint64_t(r)] = fRmsScaleOf(aSums[r], pLength, pEps);
      }
    }
  }
#endif
  for (; vRow < pCount; vRow++) {
    pScales[vRow] = fRmsScaleOf(fSquaresScalar(pRows + vRow * pStride, pLength), pLength, pEps);
  }
}

bool fXdnaSwigluBlockFits(uint64_t pFirst, uint64_t pRowLength) {
#if defined(__x86_64__)
  const XdnaVector vKind  = fVector();
  const uint64_t    vWidth = vKind == XdnaVector::cAvx512 ? 16 : vKind == XdnaVector::cAvx2 ? 8 : 0;
  return vWidth > 0 && pFirst + 8 <= pRowLength - pRowLength % vWidth;
#else
  (void) pFirst;
  (void) pRowLength;
  return false;
#endif
}

void fXdnaSwigluBlock(const uint16_t * pGate, const uint16_t * pUp, float * pOut, uint64_t pStride) {
#if defined(__x86_64__)
  const XdnaVector vKind = fVector();
  if (vKind == XdnaVector::cAvx512) {
    fSwigluBlock16(pGate, pUp, pOut, pStride);
    return;
  }
  if (vKind == XdnaVector::cAvx2) {
    fSwigluBlock8(pGate, pUp, pOut, pStride);
    return;
  }
#endif
  // never reached through fXdnaSwigluBlockFits: element by element
  for (uint32_t vLine = 0; vLine < 8; vLine++) {
    for (uint32_t i = 0; i < 8; i++) {
      const float vGate        = fBf16ToFloat(pGate[8 * vLine + i]);
      pOut[vLine * pStride + i] = fSiluScalar(vGate) * fBf16ToFloat(pUp[8 * vLine + i]);
    }
  }
}

void fXdnaSwiglu(const float * pGate, const float * pUp, float * pOut, uint32_t pCount, uint64_t pFirst,
                  uint64_t pRowLength) {
  uint32_t vDone = 0;
#if defined(__x86_64__)
  // the elements before the last vector of the row that the CPU backend takes whole go by vectors, as there
  const XdnaVector vKind  = fVector();
  const uint64_t    vWidth = vKind == XdnaVector::cAvx512 ? 16 : vKind == XdnaVector::cAvx2 ? 8 : 0;
  if (vWidth > 0) {
    const uint64_t vVectorEnd = pRowLength - pRowLength % vWidth;
    while (vDone < pCount && pFirst + vDone < vVectorEnd) {
      const uint32_t vCount = uint32_t(std::min<uint64_t>({uint64_t(pCount - vDone), vWidth, vVectorEnd - pFirst - vDone}));
      if (vKind == XdnaVector::cAvx512) {
        fSwiglu16(pGate + vDone, pUp + vDone, pOut + vDone, vCount);
      } else {
        fSwiglu8(pGate + vDone, pUp + vDone, pOut + vDone, vCount);
      }
      vDone += vCount;
    }
  }
#endif
  for (; vDone < pCount; vDone++) {
    pOut[vDone] = fSiluScalar(pGate[vDone]) * pUp[vDone];
  }
}

float fXdnaScoresMax(const uint16_t * pScores, const uint16_t * pMask, uint32_t pCount) {
  float    vMax  = -INFINITY;
  uint32_t vDone = 0;
#if defined(__x86_64__)
  const XdnaVector vKind = fVector();
  if (vKind == XdnaVector::cAvx512) {
    vMax = fScoresMax16(pScores, pMask, pCount, vDone);
  } else if (vKind == XdnaVector::cAvx2 && fHasF16c()) {
    vMax = fScoresMax8(pScores, pMask, pCount, vDone);
  }
#endif
  for (; vDone < pCount; vDone++) {
    const float vValue = fScoreScalar(pScores, pMask, vDone);
    vMax               = vValue > vMax ? vValue : vMax;
  }
  return vMax;
}

float fXdnaScoresExp(const uint16_t * pScores, const uint16_t * pMask, float pShift, float * pOut, uint32_t pCount) {
  float    vSum  = 0;
  uint32_t vDone = 0;
#if defined(__x86_64__)
  const XdnaVector vKind = fVector();
  if (vKind == XdnaVector::cAvx512) {
    vSum = fScoresExp16(pScores, pMask, pShift, pOut, pCount, vDone);
  } else if (vKind == XdnaVector::cAvx2 && fHasF16c()) {
    vSum = fScoresExp8(pScores, pMask, pShift, pOut, pCount, vDone);
  }
#endif
  for (; vDone < pCount; vDone++) {
    pOut[vDone] = expf(fScoreScalar(pScores, pMask, vDone) - pShift);
    vSum += pOut[vDone];
  }
  return vSum;
}

void fXdnaHalvesToFloats(const uint16_t * pHalves, float * pFloats, uint32_t pCount) {
  uint32_t vDone = 0;
#if defined(__x86_64__)
  if (fHasF16c() && __builtin_cpu_supports("avx2")) {
    fHalves8(pHalves, pFloats, pCount, vDone);
  }
#endif
  for (; vDone < pCount; vDone++) {
    pFloats[vDone] = fXdnaHalfToFloat(pHalves[vDone]);
  }
}

void fXdnaConvToken(const float * const aTaps[4], const float * pWeights, uint64_t pWeightStride, float * pOut,
                     uint32_t pCount, bool pSilu) {
#if defined(__x86_64__)
  const XdnaVector vKind = fVector();
  if (vKind == XdnaVector::cAvx512) {
    fConvToken16(aTaps, pWeights, pWeightStride, pOut, pCount, pSilu);
    return;
  }
  if (vKind == XdnaVector::cAvx2) {
    fConvToken8(aTaps, pWeights, pWeightStride, pOut, pCount, pSilu);
    return;
  }
#endif
  for (uint32_t i = 0; i < pCount; i++) {
    float vAcc = 0.0f;
    for (uint32_t k = 0; k < 4; k++) {
      vAcc = std::fma(aTaps[k][i], pWeights[k * pWeightStride + i], vAcc);
    }
    pOut[i] = pSilu ? fSiluScalar(vAcc) : vAcc;
  }
}

void fXdnaSilu(const float * pIn, float * pOut, uint32_t pCount) {
#if defined(__x86_64__)
  const XdnaVector vKind = fVector();
  if (vKind == XdnaVector::cAvx512) {
    fSilu16(pIn, pOut, pCount);
    return;
  }
  if (vKind == XdnaVector::cAvx2) {
    fSilu8(pIn, pOut, pCount);
    return;
  }
#endif
  for (uint32_t i = 0; i < pCount; i++) {
    pOut[i] = fSiluScalar(pIn[i]);
  }
}

bool fXdnaConvRowsBf16(const uint16_t * pStrip, uint64_t pWidth, uint64_t pFirstRow, uint64_t pEndRow,
                        const float * pWeights, uint64_t pWeightStride, float * pOut, uint64_t pOutStride,
                        uint32_t pCount, bool pStream) {
  if (pFirstRow < 3) {
    return false;
  }
#if defined(__x86_64__)
  const XdnaVector vKind   = fVector();
  const uintptr_t   vOut    = reinterpret_cast<uintptr_t>(pOut + pFirstRow * pOutStride);
  const uint64_t    vStride = pOutStride * sizeof(float);
  if (vKind == XdnaVector::cAvx512 && pCount % 16 == 0) {
    fConvRowsBf16x16(pStrip, pWidth, pFirstRow, pEndRow, pWeights, pWeightStride, pOut, pOutStride, pCount,
                     pStream && vOut % 64 == 0 && vStride % 64 == 0);
    return true;
  }
  if (vKind == XdnaVector::cAvx2 && pCount % 8 == 0) {
    fConvRowsBf16x8(pStrip, pWidth, pFirstRow, pEndRow, pWeights, pWeightStride, pOut, pOutStride, pCount,
                    pStream && vOut % 32 == 0 && vStride % 32 == 0);
    return true;
  }
#else
  (void) pStrip;
  (void) pWidth;
  (void) pEndRow;
  (void) pWeights;
  (void) pWeightStride;
  (void) pOut;
  (void) pOutStride;
  (void) pCount;
  (void) pStream;
#endif
  return false;
}
