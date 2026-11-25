#include "xdna-bfp16.h"

#include <cmath>
#include <cstring>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace {

const uint32_t cExponentMask  = 0xff;       // of a 32-bit float, after shifting it down
const uint32_t cExponentShift = 23;
const uint32_t cFractionMask  = 0x7fffff;
const uint32_t cHiddenBit     = 0x800000;   // the leading 1 of a normal float
const int      cMantissaShift = 17;         // from the 24 bits of a float to 7 bits and a sign
const int      cExponentBias  = 133;        // a mantissa of 1 with this exponent is 1.0

// What a mantissa of 1 is worth with every exponent a block can have. A mantissa times it is exact: both are
// powers of two times a small integer, and the product is the value std::ldexp gives, also when it is denormal.
struct XdnaBfp16Scales {
  float aScales[256];
  XdnaBfp16Scales() {
    for (int i = 0; i < 256; i++) {
      aScales[i] = std::ldexp(1.0f, i - cExponentBias);
    }
  }
};

const XdnaBfp16Scales cScales;

}  // namespace

bool fXdnaBfp16Encode(const float * pValues, uint8_t * pBlock, XdnaRounding pRounding) {
  uint32_t aBits[cXdnaBfp16Values];
  std::memcpy(aBits, pValues, sizeof(aBits));

  uint32_t vLargest = 0;
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    const uint32_t vExponent = (aBits[i] >> cExponentShift) & cExponentMask;
    if (vExponent == cExponentMask) {
      return false;  // infinity or not a number
    }
    vLargest = vExponent > vLargest ? vExponent : vLargest;
  }

  pBlock[0] = uint8_t(vLargest);
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    const uint32_t vExponent = (aBits[i] >> cExponentShift) & cExponentMask;
    const bool     vNegative = (aBits[i] >> 31) != 0;
    const uint32_t vFraction = (aBits[i] & cFractionMask) | (vExponent != 0 ? cHiddenBit : 0);
    const int      vShift    = int(vLargest - vExponent);  // how far below the largest value of the block it is
    int32_t        vMantissa = 0;
    if (pRounding == XdnaRounding::cNearest) {
      // the quotient of a 24-bit integer by a power of two is exact in a double, so this rounds only once
      const double vSigned = vNegative ? -double(vFraction) : double(vFraction);
      const int    vDown   = cMantissaShift + vShift < 62 ? cMantissaShift + vShift : 62;
      vMantissa = int32_t(std::nearbyint(std::ldexp(vSigned, -vDown)));
      vMantissa = vMantissa > 127 ? 127 : (vMantissa < -128 ? -128 : vMantissa);
    } else if (vShift >= 32) {
      vMantissa = vNegative ? -1 : 0;
    } else {
      // two's complement in 32 bits, its byte at the mantissa shift, and then down by the distance, keeping the
      // sign: both shifts round towards minus infinity
      const uint32_t vComplement = vNegative ? uint32_t(0) - vFraction : vFraction;
      vMantissa = int32_t(int8_t(uint8_t(vComplement >> cMantissaShift)));
      vMantissa >>= vShift < 31 ? vShift : 31;
    }
    pBlock[1 + i] = uint8_t(int8_t(vMantissa));
  }
  return true;
}

namespace {

// One block, rounded to the nearest, with integers only. A mantissa is the 24 bits of the value (its fraction
// and the leading 1 of a normal float) over a power of two, rounded to the nearest and, half way, to the even
// one: the quotient, plus 1 when what is left over is more than half the divisor, or exactly half with an odd
// quotient. Past a divisor of 2 to the 25 the result is always 0, so the shift is held at 31. The loops are over
// the 8 values of the block, for the compiler to do them at once.
inline bool fEncodeNearestBlock(const float * pValues, uint8_t * pBlock) {
  uint32_t aBits[cXdnaBfp16Values];
  uint32_t aExponents[cXdnaBfp16Values];
  std::memcpy(aBits, pValues, sizeof(aBits));

  uint32_t vLargest = 0;
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    aExponents[i] = (aBits[i] >> cExponentShift) & cExponentMask;
    vLargest      = aExponents[i] > vLargest ? aExponents[i] : vLargest;
  }
  if (vLargest == cExponentMask) {
    return false;  // infinity or not a number
  }

  int32_t aMantissas[cXdnaBfp16Values];
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    const uint32_t vFraction = (aBits[i] & cFractionMask) | (aExponents[i] != 0 ? cHiddenBit : 0);
    uint32_t       vDown     = uint32_t(cMantissaShift) + vLargest - aExponents[i];
    vDown = vDown < 31 ? vDown : 31;
    const uint32_t vQuotient = vFraction >> vDown;
    const uint32_t vRest     = vFraction & ((uint32_t(1) << vDown) - 1);
    const uint32_t vHalf     = uint32_t(1) << (vDown - 1);
    const uint32_t vUp       = (vRest > vHalf ? 1u : 0u) | ((vRest == vHalf ? 1u : 0u) & vQuotient);
    const int32_t  vRounded  = int32_t(vQuotient + vUp);
    const int32_t  vSigned   = (aBits[i] >> 31) != 0 ? -vRounded : vRounded;
    aMantissas[i] = vSigned > 127 ? 127 : vSigned;
  }
  pBlock[0] = uint8_t(vLargest);
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    pBlock[1 + i] = uint8_t(int8_t(aMantissas[i]));
  }
  return true;
}

}  // namespace

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#define XDNA_BFP16_AVX2 1

namespace {

// llama.cpp-xdna: the largest exponents from which a block goes by the product of floats (fEncodeNearestAvx2), and the
// biased exponent of 2 to the power of (133 - largest) is this minus the largest
const uint32_t cFastLargest = 8;
const uint32_t cScaleBias   = 133 + 127;

// The same, with the 8 values of a block in one 256-bit register. With pScales, the values of block i are first
// (value * pScales[i]) * pWeights[j], two products of floats in that order as the scalar code does them (an RMS norm and
// its weights, llama.cpp-xdna), so the blocks are the same bit for bit; before, the products went one value at a
// time into a temporary (the library is built for the baseline of x86-64).
__attribute__((target("avx2")))
bool fEncodeNearestAvx2(const float * pValues, size_t pStride, uint32_t pCount, uint8_t * pBlocks,
                        const float * pScales = nullptr, const float * pWeights = nullptr) {
  const __m256i cOne      = _mm256_set1_epi32(1);
  const __m256i cExponent = _mm256_set1_epi32(int(cExponentMask));
  const __m256i cFraction = _mm256_set1_epi32(int(cFractionMask));
  const __m256i cHidden   = _mm256_set1_epi32(int(cHiddenBit));
  const __m256i cShift    = _mm256_set1_epi32(cMantissaShift);
  const __m256i cShiftMax = _mm256_set1_epi32(31);
  const __m256i cTop      = _mm256_set1_epi32(127);
  const __m256  vWeights  = pScales != nullptr ? _mm256_loadu_ps(pWeights) : _mm256_setzero_ps();
  bool          vFinite   = true;
  for (uint32_t i = 0; i < pCount; i++) {
    __m256i vBits = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(pValues + i * pStride));
    if (pScales != nullptr) {
      const __m256 vScaled = _mm256_mul_ps(_mm256_castsi256_ps(vBits), _mm256_set1_ps(pScales[i]));
      vBits                = _mm256_castps_si256(_mm256_mul_ps(vScaled, vWeights));
    }
    const __m256i vExponents = _mm256_and_si256(_mm256_srli_epi32(vBits, int(cExponentShift)), cExponent);
    // the largest exponent of the 8
    __m128i vMax = _mm_max_epu32(_mm256_castsi256_si128(vExponents), _mm256_extracti128_si256(vExponents, 1));
    vMax = _mm_max_epu32(vMax, _mm_shuffle_epi32(vMax, _MM_SHUFFLE(1, 0, 3, 2)));
    vMax = _mm_max_epu32(vMax, _mm_shuffle_epi32(vMax, _MM_SHUFFLE(2, 3, 0, 1)));
    const uint32_t vLargest = uint32_t(_mm_cvtsi128_si32(vMax));
    if (vLargest == cExponentMask) {
      vFinite = false;  // infinity or not a number
      continue;
    }
    __m256i vMantissas;
    if (vLargest >= cFastLargest) {
      // llama.cpp-xdna: a mantissa is the value times 2 to the power of (133 - the largest exponent), rounded to the
      // nearest and, half way, to the even one: the product by a power of two is exact (it is under 128 in magnitude,
      // and what falls under the normal floats is far under a half, 0 either way) and the conversion rounds so in the
      // default mode of the processor; the packs below hold +128 at +127 as the integers do. The same bytes as the
      // integers below, which a denormal value can only change with a largest exponent under 8.
      const __m256 vScale = _mm256_castsi256_ps(_mm256_set1_epi32(int((cScaleBias - vLargest) << cExponentShift)));
      vMantissas          = _mm256_cvtps_epi32(_mm256_mul_ps(_mm256_castsi256_ps(vBits), vScale));
    } else {
      // the 24 bits of every value: its fraction and, when it is a normal float, the leading 1
      const __m256i vNormal   = _mm256_andnot_si256(_mm256_cmpeq_epi32(vExponents, _mm256_setzero_si256()), cHidden);
      const __m256i vFraction = _mm256_or_si256(_mm256_and_si256(vBits, cFraction), vNormal);
      // how far down each one goes, held at 31
      const __m256i vDown = _mm256_min_epu32(
          _mm256_add_epi32(_mm256_sub_epi32(_mm256_set1_epi32(int(vLargest)), vExponents), cShift), cShiftMax);
      const __m256i vQuotient = _mm256_srlv_epi32(vFraction, vDown);
      const __m256i vRest     = _mm256_sub_epi32(vFraction, _mm256_sllv_epi32(vQuotient, vDown));
      const __m256i vHalf     = _mm256_sllv_epi32(cOne, _mm256_sub_epi32(vDown, cOne));
      // one more when what is left is over half, or exactly half with an odd quotient (both are under 2 to the 31)
      const __m256i vUp = _mm256_or_si256(_mm256_and_si256(_mm256_cmpgt_epi32(vRest, vHalf), cOne),
                                          _mm256_and_si256(_mm256_cmpeq_epi32(vRest, vHalf), vQuotient));
      // the sign of the value, and a mantissa that would be +128 stays at +127
      vMantissas = _mm256_add_epi32(vQuotient, _mm256_and_si256(vUp, cOne));
      vMantissas = _mm256_sign_epi32(vMantissas, _mm256_or_si256(vBits, cOne));
      vMantissas = _mm256_min_epi32(vMantissas, cTop);
    }
    // from 32 bits to 8: the packs work on each half of the register, values 0 to 3 and 4 to 7
    const __m256i vWords = _mm256_packs_epi32(vMantissas, vMantissas);
    const __m256i vBytes = _mm256_packs_epi16(vWords, vWords);
    const int32_t vLow   = _mm_cvtsi128_si32(_mm256_castsi256_si128(vBytes));
    const int32_t vHigh  = _mm_cvtsi128_si32(_mm256_extracti128_si256(vBytes, 1));
    uint8_t *     vBlock = pBlocks + size_t(i) * cXdnaBfp16Bytes;
    vBlock[0] = uint8_t(vLargest);
    std::memcpy(vBlock + 1, &vLow, sizeof(vLow));
    std::memcpy(vBlock + 5, &vHigh, sizeof(vHigh));
  }
  return vFinite;
}

}  // namespace
#endif

bool fXdnaBfp16EncodeNearest(const float * pValues, size_t pStride, uint32_t pCount, uint8_t * pBlocks) {
#ifdef XDNA_BFP16_AVX2
  static const bool cHasAvx2 = __builtin_cpu_supports("avx2") != 0;
  if (cHasAvx2) {
    return fEncodeNearestAvx2(pValues, pStride, pCount, pBlocks);
  }
#endif
  bool vFinite = true;
  for (uint32_t i = 0; i < pCount; i++) {
    vFinite &= fEncodeNearestBlock(pValues + i * pStride, pBlocks + size_t(i) * cXdnaBfp16Bytes);
  }
  return vFinite;
}

bool fXdnaBfp16EncodeNearestNormed(const float * pValues, size_t pStride, uint32_t pCount, const float * pScales,
                                    const float * pWeights, uint8_t * pBlocks) {
#ifdef XDNA_BFP16_AVX2
  static const bool cHasAvx2 = __builtin_cpu_supports("avx2") != 0;
  if (cHasAvx2) {
    return fEncodeNearestAvx2(pValues, pStride, pCount, pBlocks, pScales, pWeights);
  }
#endif
  bool vFinite = true;
  for (uint32_t i = 0; i < pCount; i++) {
    float aNormed[cXdnaBfp16Values];
    for (uint32_t j = 0; j < cXdnaBfp16Values; j++) {
      aNormed[j] = (pValues[i * pStride + j] * pScales[i]) * pWeights[j];
    }
    vFinite &= fEncodeNearestBlock(aNormed, pBlocks + size_t(i) * cXdnaBfp16Bytes);
  }
  return vFinite;
}

void fXdnaBfp16Decode(const uint8_t * pBlock, float * pValues) {
  const float vScale = cScales.aScales[pBlock[0]];
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    pValues[i] = float(int8_t(pBlock[1 + i])) * vScale;
  }
}
