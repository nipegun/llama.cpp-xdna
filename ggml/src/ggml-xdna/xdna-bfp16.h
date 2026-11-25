#pragma once

// bfp16ebs8, the block floating point of AIE2P: 8 values share one 8-bit exponent and each has an 8-bit mantissa
// in two's complement, so 8 values take 9 bytes: the exponent and then the 8 mantissas. A value is its mantissa
// times 2 to the power of (exponent - 133); the exponent is the one of the largest value of the block as a
// 32-bit float, which then has 7 bits of mantissa and its sign.
//
// The encoding is the one of mlir-aie 1.4.5 (python/utils/bfp.py, the port of its helper.h). Rounding towards
// minus infinity gives its bytes, bit for bit. Rounding to the nearest gives the bytes of its "conv_even" except
// for one case: a negative value, or a negative zero, more than 31 binades below the largest of its block is 0
// here and -1 there.

#include <cstddef>
#include <cstdint>
#include <cstring>

const uint32_t cXdnaBfp16Values = 8;  // values of a block
const uint32_t cXdnaBfp16Bytes  = 9;  // bytes of a block

// what a mantissa becomes when the value has more bits than it holds
enum class XdnaRounding {
  cFloor,    // towards minus infinity: what the encoder of mlir-aie does by default
  cNearest,  // to the nearest, ties to even; a mantissa that would be +128 stays at +127
};

// Encodes 8 values into a block. False when one of them is not finite: the block is then not written.
bool fXdnaBfp16Encode(const float * pValues, uint8_t * pBlock, XdnaRounding pRounding);

// Encodes pCount blocks rounding to the nearest: the bytes fXdnaBfp16Encode gives with cNearest, much faster.
// The 8 values of the block i start at pValues + i * pStride, counted in floats, and its bytes go to
// pBlocks + i * 9. With the length of a row as the stride, the blocks are the same 8 columns of consecutive
// rows, which is how a tile of the kernel wants them. False when a value is not finite: the blocks are then
// not complete.
bool fXdnaBfp16EncodeNearest(const float * pValues, size_t pStride, uint32_t pCount, uint8_t * pBlocks);

// The same for values normalised first: value j of block i is (pValues[i * pStride + j] * pScales[i]) * pWeights[j],
// two products of floats in that order (an RMS norm by rows and its weights by columns), as the scalar code computes
// them, so the blocks are the same bit for bit.
bool fXdnaBfp16EncodeNearestNormed(const float * pValues, size_t pStride, uint32_t pCount, const float * pScales,
                                    const float * pWeights, uint8_t * pBlocks);

// the 8 values a block holds
void fXdnaBfp16Decode(const uint8_t * pBlock, float * pValues);

// bfloat16, how an overlay can give C back: the upper 16 bits of a 32-bit float, 8 bits of mantissa.

// The bfloat16 nearest to pValue, ties to even, as the cores round in the mode conv_even; a NaN stays a NaN.
inline uint16_t fXdnaBf16FromFloat(float pValue) {
  uint32_t vBits = 0;
  std::memcpy(&vBits, &pValue, sizeof(vBits));
  if ((vBits & 0x7fffffffu) > 0x7f800000u) {
    return uint16_t((vBits >> 16) | 0x40);  // a quiet NaN
  }
  vBits += 0x7fffu + ((vBits >> 16) & 1u);
  return uint16_t(vBits >> 16);
}

inline float fXdnaBf16ToFloat(uint16_t pValue) {
  const uint32_t vBits  = uint32_t(pValue) << 16;
  float          vValue = 0;
  std::memcpy(&vValue, &vBits, sizeof(vValue));
  return vValue;
}

// an IEEE half-precision float (the d and dmin of Q4_K) as a 32-bit float
inline float fXdnaHalfToFloat(uint16_t pHalf) {
  const uint32_t vSign     = uint32_t(pHalf & 0x8000) << 16;
  const uint32_t vExponent = (pHalf >> 10) & 0x1f;
  uint32_t       vMantissa = pHalf & 0x3ff;
  uint32_t       vBits     = 0;
  if (vExponent == 0x1f) {
    vBits = vSign | 0x7f800000u | (vMantissa << 13);
  } else if (vExponent != 0) {
    vBits = vSign | ((vExponent + 112) << 23) | (vMantissa << 13);
  } else if (vMantissa != 0) {
    // subnormal: normalise it
    uint32_t vShift = 0;
    while ((vMantissa & 0x400) == 0) {
      vMantissa <<= 1;
      vShift++;
    }
    vBits = vSign | ((113 - vShift) << 23) | ((vMantissa & 0x3ff) << 13);
  } else {
    vBits = vSign;
  }
  float vValue;
  std::memcpy(&vValue, &vBits, sizeof(vValue));
  return vValue;
}
