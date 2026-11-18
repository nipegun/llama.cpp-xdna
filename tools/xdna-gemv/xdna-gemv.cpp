// llama-xdna-gemv: multiplies a few rows by a matrix of weights on the overlay for generating tokens of the NPU
// (xdna-gemv.h) and compares the result with the CPU.
//
//   llama-xdna-gemv --overlay <main.pdi> --manifest <file> --gemv-overlay <main.pdi> --gemv-manifest <file>
//                    -M <rows> -K <K> -N <N> [--format bfp16|q4k|q5k|q6k] [--also <rows>x<K>x<N>]...
//                    [--mix <format>:<N>[,<format>:<N>]...]... [--repeat <n>] [--seed <n>] [--threads <n>]
//                    [--device <node>]
//       Y = X x W^T with random X (rows x K) and W (N x K). The context gets both overlays, the one of the prompts
//       (--overlay) and the one for generating tokens, as two compute units, the way the backend opens it; every
//       multiplication goes to the second. W is floats, which the host converts to bfp16, or random blocks of the
//       Q4_K, Q5_K or Q6_K of ggml (K a multiple of 256), which the cores decode themselves. Every --also adds a
//       multiplication of other sizes in the same format, and every --mix one of -M rows by -K values whose outputs
//       come in segments of those formats, all in one context, one after the other in every round. Each runs
//       --repeat times and is then checked.
//   llama-xdna-gemv ... --gemv-overlay <main.pdi> --gemv-manifest <file> --ffn [--next-layer] [--no-early]
//                    [--drains-first] [--sync-bo] [--ffn-scale <s>] [--repeat <n>] [--seed <n>]
//       With the chained overlay (xdna-chain.h; its manifest is version 7), the feed-forward block of a layer of
//       Qwen3.5 4B in one command of three phases, with random weights of Q4_K centred on zero as a model's are:
//       4096 -> 2560 plus a residual (h), the RMS norm of h and 2 x 9216 with their SwiGLU, and 9216 -> 2560 plus h.
//       h, the gate, the SwiGLU and the output are checked against a reference in 64-bit floats, with the operands
//       rounded as the cores round them, as gemv_chain_check.py of the lab does. --ffn-scale multiplies the weights of
//       the norms (1 by default). With --next-layer, a fourth phase as the backend sends it (fFeedForward): the RMS
//       norm of the output and the projections of a layer of linear attention of Qwen3.5 4B Q4_K_M (8192 outputs in
//       Q5_K, then 4096 + 32 + 32 in Q4_K: two segments), checked against a reference from the output the NPU gave.
//       --no-early writes the buffer descriptors of every phase after the wait for the phase before, not before it
//       (see fXdnaChainPlan), for comparing, and --drains-first queues the drains of a phase before its fills, as
//       before. --sync-bo (also with the cases above) flushes the cache around every chained command with the
//       SYNC_BO of the driver, not only on the lines written and read (see mRunChain).
//       The cases above also go to the chained overlay when it is the one given: with one row the cores round X
//       themselves.
//   llama-xdna-gemv ... --gemv-overlay <main.pdi> --gemv-manifest <file> --gdn [--tokens <n>] [--repeat <n>]
//       With the chained overlay (version 8 on), the recurrence of a layer of linear attention of Qwen3.5 4B (32 heads
//       of 128, the states in a buffer of the driver) for --tokens tokens, one command each, checked against a
//       reference in 64-bit floats; then --repeat more commands are timed (see fRunGdn).
//   llama-xdna-gemv ... --gemv-overlay <main.pdi> --gemv-manifest <file> --la [--tokens <n>] [--repeat <n>]
//       llama.cpp-xdna: with the chained overlay of version 9, a whole layer of linear attention of Qwen3.5 4B (the
//       convolution of its 8192 channels, the L2 norms, the recurrence and the gated norm) for --tokens tokens, checked
//       against a reference in 64-bit floats on the same inputs in bf16 (see fRunLa).
//
// The reference is the product of X and W as the NPU reads them, in 64-bit floats: what is left is the rounding of
// the 32-bit sums of the NPU. X, and W in bfp16, are rounded to block floating point by the host, to the nearest
// (X by the cores with the chained overlay and one row, as they do it: see fQuantizeLikeACore).
// W in the formats of ggml is decoded the way the kernel does it (kernels/xdna2-gemv-bfp16.cc), in 32-bit floats,
// and every 8 values are rounded to block floating point the way a core does it: to the nearest, the exponent of the
// block raised by one when its largest mantissa rounds up to +128 (bfp.quantize of mlir-aie, conv_even). In a rare
// block the cores round otherwise, which leaves an output one step of a weight away: in the formats of ggml, at most
// one output in 1000 may be up to 1e-3 of the largest value away. Before that, the decoding of the reference is
// checked against the one of ggml (ggml_get_type_traits), which differs only by d and d * scale rounded to bf16.

#include "xdna-bfp16.h"
#include "xdna-runner.h"

#include "ggml.h"

#include <algorithm>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace {

const double   cTolerance     = 1e-5;   // largest error allowed, as a part of the largest value of the reference
const double   cToleranceRare = 1e-3;   // in the formats of ggml, for at most one output in 1000
const double   cToleranceGgml = 0.02;   // the decoding of the reference against the one of ggml, as a part of the
                                        // largest weight of the row
const uint32_t cSuper         = 256;    // values of a block of the formats of ggml

struct Part {
  XdnaGemvFormat      vFormat  = XdnaGemvFormat::cBfp16;
  uint32_t             vOutputs = 0;
  std::vector<float>   aFloats;  // bfp16: vOutputs rows of K floats
  std::vector<uint8_t> aBlocks;  // the others: vOutputs rows of K / 256 blocks of ggml
};

struct Case {
  uint32_t            vRows = 0;
  uint32_t            vK    = 0;
  std::vector<Part>   aParts;
  std::vector<float>  aX;
  std::vector<float>  aY;
  XdnaGemvWeights    vWeights;
  XdnaChainWeights   vChainWeights;  // with the chained overlay
  std::vector<double> aNpuMs;
  std::vector<double> aPackMs;
  std::vector<double> aUnpackMs;

  uint32_t mOutputs() const {
    uint32_t vN = 0;
    for (const Part & vPart : aParts) {
      vN += vPart.vOutputs;
    }
    return vN;
  }
};

void fUsage(const char * pProgram) {
  fprintf(stderr,
          "usage: %s --overlay <main.pdi> --manifest <file> --gemv-overlay <main.pdi> --gemv-manifest <file>\n"
          "          -M <rows> -K <K> -N <N> [--format bfp16|q4k|q5k|q6k] [--also <rows>x<K>x<N>]...\n"
          "          [--mix <format>:<N>[,<format>:<N>]...]... [--repeat <n>] [--seed <n>] [--threads <n>]\n"
          "          [--sync-bo] [--device <node>]\n"
          "       %s --overlay <main.pdi> --manifest <file> --gemv-overlay <main.pdi> --gemv-manifest <file> --ffn\n"
          "          [--next-layer] [--no-early] [--drains-first] [--sync-bo] [--ffn-scale <s>] [--repeat <n>]\n"
          "          [--seed <n>] [--threads <n>] [--device <node>]\n"
          "       %s --overlay <main.pdi> --manifest <file> --gemv-overlay <main.pdi> --gemv-manifest <file> --gdn|--la\n"
          "          [--tokens <n>] [--repeat <n>] [--seed <n>] [--device <node>]\n",
          pProgram, pProgram, pProgram);
}

double fMedian(std::vector<double> aValues) {
  if (aValues.empty()) {
    return 0;
  }
  std::sort(aValues.begin(), aValues.end());
  return aValues[aValues.size() / 2];
}

std::string fDescribe(const Case & pCase) {
  std::string vText = std::to_string(pCase.vRows) + " x " + std::to_string(pCase.vK) + " x ";
  for (size_t i = 0; i < pCase.aParts.size(); i++) {
    vText += (i == 0 ? "" : "+") + std::to_string(pCase.aParts[i].vOutputs) + " " +
             fXdnaGemvFormatName(pCase.aParts[i].vFormat);
  }
  return vText;
}

// the values of a row as the NPU reads them: rounded to block floating point, to the nearest
void fAsTheNpu(const float * pRow, uint32_t pLength, std::vector<double> & aOut) {
  aOut.assign(pLength, 0.0);
  for (uint32_t vStart = 0; vStart < pLength; vStart += cXdnaBfp16Values) {
    float   aValues[cXdnaBfp16Values] = {};
    uint8_t aBlock[cXdnaBfp16Bytes];
    for (uint32_t i = 0; i < cXdnaBfp16Values && vStart + i < pLength; i++) {
      aValues[i] = pRow[vStart + i];
    }
    fXdnaBfp16Encode(aValues, aBlock, XdnaRounding::cNearest);
    fXdnaBfp16Decode(aBlock, aValues);
    for (uint32_t i = 0; i < cXdnaBfp16Values && vStart + i < pLength; i++) {
      aOut[vStart + i] = aValues[i];
    }
  }
}

// 8 values rounded to block floating point the way a core converts them: to the nearest, ties to even, and the
// exponent of the block raised by one when its largest mantissa rounds up to +128 (bfp.quantize of mlir-aie)
void fQuantizeLikeACore(const float * pValues, double * pOut) {
  uint32_t vExponent = 0;
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    uint32_t vBits = 0;
    std::memcpy(&vBits, pValues + i, sizeof(vBits));
    vExponent = std::max(vExponent, (vBits >> 23) & 0xffu);
  }
  double vStep = std::ldexp(1.0, int(vExponent) - 127 - 6);
  double aMantissas[cXdnaBfp16Values];
  double vLargest = -HUGE_VAL;
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    aMantissas[i] = std::nearbyint(double(pValues[i]) / vStep);
    vLargest      = std::max(vLargest, aMantissas[i]);
  }
  if (vLargest > 127) {
    vStep *= 2;
    for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
      aMantissas[i] = std::nearbyint(double(pValues[i]) / vStep);
    }
  }
  for (uint32_t i = 0; i < cXdnaBfp16Values; i++) {
    pOut[i] = double(float(std::clamp(aMantissas[i], -128.0, 127.0) * vStep));
  }
}

// the values of a row of floats as the cores of the chained overlay read them when they round it themselves: every 8
// values (zeros past the end) as fQuantizeLikeACore does
void fAsTheCores(const float * pRow, uint32_t pLength, std::vector<double> & aOut) {
  aOut.assign(pLength, 0.0);
  for (uint32_t vStart = 0; vStart < pLength; vStart += cXdnaBfp16Values) {
    float  aValues[cXdnaBfp16Values] = {};
    double aRounded[cXdnaBfp16Values];
    for (uint32_t i = 0; i < cXdnaBfp16Values && vStart + i < pLength; i++) {
      aValues[i] = pRow[vStart + i];
    }
    fQuantizeLikeACore(aValues, aRounded);
    for (uint32_t i = 0; i < cXdnaBfp16Values && vStart + i < pLength; i++) {
      aOut[vStart + i] = aRounded[i];
    }
  }
}

// the 6-bit scale and minimum of the sub-block pSub of a block of Q4_K or Q5_K, as get_scale_min_k4 of ggml
void fScaleMin(uint32_t pSub, const uint8_t * pBytes, uint32_t & pScale, uint32_t & pMinimum) {
  if (pSub < 4) {
    pScale   = pBytes[pSub] & 63;
    pMinimum = pBytes[pSub + 4] & 63;
  } else {
    pScale   = (pBytes[pSub + 4] & 0xf) | ((pBytes[pSub - 4] >> 6) << 4);
    pMinimum = (pBytes[pSub + 4] >> 4) | ((pBytes[pSub] >> 6) << 4);
  }
}

float fBf16(float pValue) {
  return fXdnaBf16ToFloat(fXdnaBf16FromFloat(pValue));
}

float fHalfAsBf16(const uint8_t * pFrom) {
  uint16_t vHalf = 0;
  std::memcpy(&vHalf, pFrom, 2);
  return fBf16(fXdnaHalfToFloat(vHalf));
}

// The 256 values of a block of ggml as the kernel computes them, in 32-bit floats and in its order. Q4_K and Q5_K: d
// and dmin rounded to bf16, D = bf16(d * scale), base = -128 * D - dmin * (128 + minimum) + 128 * dmin and value =
// base + (128 + q) * D. Q6_K: D = bf16(d * scale) and value = (128 + q) * D - 160 * D. Every product is exact in 32
// bits, so it makes no difference whether the compiler fuses them with the sums.
void fDecodeLikeTheKernel(XdnaGemvFormat pFormat, const uint8_t * pBlock, float * aValues) {
  if (pFormat == XdnaGemvFormat::cQ6K) {
    // block_q6_K: ql [128], qh [64], scales [16] (signed), d
    const float vD = fHalfAsBf16(pBlock + 208);
    for (uint32_t h = 0; h < 2; h++) {
      for (uint32_t k = 0; k < 4; k++) {
        for (uint32_t l = 0; l < 32; l++) {
          const uint32_t v       = 128 * h + 32 * k + l;
          const uint8_t  vByte   = pBlock[64 * h + (k % 2) * 32 + l];
          const uint32_t vLow    = k < 2 ? (vByte & 0xf) : (vByte >> 4);
          const uint32_t vTop    = (pBlock[128 + 32 * h + l] >> (2 * k)) & 3;
          const float    vFactor = fBf16(vD * float(int8_t(pBlock[192 + v / 16])));
          const float    vBase   = -160.0f * vFactor;
          aValues[v]             = vBase + (128.0f + float(vLow | (vTop << 4))) * vFactor;
        }
      }
    }
    return;
  }
  // block_q4_K: d, dmin, scales [12], qs [128]; block_q5_K: d, dmin, scales [12], qh [32], qs [128]
  const bool      vFifth = pFormat == XdnaGemvFormat::cQ5K;
  const float     vD     = fHalfAsBf16(pBlock);
  const float     vDmin  = fHalfAsBf16(pBlock + 2);
  const uint8_t * vQs    = pBlock + (vFifth ? 48 : 16);
  for (uint32_t vSub = 0; vSub < 8; vSub++) {
    uint32_t vScale   = 0;
    uint32_t vMinimum = 0;
    fScaleMin(vSub, pBlock + 4, vScale, vMinimum);
    const float vFactor = fBf16(vD * float(vScale));
    float       vBase   = -128.0f * vFactor;
    vBase               = vBase - vDmin * (128.0f + float(vMinimum));
    vBase               = vBase + vDmin * 128.0f;
    // byte l of the pair of sub-blocks c holds the value 64c + l in its low half and 64c + 32 + l in its high one;
    // the fifth bit of value 32s + l is bit s of qh[l]
    for (uint32_t l = 0; l < 32; l++) {
      const uint8_t vByte = vQs[(vSub / 2) * 32 + l];
      uint32_t      vQ    = vSub % 2 == 0 ? (vByte & 0xf) : (vByte >> 4);
      if (vFifth) {
        vQ |= ((pBlock[16 + l] >> vSub) & 1) << 4;
      }
      aValues[vSub * 32 + l] = vBase + (128.0f + float(vQ)) * vFactor;
    }
  }
}

enum ggml_type fGgmlType(XdnaGemvFormat pFormat) {
  return pFormat == XdnaGemvFormat::cQ4K ? GGML_TYPE_Q4_K :
         pFormat == XdnaGemvFormat::cQ5K ? GGML_TYPE_Q5_K : GGML_TYPE_Q6_K;
}

// The values of a row of blocks as the cores read them: decoded the way the kernel does it and rounded to block
// floating point the way a core does it. pWorst gets how far the decoding is from the one of ggml, as a part of the
// largest weight of the row.
void fAsTheNpuGgml(XdnaGemvFormat pFormat, const uint8_t * pRow, uint32_t pLength, std::vector<double> & aOut,
                   double & pWorst) {
  aOut.assign(pLength, 0.0);
  const uint32_t     vBlockBytes = fXdnaGemvBlockBytes(pFormat);
  std::vector<float> aGgml(pLength);
  ggml_get_type_traits(fGgmlType(pFormat))->to_float(pRow, aGgml.data(), pLength);
  float  aValues[cSuper];
  double vLargest = 0;
  double vError   = 0;
  for (uint32_t vBlock = 0; vBlock < pLength / cSuper; vBlock++) {
    fDecodeLikeTheKernel(pFormat, pRow + size_t(vBlock) * vBlockBytes, aValues);
    for (uint32_t i = 0; i < cSuper; i++) {
      vLargest = std::max(vLargest, std::fabs(double(aGgml[vBlock * cSuper + i])));
      vError   = std::max(vError, std::fabs(double(aValues[i]) - double(aGgml[vBlock * cSuper + i])));
    }
    for (uint32_t vStart = 0; vStart < cSuper; vStart += cXdnaBfp16Values) {
      fQuantizeLikeACore(aValues + vStart, aOut.data() + size_t(vBlock) * cSuper + vStart);
    }
  }
  pWorst = std::max(pWorst, vLargest > 0 ? vError / vLargest : vError);
}

bool fCheck(const Case & pCase, bool pCoresRoundX) {
  const uint32_t                   vN = pCase.mOutputs();
  std::vector<std::vector<double>> aX(pCase.vRows);
  for (uint32_t vRow = 0; vRow < pCase.vRows; vRow++) {
    if (pCoresRoundX) {
      fAsTheCores(pCase.aX.data() + size_t(vRow) * pCase.vK, pCase.vK, aX[vRow]);
    } else {
      fAsTheNpu(pCase.aX.data() + size_t(vRow) * pCase.vK, pCase.vK, aX[vRow]);
    }
  }
  std::vector<double> aW;
  double              vScale     = 0;
  double              vWorst     = 0;
  double              vWorstGgml = 0;
  uint64_t            vEqual     = 0;
  bool                vFinite    = true;
  bool                vGgml      = false;  // whether a format of ggml is in the case
  std::vector<double> aReference(size_t(pCase.vRows) * vN);
  uint32_t            vOutput = 0;
  for (const Part & vPart : pCase.aParts) {
    const size_t vRowBytes = size_t(pCase.vK / cSuper) * fXdnaGemvBlockBytes(vPart.vFormat);
    vGgml                  = vGgml || vPart.vFormat != XdnaGemvFormat::cBfp16;
    for (uint32_t vRowW = 0; vRowW < vPart.vOutputs; vRowW++, vOutput++) {
      if (vPart.vFormat == XdnaGemvFormat::cBfp16) {
        fAsTheNpu(vPart.aFloats.data() + size_t(vRowW) * pCase.vK, pCase.vK, aW);
      } else {
        fAsTheNpuGgml(vPart.vFormat, vPart.aBlocks.data() + size_t(vRowW) * vRowBytes, pCase.vK, aW, vWorstGgml);
      }
      for (uint32_t vRow = 0; vRow < pCase.vRows; vRow++) {
        double vSum = 0;
        for (uint32_t k = 0; k < pCase.vK; k++) {
          vSum += aX[vRow][k] * aW[k];
        }
        aReference[size_t(vRow) * vN + vOutput] = vSum;
        vScale = std::max(vScale, std::fabs(vSum));
      }
    }
  }
  uint64_t vBeyond = 0;
  for (size_t i = 0; i < aReference.size(); i++) {
    const double vValue = pCase.aY[i];
    if (!std::isfinite(vValue)) {
      vFinite = false;
      continue;
    }
    const double vError = std::fabs(vValue - aReference[i]);
    vWorst = std::max(vWorst, vError);
    vEqual += vError == 0 ? 1 : 0;
    vBeyond += vError > cTolerance * vScale ? 1 : 0;
    if (vError > cTolerance * vScale && vBeyond <= 16) {
      printf("  output %zu: %.4f, reference %.4f\n", i, vValue, aReference[i]);
    }
  }
  const bool vRare     = vGgml && vBeyond * 1000 <= aReference.size() && vWorst <= cToleranceRare * vScale;
  const bool vDecoding = vWorstGgml <= cToleranceGgml;
  const bool vOk       = vFinite && vDecoding && (vWorst <= cTolerance * vScale || vRare);
  printf("%s: largest value %.6g, %" PRIu64 " of %zu values equal to the reference, largest error %.3g (%.3g of the"
         " largest value), %" PRIu64 " beyond %g", fDescribe(pCase).c_str(), vScale, vEqual, aReference.size(), vWorst,
         vScale > 0 ? vWorst / vScale : 0.0, vBeyond, cTolerance);
  if (vGgml) {
    printf("; decoding against ggml %.3g of the largest weight", vWorstGgml);
  }
  printf(": %s\n", vOk ? "pass" : "FAIL");
  return vOk;
}

bool fParseSize(const char * pText, Case & pCase, XdnaGemvFormat pFormat) {
  uint32_t vN = 0;
  if (sscanf(pText, "%ux%ux%u", &pCase.vRows, &pCase.vK, &vN) != 3) {
    return false;
  }
  pCase.aParts.resize(1);
  pCase.aParts[0].vFormat  = pFormat;
  pCase.aParts[0].vOutputs = vN;
  return true;
}

bool fParseMix(const std::string & pText, Case & pCase) {
  for (size_t vStart = 0; vStart < pText.size();) {
    const size_t      vEnd   = pText.find(',', vStart);
    const std::string vItem  = pText.substr(vStart, vEnd == std::string::npos ? std::string::npos : vEnd - vStart);
    const size_t      vColon = vItem.find(':');
    Part              vPart;
    if (vColon == std::string::npos || !fXdnaGemvFormatFromName(vItem.substr(0, vColon), vPart.vFormat)) {
      return false;
    }
    vPart.vOutputs = uint32_t(strtoul(vItem.c_str() + vColon + 1, nullptr, 10));
    pCase.aParts.push_back(vPart);
    vStart = vEnd == std::string::npos ? pText.size() : vEnd + 1;
  }
  return !pCase.aParts.empty();
}

// d of Q6_K, and d and dmin of Q4_K and Q5_K: fp16 with either sign, d from 2^-11 to 2^-7 and dmin from 0 to 2^-8,
// subnormals included; scales, minimums and values anything
void fRandomBlocks(XdnaGemvFormat pFormat, uint32_t pRows, uint32_t pK, std::mt19937 & pRandom,
                   std::vector<uint8_t> & aBlocks) {
  std::uniform_int_distribution<uint32_t> vByte(0, 255);
  std::uniform_int_distribution<uint32_t> vMantissa(0, 1023);
  std::uniform_int_distribution<uint32_t> vExponentD(4, 7);
  std::uniform_int_distribution<uint32_t> vExponentDmin(0, 6);
  const uint32_t                          vBlockBytes = fXdnaGemvBlockBytes(pFormat);
  aBlocks.resize(size_t(pRows) * (pK / cSuper) * vBlockBytes);
  for (size_t vStart = 0; vStart < aBlocks.size(); vStart += vBlockBytes) {
    for (uint32_t i = 0; i < vBlockBytes; i++) {
      aBlocks[vStart + i] = uint8_t(vByte(pRandom));
    }
    const uint16_t vHalfD    = uint16_t(((vByte(pRandom) & 1) << 15) | (vExponentD(pRandom) << 10) | vMantissa(pRandom));
    const uint16_t vHalfDmin = uint16_t(((vByte(pRandom) & 1) << 15) | (vExponentDmin(pRandom) << 10) |
                                        vMantissa(pRandom));
    if (pFormat == XdnaGemvFormat::cQ6K) {
      std::memcpy(aBlocks.data() + vStart + 208, &vHalfD, 2);
    } else {
      std::memcpy(aBlocks.data() + vStart, &vHalfD, 2);
      std::memcpy(aBlocks.data() + vStart + 2, &vHalfDmin, 2);
    }
  }
}

// Random blocks of Q4_K centred on zero, as a model's are: dmin = d (from 2^-11 to 2^-8) and every minimum 7.5 times
// its scale (1 to 8), so that the 16 values of a sub-block go from -7.5 to 7.5 times d * scale. With the blocks of
// fRandomBlocks the gates of a feed-forward layer all come out very negative, and its SwiGLU is not tried.
void fRandomCentredQ4K(uint32_t pRows, uint32_t pK, std::mt19937 & pRandom, std::vector<uint8_t> & aBlocks) {
  std::uniform_int_distribution<uint32_t> vByte(0, 255);
  std::uniform_int_distribution<uint32_t> vMantissa(0, 1023);
  std::uniform_int_distribution<uint32_t> vExponent(4, 6);
  std::uniform_int_distribution<uint32_t> vScale(1, 8);
  const uint32_t                          vBlockBytes = fXdnaGemvBlockBytes(XdnaGemvFormat::cQ4K);
  aBlocks.resize(size_t(pRows) * (pK / cSuper) * vBlockBytes);
  for (size_t vStart = 0; vStart < aBlocks.size(); vStart += vBlockBytes) {
    uint8_t *      vBlock = aBlocks.data() + vStart;
    const uint16_t vHalfD = uint16_t((vExponent(pRandom) << 10) | vMantissa(pRandom));
    std::memcpy(vBlock, &vHalfD, 2);
    std::memcpy(vBlock + 2, &vHalfD, 2);
    uint32_t aScales[8];
    uint32_t aMinimums[8];
    for (uint32_t j = 0; j < 8; j++) {
      aScales[j]   = vScale(pRandom);
      aMinimums[j] = uint32_t(std::nearbyint(7.5 * aScales[j]));
    }
    // the 12 bytes of the 6-bit scales and minimums, as get_scale_min_k4 of ggml reads them
    for (uint32_t j = 0; j < 4; j++) {
      vBlock[4 + j]     = uint8_t(aScales[j] | ((aScales[j + 4] >> 4) << 6));
      vBlock[4 + j + 4] = uint8_t(aMinimums[j] | ((aMinimums[j + 4] >> 4) << 6));
      vBlock[4 + j + 8] = uint8_t((aScales[j + 4] & 0xf) | ((aMinimums[j + 4] & 0xf) << 4));
    }
    for (uint32_t i = 16; i < vBlockBytes; i++) {
      vBlock[i] = uint8_t(vByte(pRandom));
    }
  }
}

// a matrix of the feed-forward test: its rows in a format of ggml (Q4_K but for the projections of --next-layer) and
// its outputs
struct FfnMatrix {
  uint32_t             vOutputs = 0;
  uint32_t             vInputs  = 0;
  std::vector<uint8_t> aBlocks;
  XdnaGemvFormat      vFormat  = XdnaGemvFormat::cQ4K;

  size_t mRowBytes() const { return size_t(vInputs / cSuper) * fXdnaGemvBlockBytes(vFormat); }
};

// pX (rounded as the NPU reads it) by every row of pMatrix as the NPU reads it, in 64-bit floats
void fReferenceProduct(const FfnMatrix & pMatrix, const std::vector<double> & aX, std::vector<double> & aY,
                       double & pWorstGgml) {
  aY.assign(pMatrix.vOutputs, 0.0);
  std::vector<double> aW;
  for (uint32_t vRow = 0; vRow < pMatrix.vOutputs; vRow++) {
    fAsTheNpuGgml(pMatrix.vFormat, pMatrix.aBlocks.data() + size_t(vRow) * pMatrix.mRowBytes(), pMatrix.vInputs,
                  aW, pWorstGgml);
    double vSum = 0;
    for (uint32_t k = 0; k < pMatrix.vInputs; k++) {
      vSum += aX[k] * aW[k];
    }
    aY[vRow] = vSum;
  }
}

// how far a result of the NPU is from its reference, as a part of the largest value of the reference
bool fCompare(const char * pName, const std::vector<float> & aGot, const std::vector<double> & aReference,
              double pTolerance) {
  double vScale = 0;
  double vWorst = 0;
  double vSum   = 0;
  bool   vFinite = true;
  for (size_t i = 0; i < aReference.size(); i++) {
    vScale = std::max(vScale, std::fabs(aReference[i]));
    if (!std::isfinite(aGot[i])) {
      vFinite = false;
      continue;
    }
    const double vError = std::fabs(double(aGot[i]) - aReference[i]);
    vWorst = std::max(vWorst, vError);
    vSum += vError;
  }
  const double vRelative = vScale > 0 ? vWorst / vScale : vWorst;
  const bool   vOk       = vFinite && vRelative <= pTolerance;
  printf("%s: largest value %.5g, largest error %.3g of it, mean error %.3g, tolerance %g: %s\n", pName, vScale,
         vRelative, vScale > 0 ? vSum / double(aReference.size()) / vScale : 0.0, pTolerance, vOk ? "pass" : "FAIL");
  return vOk;
}

// The feed-forward block of a layer of Qwen3.5 4B in one command of the chained overlay (see the usage), and with
// pNext the norm and the projections of the next layer too, as a fourth phase.
bool fRunFfn(XdnaRunner & pRunner, int pRepeat, uint32_t pSeed, float pScale, bool pNext) {
  const uint32_t cEmbedding = 2560;
  const uint32_t cInner     = 4096;  // what the first matrix takes: the output of the attention or of the SSM
  const uint32_t cHidden    = 9216;
  const float    cEps       = 1e-6f;
  std::mt19937                          vRandom(pSeed);
  std::uniform_real_distribution<float> vUniform(-1.0f, 1.0f);
  FfnMatrix vFirst  = {cEmbedding, cInner, {}};
  FfnMatrix vGate   = {cHidden, cEmbedding, {}};
  FfnMatrix vUp     = {cHidden, cEmbedding, {}};
  FfnMatrix vDown   = {cEmbedding, cHidden, {}};
  for (FfnMatrix * vMatrix : {&vFirst, &vGate, &vUp, &vDown}) {
    fRandomCentredQ4K(vMatrix->vOutputs, vMatrix->vInputs, vRandom, vMatrix->aBlocks);
  }
  // the projections of a layer of linear attention of Qwen3.5 4B Q4_K_M: attn_qkv in Q5_K, then attn_gate, ssm_beta
  // and ssm_alpha in Q4_K, which make two segments
  FfnMatrix aNext[4] = {{8192, cEmbedding, {}, XdnaGemvFormat::cQ5K}, {4096, cEmbedding, {}}, {32, cEmbedding, {}},
                        {32, cEmbedding, {}}};
  if (pNext) {
    fRandomBlocks(aNext[0].vFormat, aNext[0].vOutputs, cEmbedding, vRandom, aNext[0].aBlocks);
    for (uint32_t m = 1; m < 4; m++) {
      fRandomCentredQ4K(aNext[m].vOutputs, cEmbedding, vRandom, aNext[m].aBlocks);
    }
  }
  std::vector<float> aX(cInner);
  std::vector<float> aResidual(cEmbedding);
  std::vector<float> aNorm(cEmbedding);
  std::vector<float> aNormNext(cEmbedding);
  for (float & vValue : aX) {
    vValue = vUniform(vRandom);
  }
  for (float & vValue : aResidual) {
    vValue = 2.0f * vUniform(vRandom);
  }
  for (float & vValue : aNorm) {
    vValue = pScale * (1.0f + 0.2f * vUniform(vRandom));
  }
  for (float & vValue : aNormNext) {
    vValue = pScale * (1.0f + 0.2f * vUniform(vRandom));
  }

  // the reference, in 64-bit floats, with the operands rounded as the cores round them
  double              vWorstGgml = 0;
  std::vector<double> aRounded, aProduct, aH(cEmbedding), aGate(cHidden), aUpRef(cHidden), aSwiglu(cHidden), aOut;
  fAsTheCores(aX.data(), cInner, aRounded);
  fReferenceProduct(vFirst, aRounded, aProduct, vWorstGgml);
  double vSquares = 0;
  for (uint32_t i = 0; i < cEmbedding; i++) {
    aH[i] = aProduct[i] + double(aResidual[i]);
    vSquares += aH[i] * aH[i];
  }
  const double       vScale = 1.0 / std::sqrt(vSquares / cEmbedding + double(cEps));
  std::vector<float> aNormed(cEmbedding);
  for (uint32_t i = 0; i < cEmbedding; i++) {
    aNormed[i] = float(aH[i]) * aNorm[i];
  }
  fAsTheCores(aNormed.data(), cEmbedding, aRounded);
  fReferenceProduct(vGate, aRounded, aGate, vWorstGgml);
  fReferenceProduct(vUp, aRounded, aUpRef, vWorstGgml);
  std::vector<float> aSwigluFloat(cHidden);
  for (uint32_t i = 0; i < cHidden; i++) {
    aGate[i] *= vScale;
    aSwiglu[i]      = aGate[i] / (1.0 + std::exp(-aGate[i])) * (vScale * aUpRef[i]);
    aSwigluFloat[i] = float(aSwiglu[i]);
  }
  fAsTheCores(aSwigluFloat.data(), cHidden, aRounded);
  fReferenceProduct(vDown, aRounded, aOut, vWorstGgml);
  for (uint32_t i = 0; i < cEmbedding; i++) {
    aOut[i] += aH[i];
  }

  // the three matrices in the NPU: the gate and the up as two segments of one; with pNext, the projections as a fourth
  // one, consecutive members of a format together
  auto fMember = [](const FfnMatrix & pMatrix) {
    XdnaGemvMember vMember;
    vMember.vFormat  = pMatrix.vFormat;
    vMember.vOutputs = pMatrix.vOutputs;
    const FfnMatrix * vFrom = &pMatrix;
    vMember.fRows = [vFrom](uint32_t pFirst, uint32_t pCount, void * pTo) {
      std::memcpy(pTo, vFrom->aBlocks.data() + size_t(pFirst) * vFrom->mRowBytes(), size_t(pCount) * vFrom->mRowBytes());
      return true;
    };
    return vMember;
  };
  XdnaChainWeights vWeightsFirst, vWeightsGateUp, vWeightsDown, vWeightsNext;
  std::string       vError;
  if (!pRunner.mLoadChainMatrix({fMember(vFirst)}, cInner, false, vWeightsFirst, vError) ||
      !pRunner.mLoadChainMatrix({fMember(vGate), fMember(vUp)}, cEmbedding, true, vWeightsGateUp, vError) ||
      !pRunner.mLoadChainMatrix({fMember(vDown)}, cHidden, false, vWeightsDown, vError) ||
      (pNext && !pRunner.mLoadChainMatrix({fMember(aNext[0]), fMember(aNext[1]), fMember(aNext[2]), fMember(aNext[3])},
                                          cEmbedding, false, vWeightsNext, vError))) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return false;
  }
  const uint32_t vPhases = pNext ? 4 : 3;
  XdnaChainPhase aPhases[4];
  aPhases[0].vPrologue     = XdnaChainPrologue::cFloats;
  aPhases[0].vEpilogue     = XdnaChainEpilogue::cResidual;
  aPhases[0].vMatrix       = 0;
  aPhases[1].vPrologue     = XdnaChainPrologue::cNorm;
  aPhases[1].vFromPhase    = 0;
  aPhases[1].vEpilogue     = XdnaChainEpilogue::cSwiglu;
  aPhases[1].vEps          = cEps;
  aPhases[1].vMatrix       = 1;
  aPhases[1].vSegments     = 2;
  aPhases[2].vPrologue     = XdnaChainPrologue::cFloats;
  aPhases[2].vFromPhase    = 1;
  aPhases[2].vFromSegment  = 1;
  aPhases[2].vEpilogue     = XdnaChainEpilogue::cAddH;
  aPhases[2].vMatrix       = 2;
  aPhases[3].vPrologue     = XdnaChainPrologue::cNorm;
  aPhases[3].vFromPhase    = 2;
  aPhases[3].vEpilogue     = XdnaChainEpilogue::cNone;
  aPhases[3].vEps          = cEps;
  aPhases[3].vMatrix       = 3;
  aPhases[3].vSegments     = pNext ? vWeightsNext.vMatrix.vSegments : 1;
  uint32_t vNextOutputs = 0;
  for (const FfnMatrix & vMatrix : aNext) {
    vNextOutputs += vMatrix.vOutputs;
  }
  std::vector<float> aGotH(cEmbedding), aGotGate(cHidden), aGotSwiglu(cHidden), aGotOut(cEmbedding);
  std::vector<float> aGotNext(vNextOutputs);
  XdnaGemmTarget    vTargetH      = {aGotH.data(), cEmbedding, 0, cEmbedding};
  XdnaGemmTarget    aTargetsGlu[2] = {{aGotGate.data(), cHidden, 0, cHidden}, {aGotSwiglu.data(), cHidden, cHidden, cHidden}};
  XdnaGemmTarget    vTargetOut    = {aGotOut.data(), cEmbedding, 0, cEmbedding};
  XdnaGemmTarget    aTargetsNext[4];  // the projections, one target each, as the backend gives them
  for (uint32_t m = 0, vFirstOutput = 0; m < 4; m++) {
    aTargetsNext[m] = {aGotNext.data() + vFirstOutput, aNext[m].vOutputs, vFirstOutput, aNext[m].vOutputs};
    vFirstOutput += aNext[m].vOutputs;
  }
  XdnaChainIo       aIo[4];
  aIo[0].vX        = aX.data();
  aIo[0].vStride   = cInner;
  aIo[0].vResidual = aResidual.data();
  aIo[0].aTargets  = &vTargetH;
  aIo[0].vTargets  = 1;
  aIo[1].vNorm     = aNorm.data();
  aIo[1].aTargets  = aTargetsGlu;
  aIo[1].vTargets  = 2;
  aIo[2].aTargets  = &vTargetOut;
  aIo[2].vTargets  = 1;
  aIo[3].vNorm     = aNormNext.data();
  aIo[3].aTargets  = aTargetsNext;
  aIo[3].vTargets  = 4;
  const XdnaChainWeights * aWeights[4] = {&vWeightsFirst, &vWeightsGateUp, &vWeightsDown, &vWeightsNext};
  std::vector<double> aNpuMs, aPackMs, aUnpackMs;
  for (int vRound = 0; vRound < pRepeat; vRound++) {
    XdnaGemmTimes vTimes;
    if (!pRunner.mRunChain(aPhases, aIo, vPhases, aWeights, vPhases, vTimes, vError)) {
      fprintf(stderr, "error: round %d: %s\n", vRound, vError.c_str());
      return false;
    }
    aNpuMs.push_back(vTimes.vNpuMs);
    aPackMs.push_back(vTimes.vPackMs);
    aUnpackMs.push_back(vTimes.vUnpackMs);
  }
  printf("feed-forward block %u -> %u (+ residual), norm, 2 x %u (SwiGLU), %u -> %u (+ h), Q4_K%s: NPU median %.3f ms, "
         "writing X %.3f ms, reading Y %.3f ms (%d rounds)\n", cInner, cEmbedding, cHidden, cHidden, cEmbedding,
         pNext ? ", then the norm and 8192 Q5_K + 4096 + 32 + 32 Q4_K of the next layer" : "", fMedian(aNpuMs),
         fMedian(aPackMs), fMedian(aUnpackMs), pRepeat);
  bool vOk = fCompare("h", aGotH, aH, 1e-5);
  vOk      = fCompare("gate", aGotGate, aGate, 1e-3) && vOk;
  vOk      = fCompare("swiglu", aGotSwiglu, aSwiglu, 1e-2) && vOk;
  vOk      = fCompare("output", aGotOut, aOut, 1e-2) && vOk;
  if (pNext) {
    // the reference of the fourth phase from the output the NPU gave, which is what it read: the norm and the
    // projections alone are checked
    double vSquares = 0;
    for (uint32_t i = 0; i < cEmbedding; i++) {
      vSquares += double(aGotOut[i]) * double(aGotOut[i]);
    }
    const double       vScaleNext = 1.0 / std::sqrt(vSquares / cEmbedding + double(cEps));
    std::vector<float> aNormedNext(cEmbedding);
    for (uint32_t i = 0; i < cEmbedding; i++) {
      aNormedNext[i] = aGotOut[i] * aNormNext[i];
    }
    fAsTheCores(aNormedNext.data(), cEmbedding, aRounded);
    std::vector<double> aNextRef, aOne;
    for (const FfnMatrix & vMatrix : aNext) {
      fReferenceProduct(vMatrix, aRounded, aOne, vWorstGgml);
      for (double vValue : aOne) {
        aNextRef.push_back(vScaleNext * vValue);
      }
    }
    vOk = fCompare("next layer", aGotNext, aNextRef, 1e-3) && vOk;
  }
  printf("decoding against ggml: %.3g of the largest weight\n", vWorstGgml);
  vOk = vOk && vWorstGgml <= cToleranceGgml;
  pRunner.mFreeChainWeights(vWeightsFirst);
  pRunner.mFreeChainWeights(vWeightsGateUp);
  pRunner.mFreeChainWeights(vWeightsDown);
  if (pNext) {
    pRunner.mFreeChainWeights(vWeightsNext);
  }
  printf("RESULT: %s\n", vOk ? "PASS" : "FAIL");
  return vOk;
}

// The recurrence of a layer of linear attention of Qwen3.5 4B on the chained overlay (version 8): 32 heads of values
// of 128, 16 heads of keys, a state of 64 KB per head in a buffer of the driver, as llama.cpp keeps it, for pTokens
// tokens one after the other (a command each, the state staying in the buffer), each with random q and k (of norm 1, as
// the model gives them), v, gates and betas. Every token, its outputs, and at the end the states, are checked against a
// reference in 64-bit floats from the same start; then pRepeat more commands are timed.
bool fRunGdn(XdnaRunner & pRunner, int pTokens, int pRepeat, uint32_t pSeed) {
  const uint32_t cHeads   = 32;
  const uint32_t cHeadsQK = 16;
  const uint32_t cDim     = cXdnaChainRecurrenceDim;
  const uint64_t cHead    = uint64_t(cDim) * cDim;   // floats of the state of a head
  const uint64_t cRoom    = 8192;                     // bytes after the states: the last object reads 3.5 KB past them
  const double   cScale   = 1.0 / std::sqrt(double(cDim));
  std::mt19937                          vRandom(pSeed);
  std::uniform_real_distribution<float> vUniform(-1.0f, 1.0f);
  std::uniform_real_distribution<float> vUnit(0.0f, 1.0f);

  XdnaBuffer vStates;
  std::string vError;
  if (!pRunner.mCreateShared(cHeads * cHead * sizeof(float) + cRoom, vStates, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return false;
  }
  float *             vState = static_cast<float *>(vStates.vMap);
  std::vector<double> aState(cHeads * cHead);
  for (uint64_t i = 0; i < cHeads * cHead; i++) {
    vState[i] = 0.5f * vUniform(vRandom);
    aState[i] = vState[i];
  }
  XdnaChainPhase vPhase;
  vPhase.vRecurrence.vHeads = cHeads;
  std::vector<float>     aQ(cHeadsQK * cDim), aK(cHeadsQK * cDim), aV(cHeads * cDim), aDecay(cHeads), aBeta(cHeads);
  std::vector<float>     aGot(cHeads * cDim);
  std::vector<double>    aWant(cHeads * cDim);
  XdnaChainRecurrenceIo vInputs;
  vInputs.vQ       = aQ.data();
  vInputs.vK       = aK.data();
  vInputs.vHeadsQK = cHeadsQK;
  vInputs.vV       = aV.data();
  vInputs.vDecay   = aDecay.data();
  vInputs.vBeta    = aBeta.data();
  XdnaChainIo vIo;
  vIo.vRecurrence = &vInputs;
  vIo.vOutputs    = aGot.data();
  vIo.vScale      = float(cScale);
  const XdnaBuffer * aStates[1] = {&vStates};
  const uint64_t      vFlushed   = vStates.vSize;
  auto fRandomInputs = [&]() {
    for (std::vector<float> * vVector : {&aQ, &aK}) {
      for (uint32_t h = 0; h < cHeadsQK; h++) {
        double vNorm = 0;
        for (uint32_t i = 0; i < cDim; i++) {
          (*vVector)[h * cDim + i] = vUniform(vRandom);
          vNorm += double((*vVector)[h * cDim + i]) * (*vVector)[h * cDim + i];
        }
        for (uint32_t i = 0; i < cDim; i++) {
          (*vVector)[h * cDim + i] = float((*vVector)[h * cDim + i] / std::sqrt(vNorm));
        }
      }
    }
    for (float & vValue : aV) {
      vValue = vUniform(vRandom);
    }
    for (uint32_t h = 0; h < cHeads; h++) {
      aDecay[h] = std::exp(-3.0f * vUnit(vRandom));  // e^g for a gate g from -3 to 0
      aBeta[h]  = vUnit(vRandom);
    }
  };
  bool   vOk         = true;
  double vWorstOut   = 0;
  double vLargestOut = 0;
  std::vector<double> aNpuMs;
  for (int vToken = 0; vToken < pTokens; vToken++) {
    fRandomInputs();
    // the reference: what ggml_gated_delta_net does to every row of every head, in 64-bit floats
    for (uint32_t j = 0; j < cHeads; j++) {
      const float * vQ = aQ.data() + (j % cHeadsQK) * cDim;
      const float * vK = aK.data() + (j % cHeadsQK) * cDim;
      for (uint32_t r = 0; r < cDim; r++) {
        double * vRow = aState.data() + j * cHead + uint64_t(r) * cDim;
        double   vDot = 0;
        for (uint32_t i = 0; i < cDim; i++) {
          vRow[i] *= double(aDecay[j]);
          vDot += vRow[i] * double(vK[i]);
        }
        const double vDelta = (double(aV[j * cDim + r]) - vDot) * double(aBeta[j]);
        double       vOut   = 0;
        for (uint32_t i = 0; i < cDim; i++) {
          vRow[i] += vDelta * double(vK[i]);
          vOut += vRow[i] * double(vQ[i]);
        }
        aWant[j * cDim + r] = vOut * cScale;
      }
    }
    // the states leave the cache of the processor before the command (the host wrote them first) and after it (the
    // host reads them)
    fXdnaFlushCache(vState, 1, vFlushed, vFlushed);
    XdnaGemmTimes vTimes;
    if (!pRunner.mRunChain(&vPhase, &vIo, 1, nullptr, 0, vTimes, vError, aStates, 1)) {
      fprintf(stderr, "error: token %d: %s\n", vToken, vError.c_str());
      pRunner.mFreeShared(vStates);
      return false;
    }
    fXdnaFlushCache(vState, 1, vFlushed, vFlushed);
    aNpuMs.push_back(vTimes.vNpuMs);
    for (uint64_t i = 0; i < aGot.size(); i++) {
      vLargestOut = std::max(vLargestOut, std::fabs(aWant[i]));
      vWorstOut   = std::max(vWorstOut, std::fabs(double(aGot[i]) - aWant[i]));
    }
  }
  double vWorstState = 0;
  double vLargestState = 0;
  for (uint64_t i = 0; i < aState.size(); i++) {
    vLargestState = std::max(vLargestState, std::fabs(aState[i]));
    vWorstState   = std::max(vWorstState, std::fabs(double(vState[i]) - aState[i]));
  }
  // what the last object of the last head read past the states is not written
  printf("recurrence of 32 heads of 128 (16 heads of q and k), %d tokens: NPU median %.3f ms a token\n", pTokens,
         fMedian(aNpuMs));
  printf("outputs: largest error %.3g of the largest value (%.3g)\n", vWorstOut / std::max(vLargestOut, 1e-30),
         vLargestOut);
  printf("states after the last token: largest error %.3g of the largest value (%.3g)\n",
         vWorstState / std::max(vLargestState, 1e-30), vLargestState);
  vOk = vWorstOut <= 1e-4 * vLargestOut && vWorstState <= 1e-4 * vLargestState;
  // the time of more commands, with the same inputs (the states go on changing)
  aNpuMs.clear();
  for (int vRound = 0; vRound < pRepeat; vRound++) {
    XdnaGemmTimes vTimes;
    if (!pRunner.mRunChain(&vPhase, &vIo, 1, nullptr, 0, vTimes, vError, aStates, 1)) {
      fprintf(stderr, "error: round %d: %s\n", vRound, vError.c_str());
      pRunner.mFreeShared(vStates);
      return false;
    }
    aNpuMs.push_back(vTimes.vNpuMs);
  }
  printf("timed: NPU median %.3f ms a command (%d rounds)\n", fMedian(aNpuMs), pRepeat);
  pRunner.mFreeShared(vStates);
  printf("RESULT: %s\n", vOk ? "PASS" : "FAIL");
  return vOk;
}

// llama.cpp-xdna, version 9: a whole layer of linear attention of Qwen3.5 4B on the chained overlay (one phase, see
// XdnaChainRecurrence::vLinear): 16 heads of keys and 32 value heads of 128, the 8192 channels of the convolution with
// random weights and states, z, gates and betas, for pTokens tokens one after the other (a command each; the host
// shifts the states of the convolution, and those of the recurrence stay in their buffer of the driver). Every token,
// the outputs (the input of the projection of the layer) are checked against a reference in 64-bit floats on the same
// inputs rounded to bf16 as the cores get them, and at the end the states of the recurrence; then pRepeat more
// commands are timed.
bool fRunLa(XdnaRunner & pRunner, int pTokens, int pRepeat, uint32_t pSeed) {
  const uint32_t cHeads    = cXdnaChainLinearHeads;
  const uint32_t cKeys     = cXdnaChainLinearKeys;
  const uint32_t cDim      = cXdnaChainRecurrenceDim;
  const uint32_t cChannels = cXdnaChainLinearChannels;
  const uint32_t cTaps     = cXdnaChainLinearTaps;
  const uint64_t cHead     = uint64_t(cDim) * cDim;  // floats of the state of a head
  const uint64_t cRoom     = 8192;                    // bytes after the states: the last object reads 3.5 KB past them
  const float    cEps      = 1e-6f;
  std::mt19937                          vRandom(pSeed);
  std::uniform_real_distribution<float> vUniform(-1.0f, 1.0f);
  std::uniform_real_distribution<float> vUnit(0.0f, 1.0f);
  auto fBf16 = [](float pValue) { return double(fXdnaBf16ToFloat(fXdnaBf16FromFloat(pValue))); };
  auto fSilu = [](double pValue) { return pValue / (1.0 + std::exp(-pValue)); };

  // the weights: of the convolution [channel][4] and of the gated norm
  std::vector<float> aConv(uint64_t(cChannels) * cTaps), aNorm(cDim);
  for (float & vValue : aConv) {
    vValue = 0.5f * vUniform(vRandom);
  }
  for (float & vValue : aNorm) {
    vValue = 1.0f + 0.25f * vUniform(vRandom);
  }
  std::string       vError;
  XdnaChainWeights vFixed;
  if (!pRunner.mLoadChainLinear(aConv.data(), aNorm.data(), vFixed, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return false;
  }
  XdnaBuffer vStates;
  if (!pRunner.mCreateShared(cHeads * cHead * sizeof(float) + cRoom, vStates, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    pRunner.mFreeChainWeights(vFixed);
    return false;
  }
  float *             vState = static_cast<float *>(vStates.vMap);
  std::vector<double> aState(cHeads * cHead);
  for (uint64_t i = 0; i < cHeads * cHead; i++) {
    vState[i] = 0.5f * vUniform(vRandom);
    aState[i] = vState[i];
  }
  std::vector<float> aConvStates(uint64_t(cChannels) * (cTaps - 1));
  for (float & vValue : aConvStates) {
    vValue = vUniform(vRandom);
  }
  XdnaChainPhase vPhase;
  vPhase.vRecurrence.vHeads  = cHeads;
  vPhase.vRecurrence.vLinear = 1;
  std::vector<float>  aX(cChannels), aZ(cHeads * cDim), aDecay(cHeads), aBeta(cHeads), aGot(cHeads * cDim);
  std::vector<float>  aInput(uint64_t(cTaps) * cChannels);  // as the CONCAT of the convolution by tokens has it
  std::vector<double> aWant(cHeads * cDim), aConvOut(cChannels);
  XdnaChainLinearIo  vInputs;
  vInputs.vInput    = aInput.data();
  vInputs.vStride   = cChannels;
  vInputs.vZ        = aZ.data();
  vInputs.vDecay    = aDecay.data();
  vInputs.vBeta     = aBeta.data();
  vInputs.vEpsL2    = cEps;
  vInputs.vEpsGated = cEps;
  XdnaChainIo vIo;
  vIo.vLinear  = &vInputs;
  vIo.vOutputs = aGot.data();
  vIo.vScale   = 1.0f;
  const XdnaBuffer *       aStates[1]  = {&vStates};
  const XdnaChainWeights * aWeights[1] = {&vFixed};
  const uint64_t            vFlushed    = vStates.vSize;
  bool                      vOk         = true;
  double                    vWorstOut   = 0;
  double                    vLargestOut = 0;
  std::vector<double>       aNpuMs;
  for (int vToken = 0; vToken < pTokens; vToken++) {
    for (float & vValue : aX) {
      vValue = vUniform(vRandom);
    }
    for (uint32_t c = 0; c < cChannels; c++) {
      for (uint32_t t = 0; t + 1 < cTaps; t++) {
        aInput[uint64_t(t) * cChannels + c] = aConvStates[uint64_t(c) * (cTaps - 1) + t];
      }
      aInput[uint64_t(cTaps - 1) * cChannels + c] = aX[c];
    }
    for (float & vValue : aZ) {
      vValue = 2.0f * vUniform(vRandom);
    }
    for (uint32_t h = 0; h < cHeads; h++) {
      aDecay[h] = std::exp(-3.0f * vUnit(vRandom));  // e^g for a gate g from -3 to 0
      aBeta[h]  = vUnit(vRandom);
    }
    // the reference, on the inputs as the cores get them: the convolution and its SiLU, the L2 norms, the recurrence of
    // every row of every head (ggml_gated_delta_net) and the gated norm
    for (uint32_t c = 0; c < cChannels; c++) {
      double vSum = 0;
      for (uint32_t t = 0; t + 1 < cTaps; t++) {
        vSum += fBf16(aConvStates[uint64_t(c) * (cTaps - 1) + t]) * fBf16(aConv[uint64_t(c) * cTaps + t]);
      }
      vSum += fBf16(aX[c]) * fBf16(aConv[uint64_t(c) * cTaps + cTaps - 1]);
      aConvOut[c] = fSilu(vSum);
    }
    for (uint32_t j = 0; j < 2 * cKeys; j++) {
      double * vAt   = aConvOut.data() + uint64_t(j) * cDim;  // q of the heads of keys, then their k
      double   vNorm = 0;
      for (uint32_t i = 0; i < cDim; i++) {
        vNorm += vAt[i] * vAt[i];
      }
      for (uint32_t i = 0; i < cDim; i++) {
        vAt[i] /= std::sqrt(vNorm + double(cEps));
      }
    }
    for (uint32_t j = 0; j < cHeads; j++) {
      const double * vQ = aConvOut.data() + uint64_t(j % cKeys) * cDim;
      const double * vK = aConvOut.data() + uint64_t(cKeys + j % cKeys) * cDim;
      const double * vV = aConvOut.data() + uint64_t(2 * cKeys + j) * cDim;
      std::vector<double> aOut(cDim);
      for (uint32_t r = 0; r < cDim; r++) {
        double * vRow = aState.data() + j * cHead + uint64_t(r) * cDim;
        double   vDot = 0;
        for (uint32_t i = 0; i < cDim; i++) {
          vRow[i] *= double(aDecay[j]);
          vDot += vRow[i] * vK[i];
        }
        const double vDelta = (vV[r] - vDot) * double(aBeta[j]);
        double       vOut   = 0;
        for (uint32_t i = 0; i < cDim; i++) {
          vRow[i] += vDelta * vK[i];
          vOut += vRow[i] * vQ[i];
        }
        aOut[r] = vOut / std::sqrt(double(cDim));  // as ggml scales q
      }
      double vSquares = 0;
      for (uint32_t r = 0; r < cDim; r++) {
        vSquares += aOut[r] * aOut[r];
      }
      const double vScale = 1.0 / std::sqrt(vSquares / cDim + double(cEps));
      for (uint32_t r = 0; r < cDim; r++) {
        const double vWeight = fBf16(aNorm[r] * std::sqrt(float(cDim))) / std::sqrt(double(cDim));
        aWant[j * cDim + r]  = aOut[r] * vScale * vWeight * fSilu(fBf16(aZ[j * cDim + r]));
      }
    }
    fXdnaFlushCache(vState, 1, vFlushed, vFlushed);
    XdnaGemmTimes vTimes;
    if (!pRunner.mRunChain(&vPhase, &vIo, 1, aWeights, 1, vTimes, vError, aStates, 1)) {
      fprintf(stderr, "error: token %d: %s\n", vToken, vError.c_str());
      pRunner.mFreeShared(vStates);
      pRunner.mFreeChainWeights(vFixed);
      return false;
    }
    fXdnaFlushCache(vState, 1, vFlushed, vFlushed);
    aNpuMs.push_back(vTimes.vNpuMs);
    for (uint64_t i = 0; i < aGot.size(); i++) {
      vLargestOut = std::max(vLargestOut, std::fabs(aWant[i]));
      vWorstOut   = std::max(vWorstOut, std::fabs(double(aGot[i]) - aWant[i]));
    }
    // the states of the convolution move on: [s1, s2, x]
    for (uint32_t c = 0; c < cChannels; c++) {
      float * vAt = aConvStates.data() + uint64_t(c) * (cTaps - 1);
      vAt[0]      = vAt[1];
      vAt[1]      = vAt[2];
      vAt[2]      = aX[c];
    }
  }
  double vWorstState   = 0;
  double vLargestState = 0;
  for (uint64_t i = 0; i < aState.size(); i++) {
    vLargestState = std::max(vLargestState, std::fabs(aState[i]));
    vWorstState   = std::max(vWorstState, std::fabs(double(vState[i]) - aState[i]));
  }
  printf("layer of linear attention of 32 heads of 128 (16 heads of keys, 8192 channels), %d tokens: NPU median %.3f ms "
         "a token\n",
         pTokens, fMedian(aNpuMs));
  printf("outputs: largest error %.3g of the largest value (%.3g)\n", vWorstOut / std::max(vLargestOut, 1e-30),
         vLargestOut);
  printf("states after the last token: largest error %.3g of the largest value (%.3g)\n",
         vWorstState / std::max(vLargestState, 1e-30), vLargestState);
  vOk = vWorstOut <= 0.05 * vLargestOut && vWorstState <= 0.05 * vLargestState;
  aNpuMs.clear();
  for (int vRound = 0; vRound < pRepeat; vRound++) {
    XdnaGemmTimes vTimes;
    if (!pRunner.mRunChain(&vPhase, &vIo, 1, aWeights, 1, vTimes, vError, aStates, 1)) {
      fprintf(stderr, "error: round %d: %s\n", vRound, vError.c_str());
      pRunner.mFreeShared(vStates);
      pRunner.mFreeChainWeights(vFixed);
      return false;
    }
    aNpuMs.push_back(vTimes.vNpuMs);
  }
  printf("timed: NPU median %.3f ms a command (%d rounds)\n", fMedian(aNpuMs), pRepeat);
  pRunner.mFreeShared(vStates);
  pRunner.mFreeChainWeights(vFixed);
  printf("RESULT: %s (tolerance 5 %% of the largest value: the cores compute in bf16)\n", vOk ? "PASS" : "FAIL");
  return vOk;
}

}  // namespace

int main(int pArgc, char ** pArgv) {
  std::string              vDevice = "/dev/accel/accel0";
  std::string              vOverlay, vManifest, vGemvOverlay, vGemvManifest;
  std::vector<std::string> aAlso, aMix;
  uint32_t                 vRows    = 0;
  uint32_t                 vK       = 0;
  uint32_t                 vN       = 0;
  int                      vRepeat  = 5;
  uint32_t                 vSeed    = 20261115;
  uint32_t                 vThreads = 1;
  XdnaGemvFormat          vFormat  = XdnaGemvFormat::cBfp16;
  bool                     vFfn     = false;
  bool                     vGdn     = false;
  bool                     vLa      = false;
  int                      vTokens  = 16;
  bool                     vNextLayer = false;
  bool                     vEarly     = true;
  bool                     vUserFlush = true;
  bool                     vFillsFirst = true;
  float                    vFfnScale = 1.0f;
  float                    vOutliers = 0.0f;
  std::string              vBlocksFrom;
  std::string              vXFrom;
  bool                     vReverse = false;
  for (int i = 1; i < pArgc; i++) {
    const std::string vArg = pArgv[i];
    if (vArg == "--reverse") {
      vReverse = true;
      continue;
    }
    if (vArg == "--ffn") {
      vFfn = true;
      continue;
    }
    if (vArg == "--gdn") {
      vGdn = true;
      continue;
    }
    if (vArg == "--la") {
      vLa = true;
      continue;
    }
    if (vArg == "--next-layer") {
      vNextLayer = true;
      continue;
    }
    if (vArg == "--no-early") {
      vEarly = false;
      continue;
    }
    if (vArg == "--sync-bo") {
      vUserFlush = false;
      continue;
    }
    if (vArg == "--drains-first") {
      vFillsFirst = false;
      continue;
    }
    if (i + 1 >= pArgc) {
      fUsage(pArgv[0]);
      return 2;
    }
    const char * vValue = pArgv[++i];
    if (vArg == "--overlay") {
      vOverlay = vValue;
    } else if (vArg == "--manifest") {
      vManifest = vValue;
    } else if (vArg == "--gemv-overlay") {
      vGemvOverlay = vValue;
    } else if (vArg == "--gemv-manifest") {
      vGemvManifest = vValue;
    } else if (vArg == "--device") {
      vDevice = vValue;
    } else if (vArg == "-M") {
      vRows = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-K") {
      vK = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-N") {
      vN = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--also") {
      aAlso.push_back(vValue);
    } else if (vArg == "--mix") {
      aMix.push_back(vValue);
    } else if (vArg == "--repeat") {
      vRepeat = std::atoi(vValue);
    } else if (vArg == "--tokens") {
      vTokens = std::atoi(vValue);
    } else if (vArg == "--x-from") {
      vXFrom = vValue;
    } else if (vArg == "--blocks-from") {
      vBlocksFrom = vValue;
    } else if (vArg == "--x-outliers") {
      vOutliers = strtof(vValue, nullptr);
    } else if (vArg == "--ffn-scale") {
      vFfnScale = strtof(vValue, nullptr);
    } else if (vArg == "--seed") {
      vSeed = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--threads") {
      vThreads = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--format" && fXdnaGemvFormatFromName(vValue, vFormat)) {
      // taken
    } else {
      fUsage(pArgv[0]);
      return 2;
    }
  }
  if (vOverlay.empty() || vManifest.empty() || vGemvOverlay.empty() || vGemvManifest.empty() || vRepeat < 1 ||
      vTokens < 1 || (!vFfn && !vGdn && !vLa && (vRows == 0 || vK == 0)) || (vNextLayer && !vFfn) ||
      int(vFfn) + int(vGdn) + int(vLa) > 1) {
    fUsage(pArgv[0]);
    return 2;
  }
  if (vFfn || vGdn || vLa) {
    XdnaRunner vRunner;
    std::string vError;
    if (!vRunner.mOpen(vDevice, vOverlay, vManifest, vError, vGemvOverlay, vGemvManifest)) {
      fprintf(stderr, "error: %s\n", vError.c_str());
      return 1;
    }
    if (!vRunner.mHasChain()) {
      fprintf(stderr, "error: --ffn, --gdn and --la need the chained overlay (a manifest of version 9)\n");
      return 2;
    }
    vRunner.mSetThreads(vThreads);
    vRunner.mSetEarlyDescriptors(vEarly);
    vRunner.mSetUserFlush(vUserFlush);
    vRunner.mSetFillsFirst(vFillsFirst);
    if (vGdn) {
      return fRunGdn(vRunner, vTokens, vRepeat, vSeed) ? 0 : 1;
    }
    if (vLa) {
      return fRunLa(vRunner, vTokens, vRepeat, vSeed) ? 0 : 1;
    }
    return fRunFfn(vRunner, vRepeat, vSeed, vFfnScale, vNextLayer) ? 0 : 1;
  }
  std::vector<Case> aCases;
  if (vN > 0) {
    Case vCase;
    vCase.vRows  = vRows;
    vCase.vK     = vK;
    vCase.aParts = {Part{vFormat, vN, {}, {}}};
    aCases.push_back(vCase);
  }
  for (const std::string & vText : aAlso) {
    Case vCase;
    if (!fParseSize(vText.c_str(), vCase, vFormat)) {
      fUsage(pArgv[0]);
      return 2;
    }
    aCases.push_back(vCase);
  }
  for (const std::string & vText : aMix) {
    Case vCase;
    vCase.vRows = vRows;
    vCase.vK    = vK;
    if (!fParseMix(vText, vCase)) {
      fUsage(pArgv[0]);
      return 2;
    }
    aCases.push_back(vCase);
  }
  if (aCases.empty()) {
    fUsage(pArgv[0]);
    return 2;
  }
  for (const Case & vCase : aCases) {
    for (const Part & vPart : vCase.aParts) {
      if (vCase.vRows == 0 || vCase.vK == 0 || vPart.vOutputs == 0) {
        fUsage(pArgv[0]);
        return 2;
      }
      if (vPart.vFormat != XdnaGemvFormat::cBfp16 && vCase.vK % cSuper != 0) {
        fprintf(stderr, "error: %s: in the formats of ggml, K is a multiple of %u\n", fDescribe(vCase).c_str(), cSuper);
        return 2;
      }
    }
  }

  std::mt19937                          vRandom(vSeed);
  std::uniform_real_distribution<float> vUniform(-1.0f, 1.0f);
  for (Case & vCase : aCases) {
    vCase.aX.resize(size_t(vCase.vRows) * vCase.vK);
    vCase.aY.assign(size_t(vCase.vRows) * vCase.mOutputs(), 0.0f);
    // with --x-outliers, the first value of every block of 8 that large, as the massive activations of a model are
    for (size_t vAt = 0; vAt < vCase.aX.size(); vAt++) {
      float & vValue = vCase.aX[vAt];
      vValue = vUniform(vRandom);
      if (vOutliers != 0.0f && vAt % cXdnaBfp16Values == 0) {
        vValue *= vOutliers;
      }
    }
    // with --x-from, the rows of a file (the rows a model multiplied), as many as it has
    if (!vXFrom.empty()) {
      std::ifstream vFile(vXFrom, std::ios::binary);
      vFile.read(reinterpret_cast<char *>(vCase.aX.data()), std::streamsize(vCase.aX.size() * sizeof(float)));
      printf("%lld bytes of X read from %s\n", (long long) vFile.gcount(), vXFrom.c_str());
    }
    for (Part & vPart : vCase.aParts) {
      if (vPart.vFormat == XdnaGemvFormat::cBfp16) {
        vPart.aFloats.resize(size_t(vPart.vOutputs) * vCase.vK);
        for (float & vValue : vPart.aFloats) {
          vValue = vUniform(vRandom);
        }
      } else {
        fRandomBlocks(vPart.vFormat, vPart.vOutputs, vCase.vK, vRandom, vPart.aBlocks);
        // with --blocks-from, the blocks of a file (the rows of a real matrix of a model), as many as it has
        if (!vBlocksFrom.empty()) {
          std::ifstream vFile(vBlocksFrom, std::ios::binary);
          vFile.read(reinterpret_cast<char *>(vPart.aBlocks.data()), std::streamsize(vPart.aBlocks.size()));
          printf("%lld bytes of blocks read from %s\n", (long long) vFile.gcount(), vBlocksFrom.c_str());
        }
      }
    }
  }

  XdnaRunner vRunner;
  std::string vError;
  if (!vRunner.mOpen(vDevice, vOverlay, vManifest, vError, vGemvOverlay, vGemvManifest)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  vRunner.mSetThreads(vThreads);
  vRunner.mSetUserFlush(vUserFlush);
  vRunner.mSetFillsFirst(vFillsFirst);
  for (Case & vCase : aCases) {
    std::vector<XdnaGemvMember> aMembers;
    for (const Part & vPart : vCase.aParts) {
      XdnaGemvMember vMember;
      vMember.vFormat  = vPart.vFormat;
      vMember.vOutputs = vPart.vOutputs;
      const Part *   vFrom    = &vPart;
      const uint32_t vColumns = vCase.vK;
      vMember.fRows = [vFrom, vColumns](uint32_t pFirst, uint32_t pCount, void * pTo) {
        if (vFrom->vFormat == XdnaGemvFormat::cBfp16) {
          std::memcpy(pTo, vFrom->aFloats.data() + size_t(pFirst) * vColumns, size_t(pCount) * vColumns * sizeof(float));
        } else {
          const size_t vRowBytes = size_t(vColumns / cSuper) * fXdnaGemvBlockBytes(vFrom->vFormat);
          std::memcpy(pTo, vFrom->aBlocks.data() + size_t(pFirst) * vRowBytes, size_t(pCount) * vRowBytes);
        }
        return true;
      };
      aMembers.push_back(vMember);
    }
    const bool vLoaded = vRunner.mHasChain() ?
                             vRunner.mLoadChainMatrix(aMembers, vCase.vK, false, vCase.vChainWeights, vError) :
                             vRunner.mLoadGemvMatrix(aMembers, vCase.vK, vCase.vWeights, vError);
    if (!vLoaded) {
      fprintf(stderr, "error: %s: %s\n", fDescribe(vCase).c_str(), vError.c_str());
      return 1;
    }
  }
  for (int vRound = 0; vRound < vRepeat; vRound++) {
    for (size_t vIndex = 0; vIndex < aCases.size(); vIndex++) {
      Case & vCase = aCases[vReverse ? aCases.size() - 1 - vIndex : vIndex];
      XdnaGemmTarget vTarget;
      vTarget.vData   = vCase.aY.data();
      vTarget.vStride = vCase.mOutputs();
      vTarget.vFirst  = 0;
      vTarget.vCount  = vCase.mOutputs();
      XdnaGemmTimes vTimes;
      const bool vDone =
          vRunner.mHasChain() ?
              vRunner.mChainMultiplyRows(vCase.aX.data(), vCase.vRows, vCase.vK, vCase.vChainWeights, &vTarget, 1,
                                         vTimes, vError) :
              vRunner.mMultiplyRows(vCase.aX.data(), vCase.vRows, vCase.vK, vCase.vWeights, &vTarget, 1, vTimes, vError);
      if (!vDone) {
        fprintf(stderr, "error: %s, round %d: %s\n", fDescribe(vCase).c_str(), vRound, vError.c_str());
        return 1;
      }
      vCase.aNpuMs.push_back(vTimes.vNpuMs);
      vCase.aPackMs.push_back(vTimes.vPackMs);
      vCase.aUnpackMs.push_back(vTimes.vUnpackMs);
    }
  }
  bool vOk  = true;
  bool vAny = false;
  for (Case & vCase : aCases) {
    printf("%s: NPU median %.3f ms, converting X %.3f ms, reading Y %.3f ms (%d rounds)\n", fDescribe(vCase).c_str(),
           fMedian(vCase.aNpuMs), fMedian(vCase.aPackMs), fMedian(vCase.aUnpackMs), vRepeat);
    vOk = fCheck(vCase, vRunner.mHasChain() && vCase.vRows == 1) && vOk;
    for (const Part & vPart : vCase.aParts) {
      vAny = vAny || vPart.vFormat != XdnaGemvFormat::cBfp16;
    }
    if (vRunner.mHasChain()) {
      vRunner.mFreeChainWeights(vCase.vChainWeights);
    } else {
      vRunner.mFreeGemvWeights(vCase.vWeights);
    }
  }
  printf("RESULT: %s, tolerance %g of the largest value", vOk ? "PASS" : "FAIL", cTolerance);
  if (vAny) {
    printf(" (%g for 1 output in 1000 in the formats of ggml)", cToleranceRare);
  }
  printf("\n");
  return vOk ? 0 : 1;
}
