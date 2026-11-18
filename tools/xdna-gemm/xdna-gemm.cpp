// llama-xdna-gemm: multiplies two matrices of floats on the NPU and compares the result with the CPU.
//
//   llama-xdna-gemm --overlay <main.pdi> --manifest <file> -M <M> -K <K> -N <N> [--also <M>x<K>x<N>]...
//                    [--repeat <n>] [--check-rows <n>] [--seed <n>] [--rounding floor|nearest] [--device <node>]
//                    [--threads <n>] [--diagnose <prefix> [--fill flushed|dirty|none]] [--keep-awake]
//                    [--no-flush-new]
//       C = A x B with random A (M x K) and B (K x N). It runs on the NPU --repeat times and then checks rows of
//       C against the same product computed on the CPU. Every --also adds a multiplication of other sizes: all of
//       them run in one context, one after the other in every round, with all their weights in the NPU from the
//       start, which is what a model does.
//
//   llama-xdna-gemm pack --manifest <file> -M <M> -K <K> -N <N> --a <file> --bt <file> --out-a <file>
//                    --out-b <file> [--rounding floor|nearest] [--threads <n>]
//       Converts A (M x K floats) and B transposed (N x K floats), read from files, to what the NPU takes, and
//       writes it to files. It does not touch the NPU. It says how long each conversion takes.
//
//   llama-xdna-gemm unpack --manifest <file> -M <M> -K <K> -N <N> --c <file> --out-c <file> [--threads <n>]
//       Turns the result as the NPU writes it, read from a file, into C (M x N floats). It does not touch the NPU.
//
//   llama-xdna-gemm gdn --overlay <main.pdi> --manifest <file> --tokens <n> [--key-heads <n>] [--repeat <n>]
//                    [--seed <n>] [--threads <n>] [--parts <n>] [--device <node>]
//       llama.cpp-xdna: the recurrence of a gated delta net over a batch of tokens (XdnaRunner::mRunGdn, an overlay
//       of the prompts with "gdn 1"), with random inputs like those of Qwen3.5 (16 heads of keys and 32 value heads
//       by default; any number of tokens, which the command takes in groups of 128; its rounds in up to --parts
//       commands in flight, 4 by default), --repeat times from the same states, against the recurrence token by
//       token in 64-bit floats on the same inputs rounded to bf16, as the NPU gets them: the largest error of the
//       outputs and of the final states as a part of their largest value, PASS under 2 %, and the times.
//
// --threads says how many threads share the conversions of the host; 1 if it is not given.
//
// --diagnose <prefix>, for finding faults: the buffer of C is filled with bytes 0xff before every command, and after
// the last run of the first multiplication the tool says where the buffers are for the NPU and which tiles of C the
// NPU wrote, and writes A, B transposed and C (32-bit floats) and the buffer of C as the NPU left it, to files that
// start with <prefix>.
//
// --fill says what the diagnosis does with C before every command: "flushed" (the default) fills it with bytes
// 0xff and flushes them from the cache of the processor; "dirty" fills it and leaves them in the cache, modified, so
// the words that come back as 0xff are also the ones the cache wrote over the results; "none" leaves C as it is.
//
// For experiments with the cache of the processor: --keep-awake keeps one thread spinning while every command runs,
// so the processor does not sleep and its cache is not written to memory meanwhile; --no-flush-new leaves the
// buffers the runner creates without the flush it gives them since 2026/11/13.
//
// The NPU multiplies in block floating point: the reference is the product of the two matrices as the NPU reads
// them, computed in 64-bit floats, so the only difference left is the rounding of the 32-bit sums of the NPU and,
// with an overlay that rounds C (to bfloat16, or to block floating point with every 8 rows of a column in a
// block), the rounding of every sum: then a value may be one step away from the reference rounded the same way,
// when the sum of the NPU falls on the other side of a tie.

#include "xdna-runner.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace {

const double cTolerance     = 1e-5;          // largest error allowed, as a part of the largest value of the reference
const double cToleranceBf16 = 1.0 / 256.0;   // and, with C in bfloat16, as a part of the value itself: one step

double fNowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

void fUsage(const char * pProgram) {
  fprintf(stderr,
          "usage: %s --overlay <main.pdi> --manifest <file> -M <M> -K <K> -N <N> [--also <M>x<K>x<N>]...\n"
          "          [--repeat <n>] [--check-rows <n>] [--seed <n>] [--rounding floor|nearest] [--device <node>]\n"
          "          [--threads <n>] [--diagnose <prefix> [--fill flushed|dirty|none]] [--keep-awake]\n"
          "          [--no-flush-new]\n"
          "       %s pack --manifest <file> -M <M> -K <K> -N <N> --a <file> --bt <file> --out-a <file>\n"
          "          --out-b <file> [--rounding floor|nearest] [--threads <n>]\n"
          "       %s unpack --manifest <file> -M <M> -K <K> -N <N> --c <file> --out-c <file> [--threads <n>]\n"
          "       %s gdn --overlay <main.pdi> --manifest <file> --tokens <n> [--key-heads <n>] [--repeat <n>]\n"
          "          [--seed <n>] [--threads <n>] [--parts <n>] [--device <node>]\n",
          pProgram, pProgram, pProgram, pProgram);
}

struct Options {
  bool          vPackOnly  = false;
  bool          vUnpackOnly = false;
  std::string   vFileC;
  std::string   vOutC;
  std::string   vOverlay;
  std::string   vManifest;
  std::string   vDevice    = "/dev/accel/accel0";
  std::string   vFileA;
  std::string   vFileBt;
  std::string   vOutA;
  std::string   vOutB;
  uint32_t      vM         = 0;
  uint32_t      vK         = 0;
  uint32_t      vN         = 0;
  int           vRepeat    = 1;
  uint32_t      vCheckRows = 16;
  uint32_t      vSeed      = 20261113;
  uint32_t      vThreads   = 1;
  std::string   vDiagnose;  // prefix of the files of the diagnosis; empty: no diagnosis
  std::string   vFill      = "flushed";  // what the diagnosis does with C before every command
  bool          vKeepAwake = false;
  bool          vFlushNew  = true;
  XdnaRounding vRounding  = XdnaRounding::cNearest;
  std::vector<std::array<uint32_t, 3>> aAlso;  // M, K and N of the multiplications after the first one
};

bool fParse(int pArgc, char ** pArgv, Options & pOptions) {
  int vFirst = 1;
  if (pArgc > 1 && strcmp(pArgv[1], "pack") == 0) {
    pOptions.vPackOnly = true;
    pOptions.vRounding = XdnaRounding::cFloor;
    vFirst = 2;
  } else if (pArgc > 1 && strcmp(pArgv[1], "unpack") == 0) {
    pOptions.vUnpackOnly = true;
    vFirst = 2;
  }
  for (int i = vFirst; i < pArgc; i++) {
    const std::string vArg = pArgv[i];
    // the options without a value
    if (vArg == "--keep-awake") {
      pOptions.vKeepAwake = true;
      continue;
    }
    if (vArg == "--no-flush-new") {
      pOptions.vFlushNew = false;
      continue;
    }
    if (i + 1 >= pArgc) {
      return false;
    }
    const char * vValue = pArgv[++i];
    if (vArg == "--overlay") {
      pOptions.vOverlay = vValue;
    } else if (vArg == "--manifest") {
      pOptions.vManifest = vValue;
    } else if (vArg == "--device") {
      pOptions.vDevice = vValue;
    } else if (vArg == "--a") {
      pOptions.vFileA = vValue;
    } else if (vArg == "--bt") {
      pOptions.vFileBt = vValue;
    } else if (vArg == "--out-a") {
      pOptions.vOutA = vValue;
    } else if (vArg == "--out-b") {
      pOptions.vOutB = vValue;
    } else if (vArg == "--c") {
      pOptions.vFileC = vValue;
    } else if (vArg == "--out-c") {
      pOptions.vOutC = vValue;
    } else if (vArg == "-M") {
      pOptions.vM = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-K") {
      pOptions.vK = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-N") {
      pOptions.vN = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--also") {
      std::array<uint32_t, 3> aSizes = {0, 0, 0};
      char                    vRest  = 0;
      if (sscanf(vValue, "%" SCNu32 "x%" SCNu32 "x%" SCNu32 "%c", &aSizes[0], &aSizes[1], &aSizes[2], &vRest) != 3 ||
          aSizes[0] == 0 || aSizes[1] == 0 || aSizes[2] == 0) {
        return false;
      }
      pOptions.aAlso.push_back(aSizes);
    } else if (vArg == "--repeat") {
      pOptions.vRepeat = atoi(vValue);
    } else if (vArg == "--check-rows") {
      pOptions.vCheckRows = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--seed") {
      pOptions.vSeed = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--threads") {
      pOptions.vThreads = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--diagnose") {
      pOptions.vDiagnose = vValue;
    } else if (vArg == "--fill" && (strcmp(vValue, "flushed") == 0 || strcmp(vValue, "dirty") == 0 ||
                                     strcmp(vValue, "none") == 0)) {
      pOptions.vFill = vValue;
    } else if (vArg == "--rounding" && (strcmp(vValue, "floor") == 0 || strcmp(vValue, "nearest") == 0)) {
      pOptions.vRounding = strcmp(vValue, "floor") == 0 ? XdnaRounding::cFloor : XdnaRounding::cNearest;
    } else {
      return false;
    }
  }
  return !pOptions.vManifest.empty() && pOptions.vM > 0 && pOptions.vK > 0 && pOptions.vN > 0 && pOptions.vRepeat >= 1;
}

bool fReadFloats(const std::string & pPath, uint64_t pCount, std::vector<float> & aValues) {
  std::ifstream vFile(pPath, std::ios::binary);
  aValues.resize(pCount);
  vFile.read(reinterpret_cast<char *>(aValues.data()), std::streamsize(pCount * sizeof(float)));
  return bool(vFile) && vFile.peek() == std::ifstream::traits_type::eof();
}

bool fWriteBytes(const std::string & pPath, const std::vector<uint8_t> & aBytes) {
  std::ofstream vFile(pPath, std::ios::binary | std::ios::trunc);
  vFile.write(reinterpret_cast<const char *>(aBytes.data()), std::streamsize(aBytes.size()));
  vFile.close();
  return bool(vFile);
}

int fPack(const Options & pOptions) {
  std::string  vError;
  XdnaOverlay vOverlay;
  if (!vOverlay.mLoad(pOptions.vManifest, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, pOptions.vM, pOptions.vK, pOptions.vN, vShape, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  std::vector<float> aA;
  std::vector<float> aBt;
  if (!fReadFloats(pOptions.vFileA, uint64_t(pOptions.vM) * pOptions.vK, aA) ||
      !fReadFloats(pOptions.vFileBt, uint64_t(pOptions.vN) * pOptions.vK, aBt)) {
    fprintf(stderr, "error: cannot read the matrices: A is M x K floats and B transposed N x K\n");
    return 1;
  }
  std::vector<uint8_t> aPackedA(uint64_t(vShape.vPaddedM) * vShape.vPaddedK / cXdnaBfp16Values * cXdnaBfp16Bytes);
  std::vector<uint8_t> aPackedB(uint64_t(vShape.vPaddedK) * vShape.vPaddedN / cXdnaBfp16Values * cXdnaBfp16Bytes);
  // each conversion several times, to time it: the first one also brings the memory in
  XdnaWorkers vWorkers;
  vWorkers.mSetThreads(pOptions.vThreads);
  double               vBestA = 0;
  double               vBestB = 0;
  std::vector<uint8_t> aFirstA;
  std::vector<uint8_t> aFirstB;
  for (int vRun = 0; vRun < 5; vRun++) {
    const double vStart = fNowMs();
    const bool   vOkA   = fXdnaGemmPackA(vOverlay, vShape, aA.data(), aPackedA.data(), pOptions.vRounding, &vWorkers);
    const double vMiddle = fNowMs();
    const bool   vOkB   = fXdnaGemmPackB(vOverlay, vShape, aBt.data(), aPackedB.data(), pOptions.vRounding, &vWorkers);
    const double vEnd   = fNowMs();
    if (!vOkA || !vOkB) {
      fprintf(stderr, "error: a value of the matrices is not finite\n");
      return 1;
    }
    vBestA = vRun == 0 || vMiddle - vStart < vBestA ? vMiddle - vStart : vBestA;
    vBestB = vRun == 0 || vEnd - vMiddle < vBestB ? vEnd - vMiddle : vBestB;
    // the first conversion is the one that is written; the others have to give the same bytes
    if (vRun == 0) {
      aFirstA = aPackedA;
      aFirstB = aPackedB;
    } else if (aPackedA != aFirstA || aPackedB != aFirstB) {
      fprintf(stderr, "error: conversion %d did not give the bytes of the first one\n", vRun);
      return 1;
    }
    std::fill(aPackedA.begin(), aPackedA.end(), uint8_t(0xee));
    std::fill(aPackedB.begin(), aPackedB.end(), uint8_t(0xee));
  }
  aPackedA.swap(aFirstA);
  aPackedB.swap(aFirstB);
  printf("converting A: %.3f ms (%.3f ns per value); converting B: %.3f ms (%.3f ns per value); %u threads\n", vBestA,
         vBestA * 1e6 / (double(pOptions.vM) * pOptions.vK), vBestB, vBestB * 1e6 / (double(pOptions.vN) * pOptions.vK),
         vWorkers.mThreads());
  if (!fWriteBytes(pOptions.vOutA, aPackedA) || !fWriteBytes(pOptions.vOutB, aPackedB)) {
    fprintf(stderr, "error: cannot write the output files\n");
    return 1;
  }
  printf("sizes: %zu %zu (multiplication of %u x %u x %u)\n", aPackedA.size(), aPackedB.size(), vShape.vPaddedM,
         vShape.vPaddedK, vShape.vPaddedN);
  return 0;
}

int fUnpack(const Options & pOptions) {
  std::string  vError;
  XdnaOverlay vOverlay;
  if (!vOverlay.mLoad(pOptions.vManifest, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, pOptions.vM, pOptions.vK, pOptions.vN, vShape, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  // the buffer of C holds 32-bit floats or bfloat16, as the overlay says: read as floats, it is as long
  const uint64_t     vSums  = uint64_t(vShape.vPaddedM) * vShape.vPaddedN;
  const uint64_t     vBytes = vOverlay.vOutput == XdnaGemmOutput::cBfloat16 ? vSums * 2 :
                              vOverlay.vOutput == XdnaGemmOutput::cBfp16    ? vSums / 8 * 9 :
                                                                               vSums * 4;
  std::vector<float> aBuffer;
  if (!fReadFloats(pOptions.vFileC, (vBytes + 3) / 4, aBuffer)) {
    fprintf(stderr, "error: cannot read the result of the NPU: %u x %u sums, %llu bytes\n", vShape.vPaddedM,
            vShape.vPaddedN, (unsigned long long) vBytes);
    return 1;
  }
  std::vector<float> aC(uint64_t(pOptions.vM) * pOptions.vN, 0.0f);
  XdnaWorkers       vWorkers;
  vWorkers.mSetThreads(pOptions.vThreads);
  double             vBest = 0;
  std::vector<float> aFirst;
  for (int vRun = 0; vRun < 5; vRun++) {
    const double vStart = fNowMs();
    fXdnaGemmUnpackC(vOverlay, vShape, aBuffer.data(), aC.data(), &vWorkers);
    const double vTime = fNowMs() - vStart;
    vBest = vRun == 0 || vTime < vBest ? vTime : vBest;
    // the first reading is the one that is written; the others have to give the same values
    if (vRun == 0) {
      aFirst = aC;
    } else if (std::memcmp(aC.data(), aFirst.data(), aC.size() * sizeof(float)) != 0) {
      fprintf(stderr, "error: reading %d did not give the values of the first one\n", vRun);
      return 1;
    }
    std::fill(aC.begin(), aC.end(), -1.0f);
  }
  aC.swap(aFirst);
  printf("reading C: %.3f ms (%.3f ns per value); %u threads\n", vBest,
         vBest * 1e6 / (double(pOptions.vM) * pOptions.vN), vWorkers.mThreads());
  std::ofstream vFile(pOptions.vOutC, std::ios::binary | std::ios::trunc);
  vFile.write(reinterpret_cast<const char *>(aC.data()), std::streamsize(aC.size() * sizeof(float)));
  vFile.close();
  if (!vFile) {
    fprintf(stderr, "error: cannot write %s\n", pOptions.vOutC.c_str());
    return 1;
  }
  printf("C: %u x %u floats\n", pOptions.vM, pOptions.vN);
  return 0;
}

// what the NPU reads of a row of pCount floats: every block of 8, encoded and decoded
void fAsRead(const float * pRow, uint32_t pCount, XdnaRounding pRounding, float * pOut) {
  for (uint32_t vAt = 0; vAt < pCount; vAt += cXdnaBfp16Values) {
    float   aValues[cXdnaBfp16Values] = {};
    float   aDecoded[cXdnaBfp16Values];
    uint8_t aBlock[cXdnaBfp16Bytes];
    const uint32_t vHere = std::min(cXdnaBfp16Values, pCount - vAt);
    std::memcpy(aValues, pRow + vAt, vHere * sizeof(float));
    fXdnaBfp16Encode(aValues, aBlock, pRounding);
    fXdnaBfp16Decode(aBlock, aDecoded);
    std::memcpy(pOut + vAt, aDecoded, vHere * sizeof(float));
  }
}

// one multiplication of a run: its operands, its weights in the NPU and its result
struct Case {
  uint32_t            vM = 0;
  uint32_t            vK = 0;
  uint32_t            vN = 0;
  std::vector<float>  aA;
  std::vector<float>  aBt;
  std::vector<float>  aC;
  XdnaWeights        vWeights;
  std::vector<double> aNpuMs;
};

// Checks some rows of C, spread over A, against the product of the operands as the NPU reads them, rounded as the
// overlay gives C back. With C in block floating point, the rows go in groups of 8 that share their exponents.
bool fCheck(const Case & pCase, const Options & pOptions, XdnaGemmOutput pOutput) {
  const bool     vBf16  = pOutput == XdnaGemmOutput::cBfloat16;
  const bool     vBfp16 = pOutput == XdnaGemmOutput::cBfp16;
  const uint32_t vM     = pCase.vM;
  const uint32_t vK     = pCase.vK;
  const uint32_t vN     = pCase.vN;

  // the rows to check: spread over A, or, in block floating point, groups of 8 that start at multiples of 8
  std::vector<uint32_t> aRows;
  if (vBfp16) {
    const uint32_t vGroups  = (vM + cXdnaBfp16Values - 1) / cXdnaBfp16Values;
    const uint32_t vWanted  = std::min(std::max((pOptions.vCheckRows + cXdnaBfp16Values - 1) / cXdnaBfp16Values, 1u),
                                       vGroups);
    for (uint32_t i = 0; i < vWanted; i++) {
      const uint32_t vGroup = vWanted == 1 ? 0 : uint32_t(uint64_t(i) * (vGroups - 1) / (vWanted - 1));
      for (uint32_t vRow = vGroup * cXdnaBfp16Values; vRow < std::min(vM, (vGroup + 1) * cXdnaBfp16Values); vRow++) {
        aRows.push_back(vRow);
      }
    }
  } else {
    const uint32_t vRows = std::min(std::max(pOptions.vCheckRows, 1u), vM);
    for (uint32_t i = 0; i < vRows; i++) {
      aRows.push_back(vRows == 1 ? 0 : uint32_t(uint64_t(i) * (vM - 1) / (vRows - 1)));
    }
  }

  std::vector<float> aReadBt(uint64_t(vN) * vK);
  std::vector<float> aReadRow(vK);
  for (uint32_t vOutput = 0; vOutput < vN; vOutput++) {
    fAsRead(pCase.aBt.data() + uint64_t(vOutput) * vK, vK, pOptions.vRounding, aReadBt.data() + uint64_t(vOutput) * vK);
  }
  // the reference of every value checked, row by row
  std::vector<double> aReference(aRows.size() * uint64_t(vN));
  double              vScale = 0, vExactSquares = 0, vReferenceSquares = 0;
  for (size_t i = 0; i < aRows.size(); i++) {
    const uint32_t vRow = aRows[i];
    fAsRead(pCase.aA.data() + uint64_t(vRow) * vK, vK, pOptions.vRounding, aReadRow.data());
    for (uint32_t vOutput = 0; vOutput < vN; vOutput++) {
      const float * vWeightsRow = aReadBt.data() + uint64_t(vOutput) * vK;
      const float * vExactRow   = pCase.aBt.data() + uint64_t(vOutput) * vK;
      double        vReference  = 0;
      double        vExact      = 0;  // the product of the floats themselves: what the conversion loses
      for (uint32_t k = 0; k < vK; k++) {
        vReference += double(aReadRow[k]) * double(vWeightsRow[k]);
        vExact     += double(pCase.aA[uint64_t(vRow) * vK + k]) * double(vExactRow[k]);
      }
      aReference[i * vN + vOutput] = vReference;
      vScale            = std::max(vScale, std::fabs(vReference));
      vExactSquares     += (vReference - vExact) * (vReference - vExact);
      vReferenceSquares += vExact * vExact;
    }
  }

  // Rounded, every value may be off by one step of its rounding besides the error of the sums, which is measured
  // against the largest value as without it: a step of bfloat16 of the value, or a step of the block of 8 rows,
  // which is set by its largest value.
  double   vWorst = 0, vSquares = 0;
  uint64_t vChecked = 0, vEqual = 0, vNonZero = 0, vNonFinite = 0, vOutside = 0;
  for (size_t i = 0; i < aRows.size(); i++) {
    const uint32_t vRow = aRows[i];
    for (uint32_t vOutput = 0; vOutput < vN; vOutput++) {
      const double vReference = aReference[i * vN + vOutput];
      const double vValue     = double(pCase.aC[uint64_t(vRow) * vN + vOutput]);
      const double vError2    = std::fabs(vValue - vReference);
      double       vRounded   = vReference;  // the reference as the overlay gives it back
      double       vStep      = 0;
      if (vBf16) {
        vRounded = double(fXdnaBf16ToFloat(fXdnaBf16FromFloat(float(vReference))));
        vStep    = cToleranceBf16 * std::fabs(vReference);
      } else if (vBfp16) {
        // the block of the value: the rows of its group, which follow one another in aRows
        const size_t vFirst = i - (vRow % cXdnaBfp16Values);
        float        aValues[cXdnaBfp16Values] = {};
        double       vLargest = 0;
        for (size_t j = vFirst; j < aRows.size() && j < vFirst + cXdnaBfp16Values &&
                                aRows[j] / cXdnaBfp16Values == vRow / cXdnaBfp16Values; j++) {
          aValues[j - vFirst] = float(aReference[j * vN + vOutput]);
          vLargest            = std::max(vLargest, std::fabs(aReference[j * vN + vOutput]));
        }
        uint8_t aBlock[cXdnaBfp16Bytes];
        fXdnaBfp16Encode(aValues, aBlock, XdnaRounding::cNearest);
        fXdnaBfp16Decode(aBlock, aValues);
        vRounded = double(aValues[i - vFirst]);
        vStep    = vLargest / 64;  // the largest value has a mantissa of 64 to 127
      }
      // a value that is not finite would not count as an error below: comparisons with it are false
      vNonFinite += std::isfinite(vValue) ? 0 : 1;
      vOutside   += (vBf16 || vBfp16) && !(vError2 - vStep <= cTolerance * vScale) ? 1 : 0;
      vWorst      = std::max(vWorst, vError2);
      vSquares   += vError2 * vError2;
      vChecked++;
      vEqual   += vValue == vRounded ? 1 : 0;
      vNonZero += vValue != 0 ? 1 : 0;
    }
  }
  const bool vOk = vScale > 0 && vNonZero > 0 && vNonFinite == 0 &&
                   (vBf16 || vBfp16 ? vOutside == 0 : vWorst <= cTolerance * vScale);
  const char * vRoundedTo = vBf16 ? " rounded to bfloat16" : vBfp16 ? " rounded to block floating point" : "";
  printf("%u x %u x %u: checked %zu rows of C (%" PRIu64 " values, %" PRIu64 " not zero, %" PRIu64
         " not finite): %" PRIu64 " equal to the reference%s\n",
         vM, vK, vN, aRows.size(), vChecked, vNonZero, vNonFinite, vEqual, vRoundedTo);
  if (vBf16 || vBfp16) {
    printf("  values more than one step of the rounding away from the reference: %" PRIu64 "\n", vOutside);
  }
  printf("  largest value of the reference: %.6g; largest error: %.3g (%.3g of the largest value); rms error: %.3g\n",
         vScale, vWorst, vScale > 0 ? vWorst / vScale : 0.0, std::sqrt(vSquares / double(vChecked)));
  printf("  what the conversion to block floating point loses: %.3g of the rms of the exact product: %s\n",
         vReferenceSquares > 0 ? std::sqrt(vExactSquares / vReferenceSquares) : 0.0, vOk ? "pass" : "FAIL");
  return vOk;
}

bool fWriteFile(const std::string & pPath, const void * pData, uint64_t pBytes) {
  std::ofstream vFile(pPath, std::ios::binary | std::ios::trunc);
  vFile.write(static_cast<const char *>(pData), std::streamsize(pBytes));
  vFile.close();
  return bool(vFile);
}

// Where a buffer is for the NPU, and whether it crosses a boundary of 4 GiB of addresses.
void fPrintAddress(const char * pName, uint64_t pAddress, uint64_t pBytes) {
  const uint64_t vLast = pAddress + (pBytes > 0 ? pBytes - 1 : 0);
  printf("  %-8s 0x%012" PRIx64 " to 0x%012" PRIx64 " (%" PRIu64 " bytes)", pName, pAddress, vLast + 1, pBytes);
  if ((pAddress >> 32) != (vLast >> 32)) {
    const uint64_t vBoundary = ((pAddress | uint64_t(0xffffffffu)) + 1) - pAddress;
    printf(", crosses a boundary of 4 GiB %" PRIu64 " bytes from its start", vBoundary);
  }
  printf("\n");
}

// The diagnosis of a multiplication, run with the buffer of C filled with bytes 0xff before the command: where its
// buffers are for the NPU, which tiles of C the NPU wrote, and files with its operands and its results.
bool fDiagnose(const Options & pOptions, XdnaRunner & pRunner, const Case & pCase) {
  const XdnaOverlay & vOverlay = pRunner.mOverlay();
  std::string          vError;
  XdnaGemmShape       vShape;
  if (!fXdnaGemmShape(vOverlay, pCase.vM, pCase.vK, pCase.vN, vShape, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return false;
  }
  uint64_t        vBytes = 0;
  const uint8_t * vRaw   = pRunner.mRawC(vBytes);
  printf("diagnosis of %u x %u x %u (on the NPU %u x %u x %u):\n", pCase.vM, pCase.vK, pCase.vN, vShape.vPaddedM,
         vShape.vPaddedK, vShape.vPaddedN);
  fPrintAddress("A", pRunner.mDevAddrA(),
                uint64_t(vShape.vPaddedM) * vShape.vPaddedK / cXdnaBfp16Values * cXdnaBfp16Bytes);
  fPrintAddress("weights", pCase.vWeights.vBuffer.vDevAddr, pCase.vWeights.vBuffer.vSize);
  fPrintAddress("C", pRunner.mDevAddrC(), vBytes);

  // C as the array writes it: [columns of cores][row blocks][turns][rows of cores][tile]
  const bool     vByColumns = vOverlay.vLayout == XdnaGemmLayout::cColumns;
  const uint32_t vStreamsA  = vByColumns ? vOverlay.vCoreColumns : vOverlay.vCoreRows;
  const uint32_t vStreamsB  = vByColumns ? vOverlay.vCoreRows : vOverlay.vCoreColumns;
  const uint32_t vBlocks    = vShape.vPaddedM / (vOverlay.vTileM * vStreamsA);
  const uint32_t vTurns     = vShape.vPaddedN / (vOverlay.vTileN * vStreamsB);
  const uint64_t vTileWords = uint64_t(vOverlay.vTileM) * vOverlay.vTileN;
  const uint64_t vTiles     = uint64_t(vOverlay.vCoreColumns) * vBlocks * vTurns * vOverlay.vCoreRows;
  // a word is a sum: a 32-bit float, or a bfloat16
  if (vOverlay.vOutput == XdnaGemmOutput::cBfp16) {
    fprintf(stderr, "error: the diagnosis reads C in 32-bit floats or in bfloat16 only\n");
    return false;
  }
  const bool     vBf16      = vOverlay.vOutput == XdnaGemmOutput::cBfloat16;
  const uint32_t vWordBytes = vBf16 ? 2 : 4;
  const uint32_t vFilled    = vBf16 ? 0xffffu : 0xffffffffu;
  const uint32_t vMagnitude = vBf16 ? 0x7fffu : 0x7fffffffu;
  if (vRaw == nullptr || vBytes != vTiles * vTileWords * vWordBytes) {
    fprintf(stderr, "error: the buffer of C does not have the size of the multiplication\n");
    return false;
  }
  auto fWord = [&](uint64_t pAt) {
    uint32_t vWord = 0;
    std::memcpy(&vWord, vRaw + pAt * vWordBytes, vWordBytes);
    return vWord;
  };
  std::vector<char> aMap(vTiles);
  uint64_t vUnwritten = 0, vZeros = 0, vFirstUnwritten = UINT64_MAX, vLastUnwritten = 0, vFirstZero = UINT64_MAX;
  uint64_t vLastZero = 0;
  for (uint64_t vTile = 0; vTile < vTiles; vTile++) {
    uint64_t vTileUnwritten = 0, vTileZeros = 0;
    for (uint64_t i = 0; i < vTileWords; i++) {
      const uint64_t vAt   = vTile * vTileWords + i;
      const uint32_t vWord = fWord(vAt);
      if (vWord == vFilled) {
        vTileUnwritten++;
        vFirstUnwritten = std::min(vFirstUnwritten, vAt);
        vLastUnwritten  = vAt;
      } else if ((vWord & vMagnitude) == 0) {
        vTileZeros++;
        vFirstZero = std::min(vFirstZero, vAt);
        vLastZero  = vAt;
      }
    }
    vUnwritten += vTileUnwritten;
    vZeros     += vTileZeros;
    aMap[vTile] = vTileUnwritten == vTileWords ? 'U' :
                  vTileZeros == vTileWords     ? 'Z' :
                  vTileUnwritten == 0 && vTileZeros == 0 ? '.' : 'm';
  }
  printf("  words of C: %" PRIu64 " of %u bytes; still 0xff: %" PRIu64 "; zero: %" PRIu64 "\n", vBytes / vWordBytes,
         vWordBytes, vUnwritten, vZeros);
  if (vUnwritten > 0) {
    printf("  still 0xff: bytes %" PRIu64 " to %" PRIu64 " of C\n", vFirstUnwritten * vWordBytes,
           (vLastUnwritten + 1) * vWordBytes);
  }
  if (vZeros > 0) {
    printf("  zero: bytes %" PRIu64 " to %" PRIu64 " of C\n", vFirstZero * vWordBytes, (vLastZero + 1) * vWordBytes);
  }
  printf("  tiles, one character per turn ('.' written, 'U' still 0xff, 'Z' zeros, 'm' mixed):\n");
  if (uint64_t(vOverlay.vCoreColumns) * vBlocks * vOverlay.vCoreRows <= 64) {
    for (uint32_t vColumn = 0; vColumn < vOverlay.vCoreColumns; vColumn++) {
      for (uint32_t vBlock = 0; vBlock < vBlocks; vBlock++) {
        for (uint32_t vRow = 0; vRow < vOverlay.vCoreRows; vRow++) {
          std::string vLine;
          for (uint32_t vTurn = 0; vTurn < vTurns; vTurn++) {
            vLine += aMap[((uint64_t(vColumn) * vBlocks + vBlock) * vTurns + vTurn) * vOverlay.vCoreRows + vRow];
          }
          printf("  column %u block %u row %u: %s\n", vColumn, vBlock, vRow, vLine.c_str());
        }
      }
    }
  } else {
    printf("  (too many to print: see the files)\n");
  }

  const std::string & vPrefix = pOptions.vDiagnose;
  if (!fWriteFile(vPrefix + "-a.f32", pCase.aA.data(), pCase.aA.size() * sizeof(float)) ||
      !fWriteFile(vPrefix + "-bt.f32", pCase.aBt.data(), pCase.aBt.size() * sizeof(float)) ||
      !fWriteFile(vPrefix + "-c.f32", pCase.aC.data(), pCase.aC.size() * sizeof(float)) ||
      !fWriteFile(vPrefix + "-craw.bin", vRaw, vBytes)) {
    fprintf(stderr, "error: cannot write the files of the diagnosis\n");
    return false;
  }
  printf("  files: %s-a.f32, %s-bt.f32, %s-c.f32, %s-craw.bin\n", vPrefix.c_str(), vPrefix.c_str(), vPrefix.c_str(),
         vPrefix.c_str());
  return true;
}

int fRun(const Options & pOptions) {
  // the multiplications: the one of -M, -K and -N and then the ones of --also
  std::vector<Case> aCases(1 + pOptions.aAlso.size());
  aCases[0].vM = pOptions.vM;
  aCases[0].vK = pOptions.vK;
  aCases[0].vN = pOptions.vN;
  for (size_t i = 0; i < pOptions.aAlso.size(); i++) {
    aCases[1 + i].vM = pOptions.aAlso[i][0];
    aCases[1 + i].vK = pOptions.aAlso[i][1];
    aCases[1 + i].vN = pOptions.aAlso[i][2];
  }
  for (size_t i = 0; i < aCases.size(); i++) {
    // A, and B transposed: one row of K values for every output
    Case &                                vCase = aCases[i];
    std::mt19937                          vRandom(pOptions.vSeed + uint32_t(i));
    std::uniform_real_distribution<float> vUniform(-1.0f, 1.0f);
    vCase.aA.resize(uint64_t(vCase.vM) * vCase.vK);
    vCase.aBt.resize(uint64_t(vCase.vN) * vCase.vK);
    vCase.aC.assign(uint64_t(vCase.vM) * vCase.vN, 0.0f);
    for (float & vValue : vCase.aA) {
      vValue = vUniform(vRandom);
    }
    for (float & vValue : vCase.aBt) {
      vValue = vUniform(vRandom);
    }
  }

  std::string vError;
  XdnaRunner vRunner;
  if (!vRunner.mOpen(pOptions.vDevice, pOptions.vOverlay, pOptions.vManifest, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  vRunner.mSetThreads(pOptions.vThreads);
  vRunner.mSetFillC(!pOptions.vDiagnose.empty() && pOptions.vFill != "none", pOptions.vFill != "dirty");
  vRunner.mSetFlushNewBuffers(pOptions.vFlushNew);
  // every matrix of weights is in the NPU before the first multiplication, as the weights of a model are
  for (Case & vCase : aCases) {
    XdnaGemmShape vShape;
    if (!fXdnaGemmShape(vRunner.mOverlay(), vCase.vM, vCase.vK, vCase.vN, vShape, vError)) {
      fprintf(stderr, "error: %s\n", vError.c_str());
      return 1;
    }
    printf("multiplication: %u x %u x %u; on the NPU: %u x %u x %u\n", vCase.vM, vCase.vK, vCase.vN, vShape.vPaddedM,
           vShape.vPaddedK, vShape.vPaddedN);
    if (!vRunner.mLoadWeights(vCase.aBt.data(), vCase.vN, vCase.vK, pOptions.vRounding, vCase.vWeights, vError)) {
      fprintf(stderr, "error: %s\n", vError.c_str());
      return 1;
    }
  }

  XdnaGemmTimes vTimes;
  for (int vRun = 0; vRun < pOptions.vRepeat; vRun++) {
    for (Case & vCase : aCases) {
      std::fill(vCase.aC.begin(), vCase.aC.end(), 0.0f);
      // with --keep-awake, a thread spins while the command runs
      std::atomic<bool> vAwake(pOptions.vKeepAwake);
      std::thread       vSpinner;
      if (pOptions.vKeepAwake) {
        vSpinner = std::thread([&vAwake] {
          while (vAwake.load(std::memory_order_relaxed)) {
          }
        });
      }
      const bool vMultiplied = vRunner.mMultiply(vCase.aA.data(), vCase.vM, vCase.vWeights, pOptions.vRounding,
                                                 vCase.aC.data(), vTimes, vError);
      vAwake = false;
      if (vSpinner.joinable()) {
        vSpinner.join();
      }
      if (!vMultiplied) {
        fprintf(stderr, "error: run %d, %u x %u x %u: %s\n", vRun, vCase.vM, vCase.vK, vCase.vN, vError.c_str());
        return 1;
      }
      vCase.aNpuMs.push_back(vTimes.vNpuMs);
      // the diagnosis looks at the buffer of C before the next multiplication writes it
      if (!pOptions.vDiagnose.empty() && vRun == pOptions.vRepeat - 1 && &vCase == &aCases[0] &&
          !fDiagnose(pOptions, vRunner, vCase)) {
        return 1;
      }
      if (vRun < 4) {
        printf("run %d, %u x %u x %u: plan %.3f ms, converting A %.3f ms, NPU %.3f ms, reading C %.3f ms\n", vRun,
               vCase.vM, vCase.vK, vCase.vN, vTimes.vPlanMs, vTimes.vPackMs, vTimes.vNpuMs, vTimes.vUnpackMs);
      }
    }
  }
  for (Case & vCase : aCases) {
    std::sort(vCase.aNpuMs.begin(), vCase.aNpuMs.end());
    printf("%u x %u x %u: %zu runs completed; NPU: fastest %.3f ms, median %.3f ms, slowest %.3f ms\n", vCase.vM,
           vCase.vK, vCase.vN, vCase.aNpuMs.size(), vCase.aNpuMs.front(), vCase.aNpuMs[vCase.aNpuMs.size() / 2],
           vCase.aNpuMs.back());
    vRunner.mFreeWeights(vCase.vWeights);
  }
  vRunner.mClose();

  bool vOk = true;
  for (const Case & vCase : aCases) {
    vOk = fCheck(vCase, pOptions, vRunner.mOverlay().vOutput) && vOk;
  }
  if (vRunner.mOverlay().vOutput == XdnaGemmOutput::cBfloat16) {
    printf("RESULT: %s, tolerance one step of bfloat16 of every value and %g of the largest value\n",
           vOk ? "PASS" : "FAIL", cTolerance);
  } else if (vRunner.mOverlay().vOutput == XdnaGemmOutput::cBfp16) {
    printf("RESULT: %s, tolerance one step of the block of every value and %g of the largest value\n",
           vOk ? "PASS" : "FAIL", cTolerance);
  } else {
    printf("RESULT: %s, tolerance %g of the largest value\n", vOk ? "PASS" : "FAIL", cTolerance);
  }
  return vOk ? 0 : 1;
}

}  // namespace

// llama.cpp-xdna: the recurrence of a gated delta net (see the top of this file)
static int fGdn(int pArgc, char ** pArgv) {
  std::string vOverlay;
  std::string vManifest;
  std::string vDevice   = "/dev/accel/accel0";
  uint32_t    vTokens   = 0;
  uint32_t    vKeyHeads = 16;
  int         vRepeat   = 1;
  uint32_t    vSeed     = 20261116;
  uint32_t    vThreads  = 1;
  uint32_t    vParts    = cXdnaRunnerSlotsC;
  for (int i = 2; i + 1 < pArgc; i += 2) {
    const std::string vArg   = pArgv[i];
    const char *      vValue = pArgv[i + 1];
    if (vArg == "--overlay") {
      vOverlay = vValue;
    } else if (vArg == "--manifest") {
      vManifest = vValue;
    } else if (vArg == "--device") {
      vDevice = vValue;
    } else if (vArg == "--tokens") {
      vTokens = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--key-heads") {
      vKeyHeads = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--repeat") {
      vRepeat = atoi(vValue);
    } else if (vArg == "--seed") {
      vSeed = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--threads") {
      vThreads = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--parts") {
      vParts = uint32_t(strtoul(vValue, nullptr, 10));
    } else {
      fprintf(stderr, "error: unknown option %s\n", vArg.c_str());
      return 2;
    }
  }
  if (vOverlay.empty() || vManifest.empty() || vTokens == 0 || vRepeat < 1) {
    fUsage(pArgv[0]);
    return 2;
  }
  const uint32_t vValueHeads = 2 * vKeyHeads;
  const uint32_t vDim        = cXdnaGdnDim;
  // the inputs: q and k of norm 1, v normal, beta a sigmoid, the gates from long memory to short, random states
  std::mt19937                     vRandom(vSeed);
  std::normal_distribution<float>  vNormal(0.0f, 1.0f);
  std::vector<float>               aQ(size_t(vTokens) * vKeyHeads * vDim);
  std::vector<float>               aK(aQ.size());
  std::vector<float>               aV(size_t(vTokens) * vValueHeads * vDim);
  std::vector<float>               aG(size_t(vTokens) * vValueHeads);
  std::vector<float>               aBeta(aG.size());
  std::vector<float>               aStateIn(size_t(vValueHeads) * vDim * vDim);
  std::vector<float>               aStateOut(aStateIn.size());
  std::vector<float>               aOut(aV.size());
  for (std::vector<float> * vVector : {&aQ, &aK}) {
    for (size_t vRow = 0; vRow < vVector->size() / vDim; vRow++) {
      double vNorm = 0;
      for (uint32_t vD = 0; vD < vDim; vD++) {
        const float vValue          = vNormal(vRandom);
        (*vVector)[vRow * vDim + vD] = vValue;
        vNorm += double(vValue) * vValue;
      }
      for (uint32_t vD = 0; vD < vDim; vD++) {
        (*vVector)[vRow * vDim + vD] = float((*vVector)[vRow * vDim + vD] / std::sqrt(vNorm));
      }
    }
  }
  for (float & vValue : aV) {
    vValue = vNormal(vRandom);
  }
  for (size_t i = 0; i < aG.size(); i++) {
    const uint32_t vHead = uint32_t(i % vValueHeads);
    const double   vRate = std::exp(-6.0 + 7.5 * vHead / std::max(1u, vValueHeads - 1));
    aG[i]                = float(-vRate * std::log1p(std::exp(double(vNormal(vRandom)) - 2.0)));
    aBeta[i]             = float(1.0 / (1.0 + std::exp(-double(vNormal(vRandom)))));
  }
  for (float & vValue : aStateIn) {
    vValue = 0.1f * vNormal(vRandom);
  }

  XdnaGdnIo vIo;
  vIo.vQ          = aQ.data();
  vIo.vQHead      = vDim;
  vIo.vQToken     = size_t(vKeyHeads) * vDim;
  vIo.vK          = aK.data();
  vIo.vKHead      = vDim;
  vIo.vKToken     = size_t(vKeyHeads) * vDim;
  vIo.vV          = aV.data();
  vIo.vVHead      = vDim;
  vIo.vVToken     = size_t(vValueHeads) * vDim;
  vIo.vG          = aG.data();
  vIo.vBeta       = aBeta.data();
  vIo.vStateIn    = aStateIn.data();
  vIo.vStateOut   = aStateOut.data();
  vIo.vOut        = aOut.data();
  vIo.vTokens     = vTokens;
  vIo.vKeyHeads   = vKeyHeads;
  vIo.vValueHeads = vValueHeads;
  vIo.vScale      = 1.0f / std::sqrt(float(vDim));

  std::string vError;
  XdnaRunner vRunner;
  if (!vRunner.mOpen(vDevice, vOverlay, vManifest, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  vRunner.mSetThreads(vThreads);
  vRunner.mSetGdnParts(vParts);
  if (!vRunner.mCanRunGdn()) {
    fprintf(stderr, "error: the manifest of this overlay does not say gdn 1\n");
    return 1;
  }
  std::vector<double> aMs;
  XdnaGemmTimes      vTimes;
  for (int vRun = 0; vRun < vRepeat; vRun++) {
    XdnaGemmTimes vOne;
    const double   vStart = fNowMs();
    if (!vRunner.mRunGdn(vIo, vOne, vError)) {
      fprintf(stderr, "error: %s\n", vError.c_str());
      return 1;
    }
    aMs.push_back(fNowMs() - vStart);
    if (vRun > 0) {
      vTimes.vPackMs += vOne.vPackMs;
      vTimes.vNpuMs += vOne.vNpuMs;
      vTimes.vUnpackMs += vOne.vUnpackMs;
    }
  }

  // the reference, token by token in 64-bit floats on the inputs as the NPU gets them (bf16, q scaled)
  auto fAsBf16 = [](float pValue) { return double(fXdnaBf16ToFloat(fXdnaBf16FromFloat(pValue))); };
  double vErrOut = 0, vMaxOut = 0, vErrState = 0, vMaxState = 0;
  std::vector<double> aS(size_t(vDim) * vDim);
  for (uint32_t vHead = 0; vHead < vValueHeads; vHead++) {
    const uint32_t vKey = vHead % vKeyHeads;
    for (size_t i = 0; i < aS.size(); i++) {
      aS[i] = aStateIn[size_t(vHead) * vDim * vDim + i];
    }
    std::vector<double> aDelta(vDim);
    for (uint32_t vT = 0; vT < vTokens; vT++) {
      const float * vQ = aQ.data() + (size_t(vT) * vKeyHeads + vKey) * vDim;
      const float * vK = aK.data() + (size_t(vT) * vKeyHeads + vKey) * vDim;
      const float * vV = aV.data() + (size_t(vT) * vValueHeads + vHead) * vDim;
      const double  vDecay = std::exp(double(aG[size_t(vT) * vValueHeads + vHead]));
      const double  vBeta  = aBeta[size_t(vT) * vValueHeads + vHead];
      for (uint32_t vJ = 0; vJ < vDim; vJ++) {
        double vDot = 0;
        for (uint32_t vD = 0; vD < vDim; vD++) {
          aS[size_t(vJ) * vDim + vD] *= vDecay;
          vDot += aS[size_t(vJ) * vDim + vD] * fAsBf16(vK[vD]);
        }
        aDelta[vJ] = vBeta * (fAsBf16(vV[vJ]) - vDot);
      }
      for (uint32_t vJ = 0; vJ < vDim; vJ++) {
        double vO = 0;
        for (uint32_t vD = 0; vD < vDim; vD++) {
          aS[size_t(vJ) * vDim + vD] += aDelta[vJ] * fAsBf16(vK[vD]);
          vO += aS[size_t(vJ) * vDim + vD] * fAsBf16(vQ[vD] * vIo.vScale);
        }
        const double vGot = aOut[(size_t(vT) * vValueHeads + vHead) * vDim + vJ];
        vErrOut = std::max(vErrOut, std::fabs(vGot - vO));
        vMaxOut = std::max(vMaxOut, std::fabs(vO));
      }
    }
    for (size_t i = 0; i < aS.size(); i++) {
      vErrState = std::max(vErrState, std::fabs(double(aStateOut[size_t(vHead) * vDim * vDim + i]) - aS[i]));
      vMaxState = std::max(vMaxState, std::fabs(aS[i]));
    }
  }
  std::sort(aMs.begin(), aMs.end());
  const double vRelOut   = vErrOut / std::max(vMaxOut, 1e-30);
  const double vRelState = vErrState / std::max(vMaxState, 1e-30);
  printf("recurrence: %u tokens, %u heads of keys, %u value heads; rounds of 2 heads of keys: %u\n", vTokens, vKeyHeads,
         vValueHeads, vKeyHeads / 2);
  printf("largest error: outputs %.3g, states %.3g (of their largest values %.3g and %.3g)\n", vRelOut, vRelState,
         vMaxOut, vMaxState);
  printf("time of a call: fastest %.3f ms, median %.3f ms (the first, %.3f ms, writes the instruction stream)\n",
         aMs.size() > 1 ? *std::min_element(aMs.begin() + 1, aMs.end()) : aMs[0], aMs[aMs.size() / 2], aMs[0]);
  if (vRepeat > 1) {
    printf("of it, per call after the first: writing A and B %.3f ms, the NPU %.3f ms, reading C %.3f ms\n",
           vTimes.vPackMs / (vRepeat - 1), vTimes.vNpuMs / (vRepeat - 1), vTimes.vUnpackMs / (vRepeat - 1));
  }
  const bool vPass = vRelOut < 0.02 && vRelState < 0.02;
  printf("%s\n", vPass ? "PASS" : "FAIL");
  return vPass ? 0 : 1;
}

int main(int pArgc, char ** pArgv) {
  if (pArgc > 1 && strcmp(pArgv[1], "gdn") == 0) {
    return fGdn(pArgc, pArgv);
  }
  Options vOptions;
  if (!fParse(pArgc, pArgv, vOptions)) {
    fUsage(pArgv[0]);
    return 2;
  }
  if (vOptions.vPackOnly) {
    if (vOptions.vFileA.empty() || vOptions.vFileBt.empty() || vOptions.vOutA.empty() || vOptions.vOutB.empty()) {
      fUsage(pArgv[0]);
      return 2;
    }
    return fPack(vOptions);
  }
  if (vOptions.vUnpackOnly) {
    if (vOptions.vFileC.empty() || vOptions.vOutC.empty()) {
      fUsage(pArgv[0]);
      return 2;
    }
    return fUnpack(vOptions);
  }
  if (vOptions.vOverlay.empty()) {
    fUsage(pArgv[0]);
    return 2;
  }
  return fRun(vOptions);
}
