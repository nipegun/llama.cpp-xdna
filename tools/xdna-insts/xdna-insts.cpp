// llama-xdna-insts: writes and reads instruction streams of the NPU.
//
//   llama-xdna-insts gemm --overlay <manifest> -M <M> -K <K> -N <N> --out <file>
//       writes the instruction stream of a matrix multiplication of M x K by K x N on an overlay of this project,
//       and prints the bytes of its three buffers
//   llama-xdna-insts gdn --overlay <manifest> --rounds <R> --groups <G> --out <file>
//       writes the instruction stream of the recurrence of a gated delta net of R rounds of G groups of chunks on an
//       overlay of the prompts with the recurrence (fXdnaGdnPlan), and prints the bytes of its three buffers
//   llama-xdna-insts dump <file>
//       prints the operations of an instruction stream, such as the insts.bin of an mlir-aie build
//   llama-xdna-insts empty --overlay <manifest> --out <file> [--writes <n> [--words <w>] --at <col>,<row>,<register>]
//       writes an instruction stream with no operation for the array of an overlay: a command with it measures
//       what a command costs by itself (the cores of the overlay keep waiting for their parameters); with --writes,
//       n operations that write w zeros (1: a write, more: a block write) from that register of that tile, to measure
//       what the operations cost: only a register that nothing reads meanwhile, such as the runtime parameters of a
//       core that waits for its barrier (they are written again before the core reads them); --patches 1 makes them
//       patches of the address of the first data buffer instead
//   llama-xdna-insts gemv --overlay <manifest> -M <rows> -K <K> (-N <N> [--format <format>] |
//                     --segments <format>:<N>[,<format>:<N>]...) --out <file>
//       the same as gemm for the overlay for generating tokens (xdna-gemv.h): Y = X x W^T, the outputs in up to 4
//       segments, each with its weights in a format: bfp16, q4k, q5k or q6k
//   llama-xdna-insts gemv-pack --overlay <manifest> -M <rows> -K <K> -N <N> [--format <format>] --x <file>
//                     --w <file> --x-out <file> --w-out <file>
//       converts X (M rows of K floats) and W (N rows of K floats, or in q4k, q5k and q6k N rows of K / 256 blocks
//       of ggml) into the buffers of that overlay
//   llama-xdna-insts gemv-unpack --overlay <manifest> -M <rows> -K <K> (-N <N> [--format <format>] |
//                     --segments ...) --y <file> --out <file>
//       reads its buffer of Y into M rows of the outputs of all the segments, as floats
//   llama-xdna-insts chain --overlay <manifest> (--ffn [--next-layer | --logits] | --gdn | -M <rows> -K <K> -N <N>
//                           [--format <format>]) [--no-early] [--drains-first] --out <file>
//       (--gdn: the recurrence of 32 heads of a layer of linear attention, as llama-xdna-gemv --gdn sends it; --la,
//       version 9: the whole layer, as llama-xdna-gemv --la sends it)
//       the instruction stream of a command of the chained overlay (xdna-chain.h): one multiplication, or the
//       feed-forward block of a layer of Qwen3.5 4B in Q4_K in three phases (as llama-xdna-gemv --ffn sends it; with
//       --next-layer, a fourth with the norm and the projections of the next layer, 8192 Q5_K + 4160 Q4_K; with
//       --logits, a fourth with the output norm and the logits, 248320 Q6_K, as in the last layer), and the layout of
//       its buffers and the buffer descriptors every shim tile uses at most; --no-early writes every descriptor right
//       before it is queued, and --drains-first queues the drains of a phase before its fills (see fXdnaChainPlan),
//       as before; with both, the streams of before
//
// It does not touch the NPU. The format of the streams is described in doc/XDNA2.md.

#include "xdna-chain.h"
#include "xdna-gemm.h"
#include "xdna-gemv.h"
#include "xdna-insts.h"
#include "xdna-moe.h"
#include "xdna-overlay.h"

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

namespace {

void fUsage(const char * pProgram) {
  fprintf(stderr,
          "usage: %s gemm --overlay <manifest> -M <M> -K <K> -N <N> --out <file>\n"
          "       %s gdn --overlay <manifest> --rounds <R> --groups <G> --out <file>\n"
          "       %s dump <file>\n"
          "       %s empty --overlay <manifest> --out <file> [--writes <n> [--words <w>] --at <col>,<row>,<register>]\n"
          "       %s gemv --overlay <manifest> -M <rows> -K <K> (-N <N> [--format <format>] | --segments"
          " <format>:<N>[,<format>:<N>]...) --out <file>\n"
          "       %s gemv-pack --overlay <manifest> -M <rows> -K <K> -N <N> [--format <format>] --x <file> --w <file>"
          " --x-out <file> --w-out <file>\n"
          "       %s gemv-unpack --overlay <manifest> -M <rows> -K <K> (-N <N> [--format <format>] | --segments ...)"
          " --y <file> --out <file>\n"
          "       %s chain --overlay <manifest> (--ffn [--next-layer | --logits] | --gdn | --la | -M <rows> -K <K> -N <N>"
          " [--format <format>])\n"
          "          [--no-early] [--drains-first] --out <file>\n"
          "       %s moe --overlay <manifest> [--experts <E>] [--chunks <C>] [--rows <R>] -K <K> -N <N>\n"
          "          [--format <format>] [--swiglu] --out <file>\n"
          "       formats: bfp16, q4k, q5k, q6k, q8_0, q5_1\n",
          pProgram, pProgram, pProgram, pProgram, pProgram, pProgram, pProgram, pProgram, pProgram);
}

bool fReadFloats(const std::string & pPath, uint64_t pCount, std::vector<float> & aValues) {
  std::ifstream vFile(pPath, std::ios::binary);
  if (!vFile) {
    return false;
  }
  aValues.resize(pCount);
  vFile.read(reinterpret_cast<char *>(aValues.data()), std::streamsize(pCount * sizeof(float)));
  return bool(vFile) && vFile.gcount() == std::streamsize(pCount * sizeof(float));
}

bool fWriteBytes(const std::string & pPath, const void * pData, uint64_t pBytes) {
  std::ofstream vOut(pPath, std::ios::binary | std::ios::trunc);
  vOut.write(static_cast<const char *>(pData), std::streamsize(pBytes));
  vOut.close();
  return bool(vOut);
}

// gemv, gemv-pack and gemv-unpack: the options are pairs of a name and a value
int fGemv(int pArgc, char ** pArgv) {
  const std::string vCommand = pArgv[1];
  std::string       vOverlayPath, vOutPath, vXPath, vWPath, vXOutPath, vWOutPath, vYPath;
  uint32_t          vRows = 0, vK = 0, vN = 0;
  XdnaGemvFormat   vFormat = XdnaGemvFormat::cBfp16;
  std::string       vSegments;
  for (int i = 2; i < pArgc; i += 2) {
    const std::string vArg = pArgv[i];
    if (i + 1 >= pArgc) {
      fprintf(stderr, "error: %s needs a value\n", vArg.c_str());
      return 1;
    }
    const char * vValue = pArgv[i + 1];
    if (vArg == "--overlay") {
      vOverlayPath = vValue;
    } else if (vArg == "--out") {
      vOutPath = vValue;
    } else if (vArg == "--x") {
      vXPath = vValue;
    } else if (vArg == "--w") {
      vWPath = vValue;
    } else if (vArg == "--x-out") {
      vXOutPath = vValue;
    } else if (vArg == "--w-out") {
      vWOutPath = vValue;
    } else if (vArg == "--y") {
      vYPath = vValue;
    } else if (vArg == "-M") {
      vRows = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-K") {
      vK = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-N") {
      vN = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--format" && fXdnaGemvFormatFromName(vValue, vFormat)) {
      // taken
    } else if (vArg == "--segments") {
      vSegments = vValue;
    } else {
      fprintf(stderr, "error: unknown option %s\n", vArg.c_str());
      return 1;
    }
  }
  // the segments: the one of -N and --format, or those of --segments
  std::vector<XdnaGemvPart> aParts;
  if (vSegments.empty()) {
    aParts.push_back({vFormat, vN});
  }
  for (size_t vStart = 0; vStart < vSegments.size();) {
    const size_t      vEnd   = vSegments.find(',', vStart);
    const std::string vPart  = vSegments.substr(vStart, vEnd == std::string::npos ? std::string::npos : vEnd - vStart);
    const size_t      vColon = vPart.find(':');
    XdnaGemvPart     vSegment;
    if (vColon == std::string::npos || !fXdnaGemvFormatFromName(vPart.substr(0, vColon), vSegment.vFormat)) {
      fprintf(stderr, "error: a segment is <format>:<outputs>\n");
      return 1;
    }
    vSegment.vOutputs = uint32_t(strtoul(vPart.c_str() + vColon + 1, nullptr, 10));
    aParts.push_back(vSegment);
    vStart = vEnd == std::string::npos ? vSegments.size() : vEnd + 1;
  }
  std::string      vError;
  XdnaGemvOverlay vOverlay;
  XdnaGemvShape   vShape;
  if (vOverlayPath.empty() || !vOverlay.mLoad(vOverlayPath, vError) ||
      !fXdnaGemvShape(vOverlay, vRows, vK, aParts.data(), uint32_t(aParts.size()), vShape, vError)) {
    fprintf(stderr, "error: %s\n", vOverlayPath.empty() ? "--overlay is needed" : vError.c_str());
    return 1;
  }
  vN = vShape.vN;

  if (vCommand == "gemv") {
    std::vector<uint32_t> aInstructions;
    if (vOutPath.empty() || !fXdnaGemvPlan(vOverlay, vShape, aInstructions, vError)) {
      fprintf(stderr, "error: %s\n", vOutPath.empty() ? "--out is needed" : vError.c_str());
      return 1;
    }
    if (!fWriteBytes(vOutPath, aInstructions.data(), aInstructions.size() * sizeof(uint32_t))) {
      fprintf(stderr, "error: cannot write %s\n", vOutPath.c_str());
      return 1;
    }
    printf("instructions: %zu words, %u operations\n", aInstructions.size(), aInstructions[2]);
    printf("sizes: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", vShape.vBytesX, vShape.vBytesW, vShape.vBytesY);
    return 0;
  }

  if (vCommand == "gemv-pack") {
    // W: N rows of K floats, or N rows of K / 256 blocks of ggml
    if (vShape.vSegments != 1) {
      fprintf(stderr, "error: gemv-pack takes one segment\n");
      return 1;
    }
    std::vector<float>   aX;
    std::vector<uint8_t> aW(uint64_t(vN) * fXdnaGemvRowBytes(vShape, vFormat));
    bool vRead = !vXOutPath.empty() && !vWOutPath.empty() && fReadFloats(vXPath, uint64_t(vRows) * vK, aX);
    if (vRead) {
      std::ifstream vFile(vWPath, std::ios::binary);
      vFile.read(reinterpret_cast<char *>(aW.data()), std::streamsize(aW.size()));
      vRead = bool(vFile) && vFile.gcount() == std::streamsize(aW.size());
    }
    if (!vRead) {
      fprintf(stderr, "error: --x, --w, --x-out and --w-out are needed, with M x K floats and the weights\n");
      return 1;
    }
    std::vector<uint8_t> aBufferX(vShape.vBytesX), aBufferW(vShape.vBytesW);
    bool                 vFinite = fXdnaGemvPackX(vOverlay, vShape, aX.data(), vK, aBufferX.data());
    vFinite                      = fXdnaGemvPackW(vOverlay, vShape, aW.data(), aBufferW.data()) && vFinite;
    if (!vFinite) {
      fprintf(stderr, "error: a value is not finite\n");
      return 1;
    }
    if (!fWriteBytes(vXOutPath, aBufferX.data(), aBufferX.size()) ||
        !fWriteBytes(vWOutPath, aBufferW.data(), aBufferW.size())) {
      fprintf(stderr, "error: cannot write the buffers\n");
      return 1;
    }
    printf("sizes: %" PRIu64 " %" PRIu64 "\n", vShape.vBytesX, vShape.vBytesW);
    return 0;
  }

  // gemv-unpack
  std::vector<float> aBufferY;
  if (vOutPath.empty() || !fReadFloats(vYPath, vShape.vBytesY / sizeof(float), aBufferY)) {
    fprintf(stderr, "error: --y with the whole buffer of Y and --out are needed\n");
    return 1;
  }
  std::vector<float> aY(size_t(vRows) * vN);
  XdnaGemmTarget    vTarget;
  vTarget.vData   = aY.data();
  vTarget.vStride = vN;
  vTarget.vFirst  = 0;
  vTarget.vCount  = vN;
  fXdnaGemvUnpackY(vOverlay, vShape, aBufferY.data(), &vTarget, 1);
  if (!fWriteBytes(vOutPath, aY.data(), aY.size() * sizeof(float))) {
    fprintf(stderr, "error: cannot write %s\n", vOutPath.c_str());
    return 1;
  }
  return 0;
}

bool fReadWords(const std::string & pPath, std::vector<uint32_t> & aWords) {
  std::ifstream vFile(pPath, std::ios::binary);
  if (!vFile) {
    return false;
  }
  vFile.seekg(0, std::ios::end);
  const std::streamoff vBytes = vFile.tellg();
  vFile.seekg(0, std::ios::beg);
  if (vBytes < 0 || vBytes % 4 != 0) {
    return false;
  }
  aWords.resize(size_t(vBytes) / 4);
  vFile.read(reinterpret_cast<char *>(aWords.data()), vBytes);
  return bool(vFile);
}

// an address holds the column, the row and the register of a tile
void fPrintAddress(uint32_t pAddress) {
  printf("(%u,%u) 0x%05x", pAddress >> 25, (pAddress >> 20) & 0x1f, pAddress & 0xfffff);
}

int fDump(const std::string & pPath) {
  std::vector<uint32_t> aWords;
  if (!fReadWords(pPath, aWords) || aWords.size() < 4) {
    fprintf(stderr, "error: cannot read %s as an instruction stream\n", pPath.c_str());
    return 1;
  }
  printf("header: version %u.%u, device generation %u, %u rows, %u columns, %u rows of memory tiles, "
         "%u operations, %u bytes (%zu words in the file)\n",
         aWords[0] & 0xff, (aWords[0] >> 8) & 0xff, (aWords[0] >> 16) & 0xff, aWords[0] >> 24, aWords[1] & 0xff,
         (aWords[1] >> 8) & 0xff, aWords[2], aWords[3], aWords.size());
  size_t   vAt    = 4;
  uint32_t vCount = 0;
  while (vAt < aWords.size()) {
    const uint32_t vOpcode = aWords[vAt] & 0xff;
    size_t         vLength = 0;
    // every operation says how long it is; the words it needs have to be there
    auto fHas = [&](size_t pWords) { return vAt + pWords <= aWords.size(); };
    printf("%5u @%6zu: ", vCount, vAt);
    if (vOpcode == 0 && fHas(6)) {
      vLength = 6;
      printf("write      ");
      fPrintAddress(aWords[vAt + 2]);
      printf(" = 0x%08x\n", aWords[vAt + 4]);
    } else if (vOpcode == 1 && fHas(4) && aWords[vAt + 3] % 4 == 0 && aWords[vAt + 3] >= 16 &&
               fHas(aWords[vAt + 3] / 4)) {
      vLength = aWords[vAt + 3] / 4;
      printf("blockwrite ");
      fPrintAddress(aWords[vAt + 2]);
      printf(" =");
      for (size_t i = 4; i < vLength; i++) {
        printf(" %08x", aWords[vAt + i]);
      }
      printf("\n");
    } else if ((vOpcode == 3 || vOpcode == 4) && fHas(7)) {
      vLength = 7;
      printf(vOpcode == 3 ? "maskwrite  " : "maskpoll   ");
      fPrintAddress(aWords[vAt + 2]);
      printf(vOpcode == 3 ? " = 0x%08x mask 0x%08x\n" : " == 0x%08x mask 0x%08x\n", aWords[vAt + 4], aWords[vAt + 5]);
    } else if (vOpcode == 128 && fHas(4)) {
      vLength = 4;
      const uint32_t vWhere = aWords[vAt + 2];
      const uint32_t vSpan  = aWords[vAt + 3];
      printf("sync       column %u row %u direction %u channel %u (%u columns, %u rows)\n", (vWhere >> 16) & 0xff,
             (vWhere >> 8) & 0xff, vWhere & 0xff, vSpan >> 24, (vSpan >> 16) & 0xff, (vSpan >> 8) & 0xff);
    } else if (vOpcode == 129 && fHas(12)) {
      vLength = 12;
      const uint64_t vOffset = uint64_t(aWords[vAt + 10]) | (uint64_t(aWords[vAt + 11]) << 32);
      printf("patch      ");
      fPrintAddress(aWords[vAt + 6]);
      printf(" = address of buffer %u + %" PRIu64 "\n", aWords[vAt + 8], vOffset);
    } else {
      printf("unknown operation %u, or cut short\n", vOpcode);
      return 1;
    }
    vAt += vLength;
    vCount++;
  }
  printf("%u operations decoded\n", vCount);
  return vCount == aWords[2] ? 0 : 1;
}

// gemm, and gdn (bGdn: --rounds and --groups instead of -M, -K and -N)
int fGemm(int pArgc, char ** pArgv, bool bGdn) {
  std::string vOverlayPath;
  std::string vOutPath;
  uint32_t    vM = 0, vK = 0, vN = 0;
  uint32_t    vRounds = 0, vGroups = 0;
  for (int i = 2; i < pArgc; i++) {
    const std::string vArg = pArgv[i];
    if (i + 1 >= pArgc) {
      fprintf(stderr, "error: %s needs a value\n", vArg.c_str());
      return 1;
    }
    const char * vValue = pArgv[++i];
    if (vArg == "--overlay") {
      vOverlayPath = vValue;
    } else if (vArg == "--out") {
      vOutPath = vValue;
    } else if (vArg == "-M") {
      vM = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-K") {
      vK = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-N") {
      vN = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--rounds") {
      vRounds = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--groups") {
      vGroups = uint32_t(strtoul(vValue, nullptr, 10));
    } else {
      fprintf(stderr, "error: unknown option %s\n", vArg.c_str());
      return 1;
    }
  }
  if (vOverlayPath.empty() || vOutPath.empty()) {
    fprintf(stderr, "error: --overlay and --out are needed\n");
    return 1;
  }

  std::string  vError;
  XdnaOverlay vOverlay;
  if (!vOverlay.mLoad(vOverlayPath, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  XdnaGemmPlan vPlan;
  const bool    vPlanned = bGdn ? fXdnaGdnPlan(vOverlay, vRounds, vGroups, vPlan, vError)
                                : fXdnaGemmPlan(vOverlay, vM, vK, vN, vPlan, vError);
  if (!vPlanned) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  std::ofstream vOut(vOutPath, std::ios::binary | std::ios::trunc);
  vOut.write(reinterpret_cast<const char *>(vPlan.aInstructions.data()),
             std::streamsize(vPlan.aInstructions.size() * sizeof(uint32_t)));
  vOut.close();
  if (!vOut) {
    fprintf(stderr, "error: cannot write %s\n", vOutPath.c_str());
    return 1;
  }
  printf("instructions: %zu words, %u operations\n", vPlan.aInstructions.size(), vPlan.aInstructions[2]);
  printf("sizes: %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", vPlan.vBytesA, vPlan.vBytesB, vPlan.vBytesC);
  return 0;
}

int fEmpty(int pArgc, char ** pArgv) {
  std::string vOverlayPath;
  std::string vOutPath;
  uint32_t    vWrites   = 0;
  uint32_t    vWords    = 1;
  bool        vAt       = false;
  bool        vPatches  = false;
  unsigned    vColumn   = 0;
  unsigned    vRow      = 0;
  unsigned    vRegister = 0;
  for (int i = 2; i + 1 < pArgc; i += 2) {
    const std::string vArg = pArgv[i];
    if (vArg == "--overlay") {
      vOverlayPath = pArgv[i + 1];
    } else if (vArg == "--out") {
      vOutPath = pArgv[i + 1];
    } else if (vArg == "--writes") {
      vWrites = uint32_t(strtoul(pArgv[i + 1], nullptr, 0));
    } else if (vArg == "--words") {
      vWords = uint32_t(strtoul(pArgv[i + 1], nullptr, 0));
    } else if (vArg == "--patches") {
      vPatches = strcmp(pArgv[i + 1], "1") == 0;
    } else if (vArg == "--at") {
      vAt = sscanf(pArgv[i + 1], "%u,%u,%i", &vColumn, &vRow, reinterpret_cast<int *>(&vRegister)) == 3;
    } else {
      fUsage(pArgv[0]);
      return 1;
    }
  }
  if (vOverlayPath.empty() || vOutPath.empty() || (vWrites > 0 && (!vAt || vWords == 0 || vWords > 256))) {
    fUsage(pArgv[0]);
    return 1;
  }
  std::string  vError;
  XdnaOverlay vOverlay;
  if (!vOverlay.mLoad(vOverlayPath, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  XdnaInstructions           vInstructions;
  const XdnaTile             vTile = { uint8_t(vColumn), uint8_t(vRow) };
  const std::vector<uint32_t> aZeros(vWords, 0);
  for (uint32_t i = 0; i < vWrites; ++i) {
    if (vPatches) {
      vInstructions.mPatchAddress(vTile, vRegister, 0, 0);
    } else if (vWords == 1) {
      vInstructions.mWrite(vTile, vRegister, 0);
    } else {
      vInstructions.mBlockWrite(vTile, vRegister, aZeros.data(), vWords);
    }
  }
  const std::vector<uint32_t> aWords = vInstructions.mFinish(vOverlay.vColumns, vOverlay.vRows, vOverlay.vMemoryRows);
  std::ofstream               vOut(vOutPath, std::ios::binary | std::ios::trunc);
  vOut.write(reinterpret_cast<const char *>(aWords.data()), std::streamsize(aWords.size() * sizeof(uint32_t)));
  vOut.close();
  if (!vOut) {
    fprintf(stderr, "error: cannot write %s\n", vOutPath.c_str());
    return 1;
  }
  printf("instructions: %zu words, %u operations\n", aWords.size(), aWords[2]);
  return 0;
}

int fChain(int pArgc, char ** pArgv) {
  std::string     vOverlayPath;
  std::string     vOutPath;
  bool            vFfn    = false;
  bool            vGdn    = false;
  bool            vLa     = false;
  bool            vNextLayer = false;
  bool            vLogits    = false;
  bool            vEarly     = true;
  bool            vFillsFirst = true;
  uint32_t        vRows   = 0;
  uint32_t        vK      = 0;
  uint32_t        vN      = 0;
  XdnaGemvFormat vFormat = XdnaGemvFormat::cBfp16;
  for (int i = 2; i < pArgc; i++) {
    const std::string vArg = pArgv[i];
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
    if (vArg == "--logits") {
      vLogits = true;
      continue;
    }
    if (vArg == "--no-early") {
      vEarly = false;
      continue;
    }
    if (vArg == "--drains-first") {
      vFillsFirst = false;
      continue;
    }
    if (i + 1 >= pArgc) {
      fUsage(pArgv[0]);
      return 1;
    }
    const char * vValue = pArgv[++i];
    if (vArg == "--overlay") {
      vOverlayPath = vValue;
    } else if (vArg == "--out") {
      vOutPath = vValue;
    } else if (vArg == "-M") {
      vRows = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-K") {
      vK = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-N") {
      vN = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg != "--format" || !fXdnaGemvFormatFromName(vValue, vFormat)) {
      fUsage(pArgv[0]);
      return 1;
    }
  }
  if (vOverlayPath.empty() || vOutPath.empty() || (!vFfn && !vGdn && !vLa && (vRows == 0 || vK == 0 || vN == 0)) ||
      ((vNextLayer || vLogits) && !vFfn) || (vNextLayer && vLogits) || int(vFfn) + int(vGdn) + int(vLa) > 1) {
    fUsage(pArgv[0]);
    return 1;
  }
  std::string       vError;
  XdnaChainOverlay vOverlay;
  if (!vOverlay.mLoad(vOverlayPath, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  XdnaChainMatrix aMatrices[4];
  XdnaChainPhase  aPhases[4];
  uint32_t         vPhases   = 1;
  uint32_t         vMatrices = 1;
  uint32_t         vStates   = 0;
  bool             vShaped   = true;
  if (vGdn) {
    // the recurrence of a layer of linear attention of Qwen3.5 4B, 32 heads, as llama-xdna-gemv --gdn sends it: the
    // states in buffer 1, head 0 at its start
    aPhases[0].vRecurrence.vHeads = 32;
    vMatrices                     = 0;
    vStates                       = 1;
  } else if (vLa) {
    // llama.cpp-xdna, version 9: a whole layer of linear attention of Qwen3.5 4B, as llama-xdna-gemv --la sends it:
    // its fixed objects in buffer 1 and the states in buffer 2, head 0 at its start
    aPhases[0].vRecurrence.vHeads  = 32;
    aPhases[0].vRecurrence.vLinear = 1;
    fXdnaChainLinearMatrix(vOverlay, aMatrices[0]);
    vMatrices = 1;
    vStates   = 1;
  } else if (vFfn) {
    const XdnaGemvPart aFirst[1]  = {{XdnaGemvFormat::cQ4K, 2560}};
    const XdnaGemvPart aGateUp[2] = {{XdnaGemvFormat::cQ4K, 9216}, {XdnaGemvFormat::cQ4K, 9216}};
    const XdnaGemvPart aDown[1]   = {{XdnaGemvFormat::cQ4K, 2560}};
    vShaped = fXdnaChainMatrixShape(vOverlay, 4096, aFirst, 1, aMatrices[0], vError) &&
              fXdnaChainMatrixShape(vOverlay, 2560, aGateUp, 2, aMatrices[1], vError) &&
              fXdnaChainMatrixShape(vOverlay, 9216, aDown, 1, aMatrices[2], vError);
    aPhases[0].vEpilogue    = XdnaChainEpilogue::cResidual;
    aPhases[1].vPrologue    = XdnaChainPrologue::cNorm;
    aPhases[1].vFromPhase   = 0;
    aPhases[1].vEpilogue    = XdnaChainEpilogue::cSwiglu;
    aPhases[1].vEps         = 1e-6f;
    aPhases[1].vMatrix      = 1;
    aPhases[1].vSegments    = 2;
    aPhases[2].vFromPhase   = 1;
    aPhases[2].vFromSegment = 1;
    aPhases[2].vEpilogue    = XdnaChainEpilogue::cAddH;
    aPhases[2].vMatrix      = 2;
    vPhases                 = 3;
    vMatrices               = 3;
    if (vNextLayer) {
      // the norm and the projections of the next layer, as llama-xdna-gemv --ffn --next-layer sends them
      const XdnaGemvPart aNext[2] = {{XdnaGemvFormat::cQ5K, 8192}, {XdnaGemvFormat::cQ4K, 4160}};
      vShaped = vShaped && fXdnaChainMatrixShape(vOverlay, 2560, aNext, 2, aMatrices[3], vError);
      aPhases[3].vPrologue  = XdnaChainPrologue::cNorm;
      aPhases[3].vFromPhase = 2;
      aPhases[3].vEps       = 1e-6f;
      aPhases[3].vMatrix    = 3;
      aPhases[3].vSegments  = 2;
      vPhases               = 4;
      vMatrices             = 4;
    }
    if (vLogits) {
      // the output norm and the logits, as the last layer of Qwen3.5 4B Q4_K_M sends them
      const XdnaGemvPart aLogits[1] = {{XdnaGemvFormat::cQ6K, 248320}};
      vShaped = vShaped && fXdnaChainMatrixShape(vOverlay, 2560, aLogits, 1, aMatrices[3], vError);
      aPhases[3].vPrologue  = XdnaChainPrologue::cNorm;
      aPhases[3].vFromPhase = 2;
      aPhases[3].vEps       = 1e-6f;
      aPhases[3].vMatrix    = 3;
      aPhases[3].vSegments  = 1;
      vPhases               = 4;
      vMatrices             = 4;
    }
  } else {
    const XdnaGemvPart aParts[1] = {{vFormat, vN}};
    vShaped               = fXdnaChainMatrixShape(vOverlay, vK, aParts, 1, aMatrices[0], vError);
    aPhases[0].vPrologue  = vRows == 1 ? XdnaChainPrologue::cFloats : XdnaChainPrologue::cBfp16;
    aPhases[0].vRows      = vRows;
  }
  XdnaChainCommand     vCommand;
  std::vector<uint32_t> aWords;
  if (!vShaped ||
      !fXdnaChainCommandShape(vOverlay, aPhases, vPhases, aMatrices, vMatrices, vCommand, vError, vStates) ||
      !fXdnaChainPlan(vOverlay, vCommand, aWords, vError, vEarly, vFillsFirst)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  if (!fWriteBytes(vOutPath, aWords.data(), aWords.size() * sizeof(uint32_t))) {
    fprintf(stderr, "error: cannot write %s\n", vOutPath.c_str());
    return 1;
  }
  // the buffer descriptors of the shim tiles: the lowest free one goes first, so the highest one written is the most
  // that a tile has in use at a time
  uint32_t vDescriptors = 0;
  for (size_t vAt = 4; vAt + 4 <= aWords.size();) {
    const uint32_t vOpcode = aWords[vAt] & 0xff;
    const size_t   vLength = vOpcode == 0 ? 6 : vOpcode == 1 ? aWords[vAt + 3] / 4 : (vOpcode == 3 || vOpcode == 4) ? 7 :
                             vOpcode == 128 ? 4 : vOpcode == 129 ? 12 : 0;
    if (vLength == 0) {
      break;
    }
    const uint32_t vAddress  = aWords[vAt + 2];
    const uint32_t vRegister = vAddress & 0xfffff;
    if (vOpcode == 1 && ((vAddress >> 20) & 0x1f) == 0 && vRegister >= 0x1d000 && vRegister < 0x1d200) {
      vDescriptors = std::max(vDescriptors, (vRegister - 0x1d000) / 0x20 + 1);
    }
    vAt += vLength;
  }
  printf("instructions: %zu words, %u operations\n", aWords.size(), aWords[2]);
  printf("buffer descriptors of a shim tile: %u at most\n", vDescriptors);
  printf("buffer of the host: %" PRIu64 " bytes, the host writes %" PRIu64 ", Y from %" PRIu64 "\n", vCommand.vBytesIo,
         vCommand.vBytesIn, vCommand.vOffsetY);
  for (uint32_t p = 0; p < vPhases; p++) {
    const XdnaChainPlace & vPlace = vCommand.aPlaces[p];
    printf("phase %u: K %u (padded %u), objects of X %u, of the prologue %u, X at %" PRIu64 ", residual at %" PRIu64
           ", norm at %" PRIu64 ", Y at %" PRIu64 " (%" PRIu64 " bytes)\n", p, vPlace.vK, vPlace.vPaddedK,
           vPlace.vObjectsX, vPlace.vObjectsPrologue, vPlace.vOffsetX, vPlace.vOffsetResidual, vPlace.vOffsetNorm,
           vPlace.vOffsetY[0], vPlace.vBytesY);
  }
  for (uint32_t m = 0; m < vMatrices; m++) {
    printf("matrix %u: K %u, %u tiles of a core, %" PRIu64 " bytes of a stream:", m, aMatrices[m].vK,
           aMatrices[m].vTilesCore, aMatrices[m].vPartW);
    for (uint32_t g = 0; g < aMatrices[m].vSegments; g++) {
      printf(" [%u outputs, %u blocks, %u objects]", aMatrices[m].aSegments[g].vOutputs,
             aMatrices[m].aSegments[g].vBlocks, aMatrices[m].aSegments[g].vObjects);
    }
    printf("\n");
  }
  return 0;
}

int fMoe(int pArgc, char ** pArgv) {
  std::string    vOverlayPath;
  std::string    vOutPath;
  uint32_t       vExperts = 256;
  uint32_t       vChunks  = 8;
  uint32_t       vRows    = 0;
  uint32_t       vK       = 2048;
  uint32_t       vN       = 512;
  bool           vSwiglu  = false;
  XdnaGemvFormat vFormat  = XdnaGemvFormat::cQ6K;
  for (int i = 2; i < pArgc; i++) {
    const std::string vArg = pArgv[i];
    if (vArg == "--swiglu") {
      vSwiglu = true;
      continue;
    }
    if (i + 1 >= pArgc) {
      fUsage(pArgv[0]);
      return 1;
    }
    const char * vValue = pArgv[++i];
    if (vArg == "--overlay") {
      vOverlayPath = vValue;
    } else if (vArg == "--out") {
      vOutPath = vValue;
    } else if (vArg == "--experts") {
      vExperts = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--chunks") {
      vChunks = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "--rows") {
      vRows = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-K") {
      vK = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg == "-N") {
      vN = uint32_t(strtoul(vValue, nullptr, 10));
    } else if (vArg != "--format" || !fXdnaGemvFormatFromName(vValue, vFormat)) {
      fUsage(pArgv[0]);
      return 1;
    }
  }
  if (vOverlayPath.empty() || vOutPath.empty() || vExperts == 0 || vChunks == 0) {
    fUsage(pArgv[0]);
    return 1;
  }
  std::string    vError;
  XdnaMoeOverlay vOverlay;
  if (!vOverlay.mLoad(vOverlayPath, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  const XdnaGemvPart aParts[2] = {{vFormat, vN}, {vFormat, vN}};
  XdnaMoeMatrix      vMatrix;
  if (!fXdnaMoeMatrixShape(vOverlay, vExperts, vK, aParts, vSwiglu ? 2 : 1, vSwiglu, vMatrix, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  const uint32_t             vMaxRows = fXdnaMoeMaxRows(vOverlay, vMatrix);
  std::vector<XdnaMoeChunk> aChunks(vChunks);
  for (uint32_t c = 0; c < vChunks; c++) {
    const uint32_t vChunkRows = vRows > 0 ? vRows : 1 + (c * 7) % vMaxRows;
    aChunks[c]                = {(c * 13) % vExperts, vChunkRows};
  }
  XdnaMoeCommand        vCommand;
  std::vector<uint32_t> aWords;
  if (!fXdnaMoeCommandShape(vOverlay, vMatrix, aChunks.data(), vChunks, vCommand, vError) ||
      !fXdnaMoePlan(vOverlay, vCommand, aWords, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  if (!fWriteBytes(vOutPath, aWords.data(), aWords.size() * sizeof(uint32_t))) {
    fprintf(stderr, "error: cannot write %s\n", vOutPath.c_str());
    return 1;
  }
  uint32_t vDescriptors = 0;
  for (size_t vAt = 4; vAt + 4 <= aWords.size();) {
    const uint32_t vOpcode = aWords[vAt] & 0xff;
    const size_t   vLength = vOpcode == 0 ? 6 : vOpcode == 1 ? aWords[vAt + 3] / 4 : (vOpcode == 3 || vOpcode == 4) ? 7 :
                             vOpcode == 128 ? 4 : vOpcode == 129 ? 12 : 0;
    if (vLength == 0) {
      break;
    }
    const uint32_t vAddress  = aWords[vAt + 2];
    const uint32_t vRegister = vAddress & 0xfffff;
    if (vOpcode == 1 && ((vAddress >> 20) & 0x1f) == 0 && vRegister >= 0x1d000 && vRegister < 0x1d200) {
      vDescriptors = std::max(vDescriptors, (vRegister - 0x1d000) / 0x20 + 1);
    }
    vAt += vLength;
  }
  printf("instructions: %zu words, %u operations\n", aWords.size(), aWords[2]);
  printf("buffer descriptors of a shim tile: %u at most\n", vDescriptors);
  printf("chunks: %u, total rows: %u, max rows per chunk: %u\n", vCommand.vChunks, vCommand.vTotalRows, vMaxRows);
  printf("buffer of the host: %" PRIu64 " bytes, the host writes %" PRIu64 ", Y from %" PRIu64 " (%" PRIu64 " bytes)\n",
         vCommand.vBytesIo, vCommand.vBytesIn, vCommand.vOffsetY, vCommand.vBytesY);
  printf("weights: %u experts, %" PRIu64 " bytes per expert and stream, %" PRIu64 " bytes in all\n", vMatrix.vExperts,
         vMatrix.vExpertBytes, vMatrix.vBytesW);
  return 0;
}

}  // namespace

int main(int pArgc, char ** pArgv) {
  if (pArgc >= 3 && strcmp(pArgv[1], "dump") == 0) {
    return fDump(pArgv[2]);
  }
  if (pArgc >= 2 && (strcmp(pArgv[1], "gemm") == 0 || strcmp(pArgv[1], "gdn") == 0)) {
    return fGemm(pArgc, pArgv, strcmp(pArgv[1], "gdn") == 0);
  }
  if (pArgc >= 2 && strcmp(pArgv[1], "empty") == 0) {
    return fEmpty(pArgc, pArgv);
  }
  if (pArgc >= 2 && strcmp(pArgv[1], "chain") == 0) {
    return fChain(pArgc, pArgv);
  }
  if (pArgc >= 2 && strcmp(pArgv[1], "moe") == 0) {
    return fMoe(pArgc, pArgv);
  }
  if (pArgc >= 2 && (strcmp(pArgv[1], "gemv") == 0 || strcmp(pArgv[1], "gemv-pack") == 0 ||
                     strcmp(pArgv[1], "gemv-unpack") == 0)) {
    return fGemv(pArgc, pArgv);
  }
  fUsage(pArgv[0]);
  return 1;
}
