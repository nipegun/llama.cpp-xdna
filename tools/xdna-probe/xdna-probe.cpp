// llama-xdna-probe: measures the fixed costs of the NPU that decide the design of the XDNA backend, through the
// amdxdna driver, without XRT. The instruction streams it runs should move no data (`llama-xdna-insts empty` makes
// them), so that what it measures is what a command costs by itself.
//
//   llama-xdna-probe switch --pdi <a> --insts <a> --pdi2 <b> --insts2 <b> [--repeat <n>]
//     two contexts, each with its own overlay: commands to the first only, to the second only, and alternating
//   llama-xdna-probe units --pdi <a> --insts <a> --pdi2 <b> --insts2 <b> [--repeat <n>]
//     the same with one context that has the two overlays as two compute units, picked by every command
//   llama-xdna-probe chain --pdi <a> --insts <a> [--chain <k>] [--repeat <n>]
//     one context: commands one by one, up to 4 submitted before waiting (what the driver keeps in flight for a
//     context), and in chains of k commands (ERT_CMD_CHAIN), which the firmware gets as one message
//
// A command that does not complete is never retried: the program reports its state and stops.

#include "xdna-device.h"

#include <algorithm>
#include <chrono>
#include <cinttypes>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>

// words of a request of a compute unit: the mask of compute units, the opcode (64 bits; 3 = run a transaction buffer),
// the address of the instructions (64 bits), their number of words and 5 empty buffer slots of 64 bits
constexpr uint32_t cRequestWords = 1 + 5 + 2 * 5;
// commands the driver keeps in flight for a context (HWCTX_MAX_CMDS of aie2_pci.h)
constexpr uint32_t cInFlight = 4;
// the slots of a chain go into a buffer of 4 KiB (MAX_CHAIN_CMDBUF_SIZE of aie2_msg_priv.h), about 68 bytes each
constexpr uint32_t cMaxChain = 32;

// one design: its overlay and its instruction stream, in the heap of the process, and its context
struct ProbeDesign {
  std::string          vPdiPath;
  std::string          vInstsPath;
  std::vector<uint8_t> aPdi;
  std::vector<uint8_t> aInsts;
  uint32_t             vColumns = 0;
  XdnaBuffer          vPdi;
  XdnaBuffer          vInsts;
  XdnaContext         vContext;
};

static double fNowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

static bool fReadFile(const std::string & pPath, std::vector<uint8_t> & pData) {
  std::ifstream vFile(pPath, std::ios::binary);
  if (!vFile) {
    return false;
  }
  pData.assign(std::istreambuf_iterator<char>(vFile), std::istreambuf_iterator<char>());
  return true;
}

// columns of the partition an overlay was built for, from the partition_main.json that mlir-aie leaves next to it; 0 if
// the file is not there or does not say
static uint32_t fPartitionColumns(const std::string & pPdiPath) {
  const size_t         vSlash  = pPdiPath.find_last_of('/');
  const std::string    vFolder = vSlash == std::string::npos ? "." : pPdiPath.substr(0, vSlash);
  std::vector<uint8_t> aJson;
  if (!fReadFile(vFolder + "/partition_main.json", aJson)) {
    return 0;
  }
  const std::string vText(aJson.begin(), aJson.end());
  const size_t      vKey = vText.find("\"column_width\"");
  if (vKey == std::string::npos) {
    return 0;
  }
  const size_t vColon = vText.find(':', vKey);
  if (vColon == std::string::npos) {
    return 0;
  }
  return (uint32_t) std::strtoul(vText.c_str() + vColon + 1, nullptr, 10);
}

static double fMedian(std::vector<double> aTimes) {
  if (aTimes.empty()) {
    return 0.0;
  }
  std::sort(aTimes.begin(), aTimes.end());
  return aTimes[aTimes.size() / 2];
}

static void fUsage(const char * pProgram) {
  printf("usage: %s switch|units --pdi <file> --insts <file> --pdi2 <file> --insts2 <file> [options]\n", pProgram);
  printf("       %s chain --pdi <file> --insts <file> [--chain <k>] [options]\n", pProgram);
  printf("\n");
  printf("  --pdi, --pdi2 <file>       overlays (main.pdi of an mlir-aie build); each needs its partition_main.json\n");
  printf("  --insts, --insts2 <file>   instruction streams that move no data (llama-xdna-insts empty)\n");
  printf("  --chain <k>                commands in every chain (default: 8, at most %u)\n", cMaxChain);
  printf("  --repeat <n>               commands timed in every way (default: 200)\n");
  printf("  --timeout-ms <n>           how long to wait for a command (default: 10000)\n");
  printf("  --device <node>            device node of the NPU (default: /dev/accel/accel0)\n");
}

// fills a request of compute unit pUnit that runs the instructions of pDesign, with no data buffers
static void fFillRequest(uint32_t * pWords, const ProbeDesign & pDesign, uint32_t pUnit) {
  std::memset(pWords, 0, (1 + cRequestWords) * sizeof(uint32_t));
  pWords[1] = 1u << pUnit;
  pWords[2] = 3;
  pWords[3] = 0;
  pWords[4] = (uint32_t) (pDesign.vInsts.vDevAddr & 0xffffffffu);
  pWords[5] = (uint32_t) (pDesign.vInsts.vDevAddr >> 32);
  pWords[6] = (uint32_t) (pDesign.aInsts.size() / 4);
}

int main(int pArgc, char ** pArgv) {
  if (pArgc < 2 || (strcmp(pArgv[1], "switch") != 0 && strcmp(pArgv[1], "units") != 0 && strcmp(pArgv[1], "chain") != 0)) {
    fUsage(pArgv[0]);
    return pArgc >= 2 && (strcmp(pArgv[1], "-h") == 0 || strcmp(pArgv[1], "--help") == 0) ? 0 : 2;
  }
  // units: one context with the two overlays; switch: a context for each
  const bool  vUnits     = strcmp(pArgv[1], "units") == 0;
  const bool  vSwitch    = vUnits || strcmp(pArgv[1], "switch") == 0;
  std::string vPath      = "/dev/accel/accel0";
  ProbeDesign aDesigns[2];
  uint32_t    vChain     = 8;
  int         vRepeat    = 200;
  uint32_t    vTimeoutMs = 10000;

  for (int i = 2; i < pArgc; i++) {
    const std::string vArg  = pArgv[i];
    const bool        vMore = i + 1 < pArgc;
    if (vArg == "--pdi" && vMore) {
      aDesigns[0].vPdiPath = pArgv[++i];
    } else if (vArg == "--insts" && vMore) {
      aDesigns[0].vInstsPath = pArgv[++i];
    } else if (vArg == "--pdi2" && vMore) {
      aDesigns[1].vPdiPath = pArgv[++i];
    } else if (vArg == "--insts2" && vMore) {
      aDesigns[1].vInstsPath = pArgv[++i];
    } else if (vArg == "--chain" && vMore) {
      vChain = (uint32_t) std::strtoul(pArgv[++i], nullptr, 0);
    } else if (vArg == "--repeat" && vMore) {
      vRepeat = std::atoi(pArgv[++i]);
    } else if (vArg == "--timeout-ms" && vMore) {
      vTimeoutMs = (uint32_t) std::strtoul(pArgv[++i], nullptr, 0);
    } else if (vArg == "--device" && vMore) {
      vPath = pArgv[++i];
    } else {
      fUsage(pArgv[0]);
      return 2;
    }
  }
  const uint32_t vDesigns = vSwitch ? 2 : 1;
  if (vRepeat < 1 || vChain < 1 || vChain > cMaxChain) {
    fUsage(pArgv[0]);
    return 2;
  }

  XdnaDevice vDevice;
  std::string vError;
  XdnaInfo   vInfo;
  if (!vDevice.mOpen(vPath, vError) || !vDevice.mQueryInfo(vInfo, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  for (uint32_t d = 0; d < vDesigns; d++) {
    ProbeDesign & vDesign = aDesigns[d];
    if (vDesign.vPdiPath.empty() || vDesign.vInstsPath.empty()) {
      fUsage(pArgv[0]);
      return 2;
    }
    if (!fReadFile(vDesign.vPdiPath, vDesign.aPdi) || vDesign.aPdi.empty()) {
      fprintf(stderr, "error: cannot read %s\n", vDesign.vPdiPath.c_str());
      return 1;
    }
    if (!fReadFile(vDesign.vInstsPath, vDesign.aInsts) || vDesign.aInsts.empty() || vDesign.aInsts.size() % 4 != 0) {
      fprintf(stderr, "error: cannot read %s as 32-bit words\n", vDesign.vInstsPath.c_str());
      return 1;
    }
    // the context has to be as wide as the partition the overlay was built for: a narrower one hung the NPU
    // (2026/11/12), so an overlay without its partition_main.json is not run
    vDesign.vColumns = fPartitionColumns(vDesign.vPdiPath);
    if (vDesign.vColumns == 0 || vDesign.vColumns > vInfo.vColumns) {
      fprintf(stderr, "error: no usable partition_main.json next to %s\n", vDesign.vPdiPath.c_str());
      return 1;
    }
  }

  XdnaBuffer              vHeap;
  std::vector<XdnaBuffer> aCommands;
  int                      vResult = 1;

  // everything is released in one place, in the order the driver needs: the contexts, the buffers, the heap
  auto fRelease = [&]() {
    for (uint32_t d = 0; d < vDesigns; d++) {
      vDevice.mDestroyContext(aDesigns[d].vContext);
    }
    for (XdnaBuffer & vCommand : aCommands) {
      vDevice.mDestroyBuffer(vCommand);
    }
    for (uint32_t d = 0; d < vDesigns; d++) {
      vDevice.mDestroyBuffer(aDesigns[d].vInsts);
      vDevice.mDestroyBuffer(aDesigns[d].vPdi);
    }
    vDevice.mDestroyBuffer(vHeap);
  };
  auto fFail = [&](const std::string & pWhat) {
    fprintf(stderr, "error: %s: %s\n", pWhat.c_str(), vError.c_str());
    fRelease();
    return 1;
  };

  if (!vDevice.mCreateBuffer(XdnaBufferType::cDevHeap, 64ull << 20, vHeap, vError)) {
    return fFail("heap");
  }
  for (uint32_t d = 0; d < vDesigns; d++) {
    ProbeDesign & vDesign = aDesigns[d];
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, vDesign.aPdi.size(), vDesign.vPdi, vError) ||
        !vDevice.mCreateBuffer(XdnaBufferType::cDev, vDesign.aInsts.size(), vDesign.vInsts, vError)) {
      return fFail("overlay or instruction buffer");
    }
    std::memcpy(vDesign.vPdi.vMap, vDesign.aPdi.data(), vDesign.aPdi.size());
    std::memcpy(vDesign.vInsts.vMap, vDesign.aInsts.data(), vDesign.aInsts.size());
    if (!vDevice.mSyncBuffer(vDesign.vPdi, 0, vDesign.vPdi.vSize, vError) ||
        !vDevice.mSyncBuffer(vDesign.vInsts, 0, vDesign.vInsts.vSize, vError)) {
      return fFail("sync");
    }
    printf("design %u: %s, %u columns, %zu instruction words\n", d, vDesign.vPdiPath.c_str(), vDesign.vColumns,
           vDesign.aInsts.size() / 4);
    if (vUnits && d == 1) {
      continue;
    }
    if (!vDevice.mCreateContext(vDesign.vColumns * vInfo.vCore.vRowCount, 2048, vDesign.vContext, vError)) {
      return fFail("context " + std::to_string(d));
    }
    if (!vUnits && !vDevice.mConfigureContext(vDesign.vContext, vDesign.vPdi, 0, vError)) {
      return fFail("configuration of context " + std::to_string(d));
    }
  }
  if (vUnits) {
    if (aDesigns[0].vColumns != aDesigns[1].vColumns) {
      vError = "the two overlays were built for partitions of different widths";
      return fFail("units");
    }
    if (!vDevice.mConfigureContextUnits(aDesigns[0].vContext, { &aDesigns[0].vPdi, &aDesigns[1].vPdi }, 0, vError)) {
      return fFail("configuration of the context with two compute units");
    }
  }
  // the context that runs the commands of design d, and the compute unit of it that runs them
  auto fContextOf = [&](uint32_t pDesign) -> const XdnaContext & {
    return aDesigns[vUnits ? 0 : pDesign].vContext;
  };
  auto fUnitOf = [&](uint32_t pDesign) -> uint32_t {
    return vUnits ? pDesign : 0;
  };

  // command buffers: one per command that can be in flight, and for a chain one per command of the chain plus the chain
  const uint32_t vBuffers = vSwitch ? 2 : cInFlight + vChain + 1;
  aCommands.resize(vBuffers);
  for (uint32_t b = 0; b < vBuffers; b++) {
    if (!vDevice.mCreateBuffer(XdnaBufferType::cCommand, 4096, aCommands[b], vError)) {
      return fFail("command buffer");
    }
  }
  std::vector<uint32_t> aHandles[2];
  for (uint32_t d = 0; d < vDesigns; d++) {
    aHandles[d].push_back(aDesigns[d].vInsts.vHandle);
  }

  // submits the request in pCommand to the context of pDesign; with pWait, waits for it and checks its state
  auto fSubmit = [&](uint32_t pDesign, XdnaBuffer & pCommand, uint32_t pOpcode, uint32_t pWords, bool pWait,
                     uint64_t & pSeq) {
    uint32_t * vWords = static_cast<uint32_t *>(pCommand.vMap);
    vWords[0]         = fXdnaCommandHeader(pOpcode, pWords);
    if (!vDevice.mSubmit(fContextOf(pDesign), pCommand, aHandles[pDesign], pSeq, vError)) {
      return false;
    }
    if (!pWait) {
      return true;
    }
    const XdnaWaitResult vWaited = vDevice.mWait(fContextOf(pDesign), pSeq, vTimeoutMs, vError);
    const uint32_t        vState  = fXdnaCommandState(vWords[0]);
    if (vWaited != XdnaWaitResult::cDone || vState != cXdnaStateCompleted) {
      vError = std::string("command ") + std::to_string(pSeq) + " of design " + std::to_string(pDesign) + ": " +
               (vWaited == XdnaWaitResult::cDone ? fXdnaStateName(vState) : "no answer") + " " + vError;
      return false;
    }
    return true;
  };
  // waits for a command submitted without waiting and checks its state
  auto fFinish = [&](uint32_t pDesign, XdnaBuffer & pCommand, uint64_t pSeq) {
    const XdnaWaitResult vWaited = vDevice.mWait(fContextOf(pDesign), pSeq, vTimeoutMs, vError);
    const uint32_t        vState  = fXdnaCommandState(static_cast<uint32_t *>(pCommand.vMap)[0]);
    if (vWaited != XdnaWaitResult::cDone || vState != cXdnaStateCompleted) {
      vError = std::string("command ") + std::to_string(pSeq) + ": " +
               (vWaited == XdnaWaitResult::cDone ? fXdnaStateName(vState) : "no answer") + " " + vError;
      return false;
    }
    return true;
  };

  uint64_t vSeq = 0;
  if (vSwitch) {
    for (uint32_t d = 0; d < 2; d++) {
      fFillRequest(static_cast<uint32_t *>(aCommands[d].vMap), aDesigns[d], fUnitOf(d));
    }
    // the first command of each design loads its overlay: not timed
    for (uint32_t d = 0; d < 2; d++) {
      const double vStart = fNowMs();
      if (!fSubmit(d, aCommands[d], cXdnaCommandStartCu, cRequestWords, true, vSeq)) {
        return fFail("first command");
      }
      printf("first command of design %u: %.3f ms\n", d, fNowMs() - vStart);
    }
    std::vector<double> aAlone[2];
    std::vector<double> aAlternating[2];
    for (uint32_t d = 0; d < 2; d++) {
      for (int r = 0; r < vRepeat; r++) {
        const double vStart = fNowMs();
        if (!fSubmit(d, aCommands[d], cXdnaCommandStartCu, cRequestWords, true, vSeq)) {
          return fFail("alone");
        }
        aAlone[d].push_back(fNowMs() - vStart);
      }
    }
    for (int r = 0; r < vRepeat; r++) {
      for (uint32_t d = 0; d < 2; d++) {
        const double vStart = fNowMs();
        if (!fSubmit(d, aCommands[d], cXdnaCommandStartCu, cRequestWords, true, vSeq)) {
          return fFail("alternating");
        }
        aAlternating[d].push_back(fNowMs() - vStart);
      }
    }
    for (uint32_t d = 0; d < 2; d++) {
      printf("design %u: %d commands alone, median %.3f ms; alternating with the other design, median %.3f ms\n", d,
             vRepeat, fMedian(aAlone[d]), fMedian(aAlternating[d]));
    }
    printf("cost of changing %s: about %.3f ms per command\n", vUnits ? "compute unit" : "context",
           (fMedian(aAlternating[0]) + fMedian(aAlternating[1]) - fMedian(aAlone[0]) - fMedian(aAlone[1])) / 2.0);
    vResult = 0;
  } else {
    for (uint32_t b = 0; b < cInFlight + vChain; b++) {
      fFillRequest(static_cast<uint32_t *>(aCommands[b].vMap), aDesigns[0], 0);
    }
    if (!fSubmit(0, aCommands[0], cXdnaCommandStartCu, cRequestWords, true, vSeq)) {
      return fFail("first command");
    }
    // one by one
    std::vector<double> aSingle;
    for (int r = 0; r < vRepeat; r++) {
      const double vStart = fNowMs();
      if (!fSubmit(0, aCommands[0], cXdnaCommandStartCu, cRequestWords, true, vSeq)) {
        return fFail("one by one");
      }
      aSingle.push_back(fNowMs() - vStart);
    }
    // up to cInFlight in flight: each buffer is submitted again once its previous command ended
    std::vector<uint64_t> aSeqs(cInFlight, 0);
    double                vStart = fNowMs();
    for (int r = 0; r < vRepeat; r++) {
      const uint32_t vSlot = (uint32_t) r % cInFlight;
      if (r >= (int) cInFlight && !fFinish(0, aCommands[vSlot], aSeqs[vSlot])) {
        return fFail("in flight");
      }
      if (!fSubmit(0, aCommands[vSlot], cXdnaCommandStartCu, cRequestWords, false, aSeqs[vSlot])) {
        return fFail("in flight");
      }
    }
    for (int r = std::max(0, vRepeat - (int) cInFlight); r < vRepeat; r++) {
      const uint32_t vSlot = (uint32_t) r % cInFlight;
      if (!fFinish(0, aCommands[vSlot], aSeqs[vSlot])) {
        return fFail("in flight");
      }
    }
    const double vInFlightMs = (fNowMs() - vStart) / vRepeat;
    // chains: the payload is the number of commands, two indexes the firmware fills, 3 reserved words and the handle
    // of every command (64 bits each); a chain has no mask of compute units
    XdnaBuffer & vChainBuffer = aCommands[cInFlight + vChain];
    uint32_t *    vChainWords  = static_cast<uint32_t *>(vChainBuffer.vMap);
    std::memset(vChainWords, 0, 4096);
    vChainWords[1] = vChain;
    for (uint32_t c = 0; c < vChain; c++) {
      vChainWords[7 + 2 * c] = aCommands[cInFlight + c].vHandle;
      // the header of every command of the chain, as for a command of its own
      static_cast<uint32_t *>(aCommands[cInFlight + c].vMap)[0] = fXdnaCommandHeader(cXdnaCommandStartCu, cRequestWords);
    }
    std::vector<double> aChains;
    const int           vChains = std::max(1, vRepeat / (int) vChain);
    for (int r = 0; r < vChains; r++) {
      vChainWords[2]      = 0;
      vChainWords[3]      = 0;
      const double vBegin = fNowMs();
      if (!fSubmit(0, vChainBuffer, cXdnaCommandChain, 6 + 2 * vChain, true, vSeq)) {
        return fFail("chain");
      }
      aChains.push_back((fNowMs() - vBegin) / vChain);
    }
    printf("one by one: median %.3f ms per command (%d commands)\n", fMedian(aSingle), vRepeat);
    printf("up to %u in flight: %.3f ms per command (%d commands)\n", cInFlight, vInFlightMs, vRepeat);
    printf("in chains of %u: median %.3f ms per command (%d chains)\n", vChain, fMedian(aChains), vChains);
    vResult = 0;
  }

  fRelease();
  return vResult;
}
