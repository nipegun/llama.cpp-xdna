// llama-xdna-run: runs a design compiled for the NPU (its overlay, a PDI, and its instruction stream) through the
// amdxdna driver, without XRT, and times it. For developing and measuring the kernels of the XDNA backend.
//
// The data buffers are given in the order of the arguments of the design: --in <file> is a buffer with the contents of
// a file, --out <bytes>[:<file>] a zeroed buffer of that size, written to the file after the last run.
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

// one data buffer of the design
struct RunBuffer {
  bool        vIsInput = false;
  std::string vFile;       // read from it (input) or written to it (output, may be empty)
  uint64_t    vSize = 0;   // bytes used
  XdnaBuffer vBuffer;
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

// what the driver knows, without privileges, about the contexts and about the last error of the firmware
static void fPrintDiagnostics(XdnaDevice & pDevice, const char * pWhen) {
  std::string                     vError;
  std::vector<XdnaContextStatus> aContexts;
  XdnaAsyncError                 vLast;
  if (pDevice.mQueryContexts(aContexts, vError)) {
    for (const XdnaContextStatus & vStatus : aContexts) {
      printf("%s: context %u (pid %" PRId64 "): columns %u to %u, commands submitted %" PRIu64 ", ended %" PRIu64
             ", errors %" PRIu64 ", PASID %u\n",
             pWhen, vStatus.vHandle, vStatus.vPid, vStatus.vStartColumn, vStatus.vStartColumn + vStatus.vColumns - 1,
             vStatus.vSubmitted, vStatus.vCompleted, vStatus.vErrors, vStatus.vPasid);
      printf("%s:   firmware health: control-code operation %u, program counter 0x%x, fatal type 0x%x, exception type 0x%x"
             " at 0x%x, module 0x%x\n",
             pWhen, vStatus.vTxnOpIndex, vStatus.vProgramCounter, vStatus.vFatalType, vStatus.vExceptionType,
             vStatus.vExceptionPc, vStatus.vAppModule);
    }
  } else {
    printf("%s: contexts not available (%s)\n", pWhen, vError.c_str());
  }
  if (pDevice.mQueryLastError(vLast, vError)) {
    printf("%s: last asynchronous error: code 0x%" PRIx64 ", extra code 0x%" PRIx64 ", time %" PRIu64 " us\n", pWhen,
           vLast.vCode, vLast.vExtraCode, vLast.vTimeUs);
  } else {
    printf("%s: last asynchronous error not available (%s)\n", pWhen, vError.c_str());
  }
}

static void fUsage(const char * pProgram) {
  printf("usage: %s --pdi <file> --insts <file> [options] (--in <file> | --out <bytes>[:<file>])...\n", pProgram);
  printf("\n");
  printf("  --pdi <file>            overlay of the design (main.pdi of an mlir-aie build)\n");
  printf("  --insts <file>          instruction stream of the design (insts.bin)\n");
  printf("  --in <file>             data buffer with the contents of a file\n");
  printf("  --out <bytes>[:<file>]  zeroed data buffer; written to the file after the last run\n");
  printf("  --columns <n>           columns of the partition the overlay was built for (default: what the\n");
  printf("                          partition_main.json next to the overlay says, or all of them)\n");
  printf("  --ops-per-cycle <n>     operations per cycle of the design, a hint for the driver (default: 2048)\n");
  printf("  --slots <n>             buffer slots of the kernel of the design (default: 5, as mlir-aie declares)\n");
  printf("  --repeat <n>            times to run the design (default: 1); 0 creates the context and runs nothing\n");
  printf("  --clear-outputs         zero the output buffers before every run, outside the time measured\n");
  printf("  --trace-every <n>       print the time of one run of every n (default: only the first four)\n");
  printf("  --timeout-ms <n>        how long to wait for a run (default: 10000)\n");
  printf("  --device <node>         device node of the NPU (default: /dev/accel/accel0)\n");
}

int main(int pArgc, char ** pArgv) {
  std::string            vPath        = "/dev/accel/accel0";
  std::string            vPdiPath;
  std::string            vInstsPath;
  uint32_t               vColumns     = 0;
  uint32_t               vOpsPerCycle = 2048;
  uint32_t               vSlots       = 5;
  int                    vRepeat      = 1;
  uint32_t               vTimeoutMs   = 10000;
  bool                   vClearOutputs = false;
  int                    vTraceEvery  = 0;
  std::vector<RunBuffer> aBuffers;

  for (int i = 1; i < pArgc; i++) {
    const std::string vArg  = pArgv[i];
    const bool        vMore = i + 1 < pArgc;
    if (vArg == "--pdi" && vMore) {
      vPdiPath = pArgv[++i];
    } else if (vArg == "--insts" && vMore) {
      vInstsPath = pArgv[++i];
    } else if (vArg == "--in" && vMore) {
      RunBuffer vBuffer;
      vBuffer.vIsInput = true;
      vBuffer.vFile    = pArgv[++i];
      aBuffers.push_back(vBuffer);
    } else if (vArg == "--out" && vMore) {
      RunBuffer         vBuffer;
      const std::string vValue = pArgv[++i];
      const size_t      vColon = vValue.find(':');
      vBuffer.vSize = std::strtoull(vValue.substr(0, vColon).c_str(), nullptr, 0);
      if (vColon != std::string::npos) {
        vBuffer.vFile = vValue.substr(vColon + 1);
      }
      aBuffers.push_back(vBuffer);
    } else if (vArg == "--columns" && vMore) {
      vColumns = (uint32_t) std::strtoul(pArgv[++i], nullptr, 0);
    } else if (vArg == "--ops-per-cycle" && vMore) {
      vOpsPerCycle = (uint32_t) std::strtoul(pArgv[++i], nullptr, 0);
    } else if (vArg == "--slots" && vMore) {
      vSlots = (uint32_t) std::strtoul(pArgv[++i], nullptr, 0);
    } else if (vArg == "--repeat" && vMore) {
      vRepeat = std::atoi(pArgv[++i]);
    } else if (vArg == "--clear-outputs") {
      vClearOutputs = true;
    } else if (vArg == "--trace-every" && vMore) {
      vTraceEvery = std::atoi(pArgv[++i]);
    } else if (vArg == "--timeout-ms" && vMore) {
      vTimeoutMs = (uint32_t) std::strtoul(pArgv[++i], nullptr, 0);
    } else if (vArg == "--device" && vMore) {
      vPath = pArgv[++i];
    } else {
      fUsage(pArgv[0]);
      return vArg == "-h" || vArg == "--help" ? 0 : 2;
    }
  }
  // the request of a compute unit carries at most 19 words: 5 fixed ones and 2 per buffer slot
  if (vPdiPath.empty() || vInstsPath.empty() || vRepeat < 0 || vSlots > 7 || aBuffers.size() > vSlots) {
    fUsage(pArgv[0]);
    return 2;
  }

  std::vector<uint8_t> aPdi;
  std::vector<uint8_t> aInsts;
  if (!fReadFile(vPdiPath, aPdi) || aPdi.empty()) {
    fprintf(stderr, "error: cannot read %s\n", vPdiPath.c_str());
    return 1;
  }
  if (!fReadFile(vInstsPath, aInsts) || aInsts.empty() || aInsts.size() % 4 != 0) {
    fprintf(stderr, "error: cannot read %s as 32-bit words\n", vInstsPath.c_str());
    return 1;
  }

  XdnaDevice vDevice;
  std::string vError;
  XdnaInfo   vInfo;
  if (!vDevice.mOpen(vPath, vError) || !vDevice.mQueryInfo(vInfo, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  // The context has to be as wide as the partition the overlay was built for, which is not the number of columns the
  // design computes in: a design for the whole array that uses 4 columns, run in a context of 4 columns, hung the NPU
  // and left it unusable (measured on 2026/11/12)
  const uint32_t vPartitionColumns = fPartitionColumns(vPdiPath);
  if (vColumns == 0) {
    vColumns = vPartitionColumns != 0 ? vPartitionColumns : vInfo.vColumns;
  }
  if (vPartitionColumns != 0 && vColumns != vPartitionColumns) {
    fprintf(stderr, "error: the overlay was built for a partition of %u columns; a context of %u would hang the NPU\n",
            vPartitionColumns, vColumns);
    return 1;
  }
  if (vColumns > vInfo.vColumns) {
    fprintf(stderr, "error: the design needs %u columns and the NPU has %u\n", vColumns, (unsigned) vInfo.vColumns);
    return 1;
  }
  printf("NPU: %u columns, firmware %u.%u.%u.%u; design: %u columns, PDI %zu bytes, %zu instruction words\n",
         (unsigned) vInfo.vColumns, vInfo.vFirmwareMajor, vInfo.vFirmwareMinor, vInfo.vFirmwarePatch, vInfo.vFirmwareBuild,
         vColumns, aPdi.size(), aInsts.size() / 4);

  XdnaBuffer vHeap;
  XdnaBuffer vPdi;
  XdnaBuffer vInsts;
  XdnaBuffer vCommand;
  XdnaContext vContext;
  int         vResult  = 1;

  // everything is released in one place, in the order the driver needs: the context, the pieces of the heap, the heap
  auto fRelease = [&]() {
    vDevice.mDestroyContext(vContext);
    for (RunBuffer & vBuffer : aBuffers) {
      vDevice.mDestroyBuffer(vBuffer.vBuffer);
    }
    vDevice.mDestroyBuffer(vCommand);
    vDevice.mDestroyBuffer(vInsts);
    vDevice.mDestroyBuffer(vPdi);
    vDevice.mDestroyBuffer(vHeap);
  };
  auto fFail = [&](const std::string & pWhat) {
    fprintf(stderr, "error: %s: %s\n", pWhat.c_str(), vError.c_str());
    fRelease();
    return 1;
  };

  // the device memory of the process, the overlay in it, the context and its configuration
  if (!vDevice.mCreateBuffer(XdnaBufferType::cDevHeap, 64ull << 20, vHeap, vError)) {
    return fFail("heap");
  }
  if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, aPdi.size(), vPdi, vError)) {
    return fFail("overlay buffer");
  }
  std::memcpy(vPdi.vMap, aPdi.data(), aPdi.size());
  if (!vDevice.mSyncBuffer(vPdi, 0, vPdi.vSize, vError)) {
    return fFail("overlay sync");
  }

  double vStart = fNowMs();
  if (!vDevice.mCreateContext(vColumns * vInfo.vCore.vRowCount, vOpsPerCycle, vContext, vError)) {
    return fFail("context");
  }
  const double vContextMs = fNowMs() - vStart;
  vStart = fNowMs();
  if (!vDevice.mConfigureContext(vContext, vPdi, 0, vError)) {
    return fFail("context configuration");
  }
  const double vConfigMs = fNowMs() - vStart;
  printf("context %u: created in %.2f ms, configured in %.2f ms\n", vContext.vHandle, vContextMs, vConfigMs);

  XdnaInfo vActive;
  if (vDevice.mQueryInfo(vActive, vError)) {
    printf("clocks with the context: H clock %u MHz, MP-NPU clock %u MHz\n", vActive.vHClockMhz, vActive.vNpuClockMhz);
  }

  // the instruction stream and the data
  if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, aInsts.size(), vInsts, vError)) {
    return fFail("instruction buffer");
  }
  std::memcpy(vInsts.vMap, aInsts.data(), aInsts.size());
  if (!vDevice.mSyncBuffer(vInsts, 0, vInsts.vSize, vError)) {
    return fFail("instruction sync");
  }

  std::vector<uint32_t> aHandles;
  aHandles.push_back(vInsts.vHandle);
  for (size_t i = 0; i < aBuffers.size(); i++) {
    RunBuffer &          vBuffer = aBuffers[i];
    std::vector<uint8_t> aData;
    if (vBuffer.vIsInput) {
      if (!fReadFile(vBuffer.vFile, aData) || aData.empty()) {
        vError = "cannot read " + vBuffer.vFile;
        return fFail("input");
      }
      vBuffer.vSize = aData.size();
    }
    if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vBuffer.vSize, vBuffer.vBuffer, vError)) {
      return fFail("data buffer");
    }
    std::memset(vBuffer.vBuffer.vMap, 0, vBuffer.vBuffer.vSize);
    if (vBuffer.vIsInput) {
      std::memcpy(vBuffer.vBuffer.vMap, aData.data(), aData.size());
    }
    if (!vDevice.mSyncBuffer(vBuffer.vBuffer, 0, vBuffer.vBuffer.vSize, vError)) {
      return fFail("data sync");
    }
    aHandles.push_back(vBuffer.vBuffer.vHandle);
    printf("buffer %zu: %s, %" PRIu64 " bytes, NPU address 0x%" PRIx64 "\n", i, vBuffer.vIsInput ? "input" : "output",
           vBuffer.vSize, vBuffer.vBuffer.vDevAddr);
  }

  // the command: header, mask of compute units and the register map of the kernel of an mlir-aie design: opcode (64
  // bits; 3 = run a transaction buffer), address of the instructions (64 bits), their number of words (32 bits) and
  // the address of every data buffer (64 bits each)
  if (!vDevice.mCreateBuffer(XdnaBufferType::cCommand, 4096, vCommand, vError)) {
    return fFail("command buffer");
  }
  uint32_t *     vWords = static_cast<uint32_t *>(vCommand.vMap);
  const uint32_t vCount = 1 + 5 + 2 * vSlots;
  std::memset(vWords, 0, 4096);
  vWords[1] = 0x1;
  vWords[2] = 3;
  vWords[3] = 0;
  vWords[4] = (uint32_t) (vInsts.vDevAddr & 0xffffffffu);
  vWords[5] = (uint32_t) (vInsts.vDevAddr >> 32);
  vWords[6] = (uint32_t) (aInsts.size() / 4);
  for (size_t i = 0; i < aBuffers.size(); i++) {
    vWords[7 + 2 * i] = (uint32_t) (aBuffers[i].vBuffer.vDevAddr & 0xffffffffu);
    vWords[8 + 2 * i] = (uint32_t) (aBuffers[i].vBuffer.vDevAddr >> 32);
  }

  printf("overlay at NPU address 0x%" PRIx64 ", instructions at 0x%" PRIx64 ", heap at 0x%" PRIx64 " (host %p)\n",
         vPdi.vDevAddr, vInsts.vDevAddr, vHeap.vDevAddr, vHeap.vMap);
  fPrintDiagnostics(vDevice, "before");

  std::vector<double> aTimes;
  bool                vCompleted = true;
  for (int vRun = 0; vRun < vRepeat && vCompleted; vRun++) {
    if (vClearOutputs) {
      // so that what is read at the end can only come from the last run. Not timed
      for (RunBuffer & vBuffer : aBuffers) {
        if (vBuffer.vIsInput) {
          continue;
        }
        std::memset(vBuffer.vBuffer.vMap, 0, vBuffer.vBuffer.vSize);
        if (!vDevice.mSyncBuffer(vBuffer.vBuffer, 0, vBuffer.vBuffer.vSize, vError)) {
          return fFail("output sync");
        }
      }
    }
    vWords[0] = fXdnaCommandHeader(cXdnaCommandStartCu, vCount);
    uint64_t vSeq = 0;
    vStart = fNowMs();
    if (!vDevice.mSubmit(vContext, vCommand, aHandles, vSeq, vError)) {
      return fFail("submit");
    }
    const XdnaWaitResult vWaited = vDevice.mWait(vContext, vSeq, vTimeoutMs, vError);
    const double          vMs     = fNowMs() - vStart;
    const uint32_t        vState  = fXdnaCommandState(vWords[0]);
    if (vWaited != XdnaWaitResult::cDone) {
      printf("run %d (command %" PRIu64 "): %s after %.2f ms (%s); command state: %s\n", vRun, vSeq,
             vWaited == XdnaWaitResult::cTimeout ? "no answer" : "the wait failed", vMs, vError.c_str(),
             fXdnaStateName(vState));
      vCompleted = false;
    } else if (vState != cXdnaStateCompleted) {
      printf("run %d: not completed after %.2f ms; command state: %s (first payload words: 0x%08x 0x%08x)\n", vRun, vMs,
             fXdnaStateName(vState), vWords[2], vWords[3]);
      vCompleted = false;
    } else {
      aTimes.push_back(vMs);
      if (vRun < 4 || (vTraceEvery > 0 && vRun % vTraceEvery == 0)) {
        printf("run %d: completed in %.3f ms\n", vRun, vMs);
      }
    }
  }

  if (!vCompleted) {
    fPrintDiagnostics(vDevice, "after");
  }

  if (vRepeat == 0) {
    printf("no runs asked for: the context and its buffers were created, and nothing was sent to the NPU\n");
    vResult = 0;
  } else if (vCompleted) {
    std::vector<double> aSorted = aTimes;
    std::sort(aSorted.begin(), aSorted.end());
    printf("%zu runs completed: fastest %.3f ms, median %.3f ms, slowest %.3f ms\n", aSorted.size(), aSorted.front(),
           aSorted[aSorted.size() / 2], aSorted.back());
    vResult = 0;
    for (RunBuffer & vBuffer : aBuffers) {
      if (vBuffer.vIsInput) {
        continue;
      }
      // what the NPU wrote: flush the cache before reading it
      if (!vDevice.mSyncBuffer(vBuffer.vBuffer, 0, vBuffer.vBuffer.vSize, vError)) {
        return fFail("output sync");
      }
      if (!vBuffer.vFile.empty()) {
        std::ofstream vFile(vBuffer.vFile, std::ios::binary);
        vFile.write(static_cast<const char *>(vBuffer.vBuffer.vMap), (std::streamsize) vBuffer.vSize);
        if (!vFile) {
          fprintf(stderr, "error: cannot write %s\n", vBuffer.vFile.c_str());
          vResult = 1;
        }
      }
    }
  }

  if (vDevice.mQueryInfo(vActive, vError)) {
    printf("clocks after the runs: H clock %u MHz, MP-NPU clock %u MHz\n", vActive.vHClockMhz, vActive.vNpuClockMhz);
  }
  fRelease();
  return vResult;
}
