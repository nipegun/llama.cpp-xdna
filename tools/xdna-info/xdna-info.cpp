// llama-xdna-info: what the amdxdna driver of Linux reports about the NPU of an AMD Ryzen AI processor, and a check
// of the buffers of the driver. It runs nothing in the NPU: it creates no hardware context and submits no command.

#include "xdna-device.h"

#include <cinttypes>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <sys/resource.h>

static const char * fPowerModeName(uint32_t pMode) {
  switch (pMode) {
    case 0:  return "default";
    case 1:  return "low";
    case 2:  return "medium";
    case 3:  return "high";
    case 4:  return "turbo";
    default: return "unknown";
  }
}

static void fPrintTile(const char * pName, const XdnaTileInfo & pTile) {
  printf("  %-13s rows %u to %u, %u DMA channels, %u locks\n", pName, (unsigned) pTile.vRowStart,
         (unsigned) (pTile.vRowStart + pTile.vRowCount - 1), (unsigned) pTile.vDmaChannels, (unsigned) pTile.vLocks);
}

static void fUsage(const char * pProgram) {
  printf("usage: %s [--device <node>] [--check-buffers]\n", pProgram);
  printf("\n");
  printf("  --device <node>   device node of the NPU (default: /dev/accel/accel0)\n");
  printf("  --check-buffers   also create, write, sync and destroy one buffer of every type of the driver\n");
  printf("\n");
  printf("It runs nothing in the NPU.\n");
}

static bool fPrintInfo(XdnaDevice & pDevice) {
  std::string vError;
  XdnaInfo   vInfo;
  if (!pDevice.mQueryInfo(vInfo, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return false;
  }
  printf("driver:        %s %d.%d.%d\n", vInfo.vDriverName.c_str(), vInfo.vDriverMajor, vInfo.vDriverMinor, vInfo.vDriverPatch);
  printf("firmware:      %u.%u.%u.%u\n", vInfo.vFirmwareMajor, vInfo.vFirmwareMinor, vInfo.vFirmwarePatch, vInfo.vFirmwareBuild);
  printf("AIE version:   %u.%u\n", vInfo.vAieMajor, vInfo.vAieMinor);
  printf("array:         %u columns x %u rows\n", (unsigned) vInfo.vColumns, (unsigned) vInfo.vRows);
  fPrintTile("core tiles:", vInfo.vCore);
  fPrintTile("memory tiles:", vInfo.vMem);
  fPrintTile("shim tiles:", vInfo.vShim);
  printf("clocks now:    H clock %u MHz, MP-NPU clock %u MHz (they drop when the NPU is idle)\n", vInfo.vHClockMhz,
         vInfo.vNpuClockMhz);
  printf("maximum:       H clock %" PRIu64 " MHz, %" PRIu64 " TOPS\n", vInfo.vHClockMaxMhz, vInfo.vTopsMax);
  printf("power mode:    %s\n", fPowerModeName(vInfo.vPowerMode));
  printf("contexts:      %" PRIu64 " of %" PRIu64 " in use\n", vInfo.vContextsNow, vInfo.vContextsMax);

  std::vector<XdnaSensor> aSensors;
  if (!pDevice.mQuerySensors(aSensors, vError)) {
    printf("sensors:       not available (%s)\n", vError.c_str());
    return true;
  }
  for (const XdnaSensor & vSensor : aSensors) {
    if (vSensor.vIsPower) {
      printf("power:         %u %s\n", vSensor.vValue, vSensor.vUnits.c_str());
    }
  }
  printf("column use:   ");
  for (const XdnaSensor & vSensor : aSensors) {
    if (!vSensor.vIsPower) {
      printf(" %u%s", vSensor.vValue, vSensor.vUnits.c_str());
    }
  }
  printf("\n");
  return true;
}

// fills a buffer, syncs it and compares it
static bool fExercise(XdnaDevice & pDevice, const char * pName, XdnaBuffer & pBuffer) {
  std::string vError;
  uint8_t *   vBytes = static_cast<uint8_t *>(pBuffer.vMap);
  printf("  %-8s handle %u, %" PRIu64 " KiB, host %p, NPU address 0x%" PRIx64 "\n", pName, pBuffer.vHandle,
         pBuffer.vSize / 1024, pBuffer.vMap, pBuffer.vDevAddr);
  for (uint64_t i = 0; i < pBuffer.vSize; i++) {
    vBytes[i] = (uint8_t) (i * 31 + 7);
  }
  if (!pDevice.mSyncBuffer(pBuffer, 0, pBuffer.vSize, vError)) {
    printf("           sync failed: %s\n", vError.c_str());
    return false;
  }
  for (uint64_t i = 0; i < pBuffer.vSize; i++) {
    if (vBytes[i] != (uint8_t) (i * 31 + 7)) {
      printf("           the contents changed at byte %" PRIu64 "\n", i);
      return false;
    }
  }
  return true;
}

static bool fCheckBuffers(XdnaDevice & pDevice) {
  std::string vError;
  bool        vOk = true;

  struct rlimit vLimit;
  if (getrlimit(RLIMIT_MEMLOCK, &vLimit) == 0) {
    if (vLimit.rlim_cur == RLIM_INFINITY) {
      printf("locked-memory limit of this process: none\n");
    } else {
      printf("locked-memory limit of this process: %" PRIu64 " KiB\n", (uint64_t) vLimit.rlim_cur / 1024);
    }
  }

  printf("buffers:\n");
  XdnaBuffer vHeap;
  XdnaBuffer vDev;
  XdnaBuffer vShare;
  XdnaBuffer vCommand;

  // the heap first: the device buffers are pieces of it
  if (!pDevice.mCreateBuffer(XdnaBufferType::cDevHeap, 64ull << 20, vHeap, vError)) {
    printf("  heap:    %s\n", vError.c_str());
    return false;
  }
  printf("  %-8s handle %u, %" PRIu64 " KiB, host %p, NPU address 0x%" PRIx64 "\n", "heap", vHeap.vHandle, vHeap.vSize / 1024,
         vHeap.vMap, vHeap.vDevAddr);

  if (pDevice.mCreateBuffer(XdnaBufferType::cDev, 64 << 10, vDev, vError)) {
    vOk = fExercise(pDevice, "device", vDev) && vOk;
    const uint8_t * vStart = static_cast<const uint8_t *>(vHeap.vMap);
    const uint8_t * vMap   = static_cast<const uint8_t *>(vDev.vMap);
    if (vMap < vStart || vMap + vDev.vSize > vStart + vHeap.vSize) {
      printf("           it is not inside the mapping of the heap\n");
      vOk = false;
    }
  } else {
    printf("  device:  %s\n", vError.c_str());
    vOk = false;
  }

  // larger than the usual 8 MiB limit of locked memory, to see whether the buffers of the driver count against it
  if (pDevice.mCreateBuffer(XdnaBufferType::cShare, 16ull << 20, vShare, vError)) {
    vOk = fExercise(pDevice, "share", vShare) && vOk;
  } else {
    printf("  share:   %s\n", vError.c_str());
    vOk = false;
  }

  if (pDevice.mCreateBuffer(XdnaBufferType::cCommand, 4096, vCommand, vError)) {
    vOk = fExercise(pDevice, "command", vCommand) && vOk;
  } else {
    printf("  command: %s\n", vError.c_str());
    vOk = false;
  }

  // the device buffer before the heap it lives in
  pDevice.mDestroyBuffer(vCommand);
  pDevice.mDestroyBuffer(vShare);
  pDevice.mDestroyBuffer(vDev);
  pDevice.mDestroyBuffer(vHeap);
  printf("buffers: %s\n", vOk ? "ok" : "FAILED");
  return vOk;
}

int main(int pArgc, char ** pArgv) {
  std::string vPath         = "/dev/accel/accel0";
  bool        vCheckBuffers = false;
  for (int i = 1; i < pArgc; i++) {
    if (std::strcmp(pArgv[i], "--device") == 0 && i + 1 < pArgc) {
      vPath = pArgv[++i];
    } else if (std::strcmp(pArgv[i], "--check-buffers") == 0) {
      vCheckBuffers = true;
    } else if (std::strcmp(pArgv[i], "-h") == 0 || std::strcmp(pArgv[i], "--help") == 0) {
      fUsage(pArgv[0]);
      return 0;
    } else {
      fUsage(pArgv[0]);
      return 2;
    }
  }

  XdnaDevice vDevice;
  std::string vError;
  if (!vDevice.mOpen(vPath, vError)) {
    fprintf(stderr, "error: %s\n", vError.c_str());
    return 1;
  }
  printf("device:        %s\n", vPath.c_str());
  if (!fPrintInfo(vDevice)) {
    return 1;
  }
  if (vCheckBuffers && !fCheckBuffers(vDevice)) {
    return 1;
  }
  return 0;
}
