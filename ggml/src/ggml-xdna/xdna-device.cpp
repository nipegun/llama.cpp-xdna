#include "xdna-device.h"

#include <cerrno>
#include <cstddef>
#include <cstring>
#include <ctime>

#include <cpuid.h>
#include <fcntl.h>
#include <immintrin.h>
#include <sys/ioctl.h>
#include <sys/mman.h>
#include <unistd.h>

// the interface of the amdxdna driver (uapi/amdxdna_accel.h, from Linux); it brings drm.h of the system
#include <amdxdna_accel.h>

namespace {

constexpr uintptr_t cCacheLine = 64;

// whether the processor has clflushopt (CPUID leaf 7, EBX bit 23), which flushes lines without ordering them against
// each other, as the driver uses it when it can
bool fHasFlushOpt() {
  unsigned vA = 0;
  unsigned vB = 0;
  unsigned vC = 0;
  unsigned vD = 0;
  return __get_cpuid_count(7, 0, &vA, &vB, &vC, &vD) != 0 && (vB & bit_CLFLUSHOPT) != 0;
}

__attribute__((target("clflushopt"))) void fFlushOpt(const uint8_t * pFirst, uint64_t pRecords, uint64_t pStride,
                                                     uint64_t pBytes) {
  for (uint64_t r = 0; r < pRecords; r++) {
    const uintptr_t vEnd = reinterpret_cast<uintptr_t>(pFirst + r * pStride) + pBytes;
    for (uintptr_t vLine = reinterpret_cast<uintptr_t>(pFirst + r * pStride) & ~(cCacheLine - 1); vLine < vEnd;
         vLine += cCacheLine) {
      _mm_clflushopt(reinterpret_cast<void *>(vLine));
    }
  }
}

void fFlushPlain(const uint8_t * pFirst, uint64_t pRecords, uint64_t pStride, uint64_t pBytes) {
  for (uint64_t r = 0; r < pRecords; r++) {
    const uintptr_t vEnd = reinterpret_cast<uintptr_t>(pFirst + r * pStride) + pBytes;
    for (uintptr_t vLine = reinterpret_cast<uintptr_t>(pFirst + r * pStride) & ~(cCacheLine - 1); vLine < vEnd;
         vLine += cCacheLine) {
      _mm_clflush(reinterpret_cast<void *>(vLine));
    }
  }
}

// the device memory of a process grows in units of this size (AIE2_DEVM_SIZE of the driver), and the firmware gets
// the address of every unit: the heap is mapped at a multiple of it
constexpr uint64_t cHeapUnit = 64ull << 20;

constexpr const char * cDriverName = "amdxdna_accel_driver";

std::string fErrno(const std::string & pWhat) {
  return pWhat + ": " + std::strerror(errno);
}

std::string fText(const unsigned char * pText, size_t pSize) {
  size_t vLength = 0;
  while (vLength < pSize && pText[vLength] != 0) {
    vLength++;
  }
  return std::string(reinterpret_cast<const char *>(pText), vLength);
}

XdnaTileInfo fTile(const amdxdna_drm_query_aie_tile_metadata & pTile) {
  XdnaTileInfo vTile;
  vTile.vRowCount    = pTile.row_count;
  vTile.vRowStart    = pTile.row_start;
  vTile.vDmaChannels = pTile.dma_channel_count;
  vTile.vLocks       = pTile.lock_count;
  vTile.vEventRegs   = pTile.event_reg_count;
  return vTile;
}

// DRM_AMDXDNA_HWCTX_CONFIG_CU with one compute unit: struct amdxdna_hwctx_param_config_cu ends in a flexible array,
// which C++ does not have, so its layout is spelled out here
struct ConfigOneCu {
  uint16_t          num_cus;
  uint16_t          pad[3];
  amdxdna_cu_config cu;
};
static_assert(offsetof(ConfigOneCu, cu) == sizeof(amdxdna_hwctx_param_config_cu), "layout of the CU configuration");

// glibc takes the request of ioctl as unsigned long and musl (POSIX) as int, which overflows on _IOWR constants
int fIoctl(int pFd, unsigned long pRequest, void * pArg) {
#if defined(__GLIBC__)
  return ::ioctl(pFd, pRequest, pArg);
#else
  return ::ioctl(pFd, static_cast<int>(pRequest), pArg);
#endif
}

} // namespace

uint32_t fXdnaCommandHeader(uint32_t pOpcode, uint32_t pWords) {
  // state: bits 0-3, extra compute-unit masks: bits 10-11 (none), words: bits 12-22, opcode: bits 23-27
  return (cXdnaStateNew & 0xfu) | ((pWords & 0x7ffu) << 12) | ((pOpcode & 0x1fu) << 23);
}

uint32_t fXdnaCommandState(uint32_t pHeader) {
  return pHeader & 0xfu;
}

void fXdnaFlushCache(const void * pFirst, uint64_t pRecords, uint64_t pStride, uint64_t pBytes) {
  static const bool cFlushOpt = fHasFlushOpt();
  if (pFirst == nullptr || pRecords == 0 || pBytes == 0) {
    return;
  }
  // clflushopt is ordered only by full barriers, before and after, as in drm_clflush_virt_range
  _mm_mfence();
  if (cFlushOpt) {
    fFlushOpt(static_cast<const uint8_t *>(pFirst), pRecords, pStride, pBytes);
  } else {
    fFlushPlain(static_cast<const uint8_t *>(pFirst), pRecords, pStride, pBytes);
  }
  _mm_mfence();
}

const char * fXdnaStateName(uint32_t pState) {
  switch (pState) {
    case 0:  return "invalid";
    case 1:  return "new";
    case 2:  return "queued";
    case 3:  return "running";
    case 4:  return "completed";
    case 5:  return "error";
    case 6:  return "abort";
    case 7:  return "submitted";
    case 8:  return "timeout";
    case 9:  return "no response";
    default: return "unknown";
  }
}

XdnaDevice::~XdnaDevice() {
  mClose();
}

bool XdnaDevice::mIoctl(unsigned long pRequest, void * pArg, const char * pWhat, std::string & pError) {
  if (vFd < 0) {
    pError = std::string(pWhat) + ": the device is not open";
    return false;
  }
  int vRet;
  int vAgain = 0;
  do {
    vRet = fIoctl(vFd, pRequest, pArg);
  } while (vRet == -1 && (errno == EINTR || (errno == EAGAIN && ++vAgain < 64)));
  if (vRet == -1) {
    pError = fErrno(pWhat);
    return false;
  }
  return true;
}

bool XdnaDevice::mGetInfo(uint32_t pParam, void * pBuffer, uint32_t pSize, const char * pWhat, std::string & pError) {
  amdxdna_drm_get_info vArgs;
  std::memset(&vArgs, 0, sizeof(vArgs));
  vArgs.param       = pParam;
  vArgs.buffer_size = pSize;
  vArgs.buffer      = (uint64_t) (uintptr_t) pBuffer;
  return mIoctl(DRM_IOCTL_AMDXDNA_GET_INFO, &vArgs, pWhat, pError);
}

bool XdnaDevice::mOpen(const std::string & pPath, std::string & pError) {
  mClose();
  vFd = ::open(pPath.c_str(), O_RDWR | O_CLOEXEC);
  if (vFd < 0) {
    pError = fErrno("cannot open " + pPath);
    return false;
  }
  char aName[64] = { 0 };
  drm_version vVersion;
  std::memset(&vVersion, 0, sizeof(vVersion));
  vVersion.name     = aName;
  vVersion.name_len = sizeof(aName) - 1;
  if (!mIoctl(DRM_IOCTL_VERSION, &vVersion, "DRM_IOCTL_VERSION", pError)) {
    mClose();
    return false;
  }
  if (std::strcmp(aName, cDriverName) != 0) {
    pError = pPath + " is not an AMD XDNA NPU (its driver is \"" + aName + "\")";
    mClose();
    return false;
  }
  return true;
}

void XdnaDevice::mClose() {
  if (vFd >= 0) {
    ::close(vFd);
    vFd = -1;
  }
}

bool XdnaDevice::mQueryInfo(XdnaInfo & pInfo, std::string & pError) {
  pInfo = XdnaInfo();

  char aName[64] = { 0 };
  drm_version vVersion;
  std::memset(&vVersion, 0, sizeof(vVersion));
  vVersion.name     = aName;
  vVersion.name_len = sizeof(aName) - 1;
  if (!mIoctl(DRM_IOCTL_VERSION, &vVersion, "DRM_IOCTL_VERSION", pError)) {
    return false;
  }
  pInfo.vDriverName  = aName;
  pInfo.vDriverMajor = vVersion.version_major;
  pInfo.vDriverMinor = vVersion.version_minor;
  pInfo.vDriverPatch = vVersion.version_patchlevel;

  amdxdna_drm_query_firmware_version vFirmware;
  std::memset(&vFirmware, 0, sizeof(vFirmware));
  if (!mGetInfo(DRM_AMDXDNA_QUERY_FIRMWARE_VERSION, &vFirmware, sizeof(vFirmware), "firmware version", pError)) {
    return false;
  }
  pInfo.vFirmwareMajor = vFirmware.major;
  pInfo.vFirmwareMinor = vFirmware.minor;
  pInfo.vFirmwarePatch = vFirmware.patch;
  pInfo.vFirmwareBuild = vFirmware.build;

  amdxdna_drm_query_aie_metadata vMetadata;
  std::memset(&vMetadata, 0, sizeof(vMetadata));
  if (!mGetInfo(DRM_AMDXDNA_QUERY_AIE_METADATA, &vMetadata, sizeof(vMetadata), "AIE metadata", pError)) {
    return false;
  }
  pInfo.vAieMajor         = vMetadata.version.major;
  pInfo.vAieMinor         = vMetadata.version.minor;
  pInfo.vColumns          = vMetadata.cols;
  pInfo.vRows             = vMetadata.rows;
  pInfo.vColumnStatusSize = vMetadata.col_size;
  pInfo.vCore             = fTile(vMetadata.core);
  pInfo.vMem              = fTile(vMetadata.mem);
  pInfo.vShim             = fTile(vMetadata.shim);

  amdxdna_drm_query_clock_metadata vClocks;
  std::memset(&vClocks, 0, sizeof(vClocks));
  if (!mGetInfo(DRM_AMDXDNA_QUERY_CLOCK_METADATA, &vClocks, sizeof(vClocks), "clocks", pError)) {
    return false;
  }
  pInfo.vNpuClockMhz = vClocks.mp_npu_clock.freq_mhz;
  pInfo.vHClockMhz   = vClocks.h_clock.freq_mhz;

  amdxdna_drm_get_power_mode vPower;
  std::memset(&vPower, 0, sizeof(vPower));
  if (!mGetInfo(DRM_AMDXDNA_GET_POWER_MODE, &vPower, sizeof(vPower), "power mode", pError)) {
    return false;
  }
  pInfo.vPowerMode = vPower.power_mode;

  amdxdna_drm_get_resource_info vResources;
  std::memset(&vResources, 0, sizeof(vResources));
  if (!mGetInfo(DRM_AMDXDNA_QUERY_RESOURCE_INFO, &vResources, sizeof(vResources), "resources", pError)) {
    return false;
  }
  pInfo.vHClockMaxMhz = vResources.npu_clk_max;
  pInfo.vTopsMax      = vResources.npu_tops_max;
  pInfo.vTopsNow      = vResources.npu_tops_curr;
  pInfo.vContextsMax  = vResources.npu_task_max;
  pInfo.vContextsNow  = vResources.npu_task_curr;
  return true;
}

bool XdnaDevice::mQuerySensors(std::vector<XdnaSensor> & pSensors, std::string & pError) {
  pSensors.clear();
  // the total power and the use of every column
  std::vector<amdxdna_drm_query_sensor> aSensors(64);
  std::memset(aSensors.data(), 0, aSensors.size() * sizeof(aSensors[0]));
  amdxdna_drm_get_info vArgs;
  std::memset(&vArgs, 0, sizeof(vArgs));
  vArgs.param       = DRM_AMDXDNA_QUERY_SENSORS;
  vArgs.buffer_size = (uint32_t) (aSensors.size() * sizeof(aSensors[0]));
  vArgs.buffer      = (uint64_t) (uintptr_t) aSensors.data();
  if (!mIoctl(DRM_IOCTL_AMDXDNA_GET_INFO, &vArgs, "sensors", pError)) {
    return false;
  }
  // the driver returns the size it wrote
  const size_t vCount = vArgs.buffer_size / sizeof(aSensors[0]);
  for (size_t i = 0; i < vCount && i < aSensors.size(); i++) {
    XdnaSensor vSensor;
    vSensor.vLabel        = fText(aSensors[i].label, sizeof(aSensors[i].label));
    vSensor.vUnits        = fText(aSensors[i].units, sizeof(aSensors[i].units));
    vSensor.vValue        = aSensors[i].input;
    vSensor.vUnitExponent = aSensors[i].unitm;
    vSensor.vIsPower      = aSensors[i].type == AMDXDNA_SENSOR_TYPE_POWER;
    pSensors.push_back(vSensor);
  }
  return true;
}

bool XdnaDevice::mQueryContexts(std::vector<XdnaContextStatus> & pContexts, std::string & pError) {
  pContexts.clear();
  std::vector<amdxdna_drm_hwctx_entry> aEntries(64);
  std::memset(aEntries.data(), 0, aEntries.size() * sizeof(aEntries[0]));
  amdxdna_drm_get_array vArgs;
  std::memset(&vArgs, 0, sizeof(vArgs));
  vArgs.param        = DRM_AMDXDNA_HW_CONTEXT_ALL;
  vArgs.element_size = sizeof(aEntries[0]);
  vArgs.num_element  = (uint32_t) aEntries.size();
  vArgs.buffer       = (uint64_t) (uintptr_t) aEntries.data();
  if (!mIoctl(DRM_IOCTL_AMDXDNA_GET_ARRAY, &vArgs, "contexts", pError)) {
    return false;
  }
  // the driver returns how many entries it wrote and their size, which is smaller with an older driver
  const size_t    vElement = vArgs.element_size;
  const uint8_t * vBytes   = reinterpret_cast<const uint8_t *>(aEntries.data());
  if (vElement == 0 || vElement > sizeof(aEntries[0])) {
    pError = "contexts: unexpected entry size";
    return false;
  }
  for (size_t i = 0; i < vArgs.num_element && i < aEntries.size(); i++) {
    amdxdna_drm_hwctx_entry vEntry;
    std::memset(&vEntry, 0, sizeof(vEntry));
    std::memcpy(&vEntry, vBytes + i * vElement, vElement);
    XdnaContextStatus vStatus;
    vStatus.vHandle         = vEntry.context_id;
    vStatus.vPid            = vEntry.pid;
    vStatus.vStartColumn    = vEntry.start_col;
    vStatus.vColumns        = vEntry.num_col;
    vStatus.vSubmitted      = vEntry.command_submissions;
    vStatus.vCompleted      = vEntry.command_completions;
    vStatus.vErrors         = vEntry.errors;
    vStatus.vPasid          = vEntry.pasid;
    vStatus.vTxnOpIndex     = vEntry.txn_op_idx;
    vStatus.vProgramCounter = vEntry.ctx_pc;
    vStatus.vFatalType      = vEntry.fatal_error_type;
    vStatus.vExceptionType  = vEntry.fatal_error_exception_type;
    vStatus.vExceptionPc    = vEntry.fatal_error_exception_pc;
    vStatus.vAppModule      = vEntry.fatal_error_app_module;
    pContexts.push_back(vStatus);
  }
  return true;
}

bool XdnaDevice::mQueryLastError(XdnaAsyncError & pAsyncError, std::string & pError) {
  pAsyncError = XdnaAsyncError();
  amdxdna_async_error vLast;
  std::memset(&vLast, 0, sizeof(vLast));
  amdxdna_drm_get_array vArgs;
  std::memset(&vArgs, 0, sizeof(vArgs));
  vArgs.param        = DRM_AMDXDNA_HW_LAST_ASYNC_ERR;
  vArgs.element_size = sizeof(vLast);
  vArgs.num_element  = 1;
  vArgs.buffer       = (uint64_t) (uintptr_t) &vLast;
  if (!mIoctl(DRM_IOCTL_AMDXDNA_GET_ARRAY, &vArgs, "last asynchronous error", pError)) {
    return false;
  }
  pAsyncError.vCode      = vLast.err_code;
  pAsyncError.vTimeUs    = vLast.ts_us;
  pAsyncError.vExtraCode = vLast.ex_err_code;
  return true;
}

bool XdnaDevice::mCreateBuffer(XdnaBufferType pType, uint64_t pSize, XdnaBuffer & pBuffer, std::string & pError) {
  pBuffer = XdnaBuffer();
  const uint64_t vPage = (uint64_t) ::sysconf(_SC_PAGESIZE);
  const uint64_t vSize = (pSize + vPage - 1) / vPage * vPage;
  if (vSize == 0) {
    pError = "a buffer cannot be empty";
    return false;
  }
  if (pType == XdnaBufferType::cDevHeap && vSize % cHeapUnit != 0) {
    pError = "the size of the heap has to be a multiple of 64 MiB";
    return false;
  }

  amdxdna_drm_create_bo vCreate;
  std::memset(&vCreate, 0, sizeof(vCreate));
  vCreate.size = vSize;
  vCreate.type = (uint32_t) pType;
  if (!mIoctl(DRM_IOCTL_AMDXDNA_CREATE_BO, &vCreate, "create buffer", pError)) {
    return false;
  }
  pBuffer.vHandle = vCreate.handle;
  pBuffer.vType   = pType;
  pBuffer.vSize   = vSize;

  amdxdna_drm_get_bo_info vInfo;
  std::memset(&vInfo, 0, sizeof(vInfo));
  vInfo.handle = pBuffer.vHandle;
  if (!mIoctl(DRM_IOCTL_AMDXDNA_GET_BO_INFO, &vInfo, "buffer info", pError)) {
    mDestroyBuffer(pBuffer);
    return false;
  }

  if (pType == XdnaBufferType::cDev) {
    // a piece of the heap: it is already mapped, inside the mapping of the heap
    if (vInfo.vaddr == 0) {
      pError = "the heap is not mapped: create it before the device buffers";
      mDestroyBuffer(pBuffer);
      return false;
    }
    pBuffer.vMap     = (void *) (uintptr_t) vInfo.vaddr;
    pBuffer.vDevAddr = vInfo.xdna_addr;
    return true;
  }

  void * vTarget = nullptr;
  int    vFlags  = MAP_SHARED;
  if (pType == XdnaBufferType::cDevHeap) {
    // reserve address space and take the multiple of 64 MiB inside it
    const uint64_t vReserve = vSize + cHeapUnit;
    void *         vBase    = ::mmap(nullptr, vReserve, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS | MAP_NORESERVE, -1, 0);
    if (vBase == MAP_FAILED) {
      pError = fErrno("cannot reserve address space for the heap");
      mDestroyBuffer(pBuffer);
      return false;
    }
    pBuffer.vUnmapBase = vBase;
    pBuffer.vUnmapSize = vReserve;
    vTarget = (void *) (((uintptr_t) vBase + cHeapUnit - 1) & ~((uintptr_t) cHeapUnit - 1));
    vFlags |= MAP_FIXED;
  }
  void * vMap = ::mmap(vTarget, vSize, PROT_READ | PROT_WRITE, vFlags, vFd, (off_t) vInfo.map_offset);
  if (vMap == MAP_FAILED) {
    pError = fErrno("cannot map the buffer");
    mDestroyBuffer(pBuffer);
    return false;
  }
  pBuffer.vMap = vMap;
  if (pBuffer.vUnmapBase == nullptr) {
    pBuffer.vUnmapBase = vMap;
    pBuffer.vUnmapSize = vSize;
  }

  // the driver sets the address the NPU uses when the buffer is first mapped
  std::memset(&vInfo, 0, sizeof(vInfo));
  vInfo.handle = pBuffer.vHandle;
  if (!mIoctl(DRM_IOCTL_AMDXDNA_GET_BO_INFO, &vInfo, "buffer info", pError)) {
    mDestroyBuffer(pBuffer);
    return false;
  }
  pBuffer.vDevAddr = vInfo.xdna_addr;
  return true;
}

bool XdnaDevice::mSyncBuffer(const XdnaBuffer & pBuffer, uint64_t pOffset, uint64_t pSize, std::string & pError) {
  amdxdna_drm_sync_bo vSync;
  std::memset(&vSync, 0, sizeof(vSync));
  vSync.handle    = pBuffer.vHandle;
  vSync.direction = SYNC_DIRECT_TO_DEVICE;
  vSync.offset    = pOffset;
  vSync.size      = pSize;
  return mIoctl(DRM_IOCTL_AMDXDNA_SYNC_BO, &vSync, "sync buffer", pError);
}

void XdnaDevice::mDestroyBuffer(XdnaBuffer & pBuffer) {
  if (pBuffer.vUnmapBase != nullptr) {
    ::munmap(pBuffer.vUnmapBase, pBuffer.vUnmapSize);
  }
  if (pBuffer.vHandle != 0 && vFd >= 0) {
    drm_gem_close vClose;
    std::memset(&vClose, 0, sizeof(vClose));
    vClose.handle = pBuffer.vHandle;
    fIoctl(vFd, DRM_IOCTL_GEM_CLOSE, &vClose);
  }
  pBuffer = XdnaBuffer();
}

bool XdnaDevice::mCreateContext(uint32_t pTiles, uint32_t pOpsPerCycle, XdnaContext & pContext, std::string & pError) {
  pContext = XdnaContext();
  // all zeros: no hints. The driver gives the highest clock level to a context without them
  amdxdna_qos_info vQos;
  std::memset(&vQos, 0, sizeof(vQos));
  amdxdna_drm_create_hwctx vCreate;
  std::memset(&vCreate, 0, sizeof(vCreate));
  vCreate.qos_p     = (uint64_t) (uintptr_t) &vQos;
  vCreate.max_opc   = pOpsPerCycle;
  vCreate.num_tiles = pTiles;
  if (!mIoctl(DRM_IOCTL_AMDXDNA_CREATE_HWCTX, &vCreate, "create context", pError)) {
    return false;
  }
  pContext.vHandle     = vCreate.handle;
  pContext.vSyncHandle = vCreate.syncobj_handle;
  return true;
}

bool XdnaDevice::mConfigureContext(const XdnaContext & pContext, const XdnaBuffer & pOverlay, uint8_t pFunction,
                                    std::string & pError) {
  if (pOverlay.vType != XdnaBufferType::cDev) {
    pError = "the overlay has to be in a device buffer";
    return false;
  }
  ConfigOneCu vParam;
  std::memset(&vParam, 0, sizeof(vParam));
  vParam.num_cus    = 1;
  vParam.cu.cu_bo   = pOverlay.vHandle;
  vParam.cu.cu_func = pFunction;

  amdxdna_drm_config_hwctx vConfig;
  std::memset(&vConfig, 0, sizeof(vConfig));
  vConfig.handle         = pContext.vHandle;
  vConfig.param_type     = DRM_AMDXDNA_HWCTX_CONFIG_CU;
  vConfig.param_val      = (uint64_t) (uintptr_t) &vParam;
  vConfig.param_val_size = sizeof(vParam);
  return mIoctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &vConfig, "configure context", pError);
}

bool XdnaDevice::mConfigureContextUnits(const XdnaContext & pContext, const std::vector<const XdnaBuffer *> & pOverlays,
                                         uint8_t pFunction, std::string & pError) {
  // the driver takes at most MAX_NUM_CUS (32, aie2_msg_priv.h) compute units
  if (pOverlays.empty() || pOverlays.size() > 32) {
    pError = "a context takes from 1 to 32 compute units";
    return false;
  }
  for (const XdnaBuffer * vOverlay : pOverlays) {
    if (vOverlay == nullptr || vOverlay->vType != XdnaBufferType::cDev) {
      pError = "every overlay has to be in a device buffer";
      return false;
    }
  }
  // struct amdxdna_hwctx_param_config_cu and then one struct amdxdna_cu_config per compute unit
  std::vector<uint8_t> aParam(sizeof(amdxdna_hwctx_param_config_cu) + pOverlays.size() * sizeof(amdxdna_cu_config), 0);
  amdxdna_hwctx_param_config_cu vHeader;
  std::memset(&vHeader, 0, sizeof(vHeader));
  vHeader.num_cus = (uint16_t) pOverlays.size();
  std::memcpy(aParam.data(), &vHeader, sizeof(vHeader));
  for (size_t i = 0; i < pOverlays.size(); i++) {
    amdxdna_cu_config vUnit;
    std::memset(&vUnit, 0, sizeof(vUnit));
    vUnit.cu_bo   = pOverlays[i]->vHandle;
    vUnit.cu_func = pFunction;
    std::memcpy(aParam.data() + sizeof(vHeader) + i * sizeof(vUnit), &vUnit, sizeof(vUnit));
  }

  amdxdna_drm_config_hwctx vConfig;
  std::memset(&vConfig, 0, sizeof(vConfig));
  vConfig.handle         = pContext.vHandle;
  vConfig.param_type     = DRM_AMDXDNA_HWCTX_CONFIG_CU;
  vConfig.param_val      = (uint64_t) (uintptr_t) aParam.data();
  vConfig.param_val_size = (uint32_t) aParam.size();
  return mIoctl(DRM_IOCTL_AMDXDNA_CONFIG_HWCTX, &vConfig, "configure context", pError);
}

void XdnaDevice::mDestroyContext(XdnaContext & pContext) {
  if (vFd >= 0 && pContext.vHandle != 0) {
    amdxdna_drm_destroy_hwctx vDestroy;
    std::memset(&vDestroy, 0, sizeof(vDestroy));
    vDestroy.handle = pContext.vHandle;
    fIoctl(vFd, DRM_IOCTL_AMDXDNA_DESTROY_HWCTX, &vDestroy);
  }
  // the driver leaves the synchronisation object of the context to the process
  if (vFd >= 0 && pContext.vSyncHandle != 0) {
    drm_syncobj_destroy vDestroy;
    std::memset(&vDestroy, 0, sizeof(vDestroy));
    vDestroy.handle = pContext.vSyncHandle;
    fIoctl(vFd, DRM_IOCTL_SYNCOBJ_DESTROY, &vDestroy);
  }
  pContext = XdnaContext();
}

bool XdnaDevice::mSubmit(const XdnaContext & pContext, const XdnaBuffer & pCommand,
                          const std::vector<uint32_t> & pArgHandles, uint64_t & pSeq, std::string & pError) {
  pSeq = 0;
  amdxdna_drm_exec_cmd vExec;
  std::memset(&vExec, 0, sizeof(vExec));
  vExec.hwctx       = pContext.vHandle;
  vExec.type        = AMDXDNA_CMD_SUBMIT_EXEC_BUF;
  vExec.cmd_handles = pCommand.vHandle;
  vExec.args        = (uint64_t) (uintptr_t) pArgHandles.data();
  vExec.cmd_count   = 1;
  vExec.arg_count   = (uint32_t) pArgHandles.size();
  if (!mIoctl(DRM_IOCTL_AMDXDNA_EXEC_CMD, &vExec, "submit command", pError)) {
    return false;
  }
  pSeq = vExec.seq;
  return true;
}

XdnaWaitResult XdnaDevice::mWait(const XdnaContext & pContext, uint64_t pSeq, uint32_t pTimeoutMs, std::string & pError) {
  // The driver signals the end of the command number N of a context at the point N of the timeline of the
  // synchronisation object of the context, and that is where to wait: its own ioctl, DRM_IOCTL_AMDXDNA_WAIT_CMD, is
  // not implemented for this generation of NPU (measured: EOPNOTSUPP).
  // The point 0 of a timeline means its last point, so waiting for the command number 0 waits for every command
  // submitted until then.
  if (vFd < 0 || pContext.vSyncHandle == 0) {
    pError = "wait for command: there is no context";
    return XdnaWaitResult::cFailed;
  }
  // the limit is an absolute time of the monotonic clock
  int64_t vDeadline = INT64_MAX;
  if (pTimeoutMs != 0) {
    timespec vNow;
    ::clock_gettime(CLOCK_MONOTONIC, &vNow);
    vDeadline = (int64_t) vNow.tv_sec * 1000000000ll + (int64_t) vNow.tv_nsec + (int64_t) pTimeoutMs * 1000000ll;
  }
  uint32_t vHandle = pContext.vSyncHandle;
  uint64_t vPoint  = pSeq;
  drm_syncobj_timeline_wait vWait;
  std::memset(&vWait, 0, sizeof(vWait));
  vWait.handles       = (uint64_t) (uintptr_t) &vHandle;
  vWait.points        = (uint64_t) (uintptr_t) &vPoint;
  vWait.count_handles = 1;
  int vRet;
  // First it asks without sleeping, for as long as most commands take: a thread that sleeps until the end of a
  // command wakes up some tens of microseconds after it, and more the longer it slept (measured on 2026/11/13: the
  // median of a command of 512 x 4096 x 4096 went from 0.90 to 0.75 ms with a thread kept busy meanwhile). A
  // limit of 0 asks once and returns at once.
  const int64_t cSpinNs = 2000000;
  timespec      vStart;
  ::clock_gettime(CLOCK_MONOTONIC, &vStart);
  for (;;) {
    vWait.timeout_nsec = 0;
    vRet = fIoctl(vFd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &vWait);
    if (vRet == 0) {
      return XdnaWaitResult::cDone;
    }
    if (errno != ETIME && errno != EINTR && errno != EAGAIN) {
      break;
    }
    timespec vNow;
    ::clock_gettime(CLOCK_MONOTONIC, &vNow);
    const int64_t vSpent = ((int64_t) vNow.tv_sec - vStart.tv_sec) * 1000000000ll + (vNow.tv_nsec - vStart.tv_nsec);
    if (vSpent >= cSpinNs) {
      break;
    }
  }
  // then it sleeps until the end, or until the limit
  vWait.timeout_nsec = vDeadline;
  do {
    vRet = fIoctl(vFd, DRM_IOCTL_SYNCOBJ_TIMELINE_WAIT, &vWait);
  } while (vRet == -1 && (errno == EINTR || errno == EAGAIN));
  if (vRet == 0) {
    return XdnaWaitResult::cDone;
  }
  if (errno == ETIME) {
    pError = "wait for command: it did not end in " + std::to_string(pTimeoutMs) + " ms";
    return XdnaWaitResult::cTimeout;
  }
  pError = fErrno("wait for command");
  return XdnaWaitResult::cFailed;
}
