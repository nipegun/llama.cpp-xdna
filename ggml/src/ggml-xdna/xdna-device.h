#pragma once

// Direct access to the NPU of AMD Ryzen AI processors (XDNA) through the amdxdna DRM accel driver of Linux, without
// XRT. The interface of the driver, and what was measured through it, is described in doc/XDNA2.md.

#include <cstdint>
#include <string>
#include <vector>

// one kind of tile of the array (core, memory or shim)
struct XdnaTileInfo {
  uint16_t vRowCount    = 0;
  uint16_t vRowStart    = 0;
  uint16_t vDmaChannels = 0;
  uint16_t vLocks       = 0;
  uint16_t vEventRegs   = 0;
};

// what the driver reports about the NPU
struct XdnaInfo {
  std::string   vDriverName;
  int           vDriverMajor      = 0;
  int           vDriverMinor      = 0;
  int           vDriverPatch      = 0;
  uint32_t      vAieMajor         = 0;
  uint32_t      vAieMinor         = 0;
  uint32_t      vFirmwareMajor    = 0;
  uint32_t      vFirmwareMinor    = 0;
  uint32_t      vFirmwarePatch    = 0;
  uint32_t      vFirmwareBuild    = 0;
  uint16_t      vColumns          = 0;
  uint16_t      vRows             = 0;
  uint32_t      vColumnStatusSize = 0;
  XdnaTileInfo vCore;
  XdnaTileInfo vMem;
  XdnaTileInfo vShim;
  uint32_t      vNpuClockMhz      = 0;  // now; both clocks drop when the NPU is idle
  uint32_t      vHClockMhz        = 0;
  uint32_t      vPowerMode        = 0;  // enum amdxdna_power_mode_type
  uint64_t      vHClockMaxMhz     = 0;
  uint64_t      vTopsMax          = 0;
  uint64_t      vTopsNow          = 0;
  uint64_t      vContextsMax      = 0;
  uint64_t      vContextsNow      = 0;
};

struct XdnaSensor {
  std::string vLabel;
  std::string vUnits;
  uint32_t    vValue        = 0;
  int         vUnitExponent = 0;  // vValue is in vUnits; times 10^vUnitExponent it is in the base unit (mW and -3: W)
  bool        vIsPower      = false;
};

// buffer types of the driver (enum amdxdna_bo_type)
enum class XdnaBufferType : uint32_t {
  cShare   = 1,  // host memory for data; the NPU reads and writes it at the address of its mapping
  cDevHeap = 2,  // the device memory of the process, in multiples of 64 MiB; needed before a context is created
  cDev     = 3,  // a piece of the heap, for instruction buffers and overlays
  cCommand = 4,  // one command
};

struct XdnaBuffer {
  uint32_t        vHandle       = 0;
  XdnaBufferType vType         = XdnaBufferType::cShare;
  uint64_t        vSize         = 0;
  void *          vMap          = nullptr;  // where the host reads and writes it
  uint64_t        vDevAddr      = 0;        // the address the NPU uses for it
  void *          vUnmapBase    = nullptr;  // what to unmap with the buffer (nothing for cDev: it lives inside the heap)
  uint64_t        vUnmapSize    = 0;
};

// a hardware context of the NPU
struct XdnaContext {
  uint32_t vHandle     = 0;
  uint32_t vSyncHandle = 0;  // DRM synchronisation object where the driver signals the end of the commands of the context
};

// what the driver reports about a hardware context, of any process
struct XdnaContextStatus {
  uint32_t vHandle         = 0;
  int64_t  vPid            = 0;
  uint32_t vStartColumn    = 0;
  uint32_t vColumns        = 0;
  uint64_t vSubmitted      = 0;  // commands submitted
  uint64_t vCompleted      = 0;  // commands that ended
  uint64_t vErrors         = 0;
  uint32_t vPasid          = 0;
  // health report of the firmware; all zeros with a firmware that does not have it (protocol under 6.18)
  uint32_t vTxnOpIndex     = 0;  // index of the last operation of the control code that ran
  uint32_t vProgramCounter = 0;
  uint32_t vFatalType      = 0;
  uint32_t vExceptionType  = 0;
  uint32_t vExceptionPc    = 0;
  uint32_t vAppModule      = 0;
};

// the last asynchronous error that the firmware reported; all zeros if there was none
struct XdnaAsyncError {
  uint64_t vCode      = 0;
  uint64_t vTimeUs    = 0;
  uint64_t vExtraCode = 0;
};

// how a wait for a command ended
enum class XdnaWaitResult {
  cDone,     // the command ended; its header has its state
  cTimeout,  // it did not end in the time given; it is still in the driver
  cFailed,   // the wait itself failed
};

// opcodes and states of a command (enum ert_cmd_opcode and enum ert_cmd_state of the driver)
constexpr uint32_t cXdnaCommandStartCu   = 0;
constexpr uint32_t cXdnaCommandChain     = 19;
constexpr uint32_t cXdnaCommandStartNpu  = 20;

constexpr uint32_t cXdnaStateNew         = 1;
constexpr uint32_t cXdnaStateCompleted   = 4;
constexpr uint32_t cXdnaStateError       = 5;
constexpr uint32_t cXdnaStateAbort       = 6;
constexpr uint32_t cXdnaStateTimeout     = 8;

// header of a command with pWords 32-bit words after it (the compute-unit mask and the payload), in state "new"
uint32_t fXdnaCommandHeader(uint32_t pOpcode, uint32_t pWords);

// state of a command, from its header
uint32_t fXdnaCommandState(uint32_t pHeader);

// Writes back and drops from the cache of the processor the lines that hold pRecords records of pBytes, pStride bytes
// apart, from pFirst: what the SYNC_BO of the driver does on a range of a buffer (drm_clflush_virt_range: clflushopt on
// every line between two full barriers), without a call to the driver and only on the lines given. pFirst has to be
// in a mapped buffer.
void fXdnaFlushCache(const void * pFirst, uint64_t pRecords, uint64_t pStride, uint64_t pBytes);

const char * fXdnaStateName(uint32_t pState);

class XdnaDevice {
public:
  XdnaDevice() = default;
  ~XdnaDevice();
  XdnaDevice(const XdnaDevice &) = delete;
  XdnaDevice & operator=(const XdnaDevice &) = delete;

  // opens the device node (/dev/accel/accel0) and checks that its driver is amdxdna
  bool mOpen(const std::string & pPath, std::string & pError);
  void mClose();
  bool mIsOpen() const { return vFd >= 0; }

  // informational queries: they create nothing in the NPU
  bool mQueryInfo(XdnaInfo & pInfo, std::string & pError);
  bool mQuerySensors(std::vector<XdnaSensor> & pSensors, std::string & pError);
  bool mQueryContexts(std::vector<XdnaContextStatus> & pContexts, std::string & pError);
  bool mQueryLastError(XdnaAsyncError & pAsyncError, std::string & pError);

  // buffers. pSize is rounded up to whole pages. A cDevHeap buffer has to exist before the cDev ones and before a
  // context; it is mapped at an address aligned to 64 MiB
  bool mCreateBuffer(XdnaBufferType pType, uint64_t pSize, XdnaBuffer & pBuffer, std::string & pError);
  // flushes the CPU cache of a range of a buffer (the NPU is not cache coherent): after the host writes what the NPU
  // will read, and before the host reads what the NPU wrote. The driver has a second direction, "from device", but it
  // only takes it for the debug buffer of a context (measured: EINVAL for any other buffer)
  bool mSyncBuffer(const XdnaBuffer & pBuffer, uint64_t pOffset, uint64_t pSize, std::string & pError);
  void mDestroyBuffer(XdnaBuffer & pBuffer);

  // hardware context over pTiles core tiles (4 per column). It passes no quality-of-service hints, so the driver
  // runs it at its highest clock level
  bool mCreateContext(uint32_t pTiles, uint32_t pOpsPerCycle, XdnaContext & pContext, std::string & pError);
  // gives the context its overlay (a PDI in a cDev buffer) as its only compute unit. Once per context
  bool mConfigureContext(const XdnaContext & pContext, const XdnaBuffer & pOverlay, uint8_t pFunction, std::string & pError);
  // gives the context several overlays, each as a compute unit of its own (at most 32): compute unit i runs
  // pOverlays[i], and every command picks one with its mask of compute units. Once per context
  bool mConfigureContextUnits(const XdnaContext & pContext, const std::vector<const XdnaBuffer *> & pOverlays,
                              uint8_t pFunction, std::string & pError);
  void mDestroyContext(XdnaContext & pContext);

  // submits the command of a cCommand buffer; pArgHandles are the buffers it uses, which the driver pins. pSeq is the
  // number of the command in its context: the first one is 0
  bool mSubmit(const XdnaContext & pContext, const XdnaBuffer & pCommand, const std::vector<uint32_t> & pArgHandles,
               uint64_t & pSeq, std::string & pError);
  // waits for a submitted command; pTimeoutMs = 0 waits without limit. The state of the command is in its header.
  // For the first 2 ms it asks without sleeping, as a thread that sleeps wakes up late.
  // Waiting for the command number 0 waits for every command submitted to the context until then (see mWait)
  XdnaWaitResult mWait(const XdnaContext & pContext, uint64_t pSeq, uint32_t pTimeoutMs, std::string & pError);

private:
  bool mIoctl(unsigned long pRequest, void * pArg, const char * pWhat, std::string & pError);
  bool mGetInfo(uint32_t pParam, void * pBuffer, uint32_t pSize, const char * pWhat, std::string & pError);

  int vFd = -1;
};
