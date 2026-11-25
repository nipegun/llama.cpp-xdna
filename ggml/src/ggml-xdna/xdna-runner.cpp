#include "xdna-runner.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <vector>

namespace {

const uint64_t cHeapBytes         = 64ull << 20;  // the device memory of a process comes in pieces of 64 MiB
const uint64_t cCommandBytes      = 4096;
const uint32_t cOpsPerCycle       = 2048;         // hint for the driver: what the overlays of mlir-aie declare
const uint32_t cBufferSlots       = 5;            // buffers the kernel of an mlir-aie overlay takes
const uint32_t cOpcodeTransaction = 3;            // of its register map: run an instruction stream
const uint32_t cTimeoutMs         = 10000;

double fNowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

bool fReadFile(const std::string & pPath, std::vector<uint8_t> & aBytes) {
  std::ifstream vFile(pPath, std::ios::binary);
  if (!vFile) {
    return false;
  }
  aBytes.assign(std::istreambuf_iterator<char>(vFile), std::istreambuf_iterator<char>());
  return !aBytes.empty();
}

}  // namespace

XdnaRunner::~XdnaRunner() {
  mClose();
}

bool XdnaRunner::mOpen(const std::string & pDevicePath, const std::string & pOverlayPath,
                        const std::string & pManifestPath, std::string & pError, const std::string & pGemvOverlayPath,
                        const std::string & pGemvManifestPath, const std::string & pMoeOverlayPath,
                        const std::string & pMoeManifestPath) {
  mClose();
  std::vector<uint8_t> aPdi;
  if (!fReadFile(pOverlayPath, aPdi)) {
    pError = "cannot read the overlay " + pOverlayPath;
    return false;
  }
  if (!vOverlay.mLoad(pManifestPath, pError)) {
    return false;
  }
  std::vector<uint8_t> aPdiGemv;
  std::vector<uint8_t> aPdiMoe;
  vHasGemv     = false;
  vHasChain    = false;
  vHasMoe      = false;
  vMoeUnitMask = 0;
  const bool vSecond = !pGemvOverlayPath.empty();
  if (vSecond) {
    if (!fReadFile(pGemvOverlayPath, aPdiGemv)) {
      pError = "cannot read the overlay " + pGemvOverlayPath;
      return false;
    }
    // the overlay for generating tokens or the chained one, as its manifest says (a chained one of version 5 is
    // refused by mLoad, which says so); both overlays share the context, which has to be as wide as the array either
    // was built for
    if (fXdnaGemvManifestVersion(pGemvManifestPath) >= 5) {
      if (!vChainOverlay.mLoad(pGemvManifestPath, pError)) {
        return false;
      }
      if (vChainOverlay.vColumns != vOverlay.vColumns) {
        pError = "the two overlays are for arrays of different widths";
        return false;
      }
      vHasChain = true;
    } else {
      if (!vGemvOverlay.mLoad(pGemvManifestPath, pError)) {
        return false;
      }
      if (vGemvOverlay.vColumns != vOverlay.vColumns) {
        pError = "the two overlays are for arrays of different widths";
        return false;
      }
      vHasGemv = true;
    }
  }
  const bool vThird = !pMoeOverlayPath.empty();
  if (vThird) {
    if (!fReadFile(pMoeOverlayPath, aPdiMoe)) {
      pError = "cannot read the overlay " + pMoeOverlayPath;
      return false;
    }
    if (!vMoeOverlay.mLoad(pMoeManifestPath, pError)) {
      return false;
    }
    if (vMoeOverlay.vColumns != vOverlay.vColumns) {
      pError = "the mixture-of-experts overlay is for an array of a different width";
      return false;
    }
    vHasMoe = true;
  }
  if (!vDevice.mOpen(pDevicePath, pError)) {
    return false;
  }
  vIsOpen = true;
  XdnaInfo vInfo;
  if (!vDevice.mQueryInfo(vInfo, pError)) {
    mClose();
    return false;
  }
  // The context has to be as wide as the array the overlay was built for: a narrower one hangs the NPU.
  if (vOverlay.vColumns != vInfo.vColumns) {
    pError = "the overlay is for an array of " + std::to_string(vOverlay.vColumns) + " columns and the NPU has " +
             std::to_string(vInfo.vColumns);
    mClose();
    return false;
  }

  // the device memory of the process, the overlay in it, the context and its configuration
  if (!vDevice.mCreateBuffer(XdnaBufferType::cDevHeap, cHeapBytes, vHeap, pError) ||
      !vDevice.mCreateBuffer(XdnaBufferType::cDev, aPdi.size(), vPdi, pError)) {
    mClose();
    return false;
  }
  std::memcpy(vPdi.vMap, aPdi.data(), aPdi.size());
  if (!vDevice.mSyncBuffer(vPdi, 0, vPdi.vSize, pError)) {
    mClose();
    return false;
  }
  if (vSecond) {
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, aPdiGemv.size(), vPdiGemv, pError)) {
      mClose();
      return false;
    }
    std::memcpy(vPdiGemv.vMap, aPdiGemv.data(), aPdiGemv.size());
    if (!vDevice.mSyncBuffer(vPdiGemv, 0, vPdiGemv.vSize, pError)) {
      mClose();
      return false;
    }
  }
  if (vThird) {
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, aPdiMoe.size(), vPdiMoe, pError)) {
      mClose();
      return false;
    }
    std::memcpy(vPdiMoe.vMap, aPdiMoe.data(), aPdiMoe.size());
    if (!vDevice.mSyncBuffer(vPdiMoe, 0, vPdiMoe.vSize, pError)) {
      mClose();
      return false;
    }
  }
  if (!vDevice.mCreateContext(vOverlay.vColumns * vInfo.vCore.vRowCount, cOpsPerCycle, vContext, pError) ||
      !vDevice.mConfigureContext(vContext, vPdi, 0, pError) ||
      !vDevice.mCreateBuffer(XdnaBufferType::cCommand, cCommandBytes, vCommand, pError)) {
    mClose();
    return false;
  }
  if (vSecond) {
    const uint32_t vGemvColumns = vHasChain ? vChainOverlay.vColumns : vGemvOverlay.vColumns;
    if (!vDevice.mCreateContext(vGemvColumns * vInfo.vCore.vRowCount, cOpsPerCycle, vGemvContext, pError) ||
        !vDevice.mConfigureContext(vGemvContext, vPdiGemv, 0, pError) ||
        !vDevice.mCreateBuffer(XdnaBufferType::cCommand, cCommandBytes, vGemvCommand, pError)) {
      mClose();
      return false;
    }
  }
  if (vThird) {
    vMoeUnitMask = 0x1;
    if (!vDevice.mCreateContext(vMoeOverlay.vColumns * vInfo.vCore.vRowCount, cOpsPerCycle, vMoeContext, pError) ||
        !vDevice.mConfigureContext(vMoeContext, vPdiMoe, 0, pError) ||
        !vDevice.mCreateBuffer(XdnaBufferType::cCommand, cCommandBytes, vMoeCommand, pError)) {
      mClose();
      return false;
    }
  }
  for (XdnaBuffer & vAsync : aCommandAsync) {
    if (!vDevice.mCreateBuffer(XdnaBufferType::cCommand, cCommandBytes, vAsync, pError)) {
      mClose();
      return false;
    }
  }
  return true;
}

void XdnaRunner::mClose() {
  if (!vIsOpen) {
    return;
  }
  // in the order the driver needs: the contexts, the buffers, the pieces of the heap and the heap
  vDevice.mDestroyContext(vMoeContext);
  vDevice.mDestroyContext(vGemvContext);
  vDevice.mDestroyContext(vContext);
  vDevice.mDestroyBuffer(vMoeCommand);
  vDevice.mDestroyBuffer(vGemvCommand);
  for (uint32_t i = 0; i < cXdnaRunnerSlotsC; i++) {
    vDevice.mDestroyBuffer(aBufferC[i]);
    vDevice.mDestroyBuffer(aCommandAsync[i]);
    aKeptC[i] = false;
  }
  for (uint32_t i = 0; i < cXdnaRunnerSlotsA; i++) {
    vDevice.mDestroyBuffer(aBufferA[i]);
    aPackedA[i]   = XdnaPackedA();
    aPreparedA[i] = XdnaPackedA();
  }
  vDevice.mDestroyBuffer(vBufferY);
  vDevice.mDestroyBuffer(vBufferX);
  vDevice.mDestroyBuffer(vCommand);
  for (auto & vPlan : aGemmPlans) {
    vDevice.mDestroyBuffer(vPlan.second.vBuffer);
  }
  aGemmPlans.clear();
  for (auto & vPlan : aGdnPlans) {
    vDevice.mDestroyBuffer(vPlan.second.vBuffer);
  }
  aGdnPlans.clear();
  vDevice.mDestroyBuffer(vGdnA);
  vDevice.mDestroyBuffer(vGdnB);
  vDevice.mDestroyBuffer(vGdnC);
  for (auto & vPlan : aGemvPlans) {
    vDevice.mDestroyBuffer(vPlan.second.first);
  }
  aGemvPlans.clear();
  vDevice.mDestroyBuffer(vBufferIo);
  vDevice.mDestroyBuffer(vBufferIoChain);
  for (auto & vPlan : aChainPlans) {
    vDevice.mDestroyBuffer(vPlan.second.first);
  }
  aChainPlans.clear();
  vDevice.mDestroyBuffer(aMoeIo[0]);
  vDevice.mDestroyBuffer(aMoeIo[1]);
  for (auto & vPlan : aMoeDensePlans) {
    vDevice.mDestroyBuffer(vPlan.second.first);
  }
  aMoeDensePlans.clear();
  vDevice.mDestroyBuffer(vMoeInsts);
  vDevice.mDestroyBuffer(vPdiMoe);
  vDevice.mDestroyBuffer(vPdiGemv);
  vDevice.mDestroyBuffer(vPdi);
  vDevice.mDestroyBuffer(vHeap);
  vDevice.mClose();
  vIsOpen      = false;
  vBroken      = false;
  vLastUnits   = 0;
  vSwitches    = 0;
  vHasGemv     = false;
  vHasChain    = false;
  vHasMoe      = false;
  vMoeUnitMask = 0;
  vMoeKept     = false;
}

bool XdnaRunner::mSubmitAndWait(const std::vector<uint32_t> & aHandles, XdnaWaitResult & pWaited,
                                 std::string & pError) {
  bool vSubmitted = false;
  auto fCommand   = [&]() {
    uint64_t vSeq = 0;
    mNoteUnit(vCommand);
    vSubmitted    = vDevice.mSubmit(vContext, vCommand, aHandles, vSeq, pError);
    if (vSubmitted) {
      pWaited = vDevice.mWait(vContext, vSeq, cTimeoutMs, pError);
    }
  };
  // while the NPU works, the other threads of OpenMP sleep instead of spinning (see XdnaWorkers::mWhileParked)
  if (vPark) {
    vWorkers.mWhileParked(fCommand);
  } else {
    fCommand();
  }
  return vSubmitted;
}

bool XdnaRunner::mSubmitGemvAndWait(const std::vector<uint32_t> & aHandles, XdnaWaitResult & pWaited,
                                    std::string & pError) {
  bool vSubmitted = false;
  auto fCommand   = [&]() {
    uint64_t vSeq = 0;
    if (vLastUnits != 0 && vLastUnits != 0x2u) {
      vSwitches++;
    }
    vLastUnits = 0x2u;
    vSubmitted = vDevice.mSubmit(vGemvContext, vGemvCommand, aHandles, vSeq, pError);
    if (vSubmitted) {
      pWaited = vDevice.mWait(vGemvContext, vSeq, cTimeoutMs, pError);
    }
  };
  if (vPark) {
    vWorkers.mWhileParked(fCommand);
  } else {
    fCommand();
  }
  return vSubmitted;
}

void XdnaRunner::mNoteUnit(const XdnaBuffer & pCommand) {
  // llama.cpp-xdna: word 1 of a command is the mask of the compute units it runs on, one per overlay
  const uint32_t vMask = static_cast<const uint32_t *>(pCommand.vMap)[1];
  if (vLastUnits != 0 && vMask != vLastUnits) {
    vSwitches++;
  }
  vLastUnits = vMask;
}

bool XdnaRunner::mEnsure(XdnaBuffer & pBuffer, XdnaBufferType pType, uint64_t pBytes, std::string & pError) {
  if (pBuffer.vMap != nullptr && pBuffer.vSize >= pBytes) {
    return true;
  }
  vDevice.mDestroyBuffer(pBuffer);
  if (!vDevice.mCreateBuffer(pType, pBytes, pBuffer, pError)) {
    return false;
  }
  // The driver puts every page of a new buffer in when it is mapped, and the kernel zeroes them through the cache
  // of the processor, where the zeros stay as modified data. The NPU writes C to memory behind the cache, so the
  // flush that comes before reading C would write those zeros back over its results: measured with 512 x 1024 x
  // 16128, where 38 % of C came back as zeros. They leave the cache now, before the NPU writes anything.
  return !vFlushNew || vDevice.mSyncBuffer(pBuffer, 0, pBuffer.vSize, pError);
}

bool XdnaRunner::mFlushShared(const XdnaBuffer & pBuffer, uint64_t pBytes, bool pAfterHostWrite,
                              std::string & pError) {
  if (pBytes == 0 || pBuffer.vMap == nullptr) {
    return true;
  }
  const uint64_t vFlushed = pAfterHostWrite ?
                                std::min<uint64_t>(pBuffer.vSize, (pBytes + 4095) / 4096 * 4096 + 4096) :
                                std::min<uint64_t>(pBuffer.vSize, pBytes);
  if (!vUserFlush) {
    return vDevice.mSyncBuffer(pBuffer, 0, vFlushed, pError);
  }
  const uint32_t vThreads = vWorkers.mThreads();
  if (vThreads > 1 && vFlushed >= (512ull << 10)) {
    const uint64_t  vPages = (vFlushed + 4095) / 4096;
    const uint32_t  vTasks = uint32_t(std::min<uint64_t>(vThreads, vPages));
    const uint8_t * vBase  = static_cast<const uint8_t *>(pBuffer.vMap);
    vWorkers.mForEach(vTasks, [&](uint32_t pTask) {
      const uint64_t vStartPage = vPages * pTask / vTasks;
      const uint64_t vEndPage   = vPages * (pTask + 1) / vTasks;
      const uint64_t vOffset    = vStartPage * 4096;
      const uint64_t vEnd       = std::min<uint64_t>(vFlushed, vEndPage * 4096);
      if (vEnd > vOffset) {
        fXdnaFlushCache(vBase + vOffset, 1, vEnd - vOffset, vEnd - vOffset);
      }
    });
    return true;
  }
  fXdnaFlushCache(pBuffer.vMap, 1, vFlushed, vFlushed);
  return true;
}

bool XdnaRunner::mCreateShared(uint64_t pBytes, XdnaBuffer & pBuffer, std::string & pError) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  pBuffer = XdnaBuffer();
  return mEnsure(pBuffer, XdnaBufferType::cShare, pBytes, pError);
}

void XdnaRunner::mFreeShared(XdnaBuffer & pBuffer) {
  vDevice.mDestroyBuffer(pBuffer);
}

bool XdnaRunner::mLoadWeights(const float * pWeights, uint32_t pOutputs, uint32_t pInputs, XdnaRounding pRounding,
                               XdnaWeights & pLoaded, std::string & pError, bool pReuseBuffer) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  // the buffer of B does not depend on M: any M gives the padded K and N
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, 1, pInputs, pOutputs, vShape, pError)) {
    return false;
  }
  const uint64_t vBytes = uint64_t(vShape.vPaddedK) * vShape.vPaddedN / cXdnaBfp16Values * cXdnaBfp16Bytes;
  if (pReuseBuffer) {
    pLoaded.vOutputs = 0;
    if (!mEnsure(pLoaded.vBuffer, XdnaBufferType::cShare, vBytes, pError)) {
      return false;
    }
  } else if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vBytes, pLoaded.vBuffer, pError)) {
    return false;
  }
  if (!fXdnaGemmPackB(vOverlay, vShape, pWeights, static_cast<uint8_t *>(pLoaded.vBuffer.vMap), pRounding,
                       &vWorkers)) {
    pError = "the weights have a value that is not finite";
    if (!pReuseBuffer) {
      vDevice.mDestroyBuffer(pLoaded.vBuffer);
    }
    return false;
  }
  if (!mFlushShared(pLoaded.vBuffer, vBytes, true, pError)) {
    if (!pReuseBuffer) {
      vDevice.mDestroyBuffer(pLoaded.vBuffer);
    }
    return false;
  }
  pLoaded.vOutputs       = pOutputs;
  pLoaded.vInputs        = pInputs;
  pLoaded.vPaddedOutputs = vShape.vPaddedN;
  pLoaded.vPaddedInputs  = vShape.vPaddedK;
  pLoaded.vFormat        = XdnaPromptFormat::cBfp16;
  return true;
}

bool XdnaRunner::mLoadWeightsRows(const XdnaGemmRowReader & pReadRow, uint32_t pOutputs, uint32_t pInputs,
                                   XdnaRounding pRounding, XdnaWeights & pLoaded, std::string & pError,
                                   bool pReuseBuffer) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, 1, pInputs, pOutputs, vShape, pError)) {
    return false;
  }
  const uint64_t vBytes = uint64_t(vShape.vPaddedK) * vShape.vPaddedN / cXdnaBfp16Values * cXdnaBfp16Bytes;
  if (pReuseBuffer) {
    pLoaded.vOutputs = 0;
    if (!mEnsure(pLoaded.vBuffer, XdnaBufferType::cShare, vBytes, pError)) {
      return false;
    }
  } else if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vBytes, pLoaded.vBuffer, pError)) {
    return false;
  }
  if (!fXdnaGemmPackBRows(vOverlay, vShape, pReadRow, static_cast<uint8_t *>(pLoaded.vBuffer.vMap), pRounding,
                           &vWorkers)) {
    pError = "the weights have a value that is not finite";
    if (!pReuseBuffer) {
      vDevice.mDestroyBuffer(pLoaded.vBuffer);
    }
    return false;
  }
  if (!mFlushShared(pLoaded.vBuffer, vBytes, true, pError)) {
    if (!pReuseBuffer) {
      vDevice.mDestroyBuffer(pLoaded.vBuffer);
    }
    return false;
  }
  pLoaded.vOutputs       = pOutputs;
  pLoaded.vInputs        = pInputs;
  pLoaded.vPaddedOutputs = vShape.vPaddedN;
  pLoaded.vPaddedInputs  = vShape.vPaddedK;
  pLoaded.vFormat        = XdnaPromptFormat::cBfp16;
  return true;
}

bool XdnaRunner::mLoadWeightsCompressed(XdnaPromptFormat pFormat, const XdnaGemmRowPtrReader & pRowPtr,
                                         uint32_t pOutputs, uint32_t pInputs, XdnaWeights & pLoaded,
                                         std::string & pError, bool pReuseBuffer) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  if (pFormat == XdnaPromptFormat::cBfp16 || vOverlay.vCompB != 1) {
    pError = "this overlay does not decode compressed prompt weights";
    return false;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, 1, pInputs, pOutputs, vShape, pError, pFormat)) {
    return false;
  }
  const uint64_t vBytes = fXdnaGemmBytesB(vOverlay, vShape, pFormat);
  if (pReuseBuffer) {
    pLoaded.vOutputs = 0;
    if (!mEnsure(pLoaded.vBuffer, XdnaBufferType::cShare, vBytes, pError)) {
      return false;
    }
  } else if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vBytes, pLoaded.vBuffer, pError)) {
    return false;
  }
  const bool vPacked = pFormat == XdnaPromptFormat::cQ6K ?
                           fXdnaGemmPackBQ6K(vOverlay, vShape, pRowPtr,
                                             static_cast<uint8_t *>(pLoaded.vBuffer.vMap), &vWorkers) :
                           fXdnaGemmPackBQ8_0(vOverlay, vShape, pRowPtr,
                                              static_cast<uint8_t *>(pLoaded.vBuffer.vMap), &vWorkers);
  if (!vPacked) {
    pError = "cannot pack compressed prompt weights for these dimensions";
    if (!pReuseBuffer) {
      vDevice.mDestroyBuffer(pLoaded.vBuffer);
    }
    return false;
  }
  if (!mFlushShared(pLoaded.vBuffer, vBytes, true, pError)) {
    if (!pReuseBuffer) {
      vDevice.mDestroyBuffer(pLoaded.vBuffer);
    }
    return false;
  }
  pLoaded.vOutputs       = pOutputs;
  pLoaded.vInputs        = pInputs;
  pLoaded.vPaddedOutputs = vShape.vPaddedN;
  pLoaded.vPaddedInputs  = vShape.vPaddedK;
  pLoaded.vFormat        = pFormat;
  return true;
}

const uint8_t * XdnaRunner::mRawC(uint64_t & pBytes) const {
  pBytes = aBufferC[0].vMap != nullptr ? vBytesC : 0;
  return static_cast<const uint8_t *>(aBufferC[0].vMap);
}

void XdnaRunner::mFreeWeights(XdnaWeights & pLoaded) {
  vDevice.mDestroyBuffer(pLoaded.vBuffer);
  pLoaded = XdnaWeights();
}

bool XdnaRunner::mMultiply(const float * pA, uint32_t pM, const XdnaWeights & pWeights, XdnaRounding pRounding,
                            float * pC, XdnaGemmTimes & pTimes, std::string & pError) {
  XdnaGemmTarget vWhole;
  vWhole.vData   = pC;
  vWhole.vStride = pWeights.vOutputs;
  vWhole.vFirst  = 0;
  vWhole.vCount  = pWeights.vOutputs;
  return mMultiply(pA, pM, pWeights, pRounding, &vWhole, 1, pTimes, pError);
}

bool XdnaRunner::mMultiply(const float * pA, uint32_t pM, const XdnaWeights & pWeights, XdnaRounding pRounding,
                            const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaGemmTimes & pTimes,
                            std::string & pError) {
  pTimes = XdnaGemmTimes();
  return mPackRows(pA, pM, pWeights.vInputs, pWeights.vInputs, pRounding, pTimes, pError) &&
         mMultiplyPacked(pWeights, aTargets, pTargets, false, pTimes, pError);
}

bool XdnaRunner::mPackRows(const float * pA, uint32_t pM, uint32_t pK, size_t pStride, XdnaRounding pRounding,
                            XdnaGemmTimes & pTimes, std::string & pError, uint32_t pSlotA,
                            const XdnaRowNorm * pNorm) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  if (pSlotA >= cXdnaRunnerSlotsA) {
    pError = "there are cXdnaRunnerSlotsA buffers of A";
    return false;
  }
  XdnaBuffer &  vBufferA = aBufferA[pSlotA];
  XdnaPackedA & vPacked  = aPackedA[pSlotA];
  // the layout of A depends on M and K only
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, pM, pK, 1, vShape, pError)) {
    return false;
  }
  const uint64_t vBytes = uint64_t(vShape.vPaddedM) * vShape.vPaddedK / cXdnaBfp16Values * cXdnaBfp16Bytes;
  if (!mEnsure(vBufferA, XdnaBufferType::cShare, vBytes, pError)) {
    return false;
  }
  const double vStart = fNowMs();
  vPacked.vM          = 0;  // nothing usable in A until it is complete
  if (!fXdnaGemmPackA(vOverlay, vShape, pA, static_cast<uint8_t *>(vBufferA.vMap), pRounding, &vWorkers,
                       pStride, pNorm)) {
    pError = "A has a value that is not finite";
    return false;
  }
  // the NPU is not cache coherent: what the host wrote has to leave the cache before the NPU reads it
  if (!mFlushShared(vBufferA, vBytes, true, pError)) {
    return false;
  }
  pTimes.vPackMs  += fNowMs() - vStart;
  vPacked.vM       = pM;
  vPacked.vK       = pK;
  vPacked.vPaddedM = vShape.vPaddedM;
  vPacked.vPaddedK = vShape.vPaddedK;
  vPacked.vBytes   = vBytes;
  return true;
}

bool XdnaRunner::mPrepare(const XdnaWeights & pWeights, uint32_t pSlotA, uint32_t pSlotC, XdnaBuffer & pCommand,
                           XdnaGemmShape & pShape, uint64_t & pBytesC, std::vector<uint32_t> & aHandles,
                           XdnaGemmTimes & pTimes, std::string & pError) {
  if (pSlotA >= cXdnaRunnerSlotsA || pSlotC >= cXdnaRunnerSlotsC) {
    pError = "there are cXdnaRunnerSlotsA buffers of A and cXdnaRunnerSlotsC of C";
    return false;
  }
  XdnaBuffer &        vBufferA = aBufferA[pSlotA];
  const XdnaPackedA & vPacked  = aPackedA[pSlotA];
  XdnaBuffer &        vBufferC = aBufferC[pSlotC];
  aKeptC[pSlotC]                = false;
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  if (vBroken) {
    pError = "a command did not end before: nothing else is sent to the NPU";
    return false;
  }
  if (pWeights.vBuffer.vMap == nullptr) {
    pError = "the weights are not loaded";
    return false;
  }
  if (vPacked.vM == 0 || pWeights.vInputs != vPacked.vK) {
    pError = "the rows in A were not converted for these weights";
    return false;
  }

  if (!fXdnaGemmShape(vOverlay, vPacked.vM, pWeights.vInputs, pWeights.vOutputs, pShape, pError, pWeights.vFormat)) {
    return false;
  }
  if (pShape.vPaddedK != pWeights.vPaddedInputs || pShape.vPaddedN != pWeights.vPaddedOutputs ||
      pShape.vPaddedM != vPacked.vPaddedM || pShape.vPaddedK != vPacked.vPaddedK) {
    pError = "the weights were loaded for another overlay";
    return false;
  }

  // the instruction stream of this size, written the first time it is needed (a model repeats a few sizes)
  const double vStart = fNowMs();
  const auto   vKey   = std::make_tuple(pShape.vPaddedM, pShape.vPaddedK, pShape.vPaddedN, uint8_t(pWeights.vFormat));
  auto         vFound = aGemmPlans.find(vKey);
  if (vFound == aGemmPlans.end()) {
    XdnaGemmPlan vPlan;
    if (!fXdnaGemmPlan(vOverlay, pShape.vPaddedM, pShape.vPaddedK, pShape.vPaddedN, vPlan, pError, pWeights.vFormat)) {
      return false;
    }
    if (vPlan.vBytesA > vPacked.vBytes) {
      pError = "the plan reads more of A than was converted";
      return false;
    }
    XdnaGemmCached vCached;
    const uint64_t  vBytes = vPlan.aInstructions.size() * sizeof(uint32_t);
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, vBytes, vCached.vBuffer, pError)) {
      return false;
    }
    std::memcpy(vCached.vBuffer.vMap, vPlan.aInstructions.data(), vBytes);
    if (!vDevice.mSyncBuffer(vCached.vBuffer, 0, vBytes, pError)) {
      vDevice.mDestroyBuffer(vCached.vBuffer);
      return false;
    }
    vCached.vWords  = uint32_t(vPlan.aInstructions.size());
    vCached.vBytesC = vPlan.vBytesC;
    vFound          = aGemmPlans.emplace(vKey, vCached).first;
  }
  const XdnaGemmCached & vPlan = vFound->second;
  if (!mEnsure(vBufferC, XdnaBufferType::cShare, vPlan.vBytesC, pError)) {
    return false;
  }
  pTimes.vPlanMs += fNowMs() - vStart;
  pBytesC         = vPlan.vBytesC;
  vBytesC         = vPlan.vBytesC;
  if (vFillC) {
    std::memset(vBufferC.vMap, 0xff, vPlan.vBytesC);
    if (vFlushFillC && !vDevice.mSyncBuffer(vBufferC, 0, vPlan.vBytesC, pError)) {
      return false;
    }
  }

  // the command: its header, the mask of compute units and the register map of the kernel of an mlir-aie overlay:
  // opcode (64 bits), address of the instructions (64 bits), their number of words (32 bits) and the address of
  // every buffer (64 bits each)
  uint32_t *     vWords = static_cast<uint32_t *>(pCommand.vMap);
  const uint32_t vCount = 1 + 5 + 2 * cBufferSlots;
  std::memset(vWords, 0, cCommandBytes);
  vWords[0] = fXdnaCommandHeader(cXdnaCommandStartCu, vCount);
  vWords[1] = 0x1;
  vWords[2] = cOpcodeTransaction;
  vWords[4] = uint32_t(vPlan.vBuffer.vDevAddr & 0xffffffffu);
  vWords[5] = uint32_t(vPlan.vBuffer.vDevAddr >> 32);
  vWords[6] = vPlan.vWords;
  const XdnaBuffer * aBuffers[3] = {&vBufferA, &pWeights.vBuffer, &vBufferC};
  for (uint32_t i = 0; i < 3; i++) {
    vWords[7 + 2 * i] = uint32_t(aBuffers[i]->vDevAddr & 0xffffffffu);
    vWords[8 + 2 * i] = uint32_t(aBuffers[i]->vDevAddr >> 32);
  }
  aHandles = {vPlan.vBuffer.vHandle, vBufferA.vHandle, pWeights.vBuffer.vHandle, vBufferC.vHandle};
  return true;
}

bool XdnaRunner::mCompleted(const XdnaBuffer & pCommand, XdnaWaitResult pWaited, std::string & pError) {
  const uint32_t vState = fXdnaCommandState(static_cast<const uint32_t *>(pCommand.vMap)[0]);
  if (pWaited != XdnaWaitResult::cDone || vState != cXdnaStateCompleted) {
    vBroken = true;
    pError  = std::string("the command did not complete (") +
              (pWaited == XdnaWaitResult::cTimeout ? "no answer" :
               pWaited == XdnaWaitResult::cFailed ? "the wait failed: " + pError : fXdnaStateName(vState)) +
              ")";
    return false;
  }
  return true;
}

bool XdnaRunner::mReadC(uint32_t pSlotC, const XdnaGemmShape & pShape, uint64_t pBytesC,
                         const XdnaGemmTarget * aTargets, uint32_t pTargets, bool pAdd, bool pUnpack,
                         XdnaGemmTimes & pTimes, std::string & pError) {
  XdnaBuffer & vBufferC = aBufferC[pSlotC];
  const double  vStart   = fNowMs();
  // what the NPU wrote may not be what the cache of the host holds
  if (!mFlushShared(vBufferC, pBytesC, false, pError)) {
    return false;
  }
  pTimes.vFlushMs += fNowMs() - vStart;
  if (!pUnpack) {
    aShapeC[pSlotC]   = pShape;
    aKeptC[pSlotC]    = true;
    pTimes.vUnpackMs += fNowMs() - vStart;
    return true;
  }
  fXdnaGemmUnpackC(vOverlay, pShape, vBufferC.vMap, aTargets, pTargets, &vWorkers, pAdd);
  pTimes.vUnpackMs += fNowMs() - vStart;
  return true;
}

bool XdnaRunner::mMultiplyPacked(const XdnaWeights & pWeights, const XdnaGemmTarget * aTargets, uint32_t pTargets,
                                  bool pAdd, XdnaGemmTimes & pTimes, std::string & pError, uint32_t pSlotC,
                                  bool pUnpack, uint32_t pSlotA) {
  XdnaGemmShape        vShape;
  uint64_t              vBytesNow = 0;
  std::vector<uint32_t> aHandles;
  if (!mPrepare(pWeights, pSlotA, pSlotC, vCommand, vShape, vBytesNow, aHandles, pTimes, pError)) {
    return false;
  }
  const double    vStart  = fNowMs();
  XdnaWaitResult vWaited = XdnaWaitResult::cFailed;
  if (!mSubmitAndWait(aHandles, vWaited, pError)) {
    return false;
  }
  pTimes.vNpuMs += fNowMs() - vStart;
  return mCompleted(vCommand, vWaited, pError) &&
         mReadC(pSlotC, vShape, vBytesNow, aTargets, pTargets, pAdd, pUnpack, pTimes, pError);
}

bool XdnaRunner::mSubmitPacked(const XdnaWeights & pWeights, uint32_t pSlotA, uint32_t pSlotC,
                                XdnaPending & pPending, XdnaGemmTimes & pTimes, std::string & pError) {
  pPending = XdnaPending();
  XdnaGemmShape        vShape;
  uint64_t              vBytesNow = 0;
  std::vector<uint32_t> aHandles;
  if (pSlotC >= cXdnaRunnerSlotsC) {
    pError = "there are cXdnaRunnerSlotsC buffers of C";
    return false;
  }
  if (!mPrepare(pWeights, pSlotA, pSlotC, aCommandAsync[pSlotC], vShape, vBytesNow, aHandles, pTimes, pError)) {
    return false;
  }
  uint64_t vSeq = 0;
  mNoteUnit(aCommandAsync[pSlotC]);
  if (!vDevice.mSubmit(vContext, aCommandAsync[pSlotC], aHandles, vSeq, pError)) {
    return false;
  }
  pPending.vValid  = true;
  pPending.vSeq    = vSeq;
  pPending.vSlotC  = pSlotC;
  pPending.vBytesC = vBytesNow;
  pPending.vShape  = vShape;
  return true;
}

bool XdnaRunner::mFinishPacked(XdnaPending & pPending, const XdnaGemmTarget * aTargets, uint32_t pTargets,
                                bool pAdd, XdnaGemmTimes & pTimes, std::string & pError, bool pUnpack) {
  if (!pPending.vValid) {
    pError = "no multiplication was sent";
    return false;
  }
  pPending.vValid         = false;
  const double    vStart  = fNowMs();
  XdnaWaitResult vWaited = XdnaWaitResult::cFailed;
  auto            fWait   = [&]() {
    vWaited = vDevice.mWait(vContext, pPending.vSeq, cTimeoutMs, pError);
  };
  // while the NPU works, the other threads of OpenMP sleep instead of spinning (see XdnaWorkers::mWhileParked)
  if (vPark) {
    vWorkers.mWhileParked(fWait);
  } else {
    fWait();
  }
  pTimes.vNpuMs += fNowMs() - vStart;
  return mCompleted(aCommandAsync[pPending.vSlotC], vWaited, pError) &&
         mReadC(pPending.vSlotC, pPending.vShape, pPending.vBytesC, aTargets, pTargets, pAdd, pUnpack, pTimes,
                pError);
}

bool XdnaRunner::mUnpackSwiglu(uint32_t pGate, uint32_t pUp, const XdnaGemmTarget & pTarget, XdnaGemmTimes & pTimes,
                                std::string & pError, uint64_t pRowFirst, uint64_t pRowLength) {
  if (pGate >= cXdnaRunnerSlotsC || pUp >= cXdnaRunnerSlotsC || pGate == pUp || !aKeptC[pGate] || !aKeptC[pUp]) {
    pError = "the buffers of C of the gate and the up do not hold two kept results";
    return false;
  }
  const XdnaGemmShape & vGate = aShapeC[pGate];
  const XdnaGemmShape & vUp   = aShapeC[pUp];
  if (vGate.vM != vUp.vM || vGate.vK != vUp.vK || vGate.vN != vUp.vN || vGate.vPaddedM != vUp.vPaddedM ||
      vGate.vPaddedK != vUp.vPaddedK || vGate.vPaddedN != vUp.vPaddedN) {
    pError = "the gate and the up of a SwiGLU have different shapes";
    return false;
  }
  const double vStart = fNowMs();
  if (!fXdnaGemmUnpackSwiglu(vOverlay, vGate, aBufferC[pGate].vMap, aBufferC[pUp].vMap, pTarget, &vWorkers, pRowFirst,
                              pRowLength)) {
    pError = "the overlay gives C in a format that SwiGLU does not read";
    return false;
  }
  aKeptC[pGate] = aKeptC[pUp] = false;
  pTimes.vUnpackMs += fNowMs() - vStart;
  return true;
}

bool XdnaRunner::mPrepareRowsA(uint32_t pSlotA, uint32_t pM, uint32_t pK, std::string & pError) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  if (pSlotA >= cXdnaRunnerSlotsA) {
    pError = "there are cXdnaRunnerSlotsA buffers of A";
    return false;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, pM, pK, 1, vShape, pError)) {
    return false;
  }
  const uint64_t vBytes = uint64_t(vShape.vPaddedM) * vShape.vPaddedK / cXdnaBfp16Values * cXdnaBfp16Bytes;
  aPackedA[pSlotA]      = XdnaPackedA();  // nothing usable in A until it is complete
  if (!mEnsure(aBufferA[pSlotA], XdnaBufferType::cShare, vBytes, pError)) {
    return false;
  }
  aPreparedA[pSlotA].vM       = pM;
  aPreparedA[pSlotA].vK       = pK;
  aPreparedA[pSlotA].vPaddedM = vShape.vPaddedM;
  aPreparedA[pSlotA].vPaddedK = vShape.vPaddedK;
  aPreparedA[pSlotA].vBytes   = vBytes;
  return true;
}

bool XdnaRunner::mReadyRowsA(uint32_t pSlotA, XdnaGemmTimes & pTimes, std::string & pError) {
  if (pSlotA >= cXdnaRunnerSlotsA || aPreparedA[pSlotA].vM == 0) {
    pError = "the buffer of A was not made ready for rows converted ahead";
    return false;
  }
  // the NPU is not cache coherent: what the host wrote has to leave the cache before the NPU reads it
  const double vStart = fNowMs();
  if (!mFlushShared(aBufferA[pSlotA], aPreparedA[pSlotA].vBytes, true, pError)) {
    return false;
  }
  pTimes.vFlushMs += fNowMs() - vStart;
  aPackedA[pSlotA]   = aPreparedA[pSlotA];
  aPreparedA[pSlotA] = XdnaPackedA();
  return true;
}

uint8_t * XdnaRunner::mPreparedBufferA(uint32_t pSlotA, uint32_t & pPaddedM) {
  if (pSlotA >= cXdnaRunnerSlotsA || aPreparedA[pSlotA].vM == 0) {
    pPaddedM = 0;
    return nullptr;
  }
  pPaddedM = aPreparedA[pSlotA].vPaddedM;
  return static_cast<uint8_t *>(aBufferA[pSlotA].vMap);
}

bool XdnaRunner::mMarkFlushedRowsA(uint32_t pSlotA, XdnaGemmTimes & pTimes, std::string & pError) {
  if (pSlotA >= cXdnaRunnerSlotsA || aPreparedA[pSlotA].vM == 0) {
    pError = "the buffer of A was not made ready for rows converted ahead";
    return false;
  }
  if (!vUserFlush) {
    return mReadyRowsA(pSlotA, pTimes, pError);
  }
  const double   vStart   = fNowMs();
  const uint64_t vBytes   = aPreparedA[pSlotA].vBytes;
  const uint64_t vFlushed = std::min<uint64_t>(aBufferA[pSlotA].vSize, (vBytes + 4095) / 4096 * 4096 + 4096);
  if (vFlushed > vBytes) {
    fXdnaFlushCache(static_cast<const uint8_t *>(aBufferA[pSlotA].vMap) + vBytes, 1, vFlushed - vBytes,
                    vFlushed - vBytes);
  }
  pTimes.vFlushMs += fNowMs() - vStart;
  aPackedA[pSlotA]   = aPreparedA[pSlotA];
  aPreparedA[pSlotA] = XdnaPackedA();
  return true;
}

bool XdnaRunner::mPackGatedNormRows(uint32_t pSlotA, uint32_t pM, uint32_t pK, uint64_t pFirst, const float * pO,
                                     uint64_t pStrideO, const float * pZ, uint64_t pStrideZ, const float * pW,
                                     uint32_t pHead, float pEps, XdnaRounding pRounding, XdnaGemmTimes & pTimes,
                                     std::string & pError, bool pSigmoid) {
  if (!mPrepareRowsA(pSlotA, pM, pK, pError)) {
    return false;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, pM, pK, 1, vShape, pError)) {
    return false;
  }
  const double vStart = fNowMs();
  if (!fXdnaGemmPackGatedNorm(vOverlay, vShape, pO, pStrideO, pZ, pStrideZ, pW, pHead, pEps, pFirst,
                               static_cast<uint8_t *>(aBufferA[pSlotA].vMap), pRounding, &vWorkers, pSigmoid)) {
    pError = "the gated norm does not go into those rows, or has a value that is not finite";
    return false;
  }
  pTimes.vPackMs += fNowMs() - vStart;
  return mReadyRowsA(pSlotA, pTimes, pError);
}

bool XdnaRunner::mSwigluRowsFit(uint32_t pM, uint64_t pRowFirst, uint64_t pOutputs, const uint64_t * aFirsts,
                                 const uint32_t * aCounts, uint32_t pPieces) const {
  XdnaSwigluPiece aPieces[cXdnaRunnerSlotsA];
  if (pPieces == 0 || pPieces > cXdnaRunnerSlotsA) {
    return false;
  }
  for (uint32_t i = 0; i < pPieces; i++) {
    aPieces[i].vFirst = aFirsts[i];
    aPieces[i].vCount = aCounts[i];
  }
  return fXdnaGemmSwigluFits(vOverlay, pM, pRowFirst, pOutputs, aPieces, pPieces);
}

bool XdnaRunner::mUnpackSwigluRows(uint32_t pGate, uint32_t pUp, const uint32_t * aSlotsA, const uint64_t * aFirsts,
                                    uint32_t pPieces, XdnaRounding pRounding, XdnaGemmTimes & pTimes,
                                    std::string & pError, uint64_t pRowFirst, uint64_t pRowLength) {
  if (pGate >= cXdnaRunnerSlotsC || pUp >= cXdnaRunnerSlotsC || pGate == pUp || !aKeptC[pGate] || !aKeptC[pUp]) {
    pError = "the buffers of C of the gate and the up do not hold two kept results";
    return false;
  }
  const XdnaGemmShape & vGate = aShapeC[pGate];
  const XdnaGemmShape & vUp   = aShapeC[pUp];
  if (vGate.vM != vUp.vM || vGate.vK != vUp.vK || vGate.vN != vUp.vN || vGate.vPaddedM != vUp.vPaddedM ||
      vGate.vPaddedK != vUp.vPaddedK || vGate.vPaddedN != vUp.vPaddedN) {
    pError = "the gate and the up of a SwiGLU have different shapes";
    return false;
  }
  if (pPieces == 0 || pPieces > cXdnaRunnerSlotsA) {
    pError = "the rows of a SwiGLU go into 1 to cXdnaRunnerSlotsA pieces";
    return false;
  }
  XdnaSwigluPiece aPieces[cXdnaRunnerSlotsA];
  for (uint32_t i = 0; i < pPieces; i++) {
    const uint32_t vSlot = aSlotsA[i];
    if (vSlot >= cXdnaRunnerSlotsA || aPreparedA[vSlot].vM != vGate.vM) {
      pError = "a buffer of A was not made ready for the rows of the SwiGLU";
      return false;
    }
    aPieces[i].vFirst  = aFirsts[i];
    aPieces[i].vCount  = aPreparedA[vSlot].vK;
    aPieces[i].vBuffer = static_cast<uint8_t *>(aBufferA[vSlot].vMap);
  }
  const double vStart = fNowMs();
  if (!fXdnaGemmPackSwiglu(vOverlay, vGate, aBufferC[pGate].vMap, aBufferC[pUp].vMap, pRowFirst, pRowLength, aPieces,
                            pPieces, pRounding, &vWorkers)) {
    pError = "the SwiGLU does not go into those rows, or has a value that is not finite";
    return false;
  }
  aKeptC[pGate] = aKeptC[pUp] = false;
  pTimes.vUnpackMs += fNowMs() - vStart;
  return true;
}

bool XdnaRunner::mPackAttnQ(uint32_t pSlotA, const XdnaAttnRows & pRows, uint32_t pDim, XdnaRounding pRounding,
                             XdnaGemmTimes & pTimes, std::string & pError) {
  const uint64_t vRows = uint64_t(pRows.vTokens) * pRows.vHeads;
  if (vRows == 0 || vRows > 0xffffffffu) {
    pError = "the rows of the queries of a group do not fit in a multiplication";
    return false;
  }
  if (!mPrepareRowsA(pSlotA, uint32_t(vRows), pDim, pError)) {
    return false;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, uint32_t(vRows), pDim, 1, vShape, pError)) {
    return false;
  }
  const double vStart = fNowMs();
  if (!fXdnaGemmPackAttnQ(vOverlay, vShape, pRows, static_cast<uint8_t *>(aBufferA[pSlotA].vMap), pRounding,
                           &vWorkers)) {
    pError = "the queries have a value that is not finite";
    return false;
  }
  pTimes.vPackMs += fNowMs() - vStart;
  return mReadyRowsA(pSlotA, pTimes, pError);
}

bool XdnaRunner::mLoadAttnB(const uint16_t * pData, uint64_t pStrideN, uint64_t pStrideK, uint32_t pOutputs,
                             uint32_t pInputs, uint32_t pValidInputs, XdnaRounding pRounding, XdnaWeights & pWeights,
                             XdnaGemmTimes & pTimes, std::string & pError, uint32_t pSplitDim) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  XdnaGemmShape vShape;
  if (!fXdnaGemmShape(vOverlay, 1, pInputs, pOutputs, vShape, pError)) {
    return false;
  }
  const uint64_t vBytes = uint64_t(vShape.vPaddedK) * vShape.vPaddedN / cXdnaBfp16Values * cXdnaBfp16Bytes;
  pWeights.vOutputs     = 0;  // nothing usable until it is complete
  if (!mEnsure(pWeights.vBuffer, XdnaBufferType::cShare, vBytes, pError)) {
    return false;
  }
  const double vStart = fNowMs();
  if (!fXdnaGemmPackBHalves(vOverlay, vShape, pData, pStrideN, pStrideK, pValidInputs,
                             static_cast<uint8_t *>(pWeights.vBuffer.vMap), pRounding, &vWorkers, pSplitDim)) {
    pError = "the keys or the values have a value that is not finite, or the overlay splits B";
    return false;
  }
  pTimes.vPackMs += fNowMs() - vStart;
  // the NPU is not cache coherent: what the host wrote has to leave the cache before the NPU reads it
  const double vFlushStart = fNowMs();
  if (!mFlushShared(pWeights.vBuffer, vBytes, true, pError)) {
    return false;
  }
  pTimes.vFlushMs         += fNowMs() - vFlushStart;
  pWeights.vOutputs        = pOutputs;
  pWeights.vInputs         = pInputs;
  pWeights.vPaddedOutputs  = vShape.vPaddedN;
  pWeights.vPaddedInputs   = vShape.vPaddedK;
  return true;
}

bool XdnaRunner::mPackAttnP(uint32_t pSlotS, const XdnaAttnScores & pScores, const uint32_t * aSlotsA,
                             const uint64_t * aFirsts, uint32_t pPieces, float * pSums, XdnaRounding pRounding,
                             XdnaGemmTimes & pTimes, std::string & pError) {
  if (pSlotS >= cXdnaRunnerSlotsC || !aKeptC[pSlotS]) {
    pError = "the buffer of C of the scores does not hold a kept result";
    return false;
  }
  if (pPieces == 0 || pPieces > cXdnaRunnerSlotsA) {
    pError = "P goes into 1 to cXdnaRunnerSlotsA pieces";
    return false;
  }
  const XdnaGemmShape & vShape = aShapeC[pSlotS];
  XdnaSwigluPiece       aPieces[cXdnaRunnerSlotsA];
  for (uint32_t i = 0; i < pPieces; i++) {
    const uint32_t vSlot = aSlotsA[i];
    if (vSlot >= cXdnaRunnerSlotsA || aPreparedA[vSlot].vM != vShape.vM) {
      pError = "a buffer of A was not made ready for the rows of P";
      return false;
    }
    aPieces[i].vFirst  = aFirsts[i];
    aPieces[i].vCount  = aPreparedA[vSlot].vK;
    aPieces[i].vBuffer = static_cast<uint8_t *>(aBufferA[vSlot].vMap);
  }
  XdnaAttnScores vScores = pScores;
  vScores.vS              = aBufferC[pSlotS].vMap;
  const double vStart     = fNowMs();
  if (!fXdnaGemmPackAttnP(vOverlay, vShape, vScores, aPieces, pPieces, pSums, pRounding, &vWorkers)) {
    pError = "P does not go into those rows, or has a value that is not finite";
    return false;
  }
  aKeptC[pSlotS] = false;
  pTimes.vUnpackMs += fNowMs() - vStart;
  for (uint32_t i = 0; i < pPieces; i++) {
    if (!mReadyRowsA(aSlotsA[i], pTimes, pError)) {
      return false;
    }
  }
  return true;
}

bool XdnaRunner::mLoadGemvWeights(const float * pWeights, uint32_t pOutputs, uint32_t pInputs,
                                   XdnaGemvWeights & pLoaded, std::string & pError) {
  XdnaGemvMember vMember;
  vMember.vFormat  = XdnaGemvFormat::cBfp16;
  vMember.vOutputs = pOutputs;
  vMember.fRows    = [&](uint32_t pFirst, uint32_t pCount, void * pTo) {
    std::memcpy(pTo, pWeights + size_t(pFirst) * pInputs, size_t(pCount) * pInputs * sizeof(float));
    return true;
  };
  return mLoadGemvMatrix({vMember}, pInputs, pLoaded, pError);
}

bool XdnaRunner::mLoadGemvMatrix(const std::vector<XdnaGemvMember> & aMembers, uint32_t pInputs,
                                  XdnaGemvWeights & pLoaded, std::string & pError) {
  if (!vIsOpen || !vHasGemv) {
    pError = "the overlay for generating tokens is not open";
    return false;
  }
  // the segments: consecutive members of one format, and where each starts among the members
  XdnaGemvPart         aParts[cXdnaGemvSegmentsMax];
  std::vector<uint32_t> aStarts;
  uint32_t              vParts = 0;
  for (uint32_t i = 0; i < aMembers.size(); i++) {
    if (aMembers[i].vOutputs == 0) {
      continue;
    }
    if (vParts == 0 || aParts[vParts - 1].vFormat != aMembers[i].vFormat) {
      if (vParts == cXdnaGemvSegmentsMax) {
        pError = "the matrices of weights make more segments than the overlay for generating tokens takes";
        return false;
      }
      aParts[vParts] = {aMembers[i].vFormat, 0};
      aStarts.push_back(i);
      vParts++;
    }
    aParts[vParts - 1].vOutputs += aMembers[i].vOutputs;
  }
  aStarts.push_back(uint32_t(aMembers.size()));
  // the buffer of W does not depend on the rows
  XdnaGemvShape vShape;
  if (!fXdnaGemvShape(vGemvOverlay, 1, pInputs, aParts, vParts, vShape, pError)) {
    return false;
  }
  if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vShape.vBytesW, pLoaded.vBuffer, pError)) {
    return false;
  }
  // segment by segment and stream by stream: the rows of one stream, from the members they belong to, and then its
  // part of the buffer
  std::vector<uint8_t> aRows;
  for (uint32_t vIndex = 0; vIndex < vParts; vIndex++) {
    const XdnaGemvSegment & vSegment   = vShape.aSegments[vIndex];
    const uint32_t           vPerStream = fXdnaGemvOutputsPerStream(vGemvOverlay, vShape, vIndex);
    const uint64_t           vRowBytes  = fXdnaGemvRowBytes(vShape, vSegment.vFormat);
    for (uint32_t vStream = 0; vStream < vGemvOverlay.mStreams(); vStream++) {
      const uint64_t vFirst = uint64_t(vStream) * vPerStream;
      const uint32_t vGiven =
          vFirst >= vSegment.vOutputs ? 0 : uint32_t(std::min<uint64_t>(vPerStream, vSegment.vOutputs - vFirst));
      aRows.resize(std::max<uint64_t>(vGiven, 1) * vRowBytes);
      // the rows vFirst to vFirst + vGiven of the segment, member by member
      uint32_t vDone   = 0;
      uint64_t vOffset = 0;  // the first output of the member, in the segment
      for (uint32_t m = aStarts[vIndex]; m < aStarts[vIndex + 1] && vDone < vGiven; m++) {
        const uint64_t vEnd = vOffset + aMembers[m].vOutputs;
        const uint64_t vAt  = vFirst + vDone;
        if (aMembers[m].vOutputs > 0 && vAt < vEnd) {
          const uint32_t vCount = uint32_t(std::min<uint64_t>(vEnd - vAt, vGiven - vDone));
          if (!aMembers[m].fRows(uint32_t(vAt - vOffset), vCount, aRows.data() + uint64_t(vDone) * vRowBytes)) {
            pError = "the rows of the weights could not be read";
            vDevice.mDestroyBuffer(pLoaded.vBuffer);
            return false;
          }
          vDone += vCount;
        }
        vOffset = vEnd;
      }
      if (!fXdnaGemvPackStream(vGemvOverlay, vShape, vIndex, vStream, aRows.data(), vGiven,
                                static_cast<uint8_t *>(pLoaded.vBuffer.vMap), &vWorkers)) {
        pError = "the weights have a value that is not finite";
        vDevice.mDestroyBuffer(pLoaded.vBuffer);
        return false;
      }
    }
  }
  if (!vDevice.mSyncBuffer(pLoaded.vBuffer, 0, pLoaded.vBuffer.vSize, pError)) {
    vDevice.mDestroyBuffer(pLoaded.vBuffer);
    return false;
  }
  pLoaded.vOutputs  = vShape.vN;
  pLoaded.vInputs   = pInputs;
  pLoaded.vSegments = vParts;
  for (uint32_t i = 0; i < vParts; i++) {
    pLoaded.aParts[i] = aParts[i];
  }
  return true;
}

void XdnaRunner::mFreeGemvWeights(XdnaGemvWeights & pLoaded) {
  vDevice.mDestroyBuffer(pLoaded.vBuffer);
  pLoaded = XdnaGemvWeights();
}

bool XdnaRunner::mMultiplyRows(const float * pX, uint32_t pRows, size_t pStride, const XdnaGemvWeights & pWeights,
                                const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaGemmTimes & pTimes,
                                std::string & pError) {
  if (!vIsOpen || !vHasGemv) {
    pError = "the overlay for generating tokens is not open";
    return false;
  }
  if (vBroken) {
    pError = "a command did not end before: nothing else is sent to the NPU";
    return false;
  }
  if (pWeights.vBuffer.vMap == nullptr) {
    pError = "the weights are not loaded";
    return false;
  }
  XdnaGemvShape vShape;
  if (!fXdnaGemvShape(vGemvOverlay, pRows, pWeights.vInputs, pWeights.aParts, pWeights.vSegments, vShape, pError)) {
    return false;
  }

  // the instruction stream of this size, written the first time it is needed
  double                vStart = fNowMs();
  std::vector<uint32_t> vKey   = {pRows, pWeights.vInputs, pWeights.vSegments};
  for (uint32_t i = 0; i < pWeights.vSegments; i++) {
    vKey.push_back(uint32_t(pWeights.aParts[i].vFormat));
    vKey.push_back(pWeights.aParts[i].vOutputs);
  }
  auto vPlan = aGemvPlans.find(vKey);
  if (vPlan == aGemvPlans.end()) {
    std::vector<uint32_t> aInstructions;
    if (!fXdnaGemvPlan(vGemvOverlay, vShape, aInstructions, pError)) {
      return false;
    }
    XdnaBuffer    vBuffer;
    const uint64_t vBytes = aInstructions.size() * sizeof(uint32_t);
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, vBytes, vBuffer, pError)) {
      return false;
    }
    std::memcpy(vBuffer.vMap, aInstructions.data(), vBytes);
    if (!vDevice.mSyncBuffer(vBuffer, 0, vBytes, pError)) {
      vDevice.mDestroyBuffer(vBuffer);
      return false;
    }
    vPlan = aGemvPlans.emplace(vKey, std::make_pair(vBuffer, uint32_t(aInstructions.size()))).first;
  }
  const XdnaBuffer & vInstructionsGemv = vPlan->second.first;
  const uint32_t      vWordsGemv        = vPlan->second.second;
  if (!mEnsure(vBufferX, XdnaBufferType::cShare, vShape.vBytesX, pError) ||
      !mEnsure(vBufferY, XdnaBufferType::cShare, vShape.vBytesY, pError)) {
    return false;
  }
  pTimes.vPlanMs = fNowMs() - vStart;

  vStart = fNowMs();
  if (!fXdnaGemvPackX(vGemvOverlay, vShape, pX, pStride, static_cast<uint8_t *>(vBufferX.vMap))) {
    pError = "X has a value that is not finite";
    return false;
  }
  if (!vDevice.mSyncBuffer(vBufferX, 0, vShape.vBytesX, pError)) {
    return false;
  }
  pTimes.vPackMs = fNowMs() - vStart;

  // the same command as for prompts, on vGemvContext (compute unit 0), with the buffers X, W and Y
  uint32_t *     vWords = static_cast<uint32_t *>(vGemvCommand.vMap);
  const uint32_t vCount = 1 + 5 + 2 * cBufferSlots;
  std::memset(vWords, 0, cCommandBytes);
  vWords[0] = fXdnaCommandHeader(cXdnaCommandStartCu, vCount);
  vWords[1] = 0x1;
  vWords[2] = cOpcodeTransaction;
  vWords[4] = uint32_t(vInstructionsGemv.vDevAddr & 0xffffffffu);
  vWords[5] = uint32_t(vInstructionsGemv.vDevAddr >> 32);
  vWords[6] = vWordsGemv;
  const XdnaBuffer * aBuffers[3] = {&vBufferX, &pWeights.vBuffer, &vBufferY};
  for (uint32_t i = 0; i < 3; i++) {
    vWords[7 + 2 * i] = uint32_t(aBuffers[i]->vDevAddr & 0xffffffffu);
    vWords[8 + 2 * i] = uint32_t(aBuffers[i]->vDevAddr >> 32);
  }
  const std::vector<uint32_t> aHandles = {vInstructionsGemv.vHandle, vBufferX.vHandle, pWeights.vBuffer.vHandle,
                                          vBufferY.vHandle};

  vStart = fNowMs();
  XdnaWaitResult vWaited = XdnaWaitResult::cFailed;
  if (!mSubmitGemvAndWait(aHandles, vWaited, pError)) {
    return false;
  }
  pTimes.vNpuMs = fNowMs() - vStart;
  if (!mCompleted(vGemvCommand, vWaited, pError)) {
    return false;
  }

  vStart = fNowMs();
  if (!vDevice.mSyncBuffer(vBufferY, 0, vShape.vBytesY, pError)) {
    return false;
  }
  pTimes.vFlushMs = fNowMs() - vStart;
  fXdnaGemvUnpackY(vGemvOverlay, vShape, static_cast<const float *>(vBufferY.vMap), aTargets, pTargets, &vWorkers);
  pTimes.vUnpackMs = fNowMs() - vStart;
  return true;
}

bool XdnaRunner::mLoadChainMatrix(const std::vector<XdnaGemvMember> & aMembers, uint32_t pInputs, bool pApart,
                                   XdnaChainWeights & pLoaded, std::string & pError) {
  if (!vIsOpen || !vHasChain) {
    pError = "the chained overlay for generating tokens is not open";
    return false;
  }
  // the segments: every member apart, or consecutive members of one format together, and where each starts among
  // the members
  XdnaGemvPart         aParts[cXdnaGemvSegmentsMax];
  std::vector<uint32_t> aStarts;
  uint32_t              vParts = 0;
  for (uint32_t i = 0; i < aMembers.size(); i++) {
    if (aMembers[i].vOutputs == 0) {
      continue;
    }
    if (vParts == 0 || pApart || aParts[vParts - 1].vFormat != aMembers[i].vFormat) {
      if (vParts == cXdnaGemvSegmentsMax) {
        pError = "the matrices of weights make more segments than the chained overlay takes";
        return false;
      }
      aParts[vParts] = {aMembers[i].vFormat, 0};
      aStarts.push_back(i);
      vParts++;
    }
    aParts[vParts - 1].vOutputs += aMembers[i].vOutputs;
  }
  aStarts.push_back(uint32_t(aMembers.size()));
  XdnaChainMatrix vMatrix;
  if (!fXdnaChainMatrixShape(vChainOverlay, pInputs, aParts, vParts, vMatrix, pError)) {
    return false;
  }
  if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vMatrix.vBytesW, pLoaded.vBuffer, pError)) {
    return false;
  }
  // segment by segment and stream by stream: the rows of one stream, from the members they belong to, and then its
  // part of the buffer
  std::vector<uint8_t> aRows;
  for (uint32_t vIndex = 0; vIndex < vParts; vIndex++) {
    const XdnaChainSegment & vSegment   = vMatrix.aSegments[vIndex];
    const uint32_t            vPerStream = vSegment.vPadded / vChainOverlay.mStreams();
    const uint64_t            vRowBytes  = vSegment.vFormat == XdnaGemvFormat::cBfp16 ?
                                               uint64_t(pInputs) * sizeof(float) :
                                           vSegment.vFormat == XdnaGemvFormat::cQ8_0 ?
                                               uint64_t(pInputs) / 32 * 34 :
                                           vSegment.vFormat == XdnaGemvFormat::cQ5_1 ?
                                               uint64_t(pInputs) / 32 * 24 :
                                               uint64_t(pInputs) / 256 * fXdnaGemvBlockBytes(vSegment.vFormat);
    for (uint32_t vStream = 0; vStream < vChainOverlay.mStreams(); vStream++) {
      const uint64_t vFirst = uint64_t(vStream) * vPerStream;
      const uint32_t vGiven =
          vFirst >= vSegment.vOutputs ? 0 : uint32_t(std::min<uint64_t>(vPerStream, vSegment.vOutputs - vFirst));
      aRows.resize(std::max<uint64_t>(vGiven, 1) * vRowBytes);
      // the rows vFirst to vFirst + vGiven of the segment, member by member
      uint32_t vDone   = 0;
      uint64_t vOffset = 0;  // the first output of the member, in the segment
      for (uint32_t m = aStarts[vIndex]; m < aStarts[vIndex + 1] && vDone < vGiven; m++) {
        const uint64_t vEnd = vOffset + aMembers[m].vOutputs;
        const uint64_t vAt  = vFirst + vDone;
        if (aMembers[m].vOutputs > 0 && vAt < vEnd) {
          const uint32_t vCount = uint32_t(std::min<uint64_t>(vEnd - vAt, vGiven - vDone));
          if (!aMembers[m].fRows(uint32_t(vAt - vOffset), vCount, aRows.data() + uint64_t(vDone) * vRowBytes)) {
            pError = "the rows of the weights could not be read";
            vDevice.mDestroyBuffer(pLoaded.vBuffer);
            return false;
          }
          vDone += vCount;
        }
        vOffset = vEnd;
      }
      if (!fXdnaChainPackStream(vChainOverlay, vMatrix, vIndex, vStream, aRows.data(), vGiven,
                                 static_cast<uint8_t *>(pLoaded.vBuffer.vMap), &vWorkers)) {
        pError = "the weights have a value that is not finite";
        vDevice.mDestroyBuffer(pLoaded.vBuffer);
        return false;
      }
    }
  }
  if (!vDevice.mSyncBuffer(pLoaded.vBuffer, 0, pLoaded.vBuffer.vSize, pError)) {
    vDevice.mDestroyBuffer(pLoaded.vBuffer);
    return false;
  }
  pLoaded.vMatrix = vMatrix;
  return true;
}

bool XdnaRunner::mLoadChainLinear(const float * pConv, const float * pNorm, XdnaChainWeights & pLoaded,
                                   std::string & pError) {
  if (!vIsOpen || !vHasChain) {
    pError = "the chained overlay for generating tokens is not open";
    return false;
  }
  XdnaChainMatrix vMatrix;
  fXdnaChainLinearMatrix(vChainOverlay, vMatrix);
  if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vMatrix.vBytesW, pLoaded.vBuffer, pError)) {
    return false;
  }
  if (!fXdnaChainPackLinear(vChainOverlay, pConv, pNorm, static_cast<uint8_t *>(pLoaded.vBuffer.vMap))) {
    pError = "the weights of the layer of linear attention have a value that is not finite";
    vDevice.mDestroyBuffer(pLoaded.vBuffer);
    return false;
  }
  if (!vDevice.mSyncBuffer(pLoaded.vBuffer, 0, pLoaded.vBuffer.vSize, pError)) {
    vDevice.mDestroyBuffer(pLoaded.vBuffer);
    return false;
  }
  pLoaded.vMatrix = vMatrix;
  return true;
}

void XdnaRunner::mFreeChainWeights(XdnaChainWeights & pLoaded) {
  vDevice.mDestroyBuffer(pLoaded.vBuffer);
  pLoaded = XdnaChainWeights();
}

bool XdnaRunner::mChainMultiplyRows(const float * pX, uint32_t pRows, size_t pStride, const XdnaChainWeights & pWeights,
                                     const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaGemmTimes & pTimes,
                                     std::string & pError) {
  if (!vIsOpen || !vHasChain) {
    pError = "the chained overlay for generating tokens is not open";
    return false;
  }
  const XdnaChainMatrix & vMatrix = pWeights.vMatrix;
  const uint32_t           vEach   = vChainOverlay.vSegments;
  const uint32_t           vCount  = (vMatrix.vSegments + vEach - 1) / vEach;
  if (pRows == 0 || pRows > cXdnaGemvRowsMax || vCount == 0 || vCount > cXdnaChainPhasesMax) {
    pError = "the chained overlay multiplies 1 to " + std::to_string(cXdnaGemvRowsMax) + " rows";
    return false;
  }
  // a phase for every segments of the overlay, all with the same X from the host
  XdnaChainPhase aPhases[cXdnaChainPhasesMax];
  XdnaChainIo    aIo[cXdnaChainPhasesMax];
  for (uint32_t q = 0; q < vCount; q++) {
    aPhases[q].vPrologue     = pRows == 1 ? XdnaChainPrologue::cFloats : XdnaChainPrologue::cBfp16;
    aPhases[q].vRows         = pRows;
    aPhases[q].vSameXAs      = q == 0 ? -1 : 0;
    aPhases[q].vMatrix       = 0;
    aPhases[q].vFirstSegment = q * vEach;
    aPhases[q].vSegments     = std::min(vEach, vMatrix.vSegments - q * vEach);
    aIo[q].vX                = q == 0 ? pX : nullptr;
    aIo[q].vStride           = pStride;
    aIo[q].aTargets          = aTargets;
    aIo[q].vTargets          = pTargets;
  }
  const XdnaChainWeights * aWeights[1] = {&pWeights};
  return mRunChain(aPhases, aIo, vCount, aWeights, 1, pTimes, pError);
}

bool XdnaRunner::mPrepareGdnB(uint32_t pTokens, uint32_t pKeyHeads, uint32_t pValueHeads, uint8_t *& pB,
                              uint64_t & pPartBytesB, uint32_t & pParts, XdnaGemmTimes & pTimes, std::string & pError) {
  pB          = nullptr;
  pPartBytesB = 0;
  pParts      = 0;
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  XdnaGdnIo vIo;
  vIo.vTokens     = pTokens;
  vIo.vKeyHeads   = pKeyHeads;
  vIo.vValueHeads = pValueHeads;
  uint32_t vRounds = 0;
  uint32_t vGroups = 0;
  if (!fXdnaGdnShape(vIo, vRounds, vGroups)) {
    pError = "the recurrence takes 1 to 4 value heads per head of keys, and these in pairs";
    return false;
  }
  uint32_t vParts = std::max<uint32_t>(1, std::min<uint32_t>(vGdnParts, cXdnaRunnerSlotsC));
  while (vRounds % vParts != 0) {
    vParts--;
  }
  const uint32_t vRoundsPart = vRounds / vParts;
  const double   vStart      = fNowMs();
  const auto     vKey        = std::make_pair(vRoundsPart, vGroups);
  auto           vFound      = aGdnPlans.find(vKey);
  if (vFound == aGdnPlans.end()) {
    XdnaGemmPlan vPlan;
    if (!fXdnaGdnPlan(vOverlay, vRoundsPart, vGroups, vPlan, pError)) {
      return false;
    }
    XdnaGdnCached  vCached;
    const uint64_t vBytes = vPlan.aInstructions.size() * sizeof(uint32_t);
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, vBytes, vCached.vBuffer, pError)) {
      return false;
    }
    std::memcpy(vCached.vBuffer.vMap, vPlan.aInstructions.data(), vBytes);
    if (!vDevice.mSyncBuffer(vCached.vBuffer, 0, vBytes, pError)) {
      vDevice.mDestroyBuffer(vCached.vBuffer);
      return false;
    }
    vCached.vWords  = uint32_t(vPlan.aInstructions.size());
    vCached.vBytesA = vPlan.vBytesA;
    vCached.vBytesB = vPlan.vBytesB;
    vCached.vBytesC = vPlan.vBytesC;
    vFound          = aGdnPlans.emplace(vKey, vCached).first;
  }
  const XdnaGdnCached & vPlan = vFound->second;
  if (!mEnsure(vGdnB, XdnaBufferType::cShare, vParts * vPlan.vBytesB, pError)) {
    return false;
  }
  pTimes.vPlanMs += fNowMs() - vStart;
  pB          = static_cast<uint8_t *>(vGdnB.vMap);
  pPartBytesB = vPlan.vBytesB;
  pParts      = vParts;
  return true;
}

// llama.cpp-xdna: see xdna-runner.h
bool XdnaRunner::mRunGdn(const XdnaGdnIo & pIo, XdnaGemmTimes & pTimes, std::string & pError) {
  if (!vIsOpen) {
    pError = "the NPU is not open";
    return false;
  }
  if (vBroken) {
    pError = "a command did not end before: nothing else is sent to the NPU";
    return false;
  }
  uint32_t vRounds = 0;
  uint32_t vGroups = 0;
  if (!fXdnaGdnShape(pIo, vRounds, vGroups)) {
    pError = "the recurrence takes twice as many value heads as heads of keys, and these in pairs";
    return false;
  }
  // as many parts as divide the rounds, up to the commands the runner keeps in flight
  uint32_t vParts = std::max<uint32_t>(1, std::min<uint32_t>(vGdnParts, cXdnaRunnerSlotsC));
  while (vRounds % vParts != 0) {
    vParts--;
  }
  const uint32_t vRoundsPart = vRounds / vParts;
  // the instruction stream of the rounds of a part and these groups, written the first time
  const double vStart = fNowMs();
  const auto   vKey   = std::make_pair(vRoundsPart, vGroups);
  auto         vFound = aGdnPlans.find(vKey);
  if (vFound == aGdnPlans.end()) {
    XdnaGemmPlan vPlan;
    if (!fXdnaGdnPlan(vOverlay, vRoundsPart, vGroups, vPlan, pError)) {
      return false;
    }
    XdnaGdnCached vCached;
    const uint64_t vBytes = vPlan.aInstructions.size() * sizeof(uint32_t);
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, vBytes, vCached.vBuffer, pError)) {
      return false;
    }
    std::memcpy(vCached.vBuffer.vMap, vPlan.aInstructions.data(), vBytes);
    if (!vDevice.mSyncBuffer(vCached.vBuffer, 0, vBytes, pError)) {
      vDevice.mDestroyBuffer(vCached.vBuffer);
      return false;
    }
    vCached.vWords  = uint32_t(vPlan.aInstructions.size());
    vCached.vBytesA = vPlan.vBytesA;
    vCached.vBytesB = vPlan.vBytesB;
    vCached.vBytesC = vPlan.vBytesC;
    vFound          = aGdnPlans.emplace(vKey, vCached).first;
  }
  const XdnaGdnCached & vPlan = vFound->second;
  if (!mEnsure(vGdnA, XdnaBufferType::cShare, vParts * vPlan.vBytesA, pError) ||
      !mEnsure(vGdnB, XdnaBufferType::cShare, vParts * vPlan.vBytesB, pError) ||
      !mEnsure(vGdnC, XdnaBufferType::cShare, vParts * vPlan.vBytesC, pError)) {
    return false;
  }
  pTimes.vPlanMs += fNowMs() - vStart;

  // every part: its bands and chunks (every task of the threads takes what it wrote out of the cache of the
  // processor), and its command, as the one of a multiplication (see mPrepare) with its parts of the buffers A, B and
  // C of the recurrence, sent without waiting
  const XdnaGdnFlush vFlush = [](const void * pFirst, uint64_t pBytes) { fXdnaFlushCache(pFirst, 1, pBytes, pBytes); };
  const std::vector<uint32_t> aHandles = {vPlan.vBuffer.vHandle, vGdnA.vHandle, vGdnB.vHandle, vGdnC.vHandle};
  uint64_t                    aSeq[cXdnaRunnerSlotsC] = {};
  uint32_t                    vSent                    = 0;
  bool                        vOk                      = true;
  for (uint32_t vPart = 0; vPart < vParts; vPart++) {
    const double vPackStart = fNowMs();
    fXdnaGdnPack(pIo, vPart * vRoundsPart, vRoundsPart, static_cast<uint8_t *>(vGdnA.vMap) + vPart * vPlan.vBytesA,
                  static_cast<uint8_t *>(vGdnB.vMap) + vPart * vPlan.vBytesB, &vWorkers, vFlush);
    pTimes.vPackMs += fNowMs() - vPackStart;
    uint32_t *     vWords = static_cast<uint32_t *>(aCommandAsync[vPart].vMap);
    const uint32_t vCount = 1 + 5 + 2 * cBufferSlots;
    std::memset(vWords, 0, cCommandBytes);
    vWords[0] = fXdnaCommandHeader(cXdnaCommandStartCu, vCount);
    vWords[1] = 0x1;
    vWords[2] = cOpcodeTransaction;
    vWords[4] = uint32_t(vPlan.vBuffer.vDevAddr & 0xffffffffu);
    vWords[5] = uint32_t(vPlan.vBuffer.vDevAddr >> 32);
    vWords[6] = vPlan.vWords;
    const uint64_t aAddresses[3] = {vGdnA.vDevAddr + vPart * vPlan.vBytesA, vGdnB.vDevAddr + vPart * vPlan.vBytesB,
                                    vGdnC.vDevAddr + vPart * vPlan.vBytesC};
    for (uint32_t i = 0; i < 3; i++) {
      vWords[7 + 2 * i] = uint32_t(aAddresses[i] & 0xffffffffu);
      vWords[8 + 2 * i] = uint32_t(aAddresses[i] >> 32);
    }
    mNoteUnit(aCommandAsync[vPart]);
    if (!vDevice.mSubmit(vContext, aCommandAsync[vPart], aHandles, aSeq[vPart], pError)) {
      vOk = false;
      break;
    }
    vSent++;
  }

  // every command sent is waited for, in order and also after an error, before its buffers are used again; what the
  // NPU wrote may not be what the cache of the host holds, so every task flushes its piece before reading it
  for (uint32_t vPart = 0; vPart < vSent; vPart++) {
    const double    vNpuStart = fNowMs();
    XdnaWaitResult vWaited   = XdnaWaitResult::cFailed;
    std::string     vError;
    auto            fWait = [&]() {
      vWaited = vDevice.mWait(vContext, aSeq[vPart], cTimeoutMs, vError);
    };
    // while the NPU works, the other threads of OpenMP sleep instead of spinning (see XdnaWorkers::mWhileParked)
    if (vPark) {
      vWorkers.mWhileParked(fWait);
    } else {
      fWait();
    }
    pTimes.vNpuMs += fNowMs() - vNpuStart;
    if (!mCompleted(aCommandAsync[vPart], vWaited, vError)) {
      if (vOk) {
        pError = vError;
      }
      vOk = false;
    }
    if (!vOk) {
      continue;
    }
    const double vReadStart = fNowMs();
    if (!fXdnaGdnUnpack(pIo, vPart * vRoundsPart, vRoundsPart,
                        static_cast<const uint8_t *>(vGdnC.vMap) + vPart * vPlan.vBytesC, &vWorkers, vFlush)) {
      if (vOk) {
        pError = "the gated norm has a value that is not finite";
      }
      vOk = false;
    }
    pTimes.vUnpackMs += fNowMs() - vReadStart;
  }
  return vOk;
}

bool XdnaRunner::mRunChain(const XdnaChainPhase * aPhases, const XdnaChainIo * aIo, uint32_t pPhases,
                            const XdnaChainWeights * const * aWeights, uint32_t pMatrices, XdnaGemmTimes & pTimes,
                            std::string & pError, const XdnaBuffer * const * aStates, uint32_t pStates) {
  if (!vIsOpen || !vHasChain) {
    pError = "the chained overlay for generating tokens is not open";
    return false;
  }
  if (vBroken) {
    pError = "a command did not end before: nothing else is sent to the NPU";
    return false;
  }
  if (pMatrices + pStates == 0 || pMatrices > cXdnaChainMatricesMax || 1 + pMatrices + pStates > cBufferSlots) {
    pError = "a command of the chained overlay runs 1 to 4 matrices and buffers of states";
    return false;
  }
  for (uint32_t i = 0; i < pStates; i++) {
    if (aStates == nullptr || aStates[i] == nullptr || aStates[i]->vMap == nullptr) {
      pError = "a buffer of states is not there";
      return false;
    }
  }
  XdnaChainMatrix aMatrices[cXdnaChainMatricesMax];
  for (uint32_t m = 0; m < pMatrices; m++) {
    if (aWeights[m] == nullptr || aWeights[m]->vBuffer.vMap == nullptr) {
      pError = "the weights are not loaded";
      return false;
    }
    aMatrices[m] = aWeights[m]->vMatrix;
  }
  double            vStart = fNowMs();
  XdnaChainCommand vChain;
  if (!fXdnaChainCommandShape(vChainOverlay, aPhases, pPhases, aMatrices, pMatrices, vChain, pError, pStates)) {
    return false;
  }
  // the instruction stream of this command, written the first time it is needed
  const std::vector<uint32_t> aKey  = fXdnaChainKey(vChain);
  auto                        vPlan = aChainPlans.find(aKey);
  if (vPlan == aChainPlans.end()) {
    std::vector<uint32_t> aInstructions;
    if (!fXdnaChainPlan(vChainOverlay, vChain, aInstructions, pError, vEarly, vFillsFirst)) {
      return false;
    }
    XdnaBuffer    vBuffer;
    const uint64_t vBytes = aInstructions.size() * sizeof(uint32_t);
    if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, vBytes, vBuffer, pError)) {
      return false;
    }
    std::memcpy(vBuffer.vMap, aInstructions.data(), vBytes);
    if (!vDevice.mSyncBuffer(vBuffer, 0, vBytes, pError)) {
      vDevice.mDestroyBuffer(vBuffer);
      return false;
    }
    vPlan = aChainPlans.emplace(aKey, std::make_pair(vBuffer, uint32_t(aInstructions.size()))).first;
  }
  const XdnaBuffer & vInstructions = vPlan->second.first;
  const uint32_t      vWords        = vPlan->second.second;
  XdnaBuffer & vIoBuffer = pPhases > 1 ? vBufferIoChain : vBufferIo;
  if (!mEnsure(vIoBuffer, XdnaBufferType::cShare, vChain.vBytesIo, pError)) {
    return false;
  }
  pTimes.vPlanMs += fNowMs() - vStart;

  // what the host gives: the parameters, and the X, residuals and weights of norms of the phases
  vStart        = fNowMs();
  uint8_t * vIo = static_cast<uint8_t *>(vIoBuffer.vMap);
  fXdnaChainWriteParameters(vChainOverlay, vChain, vIo);
  for (uint32_t p = 0; p < pPhases; p++) {
    const XdnaChainPhase & vPhase = aPhases[p];
    if (vPhase.vRecurrence.vLinear != 0) {
      if (aIo[p].vLinear == nullptr || !fXdnaChainWriteLinear(vChainOverlay, vChain, p, *aIo[p].vLinear, vIo)) {
        pError = "the inputs of a phase of linear attention are not given, or have a value that is not finite";
        return false;
      }
      continue;
    }
    if (vPhase.vRecurrence.vHeads != 0) {
      if (aIo[p].vRecurrence == nullptr || !fXdnaChainWriteRecurrence(vChainOverlay, vChain, p, *aIo[p].vRecurrence, vIo)) {
        pError = "the inputs of a phase of the recurrence are not given, or have a value that is not finite";
        return false;
      }
      continue;
    }
    if (vPhase.vFromPhase < 0 && vPhase.vSameXAs < 0) {
      if (aIo[p].vX == nullptr) {
        pError = "a phase takes its X from the host, and it is not given";
        return false;
      }
      if (!fXdnaChainWriteX(vChainOverlay, vChain, p, aIo[p].vX, aIo[p].vStride, vIo)) {
        pError = "X has a value that is not finite";
        return false;
      }
    }
    if (vPhase.vEpilogue == XdnaChainEpilogue::cResidual) {
      if (aIo[p].vResidual == nullptr) {
        pError = "a phase adds a residual, and it is not given";
        return false;
      }
      fXdnaChainWriteResidual(vChainOverlay, vChain, p, aIo[p].vResidual, vIo);
    }
    if (vPhase.vPrologue == XdnaChainPrologue::cNorm) {
      if (aIo[p].vNorm == nullptr) {
        pError = "a phase has a norm, and its weights are not given";
        return false;
      }
      fXdnaChainWriteNorm(vChainOverlay, vChain, p, aIo[p].vNorm, vIo);
    }
  }
  // What the host wrote leaves the cache, and so does what comes after it up to the end of its page and the page
  // after it. After a run of stores, the processor leaves the next cache line modified, with what it held, though
  // nothing wrote it (measured on 2026/11/15, likely its prefetching for stores, which leaves lines modified on AMD
  // processors): the flush after a later command whose Y covers that line wrote it back over its results. It took a
  // command of K = 9216, whose input ends at byte 46080, before the logits (whose Y covers that byte): one cache line
  // of logits, outputs 544 to 551 of Qwen3.5 4B, came back as what the command before had left there. Prefetching
  // does not cross pages: flushing to the end of the page, and one more for safety, covers it.
  // The flush is that of the driver (drm_clflush_virt_range) done here: a call to the driver costs some microseconds,
  // and a command flushes twice.
  const uint64_t vFlushed = std::min<uint64_t>(vIoBuffer.vSize, (vChain.vBytesIn + 4095) / 4096 * 4096 + 4096);
  if (vUserFlush) {
    fXdnaFlushCache(vIo, 1, vFlushed, vFlushed);
  } else if (!vDevice.mSyncBuffer(vIoBuffer, 0, vFlushed, pError)) {
    return false;
  }
  pTimes.vPackMs += fNowMs() - vStart;

  // the command, on vGemvContext (compute unit 0), with the buffer of the host and those of the matrices
  uint32_t *     vCommandWords = static_cast<uint32_t *>(vGemvCommand.vMap);
  const uint32_t vCount        = 1 + 5 + 2 * cBufferSlots;
  std::memset(vCommandWords, 0, cCommandBytes);
  vCommandWords[0] = fXdnaCommandHeader(cXdnaCommandStartCu, vCount);
  vCommandWords[1] = 0x1;
  vCommandWords[2] = cOpcodeTransaction;
  vCommandWords[4] = uint32_t(vInstructions.vDevAddr & 0xffffffffu);
  vCommandWords[5] = uint32_t(vInstructions.vDevAddr >> 32);
  vCommandWords[6] = vWords;
  std::vector<uint32_t>             aHandles = {vInstructions.vHandle, vIoBuffer.vHandle};
  std::vector<const XdnaBuffer *> aBuffers = {&vIoBuffer};
  for (uint32_t m = 0; m < pMatrices; m++) {
    aBuffers.push_back(&aWeights[m]->vBuffer);
    if (std::find(aHandles.begin(), aHandles.end(), aWeights[m]->vBuffer.vHandle) == aHandles.end()) {
      aHandles.push_back(aWeights[m]->vBuffer.vHandle);
    }
  }
  for (uint32_t i = 0; i < pStates; i++) {
    aBuffers.push_back(aStates[i]);
    if (std::find(aHandles.begin(), aHandles.end(), aStates[i]->vHandle) == aHandles.end()) {
      aHandles.push_back(aStates[i]->vHandle);
    }
  }
  for (uint32_t i = 0; i < aBuffers.size(); i++) {
    vCommandWords[7 + 2 * i] = uint32_t(aBuffers[i]->vDevAddr & 0xffffffffu);
    vCommandWords[8 + 2 * i] = uint32_t(aBuffers[i]->vDevAddr >> 32);
  }
  vStart                  = fNowMs();
  XdnaWaitResult vWaited = XdnaWaitResult::cFailed;
  if (!mSubmitGemvAndWait(aHandles, vWaited, pError)) {
    return false;
  }
  pTimes.vNpuMs += fNowMs() - vStart;
  if (!mCompleted(vGemvCommand, vWaited, pError)) {
    return false;
  }

  // What it gives back: the Y of the phases that have targets, out of the cache of the processor first. Only the lines
  // that will be read leave it (fXdnaChainReads: with version 7 and one row, the outputs in their order; the logits of
  // Qwen3.5 4B are 1 MB). The others may stay in the cache as they were, as nothing reads them, and nothing wrote them:
  // what the host writes leaves the cache before every command.
  vStart = fNowMs();
  if (vUserFlush) {
    for (uint32_t p = 0; p < pPhases; p++) {
      if (aIo[p].vTargets == 0 && aIo[p].vOutputs == nullptr) {
        continue;
      }
      XdnaChainRead aReads[cXdnaChainSegmentsMax];
      const uint32_t vReads = fXdnaChainReads(vChain, p, aReads);
      for (uint32_t r = 0; r < vReads; r++) {
        fXdnaFlushCache(vIo + aReads[r].vOffset, aReads[r].vRecords, aReads[r].vStride, aReads[r].vBytes);
      }
    }
  } else {
    uint64_t vFirst = vChain.vBytesIo;
    uint64_t vEnd   = 0;
    for (uint32_t p = 0; p < pPhases; p++) {
      if (aIo[p].vTargets > 0 || aIo[p].vOutputs != nullptr) {
        vFirst = std::min(vFirst, vChain.aPlaces[p].vOffsetY[0]);
        vEnd   = std::max(vEnd, vChain.aPlaces[p].vOffsetY[0] + vChain.aPlaces[p].vBytesY);
      }
    }
    if (vEnd > vFirst && !vDevice.mSyncBuffer(vIoBuffer, vFirst, vEnd - vFirst, pError)) {
      return false;
    }
  }
  pTimes.vFlushMs += fNowMs() - vStart;
  for (uint32_t p = 0; p < pPhases; p++) {
    if (aPhases[p].vRecurrence.vHeads != 0) {
      if (aIo[p].vOutputs != nullptr) {
        fXdnaChainReadRecurrence(vChain, p, vIo, aIo[p].vOutputs, aIo[p].vScale);
      }
      continue;
    }
    if (aIo[p].vTargets > 0) {
      fXdnaChainReadY(vChainOverlay, vChain, p, vIo, aIo[p].aTargets, aIo[p].vTargets);
    }
  }
  pTimes.vUnpackMs += fNowMs() - vStart;
  return true;
}

bool XdnaRunner::mLoadMoeMatrix(uint32_t pExperts, uint32_t pInputs, const XdnaMoeSegmentSource * aSources,
                                 uint32_t pSources, bool pSwiglu, XdnaMoeWeights & pLoaded, std::string & pError) {
  if (!vIsOpen || !vHasMoe) {
    pError = "the mixture-of-experts overlay is not open";
    return false;
  }
  if (pSources == 0 || pSources > cXdnaMoeSegmentsMax || aSources == nullptr) {
    pError = "the mixture-of-experts matrix has 1 or 2 segments";
    return false;
  }
  XdnaGemvPart aParts[cXdnaMoeSegmentsMax];
  for (uint32_t s = 0; s < pSources; s++) {
    aParts[s] = {aSources[s].vFormat, aSources[s].vOutputs};
  }
  XdnaMoeMatrix vMatrix;
  if (!fXdnaMoeMatrixShape(vMoeOverlay, pExperts, pInputs, aParts, pSources, pSwiglu, vMatrix, pError)) {
    return false;
  }
  pLoaded = XdnaMoeWeights();
  if (!vDevice.mCreateBuffer(XdnaBufferType::cShare, vMatrix.vBytesW, pLoaded.vBuffer, pError)) {
    return false;
  }
  uint8_t *         vBuffer = static_cast<uint8_t *>(pLoaded.vBuffer.vMap);
  std::atomic<bool> vOk(true);
  vWorkers.mForEach(pExperts, [&](uint32_t pExpert) {
    std::vector<uint8_t> aScratch;
    for (uint32_t s = 0; s < pSources; s++) {
      if (!vOk.load(std::memory_order_relaxed)) {
        return;
      }
      if (aSources[s].vNeedsScratch) {
        const uint64_t vScratchBytes = uint64_t(aSources[s].vOutputs) * aSources[s].vRowBytes;
        if (aScratch.size() < vScratchBytes) {
          aScratch.resize(vScratchBytes);
        }
      }
      const void * vRows = aSources[s].fExpertRows(pExpert, aSources[s].vNeedsScratch ? aScratch.data() : nullptr);
      if (vRows == nullptr ||
          !fXdnaMoePackExpertSegment(vMoeOverlay, vMatrix, pExpert, s, vRows, aSources[s].vRowBytes, vBuffer)) {
        vOk.store(false, std::memory_order_relaxed);
        return;
      }
    }
  });
  if (!vOk) {
    pError = "the expert weights have a value that is not finite";
    vDevice.mDestroyBuffer(pLoaded.vBuffer);
    return false;
  }
  if (!vDevice.mSyncBuffer(pLoaded.vBuffer, 0, pLoaded.vBuffer.vSize, pError)) {
    vDevice.mDestroyBuffer(pLoaded.vBuffer);
    return false;
  }
  pLoaded.vMatrix = vMatrix;
  return true;
}

void XdnaRunner::mFreeMoeWeights(XdnaMoeWeights & pLoaded) {
  vDevice.mDestroyBuffer(pLoaded.vBuffer);
  pLoaded = XdnaMoeWeights();
}

bool XdnaRunner::mSubmitMoeCommand(const XdnaMoeCommand & pCommand, XdnaBuffer & pIoBuffer,
                                    const XdnaMoeWeights & pWeights, XdnaGemmTimes & pTimes, std::string & pError) {
  constexpr uint64_t cMoeInstsMaxBytes = 8ull << 20;
  double             vStart            = fNowMs();
  const XdnaBuffer * vInstBuf          = nullptr;
  uint32_t           vInstWords        = 0;
  if (pCommand.vChunks == 1 && pCommand.aChunks[0].vExpert == 0) {
    const std::vector<uint32_t> aKey = {
      pCommand.vMatrix.vK,
      pCommand.vMatrix.vOutputs,
      pCommand.vMatrix.vSegments,
      uint32_t(pCommand.vMatrix.vSwiglu),
      uint32_t(pCommand.vMatrix.aSegments[0].vFormat),
      uint32_t(pCommand.vMatrix.aSegments[1].vFormat),
      pCommand.aChunks[0].vRows,
    };
    auto vPlan = aMoeDensePlans.find(aKey);
    if (vPlan == aMoeDensePlans.end()) {
      std::vector<uint32_t> aInstructions;
      if (!fXdnaMoePlan(vMoeOverlay, pCommand, aInstructions, pError)) {
        return false;
      }
      XdnaBuffer     vBuffer;
      const uint64_t vBytes = aInstructions.size() * sizeof(uint32_t);
      if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, vBytes, vBuffer, pError)) {
        return false;
      }
      std::memcpy(vBuffer.vMap, aInstructions.data(), vBytes);
      if (!vDevice.mSyncBuffer(vBuffer, 0, vBytes, pError)) {
        vDevice.mDestroyBuffer(vBuffer);
        return false;
      }
      vPlan = aMoeDensePlans.emplace(aKey, std::make_pair(vBuffer, uint32_t(aInstructions.size()))).first;
    }
    vInstBuf   = &vPlan->second.first;
    vInstWords = vPlan->second.second;
  } else {
    if (!fXdnaMoePlan(vMoeOverlay, pCommand, aMoeInstScratch, pError)) {
      return false;
    }
    const uint64_t vInstBytes = aMoeInstScratch.size() * sizeof(uint32_t);
    if (vMoeInsts.vMap == nullptr || vMoeInsts.vSize < vInstBytes) {
      vDevice.mDestroyBuffer(vMoeInsts);
      if (!vDevice.mCreateBuffer(XdnaBufferType::cDev, std::max<uint64_t>(cMoeInstsMaxBytes, vInstBytes), vMoeInsts,
                                 pError)) {
        return false;
      }
    }
    std::memcpy(vMoeInsts.vMap, aMoeInstScratch.data(), vInstBytes);
    if (vUserFlush) {
      if (!mFlushShared(vMoeInsts, vInstBytes, true, pError)) {
        return false;
      }
    } else if (!vDevice.mSyncBuffer(vMoeInsts, 0, vInstBytes, pError)) {
      return false;
    }
    vInstBuf   = &vMoeInsts;
    vInstWords = uint32_t(aMoeInstScratch.size());
  }
  pTimes.vPlanMs += fNowMs() - vStart;

  vStart                    = fNowMs();
  uint8_t *      vIo        = static_cast<uint8_t *>(pIoBuffer.vMap);
  const uint64_t vFlushedIn = std::min<uint64_t>(pIoBuffer.vSize, (pCommand.vBytesIn + 4095) / 4096 * 4096 + 4096);
  if (vUserFlush) {
    if (!mFlushShared(pIoBuffer, pCommand.vBytesIn, true, pError)) {
      return false;
    }
    (void) vIo;
  } else if (!vDevice.mSyncBuffer(pIoBuffer, 0, vFlushedIn, pError)) {
    return false;
  }
  pTimes.vPackMs += fNowMs() - vStart;

  uint32_t *     vCommandWords = static_cast<uint32_t *>(vMoeCommand.vMap);
  const uint32_t vCount        = 1 + 5 + 2 * cBufferSlots;
  std::memset(vCommandWords, 0, cCommandBytes);
  vCommandWords[0]  = fXdnaCommandHeader(cXdnaCommandStartCu, vCount);
  vCommandWords[1]  = vMoeUnitMask;
  vCommandWords[2]  = cOpcodeTransaction;
  vCommandWords[4]  = uint32_t(vInstBuf->vDevAddr & 0xffffffffu);
  vCommandWords[5]  = uint32_t(vInstBuf->vDevAddr >> 32);
  vCommandWords[6]  = vInstWords;
  vCommandWords[7]  = uint32_t(pIoBuffer.vDevAddr & 0xffffffffu);
  vCommandWords[8]  = uint32_t(pIoBuffer.vDevAddr >> 32);
  vCommandWords[9]  = uint32_t(pWeights.vBuffer.vDevAddr & 0xffffffffu);
  vCommandWords[10] = uint32_t(pWeights.vBuffer.vDevAddr >> 32);
  const std::vector<uint32_t> aHandles = {vInstBuf->vHandle, pIoBuffer.vHandle, pWeights.vBuffer.vHandle};

  vStart                  = fNowMs();
  XdnaWaitResult vWaited = XdnaWaitResult::cFailed;
  bool           vSent   = false;
  auto           fRun    = [&]() {
    uint64_t vSeq = 0;
    if (vLastUnits != 0 && vLastUnits != 0x100u) {
      vSwitches++;
    }
    vLastUnits = 0x100u;
    vSent      = vDevice.mSubmit(vMoeContext, vMoeCommand, aHandles, vSeq, pError);
    if (vSent) {
      vWaited = vDevice.mWait(vMoeContext, vSeq, cTimeoutMs, pError);
    }
  };
  if (vPark) {
    vWorkers.mWhileParked(fRun);
  } else {
    fRun();
  }
  if (!vSent) {
    return false;
  }
  pTimes.vNpuMs += fNowMs() - vStart;
  return mCompleted(vMoeCommand, vWaited, pError);
}

bool XdnaRunner::mFlushMoeY(const XdnaMoeCommand & pCommand, XdnaBuffer & pIoBuffer, XdnaGemmTimes & pTimes,
                             std::string & pError) {
  const double vStart = fNowMs();
  if (pCommand.vBytesY > 0) {
    if (vUserFlush) {
      uint8_t *      vBase    = static_cast<uint8_t *>(pIoBuffer.vMap) + pCommand.vOffsetY;
      const uint32_t vStreams = vMoeOverlay.mStreams();
      if (pCommand.vBytesY >= (256u << 10)) {
        vWorkers.mForEach(vStreams, [&](uint32_t pStream) {
          fXdnaFlushCache(vBase + uint64_t(pStream) * pCommand.vStreamBytesY, 1, pCommand.vStreamBytesY,
                          pCommand.vStreamBytesY);
        });
      } else {
        fXdnaFlushCache(vBase, 1, pCommand.vBytesY, pCommand.vBytesY);
      }
    } else if (!vDevice.mSyncBuffer(pIoBuffer, pCommand.vOffsetY, pCommand.vBytesY, pError)) {
      return false;
    }
  }
  const double vElapsed = fNowMs() - vStart;
  pTimes.vFlushMs += vElapsed;
  pTimes.vUnpackMs += vElapsed;
  return true;
}

bool XdnaRunner::mRunMoe(const XdnaMoeWeights & pWeights, const XdnaMoeChunk * aChunks, uint32_t pChunks,
                          const float * pSharedRows, size_t pSharedRowStride, uint32_t pSharedRowCount,
                          const uint32_t * aRowIndices, const float * const * aRowPtrs, const XdnaMoeRowOut * aRowsOut,
                          XdnaGemmTimes & pTimes, std::string & pError, XdnaMoeCommand * pKeptCommand,
                          const XdnaRowNorm * pSharedNorm, const XdnaGemmTarget * aTargets, uint32_t pTargets,
                          const float * pInitSrc, float * pInitDst, uint32_t pInitRows, uint64_t pInitStride) {
  if (!vIsOpen || !vHasMoe) {
    pError = "the mixture-of-experts overlay is not open";
    return false;
  }
  if (vBroken) {
    pError = "a command did not end before: nothing else is sent to the NPU";
    return false;
  }
  if (pWeights.vBuffer.vMap == nullptr) {
    pError = "the expert weights are not loaded";
    return false;
  }
  vMoeKept = false;
  double          vStart = fNowMs();
  XdnaMoeCommand vCmd;
  if (!fXdnaMoeCommandShape(vMoeOverlay, pWeights.vMatrix, aChunks, pChunks, vCmd, pError)) {
    return false;
  }
  XdnaBuffer & vIoBuf = aMoeIo[0];
  if (!mEnsure(vIoBuf, XdnaBufferType::cShare, vCmd.vBytesIo, pError)) {
    return false;
  }
  pTimes.vPlanMs += fNowMs() - vStart;

  vStart        = fNowMs();
  uint8_t * vIo = static_cast<uint8_t *>(vIoBuf.vMap);
  fXdnaMoeWriteParameters(vMoeOverlay, vCmd, vIo);
  std::atomic<bool> vFinite(true);
  if (pSharedRows != nullptr && aRowIndices != nullptr && pSharedRowCount > 0) {
    const uint64_t vRowBytes = fXdnaMoeRowEncodedBytes(vCmd.vMatrix);
    aMoeEncodedScratch.resize(uint64_t(pSharedRowCount) * vRowBytes);
    auto fEncodeRow = [&](uint32_t pRow) {
      const float * vNormWeights = pSharedNorm != nullptr ? pSharedNorm->vWeights : nullptr;
      const float   vNormScale   = pSharedNorm != nullptr ? pSharedNorm->vScales[pRow] : 1.0f;
      if (!fXdnaMoeEncodeRowBfp16(vCmd.vMatrix, pSharedRows + size_t(pRow) * pSharedRowStride,
                                   aMoeEncodedScratch.data() + uint64_t(pRow) * vRowBytes, vNormWeights, vNormScale)) {
        vFinite.store(false, std::memory_order_relaxed);
      }
    };
    if (pSharedRowCount >= 8) {
      vWorkers.mForEach(pSharedRowCount, fEncodeRow);
    } else {
      for (uint32_t r = 0; r < pSharedRowCount; r++) {
        fEncodeRow(r);
      }
    }
    if (!vFinite) {
      pError = "an input row of the experts has a value that is not finite";
      return false;
    }
    auto fPackChunk = [&](uint32_t pChunk) {
      const uint8_t *      aChunkRows[24];
      const uint32_t       vFirst = vCmd.aPlaces[pChunk].vFirstRowY;
      const uint32_t       vRows  = vCmd.aChunks[pChunk].vRows;
      for (uint32_t m = 0; m < vRows; m++) {
        aChunkRows[m] = aMoeEncodedScratch.data() + uint64_t(aRowIndices[vFirst + m]) * vRowBytes;
      }
      fXdnaMoeWriteChunkEncodedX(vMoeOverlay, vCmd, pChunk, aChunkRows, vIo);
    };
    if (pChunks >= 8) {
      vWorkers.mForEach(pChunks, fPackChunk);
    } else {
      for (uint32_t c = 0; c < pChunks; c++) {
        fPackChunk(c);
      }
    }
  } else if (aRowPtrs != nullptr) {
    auto fPackChunk = [&](uint32_t pChunk) {
      if (!fXdnaMoeWriteChunkX(vMoeOverlay, vCmd, pChunk, aRowPtrs + vCmd.aPlaces[pChunk].vFirstRowY, vIo)) {
        vFinite.store(false, std::memory_order_relaxed);
      }
    };
    if (pChunks >= 8) {
      vWorkers.mForEach(pChunks, fPackChunk);
    } else {
      for (uint32_t c = 0; c < pChunks; c++) {
        fPackChunk(c);
      }
    }
    if (!vFinite) {
      pError = "an input row of the experts has a value that is not finite";
      return false;
    }
  } else {
    pError = "no input rows given to mRunMoe";
    return false;
  }
  pTimes.vPackMs += fNowMs() - vStart;

  if (!mSubmitMoeCommand(vCmd, vIoBuf, pWeights, pTimes, pError)) {
    return false;
  }
  if (aRowsOut == nullptr && aTargets == nullptr) {
    if (pKeptCommand != nullptr) {
      *pKeptCommand = vCmd;
      vMoeKept      = true;
    }
    return true;
  }
  if (!mFlushMoeY(vCmd, vIoBuf, pTimes, pError)) {
    return false;
  }
  vStart = fNowMs();
  if (aTargets != nullptr) {
    fXdnaMoeReadYTargets(vMoeOverlay, vCmd, vIo, aTargets, pTargets, &vWorkers);
  } else {
    fXdnaMoeReadY(vMoeOverlay, vCmd, vIo, aRowsOut, &vWorkers, pInitSrc, pInitDst, pInitRows, pInitStride);
  }
  pTimes.vUnpackMs += fNowMs() - vStart;
  return true;
}

bool XdnaRunner::mRunMoeFromKept(const XdnaMoeCommand & pPrevCommand, const XdnaMoeWeights & pWeights,
                                  const float * aRowScales, const XdnaMoeRowOut * aRowsOut, XdnaGemmTimes & pTimes,
                                  std::string & pError, const XdnaGemmTarget * aTargets, uint32_t pTargets,
                                  const float * pInitSrc, float * pInitDst, uint32_t pInitRows,
                                  uint64_t pInitStride) {
  if (!vIsOpen || !vHasMoe || !vMoeKept) {
    pError = "there is no kept mixture-of-experts result to continue from";
    return false;
  }
  if (vBroken) {
    pError = "a command did not end before: nothing else is sent to the NPU";
    return false;
  }
  if (pWeights.vBuffer.vMap == nullptr || (aRowsOut == nullptr && aTargets == nullptr)) {
    pError = "the expert weights or outputs of mRunMoeFromKept are missing";
    return false;
  }
  vMoeKept = false;
  if (!mFlushMoeY(pPrevCommand, aMoeIo[0], pTimes, pError)) {
    return false;
  }

  double          vStart = fNowMs();
  XdnaMoeCommand vCmd;
  if (!fXdnaMoeCommandShape(vMoeOverlay, pWeights.vMatrix, pPrevCommand.aChunks.data(), pPrevCommand.vChunks, vCmd,
                             pError)) {
    return false;
  }
  XdnaBuffer & vIoBuf = aMoeIo[1];
  if (!mEnsure(vIoBuf, XdnaBufferType::cShare, vCmd.vBytesIo, pError)) {
    return false;
  }
  pTimes.vPlanMs += fNowMs() - vStart;

  vStart        = fNowMs();
  uint8_t * vIo = static_cast<uint8_t *>(vIoBuf.vMap);
  fXdnaMoeWriteParameters(vMoeOverlay, vCmd, vIo);
  if (!fXdnaMoeTranscodeYToX(vMoeOverlay, pPrevCommand, static_cast<const uint8_t *>(aMoeIo[0].vMap), vCmd, vIo,
                              aRowScales, &vWorkers)) {
    pError = "the intermediate activations of the experts are not finite";
    return false;
  }
  pTimes.vPackMs += fNowMs() - vStart;

  if (!mSubmitMoeCommand(vCmd, vIoBuf, pWeights, pTimes, pError)) {
    return false;
  }
  if (!mFlushMoeY(vCmd, vIoBuf, pTimes, pError)) {
    return false;
  }
  vStart = fNowMs();
  if (aTargets != nullptr) {
    fXdnaMoeReadYTargets(vMoeOverlay, vCmd, vIo, aTargets, pTargets, &vWorkers);
  } else {
    fXdnaMoeReadY(vMoeOverlay, vCmd, vIo, aRowsOut, &vWorkers, pInitSrc, pInitDst, pInitRows, pInitStride);
  }
  pTimes.vUnpackMs += fNowMs() - vStart;
  return true;
}
