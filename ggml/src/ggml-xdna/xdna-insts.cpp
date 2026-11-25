#include "xdna-insts.h"

namespace {

// operations of the transaction format of the firmware
const uint32_t cOpWrite      = 0;
const uint32_t cOpBlockWrite = 1;
const uint32_t cOpMaskWrite  = 3;
const uint32_t cOpMaskPoll   = 4;
const uint32_t cOpSync       = 128;
const uint32_t cOpPatch      = 129;

const size_t   cHeaderWords      = 4;
const uint32_t cDeviceGeneration = 4;  // the NPU of AIE2P (npu2 of mlir-aie)
const uint32_t cVersionMajor     = 0;
const uint32_t cVersionMinor     = 1;

// an address holds the column, the row and the register of the tile
const uint32_t cColumnShift = 25;
const uint32_t cRowShift    = 20;

// DMA of a shim tile of AIE2P
const uint32_t cShimDescriptors      = 16;        // buffer descriptors of a tile
const uint32_t cShimDescriptorBase   = 0x1d000;   // registers of the first one
const uint32_t cShimDescriptorStep   = 0x20;      // and bytes to the next
const uint32_t cShimDescriptorWords  = 8;
const uint32_t cShimChannels         = 2;         // per direction
const uint32_t cShimControlToMemory  = 0x1d200;   // control register of channel 0; the queue is 4 bytes after it
const uint32_t cShimControlFromMemory = 0x1d210;
const uint32_t cShimChannelStep      = 8;
const uint32_t cShimStatusToMemory   = 0x1d220;   // status register of channel 0
const uint32_t cShimStatusFromMemory = 0x1d228;
const uint32_t cShimStatusStep       = 4;
const uint32_t cQueueDepth           = 4;         // tasks a channel holds
const uint32_t cQueueFullBit         = 0x00400000;  // of the status register: 4 tasks queued
const uint32_t cQueueTokenBit        = 0x80000000;  // of a push: issue a token at the end of the task
const uint32_t cQueueRepeatShift     = 16;
const uint32_t cShimSizeShift        = 20;        // of words 3, 4 and 6 of a descriptor: sizes and iterations
const uint32_t cControllerMask       = 0x00001f00;  // of the control register: who gets the tokens
const uint32_t cControllerValue      = 0x00000f00;

// words 2 to 7 of a buffer descriptor of a transfer of consecutive bytes
const uint32_t cDescriptorBurst = 0xc0000000;  // word 4: the longest bursts
const uint32_t cDescriptorCache = 0x02000000;  // word 5
const uint32_t cDescriptorValid = 0x02000000;  // word 7: the descriptor is valid and has no successor

uint32_t fAddress(XdnaTile pTile, uint32_t pRegister) {
  return (uint32_t(pTile.vColumn) << cColumnShift) | (uint32_t(pTile.vRow) << cRowShift) | pRegister;
}

}  // namespace

XdnaInstructions::XdnaInstructions() : aWords(cHeaderWords, 0), vOperations(0) {}

void XdnaInstructions::mReserve(size_t pWords) {
  aWords.reserve(cHeaderWords + pWords);
}

void XdnaInstructions::mWrite(XdnaTile pTile, uint32_t pRegister, uint32_t pValue) {
  const uint32_t aOperation[6] = {cOpWrite, 0, fAddress(pTile, pRegister), 0, pValue, 6 * sizeof(uint32_t)};
  aWords.insert(aWords.end(), aOperation, aOperation + 6);
  vOperations++;
}

void XdnaInstructions::mMaskWrite(XdnaTile pTile, uint32_t pRegister, uint32_t pValue, uint32_t pMask) {
  const uint32_t aOperation[7] = {cOpMaskWrite, 0, fAddress(pTile, pRegister), 0, pValue, pMask, 7 * sizeof(uint32_t)};
  aWords.insert(aWords.end(), aOperation, aOperation + 7);
  vOperations++;
}

void XdnaInstructions::mMaskPoll(XdnaTile pTile, uint32_t pRegister, uint32_t pValue, uint32_t pMask) {
  const uint32_t aOperation[7] = {cOpMaskPoll, 0, fAddress(pTile, pRegister), 0, pValue, pMask, 7 * sizeof(uint32_t)};
  aWords.insert(aWords.end(), aOperation, aOperation + 7);
  vOperations++;
}

void XdnaInstructions::mBlockWrite(XdnaTile pTile, uint32_t pRegister, const uint32_t * pValues, size_t pCount) {
  // the second word could name the tile again; mlir-aie leaves it at 0 and so does this
  const uint32_t aHeader[4] = {cOpBlockWrite, 0, fAddress(pTile, pRegister), uint32_t((4 + pCount) * sizeof(uint32_t))};
  aWords.insert(aWords.end(), aHeader, aHeader + 4);
  aWords.insert(aWords.end(), pValues, pValues + pCount);
  vOperations++;
}

void XdnaInstructions::mPatchAddress(XdnaTile pTile, uint32_t pRegister, uint32_t pBuffer, uint64_t pOffset) {
  const uint32_t aOperation[12] = {
    cOpPatch, 12 * sizeof(uint32_t), 0, 0, 0, 0,
    fAddress(pTile, pRegister), 0,
    pBuffer, 0,
    uint32_t(pOffset & 0xffffffffu), uint32_t(pOffset >> 32),
  };
  aWords.insert(aWords.end(), aOperation, aOperation + 12);
  vOperations++;
}

void XdnaInstructions::mSync(XdnaTile pTile, XdnaDirection pDirection, uint32_t pChannel) {
  // one tile: 1 column and 1 row
  const uint32_t aOperation[4] = {
    cOpSync, 4 * sizeof(uint32_t),
    uint32_t(pDirection) | (uint32_t(pTile.vRow) << 8) | (uint32_t(pTile.vColumn) << 16),
    (1u << 8) | (1u << 16) | (pChannel << 24),
  };
  aWords.insert(aWords.end(), aOperation, aOperation + 4);
  vOperations++;
}

std::vector<uint32_t> XdnaInstructions::mFinish(uint32_t pColumns, uint32_t pRows, uint32_t pMemoryRows) const {
  std::vector<uint32_t> aStream = aWords;
  aStream[0] = (pRows << 24) | (cDeviceGeneration << 16) | (cVersionMinor << 8) | cVersionMajor;
  aStream[1] = (pMemoryRows << 8) | pColumns;
  aStream[2] = vOperations;
  aStream[3] = uint32_t(aStream.size() * sizeof(uint32_t));
  return aStream;
}

void XdnaInstructions::mFinishInto(uint32_t pColumns, uint32_t pRows, uint32_t pMemoryRows,
                                    std::vector<uint32_t> & aStream) {
  aStream    = std::move(aWords);
  aStream[0] = (pRows << 24) | (cDeviceGeneration << 16) | (cVersionMinor << 8) | cVersionMajor;
  aStream[1] = (pMemoryRows << 8) | pColumns;
  aStream[2] = vOperations;
  aStream[3] = uint32_t(aStream.size() * sizeof(uint32_t));
  aWords.assign(cHeaderWords, 0);
  vOperations = 0;
}

XdnaShimDma::XdnaShimDma(XdnaInstructions & pInstructions, uint32_t pColumns)
    : vInstructions(pInstructions),
      aFreeDescriptors(pColumns, uint16_t((1u << cShimDescriptors) - 1)),
      aQueues(size_t(pColumns) * 2 * cShimChannels) {}

XdnaShimDma::XdnaQueue & XdnaShimDma::mQueue(uint8_t pColumn, XdnaDirection pDirection, uint8_t pChannel) {
  return aQueues[(size_t(pColumn) * 2 + size_t(pDirection)) * cShimChannels + pChannel];
}

bool XdnaShimDma::mPush(const XdnaShimTransfer & pTransfer, XdnaShimTask & pTask, std::string & pError) {
  if (!mPrepare(pTransfer, pTask, pError)) {
    return false;
  }
  mStart(pTask);
  return true;
}

bool XdnaShimDma::mPrepare(const XdnaShimTransfer & pTransfer, XdnaShimTask & pTask, std::string & pError) {
  if (pTransfer.vColumn >= aFreeDescriptors.size() || pTransfer.vChannel >= cShimChannels) {
    pError = "a transfer names a shim channel that does not exist";
    return false;
  }
  if (pTransfer.vBytes == 0 || pTransfer.vBytes % 4 != 0 || pTransfer.vOffset % 4 != 0 ||
      pTransfer.vBytes / 4 > 0xffffffffu) {
    pError = "a transfer has to move a whole number of 32-bit words, at a 32-bit boundary of its buffer";
    return false;
  }
  if (pTransfer.vRepeat < 1 || pTransfer.vRepeat > cXdnaShimRepeatMax) {
    pError = "a channel repeats a transfer " + std::to_string(cXdnaShimRepeatMax) + " times at most";
    return false;
  }
  // a pattern: inner dimensions, iterations or both (without inner dimensions, consecutive words every iteration)
  const bool vPattern = pTransfer.vSizes[0] != 0 || pTransfer.vIterations > 1;
  if (vPattern) {
    bool vFits = pTransfer.vSizes[0] <= cXdnaShimSizeMax && pTransfer.vSizes[1] <= cXdnaShimSizeMax &&
                 (pTransfer.vSizes[0] != 0 || pTransfer.vSizes[1] == 0) &&
                 pTransfer.vIterations >= 1 && pTransfer.vIterations <= cXdnaShimIterationsMax &&
                 (pTransfer.vIterations == 1 || pTransfer.vRepeat == 1) && pTransfer.vIterationStride >= 1 &&
                 pTransfer.vIterationStride <= cXdnaShimStrideMax;
    for (uint32_t vStride : pTransfer.vStrides) {
      vFits = vFits && vStride >= 1 && vStride <= cXdnaShimStrideMax;
    }
    if (!vFits) {
      pError = "a pattern of a transfer does not fit in a buffer descriptor of a shim tile";
      return false;
    }
  }

  // the lowest free descriptor of the tile
  uint16_t & vFree = aFreeDescriptors[pTransfer.vColumn];
  uint32_t vDescriptor = 0;
  while (vDescriptor < cShimDescriptors && !(vFree & (1u << vDescriptor))) {
    vDescriptor++;
  }
  if (vDescriptor == cShimDescriptors) {
    pError = "the shim tile of column " + std::to_string(pTransfer.vColumn) + " has no buffer descriptor left";
    return false;
  }
  vFree = uint16_t(vFree & ~(1u << vDescriptor));

  const XdnaTile vTile = {pTransfer.vColumn, 0};
  const uint32_t vDescriptorAt = cShimDescriptorBase + vDescriptor * cShimDescriptorStep;
  // The second word is the address: the firmware puts there the address of the buffer plus the offset. The offset
  // is written too, as mlir-aie does. A pattern goes in words 3 to 6 (sizes, and strides minus one), as mlir-aie
  // writes it; without one they are those of consecutive words.
  uint32_t aDescriptor[cShimDescriptorWords] = {
    uint32_t(pTransfer.vBytes / 4), uint32_t(pTransfer.vOffset & 0xffffffffu), 0, 0,
    cDescriptorBurst, cDescriptorCache, 0, cDescriptorValid,
  };
  if (vPattern) {
    aDescriptor[3] = (pTransfer.vSizes[0] << cShimSizeShift) | (pTransfer.vStrides[0] - 1);
    aDescriptor[4] = cDescriptorBurst | (pTransfer.vSizes[1] << cShimSizeShift) | (pTransfer.vStrides[1] - 1);
    aDescriptor[5] = cDescriptorCache | (pTransfer.vStrides[2] - 1);
    aDescriptor[6] = ((pTransfer.vIterations - 1) << cShimSizeShift) | (pTransfer.vIterationStride - 1);
  }
  vInstructions.mBlockWrite(vTile, vDescriptorAt, aDescriptor, cShimDescriptorWords);
  vInstructions.mPatchAddress(vTile, vDescriptorAt + 4, pTransfer.vBuffer, pTransfer.vOffset);

  pTask.vColumn     = pTransfer.vColumn;
  pTask.vDirection  = pTransfer.vDirection;
  pTask.vChannel    = pTransfer.vChannel;
  pTask.vDescriptor = uint8_t(vDescriptor);
  pTask.vToken      = pTransfer.vToken;
  pTask.vRuns       = vPattern && pTransfer.vIterations > 1 ? pTransfer.vIterations : pTransfer.vRepeat;
  return true;
}

void XdnaShimDma::mStart(const XdnaShimTask & pTask) {
  const XdnaTile vTile     = {pTask.vColumn, 0};
  const bool      vToMemory = pTask.vDirection == XdnaDirection::cToMemory;
  const uint32_t  vControl  = (vToMemory ? cShimControlToMemory : cShimControlFromMemory) + pTask.vChannel * cShimChannelStep;
  const uint32_t  vStatus   = (vToMemory ? cShimStatusToMemory : cShimStatusFromMemory) + pTask.vChannel * cShimStatusStep;

  // A push onto a full queue is lost. When 4 pushes of the channel are not known to have ended, wait for room.
  XdnaQueue & vQueue = mQueue(pTask.vColumn, pTask.vDirection, pTask.vChannel);
  if (vQueue.aPushes.size() >= cQueueDepth) {
    vInstructions.mMaskPoll(vTile, vStatus, 0, cQueueFullBit);
    // room for one: the oldest pushes are gone, but not which of them issued tokens
    const size_t vGone = vQueue.aPushes.size() - (cQueueDepth - 1);
    for (size_t i = 0; i < vGone; i++) {
      vQueue.vUnknownTokens = vQueue.vUnknownTokens || vQueue.aPushes[i];
    }
    vQueue.aPushes.erase(vQueue.aPushes.begin(), vQueue.aPushes.begin() + vGone);
    if (vQueue.vUnknownTokens) {
      vQueue.aPushes.assign(vQueue.aPushes.size(), false);
    }
  }
  vQueue.aPushes.push_back(pTask.vToken && !vQueue.vUnknownTokens);

  if (pTask.vToken) {
    // the tokens of the channel go to the firmware
    vInstructions.mMaskWrite(vTile, vControl, cControllerValue, cControllerMask);
  }
  uint32_t vPush = pTask.vDescriptor | ((pTask.vRuns - 1) << cQueueRepeatShift);
  if (pTask.vToken) {
    vPush |= cQueueTokenBit;
  }
  vInstructions.mWrite(vTile, vControl + 4, vPush);
}

void XdnaShimDma::mSettle(uint8_t pColumn, XdnaDirection pDirection, uint8_t pChannel) {
  XdnaQueue & vQueue = mQueue(pColumn, pDirection, pChannel);
  vQueue.aPushes.clear();
  vQueue.vUnknownTokens = false;
}

void XdnaShimDma::mAwait(const XdnaShimTask & pTask) {
  vInstructions.mSync({pTask.vColumn, 0}, pTask.vDirection, pTask.vChannel);
  // The channel gives its tokens in order: this one is of its oldest push with a token, and what was queued before
  // it has ended too.
  XdnaQueue & vQueue = mQueue(pTask.vColumn, pTask.vDirection, pTask.vChannel);
  if (vQueue.vUnknownTokens) {
    return;
  }
  for (size_t i = 0; i < vQueue.aPushes.size(); i++) {
    if (vQueue.aPushes[i]) {
      vQueue.aPushes.erase(vQueue.aPushes.begin(), vQueue.aPushes.begin() + i + 1);
      break;
    }
  }
}

void XdnaShimDma::mFree(const XdnaShimTask & pTask) {
  aFreeDescriptors[pTask.vColumn] = uint16_t(aFreeDescriptors[pTask.vColumn] | (1u << pTask.vDescriptor));
}

uint32_t XdnaShimDma::mFreeDescriptors(uint8_t pColumn) const {
  return pColumn < aFreeDescriptors.size() ? uint32_t(__builtin_popcount(aFreeDescriptors[pColumn])) : 0;
}
