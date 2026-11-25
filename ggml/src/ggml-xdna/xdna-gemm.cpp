#include "xdna-gemm.h"

#include "xdna-gdnp.h"
#include "xdna-glu.h"
#include "xdna-insts.h"
#include "xdna-workers.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <string>
#include <type_traits>
#include <vector>

#if defined(__x86_64__)
#include <immintrin.h>
#endif

namespace {

const uint32_t cBlock      = 8;  // values of A and of B that share an exponent
const uint32_t cBlockBytes = 9;  // and the bytes they take
const uint32_t cPair       = 16;  // rows, columns and values of K the kernel takes at a time
const uint32_t cInFlight   = 4;  // row blocks whose transfers are queued at once: what the queue of a channel holds

// buffers of the command
const uint32_t cBufferA = 0;
const uint32_t cBufferB = 1;
const uint32_t cBufferC = 2;

// the value of the lock number 0 of a core tile, and bytes to the next
const uint32_t cCoreLockBase = 0x1f000;
const uint32_t cCoreLockStep = 0x10;

// DMA and locks of a memory tile of AIE2P (read in the register database of mlir-aie)
const uint32_t cMemDescriptorBase    = 0xa0000;  // registers of the first buffer descriptor
const uint32_t cMemDescriptorStep    = 0x20;     // and bytes to the next
const uint32_t cMemDescriptorLocks   = 0x1c;     // its last word: the locks it takes and gives
const uint32_t cMemControlToMemory   = 0xa0600;  // control register of channel 0; the queue is 4 bytes after it
const uint32_t cMemControlFromMemory = 0xa0630;
const uint32_t cMemChannelStep       = 8;
const uint32_t cMemChannelReset      = 0x2;      // of a control register
const uint32_t cMemLockBase          = 0xc0000;  // value of the lock number 0
const uint32_t cMemLockStep          = 0x10;
const uint32_t cMemOwnLocks          = 64;       // a descriptor numbers the locks of its own tile from here
const uint32_t cMemLengthMax         = 0x1ffff;  // words of a descriptor
const uint32_t cMemSendsMax          = 63;       // a lock counts up to here
const uint32_t cDescriptorValid      = 0x80000000;
const uint32_t cDescriptorAcquire    = 0x00008000;
const uint32_t cQueueRepeatShift     = 16;

// Sets, in the memory tile that keeps the band of A of a stream, how long the band is and how many times it goes
// to the cores before the next one may come in. The two channels of the band wait with what the overlay, or the
// multiplication before, left in them, so they are stopped, set and started again, in the order mlir-aie uses
// to re-arm a fifo that lives in a memory tile.
void fArmBand(XdnaInstructions & pInstructions, const XdnaOverlayBand & pBand, uint32_t pWords, uint32_t pSends) {
  const XdnaTile vTile = {pBand.vTileColumn, pBand.vTileRow};
  const uint32_t  vControlIn     = cMemControlToMemory + pBand.vChannelIn * cMemChannelStep;
  const uint32_t  vControlOut    = cMemControlFromMemory + pBand.vChannelOut * cMemChannelStep;
  const uint32_t  vDescriptorIn  = cMemDescriptorBase + pBand.vDescriptorIn * cMemDescriptorStep;
  const uint32_t  vDescriptorOut = cMemDescriptorBase + pBand.vDescriptorOut * cMemDescriptorStep;

  // both channels stop and forget what they were waiting for
  pInstructions.mMaskWrite(vTile, vControlIn, cMemChannelReset, cMemChannelReset);
  pInstructions.mMaskWrite(vTile, vControlIn, 0, cMemChannelReset);
  pInstructions.mMaskWrite(vTile, vControlOut, cMemChannelReset, cMemChannelReset);
  pInstructions.mMaskWrite(vTile, vControlOut, 0, cMemChannelReset);

  // The band is pWords long, coming in and going out. The descriptor that receives it takes pSends from the lock
  // of free sends and gives pSends to the lock of owed sends; the one that sends it takes one and gives one back,
  // which the overlay already says.
  pInstructions.mWrite(vTile, vDescriptorIn, pWords);
  pInstructions.mWrite(vTile, vDescriptorIn + cMemDescriptorLocks,
                       cDescriptorValid | (pSends << 24) | ((cMemOwnLocks + pBand.vLockFull) << 16) |
                           cDescriptorAcquire | (((0u - pSends) & 0x7f) << 8) | (cMemOwnLocks + pBand.vLockFree));
  pInstructions.mWrite(vTile, vDescriptorOut, pWords);

  // no send owed and all of them free: a band may come in
  pInstructions.mWrite(vTile, cMemLockBase + pBand.vLockFree * cMemLockStep, pSends);
  pInstructions.mWrite(vTile, cMemLockBase + pBand.vLockFull * cMemLockStep, 0);

  // a channel of AIE2P starts when a task is queued
  pInstructions.mWrite(vTile, vControlIn + 4, pBand.vDescriptorIn);
  pInstructions.mWrite(vTile, vControlOut + 4, pBand.vDescriptorOut | ((pSends - 1) << cQueueRepeatShift));
}

// bytes of pSums sums of C, a multiple of 8, as the overlay gives them back
uint64_t fOutputBytes(const XdnaOverlay & pOverlay, uint64_t pSums) {
  switch (pOverlay.vOutput) {
    case XdnaGemmOutput::cBfloat16:
      return pSums * 2;
    case XdnaGemmOutput::cBfp16:
      return pSums / cBlock * cBlockBytes;
    default:
      return pSums * 4;
  }
}

}  // namespace

// Tiles of K every tile of C takes outside of the loop of a core: the first one, which starts the sums, and with C
// in bfloat16 also the last one, which writes them rounded.
uint32_t fXdnaGemmTilesKOutside(const XdnaOverlay & pOverlay) {
  return pOverlay.vOutput == XdnaGemmOutput::cFloat ? 1 : 2;
}

bool fXdnaGemmPlan(const XdnaOverlay & pOverlay, uint32_t pM, uint32_t pK, uint32_t pN, XdnaGemmPlan & pPlan,
                    std::string & pError, XdnaPromptFormat pFormat) {
  const uint32_t vTileM = pOverlay.vTileM;
  const uint32_t vTileK = pOverlay.vTileK;
  const uint32_t vTileN = pOverlay.vTileN;
  const bool     vByColumns = pOverlay.vLayout == XdnaGemmLayout::cColumns;
  // A goes to one side of the grid of cores and B to the other; C comes back by columns of cores
  const uint32_t vStreamsA = vByColumns ? pOverlay.vCoreColumns : pOverlay.vCoreRows;
  const uint32_t vStreamsB = vByColumns ? pOverlay.vCoreRows : pOverlay.vCoreColumns;
  const uint32_t vStreamsC = pOverlay.vCoreColumns;
  // Split, every tile of B comes in pieces, each through a stream of its own (the pieces of the side s of the grid
  // go through the streams s * vSplit ...), and the last row of cores sends its tiles of C straight to the host.
  const uint32_t vSplit      = pOverlay.vSplit;
  const bool     vDirect     = vSplit == 2;
  const uint32_t vRowsJoined = pOverlay.vCoreRows - (vDirect ? 1 : 0);  // rows of cores whose C a memory tile joins

  if (pOverlay.aStreamsA.size() != vStreamsA || pOverlay.aStreamsB.size() != size_t(vStreamsB) * vSplit ||
      pOverlay.aStreamsC.size() != vStreamsC || pOverlay.aStreamsD.size() != (vDirect ? vStreamsC : 0) ||
      (vSplit != 1 && (!vDirect || !vByColumns || vRowsJoined == 0))) {
    pError = "the streams of the overlay do not match its layout";
    return false;
  }
  if (vTileM % cPair != 0 || vTileK % cPair != 0 || vTileN % cPair != 0) {
    pError = "the tiles of the overlay are not multiples of " + std::to_string(cPair);
    return false;
  }
  if (pM == 0 || pK == 0 || pN == 0 || pM % (vTileM * vStreamsA) != 0 || pK % vTileK != 0 ||
      pN % (vTileN * vStreamsB) != 0) {
    pError = "this overlay multiplies M in multiples of " + std::to_string(vTileM * vStreamsA) + ", K of " +
             std::to_string(vTileK) + " and N of " + std::to_string(vTileN * vStreamsB);
    return false;
  }
  const uint32_t vOutside = fXdnaGemmTilesKOutside(pOverlay);
  if (pK / vTileK < vOutside) {
    pError = "this overlay multiplies K of " + std::to_string(vOutside) + " tiles of " + std::to_string(vTileK) +
             " at least";
    return false;
  }
  if ((pOverlay.vFixedK != 0 && pOverlay.vFixedK != pK) || (pOverlay.vFixedN != 0 && pOverlay.vFixedN != pN)) {
    pError = "this overlay is built for K = " + std::to_string(pOverlay.vFixedK) + " and N = " +
             std::to_string(pOverlay.vFixedN);
    return false;
  }

  const uint64_t vBlocks   = pM / (vTileM * vStreamsA);  // row blocks: each one is a pass of the whole B
  const uint64_t vTilesK   = pK / vTileK;                // tiles a core adds up into one tile of C
  const uint64_t vTilesOut = pN / (vTileN * vStreamsB);  // tiles of C a core produces in one row block
  if (pFormat != XdnaPromptFormat::cBfp16) {
    if (pOverlay.vCompB != 1 || pOverlay.vRowsC != 1 || vSplit != 1 || !vByColumns) {
      pError = "this overlay does not decode compressed prompt weights (comp 1)";
      return false;
    }
    if (pFormat == XdnaPromptFormat::cQ6K && (vTilesK < 4 || vTilesK % 4 != 0)) {
      pError = "Q6_K prompt weights require K to be a multiple of 512";
      return false;
    }
  }
  const uint64_t vObjectsK   = pFormat == XdnaPromptFormat::cQ6K ? (vTilesK / 4 * 3) : vTilesK;
  const uint64_t vTileBytesA = uint64_t(vTileM) * vTileK / cBlock * cBlockBytes;
  const uint64_t vTileBytesB = uint64_t(vTileK) * vTileN / cBlock * cBlockBytes;
  const uint64_t vTileBytesC = fOutputBytes(pOverlay, uint64_t(vTileM) * vTileN);
  const uint64_t vBandBytes   = vTilesK * vTileBytesA;                           // a band of A: vTileM rows
  const uint64_t vPartBytesB  = vTilesOut * vObjectsK * vTileBytesB / vSplit;    // what a stream of B carries of B
  // what a column gives in a row block: through its memory tile, and, split, from its last core
  const uint64_t vColumnBytes = vTilesOut * vRowsJoined * vTileBytesC;
  const uint64_t vDirectBytes = vTilesOut * vTileBytesC;
  // where the stretches of the last cores begin in C: after all the ones of the memory tiles
  const uint64_t vDirectStart = uint64_t(vStreamsC) * vBlocks * vColumnBytes;

  if (vTileBytesB % (uint64_t(vSplit) * 4) != 0) {
    pError = "the pieces of a tile of B are not whole 32-bit words";
    return false;
  }
  if (vTilesK - vOutside > 0x7fffffffu || vBlocks * vTilesOut > 0x7fffffffu) {
    pError = "the multiplication is too large for the counters of the cores";
    return false;
  }
  // llama.cpp-xdna: with strips of C, the turns of a row block are the iterations of the descriptor of its drain,
  // and 4 strips the step from one turn to the next
  if (pOverlay.vRowsC == 1 &&
      (!vByColumns || vDirect || vTilesOut > cXdnaShimIterationsMax ||
       vStreamsB * fOutputBytes(pOverlay, uint64_t(pM) * vTileN) / 4 > cXdnaShimStrideMax)) {
    pError = "strips of C take the layout columns, no split, N up to " +
             std::to_string(cXdnaShimIterationsMax * vTileN * vStreamsB) + " and M up to " +
             std::to_string(uint64_t(cXdnaShimStrideMax) * 4 / (vStreamsB * fOutputBytes(pOverlay, vTileN)));
    return false;
  }
  // with the bands of A kept in memory tiles that the host can set, the band has to fit and the sends to count
  const bool vArmBands = vByColumns && !pOverlay.aBands.empty();
  if (vArmBands) {
    if (pOverlay.aBands.size() != vStreamsA) {
      pError = "the bands of the overlay do not match its layout";
      return false;
    }
    if (vBandBytes / 4 > pOverlay.vBandWords || vBandBytes / 4 > cMemLengthMax) {
      pError = "a band of A for K = " + std::to_string(pK) + " does not fit in the memory tiles of this overlay";
      return false;
    }
    if (vTilesOut > cMemSendsMax) {
      pError = "this overlay multiplies N up to " + std::to_string(cMemSendsMax * vTileN * vStreamsB);
      return false;
    }
  }

  pPlan.vBytesA = uint64_t(pM) * pK / cBlock * cBlockBytes;
  pPlan.vBytesB = uint64_t(vStreamsB) * vSplit * vPartBytesB;
  pPlan.vBytesC = fOutputBytes(pOverlay, uint64_t(pM) * pN);

  XdnaInstructions vInstructions;

  if (vArmBands) {
    for (const XdnaOverlayBand & vBand : pOverlay.aBands) {
      fArmBand(vInstructions, vBand, uint32_t(vBandBytes / 4), uint32_t(vTilesOut));
    }
  }

  // Every core gets its two counters and then the lock that lets it start: the tiles of K of every tile of C that
  // go through its loop, and the tiles of C.
  for (uint32_t vRow = 0; vRow < pOverlay.vCoreRows; vRow++) {
    for (uint32_t vColumn = 0; vColumn < pOverlay.vCoreColumns; vColumn++) {
      const XdnaOverlayCore & vCore = pOverlay.aCores[size_t(vRow) * pOverlay.vCoreColumns + vColumn];
      const XdnaTile          vTile = {vCore.vTileColumn, vCore.vTileRow};
      if (pFormat == XdnaPromptFormat::cQ6K) {
        vInstructions.mWrite(vTile, vCore.vParameters, 0);
        vInstructions.mWrite(vTile, vCore.vParameters + 4, 0);
        vInstructions.mWrite(vTile, vCore.vParameters + 16, uint32_t(vBlocks * vTilesOut));
        vInstructions.mWrite(vTile, vCore.vParameters + 20, uint32_t(vTilesK / 4));
      } else if (pFormat == XdnaPromptFormat::cQ8_0) {
        vInstructions.mWrite(vTile, vCore.vParameters, 0);
        vInstructions.mWrite(vTile, vCore.vParameters + 4, 0);
        vInstructions.mWrite(vTile, vCore.vParameters + 20, uint32_t(vTilesK));
        vInstructions.mWrite(vTile, vCore.vParameters + 24, uint32_t(vBlocks * vTilesOut));
      } else {
        vInstructions.mWrite(vTile, vCore.vParameters, uint32_t(vTilesK - vOutside));
        vInstructions.mWrite(vTile, vCore.vParameters + 4, uint32_t(vBlocks * vTilesOut));
      }
      vInstructions.mWrite(vTile, cCoreLockBase + vCore.vLock * cCoreLockStep, 1);
    }
  }

  XdnaShimDma vDma(vInstructions, pOverlay.vColumns);

  // The transfers of a row block are queued together, and those of cInFlight row blocks are in the queues at
  // once: the ones of a row block are waited for once the next cInFlight - 1 have been queued.
  std::deque<std::vector<XdnaShimTask>> aInFlight;
  auto fFinish = [&](const std::vector<XdnaShimTask> & aTasks) {
    for (const XdnaShimTask & vTask : aTasks) {
      if (vTask.vToken) {
        vDma.mAwait(vTask);
      }
    }
    for (const XdnaShimTask & vTask : aTasks) {
      vDma.mFree(vTask);
    }
  };

  for (uint64_t vBlock = 0; vBlock < vBlocks; vBlock++) {
    std::vector<XdnaShimTask> aTasks;
    auto fPush = [&](const XdnaOverlayStream & pStream, XdnaDirection pDirection, uint32_t pBuffer,
                     uint64_t pOffset, uint64_t pBytes, uint32_t pRepeat, bool pToken) {
      XdnaShimTransfer vTransfer;
      vTransfer.vColumn    = pStream.vColumn;
      vTransfer.vDirection = pDirection;
      vTransfer.vChannel   = pStream.vChannel;
      vTransfer.vBuffer    = pBuffer;
      vTransfer.vOffset    = pOffset;
      vTransfer.vBytes     = pBytes;
      vTransfer.vRepeat    = pRepeat;
      vTransfer.vToken     = pToken;
      XdnaShimTask vTask;
      if (!vDma.mPush(vTransfer, vTask, pError)) {
        return false;
      }
      aTasks.push_back(vTask);
      return true;
    };
    // what a column of cores gives back in this row block: its part of C, which is waited for
    auto fPushC = [&](uint32_t pColumn) {
      if (pOverlay.vRowsC == 1) {
        // llama.cpp-xdna: as strips of C, [N / vTileN][M][vTileN]. A turn is the tiles of the 4 rows of cores one
        // after the other, every tile as its rows, so every tile is its 64 rows of a strip, 8 KB in a row: the rows
        // of a tile go one after the other, every tile to the strip of the row of cores before plus one, and every
        // turn 4 strips on. (Written as the rows of C, in runs of one row of a tile, 128 bytes, the shims were 49 %
        // slower than one run a column, with 512 x 2560 x 9216.)
        const uint64_t    vRowBytes   = fOutputBytes(pOverlay, vTileN);
        const uint64_t    vStripBytes = fOutputBytes(pOverlay, uint64_t(pM) * vTileN);
        XdnaShimTransfer vTransfer;
        vTransfer.vColumn          = pOverlay.aStreamsC[pColumn].vColumn;
        vTransfer.vDirection       = XdnaDirection::cToMemory;
        vTransfer.vChannel         = pOverlay.aStreamsC[pColumn].vChannel;
        vTransfer.vBuffer          = cBufferC;
        vTransfer.vOffset          = (vBlock * vStreamsA + pColumn) * vTileM * vRowBytes;
        vTransfer.vBytes           = vRowsJoined * vTileBytesC;
        vTransfer.vToken           = true;
        vTransfer.vSizes[0]        = uint32_t(vRowBytes / 4);
        vTransfer.vSizes[1]        = vTileM;
        vTransfer.vStrides[0]      = 1;
        vTransfer.vStrides[1]      = uint32_t(vRowBytes / 4);
        vTransfer.vStrides[2]      = uint32_t(vStripBytes / 4);
        vTransfer.vIterations      = uint32_t(vTilesOut);
        vTransfer.vIterationStride = uint32_t(vStreamsB * vStripBytes / 4);
        XdnaShimTask vTask;
        if (!vDma.mPush(vTransfer, vTask, pError)) {
          return false;
        }
        aTasks.push_back(vTask);
        return true;
      }
      return fPush(pOverlay.aStreamsC[pColumn], XdnaDirection::cToMemory, cBufferC,
                   (uint64_t(pColumn) * vBlocks + vBlock) * vColumnBytes, vColumnBytes, 1, true);
    };
    // split, what its last core gives back straight
    auto fPushD = [&](uint32_t pColumn) {
      return fPush(pOverlay.aStreamsD[pColumn], XdnaDirection::cToMemory, cBufferC,
                   vDirectStart + (uint64_t(pColumn) * vBlocks + vBlock) * vDirectBytes, vDirectBytes, 1, true);
    };
    // the band of A of a stream in this row block
    auto fPushA = [&](uint32_t pStream, uint32_t pRepeat) {
      return fPush(pOverlay.aStreamsA[pStream], XdnaDirection::cFromMemory, cBufferA,
                   (vBlock * vStreamsA + pStream) * vBandBytes, vBandBytes, pRepeat, false);
    };
    // the part of B of a stream (split, of a piece of every tile), whole, for every row block
    auto fPushB = [&](uint32_t pStream) {
      return fPush(pOverlay.aStreamsB[pStream], XdnaDirection::cFromMemory, cBufferB,
                   uint64_t(pStream) * vPartBytesB, vPartBytesB, 1, false);
    };

    bool vOk = true;
    if (vByColumns) {
      // the memory tile of every column keeps its band and repeats it: the host sends it once
      for (uint32_t vColumn = 0; vOk && vColumn < vStreamsC; vColumn++) {
        vOk = fPushC(vColumn) && (!vDirect || fPushD(vColumn)) && fPushA(vColumn, 1);
      }
      for (uint32_t vStream = 0; vOk && vStream < vStreamsB * vSplit; vStream++) {
        vOk = fPushB(vStream);
      }
    } else {
      // the host sends the band of every row of cores once for every tile of C the cores produce with it
      if (vTilesOut > 0xffffffffu) {
        pError = "the multiplication is too large";
        return false;
      }
      for (uint32_t vColumn = 0; vOk && vColumn < vStreamsC; vColumn++) {
        vOk = fPushC(vColumn);
        if (vOk && vColumn < vStreamsA) {
          vOk = fPushA(vColumn, uint32_t(vTilesOut));
        }
        vOk = vOk && fPushB(vColumn);
      }
    }
    if (!vOk) {
      return false;
    }

    aInFlight.push_back(aTasks);
    if (aInFlight.size() == cInFlight) {
      fFinish(aInFlight.front());
      aInFlight.pop_front();
    }
  }
  while (!aInFlight.empty()) {
    fFinish(aInFlight.front());
    aInFlight.pop_front();
  }
  if (pFormat != XdnaPromptFormat::cBfp16) {
    for (uint32_t vRow = 0; vRow < pOverlay.vCoreRows; vRow++) {
      for (uint32_t vColumn = 0; vColumn < pOverlay.vCoreColumns; vColumn++) {
        const XdnaOverlayCore & vCore = pOverlay.aCores[size_t(vRow) * pOverlay.vCoreColumns + vColumn];
        const XdnaTile          vTile = {vCore.vTileColumn, vCore.vTileRow};
        if (pFormat == XdnaPromptFormat::cQ6K) {
          vInstructions.mWrite(vTile, vCore.vParameters + 16, 0);
          vInstructions.mWrite(vTile, vCore.vParameters + 20, 0);
        } else {
          vInstructions.mWrite(vTile, vCore.vParameters + 20, 0);
          vInstructions.mWrite(vTile, vCore.vParameters + 24, 0);
        }
      }
    }
  }

  pPlan.aInstructions = vInstructions.mFinish(pOverlay.vColumns, pOverlay.vRows, pOverlay.vMemoryRows);
  return true;
}

// llama.cpp-xdna: see xdna-gemm.h
bool fXdnaGdnPlan(const XdnaOverlay & pOverlay, uint32_t pRounds, uint32_t pGroups, XdnaGemmPlan & pPlan,
                   std::string & pError) {
  const uint32_t vRows    = 4;
  const uint32_t vColumns = 8;
  if (pOverlay.vGdn != 1 || pOverlay.vLayout != XdnaGemmLayout::cColumns || pOverlay.vSplit != 1 ||
      pOverlay.vCoreRows != vRows || pOverlay.vCoreColumns != vColumns || pOverlay.vTileM != 64 ||
      pOverlay.vTileK != 128 || pOverlay.vTileN != 64 || pOverlay.vOutput != XdnaGemmOutput::cBfloat16 ||
      pOverlay.aStreamsA.size() != vColumns || pOverlay.aStreamsB.size() != vRows ||
      pOverlay.aStreamsC.size() != vColumns || !pOverlay.aStreamsD.empty()) {
    pError = "this overlay does not run the recurrence (bfpmm_resident.py --gdn 1, tiles of 64 x 128 x 64, C in bf16)";
    return false;
  }
  if (pRounds == 0 || pGroups == 0 || uint64_t(pRounds) * pGroups > 0x7fffffffu) {
    pError = "the recurrence needs rounds and groups";
    return false;
  }
  // with bands the host can set, a band of 6 objects sent once; without them, the overlay has to be built for it
  // (-K 768 -N 256, which the tests of bfpmm_resident.py --test gdn use)
  const uint32_t vBandWords = cXdnaGdnBand * cXdnaGdnTile / 4;
  const bool     vArmBands  = !pOverlay.aBands.empty();
  if (vArmBands) {
    if (pOverlay.aBands.size() != vColumns || vBandWords > pOverlay.vBandWords) {
      pError = "the bands of the overlay do not take the ones of the recurrence";
      return false;
    }
  } else if (pOverlay.vFixedK != cXdnaGdnBand * pOverlay.vTileK || pOverlay.vFixedN != vRows * pOverlay.vTileN) {
    pError = "this overlay sets neither its bands nor the ones of the recurrence";
    return false;
  }

  const uint64_t vChunks      = uint64_t(pGroups) * cXdnaGdnChunksOut;
  const uint64_t vHeadBytes   = vChunks * 2 * cXdnaGdnTile;                       // the stream of a head of keys
  const uint64_t vColumnBytesA = uint64_t(pRounds) * cXdnaGdnBand * cXdnaGdnTile;  // the bands of a column
  const uint64_t vColumnBytesC = uint64_t(pRounds) * (pGroups + 1) * vRows * cXdnaGdnObjectC;
  pPlan.vBytesA = vColumns * vColumnBytesA;
  pPlan.vBytesB = 2 * uint64_t(pRounds) * vHeadBytes;
  pPlan.vBytesC = vColumns * vColumnBytesC;

  XdnaInstructions vInstructions;
  if (vArmBands) {
    for (const XdnaOverlayBand & vBand : pOverlay.aBands) {
      fArmBand(vInstructions, vBand, vBandWords, 1);
    }
  }
  // every core: no tile of the multiplication, the rounds and the groups, and the lock that lets it start
  for (uint32_t vRow = 0; vRow < vRows; vRow++) {
    for (uint32_t vColumn = 0; vColumn < vColumns; vColumn++) {
      const XdnaOverlayCore & vCore = pOverlay.aCores[size_t(vRow) * vColumns + vColumn];
      const XdnaTile          vTile = {vCore.vTileColumn, vCore.vTileRow};
      vInstructions.mWrite(vTile, vCore.vParameters, 0);
      vInstructions.mWrite(vTile, vCore.vParameters + 4, 0);
      vInstructions.mWrite(vTile, vCore.vParameters + 8, pRounds);
      vInstructions.mWrite(vTile, vCore.vParameters + 12, pGroups);
      vInstructions.mWrite(vTile, cCoreLockBase + vCore.vLock * cCoreLockStep, 1);
    }
  }

  XdnaShimDma               vDma(vInstructions, pOverlay.vColumns);
  std::vector<XdnaShimTask> aTasks;
  auto fPush = [&](const XdnaOverlayStream & pStream, XdnaDirection pDirection, uint32_t pBuffer, uint64_t pOffset,
                   uint64_t pBytes, uint32_t pIterations, uint64_t pIterationBytes, bool pToken) {
    XdnaShimTransfer vTransfer;
    vTransfer.vColumn    = pStream.vColumn;
    vTransfer.vDirection = pDirection;
    vTransfer.vChannel   = pStream.vChannel;
    vTransfer.vBuffer    = pBuffer;
    vTransfer.vOffset    = pOffset;
    vTransfer.vBytes     = pBytes;
    vTransfer.vToken     = pToken;
    if (pIterations > 1) {
      vTransfer.vIterations      = pIterations;
      vTransfer.vIterationStride = uint32_t(pIterationBytes / 4);
    }
    XdnaShimTask vTask;
    if (!vDma.mPush(vTransfer, vTask, pError)) {
      return false;
    }
    aTasks.push_back(vTask);
    return true;
  };
  // what every column gives back, then its bands, then the chunks of every row: those of the head of keys r / 2 of
  // every round, which both rows of a head of keys read
  for (uint32_t vColumn = 0; vColumn < vColumns; vColumn++) {
    if (!fPush(pOverlay.aStreamsC[vColumn], XdnaDirection::cToMemory, cBufferC, vColumn * vColumnBytesC,
               vColumnBytesC, 1, 0, true)) {
      return false;
    }
  }
  for (uint32_t vColumn = 0; vColumn < vColumns; vColumn++) {
    if (!fPush(pOverlay.aStreamsA[vColumn], XdnaDirection::cFromMemory, cBufferA, vColumn * vColumnBytesA,
               vColumnBytesA, 1, 0, false)) {
      return false;
    }
  }
  for (uint32_t vRow = 0; vRow < vRows; vRow++) {
    if (!fPush(pOverlay.aStreamsB[vRow], XdnaDirection::cFromMemory, cBufferB, (vRow / 2) * vHeadBytes, vHeadBytes,
               pRounds, 2 * vHeadBytes, false)) {
      return false;
    }
  }
  for (const XdnaShimTask & vTask : aTasks) {
    if (vTask.vToken) {
      vDma.mAwait(vTask);
    }
  }
  for (const XdnaShimTask & vTask : aTasks) {
    vDma.mFree(vTask);
  }
  // the command has ended: the rounds and the groups back to 0
  for (uint32_t vRow = 0; vRow < vRows; vRow++) {
    for (uint32_t vColumn = 0; vColumn < vColumns; vColumn++) {
      const XdnaOverlayCore & vCore = pOverlay.aCores[size_t(vRow) * vColumns + vColumn];
      const XdnaTile          vTile = {vCore.vTileColumn, vCore.vTileRow};
      vInstructions.mWrite(vTile, vCore.vParameters + 8, 0);
      vInstructions.mWrite(vTile, vCore.vParameters + 12, 0);
    }
  }
  pPlan.aInstructions = vInstructions.mFinish(pOverlay.vColumns, pOverlay.vRows, pOverlay.vMemoryRows);
  return true;
}

namespace {

// A goes to one side of the grid of cores and B to the other
uint32_t fStreamsA(const XdnaOverlay & pOverlay) {
  return pOverlay.vLayout == XdnaGemmLayout::cColumns ? pOverlay.vCoreColumns : pOverlay.vCoreRows;
}

uint32_t fStreamsB(const XdnaOverlay & pOverlay) {
  return pOverlay.vLayout == XdnaGemmLayout::cColumns ? pOverlay.vCoreRows : pOverlay.vCoreColumns;
}

// The order of the groups of 8 x 8 values in a tile of A or of B transposed, as the kernel of the overlay reads them
// (see its source). A group is 8 rows (of A, or of B transposed: outputs) by 8 values along K.
enum class XdnaTileOrder {
  cPairs,  // xdna2_mm_bfp16: [rows / 16][K / 16][2 groups of rows][2 groups along K]
  cRows,   // xdna2_mm_bfp16_s3, A: [rows / 8][K / 8]
  cSets,   // xdna2_mm_bfp16_s3, B: [sets of 3 groups of rows, the last one of 2][K / 8][groups of the set]
};

// whether the kernel of the overlay works in steps of three sets of sums (kernels/xdna2-mm-bfp16-s3.cc)
bool fSteps3(const XdnaOverlay & pOverlay) {
  return pOverlay.vKernel == "xdna2_mm_bfp16_s3";
}

// Calls pBlock(group of rows, group of columns) for every 8 x 8 of a tile of C, in the order the kernel of the
// overlay writes them: by pairs of groups (xdna2_mm_bfp16), or by groups of rows and sets of 3 groups of columns,
// with the last set of 2 after all of them (xdna2_mm_bfp16_s3).
template <typename F>
void fForEachBlock(bool pSteps3, uint32_t pRowGroups, uint32_t pColumnGroups, F && pBlock) {
  if (pSteps3) {
    const uint32_t vSets = pColumnGroups / 3;
    for (uint32_t vRowGroup = 0; vRowGroup < pRowGroups; vRowGroup++) {
      for (uint32_t vSet = 0; vSet < vSets; vSet++) {
        for (uint32_t j = 0; j < 3; j++) {
          pBlock(vRowGroup, vSet * 3 + j);
        }
      }
    }
    for (uint32_t vRowGroup = 0; vRowGroup < pRowGroups; vRowGroup++) {
      for (uint32_t j = 0; j < pColumnGroups % 3; j++) {
        pBlock(vRowGroup, vSets * 3 + j);
      }
    }
    return;
  }
  for (uint32_t vRowPair = 0; vRowPair < pRowGroups / 2; vRowPair++) {
    for (uint32_t vColumnPair = 0; vColumnPair < pColumnGroups / 2; vColumnPair++) {
      for (uint32_t vRowGroup = 0; vRowGroup < 2; vRowGroup++) {
        for (uint32_t vColumnGroup = 0; vColumnGroup < 2; vColumnGroup++) {
          pBlock(vRowPair * 2 + vRowGroup, vColumnPair * 2 + vColumnGroup);
        }
      }
    }
  }
}

// pValue rounded up to a multiple of pStep; 0 when it does not fit in 32 bits
uint32_t fRoundUp(uint32_t pValue, uint32_t pStep) {
  const uint64_t vRounded = (uint64_t(pValue) + pStep - 1) / pStep * pStep;
  return vRounded > 0xffffffffu ? 0 : uint32_t(vRounded);
}

// Encodes the 8 values of a matrix of pRows x pColumns floats, its rows pStride floats apart, that start at the row
// pRow and the column pColumn. What falls outside the matrix is zeros.
bool fPackBlock(const float * pMatrix, uint32_t pRows, uint32_t pColumns, uint64_t pStride, uint64_t pRow,
                uint64_t pColumn, uint8_t * pBlock, XdnaRounding pRounding, const XdnaRowNorm * pNorm = nullptr) {
  if (pRow < pRows && pColumn + cBlock <= pColumns) {
    if (pNorm == nullptr) {
      return fXdnaBfp16Encode(pMatrix + pRow * pStride + pColumn, pBlock, pRounding);
    }
    float aNormed[cBlock];
    for (uint32_t i = 0; i < cBlock; i++) {
      aNormed[i] = (pMatrix[pRow * pStride + pColumn + i] * pNorm->vScales[pRow]) * pNorm->vWeights[pColumn + i];
    }
    return fXdnaBfp16Encode(aNormed, pBlock, pRounding);
  }
  if (pRow >= pRows || pColumn >= pColumns) {
    // Outside the matrix: 8 zeros, which either rounding encodes as an exponent of 0 and mantissas of 0. Most of
    // a multiplication of a few rows is this, so it does not go through the encoder.
    std::memset(pBlock, 0, cBlockBytes);
    return true;
  }
  float aValues[cBlock] = {};
  for (uint32_t i = 0; i < cBlock && pColumn + i < pColumns; i++) {
    aValues[i] = pMatrix[pRow * pStride + pColumn + i];
    if (pNorm != nullptr) {
      aValues[i] = (aValues[i] * pNorm->vScales[pRow]) * pNorm->vWeights[pColumn + i];
    }
  }
  return fXdnaBfp16Encode(aValues, pBlock, pRounding);
}

// Writes the tile of a matrix (its rows pStride floats apart) whose first row is pRow and whose first column, along
// K, is pColumn, group by group of 8 rows by 8 values of K in the order pOrder (see XdnaTileOrder), every group as
// its 8 rows. It returns where the next tile goes, or nullptr when a value is not finite.
uint8_t * fPackTile(const float * pMatrix, uint32_t pRows, uint32_t pColumns, uint64_t pStride, uint64_t pRow,
                    uint64_t pColumn, uint32_t pTileRows, uint32_t pTileK, uint8_t * pAt, XdnaRounding pRounding,
                    XdnaTileOrder pOrder, const XdnaRowNorm * pNorm = nullptr) {
  if (pRow >= pRows || pColumn >= pColumns) {
    // the whole tile is outside the matrix: zeros
    const size_t vBytes = size_t(pTileRows) * pTileK / cBlock * cBlockBytes;
    std::memset(pAt, 0, vBytes);
    return pAt + vBytes;
  }
  bool vFinite = true;
  auto fGroup  = [&](uint32_t pRowGroup, uint32_t pGroupK) {
    const uint64_t vRow    = pRow + uint64_t(pRowGroup) * cBlock;
    const uint64_t vColumn = pColumn + uint64_t(pGroupK) * cBlock;
    if (pRounding == XdnaRounding::cNearest && vRow + cBlock <= pRows && vColumn + cBlock <= pColumns) {
      // 8 rows of the same 8 columns, all of them inside the matrix: the fast encoder takes them at once
      if (pNorm != nullptr) {
        // normalised first, (value * scale of its row) * weight of its column, as the encoder reads them (llama.cpp-
        // xdna; before, value by value into a temporary, with scalar products)
        vFinite = fXdnaBfp16EncodeNearestNormed(pMatrix + vRow * pStride + vColumn, pStride, cBlock,
                                                 pNorm->vScales + vRow, pNorm->vWeights + vColumn, pAt) &&
                  vFinite;
      } else {
        vFinite = fXdnaBfp16EncodeNearest(pMatrix + vRow * pStride + vColumn, pStride, cBlock, pAt) && vFinite;
      }
      pAt += cBlock * cBlockBytes;
      return;
    }
    for (uint32_t vLine = 0; vLine < cBlock; vLine++) {
      vFinite = fPackBlock(pMatrix, pRows, pColumns, pStride, vRow + vLine, vColumn, pAt, pRounding, pNorm) && vFinite;
      pAt += cBlockBytes;
    }
  };
  const uint32_t vRowGroups = pTileRows / cBlock;
  const uint32_t vGroupsK   = pTileK / cBlock;
  switch (pOrder) {
    case XdnaTileOrder::cRows:
      for (uint32_t vRowGroup = 0; vRowGroup < vRowGroups; vRowGroup++) {
        for (uint32_t vGroupK = 0; vGroupK < vGroupsK; vGroupK++) {
          fGroup(vRowGroup, vGroupK);
        }
      }
      break;
    case XdnaTileOrder::cSets:
      for (uint32_t vFirst = 0; vFirst < vRowGroups; vFirst += 3) {
        const uint32_t vCount = std::min(3u, vRowGroups - vFirst);
        for (uint32_t vGroupK = 0; vGroupK < vGroupsK; vGroupK++) {
          for (uint32_t j = 0; j < vCount; j++) {
            fGroup(vFirst + j, vGroupK);
          }
        }
      }
      break;
    default:
      for (uint32_t vRowPair = 0; vRowPair < vRowGroups / 2; vRowPair++) {
        for (uint32_t vPairK = 0; vPairK < vGroupsK / 2; vPairK++) {
          for (uint32_t vRowGroup = 0; vRowGroup < 2; vRowGroup++) {
            for (uint32_t vGroupK = 0; vGroupK < 2; vGroupK++) {
              fGroup(vRowPair * 2 + vRowGroup, vPairK * 2 + vGroupK);
            }
          }
        }
      }
      break;
  }
  return vFinite ? pAt : nullptr;
}

// values from which a conversion is shared between the threads: under it, waking them costs more than it saves
const uint64_t cSharedValues = 1u << 18;

// Pieces to cut every one of pParts parts of a conversion into, each of pSteps steps, so that the threads of pWorkers
// get about 4 pieces each: with a batch of 512 rows a conversion has 8 parts (bands of A, columns of cores), one per
// thread, and the threads that run slower (half of the cores of this processor) left the others waiting.
uint32_t fPieces(const XdnaWorkers * pWorkers, uint32_t pParts, uint32_t pSteps) {
  const uint32_t vThreads = pWorkers != nullptr ? pWorkers->mThreads() : 1;
  if (vThreads <= 1 || pParts == 0 || pSteps <= 1) {
    return 1;
  }
  const uint32_t vWanted = (4 * vThreads + pParts - 1) / pParts;
  return std::min(vWanted, pSteps);
}

// runs pTask(0) ... pTask(pCount - 1): on the threads of pWorkers when there are and pShared says so
void fForEach(XdnaWorkers * pWorkers, bool pShared, uint32_t pCount, const std::function<void(uint32_t)> & pTask) {
  if (pWorkers != nullptr && pShared) {
    pWorkers->mForEach(pCount, pTask);
    return;
  }
  for (uint32_t i = 0; i < pCount; i++) {
    pTask(i);
  }
}

}  // namespace

uint8_t * fXdnaPackTileRows(const float * pMatrix, uint32_t pRows, uint32_t pColumns, uint64_t pRow, uint64_t pColumn,
                             uint32_t pTileRows, uint32_t pTileK, uint8_t * pAt, XdnaRounding pRounding) {
  return fPackTile(pMatrix, pRows, pColumns, pColumns, pRow, pColumn, pTileRows, pTileK, pAt, pRounding,
                   XdnaTileOrder::cRows);
}

bool fXdnaGemmSteps3(const XdnaOverlay & pOverlay) {
  return fSteps3(pOverlay);
}

uint8_t * fXdnaPackTileA(const float * pTile, uint32_t pRows, uint32_t pTileM, uint32_t pTileK, uint8_t * pAt,
                         XdnaRounding pRounding, bool pSteps3) {
  return fPackTile(pTile, pRows, pTileK, pTileK, 0, 0, pTileM, pTileK, pAt, pRounding,
                   pSteps3 ? XdnaTileOrder::cRows : XdnaTileOrder::cPairs);
}

bool fXdnaGemmLimits(const XdnaOverlay & pOverlay, uint32_t & pMaxK, uint32_t & pMaxN, std::string & pError) {
  const uint32_t vStreamsA = fStreamsA(pOverlay);
  const uint32_t vStreamsB = fStreamsB(pOverlay);
  if (pOverlay.vTileM == 0 || pOverlay.vTileK == 0 || pOverlay.vTileN == 0 || vStreamsA == 0 || vStreamsB == 0) {
    pError = "the overlay has no tiles or no streams";
    return false;
  }
  if (pOverlay.vFixedK != 0 || pOverlay.vFixedN != 0) {
    pError = "the overlay is built for K = " + std::to_string(pOverlay.vFixedK) + " and N = " +
             std::to_string(pOverlay.vFixedN) + " only";
    return false;
  }

  // what the counters of the cores and the sizes of the buffers hold with room to spare
  const uint64_t cLargest = 0x40000000u;
  const uint64_t vStepN   = uint64_t(pOverlay.vTileN) * vStreamsB;
  uint64_t       vMaxK    = cLargest / pOverlay.vTileK * pOverlay.vTileK;
  uint64_t       vMaxN    = cLargest / vStepN * vStepN;

  uint64_t vSends = cXdnaShimRepeatMax;  // with the bands of A coming from the host, it repeats every one
  if (pOverlay.vLayout == XdnaGemmLayout::cColumns) {
    if (pOverlay.aBands.empty()) {
      pError = "the overlay keeps the bands of A in its memory tiles but does not say how to set them";
      return false;
    }
    // A band of A, vTileM rows of K values, has to fit in the buffer of its memory tile and in the length of a
    // buffer descriptor, and a lock counts how many times it goes to the cores.
    const uint64_t vWords = pOverlay.vBandWords < cMemLengthMax ? pOverlay.vBandWords : cMemLengthMax;
    const uint64_t vFits  = vWords * 4 * cBlock / (uint64_t(cBlockBytes) * pOverlay.vTileM) / pOverlay.vTileK *
                            pOverlay.vTileK;
    vMaxK  = vFits < vMaxK ? vFits : vMaxK;
    vSends = cMemSendsMax;
  }
  vMaxN = vSends * vStepN < vMaxN ? vSends * vStepN : vMaxN;

  if (vMaxK == 0 || vMaxN == 0) {
    pError = "the memory tiles of the overlay do not hold a band of A";
    return false;
  }
  pMaxK = uint32_t(vMaxK);
  pMaxN = uint32_t(vMaxN);
  return true;
}

bool fXdnaGemmShape(const XdnaOverlay & pOverlay, uint32_t pM, uint32_t pK, uint32_t pN, XdnaGemmShape & pShape,
                     std::string & pError, XdnaPromptFormat pFormat) {
  if (pM == 0 || pK == 0 || pN == 0) {
    pError = "the sizes of a multiplication cannot be 0";
    return false;
  }
  if (pOverlay.vTileM == 0 || pOverlay.vTileK == 0 || pOverlay.vTileN == 0 || fStreamsA(pOverlay) == 0 ||
      fStreamsB(pOverlay) == 0) {
    pError = "the overlay has no tiles or no streams";
    return false;
  }
  const uint32_t vStepK = pFormat == XdnaPromptFormat::cQ6K ? pOverlay.vTileK * 4 : pOverlay.vTileK;
  pShape.vM       = pM;
  pShape.vK       = pK;
  pShape.vN       = pN;
  pShape.vPaddedM = fRoundUp(pM, pOverlay.vTileM * fStreamsA(pOverlay));
  // with fewer tiles of K than the cores take outside of their loop, the rest are zeros
  pShape.vPaddedK = fRoundUp(std::max(pK, pOverlay.vTileK * fXdnaGemmTilesKOutside(pOverlay)), vStepK);
  pShape.vPaddedN = fRoundUp(pN, pOverlay.vTileN * fStreamsB(pOverlay));
  if (pShape.vPaddedM == 0 || pShape.vPaddedK == 0 || pShape.vPaddedN == 0) {
    pError = "the multiplication is too large";
    return false;
  }
  return true;
}

uint64_t fXdnaGemmBytesB(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, XdnaPromptFormat pFormat) {
  (void) pOverlay;
  const uint64_t vBfpBytes = uint64_t(pShape.vPaddedK) * pShape.vPaddedN / cBlock * cBlockBytes;
  return pFormat == XdnaPromptFormat::cQ6K ? vBfpBytes / 4 * 3 : vBfpBytes;
}

bool fXdnaGemmPackA(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const float * pA,
                     uint8_t * pBuffer, XdnaRounding pRounding, XdnaWorkers * pWorkers, uint64_t pStride,
                     const XdnaRowNorm * pNorm) {
  const uint64_t vStride = pStride != 0 ? pStride : pShape.vK;
  // [bands of vTileM rows][tiles along K][tile]: every band is a stretch of the buffer of its own
  const uint32_t    vBands     = pShape.vPaddedM / pOverlay.vTileM;
  const uint32_t    vTiles     = pShape.vPaddedK / pOverlay.vTileK;
  const uint64_t    vTileBytes = uint64_t(pOverlay.vTileM) * pOverlay.vTileK / cBlock * cBlockBytes;
  const uint64_t    vBandBytes = uint64_t(vTiles) * vTileBytes;
  const bool        vShared    = uint64_t(pShape.vM) * pShape.vK >= cSharedValues;
  // every band in pieces of consecutive tiles along K, each a stretch of the buffer of its own
  const uint32_t    vPieces    = vShared ? fPieces(pWorkers, vBands, vTiles) : 1;
  const uint32_t    vPerPiece  = (vTiles + vPieces - 1) / vPieces;
  std::atomic<bool> vFinite(true);
  auto fPiece = [&](uint32_t pIndex) {
    const uint32_t vBand  = pIndex / vPieces;
    const uint32_t vFirst = (pIndex % vPieces) * vPerPiece;
    const uint32_t vEnd   = std::min(vTiles, vFirst + vPerPiece);
    uint8_t *      vAt    = pBuffer + vBand * vBandBytes + vFirst * vTileBytes;
    for (uint32_t vTile = vFirst; vAt != nullptr && vTile < vEnd; vTile++) {
      vAt = fPackTile(pA, pShape.vM, pShape.vK, vStride, uint64_t(vBand) * pOverlay.vTileM,
                      uint64_t(vTile) * pOverlay.vTileK,
                      pOverlay.vTileM, pOverlay.vTileK, vAt, pRounding,
                      fSteps3(pOverlay) ? XdnaTileOrder::cRows : XdnaTileOrder::cPairs, pNorm);
    }
    if (vAt == nullptr) {
      vFinite = false;
    }
  };
  fForEach(pWorkers, vShared, vBands * vPieces, fPiece);
  return vFinite;
}

bool fXdnaGemmPackB(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const float * pTransposedB,
                     uint8_t * pBuffer, XdnaRounding pRounding, XdnaWorkers * pWorkers) {
  // [streams of B][turns of the stream][tiles along K][tile]: the stream s carries the tile columns s, s + streams...
  // Every tile column is a stretch of the buffer of its own. Split, [streams of B][pieces][turns][tiles along K]
  // [piece of a tile]: every piece of the tiles of a stream goes through a stream of the overlay of its own.
  const uint32_t    vStreams     = fStreamsB(pOverlay);
  const uint32_t    vSplit       = pOverlay.vSplit;
  const uint32_t    vTurns       = pShape.vPaddedN / (pOverlay.vTileN * vStreams);
  const uint32_t    vTiles       = pShape.vPaddedK / pOverlay.vTileK;
  const uint64_t    vTileBytes   = uint64_t(pOverlay.vTileN) * pOverlay.vTileK / cBlock * cBlockBytes;
  const uint64_t    vColumnBytes = uint64_t(vTiles) * vTileBytes;
  const uint64_t    vPieceBytes  = vTileBytes / vSplit;
  const uint64_t    vPieceRun    = uint64_t(vTurns) * vTiles * vPieceBytes;  // a piece of all the tiles of a stream
  const XdnaTileOrder vOrder    = fSteps3(pOverlay) ? XdnaTileOrder::cSets : XdnaTileOrder::cPairs;
  std::atomic<bool> vFinite(true);
  auto fColumn = [&](uint32_t pIndex) {
    const uint32_t vStream    = pIndex / vTurns;
    const uint32_t vTurn      = pIndex % vTurns;
    const uint64_t vColumnOfB = (uint64_t(vTurn) * vStreams + vStream) * pOverlay.vTileN;
    if (vSplit == 1) {
      uint8_t * vAt = pBuffer + pIndex * vColumnBytes;
      for (uint32_t vTile = 0; vAt != nullptr && vTile < vTiles; vTile++) {
        vAt = fPackTile(pTransposedB, pShape.vN, pShape.vK, pShape.vK, vColumnOfB, uint64_t(vTile) * pOverlay.vTileK,
                        pOverlay.vTileN, pOverlay.vTileK, vAt, pRounding, vOrder);
      }
      if (vAt == nullptr) {
        vFinite = false;
      }
      return;
    }
    // split: every tile is written whole here and its pieces go to their places
    std::vector<uint8_t> aTile(vTileBytes);
    for (uint32_t vTile = 0; vTile < vTiles; vTile++) {
      if (fPackTile(pTransposedB, pShape.vN, pShape.vK, pShape.vK, vColumnOfB, uint64_t(vTile) * pOverlay.vTileK,
                    pOverlay.vTileN, pOverlay.vTileK, aTile.data(), pRounding, vOrder) == nullptr) {
        vFinite = false;
        return;
      }
      for (uint32_t vPiece = 0; vPiece < vSplit; vPiece++) {
        uint8_t * vTo = pBuffer + (uint64_t(vStream) * vSplit + vPiece) * vPieceRun +
                        (uint64_t(vTurn) * vTiles + vTile) * vPieceBytes;
        std::memcpy(vTo, aTile.data() + vPiece * vPieceBytes, vPieceBytes);
      }
    }
  };
  fForEach(pWorkers, uint64_t(pShape.vN) * pShape.vK >= cSharedValues, vStreams * vTurns, fColumn);
  return vFinite;
}

bool fXdnaGemmPackBRows(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape,
                         const XdnaGemmRowReader & pReadRow, uint8_t * pBuffer, XdnaRounding pRounding,
                         XdnaWorkers * pWorkers) {
  const uint32_t      vStreams     = fStreamsB(pOverlay);
  const uint32_t      vSplit       = pOverlay.vSplit;
  const uint32_t      vTurns       = pShape.vPaddedN / (pOverlay.vTileN * vStreams);
  const uint32_t      vTiles       = pShape.vPaddedK / pOverlay.vTileK;
  const uint64_t      vTileBytes   = uint64_t(pOverlay.vTileN) * pOverlay.vTileK / cBlock * cBlockBytes;
  const uint64_t      vColumnBytes = uint64_t(vTiles) * vTileBytes;
  const uint64_t      vPieceBytes  = vTileBytes / vSplit;
  const uint64_t      vPieceRun    = uint64_t(vTurns) * vTiles * vPieceBytes;
  const XdnaTileOrder vOrder       = fSteps3(pOverlay) ? XdnaTileOrder::cSets : XdnaTileOrder::cPairs;
  std::atomic<bool>   vFinite(true);
  auto fColumn = [&](uint32_t pIndex) {
    const uint32_t                  vStream    = pIndex / vTurns;
    const uint32_t                  vTurn      = pIndex % vTurns;
    const uint64_t                  vColumnOfB = (uint64_t(vTurn) * vStreams + vStream) * pOverlay.vTileN;
    const uint32_t                  vRows      =
        vColumnOfB >= pShape.vN ? 0 : uint32_t(std::min<uint64_t>(pOverlay.vTileN, uint64_t(pShape.vN) - vColumnOfB));
    thread_local std::vector<float> aStrip;
    aStrip.resize(size_t(pOverlay.vTileN) * pShape.vK);
    for (uint32_t r = 0; r < vRows; r++) {
      pReadRow(uint32_t(vColumnOfB + r), aStrip.data() + size_t(r) * pShape.vK);
    }
    if (vSplit == 1) {
      uint8_t * vAt = pBuffer + pIndex * vColumnBytes;
      for (uint32_t vTile = 0; vAt != nullptr && vTile < vTiles; vTile++) {
        vAt = fPackTile(aStrip.data(), vRows, pShape.vK, pShape.vK, 0, uint64_t(vTile) * pOverlay.vTileK,
                        pOverlay.vTileN, pOverlay.vTileK, vAt, pRounding, vOrder);
      }
      if (vAt == nullptr) {
        vFinite = false;
      }
      return;
    }
    std::vector<uint8_t> aTile(vTileBytes);
    for (uint32_t vTile = 0; vTile < vTiles; vTile++) {
      if (fPackTile(aStrip.data(), vRows, pShape.vK, pShape.vK, 0, uint64_t(vTile) * pOverlay.vTileK,
                    pOverlay.vTileN, pOverlay.vTileK, aTile.data(), pRounding, vOrder) == nullptr) {
        vFinite = false;
        return;
      }
      for (uint32_t vPiece = 0; vPiece < vSplit; vPiece++) {
        uint8_t * vTo = pBuffer + (uint64_t(vStream) * vSplit + vPiece) * vPieceRun +
                        (uint64_t(vTurn) * vTiles + vTile) * vPieceBytes;
        std::memcpy(vTo, aTile.data() + vPiece * vPieceBytes, vPieceBytes);
      }
    }
  };
  fForEach(pWorkers, uint64_t(pShape.vN) * pShape.vK >= cSharedValues, vStreams * vTurns, fColumn);
  return vFinite;
}

namespace {

uint16_t fGemmHalfAsBf16(const uint8_t * pFrom) {
  uint16_t vHalf = 0;
  std::memcpy(&vHalf, pFrom, 2);
  return fXdnaBf16FromFloat(fXdnaHalfToFloat(vHalf));
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
__attribute__((target("avx2"))) void fGemmPackSliceQ6K(__m128i pLo, __m128i pHi, uint32_t * pOut) {
  const __m128i vPack = _mm_set1_epi16(0x1001);
  _mm_storeu_si128(reinterpret_cast<__m128i *>(pOut),
                   _mm_packus_epi16(_mm_maddubs_epi16(pLo, vPack), _mm_maddubs_epi16(pHi, vPack)));
}

__attribute__((target("avx2"))) void fPackColQ6KAvx2(const uint8_t * const * aBlocks, uint32_t pHalf, uint8_t * pCol) {
  alignas(32) uint32_t aLow32[8][16];
  alignas(32) uint32_t aTop32[4][16];
  uint16_t             aD[16] = {};
  const __m128i        vMask4 = _mm_set1_epi8(0x0f);
  const __m256i        vMask2 = _mm256_set1_epi8(0x03);
  std::memset(pCol + 32, 0, 64);
  for (uint32_t j = 0; j < 8; j++) {
    if (aBlocks[j] == nullptr) {
      std::memset(aLow32[j], 0, sizeof(aLow32[j]));
      continue;
    }
    const uint8_t * vBlock = aBlocks[j];
    aD[j]                  = fGemmHalfAsBf16(vBlock + 208);
    for (uint32_t s = 0; s < 8; s++) {
      pCol[32 + s * 8 + j] = vBlock[192 + 8 * pHalf + s];
    }
    const uint8_t * vQl = vBlock + 64 * pHalf;
    const __m128i   vQ0 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl));
    const __m128i   vQ1 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl + 16));
    const __m128i   vQ2 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl + 32));
    const __m128i   vQ3 = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vQl + 48));
    fGemmPackSliceQ6K(_mm_and_si128(vQ0, vMask4), _mm_and_si128(vQ1, vMask4), &aLow32[j][0]);
    fGemmPackSliceQ6K(_mm_and_si128(vQ2, vMask4), _mm_and_si128(vQ3, vMask4), &aLow32[j][4]);
    fGemmPackSliceQ6K(_mm_and_si128(_mm_srli_epi16(vQ0, 4), vMask4),
                      _mm_and_si128(_mm_srli_epi16(vQ1, 4), vMask4), &aLow32[j][8]);
    fGemmPackSliceQ6K(_mm_and_si128(_mm_srli_epi16(vQ2, 4), vMask4),
                      _mm_and_si128(_mm_srli_epi16(vQ3, 4), vMask4), &aLow32[j][12]);
  }
  for (uint32_t j = 0; j < 4; j++) {
    const __m256i vLo = aBlocks[j] != nullptr ?
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(aBlocks[j] + 128 + 32 * pHalf)) :
                            _mm256_setzero_si256();
    const __m256i vHi = aBlocks[j + 4] != nullptr ?
                            _mm256_loadu_si256(reinterpret_cast<const __m256i *>(aBlocks[j + 4] + 128 + 32 * pHalf)) :
                            _mm256_setzero_si256();
    const __m256i aFields[4] = {
      _mm256_or_si256(_mm256_and_si256(vLo, vMask2), _mm256_slli_epi16(_mm256_and_si256(vHi, vMask2), 2)),
      _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vLo, 2), vMask2),
                      _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vHi, 2), vMask2), 2)),
      _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vLo, 4), vMask2),
                      _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vHi, 4), vMask2), 2)),
      _mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vLo, 6), vMask2),
                      _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vHi, 6), vMask2), 2)),
    };
    for (uint32_t k = 0; k < 4; k++) {
      fGemmPackSliceQ6K(_mm256_castsi256_si128(aFields[k]), _mm256_extracti128_si256(aFields[k], 1),
                        &aTop32[j][4 * k]);
    }
  }
  std::memcpy(pCol, aD, sizeof(aD));
  for (uint32_t vGroup = 0; vGroup < 16; vGroup++) {
    uint32_t aOutLow[8];
    uint32_t aOutTop[4];
    for (uint32_t j = 0; j < 8; j++) {
      aOutLow[j] = aLow32[j][vGroup];
    }
    for (uint32_t j = 0; j < 4; j++) {
      aOutTop[j] = aTop32[j][vGroup];
    }
    std::memcpy(pCol + 96 + vGroup * 32, aOutLow, sizeof(aOutLow));
    std::memcpy(pCol + 608 + vGroup * 16, aOutTop, sizeof(aOutTop));
  }
}
#endif

void fPackColQ6K(const uint8_t * const * aBlocks, uint32_t pHalf, uint8_t * pCol) {
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  static const bool cHasAvx2 = __builtin_cpu_supports("avx2");
  if (cHasAvx2) {
    fPackColQ6KAvx2(aBlocks, pHalf, pCol);
    return;
  }
#endif
  uint8_t  aLow[8][128] = {};
  uint8_t  aTop[8][128] = {};
  uint16_t aD[16]       = {};
  std::memset(pCol, 0, 864);
  for (uint32_t j = 0; j < 8; j++) {
    if (aBlocks[j] == nullptr) {
      continue;
    }
    const uint8_t * vBlock = aBlocks[j];
    const uint8_t * vQl    = vBlock + 64 * pHalf;
    const uint8_t * vQh    = vBlock + 128 + 32 * pHalf;
    aD[j]                  = fGemmHalfAsBf16(vBlock + 208);
    for (uint32_t s = 0; s < 8; s++) {
      pCol[32 + s * 8 + j] = vBlock[192 + 8 * pHalf + s];
    }
    for (uint32_t k = 0; k < 4; k++) {
      for (uint32_t l = 0; l < 32; l++) {
        const uint32_t v     = 32 * k + l;
        const uint8_t  vByte = vQl[(k % 2) * 32 + l];
        aLow[j][v]           = k < 2 ? (vByte & 0xf) : (vByte >> 4);
        aTop[j][v]           = (vQh[l] >> (2 * k)) & 3;
      }
    }
  }
  std::memcpy(pCol, aD, sizeof(aD));
  for (uint32_t vGroup = 0; vGroup < 16; vGroup++) {
    uint8_t * vToLow = pCol + 96 + vGroup * 32;
    for (uint32_t n = 0; n < 64; n += 2) {
      const uint32_t vFirst  = aLow[n / 8][vGroup * 8 + n % 8];
      const uint32_t vSecond = aLow[(n + 1) / 8][vGroup * 8 + (n + 1) % 8];
      vToLow[n / 2]          = uint8_t((vFirst & 0xf) | ((vSecond & 0xf) << 4));
    }
    uint8_t * vToTop = pCol + 608 + vGroup * 16;
    for (uint32_t k = 0; k < 32; k++) {
      const uint32_t vLow   = aTop[k / 8][vGroup * 8 + k % 8];
      const uint32_t vHigh  = aTop[(k + 32) / 8][vGroup * 8 + (k + 32) % 8];
      const uint32_t vField = vLow | (vHigh << 2);
      vToTop[k / 2] |= uint8_t(k % 2 == 0 ? vField : vField << 4);
    }
  }
}

}  // namespace

bool fXdnaGemmPackBQ6K(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape,
                        const XdnaGemmRowPtrReader & pRowPtr, uint8_t * pBuffer, XdnaWorkers * pWorkers) {
  if (pOverlay.vCompB != 1 || pOverlay.vSplit != 1 || pOverlay.vTileK != 128 || pOverlay.vTileN != 64 ||
      pShape.vK % 256 != 0 || pShape.vPaddedK % 512 != 0) {
    return false;
  }
  const uint32_t vStreams     = fStreamsB(pOverlay);
  const uint32_t vTurns       = pShape.vPaddedN / (pOverlay.vTileN * vStreams);
  const uint32_t vTiles       = pShape.vPaddedK / pOverlay.vTileK;
  const uint64_t vTileBytes   = 6912;
  const uint64_t vColumnBytes = uint64_t(vTiles) * vTileBytes;
  const uint32_t vSupersValid = pShape.vK / 256;
  auto           fColumn      = [&](uint32_t pIndex) {
    const uint32_t  vStream    = pIndex / vTurns;
    const uint32_t  vTurn      = pIndex % vTurns;
    const uint64_t  vColumnOfB = (uint64_t(vTurn) * vStreams + vStream) * pOverlay.vTileN;
    const uint8_t * aRows[64]  = {};
    for (uint32_t r = 0; r < 64; r++) {
      aRows[r] = vColumnOfB + r < pShape.vN ? pRowPtr(uint32_t(vColumnOfB + r)) : nullptr;
    }
    uint8_t * vColOut = pBuffer + pIndex * vColumnBytes;
    for (uint32_t vTile = 0; vTile < vTiles; vTile++) {
      uint8_t *      vTileAt = vColOut + uint64_t(vTile) * vTileBytes;
      const uint32_t vSuper  = vTile / 2;
      const uint32_t vHalf   = vTile % 2;
      if (vSuper >= vSupersValid) {
        std::memset(vTileAt, 0, vTileBytes);
        continue;
      }
      for (uint32_t g = 0; g < 8; g++) {
        const uint8_t * aBlocks[8];
        for (uint32_t j = 0; j < 8; j++) {
          const uint8_t * vRowBase = aRows[g * 8 + j];
          aBlocks[j]               = vRowBase != nullptr ? vRowBase + size_t(vSuper) * 210 : nullptr;
        }
        fPackColQ6K(aBlocks, vHalf, vTileAt + g * 864);
      }
    }
  };
  fForEach(pWorkers, uint64_t(pShape.vN) * pShape.vK >= cSharedValues, vStreams * vTurns, fColumn);
  return true;
}

bool fXdnaGemmPackBQ8_0(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape,
                         const XdnaGemmRowPtrReader & pRowPtr, uint8_t * pBuffer, XdnaWorkers * pWorkers) {
  if (pOverlay.vCompB != 1 || pOverlay.vSplit != 1 || pOverlay.vTileK != 128 || pOverlay.vTileN != 64 ||
      pShape.vK % 32 != 0) {
    return false;
  }
  const uint32_t vStreams     = fStreamsB(pOverlay);
  const uint32_t vTurns       = pShape.vPaddedN / (pOverlay.vTileN * vStreams);
  const uint32_t vTiles       = pShape.vPaddedK / pOverlay.vTileK;
  const uint64_t vTileBytes   = 9216;
  const uint64_t vColumnBytes = uint64_t(vTiles) * vTileBytes;
  const uint32_t vSubsTotal   = pShape.vK / 32;
  auto           fColumn      = [&](uint32_t pIndex) {
    const uint32_t  vStream    = pIndex / vTurns;
    const uint32_t  vTurn      = pIndex % vTurns;
    const uint64_t  vColumnOfB = (uint64_t(vTurn) * vStreams + vStream) * pOverlay.vTileN;
    const uint8_t * aRows[64]  = {};
    for (uint32_t r = 0; r < 64; r++) {
      aRows[r] = vColumnOfB + r < pShape.vN ? pRowPtr(uint32_t(vColumnOfB + r)) : nullptr;
    }
    uint8_t * vColOut = pBuffer + pIndex * vColumnBytes;
    for (uint32_t vTile = 0; vTile < vTiles; vTile++) {
      uint8_t *      vTileAt  = vColOut + uint64_t(vTile) * vTileBytes;
      const uint32_t vSubBase = vTile * 4;
      std::memset(vTileAt + 8704, 0, 512);
      if (vSubBase >= vSubsTotal) {
        std::memset(vTileAt, 0, 8704);
        continue;
      }
      const uint32_t vValidSubs = std::min<uint32_t>(4u, vSubsTotal - vSubBase);
      for (uint32_t g = 0; g < 8; g++) {
        uint8_t *  vCol    = vTileAt + g * 1088;
        uint16_t * vScales = reinterpret_cast<uint16_t *>(vCol);
        uint64_t * vValues = reinterpret_cast<uint64_t *>(vCol + 64);
        for (uint32_t j = 0; j < 8; j++) {
          const uint8_t * vRowBase = aRows[g * 8 + j];
          const uint32_t  vSubCnt  = vRowBase != nullptr ? vValidSubs : 0u;
          for (uint32_t s = 0; s < vSubCnt; s++) {
            const uint8_t * vBlock = vRowBase + size_t(vSubBase + s) * 34;
            vScales[s * 8 + j]     = fGemmHalfAsBf16(vBlock);
            uint64_t aRuns[4];
            std::memcpy(aRuns, vBlock + 2, 32);
            vValues[(s * 4 + 0) * 8 + j] = aRuns[0];
            vValues[(s * 4 + 1) * 8 + j] = aRuns[1];
            vValues[(s * 4 + 2) * 8 + j] = aRuns[2];
            vValues[(s * 4 + 3) * 8 + j] = aRuns[3];
          }
          for (uint32_t s = vSubCnt; s < 4; s++) {
            vScales[s * 8 + j]           = 0;
            vValues[(s * 4 + 0) * 8 + j] = 0;
            vValues[(s * 4 + 1) * 8 + j] = 0;
            vValues[(s * 4 + 2) * 8 + j] = 0;
            vValues[(s * 4 + 3) * 8 + j] = 0;
          }
        }
      }
    }
  };
  fForEach(pWorkers, uint64_t(pShape.vN) * pShape.vK >= cSharedValues, vStreams * vTurns, fColumn);
  return true;
}

namespace {

// a sum of C, as the overlay gives it back, as a float
inline float fSumOf(const float * pAt) {
  return *pAt;
}

inline float fSumOf(const uint16_t * pAt) {
  return fXdnaBf16ToFloat(*pAt);
}

// Writes 8 sums of C as floats at pTo, which is aligned to 16 bytes: past the cache when it can, as C is written
// once and is larger than the cache, and a store into the cache first reads the line it goes to.
inline void fWriteEight(float * pTo, const float * pFrom) {
#if defined(__x86_64__)
  _mm_stream_ps(pTo, _mm_loadu_ps(pFrom));
  _mm_stream_ps(pTo + 4, _mm_loadu_ps(pFrom + 4));
#else
  std::memcpy(pTo, pFrom, 8 * sizeof(float));
#endif
}

inline void fWriteEight(float * pTo, const uint16_t * pFrom) {
#if defined(__x86_64__)
  // a bfloat16 is the upper half of a float: next to a zero below it, it is the float
  const __m128i vBits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom));
  const __m128i vZero = _mm_setzero_si128();
  _mm_stream_ps(pTo, _mm_castsi128_ps(_mm_unpacklo_epi16(vZero, vBits)));
  _mm_stream_ps(pTo + 4, _mm_castsi128_ps(_mm_unpackhi_epi16(vZero, vBits)));
#else
  for (uint32_t i = 0; i < 8; i++) {
    pTo[i] = fXdnaBf16ToFloat(pFrom[i]);
  }
#endif
}

// Adds 8 sums of C to the 8 floats at pTo: the pieces of a matrix along its inputs after the first, in the order
// they come (the first one is written, the others added one after the other).
inline void fAddEight(float * pTo, const float * pFrom) {
  for (uint32_t i = 0; i < 8; i++) {
    pTo[i] += pFrom[i];
  }
}

inline void fAddEight(float * pTo, const uint16_t * pFrom) {
#if defined(__x86_64__)
  const __m128i vBits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom));
  const __m128i vZero = _mm_setzero_si128();
  _mm_storeu_ps(pTo, _mm_add_ps(_mm_loadu_ps(pTo), _mm_castsi128_ps(_mm_unpacklo_epi16(vZero, vBits))));
  _mm_storeu_ps(pTo + 4, _mm_add_ps(_mm_loadu_ps(pTo + 4), _mm_castsi128_ps(_mm_unpackhi_epi16(vZero, vBits))));
#else
  for (uint32_t i = 0; i < 8; i++) {
    pTo[i] += fXdnaBf16ToFloat(pFrom[i]);
  }
#endif
}

// llama.cpp-xdna: 8 sums of C with a residual sum: pTo[i] = (pBase[i] + sum) + pAlso[i], or sum + pAlso[i] when
// pBase is null (see XdnaGemmTarget). pTo can be pBase or pAlso: the 8 values are read before they are written.
inline void fResidualEight(float * pTo, const float * pBase, const float * pFrom, const float * pAlso) {
  float aValues[8];
  for (uint32_t i = 0; i < 8; i++) {
    aValues[i] = pBase != nullptr ? pBase[i] + pFrom[i] : pFrom[i];
    aValues[i] = aValues[i] + pAlso[i];
  }
  std::memcpy(pTo, aValues, sizeof(aValues));
}

inline void fResidualEight(float * pTo, const float * pBase, const uint16_t * pFrom, const float * pAlso) {
#if defined(__x86_64__)
  const __m128i vBits = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom));
  const __m128i vZero = _mm_setzero_si128();
  __m128        vLow  = _mm_castsi128_ps(_mm_unpacklo_epi16(vZero, vBits));
  __m128        vHigh = _mm_castsi128_ps(_mm_unpackhi_epi16(vZero, vBits));
  if (pBase != nullptr) {
    vLow  = _mm_add_ps(_mm_loadu_ps(pBase), vLow);
    vHigh = _mm_add_ps(_mm_loadu_ps(pBase + 4), vHigh);
  }
  vLow  = _mm_add_ps(vLow, _mm_loadu_ps(pAlso));
  vHigh = _mm_add_ps(vHigh, _mm_loadu_ps(pAlso + 4));
  _mm_storeu_ps(pTo, vLow);
  _mm_storeu_ps(pTo + 4, vHigh);
#else
  float aValues[8];
  for (uint32_t i = 0; i < 8; i++) {
    aValues[i] = pBase != nullptr ? pBase[i] + fXdnaBf16ToFloat(pFrom[i]) : fXdnaBf16ToFloat(pFrom[i]);
    aValues[i] = aValues[i] + pAlso[i];
  }
  std::memcpy(pTo, aValues, sizeof(aValues));
#endif
}

// the target that takes the output pOutput, or none
inline const XdnaGemmTarget * fTargetOf(const XdnaGemmTarget * aTargets, uint32_t pTargets, uint64_t pOutput) {
  for (uint32_t i = 0; i < pTargets; i++) {
    if (pOutput >= aTargets[i].vFirst && pOutput - aTargets[i].vFirst < aTargets[i].vCount) {
      return &aTargets[i];
    }
  }
  return nullptr;
}

#if defined(__x86_64__)
const bool cHasGemmAvx2 = __builtin_cpu_supports("avx2") != 0;

__attribute__((target("avx2"))) inline void fWriteSixteenBf16Avx2(float * pTo, const uint16_t * pFrom) {
  const __m128i vLo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom));
  const __m128i vHi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom + 8));
  _mm256_stream_ps(pTo, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vLo), 16)));
  _mm256_stream_ps(pTo + 8, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vHi), 16)));
}

__attribute__((target("avx2"))) inline void fAddSixteenBf16Avx2(float * pTo, const uint16_t * pFrom) {
  const __m128i vLo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom));
  const __m128i vHi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom + 8));
  _mm256_storeu_ps(
      pTo, _mm256_add_ps(_mm256_loadu_ps(pTo), _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vLo), 16))));
  _mm256_storeu_ps(pTo + 8, _mm256_add_ps(_mm256_loadu_ps(pTo + 8),
                                          _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vHi), 16))));
}

__attribute__((target("avx2"))) inline void fResidualSixteenBf16Avx2(float * pTo, const float * pBase,
                                                                     const uint16_t * pFrom, const float * pAlso) {
  __m256 vLo = _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom))), 16));
  __m256 vHi = _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_cvtepu16_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i *>(pFrom + 8))), 16));
  if (pBase != nullptr) {
    vLo = _mm256_add_ps(_mm256_loadu_ps(pBase), vLo);
    vHi = _mm256_add_ps(_mm256_loadu_ps(pBase + 8), vHi);
  }
  vLo = _mm256_add_ps(vLo, _mm256_loadu_ps(pAlso));
  vHi = _mm256_add_ps(vHi, _mm256_loadu_ps(pAlso + 8));
  _mm256_storeu_ps(pTo, vLo);
  _mm256_storeu_ps(pTo + 8, vHi);
}

__attribute__((target("avx2"))) inline void fLoadBf16PairRunAvx2(const uint16_t * pGate, const uint16_t * pUp,
                                                                 float * pGateOut, float * pUpOut, uint32_t pRun) {
  uint32_t i = 0;
  for (; i + 16 <= pRun; i += 16) {
    const __m128i vGLo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pGate + i));
    const __m128i vGHi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pGate + i + 8));
    const __m128i vULo = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pUp + i));
    const __m128i vUHi = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pUp + i + 8));
    _mm256_storeu_ps(pGateOut + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vGLo), 16)));
    _mm256_storeu_ps(pGateOut + i + 8, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vGHi), 16)));
    _mm256_storeu_ps(pUpOut + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vULo), 16)));
    _mm256_storeu_ps(pUpOut + i + 8, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vUHi), 16)));
  }
  for (; i + 8 <= pRun; i += 8) {
    const __m128i vG = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pGate + i));
    const __m128i vU = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pUp + i));
    _mm256_storeu_ps(pGateOut + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vG), 16)));
    _mm256_storeu_ps(pUpOut + i, _mm256_castsi256_ps(_mm256_slli_epi32(_mm256_cvtepu16_epi32(vU), 16)));
  }
  for (; i < pRun; i++) {
    pGateOut[i] = fXdnaBf16ToFloat(pGate[i]);
    pUpOut[i]   = fXdnaBf16ToFloat(pUp[i]);
  }
}
#endif

// llama.cpp-xdna: a row of pCount sums of C into the floats at pTo; past the cache from the first float aligned to
// 16 bytes (see fWriteEight)
template <typename T>
void fWriteRow(float * pTo, const T * pFrom, uint64_t pCount) {
  uint64_t i = 0;
#if defined(__x86_64__)
  if constexpr (std::is_same_v<T, uint16_t>) {
    if (cHasGemmAvx2) {
      for (; i < pCount && reinterpret_cast<uintptr_t>(pTo + i) % 32 != 0; i++) {
        pTo[i] = fSumOf(pFrom + i);
      }
      for (; i + 16 <= pCount; i += 16) {
        fWriteSixteenBf16Avx2(pTo + i, pFrom + i);
      }
    }
  }
#endif
  for (; i < pCount && reinterpret_cast<uintptr_t>(pTo + i) % 16 != 0; i++) {
    pTo[i] = fSumOf(pFrom + i);
  }
  for (; i + cBlock <= pCount; i += cBlock) {
    fWriteEight(pTo + i, pFrom + i);
  }
  for (; i < pCount; i++) {
    pTo[i] = fSumOf(pFrom + i);
  }
}

// the same added to the floats at pTo
template <typename T>
void fAddRow(float * pTo, const T * pFrom, uint64_t pCount) {
  uint64_t i = 0;
#if defined(__x86_64__)
  if constexpr (std::is_same_v<T, uint16_t>) {
    if (cHasGemmAvx2) {
      for (; i + 16 <= pCount; i += 16) {
        fAddSixteenBf16Avx2(pTo + i, pFrom + i);
      }
    }
  }
#endif
  for (; i + cBlock <= pCount; i += cBlock) {
    fAddEight(pTo + i, pFrom + i);
  }
  for (; i < pCount; i++) {
    pTo[i] += fSumOf(pFrom + i);
  }
}

// the same with a residual sum: pTo[i] = (pBase[i] + sum) + pAlso[i], or sum + pAlso[i] when pBase is null, as
// fResidualEight
template <typename T>
void fResidualRow(float * pTo, const float * pBase, const T * pFrom, const float * pAlso, uint64_t pCount) {
  uint64_t i = 0;
#if defined(__x86_64__)
  if constexpr (std::is_same_v<T, uint16_t>) {
    if (cHasGemmAvx2) {
      for (; i + 16 <= pCount; i += 16) {
        fResidualSixteenBf16Avx2(pTo + i, pBase != nullptr ? pBase + i : nullptr, pFrom + i, pAlso + i);
      }
    }
  }
#endif
  for (; i + cBlock <= pCount; i += cBlock) {
    fResidualEight(pTo + i, pBase != nullptr ? pBase + i : nullptr, pFrom + i, pAlso + i);
  }
  for (; i < pCount; i++) {
    const float vSum = pBase != nullptr ? pBase[i] + fSumOf(pFrom + i) : fSumOf(pFrom + i);
    pTo[i]           = vSum + pAlso[i];
  }
}

// llama.cpp-xdna: C as strips of the width of a tile (XdnaOverlay::vRowsC), [padded N / pWidth][padded M][pWidth]
// sums: pieces of groups of cStripGroup strips and of their rows for the threads, about 4 a thread, and every piece
// calls pRun(first row, end row, first strip, end strip). A row of a group is then a run of cStripGroup * pWidth
// values of its row of a target (2 KB of floats with tiles of 64), not one of pWidth: a run of 256 bytes in every row
// read C 0.51 ms for 512 x 9216, the group of 8, 0.40.
constexpr uint64_t cStripGroup = 8;

template <typename F>
void fForEachStrip(const XdnaGemmShape & pShape, uint64_t pWidth, XdnaWorkers * pWorkers, bool pAlignGdnChunks,
                   F && pRun) {
  const uint64_t vStrips  = (pShape.vN + pWidth - 1) / pWidth;
  const uint64_t vGroups  = (vStrips + cStripGroup - 1) / cStripGroup;
  const bool     vShared  = uint64_t(pShape.vM) * pShape.vN >= cSharedValues;
  const uint64_t vThreads = vShared && pWorkers != nullptr ? pWorkers->mThreads() : 1;
  const uint64_t vRowBlocks = (pShape.vM + cXdnaGdnChunk - 1) / cXdnaGdnChunk;
  uint64_t       vChunks    = std::max<uint64_t>(1, std::min<uint64_t>(std::max<uint64_t>(1, pShape.vM / 64),
                                                                       (4 * vThreads + vGroups - 1) / vGroups));
  if (pAlignGdnChunks) {
    vChunks = std::max<uint64_t>(1, std::min<uint64_t>(vChunks, vRowBlocks));
  }
  fForEach(pWorkers, vShared, uint32_t(vGroups * vChunks), [&](uint32_t pPiece) {
    const uint64_t vGroup = pPiece / vChunks;
    const uint64_t vChunk = pPiece % vChunks;
    const uint64_t vFirstRow = pAlignGdnChunks ? (vRowBlocks * vChunk / vChunks) * cXdnaGdnChunk :
                                                 uint64_t(pShape.vM) * vChunk / vChunks;
    const uint64_t vEndRow   = pAlignGdnChunks ?
                                   std::min<uint64_t>(pShape.vM, (vRowBlocks * (vChunk + 1) / vChunks) * cXdnaGdnChunk) :
                                   uint64_t(pShape.vM) * (vChunk + 1) / vChunks;
    if (vFirstRow < vEndRow) {
      pRun(vFirstRow, vEndRow, vGroup * cStripGroup, std::min(vStrips, (vGroup + 1) * cStripGroup));
    }
#if defined(__x86_64__)
    // what went past the cache is in memory before anyone reads it
    _mm_sfence();
#endif
  });
}

// llama.cpp-xdna: the most columns a strip can have for the short convolution of a target (fConvStrips)
constexpr uint64_t cConvMaxWidth = 64;

// llama.cpp-xdna: the short convolution of a target (XdnaGemmTarget::vConvWeights) and its SiLU for the rows
// pFirstRow to pEndRow of the strips pFirstStrip to pEndStrip, strip after strip: with bfloat16 and vectors, the rows
// from the fourth on in one call (fXdnaConvRowsBf16: every value of C read by the 4 rows whose tap it is, the 3 after
// the first time from the cache, and converted in the registers); the first 3 rows, which take the old states, and
// every row without such vectors, as floats one by one; and the last 3 rows as they are into vConvTail
template <typename T>
void fConvStrips(const XdnaGemmShape & pShape, const T * pBuffer, uint64_t pWidth, const XdnaGemmTarget & pTarget,
                 uint64_t pFirstRow, uint64_t pEndRow, uint64_t pFirstStrip, uint64_t pEndStrip) {
  alignas(64) float aRows[4][cConvMaxWidth];
  if (pTarget.vGdnPrepack != nullptr && pWidth == cConvMaxWidth && pTarget.vFirst == 0 && pFirstStrip % 2 == 0 &&
      pFirstRow % cXdnaGdnChunk == 0) {
    alignas(64) float aHead[cXdnaGdnChunk][cXdnaGdnDim];
    for (uint64_t vStrip = pFirstStrip; vStrip + 1 < pEndStrip; vStrip += 2) {
      const uint64_t vChannel = vStrip * pWidth;
      if (vChannel + cXdnaGdnDim > pTarget.vCount || vChannel + cXdnaGdnDim > pShape.vN) {
        break;
      }
      for (uint64_t vHalf = 0; vHalf < 2; vHalf++) {
        const uint64_t vOutput  = vChannel + vHalf * pWidth;
        const T *      vStripAt = pBuffer + (vStrip + vHalf) * pShape.vPaddedM * pWidth;
        for (uint64_t vRow = std::max<uint64_t>(pFirstRow, uint64_t(pShape.vM) - 3); vRow < pEndRow; vRow++) {
          float *   vTail = pTarget.vConvTail + (vRow + 3 - uint64_t(pShape.vM)) * pTarget.vConvStride + vOutput;
          const T * vFrom = vStripAt + vRow * pWidth;
          for (uint64_t i = 0; i < pWidth; i++) {
            vTail[i] = fSumOf(vFrom + i);
          }
        }
      }
      for (uint64_t vChunkRow = pFirstRow; vChunkRow < pEndRow; vChunkRow += cXdnaGdnChunk) {
        const uint64_t vChunkEnd = std::min<uint64_t>(pEndRow, vChunkRow + cXdnaGdnChunk);
        for (uint64_t vHalf = 0; vHalf < 2; vHalf++) {
          const uint64_t vOutput     = vChannel + vHalf * pWidth;
          const T *      vStripAt    = pBuffer + (vStrip + vHalf) * pShape.vPaddedM * pWidth;
          uint64_t       vVectorFrom = vChunkEnd;
          if constexpr (std::is_same_v<T, uint16_t>) {
            const uint64_t vFrom = std::max<uint64_t>(vChunkRow, 3);
            if (vFrom < vChunkEnd &&
                fXdnaConvChunkBf16(vStripAt, pWidth, vFrom, vChunkEnd, pTarget.vConvWeights + vOutput,
                                   pTarget.vConvStride, &aHead[vFrom - vChunkRow][vHalf * pWidth], cXdnaGdnDim,
                                   uint32_t(pWidth))) {
              vVectorFrom = vFrom;
            }
          }
          for (uint64_t vRow = vChunkRow; vRow < vVectorFrom; vRow++) {
            const float * aTaps[4];
            for (uint64_t k = 0; k < 4; k++) {
              if (vRow + k < 3) {
                aTaps[k] = pTarget.vConvHead + (vRow + k) * pTarget.vConvStride + vOutput;
                continue;
              }
              const T * vFrom = vStripAt + (vRow + k - 3) * pWidth;
              for (uint64_t i = 0; i < pWidth; i++) {
                aRows[k][i] = fSumOf(vFrom + i);
              }
              aTaps[k] = aRows[k];
            }
            fXdnaConvToken(aTaps, pTarget.vConvWeights + vOutput, pTarget.vConvStride,
                           &aHead[vRow - vChunkRow][vHalf * pWidth], uint32_t(pWidth), true);
          }
        }
        fXdnaGdnPackConvHeadChunk(*pTarget.vGdnPrepack, uint32_t(vChannel / cXdnaGdnDim),
                                  uint32_t(vChunkRow / cXdnaGdnChunk), &aHead[0][0], vChunkEnd == pShape.vM);
      }
    }
    return;
  }
  for (uint64_t vStrip = pFirstStrip; vStrip < pEndStrip; vStrip++) {
    const uint64_t vColumn = vStrip * pWidth;
    const uint64_t vBegin  = std::max<uint64_t>(vColumn, pTarget.vFirst);
    const uint64_t vEnd = std::min<uint64_t>({vColumn + pWidth, pTarget.vFirst + pTarget.vCount, uint64_t(pShape.vN)});
    if (vBegin >= vEnd) {
      continue;
    }
    const uint64_t vCount  = vEnd - vBegin;
    const uint64_t vOutput = vBegin - pTarget.vFirst;  // of the target
    const T *      vStripAt = pBuffer + vStrip * pShape.vPaddedM * pWidth + (vBegin - vColumn);  // its row 0
    uint64_t       vVectorFrom = pEndRow;  // the rows from here to pEndRow went by vectors
    if constexpr (std::is_same_v<T, uint16_t>) {
      const uint64_t vFrom = std::max<uint64_t>(pFirstRow, 3);
      if (vFrom < pEndRow &&
          fXdnaConvRowsBf16(vStripAt, pWidth, vFrom, pEndRow, pTarget.vConvWeights + vOutput, pTarget.vConvStride,
                             pTarget.vData + vOutput, pTarget.vStride, uint32_t(vCount), true)) {
        vVectorFrom = vFrom;
      }
    }
    for (uint64_t vRow = pFirstRow; vRow < vVectorFrom; vRow++) {
      const float * aTaps[4];
      for (uint64_t k = 0; k < 4; k++) {
        // tap k of row t is the row t + k - 3: before the first row, a row of the old states
        if (vRow + k < 3) {
          aTaps[k] = pTarget.vConvHead + (vRow + k) * pTarget.vConvStride + vOutput;
          continue;
        }
        const T * vFrom = vStripAt + (vRow + k - 3) * pWidth;
        for (uint64_t i = 0; i < vCount; i++) {
          aRows[k][i] = fSumOf(vFrom + i);
        }
        aTaps[k] = aRows[k];
      }
      fXdnaConvToken(aTaps, pTarget.vConvWeights + vOutput, pTarget.vConvStride,
                      pTarget.vData + vRow * pTarget.vStride + vOutput, uint32_t(vCount), true);
    }
    // the last 3 rows as they are, for the copy of the new states into the cache
    for (uint64_t vRow = std::max<uint64_t>(pFirstRow, uint64_t(pShape.vM) - 3); vRow < pEndRow; vRow++) {
      float *   vTail = pTarget.vConvTail + (vRow + 3 - uint64_t(pShape.vM)) * pTarget.vConvStride + vOutput;
      const T * vFrom = vStripAt + vRow * pWidth;
      for (uint64_t i = 0; i < vCount; i++) {
        vTail[i] = fSumOf(vFrom + i);
      }
    }
  }
}

// every row of every target from its strips of C, with the same arithmetic as fUnpackC, so the same floats: a strip
// is read in order, and every row of it goes to a run of its row of a target
template <typename T>
void fUnpackStrips(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const T * pBuffer,
                   const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers, bool pAdd) {
  const uint64_t vWidth   = pOverlay.vTileN;
  bool           vAlign16 = false;
  for (uint32_t t = 0; !pAdd && t < pTargets; t++) {
    vAlign16 = vAlign16 || aTargets[t].vGdnPrepack != nullptr;
  }
  fForEachStrip(pShape, vWidth, pWorkers, vAlign16, [&](uint64_t pFirstRow, uint64_t pEndRow, uint64_t pFirstStrip,
                                                        uint64_t pEndStrip) {
    for (uint32_t t = 0; t < pTargets; t++) {
      const XdnaGemmTarget & vTarget = aTargets[t];
      if (vTarget.vConvWeights != nullptr && !pAdd && vWidth <= cConvMaxWidth) {
        fConvStrips(pShape, pBuffer, vWidth, vTarget, pFirstRow, pEndRow, pFirstStrip, pEndStrip);
        continue;
      }
      for (uint64_t vRow = pFirstRow; vRow < pEndRow; vRow++) {
        for (uint64_t vStrip = pFirstStrip; vStrip < pEndStrip; vStrip++) {
          const uint64_t vColumn = vStrip * vWidth;
          const uint64_t vBegin  = std::max<uint64_t>(vColumn, vTarget.vFirst);
          const uint64_t vEnd =
              std::min<uint64_t>({vColumn + vWidth, vTarget.vFirst + vTarget.vCount, uint64_t(pShape.vN)});
          if (vBegin >= vEnd) {
            continue;
          }
          const T *      vFrom = pBuffer + (vStrip * pShape.vPaddedM + vRow) * vWidth + (vBegin - vColumn);
          const uint64_t vAt   = vRow * vTarget.vStride + (vBegin - vTarget.vFirst);
          if (vTarget.vAlso != nullptr) {
            const float * vBase =
                !pAdd ? nullptr : vTarget.vBase != nullptr ? vTarget.vBase + vAt : vTarget.vData + vAt;
            fResidualRow(vTarget.vData + vAt, vBase, vFrom, vTarget.vAlso + vAt, vEnd - vBegin);
          } else if (pAdd) {
            fAddRow(vTarget.vData + vAt, vFrom, vEnd - vBegin);
          } else {
            fWriteRow(vTarget.vData + vAt, vFrom, vEnd - vBegin);
          }
        }
      }
    }
  });
}

// T is the type of a sum of C as the overlay gives it back: float, uint16_t for bfloat16, or uint8_t for block
// floating point, where every 8 x 8 sums are 8 blocks of 9 bytes, one for the 8 rows of every column.
template <typename T>
void fUnpackC(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const T * pBuffer,
              const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers, bool pAdd) {
  if constexpr (!std::is_same_v<T, uint8_t>) {
    if (pOverlay.vRowsC == 1) {
      fUnpackStrips(pOverlay, pShape, pBuffer, aTargets, pTargets, pWorkers, pAdd);
      return;
    }
  }
  // [columns of cores][row blocks][turns][rows of cores][tile], the tile as the kernel keeps it:
  // [rows / 16][columns / 16][2 groups of rows][2 groups of columns][8 rows][8 columns]
  // (in block floating point, [8 columns][8 rows] in every 8 x 8)
  // What a column of cores gives in a row block is a stretch of the buffer of its own, and so are the values of
  // C it holds: with A by columns, the rows of a band; with A by rows, the columns of its part of B.
  // Split, the last row of cores gives its tiles in stretches of their own, [columns of cores][row blocks][turns]
  // [tile], after all the others, which then hold the tiles of the other rows of cores.
  const bool     vByColumns = pOverlay.vLayout == XdnaGemmLayout::cColumns;
  const uint32_t vStreamsA  = fStreamsA(pOverlay);
  const uint32_t vStreamsB  = fStreamsB(pOverlay);
  const uint32_t vBlocks    = pShape.vPaddedM / (pOverlay.vTileM * vStreamsA);
  const uint32_t vTurns     = pShape.vPaddedN / (pOverlay.vTileN * vStreamsB);
  const uint32_t vRowsJoined = pOverlay.vCoreRows - (pOverlay.vSplit == 2 ? 1 : 0);
  constexpr bool     cBlocks     = std::is_same_v<T, uint8_t>;
  const bool         vSteps3     = fSteps3(pOverlay);
  constexpr uint64_t cBlockItems = cBlocks ? cBlock * cBlockBytes : cBlock * cBlock;  // items of an 8 x 8
  const uint64_t     vTileSums   = uint64_t(pOverlay.vTileM) * pOverlay.vTileN / (cBlock * cBlock) * cBlockItems;
  const uint64_t     vDirectStart =
      uint64_t(pOverlay.vCoreColumns) * vBlocks * vTurns * vRowsJoined * vTileSums;
  // the part of a column of cores in a row block, in pieces of consecutive turns, each to outputs of its own
  const bool     vShared   = uint64_t(pShape.vM) * pShape.vN >= cSharedValues;
  const uint32_t vParts    = pOverlay.vCoreColumns * vBlocks;
  const uint32_t vPieces   = vShared ? fPieces(pWorkers, vParts, vTurns) : 1;
  const uint32_t vPerPiece = (vTurns + vPieces - 1) / vPieces;
  auto fPart = [&](uint32_t pPiece) {
    const uint32_t vPart       = pPiece / vPieces;
    const uint32_t vCoreColumn = vPart / vBlocks;
    const uint32_t vBlock      = vPart % vBlocks;
    const uint32_t vFirstTurn  = (pPiece % vPieces) * vPerPiece;
    const uint32_t vEndTurn    = std::min(vTurns, vFirstTurn + vPerPiece);
    float          aDecoded[cBlock * cBlock];  // in block floating point, the 8 x 8 sums, 8 rows of 8
    for (uint32_t vTurn = vFirstTurn; vTurn < vEndTurn; vTurn++) {
      for (uint32_t vCoreRow = 0; vCoreRow < pOverlay.vCoreRows; vCoreRow++) {
        // the core of this row and this column holds one band of A and one tile column of B
        const uint32_t vStreamA    = vByColumns ? vCoreColumn : vCoreRow;
        const uint32_t vStreamB    = vByColumns ? vCoreRow : vCoreColumn;
        const uint64_t vTileRow    = (uint64_t(vBlock) * vStreamsA + vStreamA) * pOverlay.vTileM;
        const uint64_t vTileColumn = (uint64_t(vTurn) * vStreamsB + vStreamB) * pOverlay.vTileN;
        if (vTileRow >= pShape.vM || vTileColumn >= pShape.vN) {
          continue;  // the whole tile is outside C
        }
        const T * vAt = vCoreRow < vRowsJoined ?
                            pBuffer + ((uint64_t(vPart) * vTurns + vTurn) * vRowsJoined + vCoreRow) * vTileSums :
                            pBuffer + vDirectStart + (uint64_t(vPart) * vTurns + vTurn) * vTileSums;
        const bool vInside = vTileRow + pOverlay.vTileM <= pShape.vM && vTileColumn + pOverlay.vTileN <= pShape.vN;
        fForEachBlock(vSteps3, pOverlay.vTileM / cBlock, pOverlay.vTileN / cBlock,
                      [&](uint32_t pRowGroup, uint32_t pColumnGroup) {
                const uint64_t vRow    = vTileRow + uint64_t(pRowGroup) * cBlock;
                const uint64_t vColumn = vTileColumn + uint64_t(pColumnGroup) * cBlock;
                const XdnaGemmTarget * vTarget = fTargetOf(aTargets, pTargets, vColumn);
                float * vTo = vTarget != nullptr ?
                                  vTarget->vData + vRow * vTarget->vStride + (vColumn - vTarget->vFirst) :
                                  nullptr;
                // the 8 x 8 sums, from pFrom, 8 rows of 8, each row to its place
                auto fWrite = [&](const auto * pFrom) {
                  if (vInside && vTarget != nullptr && vTarget->vAlso != nullptr &&
                      vColumn + cBlock - vTarget->vFirst <= vTarget->vCount) {
                    // with a residual sum (llama.cpp-xdna)
                    const uint64_t vOffset = uint64_t(vTo - vTarget->vData);
                    for (uint32_t vLine = 0; vLine < cBlock; vLine++) {
                      const uint64_t vAt   = vOffset + uint64_t(vLine) * vTarget->vStride;
                      const float *  vBase = !pAdd ? nullptr :
                                             vTarget->vBase != nullptr ? vTarget->vBase + vAt : vTarget->vData + vAt;
                      fResidualEight(vTarget->vData + vAt, vBase, pFrom + vLine * cBlock, vTarget->vAlso + vAt);
                    }
                    return;
                  }
                  if (vInside && vTarget != nullptr && vColumn + cBlock - vTarget->vFirst <= vTarget->vCount &&
                      reinterpret_cast<uintptr_t>(vTo) % 16 == 0 && vTarget->vStride % 4 == 0) {
                    for (uint32_t vLine = 0; vLine < cBlock; vLine++) {
                      if (pAdd) {
                        fAddEight(vTo + uint64_t(vLine) * vTarget->vStride, pFrom + vLine * cBlock);
                      } else {
                        fWriteEight(vTo + uint64_t(vLine) * vTarget->vStride, pFrom + vLine * cBlock);
                      }
                    }
                    return;
                  }
                  // at an edge of C or of a target: value by value
                  for (uint32_t i = 0; i < cBlock && vColumn + i < pShape.vN; i++) {
                    const XdnaGemmTarget * vHere = fTargetOf(aTargets, pTargets, vColumn + i);
                    if (vHere == nullptr) {
                      continue;
                    }
                    for (uint32_t vLine = 0; vLine < cBlock && vRow + vLine < pShape.vM; vLine++) {
                      const uint64_t vAt    = (vRow + vLine) * vHere->vStride + (vColumn + i - vHere->vFirst);
                      float &        vValue = vHere->vData[vAt];
                      if (vHere->vAlso != nullptr) {
                        // with a residual sum (llama.cpp-xdna)
                        float vSum = fSumOf(pFrom + vLine * cBlock + i);
                        if (pAdd) {
                          vSum = (vHere->vBase != nullptr ? vHere->vBase[vAt] : vValue) + vSum;
                        }
                        vValue = vSum + vHere->vAlso[vAt];
                      } else if (pAdd) {
                        vValue += fSumOf(pFrom + vLine * cBlock + i);
                      } else {
                        vValue = fSumOf(pFrom + vLine * cBlock + i);
                      }
                    }
                  }
                };
                if constexpr (cBlocks) {
                  // 8 blocks, one for the 8 rows of every column
                  for (uint32_t j = 0; j < cBlock; j++) {
                    float aColumn[cBlock];
                    fXdnaBfp16Decode(vAt + j * cBlockBytes, aColumn);
                    for (uint32_t r = 0; r < cBlock; r++) {
                      aDecoded[r * cBlock + j] = aColumn[r];
                    }
                  }
                  fWrite(static_cast<const float *>(aDecoded));
                } else {
                  fWrite(vAt);
                }
                vAt += cBlockItems;
        });
      }
    }
  };
  fForEach(pWorkers, vShared, vParts * vPieces,
           [&](uint32_t pIndex) {
             fPart(pIndex);
#if defined(__x86_64__)
             // what went past the cache is in memory before anyone reads it
             _mm_sfence();
#endif
           });
}

// The C of two multiplications of the same shape, the gate and the up of a feed-forward layer, read together: their
// buffers have the same layout, so every 8 x 8 of one is at the same place in the other. silu(gate) * up of every
// value goes into the target (see fXdnaGemmUnpackSwiglu).
// llama.cpp-xdna: the same for C as strips (XdnaOverlay::vRowsC): every run of a row of the target from the same
// run of the gate and of the up, made floats for fXdnaSwiglu, which picks the vector or the scalar arithmetic of
// every value by its column, so the runs give the same floats as the whole row
template <typename T>
void fUnpackSwigluStrips(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const T * pGate, const T * pUp,
                         const XdnaGemmTarget & pTarget, XdnaWorkers * pWorkers, uint64_t pRowFirst,
                         uint64_t pRowLength) {
  const uint64_t vWidth = pOverlay.vTileN;
  fForEachStrip(pShape, vWidth, pWorkers, false, [&](uint64_t pFirstRow, uint64_t pEndRow, uint64_t pFirstStrip,
                                                     uint64_t pEndStrip) {
    constexpr uint32_t cRun = 256;
    float              aGate[cRun];
    float              aUp[cRun];
    for (uint64_t vRow = pFirstRow; vRow < pEndRow; vRow++) {
      for (uint64_t vStrip = pFirstStrip; vStrip < pEndStrip; vStrip++) {
        const uint64_t vColumn = vStrip * vWidth;
        const uint64_t vBegin  = std::max<uint64_t>(vColumn, pTarget.vFirst);
        const uint64_t vEnd =
            std::min<uint64_t>({vColumn + vWidth, pTarget.vFirst + pTarget.vCount, uint64_t(pShape.vN)});
        if (vBegin >= vEnd || vEnd - vBegin > cRun) {
          continue;  // the second never: a strip is the width of a tile
        }
        const uint32_t vCount = uint32_t(vEnd - vBegin);
        const uint64_t vAt    = (vStrip * pShape.vPaddedM + vRow) * vWidth + (vBegin - vColumn);
        float *        vTo    = pTarget.vData + vRow * pTarget.vStride + (vBegin - pTarget.vFirst);
        if constexpr (std::is_same_v<T, float>) {
          fXdnaSwiglu(pGate + vAt, pUp + vAt, vTo, vCount, pRowFirst + vBegin, pRowLength);
        } else {
          for (uint32_t i = 0; i < vCount; i++) {
            aGate[i] = fSumOf(pGate + vAt + i);
            aUp[i]   = fSumOf(pUp + vAt + i);
          }
          fXdnaSwiglu(aGate, aUp, vTo, vCount, pRowFirst + vBegin, pRowLength);
        }
      }
    }
  });
}

template <typename T>
void fUnpackSwiglu(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const T * pGate, const T * pUp,
                   const XdnaGemmTarget & pTarget, XdnaWorkers * pWorkers, uint64_t pRowFirst, uint64_t pRowLength) {
  if (pOverlay.vRowsC == 1) {
    fUnpackSwigluStrips(pOverlay, pShape, pGate, pUp, pTarget, pWorkers, pRowFirst, pRowLength);
    return;
  }
  // the layout of fUnpackC
  const bool     vByColumns  = pOverlay.vLayout == XdnaGemmLayout::cColumns;
  const uint32_t vStreamsA   = fStreamsA(pOverlay);
  const uint32_t vStreamsB   = fStreamsB(pOverlay);
  const uint32_t vBlocks     = pShape.vPaddedM / (pOverlay.vTileM * vStreamsA);
  const uint32_t vTurns      = pShape.vPaddedN / (pOverlay.vTileN * vStreamsB);
  const uint32_t vRowsJoined = pOverlay.vCoreRows - (pOverlay.vSplit == 2 ? 1 : 0);
  const bool     vSteps3     = fSteps3(pOverlay);
  constexpr uint64_t cBlockItems  = cBlock * cBlock;
  const uint64_t     vTileSums    = uint64_t(pOverlay.vTileM) * pOverlay.vTileN / (cBlock * cBlock) * cBlockItems;
  const uint64_t     vDirectStart = uint64_t(pOverlay.vCoreColumns) * vBlocks * vTurns * vRowsJoined * vTileSums;
  const bool         vShared      = uint64_t(pShape.vM) * pShape.vN >= cSharedValues;
  const uint32_t     vParts       = pOverlay.vCoreColumns * vBlocks;
  const uint32_t     vPieces      = vShared ? fPieces(pWorkers, vParts, vTurns) : 1;
  const uint32_t     vPerPiece    = (vTurns + vPieces - 1) / vPieces;
  auto fPart = [&](uint32_t pPiece) {
    const uint32_t vPart       = pPiece / vPieces;
    const uint32_t vCoreColumn = vPart / vBlocks;
    const uint32_t vBlock      = vPart % vBlocks;
    const uint32_t vFirstTurn  = (pPiece % vPieces) * vPerPiece;
    const uint32_t vEndTurn    = std::min(vTurns, vFirstTurn + vPerPiece);
    float          aGate[cBlock * cBlock];
    float          aUp[cBlock * cBlock];
    for (uint32_t vTurn = vFirstTurn; vTurn < vEndTurn; vTurn++) {
      for (uint32_t vCoreRow = 0; vCoreRow < pOverlay.vCoreRows; vCoreRow++) {
        const uint32_t vStreamA    = vByColumns ? vCoreColumn : vCoreRow;
        const uint32_t vStreamB    = vByColumns ? vCoreRow : vCoreColumn;
        const uint64_t vTileRow    = (uint64_t(vBlock) * vStreamsA + vStreamA) * pOverlay.vTileM;
        const uint64_t vTileColumn = (uint64_t(vTurn) * vStreamsB + vStreamB) * pOverlay.vTileN;
        if (vTileRow >= pShape.vM || vTileColumn >= pShape.vN) {
          continue;
        }
        const uint64_t vOffset = vCoreRow < vRowsJoined ?
                                     ((uint64_t(vPart) * vTurns + vTurn) * vRowsJoined + vCoreRow) * vTileSums :
                                     vDirectStart + (uint64_t(vPart) * vTurns + vTurn) * vTileSums;
        const T * vAtGate = pGate + vOffset;
        const T * vAtUp   = pUp + vOffset;
        fForEachBlock(vSteps3, pOverlay.vTileM / cBlock, pOverlay.vTileN / cBlock,
                      [&](uint32_t pRowGroup, uint32_t pColumnGroup) {
                const uint64_t vRow    = vTileRow + uint64_t(pRowGroup) * cBlock;
                const uint64_t vColumn = vTileColumn + uint64_t(pColumnGroup) * cBlock;
                if constexpr (std::is_same_v<T, uint16_t>) {
                  // a whole 8 x 8 inside C and inside the target, where the CPU backend uses vectors: at once
                  if (vRow + cBlock <= pShape.vM && vColumn + cBlock <= pShape.vN && vColumn >= pTarget.vFirst &&
                      vColumn + cBlock <= pTarget.vFirst + pTarget.vCount &&
                      fXdnaSwigluBlockFits(pRowFirst + vColumn, pRowLength)) {
                    fXdnaSwigluBlock(vAtGate, vAtUp, pTarget.vData + vRow * pTarget.vStride + (vColumn - pTarget.vFirst),
                                      pTarget.vStride);
                    vAtGate += cBlockItems;
                    vAtUp += cBlockItems;
                    return;
                  }
                }
                if (vRow < pShape.vM && vColumn < pShape.vN && vColumn >= pTarget.vFirst &&
                    vColumn - pTarget.vFirst < pTarget.vCount) {
                  for (uint32_t i = 0; i < cBlock * cBlock; i++) {
                    aGate[i] = fSumOf(vAtGate + i);
                    aUp[i]   = fSumOf(vAtUp + i);
                  }
                  const uint32_t vCount = uint32_t(std::min<uint64_t>(
                      {uint64_t(cBlock), pShape.vN - vColumn, pTarget.vFirst + pTarget.vCount - vColumn}));
                  for (uint32_t vLine = 0; vLine < cBlock && vRow + vLine < pShape.vM; vLine++) {
                    float * vTo = pTarget.vData + (vRow + vLine) * pTarget.vStride + (vColumn - pTarget.vFirst);
                    fXdnaSwiglu(aGate + vLine * cBlock, aUp + vLine * cBlock, vTo, vCount, pRowFirst + vColumn,
                                 pRowLength);
                  }
                }
                vAtGate += cBlockItems;
                vAtUp += cBlockItems;
        });
      }
    }
  };
  fForEach(pWorkers, vShared, vParts * vPieces, fPart);
}

// llama.cpp-xdna: whether a piece of the rows of a SwiGLU is whole tiles of K for pM rows (see fXdnaGemmPackSwiglu),
// and its shape as A
bool fSwigluPieceShape(const XdnaOverlay & pOverlay, uint32_t pM, const XdnaSwigluPiece & pPiece,
                       XdnaGemmShape & pShapeA) {
  std::string vIgnored;
  return pPiece.vCount > 0 && pPiece.vCount % pOverlay.vTileK == 0 && pPiece.vFirst % pOverlay.vTileK == 0 &&
         fXdnaGemmShape(pOverlay, pM, pPiece.vCount, 1, pShapeA, vIgnored) && pShapeA.vPaddedK == pPiece.vCount;
}

// llama.cpp-xdna: the tiles of K of a piece inside the columns pRowFirst to pRowFirst + pOutputs, pFirstTile to
// pEndTile; false when one of them is only partly inside
bool fSwigluTiles(const XdnaOverlay & pOverlay, const XdnaSwigluPiece & pPiece, uint64_t pRowFirst,
                  uint64_t pOutputs, uint32_t & pFirstTile, uint32_t & pEndTile) {
  const uint64_t vTileK = pOverlay.vTileK;
  pFirstTile            = 0;
  pEndTile              = 0;
  for (uint64_t t = 0; t < pPiece.vCount / vTileK; t++) {
    const uint64_t vBegin   = pPiece.vFirst + t * vTileK;
    const uint64_t vEnd     = vBegin + vTileK;
    const bool     vInside  = vBegin >= pRowFirst && vEnd <= pRowFirst + pOutputs;
    const bool     vOutside = vEnd <= pRowFirst || vBegin >= pRowFirst + pOutputs;
    if (!vInside && !vOutside) {
      return false;
    }
    if (vInside) {
      if (pEndTile == pFirstTile) {
        pFirstTile = uint32_t(t);
      }
      pEndTile = uint32_t(t + 1);
    }
  }
  return true;
}

// llama.cpp-xdna: see fXdnaGemmPackSwiglu. Every tile of A of a piece inside the columns of this C is a task: its
// rows, from the strips of the gate and the up in runs of a strip made floats, into fXdnaSwiglu by rows of a tile of K
// (which picks the vector or the scalar arithmetic of every value by its column, as fUnpackSwigluStrips relies on), and
// the tile of floats into fPackTile as fXdnaGemmPackA calls it, rows past pShape.vM as zeros.
template <typename T>
bool fPackSwiglu(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const T * pGate, const T * pUp,
                 uint64_t pRowFirst, uint64_t pRowLength, const XdnaSwigluPiece * aPieces, uint32_t pPieces,
                 XdnaRounding pRounding, XdnaWorkers * pWorkers) {
  const uint64_t       vWidth     = pOverlay.vTileN;
  const uint32_t       vTileM     = pOverlay.vTileM;
  const uint32_t       vTileK     = pOverlay.vTileK;
  const uint64_t       vTileBytes = uint64_t(vTileM) * vTileK / cBlock * cBlockBytes;
  const XdnaTileOrder vOrder     = fSteps3(pOverlay) ? XdnaTileOrder::cRows : XdnaTileOrder::cPairs;
  struct XdnaTask {
    uint8_t * vAt;      // where the tile goes
    uint64_t  vColumn;  // its first column in this C
    uint64_t  vRow;     // its first row
  };
  std::vector<XdnaTask> aTasks;
  for (uint32_t p = 0; p < pPieces; p++) {
    XdnaGemmShape vShapeA;
    uint32_t       vFirstTile;
    uint32_t       vEndTile;
    if (aPieces[p].vBuffer == nullptr || !fSwigluPieceShape(pOverlay, pShape.vM, aPieces[p], vShapeA) ||
        !fSwigluTiles(pOverlay, aPieces[p], pRowFirst, pShape.vN, vFirstTile, vEndTile)) {
      return false;
    }
    const uint64_t vBandBytes = uint64_t(vShapeA.vPaddedK / vTileK) * vTileBytes;
    for (uint32_t vBand = 0; vBand < vShapeA.vPaddedM / vTileM; vBand++) {
      for (uint32_t t = vFirstTile; t < vEndTile; t++) {
        aTasks.push_back({aPieces[p].vBuffer + vBand * vBandBytes + t * vTileBytes,
                          aPieces[p].vFirst + uint64_t(t) * vTileK - pRowFirst, uint64_t(vBand) * vTileM});
      }
    }
  }
  std::atomic<bool> vFinite(true);
  fForEach(pWorkers, uint64_t(pShape.vM) * pShape.vN >= cSharedValues, uint32_t(aTasks.size()), [&](uint32_t pTask) {
    const XdnaTask &               vTask = aTasks[pTask];
    thread_local std::vector<float> aTile;
    thread_local std::vector<float> aGate;
    thread_local std::vector<float> aUp;
    aTile.resize(size_t(vTileM) * vTileK);
    aGate.resize(vTileK);
    aUp.resize(vTileK);
    const uint32_t vRows =
        vTask.vRow >= pShape.vM ? 0 : uint32_t(std::min<uint64_t>(vTileM, uint64_t(pShape.vM) - vTask.vRow));
    for (uint32_t r = 0; r < vRows; r++) {
      const uint64_t vRow = vTask.vRow + r;
      for (uint32_t j = 0; j < vTileK;) {
        // a run inside one strip
        const uint64_t vColumn = vTask.vColumn + j;
        const uint32_t vRun    = uint32_t(std::min<uint64_t>(vWidth - vColumn % vWidth, vTileK - j));
        const uint64_t vAt     = ((vColumn / vWidth) * pShape.vPaddedM + vRow) * vWidth + vColumn % vWidth;
#if defined(__x86_64__)
        if constexpr (std::is_same_v<T, uint16_t>) {
          if (cHasGemmAvx2) {
            fLoadBf16PairRunAvx2(pGate + vAt, pUp + vAt, aGate.data() + j, aUp.data() + j, vRun);
            j += vRun;
            continue;
          }
        }
#endif
        for (uint32_t i = 0; i < vRun; i++) {
          aGate[j + i] = fSumOf(pGate + vAt + i);
          aUp[j + i]   = fSumOf(pUp + vAt + i);
        }
        j += vRun;
      }
      fXdnaSwiglu(aGate.data(), aUp.data(), aTile.data() + size_t(r) * vTileK, vTileK, pRowFirst + vTask.vColumn,
                   pRowLength);
    }
    if (fPackTile(aTile.data(), vRows, vTileK, vTileK, 0, 0, vTileM, vTileK, vTask.vAt, pRounding, vOrder) ==
        nullptr) {
      vFinite = false;
    }
  });
  return vFinite;
}

}  // namespace

bool fXdnaGemmUnpackSwiglu(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pGate,
                            const void * pUp, const XdnaGemmTarget & pTarget, XdnaWorkers * pWorkers,
                            uint64_t pRowFirst, uint64_t pRowLength) {
  const uint64_t vRowLength = pRowLength != 0 ? pRowLength : pShape.vN;
  if (pOverlay.vOutput == XdnaGemmOutput::cBfloat16) {
    fUnpackSwiglu(pOverlay, pShape, static_cast<const uint16_t *>(pGate), static_cast<const uint16_t *>(pUp), pTarget,
                  pWorkers, pRowFirst, vRowLength);
    return true;
  }
  if (pOverlay.vOutput == XdnaGemmOutput::cFloat) {
    fUnpackSwiglu(pOverlay, pShape, static_cast<const float *>(pGate), static_cast<const float *>(pUp), pTarget,
                  pWorkers, pRowFirst, vRowLength);
    return true;
  }
  return false;
}

bool fXdnaGemmPackSwiglu(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pGate,
                          const void * pUp, uint64_t pRowFirst, uint64_t pRowLength, const XdnaSwigluPiece * aPieces,
                          uint32_t pPieces, XdnaRounding pRounding, XdnaWorkers * pWorkers) {
  const uint64_t vRowLength = pRowLength != 0 ? pRowLength : pShape.vN;
  if (pOverlay.vRowsC != 1) {
    return false;
  }
  if (pOverlay.vOutput == XdnaGemmOutput::cBfloat16) {
    return fPackSwiglu(pOverlay, pShape, static_cast<const uint16_t *>(pGate), static_cast<const uint16_t *>(pUp),
                       pRowFirst, vRowLength, aPieces, pPieces, pRounding, pWorkers);
  }
  if (pOverlay.vOutput == XdnaGemmOutput::cFloat) {
    return fPackSwiglu(pOverlay, pShape, static_cast<const float *>(pGate), static_cast<const float *>(pUp),
                       pRowFirst, vRowLength, aPieces, pPieces, pRounding, pWorkers);
  }
  return false;
}

bool fXdnaGemmPackGatedNorm(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const float * pO,
                             uint64_t pStrideO, const float * pZ, uint64_t pStrideZ, const float * pW, uint32_t pHead,
                             float pEps, uint64_t pFirst, uint8_t * pBuffer, XdnaRounding pRounding,
                             XdnaWorkers * pWorkers, bool pSigmoid) {
  const uint32_t vTileM = pOverlay.vTileM;
  const uint32_t vTileK = pOverlay.vTileK;
  if (pHead == 0 || vTileK % pHead != 0 || pFirst % vTileK != 0 || pShape.vK % vTileK != 0 ||
      pShape.vPaddedK != pShape.vK) {
    return false;
  }
  const uint64_t       vTileBytes = uint64_t(vTileM) * vTileK / cBlock * cBlockBytes;
  const uint32_t       vTiles     = pShape.vPaddedK / vTileK;
  const uint64_t       vBandBytes = uint64_t(vTiles) * vTileBytes;
  const XdnaTileOrder vOrder     = fSteps3(pOverlay) ? XdnaTileOrder::cRows : XdnaTileOrder::cPairs;
  std::atomic<bool>    vFinite(true);
  fForEach(pWorkers, uint64_t(pShape.vM) * pShape.vK >= cSharedValues, (pShape.vPaddedM / vTileM) * vTiles,
           [&](uint32_t pTask) {
    const uint32_t                  vBand = pTask / vTiles;
    const uint32_t                  vTile = pTask % vTiles;
    thread_local std::vector<float> aTile;
    thread_local std::vector<float> aNormed;
    thread_local std::vector<float> aScales;
    aTile.resize(size_t(vTileM) * vTileK);
    aNormed.resize(pHead);
    aScales.resize(vTileM);
    const uint64_t vRow0 = uint64_t(vBand) * vTileM;
    const uint32_t vRows =
        vRow0 >= pShape.vM ? 0 : uint32_t(std::min<uint64_t>(vTileM, uint64_t(pShape.vM) - vRow0));
    for (uint32_t vHead = 0; vRows > 0 && vHead < vTileK / pHead; vHead++) {
      const uint64_t vColumn = pFirst + uint64_t(vTile) * vTileK + uint64_t(vHead) * pHead;  // in a row
      fXdnaRmsScales(pO + vRow0 * pStrideO + vColumn, vRows, pHead, pStrideO, pEps, aScales.data());
      for (uint32_t r = 0; r < vRows; r++) {
        const float * vX     = pO + (vRow0 + r) * pStrideO + vColumn;
        const float   vScale = aScales[r];
        for (uint32_t i = 0; i < pHead; i++) {
          aNormed[i] = vX[i] * vScale * pW[i];
        }
        float * vDst = aTile.data() + size_t(r) * vTileK + size_t(vHead) * pHead;
        if (pSigmoid) {
          fXdnaSigmoidMul(pZ + (vRow0 + r) * pStrideZ + vColumn, aNormed.data(), vDst, pHead, 0, pHead);
        } else {
          fXdnaSwiglu(pZ + (vRow0 + r) * pStrideZ + vColumn, aNormed.data(), vDst, pHead, 0, pHead);
        }
      }
    }
    if (fPackTile(aTile.data(), vRows, vTileK, vTileK, 0, 0, vTileM, vTileK, pBuffer + vBand * vBandBytes +
                  uint64_t(vTile) * vTileBytes, pRounding, vOrder) == nullptr) {
      vFinite = false;
    }
  });
  return vFinite;
}

bool fXdnaGemmSwigluFits(const XdnaOverlay & pOverlay, uint32_t pM, uint64_t pRowFirst, uint64_t pOutputs,
                          const XdnaSwigluPiece * aPieces, uint32_t pPieces) {
  if (pOverlay.vRowsC != 1 ||
      (pOverlay.vOutput != XdnaGemmOutput::cBfloat16 && pOverlay.vOutput != XdnaGemmOutput::cFloat)) {
    return false;
  }
  for (uint32_t p = 0; p < pPieces; p++) {
    XdnaGemmShape vShapeA;
    uint32_t       vFirstTile;
    uint32_t       vEndTile;
    if (!fSwigluPieceShape(pOverlay, pM, aPieces[p], vShapeA) ||
        !fSwigluTiles(pOverlay, aPieces[p], pRowFirst, pOutputs, vFirstTile, vEndTile)) {
      return false;
    }
  }
  return true;
}

void fXdnaGemmUnpackC(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pBuffer,
                       const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers, bool pAdd) {
  if (pOverlay.vOutput == XdnaGemmOutput::cBfloat16) {
    fUnpackC(pOverlay, pShape, static_cast<const uint16_t *>(pBuffer), aTargets, pTargets, pWorkers, pAdd);
  } else if (pOverlay.vOutput == XdnaGemmOutput::cBfp16) {
    fUnpackC(pOverlay, pShape, static_cast<const uint8_t *>(pBuffer), aTargets, pTargets, pWorkers, pAdd);
  } else {
    fUnpackC(pOverlay, pShape, static_cast<const float *>(pBuffer), aTargets, pTargets, pWorkers, pAdd);
  }
}

void fXdnaGemmUnpackC(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pBuffer,
                       float * pC, XdnaWorkers * pWorkers) {
  XdnaGemmTarget vWhole;
  vWhole.vData   = pC;
  vWhole.vStride = pShape.vN;
  vWhole.vFirst  = 0;
  vWhole.vCount  = pShape.vN;
  fXdnaGemmUnpackC(pOverlay, pShape, pBuffer, &vWhole, 1, pWorkers);
}

// llama.cpp-xdna: the attention of a batch (see XdnaAttnRows in xdna-gemm.h)

void fXdnaAttnSplit(const float * pValues, uint32_t pCount, float * pHigh, float * pLow) {
  // all the blocks of the row at once with the fast encoder (the same bytes as fXdnaBfp16Encode to the nearest)
  thread_local std::vector<uint8_t> aBlocks;
  const uint32_t                    vBlocks = pCount / cBlock;
  aBlocks.resize(size_t(vBlocks) * cBlockBytes);
  if (!fXdnaBfp16EncodeNearest(pValues, cBlock, vBlocks, aBlocks.data())) {
    // a value that is not finite: the row goes whole into the high part, and the encoder of the tile says so
    std::memcpy(pHigh, pValues, size_t(pCount) * sizeof(float));
    std::memset(pLow, 0, size_t(pCount) * sizeof(float));
    return;
  }
  for (uint32_t b = 0; b < vBlocks; b++) {
    fXdnaBfp16Decode(aBlocks.data() + size_t(b) * cBlockBytes, pHigh + size_t(b) * cBlock);
  }
  for (uint32_t i = 0; i < vBlocks * cBlock; i++) {
    pLow[i] = pValues[i] - pHigh[i];
  }
}

bool fXdnaGemmPackAttnQ(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const XdnaAttnRows & pRows,
                         uint8_t * pBuffer, XdnaRounding pRounding, XdnaWorkers * pWorkers) {
  const uint32_t vTileM = pOverlay.vTileM;
  const uint32_t vTileK = pOverlay.vTileK;
  if (pRows.vQ == nullptr || pRows.vHeads == 0 || uint64_t(pRows.vTokens) * pRows.vHeads != pShape.vM ||
      pRows.vDim == 0 || pShape.vK != (pRows.vSplit ? 3 * pRows.vDim : pRows.vDim) ||
      (pRows.vSplit && pRows.vDim % cBlock != 0)) {
    return false;
  }
  // [bands of vTileM rows][tiles along K][tile], as fXdnaGemmPackA writes A: every tile is a task, its rows gathered
  // from Q (times the scale) into a tile of floats and converted from there as fXdnaGemmPackA converts them
  const uint32_t       vBands     = pShape.vPaddedM / vTileM;
  const uint32_t       vTiles     = pShape.vPaddedK / vTileK;
  const uint64_t       vTileBytes = uint64_t(vTileM) * vTileK / cBlock * cBlockBytes;
  const uint64_t       vBandBytes = uint64_t(vTiles) * vTileBytes;
  const XdnaTileOrder vOrder     = fSteps3(pOverlay) ? XdnaTileOrder::cRows : XdnaTileOrder::cPairs;
  std::atomic<bool>    vFinite(true);
  fForEach(pWorkers, uint64_t(pShape.vM) * pShape.vK >= cSharedValues, vBands * vTiles, [&](uint32_t pTask) {
    const uint32_t                  vBand = pTask / vTiles;
    const uint32_t                  vTile = pTask % vTiles;
    thread_local std::vector<float> aTile;
    thread_local std::vector<float> aParts;
    aTile.resize(size_t(vTileM) * vTileK);
    aParts.resize(size_t(3) * pRows.vDim);
    const uint64_t vRow0    = uint64_t(vBand) * vTileM;
    const uint64_t vColumn0 = uint64_t(vTile) * vTileK;
    const uint32_t vRows =
        vRow0 >= pShape.vM ? 0 : uint32_t(std::min<uint64_t>(vTileM, uint64_t(pShape.vM) - vRow0));
    const uint32_t vColumns =
        vColumn0 >= pShape.vK ? 0 : uint32_t(std::min<uint64_t>(vTileK, uint64_t(pShape.vK) - vColumn0));
    for (uint32_t r = 0; r < vRows; r++) {
      const uint64_t vRow  = vRow0 + r;
      const float *  vFrom = pRows.vQ + (vRow / pRows.vHeads) * pRows.vTokenStride + (vRow % pRows.vHeads) * pRows.vHeadStride;
      float *        vTo   = aTile.data() + size_t(r) * vTileK;
      if (!pRows.vSplit) {
        for (uint32_t c = 0; c < vColumns; c++) {
          vTo[c] = vFrom[vColumn0 + c] * pRows.vScale;
        }
        continue;
      }
      // [high, high, low]: the scaled row, then its two parts
      float * vScaled = aParts.data();
      float * vHigh   = aParts.data() + pRows.vDim;
      float * vLow    = aParts.data() + 2 * size_t(pRows.vDim);
      for (uint32_t d = 0; d < pRows.vDim; d++) {
        vScaled[d] = vFrom[d] * pRows.vScale;
      }
      fXdnaAttnSplit(vScaled, pRows.vDim, vHigh, vLow);
      for (uint32_t c = 0; c < vColumns; c++) {
        const uint64_t vColumn = vColumn0 + c;
        const uint64_t vPart   = vColumn / pRows.vDim;
        vTo[c]                 = (vPart == 2 ? vLow : vHigh)[vColumn % pRows.vDim];
      }
    }
    if (fPackTile(aTile.data(), vRows, vColumns, vTileK, 0, 0, vTileM, vTileK,
                  pBuffer + vBand * vBandBytes + uint64_t(vTile) * vTileBytes, pRounding, vOrder) == nullptr) {
      vFinite = false;
    }
  });
  return vFinite;
}

bool fXdnaGemmPackBHalves(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const uint16_t * pData,
                           uint64_t pStrideN, uint64_t pStrideK, uint32_t pValidK, uint8_t * pBuffer,
                           XdnaRounding pRounding, XdnaWorkers * pWorkers, uint32_t pSplitDim) {
  if (pOverlay.vSplit != 1 || pData == nullptr || pValidK > pShape.vK ||
      (pSplitDim != 0 && (pStrideK != 1 || pSplitDim % cBlock != 0 || pShape.vK != 3 * pSplitDim ||
                          pValidK != 3 * pSplitDim))) {
    return false;
  }
  // the layout of fXdnaGemmPackB without a split: [streams of B][turns of the stream][tiles along K][tile], every
  // tile a task
  const uint32_t       vTileN       = pOverlay.vTileN;
  const uint32_t       vTileK       = pOverlay.vTileK;
  const uint32_t       vStreams     = fStreamsB(pOverlay);
  const uint32_t       vTurns       = pShape.vPaddedN / (vTileN * vStreams);
  const uint32_t       vTiles       = pShape.vPaddedK / vTileK;
  const uint64_t       vTileBytes   = uint64_t(vTileN) * vTileK / cBlock * cBlockBytes;
  const uint64_t       vColumnBytes = uint64_t(vTiles) * vTileBytes;
  const XdnaTileOrder vOrder       = fSteps3(pOverlay) ? XdnaTileOrder::cSets : XdnaTileOrder::cPairs;
  std::atomic<bool>    vFinite(true);
  fForEach(pWorkers, uint64_t(pShape.vN) * pShape.vK >= cSharedValues, vStreams * vTurns * vTiles, [&](uint32_t pTask) {
    const uint32_t                  vColumn = pTask / vTiles;  // as the index of fXdnaGemmPackB
    const uint32_t                  vTile   = pTask % vTiles;
    const uint32_t                  vStream = vColumn / vTurns;
    const uint32_t                  vTurn   = vColumn % vTurns;
    thread_local std::vector<float> aTile;
    thread_local std::vector<float> aLine;
    aTile.resize(size_t(vTileN) * vTileK);
    aLine.resize(std::max(vTileN, vTileK));
    const uint64_t vOutput0 = (uint64_t(vTurn) * vStreams + vStream) * vTileN;
    const uint64_t vInput0  = uint64_t(vTile) * vTileK;
    const uint32_t vRows =
        vOutput0 >= pShape.vN ? 0 : uint32_t(std::min<uint64_t>(vTileN, uint64_t(pShape.vN) - vOutput0));
    const uint32_t vColumns =
        vInput0 >= pValidK ? 0 : uint32_t(std::min<uint64_t>(vTileK, uint64_t(pValidK) - vInput0));
    if (vRows > 0 && vColumns > 0) {
      if (pSplitDim != 0) {
        // an output a run of halves, in two parts: [high, low, high] of its keys (see XdnaAttnRows)
        thread_local std::vector<float> aParts;
        aParts.resize(size_t(3) * pSplitDim);
        float * vValues = aParts.data();
        float * vHigh   = aParts.data() + pSplitDim;
        float * vLow    = aParts.data() + 2 * size_t(pSplitDim);
        for (uint32_t r = 0; r < vRows; r++) {
          fXdnaHalvesToFloats(pData + (vOutput0 + r) * pStrideN, vValues, pSplitDim);
          fXdnaAttnSplit(vValues, pSplitDim, vHigh, vLow);
          float * vTo = aTile.data() + size_t(r) * vTileK;
          for (uint32_t c = 0; c < vColumns; c++) {
            const uint64_t vInput = vInput0 + c;
            vTo[c]                = (vInput / pSplitDim == 1 ? vLow : vHigh)[vInput % pSplitDim];
          }
        }
      } else if (pStrideK == 1) {
        // an output a run of halves: K by keys
        for (uint32_t r = 0; r < vRows; r++) {
          fXdnaHalvesToFloats(pData + (vOutput0 + r) * pStrideN + vInput0, aTile.data() + size_t(r) * vTileK,
                               vColumns);
        }
      } else if (pStrideN == 1) {
        // an input a run of halves: V by keys, every key into a column of the tile
        for (uint32_t c = 0; c < vColumns; c++) {
          fXdnaHalvesToFloats(pData + (vInput0 + c) * pStrideK + vOutput0, aLine.data(), vRows);
          for (uint32_t r = 0; r < vRows; r++) {
            aTile[size_t(r) * vTileK + c] = aLine[r];
          }
        }
      } else {
        for (uint32_t r = 0; r < vRows; r++) {
          for (uint32_t c = 0; c < vColumns; c++) {
            aTile[size_t(r) * vTileK + c] = fXdnaHalfToFloat(pData[(vOutput0 + r) * pStrideN + (vInput0 + c) * pStrideK]);
          }
        }
      }
    }
    if (fPackTile(aTile.data(), vRows, vColumns, vTileK, 0, 0, vTileN, vTileK,
                  pBuffer + uint64_t(vColumn) * vColumnBytes + uint64_t(vTile) * vTileBytes, pRounding, vOrder) ==
        nullptr) {
      vFinite = false;
    }
  });
  return vFinite;
}

void fXdnaAttnMaskKinds(const XdnaOverlay & pOverlay, const uint16_t * pMask, uint64_t pMaskStride, uint32_t pTokens,
                         uint32_t pKeys, uint8_t * pKinds, XdnaWorkers * pWorkers) {
  const uint64_t vWidth    = pOverlay.vTileN;
  const uint64_t vStrips   = (uint64_t(pKeys) + vWidth - 1) / vWidth;
  const uint16_t cMinusInf = 0xfc00;  // minus infinity as a half of IEEE
  fForEach(pWorkers, uint64_t(pTokens) * pKeys >= cSharedValues, pTokens, [&](uint32_t pToken) {
    const uint16_t * vRow = pMask + uint64_t(pToken) * pMaskStride;
    for (uint64_t s = 0; s < vStrips; s++) {
      const uint64_t vFirst  = s * vWidth;
      const uint64_t vCount  = std::min<uint64_t>(vWidth, uint64_t(pKeys) - vFirst);
      bool           vMasked = true;
      bool           vOpen   = true;
      for (uint64_t j = 0; j < vCount; j++) {
        const uint16_t vHalf = vRow[vFirst + j];
        vMasked              = vMasked && vHalf == cMinusInf;
        vOpen                = vOpen && (vHalf & 0x7fffu) == 0;
      }
      pKinds[uint64_t(pToken) * vStrips + s] = vMasked ? cXdnaAttnMasked : vOpen ? cXdnaAttnOpen : cXdnaAttnMixed;
    }
  });
}

bool fXdnaGemmPackAttnP(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const XdnaAttnScores & pScores,
                         const XdnaSwigluPiece * aPieces, uint32_t pPieces, float * pSums, XdnaRounding pRounding,
                         XdnaWorkers * pWorkers) {
  // without a mask every key is open, and the kinds are not read
  if (pOverlay.vRowsC != 1 || pOverlay.vOutput != XdnaGemmOutput::cBfloat16 || pScores.vS == nullptr ||
      pScores.vHeads == 0 || (pScores.vMask != nullptr) != (pScores.vKinds != nullptr)) {
    return false;
  }
  if (pScores.vSinks != nullptr) {
    for (uint32_t h = 0; h < pScores.vHeads; h++) {
      if (std::isnan(pScores.vSinks[h]) || pScores.vSinks[h] == INFINITY) {
        return false;
      }
    }
  }
  const uint16_t *     vS         = static_cast<const uint16_t *>(pScores.vS);
  const uint64_t       vWidth     = pOverlay.vTileN;
  const uint32_t       vTileM     = pOverlay.vTileM;
  const uint32_t       vTileK     = pOverlay.vTileK;
  const uint64_t       vKeys      = pShape.vN;
  const uint64_t       vStrips    = (vKeys + vWidth - 1) / vWidth;
  const uint64_t       vTileBytes = uint64_t(vTileM) * vTileK / cBlock * cBlockBytes;
  const XdnaTileOrder vOrder     = fSteps3(pOverlay) ? XdnaTileOrder::cRows : XdnaTileOrder::cPairs;
  const bool           vShared    = uint64_t(pShape.vM) * pShape.vN >= cSharedValues;
  if (pScores.vKinds != nullptr && pScores.vKindsStride != vStrips) {
    return false;
  }
  // what the mask holds in the strip of a row, and the mask of its keys (none when they are all 0)
  auto fKind = [&](uint64_t pRow, uint64_t pStrip) {
    return pScores.vKinds == nullptr ? cXdnaAttnOpen :
                                       pScores.vKinds[(pRow / pScores.vHeads) * pScores.vKindsStride + pStrip];
  };
  auto fMaskOf = [&](uint64_t pRow, uint64_t pKey) -> const uint16_t * {
    return pScores.vMask + (pRow / pScores.vHeads) * pScores.vMaskStride + pKey;
  };

  // the pieces as A: whole tiles of K for these rows, laid out as fXdnaGemmPackA lays A out
  std::vector<uint64_t> aBandBytes(pPieces);
  uint32_t              vBands = 0;
  for (uint32_t p = 0; p < pPieces; p++) {
    XdnaGemmShape vShapeA;
    std::string    vIgnored;
    if (aPieces[p].vBuffer == nullptr || aPieces[p].vCount == 0 || aPieces[p].vCount % vTileK != 0 ||
        aPieces[p].vFirst % vTileK != 0 ||
        !fXdnaGemmShape(pOverlay, pShape.vM, aPieces[p].vCount, 1, vShapeA, vIgnored) ||
        vShapeA.vPaddedK != aPieces[p].vCount) {
      return false;
    }
    aBandBytes[p] = uint64_t(vShapeA.vPaddedK / vTileK) * vTileBytes;
    vBands        = vShapeA.vPaddedM / vTileM;
  }

  // Every band of rows is a task: the largest score of each of its rows, from all its strips (every strip of a band is
  // a stretch of the buffer of its own), and then its tiles of P, piece after piece in the order of the keys, with
  // that part of S still in the cache, and the sums of its rows in that order. The bands past the rows are zeros.
  std::atomic<bool> vFinite(true);
  fForEach(pWorkers, vShared, vBands, [&](uint32_t pBand) {
    thread_local std::vector<float> aTile;
    thread_local std::vector<float> aMax;
    thread_local std::vector<float> aSum;
    aTile.resize(size_t(vTileM) * vTileK);
    aMax.assign(vTileM, -INFINITY);
    aSum.assign(vTileM, 0.0f);
    const uint64_t vRow0 = uint64_t(pBand) * vTileM;
    const uint32_t vRows =
        vRow0 >= pShape.vM ? 0 : uint32_t(std::min<uint64_t>(vTileM, uint64_t(pShape.vM) - vRow0));
    for (uint64_t s = 0; s < vStrips && vRows > 0; s++) {
      const uint32_t vCount = uint32_t(std::min<uint64_t>(vWidth, vKeys - s * vWidth));
      for (uint32_t r = 0; r < vRows; r++) {
        const uint64_t vRow  = vRow0 + r;
        const uint8_t  vKind = fKind(vRow, s);
        if (vKind == cXdnaAttnMasked) {
          continue;
        }
        const float vMax = fXdnaScoresMax(vS + (s * pShape.vPaddedM + vRow) * vWidth,
                                           vKind == cXdnaAttnMixed ? fMaskOf(vRow, s * vWidth) : nullptr, vCount);
        aMax[r] = vMax > aMax[r] ? vMax : aMax[r];
      }
    }
    for (uint32_t p = 0; p < pPieces; p++) {
      for (uint32_t t = 0; t < aPieces[p].vCount / vTileK; t++) {
        const uint64_t vKey0    = aPieces[p].vFirst + uint64_t(t) * vTileK;
        const uint32_t vColumns = vKey0 >= vKeys ? 0 : uint32_t(std::min<uint64_t>(vTileK, vKeys - vKey0));
        for (uint32_t r = 0; r < vRows && vColumns > 0; r++) {
          const uint64_t vRow = vRow0 + r;
          const float    vMax = aMax[r];
          float *        vTo  = aTile.data() + size_t(r) * vTileK;
          if (vMax == -INFINITY) {
            std::memset(vTo, 0, size_t(vColumns) * sizeof(float));
            continue;
          }
          for (uint32_t j = 0; j < vColumns;) {
            // a run inside one strip
            const uint64_t vKey   = vKey0 + j;
            const uint64_t vStrip = vKey / vWidth;
            const uint32_t vRun   = uint32_t(std::min<uint64_t>(vWidth - vKey % vWidth, vColumns - j));
            const uint8_t  vKind  = fKind(vRow, vStrip);
            if (vKind == cXdnaAttnMasked) {
              std::memset(vTo + j, 0, size_t(vRun) * sizeof(float));
            } else {
              aSum[r] += fXdnaScoresExp(vS + (vStrip * pShape.vPaddedM + vRow) * vWidth + vKey % vWidth,
                                         vKind == cXdnaAttnMixed ? fMaskOf(vRow, vKey) : nullptr, vMax, vTo + j,
                                         vRun);
            }
            j += vRun;
          }
        }
        if (fPackTile(aTile.data(), vColumns > 0 ? vRows : 0, vColumns, vTileK, 0, 0, vTileM, vTileK,
                      aPieces[p].vBuffer + pBand * aBandBytes[p] + uint64_t(t) * vTileBytes, pRounding, vOrder) ==
            nullptr) {
          vFinite = false;
        }
      }
    }
    for (uint32_t r = 0; r < vRows; r++) {
      float vSum = aSum[r];
      if (pScores.vSinks != nullptr && aMax[r] > -INFINITY) {
        vSum += std::exp(pScores.vSinks[(vRow0 + r) % pScores.vHeads] - aMax[r]);
      }
      pSums[vRow0 + r] = vSum;
    }
  });
  return vFinite;
}
