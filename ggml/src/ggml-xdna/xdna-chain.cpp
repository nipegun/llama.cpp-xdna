#include "xdna-chain.h"

#include "xdna-bfp16.h"
#include "xdna-insts.h"
#include "xdna-workers.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>

namespace {

const uint32_t cManifestVersion = 9;
const uint32_t cMaxStreams      = 64;
const uint32_t cBlock           = 8;     // values that share an exponent
const uint32_t cBlockBytes      = 9;     // and the bytes they take
const uint32_t cGroupOutputs    = 8;     // outputs of a group of sums, and of a piece
const uint32_t cGroupsOut       = 4;     // groups of a block of a stream
const uint32_t cBlockOutputs    = cGroupOutputs * cGroupsOut;
const uint32_t cSuper           = 256;   // values of K of a tile, and of a block of the formats of ggml
const uint32_t cCoreBytes       = 4608;  // the half of an object that a core takes, as the kernel has it
const uint32_t cChunk           = cCoreBytes / sizeof(float);  // floats of X (or h) of a core in an object
const uint32_t cChunkGroups     = cChunk / cBlock;             // and its groups of 8
const uint32_t cRowFloats       = cGroupsOut * cGroupOutputs;  // floats of an object of Y: a row of a block
                                                               // (version 7; with version 6, the block with room
                                                               // for 8 rows, [4 groups][8 rows][8 outputs])
const uint32_t cParamsRows      = 5;     // the parameters: a row per phase and one with the number of phases
const uint32_t cParamsColumns   = 32;
const uint32_t cResidualMax     = 320;   // floats of the residual (and of h) that core 0 keeps per stream
const uint32_t cGateMax         = 1024;  // and of the gate
const uint64_t cPage            = 4096;

// buffers of a command
const uint32_t cBufferIo      = 0;
const uint32_t cBufferWeights = 1;  // of the first matrix

// items of a row of the parameters (see the kernel)
const uint32_t cItemPrologue = 0;
const uint32_t cItemObjects  = 1;
const uint32_t cItemRows     = 2;
const uint32_t cItemTiles    = 3;
const uint32_t cItemEpilogue = 4;
const uint32_t cItemResidual = 5;
const uint32_t cItemEps      = 6;
const uint32_t cItemValues   = 7;
const uint32_t cItemSegments = 8;  // and 4 per segment: blocks, objects of a block, format, 0
const uint32_t cItemRecurrence = 24;  // version 8: objects of the states of a core for the recurrence (0: none)
const uint32_t cItemInverse    = 25;  // version 8: 1 / K, for the mean of the squares of an RMS norm (a float)
const uint32_t cItemLinear     = 26;  // version 9: 1 in a phase of a whole layer of linear attention
const uint32_t cPrologueLinear = 3;   // version 9: the prologue of such a phase, copied from X[1] on
// version 9: the objects of the prologue of a phase of linear attention (the kernel says what is where): X[1] and X[2]
// from the host every token, X[3] from the matrix of the phase; in the half of every core, in bytes
const uint32_t cLinearObjects   = 3;
const uint32_t cLinearInput     = 64;    // X[1], core 1: x of the 512 channels of the pair, bf16
const uint32_t cLinearStates    = 1088;  // and their 3 old states by tap, [3][512] bf16
const uint32_t cLinearZ         = 256;   // X[2], core 1: z of head s + 16, then of head s, bf16 (bytes of one)
const uint32_t cLinearWeights   = 256;   // X[3], core 1: the weights of the gated norm, then those of the convolution
const uint32_t cLinearChannels  = 512;   // channels of a pair
const uint32_t cLinearScalars   = 5;     // X[1]: beta, e^g, beta e^g, E, eps of the L2 norms

// the recurrence (version 8): rows of the state of a head, and its rows and objects in the half of an object of a core
const uint32_t cRecurrenceDim     = cXdnaChainRecurrenceDim;
const uint32_t cRecurrenceRowBytes = cRecurrenceDim * sizeof(float);
const uint32_t cRecurrenceRows    = cCoreBytes / cRecurrenceRowBytes;                        // 9
const uint32_t cRecurrenceObjects = (cRecurrenceDim + cRecurrenceRows - 1) / cRecurrenceRows;  // 15
const uint64_t cRecurrenceHead    = uint64_t(cRecurrenceDim) * cRecurrenceRowBytes;          // 64 KB
// the floats of the inputs of a head in its object: q, k, beta v, the decay e^g and beta e^g (the kernel takes delta =
// beta v[j] - beta e^g (row . k), with the row before its decay)
const uint32_t cRecurrenceQ          = 0;
const uint32_t cRecurrenceK          = 128;
const uint32_t cRecurrenceBetaV      = 256;
const uint32_t cRecurrenceDecay      = 384;
const uint32_t cRecurrenceBetaDecay  = 385;
const uint32_t cBuffersMax      = 5;  // of a command: the one of the host, the matrices and the states

// puts a stream at its number, growing the list
bool fSetStream(std::vector<XdnaOverlayStream> & aStreams, uint32_t pNumber, uint32_t pColumn, uint32_t pChannel) {
  if (pNumber >= cMaxStreams || pColumn > 255 || pChannel > 255) {
    return false;
  }
  if (aStreams.size() <= pNumber) {
    aStreams.resize(pNumber + 1, XdnaOverlayStream{255, 255});
  }
  aStreams[pNumber] = {uint8_t(pColumn), uint8_t(pChannel)};
  return true;
}

bool fComplete(const std::vector<XdnaOverlayStream> & aStreams) {
  for (const XdnaOverlayStream & vStream : aStreams) {
    if (vStream.vColumn == 255) {
      return false;
    }
  }
  return !aStreams.empty();
}

uint64_t fRoundUp(uint64_t pValue, uint64_t pStep) {
  return (pValue + pStep - 1) / pStep * pStep;
}

// bytes of a row of pK values in pFormat as the host gives it: floats for bfp16, blocks of ggml otherwise
uint64_t fRowBytes(uint32_t pK, XdnaGemvFormat pFormat) {
  if (pFormat == XdnaGemvFormat::cBfp16) {
    return uint64_t(pK) * sizeof(float);
  }
  if (pFormat == XdnaGemvFormat::cQ8_0) {
    return uint64_t(pK) / 32 * 34;
  }
  if (pFormat == XdnaGemvFormat::cQ5_1) {
    return uint64_t(pK) / 32 * 24;
  }
  return uint64_t(pK) / cSuper * fXdnaGemvBlockBytes(pFormat);
}

// bytes of a group of 8 values of K of X in bfp16: its rows, padded to a multiple of 4
uint32_t fGroupBytesX(uint32_t pRows) {
  return (pRows * cBlockBytes + 3) / 4 * 4;
}

// the objects of a chunk of floats of every core for pHalf values of K of a core
uint32_t fChunks(uint32_t pHalf) {
  return (pHalf + cChunk - 1) / cChunk;
}

std::string fPhaseText(uint32_t pPhase) {
  return "phase " + std::to_string(pPhase) + " of the command";
}

}  // namespace

uint32_t fXdnaGemvManifestVersion(const std::string & pPath) {
  std::ifstream vFile(pPath);
  std::string   vLine;
  while (std::getline(vFile, vLine)) {
    std::istringstream vWords(vLine);
    std::string        vKey;
    uint32_t           vVersion = 0;
    if ((vWords >> vKey) && vKey == "xdna2-gemv" && (vWords >> vVersion)) {
      return vVersion;
    }
  }
  return 0;
}

bool XdnaChainOverlay::mLoad(const std::string & pPath, std::string & pError) {
  std::ifstream vFile(pPath);
  if (!vFile) {
    pError = "cannot open the manifest " + pPath;
    return false;
  }
  *this = XdnaChainOverlay();

  bool        vHasVersion = false;
  std::string vLine;
  uint32_t    vLineNumber = 0;
  while (std::getline(vFile, vLine)) {
    vLineNumber++;
    std::istringstream vWords(vLine);
    std::string        vKey;
    if (!(vWords >> vKey) || vKey[0] == '#') {
      continue;
    }
    bool vOk = true;
    if (vKey == "xdna2-gemv") {
      uint32_t vVersion = 0;
      vOk               = bool(vWords >> vVersion);
      if (vOk && vVersion != cManifestVersion) {
        pError = "the manifest " + pPath + " is of version " + std::to_string(vVersion) + ", and this build takes the " +
                 "chained overlay of version " + std::to_string(cManifestVersion) + ": build it again (gemv_chain.py)";
        return false;
      }
      vHasVersion = vOk;
    } else if (vKey == "array") {
      vOk = bool(vWords >> vColumns >> vRows >> vMemoryRows);
    } else if (vKey == "kernel") {
      vOk = bool(vWords >> vKernel);
    } else if (vKey == "object") {
      vOk = bool(vWords >> vObjectBytes >> vCoreBytes);
    } else if (vKey == "x") {
      vOk = bool(vWords >> vKeptX);
    } else if (vKey == "segments") {
      vOk = bool(vWords >> vSegments) && vSegments >= 1 && vSegments <= cXdnaChainSegmentsMax;
    } else if (vKey == "phases") {
      vOk = bool(vWords >> vPhases) && vPhases >= 1 && vPhases <= cXdnaChainPhasesMax;
    } else if (vKey == "formats") {
      std::string vName;
      while (vOk && (vWords >> vName)) {
        XdnaGemvFormat vFormat = XdnaGemvFormat::cBfp16;
        vOk                     = fXdnaGemvFormatFromName(vName, vFormat);
        vFormats |= vOk ? 1u << uint32_t(vFormat) : 0u;
      }
    } else if (vKey == "stream") {
      std::string vName;
      uint32_t    vNumber  = 0;
      uint32_t    vColumn  = 0;
      uint32_t    vChannel = 0;
      vOk = bool(vWords >> vName >> vNumber >> vColumn >> vChannel) && (vName == "W" || vName == "Y") &&
            fSetStream(vName == "W" ? aStreamsW : aStreamsY, vNumber, vColumn, vChannel);
    } else {
      vOk = false;
    }
    if (!vOk) {
      pError = "line " + std::to_string(vLineNumber) + " of the manifest " + pPath + " is not understood";
      return false;
    }
  }
  if (!vHasVersion || vColumns == 0 || vKernel.empty() || vObjectBytes == 0 || vKeptX == 0 || vSegments == 0 ||
      vPhases == 0 || !mTakes(XdnaGemvFormat::cBfp16) || !fComplete(aStreamsW) || !fComplete(aStreamsY)) {
    pError = "the manifest " + pPath + " is not complete";
    return false;
  }
  // an object is the halves of the two cores of a pair, which the kernel takes as 4608 bytes; X has room for one
  if (aStreamsW.size() != aStreamsY.size() || vCoreBytes != cCoreBytes || vObjectBytes != 2 * vCoreBytes ||
      vKeptX < vCoreBytes) {
    pError = "the manifest " + pPath + " does not describe a chained overlay for generating tokens";
    return false;
  }
  return true;
}

bool fXdnaChainMatrixShape(const XdnaChainOverlay & pOverlay, uint32_t pK, const XdnaGemvPart * aParts,
                            uint32_t pParts, XdnaChainMatrix & pMatrix, std::string & pError) {
  const uint64_t vStreams = pOverlay.mStreams();
  if (pK == 0 || vStreams == 0) {
    pError = "the matrix of weights is empty";
    return false;
  }
  if (pParts == 0 || pParts > cXdnaGemvSegmentsMax) {
    pError = "a matrix of the chained overlay has 1 to " + std::to_string(cXdnaGemvSegmentsMax) + " segments";
    return false;
  }
  const uint64_t vPaddedK = fRoundUp(pK, 2 * cSuper);
  pMatrix                 = XdnaChainMatrix();
  pMatrix.vK              = pK;
  pMatrix.vPaddedK        = uint32_t(vPaddedK);
  pMatrix.vTilesCore      = uint32_t(vPaddedK / (2 * cSuper));
  pMatrix.vSegments       = pParts;
  uint64_t vFirst         = 0;
  uint64_t vPartW         = 0;
  for (uint32_t i = 0; i < pParts; i++) {
    const XdnaGemvPart & vPart = aParts[i];
    if (!pOverlay.mTakes(vPart.vFormat) || vPart.vOutputs == 0) {
      pError = "the chained overlay does not take this format of weights";
      return false;
    }
    const uint32_t vStepK =
        (vPart.vFormat == XdnaGemvFormat::cQ8_0 || vPart.vFormat == XdnaGemvFormat::cQ5_1) ? 32u : cSuper;
    if (vPart.vFormat != XdnaGemvFormat::cBfp16 && pK % vStepK != 0) {
      pError = "weights in the formats of ggml come in blocks of 256 (or 32 for Q8_0 and Q5_1) values of K";
      return false;
    }
    // a block of a core: its 4 groups of 8 outputs by its tiles of K, a piece each; the blocks of the segment one after
    // the other, in whole halves of objects
    const uint64_t      vCoreBlock = uint64_t(cGroupsOut) * pMatrix.vTilesCore * fXdnaGemvPieceBytes(vPart.vFormat);
    XdnaChainSegment & vSegment   = pMatrix.aSegments[i];
    vSegment.vFormat     = vPart.vFormat;
    vSegment.vOutputs    = vPart.vOutputs;
    vSegment.vPadded     = uint32_t(fRoundUp(vPart.vOutputs, vStreams * cBlockOutputs));
    vSegment.vFirst      = uint32_t(vFirst);
    vSegment.vBlocks     = uint32_t(vSegment.vPadded / (vStreams * cBlockOutputs));
    vSegment.vCoreBlock  = vCoreBlock;
    const uint64_t vObjects = (vSegment.vBlocks * vCoreBlock + pOverlay.vCoreBytes - 1) / pOverlay.vCoreBytes;
    if (vObjects > 0x7fffffffu) {
      pError = "the matrix of weights is too large";
      return false;
    }
    vSegment.vObjects = uint32_t(vObjects);
    vSegment.vOffsetW = vPartW;
    vFirst += vPart.vOutputs;
    vPartW += uint64_t(vSegment.vObjects) * pOverlay.vObjectBytes;
  }
  if (vPaddedK > 0x40000000u || vFirst > 0x40000000u) {
    pError = "the matrix of weights is too large";
    return false;
  }
  pMatrix.vN      = uint32_t(vFirst);
  pMatrix.vPartW  = vPartW;
  pMatrix.vBytesW = vStreams * vPartW;
  return true;
}

bool fXdnaChainPackStream(const XdnaChainOverlay & pOverlay, const XdnaChainMatrix & pMatrix, uint32_t pSegment,
                           uint32_t pStream, const void * pRows, uint32_t pGiven, uint8_t * pBuffer,
                           XdnaWorkers * pWorkers) {
  const XdnaChainSegment & vSegment    = pMatrix.aSegments[pSegment];
  const uint32_t            vPerStream  = vSegment.vPadded / pOverlay.mStreams();
  const uint32_t            vGiven      = std::min(pGiven, vPerStream);
  const uint32_t            vPieceBytes = fXdnaGemvPieceBytes(vSegment.vFormat);
  const uint64_t            vRowBytes   = fRowBytes(pMatrix.vK, vSegment.vFormat);
  uint8_t *                 vPart       = pBuffer + uint64_t(pStream) * pMatrix.vPartW + vSegment.vOffsetW;
  std::atomic<bool>         vFinite(true);
  // the padding after the last block, in the last object: zeros, before the blocks are written
  if (vSegment.vObjects > 0) {
    std::memset(vPart + uint64_t(vSegment.vObjects - 1) * pOverlay.vObjectBytes, 0, pOverlay.vObjectBytes);
  }
  // One task per block of outputs of the stream; its rows count from the first output of the stream. The pieces of a
  // core for block b start at byte b * vCoreBlock of its halves of objects: two tasks may write the same object, never
  // the same bytes.
  auto fBlock = [&](uint32_t pBlock) {
    alignas(64) uint8_t aPiece[2304];
    for (uint32_t vCore = 0; vCore < 2; vCore++) {
      // the pieces of this core: its half of the tiles of K, tile after tile and group of 8 outputs after group,
      // cut into halves of objects, a piece that does not fit going on in the next one
      uint64_t vInCore = uint64_t(pBlock) * vSegment.vCoreBlock;
      for (uint32_t vTile = 0; vTile < pMatrix.vTilesCore; vTile++) {
        for (uint32_t vGroup = 0; vGroup < cGroupsOut; vGroup++) {
          const uint32_t vRow = pBlock * cBlockOutputs + vGroup * cGroupOutputs;
          if (!fXdnaGemvPiece(vSegment.vFormat, pRows, vGiven, pMatrix.vK, vRowBytes, vRow,
                               vCore * pMatrix.vTilesCore + vTile, aPiece)) {
            vFinite = false;
            return;
          }
          for (uint32_t vDone = 0; vDone < vPieceBytes;) {
            const uint64_t vObject = vInCore / pOverlay.vCoreBytes;
            const uint64_t vWithin = vInCore % pOverlay.vCoreBytes;
            const uint64_t vBytes  = std::min<uint64_t>(vPieceBytes - vDone, pOverlay.vCoreBytes - vWithin);
            std::memcpy(vPart + vObject * pOverlay.vObjectBytes + vCore * pOverlay.vCoreBytes + vWithin, aPiece + vDone,
                        vBytes);
            vDone += uint32_t(vBytes);
            vInCore += vBytes;
          }
        }
      }
    }
  };
  if (pWorkers != nullptr) {
    pWorkers->mForEach(vSegment.vBlocks, fBlock);
  } else {
    for (uint32_t vBlock = 0; vBlock < vSegment.vBlocks; vBlock++) {
      fBlock(vBlock);
    }
  }
  return vFinite;
}

bool fXdnaChainCommandShape(const XdnaChainOverlay & pOverlay, const XdnaChainPhase * aPhases, uint32_t pPhases,
                             const XdnaChainMatrix * aMatrices, uint32_t pMatrices, XdnaChainCommand & pCommand,
                             std::string & pError, uint32_t pStates) {
  const uint64_t vStreams = pOverlay.mStreams();
  if (pPhases == 0 || pPhases > pOverlay.vPhases) {
    pError = "the chained overlay runs 1 to " + std::to_string(pOverlay.vPhases) + " phases a command";
    return false;
  }
  if (pMatrices > cXdnaChainMatricesMax || 1 + pMatrices + pStates > cBuffersMax) {
    pError = "a command of the chained overlay has " + std::to_string(cXdnaChainMatricesMax) + " matrices at most, and " +
             std::to_string(cBuffersMax - 1) + " matrices and buffers of states at most";
    return false;
  }
  pCommand           = XdnaChainCommand();
  pCommand.vPhases   = pPhases;
  pCommand.vMatrices = pMatrices;
  pCommand.vStates   = pStates;
  for (uint32_t i = 0; i < pMatrices; i++) {
    pCommand.aMatrices[i] = aMatrices[i];
  }
  const uint32_t vObjectsKept = pOverlay.vKeptX / pOverlay.vCoreBytes;  // whole halves of objects of X a core keeps
  uint64_t       vAt          = pOverlay.vObjectBytes;                  // after the parameters
  uint64_t       vY           = 0;  // from the start of the Y
  uint64_t       vYEnd        = 0;  // the furthest a phase may read a Y, from there
  for (uint32_t p = 0; p < pPhases; p++) {
    const XdnaChainPhase & vPhase = aPhases[p];
    XdnaChainPlace &       vPlace = pCommand.aPlaces[p];
    pCommand.aPhases[p]            = vPhase;
    if (vPhase.vRecurrence.vLinear != 0) {
      // version 9, a whole layer of linear attention: 32 heads, one per core; X[1] and X[2] of every stream from the
      // host, X[3] from the matrix of the phase (an object per stream); its outputs, [head][128] floats, in its Y
      const XdnaChainRecurrence & vRecurrence = vPhase.vRecurrence;
      if (vRecurrence.vHeads != cXdnaChainLinearHeads || 2 * vStreams != cXdnaChainLinearHeads ||
          vRecurrence.vState >= pStates || vRecurrence.vStateOffset % sizeof(uint32_t) != 0 ||
          vPhase.vMatrix >= pMatrices || aMatrices[vPhase.vMatrix].vSegments != 0 ||
          aMatrices[vPhase.vMatrix].vPartW != pOverlay.vObjectBytes || vPhase.vFromPhase >= 0 || vPhase.vSameXAs >= 0 ||
          vObjectsKept < 1 + cLinearObjects + 3) {
        pError = fPhaseText(p) + ": a layer of linear attention takes 32 heads on 16 streams, a buffer of states, the " +
                 "matrix of its fixed objects and its inputs from the host";
        return false;
      }
      pCommand.aPhases[p].vSegments = 1;
      vPlace.vObjectsX              = 1;
      vPlace.vObjectsPrologue       = cLinearObjects;
      vPlace.vOffsetX               = vAt;
      vAt += vStreams * 2 * pOverlay.vObjectBytes;
      vPlace.vOffsetY[0] = vY;
      vY += uint64_t(vRecurrence.vHeads) * cRecurrenceRowBytes;
      vPlace.vBytesY = vY - vPlace.vOffsetY[0];
      continue;
    }
    if (vPhase.vRecurrence.vHeads != 0) {
      // the recurrence: the inputs of every head of a core, an object each for its stream, kept in X; its outputs,
      // [head][128] floats, in its Y
      const XdnaChainRecurrence & vRecurrence = vPhase.vRecurrence;
      const uint32_t               vCores      = uint32_t(vStreams) * 2;
      const uint32_t               vPerCore    = vRecurrence.vHeads / vCores;
      if (vRecurrence.vHeads % vCores != 0 || vPerCore > vObjectsKept || vRecurrence.vState >= pStates ||
          vRecurrence.vStateOffset % sizeof(uint32_t) != 0 || vPhase.vFromPhase >= 0 || vPhase.vSameXAs >= 0) {
        pError = fPhaseText(p) + ": the recurrence takes " + std::to_string(vCores) + " to " +
                 std::to_string(vCores * vObjectsKept) + " heads (a multiple of " + std::to_string(vCores) +
                 "), a buffer of states of the command and its inputs from the host";
        return false;
      }
      pCommand.aPhases[p].vSegments = 1;
      vPlace.vObjectsX              = vPerCore;
      vPlace.vObjectsPrologue       = vPerCore;
      vPlace.vOffsetX               = vAt;
      vAt += vStreams * vPerCore * pOverlay.vObjectBytes;
      vPlace.vOffsetY[0] = vY;
      vY += uint64_t(vRecurrence.vHeads) * cRecurrenceRowBytes;
      vPlace.vBytesY = vY - vPlace.vOffsetY[0];
      continue;
    }
    if (vPhase.vMatrix >= pMatrices) {
      pError = fPhaseText(p) + " runs a matrix the command does not have";
      return false;
    }
    const XdnaChainMatrix & vMatrix = aMatrices[vPhase.vMatrix];
    if (vPhase.vSegments == 0 || vPhase.vSegments > pOverlay.vSegments ||
        vPhase.vFirstSegment + vPhase.vSegments > vMatrix.vSegments) {
      pError = fPhaseText(p) + " runs 1 to " + std::to_string(pOverlay.vSegments) + " segments of its matrix";
      return false;
    }
    const XdnaChainSegment & vFirst = vMatrix.aSegments[vPhase.vFirstSegment];
    const uint32_t            vHalf  = vMatrix.vPaddedK / 2;  // values of K of a core
    vPlace.vK                        = vMatrix.vK;
    vPlace.vPaddedK                  = vMatrix.vPaddedK;
    for (uint32_t g = 0; g < vPhase.vSegments; g++) {
      vPlace.vBlocks[g] = vMatrix.aSegments[vPhase.vFirstSegment + g].vBlocks;
    }

    // rows and epilogues: the epilogues work on row 0, with the outputs of a stream kept by core 0
    const bool vOneRow = vPhase.vPrologue != XdnaChainPrologue::cBfp16;
    if (vPhase.vRows == 0 || vPhase.vRows > cXdnaGemvRowsMax || (vOneRow && vPhase.vRows != 1) ||
        (vPhase.vEpilogue != XdnaChainEpilogue::cNone && vPhase.vRows != 1)) {
      pError = fPhaseText(p) + ": 1 to 8 rows with X in bfp16, 1 row otherwise and with an epilogue";
      return false;
    }
    const uint32_t vPerStream = vFirst.vPadded / uint32_t(vStreams);
    if (vPhase.vEpilogue == XdnaChainEpilogue::cResidual || vPhase.vEpilogue == XdnaChainEpilogue::cAddH) {
      if (vPhase.vSegments != 1 || vPerStream > cResidualMax) {
        pError = fPhaseText(p) + ": a residual or h goes with one segment of " +
                 std::to_string(cResidualMax * vStreams) + " outputs at most";
        return false;
      }
    }
    if (vPhase.vEpilogue == XdnaChainEpilogue::cAddH) {
      // h comes from the last phase before with a residual, with the same outputs
      int32_t vKept = -1;
      for (uint32_t q = 0; q < p; q++) {
        vKept = aPhases[q].vEpilogue == XdnaChainEpilogue::cResidual ? int32_t(q) : vKept;
      }
      if (vKept < 0 || pCommand.aPlaces[vKept].vBlocks[0] != vPlace.vBlocks[0]) {
        pError = fPhaseText(p) + " adds h, which no phase before with the same outputs keeps";
        return false;
      }
    }
    if (vPhase.vEpilogue == XdnaChainEpilogue::cSwiglu) {
      const uint32_t vGateMax = pOverlay.mTakes(XdnaGemvFormat::cQ8_0) ? 1088u : cGateMax;
      if (vPhase.vSegments != 2 || vPlace.vBlocks[0] != vPlace.vBlocks[1] || vPerStream > vGateMax) {
        pError = fPhaseText(p) + ": the SwiGLU takes two segments of the same " + std::to_string(vGateMax * vStreams) +
                 " outputs at most";
        return false;
      }
    }

    // the prologue: the residual, X (or h) and the weights of the norm
    const bool vResidual = vPhase.vEpilogue == XdnaChainEpilogue::cResidual;
    if (vPhase.vPrologue == XdnaChainPrologue::cBfp16) {
      const uint64_t vBytesCore = uint64_t(vHalf / cBlock) * fGroupBytesX(vPhase.vRows);
      vPlace.vObjectsX          = uint32_t((vBytesCore + pOverlay.vCoreBytes - 1) / pOverlay.vCoreBytes);
      if (vPlace.vObjectsX > vObjectsKept) {
        pError = std::to_string(vPhase.vRows) + " rows of " + std::to_string(vMatrix.vK) +
                 " values do not fit in the cores of the chained overlay";
        return false;
      }
    } else if (vPhase.vPrologue == XdnaChainPrologue::cFloats) {
      // the core keeps its half as bfp16 of one row: 12 bytes for 8 values
      if (uint64_t(vHalf / cBlock) * fGroupBytesX(1) > pOverlay.vKeptX) {
        pError = std::to_string(vMatrix.vK) + " values do not fit in the cores of the chained overlay";
        return false;
      }
      vPlace.vObjectsX = fChunks(vHalf);
    } else if (vPhase.vPrologue == XdnaChainPrologue::cNorm) {
      // the core keeps its half of h in floats; the mean of the squares counts the padded values of K too
      if (uint64_t(vHalf) * sizeof(float) > pOverlay.vKeptX || vMatrix.vPaddedK != vMatrix.vK || vPhase.vEps < 0 ||
          !std::isfinite(vPhase.vEps)) {
        pError = fPhaseText(p) + ": the norm takes K a multiple of 512 whose half fits in a core in floats";
        return false;
      }
      vPlace.vObjectsX = fChunks(vHalf);
    } else {
      pError = fPhaseText(p) + " has a prologue the chained overlay does not have";
      return false;
    }
    if (vPhase.vFromPhase >= 0) {
      if (vPhase.vPrologue == XdnaChainPrologue::cBfp16 || uint32_t(vPhase.vFromPhase) >= p ||
          vPhase.vFromSegment >= aPhases[vPhase.vFromPhase].vSegments || vPhase.vSameXAs >= 0) {
        pError = fPhaseText(p) + " takes its X from a Y that is not one of a phase before it";
        return false;
      }
      // value k of X is output k of that segment, which has to have K outputs (its padding is zeros, as is that of K)
      const XdnaChainPhase &   vFrom   = aPhases[vPhase.vFromPhase];
      const XdnaChainSegment & vSource = aMatrices[vFrom.vMatrix].aSegments[vFrom.vFirstSegment + vPhase.vFromSegment];
      if (vSource.vOutputs != vMatrix.vK || vSource.vPadded < vMatrix.vPaddedK) {
        pError = fPhaseText(p) + " takes its X from a Y that does not have its K outputs";
        return false;
      }
      // the pattern reads whole chunks of both halves: past K, and so past the end of that Y (of one row: output o is
      // float o of it)
      const uint64_t vBase = pCommand.aPlaces[vPhase.vFromPhase].vOffsetY[vPhase.vFromSegment];
      const uint64_t vEnd  = vBase + (uint64_t(vHalf) + uint64_t(vPlace.vObjectsX) * cChunk) * sizeof(float);
      vYEnd                = std::max(vYEnd, vEnd);
    } else if (vPhase.vSameXAs >= 0) {
      const XdnaChainPhase & vSame = aPhases[vPhase.vSameXAs];
      if (uint32_t(vPhase.vSameXAs) >= p || vSame.vFromPhase >= 0 || vSame.vPrologue != vPhase.vPrologue ||
          vSame.vRows != vPhase.vRows || pCommand.aPlaces[vPhase.vSameXAs].vK != vMatrix.vK ||
          pCommand.aPlaces[vPhase.vSameXAs].vPaddedK != vMatrix.vPaddedK) {
        pError = fPhaseText(p) + " takes the X of a phase that does not have the same X";
        return false;
      }
      vPlace.vOffsetX = pCommand.aPlaces[vPhase.vSameXAs].vOffsetX;
    }
    if (vResidual) {
      vPlace.vOffsetResidual = vAt;
      vAt += vStreams * pOverlay.vObjectBytes;
    }
    if (vPhase.vFromPhase < 0 && vPhase.vSameXAs < 0) {
      vPlace.vOffsetX = vAt;
      vAt += uint64_t(vPlace.vObjectsX) * pOverlay.vObjectBytes;
    }
    uint32_t vObjectsNorm = 0;
    if (vPhase.vPrologue == XdnaChainPrologue::cNorm) {
      vObjectsNorm       = vPlace.vObjectsX;
      vPlace.vOffsetNorm = vAt;
      vAt += uint64_t(vObjectsNorm) * pOverlay.vObjectBytes;
    }
    vPlace.vObjectsPrologue = (vResidual ? 1 : 0) + vPlace.vObjectsX + vObjectsNorm;

    // the Y of its segments, [segment][stream][block][row][32 outputs]
    for (uint32_t g = 0; g < vPhase.vSegments; g++) {
      vPlace.vOffsetY[g] = vY;
      vY += vStreams * vPlace.vBlocks[g] * vPhase.vRows * cRowFloats * sizeof(float);
    }
    vPlace.vBytesY = vY - vPlace.vOffsetY[0];
  }
  // the Y on pages of their own, after what the host writes
  pCommand.vBytesIn = vAt;
  pCommand.vOffsetY = fRoundUp(vAt, cPage);
  for (uint32_t p = 0; p < pPhases; p++) {
    for (uint32_t g = 0; g < pCommand.aPhases[p].vSegments; g++) {
      pCommand.aPlaces[p].vOffsetY[g] += pCommand.vOffsetY;
    }
  }
  pCommand.vBytesIo = pCommand.vOffsetY + fRoundUp(std::max(vY, vYEnd), cPage);
  if (pCommand.vBytesIo > (1ull << 32)) {
    pError = "the command is too large";
    return false;
  }
  return true;
}

std::vector<uint32_t> fXdnaChainKey(const XdnaChainCommand & pCommand) {
  std::vector<uint32_t> aKey = {pCommand.vPhases, pCommand.vMatrices, pCommand.vStates};
  for (uint32_t p = 0; p < pCommand.vPhases; p++) {
    const XdnaChainPhase & vPhase = pCommand.aPhases[p];
    aKey.insert(aKey.end(), {uint32_t(vPhase.vPrologue), vPhase.vRows, uint32_t(vPhase.vFromPhase),
                             vPhase.vFromSegment, uint32_t(vPhase.vSameXAs), uint32_t(vPhase.vEpilogue),
                             vPhase.vMatrix, vPhase.vFirstSegment, vPhase.vSegments, vPhase.vRecurrence.vHeads,
                             vPhase.vRecurrence.vState, uint32_t(vPhase.vRecurrence.vStateOffset & 0xffffffffu),
                             uint32_t(vPhase.vRecurrence.vStateOffset >> 32), vPhase.vRecurrence.vLinear});
  }
  for (uint32_t m = 0; m < pCommand.vMatrices; m++) {
    const XdnaChainMatrix & vMatrix = pCommand.aMatrices[m];
    aKey.insert(aKey.end(), {vMatrix.vK, vMatrix.vSegments});
    for (uint32_t g = 0; g < vMatrix.vSegments; g++) {
      aKey.insert(aKey.end(), {uint32_t(vMatrix.aSegments[g].vFormat), vMatrix.aSegments[g].vOutputs});
    }
  }
  return aKey;
}

bool fXdnaChainPlan(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand,
                     std::vector<uint32_t> & aInstructions, std::string & pError, bool pEarly, bool pFillsFirst) {
  const uint32_t vStreams = pOverlay.mStreams();
  const uint64_t vObject  = pOverlay.vObjectBytes;
  if (pCommand.vPhases == 0) {
    pError = "the command has no phases";
    return false;
  }
  XdnaInstructions          vInstructions;
  XdnaShimDma               vDma(vInstructions, pOverlay.vColumns);
  std::vector<XdnaShimTask> aDrains;  // not waited for yet
  std::vector<XdnaShimTask> aFills;   // not given back yet
  uint32_t                   vWaited = 0;  // phases whose Y has been waited for
  // the transfers of a phase, in the order they are queued, and whether each one drains Y (and is waited for)
  struct XdnaPlanned {
    XdnaShimTransfer vTransfer;
    bool              vDrain = false;
  };
  std::vector<XdnaPlanned>  aPlanned;
  std::vector<XdnaShimTask> aTasks;
  auto fPush = [&](const XdnaShimTransfer & pTransfer, std::vector<XdnaShimTask> & aInto) {
    aPlanned.push_back({pTransfer, &aInto == &aDrains});
    return true;
  };
  // Waits for the Y of every phase sent so far. Then every object those phases took has gone through the cores, so
  // the descriptors of their transfers are free and the queues of the channels of W are empty.
  auto fWaitAll = [&](uint32_t pPhasesSent) {
    for (const XdnaShimTask & vTask : aDrains) {
      if (vTask.vToken) {
        vDma.mAwait(vTask);
      }
    }
    for (const XdnaShimTask & vTask : aDrains) {
      vDma.mFree(vTask);
    }
    for (const XdnaShimTask & vTask : aFills) {
      vDma.mFree(vTask);
    }
    aDrains.clear();
    aFills.clear();
    for (const XdnaOverlayStream & vStream : pOverlay.aStreamsW) {
      vDma.mSettle(vStream.vColumn, XdnaDirection::cFromMemory, vStream.vChannel);
    }
    vWaited = pPhasesSent;
  };
  auto fLinear = [&](const XdnaOverlayStream & pStream, XdnaDirection pDirection, uint32_t pBuffer, uint64_t pOffset,
                     uint64_t pBytes, bool pToken) {
    XdnaShimTransfer vTransfer;
    vTransfer.vColumn    = pStream.vColumn;
    vTransfer.vDirection = pDirection;
    vTransfer.vChannel   = pStream.vChannel;
    vTransfer.vBuffer    = pBuffer;
    vTransfer.vOffset    = pOffset;
    vTransfer.vBytes     = pBytes;
    vTransfer.vToken     = pToken;
    return vTransfer;
  };

  for (uint32_t p = 0; p < pCommand.vPhases; p++) {
    const XdnaChainPhase &  vPhase  = pCommand.aPhases[p];
    const XdnaChainPlace &  vPlace  = pCommand.aPlaces[p];
    const XdnaChainMatrix & vMatrix = pCommand.aMatrices[vPhase.vMatrix];
    const uint32_t           vHalf   = vMatrix.vPaddedK / 2;
    // X from a Y: that Y has to be in memory
    const bool vWait = vPhase.vFromPhase >= 0 && uint32_t(vPhase.vFromPhase) >= vWaited;
    aPlanned.clear();
    const bool     vRecurrence    = vPhase.vRecurrence.vHeads != 0;
    const uint32_t vMatrixStreams = vRecurrence ? 0 : vStreams;  // the streams that run a matrix in the phase

    if (vRecurrence && vPhase.vRecurrence.vLinear != 0) {
      // Version 9, a whole layer of linear attention: heads s (core 0) and s + 16 (core 1) on stream s. Every stream
      // drains row j of both heads one after the other back where they came from, and then the outputs of both heads
      // to their places in the Y of the phase; and takes the parameters (the first phase), X[1] and X[2] of the
      // stream from the host, X[3] from the matrix of the phase, and the objects of the states of its heads, [rows 9 o
      // to 9 o + 8 of head s | the same rows of head s + 16] for o from 0 to 14 (from row 128 on, padding read past
      // the head).
      const XdnaChainRecurrence & vChain     = vPhase.vRecurrence;
      const uint32_t               vStates    = cBufferWeights + pCommand.vMatrices + vChain.vState;
      const uint32_t               vHeadWords = uint32_t(cRecurrenceHead / sizeof(uint32_t));
      const uint32_t               vPair      = vStreams * vHeadWords;  // from head s to head s + 16, in words
      for (uint32_t s = 0; s < vStreams; s++) {
        XdnaShimTransfer vRows = fLinear(pOverlay.aStreamsY[s], XdnaDirection::cToMemory, vStates,
                                          vChain.vStateOffset + s * cRecurrenceHead, 2 * cRecurrenceHead, false);
        vRows.vSizes[0]   = cRecurrenceDim;
        vRows.vSizes[1]   = 2;
        vRows.vStrides[0] = 1;
        vRows.vStrides[1] = vPair;
        vRows.vStrides[2] = cRecurrenceDim;
        XdnaShimTransfer vOutputs = fLinear(pOverlay.aStreamsY[s], XdnaDirection::cToMemory, cBufferIo,
                                             vPlace.vOffsetY[0] + s * cRecurrenceRowBytes, 2 * cRecurrenceRowBytes, true);
        vOutputs.vSizes[0]   = cRecurrenceDim;
        vOutputs.vSizes[1]   = 2;
        vOutputs.vStrides[0] = 1;
        vOutputs.vStrides[1] = vStreams * cRecurrenceDim;
        vOutputs.vStrides[2] = 1;
        if (!fPush(vRows, aDrains) || !fPush(vOutputs, aDrains)) {
          return false;
        }
      }
      for (uint32_t s = 0; s < vStreams; s++) {
        const XdnaOverlayStream & vStream = pOverlay.aStreamsW[s];
        if (p == 0 && !fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, 0, vObject, false), aFills)) {
          return false;
        }
        XdnaShimTransfer vObjects = fLinear(vStream, XdnaDirection::cFromMemory, vStates,
                                             vChain.vStateOffset + s * cRecurrenceHead, vObject, false);
        vObjects.vSizes[0]        = cRecurrenceDim;
        vObjects.vSizes[1]        = cRecurrenceRows;
        vObjects.vStrides[0]      = 1;
        vObjects.vStrides[1]      = cRecurrenceDim;
        vObjects.vStrides[2]      = vPair;
        vObjects.vIterations      = cRecurrenceObjects;
        vObjects.vIterationStride = cRecurrenceRows * cRecurrenceDim;
        if (!fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, vPlace.vOffsetX + uint64_t(s) * 2 * vObject,
                           2 * vObject, false),
                   aFills) ||
            !fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferWeights + vPhase.vMatrix, uint64_t(s) * vObject,
                           vObject, false),
                   aFills) ||
            !fPush(vObjects, aFills)) {
          return false;
        }
      }
    } else if (vRecurrence) {
      // The recurrence (version 8). Every stream drains the rows of the states of its heads, row j of both heads of a
      // pair one after the other, back where they came from, and then their outputs; and takes the parameters (the
      // first phase), the inputs of its heads and the objects of their states, [rows 9 o to 9 o + 8 of head 2 s | the
      // same rows of head 2 s + 1] for o from 0 to 14 (from row 128 on, padding read past the head).
      const XdnaChainRecurrence & vChain    = vPhase.vRecurrence;
      const uint32_t               vPerCore  = vPlace.vObjectsX;
      const uint32_t               vStates   = cBufferWeights + pCommand.vMatrices + vChain.vState;
      const uint32_t               vHeadWords = uint32_t(cRecurrenceHead / sizeof(uint32_t));
      for (uint32_t s = 0; s < vStreams; s++) {
        for (uint32_t h = 0; h < vPerCore; h++) {
          const uint64_t    vHead = uint64_t(h) * 2 * vStreams + 2 * s;
          XdnaShimTransfer vRows = fLinear(pOverlay.aStreamsY[s], XdnaDirection::cToMemory, vStates,
                                            vChain.vStateOffset + vHead * cRecurrenceHead, 2 * cRecurrenceHead, false);
          vRows.vSizes[0]   = cRecurrenceDim;
          vRows.vSizes[1]   = 2;
          vRows.vStrides[0] = 1;
          vRows.vStrides[1] = vHeadWords;
          vRows.vStrides[2] = cRecurrenceDim;
          if (!fPush(vRows, aDrains) ||
              !fPush(fLinear(pOverlay.aStreamsY[s], XdnaDirection::cToMemory, cBufferIo,
                             vPlace.vOffsetY[0] + vHead * cRecurrenceRowBytes, 2 * cRecurrenceRowBytes, h + 1 == vPerCore),
                     aDrains)) {
            return false;
          }
        }
      }
      for (uint32_t s = 0; s < vStreams; s++) {
        const XdnaOverlayStream & vStream = pOverlay.aStreamsW[s];
        const uint64_t             vInputs = vPlace.vOffsetX + uint64_t(s) * vPerCore * vObject;
        if (p == 0 && vPerCore == 1) {
          // the parameters and the inputs of the stream, two objects of one descriptor
          XdnaShimTransfer vTransfer = fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, 0, vObject, false);
          vTransfer.vIterations       = 2;
          vTransfer.vIterationStride  = uint32_t(vInputs / sizeof(uint32_t));
          if (!fPush(vTransfer, aFills)) {
            return false;
          }
        } else {
          if (p == 0 && !fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, 0, vObject, false), aFills)) {
            return false;
          }
          if (!fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, vInputs, vPerCore * vObject, false),
                     aFills)) {
            return false;
          }
        }
        for (uint32_t h = 0; h < vPerCore; h++) {
          const uint64_t    vHead    = uint64_t(h) * 2 * vStreams + 2 * s;
          XdnaShimTransfer vObjects = fLinear(vStream, XdnaDirection::cFromMemory, vStates,
                                               vChain.vStateOffset + vHead * cRecurrenceHead, vObject, false);
          vObjects.vSizes[0]        = cRecurrenceDim;
          vObjects.vSizes[1]        = cRecurrenceRows;
          vObjects.vStrides[0]      = 1;
          vObjects.vStrides[1]      = cRecurrenceDim;
          vObjects.vStrides[2]      = vHeadWords;
          vObjects.vIterations      = cRecurrenceObjects;
          vObjects.vIterationStride = cRecurrenceRows * cRecurrenceDim;
          if (!fPush(vObjects, aFills)) {
            return false;
          }
        }
      }
    }

    // the Y of every stream, which are waited for: [segment][stream][block], one descriptor for two segments of the
    // same blocks, two otherwise
    for (uint32_t s = 0; s < vMatrixStreams; s++) {
      const XdnaOverlayStream & vStream = pOverlay.aStreamsY[s];
      const uint64_t             vBytes0 = uint64_t(vPlace.vBlocks[0]) * vPhase.vRows * cRowFloats * sizeof(float);
      XdnaShimTransfer vTransfer = fLinear(vStream, XdnaDirection::cToMemory, cBufferIo, vPlace.vOffsetY[0] + s * vBytes0,
                                            vBytes0, vPhase.vSegments == 1);
      if (vPhase.vSegments == 2 && vPlace.vBlocks[1] == vPlace.vBlocks[0]) {
        vTransfer.vToken           = true;
        vTransfer.vIterations      = 2;
        vTransfer.vIterationStride = uint32_t((vPlace.vOffsetY[1] - vPlace.vOffsetY[0]) / sizeof(uint32_t));
        if (!fPush(vTransfer, aDrains)) {
          return false;
        }
        continue;
      }
      if (!fPush(vTransfer, aDrains)) {
        return false;
      }
      if (vPhase.vSegments == 2) {
        const uint64_t vBytes1 = uint64_t(vPlace.vBlocks[1]) * vPhase.vRows * cRowFloats * sizeof(float);
        if (!fPush(fLinear(vStream, XdnaDirection::cToMemory, cBufferIo, vPlace.vOffsetY[1] + s * vBytes1, vBytes1,
                           true),
                   aDrains)) {
          return false;
        }
      }
    }

    // the objects of every stream, in the order the cores take them: the parameters (the first phase), the residual,
    // X (or h), the weights of the norm and the weights
    const bool vResidual = vPhase.vEpilogue == XdnaChainEpilogue::cResidual;
    const bool vHostX    = vPhase.vFromPhase < 0;
    for (uint32_t s = 0; s < vMatrixStreams; s++) {
      const XdnaOverlayStream & vStream = pOverlay.aStreamsW[s];
      bool                       vXSent  = false;
      if (p == 0) {
        if (vResidual) {
          // the parameters and the residual of the stream, two objects of one descriptor
          XdnaShimTransfer vTransfer = fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, 0, vObject, false);
          vTransfer.vIterations       = 2;
          vTransfer.vIterationStride  = uint32_t((vPlace.vOffsetResidual + s * vObject) / sizeof(uint32_t));
          if (!fPush(vTransfer, aFills)) {
            return false;
          }
        } else {
          // the parameters, and X right after them when it comes from the host
          vXSent = vHostX && vPlace.vOffsetX == vObject;
          const uint64_t vBytes = vObject * (1 + (vXSent ? vPlace.vObjectsX : 0));
          if (!fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, 0, vBytes, false), aFills)) {
            return false;
          }
        }
      } else if (vResidual) {
        if (!fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, vPlace.vOffsetResidual + s * vObject,
                           vObject, false),
                   aFills)) {
          return false;
        }
      }
      if (!vHostX) {
        // the Y of that segment, of one row: output o is float o of it. Object m is [chunk m of the first half of K |
        // chunk m of the second half], 1152 consecutive floats each (8 at a time)
        const uint64_t    vBase     = pCommand.aPlaces[vPhase.vFromPhase].vOffsetY[vPhase.vFromSegment];
        XdnaShimTransfer vTransfer = fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, vBase, vObject, false);
        vTransfer.vSizes[0]        = cGroupOutputs;
        vTransfer.vSizes[1]        = cChunkGroups;
        vTransfer.vStrides[0]      = 1;
        vTransfer.vStrides[1]      = cGroupOutputs;
        vTransfer.vStrides[2]      = vHalf;
        vTransfer.vIterations      = vPlace.vObjectsX;
        vTransfer.vIterationStride = cChunk;
        if (!fPush(vTransfer, aFills)) {
          return false;
        }
      } else if (!vXSent) {
        if (!fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, vPlace.vOffsetX,
                           vObject * vPlace.vObjectsX, false),
                   aFills)) {
          return false;
        }
      }
      if (vPhase.vPrologue == XdnaChainPrologue::cNorm) {
        if (!fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferIo, vPlace.vOffsetNorm,
                           vObject * vPlace.vObjectsX, false),
                   aFills)) {
          return false;
        }
      }
      // the weights of its segments, one after the other in the part of the stream
      uint64_t vBytesW = 0;
      for (uint32_t g = 0; g < vPhase.vSegments; g++) {
        vBytesW += uint64_t(vMatrix.aSegments[vPhase.vFirstSegment + g].vObjects) * vObject;
      }
      if (!fPush(fLinear(vStream, XdnaDirection::cFromMemory, cBufferWeights + vPhase.vMatrix,
                         uint64_t(s) * vMatrix.vPartW + vMatrix.aSegments[vPhase.vFirstSegment].vOffsetW, vBytesW,
                         false),
                 aFills)) {
        return false;
      }
    }

    // Queued. When the phase waits for a Y, the descriptors of its transfers are written before the wait, while the
    // array still works on the phases before it, if every shim tile has them free besides those the phases before
    // still hold; after the wait only the queues are pushed. Writing a descriptor (a block write and a patch of its
    // address) costs the firmware some 0.3 µs, and a phase has some 64 (doc/XDNA2.md, "Descriptors before the wait").
    // Otherwise, or without pEarly, every descriptor is written right before it is queued.
    aTasks.assign(aPlanned.size(), XdnaShimTask());
    bool vPrepared = false;
    if (vWait) {
      std::vector<uint32_t> aNeeded(pOverlay.vColumns, 0);
      bool                  vFits = pEarly;
      for (const XdnaPlanned & vPlanned : aPlanned) {
        vFits = vFits && vPlanned.vTransfer.vColumn < aNeeded.size();
        if (vFits) {
          aNeeded[vPlanned.vTransfer.vColumn]++;
        }
      }
      for (uint32_t c = 0; vFits && c < aNeeded.size(); c++) {
        vFits = aNeeded[c] <= vDma.mFreeDescriptors(uint8_t(c));
      }
      if (vFits) {
        for (size_t i = 0; i < aPlanned.size(); i++) {
          if (!vDma.mPrepare(aPlanned[i].vTransfer, aTasks[i], pError)) {
            return false;
          }
        }
        vPrepared = true;
      }
      fWaitAll(p);
    }
    // With pFillsFirst the fills go before the drains: the cores take their next objects as soon as they come, while
    // a drain is needed only when a block of Y is done, several microseconds later. In the order planned, the 16 drains
    // went first (a write of the register of the controller of the tokens and a push each, after the wait, and their
    // descriptors too at the start of a command), while the cores waited for data (doc/XDNA2.md, "The transitions").
    std::vector<size_t> aOrder;
    for (int vPass = 0; vPass < 2; vPass++) {
      for (size_t i = 0; i < aPlanned.size(); i++) {
        if (!pFillsFirst ? vPass == 0 : aPlanned[i].vDrain == (vPass == 1)) {
          aOrder.push_back(i);
        }
      }
    }
    for (size_t i : aOrder) {
      if (!vPrepared && !vDma.mPrepare(aPlanned[i].vTransfer, aTasks[i], pError)) {
        return false;
      }
      vDma.mStart(aTasks[i]);
      (aPlanned[i].vDrain ? aDrains : aFills).push_back(aTasks[i]);
    }
  }
  fWaitAll(pCommand.vPhases);
  aInstructions = vInstructions.mFinish(pOverlay.vColumns, pOverlay.vRows, pOverlay.vMemoryRows);
  return true;
}

void fXdnaChainWriteParameters(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand,
                                uint8_t * pBuffer) {
  int32_t aParams[cParamsRows][cParamsColumns] = {};
  for (uint32_t p = 0; p < pCommand.vPhases; p++) {
    const XdnaChainPhase &  vPhase  = pCommand.aPhases[p];
    const XdnaChainPlace &  vPlace  = pCommand.aPlaces[p];
    const XdnaChainMatrix & vMatrix = pCommand.aMatrices[vPhase.vMatrix];
    int32_t *                vRow    = aParams[p];
    int32_t                  vEps    = 0;
    if (vPhase.vRecurrence.vLinear != 0) {
      // version 9: the 3 objects of a layer of linear attention, copied from X[1] on, the objects of the states of the
      // head of a core, and the convolution and the gated norm
      vRow[cItemPrologue]   = int32_t(cPrologueLinear);
      vRow[cItemObjects]    = int32_t(cLinearObjects);
      vRow[cItemRows]       = 1;
      vRow[cItemRecurrence] = int32_t(cRecurrenceObjects);
      vRow[cItemLinear]     = 1;
      continue;
    }
    if (vPhase.vRecurrence.vHeads != 0) {
      // the inputs of the heads of a core, copied as they are (the prologue of X in bfp16), and the objects of their
      // states
      vRow[cItemPrologue]   = int32_t(XdnaChainPrologue::cBfp16);
      vRow[cItemObjects]    = int32_t(vPlace.vObjectsPrologue);
      vRow[cItemRows]       = 1;
      vRow[cItemRecurrence] = int32_t(vPlace.vObjectsX * cRecurrenceObjects);
      continue;
    }
    // the mean of the squares of a norm is their sum times 1 / K (K padded, as the zeros of the padding add nothing)
    const float vInverse = 1.0f / float(vMatrix.vPaddedK);
    std::memcpy(&vRow[cItemInverse], &vInverse, sizeof(vInverse));
    std::memcpy(&vEps, &vPhase.vEps, sizeof(vEps));
    vRow[cItemPrologue] = int32_t(vPhase.vPrologue);
    vRow[cItemObjects]  = int32_t(vPlace.vObjectsPrologue);
    vRow[cItemRows]     = int32_t(vPhase.vRows);
    vRow[cItemTiles]    = int32_t(vMatrix.vTilesCore);
    vRow[cItemEpilogue] = int32_t(vPhase.vEpilogue);
    vRow[cItemResidual] = vPhase.vEpilogue == XdnaChainEpilogue::cResidual ? 1 : 0;
    vRow[cItemEps]      = vEps;
    vRow[cItemValues]   = int32_t(vMatrix.vPaddedK / 2);
    for (uint32_t g = 0; g < vPhase.vSegments; g++) {
      const XdnaChainSegment & vSegment  = vMatrix.aSegments[vPhase.vFirstSegment + g];
      vRow[cItemSegments + 4 * g]     = int32_t(vSegment.vBlocks);
      vRow[cItemSegments + 4 * g + 1] = int32_t(vSegment.vObjects);
      vRow[cItemSegments + 4 * g + 2] = int32_t(vSegment.vFormat);
    }
  }
  aParams[cParamsRows - 1][0] = int32_t(pCommand.vPhases);
  // the same in the halves of both cores
  std::memcpy(pBuffer, aParams, sizeof(aParams));
  std::memcpy(pBuffer + pOverlay.vCoreBytes, aParams, sizeof(aParams));
}

bool fXdnaChainWriteX(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                       const float * pX, size_t pStride, uint8_t * pBuffer) {
  const XdnaChainPhase & vPhase = pCommand.aPhases[pPhase];
  const XdnaChainPlace & vPlace = pCommand.aPlaces[pPhase];
  const uint32_t          vK     = vPlace.vK;
  const uint32_t          vHalf  = vPlace.vPaddedK / 2;
  uint8_t *               vAt    = pBuffer + vPlace.vOffsetX;
  bool                    vFinite = true;
  if (vPhase.vPrologue != XdnaChainPrologue::cBfp16) {
    // object m is [chunk m of the first half of K | chunk m of the second half], 1152 floats each, zeros past K
    for (uint32_t vObject = 0; vObject < vPlace.vObjectsX; vObject++) {
      for (uint32_t vCore = 0; vCore < 2; vCore++) {
        float *        vTo    = reinterpret_cast<float *>(vAt + vObject * pOverlay.vObjectBytes + vCore * pOverlay.vCoreBytes);
        const uint64_t vFirst = uint64_t(vCore) * vHalf + uint64_t(vObject) * cChunk;
        const uint64_t vEnd   = std::min<uint64_t>(uint64_t(vCore) * vHalf + std::min<uint64_t>(uint64_t(vObject + 1) * cChunk, vHalf), vK);
        const uint32_t vCount = vEnd > vFirst ? uint32_t(vEnd - vFirst) : 0;
        if (vCount > 0) {
          std::memcpy(vTo, pX + vFirst, size_t(vCount) * sizeof(float));
          for (uint32_t i = 0; i < vCount; i++) {
            vFinite = vFinite && std::isfinite(vTo[i]);
          }
        }
        std::memset(vTo + vCount, 0, size_t(cChunk - vCount) * sizeof(float));
      }
    }
    return vFinite;
  }
  // in bfp16: the X of every core is [K / 2 / 8 groups][rows * 9 bytes, padded to a multiple of 4], cut into halves
  // of objects; written for one core at a time and then put in its halves
  const uint32_t       vGroupBytes = fGroupBytesX(vPhase.vRows);
  std::vector<uint8_t> aCore(uint64_t(vPlace.vObjectsX) * pOverlay.vCoreBytes);
  for (uint32_t vCore = 0; vCore < 2; vCore++) {
    std::fill(aCore.begin(), aCore.end(), uint8_t(0));
    for (uint32_t vGroup = 0; vGroup < vHalf / cBlock; vGroup++) {
      const uint64_t vColumn = uint64_t(vCore) * vHalf + uint64_t(vGroup) * cBlock;
      uint8_t *      vTo     = aCore.data() + uint64_t(vGroup) * vGroupBytes;
      if (vColumn >= vK) {
        continue;  // zeros, already there
      }
      if (vColumn + cBlock <= vK) {
        vFinite = fXdnaBfp16EncodeNearest(pX + vColumn, pStride, vPhase.vRows, vTo) && vFinite;
        continue;
      }
      for (uint32_t vRow = 0; vRow < vPhase.vRows; vRow++) {
        float aValues[cBlock] = {};
        for (uint32_t i = 0; vColumn + i < vK; i++) {
          aValues[i] = pX[vRow * pStride + vColumn + i];
        }
        vFinite = fXdnaBfp16Encode(aValues, vTo + vRow * cBlockBytes, XdnaRounding::cNearest) && vFinite;
      }
    }
    for (uint32_t vObject = 0; vObject < vPlace.vObjectsX; vObject++) {
      std::memcpy(vAt + vObject * pOverlay.vObjectBytes + vCore * pOverlay.vCoreBytes,
                  aCore.data() + uint64_t(vObject) * pOverlay.vCoreBytes, pOverlay.vCoreBytes);
    }
  }
  return vFinite;
}

void fXdnaChainWriteResidual(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                              const float * pResidual, uint8_t * pBuffer) {
  const XdnaChainPhase &   vPhase     = pCommand.aPhases[pPhase];
  const XdnaChainPlace &   vPlace     = pCommand.aPlaces[pPhase];
  const XdnaChainSegment & vSegment   = pCommand.aMatrices[vPhase.vMatrix].aSegments[vPhase.vFirstSegment];
  const uint32_t            vPerStream = vSegment.vPadded / pOverlay.mStreams();
  // the outputs of stream s, in the half of core 0 of its object; zeros past the last output
  for (uint32_t s = 0; s < pOverlay.mStreams(); s++) {
    float *        vTo    = reinterpret_cast<float *>(pBuffer + vPlace.vOffsetResidual + uint64_t(s) * pOverlay.vObjectBytes);
    const uint64_t vFirst = uint64_t(s) * vPerStream;
    const uint32_t vCount = vFirst >= vSegment.vOutputs ? 0 : uint32_t(std::min<uint64_t>(vPerStream, vSegment.vOutputs - vFirst));
    if (vCount > 0) {
      std::memcpy(vTo, pResidual + vFirst, size_t(vCount) * sizeof(float));
    }
    std::memset(vTo + vCount, 0, size_t(vPerStream - vCount) * sizeof(float));
  }
}

void fXdnaChainWriteNorm(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                          const float * pWeights, uint8_t * pBuffer) {
  const XdnaChainPlace & vPlace = pCommand.aPlaces[pPhase];
  const uint32_t          vHalf  = vPlace.vPaddedK / 2;
  // as X in floats: object m is [chunk m of the first half | chunk m of the second half], zeros past the half
  for (uint32_t vObject = 0; vObject < vPlace.vObjectsX; vObject++) {
    for (uint32_t vCore = 0; vCore < 2; vCore++) {
      float *        vTo    = reinterpret_cast<float *>(pBuffer + vPlace.vOffsetNorm + vObject * pOverlay.vObjectBytes +
                                                        vCore * pOverlay.vCoreBytes);
      const uint64_t vInHalf = uint64_t(vObject) * cChunk;
      const uint32_t vCount  = vInHalf >= vHalf ? 0 : uint32_t(std::min<uint64_t>(cChunk, vHalf - vInHalf));
      if (vCount > 0) {
        std::memcpy(vTo, pWeights + uint64_t(vCore) * vHalf + vInHalf, size_t(vCount) * sizeof(float));
      }
      std::memset(vTo + vCount, 0, size_t(cChunk - vCount) * sizeof(float));
    }
  }
}

bool fXdnaChainWriteRecurrence(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand,
                                uint32_t pPhase, const XdnaChainRecurrenceIo & pIo, uint8_t * pBuffer) {
  const XdnaChainPlace & vPlace   = pCommand.aPlaces[pPhase];
  const uint32_t          vStreams = pOverlay.mStreams();
  const uint32_t          vPerCore = vPlace.vObjectsX;
  const size_t            vBytes   = cRecurrenceDim * sizeof(float);
  if (pIo.vQ == nullptr || pIo.vK == nullptr || pIo.vV == nullptr || pIo.vDecay == nullptr || pIo.vBeta == nullptr ||
      pIo.vHeadsQK == 0) {
    return false;
  }
  bool vFinite = true;
  // head 32 h + 2 s + c in the half of core c of object h of stream s
  for (uint32_t s = 0; s < vStreams; s++) {
    for (uint32_t h = 0; h < vPerCore; h++) {
      for (uint32_t c = 0; c < 2; c++) {
        const uint64_t vHead = uint64_t(h) * 2 * vStreams + 2 * s + c;
        float *        vTo   = reinterpret_cast<float *>(pBuffer + vPlace.vOffsetX +
                                                (uint64_t(s) * vPerCore + h) * pOverlay.vObjectBytes +
                                                uint64_t(c) * pOverlay.vCoreBytes);
        std::memcpy(vTo + cRecurrenceQ, pIo.vQ + (vHead % pIo.vHeadsQK) * pIo.vStrideQ, vBytes);
        std::memcpy(vTo + cRecurrenceK, pIo.vK + (vHead % pIo.vHeadsQK) * pIo.vStrideK, vBytes);
        const float   vBeta = pIo.vBeta[vHead];
        const float * vV    = pIo.vV + vHead * pIo.vStrideV;
        for (uint32_t i = 0; i < cRecurrenceDim; i++) {
          vTo[cRecurrenceBetaV + i] = vBeta * vV[i];
        }
        vTo[cRecurrenceDecay]     = pIo.vDecay[vHead];
        vTo[cRecurrenceBetaDecay] = vBeta * pIo.vDecay[vHead];
        vFinite = vFinite && std::isfinite(vTo[cRecurrenceDecay]) && std::isfinite(vTo[cRecurrenceBetaDecay]);
      }
    }
  }
  return vFinite;
}

namespace {

// The channel of the convolution that core 1 of stream s takes as its channel c of 512: q and k of the head of keys s,
// v of head s and v of head s + 16
inline uint32_t fLinearChannel(uint32_t pStream, uint32_t pChannel) {
  const uint32_t vPart = pChannel / cRecurrenceDim;
  const uint32_t vAt   = pChannel % cRecurrenceDim;
  const uint32_t vKeys = cXdnaChainLinearKeys * cRecurrenceDim;
  switch (vPart) {
    case 0:
      return pStream * cRecurrenceDim + vAt;
    case 1:
      return vKeys + pStream * cRecurrenceDim + vAt;
    case 2:
      return 2 * vKeys + pStream * cRecurrenceDim + vAt;
    default:
      return 2 * vKeys + (pStream + cXdnaChainLinearKeys) * cRecurrenceDim + vAt;
  }
}

}  // namespace

bool fXdnaChainWriteLinear(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                            const XdnaChainLinearIo & pIo, uint8_t * pBuffer) {
  const XdnaChainPlace & vPlace   = pCommand.aPlaces[pPhase];
  const uint32_t          vStreams = pOverlay.mStreams();
  if (pIo.vInput == nullptr || pIo.vStride < cXdnaChainLinearChannels || pIo.vZ == nullptr ||
      pIo.vDecay == nullptr || pIo.vBeta == nullptr) {
    return false;
  }
  // E of the gated norm: the outputs of the recurrence lack the 1 / sqrt(128) of ggml (see the kernel)
  const float vEpsGated = pIo.vEpsGated * float(cRecurrenceDim * cRecurrenceDim);
  bool        vFinite   = std::isfinite(vEpsGated) && std::isfinite(pIo.vEpsL2);
  for (uint32_t s = 0; s < vStreams; s++) {
    uint8_t * vObject1 = pBuffer + vPlace.vOffsetX + uint64_t(s) * 2 * pOverlay.vObjectBytes;
    uint8_t * vObject2 = vObject1 + pOverlay.vObjectBytes;
    for (uint32_t c = 0; c < 2; c++) {
      // the scalars of the head of the core
      const uint32_t vHead     = s + c * vStreams;
      float          aScalars[cLinearScalars] = {pIo.vBeta[vHead], pIo.vDecay[vHead], pIo.vBeta[vHead] * pIo.vDecay[vHead],
                                                 vEpsGated, pIo.vEpsL2};
      for (float vValue : aScalars) {
        vFinite = vFinite && std::isfinite(vValue);
      }
      std::memcpy(vObject1 + uint64_t(c) * pOverlay.vCoreBytes, aScalars, sizeof(aScalars));
    }
    // core 1: x of the channels of the pair, their old states by tap, and z of heads s + 16 and s
    uint16_t * vInput  = reinterpret_cast<uint16_t *>(vObject1 + pOverlay.vCoreBytes + cLinearInput);
    uint16_t * vStates = reinterpret_cast<uint16_t *>(vObject1 + pOverlay.vCoreBytes + cLinearStates);
    const float * vToken = pIo.vInput + (cXdnaChainLinearTaps - 1) * pIo.vStride;
    for (uint32_t i = 0; i < cLinearChannels; i++) {
      const uint32_t vChannel = fLinearChannel(s, i);
      const float    vX       = vToken[vChannel];
      vFinite                 = vFinite && std::isfinite(vX);
      vInput[i]               = fXdnaBf16FromFloat(vX);
      for (uint32_t t = 0; t + 1 < cXdnaChainLinearTaps; t++) {
        const float vState = pIo.vInput[t * pIo.vStride + vChannel];
        vFinite            = vFinite && std::isfinite(vState);
        vStates[t * cLinearChannels + i] = fXdnaBf16FromFloat(vState);
      }
    }
    uint16_t * vZ = reinterpret_cast<uint16_t *>(vObject2 + pOverlay.vCoreBytes);
    for (uint32_t i = 0; i < cRecurrenceDim; i++) {
      const float vNear = pIo.vZ[uint64_t(s + vStreams) * cRecurrenceDim + i];
      const float vFar  = pIo.vZ[uint64_t(s) * cRecurrenceDim + i];
      vFinite           = vFinite && std::isfinite(vNear) && std::isfinite(vFar);
      vZ[i]                         = fXdnaBf16FromFloat(vNear);
      vZ[cLinearZ / 2 + i]          = fXdnaBf16FromFloat(vFar);
    }
  }
  return vFinite;
}

void fXdnaChainLinearMatrix(const XdnaChainOverlay & pOverlay, XdnaChainMatrix & pMatrix) {
  pMatrix         = XdnaChainMatrix();
  pMatrix.vPartW  = pOverlay.vObjectBytes;
  pMatrix.vBytesW = uint64_t(pOverlay.mStreams()) * pOverlay.vObjectBytes;
}

bool fXdnaChainPackLinear(const XdnaChainOverlay & pOverlay, const float * pConv, const float * pNorm,
                           uint8_t * pBuffer) {
  const uint32_t vStreams = pOverlay.mStreams();
  std::memset(pBuffer, 0, uint64_t(vStreams) * pOverlay.vObjectBytes);
  // the weights of the gated norm times sqrt(128) (see the kernel)
  const float vRoot   = std::sqrt(float(cRecurrenceDim));
  bool        vFinite = true;
  for (uint32_t s = 0; s < vStreams; s++) {
    uint16_t * vNorm    = reinterpret_cast<uint16_t *>(pBuffer + uint64_t(s) * pOverlay.vObjectBytes + pOverlay.vCoreBytes);
    uint16_t * vWeights = vNorm + cLinearWeights / 2;
    for (uint32_t i = 0; i < cRecurrenceDim; i++) {
      vFinite  = vFinite && std::isfinite(pNorm[i]);
      vNorm[i] = fXdnaBf16FromFloat(pNorm[i] * vRoot);
    }
    for (uint32_t i = 0; i < cLinearChannels; i++) {
      const uint32_t vChannel = fLinearChannel(s, i);
      for (uint32_t t = 0; t < cXdnaChainLinearTaps; t++) {
        const float vWeight = pConv[uint64_t(vChannel) * cXdnaChainLinearTaps + t];
        vFinite             = vFinite && std::isfinite(vWeight);
        vWeights[t * cLinearChannels + i] = fXdnaBf16FromFloat(vWeight);
      }
    }
  }
  return vFinite;
}

void fXdnaChainReadRecurrence(const XdnaChainCommand & pCommand, uint32_t pPhase, const uint8_t * pBuffer,
                               float * pOutputs, float pScale) {
  const XdnaChainPhase & vPhase = pCommand.aPhases[pPhase];
  const float *           vFrom  = reinterpret_cast<const float *>(pBuffer + pCommand.aPlaces[pPhase].vOffsetY[0]);
  const uint64_t          vCount = uint64_t(vPhase.vRecurrence.vHeads) * cRecurrenceDim;
  for (uint64_t i = 0; i < vCount; i++) {
    pOutputs[i] = vFrom[i] * pScale;
  }
}

uint32_t fXdnaChainReads(const XdnaChainCommand & pCommand, uint32_t pPhase, XdnaChainRead * aReads) {
  const XdnaChainPhase &  vPhase  = pCommand.aPhases[pPhase];
  const XdnaChainPlace &  vPlace  = pCommand.aPlaces[pPhase];
  const XdnaChainMatrix & vMatrix = pCommand.aMatrices[vPhase.vMatrix];
  if (vPhase.vRecurrence.vHeads != 0) {
    // the outputs of the recurrence, [head][128]
    aReads[0].vOffset  = vPlace.vOffsetY[0];
    aReads[0].vRecords = 1;
    aReads[0].vBytes   = vPlace.vBytesY;
    aReads[0].vStride  = vPlace.vBytesY;
    return 1;
  }
  // as fXdnaChainReadY goes: the blocks up to the last given output of every segment, all their rows, in one piece
  for (uint32_t g = 0; g < vPhase.vSegments; g++) {
    const XdnaChainSegment & vSegment = vMatrix.aSegments[vPhase.vFirstSegment + g];
    aReads[g].vOffset                  = vPlace.vOffsetY[g];
    aReads[g].vRecords                 = 1;
    aReads[g].vBytes = uint64_t((vSegment.vOutputs + cRowFloats - 1) / cRowFloats) * vPhase.vRows * cRowFloats *
                       sizeof(float);
    aReads[g].vStride = aReads[g].vBytes;
  }
  return vPhase.vSegments;
}

void fXdnaChainReadY(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                      const uint8_t * pBuffer, const XdnaGemmTarget * aTargets, uint32_t pTargets) {
  const XdnaChainPhase &  vPhase  = pCommand.aPhases[pPhase];
  const XdnaChainPlace &  vPlace  = pCommand.aPlaces[pPhase];
  const XdnaChainMatrix & vMatrix = pCommand.aMatrices[vPhase.vMatrix];
  (void) pOverlay;
  // A segment is [block][row][32 outputs], the blocks of all its streams one after the other (stream s makes the
  // blocks s * B to (s + 1) * B - 1), so output o of it is at float (o / 32 * rows + row) * 32 + o % 32: with one row,
  // at float o, and what a target takes of the segment is copied at once. Version 6 had room for 8 rows in every block,
  // [4 groups][8 rows][8 outputs]: with one row 7 of every 8 floats were sent for nothing, and the 8 that are read came
  // a cache line in four.
  for (uint32_t g = 0; g < vPhase.vSegments; g++) {
    const XdnaChainSegment & vSegment = vMatrix.aSegments[vPhase.vFirstSegment + g];
    const float *             vBase    = reinterpret_cast<const float *>(pBuffer + vPlace.vOffsetY[g]);
    const uint32_t            vRows    = vPhase.vRows;
    for (uint32_t t = 0; t < pTargets; t++) {
      const XdnaGemmTarget & vInto = aTargets[t];
      // the outputs of the segment that the target takes, in the numbering of the targets
      const uint64_t vLo = std::max<uint64_t>(vInto.vFirst, vSegment.vFirst);
      const uint64_t vHi = std::min<uint64_t>(uint64_t(vInto.vFirst) + vInto.vCount,
                                              uint64_t(vSegment.vFirst) + vSegment.vOutputs);
      if (vLo >= vHi) {
        continue;
      }
      if (vRows == 1) {
        std::memcpy(vInto.vData + (vLo - vInto.vFirst), vBase + (vLo - vSegment.vFirst), size_t(vHi - vLo) * sizeof(float));
        continue;
      }
      // several rows: block by block, every row of what the target takes of the block at once
      for (uint64_t vOut = vLo; vOut < vHi;) {
        const uint64_t vIn    = vOut - vSegment.vFirst;
        const uint64_t vBlock = vIn / cRowFloats;
        const uint64_t vEnd   = std::min<uint64_t>(vHi, uint64_t(vSegment.vFirst) + (vBlock + 1) * cRowFloats);
        for (uint32_t r = 0; r < vRows; r++) {
          std::memcpy(vInto.vData + r * vInto.vStride + (vOut - vInto.vFirst),
                      vBase + (vBlock * vRows + r) * cRowFloats + vIn % cRowFloats, size_t(vEnd - vOut) * sizeof(float));
        }
        vOut = vEnd;
      }
    }
  }
}
