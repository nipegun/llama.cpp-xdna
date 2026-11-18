#pragma once

// Matrix multiplications on the NPU, from floats to floats: the piece that ties the device layer, an overlay of
// this project and the plan of a multiplication together. It converts the operands to the block floating point
// of the NPU, in the order the array consumes them, sends the command and reads the result back.
//
// The weights are converted once and stay in a buffer of the NPU; the other operand is converted for every
// multiplication. The sizes can be any: they are rounded up to what the overlay takes at a time, with zeros.

#include "xdna-chain.h"
#include "xdna-device.h"
#include "xdna-gdnp.h"
#include "xdna-gemm.h"
#include "xdna-gemv.h"
#include "xdna-overlay.h"
#include "xdna-workers.h"

#include <cstdint>
#include <functional>
#include <map>
#include <string>
#include <tuple>

// a matrix of weights as the NPU reads it, kept in a buffer of the NPU
struct XdnaWeights {
  XdnaBuffer vBuffer;
  uint32_t    vOutputs       = 0;  // rows of the matrix it came from: the N of a multiplication
  uint32_t    vInputs        = 0;  // its columns: the K
  uint32_t    vPaddedOutputs = 0;  // the sizes it has in the buffer
  uint32_t    vPaddedInputs  = 0;
};

// a matrix of weights as the overlay for generating tokens reads it (xdna-gemv.h), kept in a buffer of the NPU: up
// to the segments of the overlay, each in a format
struct XdnaGemvWeights {
  XdnaBuffer   vBuffer;
  uint32_t      vOutputs  = 0;  // rows of the matrices it came from: the N of a multiplication
  uint32_t      vInputs   = 0;  // their columns: the K
  uint32_t      vSegments = 0;
  XdnaGemvPart aParts[cXdnaGemvSegmentsMax];  // the format and the outputs of every segment
};

// One of the matrices that a matrix of weights for generating tokens puts together, in a format: fRows(first output,
// count, where) writes count rows of it from that output on, as floats for bfp16 and as the blocks of ggml for the
// other formats. Consecutive members of one format are one segment.
struct XdnaGemvMember {
  XdnaGemvFormat                                 vFormat  = XdnaGemvFormat::cBfp16;
  uint32_t                                        vOutputs = 0;
  std::function<bool(uint32_t, uint32_t, void *)> fRows;
};

// a matrix of weights as the chained overlay for generating tokens reads it (xdna-chain.h), kept in a buffer of the
// NPU
struct XdnaChainWeights {
  XdnaBuffer      vBuffer;
  XdnaChainMatrix vMatrix;
};

// what the host gives a phase of a command of the chained overlay and takes from it (see XdnaChainPhase)
struct XdnaChainIo {
  const float *           vX        = nullptr;  // X from the host: rows of K floats, vStride apart (not with vSameXAs)
  size_t                  vStride   = 0;
  const float *           vResidual = nullptr;  // cResidual: a value per output of its segment
  const float *           vNorm     = nullptr;  // cNorm: the weights of the norm, K values
  const XdnaGemmTarget * aTargets  = nullptr;  // where its Y goes (see fXdnaChainReadY); none: it is not read
  uint32_t                vTargets  = 0;
  // a phase of the recurrence (version 8): its inputs, and where its outputs go ([head][128], times vScale; none: they
  // are not read)
  const XdnaChainRecurrenceIo * vRecurrence = nullptr;
  float *                        vOutputs    = nullptr;
  float                          vScale      = 1.0f;
  // version 9: a phase of a whole layer of linear attention: its inputs (its outputs as those of the recurrence, with a
  // scale of 1)
  const XdnaChainLinearIo * vLinear = nullptr;
};

// where the time of one multiplication went, in milliseconds
struct XdnaGemmTimes {
  double vPlanMs   = 0;  // writing the instruction stream
  double vPackMs   = 0;  // converting A
  double vNpuMs    = 0;  // from the submission of the command to its end
  double vUnpackMs = 0;  // reading C back
  double vFlushMs  = 0;  // of it, taking C out of the cache of the processor before reading it
};

// buffers of C of a runner (and of commands sent without waiting, one per buffer of C)
const uint32_t cXdnaRunnerSlotsC = 4;
// llama.cpp-xdna: buffers of A: 2 for the rows a multiplication converts piece by piece along the inputs, and from
// cXdnaRunnerFirstReadyA on, 3 for the rows converted ahead by mUnpackSwigluRows (the pieces of ffn_down)
const uint32_t cXdnaRunnerSlotsA      = 5;
const uint32_t cXdnaRunnerFirstReadyA = 2;

class XdnaRunner {
public:
  XdnaRunner() = default;
  ~XdnaRunner();
  XdnaRunner(const XdnaRunner &) = delete;
  XdnaRunner & operator=(const XdnaRunner &) = delete;

  // Opens the NPU and creates a context with the overlay (the main.pdi of its build), described by its manifest.
  // With pGemvOverlayPath and pGemvManifestPath, the overlay for generating tokens goes into the same context as a
  // second compute unit: a command for one after a command for the other costs about 2.5 ms more, as the firmware
  // loads the overlay again. That overlay is the one of xdna-gemv.h or the chained one of xdna-chain.h, as its
  // manifest says (versions 4 and 5).
  bool mOpen(const std::string & pDevicePath, const std::string & pOverlayPath, const std::string & pManifestPath,
             std::string & pError, const std::string & pGemvOverlayPath = std::string(),
             const std::string & pGemvManifestPath = std::string());
  void mClose();

  const XdnaOverlay & mOverlay() const { return vOverlay; }
  // whether the overlay for generating tokens is there, and its description
  bool                     mHasGemv() const { return vHasGemv; }
  const XdnaGemvOverlay & mGemvOverlay() const { return vGemvOverlay; }
  // the same for the chained one
  bool                      mHasChain() const { return vHasChain; }
  const XdnaChainOverlay & mChainOverlay() const { return vChainOverlay; }

  // Threads that share the conversions of the host, counting the one that calls; 1 if it is never set. They are
  // there for whoever uses the runner too, between its calls.
  void           mSetThreads(uint32_t pThreads) { vWorkers.mSetThreads(pThreads); }
  XdnaWorkers & mWorkers() { return vWorkers; }

  // For finding faults. With pOn, every multiplication first fills the buffer of C with bytes 0xff, which no sum of
  // finite values gives, so the words that still hold them afterwards are the ones the NPU did not write. It costs
  // a pass over C. With pFlush false the bytes stay in the cache of the processor, modified, as the zeros of a new
  // buffer did: the words that come back as 0xff are then also the ones the cache wrote over the results.
  void mSetFillC(bool pOn, bool pFlush = true) {
    vFillC      = pOn;
    vFlushFillC = pFlush;
  }

  // For experiments only: with pOn false, the buffers the runner creates are not flushed from the cache of the
  // processor (see mEnsure), as before 2026/11/13.
  void mSetFlushNewBuffers(bool pOn) { vFlushNew = pOn; }
  // For comparing: with pOn false, the other threads of OpenMP are not put to sleep while the NPU works (see
  // XdnaWorkers::mWhileParked).
  void mSetPark(bool pOn) { vPark = pOn; }
  // For comparing: with pOn false, the commands of the chained overlay write every buffer descriptor right before it is
  // queued, not before the wait for the phase before (see fXdnaChainPlan). Before the first command.
  void mSetEarlyDescriptors(bool pOn) { vEarly = pOn; }
  // For comparing: with pOn false, a command of the chained overlay flushes the cache with the SYNC_BO of the driver,
  // what the host wrote before it and the whole Y it reads after it, not with fXdnaFlushCache on the lines the host
  // wrote and those it reads (see mRunChain).
  void mSetUserFlush(bool pOn) { vUserFlush = pOn; }
  // For comparing: with pOn false, a phase of the chained overlay queues the drains of its Y before the transfers that
  // feed its cores, as before (see fXdnaChainPlan). Before the first command.
  void mSetFillsFirst(bool pOn) { vFillsFirst = pOn; }

  // A buffer of the driver of at least pBytes (whole pages) that the host and the NPU share, for data that lives as
  // long as its user wants: the recurrent states of a model. It leaves the cache of the processor before it is given
  // (see mEnsure). It is freed with mFreeShared, before the runner is closed.
  bool mCreateShared(uint64_t pBytes, XdnaBuffer & pBuffer, std::string & pError);
  void mFreeShared(XdnaBuffer & pBuffer);

  // llama.cpp-xdna: how many times a command ran on the other overlay than the command before it since the NPU was
  // opened: every change makes the firmware load the overlay again, some 2.5 ms
  uint64_t mOverlaySwitches() const { return vSwitches; }

  // The buffer of C as the NPU left it in the last multiplication, in the order the array writes it, and its size.
  const uint8_t * mRawC(uint64_t & pBytes) const;

  // The addresses the NPU used for A and for C in the last multiplication.
  uint64_t mDevAddrA() const { return aBufferA[0].vDevAddr; }
  uint64_t mDevAddrC() const { return aBufferC[0].vDevAddr; }

  // Converts a matrix of weights, pOutputs rows of pInputs floats (one row for every output), and keeps it in a
  // buffer of the NPU.
  bool mLoadWeights(const float * pWeights, uint32_t pOutputs, uint32_t pInputs, XdnaRounding pRounding,
                    XdnaWeights & pLoaded, std::string & pError);
  void mFreeWeights(XdnaWeights & pLoaded);

  // pC[m][n] = sum over k of pA[m][k] * weights[n][k]: pA is pM rows of as many floats as the weights have
  // inputs, and pC gets pM rows of as many floats as they have outputs.
  //
  // A command that does not end is not sent again, and neither is any other: the NPU may be unusable after it.
  bool mMultiply(const float * pA, uint32_t pM, const XdnaWeights & pWeights, XdnaRounding pRounding, float * pC,
                 XdnaGemmTimes & pTimes, std::string & pError);

  // The same, with the outputs into pTargets targets (see XdnaGemmTarget), counted from the first output of the
  // weights.
  bool mMultiply(const float * pA, uint32_t pM, const XdnaWeights & pWeights, XdnaRounding pRounding,
                 const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaGemmTimes & pTimes, std::string & pError);

  // The same in two steps, so that rows multiplied by several pieces of weights are converted once. mPackRows
  // converts pM rows of pK floats, pStride floats apart (a range of the columns of a wider matrix when pStride is
  // larger than pK), into the buffer of A pSlotA (cXdnaRunnerSlotsA); mMultiplyPacked multiplies the rows of a slot by
  // pWeights, of pK inputs, into the targets, adding to what they hold with pAdd (the pieces of a matrix along its
  // inputs after the first). Both add their times to pTimes. With pNorm, the rows are normalised as they are
  // converted (see XdnaRowNorm: its scales for the pM rows, its weights for the pK columns).
  bool mPackRows(const float * pA, uint32_t pM, uint32_t pK, size_t pStride, XdnaRounding pRounding,
                 XdnaGemmTimes & pTimes, std::string & pError, uint32_t pSlotA = 0,
                 const XdnaRowNorm * pNorm = nullptr);
  // pSlotC says which of the buffers of C the NPU writes (cXdnaRunnerSlotsC); with pUnpack false, C stays there for
  // mUnpackSwiglu and the targets are not written.
  bool mMultiplyPacked(const XdnaWeights & pWeights, const XdnaGemmTarget * aTargets, uint32_t pTargets, bool pAdd,
                       XdnaGemmTimes & pTimes, std::string & pError, uint32_t pSlotC = 0, bool pUnpack = true,
                       uint32_t pSlotA = 0);

  // The same without waiting, so that the host converts and reads while the NPU works: mSubmitPacked sends the
  // multiplication of the rows of the slot pSlotA by pWeights, which writes the buffer of C pSlotC, and returns at
  // once; mFinishPacked waits for it and reads its C into the targets (adding with pAdd), or keeps it in its slot for
  // mUnpackSwiglu with pUnpack false. One at a time per buffer of C, finished in the order they were sent (the driver
  // keeps 4 per context); the slots of A and C of one cannot be used for anything else until it is finished.
  struct XdnaPending {
    bool           vValid  = false;
    uint64_t       vSeq    = 0;
    uint32_t       vSlotC  = 0;
    uint64_t       vBytesC = 0;
    XdnaGemmShape vShape;
  };
  bool mSubmitPacked(const XdnaWeights & pWeights, uint32_t pSlotA, uint32_t pSlotC, XdnaPending & pPending,
                     XdnaGemmTimes & pTimes, std::string & pError);
  bool mFinishPacked(XdnaPending & pPending, const XdnaGemmTarget * aTargets, uint32_t pTargets, bool pAdd,
                     XdnaGemmTimes & pTimes, std::string & pError, bool pUnpack = true);
  // Reads the C left in the slots pGate and pUp by two multiplications of the same shape (the gate and the up of a
  // feed-forward layer) and writes silu(gate) * up into pTarget, as the SwiGLU of ggml (see fXdnaGemmUnpackSwiglu,
  // with pRowFirst and pRowLength for a part of the rows of the SwiGLU).
  bool mUnpackSwiglu(uint32_t pGate, uint32_t pUp, const XdnaGemmTarget & pTarget, XdnaGemmTimes & pTimes,
                     std::string & pError, uint64_t pRowFirst = 0, uint64_t pRowLength = 0);
  // llama.cpp-xdna: the rows of a multiplication converted ahead into the buffer of A pSlotA: mPrepareRowsA makes it
  // ready for pM rows of pK inputs (nothing in it is usable then), mUnpackSwigluRows writes its parts, and mReadyRowsA
  // makes it usable by mSubmitPacked (out of the cache of the processor)
  bool mPrepareRowsA(uint32_t pSlotA, uint32_t pM, uint32_t pK, std::string & pError);
  bool mReadyRowsA(uint32_t pSlotA, XdnaGemmTimes & pTimes, std::string & pError);
  // llama.cpp-xdna: mUnpackSwiglu with the SwiGLU converted straight into the rows of the multiplication that reads it
  // (fXdnaGemmPackSwiglu) instead of written as floats: its pieces along the inputs, piece i from column aFirsts[i] of
  // the SwiGLU into the buffer of A aSlotsA[i], made ready by mPrepareRowsA; mSwigluRowsFit says whether a C of
  // pOutputs columns from pRowFirst can go so into pieces of aCounts inputs
  bool mUnpackSwigluRows(uint32_t pGate, uint32_t pUp, const uint32_t * aSlotsA, const uint64_t * aFirsts,
                         uint32_t pPieces, XdnaRounding pRounding, XdnaGemmTimes & pTimes, std::string & pError,
                         uint64_t pRowFirst, uint64_t pRowLength);
  bool mSwigluRowsFit(uint32_t pM, uint64_t pRowFirst, uint64_t pOutputs, const uint64_t * aFirsts,
                      const uint32_t * aCounts, uint32_t pPieces) const;
  // llama.cpp-xdna: the gated norm of Qwen3.5 converted ahead into the buffer of A pSlotA as the rows of the
  // multiplication that reads it (fXdnaGemmPackGatedNorm): pM rows of the columns pFirst to pFirst + pK, from the
  // outputs pO of the recurrence, z (pZ), the weights pW of pHead values and pEps; made usable by mSubmitPacked
  bool mPackGatedNormRows(uint32_t pSlotA, uint32_t pM, uint32_t pK, uint64_t pFirst, const float * pO,
                          uint64_t pStrideO, const float * pZ, uint64_t pStrideZ, const float * pW, uint32_t pHead,
                          float pEps, XdnaRounding pRounding, XdnaGemmTimes & pTimes, std::string & pError);

  // llama.cpp-xdna: the attention of a batch on the overlay of the prompts (see XdnaAttnRows in xdna-gemm.h).
  // mPackAttnQ converts the rows of Q of a group, pDim values each, into the buffer of A pSlotA and makes it usable by
  // mSubmitPacked. mLoadAttnB converts halves into the weights pWeights, pOutputs by pInputs, the first pValidInputs
  // of them from pData (see fXdnaGemmPackBHalves), keeping their buffer when it is large enough: no command that reads
  // them can be in flight. mPackAttnP converts P from the scores kept in the buffer of C pSlotS (mFinishPacked with
  // pUnpack false) into the buffers of A aSlotsA, made ready by mPrepareRowsA for the rows of the scores and the keys of
  // every piece (from aFirsts[i]), writes the sums of the rows of P into pSums and makes those buffers usable.
  bool mPackAttnQ(uint32_t pSlotA, const XdnaAttnRows & pRows, uint32_t pDim, XdnaRounding pRounding,
                  XdnaGemmTimes & pTimes, std::string & pError);
  bool mLoadAttnB(const uint16_t * pData, uint64_t pStrideN, uint64_t pStrideK, uint32_t pOutputs, uint32_t pInputs,
                  uint32_t pValidInputs, XdnaRounding pRounding, XdnaWeights & pWeights, XdnaGemmTimes & pTimes,
                  std::string & pError, uint32_t pSplitDim = 0);
  bool mPackAttnP(uint32_t pSlotS, const XdnaAttnScores & pScores, const uint32_t * aSlotsA, const uint64_t * aFirsts,
                  uint32_t pPieces, float * pSums, XdnaRounding pRounding, XdnaGemmTimes & pTimes,
                  std::string & pError);

  // The same for the overlay for generating tokens: the weights in its own layout, and 1 to 8 rows of X, pStride
  // floats apart, rounded to the nearest. Its instruction streams are kept by size, since generating repeats them.
  bool mLoadGemvWeights(const float * pWeights, uint32_t pOutputs, uint32_t pInputs, XdnaGemvWeights & pLoaded,
                        std::string & pError);
  // The same for the members aMembers, one after another, with pInputs columns each (a multiple of 256 when a
  // member is in a format of ggml, whose blocks the cores decode themselves): converted one segment and stream at a
  // time, so that only the rows of one stream are read at once (the output matrix of Qwen3.5 4B would be 2.5 GB of
  // floats whole). False when there are more segments than the overlay takes.
  bool mLoadGemvMatrix(const std::vector<XdnaGemvMember> & aMembers, uint32_t pInputs, XdnaGemvWeights & pLoaded,
                       std::string & pError);
  void mFreeGemvWeights(XdnaGemvWeights & pLoaded);
  bool mMultiplyRows(const float * pX, uint32_t pRows, size_t pStride, const XdnaGemvWeights & pWeights,
                     const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaGemmTimes & pTimes, std::string & pError);

  // The same for the chained overlay. With pApart every member is a segment of its own (4 at most), as a SwiGLU
  // needs its gate and its up; without it, consecutive members of one format are one segment.
  bool mLoadChainMatrix(const std::vector<XdnaGemvMember> & aMembers, uint32_t pInputs, bool pApart,
                        XdnaChainWeights & pLoaded, std::string & pError);
  void mFreeChainWeights(XdnaChainWeights & pLoaded);
  // version 9: the fixed objects of a phase of a whole layer of linear attention, from the weights of its convolution
  // ([8192][4] floats) and of its gated norm (128 floats), as the matrix of the phase
  bool mLoadChainLinear(const float * pConv, const float * pNorm, XdnaChainWeights & pLoaded, std::string & pError);
  // 1 to 8 rows of X, pStride floats apart, by a matrix: one command, of a phase for every two segments, with X in
  // floats for one row (the cores round it) and in bfp16 rounded to the nearest by the host for more
  bool mChainMultiplyRows(const float * pX, uint32_t pRows, size_t pStride, const XdnaChainWeights & pWeights,
                          const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaGemmTimes & pTimes,
                          std::string & pError);
  // A command of pPhases phases (aPhases, with what the host gives each and takes from each in aIo) that run the
  // matrices aWeights (pMatrices of them, buffers 1 to 4 of the command) and, version 8, the recurrence on the states
  // of the buffers aStates (pStates of them, buffers of the driver after the matrices). Its instruction stream is kept
  // by its key. The states are not flushed from the cache of the processor: the caller makes sure that no line of them
  // is modified in it before the command, and that none is in it before the processor reads them after it.
  bool mRunChain(const XdnaChainPhase * aPhases, const XdnaChainIo * aIo, uint32_t pPhases,
                 const XdnaChainWeights * const * aWeights, uint32_t pMatrices, XdnaGemmTimes & pTimes,
                 std::string & pError, const XdnaBuffer * const * aStates = nullptr, uint32_t pStates = 0);

  // llama.cpp-xdna: the recurrence of a gated delta net over a batch of tokens on the overlay of the prompts, when
  // its manifest says "gdn 1" (see xdna-gdnp.h), which reads the states and writes the outputs and the final states.
  // Its rounds go in up to mSetGdnParts commands (as many as divide them), all in flight at once: the host packs a
  // part while the NPU works on the ones before, and reads a part while the NPU works on the ones after. Every
  // command has the same instruction stream, kept by rounds and groups, and its part of the same three buffers.
  bool mCanRunGdn() const { return vIsOpen && vOverlay.vGdn == 1; }
  bool mRunGdn(const XdnaGdnIo & pIo, XdnaGemmTimes & pTimes, std::string & pError);
  // 1 to cXdnaRunnerSlotsC; 4 when it is never set
  void mSetGdnParts(uint32_t pParts) { vGdnParts = pParts; }

private:
  // llama.cpp-xdna: counts a change of overlay when pCommand runs on other compute units than the command before
  void mNoteUnit(const XdnaBuffer & pCommand);
  // makes a buffer at least pBytes long, keeping it when it already is
  bool mEnsure(XdnaBuffer & pBuffer, XdnaBufferType pType, uint64_t pBytes, std::string & pError);
  // submits the command in vCommand and waits for it; false when it could not be submitted
  bool mSubmitAndWait(const std::vector<uint32_t> & aHandles, XdnaWaitResult & pWaited, std::string & pError);
  // the checks, the instruction stream, the buffer of C and the command in pCommand of a multiplication of the rows
  // of the slot pSlotA by pWeights into the buffer of C pSlotC, with the buffers it uses in aHandles
  bool mPrepare(const XdnaWeights & pWeights, uint32_t pSlotA, uint32_t pSlotC, XdnaBuffer & pCommand,
                XdnaGemmShape & pShape, uint64_t & pBytesC, std::vector<uint32_t> & aHandles, XdnaGemmTimes & pTimes,
                std::string & pError);
  // after waiting for the command in pCommand: whether it completed (if not, nothing else is sent: vBroken)
  bool mCompleted(const XdnaBuffer & pCommand, XdnaWaitResult pWaited, std::string & pError);
  // flushes the buffer of C pSlotC and reads it into the targets, or keeps it with pUnpack false
  bool mReadC(uint32_t pSlotC, const XdnaGemmShape & pShape, uint64_t pBytesC, const XdnaGemmTarget * aTargets,
              uint32_t pTargets, bool pAdd, bool pUnpack, XdnaGemmTimes & pTimes, std::string & pError);

  XdnaWorkers vWorkers;
  XdnaDevice  vDevice;
  XdnaOverlay vOverlay;
  XdnaContext vContext;
  XdnaBuffer  vHeap;          // the device memory of the process
  XdnaBuffer  vPdi;           // the overlay, in it
  XdnaBuffer  vCommand;
  XdnaBuffer  aCommandAsync[cXdnaRunnerSlotsC];  // of the multiplications sent with mSubmitPacked, one per buffer of C
  XdnaBuffer  aBufferA[cXdnaRunnerSlotsA];
  XdnaBuffer  aBufferC[cXdnaRunnerSlotsC];
  uint64_t     vBytesC = 0;      // of the last multiplication
  // what each buffer of C holds when a multiplication left it there for a later reading (mUnpackSwiglu)
  XdnaGemmShape aShapeC[cXdnaRunnerSlotsC];
  bool           aKeptC[cXdnaRunnerSlotsC] = {};
  // what is in each buffer of A: the rows and the inputs it was converted for, as given and as padded
  struct XdnaPackedA {
    uint32_t vM       = 0;
    uint32_t vK       = 0;
    uint32_t vPaddedM = 0;
    uint32_t vPaddedK = 0;
    uint64_t vBytes   = 0;
  };
  XdnaPackedA aPackedA[cXdnaRunnerSlotsA];
  XdnaPackedA aPreparedA[cXdnaRunnerSlotsA];  // llama.cpp-xdna: what mPrepareRowsA made a buffer ready for
  // the instruction streams of the overlay of the prompts, by padded M, K and N, each in a buffer of the heap, with
  // its number of words and the bytes of its buffer of C
  struct XdnaGemmCached {
    XdnaBuffer vBuffer;
    uint32_t    vWords  = 0;
    uint64_t    vBytesC = 0;
  };
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, XdnaGemmCached> aGemmPlans;
  // those of the recurrence, by rounds and groups, with the bytes of its three buffers, and those buffers
  struct XdnaGdnCached {
    XdnaBuffer vBuffer;
    uint32_t    vWords  = 0;
    uint64_t    vBytesA = 0;
    uint64_t    vBytesB = 0;
    uint64_t    vBytesC = 0;
  };
  std::map<std::pair<uint32_t, uint32_t>, XdnaGdnCached> aGdnPlans;
  XdnaBuffer                                             vGdnA;
  XdnaBuffer                                             vGdnB;
  XdnaBuffer                                             vGdnC;

  // the overlay for generating tokens: compute unit 1 of the context
  XdnaGemvOverlay vGemvOverlay;
  bool             vHasGemv = false;
  XdnaBuffer      vPdiGemv;
  XdnaBuffer      vBufferX;
  XdnaBuffer      vBufferY;
  // its instruction streams, by rows, K and the format and outputs of every segment, each in a buffer of the heap,
  // with its number of words
  std::map<std::vector<uint32_t>, std::pair<XdnaBuffer, uint32_t>> aGemvPlans;

  // or the chained overlay, in its place
  XdnaChainOverlay vChainOverlay;
  bool              vHasChain = false;
  XdnaBuffer       vBufferIo;       // the input and output of the host of its commands of one phase
  XdnaBuffer       vBufferIoChain;  // and of those of more phases
  // its instruction streams, by the key of their command (fXdnaChainKey), as those of the other one
  std::map<std::vector<uint32_t>, std::pair<XdnaBuffer, uint32_t>> aChainPlans;

  bool         vIsOpen = false;
  bool         vBroken = false;  // a command did not end
  bool         vFillC      = false;
  bool         vFlushFillC = true;
  bool         vFlushNew   = true;
  bool         vPark       = true;
  bool         vEarly      = true;
  bool         vUserFlush  = true;
  bool         vFillsFirst = true;
  uint32_t     vGdnParts   = cXdnaRunnerSlotsC;
  uint32_t     vLastUnits  = 0;  // llama.cpp-xdna: the compute units of the last command sent
  uint64_t     vSwitches   = 0;  // and the changes of overlay (mOverlaySwitches)
};
