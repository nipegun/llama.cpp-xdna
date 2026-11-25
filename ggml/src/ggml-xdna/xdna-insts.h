#pragma once

// Instruction streams of the NPU: what a command carries, next to an overlay, to tell the array what to move and
// when. The stream is a list of register writes and waits that the firmware of the NPU carries out.
//
// The format is the transaction format of the firmware, as mlir-aie 1.4.5 writes it (read in its
// include/aie/Runtime/TxnEncoding.h), and the registers are the ones of the tiles of AIE2P that a stream of
// mlir-aie for this NPU writes (read in the streams it generates). Both are described in doc/XDNA2.md.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

// where a tile is in the array: row 0 holds the shim tiles, row 1 the memory tiles and the rest the core tiles
struct XdnaTile {
  uint8_t vColumn = 0;
  uint8_t vRow    = 0;
};

// direction of a DMA channel, with the numbers the firmware uses for it
enum class XdnaDirection : uint32_t {
  cToMemory   = 0,  // from the streams of the array into memory: in a shim tile, into a buffer of the host
  cFromMemory = 1,  // from memory into the streams of the array: in a shim tile, from a buffer of the host
};

// One instruction stream being written.
class XdnaInstructions {
 public:
  XdnaInstructions();

  // the register pRegister of pTile takes pValue
  void mWrite(XdnaTile pTile, uint32_t pRegister, uint32_t pValue);
  // the bits of pMask of the register take the ones of pValue
  void mMaskWrite(XdnaTile pTile, uint32_t pRegister, uint32_t pValue, uint32_t pMask);
  // waits until the bits of pMask of the register are the ones of pValue
  void mMaskPoll(XdnaTile pTile, uint32_t pRegister, uint32_t pValue, uint32_t pMask);
  // pCount consecutive registers, from pRegister on, take the values of pValues
  void mBlockWrite(XdnaTile pTile, uint32_t pRegister, const uint32_t * pValues, size_t pCount);
  // the register takes the address of the buffer number pBuffer of the command, plus pOffset bytes
  void mPatchAddress(XdnaTile pTile, uint32_t pRegister, uint32_t pBuffer, uint64_t pOffset);
  // waits for the token that a channel issues when it ends a task that asked for one
  void mSync(XdnaTile pTile, XdnaDirection pDirection, uint32_t pChannel);

  // pre-allocates room for at least pWords 32-bit words to avoid reallocations on large streams
  void mReserve(size_t pWords);

  // The stream, with its header. pColumns, pRows and pMemoryRows describe the array it is for.
  std::vector<uint32_t> mFinish(uint32_t pColumns, uint32_t pRows, uint32_t pMemoryRows) const;
  void                  mFinishInto(uint32_t pColumns, uint32_t pRows, uint32_t pMemoryRows,
                                    std::vector<uint32_t> & aStream);

 private:
  std::vector<uint32_t> aWords;       // the operations, with room for the header at the front
  uint32_t              vOperations;  // how many there are
};

// times the queue of a channel of a shim tile can repeat one transfer
const uint32_t cXdnaShimRepeatMax = 256;

// limits of the patterns of a buffer descriptor of a shim tile: sizes of the two inner dimensions, iterations and
// strides (in 32-bit words)
const uint32_t cXdnaShimSizeMax       = 1023;
const uint32_t cXdnaShimIterationsMax = 64;
const uint32_t cXdnaShimStrideMax     = 1u << 20;

// One transfer between a buffer of the host and a channel of a shim tile.
//
// Without a pattern it moves vBytes consecutive bytes. With one (vSizes[0] not 0), in 32-bit words as mlir-aie writes
// them: the innermost dimension takes vSizes[0] words vStrides[0] apart, the next one vSizes[1] of those vStrides[1]
// apart (0: none), the third as many of those as vBytes holds, vStrides[2] apart, and the whole goes vIterations
// times, vIterationStride words further every time (the queue of the channel runs the descriptor that many times; the
// descriptor counts them). Strides are of 1 word or more: the hardware keeps them minus one.
struct XdnaShimTransfer {
  uint8_t        vColumn    = 0;  // the shim tile
  XdnaDirection vDirection = XdnaDirection::cFromMemory;
  uint8_t        vChannel   = 0;
  uint32_t       vBuffer    = 0;  // which buffer of the command
  uint64_t       vOffset    = 0;  // bytes from its start
  uint64_t       vBytes     = 0;  // a multiple of 4; with vIterations, the bytes of one iteration
  uint32_t       vRepeat    = 1;  // times the same bytes go (1 with vIterations above 1)
  bool           vToken     = false;  // whether its end can be waited for
  uint32_t       vSizes[2]        = {0, 0};
  uint32_t       vStrides[3]      = {1, 1, 1};
  uint32_t       vIterations      = 1;
  uint32_t       vIterationStride = 1;
};

// A transfer that has a buffer descriptor, queued or not yet.
struct XdnaShimTask {
  uint8_t        vColumn     = 0;
  XdnaDirection vDirection  = XdnaDirection::cFromMemory;
  uint8_t        vChannel    = 0;
  uint8_t        vDescriptor = 0;  // the buffer descriptor it holds
  bool           vToken      = false;
  uint32_t       vRuns       = 1;  // times the queue runs the descriptor
};

// The DMA of the shim tiles, as a sequence of transfers has to see it: every tile has 16 buffer descriptors, which
// its four channels share, and every channel a queue of 4 tasks. A push onto a full queue is lost, so the stream
// waits for room whenever the queue may be full.
//
// The descriptors are handed out and the waits placed exactly as mlir-aie 1.4.5 does (the lowest free descriptor
// of the tile; a wait for room before a push when 4 pushes of the channel are not known to have ended), so that
// a sequence written here and the same one written with mlir-aie give the same stream, byte for byte.
class XdnaShimDma {
 public:
  XdnaShimDma(XdnaInstructions & pInstructions, uint32_t pColumns);

  // Queues a transfer. False, with the reason in pError, when it cannot be: no descriptor left, bad sizes.
  bool mPush(const XdnaShimTransfer & pTransfer, XdnaShimTask & pTask, std::string & pError);
  // The same in two steps: mPrepare writes the buffer descriptor of the transfer and mStart queues it, so that the
  // descriptors can be written while the array still works on what was queued before.
  bool mPrepare(const XdnaShimTransfer & pTransfer, XdnaShimTask & pTask, std::string & pError);
  void mStart(const XdnaShimTask & pTask);
  // waits for the end of a task that was queued with a token
  void mAwait(const XdnaShimTask & pTask);
  // Every task queued on a channel is known to have ended, as the caller knows that what it fed was waited for: the
  // next pushes need not wait for room in its queue.
  void mSettle(uint8_t pColumn, XdnaDirection pDirection, uint8_t pChannel);
  // gives the descriptor of a task back
  void mFree(const XdnaShimTask & pTask);
  // the buffer descriptors of a shim tile that are free
  uint32_t mFreeDescriptors(uint8_t pColumn) const;

 private:
  // what is known about the queue of a channel: its pushes not known to have ended, oldest first
  struct XdnaQueue {
    std::vector<bool> aPushes;  // true: the push issues a token
    bool              vUnknownTokens = false;
  };

  XdnaQueue & mQueue(uint8_t pColumn, XdnaDirection pDirection, uint8_t pChannel);

  XdnaInstructions &     vInstructions;
  std::vector<uint16_t>   aFreeDescriptors;  // per shim tile, one bit per descriptor
  std::vector<XdnaQueue> aQueues;           // per shim tile, direction and channel
};
