#pragma once

// llama.cpp-xdna: the mixture-of-experts overlay (xdna2-moe 1), for GGML_OP_MUL_MAT_ID and the routed experts of a
// feed-forward block (gate, up, SwiGLU and down). It multiplies C chunks of 1 to 24 rows (1 to 3 groups of 8 rows) by
// their chosen expert matrices in one command across the 16 input channels of the shims and all 32 cores, with 32-bit
// float sums and the weights decoded in the cores from bfp16, Q4_K, Q5_K or Q6_K.
//
// Design: overlays/designs/moe_experts.py; manifest writer: overlays/designs/moe_manifest.py; kernel:
// kernels/xdna2-moe.cc. Its manifest is:
//
//   xdna2-moe 1
//   array <columns> <rows> <rows of memory tiles>
//   kernel <name>
//   object <bytes of an object of a stream> <bytes of the half of a core>
//   x <bytes of X a core keeps> <bytes available for the rows of a chunk>
//   rows <maximum rows of a chunk>
//   chunks <maximum chunks of a command>
//   segments <segments of an expert>
//   chain <floats of gate a core keeps for SwiGLU>
//   formats <name>...                                                   (bfp16, q4k, q5k, q6k)
//   stream W|Y <stream> <shim column> <channel>                         (one per stream)

#include "xdna-gemm.h"
#include "xdna-gemv.h"
#include "xdna-overlay.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class XdnaWorkers;

constexpr uint32_t cXdnaMoeSegmentsMax = 2;

struct XdnaMoeOverlay {
  uint32_t                        vColumns     = 0;
  uint32_t                        vRows        = 0;
  uint32_t                        vMemoryRows  = 0;
  std::string                     vKernel;
  uint32_t                        vObjectBytes = 0;  // 9216 bytes per stream object
  uint32_t                        vCoreBytes   = 0;  // 4608 bytes per core half-object
  uint32_t                        vKeptX       = 0;  // 32256 bytes in each core
  uint32_t                        vRoomX       = 0;  // 27648 bytes for the rows of X of a chunk
  uint32_t                        vRowsMax     = 0;  // 24 rows per chunk at most
  uint32_t                        vChunksMax   = 0;  // 1136 chunks per command at most
  uint32_t                        vSegments    = 0;  // up to 2 segments per expert (gate and up)
  uint32_t                        vChainFloats = 0;  // floats of gate kept in core 0 for SwiGLU
  uint32_t                        vFormats     = 0;  // bit 1 << XdnaGemvFormat
  std::vector<XdnaOverlayStream> aStreamsW;
  std::vector<XdnaOverlayStream> aStreamsY;

  bool     mLoad(const std::string & pPath, std::string & pError);
  uint32_t mStreams() const { return uint32_t(aStreamsW.size()); }
  bool     mTakes(XdnaGemvFormat pFormat) const { return (vFormats & (1u << uint32_t(pFormat))) != 0; }
};

struct XdnaMoeSegment {
  XdnaGemvFormat vFormat       = XdnaGemvFormat::cBfp16;
  uint32_t        vOutputs      = 0;
  uint32_t        vPadded       = 0;
  uint32_t        vBlocks       = 0;  // blocks of up to 32 outputs per stream
  uint32_t        vTailGroups   = 4;  // groups of 8 outputs in the last block (1 to 4)
  uint64_t        vCoreBlock    = 0;  // bytes of the pieces of one full block on one core
  uint32_t        vObjects      = 0;  // stream objects of this segment for one expert
  uint64_t        vOffsetExpert = 0;  // byte offset inside the slice of one expert on a stream
};

// All experts of a tensor (or a {gate, up} pair when vSwiglu is true), laid out as [stream][expert][segments][objects]
// so that stream s reads one contiguous slice of vExpertBytes for any chosen expert.
struct XdnaMoeMatrix {
  uint32_t        vExperts        = 0;
  uint32_t        vK              = 0;
  uint32_t        vPaddedK        = 0;
  uint32_t        vTilesCore      = 0;  // tiles of up to 256 values of K per core
  uint32_t        vTailSubs       = 8;  // 32-value sub-blocks in the last tile of K per core (2, 4, 6 or 8)
  uint32_t        vOutputs        = 0;  // emitted outputs per row (those of segment 0; both segments match with SwiGLU)
  uint32_t        vPaddedOut      = 0;
  uint32_t        vBlocksEmit     = 0;  // blocks of 32 outputs emitted per stream and row
  uint32_t        vTailGroupsEmit = 4;  // groups of 8 valid outputs in the last emitted block (1 to 4)
  uint32_t        vSegments       = 0;
  bool            vSwiglu         = false;
  XdnaMoeSegment aSegments[cXdnaMoeSegmentsMax];
  uint64_t        vExpertBytes    = 0;  // bytes of one expert on one stream
  uint64_t        vPartW          = 0;  // bytes of all experts on one stream
  uint64_t        vBytesW         = 0;  // bytes of the whole buffer across all streams
};

bool fXdnaMoeMatrixShape(const XdnaMoeOverlay & pOverlay, uint32_t pExperts, uint32_t pK,
                          const XdnaGemvPart * aParts, uint32_t pParts, bool pSwiglu, XdnaMoeMatrix & pMatrix,
                          std::string & pError);

// Maximum rows a chunk of pMatrix can hold on pOverlay (1 to 24).
uint32_t fXdnaMoeMaxRows(const XdnaMoeOverlay & pOverlay, const XdnaMoeMatrix & pMatrix);

// Packs segment pSegment of expert pExpert (pMatrix.aSegments[pSegment].vOutputs rows, pRowBytes apart) into pBuffer.
// Safe to call in parallel for distinct pExpert values.
bool fXdnaMoePackExpertSegment(const XdnaMoeOverlay & pOverlay, const XdnaMoeMatrix & pMatrix, uint32_t pExpert,
                                uint32_t pSegment, const void * pRows, uint64_t pRowBytes, uint8_t * pBuffer);

struct XdnaMoeChunk {
  uint32_t vExpert = 0;
  uint32_t vRows   = 0;
};

struct XdnaMoeChunkPlace {
  uint32_t vRowGroups = 0;
  uint32_t vObjectsX  = 0;
  uint32_t vFirstRowY = 0;
  uint64_t vOffsetX   = 0;
};

struct XdnaMoeCommand {
  XdnaMoeMatrix                  vMatrix;
  uint32_t                        vChunks       = 0;
  std::vector<XdnaMoeChunk>      aChunks;
  std::vector<XdnaMoeChunkPlace> aPlaces;
  uint32_t                        vTotalRows    = 0;
  uint64_t                        vBytesIn      = 0;  // [0, vBytesIn) written by the host
  uint64_t                        vOffsetY      = 0;  // page-aligned start of Y, after a guard page
  uint64_t                        vStreamBytesY = 0;  // bytes emitted by one stream across all chunks
  uint64_t                        vBytesY       = 0;  // bytes emitted across all streams
  uint64_t                        vBytesIo      = 0;  // total bytes of buffer 0
};

bool fXdnaMoeCommandShape(const XdnaMoeOverlay & pOverlay, const XdnaMoeMatrix & pMatrix,
                           const XdnaMoeChunk * aChunks, uint32_t pChunks, XdnaMoeCommand & pCommand,
                           std::string & pError);

// Builds the instruction stream for pCommand (buffer 0: host I/O, buffer 1: expert weights), keeping two chunks queued
// per input channel in a 3-slot ring of shim DMA descriptors while draining Y with one linear transfer per stream.
bool fXdnaMoePlan(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand,
                   std::vector<uint32_t> & aInstructions, std::string & pError);

// Writes object 0 (the 16 int32 parameters and the table of chunks) into both core halves at the start of pBuffer.
void fXdnaMoeWriteParameters(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, uint8_t * pBuffer);

// Pre-encodes one row of pMatrix.vK floats (padded with zeros to vPaddedK) into fXdnaMoeRowEncodedBytes(pMatrix) bytes
// of bfp16ebs8, so a row shared by several routed experts is encoded only once; with pNormWeights, normalises every
// value k as (pRow[k] * pNormScale) * pNormWeights[k] on the way.
uint64_t fXdnaMoeRowEncodedBytes(const XdnaMoeMatrix & pMatrix);
bool     fXdnaMoeEncodeRowBfp16(const XdnaMoeMatrix & pMatrix, const float * pRow, uint8_t * pEncoded,
                                const float * pNormWeights = nullptr, float pNormScale = 1.0f);

// Writes the X objects of chunk pChunk from pre-encoded bfp16 rows or directly from float rows.
void fXdnaMoeWriteChunkEncodedX(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, uint32_t pChunk,
                                 const uint8_t * const * aEncodedRows, uint8_t * pBuffer);
bool fXdnaMoeWriteChunkX(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, uint32_t pChunk,
                          const float * const * aRows, uint8_t * pBuffer);

// Encodes the Y emitted by pPrevCommand (e.g. gate + up + SwiGLU) directly into the X objects of pNextCommand (e.g.
// down_exps) for the same chunks, optionally scaling every row r of vTotalRows by aRowScales[r] (nullptr: scale 1).
bool fXdnaMoeTranscodeYToX(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pPrevCommand,
                            const uint8_t * pPrevBuffer, const XdnaMoeCommand & pNextCommand, uint8_t * pNextBuffer,
                            const float * aRowScales = nullptr, XdnaWorkers * pWorkers = nullptr);

struct XdnaMoeRowOut {
  float * vDst        = nullptr;  // pMatrix.vOutputs floats; nullptr skips the row
  float   vScale      = 1.0f;
  bool    vAccumulate = false;    // false: vDst[i] = vScale * y[i]; true: vDst[i] += vScale * y[i]
};

// Reads Y of all chunks of pCommand into aRowsOut (pCommand.vTotalRows entries, in chunk order). Parallelized across
// the 16 output streams, which own disjoint column ranges of vDst and therefore never race even when rows accumulate
// into the same token buffer. With pInitSrc and pInitDst, copies pInitRows rows of pMatrix.vOutputs floats from
// pInitSrc into pInitDst inside each stream's column slice before accumulating the chunks.
void fXdnaMoeReadY(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, const uint8_t * pBuffer,
                    const XdnaMoeRowOut * aRowsOut, XdnaWorkers * pWorkers = nullptr,
                    const float * pInitSrc = nullptr, float * pInitDst = nullptr, uint32_t pInitRows = 0,
                    uint64_t pInitStride = 0);

// Reads Y of a single-chunk pCommand into pTargets targets (aTargets, sorted by vFirst and not overlapping), adding
// vTarget.vAlso when set. Parallelized across the 16 output streams.
void fXdnaMoeReadYTargets(const XdnaMoeOverlay & pOverlay, const XdnaMoeCommand & pCommand, const uint8_t * pBuffer,
                           const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers = nullptr);
