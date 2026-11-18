#pragma once

// The overlay for generating tokens: Y = X x W^T for 1 to 8 rows of X, what a token multiplies by every matrix of
// weights. The weights come through the sixteen input channels of the shims, in objects of 9216 bytes. Every channel
// feeds S cores (the split): with S = 2, the memory tile of its column gives every core one half of every object.
// The outputs of a command come in up to 4 segments, each with its weights in a format: bfp16 (block floating point,
// 9 bytes for 8 values), or the Q4_K, Q5_K or Q6_K of ggml, which the cores decode themselves. Every stream makes 1 /
// 16 of the outputs of every segment, in blocks of 32 outputs, and every core 32 / S of a block, from a stream of
// pieces (8 outputs by 256 values of K, one per group of 8 outputs and tile of K) cut into objects of a core, every
// block starting a new object. Every stream carries X first, in objects of a core, each S times in a row, which every
// core keeps for the whole command. The cores give back 32-bit sums: 8 rows by 32 outputs per block (joined in the
// memory tile with S = 2); the rows from the last of X on are not meaningful. Design: _/xdna2-lab/
// gemv_stream.py; kernel: kernels/xdna2-gemv-bfp16.cc, which describes the layouts of the pieces; described in
// doc/XDNA2.md.
//
// Its manifest, written next to the overlay by _/xdna2-lab/gemv_manifest.py, is a text file:
//
//   xdna2-gemv 4
//   array <columns> <rows> <rows of memory tiles>
//   kernel <name>
//   object <bytes> <outputs> <values of K> <objects of X a core keeps>   (an object and a block of a stream, a tile)
//   split <cores per stream>
//   segments <segments a command can have>
//   formats <name>...                                                   (bfp16, q4k, q5k, q6k)
//   core <core> <tile column> <tile row> <address of its parameters> <lock that starts it>
//                                                    (one per core; core c takes part c % S of stream c / S)
//   stream W|Y <stream> <shim column> <channel>                         (one per stream)
//
// Every core takes 2 + 4 * segments parameters: the objects of X after the first, the rows, and for every segment its
// blocks, the objects of every block after the first, the tiles of K and the format.
//
// The buffers of a command, in this order: X [objects of X][S copies][object of a core: bytes / S], whose objects
// hold [K / 8][rows * 9 bytes, rounded up to a multiple of 4] padded with zeros (every stream reads the same), W
// [stream][segment][block of 32 outputs][objects of the block][object] and Y [stream][segment][block][4 groups of
// outputs][8 rows][8 outputs]. Stream s makes the outputs s * n / 16 to (s + 1) * n / 16 - 1 of a segment of n
// outputs. The manifests of before the segments (versions 1 to 3) are not taken: their kernel is another.

#include "xdna-bfp16.h"
#include "xdna-gemm.h"
#include "xdna-overlay.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class XdnaWorkers;

constexpr uint32_t cXdnaGemvRowsMax     = 8;  // rows of X a command takes, and rows of every object of sums
constexpr uint32_t cXdnaGemvSegmentsMax = 4;  // segments a shape can have

// what the weights of a segment are; the number is what the cores get
enum class XdnaGemvFormat : uint32_t {
  cBfp16 = 0,  // bfp16ebs8, converted from floats by the host
  cQ4K   = 1,  // the blocks of ggml, laid out for the cores (see the kernel)
  cQ5K   = 2,
  cQ6K   = 3,
};

constexpr uint32_t cXdnaGemvFormats = 4;

// bytes of a block of ggml (256 values) of the formats the cores decode, and 0 for bfp16
uint32_t fXdnaGemvBlockBytes(XdnaGemvFormat pFormat);

// bytes of a piece (8 outputs by 256 values of K) of every format
uint32_t fXdnaGemvPieceBytes(XdnaGemvFormat pFormat);

// the name of a format in a manifest and in the tools: bfp16, q4k, q5k, q6k; false when pName is none of them
const char * fXdnaGemvFormatName(XdnaGemvFormat pFormat);
bool         fXdnaGemvFormatFromName(const std::string & pName, XdnaGemvFormat & pFormat);

struct XdnaGemvOverlay {
  uint32_t                        vColumns     = 0;  // of the array
  uint32_t                        vRows        = 0;
  uint32_t                        vMemoryRows  = 0;
  std::string                     vKernel;
  uint32_t                        vObjectBytes = 0;  // of an object of weights of a stream
  uint32_t                        vOutputs     = 0;  // outputs of a block of a stream
  uint32_t                        vTileK       = 0;  // values of K of a tile
  uint32_t                        vObjectsX    = 0;  // objects of X a core can keep, each vObjectBytes / vSplit
  uint32_t                        vSplit       = 1;  // cores every stream feeds
  uint32_t                        vSegments    = 0;  // segments a command can have
  uint32_t                        vFormats     = 0;  // the formats it takes: bit 1 << XdnaGemvFormat
  std::vector<XdnaOverlayCore>   aCores;     // per core: vSplit per stream
  std::vector<XdnaOverlayStream> aStreamsW;  // host to cores, per stream
  std::vector<XdnaOverlayStream> aStreamsY;  // cores to host, per stream

  // Reads a manifest. False, with the reason in pError, when it cannot be read or is not complete.
  bool mLoad(const std::string & pPath, std::string & pError);

  uint32_t mStreams() const { return uint32_t(aStreamsW.size()); }
  bool     mTakes(XdnaGemvFormat pFormat) const { return (vFormats & (1u << uint32_t(pFormat))) != 0; }
};

// what a segment of a multiplication is asked to be: its format and outputs
struct XdnaGemvPart {
  XdnaGemvFormat vFormat  = XdnaGemvFormat::cBfp16;
  uint32_t        vOutputs = 0;
};

// a segment as the array runs it: its outputs rounded up to the streams times the outputs of a block (zeros in W,
// which add nothing to Y)
struct XdnaGemvSegment {
  XdnaGemvFormat vFormat      = XdnaGemvFormat::cBfp16;
  uint32_t        vOutputs     = 0;  // as given
  uint32_t        vPadded      = 0;
  uint32_t        vFirst       = 0;  // its first output, counting the given outputs of the segments before it
  uint32_t        vBlocks      = 0;  // blocks of every stream
  uint32_t        vObjects     = 0;  // objects of a stream for a block
  uint32_t        vFirstBlock  = 0;  // blocks of every stream of the segments before it: where its sums start
  uint64_t        vOffsetW     = 0;  // where its weights start in the part of W of a stream, in bytes
};

// the sizes of one multiplication, as the array runs it: K rounded up to a tile (zeros in X and W)
struct XdnaGemvShape {
  uint32_t         vRows     = 0;  // of X
  uint32_t         vK        = 0;  // as given
  uint32_t         vN        = 0;  // the given outputs of all the segments
  uint32_t         vPaddedK  = 0;
  uint32_t         vTilesK   = 0;  // tiles of K
  uint32_t         vGroupX   = 0;  // bytes of a group of 8 values of K of X: its rows, padded to a multiple of 4
  uint32_t         vObjectsX = 0;  // objects of X of a core, at the start of every stream (each split times)
  uint32_t         vSegments = 0;
  XdnaGemvSegment aSegments[cXdnaGemvSegmentsMax];
  uint32_t         vBlocksY  = 0;  // blocks of sums of every stream
  uint64_t         vPartW    = 0;  // bytes of W of every stream
  uint64_t         vBytesX   = 0;  // of the three buffers of the command
  uint64_t         vBytesW   = 0;
  uint64_t         vBytesY   = 0;
};

// False, with the reason in pError, when the overlay cannot do it: 1 to 8 rows, X has to fit in a core, 1 to the
// segments of the overlay, and the overlay has to take their formats. With the formats of ggml, K is a multiple of
// 256 (their blocks).
bool fXdnaGemvShape(const XdnaGemvOverlay & pOverlay, uint32_t pRows, uint32_t pK, const XdnaGemvPart * aParts,
                     uint32_t pParts, XdnaGemvShape & pShape, std::string & pError);

// The same with one segment.
bool fXdnaGemvShape(const XdnaGemvOverlay & pOverlay, uint32_t pRows, uint32_t pK, uint32_t pN, XdnaGemvFormat pFormat,
                     XdnaGemvShape & pShape, std::string & pError);

// The instruction stream of the multiplication of pShape.
bool fXdnaGemvPlan(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, std::vector<uint32_t> & aInstructions,
                    std::string & pError);

// the outputs every stream makes of the segment pSegment of pShape
inline uint32_t fXdnaGemvOutputsPerStream(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape,
                                           uint32_t pSegment) {
  return pOverlay.mStreams() == 0 ? 0 : pShape.aSegments[pSegment].vPadded / pOverlay.mStreams();
}

// bytes of a row of pShape.vK values in pFormat as the host gives it: floats for bfp16, blocks of ggml otherwise
uint64_t fXdnaGemvRowBytes(const XdnaGemvShape & pShape, XdnaGemvFormat pFormat);

// Writes one piece (fXdnaGemvPieceBytes(pFormat) bytes, see the kernel): the 8 outputs from pRow on by the tile
// pTile of K (256 values) of pGiven rows of pK values in pFormat, pRowBytes apart (floats for bfp16, rounded to the
// nearest, the blocks of ggml otherwise, pK a multiple of 256). Rows from pGiven on, and values from pK on, are zeros.
// False when a float is not finite. The chained overlay (xdna-chain.h) takes the same pieces.
bool fXdnaGemvPiece(XdnaGemvFormat pFormat, const void * pRows, uint32_t pGiven, uint32_t pK, uint64_t pRowBytes,
                     uint32_t pRow, uint32_t pTile, uint8_t * pPiece);

// Writes the weights of the stream pStream for the segment pSegment into its part of the buffer of W: its outputs
// are pStream * fXdnaGemvOutputsPerStream on, of which pRows holds the first pGiven (the rest, past the last output
// of the segment, are zeros), one after another, fXdnaGemvRowBytes each: floats for bfp16, rounded to the nearest,
// the blocks of ggml for the others. False when a float is not finite. It shares the work between the threads of
// pWorkers when it is given.
bool fXdnaGemvPackStream(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, uint32_t pSegment,
                          uint32_t pStream, const void * pRows, uint32_t pGiven, uint8_t * pBuffer,
                          XdnaWorkers * pWorkers = nullptr);

// Writes the whole W of a shape of one segment, its pShape.vN rows one after another in its format.
bool fXdnaGemvPackW(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, const void * pRows,
                     uint8_t * pBuffer, XdnaWorkers * pWorkers = nullptr);

// Writes X, pShape.vRows rows of pShape.vK floats, pStride floats apart, into the buffer of X, every object of a core
// as many times as a stream feeds cores. False when a value is not finite.
bool fXdnaGemvPackX(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, const float * pX, size_t pStride,
                     uint8_t * pBuffer);

// Reads Y, pShape.vRows rows of the given outputs of all the segments (the first output of a segment is its vFirst),
// from the buffer of Y into pTargets targets (see XdnaGemmTarget), sorted by their first output and not overlapping.
// An output that no target takes is not read.
void fXdnaGemvUnpackY(const XdnaGemvOverlay & pOverlay, const XdnaGemvShape & pShape, const float * pBuffer,
                       const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers = nullptr);
