#pragma once

// The chained overlay for generating tokens (xdna2-gemv 9). Like the overlay of xdna-gemv.h it computes Y = X x W^T
// for 1 to 8 rows of X with 32-bit float sums, the weights coming through the sixteen input channels of the shims in
// objects of 9216 bytes as pieces of 8 outputs by 256 values of K (bfp16, or the Q4_K, Q5_K or Q6_K of ggml, which
// the cores decode), but:
//
//   - The two cores fed by a channel share the work along K: core 0 of the pair makes the first half of the tiles of
//     K of every block of 32 outputs and core 1 the second half, which it sends to core 0 over the cascade; core 0
//     adds them, finishes them and sends the block of Y. Every object of a channel is [4608 bytes for core 0 | 4608
//     bytes for core 1], and every core keeps only its half of X.
//   - A command runs up to 4 phases one after the other. A phase takes its X from the host (in bfp16, 1 to 8 rows, or
//     in 32-bit floats, 1 row, which the cores round to bfp16) or from row 0 of the Y of a phase before it, and may
//     start with an RMS norm (h and the weights of the norm; the cores apply the scale of the norm to the sums) and
//     end with an epilogue on row 0 (a residual added and kept as h, h added, or the SwiGLU of its two segments).
//   - The parameters come in the first object of every command, the same for every channel: no runtime parameters
//     and no writes of registers of the cores.
//   - Version 6: the blocks of a segment follow each other in the objects with no padding between them (the cores stop
//     where a block ends inside an object), and a core keeps 7 objects of X (32256 bytes; 3 with version 5), so that 8
//     rows of K = 4096 and 6 of K = 9216 fit; the epilogues take 320 outputs a stream (5120 in all; 512 before).
//   - Version 7: an object of Y is one row of a block, its 32 outputs, so the Y of a segment is [block][row][32
//     outputs] (with one row, the outputs in their order); before, every block had room for 8 rows.
//   - Version 8: a phase can run the recurrence of a layer of linear attention (see XdnaChainRecurrence), and the RMS
//     norm of a prologue takes 1 / K from the host (the cores have no room for a division of floats).
//   - Version 9: a phase can run a whole layer of linear attention of Qwen3.5 (XdnaChainRecurrence::vLinear): the
//     convolution of q, k and v, its SiLU, the L2 norms of q and k, the recurrence and the gated norm.
//
// Design: _/xdna2-lab/gemv_chain.py; kernel: kernels/xdna2-gemv-chain.cc, which describes the parameters, the
// objects and the epilogues; described in doc/XDNA2.md.
//
// Its manifest, written next to the overlay by _/xdna2-lab/gemv_chain_manifest.py, is a text file (the host
// takes version 9 only):
//
//   xdna2-gemv 9
//   array <columns> <rows> <rows of memory tiles>
//   kernel <name>
//   object <bytes of an object of a stream> <bytes of the half of a core>
//   x <bytes of X a core keeps>
//   segments <segments of a phase>
//   phases <phases of a command>
//   formats <name>...                                                   (bfp16, q4k, q5k, q6k)
//   stream W|Y <stream> <shim column> <channel>                         (one per stream)
//
// The buffers of a command: 0, the input and output of the host (see XdnaChainCommand: the parameters, the
// residuals, the X that come from the host and the weights of the norms, and after them the Y of every phase), and 1
// to 4, the matrices of weights its phases run, each laid out as XdnaChainMatrix says.

#include "xdna-gemm.h"
#include "xdna-gemv.h"
#include "xdna-overlay.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

class XdnaWorkers;

constexpr uint32_t cXdnaChainPhasesMax   = 4;  // phases of a command
constexpr uint32_t cXdnaChainMatricesMax = 4;  // matrices of weights of a command
constexpr uint32_t cXdnaChainSegmentsMax = 2;  // segments of a phase

struct XdnaChainOverlay {
  uint32_t                        vColumns     = 0;  // of the array
  uint32_t                        vRows        = 0;
  uint32_t                        vMemoryRows  = 0;
  std::string                     vKernel;
  uint32_t                        vObjectBytes = 0;  // of an object of a stream
  uint32_t                        vCoreBytes   = 0;  // of the half of it that a core gets
  uint32_t                        vKeptX       = 0;  // bytes of X a core keeps
  uint32_t                        vSegments    = 0;  // segments of a phase
  uint32_t                        vPhases      = 0;  // phases of a command
  uint32_t                        vFormats     = 0;  // the formats it takes: bit 1 << XdnaGemvFormat
  std::vector<XdnaOverlayStream> aStreamsW;  // host to cores, per stream
  std::vector<XdnaOverlayStream> aStreamsY;  // cores to host, per stream

  // Reads a manifest. False, with the reason in pError, when it cannot be read or is not complete.
  bool mLoad(const std::string & pPath, std::string & pError);

  uint32_t mStreams() const { return uint32_t(aStreamsW.size()); }
  bool     mTakes(XdnaGemvFormat pFormat) const { return (vFormats & (1u << uint32_t(pFormat))) != 0; }
};

// the version a manifest of an overlay for generating tokens says it is (4: xdna-gemv.h, 5 and 6: this one, of which
// the host takes 6), 0 if none
uint32_t fXdnaGemvManifestVersion(const std::string & pPath);

// a segment of a matrix of weights: its outputs rounded up to the streams times the 32 outputs of a block (zeros in
// W, which add nothing to Y)
struct XdnaChainSegment {
  XdnaGemvFormat vFormat    = XdnaGemvFormat::cBfp16;
  uint32_t        vOutputs   = 0;  // as given
  uint32_t        vPadded    = 0;
  uint32_t        vFirst     = 0;  // its first output, counting the given outputs of the segments before it
  uint32_t        vBlocks    = 0;  // blocks of every stream
  uint64_t        vCoreBlock = 0;  // bytes of the pieces of a block for a core
  uint32_t        vObjects   = 0;  // objects of a stream for the whole segment (half of each for every core)
  uint64_t        vOffsetW   = 0;  // where its weights start in the part of a stream, in bytes
};

// A matrix of weights as the chained overlay reads it, up to 4 segments of outputs, each in a format: [stream][segment]
// [objects], where the halves of the objects of a core hold the pieces of the blocks of 32 outputs of the segment one
// after the other (version 6; with version 5 every block took whole objects), and only the last object of a segment
// has padding. K is rounded up to a multiple of 512 (zeros), so that every core gets whole tiles: the first half of the
// tiles goes to core 0 and the second to core 1. Stream s makes the outputs s * n / 16 to (s + 1) * n / 16 - 1 of a
// segment of n padded outputs.
struct XdnaChainMatrix {
  uint32_t          vK         = 0;  // as given
  uint32_t          vPaddedK   = 0;
  uint32_t          vTilesCore = 0;  // tiles of 256 values of K of a core
  uint32_t          vN         = 0;  // the given outputs of all the segments
  uint32_t          vSegments  = 0;
  XdnaChainSegment aSegments[cXdnaGemvSegmentsMax];
  uint64_t          vPartW     = 0;  // bytes of W of every stream
  uint64_t          vBytesW    = 0;
};

// False, with the reason in pError, when the overlay cannot take it: 1 to 4 segments in formats the overlay takes,
// K a multiple of 256 when a segment is in a format of ggml.
bool fXdnaChainMatrixShape(const XdnaChainOverlay & pOverlay, uint32_t pK, const XdnaGemvPart * aParts,
                            uint32_t pParts, XdnaChainMatrix & pMatrix, std::string & pError);

// Writes the weights of the stream pStream for the segment pSegment into its part of the buffer of the matrix: its
// outputs are pStream * (padded outputs / streams) on, of which pRows holds the first pGiven (the rest are zeros),
// one after another, fXdnaGemvRowBytes each (floats for bfp16, rounded to the nearest, the blocks of ggml for the
// others). False when a float is not finite. It shares the work between the threads of pWorkers when it is given.
bool fXdnaChainPackStream(const XdnaChainOverlay & pOverlay, const XdnaChainMatrix & pMatrix, uint32_t pSegment,
                           uint32_t pStream, const void * pRows, uint32_t pGiven, uint8_t * pBuffer,
                           XdnaWorkers * pWorkers = nullptr);

// how a phase gets its X
enum class XdnaChainPrologue : uint32_t {
  cBfp16  = 0,  // from the host, in bfp16, 1 to 8 rows
  cFloats = 1,  // in 32-bit floats, 1 row: from the host or the Y of a phase before; the cores round it to bfp16
  cNorm   = 2,  // the RMS norm of h (1 row, from the host or the Y of a phase before) by its weights (from the host)
};

// what a phase does with its sums (core 0, row 0) before they go to Y
enum class XdnaChainEpilogue : uint32_t {
  cNone     = 0,
  cResidual = 1,  // Y + r, r a residual from the host; the result is kept as h
  cAddH     = 2,  // Y + h, h kept by a phase before with cResidual (its outputs are those of this phase)
  cSwiglu   = 3,  // two segments of the same outputs: the first (the gate) stays, the second becomes silu(gate) * up
};

// Version 8: a phase can run the recurrence of a layer of linear attention instead of a matrix: the gated delta net of
// ggml (ggml_gated_delta_net with heads of 128 values and one gate per head, as Qwen3.5 has), one head of values per
// core, head 32 h + 2 s + c on core c of stream s for h from 0 to vHeads / 32 - 1. The states live in a buffer of the
// command (one of the driver, which the host shares with the processor), head j at vStateOffset + j * 64 KB, as
// llama.cpp keeps them in a cell of its recurrent cache (row i of a head is column i of the state, 128 floats); they
// come through the channels of the weights, 9 rows to the half of an object of a core and 15 objects a head (the last
// one with 2 rows: the other 7, read past the head, are padding, so that the buffer needs 3.5 KB of room after the
// last head), and go back where they were. The inputs of a head (q, k, v, the decay e^g of its gate and beta) come
// from the host, an object of the prologue per head of a core; its outputs (row . q for every row, without the
// 1 / sqrt(128) of ggml) go to the Y of the phase, [head][128].
const uint32_t cXdnaChainRecurrenceDim = 128;  // values of a head

// Version 9, vLinear: the phase runs a whole layer of linear attention of Qwen3.5 when generating (qwen35.cpp,
// build_layer_attn_linear): 16 heads of keys and 32 value heads of 128, the 8192 channels of the convolution [q of the
// heads of keys | their k | v of the value heads], head s on core 0 and head s + 16 on core 1 of stream s (they share
// the head of keys s). The host gives every token x of the channels (the projection before the convolution), the 3 old
// states of the convolution of every channel (which it then shifts in its cache), z of every head, and the decay e^g
// and beta of every head (it computes them from alpha and beta of the projections: softplus, exp and sigmoid), in two
// objects of the prologue per stream (fXdnaChainWriteLinear); the weights of the convolution and of the gated norm,
// the same every token, come in a third one from the matrix vMatrix of the phase, one object per stream
// (fXdnaChainLinearMatrix, fXdnaChainPackLinear). The cores convolve, take the SiLU and the L2 norms of q and k, run
// the recurrence with the states in their buffer as above, and end every head with its gated norm: the Y of the phase
// is the input of the projection of the layer, [head][128] (read it with fXdnaChainReadRecurrence and a scale of 1).
const uint32_t cXdnaChainLinearHeads    = 32;    // value heads of a layer of linear attention
const uint32_t cXdnaChainLinearKeys     = 16;    // heads of keys
const uint32_t cXdnaChainLinearChannels = 8192;  // channels of its convolution
const uint32_t cXdnaChainLinearTaps     = 4;     // its taps: 3 old states and the token

struct XdnaChainRecurrence {
  uint32_t vHeads       = 0;  // 0: the phase runs a matrix; otherwise a multiple of the cores (32)
  uint32_t vState       = 0;  // the buffer of the states among those of the command after its matrices
  uint64_t vStateOffset = 0;  // where head 0 starts in it (bytes, a multiple of 4)
  uint32_t vLinear      = 0;  // version 9: 1 for a whole layer of linear attention (vHeads 32, see above)
};

// one phase of a command, as it is asked for
struct XdnaChainPhase {
  XdnaChainPrologue vPrologue     = XdnaChainPrologue::cFloats;
  uint32_t           vRows         = 1;   // 1 to 8 with cBfp16, 1 otherwise
  int32_t            vFromPhase    = -1;  // cFloats and cNorm: X (h for cNorm) is row 0 of the Y of the segment
  uint32_t           vFromSegment  = 0;   // vFromSegment of that phase before; -1: from the host
  int32_t            vSameXAs      = -1;  // from the host: the X of that phase before again (same prologue, K, rows)
  XdnaChainEpilogue vEpilogue     = XdnaChainEpilogue::cNone;
  float              vEps          = 0;   // cNorm
  uint32_t           vMatrix       = 0;   // the matrix of the command it runs (buffer 1 + vMatrix)
  uint32_t           vFirstSegment = 0;   // and its segments: 1 or 2 (those of the overlay), one after the other
  uint32_t           vSegments     = 1;
  XdnaChainRecurrence vRecurrence;       // version 8: the recurrence instead of a matrix (vHeads not 0); X from the
                                          // host, and the prologue, the epilogue, the matrix and its segments unused
};

// where a phase puts things, in the buffer of the host (bytes from its start)
struct XdnaChainPlace {
  uint32_t vK               = 0;  // the K of its matrix, as given and padded
  uint32_t vPaddedK         = 0;
  uint32_t vObjectsX        = 0;  // objects of its X (or of h) in its prologue
  uint32_t vObjectsPrologue = 0;  // all the objects of its prologue: the residual, X (or h) and the weights of the norm
  uint64_t vOffsetX         = 0;  // its X from the host (that of vSameXAs with it)
  uint64_t vOffsetResidual  = 0;  // its residuals: one object per stream
  uint64_t vOffsetNorm      = 0;  // the weights of its norm
  uint32_t vBlocks[cXdnaChainSegmentsMax]  = {};  // blocks of every stream of its segments
  uint64_t vOffsetY[cXdnaChainSegmentsMax] = {};  // where the Y of its segments start: [stream][block][row][32
                                                   // outputs] each (with one row, the outputs in their order)
  uint64_t vBytesY          = 0;
};

// A command of the chained overlay: its phases, the matrices they run and the layout of the buffer of the host, which
// is the parameters (an object), then for every phase its residuals, its X from the host and the weights of its
// norm, and from vOffsetY (on a page of its own) the Y of the phases one after the other, with room after them for
// what a phase that reads a Y may read past its end.
struct XdnaChainCommand {
  uint32_t         vPhases   = 0;
  XdnaChainPhase  aPhases[cXdnaChainPhasesMax];
  XdnaChainPlace  aPlaces[cXdnaChainPhasesMax];
  uint32_t         vMatrices = 0;
  XdnaChainMatrix aMatrices[cXdnaChainMatricesMax];
  uint32_t         vStates   = 0;  // buffers of states (version 8), after the matrices: buffers 1 + vMatrices on
  uint64_t         vBytesIn  = 0;  // what the host writes: [0, vBytesIn)
  uint64_t         vOffsetY  = 0;
  uint64_t         vBytesIo  = 0;  // the whole buffer
};

// Checks the phases against the overlay, the matrices and the buffers of states and lays the buffer of the host out.
// False, with the reason in pError, when the overlay cannot run them. The buffers of a command are 5 at most: the one of
// the host, the matrices and the buffers of states.
bool fXdnaChainCommandShape(const XdnaChainOverlay & pOverlay, const XdnaChainPhase * aPhases, uint32_t pPhases,
                             const XdnaChainMatrix * aMatrices, uint32_t pMatrices, XdnaChainCommand & pCommand,
                             std::string & pError, uint32_t pStates = 0);

// a key that tells two commands with the same instruction stream apart from the rest
std::vector<uint32_t> fXdnaChainKey(const XdnaChainCommand & pCommand);

// The instruction stream of a command. With pEarly, a phase that waits for the Y of the phases before it has the
// buffer descriptors of its transfers written before the wait, while the array works, when the shim tiles have them
// free (16 per tile, which two consecutive phases of a feed-forward block share); without it, every descriptor is
// written right before it is queued, as mlir-aie does. With pFillsFirst the transfers that feed the cores of a phase
// are queued before those that drain its Y; without it the drains go first, as before.
bool fXdnaChainPlan(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand,
                     std::vector<uint32_t> & aInstructions, std::string & pError, bool pEarly = true,
                     bool pFillsFirst = true);

// Writes the parameters of the command at the start of the buffer of the host.
void fXdnaChainWriteParameters(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand,
                                uint8_t * pBuffer);

// Writes the X from the host of the phase pPhase: its rows of K floats (as given), pStride floats apart, rounded to
// bfp16 to the nearest with cBfp16 or as floats with cFloats. False when a value is not finite.
bool fXdnaChainWriteX(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                       const float * pX, size_t pStride, uint8_t * pBuffer);

// Writes the residual of the phase pPhase (cResidual): one value per output of its first segment, as given.
void fXdnaChainWriteResidual(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                              const float * pResidual, uint8_t * pBuffer);

// Writes the weights of the norm of the phase pPhase (cNorm): K values, as given.
void fXdnaChainWriteNorm(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                          const float * pWeights, uint8_t * pBuffer);

// The inputs of a phase of the recurrence, in the floats of ggml: q and k of vHeadsQK heads (head j of the values takes
// those of head j % vHeadsQK, as ggml_gated_delta_net does), v of every head, and for every head the decay of its gate
// (e^g, which the host computes: the cores have no scalar floating point) and its beta. The strides are in floats,
// from one head to the next.
struct XdnaChainRecurrenceIo {
  const float * vQ       = nullptr;
  size_t        vStrideQ = cXdnaChainRecurrenceDim;
  const float * vK       = nullptr;
  size_t        vStrideK = cXdnaChainRecurrenceDim;
  uint32_t      vHeadsQK = 0;
  const float * vV       = nullptr;
  size_t        vStrideV = cXdnaChainRecurrenceDim;
  const float * vDecay   = nullptr;
  const float * vBeta    = nullptr;
};

// Writes the inputs of the phase of the recurrence pPhase. False when a value is not finite.
bool fXdnaChainWriteRecurrence(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand,
                                uint32_t pPhase, const XdnaChainRecurrenceIo & pIo, uint8_t * pBuffer);

// Version 9: what a phase of a whole layer of linear attention takes from the host every token (see
// XdnaChainRecurrence), in the floats of ggml: the input of the short convolution as the CONCAT of the convolution by
// tokens of llama.cpp-xdna has it (build_conv_state): 4 rows of the 8192 channels, vStride floats apart, the 3 old
// states (the oldest first) and then the projection x of the token; z of every value head ([head][128]), e^g and beta
// of every value head, the eps of the L2 norms of q and k (x / sqrt(sum of the squares + eps)) and that of the gated
// norm (f_norm_rms_eps).
struct XdnaChainLinearIo {
  const float * vInput    = nullptr;
  size_t        vStride   = 0;
  const float * vZ        = nullptr;
  const float * vDecay    = nullptr;
  const float * vBeta     = nullptr;
  float         vEpsL2    = 0;
  float         vEpsGated = 0;
};

// Writes what the phase pPhase of a layer of linear attention takes from the host. False when a value is not finite.
bool fXdnaChainWriteLinear(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                            const XdnaChainLinearIo & pIo, uint8_t * pBuffer);

// The matrix of the fixed object of a layer of linear attention: one object per stream.
void fXdnaChainLinearMatrix(const XdnaChainOverlay & pOverlay, XdnaChainMatrix & pMatrix);

// Writes it: the weights of the convolution ([channel][4] floats, as ggml keeps ssm_conv1d) and of the gated norm (128
// floats, ssm_norm), in bf16 as the cores take them, into pBuffer (vBytesW of fXdnaChainLinearMatrix). False when a
// value is not finite.
bool fXdnaChainPackLinear(const XdnaChainOverlay & pOverlay, const float * pConv, const float * pNorm,
                           uint8_t * pBuffer);

// Reads the outputs of the phase of the recurrence pPhase into pOutputs, [head][128], each times pScale (ggml takes
// 1 / sqrt(128)).
void fXdnaChainReadRecurrence(const XdnaChainCommand & pCommand, uint32_t pPhase, const uint8_t * pBuffer,
                               float * pOutputs, float pScale);

// Reads the rows of the Y of the phase pPhase (as many as it has) into pTargets targets (see XdnaGemmTarget,
// sorted by their first output and not overlapping), the outputs counted as in its matrix: the first output of a
// segment is its vFirst. An output that no target takes is not read.
void fXdnaChainReadY(const XdnaChainOverlay & pOverlay, const XdnaChainCommand & pCommand, uint32_t pPhase,
                      const uint8_t * pBuffer, const XdnaGemmTarget * aTargets, uint32_t pTargets);

// What fXdnaChainReadY may read of the Y of a segment: vRecords records of vBytes, vStride bytes apart, from vOffset
// in the buffer of the host (with version 7, one piece: the blocks up to the last given output, all their rows; with
// version 6 it was 32 bytes of every 256 with one row). Those are the lines that have to leave the cache of the
// processor before it reads them.
struct XdnaChainRead {
  uint64_t vOffset  = 0;
  uint64_t vRecords = 0;
  uint64_t vStride  = 0;
  uint64_t vBytes   = 0;
};

// The reads of the phase pPhase, one per segment, into aReads (cXdnaChainSegmentsMax of them at most). Returns how
// many.
uint32_t fXdnaChainReads(const XdnaChainCommand & pCommand, uint32_t pPhase, XdnaChainRead * aReads);
