#pragma once

// A matrix multiplication C = A x B on an overlay of this project: what the host has to send with the command.
// A and B are block floating point (bfp16ebs8: 8 values of 8 bits share an exponent, 9 bytes for 8 values) and C
// 32-bit floats or bfloat16, as the overlay says, all three laid out in the order the array consumes them. The
// layouts and the kernel are described in doc/XDNA2.md.

#include "xdna-bfp16.h"
#include "xdna-overlay.h"

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

class XdnaWorkers;
struct XdnaGdnPrepack;

// llama.cpp-xdna: format of the weights in B on the prompt overlay (cBfp16 on all overlays; cQ6K 4-tiles-in-3-objects
// and cQ8_0 1-tile-in-1-object when the manifest says "comp 1")
enum class XdnaPromptFormat : uint8_t {
  cBfp16 = 0,
  cQ6K   = 1,
  cQ8_0  = 2,
};

// what one multiplication needs: three buffers, in this order in the command, and an instruction stream
struct XdnaGemmPlan {
  uint64_t              vBytesA = 0;
  uint64_t              vBytesB = 0;
  uint64_t              vBytesC = 0;
  std::vector<uint32_t> aInstructions;
};

// Tiles of K every tile of C takes outside of the loop of a core: K has to have that many tiles at least.
uint32_t fXdnaGemmTilesKOutside(const XdnaOverlay & pOverlay);

// Plans the multiplication of pM x pK by pK x pN on pOverlay. False, with the reason in pError, when the overlay
// cannot do it: the sizes have to be multiples of what the overlay takes at a time.
bool fXdnaGemmPlan(const XdnaOverlay & pOverlay, uint32_t pM, uint32_t pK, uint32_t pN, XdnaGemmPlan & pPlan,
                    std::string & pError, XdnaPromptFormat pFormat = XdnaPromptFormat::cBfp16);

// llama.cpp-xdna: the recurrence of a gated delta net over a batch of tokens on an overlay of the prompts with
// "gdn 1" (bfpmm_resident.py --gdn 1, kernels/xdna2-gdn-prompt.cc), heads of 128 values with one gate per head and
// two value heads per head of keys (Qwen3.5: value head h takes the keys of h % 16): pRounds rounds of 2 heads of
// keys over pGroups groups of 8 chunks of 16 tokens (128 tokens a group). The buffers of the command, which
// fXdnaGdnPack and fXdnaGdnUnpack (xdna-gdnp.h) fill and read:
//
//   A  the bands: [8 columns][rounds][6 objects of 9216 bytes]: the states of the 4 rows, then two objects of work
//   B  the chunks: [2 x rounds heads of keys][chunks][2 objects of 9216 bytes], which both rows of a head of keys read
//   C  the outputs and the states: [8 columns][rounds][groups + 1][4 rows][8192 bytes]
//
// The command sets the last two runtime parameters of every core back to 0 after it ends, so that a multiplication
// after it needs not touch them.
const uint32_t cXdnaGdnTile      = 9216;  // bytes of an object of A or of B
const uint32_t cXdnaGdnBand      = 6;     // objects of the band of a round
const uint32_t cXdnaGdnChunk     = 16;    // tokens of a chunk
const uint32_t cXdnaGdnChunksOut = 8;     // chunks of a group: an object of C of every core
const uint32_t cXdnaGdnObjectC   = 8192;  // bytes of an object of C of a core
const uint32_t cXdnaGdnDim       = 128;   // values of a head

bool fXdnaGdnPlan(const XdnaOverlay & pOverlay, uint32_t pRounds, uint32_t pGroups, XdnaGemmPlan & pPlan,
                   std::string & pError);

// The largest K and the largest N one multiplication can have on pOverlay, as the array runs them (multiples of
// what the overlay takes at a time). A matrix with more inputs or more outputs has to be multiplied in pieces.
// False, with the reason in pError, when the overlay does not serve any size: it is built for one K and one N.
bool fXdnaGemmLimits(const XdnaOverlay & pOverlay, uint32_t & pMaxK, uint32_t & pMaxN, std::string & pError);

// The sizes of a multiplication and the ones the array runs it with: each rounded up to the next multiple of what
// the overlay takes at a time, and K to the tiles fXdnaGemmTilesKOutside asks for. What is added is zeros in A
// and in B, which add nothing to C.
struct XdnaGemmShape {
  uint32_t vM       = 0;
  uint32_t vK       = 0;
  uint32_t vN       = 0;
  uint32_t vPaddedM = 0;
  uint32_t vPaddedK = 0;
  uint32_t vPaddedN = 0;
};

// False when a size is 0 or its multiple does not fit in 32 bits.
bool     fXdnaGemmShape(const XdnaOverlay & pOverlay, uint32_t pM, uint32_t pK, uint32_t pN, XdnaGemmShape & pShape,
                         std::string & pError, XdnaPromptFormat pFormat = XdnaPromptFormat::cBfp16);
uint64_t fXdnaGemmBytesB(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape,
                          XdnaPromptFormat pFormat = XdnaPromptFormat::cBfp16);

// The three that follow share their work between the threads of pWorkers when it is given and the matrix is
// large enough for that to pay; what they write does not depend on it.

// llama.cpp-xdna: an RMS norm and the multiplication by its weights, applied to the rows of A as they are written:
// every value becomes (value * vScales[row]) * vWeights[column], the float the CPU backend gives with RMS_NORM and
// MUL, so that the normalised rows need not be written first
struct XdnaRowNorm {
  const float * vScales  = nullptr;  // one for every row of A
  const float * vWeights = nullptr;  // one for every column of A
};

// Writes A, pM rows of pK floats, into the buffer of A of the plan of the padded sizes: block floating point, in
// the order the array consumes it. The rows are pStride floats apart (pK when it is 0): a range of the columns of
// a wider matrix can be written without copying it first. With pNorm, the rows normalised as it says. False when a
// value is not finite.
bool fXdnaGemmPackA(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const float * pA,
                     uint8_t * pBuffer, XdnaRounding pRounding, XdnaWorkers * pWorkers = nullptr,
                     uint64_t pStride = 0, const XdnaRowNorm * pNorm = nullptr);

// The same for B, given transposed: pN rows of pK floats, one row for every column of B (every output), or decoded on
// demand strip by strip in cache via pReadRow(row, dst).
using XdnaGemmRowReader    = std::function<void(uint32_t pRow, float * pDst)>;
using XdnaGemmRowPtrReader = std::function<const uint8_t *(uint32_t pRow)>;

bool fXdnaGemmPackB(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const float * pTransposedB,
                     uint8_t * pBuffer, XdnaRounding pRounding, XdnaWorkers * pWorkers = nullptr);
bool fXdnaGemmPackBRows(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape,
                         const XdnaGemmRowReader & pReadRow, uint8_t * pBuffer, XdnaRounding pRounding,
                         XdnaWorkers * pWorkers = nullptr);
bool fXdnaGemmPackBQ6K(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape,
                        const XdnaGemmRowPtrReader & pRowPtr, uint8_t * pBuffer, XdnaWorkers * pWorkers = nullptr);
bool fXdnaGemmPackBQ8_0(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape,
                         const XdnaGemmRowPtrReader & pRowPtr, uint8_t * pBuffer, XdnaWorkers * pWorkers = nullptr);

// Reads C, pM rows of pN floats, from the buffer of C of the plan of the padded sizes, in the format of the overlay.
void fXdnaGemmUnpackC(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pBuffer,
                       float * pC, XdnaWorkers * pWorkers = nullptr);

// Where a range of the outputs of a multiplication (columns of C) goes: a matrix of its own, row after row. The
// outputs of one multiplication can go to several, one after the other: those of the matrices of weights that
// multiply the same rows in one command.
struct XdnaGemmTarget {
  float *  vData   = nullptr;  // where the first of its outputs goes, in the first row
  uint64_t vStride = 0;        // floats from a row to the next
  uint32_t vFirst  = 0;        // the first output it takes
  uint32_t vCount  = 0;        // and how many
  // llama.cpp-xdna: a residual sum read with C (the ADD after a multiplication). With vAlso, every output becomes
  // (base + sum) + vAlso[...], base being, with pAdd, the sums of the pieces before along the inputs (in vBase, or
  // in vData when vBase is null) and, without it, nothing: the same float as the multiplication and then the ADD of
  // the CPU backend. vBase and vAlso are laid out as vData (the same place for an output, vStride floats a row).
  const float * vBase = nullptr;
  const float * vAlso = nullptr;
  // llama.cpp-xdna: the short convolution of Qwen3.5 by tokens and its SiLU read with C (the SSM_CONV and the SILU
  // after the CONCAT of a projection, doc/XDNA2.md "The short convolution while C is read"). With vConvWeights, output
  // i of row t gets the SiLU of the convolution of its column of C along the rows (fXdnaConvToken: the rows t - 3 to
  // t of C, the ones before the first from vConvHead), and the last 3 rows of C go as they are into vConvTail. Only
  // for C as strips, in one command along the rows and without pAdd. vConvWeights is [4 taps][vConvStride], and
  // vConvHead and vConvTail [3 rows][vConvStride], all from the first output of the target.
  const float * vConvWeights = nullptr;
  const float * vConvHead    = nullptr;
  float *       vConvTail    = nullptr;
  uint64_t      vConvStride  = 0;
  // llama.cpp-xdna: when set along with vConvWeights, fConvStrips packs K, Q (with L2 norms) and V^T straight into
  // vGdnB by 16-token chunks and 128-channel heads without writing the SiLU tensor to memory
  const XdnaGdnPrepack * vGdnPrepack = nullptr;
};

// The same, with the outputs into pTargets targets, sorted by their first output and not overlapping. An output
// that no target takes is not read. With pAdd the sums are added to what the targets hold: the pieces of a matrix
// along its inputs after the first.
void fXdnaGemmUnpackC(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pBuffer,
                       const XdnaGemmTarget * aTargets, uint32_t pTargets, XdnaWorkers * pWorkers = nullptr,
                       bool pAdd = false);

// Reads the C of two multiplications of pShape, the gate and the up of a feed-forward layer (two pieces of weights by
// the same rows), and writes silu(gate) * up of every value into pTarget, as the CPU backend of ggml computes
// SwiGLU (xdna-glu.h): the SwiGLU of a model without writing its two operands as floats first. False when the
// overlay gives C in a format this does not read (block floating point). When the two are a part of the rows of the
// SwiGLU (pieces of its weights along the outputs), pRowFirst is the column of their first output in a row of
// pRowLength (0: the outputs of pShape), so that every column takes the vectors it takes in the CPU backend.
bool fXdnaGemmUnpackSwiglu(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pGate,
                            const void * pUp, const XdnaGemmTarget & pTarget, XdnaWorkers * pWorkers = nullptr,
                            uint64_t pRowFirst = 0, uint64_t pRowLength = 0);

// llama.cpp-xdna: a piece along the inputs of the multiplication whose rows are a SwiGLU (ffn_down of a feed-forward
// layer): its inputs vFirst to vFirst + vCount, columns of the SwiGLU, converted into vBuffer as fXdnaGemmPackA
// converts the rows of the SwiGLU of those columns (see fXdnaGemmPackSwiglu)
struct XdnaSwigluPiece {
  uint64_t  vFirst  = 0;
  uint32_t  vCount  = 0;
  uint8_t * vBuffer = nullptr;
};

// llama.cpp-xdna: the SwiGLU of the C of the gate and the up as fXdnaGemmUnpackSwiglu computes it (C as strips,
// XdnaOverlay::vRowsC, its columns pRowFirst to pRowFirst + pShape.vN of rows of pRowLength values), written straight
// as the rows of the multiplication that reads it, into its pieces along the inputs (aPieces): every tile of A of a
// piece whose columns are among these is computed into a tile of floats and converted from there with pRounding as
// fXdnaGemmPackA converts it (the same floats into the same encoder: the same bytes), and the SwiGLU is not written as
// floats. A piece has to be whole tiles of K (vCount, its padded K and vFirst multiples of a tile of K), and each of
// its tiles inside these columns or outside them (fXdnaGemmSwigluFits); false when it is not, when C is not as
// strips or when a value is not finite.
bool fXdnaGemmPackSwiglu(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const void * pGate,
                          const void * pUp, uint64_t pRowFirst, uint64_t pRowLength, const XdnaSwigluPiece * aPieces,
                          uint32_t pPieces, XdnaRounding pRounding, XdnaWorkers * pWorkers = nullptr);

// llama.cpp-xdna: the gated norm of a layer of linear attention of Qwen3.5 (build_norm_gated in
// src/models/qwen35.cpp) for pShape.vM rows, converted straight into the rows of the multiplication that reads it
// (ssm_out) as fXdnaGemmPackA converts them into pBuffer: for every head of pHead values of a row, its RMS norm from
// the outputs pO of the recurrence times the weights pW (x * scale * w[i], the scale as the CPU backend computes it,
// fXdnaRmsScales), times silu(z) of pZ (fXdnaSwiglu with the vectors of the CPU backend for rows of pHead), the
// floats of RMS_NORM, MUL, SILU and MUL of the CPU backend, into the same encoder. The rows of pO and pZ are pStrideO
// and pStrideZ floats apart; the columns are pFirst to pFirst + pShape.vK of a row, whole heads and whole tiles of K
// (and no padding along K); false when they are not or a value is not finite.
bool fXdnaGemmPackGatedNorm(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const float * pO,
                             uint64_t pStrideO, const float * pZ, uint64_t pStrideZ, const float * pW, uint32_t pHead,
                             float pEps, uint64_t pFirst, uint8_t * pBuffer, XdnaRounding pRounding,
                             XdnaWorkers * pWorkers = nullptr, bool pSigmoid = false);

// llama.cpp-xdna: the attention of a batch (FLASH_ATTN_EXT of ggml) on the overlay of the prompts, as two
// multiplications for every head of keys and values: S = Q·Kᵀ, with the queries of the heads that share that head
// (GQA) as its rows, and O = P·V, with P = exp(S + mask - the largest of its row) converted straight into the rows of
// the second one (as fXdnaGemmPackSwiglu does with a SwiGLU), and O divided by the sum of every row of P when it is
// read. The rows go by tokens: row r is the token r / vHeads of the head r % vHeads of the group.
struct XdnaAttnRows {
  const float * vQ           = nullptr;  // the query of the first token for the first head of the group
  uint64_t      vTokenStride = 0;        // floats from a token to the next
  uint64_t      vHeadStride  = 0;        // floats from a head to the next
  uint32_t      vTokens      = 0;
  uint32_t      vHeads       = 0;        // of the group
  float         vScale       = 1.0f;     // every value of Q times this before it is converted
  // Q in two parts of bfp16 (fXdnaAttnSplit): a row of vDim values is then 3 vDim inputs, [high, high, low], and the
  // keys [high, low, high] (fXdnaGemmPackBHalves with pSplitDim), so that S = Qh·Khᵀ + Qh·Klᵀ + Ql·Khᵀ, some 14 bits
  // for the small values of a block next to a large one; without it, vDim inputs
  bool          vSplit       = false;
  uint32_t      vDim         = 0;
};

// The two parts of bfp16 of pCount values (a multiple of 8): pHigh what a block keeps of them, rounded to the nearest,
// and pLow what is left, pValues - pHigh.
void fXdnaAttnSplit(const float * pValues, uint32_t pCount, float * pHigh, float * pLow);

// Writes the rows of Q, pShape.vM = vTokens * vHeads rows of pShape.vK values, into the buffer of A of the plan.
bool fXdnaGemmPackAttnQ(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const XdnaAttnRows & pRows,
                         uint8_t * pBuffer, XdnaRounding pRounding, XdnaWorkers * pWorkers = nullptr);

// Writes B from halves of IEEE (the F16 cache of ggml): the value of the output n and the input k is at
// pData + n * pStrideN + k * pStrideK halves, for the pShape.vN outputs and the first pValidK of the pShape.vK inputs
// (the others are zeros). The keys for S = Q·Kᵀ are a row each (pStrideK 1); for O = P·V, a dimension of V is an
// output (pStrideN 1). False when the overlay splits B or a value is not finite.
// With pSplitDim, the keys in two parts (see XdnaAttnRows): the inputs of an output are [high, low, high] of its first
// pSplitDim values (pStrideK 1, pShape.vK and pValidK 3 pSplitDim).
bool fXdnaGemmPackBHalves(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const uint16_t * pData,
                           uint64_t pStrideN, uint64_t pStrideK, uint32_t pValidK, uint8_t * pBuffer,
                           XdnaRounding pRounding, XdnaWorkers * pWorkers = nullptr, uint32_t pSplitDim = 0);

// What the KQ mask of a batch holds in every strip of a token, the width of a tile of N (XdnaOverlay::vTileN): every key
// masked (minus infinity), every one 0, or anything else. pKinds gets pTokens rows of (pKeys + width - 1) / width.
const uint8_t cXdnaAttnMasked = 0;
const uint8_t cXdnaAttnOpen   = 1;
const uint8_t cXdnaAttnMixed  = 2;
void fXdnaAttnMaskKinds(const XdnaOverlay & pOverlay, const uint16_t * pMask, uint64_t pMaskStride, uint32_t pTokens,
                         uint32_t pKeys, uint8_t * pKinds, XdnaWorkers * pWorkers = nullptr);

// The scores the first multiplication left (its C as strips of bfloat16), the mask of their tokens, and the attention
// sinks of the heads of the group (vHeads floats, one per head, or none)
struct XdnaAttnScores {
  const void *     vS           = nullptr;
  const uint16_t * vMask        = nullptr;  // [token][key], halves of IEEE, from the first token of the rows
  uint64_t         vMaskStride  = 0;        // halves from a token to the next
  const uint8_t *  vKinds       = nullptr;  // fXdnaAttnMaskKinds of that mask, from the same token
  uint32_t         vKindsStride = 0;        // strips of a token in vKinds
  const float *    vSinks       = nullptr;  // [head], floats, from the first head of the group
  uint32_t         vHeads       = 0;        // rows of a token
};

// P of the scores of pShape (its rows, and pShape.vN keys) into the pieces along the inputs of the second
// multiplication (as for a SwiGLU: piece i takes the keys vFirst to vFirst + vCount, whole tiles of K and at least the
// tiles a core takes outside of its loop, and keys past pShape.vN are zeros), and the sum of every row of P, before it
// is rounded (plus exp(sink - max) for the head of the row when pScores.vSinks is given), into pSums (pShape.vM
// floats). A row whose keys are all masked is zeros, with a sum of 0. False when C is not as strips of bfloat16, a
// piece is not whole tiles or a value is not finite (minus infinity is allowed as a sink).
bool fXdnaGemmPackAttnP(const XdnaOverlay & pOverlay, const XdnaGemmShape & pShape, const XdnaAttnScores & pScores,
                         const XdnaSwigluPiece * aPieces, uint32_t pPieces, float * pSums, XdnaRounding pRounding,
                         XdnaWorkers * pWorkers = nullptr);

// llama.cpp-xdna: whether fXdnaGemmPackSwiglu takes the SwiGLU of pM rows whose C has the columns pRowFirst to
// pRowFirst + pOutputs into those pieces (their buffers aside)
bool fXdnaGemmSwigluFits(const XdnaOverlay & pOverlay, uint32_t pM, uint64_t pRowFirst, uint64_t pOutputs,
                          const XdnaSwigluPiece * aPieces, uint32_t pPieces);

// Writes a tile of pTileRows rows by pTileK values of K (multiples of 8) of a matrix of pRows rows of pColumns
// floats, whose first row is pRow and whose first column is pColumn, as groups of 8 rows by 8 values of K: group of
// rows after group of rows and, inside each, along K; every group as its 8 rows, a block each. What falls outside
// the matrix is zeros. It returns where the next tile goes, or nullptr when a value is not finite. It is an object
// of weights of the overlay for generating tokens (xdna-gemv.h).
uint8_t * fXdnaPackTileRows(const float * pMatrix, uint32_t pRows, uint32_t pColumns, uint64_t pRow, uint64_t pColumn,
                             uint32_t pTileRows, uint32_t pTileK, uint8_t * pAt, XdnaRounding pRounding);

// Whether pOverlay uses the 3-step tile order of xdna2_mm_bfp16_s3 (cRows for A, cSets for B), and packs one tile of A
// of pTileM by pTileK values from contiguous [pRows][pTileK] floats in that order or in 2 x 2 pairs.
bool      fXdnaGemmSteps3(const XdnaOverlay & pOverlay);
uint8_t * fXdnaPackTileA(const float * pTile, uint32_t pRows, uint32_t pTileM, uint32_t pTileK, uint8_t * pAt,
                         XdnaRounding pRounding, bool pSteps3);
