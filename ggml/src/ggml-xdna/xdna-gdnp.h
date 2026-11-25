#pragma once

// llama.cpp-xdna: the host side of the recurrence of a gated delta net over a batch of tokens on the overlay of the
// prompts (GATED_DELTA_NET of Qwen3.5 with one gate per head; fXdnaGdnPlan in xdna-gemm.h, the kernel in
// kernels/xdna2-gdn-prompt.cc): fXdnaGdnPack writes the buffers A and B of the command from the tensors of the
// operation, and fXdnaGdnUnpack writes its outputs and its final states from the buffer C.
//
// The tensors, as ggml gives them (rows of 128 floats one after another; heads and tokens at any distance):
//
//   q, k   [tokens][heads of keys][128]        value head h takes the keys and queries of head h % heads of keys
//   v      [tokens][value heads][128]
//   g      [tokens][value heads]               the logarithm of the decay of every token (<= 0)
//   beta   [tokens][value heads]
//   state  [value heads][128 j][128 d]         row j of a head is column j of S, as ggml keeps it
//   out    [tokens][value heads][128]
//
// q is multiplied by vScale (1 / sqrt(128) for ggml) before it is rounded to bf16; the outputs are linear in it.
// Per head and chunk of 16 tokens the host computes e^g of every token, and from them by products (no more
// exponentials, and no division that could overflow) Gamma[t][t'] = e^(cs_t - cs_t'), E_t = e^cs_t, G_t =
// e^(cs_15 - cs_t) and the decay of the chunk e^cs_15, with cs the cumulative sum of g in the chunk.
//
// The command takes the tokens in groups of 128. The tokens that complete the last group are empty: k, q, v and beta
// 0, and g 0 (no decay). They change neither the state nor the outputs of the tokens before them (the matrix of the
// chunk is strictly lower triangular, and their keys and new values are zero), and their outputs are not read.
//
// A command can take a part of the rounds (pFirstRound, pRounds): its buffers are then those of a command of pRounds
// rounds, for the heads of keys 2 pFirstRound to 2 (pFirstRound + pRounds) - 1 and their value heads, so that the
// host packs one part and reads another while the NPU works on a third.

#include "xdna-bfp16.h"

#include <cstddef>
#include <cstdint>
#include <functional>

class XdnaWorkers;

// llama.cpp-xdna: parameters for packing K, Q (with L2 norms) and V^T of a linear-attention batch directly into vGdnB
// while fConvStrips reads C of the projection and computes the short convolution and its SiLU (2026/11/25)
struct XdnaGdnPrepack {
  uint8_t * vB          = nullptr;
  uint64_t  vPartBytesB = 0;
  uint32_t  vParts      = 0;
  uint32_t  vTokens     = 0;
  uint32_t  vKeyHeads   = 0;
  uint32_t  vValueHeads = 0;
  float     vScale      = 1.0f;
  bool      vHasL2      = false;
  float     vL2Eps      = 0.0f;
  float     vNormScale  = 1.0f;
};

// llama.cpp-xdna: a piece along the inputs of ssm_out whose A buffer receives the gated norm of the recurrence outputs
// directly inside fXdnaGdnUnpack without writing the intermediate float tensor (2026/11/25)
struct XdnaGdnGatedPiece {
  uint32_t  vFirstHead = 0;
  uint32_t  vHeads     = 0;
  uint32_t  vPaddedM   = 0;
  uint8_t * vBufferA   = nullptr;
};

struct XdnaGdnGated {
  const float *             vZ        = nullptr;
  size_t                    vStrideZ  = 0;
  const float *             vWeights  = nullptr;
  float                     vEps      = 0.0f;
  bool                      vSigmoid  = false;
  bool                      vSteps3   = true;
  XdnaRounding              vRounding = XdnaRounding::cNearest;
  const XdnaGdnGatedPiece * aPieces   = nullptr;
  uint32_t                  vPieces   = 0;
};

struct XdnaGdnIo {
  const float * vQ          = nullptr;
  size_t        vQHead      = 0;  // floats from one head of keys to the next, and from one token to the next
  size_t        vQToken     = 0;
  const float * vK          = nullptr;
  size_t        vKHead      = 0;
  size_t        vKToken     = 0;
  const float * vV          = nullptr;
  size_t        vVHead      = 0;
  size_t        vVToken     = 0;
  const float * vG          = nullptr;  // contiguous: [tokens][value heads]
  const float * vBeta       = nullptr;
  const float * vStateIn    = nullptr;  // contiguous
  float *       vStateOut   = nullptr;  // contiguous; it may be vStateIn
  float *       vOut        = nullptr;  // contiguous
  uint32_t      vTokens     = 0;        // any number; the command takes them in groups of 128
  uint32_t      vKeyHeads   = 0;        // even
  uint32_t      vValueHeads = 0;        // 1 to 4 times the heads of keys
  float         vScale      = 1.0f;
  // llama.cpp-xdna: with these, vQ and vK are the inputs of their L2 norms (build_gdn_l2_norm: an RMS_NORM and a SCALE
  // by vNormScale), and every row of q or k is taken times its scale of the RMS_NORM, [head of keys][token]
  // (fXdnaRmsScales), and then times vNormScale, as the CPU backend computes the two (2026/11/16)
  const float *        vQNorms       = nullptr;
  const float *        vKNorms       = nullptr;
  float                vNormScale    = 1.0f;
  bool                 vL2InPack     = false;  // compute L2 RMS scales per 16-token chunk inside fPackChunk
  float                vL2Eps        = 0.0f;
  bool                 vPrepackedKQV = false;  // K, Q and V^T were already written into B by fXdnaGdnPackConvHeadChunk
  const XdnaGdnGated * vGated        = nullptr;
};

// rounds of 2 heads of keys and groups of 8 chunks of all the commands for pIo; false when the overlay cannot take it
bool fXdnaGdnShape(const XdnaGdnIo & pIo, uint32_t & pRounds, uint32_t & pGroups);

// pFlush(first, bytes) takes a range of a buffer of the driver out of the cache of the processor (fXdnaFlushCache):
// every task of the threads flushes what it wrote, or what it is about to read, so that no call to the driver flushes
// the buffers whole on one thread.
using XdnaGdnFlush = std::function<void(const void *, uint64_t)>;

// packs one 16-token chunk pChunk of channel head pChannelHead (0 .. 2 * vKeyHeads + vValueHeads - 1) from contiguous
// [vReal][128] floats pRows straight into pPrepack.vB, and zeroes trailing empty chunks when pZeroTail is true
void fXdnaGdnPackConvHeadChunk(const XdnaGdnPrepack & pPrepack, uint32_t pChannelHead, uint32_t pChunk,
                               const float * pRows, bool pZeroTail);

// writes the bands (buffer A, the states of every round) and the chunks (buffer B) of the command for the rounds
// pFirstRound to pFirstRound + pRounds - 1, and flushes them; the objects of work of the bands are left as they are
void fXdnaGdnPack(const XdnaGdnIo & pIo, uint32_t pFirstRound, uint32_t pRounds, uint8_t * pA, uint8_t * pB,
                   XdnaWorkers * pWorkers, const XdnaGdnFlush & pFlush = nullptr);

// writes the outputs (or packs their gated norm into pIo.vGated) and the final states of those rounds from the buffer C
// of their command, flushing every piece before reading it; false only when pIo.vGated has a non-finite value
bool fXdnaGdnUnpack(const XdnaGdnIo & pIo, uint32_t pFirstRound, uint32_t pRounds, const uint8_t * pC,
                     XdnaWorkers * pWorkers, const XdnaGdnFlush & pFlush = nullptr);
