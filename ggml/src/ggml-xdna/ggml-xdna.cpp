// The XDNA backend of ggml: the NPU of AMD Ryzen AI processors.
//
// It is an accelerator device, as the BLAS backend of llama.cpp was: llama.cpp adds it to every context in front
// of the processor, and the scheduler of ggml gives it the operations it says it can do. It takes one: the
// multiplication of rows of 32-bit floats by a matrix of weights of a model (GGML_OP_MUL_MAT). Everything else
// stays in the processor.
//
// Its buffers are plain host memory, so the processor reads and writes them as its own and nothing is copied
// between the two. It has a buffer type of its own all the same, for two reasons: llama.cpp puts in the buffers
// of an accelerator the weights whose operation the accelerator takes, which keeps them out of the buffers where
// the processor repacks weights for itself; and the backend learns when the weights of a model are freed.
//
// The first time a matrix of weights is used it is converted to the block floating point of the NPU (bfp16) and
// kept in a buffer of the NPU, in pieces when it has more inputs or more outputs than one multiplication of the
// overlay takes. It is freed with the buffer of the model it lives in. The rows are converted for every
// multiplication, and the sums come back as 32-bit floats.
//
// All the backends of a process share one context of the NPU, and one multiplication runs at a time.
//
// When the folder of the overlay also holds the overlay for generating tokens (xdna2-gemv-bfp16.pdi and its
// manifest, see xdna-gemv.h), it goes into the same context as a second compute unit, and the multiplications of
// 1 to 8 rows (generating tokens) go to it instead of staying with the processor: its weights are converted again,
// in its own layout, the first time a matrix multiplies so few rows. Its sums come back as 32-bit floats, so it
// also takes the logits. GGML_XDNA_NO_GEMV leaves it out.
//
// The backend also takes the SwiGLU of a feed-forward layer: with many rows it computes it while it reads the C of
// the gate and the up (fMulMatGlu), and otherwise on the host from their results, as the CPU backend computes it.
// GGML_XDNA_NO_GLU leaves it to the CPU backend, for comparing.
//
// The commands of the overlay of the prompts go two at a time: the host converts the rows of the next piece of a
// matrix and reads the results of the last command while the NPU works (fMultiply). GGML_XDNA_NO_OVERLAP waits for
// every command before going on, for comparing: the same commands in the same order.
//
// It also takes the residual ADD after a multiplication it takes: on the NPU of the prompts, the ADD is computed
// while C is read, into its result (the result of the multiplication is not written), and otherwise on the host.
// And the concatenation of the input of the short convolution of Qwen3.5 by tokens (build_conv_state): the states,
// then the result of a multiplication it takes; when it follows the multiplications, the result is written straight
// into it. And the RMS_NORM of the residual stream with the MUL by its weights, when the multiplications that use
// its result follow: the rows are normalised as they are converted for the NPU, and the normalised rows are not
// written. GGML_XDNA_NO_FUSE leaves all of them to the CPU backend, for comparing.
//
// llama.cpp-xdna: with GGML_XDNA_LINEAR and the chained overlay of version 9, it also takes, while a token is
// generated, the operations of every layer of linear attention of Qwen3.5 4B from the short convolution to the gated
// norm (fLinearTaken), and gives them to the NPU as one phase of the chained overlay (fLinearAttention), with the
// recurrent states in their cell of a buffer of the driver; they go one by one on the host when they do not make a
// whole layer (fLinearOnHost).
//
// llama.cpp-xdna: with many rows, the SwiGLU of a feed-forward layer goes straight into the rows of ffn_down
// (fMulMatGlu), and the gated norm of a layer of linear attention into those of ssm_out (fGatedNorm): converted
// tile by tile into the buffers of A of the NPU, without writing them as floats (GGML_XDNA_NO_PACK_GLU and
// GGML_XDNA_NO_GATED write them as before).
//
// With GGML_XDNA_EMULATE the processor does what the NPU would do, with the same numbers: the operands rounded
// to the block floating point and the sums in 32-bit floats. It is there to check the backend and to measure
// what a number format costs in precision without the NPU.

#include "ggml-xdna.h"

#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "xdna-glu.h"
#include "xdna-runner.h"

#include <sys/mman.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
#include <immintrin.h>
#endif

namespace {

const char * const cName          = "XDNA";
const char * const cDefaultDevice = "/dev/accel/accel0";
const char * const cOverlayFile   = "xdna2-gemm-bfp16.pdi";
const char * const cManifestFile  = "xdna2-gemm-bfp16.manifest";
const char * const cGemvOverlayFile  = "xdna2-gemv-bfp16.pdi";       // the overlay for generating tokens
const char * const cGemvManifestFile = "xdna2-gemv-bfp16.manifest";
const char * const cChainOverlayFile  = "xdna2-gemv-chain.pdi";      // or the chained one, which goes first
const char * const cChainManifestFile = "xdna2-gemv-chain.manifest";
const char * const cMoeOverlayFile    = "xdna2-moe.pdi";             // the mixture-of-experts overlay (MUL_MAT_ID)
const char * const cMoeManifestFile   = "xdna2-moe.manifest";

const XdnaRounding cRounding       = XdnaRounding::cNearest;  // of the weights and of the rows
const uint32_t      cRowsPerCommand = 4096;                     // rows one command multiplies at most
const int64_t       cLargestSide    = 1 << 24;                  // inputs, outputs or rows of a multiplication
const uint64_t      cThreadedWork   = 1ull << 24;               // products from which the emulation uses threads
const uint32_t      cRowsPerTask    = 64;                       // rows of weights a thread converts at a time
// Rows a multiplication needs for the backend to take it, if GGML_XDNA_MIN_BATCH does not say otherwise (from 1 to 8,
// the overlay for generating tokens takes them). Measured again on 2026/11/16 (it was 72, from Qwen3.5 0.8B and the
// overlay of 2026/11/13): with batches of 32 tokens the processor is faster with Qwen3.5 0.8B Q8_0 and 4B Q4_K_M, and
// with 40 the NPU is, by 8 and 20 % (by more than half with 64) (doc/XDNA2.md). It is also what the attention of a
// batch needs (fAttnTaken).
const uint64_t      cDefaultMinRows = 40;
const uint64_t      cAttnSplitMinKeys = 1024;  // llama.cpp-xdna: the keys an attention with Q and K in two parts needs (fAttnTaken)
const int           cDefaultThreads = 4;

double fNowMs() {
  return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// whether an environment variable is set to something that is not 0
bool fFlag(const char * pName) {
  const char * vValue = std::getenv(pName);
  return vValue != nullptr && vValue[0] != '\0' && std::strcmp(vValue, "0") != 0;
}

std::string fText(const char * pName, const std::string & pDefault) {
  const char * vValue = std::getenv(pName);
  return vValue != nullptr && vValue[0] != '\0' ? std::string(vValue) : pDefault;
}

// the folder of the program that is running; empty when it is not known
std::string fProgramFolder() {
  std::vector<char> aPath(4096);
  const ssize_t     vLength = readlink("/proc/self/exe", aPath.data(), aPath.size() - 1);
  if (vLength <= 0) {
    return std::string();
  }
  const std::string vPath(aPath.data(), size_t(vLength));
  const size_t      vSlash = vPath.find_last_of('/');
  return vSlash == std::string::npos || vSlash == 0 ? std::string() : vPath.substr(0, vSlash);
}

// pTotal in ranges of pLargest at most: where each one starts and how long it is
std::vector<std::pair<uint32_t, uint32_t>> fRanges(uint32_t pTotal, uint32_t pLargest) {
  std::vector<std::pair<uint32_t, uint32_t>> aRanges;
  for (uint32_t vFirst = 0; vFirst < pTotal; vFirst += pLargest) {
    aRanges.emplace_back(vFirst, std::min(pLargest, pTotal - vFirst));
  }
  return aRanges;
}

// The outputs of the members of a matrix (aOutputs, one after another) in ranges of pLargest at most: whole members
// together where that takes no more ranges than cutting the total, so that no range ends inside a member (the gate
// and the up of a feed-forward layer then go in a command each, which fMulMatGlu reads together); otherwise the
// total, cut.
std::vector<std::pair<uint32_t, uint32_t>> fRangesOfMembers(const std::vector<uint32_t> & aOutputs, uint32_t pLargest) {
  uint32_t vTotal = 0;
  for (uint32_t vOutputs : aOutputs) {
    vTotal += vOutputs;
  }
  const auto aCut = fRanges(vTotal, pLargest);
  std::vector<std::pair<uint32_t, uint32_t>> aWhole;
  uint32_t vFirst = 0;
  uint32_t vCount = 0;
  for (uint32_t vOutputs : aOutputs) {
    if (vOutputs > pLargest) {
      return aCut;
    }
    if (vCount + vOutputs > pLargest) {
      aWhole.emplace_back(vFirst, vCount);
      vFirst += vCount;
      vCount = 0;
    }
    vCount += vOutputs;
  }
  if (vCount > 0) {
    aWhole.emplace_back(vFirst, vCount);
  }
  return aWhole.size() <= aCut.size() ? aWhole : aCut;
}

// llama.cpp-xdna (2026/11/15): a matrix that would go in one command, of several members, the first one of
// cOverlapFirst outputs or more and the rest of cOverlapRest or more, goes in two, cut after the first member, so that
// the host reads the C of the first (the larger one) while the NPU multiplies the second (fMultiply): the input of
// the linear attention of Qwen3.5 4B (attn_qkv, and attn_gate, ssm_beta and ssm_alpha) and attn_q, attn_k and attn_v.
// Every output adds the same products either way. With GGML_XDNA_NO_OVERLAP it stays in one.
const uint32_t cOverlapFirst = 4096;
const uint32_t cOverlapRest  = 1024;

std::vector<std::pair<uint32_t, uint32_t>> fRangesForOverlap(const std::vector<uint32_t> & aOutputs,
                                                             std::vector<std::pair<uint32_t, uint32_t>> aRanges,
                                                             uint32_t pLargest) {
  // Two members of the same size (the gate and the up of a feed-forward layer) go in two halves aligned to 256 outputs
  // (a whole tile of K of ffn_down, see fSwigluTiles) whenever both halves fit in pLargest, even if the two members fit
  // together in one range (0.8B, 2B, TTS-0.6B, TTS-1.7B) or exceeded one range (27B): fMulMatGlu then fuses SwiGLU
  // straight into the input buffers of ffn_down and overlaps the first half with the second on the NPU.
  if (aOutputs.size() == 2 && aOutputs[0] == aOutputs[1] && aOutputs[0] >= cOverlapRest) {
    const uint32_t vMember = aOutputs[0];
    const uint32_t vHalf   = vMember / 2 / 256 * 256;
    if (vHalf >= cOverlapRest && vHalf <= pLargest && vMember - vHalf <= pLargest) {
      return {{0, vHalf}, {vHalf, vMember - vHalf}, {vMember, vHalf}, {vMember + vHalf, vMember - vHalf}};
    }
    if (vMember <= pLargest) {
      return {{0, vMember}, {vMember, vMember}};
    }
  }
  if (aRanges.size() != 1 || aOutputs.size() < 2 || aOutputs[0] < cOverlapFirst ||
      aRanges[0].second - aOutputs[0] < cOverlapRest) {
    return aRanges;
  }
  const uint32_t vTotal = aRanges[0].second;
  return {{0, aOutputs[0]}, {aOutputs[0], vTotal - aOutputs[0]}};
}

//
// What the NPU does, on the processor
//

// blocks of 8 values a row of pColumns values takes
uint32_t fBlocksOf(uint32_t pColumns) {
  return (pColumns + cXdnaBfp16Values - 1) / cXdnaBfp16Values;
}

// Encodes pRows rows of pColumns floats as the NPU reads them: blocks of 8 values along every row, the last one
// filled with zeros. False when a value is not finite.
bool fEncodeRows(const float * pValues, uint64_t pRows, uint32_t pColumns, uint8_t * pBlocks) {
  const uint32_t vWhole = cRounding == XdnaRounding::cNearest ? pColumns / cXdnaBfp16Values : 0;
  for (uint64_t vRow = 0; vRow < pRows; vRow++) {
    const float * vFrom = pValues + vRow * pColumns;
    // the blocks that are whole inside the row, one after the other, and then what is left
    if (!fXdnaBfp16EncodeNearest(vFrom, cXdnaBfp16Values, vWhole, pBlocks)) {
      return false;
    }
    pBlocks += size_t(vWhole) * cXdnaBfp16Bytes;
    for (uint32_t vColumn = vWhole * cXdnaBfp16Values; vColumn < pColumns; vColumn += cXdnaBfp16Values) {
      float          aValues[cXdnaBfp16Values] = {};
      const uint32_t vCount = std::min(cXdnaBfp16Values, pColumns - vColumn);
      std::memcpy(aValues, vFrom + vColumn, vCount * sizeof(float));
      if (!fXdnaBfp16Encode(aValues, pBlocks, cRounding)) {
        return false;
      }
      pBlocks += cXdnaBfp16Bytes;
    }
  }
  return true;
}

// the values of pBlocks blocks, 8 floats for each
void fDecodeBlocks(const uint8_t * pEncoded, uint32_t pBlocks, float * pValues) {
  for (uint32_t vBlock = 0; vBlock < pBlocks; vBlock++) {
    fXdnaBfp16Decode(pEncoded + size_t(vBlock) * cXdnaBfp16Bytes, pValues + size_t(vBlock) * cXdnaBfp16Values);
  }
}

// llama.cpp-xdna: the clones of a function are picked when the program starts by an indirect function (ifunc) of
// glibc, which musl (Alpine) does not have: there the build failed with "the call requires 'ifunc'"
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__)) && defined(__GLIBC__)
#  define XDNA_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#  define XDNA_CLONES
#endif

// Partial sums of the emulated dot product, a divisor of 16. Building with another number changes only the order
// of the sums, which is how doc/XDNA2.md measures what that order alone does to the results of a model.
#ifndef XDNA_EMULATION_LANES
#  define XDNA_EMULATION_LANES 16
#endif
static_assert(16 % XDNA_EMULATION_LANES == 0, "the partial sums of the emulation have to divide 16");

// the sum of the products of two rows of pCount values, a multiple of 16, in 32-bit floats as the cores add up
XDNA_CLONES float fDot(const float * pA, const float * pB, uint32_t pCount) {
  float aSums[XDNA_EMULATION_LANES] = {};
  for (uint32_t i = 0; i < pCount; i += XDNA_EMULATION_LANES) {
    for (uint32_t j = 0; j < XDNA_EMULATION_LANES; j++) {
      aSums[j] += pA[i + j] * pB[i + j];
    }
  }
  float vSum = 0;
  for (uint32_t j = 0; j < XDNA_EMULATION_LANES; j++) {
    vSum += aSums[j];
  }
  return vSum;
}

XDNA_CLONES float fDotRow(const float * pA, const float * pB, uint32_t pCount) {
  const uint32_t vAligned = pCount & ~15u;
  float          vSum     = vAligned > 0 ? fDot(pA, pB, vAligned) : 0.0f;
  for (uint32_t i = vAligned; i < pCount; i++) {
    vSum += pA[i] * pB[i];
  }
  return vSum;
}

//
// What the backends of a process share
//

struct XdnaSettings {
  bool        vEmulate = false;
  std::string vDevicePath;
  std::string vOverlayPath;
  std::string vManifestPath;
  std::string vGemvOverlayPath;   // the overlay for generating tokens; empty when there is none
  std::string vGemvManifestPath;
  std::string vMoeOverlayPath;    // the mixture-of-experts overlay; empty when there is none
  std::string vMoeManifestPath;
  uint64_t    vMinRows = 1;  // rows a multiplication needs for the backend to take it
  bool        vNoGlu   = false;  // GGML_XDNA_NO_GLU: the SwiGLUs stay with the CPU backend
  bool        vNoOverlap = false;  // GGML_XDNA_NO_OVERLAP: every command is waited for before going on
  bool        vNoFuse    = false;  // GGML_XDNA_NO_FUSE: the ADDs, CONCATs, RMS_NORMs and MULs stay with the CPU
  bool        vNoChain   = false;  // GGML_XDNA_NO_CHAIN_FFN: with the chained overlay, the feed-forward blocks go
                                   // as separate multiplications
  bool        vNoChainNext = false;  // GGML_XDNA_NO_CHAIN_NEXT: and without the norm and the multiplications of the
                                     // next layer
  bool        vNoStateBuffers = false;  // GGML_XDNA_NO_STATE_BUFFERS: the recurrent states stay in buffers of the CPU
  bool        vNoGdn = false;  // llama.cpp-xdna, GGML_XDNA_NO_GDN: the recurrences of a batch stay with the CPU
  uint32_t    vGdnParts = 4;   // GGML_XDNA_GDN_PARTS: the commands of a recurrence of a batch (1 to 4)
  bool        vNoPackGlu = false;  // llama.cpp-xdna, GGML_XDNA_NO_PACK_GLU: the SwiGLU of a batch written as floats,
                                   // and the rows of ffn_down converted from them (fMulMatGlu)
  bool        vNoGated   = false;  // llama.cpp-xdna, GGML_XDNA_NO_GATED: the gated norm of a batch with the CPU
                                   // backend (fGatedTaken)
  bool        vNoL2      = false;  // llama.cpp-xdna, GGML_XDNA_NO_L2: the L2 norms of q and k of a batch with the CPU
                                   // backend (fL2Taken)
  bool        vLinear   = false;  // llama.cpp-xdna, GGML_XDNA_LINEAR: the layers of linear attention on the NPU
                                  // while a token is generated (fLinearTaken)
  bool        vNoAttn   = false;  // llama.cpp-xdna, GGML_XDNA_NO_ATTN: the attention of a batch with the CPU backend
                                  // (fAttnTaken)
  bool        vNoAttnOverlap = false;  // GGML_XDNA_NO_ATTN_OVERLAP: its items one after the other (fFlashAttention)
  bool        vNoConv = false;  // llama.cpp-xdna, GGML_XDNA_NO_CONV: the short convolution of a batch, its SiLU and
                                // the copy of its states with the CPU backend (fConvTaken)
  bool        vNoConvFold = false;  // GGML_XDNA_NO_CONV_FOLD: and those on the host, not while C is read (fConvFold)
  bool        vNoGdnFold  = false;  // GGML_XDNA_NO_GDN_FOLD: disable direct GDN prepack, overlapped L2 and gated-norm unpack
  uint64_t    vWeightsLimit    = 0;  // llama.cpp-xdna, GGML_XDNA_WEIGHTS_GB: the bytes the copies of the weights for the
                                     // NPU may take (fPlaceWeights); 0 for no limit
  uint64_t    vMoeWeightsLimit   = 0;  // total NPU weight budget when MADV_DONTNEED drops mapped expert pages
  uint64_t    vDenseTotalLimit   = 0;  // total NPU weight budget for dense models when MADV_DONTNEED drops mapped pages
  uint64_t    vDenseChainLimit   = 0;  // chain weight budget for two-tier full-NPU dense models (e.g. 27B)
  bool        vWeightsLimitExplicit = false;
  bool        vCopyWeights          = false;
  bool        vMoeGemv = false;  // llama.cpp-xdna, GGML_XDNA_MOE_GEMV: with a mixture of experts, the multiplications of
                                 // a few rows on the NPU too (fSawExperts)
  int         vAttnSplit = -1;  // GGML_XDNA_ATTN_SPLIT: 1 Q and K always in two parts of bfp16, 0 never, and by
                                // default when the queries do not come from an RMS norm (fNormedQueries)
  bool        vNoMoe        = false;  // llama.cpp-xdna, GGML_XDNA_NO_MOE: MUL_MAT_ID stays with the CPU backend
  uint64_t    vMoeMinTokens = 1;      // GGML_XDNA_MOE_MIN_TOKENS: minimum tokens for MUL_MAT_ID on the NPU
  bool        vNoMoeDense   = false;  // GGML_XDNA_NO_MOE_DENSE: dense and shared-expert generation stays with the CPU
  bool        vMoeAttnOut   = false;  // GGML_XDNA_MOE_ATTN_OUT: also run ssm_out and attn_output of MoE layers on NPU
  int         vPromptFormat = -1;     // GGML_XDNA_PROMPT_FORMAT: -1 auto, 0 bfp16, 1 q6k, 2 q8_0
  int         vFull         = 0;      // GGML_XDNA_FULL: 1 single XDNA subgraph (0 splits), 2 also 1-token attn on GEMM
};

// A matrix of weights converted for the overlay for generating tokens. Like XdnaMatrix, it can hold the matrices
// of several tensors that multiply the same rows; it is never cut in pieces.
struct XdnaGemvMatrix {
  uint32_t                    vInputs  = 0;
  uint32_t                    vOutputs = 0;
  XdnaGemvWeights            vWeights;
  XdnaChainWeights           vChainWeights;  // in their place with the chained overlay
  std::vector<enum ggml_type> aTypes;    // of every tensor it came from
  std::vector<uint32_t>       aOutputs;  // and the outputs of each
};

// One piece of a matrix of weights: some of its inputs and some of its outputs, as the NPU reads them.
struct XdnaPiece {
  uint32_t             vFirstInput  = 0;
  uint32_t             vInputs      = 0;
  uint32_t             vFirstOutput = 0;
  uint32_t             vOutputs     = 0;
  XdnaWeights         vWeights;  // in a buffer of the NPU
  std::vector<uint8_t> aBlocks;   // or, emulating, here: vOutputs rows of blocks of 8 values along the inputs
};

// A matrix of weights, converted. The pieces are not freed with it: fFreeMatrix does it. It can hold the matrices
// of several tensors of weights that multiply the same rows, one after the other along the outputs (see
// fGraphCompute).
struct XdnaMatrix {
  uint32_t                    vInputs  = 0;
  uint32_t                    vOutputs = 0;
  uint32_t                    vRangesK = 0;  // pieces along the inputs and along the outputs
  uint32_t                    vRangesN = 0;
  std::vector<XdnaPiece>     aPieces;       // [range of inputs * vRangesN + range of outputs]
  std::vector<enum ggml_type> aTypes;        // of every tensor it came from
  std::vector<uint32_t>       aOutputs;      // and the outputs of each
};

// what a backend did, to report it when it is freed
struct XdnaTotals {
  uint64_t vOperations = 0;  // multiplications of a graph
  uint64_t vCommands   = 0;  // multiplications of the NPU they took
  uint64_t vRows       = 0;
  double   vPlanMs     = 0;  // writing instruction streams
  double   vPackMs     = 0;  // converting rows
  double   vNpuMs      = 0;  // from the submission of a command to its end
  double   vUnpackMs   = 0;  // reading results back
  double   vFlushMs    = 0;  // of it, taking them out of the cache of the processor
  double   vTotalMs    = 0;  // inside the backend: the above, the conversion of weights and what is around them
  uint64_t vMatrices   = 0;  // matrices of weights it converted
  uint64_t vWeights    = 0;
  double   vWeightsMs  = 0;
  uint64_t vCommandsRows = 0;  // of the commands, those of the overlay for generating tokens
  double   vNpuRowsMs    = 0;  // and their time in the NPU
  uint64_t vGlusFused    = 0;  // SwiGLUs computed while the C of their multiplications was read (fMulMatGlu)
  uint64_t vGlusHost     = 0;  // and on the host from their results (fGluOnHost)
  double   vGluMs        = 0;  // the time of the latter
  uint64_t vAddsFused    = 0;  // residual ADDs computed while C was read
  uint64_t vAddsHost     = 0;  // and on the host (fAddOnHost)
  double   vAddMs        = 0;  // the time of the latter
  uint64_t vConcatsFused = 0;  // CONCATs whose multiplication wrote its result straight into them
  uint64_t vConcatsHost  = 0;  // and on the host (fConcatOnHost)
  uint64_t vNormsFused   = 0;  // RMS_NORMs with their MUL applied to the rows as they were converted
  uint64_t vNormsHost    = 0;  // and RMS_NORMs or MULs on the host
  uint64_t vChainsFfn    = 0;  // feed-forward blocks in one command of the chained overlay (fFeedForward)
  double   vChainsFfnMs  = 0;  // and their time in the NPU
  uint64_t vChainsNext   = 0;  // of them, those with the norm and the multiplications of the next layer
  uint64_t vChainsLogits = 0;  // and of those, the ones whose next layer is the output norm and the logits
  uint64_t vRowsHost     = 0;  // GET_ROWS of the only row of a result copied on the host (fRowsOnHost)
  uint64_t vSplitRows    = 0;  // commands after the first for rows that did not fit in one (fRowsPerCommand)
  uint64_t vGdns         = 0;  // llama.cpp-xdna: recurrences of a batch of tokens on the NPU (fGatedDeltaNet)
  double   vGdnNpuMs     = 0;
  double   vGdnPackMs    = 0;  // writing their inputs
  double   vGdnUnpackMs  = 0;  // reading their results
  double   vDownPackMs     = 0;  // llama.cpp-xdna: of the above, the multiplications of many rows whose rows are a
  double   vDownNpuMs      = 0;  // SwiGLU (ffn_down of a feed-forward layer): converting their rows, the NPU and
  double   vDownUnpackMs   = 0;  // reading their results (with the sums of their pieces along the inputs)
  uint64_t vDowns          = 0;
  double   vSwigluUnpackMs = 0;  // reading the C of the gate and the up with their SwiGLU (fMulMatGlu)
  uint64_t vGlusPacked     = 0;  // of those SwiGLUs, the ones converted straight into the rows of ffn_down
  uint64_t vGatedPacked    = 0;  // gated norms of a batch converted straight into the rows of ssm_out (fGatedNorm)
  double   vGatedMs        = 0;  // and their time in the backend: converting, and flushing the cache
  double   vGatedPackMs    = 0;
  double   vGatedFlushMs   = 0;
  uint64_t vGatedHost      = 0;  // operations of gated norms on the host (fGatedOnHost)
  uint64_t vL2Folded       = 0;  // recurrences of a batch that took the L2 norms of their q and k (fL2Fold)
  uint64_t vL2Host         = 0;  // parts of L2 norms on the host (fL2OnHost)
  uint64_t vGateHost       = 0;  // operations of the gates of recurrences of a batch on the host (fGateOnHost)
  // llama.cpp-xdna: the short convolutions of a batch computed while the C of their projection was read (fConvFold),
  // the ones on the host (fConvOnHost) with the time they took, the SiLUs of those alone, and the copies of their states
  uint64_t vConvsFused  = 0;
  uint64_t vConvsHost   = 0;
  double   vConvHostMs  = 0;
  uint64_t vConvSilus   = 0;
  uint64_t vConvCopies  = 0;
  uint64_t vLinears      = 0;  // llama.cpp-xdna: layers of linear attention in one phase of the chained overlay
  double   vLinearNpuMs  = 0;  // (fLinearAttention), their time in the NPU
  double   vLinearMs     = 0;  // and in the backend
  uint64_t vLinearHost   = 0;  // operations of such layers on the host (fLinearOnHost)
  uint64_t vAttns        = 0;  // llama.cpp-xdna: attentions of a batch on the NPU (fFlashAttention), and their
  uint64_t vAttnCommands = 0;  // commands; their time in the NPU, converting Q, K and V, converting P from S,
  double   vAttnNpuMs    = 0;  // reading O and in the backend
  double   vAttnPackMs   = 0;
  double   vAttnPMs      = 0;
  double   vAttnUnpackMs = 0;
  double   vAttnFlushMs  = 0;  // of all of it, flushing the cache
  double   vAttnMs       = 0;
  uint64_t vMoeOps          = 0;  // llama.cpp-xdna: MUL_MAT_ID operations of routed experts on the NPU
  uint64_t vMoeCommands     = 0;
  uint64_t vMoeFusedGlu     = 0;  // gate + up + SwiGLU fused in one MoE command
  uint64_t vMoeTailFused    = 0;  // weighted expert reduction fused into reading Y of down_exps
  uint64_t vMoeFusedShexp   = 0;  // shared experts folded as virtual experts into MUL_MAT_ID commands
  uint64_t vMoeTailHost     = 0;
  uint64_t vMoeMatrices     = 0;
  double   vMoeWeightsMs    = 0;
  double   vMoeNpuMs        = 0;
  double   vMoePackMs       = 0;
  double   vMoeUnpackMs     = 0;
  double   vMoePlanMs       = 0;
  double   vMoeMs           = 0;
  uint64_t vMoeDenseCmds    = 0;  // dense projections of 1 to 8 rows run on xdna2-moe without overlay switches
  double   vMoeDenseNpuMs   = 0;
  uint64_t vChainsShexp     = 0;  // shared-expert blocks chained with next-layer norm and projections
  double   vChainsShexpMs   = 0;
  uint64_t vFullOpsHost     = 0;  // GGML_XDNA_FULL glue operations executed inside the XDNA subgraph
  double   vFullOpsMs       = 0;
};

struct XdnaShared {
  // what was found when the backend was registered; it does not change afterwards
  bool          vUsable = false;
  XdnaSettings vSettings;
  std::string   vDescription;
  XdnaOverlay  vOverlay;
  uint32_t      vMaxInputs  = 0;  // of one multiplication of the overlay
  uint32_t      vMaxOutputs = 0;
  bool             vHasGemv = false;  // the overlay for generating tokens is there (either of the two)
  XdnaGemvOverlay vGemvOverlay;
  bool              vHasChain = false;  // and it is the chained one
  XdnaChainOverlay vChainOverlay;
  bool              vHasMoe   = false;  // the mixture-of-experts overlay is there
  XdnaMoeOverlay   vMoeOverlay;

  // the NPU and what is kept in it: one thread at a time
  std::mutex  vMutex;
  XdnaRunner vRunner;
  bool        vIsOpen   = false;
  uint32_t    vBackends = 0;
  // by where the weights of the model are: one place for a matrix, or the places of the tensors held together
  std::map<std::vector<const void *>, XdnaMatrix> dMatrices;
  // the same for the overlay for generating tokens
  std::map<std::vector<const void *>, XdnaGemvMatrix> dGemvMatrices;
  // llama.cpp-xdna: expert matrices (or {gate, up} pairs) converted for the mixture-of-experts overlay
  std::map<std::vector<const void *>, XdnaMoeWeights> dMoeMatrices;
  // llama.cpp-xdna: foreign host mappings (fDeviceBufferFromHostPtr) -> (size, is PROT_READ | MAP_SHARED file mapping)
  std::map<const void *, std::pair<size_t, bool>> dForeignMappings;
  // buffers of the driver that hold recurrent states of a model (fStateBufferType), by where the host sees them
  std::map<const void *, XdnaBuffer> dStateBuffers;
  // llama.cpp-xdna: the fixed objects of every layer of linear attention (fLinearAttention), by where the weights of
  // its convolution and of its gated norm are; freed with the buffers of weights of this backend
  std::map<std::pair<const void *, const void *>, XdnaChainWeights> dLinear;
  // llama.cpp-xdna: the keys and the pieces of the values of the attention of a batch as the NPU reads them
  // (fFlashAttention), converted again for every head; kept while the NPU is open
  XdnaWeights vAttnK;
  XdnaWeights aAttnV[2][cXdnaRunnerSlotsC];  // two sets: the head being multiplied and the next one
  // llama.cpp-xdna: cached prompt bytes, per-tensor mapped-page release tracking, and two reusable stream matrices for
  // dense models whose full bfp16 + chain copies exceed host memory (e.g. Qwen3.8 27B)
  uint64_t                        vCachedPromptBytes = 0;
  XdnaMatrix                      aStreamMatrices[2];
  std::vector<const void *>       aStreamKeys[2];
  std::unordered_set<const void *> sPromptCached;
  std::unordered_set<const void *> sGemvCached;
  std::unordered_set<const void *> sCleanStates;
};

// Looks for the overlay and for the NPU, to know whether the backend has a device to offer. It only queries.
void fProbe(XdnaShared & pShared) {
  if (fFlag("GGML_XDNA_DISABLE")) {
    GGML_LOG_DEBUG("%s: disabled by GGML_XDNA_DISABLE\n", cName);
    return;
  }
  XdnaSettings & vSettings = pShared.vSettings;
  vSettings.vEmulate    = fFlag("GGML_XDNA_EMULATE");
  vSettings.vNoGlu      = fFlag("GGML_XDNA_NO_GLU");
  vSettings.vNoOverlap  = fFlag("GGML_XDNA_NO_OVERLAP");
  vSettings.vNoFuse     = fFlag("GGML_XDNA_NO_FUSE");
  vSettings.vNoChain    = fFlag("GGML_XDNA_NO_CHAIN_FFN");
  vSettings.vNoChainNext = fFlag("GGML_XDNA_NO_CHAIN_NEXT");
  vSettings.vNoStateBuffers = fFlag("GGML_XDNA_NO_STATE_BUFFERS");
  vSettings.vNoGdn          = fFlag("GGML_XDNA_NO_GDN");
  vSettings.vLinear         = fFlag("GGML_XDNA_LINEAR");
  vSettings.vNoPackGlu      = fFlag("GGML_XDNA_NO_PACK_GLU");
  vSettings.vNoGated        = fFlag("GGML_XDNA_NO_GATED");
  vSettings.vNoL2           = fFlag("GGML_XDNA_NO_L2");
  vSettings.vNoConv         = fFlag("GGML_XDNA_NO_CONV");
  vSettings.vNoConvFold     = fFlag("GGML_XDNA_NO_CONV_FOLD");
  vSettings.vNoGdnFold      = fFlag("GGML_XDNA_NO_GDN_FOLD");
  vSettings.vMoeGemv        = fFlag("GGML_XDNA_MOE_GEMV");
  vSettings.vNoMoe          = fFlag("GGML_XDNA_NO_MOE");
  vSettings.vNoMoeDense     = fFlag("GGML_XDNA_NO_MOE_DENSE");
  vSettings.vMoeAttnOut     = fFlag("GGML_XDNA_MOE_ATTN_OUT");
  vSettings.vCopyWeights    = fFlag("GGML_XDNA_COPY_WEIGHTS");
  vSettings.vMoeMinTokens   = std::max<unsigned long long>(
      1, std::strtoull(fText("GGML_XDNA_MOE_MIN_TOKENS", "1").c_str(), nullptr, 10));
  // llama.cpp-xdna: 40 % of the memory of the machine by default for copies that live beside the mapped file; when
  // MADV_DONTNEED releases the mapped file pages of every taken 3D expert tensor as soon as its single xdna2-moe copy
  // is built (vMoeMinTokens <= 1 and !vCopyWeights), each expert copy replaces its file pages instead of adding on top
  // of them, so up to 76 % of host memory may hold total NPU weight buffers while leaving room for the system, the
  // caches and the remaining mapped pages (fPlaceWeights, fPlaceMoeWeights)
  {
    const std::string vLimit = fText("GGML_XDNA_WEIGHTS_GB", "");
    if (!vLimit.empty()) {
      const double vGigabytes         = std::strtod(vLimit.c_str(), nullptr);
      vSettings.vWeightsLimit         = vGigabytes > 0 ? uint64_t(vGigabytes * 1e9) : 0;
      vSettings.vMoeWeightsLimit      = vSettings.vWeightsLimit;
      vSettings.vWeightsLimitExplicit = true;
    } else {
      const long   vPages    = sysconf(_SC_PHYS_PAGES);
      const long   vPageSize = sysconf(_SC_PAGE_SIZE);
      const double vRamBytes = vPages > 0 && vPageSize > 0 ? double(vPages) * double(vPageSize) : 0.0;
      vSettings.vWeightsLimit    = vRamBytes > 0.0 ? uint64_t(vRamBytes * 0.4) : 0;
      const double vMoeRatio = vRamBytes >= 58.0e9 ? 0.78 : 0.76;
      vSettings.vMoeWeightsLimit =
          !vSettings.vEmulate && !vSettings.vCopyWeights && vSettings.vMoeMinTokens <= 1 && vRamBytes > 0.0 ?
              uint64_t(vRamBytes * vMoeRatio) :
              vSettings.vWeightsLimit;
      if (!vSettings.vEmulate && !vSettings.vCopyWeights && vRamBytes > 0.0) {
        vSettings.vDenseTotalLimit = uint64_t(vRamBytes * 0.72);
        vSettings.vDenseChainLimit = uint64_t(vRamBytes * 0.48);
      }
    }
  }
  {
    const std::string vPromptFmt = fText("GGML_XDNA_PROMPT_FORMAT", "auto");
    vSettings.vPromptFormat      = vPromptFmt == "bfp16" ? 0 :
                                   vPromptFmt == "q6k"   ? 1 :
                                   vPromptFmt == "q8_0"  ? 2 :
                                                           -1;
  }
  {
    const std::string vFullStr = fText("GGML_XDNA_FULL", "");
    vSettings.vFull            = vFullStr == "2" ? 2 : (fFlag("GGML_XDNA_FULL") ? 1 : 0);
    if (vSettings.vFull >= 1 && !vSettings.vEmulate && !vSettings.vNoMoeDense) {
      vSettings.vMoeAttnOut = true;
    }
  }
  vSettings.vNoAttn         = fFlag("GGML_XDNA_NO_ATTN");
  vSettings.vNoAttnOverlap  = fFlag("GGML_XDNA_NO_ATTN_OVERLAP");
  {
    const std::string vSplit = fText("GGML_XDNA_ATTN_SPLIT", "");
    vSettings.vAttnSplit     = vSplit == "1" ? 1 : vSplit == "0" ? 0 : -1;
  }
  vSettings.vGdnParts       = uint32_t(std::min<unsigned long long>(
    4, std::max<unsigned long long>(1, std::strtoull(fText("GGML_XDNA_GDN_PARTS", "4").c_str(), nullptr, 10))));
  vSettings.vDevicePath = fText("GGML_XDNA_DEVICE", cDefaultDevice);
  // The overlay configures the NPU, and a wrong one can leave it unusable until the computer is powered off:
  // it is looked for where the program is, or where the variable says, never in the current folder.
  const std::string vFolder = fText("GGML_XDNA_OVERLAY_DIR", fProgramFolder());
  vSettings.vOverlayPath  = vFolder + "/" + cOverlayFile;
  vSettings.vManifestPath = vFolder + "/" + cManifestFile;
  const std::string vMinRows = fText("GGML_XDNA_MIN_BATCH", std::to_string(cDefaultMinRows));
  vSettings.vMinRows = std::max<unsigned long long>(1, std::strtoull(vMinRows.c_str(), nullptr, 10));

  std::string vError;
  if (vFolder.empty()) {
    vError = "the folder of the program is not known; GGML_XDNA_OVERLAY_DIR says where the overlay is";
  } else if (pShared.vOverlay.mLoad(vSettings.vManifestPath, vError) &&
             fXdnaGemmLimits(pShared.vOverlay, pShared.vMaxInputs, pShared.vMaxOutputs, vError)) {
    if (vSettings.vEmulate) {
      pShared.vDescription = "AMD Ryzen AI NPU (XDNA2), emulated on the processor";
      pShared.vUsable      = true;
    } else if (access(vSettings.vOverlayPath.c_str(), R_OK) != 0) {
      vError = "cannot read the overlay " + vSettings.vOverlayPath;
    } else {
      XdnaDevice vDevice;
      XdnaInfo   vInfo;
      if (vDevice.mOpen(vSettings.vDevicePath, vError) && vDevice.mQueryInfo(vInfo, vError)) {
        if (vInfo.vColumns != pShared.vOverlay.vColumns) {
          vError = "the overlay is for an array of " + std::to_string(pShared.vOverlay.vColumns) +
                   " columns and the NPU has " + std::to_string(vInfo.vColumns);
        } else {
          pShared.vDescription = "AMD Ryzen AI NPU (XDNA2), " + std::to_string(vInfo.vColumns) + " x " +
                                 std::to_string(vInfo.vRows) + " tiles, firmware " +
                                 std::to_string(vInfo.vFirmwareMajor) + "." + std::to_string(vInfo.vFirmwareMinor) +
                                 "." + std::to_string(vInfo.vFirmwarePatch) + "." +
                                 std::to_string(vInfo.vFirmwareBuild);
          pShared.vUsable      = true;
        }
      }
    }
  }
  if (!pShared.vUsable) {
    GGML_LOG_INFO("%s: the NPU is not used: %s\n", cName, vError.c_str());
    return;
  }
  if (!vSettings.vEmulate && !vSettings.vNoMoe) {
    const std::string vMoeOverlay  = vFolder + "/" + cMoeOverlayFile;
    const std::string vMoeManifest = vFolder + "/" + cMoeManifestFile;
    if (access(vMoeOverlay.c_str(), R_OK) == 0) {
      std::string vMoeError;
      if (pShared.vMoeOverlay.mLoad(vMoeManifest, vMoeError) &&
          pShared.vMoeOverlay.vColumns == pShared.vOverlay.vColumns) {
        vSettings.vMoeOverlayPath  = vMoeOverlay;
        vSettings.vMoeManifestPath = vMoeManifest;
        pShared.vHasMoe            = true;
      } else {
        GGML_LOG_INFO("%s: the mixture-of-experts overlay is not used: %s\n", cName,
                      vMoeError.empty() ? "it is for an array of another width" : vMoeError.c_str());
      }
    }
  }
  // The overlay for generating tokens, next to the other one: the chained one if it is there (and GGML_XDNA_NO_CHAIN
  // does not say otherwise), the other one if not. Emulating, the processor does the multiplications of few rows as
  // before.
  if (vSettings.vEmulate || fFlag("GGML_XDNA_NO_GEMV")) {
    GGML_LOG_DEBUG("%s: no overlay for generating tokens\n", cName);
    return;
  }
  const std::string vChainOverlay  = vFolder + "/" + cChainOverlayFile;
  const std::string vChainManifest = vFolder + "/" + cChainManifestFile;
  if (!fFlag("GGML_XDNA_NO_CHAIN") && access(vChainOverlay.c_str(), R_OK) == 0) {
    std::string vChainError;
    if (pShared.vChainOverlay.mLoad(vChainManifest, vChainError) &&
        pShared.vChainOverlay.vColumns == pShared.vOverlay.vColumns) {
      vSettings.vGemvOverlayPath  = vChainOverlay;
      vSettings.vGemvManifestPath = vChainManifest;
      pShared.vHasGemv            = true;
      pShared.vHasChain           = true;
      return;
    }
    GGML_LOG_INFO("%s: the chained overlay for generating tokens is not used: %s\n", cName,
                  vChainError.empty() ? "it is for an array of another width" : vChainError.c_str());
  }
  const std::string vGemvOverlay  = vFolder + "/" + cGemvOverlayFile;
  const std::string vGemvManifest = vFolder + "/" + cGemvManifestFile;
  if (access(vGemvOverlay.c_str(), R_OK) != 0) {
    GGML_LOG_DEBUG("%s: no overlay for generating tokens\n", cName);
    return;
  }
  std::string vGemvError;
  if (!pShared.vGemvOverlay.mLoad(vGemvManifest, vGemvError) ||
      pShared.vGemvOverlay.vColumns != pShared.vOverlay.vColumns) {
    GGML_LOG_INFO("%s: the overlay for generating tokens is not used: %s\n", cName,
                  vGemvError.empty() ? "it is for an array of another width" : vGemvError.c_str());
    return;
  }
  vSettings.vGemvOverlayPath  = vGemvOverlay;
  vSettings.vGemvManifestPath = vGemvManifest;
  pShared.vHasGemv            = true;
}

// It is never destroyed: buffers and backends may be freed while the process ends, after the static objects.
XdnaShared & fShared() {
  static XdnaShared * vShared = [] {
    XdnaShared * vNew = new XdnaShared;
    fProbe(*vNew);
    return vNew;
  }();
  return *vShared;
}

void fFreeMatrix(XdnaShared & pShared, XdnaMatrix & pMatrix) {
  for (XdnaPiece & vPiece : pMatrix.aPieces) {
    if (vPiece.vWeights.vBuffer.vMap != nullptr) {
      pShared.vRunner.mFreeWeights(vPiece.vWeights);
    }
  }
  pMatrix = XdnaMatrix();
}

void fFreeGemvMatrix(XdnaShared & pShared, XdnaGemvMatrix & pMatrix) {
  if (pMatrix.vWeights.vBuffer.vMap != nullptr) {
    pShared.vRunner.mFreeGemvWeights(pMatrix.vWeights);
  }
  if (pMatrix.vChainWeights.vBuffer.vMap != nullptr) {
    pShared.vRunner.mFreeChainWeights(pMatrix.vChainWeights);
  }
  pMatrix = XdnaGemvMatrix();
}

// Whether one command of the overlay for generating tokens takes pRows rows of pInputs values: 1 to 8 rows, which fit
// in its cores.
bool fRowsFit(const XdnaShared & pShared, uint64_t pRows, uint32_t pInputs) {
  if (!pShared.vHasGemv || pRows == 0 || pRows > cXdnaGemvRowsMax) {
    return false;
  }
  std::string vError;
  if (pShared.vHasChain) {
    // as the runner sends it (mChainMultiplyRows): X in floats for one row, in bfp16 for more
    const XdnaGemvPart vPart = {XdnaGemvFormat::cBfp16, 1};
    XdnaChainMatrix    vMatrix;
    XdnaChainPhase     vPhase;
    XdnaChainCommand   vCommand;
    vPhase.vPrologue = pRows == 1 ? XdnaChainPrologue::cFloats : XdnaChainPrologue::cBfp16;
    vPhase.vRows     = uint32_t(pRows);
    return fXdnaChainMatrixShape(pShared.vChainOverlay, pInputs, &vPart, 1, vMatrix, vError) &&
           fXdnaChainCommandShape(pShared.vChainOverlay, &vPhase, 1, &vMatrix, 1, vCommand, vError);
  }
  XdnaGemvShape vShape;
  return fXdnaGemvShape(pShared.vGemvOverlay, uint32_t(pRows), pInputs, 1, XdnaGemvFormat::cBfp16, vShape, vError);
}

// The rows of every command when the chained overlay multiplies pRows rows (1 to 8) of pInputs values: all of them in
// one when they fit, otherwise as few commands as fit, with the rows shared out evenly (8 rows of K = 9216 do not fit
// in the 32256 bytes of X of a core: two commands of 4, each reading all the weights, take less than the processor
// with the 8). 0 when not even one row fits, or with the overlay before the chained one, which takes what fits.
uint64_t fRowsPerCommand(const XdnaShared & pShared, uint64_t pRows, uint32_t pInputs) {
  if (fRowsFit(pShared, pRows, pInputs)) {
    return pRows;
  }
  if (!pShared.vHasChain || pRows == 0 || pRows > cXdnaGemvRowsMax) {
    return 0;
  }
  for (uint64_t vEach = pRows - 1; vEach >= 1; vEach--) {
    if (fRowsFit(pShared, vEach, pInputs)) {
      const uint64_t vCommands = (pRows + vEach - 1) / vEach;
      return (pRows + vCommands - 1) / vCommands;
    }
  }
  return 0;
}

// Whether a multiplication of pRows rows of pInputs values goes to the overlay for generating tokens: in one command,
// or with the chained overlay in a few (fRowsPerCommand).
bool fByRows(const XdnaShared & pShared, uint64_t pRows, uint32_t pInputs) {
  return fRowsPerCommand(pShared, pRows, pInputs) > 0;
}

// The NPU is held while a backend, the weights of a model or its recurrent states need it: the buffers of the driver
// serve the NPU only while its file stays open.
void fCloseIfIdle(XdnaShared & pShared) {
  if (pShared.vIsOpen && pShared.vBackends == 0 && pShared.dMatrices.empty() && pShared.dGemvMatrices.empty() &&
      pShared.dMoeMatrices.empty() && pShared.dStateBuffers.empty()) {
    // llama.cpp-xdna: the fixed objects of the layers of linear attention of a model that is gone
    for (auto & vEntry : pShared.dLinear) {
      pShared.vRunner.mFreeChainWeights(vEntry.second);
    }
    pShared.dLinear.clear();
    // llama.cpp-xdna: and those of the attention of a batch
    pShared.vRunner.mFreeWeights(pShared.vAttnK);
    for (auto & aSet : pShared.aAttnV) {
      for (XdnaWeights & vPiece : aSet) {
        pShared.vRunner.mFreeWeights(vPiece);
      }
    }
    pShared.vRunner.mClose();
    pShared.vIsOpen = false;
  }
}

// Opens the NPU, with the overlay for generating tokens if it is there, unless it is open already. With the mutex of
// pShared held, and not emulating.
bool fOpenRunner(XdnaShared & pShared, std::string & pError) {
  if (pShared.vIsOpen) {
    return true;
  }
  // with the overlay for generating tokens, if it is there, as the second compute unit of the context
  if (!pShared.vRunner.mOpen(pShared.vSettings.vDevicePath, pShared.vSettings.vOverlayPath,
                             pShared.vSettings.vManifestPath, pError, pShared.vSettings.vGemvOverlayPath,
                             pShared.vSettings.vGemvManifestPath, pShared.vSettings.vMoeOverlayPath,
                             pShared.vSettings.vMoeManifestPath)) {
    return false;
  }
  // GGML_XDNA_NO_PARK leaves the threads of OpenMP spinning while the NPU works, for comparing
  pShared.vRunner.mSetPark(!fFlag("GGML_XDNA_NO_PARK"));
  pShared.vRunner.mSetGdnParts(pShared.vSettings.vGdnParts);
  // GGML_XDNA_NO_EARLY_DESCRIPTORS writes the descriptors of a phase after the wait for the phase before
  pShared.vRunner.mSetEarlyDescriptors(!fFlag("GGML_XDNA_NO_EARLY_DESCRIPTORS"));
  // GGML_XDNA_NO_USER_FLUSH flushes the cache around a chained command with the SYNC_BO of the driver
  pShared.vRunner.mSetUserFlush(!fFlag("GGML_XDNA_NO_USER_FLUSH"));
  // GGML_XDNA_NO_FILLS_FIRST queues the drains of a phase before its fills, as before
  pShared.vRunner.mSetFillsFirst(!fFlag("GGML_XDNA_NO_FILLS_FIRST"));
  pShared.vIsOpen = true;
  return true;
}

//
// Backend (stream)
//

// llama.cpp-xdna: the operations of a layer of linear attention of Qwen3.5 while a token is generated, as
// build_layer_attn_linear makes them (src/models/qwen35.cpp), from the short convolution to the gated norm, and what
// they read: one phase of the chained overlay of version 9 (fMatchLinear, fLinearAttention)
struct XdnaLinearBlock {
  const ggml_tensor * vConv     = nullptr;  // SSM_CONV of the CONCAT of the convolution by tokens
  const ggml_tensor * vSilu     = nullptr;  // its SiLU, whose views are q, k and v
  const ggml_tensor * vNormQ    = nullptr;  // the L2 norms of q and k: an RMS_NORM and a SCALE each
  const ggml_tensor * vScaleQ   = nullptr;
  const ggml_tensor * vNormK    = nullptr;
  const ggml_tensor * vScaleK   = nullptr;
  const ggml_tensor * vBiased   = nullptr;  // alpha + ssm_dt
  const ggml_tensor * vSoftplus = nullptr;  // and its softplus
  const ggml_tensor * vGate     = nullptr;  // g = softplus(alpha + ssm_dt) * ssm_a
  const ggml_tensor * vBeta     = nullptr;  // the sigmoid of the projection of beta
  const ggml_tensor * vGdn      = nullptr;  // the recurrence, in the cell of the cache
  const ggml_tensor * vNormO    = nullptr;  // the gated norm: the RMS_NORM of its outputs, by ssm_norm, by silu(z)
  const ggml_tensor * vWeighted = nullptr;
  const ggml_tensor * vSiluZ    = nullptr;
  const ggml_tensor * vOut      = nullptr;  // the last of them: the layer runs when the graph gets to it
  const float *       vInput    = nullptr;  // the CONCAT: [4][8192], the 3 old states and the token
  const float *       vZ        = nullptr;  // [32][128]
  const float *       vAlpha    = nullptr;  // the projections of alpha and of beta, 32 each
  const float *       vBetaIn   = nullptr;
  const float *       vDt       = nullptr;  // ssm_dt and ssm_a, 32 each
  const float *       vA        = nullptr;
  const ggml_tensor * vKernel   = nullptr;  // the weights of the convolution, [8192][4], and ssm_norm, 128
  const ggml_tensor * vNorm     = nullptr;
  float               vEpsL2    = 0;  // x / sqrt(the sum of the squares + vEpsL2): the L2 norms of q and k
  float               vEpsGated = 0;  // that of the RMS_NORM of the gated norm
};

// llama.cpp-xdna: the short convolution by tokens of a batch and its SiLU computed while the multiplication of its
// projection reads C (fConvFold): the nodes, and the CONCAT whose first 3 rows are the old states and whose last 3
// go to the cache
struct XdnaConvFold {
  const ggml_tensor * vConv     = nullptr;  // SSM_CONV
  const ggml_tensor * vSilu     = nullptr;  // its SILU, where the result goes
  const ggml_tensor * vConcat   = nullptr;
  int64_t             vTokens   = 0;
  int64_t             vChannels = 0;
};

struct XdnaBackend {
  int         vThreads = cDefaultThreads;  // that share what the host does: llama.cpp says how many
  XdnaTotals vTotals;
  // room that is kept between multiplications
  std::vector<float> aFloats;   // weights, as floats
  std::vector<float> aPartB;    // the inputs of a piece of them
  std::vector<float> aRowsA;    // rows gathered from a tensor that does not hold them one after the other
  std::vector<float> aRowsC;    // and their results
  std::vector<float> aPartA;    // the inputs of a piece, of the rows of a command
  std::vector<float> aPartC;    // the outputs of a piece, of the rows of a command
  std::vector<float> aRounded;  // emulating: the rows as the NPU would read them
  std::vector<XdnaGemmTarget> aTargets;       // where the outputs of a multiplication go
  std::vector<XdnaGemmTarget> aPieceTargets;  // and those of a piece of its matrix, for the rows of a command
  std::vector<ggml_tensor *>   aGroup;         // multiplications of a graph that go together
  // a residual ADD read with the C of a multiplication of one result (fMultiply): its result, the result of the
  // multiplication (where the pieces along the inputs before the last add up) and the other operand, laid out alike
  struct XdnaResidual {
    bool          vActive  = false;
    float *       vDst     = nullptr;
    const float * vProduct = nullptr;
    const float * vAlso    = nullptr;
  } vResidual;
  // a result of the multiplications of a group that goes straight into a CONCAT (fMulMat): which one, and where
  struct XdnaRedirect {
    bool                vActive = false;
    const ggml_tensor * vMember = nullptr;
    float *             vData   = nullptr;
    // llama.cpp-xdna: and with vConvOn its short convolution and SiLU into the SILU instead, and the last 3 rows of
    // C into the CONCAT (fConvFold)
    bool           vConvOn = false;
    XdnaConvFold   vConv;
    bool           vGdnPrepackOn = false;
    XdnaGdnPrepack vGdnPrepack;
  } vRedirect;
  const ggml_tensor * vPrepackedGdn = nullptr;
  // llama.cpp-xdna: for a short convolution, its weights by tap [4][channels] and the old states [3][channels]
  // (fConvPrepare); and the nodes of the convolutions of the part of a graph being computed that went with their
  // projection, which are then not computed
  std::vector<float>                      aConvWeights;
  std::vector<float>                      aConvHead;
  std::unordered_set<const ggml_tensor *> sConvDone;
  // an RMS_NORM and its MUL whose result (vRows) the next multiplications read from the rows the norm reads, scaled
  // as they are converted (fMulMat, fMulMatGlu)
  struct XdnaNormed {
    bool                vActive  = false;
    const ggml_tensor * vRows    = nullptr;  // the result of the MUL, which is not written
    const float *       vX       = nullptr;  // the rows the RMS_NORM reads, one after the other
    const float *       vWeights = nullptr;  // the weights of the MUL, one per column
  } vNormed;
  std::vector<float> aScales;  // the scales of its rows
  // llama.cpp-xdna: the rows of the multiplication right after a SwiGLU (vRows, its result, which is not written),
  // converted ahead by fMulMatGlu into the buffers of A cXdnaRunnerFirstReadyA on, one per piece along its inputs
  struct XdnaPacked {
    bool                    vActive            = false;
    const ggml_tensor *     vRows              = nullptr;
    XdnaRunner::XdnaPending vEarlySent;
    uint32_t                vEarlyWeightHandle = 0;
  } vPacked;
  // llama.cpp-xdna: the recurrences of a batch of the part of a graph being computed that take the L2 norms of their
  // q and k (fL2Fold): the inputs of the norms, their eps and the factor of their SCALE; the nodes of those norms,
  // which are then not computed; and the scales of the norms of a recurrence, [head of keys][token]
  struct XdnaL2Fold {
    const ggml_tensor * vQ      = nullptr;
    const ggml_tensor * vK      = nullptr;
    float               vEps    = 0;
    float               vFactor = 1.0f;
  };
  std::unordered_map<const ggml_tensor *, XdnaL2Fold> dL2Folds;
  std::unordered_set<const ggml_tensor *>              sL2Folded;
  std::vector<float>                                   aQNorms;
  std::vector<float>                                   aKNorms;
  // llama.cpp-xdna: the layers of linear attention of the part of a graph being computed that go to the NPU whole
  // (fGraphCompute), and the layer of every operation of them
  std::vector<XdnaLinearBlock>                aLinear;
  std::unordered_map<const ggml_tensor *, int> dLinearOf;
  // llama.cpp-xdna: the attention of a batch (fFlashAttention): O of a block of rows, the sums of the rows of P, and
  // what the mask holds in every strip of every token
  std::vector<float>   aAttnO;
  std::vector<float>   aAttnSums;
  std::vector<uint8_t> aAttnKinds;
};

ggml_backend_buffer_type_t fBufferType();
ggml_backend_buffer_type_t fStateBufferType();

constexpr uintptr_t cWeightViewKeyTag = uintptr_t(1) << 62;

void fPushWeightKey(std::vector<const void *> & aKey, const ggml_tensor * pWeights) {
  aKey.push_back(pWeights->data);
  if (pWeights->view_src != nullptr) {
    aKey.push_back(reinterpret_cast<const void *>(uintptr_t(pWeights->ne[1]) | cWeightViewKeyTag));
  }
}

bool fGivesLogitsBase(const ggml_tensor * pWeights) {
  if (pWeights == nullptr) {
    return false;
  }
  const char * vName = pWeights->name;
  return std::strcmp(vName, "output.weight") == 0 || std::strcmp(vName, "token_embd.weight") == 0 ||
         (std::strncmp(vName, "blk.", 4) == 0 && std::strstr(vName, ".nextn.shared_head_head.weight") != nullptr);
}

// Whether a matrix of weights turns the hidden states of a model into the logits of its vocabulary, by the names
// llama.cpp gives it: the output matrix, the embeddings when the model uses them for it too, and the head of a
// layer of multi-token prediction (or a zero-offset row prefix view of one, for --mtp-vocab).
bool fGivesLogits(const ggml_tensor * pWeights) {
  if (pWeights != nullptr && pWeights->op == GGML_OP_VIEW && pWeights->view_src != nullptr &&
      pWeights->view_offs == 0) {
    return fGivesLogitsBase(pWeights->view_src);
  }
  return fGivesLogitsBase(pWeights);
}

const ggml_tensor * fWeightTensorBase(const ggml_tensor * pWeights) {
  if (pWeights == nullptr) {
    return nullptr;
  }
  if (pWeights->op == GGML_OP_NONE && pWeights->view_src == nullptr) {
    return pWeights;
  }
  const ggml_tensor * vBase = pWeights->view_src;
  if (pWeights->op == GGML_OP_VIEW && vBase != nullptr && pWeights->view_offs == 0 && vBase->op == GGML_OP_NONE &&
      vBase->view_src == nullptr && fGivesLogitsBase(vBase) && pWeights->type == vBase->type &&
      pWeights->ne[0] == vBase->ne[0] && pWeights->ne[1] <= vBase->ne[1] && pWeights->nb[0] == vBase->nb[0] &&
      pWeights->nb[1] == vBase->nb[1]) {
    return vBase;
  }
  return nullptr;
}

// llama.cpp-xdna: the scalar gate of a shared expert (qwen35moe: blk.<il>.ffn_gate_inp_shexp.weight, 1 output row),
// distinct from the discrete router of the routed experts (blk.<il>.ffn_gate_inp.weight)
bool fIsSharedGate(const ggml_tensor * pWeights) {
  return pWeights != nullptr && std::strncmp(pWeights->name, "blk.", 4) == 0 &&
         std::strstr(pWeights->name, ".ffn_gate_inp_shexp") != nullptr;
}

bool fIsSharedExpertWeight(const ggml_tensor * pWeights) {
  return pWeights != nullptr && std::strncmp(pWeights->name, "blk.", 4) == 0 &&
         std::strstr(pWeights->name, "_shexp.") != nullptr;
}

bool fIsAttnOutWeight(const ggml_tensor * pWeights) {
  return pWeights != nullptr && std::strncmp(pWeights->name, "blk.", 4) == 0 &&
         (std::strstr(pWeights->name, ".ssm_out.") != nullptr ||
          std::strstr(pWeights->name, ".attn_output.") != nullptr);
}

std::string fMoeLayerKey(const char * pName);

// llama.cpp-xdna: whether a matrix of weights is the discrete router of a mixture of experts: it stays with the
// processor, since in block floating point the router could choose other experts, and its few outputs cost little
bool fIsRouter(const ggml_tensor * pWeights) {
  return std::strncmp(pWeights->name, "blk.", 4) == 0 && std::strstr(pWeights->name, ".ffn_gate_inp") != nullptr &&
         !fIsSharedGate(pWeights);
}

// llama.cpp-xdna: the weights placed in this backend while llama.cpp loads models, against the budget for the copies the
// NPU keeps of them (XdnaSettings::vWeightsLimit): every matrix by its name, type and shape, with the decision taken,
// so that a model loaded twice (llama.cpp loads it once without memory to fit it to the devices) is counted once; and
// once a matrix did not fit, every matrix after it stays with the processor, so that the layers on the NPU are whole.
// Everything is forgotten when the last buffer of the backend is freed (fForgetBuffer).
struct XdnaPlacement {
  std::mutex                               vMutex;
  std::unordered_map<std::string, bool>    dTaken;
  std::unordered_map<std::string, bool>    dMoeLayer;       // whole-layer decision by "blk.<il>."
  std::unordered_set<std::string>          sExpertLayers;   // layers "blk.<il>." that have a router
  uint64_t                                 vPrompt  = 0;  // bytes of the copies for the overlay of the prompts
  uint64_t                                 vChain   = 0;  // and for the overlay for generating tokens
  uint64_t                                 vLogits  = 0;  // of those, the ones of the logits
  uint64_t                                 vMoe     = 0;  // bytes of the single copies for the experts (xdna2-moe)
  bool                                     vFull    = false;
  bool                                     vFullMoe = false;
  uint64_t                                 vRefused      = 0;
  int64_t                                  vBuffers      = 0;  // of this backend with memory, weights and computing
  int64_t                                  vDummyBuffers = 0;  // zero-size fit-probe buffers (common_fit_params)
  // a model that did not fit whole: generating tokens goes with the processor but the logits, and the budget counts
  // the copies for the prompts (and for the logits) only (fMulMatTaken)
  std::atomic<bool>                        vPartial{false};
  // a dense model whose chain copies fit in vDenseChainLimit while all bfp16 + chain copies exceed vWeightsLimit: all
  // layers stay on XDNA for chained generation, and prompt matrices past the cached budget stream through reusable BOs
  std::atomic<bool>                        vStreamPrompt{false};
  std::atomic<bool>                        vCompPrompt{false};
};

XdnaPlacement & fPlacement() {
  static XdnaPlacement vPlacement;
  return vPlacement;
}

bool fLayerHasExperts(const char * pName) {
  if (std::strncmp(pName, "blk.", 4) != 0) {
    return false;
  }
  XdnaPlacement &             vPlacement = fPlacement();
  std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
  return vPlacement.sExpertLayers.count(fMoeLayerKey(pName)) != 0;
}

bool fLayerHasNpuMoe(const char * pName) {
  if (std::strncmp(pName, "blk.", 4) != 0) {
    return false;
  }
  XdnaPlacement &             vPlacement = fPlacement();
  std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
  const auto                  vAt = vPlacement.dMoeLayer.find(fMoeLayerKey(pName));
  return vAt != vPlacement.dMoeLayer.end() && vAt->second;
}

bool fMoeActiveForRows(const XdnaShared & pShared, uint64_t pRows) {
  return !pShared.vSettings.vEmulate && pShared.vHasMoe && !pShared.vSettings.vNoMoe && fPlacement().vMoe > 0 &&
         pRows >= pShared.vSettings.vMoeMinTokens;
}

// llama.cpp-xdna: what the NPU keeps of a matrix of weights: its copy in block floating point for the overlay of the
// prompts (but the logits with an overlay that rounds C, fMulMatTaken), in pPrompt, and its copy for the overlay for
// generating tokens (the blocks of the file for Q4_K, Q5_K and Q6_K, block floating point otherwise), in pChain; without
// the padding
void fCopiesBytes(const XdnaShared & pShared, const ggml_tensor * pWeights, uint64_t & pPrompt, uint64_t & pChain,
                  bool pCompPrompt = false) {
  const uint64_t vValues = uint64_t(pWeights->ne[0]) * uint64_t(pWeights->ne[1]);
  const uint64_t vBfp16  = vValues / 8 * 9;
  const bool     vUseQ6K = (pShared.vSettings.vPromptFormat == 1 || pCompPrompt) && pShared.vOverlay.vCompB == 1 &&
                           pWeights->type == GGML_TYPE_Q6_K && pWeights->ne[0] % 512 == 0;
  const uint64_t vPromptRaw = vUseQ6K ? (vBfp16 / 4 * 3) : vBfp16;
  pPrompt = pShared.vOverlay.vOutput != XdnaGemmOutput::cFloat && fGivesLogits(pWeights) ? 0 : vPromptRaw;
  pChain  = 0;
  if (pShared.vHasGemv) {
    const bool vQ8Taken = pShared.vHasChain ? pShared.vChainOverlay.mTakes(XdnaGemvFormat::cQ8_0) :
                                              pShared.vGemvOverlay.mTakes(XdnaGemvFormat::cQ8_0);
    const bool vNative  = ((pWeights->type == GGML_TYPE_Q4_K || pWeights->type == GGML_TYPE_Q5_K ||
                            pWeights->type == GGML_TYPE_Q6_K) &&
                           pWeights->ne[0] % 256 == 0) ||
                          (pWeights->type == GGML_TYPE_Q8_0 && vQ8Taken && pWeights->ne[0] % 32 == 0);
    pChain = vNative ? ggml_nbytes(pWeights) : vBfp16;
  }
}

std::atomic<bool> & fSawExperts();

// llama.cpp-xdna: whether a matrix of weights that llama.cpp is placing goes to this backend within the budget
bool fPlaceWeights(const XdnaShared & pShared, const ggml_tensor * pWeights) {
  if (pShared.vSettings.vWeightsLimit == 0) {
    return true;
  }
  XdnaPlacement &            vPlacement = fPlacement();
  std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
  const std::string vKey = std::string(pWeights->name) + "|" + std::to_string(int(pWeights->type)) + "|" +
                           std::to_string(pWeights->ne[0]) + "|" + std::to_string(pWeights->ne[1]);
  const auto vAt = vPlacement.dTaken.find(vKey);
  if (vAt != vPlacement.dTaken.end()) {
    return vAt->second;
  }
  uint64_t       vPrompt = 0;
  uint64_t       vChain  = 0;
  fCopiesBytes(pShared, pWeights, vPrompt, vChain, vPlacement.vCompPrompt.load());
  const bool     vCanStreamDense = !fSawExperts().load() && pShared.vHasChain && !pShared.vSettings.vNoChain &&
                                   pShared.vSettings.vDenseTotalLimit > 0 && pShared.vSettings.vDenseChainLimit > 0;
  uint64_t       vLimit          = (pShared.vSettings.vPromptFormat == 1 || vPlacement.vCompPrompt.load()) &&
                                           vCanStreamDense && !pShared.vSettings.vWeightsLimitExplicit ?
                                       pShared.vSettings.vDenseTotalLimit :
                                       pShared.vSettings.vWeightsLimit;
  const uint64_t vMoeLimit       = pShared.vSettings.vMoeWeightsLimit;
  const bool     vMoeReplacesMap = !pShared.vSettings.vWeightsLimitExplicit && !pShared.vSettings.vEmulate &&
                                   !pShared.vSettings.vCopyWeights && pShared.vSettings.vMoeMinTokens <= 1;
  const uint64_t vMoeExtra       = vMoeReplacesMap ? 0 : vPlacement.vMoe;
  const bool     vLogits         = fGivesLogits(pWeights);
  if (!vPlacement.vFull && !vPlacement.vPartial.load() && !vPlacement.vCompPrompt.load() &&
      pShared.vSettings.vWeightsLimitExplicit && pShared.vOverlay.vCompB == 1 &&
      pShared.vSettings.vPromptFormat < 0 && pWeights->type == GGML_TYPE_Q6_K &&
      vPlacement.vPrompt + vMoeExtra + vPlacement.vChain + vPrompt + vChain > vLimit &&
      (vPlacement.vPrompt / 4 * 3) + vMoeExtra + vPlacement.vChain + (vPrompt / 4 * 3) + vChain <= vLimit) {
    vPlacement.vCompPrompt.store(true);
    vPlacement.vPrompt = vPlacement.vPrompt / 4 * 3;
    fCopiesBytes(pShared, pWeights, vPrompt, vChain, true);
    GGML_LOG_INFO("%s: dense Q6_K model exceeds %.1f GB in bfp16 + chain: keeping resident Q6_K 4-in-3 prompt weights "
                  "on the NPU within the %.1f GB budget\n",
                  cName, double(pShared.vSettings.vWeightsLimit) * 1e-9, double(vLimit) * 1e-9);
  }
  if (!vPlacement.vFull && !vPlacement.vPartial.load() &&
      (vPlacement.vPrompt + vMoeExtra + vPlacement.vChain + vPrompt + vChain > vLimit ||
       (vMoeLimit > 0 && vPlacement.vPrompt + vPlacement.vMoe + vPlacement.vChain + vPrompt + vChain > vMoeLimit))) {
    if (vCanStreamDense && vPlacement.vChain + vChain <= pShared.vSettings.vDenseChainLimit) {
      if (!vPlacement.vStreamPrompt.exchange(true)) {
        GGML_LOG_INFO("%s: dense model exceeds %.1f GB in two static copies: keeping 100%% of layers in the chained "
                      "overlay and streaming prompt matrices past %.1f GB of total NPU buffers\n",
                      cName, double(vLimit) * 1e-9, double(pShared.vSettings.vDenseTotalLimit) * 1e-9);
      }
    } else if (!vPlacement.vFullMoe || vPlacement.vPrompt + vPlacement.vChain + vPrompt + vChain > vLimit) {
      // the model does not fit whole: from now on generating tokens goes with the processor but the logits, and only
      // the copies for the prompts and for the logits count
      const uint64_t vHitLimit = vPlacement.vPrompt + vMoeExtra + vPlacement.vChain + vPrompt + vChain > vLimit ?
                                     vLimit :
                                     vMoeLimit;
      vPlacement.vStreamPrompt.store(false);
      vPlacement.vPartial.store(true);
      GGML_LOG_INFO("%s: the copies of the weights of the model for the NPU would take more than %.1f GB: generating "
                    "tokens goes with the processor but the logits (GGML_XDNA_WEIGHTS_GB sets that memory)\n",
                    cName, double(vHitLimit) * 1e-9);
    }
  }
  if (vPlacement.vStreamPrompt.load() && !vPlacement.vPartial.load()) {
    const bool vTaken = !vPlacement.vFull && vPlacement.vChain + vChain <= pShared.vSettings.vDenseChainLimit;
    if (vTaken) {
      if (vPlacement.vPrompt + pShared.vSettings.vDenseChainLimit + vPrompt <= pShared.vSettings.vDenseTotalLimit) {
        vPlacement.vPrompt += vPrompt;
      }
      vPlacement.vChain += vChain;
      vPlacement.vLogits += vLogits ? vChain : 0;
    } else {
      vPlacement.vFull = true;
      vPlacement.vRefused++;
    }
    vPlacement.dTaken[vKey] = vTaken;
    return vTaken;
  }
  const uint64_t vChainNow   = vPlacement.vPartial.load() ? (vLogits ? vChain : 0) : vChain;
  const uint64_t vChainSoFar = vPlacement.vPartial.load() ? vPlacement.vLogits : vPlacement.vChain;
  const uint64_t vSoFar      = vPlacement.vPrompt + vMoeExtra + vChainSoFar;
  const uint64_t vTotalSoFar = vPlacement.vPrompt + vPlacement.vMoe + vChainSoFar;
  const bool     vTaken      = !vPlacement.vFull && vSoFar + vPrompt + vChainNow <= vLimit &&
                               (vMoeLimit == 0 || vTotalSoFar + vPrompt + vChainNow <= vMoeLimit);
  if (vTaken) {
    vPlacement.vPrompt += vPrompt;
    vPlacement.vChain += vChainNow;
    vPlacement.vLogits += vLogits ? vChainNow : 0;
  } else {
    if (!vPlacement.vFull) {
      const uint64_t vHitLimit = vSoFar + vPrompt + vChainNow > vLimit ? vLimit : vMoeLimit;
      GGML_LOG_WARN("%s: the copies of the weights for the NPU would take more than %.1f GB: %s and the weights after it "
                    "stay with the processor (GGML_XDNA_WEIGHTS_GB sets that memory)\n",
                    cName, double(vHitLimit) * 1e-9, pWeights->name);
    }
    vPlacement.vFull = true;
    vPlacement.vRefused++;
  }
  vPlacement.dTaken[vKey] = vTaken;
  return vTaken;
}

// llama.cpp-xdna: whether the backend has been asked about the router of a mixture of experts (when llama.cpp places
// the weights of a model): then a model of this process has a block of experts that stays with the processor, which
// the overlay for generating tokens cannot chain with the rest of its layer (fMulMatTaken)
std::atomic<bool> & fSawExperts() {
  static std::atomic<bool> vSaw{false};
  return vSaw;
}

// llama.cpp-xdna: whether a result comes from the experts of a mixture of experts (MUL_MAT_ID, maybe scaled or seen
// through a reshape), which stay with the processor
bool fFromExperts(const ggml_tensor * pTensor) {
  for (int i = 0; i < 4 && pTensor != nullptr; i++) {
    if (pTensor->op == GGML_OP_MUL_MAT_ID) {
      return true;
    }
    if (pTensor->op != GGML_OP_MUL && pTensor->op != GGML_OP_RESHAPE && pTensor->op != GGML_OP_VIEW) {
      return false;
    }
    pTensor = pTensor->src[0];
  }
  return false;
}

bool fMoeTaken(const XdnaShared & pShared, const ggml_tensor * pNode);

// Whether pNode is a SwiGLU the backend takes: silu(gate) * up of two results of the same shape (ggml_swiglu_split),
// 32-bit floats one row after another. The scheduler gives it the one that follows the multiplications of a
// feed-forward layer, and the backend computes it while it reads their C (fMulMatGlu), in the cores of xdna2-moe for
// routed experts, or else on the host from the two results (fGluOnHost), as the CPU backend does.
bool fGluTaken(const ggml_tensor * pNode) {
  if (pNode->op != GGML_OP_GLU || ggml_get_glu_op(pNode) != GGML_GLU_OP_SWIGLU ||
      ggml_get_op_params_i32(pNode, 1) != 0) {
    return false;
  }
  const ggml_tensor * vGate = pNode->src[0];
  const ggml_tensor * vUp   = pNode->src[1];
  if (vGate == nullptr || vUp == nullptr || vGate->type != GGML_TYPE_F32 || vUp->type != GGML_TYPE_F32 ||
      pNode->type != GGML_TYPE_F32 || !ggml_are_same_shape(vGate, vUp) || !ggml_are_same_shape(vGate, pNode) ||
      !ggml_is_contiguous(vGate) || !ggml_is_contiguous(vUp) || !ggml_is_contiguous(pNode)) {
    return false;
  }
  if (vGate->op == GGML_OP_MUL_MAT_ID && vUp->op == GGML_OP_MUL_MAT_ID) {
    return !fShared().vSettings.vNoFuse && fMoeTaken(fShared(), vGate) && fMoeTaken(fShared(), vUp);
  }
  return !fFromExperts(vGate) && !fFromExperts(vUp);
}

// Whether a node has nothing to compute: it only looks at the memory of another one in another way.
bool fDoesNothing(const ggml_tensor * pNode) {
  switch (pNode->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
      return true;
    default:
      return false;
  }
}

// llama.cpp-xdna: the tensor whose memory pTensor shows, through reshapes and contiguous zero-offset views
const ggml_tensor * fThroughReshapes(const ggml_tensor * pTensor) {
  while (pTensor != nullptr &&
         (pTensor->op == GGML_OP_RESHAPE ||
          (pTensor->op == GGML_OP_VIEW && pTensor->view_offs == 0 && ggml_is_contiguous(pTensor))) &&
         pTensor->view_src != nullptr) {
    pTensor = pTensor->src[0];
  }
  return pTensor;
}

// llama.cpp-xdna: whether pNode is a GET_ROWS the backend takes: of 32-bit floats, one id, and the only row of the
// result of an operation (not of a cache, whose rows are not computed), so that its result is that row as it is. It
// is how llama.cpp picks the rows it outputs in the last layer of a model, which are all of them when a token is
// generated (the last layer of Qwen3.5, doc/XDNA2.md). The backend copies the row (fRowsOnHost), or reads it where it
// is when the GET_ROWS goes with a command (fFeedForward).
bool fRowsTaken(const ggml_tensor * pNode) {
  if (pNode->op != GGML_OP_GET_ROWS || pNode->type != GGML_TYPE_F32) {
    return false;
  }
  const ggml_tensor * vRows = pNode->src[0];
  const ggml_tensor * vIds  = pNode->src[1];
  return vRows != nullptr && vIds != nullptr && !fDoesNothing(vRows) && vRows->type == GGML_TYPE_F32 &&
         vIds->type == GGML_TYPE_I32 && ggml_nelements(vIds) == 1 && vRows->ne[1] == 1 && vRows->ne[2] == 1 &&
         vRows->ne[3] == 1 && ggml_are_same_shape(vRows, pNode) && ggml_is_contiguous(vRows) &&
         ggml_is_contiguous(pNode);
}

// llama.cpp-xdna: the tensor whose values pTensor has, through reshapes and GET_ROWS the backend takes
const ggml_tensor * fThroughIdentities(const ggml_tensor * pTensor) {
  while (pTensor != nullptr &&
         ((pTensor->op == GGML_OP_RESHAPE && pTensor->view_src != nullptr) || fRowsTaken(pTensor))) {
    pTensor = pTensor->src[0];
  }
  return pTensor;
}

// llama.cpp-xdna: whether pTensor is a weight of the model (or a view of one): a tensor of a buffer of weights
bool fIsWeight(const ggml_tensor * pTensor) {
  const ggml_tensor * vBase = pTensor->view_src != nullptr ? pTensor->view_src : pTensor;
  return vBase->op == GGML_OP_NONE && vBase->buffer != nullptr &&
         ggml_backend_buffer_get_usage(vBase->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
}

bool fMulMatTaken(const XdnaShared & pShared, const ggml_tensor * pNode);
bool fSharedGateOpTaken(const XdnaShared & pShared, const ggml_tensor * pNode);
bool fMoeAmaxTaken(const XdnaShared & pShared, const ggml_tensor * pNode);
XdnaGemvFormat fMoeFormatOf(const XdnaShared & pShared, const ggml_tensor * pWeights);
XdnaMoeWeights * fEnsureMoeWeights(XdnaShared & pShared, XdnaBackend & pBackend,
                                   const std::vector<const ggml_tensor *> & aTensors, bool pSwiglu,
                                   std::string & pError);
XdnaMoeWeights * fEnsureMoeDenseGroup(XdnaShared & pShared, XdnaBackend & pBackend,
                                      const std::vector<const ggml_tensor *> & aWeights, std::string & pError);

bool fMoeDenseFits(const XdnaShared & pShared, const ggml_tensor * pWeights, uint32_t pRows, bool pSwigluPair = false) {
  if (!pShared.vHasMoe || pRows == 0) {
    return false;
  }
  const XdnaGemvFormat vFormat = fMoeFormatOf(pShared, pWeights);
  const uint32_t       vOut    = uint32_t(pWeights->ne[1]);
  const XdnaGemvPart   aParts[2] = {{vFormat, vOut}, {vFormat, vOut}};
  XdnaMoeMatrix        vMatrix;
  std::string          vError;
  return fXdnaMoeMatrixShape(pShared.vMoeOverlay, 1, uint32_t(pWeights->ne[0]), aParts, pSwigluPair ? 2 : 1, pSwigluPair,
                             vMatrix, vError) &&
         pRows <= fXdnaMoeMaxRows(pShared.vMoeOverlay, vMatrix);
}

bool fIsSharedExpertSum(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pNode == nullptr || pNode->op != GGML_OP_ADD || pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode)) {
    return false;
  }
  for (int vSide = 0; vSide < 2; vSide++) {
    const ggml_tensor * vPart = pNode->src[vSide];
    if (vPart == nullptr) {
      continue;
    }
    if (vPart->op == GGML_OP_MUL && fSharedGateOpTaken(pShared, vPart) && !fIsWeight(pNode->src[1 - vSide])) {
      return true;
    }
    if (vPart->op == GGML_OP_MUL_MAT && vPart->src[0] != nullptr && fIsSharedExpertWeight(vPart->src[0]) &&
        fMulMatTaken(pShared, vPart) && !fIsWeight(pNode->src[1 - vSide])) {
      return true;
    }
  }
  return false;
}

// llama.cpp-xdna: whether pNode is the SIGMOID or scaling MUL of the scalar gate of a shared expert (qwen35moe)
bool fSharedGateOpTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vNoFuse || pShared.vSettings.vNoMoeDense || pShared.vSettings.vEmulate || pNode == nullptr ||
      pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode)) {
    return false;
  }
  if (pNode->op == GGML_OP_UNARY && ggml_get_unary_op(pNode) == GGML_UNARY_OP_SIGMOID) {
    const ggml_tensor * vGateMm = pNode->src[0];
    return vGateMm != nullptr && vGateMm->op == GGML_OP_MUL_MAT && fIsSharedGate(vGateMm->src[0]) &&
           ggml_are_same_shape(vGateMm, pNode) && fMulMatTaken(pShared, vGateMm);
  }
  if (pNode->op == GGML_OP_MUL) {
    for (int vSide = 0; vSide < 2; vSide++) {
      const ggml_tensor * vDown    = pNode->src[vSide];
      const ggml_tensor * vGateSig = pNode->src[1 - vSide];
      if (vDown != nullptr && vGateSig != nullptr && vDown->op == GGML_OP_MUL_MAT && vDown->src[0] != nullptr &&
          fIsSharedExpertWeight(vDown->src[0]) && ggml_are_same_shape(vDown, pNode) && fMulMatTaken(pShared, vDown) &&
          vGateSig->op == GGML_OP_UNARY && vGateSig->ne[0] == 1 && vGateSig->ne[1] == pNode->ne[1] &&
          vGateSig->ne[2] == 1 && vGateSig->ne[3] == 1 && fSharedGateOpTaken(pShared, vGateSig)) {
        return true;
      }
    }
  }
  return false;
}

// llama.cpp-xdna: whether pNode is a residual ADD the backend takes: two operands of 32-bit floats of its shape, one
// row after the other, one of them (through reshapes and GET_ROWS the backend takes) the result of a multiplication
// the backend takes, and the other one not a weight of the model. A bias goes with the operations of the processor
// around it: with Qwen3.5, the ADD of alpha and the bias ssm_dt of every layer of linear attention (32 values, the
// bias in a buffer of this backend because llama.cpp places it as a matrix) made two more parts of the scheduler
// per layer when the backend took it (seen on 2026/11/15, doc/XDNA2.md).
bool fAddTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pNode->op != GGML_OP_ADD || pNode->type != GGML_TYPE_F32) {
    return false;
  }
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (vA == nullptr || vB == nullptr || vA->type != GGML_TYPE_F32 || vB->type != GGML_TYPE_F32 ||
      !ggml_are_same_shape(vA, pNode) || !ggml_are_same_shape(vB, pNode) || !ggml_is_contiguous(vA) ||
      !ggml_is_contiguous(vB) || !ggml_is_contiguous(pNode)) {
    return false;
  }
  for (int vSide = 0; vSide < 2; vSide++) {
    const ggml_tensor * vProduct = fThroughIdentities(pNode->src[vSide]);
    if (vProduct != nullptr && vProduct->op == GGML_OP_MUL_MAT && !fIsSharedGate(vProduct->src[0]) &&
        fMulMatTaken(pShared, vProduct) && !fIsWeight(pNode->src[1 - vSide])) {
      return true;
    }
    if (!pShared.vSettings.vNoMoeDense && !fIsWeight(pNode->src[1 - vSide]) &&
        (fSharedGateOpTaken(pShared, pNode->src[vSide]) || fIsSharedExpertSum(pShared, pNode->src[vSide]))) {
      return true;
    }
  }
  return false;
}

// llama.cpp-xdna: whether pNode is a CONCAT the backend takes: along dimension 1 (rows) of one sequence, 32-bit
// floats, the result one row after the other, the second operand (or a reshape of it) the result of a multiplication
// the backend takes, one row after the other, and the first one of the same rows, laid out in any way (the states of
// the short convolution of Qwen3.5, transposed, see build_conv_state)
bool fConcatTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pNode->op != GGML_OP_CONCAT || ggml_get_op_params_i32(pNode, 0) != 1 || pNode->type != GGML_TYPE_F32 ||
      pNode->ne[2] != 1 || pNode->ne[3] != 1 || !ggml_is_contiguous(pNode)) {
    return false;
  }
  const ggml_tensor * vHead = pNode->src[0];
  const ggml_tensor * vRest = pNode->src[1];
  if (vHead == nullptr || vRest == nullptr || vHead->type != GGML_TYPE_F32 || vRest->type != GGML_TYPE_F32 ||
      vHead->ne[0] != pNode->ne[0] || vRest->ne[0] != pNode->ne[0] || vHead->ne[1] + vRest->ne[1] != pNode->ne[1] ||
      vHead->ne[2] != 1 || vHead->ne[3] != 1 || !ggml_is_contiguous(vRest)) {
    return false;
  }
  const ggml_tensor * vProduct = fThroughReshapes(vRest);
  return vProduct != nullptr && vProduct->op == GGML_OP_MUL_MAT && fMulMatTaken(pShared, vProduct);
}

// llama.cpp-xdna: the rows of the first operand of such a CONCAT, into its first rows (the rest is the result of the
// multiplication), value by value from where they are
void fConcatHead(const ggml_tensor * pNode) {
  const ggml_tensor * vHead  = pNode->src[0];
  const int64_t       vWidth = pNode->ne[0];
  float *             vTo    = static_cast<float *>(pNode->data);
  for (int64_t vRow = 0; vRow < vHead->ne[1]; vRow++) {
    for (int64_t vColumn = 0; vColumn < vWidth; vColumn++) {
      vTo[vRow * vWidth + vColumn] = *reinterpret_cast<const float *>(static_cast<const char *>(vHead->data) +
                                                                      vColumn * vHead->nb[0] + vRow * vHead->nb[1]);
    }
  }
}

// llama.cpp-xdna: whether pNode is an RMS_NORM the backend takes: of the residual stream (its rows come from an ADD
// or from the embeddings), 32-bit floats one row after the other; it feeds the multiplications of a layer
bool fRmsNormTaken(const ggml_tensor * pNode) {
  const ggml_tensor * vX = pNode->src[0];
  float               vEps;
  memcpy(&vEps, pNode->op_params, sizeof(float));
  if (pNode->op != GGML_OP_RMS_NORM || pNode->type != GGML_TYPE_F32 || vX == nullptr ||
      vX->type != GGML_TYPE_F32 || (vX->op != GGML_OP_ADD && vX->op != GGML_OP_GET_ROWS) ||
      !ggml_are_same_shape(vX, pNode) || !ggml_is_contiguous(vX) || !ggml_is_contiguous(pNode) || vEps < 0.0f) {
    return false;
  }
  const XdnaShared & vShared = fShared();
  const uint64_t     vRows   = uint64_t(ggml_nrows(pNode));
  if (vRows < vShared.vSettings.vMinRows) {
    if (fPlacement().vPartial.load()) {
      return false;
    }
    if (fSawExperts().load() && !vShared.vSettings.vMoeGemv) {
      if (vShared.vSettings.vNoMoeDense || vShared.vSettings.vEmulate ||
          (!fMoeActiveForRows(vShared, vRows) && !vShared.vHasChain)) {
        return false;
      }
      if (vX->op == GGML_OP_ADD) {
        for (int vSide = 0; vSide < 2; vSide++) {
          const ggml_tensor * vProd = fThroughIdentities(vX->src[vSide]);
          if (vProd != nullptr && vProd->op == GGML_OP_MUL_MAT && vProd->src[0] != nullptr &&
              fIsAttnOutWeight(vProd->src[0]) && !fMulMatTaken(vShared, vProd)) {
            return false;
          }
        }
      }
    }
  }
  return true;
}

// llama.cpp-xdna: the weights of pNode if it is the MUL of such an RMS_NORM by one weight per column (32-bit
// floats), or null
const ggml_tensor * fNormWeightsOf(const ggml_tensor * pNode) {
  if (pNode->op != GGML_OP_MUL || pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode)) {
    return nullptr;
  }
  for (int vSide = 0; vSide < 2; vSide++) {
    const ggml_tensor * vNorm    = pNode->src[vSide];
    const ggml_tensor * vWeights = pNode->src[1 - vSide];
    if (vNorm != nullptr && vWeights != nullptr && vNorm->op == GGML_OP_RMS_NORM && fRmsNormTaken(vNorm) &&
        ggml_are_same_shape(vNorm, pNode) && vWeights->type == GGML_TYPE_F32 && ggml_is_contiguous(vWeights) &&
        vWeights->ne[0] == pNode->ne[0] && ggml_nelements(vWeights) == vWeights->ne[0]) {
      return vWeights;
    }
  }
  return nullptr;
}

// llama.cpp-xdna: the uses of pTensor in the graph pGraph is a part of (the counts of the whole graph)
int fUseCount(const struct ggml_cgraph * pGraph, const ggml_tensor * pTensor) {
  const size_t vPosition = ggml_hash_find(&pGraph->visited_hash_set, pTensor);
  if (vPosition == GGML_HASHSET_FULL || !ggml_bitset_get(pGraph->visited_hash_set.used, vPosition)) {
    return 0;
  }
  return pGraph->use_counts[vPosition];
}

// Whether pNode is a multiplication the backend takes: rows of 32-bit floats by a whole matrix of weights that
// lives in one of its buffers. Everything the conversion and the multiplication rely on is checked here.
bool fMulMatTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  const ggml_tensor * vWeights = pNode->src[0];
  const ggml_tensor * vRows    = pNode->src[1];
  if (vWeights == nullptr || vRows == nullptr) {
    return false;
  }
  // the fast transform of a rotation is the processor's
  if (ggml_get_op_params_i32(pNode, 1) != GGML_HINT_NONE) {
    return false;
  }
  // llama.cpp-xdna: the discrete router of a mixture of experts stays with the processor unless GGML_XDNA_FULL is set
  if (fIsRouter(vWeights)) {
    fSawExperts().store(true);
    {
      XdnaPlacement &             vPlacement = fPlacement();
      std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
      vPlacement.sExpertLayers.insert(fMoeLayerKey(vWeights->name));
    }
    if (pShared.vSettings.vFull <= 0) {
      return false;
    }
  }
  if (fIsSharedGate(vWeights) &&
      (pShared.vSettings.vNoMoeDense || pShared.vSettings.vNoFuse || pShared.vSettings.vEmulate)) {
    return false;
  }
  // the rows are rounded to 8 bits with a shared exponent: not when the model asks for more than that
  const enum ggml_prec vPrecision = ggml_prec_get_src(pNode, 1);
  if (vPrecision != GGML_PREC_UNDEFINED && vPrecision != GGML_PREC_Q8 && vPrecision != GGML_PREC_Q4) {
    return false;
  }

  // the weights: a matrix (or a zero-offset row prefix view of the logits head), in a buffer of this backend
  const ggml_tensor *         vWeightBase = fWeightTensorBase(vWeights);
  const ggml_backend_buffer_t vWeightBuf  = vWeights->buffer != nullptr ?
                                                vWeights->buffer :
                                                (vWeightBase != nullptr ? vWeightBase->buffer : nullptr);
  if (vWeightBase == nullptr || vWeightBuf == nullptr ||
      ggml_backend_buffer_get_type(vWeightBuf) != fBufferType()) {
    return false;
  }
  const int64_t vInputs  = vWeights->ne[0];
  const int64_t vOutputs = vWeights->ne[1];
  if (vWeights->ne[2] != 1 || vWeights->ne[3] != 1 || vInputs < 1 || vInputs > cLargestSide || vOutputs < 1 ||
      vOutputs > cLargestSide) {
    return false;
  }
  // of floats, or of something that ggml converts to floats row by row
  if (vWeights->type < 0 || vWeights->type >= GGML_TYPE_COUNT) {
    return false;
  }
  const struct ggml_type_traits * vTraits = ggml_get_type_traits(vWeights->type);
  if (vWeights->type != GGML_TYPE_F32 && vTraits->to_float == nullptr) {
    return false;
  }
  if (vTraits->blck_size < 1 || vInputs % vTraits->blck_size != 0 || vWeights->nb[0] != vTraits->type_size ||
      vWeights->nb[1] != ggml_row_size(vWeights->type, vInputs)) {
    return false;
  }

  // the rows: 32-bit floats, one after the other in every row, as many as the weights have inputs
  if (vRows->type != GGML_TYPE_F32 || vRows->nb[0] != sizeof(float) || vRows->ne[0] != vInputs) {
    return false;
  }
  int64_t vCount = 1;
  for (int i = 1; i < GGML_MAX_DIMS; i++) {
    if (vRows->ne[i] < 1 || vRows->ne[i] > cLargestSide / vCount) {
      return false;
    }
    vCount *= vRows->ne[i];
  }
  // llama.cpp asks with an empty buffer of a type when it decides where a weight goes: that is not a batch
  const bool vPlacing = ggml_backend_buffer_get_size(vWeightBuf) == 0;
  const bool vByRows  = fByRows(pShared, uint64_t(vCount), uint32_t(vInputs));
  // An overlay that rounds C to 16 bits or fewer does not get the logits: rounded, they cost a model as much precision
  // as all the other multiplications together (doc/XDNA2.md, "C in bfloat16"). Only a model that wants the logits of
  // a whole batch, as llama-perplexity does, multiplies them by many rows. The overlay for generating tokens gives its
  // sums in 32-bit floats: with it the logits of few rows go to the NPU, so the weights that give them live in the
  // buffers of this backend.
  if (pShared.vOverlay.vOutput != XdnaGemmOutput::cFloat && fGivesLogits(vWeights) && !vByRows &&
      !(vPlacing && pShared.vHasGemv)) {
    return false;
  }
  const bool vShexpOnMoe = !vPlacing && !pShared.vSettings.vNoMoeDense && !pShared.vSettings.vNoFuse &&
                           !pShared.vSettings.vEmulate && !pShared.vSettings.vMoeGemv &&
                           !fPlacement().vPartial.load() &&
                           (fIsSharedExpertWeight(vWeights) || fIsSharedGate(vWeights)) &&
                           fMoeActiveForRows(pShared, uint64_t(vCount)) && fLayerHasNpuMoe(vWeights->name);
  if (uint64_t(vCount) < pShared.vSettings.vMinRows && pShared.vSettings.vFull <= 0 && !vPlacing && !vByRows &&
      !vShexpOnMoe) {
    return false;
  }
  // llama.cpp-xdna: with a mixture of experts, dense and shared-expert projections of few rows run on the same
  // overlay as the rest of the token (xdna2-moe when routed experts run there, or xdna2-gemv-chain when they run on
  // the processor) so the token never switches overlays; GGML_XDNA_NO_MOE_DENSE leaves them with the processor
  if (!vPlacing && fIsSharedGate(vWeights)) {
    if (fPlacement().vPartial.load()) {
      return false;
    }
    if (vByRows) {
      if (fMoeActiveForRows(pShared, uint64_t(vCount))) {
        if (!fLayerHasNpuMoe(vWeights->name)) {
          return false;
        }
      } else if (!pShared.vHasChain || pShared.vSettings.vNoChain || vCount != 1) {
        return false;
      }
    }
  } else if (vByRows && !vPlacing && fSawExperts().load() && !pShared.vSettings.vMoeGemv && !fGivesLogits(vWeights)) {
    if (pShared.vSettings.vNoMoeDense || pShared.vSettings.vEmulate || fPlacement().vPartial.load()) {
      return false;
    }
    const bool vMoeActive = fMoeActiveForRows(pShared, uint64_t(vCount));
    if (!fLayerHasExperts(vWeights->name)) {
      if (vMoeActive) {
        const bool vFfnPair = std::strstr(vWeights->name, ".ffn_gate.") != nullptr ||
                              std::strstr(vWeights->name, ".ffn_up.") != nullptr;
        if (!fMoeDenseFits(pShared, vWeights, uint32_t(vCount), vFfnPair)) {
          return false;
        }
      } else if (!pShared.vHasChain) {
        return false;
      }
    } else if (fIsAttnOutWeight(vWeights)) {
      if (!pShared.vSettings.vMoeAttnOut ||
          (vMoeActive && !fMoeDenseFits(pShared, vWeights, uint32_t(vCount), false)) ||
          (!vMoeActive && !pShared.vHasChain)) {
        return false;
      }
    } else if (fIsSharedExpertWeight(vWeights)) {
      if (vMoeActive) {
        if (pShared.vSettings.vNoFuse || !fLayerHasNpuMoe(vWeights->name)) {
          return false;
        }
      } else if (!pShared.vHasChain || pShared.vSettings.vNoChain || pShared.vSettings.vNoFuse || vCount != 1) {
        return false;
      }
    } else {
      if (vMoeActive) {
        if ((vInputs < 1024 && !fIsRouter(vWeights)) || std::strstr(vWeights->name, ".indexer.") != nullptr ||
            !fMoeDenseFits(pShared, vWeights, uint32_t(vCount), false)) {
          return false;
        }
      } else if (!pShared.vHasChain) {
        return false;
      }
    }
  }
  // llama.cpp-xdna: and so it does when the model did not fit whole in the budget of its copies (fPlaceWeights)
  if (vByRows && !vPlacing && fPlacement().vPartial.load() && !fGivesLogits(vWeights)) {
    return false;
  }

  // the result: a row of as many 32-bit floats as the weights have outputs, for every row
  if (pNode->type != GGML_TYPE_F32 || pNode->nb[0] != sizeof(float) || pNode->ne[0] != vOutputs) {
    return false;
  }
  for (int i = 1; i < GGML_MAX_DIMS; i++) {
    if (pNode->ne[i] != vRows->ne[i]) {
      return false;
    }
  }
  // llama.cpp-xdna: the copies of the weights for the NPU within their budget (fPlaceWeights), when llama.cpp places
  // the weights of a model: a matrix past it stays with the processor, so that a large model goes part on the NPU and
  // part on the processor instead of taking more memory than the machine has
  return !vPlacing || fPlaceWeights(pShared, vWeights);
}

// where the row pRow of a tensor starts: its dimensions after the first one, counted together
char * fRowOf(const ggml_tensor * pTensor, uint64_t pRow) {
  const uint64_t vSide1 = uint64_t(pTensor->ne[1]);
  const uint64_t vSide2 = uint64_t(pTensor->ne[2]);
  return static_cast<char *>(pTensor->data) + (pRow % vSide1) * pTensor->nb[1] +
         (pRow / vSide1 % vSide2) * pTensor->nb[2] + (pRow / (vSide1 * vSide2)) * pTensor->nb[3];
}

void fFreeMatrix(XdnaShared & pShared, XdnaMatrix & pMatrix);
void fReleaseMappedExpertPages(XdnaShared & pShared, const ggml_tensor * pWeights);

void fNoteDenseCached(XdnaShared & pShared, const std::vector<const ggml_tensor *> & aWeights, bool pPrompt) {
  if (pShared.vSettings.vEmulate || pShared.vSettings.vCopyWeights || fPlacement().vPartial.load()) {
    return;
  }
  for (const ggml_tensor * vWeights : aWeights) {
    if (vWeights == nullptr || vWeights->data == nullptr || std::strncmp(vWeights->name, "blk.", 4) != 0 ||
        fIsRouter(vWeights) || fIsSharedGate(vWeights) || fGivesLogits(vWeights)) {
      continue;
    }
    if (pPrompt) {
      pShared.sPromptCached.insert(vWeights->data);
    } else {
      pShared.sGemvCached.insert(vWeights->data);
    }
    if (pShared.sPromptCached.count(vWeights->data) != 0 && pShared.sGemvCached.count(vWeights->data) != 0) {
      fReleaseMappedExpertPages(pShared, vWeights);
    }
  }
}

// Converts matrices of weights that multiply the same rows into one matrix, one after the other along the outputs:
// their floats, piece by piece, to what the NPU reads. With one tensor, it is its matrix as it is.
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
const bool cHasDequantAvx2F16c = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("f16c");

__attribute__((target("avx2,f16c"))) void fDequantizeQ8_0Avx2(const char * pSrc, float * pDst, uint32_t pValues) {
  const uint32_t vBlocks = pValues / 32;
  for (uint32_t b = 0; b < vBlocks; b++) {
    const uint8_t * vBlock = reinterpret_cast<const uint8_t *>(pSrc) + size_t(b) * 34;
    uint16_t        vHalf  = 0;
    std::memcpy(&vHalf, vBlock, 2);
    const __m256 vD  = _mm256_broadcastss_ps(_mm_cvtph_ps(_mm_cvtsi32_si128(vHalf)));
    const char * vQs = reinterpret_cast<const char *>(vBlock + 2);
    for (uint32_t g = 0; g < 4; g++) {
      const __m128i vBytes = _mm_loadl_epi64(reinterpret_cast<const __m128i *>(vQs + g * 8));
      const __m256  vQ     = _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vBytes));
      _mm256_storeu_ps(pDst + b * 32 + g * 8, _mm256_mul_ps(vQ, vD));
    }
  }
}

__attribute__((target("avx2,f16c"))) void fDequantizeQ6KAvx2(const char * pSrc, float * pDst, uint32_t pValues) {
  const uint32_t vBlocks = pValues / 256;
  const __m256i  vMask4  = _mm256_set1_epi8(0x0f);
  const __m256i  vMask2  = _mm256_set1_epi8(0x03);
  const __m256i  vBias32 = _mm256_set1_epi8(32);
  for (uint32_t b = 0; b < vBlocks; b++) {
    const uint8_t * vBlock = reinterpret_cast<const uint8_t *>(pSrc) + size_t(b) * 210;
    uint16_t        vHalf  = 0;
    std::memcpy(&vHalf, vBlock + 208, 2);
    const float    vD  = _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128(vHalf)));
    const int8_t * vSc = reinterpret_cast<const int8_t *>(vBlock + 192);
    for (uint32_t h = 0; h < 2; h++) {
      const __m256i vQl0  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vBlock + 64 * h));
      const __m256i vQl1  = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vBlock + 64 * h + 32));
      const __m256i vQh   = _mm256_loadu_si256(reinterpret_cast<const __m256i *>(vBlock + 128 + 32 * h));
      const __m256i aQ[4] = {
        _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(vQl0, vMask4),
                                        _mm256_slli_epi16(_mm256_and_si256(vQh, vMask2), 4)),
                        vBias32),
        _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(vQl1, vMask4),
                                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vQh, 2), vMask2), 4)),
                        vBias32),
        _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vQl0, 4), vMask4),
                                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vQh, 4), vMask2), 4)),
                        vBias32),
        _mm256_sub_epi8(_mm256_or_si256(_mm256_and_si256(_mm256_srli_epi16(vQl1, 4), vMask4),
                                        _mm256_slli_epi16(_mm256_and_si256(_mm256_srli_epi16(vQh, 6), vMask2), 4)),
                        vBias32),
      };
      float *        vOut  = pDst + b * 256 + h * 128;
      const int8_t * vScH  = vSc + h * 8;
      for (uint32_t k = 0; k < 4; k++) {
        const __m128i vLo16 = _mm256_castsi256_si128(aQ[k]);
        const __m128i vHi16 = _mm256_extracti128_si256(aQ[k], 1);
        const __m256  vS0   = _mm256_set1_ps(vD * float(vScH[2 * k]));
        const __m256  vS1   = _mm256_set1_ps(vD * float(vScH[2 * k + 1]));
        _mm256_storeu_ps(vOut + 32 * k, _mm256_mul_ps(vS0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vLo16))));
        _mm256_storeu_ps(vOut + 32 * k + 8,
                         _mm256_mul_ps(vS0, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vLo16, 8)))));
        _mm256_storeu_ps(vOut + 32 * k + 16, _mm256_mul_ps(vS1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(vHi16))));
        _mm256_storeu_ps(vOut + 32 * k + 24,
                         _mm256_mul_ps(vS1, _mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_srli_si128(vHi16, 8)))));
      }
    }
  }
}
__attribute__((target("avx2,f16c"))) void fDequantizeQ5_1Avx2(const char * pSrc, float * pDst, uint32_t pValues) {
  const uint32_t vBlocks  = pValues / 32;
  const __m128i  vMask4   = _mm_set1_epi8(0x0f);
  const __m256i  vBit4    = _mm256_set1_epi32(0x10);
  const __m256i  vShifts0 = _mm256_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7);
  const __m256i  vShifts1 = _mm256_setr_epi32(8, 9, 10, 11, 12, 13, 14, 15);
  const __m256i  vShifts2 = _mm256_setr_epi32(16, 17, 18, 19, 20, 21, 22, 23);
  const __m256i  vShifts3 = _mm256_setr_epi32(24, 25, 26, 27, 28, 29, 30, 31);
  for (uint32_t b = 0; b < vBlocks; b++) {
    const uint8_t * vBlock = reinterpret_cast<const uint8_t *>(pSrc) + size_t(b) * 24;
    uint32_t        vDm    = 0;
    uint32_t        vQhRaw = 0;
    std::memcpy(&vDm, vBlock, 4);
    std::memcpy(&vQhRaw, vBlock + 4, 4);
    const __m128  vPair  = _mm_cvtph_ps(_mm_cvtsi32_si128(int(vDm)));
    const __m256  vD     = _mm256_broadcastss_ps(vPair);
    const __m256  vM     = _mm256_broadcastss_ps(_mm_shuffle_ps(vPair, vPair, 0x55));
    const __m256i vQh    = _mm256_set1_epi32(int(vQhRaw));
    const __m128i vQs    = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vBlock + 8));
    const __m128i vLo16  = _mm_and_si128(vQs, vMask4);
    const __m128i vHi16  = _mm_and_si128(_mm_srli_epi16(vQs, 4), vMask4);
    const __m256i vQ0    = _mm256_or_si256(_mm256_cvtepu8_epi32(vLo16),
                                           _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(vQh, vShifts0), 4), vBit4));
    const __m256i vQ1    = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(vLo16, 8)),
                                           _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(vQh, vShifts1), 4), vBit4));
    const __m256i vQ2    = _mm256_or_si256(_mm256_cvtepu8_epi32(vHi16),
                                           _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(vQh, vShifts2), 4), vBit4));
    const __m256i vQ3    = _mm256_or_si256(_mm256_cvtepu8_epi32(_mm_srli_si128(vHi16, 8)),
                                           _mm256_and_si256(_mm256_slli_epi32(_mm256_srlv_epi32(vQh, vShifts3), 4), vBit4));
    float *       vOut   = pDst + b * 32;
    _mm256_storeu_ps(vOut + 0, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vQ0), vD), vM));
    _mm256_storeu_ps(vOut + 8, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vQ1), vD), vM));
    _mm256_storeu_ps(vOut + 16, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vQ2), vD), vM));
    _mm256_storeu_ps(vOut + 24, _mm256_add_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vQ3), vD), vM));
  }
}

__attribute__((target("avx2"))) void fDequantizeBf16Avx2(const char * pSrc, float * pDst, uint32_t pValues) {
  const uint16_t * vHalves = reinterpret_cast<const uint16_t *>(pSrc);
  uint32_t         i       = 0;
  for (; i + 8 <= pValues; i += 8) {
    const __m128i vRaw  = _mm_loadu_si128(reinterpret_cast<const __m128i *>(vHalves + i));
    const __m256i vBits = _mm256_slli_epi32(_mm256_cvtepu16_epi32(vRaw), 16);
    _mm256_storeu_ps(pDst + i, _mm256_castsi256_ps(vBits));
  }
  for (; i < pValues; i++) {
    const uint32_t vBits = uint32_t(vHalves[i]) << 16;
    std::memcpy(pDst + i, &vBits, sizeof(float));
  }
}
#endif

void fDequantizeWeightRow(ggml_type pType, const char * pFrom, float * pDst, uint32_t pValues) {
  if (pType == GGML_TYPE_F32) {
    std::memcpy(pDst, pFrom, size_t(pValues) * sizeof(float));
    return;
  }
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
  if (cHasDequantAvx2F16c) {
    if (pType == GGML_TYPE_Q8_0) {
      fDequantizeQ8_0Avx2(pFrom, pDst, pValues);
      return;
    }
    if (pType == GGML_TYPE_Q6_K) {
      fDequantizeQ6KAvx2(pFrom, pDst, pValues);
      return;
    }
    if (pType == GGML_TYPE_Q5_1) {
      fDequantizeQ5_1Avx2(pFrom, pDst, pValues);
      return;
    }
    if (pType == GGML_TYPE_BF16) {
      fDequantizeBf16Avx2(pFrom, pDst, pValues);
      return;
    }
  }
#endif
  ggml_get_type_traits(pType)->to_float(pFrom, pDst, pValues);
}

bool fConvert(XdnaShared & pShared, XdnaBackend & pBackend, const std::vector<const ggml_tensor *> & aWeights,
              XdnaMatrix & pMatrix, std::string & pError, bool pReuse = false) {
  const double   vStart   = fNowMs();
  const uint32_t vInputs  = uint32_t(aWeights[0]->ne[0]);
  uint32_t       vOutputs = 0;
  pMatrix.aTypes.clear();
  pMatrix.aOutputs.clear();
  for (const ggml_tensor * vWeights : aWeights) {
    pMatrix.aTypes.push_back(vWeights->type);
    pMatrix.aOutputs.push_back(uint32_t(vWeights->ne[1]));
    vOutputs += uint32_t(vWeights->ne[1]);
  }
  const auto     aRangesK = fRanges(vInputs, pShared.vMaxInputs);
  const auto     aRangesN = pShared.vSettings.vNoOverlap ?
                                fRangesOfMembers(pMatrix.aOutputs, pShared.vMaxOutputs) :
                                fRangesForOverlap(pMatrix.aOutputs, fRangesOfMembers(pMatrix.aOutputs,
                                                                                     pShared.vMaxOutputs),
                                                  pShared.vMaxOutputs);
  XdnaWorkers & vWorkers = pShared.vRunner.mWorkers();

  pMatrix.vInputs  = vInputs;
  pMatrix.vOutputs = vOutputs;
  pMatrix.vRangesK = uint32_t(aRangesK.size());
  pMatrix.vRangesN = uint32_t(aRangesN.size());
  const size_t vPiecesNeed = aRangesK.size() * aRangesN.size();
  if (pReuse && pMatrix.aPieces.size() > vPiecesNeed) {
    for (size_t k = vPiecesNeed; k < pMatrix.aPieces.size(); k++) {
      pShared.vRunner.mFreeWeights(pMatrix.aPieces[k].vWeights);
    }
  }
  pMatrix.aPieces.resize(vPiecesNeed);

  const bool vAllowComp = !pShared.vSettings.vEmulate && pShared.vOverlay.vCompB == 1 &&
                          pShared.vSettings.vPromptFormat != 0;
  const bool vWantQ8    = vAllowComp && pShared.vSettings.vPromptFormat == 2;
  const bool vWantQ6K   = vAllowComp && (pShared.vSettings.vPromptFormat == 1 ||
                                         (pShared.vSettings.vPromptFormat < 0 && fPlacement().vCompPrompt.load()));

  for (size_t vRangeN = 0; vRangeN < aRangesN.size(); vRangeN++) {
    const uint32_t vFirstOutput = aRangesN[vRangeN].first;
    const uint32_t vCount       = aRangesN[vRangeN].second;
    const uint32_t vTasks       = (vCount + cRowsPerTask - 1) / cRowsPerTask;

    XdnaPromptFormat vPromptFmt = XdnaPromptFormat::cBfp16;
    if (vWantQ6K || vWantQ8) {
      const ggml_type vExpectedType = vWantQ6K ? GGML_TYPE_Q6_K : GGML_TYPE_Q8_0;
      const uint32_t  vBlockAlign   = vWantQ6K ? 256u : 32u;
      const uint32_t  vLenAlign     = vWantQ6K ? 512u : 128u;
      bool            vAllMatch     = true;
      uint32_t        vMemberStart  = 0;
      for (size_t vM = 0; vAllMatch && vM < aWeights.size(); vM++) {
        const uint32_t vMemberEnd = vMemberStart + pMatrix.aOutputs[vM];
        if (vMemberEnd > vFirstOutput && vMemberStart < vFirstOutput + vCount) {
          vAllMatch = aWeights[vM]->type == vExpectedType;
        }
        vMemberStart = vMemberEnd;
      }
      for (size_t vRangeK = 0; vAllMatch && vRangeK < aRangesK.size(); vRangeK++) {
        vAllMatch = aRangesK[vRangeK].first % vBlockAlign == 0 && aRangesK[vRangeK].second % vLenAlign == 0;
      }
      if (vAllMatch) {
        vPromptFmt = vWantQ6K ? XdnaPromptFormat::cQ6K : XdnaPromptFormat::cQ8_0;
      }
    }

    if (vPromptFmt != XdnaPromptFormat::cBfp16) {
      for (size_t vRangeK = 0; vRangeK < aRangesK.size(); vRangeK++) {
        const uint32_t vFirstInput = aRangesK[vRangeK].first;
        const uint32_t vLength     = aRangesK[vRangeK].second;
        XdnaPiece &    vPiece      = pMatrix.aPieces[vRangeK * aRangesN.size() + vRangeN];
        vPiece.vFirstInput         = vFirstInput;
        vPiece.vInputs             = vLength;
        vPiece.vFirstOutput        = vFirstOutput;
        vPiece.vOutputs            = vCount;
        auto fRowPtr = [&](uint32_t pRow) -> const uint8_t * {
          uint32_t vOutput = vFirstOutput + pRow;
          size_t   vMember = 0;
          while (vOutput >= pMatrix.aOutputs[vMember]) {
            vOutput -= pMatrix.aOutputs[vMember];
            vMember++;
          }
          const ggml_tensor * vWeights   = aWeights[vMember];
          const uint32_t      vBlockSize = uint32_t(ggml_blck_size(vWeights->type));
          const size_t        vTypeSize  = ggml_type_size(vWeights->type);
          return static_cast<const uint8_t *>(vWeights->data) + size_t(vOutput) * vWeights->nb[1] +
                 size_t(vFirstInput / vBlockSize) * vTypeSize;
        };
        if (!pShared.vRunner.mLoadWeightsCompressed(vPromptFmt, fRowPtr, vCount, vLength, vPiece.vWeights, pError,
                                                    pReuse)) {
          fFreeMatrix(pShared, pMatrix);
          return false;
        }
      }
      continue;
    }

    bool vCanReadRowsDirect = !pShared.vSettings.vEmulate;
    for (size_t vRangeK = 0; vCanReadRowsDirect && vRangeK < aRangesK.size(); vRangeK++) {
      for (const ggml_tensor * vWeights : aWeights) {
        const int64_t vBlockSize = ggml_blck_size(vWeights->type);
        if (vBlockSize <= 0 || aRangesK[vRangeK].first % uint32_t(vBlockSize) != 0 ||
            aRangesK[vRangeK].second % uint32_t(vBlockSize) != 0) {
          vCanReadRowsDirect = false;
        }
      }
    }
    if (vCanReadRowsDirect) {
      for (size_t vRangeK = 0; vRangeK < aRangesK.size(); vRangeK++) {
        const uint32_t vFirstInput = aRangesK[vRangeK].first;
        const uint32_t vLength     = aRangesK[vRangeK].second;
        XdnaPiece &    vPiece      = pMatrix.aPieces[vRangeK * aRangesN.size() + vRangeN];
        vPiece.vFirstInput         = vFirstInput;
        vPiece.vInputs             = vLength;
        vPiece.vFirstOutput        = vFirstOutput;
        vPiece.vOutputs            = vCount;
        auto fReadRow = [&](uint32_t pRow, float * pDst) {
          uint32_t vOutput = vFirstOutput + pRow;
          size_t   vMember = 0;
          while (vOutput >= pMatrix.aOutputs[vMember]) {
            vOutput -= pMatrix.aOutputs[vMember];
            vMember++;
          }
          const ggml_tensor * vWeights   = aWeights[vMember];
          const uint32_t      vBlockSize = uint32_t(ggml_blck_size(vWeights->type));
          const size_t        vTypeSize  = ggml_type_size(vWeights->type);
          const char *        vFrom      = static_cast<const char *>(vWeights->data) + size_t(vOutput) * vWeights->nb[1] +
                                           size_t(vFirstInput / vBlockSize) * vTypeSize;
          fDequantizeWeightRow(vWeights->type, vFrom, pDst, vLength);
        };
        if (!pShared.vRunner.mLoadWeightsRows(fReadRow, vCount, vLength, cRounding, vPiece.vWeights, pError, pReuse)) {
          fFreeMatrix(pShared, pMatrix);
          return false;
        }
      }
      continue;
    }

    // the outputs of the range, as rows of floats, each from the tensor it belongs to; the floats of one tensor
    // of floats are used where they are
    const float * vFloats = nullptr;
    if (aWeights.size() == 1 && aWeights[0]->type == GGML_TYPE_F32) {
      vFloats = reinterpret_cast<const float *>(static_cast<const char *>(aWeights[0]->data) +
                                                size_t(vFirstOutput) * aWeights[0]->nb[1]);
    } else {
      pBackend.aFloats.resize(size_t(vCount) * vInputs);
      vWorkers.mForEach(vTasks, [&](uint32_t pTask) {
        const uint32_t vLast = std::min(vCount, (pTask + 1) * cRowsPerTask);
        for (uint32_t vRow = pTask * cRowsPerTask; vRow < vLast; vRow++) {
          uint32_t vOutput = vFirstOutput + vRow;
          size_t   vMember = 0;
          while (vOutput >= pMatrix.aOutputs[vMember]) {
            vOutput -= pMatrix.aOutputs[vMember];
            vMember++;
          }
          const ggml_tensor * vWeights = aWeights[vMember];
          const char *        vFrom    = static_cast<const char *>(vWeights->data) + size_t(vOutput) * vWeights->nb[1];
          float *             vTo      = pBackend.aFloats.data() + size_t(vRow) * vInputs;
          if (vWeights->type == GGML_TYPE_F32) {
            std::memcpy(vTo, vFrom, size_t(vInputs) * sizeof(float));
          } else {
            ggml_get_type_traits(vWeights->type)->to_float(vFrom, vTo, vInputs);
          }
        }
      });
      vFloats = pBackend.aFloats.data();
    }

    for (size_t vRangeK = 0; vRangeK < aRangesK.size(); vRangeK++) {
      const uint32_t vFirstInput = aRangesK[vRangeK].first;
      const uint32_t vLength     = aRangesK[vRangeK].second;
      const float *  vPart       = vFloats;
      if (aRangesK.size() > 1) {
        pBackend.aPartB.resize(size_t(vCount) * vLength);
        for (uint32_t vRow = 0; vRow < vCount; vRow++) {
          std::memcpy(pBackend.aPartB.data() + size_t(vRow) * vLength,
                      vFloats + size_t(vRow) * vInputs + vFirstInput, vLength * sizeof(float));
        }
        vPart = pBackend.aPartB.data();
      }

      XdnaPiece & vPiece = pMatrix.aPieces[vRangeK * aRangesN.size() + vRangeN];
      vPiece.vFirstInput  = vFirstInput;
      vPiece.vInputs      = vLength;
      vPiece.vFirstOutput = vFirstOutput;
      vPiece.vOutputs     = vCount;
      bool vOk = true;
      if (pShared.vSettings.vEmulate) {
        const size_t      vRowBytes = size_t(fBlocksOf(vLength)) * cXdnaBfp16Bytes;
        std::atomic<bool> vFinite(true);
        vPiece.aBlocks.resize(size_t(vCount) * vRowBytes);
        vWorkers.mForEach(vTasks, [&](uint32_t pTask) {
          const uint32_t vFirst = pTask * cRowsPerTask;
          const uint32_t vRows  = std::min(vCount, vFirst + cRowsPerTask) - vFirst;
          if (!fEncodeRows(vPart + size_t(vFirst) * vLength, vRows, vLength,
                           vPiece.aBlocks.data() + size_t(vFirst) * vRowBytes)) {
            vFinite = false;
          }
        });
        vOk = vFinite;
        if (!vOk) {
          pError = "the weights have a value that is not finite";
        }
      } else {
        vOk = pShared.vRunner.mLoadWeights(vPart, vCount, vLength, cRounding, vPiece.vWeights, pError, pReuse);
      }
      if (!vOk) {
        fFreeMatrix(pShared, pMatrix);
        return false;
      }
    }
  }

  std::string vNames;
  for (const ggml_tensor * vWeights : aWeights) {
    vNames += (vNames.empty() ? "" : " + ") + std::string(vWeights->name) + " (" + ggml_type_name(vWeights->type) + ")";
  }
  GGML_LOG_DEBUG("%s: weights %s: %u inputs, %u outputs, %zu pieces\n", cName, vNames.c_str(), vInputs, vOutputs,
                 pMatrix.aPieces.size());
  pBackend.vTotals.vMatrices += aWeights.size();
  pBackend.vTotals.vWeights   += uint64_t(vInputs) * vOutputs;
  pBackend.vTotals.vWeightsMs += fNowMs() - vStart;
  return true;
}

// The converted matrix of some tensors of weights: the one kept from before or, the first time, a new one. What is
// not the weights of a model is converted into pTemporary, which the caller frees: it may change between graphs.
XdnaMatrix * fMatrixOf(XdnaShared & pShared, XdnaBackend & pBackend, const std::vector<const ggml_tensor *> & aWeights,
                        XdnaMatrix & pTemporary, std::string & pError, uint32_t pStreamSlot = 0) {
  std::vector<const void *> aKey;
  bool                      vKept = true;
  uint64_t                  vOutputsTotal = 0;
  for (const ggml_tensor * vWeights : aWeights) {
    const ggml_backend_buffer_t vBuf = vWeights->buffer != nullptr ?
                                           vWeights->buffer :
                                           (vWeights->view_src != nullptr ? vWeights->view_src->buffer : nullptr);
    fPushWeightKey(aKey, vWeights);
    vKept = vKept && vBuf != nullptr && ggml_backend_buffer_get_usage(vBuf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    vOutputsTotal += uint64_t(vWeights->ne[1]);
  }
  if (!vKept) {
    return fConvert(pShared, pBackend, aWeights, pTemporary, pError) ? &pTemporary : nullptr;
  }
  const auto vFound = pShared.dMatrices.find(aKey);
  if (vFound != pShared.dMatrices.end()) {
    XdnaMatrix & vMatrix = vFound->second;
    bool          vSame   = vMatrix.vInputs == uint32_t(aWeights[0]->ne[0]) && vMatrix.aTypes.size() == aWeights.size();
    for (size_t i = 0; vSame && i < aWeights.size(); i++) {
      vSame = vMatrix.aTypes[i] == aWeights[i]->type && vMatrix.aOutputs[i] == uint32_t(aWeights[i]->ne[1]);
    }
    if (vSame) {
      return &vMatrix;
    }
    fFreeMatrix(pShared, vMatrix);
    pShared.dMatrices.erase(vFound);
  }
  for (uint32_t s = 0; s < 2; s++) {
    if (pShared.aStreamKeys[s] == aKey) {
      XdnaMatrix & vMatrix = pShared.aStreamMatrices[s];
      bool         vSame   = vMatrix.vInputs == uint32_t(aWeights[0]->ne[0]) && vMatrix.aTypes.size() == aWeights.size();
      for (size_t i = 0; vSame && i < aWeights.size(); i++) {
        vSame = vMatrix.aTypes[i] == aWeights[i]->type && vMatrix.aOutputs[i] == uint32_t(aWeights[i]->ne[1]);
      }
      if (vSame) {
        return &vMatrix;
      }
    }
  }
  const bool     vWantComp = (pShared.vSettings.vPromptFormat == 1 ||
                              (pShared.vSettings.vPromptFormat < 0 && fPlacement().vCompPrompt.load())) &&
                             pShared.vOverlay.vCompB == 1 && aWeights[0]->type == GGML_TYPE_Q6_K &&
                             aWeights[0]->ne[0] % 512 == 0;
  const uint64_t vBfp16Bytes = (uint64_t(aWeights[0]->ne[0]) * vOutputsTotal / 8 * 9) * (vWantComp ? 3 : 4) / 4;
  const uint64_t vChainReserved = fPlacement().vChain + (512ull << 20);
  const bool     vCacheStatic = !fPlacement().vStreamPrompt.load() ||
                                (pShared.vSettings.vDenseTotalLimit > vChainReserved &&
                                 pShared.vCachedPromptBytes + vBfp16Bytes <= pShared.vSettings.vDenseTotalLimit - vChainReserved);
  if (!vCacheStatic) {
    const uint32_t vSlot = std::min<uint32_t>(1, pStreamSlot);
    pShared.aStreamKeys[vSlot].clear();
    if (!fConvert(pShared, pBackend, aWeights, pShared.aStreamMatrices[vSlot], pError, true)) {
      return nullptr;
    }
    pShared.aStreamKeys[vSlot] = std::move(aKey);
    return &pShared.aStreamMatrices[vSlot];
  }
  XdnaMatrix vMatrix;
  if (!fConvert(pShared, pBackend, aWeights, vMatrix, pError)) {
    return nullptr;
  }
  pShared.vCachedPromptBytes += vBfp16Bytes;
  fNoteDenseCached(pShared, aWeights, true);
  return &pShared.dMatrices.emplace(std::move(aKey), std::move(vMatrix)).first->second;
}

// The format a tensor of weights goes in to the overlay for generating tokens: Q4_K, Q5_K and Q6_K as they are, which
// the cores decode themselves (from half to three quarters of the bytes of bfp16), when the overlay takes them; any
// other type converted to bfp16.
XdnaGemvFormat fGemvFormatOf(const XdnaShared & pShared, const ggml_tensor * pWeights) {
  XdnaGemvFormat vFormat = XdnaGemvFormat::cBfp16;
  if (pWeights->type == GGML_TYPE_Q4_K) {
    vFormat = XdnaGemvFormat::cQ4K;
  } else if (pWeights->type == GGML_TYPE_Q5_K) {
    vFormat = XdnaGemvFormat::cQ5K;
  } else if (pWeights->type == GGML_TYPE_Q6_K) {
    vFormat = XdnaGemvFormat::cQ6K;
  } else if (pWeights->type == GGML_TYPE_Q8_0) {
    vFormat = XdnaGemvFormat::cQ8_0;
  }
  const bool     vTaken  = pShared.vHasChain ? pShared.vChainOverlay.mTakes(vFormat) : pShared.vGemvOverlay.mTakes(vFormat);
  const int64_t  vStepK  = vFormat == XdnaGemvFormat::cQ8_0 ? 32 : 256;
  if (vFormat != XdnaGemvFormat::cBfp16 && (!vTaken || pWeights->ne[0] % vStepK != 0)) {
    vFormat = XdnaGemvFormat::cBfp16;
  }
  return vFormat;
}

// The same as fConvert for the overlay for generating tokens: every tensor is a member in its format, and the members
// of one format one after another are a segment, which the runner converts one stream at a time. When the tensors
// make more segments than a command takes, they all go in bfp16. With the chained overlay, two tensors are two
// segments whatever their formats (a SwiGLU needs its gate and its up apart, see fFeedForward), and more tensors go
// as with the other one; a matrix takes 4 segments, in two phases of its command.
bool fGemvConvert(XdnaShared & pShared, XdnaBackend & pBackend, const std::vector<const ggml_tensor *> & aWeights,
                  XdnaGemvMatrix & pMatrix, std::string & pError) {
  const double   vStart   = fNowMs();
  const uint32_t vInputs  = uint32_t(aWeights[0]->ne[0]);
  uint32_t       vOutputs = 0;
  pMatrix.aTypes.clear();
  pMatrix.aOutputs.clear();
  std::vector<XdnaGemvFormat> aFormats;
  uint32_t                     vSegments = 0;
  for (const ggml_tensor * vWeights : aWeights) {
    pMatrix.aTypes.push_back(vWeights->type);
    pMatrix.aOutputs.push_back(uint32_t(vWeights->ne[1]));
    vOutputs += uint32_t(vWeights->ne[1]);
    aFormats.push_back(fGemvFormatOf(pShared, vWeights));
    vSegments += aFormats.size() == 1 || aFormats[aFormats.size() - 2] != aFormats.back() ? 1 : 0;
  }
  const bool     vApart       = pShared.vHasChain && aWeights.size() <= cXdnaChainSegmentsMax;
  const uint32_t vSegmentsMax = pShared.vHasChain ? cXdnaGemvSegmentsMax : pShared.vGemvOverlay.vSegments;
  if (vSegments > vSegmentsMax) {
    std::fill(aFormats.begin(), aFormats.end(), XdnaGemvFormat::cBfp16);
  }
  pMatrix.vInputs  = vInputs;
  pMatrix.vOutputs = vOutputs;
  XdnaWorkers & vWorkers = pShared.vRunner.mWorkers();

  std::vector<XdnaGemvMember> aMembers;
  std::string                  vFormats;
  for (size_t i = 0; i < aWeights.size(); i++) {
    const ggml_tensor * vWeights = aWeights[i];
    XdnaGemvMember     vMember;
    vMember.vFormat  = aFormats[i];
    vMember.vOutputs = uint32_t(vWeights->ne[1]);
    if (aFormats[i] == XdnaGemvFormat::cBfp16) {
      // pCount rows from pFirst on, as floats
      vMember.fRows = [vWeights, vInputs, &vWorkers](uint32_t pFirst, uint32_t pCount, void * pTo) {
        const uint32_t vTasks = (pCount + cRowsPerTask - 1) / cRowsPerTask;
        vWorkers.mForEach(vTasks, [&](uint32_t pTask) {
          const uint32_t vLast = std::min(pCount, (pTask + 1) * cRowsPerTask);
          for (uint32_t vRow = pTask * cRowsPerTask; vRow < vLast; vRow++) {
            const char * vFrom = static_cast<const char *>(vWeights->data) + size_t(pFirst + vRow) * vWeights->nb[1];
            float *      vTo   = static_cast<float *>(pTo) + size_t(vRow) * vInputs;
            if (vWeights->type == GGML_TYPE_F32) {
              std::memcpy(vTo, vFrom, size_t(vInputs) * sizeof(float));
            } else {
              ggml_get_type_traits(vWeights->type)->to_float(vFrom, vTo, vInputs);
            }
          }
        });
        return true;
      };
    } else {
      // the blocks of ggml as they are
      vMember.fRows = [vWeights, vInputs](uint32_t pFirst, uint32_t pCount, void * pTo) {
        const size_t vRowBytes = ggml_row_size(vWeights->type, vInputs);
        for (uint32_t vRow = 0; vRow < pCount; vRow++) {
          std::memcpy(static_cast<char *>(pTo) + size_t(vRow) * vRowBytes,
                      static_cast<const char *>(vWeights->data) + size_t(pFirst + vRow) * vWeights->nb[1], vRowBytes);
        }
        return true;
      };
    }
    vFormats += std::string(i == 0 ? "" : " ") + fXdnaGemvFormatName(aFormats[i]);
    aMembers.push_back(std::move(vMember));
  }
  const bool vLoaded = pShared.vHasChain ?
                           pShared.vRunner.mLoadChainMatrix(aMembers, vInputs, vApart, pMatrix.vChainWeights, pError) :
                           pShared.vRunner.mLoadGemvMatrix(aMembers, vInputs, pMatrix.vWeights, pError);
  if (!vLoaded) {
    return false;
  }
  pBackend.vTotals.vMatrices++;
  pBackend.vTotals.vWeights   += uint64_t(vInputs) * vOutputs;
  pBackend.vTotals.vWeightsMs += fNowMs() - vStart;
  GGML_LOG_DEBUG("%s: %s and %zu more taken for generating tokens (%s)\n", cName, aWeights[0]->name,
                 aWeights.size() - 1, vFormats.c_str());
  return true;
}

// The same as fMatrixOf for the overlay for generating tokens.
XdnaGemvMatrix * fGemvMatrixOf(XdnaShared & pShared, XdnaBackend & pBackend,
                                const std::vector<const ggml_tensor *> & aWeights, XdnaGemvMatrix & pTemporary,
                                std::string & pError) {
  std::vector<const void *> aKey;
  bool                      vKept = true;
  for (const ggml_tensor * vWeights : aWeights) {
    const ggml_backend_buffer_t vBuf = vWeights->buffer != nullptr ?
                                           vWeights->buffer :
                                           (vWeights->view_src != nullptr ? vWeights->view_src->buffer : nullptr);
    fPushWeightKey(aKey, vWeights);
    vKept = vKept && vBuf != nullptr && ggml_backend_buffer_get_usage(vBuf) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
  }
  if (!vKept) {
    return fGemvConvert(pShared, pBackend, aWeights, pTemporary, pError) ? &pTemporary : nullptr;
  }
  const auto vFound = pShared.dGemvMatrices.find(aKey);
  if (vFound != pShared.dGemvMatrices.end()) {
    XdnaGemvMatrix & vMatrix = vFound->second;
    bool vSame = vMatrix.vInputs == uint32_t(aWeights[0]->ne[0]) && vMatrix.aTypes.size() == aWeights.size();
    for (size_t i = 0; vSame && i < aWeights.size(); i++) {
      vSame = vMatrix.aTypes[i] == aWeights[i]->type && vMatrix.aOutputs[i] == uint32_t(aWeights[i]->ne[1]);
    }
    if (vSame) {
      return &vMatrix;
    }
    fFreeGemvMatrix(pShared, vMatrix);
    pShared.dGemvMatrices.erase(vFound);
  }
  XdnaGemvMatrix vMatrix;
  if (!fGemvConvert(pShared, pBackend, aWeights, vMatrix, pError)) {
    return nullptr;
  }
  fNoteDenseCached(pShared, aWeights, false);
  return &pShared.dGemvMatrices.emplace(std::move(aKey), std::move(vMatrix)).first->second;
}

// The rows of pA, pCount of them (1 to 8), by a matrix converted for generating tokens, into the targets of its
// outputs (aTargets, counted from the first output of the matrix): one command, or a few when the rows do not fit in
// one (fRowsPerCommand), each with its rows and the same targets moved down by them.
bool fMultiplyRows(XdnaShared & pShared, XdnaBackend & pBackend, const float * pA, uint64_t pCount,
                   const XdnaGemvMatrix & pMatrix, const std::vector<XdnaGemmTarget> & aTargets,
                   std::string & pError) {
  const uint64_t vEach = fRowsPerCommand(pShared, pCount, pMatrix.vInputs);
  if (vEach == 0) {
    pError = std::to_string(pCount) + " rows of " + std::to_string(pMatrix.vInputs) +
             " values do not fit in the overlay for generating tokens";
    return false;
  }
  std::vector<XdnaGemmTarget> aMoved(aTargets);
  for (uint64_t vFirst = 0; vFirst < pCount; vFirst += vEach) {
    const uint64_t vRows = std::min(vEach, pCount - vFirst);
    for (size_t t = 0; t < aTargets.size(); t++) {
      aMoved[t].vData = aTargets[t].vData + vFirst * aTargets[t].vStride;
    }
    pBackend.vTotals.vCommands++;
    pBackend.vTotals.vCommandsRows++;
    pBackend.vTotals.vSplitRows += vFirst > 0 ? 1 : 0;
    XdnaGemmTimes vTimes;
    const float *  vA    = pA + vFirst * pMatrix.vInputs;
    const bool     vDone = pShared.vHasChain ?
                               pShared.vRunner.mChainMultiplyRows(vA, uint32_t(vRows), pMatrix.vInputs,
                                                                  pMatrix.vChainWeights, aMoved.data(),
                                                                  uint32_t(aMoved.size()), vTimes, pError) :
                               pShared.vRunner.mMultiplyRows(vA, uint32_t(vRows), pMatrix.vInputs, pMatrix.vWeights,
                                                             aMoved.data(), uint32_t(aMoved.size()), vTimes, pError);
    if (!vDone) {
      return false;
    }
    pBackend.vTotals.vPlanMs    += vTimes.vPlanMs;
    pBackend.vTotals.vPackMs    += vTimes.vPackMs;
    pBackend.vTotals.vNpuMs     += vTimes.vNpuMs;
    pBackend.vTotals.vNpuRowsMs += vTimes.vNpuMs;
    pBackend.vTotals.vUnpackMs  += vTimes.vUnpackMs;
    pBackend.vTotals.vFlushMs   += vTimes.vFlushMs;
  }
  return true;
}

// the target that takes the output pOutput, or none
const XdnaGemmTarget * fTargetOf(const XdnaGemmTarget * aTargets, uint32_t pTargets, uint32_t pOutput) {
  for (uint32_t i = 0; i < pTargets; i++) {
    if (pOutput >= aTargets[i].vFirst && pOutput - aTargets[i].vFirst < aTargets[i].vCount) {
      return &aTargets[i];
    }
  }
  return nullptr;
}

// What the NPU computes for pRows rows and a piece of weights, on the processor, into the targets of its outputs.
bool fEmulate(XdnaShared & pShared, XdnaBackend & pBackend, const float * pA, uint32_t pRows,
              const XdnaPiece & pPiece, const XdnaGemmTarget * aTargets, uint32_t pTargets, std::string & pError) {
  // the rows, rounded as the NPU would read them; the last block of each is filled with zeros, which add nothing
  double         vStart  = fNowMs();
  const uint32_t vBlocks = fBlocksOf(pPiece.vInputs);
  const uint32_t vPadded = (vBlocks * cXdnaBfp16Values + 15) / 16 * 16;  // a multiple of what fDot takes
  std::vector<uint8_t> aEncoded(size_t(vBlocks) * cXdnaBfp16Bytes);
  pBackend.aRounded.assign(size_t(pRows) * vPadded, 0.0f);
  for (uint32_t vRow = 0; vRow < pRows; vRow++) {
    if (!fEncodeRows(pA + size_t(vRow) * pPiece.vInputs, 1, pPiece.vInputs, aEncoded.data())) {
      pError = "a row has a value that is not finite";
      return false;
    }
    fDecodeBlocks(aEncoded.data(), vBlocks, pBackend.aRounded.data() + size_t(vRow) * vPadded);
  }
  pBackend.vTotals.vPackMs += fNowMs() - vStart;

  vStart = fNowMs();
  const float * vRounded = pBackend.aRounded.data();
  // the sums, rounded as the overlay gives them back
  const bool    vToBf16  = pShared.vOverlay.vOutput == XdnaGemmOutput::cBfloat16;
  auto fWork = [&](uint32_t pFirst, uint32_t pLast) {
    std::vector<float> aWeights(vPadded, 0.0f);
    for (uint32_t vOutput = pFirst; vOutput < pLast; vOutput++) {
      const XdnaGemmTarget * vTarget = fTargetOf(aTargets, pTargets, vOutput);
      if (vTarget == nullptr) {
        continue;
      }
      float * vTo = vTarget->vData + (vOutput - vTarget->vFirst);
      fDecodeBlocks(pPiece.aBlocks.data() + size_t(vOutput) * vBlocks * cXdnaBfp16Bytes, vBlocks, aWeights.data());
      for (uint32_t vRow = 0; vRow < pRows; vRow++) {
        const float vSum = fDot(vRounded + size_t(vRow) * vPadded, aWeights.data(), vPadded);
        vTo[size_t(vRow) * vTarget->vStride] = vToBf16 ? fXdnaBf16ToFloat(fXdnaBf16FromFloat(vSum)) : vSum;
      }
    }
  };
  if (uint64_t(pRows) * pPiece.vOutputs * vPadded < cThreadedWork) {
    fWork(0, pPiece.vOutputs);
  } else {
    // some outputs for every thread, and several turns for each, so that none is left waiting for the slowest
    const uint32_t vTasks = std::min(pPiece.vOutputs, pShared.vRunner.mWorkers().mThreads() * 4);
    pShared.vRunner.mWorkers().mForEach(vTasks, [&](uint32_t pTask) {
      fWork(uint32_t(uint64_t(pPiece.vOutputs) * pTask / vTasks),
            uint32_t(uint64_t(pPiece.vOutputs) * (pTask + 1) / vTasks));
    });
  }
  // In block floating point every 8 rows of an output share an exponent, from the first row on, as the cores give
  // the 8 x 8 sums of a multiply transposed: 8 tokens of one output have values of a similar size.
  if (pShared.vOverlay.vOutput == XdnaGemmOutput::cBfp16) {
    for (uint32_t vOutput = 0; vOutput < pPiece.vOutputs; vOutput++) {
      const XdnaGemmTarget * vTarget = fTargetOf(aTargets, pTargets, vOutput);
      if (vTarget == nullptr) {
        continue;
      }
      float * vBase = vTarget->vData + (vOutput - vTarget->vFirst);
      for (uint32_t vGroup = 0; vGroup < pRows; vGroup += cXdnaBfp16Values) {
        float aValues[cXdnaBfp16Values] = {};
        for (uint32_t i = 0; i < cXdnaBfp16Values && vGroup + i < pRows; i++) {
          aValues[i] = vBase[size_t(vGroup + i) * vTarget->vStride];
        }
        uint8_t aBlock[cXdnaBfp16Bytes];
        if (!fXdnaBfp16EncodeNearest(aValues, cXdnaBfp16Values, 1, aBlock)) {
          pError = "a result is not finite";
          return false;
        }
        fXdnaBfp16Decode(aBlock, aValues);
        for (uint32_t i = 0; i < cXdnaBfp16Values && vGroup + i < pRows; i++) {
          vBase[size_t(vGroup + i) * vTarget->vStride] = aValues[i];
        }
      }
    }
  }
  pBackend.vTotals.vNpuMs += fNowMs() - vStart;
  return true;
}

// Sums over the inputs of pA[row][input] * weights[output][input], for one piece of weights, into the targets of
// its outputs (counted from the first output of the piece).
bool fMultiplyPiece(XdnaShared & pShared, XdnaBackend & pBackend, const float * pA, uint32_t pRows,
                    const XdnaPiece & pPiece, const XdnaGemmTarget * aTargets, uint32_t pTargets,
                    std::string & pError) {
  pBackend.vTotals.vCommands++;
  if (pShared.vSettings.vEmulate) {
    return fEmulate(pShared, pBackend, pA, pRows, pPiece, aTargets, pTargets, pError);
  }
  XdnaGemmTimes vTimes;
  if (!pShared.vRunner.mMultiply(pA, pRows, pPiece.vWeights, cRounding, aTargets, pTargets, vTimes, pError)) {
    return false;
  }
  pBackend.vTotals.vPlanMs   += vTimes.vPlanMs;
  pBackend.vTotals.vPackMs   += vTimes.vPackMs;
  pBackend.vTotals.vNpuMs    += vTimes.vNpuMs;
  pBackend.vTotals.vUnpackMs += vTimes.vUnpackMs;
  pBackend.vTotals.vFlushMs  += vTimes.vFlushMs;
  return true;
}

void fDrainEarlyPacked(XdnaShared & pShared, XdnaBackend & pBackend) {
  if (pBackend.vPacked.vEarlySent.vValid) {
    XdnaGemmTimes vIgnoredTimes;
    std::string   vIgnoredError;
    pShared.vRunner.mFinishPacked(pBackend.vPacked.vEarlySent, nullptr, 0, false, vIgnoredTimes, vIgnoredError, false);
  }
  pBackend.vPacked = XdnaBackend::XdnaPacked();
}

// The rows of pA, pCount of them, by a converted matrix, into the targets of its outputs (pBackend.aTargets, sorted
// and not overlapping): in commands of cRowsPerCommand rows at most and piece by piece. The pieces along the
// outputs fill different results, and the ones along the inputs add up.
bool fMultiply(XdnaShared & pShared, XdnaBackend & pBackend, const float * pA, uint64_t pCount,
               const XdnaMatrix & pMatrix, std::string & pError, const XdnaRowNorm * pNorm = nullptr,
               bool pPacked = false, XdnaRunner::XdnaPending * pEarlySent = nullptr) {
  const std::vector<XdnaGemmTarget> & aTargets = pBackend.aTargets;
  // llama.cpp-xdna: with pPacked, the rows are already in the buffers of A cXdnaRunnerFirstReadyA on, one per piece
  // along the inputs (fMulMatGlu converted them from its SwiGLU), and pA is not read
  if (pPacked && (pShared.vSettings.vEmulate || pCount > cRowsPerCommand ||
                  pMatrix.vRangesK > cXdnaRunnerSlotsA - cXdnaRunnerFirstReadyA)) {
    pError = "the rows converted ahead do not fit this multiplication";
    return false;
  }
  if (!pShared.vSettings.vEmulate) {
    // On the NPU, the rows of a command and a range of inputs are converted once, straight from their columns, for
    // every piece along the outputs; the pieces after the first along the inputs add to the targets as C is read.
    // (Before, every piece converted the rows again, a range of inputs was copied out first, and its sums went to a
    // buffer of their own and were added afterwards.) Up to two commands go at once (2026/11/15): every one is sent
    // before the C of the one before it is read, and the rows of the next range of inputs are converted while the
    // NPU works, into the other buffer of A, so the host reads and converts while the NPU multiplies; C is still
    // read in the order the commands were sent, so the pieces along the inputs add up in the same order as before.
    XdnaRunner::XdnaPending    vPending;  // the command sent whose C is not read yet
    std::vector<XdnaGemmTarget> aPendingTargets;
    bool                         vPendingAdd = false;
    uint32_t                     vSlotA      = 0;
    uint32_t                     vSlotC      = 0;
    const auto                   fAddTimes   = [&](const XdnaGemmTimes & pTimes) {
      pBackend.vTotals.vPlanMs   += pTimes.vPlanMs;
      pBackend.vTotals.vPackMs   += pTimes.vPackMs;
      pBackend.vTotals.vNpuMs    += pTimes.vNpuMs;
      pBackend.vTotals.vUnpackMs += pTimes.vUnpackMs;
      pBackend.vTotals.vFlushMs  += pTimes.vFlushMs;
    };
    const auto fFinish = [&]() -> bool {
      if (!vPending.vValid) {
        return true;
      }
      XdnaGemmTimes vTimes;
      const bool     vOk = pShared.vRunner.mFinishPacked(vPending, aPendingTargets.data(),
                                                         uint32_t(aPendingTargets.size()), vPendingAdd, vTimes, pError);
      fAddTimes(vTimes);
      return vOk;
    };
    for (uint64_t vFirst = 0; vFirst < pCount; vFirst += cRowsPerCommand) {
      const uint32_t vRows = uint32_t(std::min<uint64_t>(cRowsPerCommand, pCount - vFirst));
      for (uint32_t vRangeK = 0; vRangeK < pMatrix.vRangesK; vRangeK++) {
        const XdnaPiece & vAlongK = pMatrix.aPieces[size_t(vRangeK) * pMatrix.vRangesN];
        XdnaGemmTimes     vPacked;
        // the rows normalised as they are converted, when an RMS_NORM goes with them (llama.cpp-xdna)
        XdnaRowNorm vPartNorm;
        if (pNorm != nullptr) {
          vPartNorm.vScales  = pNorm->vScales + vFirst;
          vPartNorm.vWeights = pNorm->vWeights + vAlongK.vFirstInput;
        }
        const uint32_t vSlotNow = pPacked ? cXdnaRunnerFirstReadyA + vRangeK : vSlotA;
        if (!pPacked &&
            !pShared.vRunner.mPackRows(pA + vFirst * pMatrix.vInputs + vAlongK.vFirstInput, vRows, vAlongK.vInputs,
                                       pMatrix.vInputs, cRounding, vPacked, pError, vSlotA,
                                       pNorm != nullptr ? &vPartNorm : nullptr)) {
          std::string vIgnored;
          std::swap(vIgnored, pError);
          fFinish();
          std::swap(vIgnored, pError);
          return false;
        }
        fAddTimes(vPacked);
        for (uint32_t vRangeN = 0; vRangeN < pMatrix.vRangesN; vRangeN++) {
          const XdnaPiece &             vPiece        = pMatrix.aPieces[size_t(vRangeK) * pMatrix.vRangesN + vRangeN];
          std::vector<XdnaGemmTarget> & aPieceTargets = pBackend.aPieceTargets;
          aPieceTargets.clear();
          for (const XdnaGemmTarget & vTarget : aTargets) {
            const uint32_t vFrom = std::max(vTarget.vFirst, vPiece.vFirstOutput);
            const uint32_t vTo   = std::min(vTarget.vFirst + vTarget.vCount, vPiece.vFirstOutput + vPiece.vOutputs);
            if (vFrom < vTo) {
              XdnaGemmTarget vPart;
              vPart.vData   = vTarget.vData + vFirst * vTarget.vStride + (vFrom - vTarget.vFirst);
              vPart.vStride = vTarget.vStride;
              vPart.vFirst  = vFrom - vPiece.vFirstOutput;
              vPart.vCount  = vTo - vFrom;
              if (vTarget.vConvWeights != nullptr) {
                // llama.cpp-xdna: a short convolution (fConvFold: one command of rows, so vFirst is 0)
                const uint64_t vOffset = vFrom - vTarget.vFirst;
                vPart.vConvWeights     = vTarget.vConvWeights + vOffset;
                vPart.vConvHead        = vTarget.vConvHead + vOffset;
                vPart.vConvTail        = vTarget.vConvTail + vOffset;
                vPart.vConvStride      = vTarget.vConvStride;
                if (vOffset == 0 && vPart.vCount == vTarget.vCount) {
                  vPart.vGdnPrepack = vTarget.vGdnPrepack;
                }
              }
              aPieceTargets.push_back(vPart);
            }
          }
          if (pBackend.vResidual.vActive && vRangeK + 1 == pMatrix.vRangesK) {
            // the last piece along the inputs: the residual ADD, into its own result (llama.cpp-xdna); the pieces
            // before it added up in the result of the multiplication
            for (XdnaGemmTarget & vPart : aPieceTargets) {
              const uint64_t vOffset = uint64_t(vPart.vData - pBackend.vResidual.vProduct);
              vPart.vBase = pBackend.vResidual.vDst != pBackend.vResidual.vProduct ?
                                pBackend.vResidual.vProduct + vOffset : nullptr;
              vPart.vData = pBackend.vResidual.vDst + vOffset;
              vPart.vAlso = pBackend.vResidual.vAlso + vOffset;
            }
          }
          pBackend.vTotals.vCommands++;
          XdnaRunner::XdnaPending vSent;
          XdnaGemmTimes            vTimes;
          if (pEarlySent != nullptr && pEarlySent->vValid && vFirst == 0 && vRangeK == 0 && vRangeN == 0) {
            vSent       = *pEarlySent;
            *pEarlySent = XdnaRunner::XdnaPending();
          } else if (!pShared.vRunner.mSubmitPacked(vPiece.vWeights, vSlotNow, vSlotC, vSent, vTimes, pError)) {
            std::string vIgnored;
            std::swap(vIgnored, pError);
            fFinish();
            std::swap(vIgnored, pError);
            return false;
          }
          fAddTimes(vTimes);
          // the one before, while this one runs
          if (!fFinish()) {
            std::string vIgnored;
            std::swap(vIgnored, pError);
            vPending = vSent;
            fFinish();
            std::swap(vIgnored, pError);
            return false;
          }
          vPending    = vSent;
          vPendingAdd = vRangeK > 0;
          aPendingTargets.swap(aPieceTargets);
          vSlotC ^= 1;
          if (pShared.vSettings.vNoOverlap && !fFinish()) {
            return false;
          }
        }
        if (!pPacked) {
          vSlotA ^= 1;
        }
      }
    }
    return fFinish();
  }
  for (uint64_t vFirst = 0; vFirst < pCount; vFirst += cRowsPerCommand) {
    const uint32_t vRows = uint32_t(std::min<uint64_t>(cRowsPerCommand, pCount - vFirst));
    for (uint32_t vRangeK = 0; vRangeK < pMatrix.vRangesK; vRangeK++) {
      const XdnaPiece & vAlongK = pMatrix.aPieces[size_t(vRangeK) * pMatrix.vRangesN];
      const float *      vPartA  = pA + vFirst * pMatrix.vInputs;
      if (pMatrix.vRangesK > 1) {
        pBackend.aPartA.resize(size_t(vRows) * vAlongK.vInputs);
        for (uint32_t vRow = 0; vRow < vRows; vRow++) {
          std::memcpy(pBackend.aPartA.data() + size_t(vRow) * vAlongK.vInputs,
                      pA + (vFirst + vRow) * pMatrix.vInputs + vAlongK.vFirstInput, vAlongK.vInputs * sizeof(float));
        }
        vPartA = pBackend.aPartA.data();
      }
      for (uint32_t vRangeN = 0; vRangeN < pMatrix.vRangesN; vRangeN++) {
        const XdnaPiece & vPiece = pMatrix.aPieces[size_t(vRangeK) * pMatrix.vRangesN + vRangeN];
        // the targets of the outputs of the piece, for the rows of the command, counted from its first output
        std::vector<XdnaGemmTarget> & aPieceTargets = pBackend.aPieceTargets;
        aPieceTargets.clear();
        for (const XdnaGemmTarget & vTarget : aTargets) {
          const uint32_t vFrom = std::max(vTarget.vFirst, vPiece.vFirstOutput);
          const uint32_t vTo   = std::min(vTarget.vFirst + vTarget.vCount, vPiece.vFirstOutput + vPiece.vOutputs);
          if (vFrom < vTo) {
            XdnaGemmTarget vPart;
            vPart.vData   = vTarget.vData + vFirst * vTarget.vStride + (vFrom - vTarget.vFirst);
            vPart.vStride = vTarget.vStride;
            vPart.vFirst  = vFrom - vPiece.vFirstOutput;
            vPart.vCount  = vTo - vFrom;
            aPieceTargets.push_back(vPart);
          }
        }
        if (vRangeK == 0) {
          if (!fMultiplyPiece(pShared, pBackend, vPartA, vRows, vPiece, aPieceTargets.data(),
                              uint32_t(aPieceTargets.size()), pError)) {
            return false;
          }
          continue;
        }
        // the pieces after the first along the inputs add up to what the ones before left
        pBackend.aPartC.resize(size_t(vRows) * vPiece.vOutputs);
        XdnaGemmTarget vWhole;
        vWhole.vData   = pBackend.aPartC.data();
        vWhole.vStride = vPiece.vOutputs;
        vWhole.vFirst  = 0;
        vWhole.vCount  = vPiece.vOutputs;
        if (!fMultiplyPiece(pShared, pBackend, vPartA, vRows, vPiece, &vWhole, 1, pError)) {
          return false;
        }
        for (const XdnaGemmTarget & vPart : aPieceTargets) {
          for (uint32_t vRow = 0; vRow < vRows; vRow++) {
            const float * vFrom = pBackend.aPartC.data() + size_t(vRow) * vPiece.vOutputs + vPart.vFirst;
            float *       vTo   = vPart.vData + size_t(vRow) * vPart.vStride;
            for (uint32_t i = 0; i < vPart.vCount; i++) {
              vTo[i] += vFrom[i];
            }
          }
        }
      }
    }
  }
  return true;
}

// Multiplies the rows of the second operand of pNodes[0] by the weights of every node of pNodes, pCount of them,
// which all have those same rows: as one matrix with the outputs of all, so that the rows are converted once and
// the NPU gets one command for all of them (see fGraphCompute). Every result gets its own outputs.
bool fMulMat(XdnaBackend & pBackend, ggml_tensor * const * pNodes, uint32_t pCount, std::string & pError) {
  XdnaShared &       vShared = fShared();
  const ggml_tensor * vRows   = pNodes[0]->src[1];
  std::vector<const ggml_tensor *> aWeights;
  for (uint32_t i = 0; i < pCount; i++) {
    if (!fMulMatTaken(vShared, pNodes[i])) {
      pError = "it is not a multiplication this backend takes";
      return false;
    }
    if (pNodes[i]->src[1] != vRows) {
      pError = "the multiplications held together do not have the same rows";
      return false;
    }
    if (pNodes[i]->src[0]->data == nullptr || pNodes[i]->data == nullptr) {
      pError = "a tensor has no memory";
      return false;
    }
    // held together, every result is written where it is
    if (pCount > 1 && !ggml_is_contiguous(pNodes[i])) {
      pError = "a result of the multiplications held together is not one row after the other";
      return false;
    }
    aWeights.push_back(pNodes[i]->src[0]);
  }
  if (vRows->data == nullptr) {
    pError = "a tensor has no memory";
    return false;
  }
  const double   vStart  = fNowMs();
  const uint32_t vInputs = uint32_t(aWeights[0]->ne[0]);
  const uint64_t vCount  = uint64_t(vRows->ne[1]) * uint64_t(vRows->ne[2]) * uint64_t(vRows->ne[3]);

  // the rows, each as one row after the other; with an RMS_NORM that goes with them (llama.cpp-xdna), the rows it
  // reads, normalised as they are converted, or normalised here first with few rows or emulating
  const bool           vMoeByRows = fByRows(vShared, vCount, vInputs) && fMoeActiveForRows(vShared, vCount) &&
                                    !vShared.vSettings.vMoeGemv &&
                                    (!vShared.vSettings.vNoMoeDense || (pCount == 1 && fGivesLogits(aWeights[0])));
  const float *        vA     = static_cast<const float *>(vRows->data);
  XdnaRowNorm         vNorm;
  const XdnaRowNorm * vByRow = nullptr;
  if (pBackend.vNormed.vActive && vRows == pBackend.vNormed.vRows) {
    vNorm.vScales  = pBackend.aScales.data();
    vNorm.vWeights = pBackend.vNormed.vWeights;
    if ((vShared.vSettings.vEmulate || fByRows(vShared, vCount, vInputs) ||
         (pCount == 1 && fIsSharedGate(aWeights[0]))) &&
        !vMoeByRows) {
      pBackend.aRowsA.resize(vCount * vInputs);
      for (uint64_t vRow = 0; vRow < vCount; vRow++) {
        const float * vFrom = pBackend.vNormed.vX + vRow * vInputs;
        float *       vTo   = pBackend.aRowsA.data() + vRow * vInputs;
        for (uint32_t vColumn = 0; vColumn < vInputs; vColumn++) {
          vTo[vColumn] = (vFrom[vColumn] * vNorm.vScales[vRow]) * vNorm.vWeights[vColumn];
        }
      }
      vA = pBackend.aRowsA.data();
    } else {
      vA     = pBackend.vNormed.vX;
      vByRow = &vNorm;
    }
  } else if (!ggml_is_contiguous(vRows)) {
    pBackend.aRowsA.resize(vCount * vInputs);
    for (uint64_t vRow = 0; vRow < vCount; vRow++) {
      std::memcpy(pBackend.aRowsA.data() + vRow * vInputs, fRowOf(vRows, vRow), vInputs * sizeof(float));
    }
    vA = pBackend.aRowsA.data();
  }

  // where the outputs go: every result, one after the other; a lone result whose rows are not one after the other
  // gets them first in a room of its own
  pBackend.aTargets.clear();
  float *  vGathered = nullptr;
  uint32_t vFirst    = 0;
  for (uint32_t i = 0; i < pCount; i++) {
    const uint32_t  vOutputs = uint32_t(aWeights[i]->ne[1]);
    XdnaGemmTarget vTarget;
    vTarget.vData   = static_cast<float *>(pNodes[i]->data);
    vTarget.vStride = vOutputs;
    vTarget.vFirst  = vFirst;
    vTarget.vCount  = vOutputs;
    if (pBackend.vRedirect.vActive && pBackend.vRedirect.vMember == pNodes[i]) {
      vTarget.vData = pBackend.vRedirect.vData;  // straight into a CONCAT (llama.cpp-xdna)
      if (pBackend.vRedirect.vConvOn) {
        // llama.cpp-xdna: its short convolution with the SiLU into the SILU instead, and the last 3 rows into the
        // CONCAT (fConvFold)
        const XdnaConvFold & vFold = pBackend.vRedirect.vConv;
        vTarget.vData        = static_cast<float *>(vFold.vSilu->data);
        vTarget.vStride      = uint64_t(vFold.vChannels);
        vTarget.vConvWeights = pBackend.aConvWeights.data();
        vTarget.vConvHead    = pBackend.aConvHead.data();
        vTarget.vConvTail    = pBackend.vRedirect.vData + uint64_t(vFold.vTokens - 3) * uint64_t(vFold.vChannels);
        vTarget.vConvStride  = uint64_t(vFold.vChannels);
        if (pBackend.vRedirect.vGdnPrepackOn && vFirst == 0) {
          vTarget.vGdnPrepack = &pBackend.vRedirect.vGdnPrepack;
        }
      }
    }
    if (!ggml_is_contiguous(pNodes[i])) {
      pBackend.aRowsC.resize(vCount * vOutputs);
      vGathered     = pBackend.aRowsC.data();
      vTarget.vData = vGathered;
    }
    pBackend.aTargets.push_back(vTarget);
    vFirst += vOutputs;
  }

  bool vOk = false;
  if (pCount == 1 && fIsSharedGate(aWeights[0]) && aWeights[0]->ne[1] == 1) {
    std::vector<float> aGateWeights;
    const float *      vGateW = nullptr;
    if (aWeights[0]->type == GGML_TYPE_F32) {
      vGateW = static_cast<const float *>(aWeights[0]->data);
    } else {
      aGateWeights.resize(vInputs);
      ggml_get_type_traits(aWeights[0]->type)->to_float(aWeights[0]->data, aGateWeights.data(), vInputs);
      vGateW = aGateWeights.data();
    }
    float * vOutData = pBackend.aTargets[0].vData;
    auto    fGateRow = [&](uint32_t pRow) {
      vOutData[pRow] = fDotRow(vA + size_t(pRow) * vInputs, vGateW, vInputs);
    };
    if (vCount >= 16) {
      std::lock_guard<std::mutex> vLock(vShared.vMutex);
      vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
      vShared.vRunner.mWorkers().mForEach(uint32_t(vCount), fGateRow);
    } else {
      for (uint32_t r = 0; r < vCount; r++) {
        fGateRow(r);
      }
    }
    vOk = true;
  }
  if (!vOk) {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    if (fByRows(vShared, vCount, vInputs)) {
      // llama.cpp-xdna: when a token already runs its routed experts on the mixture-of-experts overlay, run its dense
      // projections and output.weight there too as 1 expert so the token never switches between overlays
      if (vMoeByRows && !(pBackend.vRedirect.vActive && pBackend.vRedirect.vConvOn)) {
        XdnaMoeWeights * vMoeGroup = fEnsureMoeDenseGroup(vShared, pBackend, aWeights, pError);
        if (vMoeGroup != nullptr && vCount <= fXdnaMoeMaxRows(vShared.vMoeOverlay, vMoeGroup->vMatrix)) {
          const XdnaMoeChunk          vChunk = {0, uint32_t(vCount)};
          std::vector<uint32_t>       aIdx(vCount);
          std::vector<XdnaGemmTarget> aMoeTargets(pBackend.aTargets);
          if (pBackend.vResidual.vActive && pCount == 1) {
            aMoeTargets[0].vData = pBackend.vResidual.vDst;
            aMoeTargets[0].vAlso = pBackend.vResidual.vAlso;
          }
          for (uint64_t r = 0; r < vCount; r++) {
            aIdx[r] = uint32_t(r);
          }
          XdnaGemmTimes vTimes;
          vOk = vShared.vRunner.mRunMoe(*vMoeGroup, &vChunk, 1, vA, vInputs, uint32_t(vCount), aIdx.data(), nullptr,
                                        nullptr, vTimes, pError, nullptr, vByRow, aMoeTargets.data(),
                                        uint32_t(aMoeTargets.size()));
          if (vOk) {
            pBackend.vTotals.vCommands++;
            pBackend.vTotals.vCommandsRows++;
            pBackend.vTotals.vMoeDenseCmds++;
            pBackend.vTotals.vMoeDenseNpuMs += vTimes.vNpuMs;
            pBackend.vTotals.vNpuMs         += vTimes.vNpuMs;
            pBackend.vTotals.vNpuRowsMs     += vTimes.vNpuMs;
            pBackend.vTotals.vPackMs        += vTimes.vPackMs;
            pBackend.vTotals.vUnpackMs      += vTimes.vUnpackMs;
            pBackend.vTotals.vFlushMs       += vTimes.vFlushMs;
            pBackend.vTotals.vPlanMs        += vTimes.vPlanMs;
          }
        }
      }
      if (!vOk && pError.empty()) {
        // A few rows, generating tokens: the overlay for that, one command for all the results. A command takes up
        // to 4 segments of outputs, each in a format, so the tensors of a group go each in its own format (Q4_K,
        // Q5_K, Q6_K as they are, the rest in bfp16), one segment for every run of one format (fGemvConvert). One
        // command per format was measured on 2026/11/15 with Qwen3.5 4B Q4_K_M, before the segments: 30 more
        // commands per token, which cost what the bytes saved.
        XdnaGemvMatrix   vTemporary;
        XdnaGemvMatrix * vMatrix = fGemvMatrixOf(vShared, pBackend, aWeights, vTemporary, pError);
        if (vMatrix != nullptr) {
          vOk = fMultiplyRows(vShared, pBackend, vA, vCount, *vMatrix, pBackend.aTargets, pError);
          if (vMatrix == &vTemporary) {
            fFreeGemvMatrix(vShared, vTemporary);
          }
        }
      }
    } else {
      XdnaMatrix   vTemporary;
      XdnaMatrix * vMatrix = fMatrixOf(vShared, pBackend, aWeights, vTemporary, pError);
      if (vMatrix != nullptr) {
        const XdnaTotals vBefore = pBackend.vTotals;
        // llama.cpp-xdna: the rows of a SwiGLU that fMulMatGlu converted ahead (XdnaBackend::vPacked)
        const bool              vReady = pBackend.vPacked.vActive && pBackend.vPacked.vRows == vRows && pCount == 1;
        XdnaRunner::XdnaPending vEarlySent;
        if (vReady && pBackend.vPacked.vEarlySent.vValid && !vMatrix->aPieces.empty() &&
            vMatrix->aPieces[0].vWeights.vBuffer.vHandle == pBackend.vPacked.vEarlyWeightHandle) {
          vEarlySent                   = pBackend.vPacked.vEarlySent;
          pBackend.vPacked.vEarlySent  = XdnaRunner::XdnaPending();
        }
        fDrainEarlyPacked(vShared, pBackend);
        vOk = fMultiply(vShared, pBackend, vA, vCount, *vMatrix, pError, vByRow, vReady,
                        vEarlySent.vValid ? &vEarlySent : nullptr);
        if (vEarlySent.vValid) {
          XdnaGemmTimes vIgnoredTimes;
          std::string   vIgnoredError;
          vShared.vRunner.mFinishPacked(vEarlySent, nullptr, 0, false, vIgnoredTimes, vIgnoredError, false);
        }
        // llama.cpp-xdna: what the multiplication of the rows of a SwiGLU takes (fBackendFree)
        if (pCount == 1 && vRows->op == GGML_OP_GLU) {
          pBackend.vTotals.vDownPackMs   += pBackend.vTotals.vPackMs - vBefore.vPackMs;
          pBackend.vTotals.vDownNpuMs    += pBackend.vTotals.vNpuMs - vBefore.vNpuMs;
          pBackend.vTotals.vDownUnpackMs += pBackend.vTotals.vUnpackMs - vBefore.vUnpackMs;
          pBackend.vTotals.vDowns++;
        }
        if (vMatrix == &vTemporary) {
          fFreeMatrix(vShared, vTemporary);
        }
      }
    }
  }
  if (!vOk) {
    return false;
  }

  if (vGathered != nullptr) {
    const uint32_t vOutputs = uint32_t(aWeights[0]->ne[1]);
    for (uint64_t vRow = 0; vRow < vCount; vRow++) {
      std::memcpy(fRowOf(pNodes[0], vRow), vGathered + vRow * vOutputs, vOutputs * sizeof(float));
    }
  }
  pBackend.vTotals.vOperations += pCount;
  pBackend.vTotals.vRows       += vCount * pCount;
  pBackend.vTotals.vTotalMs    += fNowMs() - vStart;
  return true;
}

// A SwiGLU on the host, from its two results, as the CPU backend computes it (xdna-glu.h): one that could not be
// computed while the C of its multiplications was read (fMulMatGlu), as when generating tokens.
bool fGluOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vGate = pNode->src[0];
  const ggml_tensor * vUp   = pNode->src[1];
  if (vGate->data == nullptr || vUp->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor has no memory";
    return false;
  }
  const double   vStart  = fNowMs();
  const uint64_t vLength = uint64_t(pNode->ne[0]);
  const uint64_t vRows   = uint64_t(ggml_nrows(pNode));
  const float *  vFromG  = static_cast<const float *>(vGate->data);
  const float *  vFromU  = static_cast<const float *>(vUp->data);
  float *        vTo     = static_cast<float *>(pNode->data);
  auto fRow = [&](uint32_t pRow) {
    fXdnaSwiglu(vFromG + pRow * vLength, vFromU + pRow * vLength, vTo + pRow * vLength, uint32_t(vLength), 0, vLength);
  };
  if (vRows >= 16) {
    XdnaShared &               vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    vShared.vRunner.mWorkers().mForEach(uint32_t(vRows), fRow);
  } else {
    for (uint32_t vRow = 0; vRow < vRows; vRow++) {
      fRow(vRow);
    }
  }
  pBackend.vTotals.vGlusHost++;
  pBackend.vTotals.vGluMs += fNowMs() - vStart;
  return true;
}

// llama.cpp-xdna: the scales of RMS_NORM of pRows rows of pLength floats, one after the other, into
// pBackend.aScales, as the CPU backend computes them (fXdnaRmsScales), shared between the threads with many rows
void fRowScales(XdnaBackend & pBackend, const float * pX, uint64_t pRows, uint64_t pLength, float pEps) {
  pBackend.aScales.resize(pRows);
  constexpr uint64_t cRowsPerTask = 64;
  const uint64_t     vTasks       = (pRows + cRowsPerTask - 1) / cRowsPerTask;
  auto fTask = [&](uint32_t pTask) {
    const uint64_t vFirst = uint64_t(pTask) * cRowsPerTask;
    const uint64_t vCount = std::min(cRowsPerTask, pRows - vFirst);
    fXdnaRmsScales(pX + vFirst * pLength, vCount, pLength, pLength, pEps, pBackend.aScales.data() + vFirst);
  };
  if (vTasks > 1) {
    XdnaShared &               vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    vShared.vRunner.mWorkers().mForEach(uint32_t(vTasks), fTask);
  } else if (vTasks == 1) {
    fTask(0);
  }
}

// llama.cpp-xdna: an RMS_NORM on the host, as the CPU backend computes it alone: every value times the scale of its
// row (one that did not go with the multiplications, as when generating tokens)
bool fRmsNormOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vX = pNode->src[0];
  if (vX->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor has no memory";
    return false;
  }
  float vEps;
  memcpy(&vEps, pNode->op_params, sizeof(float));
  const uint64_t vLength = uint64_t(pNode->ne[0]);
  const uint64_t vRows   = uint64_t(ggml_nrows(pNode));
  const float *  vFrom   = static_cast<const float *>(vX->data);
  float *        vTo     = static_cast<float *>(pNode->data);
  fRowScales(pBackend, vFrom, vRows, vLength, vEps);
  for (uint64_t vRow = 0; vRow < vRows; vRow++) {
    const float vScale = pBackend.aScales[vRow];
    for (uint64_t vColumn = 0; vColumn < vLength; vColumn++) {
      vTo[vRow * vLength + vColumn] = vFrom[vRow * vLength + vColumn] * vScale;
    }
  }
  pBackend.vTotals.vNormsHost++;
  return true;
}

// llama.cpp-xdna: the MUL of such an RMS_NORM by its weights on the host, as the CPU backend computes it
bool fMulOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vWeights = fNormWeightsOf(pNode);
  const ggml_tensor * vNorm    = pNode->src[0] == vWeights ? pNode->src[1] : pNode->src[0];
  if (vWeights == nullptr || vNorm->data == nullptr || vWeights->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor has no memory";
    return false;
  }
  const uint64_t vLength = uint64_t(pNode->ne[0]);
  const uint64_t vRows   = uint64_t(ggml_nrows(pNode));
  const float *  vFrom   = static_cast<const float *>(vNorm->data);
  const float *  vBy     = static_cast<const float *>(vWeights->data);
  float *        vTo     = static_cast<float *>(pNode->data);
  for (uint64_t vRow = 0; vRow < vRows; vRow++) {
    for (uint64_t vColumn = 0; vColumn < vLength; vColumn++) {
      vTo[vRow * vLength + vColumn] = vFrom[vRow * vLength + vColumn] * vBy[vColumn];
    }
  }
  pBackend.vTotals.vNormsHost++;
  return true;
}

// llama.cpp-xdna: a residual ADD on the host, as the CPU backend computes it (a 32-bit float sum for every value):
// one that was not read with the C of its multiplication, as when generating tokens
bool fAddOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (vA->data == nullptr || vB->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor has no memory";
    return false;
  }
  const double       vStart  = fNowMs();
  const uint64_t     vCount  = uint64_t(ggml_nelements(pNode));
  const float *      vFromA  = static_cast<const float *>(vA->data);
  const float *      vFromB  = static_cast<const float *>(vB->data);
  float *            vTo     = static_cast<float *>(pNode->data);
  constexpr uint64_t cChunk  = 16384;
  const uint64_t     vChunks = (vCount + cChunk - 1) / cChunk;
  auto fChunk = [&](uint32_t pChunk) {
    const uint64_t vLast = std::min(vCount, (uint64_t(pChunk) + 1) * cChunk);
    for (uint64_t i = uint64_t(pChunk) * cChunk; i < vLast; i++) {
      vTo[i] = vFromA[i] + vFromB[i];
    }
  };
  if (vChunks > 1) {
    XdnaShared &               vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    vShared.vRunner.mWorkers().mForEach(uint32_t(vChunks), fChunk);
  } else if (vChunks == 1) {
    fChunk(0);
  }
  pBackend.vTotals.vAddsHost++;
  pBackend.vTotals.vAddMs += fNowMs() - vStart;
  return true;
}

// llama.cpp-xdna: the tokens of a recurrence of a batch that the NPU takes: at least 128, and at least 9/16 of the
// groups of 128 it computes (the others are empty tokens, see xdna-gdnp.h). Measured on 2026/11/16 with Qwen3.5 4B
// (32 value heads, 4 commands): 0.78, 1.19, 1.60 and 2.08 ms for 128, 256, 384 and 512 tokens computed, and about
// 4.25 ms for 512 tokens in the CPU, so that fewer go faster in the CPU.
bool fGdnTokensTaken(int64_t pTokens) {
  const int64_t vGroup = 128;
  const int64_t vTaken = (pTokens + vGroup - 1) / vGroup * vGroup;
  return pTokens >= vGroup && pTokens <= 0x7fffff80 && 16 * pTokens >= 9 * vTaken;
}

// llama.cpp-xdna: whether pNode is a GATED_DELTA_NET the backend takes: a batch of tokens like those of Qwen3.5 (heads
// of 128, one gate per head, twice as many value heads as heads of keys, one sequence, K == 1; fGdnTokensTaken), on
// an overlay of the prompts with the recurrence ("gdn 1", xdna-gdnp.h); never emulated.
bool fGdnTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pNode->op != GGML_OP_GATED_DELTA_NET || pShared.vSettings.vEmulate || pShared.vSettings.vNoGdn ||
      pShared.vOverlay.vGdn != 1) {
    return false;
  }
  const ggml_tensor * vQ = pNode->src[0];
  const ggml_tensor * vK = pNode->src[1];
  const ggml_tensor * vV = pNode->src[2];
  const ggml_tensor * vG = pNode->src[3];
  const ggml_tensor * vB = pNode->src[4];
  const ggml_tensor * vS = pNode->src[5];
  if (vQ == nullptr || vK == nullptr || vV == nullptr || vG == nullptr || vB == nullptr || vS == nullptr) {
    return false;
  }
  for (const ggml_tensor * vTensor : {vQ, vK, vV, vG, vB, vS, pNode}) {
    if (vTensor->type != GGML_TYPE_F32) {
      return false;
    }
  }
  const int64_t vKeys   = vQ->ne[1];
  const int64_t vValues = vV->ne[1];
  const int64_t vTokens = vV->ne[2];
  return vQ->ne[0] == 128 && vK->ne[0] == 128 && vV->ne[0] == 128 && vK->ne[1] == vKeys && vKeys % 2 == 0 &&
         vValues >= vKeys && vValues % vKeys == 0 && vValues <= 4 * vKeys && vQ->ne[2] == vTokens &&
         vK->ne[2] == vTokens && fGdnTokensTaken(vTokens) &&
         vQ->ne[3] == 1 && vK->ne[3] == 1 && vV->ne[3] == 1 &&
         vQ->nb[0] == sizeof(float) && vK->nb[0] == sizeof(float) && vV->nb[0] == sizeof(float) && vG->ne[0] == 1 &&
         vG->ne[1] == vValues && vG->ne[2] == vTokens && vG->ne[3] == 1 && ggml_is_contiguous(vG) && vB->ne[0] == 1 &&
         vB->ne[1] == vValues && vB->ne[2] == vTokens && vB->ne[3] == 1 && ggml_is_contiguous(vB) &&
         vS->ne[0] == 128 && vS->ne[1] == 128 && vS->ne[2] == vValues && vS->ne[3] == 1 && ggml_is_contiguous(vS) &&
         ggml_is_contiguous(pNode) && ggml_get_op_params_i32(pNode, 0) == 1;
}

// llama.cpp-xdna: a GET_ROWS the backend takes (fRowsTaken) on the host: its id has to be 0, the only row, which is
// copied as it is
bool fRowsOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vRows = pNode->src[0];
  const ggml_tensor * vIds  = pNode->src[1];
  if (!fRowsTaken(pNode) || pNode->data == nullptr || vRows->data == nullptr || vIds->data == nullptr) {
    pError = "it is not a GET_ROWS of this backend";
    return false;
  }
  if (*static_cast<const int32_t *>(vIds->data) != 0) {
    pError = "it asks for a row that its operand does not have";
    return false;
  }
  if (pNode->data != vRows->data) {
    std::memmove(pNode->data, vRows->data, ggml_nbytes(pNode));
  }
  pBackend.vTotals.vRowsHost++;
  return true;
}

bool fIsUnary(const ggml_tensor * pTensor, enum ggml_unary_op pOp);

// llama.cpp-xdna: whether pNode is a part of the L2 norm of q or k of a layer of linear attention of Qwen3.5 in a batch
// that the backend takes (build_gdn_l2_norm: an RMS_NORM of a view of the SiLU of the convolution, heads of 128, and a
// SCALE of it without a bias), for the batches whose recurrence it takes (fGdnTokensTaken): the recurrence then takes
// the norm into the writing of its inputs (XdnaGdnIo::vQNorms), and the two are not computed; otherwise they are
// computed on the host as the CPU backend computes them (fL2OnHost). llama_context::graph_get_cb gives them to the
// backend, as the scheduler would leave them with the processor. GGML_XDNA_NO_L2 leaves them there.
bool fL2Taken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  const int64_t cDim = cXdnaChainRecurrenceDim;
  if (pShared.vSettings.vNoL2 || pShared.vSettings.vNoGdn || pShared.vSettings.vEmulate || pShared.vOverlay.vGdn != 1 ||
      pNode->type != GGML_TYPE_F32 || pNode->ne[0] != cDim || pNode->ne[3] != 1 || !fGdnTokensTaken(pNode->ne[2]) ||
      !ggml_is_contiguous(pNode)) {
    return false;
  }
  const ggml_tensor * vA = pNode->src[0];
  if (vA == nullptr || vA->type != GGML_TYPE_F32 || !ggml_are_same_shape(vA, pNode)) {
    return false;
  }
  if (pNode->op == GGML_OP_RMS_NORM) {
    return vA->op == GGML_OP_VIEW && fIsUnary(vA->view_src, GGML_UNARY_OP_SILU) && vA->nb[0] == sizeof(float) &&
           vA->nb[1] == cDim * sizeof(float);
  }
  if (pNode->op == GGML_OP_SCALE) {
    float aParams[2];
    memcpy(aParams, pNode->op_params, sizeof(aParams));
    return aParams[1] == 0.0f && vA->op == GGML_OP_RMS_NORM && fL2Taken(pShared, vA);
  }
  return false;
}

// llama.cpp-xdna: whether pNode is an operation of the gate of a layer of linear attention of Qwen3.5 in a batch whose
// recurrence the backend takes (fGdnTokensTaken): g = softplus(alpha + ssm_dt) * ssm_a (the ADD, the SOFTPLUS and the
// MUL) and beta = sigmoid(its projection), a value per value head and token. The backend computes them on the host as
// the CPU backend does (fGateOnHost): with them, the L2 norms and the recurrence are one part of the graph, so that the
// recurrence can take the norms in (fL2Fold). With GGML_XDNA_NO_L2, as the norms, they stay with the CPU backend.
bool fGateTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vNoL2 || pShared.vSettings.vNoGdn || pShared.vSettings.vEmulate || pShared.vOverlay.vGdn != 1 ||
      pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode) || pNode->ne[3] != 1) {
    return false;
  }
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (vA == nullptr || vA->type != GGML_TYPE_F32 || !ggml_is_contiguous(vA) ||
      (vB != nullptr && (vB->type != GGML_TYPE_F32 || !ggml_is_contiguous(vB)))) {
    return false;
  }
  // alpha, g and the values before them: [value heads][tokens]; beta: [1][value heads][tokens]
  const bool vByHeads = pNode->ne[2] == 1 && fGdnTokensTaken(pNode->ne[1]);
  switch (pNode->op) {
    case GGML_OP_ADD:
      return vByHeads && vB != nullptr && ggml_are_same_shape(vA, pNode) &&
             fThroughReshapes(vA)->op == GGML_OP_MUL_MAT && fIsWeight(vB) && ggml_nelements(vB) == pNode->ne[0];
    case GGML_OP_UNARY:
      if (ggml_get_unary_op(pNode) == GGML_UNARY_OP_SOFTPLUS) {
        return vByHeads && ggml_are_same_shape(vA, pNode) && vA->op == GGML_OP_ADD && fGateTaken(pShared, vA);
      }
      return ggml_get_unary_op(pNode) == GGML_UNARY_OP_SIGMOID && pNode->ne[0] == 1 &&
             fGdnTokensTaken(pNode->ne[2]) && ggml_are_same_shape(vA, pNode) &&
             fThroughReshapes(vA)->op == GGML_OP_MUL_MAT;
    case GGML_OP_MUL:
      return vByHeads && vB != nullptr && ggml_are_same_shape(vA, pNode) &&
             fIsUnary(vA, GGML_UNARY_OP_SOFTPLUS) && fGateTaken(pShared, vA) && fIsWeight(vB) &&
             ggml_nelements(vB) == pNode->ne[0];
    default:
      return false;
  }
}

// llama.cpp-xdna: such an operation of the gate (fGateTaken) on the host, value by value as the CPU backend computes it
// (binary-ops.cpp, unary-ops.cpp: a + b, a * b, softplus as op_softplus, sigmoid as op_sigmoid)
bool fGateOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (pNode->data == nullptr || vA->data == nullptr || (vB != nullptr && vB->data == nullptr)) {
    pError = "a tensor has no memory";
    return false;
  }
  const int64_t vCount = ggml_nelements(pNode);
  const float * vFrom  = static_cast<const float *>(vA->data);
  const float * vOther = vB != nullptr ? static_cast<const float *>(vB->data) : nullptr;
  const int64_t vEvery = vB != nullptr ? ggml_nelements(vB) : 1;
  float *       vTo    = static_cast<float *>(pNode->data);
  const bool    vPlus  = pNode->op == GGML_OP_UNARY && ggml_get_unary_op(pNode) == GGML_UNARY_OP_SOFTPLUS;
  // in pieces for the threads: an exponential and a logarithm per value cost some 15 ns, the 16 K values of a batch of
  // 512 some 0.25 ms on one thread (measured on 2026/11/16: on one thread the gates took what the L2 norms saved)
  constexpr int64_t cPiece  = 1024;
  const int64_t     vPieces = (vCount + cPiece - 1) / cPiece;
  auto              fPiece  = [&](uint32_t pPiece) {
    const int64_t vEnd = std::min(vCount, (int64_t(pPiece) + 1) * cPiece);
    for (int64_t n = int64_t(pPiece) * cPiece; n < vEnd; n++) {
      const float vX = vFrom[n];
      switch (pNode->op) {
        case GGML_OP_ADD:
          vTo[n] = vX + vOther[n % vEvery];
          break;
        case GGML_OP_MUL:
          vTo[n] = vX * vOther[n % vEvery];
          break;
        default:
          vTo[n] = vPlus ? ((vX > 20.0f) ? vX : log1pf(expf(vX))) : 1.f / (1.f + expf(-vX));
          break;
      }
    }
  };
  if (vPieces > 1 && pNode->op == GGML_OP_UNARY) {
    XdnaShared &               vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    vShared.vRunner.mWorkers().mForEach(uint32_t(vPieces), fPiece);
  } else {
    for (int64_t p = 0; p < vPieces; p++) {
      fPiece(uint32_t(p));
    }
  }
  pBackend.vTotals.vGateHost++;
  return true;
}

// llama.cpp-xdna: a part of an L2 norm (fL2Taken) on its own, on the host, as the CPU backend computes it: the RMS_NORM
// as x * its scale (fXdnaRmsScales), every head of keys apart, and the SCALE as x * its factor
bool fL2OnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  if (pNode->data == nullptr || vA->data == nullptr) {
    pError = "a tensor has no memory";
    return false;
  }
  const int64_t vLength = pNode->ne[0];
  const int64_t vHeads  = pNode->ne[1];
  const int64_t vTokens = pNode->ne[2];
  float *       vTo     = static_cast<float *>(pNode->data);
  if (pNode->op == GGML_OP_RMS_NORM) {
    float vEps;
    memcpy(&vEps, pNode->op_params, sizeof(float));
    const size_t       vHeadStride  = vA->nb[1] / sizeof(float);
    const size_t       vTokenStride = vA->nb[2] / sizeof(float);
    std::vector<float> aScales(static_cast<size_t>(vTokens));
    for (int64_t h = 0; h < vHeads; h++) {
      const float * vFrom = static_cast<const float *>(vA->data) + size_t(h) * vHeadStride;
      fXdnaRmsScales(vFrom, uint64_t(vTokens), uint64_t(vLength), vTokenStride, vEps, aScales.data());
      for (int64_t t = 0; t < vTokens; t++) {
        for (int64_t i = 0; i < vLength; i++) {
          vTo[(t * vHeads + h) * vLength + i] = vFrom[size_t(t) * vTokenStride + size_t(i)] * aScales[size_t(t)];
        }
      }
    }
  } else {
    float aParams[2];
    memcpy(aParams, pNode->op_params, sizeof(aParams));
    const float * vFrom = static_cast<const float *>(vA->data);
    for (int64_t n = 0; n < ggml_nelements(pNode); n++) {
      vTo[n] = vFrom[n] * aParams[0];
    }
  }
  pBackend.vTotals.vL2Host++;
  return true;
}

// llama.cpp-xdna: whether the recurrence pGdn of a batch, which the backend takes (fGdnTaken), takes the L2 norms of
// its q and k into the writing of its inputs: both an RMS_NORM and a SCALE the backend takes (fL2Taken) in this part
// of the graph (sInPart), used only on the way to it and not outputs of the graph, alike (the same eps and factor).
// pFold then holds the inputs of the norms.
bool fL2Fold(const struct ggml_cgraph * pGraph, const ggml_tensor * pGdn,
             const std::unordered_set<const ggml_tensor *> & sInPart, XdnaBackend::XdnaL2Fold & pFold) {
  const XdnaShared & vShared = fShared();
  float               aEps[2];
  float               aFactor[2];
  const ggml_tensor * aInputs[2];
  for (int vSide = 0; vSide < 2; vSide++) {
    const ggml_tensor * vScale = pGdn->src[vSide];
    if (vScale == nullptr || vScale->op != GGML_OP_SCALE || !fL2Taken(vShared, vScale)) {
      return false;
    }
    const ggml_tensor * vNorm = vScale->src[0];
    if (sInPart.count(vScale) == 0 || sInPart.count(vNorm) == 0 || fUseCount(pGraph, vScale) != 1 ||
        fUseCount(pGraph, vNorm) != 1 || (vScale->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
        (vNorm->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || vNorm->src[0]->data == nullptr) {
      return false;
    }
    memcpy(&aEps[vSide], vNorm->op_params, sizeof(float));
    memcpy(&aFactor[vSide], vScale->op_params, sizeof(float));
    aInputs[vSide] = vNorm->src[0];
  }
  if (aEps[0] != aEps[1] || aFactor[0] != aFactor[1]) {
    return false;
  }
  pFold.vQ      = aInputs[0];
  pFold.vK      = aInputs[1];
  pFold.vEps    = aEps[0];
  pFold.vFactor = aFactor[0];
  return true;
}

int  fNextNode(const struct ggml_cgraph * pGraph, int pIndex);
bool fGatedTaken(const XdnaShared & pShared, const ggml_tensor * pNode);
bool fStateOpTaken(const XdnaShared & pShared, const ggml_tensor * pNode);
bool fStateOpOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError);

// llama.cpp-xdna: a GATED_DELTA_NET the backend takes (fGdnTaken), in one command of the overlay of the prompts: the
// outputs into the result (or their gated norm straight into the A buffers of ssm_out), and the final states into the
// cell of the cache (ggml_gated_delta_net_set_state_in_place, or a following CPY into XDNA_State) or after the outputs
bool fGatedDeltaNet(XdnaBackend & pBackend, const struct ggml_cgraph * pGraph, int pIndex, int & pLast,
                    std::string & pError) {
  pLast                 = pIndex;
  ggml_tensor * pNode   = pGraph->nodes[pIndex];
  XdnaShared &  vShared = fShared();
  if (!fGdnTaken(vShared, pNode) || pNode->data == nullptr) {
    pError = "it is not a recurrence of this backend";
    return false;
  }
  const ggml_tensor * vQ = pNode->src[0];
  const ggml_tensor * vK = pNode->src[1];
  const ggml_tensor * vV = pNode->src[2];
  const ggml_tensor * vG = pNode->src[3];
  const ggml_tensor * vB = pNode->src[4];
  ggml_tensor *       vS = pNode->src[5];
  if (vQ->data == nullptr || vK->data == nullptr || vV->data == nullptr || vG->data == nullptr ||
      vB->data == nullptr || vS->data == nullptr) {
    pError = "a tensor of the recurrence has no memory";
    return false;
  }
  XdnaGdnIo vIo;
  vIo.vQ             = static_cast<const float *>(vQ->data);
  vIo.vQHead         = vQ->nb[1] / sizeof(float);
  vIo.vQToken        = vQ->nb[2] / sizeof(float);
  vIo.vK             = static_cast<const float *>(vK->data);
  vIo.vKHead         = vK->nb[1] / sizeof(float);
  vIo.vKToken        = vK->nb[2] / sizeof(float);
  vIo.vV             = static_cast<const float *>(vV->data);
  vIo.vVHead         = vV->nb[1] / sizeof(float);
  vIo.vVToken        = vV->nb[2] / sizeof(float);
  vIo.vG             = static_cast<const float *>(vG->data);
  vIo.vBeta          = static_cast<const float *>(vB->data);
  vIo.vStateIn       = static_cast<const float *>(vS->data);
  vIo.vOut           = static_cast<float *>(pNode->data);
  vIo.vTokens        = uint32_t(vV->ne[2]);
  vIo.vKeyHeads      = uint32_t(vQ->ne[1]);
  vIo.vValueHeads    = uint32_t(vV->ne[1]);
  vIo.vScale         = 1.0f / std::sqrt(128.0f);
  vIo.vPrepackedKQV  = pBackend.vPrepackedGdn == pNode;
  pBackend.vPrepackedGdn = nullptr;
  const bool   vInPlace = ggml_get_op_params_i32(pNode, 1) != 0;
  const size_t vScores  = size_t(128) * size_t(vV->ne[1]) * size_t(vV->ne[2]);
  vIo.vStateOut = vInPlace ? static_cast<float *>(vS->data) : static_cast<float *>(pNode->data) + vScores;

  const ggml_tensor * vFoldedCpy = nullptr;
  int                 vStateLast = pIndex;
  if (!vInPlace && !vShared.vSettings.vNoGdnFold && !vShared.vSettings.vEmulate) {
    const int vAtCpy = fNextNode(pGraph, pIndex);
    if (vAtCpy < pGraph->n_nodes) {
      const ggml_tensor * vCpy     = pGraph->nodes[vAtCpy];
      const ggml_tensor * vNewView = vCpy->op == GGML_OP_CPY ? vCpy->src[0] : nullptr;
      const ggml_tensor * vDstView = vCpy->op == GGML_OP_CPY ? vCpy->src[1] : nullptr;
      const int64_t       vStateEl = int64_t(128) * int64_t(128) * int64_t(vIo.vValueHeads) * int64_t(vS->ne[3]);
      if (vNewView != nullptr && vDstView != nullptr && fStateOpTaken(vShared, vCpy) &&
          vNewView->op == GGML_OP_VIEW && vNewView->view_src == pNode &&
          vNewView->view_offs == vScores * sizeof(float) && ggml_is_contiguous(vNewView) &&
          ggml_is_contiguous(vDstView) && vDstView->data != nullptr && ggml_nelements(vCpy) == vStateEl &&
          fUseCount(pGraph, vNewView) == 1 && (vNewView->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
          fUseCount(pGraph, vCpy) == 0 && (vCpy->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
        vIo.vStateOut = static_cast<float *>(vDstView->data);
        vFoldedCpy    = vCpy;
        vStateLast    = vAtCpy;
      }
    }
  }

  // llama.cpp-xdna: check whether the gated norm right after this recurrence (past any state bookkeeping ops of the
  // next layer) can be packed straight from vGdnC into the A buffers of ssm_out while the NPU runs the remaining parts
  const ggml_tensor *        vGatedRows    = nullptr;
  const ggml_tensor *        vGatedWeights = nullptr;
  const ggml_tensor *        vGatedZ       = nullptr;
  const ggml_tensor *        vGatedMat     = nullptr;
  float                      vGatedEps     = 0.0f;
  bool                       vGatedSigmoid = false;
  int                        vGatedLast    = vStateLast;
  std::vector<ggml_tensor *> aMidStateOps;
  if (!vShared.vSettings.vNoGdnFold && !vShared.vSettings.vNoGated && !vShared.vSettings.vNoFuse &&
      !vShared.vSettings.vEmulate && vShared.vOverlay.vTileM == 64 && vShared.vOverlay.vTileK == 128 &&
      (pNode->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
    int vScan = fNextNode(pGraph, vStateLast);
    while (vScan < pGraph->n_nodes && fStateOpTaken(vShared, pGraph->nodes[vScan])) {
      ggml_tensor *       vMid    = pGraph->nodes[vScan];
      const ggml_tensor * vMidSrc = vMid->src[0];
      if (vMidSrc == pNode || (vMidSrc != nullptr && vMidSrc->view_src == pNode)) {
        break;
      }
      aMidStateOps.push_back(vMid);
      vScan = fNextNode(pGraph, vScan);
    }
    int  aAt[5]   = {};
    aAt[0]        = vScan;
    bool vChainOk = aAt[0] < pGraph->n_nodes;
    for (int k = 1; vChainOk && k < 5; k++) {
      aAt[k]   = fNextNode(pGraph, aAt[k - 1]);
      vChainOk = aAt[k] < pGraph->n_nodes;
    }
    if (vChainOk) {
      const ggml_tensor * vNorm     = pGraph->nodes[aAt[0]];
      const ggml_tensor * vWeighted = pGraph->nodes[aAt[1]];
      const ggml_tensor * vSiluZ    = pGraph->nodes[aAt[2]];
      const ggml_tensor * vOut      = pGraph->nodes[aAt[3]];
      const ggml_tensor * vMat      = pGraph->nodes[aAt[4]];
      const ggml_tensor * vViewO    = vNorm->op == GGML_OP_RMS_NORM ? vNorm->src[0] : nullptr;
      if (vViewO != nullptr && vViewO->op == GGML_OP_VIEW && vViewO->view_src == pNode && vViewO->view_offs == 0 &&
          fGatedTaken(vShared, vNorm) && vWeighted->op == GGML_OP_MUL && vSiluZ->op == GGML_OP_UNARY &&
          vOut->op == GGML_OP_MUL && vMat->op == GGML_OP_MUL_MAT && fGatedTaken(vShared, vWeighted) &&
          fGatedTaken(vShared, vSiluZ) && fGatedTaken(vShared, vOut) &&
          (vWeighted->src[0] == vNorm || vWeighted->src[1] == vNorm) &&
          ((vOut->src[0] == vWeighted && vOut->src[1] == vSiluZ) ||
           (vOut->src[0] == vSiluZ && vOut->src[1] == vWeighted)) &&
          fUseCount(pGraph, vViewO) == 1 && (vViewO->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
          fUseCount(pGraph, vNorm) == 1 && (vNorm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
          fUseCount(pGraph, vWeighted) == 1 && (vWeighted->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
          fUseCount(pGraph, vSiluZ) == 1 && (vSiluZ->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
        bool vGdnOnlyStateOther = vInPlace ? fUseCount(pGraph, pNode) == 1 :
                                  vFoldedCpy != nullptr ? fUseCount(pGraph, pNode) == 2 : true;
        if (!vInPlace && vFoldedCpy == nullptr) {
          for (int j = 0; vGdnOnlyStateOther && j < pGraph->n_nodes; j++) {
            const ggml_tensor * vConsumer = pGraph->nodes[j];
            if (vConsumer == vViewO) {
              continue;
            }
            for (int s = 0; s < GGML_MAX_SRC; s++) {
              if (vConsumer->src[s] == pNode) {
                vGdnOnlyStateOther = vConsumer->op == GGML_OP_VIEW &&
                                     vConsumer->view_offs >= vScores * sizeof(float);
              }
            }
          }
        }
        const ggml_tensor * vRows = vMat->src[1];
        if (vGdnOnlyStateOther && fThroughReshapes(vRows) == vOut && fMulMatTaken(vShared, vMat) &&
            ggml_is_contiguous(vRows)) {
          bool vRowsAlone = true;
          for (const ggml_tensor * vStep = vRows; vRowsAlone && vStep != nullptr;
               vStep = vStep == vOut ? nullptr : vStep->src[0]) {
            vRowsAlone = fUseCount(pGraph, vStep) == 1 && (vStep->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
          }
          const uint64_t vCount  = uint64_t(vRows->ne[1]) * uint64_t(vRows->ne[2]) * uint64_t(vRows->ne[3]);
          const uint32_t vInputs = uint32_t(vMat->src[0]->ne[0]);
          const ggml_tensor * vWeights = vWeighted->src[0] == vNorm ? vWeighted->src[1] : vWeighted->src[0];
          const ggml_tensor * vZ       = vSiluZ->src[0];
          if (vRowsAlone && vInputs == uint64_t(vOut->ne[0] * vOut->ne[1]) && vCount == uint64_t(vOut->ne[2]) &&
              vCount <= cRowsPerCommand && !fByRows(vShared, vCount, vInputs) && vWeights->data != nullptr &&
              vZ->data != nullptr) {
            vGatedRows    = vRows;
            vGatedWeights = vWeights;
            vGatedZ       = vZ;
            vGatedMat     = vMat;
            memcpy(&vGatedEps, vNorm->op_params, sizeof(float));
            vGatedSigmoid = ggml_get_unary_op(vSiluZ) == GGML_UNARY_OP_SIGMOID;
            vGatedLast    = aAt[3];
          }
        }
      }
    }
  }

  // llama.cpp-xdna: the L2 norms of q and k taken in (fL2Fold)
  const auto    vFold = pBackend.dL2Folds.find(pNode);
  XdnaGemmTimes vTimes;
  XdnaGemmTimes vGatedFlushTimes;
  uint32_t      vGatedPieceCount = 0;
  {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    if (vFold != pBackend.dL2Folds.end()) {
      const XdnaBackend::XdnaL2Fold & vNorms  = vFold->second;
      const uint32_t                  vKeys   = vIo.vKeyHeads;
      const uint32_t                  vTokens = vIo.vTokens;
      vIo.vQ         = static_cast<const float *>(vNorms.vQ->data);
      vIo.vQHead     = vNorms.vQ->nb[1] / sizeof(float);
      vIo.vQToken    = vNorms.vQ->nb[2] / sizeof(float);
      vIo.vK         = static_cast<const float *>(vNorms.vK->data);
      vIo.vKHead     = vNorms.vK->nb[1] / sizeof(float);
      vIo.vKToken    = vNorms.vK->nb[2] / sizeof(float);
      vIo.vNormScale = vNorms.vFactor;
      if (vIo.vPrepackedKQV) {
        // already computed and packed into vGdnB during fConvStrips
      } else if (!vShared.vSettings.vNoGdnFold) {
        vIo.vL2InPack = true;
        vIo.vL2Eps    = vNorms.vEps;
      } else {
        pBackend.aQNorms.resize(size_t(vKeys) * vTokens);
        pBackend.aKNorms.resize(size_t(vKeys) * vTokens);
        const double vStart = fNowMs();
        vShared.vRunner.mWorkers().mForEach(2 * vKeys, [&](uint32_t pTask) {
          const uint32_t vHead = pTask % vKeys;
          const bool     vIsK  = pTask >= vKeys;
          fXdnaRmsScales((vIsK ? vIo.vK : vIo.vQ) + size_t(vHead) * (vIsK ? vIo.vKHead : vIo.vQHead), vTokens,
                         cXdnaChainRecurrenceDim, vIsK ? vIo.vKToken : vIo.vQToken, vNorms.vEps,
                         (vIsK ? pBackend.aKNorms : pBackend.aQNorms).data() + size_t(vHead) * vTokens);
        });
        vIo.vQNorms = pBackend.aQNorms.data();
        vIo.vKNorms = pBackend.aKNorms.data();
        pBackend.vTotals.vGdnPackMs += fNowMs() - vStart;
      }
      pBackend.vTotals.vL2Folded++;
    }

    XdnaGdnGatedPiece aGatedPieces[cXdnaRunnerSlotsA];
    XdnaGdnGated      vGatedSpec;
    if (vGatedMat != nullptr) {
      XdnaMatrix   vTemporary;
      XdnaMatrix * vMatrix = fMatrixOf(vShared, pBackend, {vGatedMat->src[0]}, vTemporary, pError);
      if (vMatrix == nullptr) {
        return false;
      }
      bool vFits = vMatrix != &vTemporary && vMatrix->vRangesK <= cXdnaRunnerSlotsA - cXdnaRunnerFirstReadyA;
      for (uint32_t r = 0; vFits && r < vMatrix->vRangesK; r++) {
        const XdnaPiece & vPiece = vMatrix->aPieces[size_t(r) * vMatrix->vRangesN];
        vFits = vPiece.vFirstInput % 128 == 0 && vPiece.vInputs % 128 == 0;
      }
      if (vFits) {
        fDrainEarlyPacked(vShared, pBackend);
        for (uint32_t r = 0; vFits && r < vMatrix->vRangesK; r++) {
          const XdnaPiece & vPiece = vMatrix->aPieces[size_t(r) * vMatrix->vRangesN];
          if (!vShared.vRunner.mPrepareRowsA(cXdnaRunnerFirstReadyA + r, vIo.vTokens, vPiece.vInputs, pError)) {
            return false;
          }
          uint32_t  vPaddedM = 0;
          uint8_t * vBufA    = vShared.vRunner.mPreparedBufferA(cXdnaRunnerFirstReadyA + r, vPaddedM);
          aGatedPieces[r].vFirstHead = vPiece.vFirstInput / 128;
          aGatedPieces[r].vHeads     = vPiece.vInputs / 128;
          aGatedPieces[r].vPaddedM   = vPaddedM;
          aGatedPieces[r].vBufferA   = vBufA;
        }
        vGatedPieceCount     = vMatrix->vRangesK;
        vGatedSpec.vZ        = static_cast<const float *>(vGatedZ->data);
        vGatedSpec.vStrideZ  = size_t(vIo.vValueHeads) * 128;
        vGatedSpec.vWeights  = static_cast<const float *>(vGatedWeights->data);
        vGatedSpec.vEps      = vGatedEps;
        vGatedSpec.vSigmoid  = vGatedSigmoid;
        vGatedSpec.vSteps3   = fXdnaGemmSteps3(vShared.vOverlay);
        vGatedSpec.vRounding = cRounding;
        vGatedSpec.aPieces   = aGatedPieces;
        vGatedSpec.vPieces   = vGatedPieceCount;
        vIo.vGated           = &vGatedSpec;
      } else if (vMatrix == &vTemporary) {
        fFreeMatrix(vShared, vTemporary);
      }
    }

    if (!vShared.vRunner.mRunGdn(vIo, vTimes, pError)) {
      return false;
    }
    for (uint32_t r = 0; r < vGatedPieceCount; r++) {
      if (!vShared.vRunner.mMarkFlushedRowsA(cXdnaRunnerFirstReadyA + r, vGatedFlushTimes, pError)) {
        return false;
      }
    }
  }
  if (vGatedPieceCount > 0) {
    for (ggml_tensor * vMid : aMidStateOps) {
      if (!fStateOpOnHost(pBackend, vMid, pError)) {
        return false;
      }
    }
    pBackend.vPacked.vActive = true;
    pBackend.vPacked.vRows   = vGatedRows;
    pBackend.vTotals.vGatedPacked++;
    pBackend.vTotals.vGatedFlushMs += vGatedFlushTimes.vFlushMs;
    pBackend.vTotals.vPackMs       += vGatedFlushTimes.vFlushMs;
    pLast = vGatedLast;
  } else if (vFoldedCpy != nullptr) {
    pLast = vStateLast;
  }
  pBackend.vTotals.vGdns++;
  pBackend.vTotals.vGdnNpuMs    += vTimes.vNpuMs;
  pBackend.vTotals.vGdnPackMs   += vTimes.vPackMs;
  pBackend.vTotals.vGdnUnpackMs += vTimes.vUnpackMs;
  return true;
}

// llama.cpp-xdna: whether the queries of an attention come from an RMS norm (the norm of q of Qwen3.5, QK-norm) through
// views and the RoPE, as build_norm leaves them (RMS_NORM and the MUL by its weights). Then their values are of one
// size, and their blocks of 8 in bfp16 are precise enough (KL 0.000042 with Qwen3.5 0.8B in F32); without it, a few
// dimensions can be much larger than the others, and the others lose their bits in the blocks they share with them (KL
// 0.0039 with Granite 4.2 3B): Q and K go in two parts (fFlashAttention).
bool fNormedQueries(const ggml_tensor * pQ) {
  const ggml_tensor * vAt = pQ;
  for (int i = 0; i < 8 && vAt != nullptr; i++) {
    switch (vAt->op) {
      case GGML_OP_VIEW:
      case GGML_OP_PERMUTE:
      case GGML_OP_RESHAPE:
      case GGML_OP_TRANSPOSE:
      case GGML_OP_CONT:
      case GGML_OP_ROPE:
        vAt = vAt->src[0];
        break;
      case GGML_OP_MUL:
        return vAt->src[0] != nullptr && vAt->src[0]->op == GGML_OP_RMS_NORM;
      case GGML_OP_RMS_NORM:
        return true;
      default:
        return false;
    }
  }
  return false;
}

// llama.cpp-xdna: whether pNode is a FLASH_ATTN_EXT the backend takes: the attention of a batch of a prompt (as many
// tokens as a multiplication needs for the backend to take it), with the keys and the values in F16 (the cache of
// llama.cpp), at most as many keys as one multiplication has outputs, optional attention sinks in F32 (one per head),
// and no ALiBi, soft cap or rows of the cache (ggml_flash_attn_ext_rows), on an overlay of the prompts that gives C as
// strips of bfloat16 without splitting B; never emulated. The probe of resolve_fused_ops (src/llama-context.cpp) builds
// it with one token, which stays with the CPU, so flash attention stays on.
bool fAttnTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pNode->op != GGML_OP_FLASH_ATTN_EXT || pShared.vSettings.vEmulate || pShared.vSettings.vNoAttn ||
      pShared.vOverlay.vRowsC != 1 || pShared.vOverlay.vOutput != XdnaGemmOutput::cBfloat16 ||
      pShared.vOverlay.vSplit != 1) {
    return false;
  }
  const ggml_tensor * vQ     = pNode->src[0];
  const ggml_tensor * vK     = pNode->src[1];
  const ggml_tensor * vV     = pNode->src[2];
  const ggml_tensor * vMask  = pNode->src[3];
  const ggml_tensor * vSinks = pNode->src[4];
  if (vQ == nullptr || vK == nullptr || vV == nullptr || pNode->src[5] != nullptr ||
      vQ->type != GGML_TYPE_F32 || vK->type != GGML_TYPE_F16 || vV->type != GGML_TYPE_F16 ||
      pNode->type != GGML_TYPE_F32) {
    return false;
  }
  float vScale   = 0;
  float vMaxBias = 0;
  float vSoftCap = 0;
  std::memcpy(&vScale, reinterpret_cast<const float *>(pNode->op_params) + 0, sizeof(float));
  std::memcpy(&vMaxBias, reinterpret_cast<const float *>(pNode->op_params) + 1, sizeof(float));
  std::memcpy(&vSoftCap, reinterpret_cast<const float *>(pNode->op_params) + 2, sizeof(float));
  if (!std::isfinite(vScale) || vMaxBias != 0.0f || vSoftCap != 0.0f) {
    return false;
  }
  const int64_t vDk     = vQ->ne[0];
  const int64_t vDv     = vV->ne[0];
  const int64_t vTokens = vQ->ne[1];
  const int64_t vHeads  = vQ->ne[2];
  const int64_t vKeys   = vK->ne[1];
  const int64_t vKvHeads = vK->ne[2];
  // With Q and K in two parts (queries not normalised, fNormedQueries) the conversions cost more than the attention of
  // the processor with few keys: measured with Granite 4.2 3B Q5_K_M (2026/11/16, a batch of 512, tokens per second,
  // the processor / the NPU), 988 / 807 with no context, 602 / 603 with 512 tokens before it, 419 / 587 with 1024.
  const bool vSplit = pShared.vSettings.vAttnSplit == 1 || (pShared.vSettings.vAttnSplit == -1 && !fNormedQueries(vQ));
  if (vSplit && vKeys < int64_t(cAttnSplitMinKeys) && pShared.vSettings.vFull == 0) {
    return false;
  }
  if (vMask != nullptr && (vMask->type != GGML_TYPE_F16 || vMask->ne[0] < vKeys || vMask->ne[1] < vTokens ||
                           vMask->ne[2] != 1 || vMask->ne[3] != 1 || vMask->nb[0] != sizeof(uint16_t) ||
                           vMask->nb[1] % sizeof(uint16_t) != 0)) {
    return false;
  }
  if (vSinks != nullptr && (vSinks->type != GGML_TYPE_F32 || vSinks->ne[0] != vHeads || vSinks->ne[1] != 1 ||
                            vSinks->ne[2] != 1 || vSinks->ne[3] != 1 || vSinks->nb[0] != sizeof(float))) {
    return false;
  }
  const uint64_t vMinAttnTokens = pShared.vSettings.vFull >= 2 ? 1u : pShared.vSettings.vMinRows;
  return vDk == vK->ne[0] && vDk > 0 && vDk % 8 == 0 && vDv > 0 && vDv % 8 == 0 && 3 * vDk <= pShared.vMaxInputs &&
         vDv <= pShared.vMaxOutputs && vV->ne[1] == vKeys && vV->ne[2] == vKvHeads && vKvHeads > 0 &&
         vHeads % vKvHeads == 0 && vQ->ne[3] == 1 && vK->ne[3] == 1 && vV->ne[3] == 1 &&
         uint64_t(vTokens) >= vMinAttnTokens && vTokens <= (int64_t(1) << 20) && vKeys >= 1 &&
         vKeys <= int64_t(pShared.vMaxOutputs) &&
         (vKeys + pShared.vMaxInputs - 1) / pShared.vMaxInputs < int64_t(cXdnaRunnerSlotsA) &&
         vQ->nb[0] == sizeof(float) && vK->nb[0] == sizeof(uint16_t) && vV->nb[0] == sizeof(uint16_t) &&
         vQ->nb[1] % sizeof(float) == 0 && vQ->nb[2] % sizeof(float) == 0 && vK->nb[1] % sizeof(uint16_t) == 0 &&
         vK->nb[2] % sizeof(uint16_t) == 0 && vV->nb[1] % sizeof(uint16_t) == 0 &&
         vV->nb[2] % sizeof(uint16_t) == 0 && pNode->ne[0] == vDv && pNode->ne[1] == vHeads &&
         pNode->ne[2] == vTokens && pNode->ne[3] == 1 && ggml_is_contiguous(pNode);
}

// llama.cpp-xdna: the attention of a batch on the overlay of the prompts (see XdnaAttnRows in xdna-gemm.h): for every
// head of keys and values, its keys and its values (in pieces of keys of at most the inputs of a multiplication)
// converted once, and for every block of at most 4096 rows (1024 tokens of 4 heads), S = Q·Kᵀ in one command, P from S
// into the rows of the pieces of the second multiplication, O = P·V in one command a piece, all of them in flight at
// once, and O over the sums of the rows of P into the result. Measured precision (2026/11/16, Qwen3.5 0.8B in F32 with
// a context of 2,048, an emulation in the CPU backend): KL 0.000120 against the CPU, against some 0.0011 that bfp16
// costs in the multiplications by weights.
bool fFlashAttention(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  XdnaShared & vShared = fShared();
  if (!fAttnTaken(vShared, pNode) || pNode->data == nullptr) {
    pError = "it is not an attention of this backend";
    return false;
  }
  const ggml_tensor * vQ     = pNode->src[0];
  const ggml_tensor * vK     = pNode->src[1];
  const ggml_tensor * vV     = pNode->src[2];
  const ggml_tensor * vMask  = pNode->src[3];
  const ggml_tensor * vSinks = pNode->src[4];
  if (vQ->data == nullptr || vK->data == nullptr || vV->data == nullptr ||
      (vMask != nullptr && vMask->data == nullptr) || (vSinks != nullptr && vSinks->data == nullptr)) {
    pError = "a tensor of the attention has no memory";
    return false;
  }
  const double   vStart   = fNowMs();
  const uint32_t vDk      = uint32_t(vQ->ne[0]);
  const uint32_t vDv      = uint32_t(vV->ne[0]);
  const uint32_t vTokens  = uint32_t(vQ->ne[1]);
  const uint32_t vHeads   = uint32_t(vQ->ne[2]);
  const uint32_t vKeys    = uint32_t(vK->ne[1]);
  const uint32_t vKvHeads = uint32_t(vK->ne[2]);
  const uint32_t vGroup   = vHeads / vKvHeads;
  float          vScale   = 1.0f;
  std::memcpy(&vScale, pNode->op_params, sizeof(float));
  const XdnaOverlay & vOverlay = vShared.vOverlay;
  // Q and K in two parts of bfp16 when the queries are not normalised (fNormedQueries), or as GGML_XDNA_ATTN_SPLIT says:
  // the inputs of S = Q·Kᵀ are then three times the dimension of a head
  const bool     vSplit   = vShared.vSettings.vAttnSplit == 1 ||
                            (vShared.vSettings.vAttnSplit == -1 && !fNormedQueries(vQ));
  const uint32_t vInputsS = vSplit ? 3 * vDk : vDk;

  // the pieces of the keys for O = P·V: as few as fit in the inputs of a multiplication, as even as whole tiles of K
  // let them be, each at least the tiles a core takes outside of its loop
  const uint32_t vTileK    = vOverlay.vTileK;
  const uint32_t vMinK     = vTileK * fXdnaGemmTilesKOutside(vOverlay);
  const uint32_t vPieces   = (vKeys + vShared.vMaxInputs - 1) / vShared.vMaxInputs;
  const uint32_t vPerPiece = ((vKeys + vPieces - 1) / vPieces + vTileK - 1) / vTileK * vTileK;
  uint64_t       aFirsts[cXdnaRunnerSlotsC];
  uint32_t       aCounts[cXdnaRunnerSlotsC];
  uint32_t       aValid[cXdnaRunnerSlotsC];
  uint32_t       aSlotsA[cXdnaRunnerSlotsC];
  uint32_t       vUsed = 0;
  for (uint32_t p = 0; p < vPieces && p < cXdnaRunnerSlotsC; p++) {
    const uint64_t vFirst = uint64_t(p) * vPerPiece;
    if (vFirst >= vKeys) {
      break;
    }
    aValid[vUsed]  = uint32_t(std::min<uint64_t>(vPerPiece, vKeys - vFirst));
    aCounts[vUsed] = std::max(vMinK, (aValid[vUsed] + vTileK - 1) / vTileK * vTileK);
    aFirsts[vUsed] = vFirst;
    aSlotsA[vUsed] = 1 + vUsed;  // slot 0 holds Q
    vUsed++;
  }
  if (vUsed == 0 || vUsed * uint64_t(vPerPiece) < vKeys || vUsed + 1 > cXdnaRunnerSlotsA) {
    pError = "the keys do not go into the pieces of a multiplication";
    return false;
  }

  // blocks of tokens of at most the rows of one command (with 5 heads a group, as Granite 4.2, 2048 rows cut a batch of
  // 512 in two, twice the commands)
  const uint32_t vBlockTokens = std::max<uint32_t>(1, cRowsPerCommand / vGroup);
  const uint32_t vBlockRows   = std::min(vTokens, vBlockTokens) * vGroup;
  pBackend.aAttnO.resize(size_t(vBlockRows) * vDv);
  pBackend.aAttnSums.resize(vBlockRows);
  const uint64_t   vStrips     = (uint64_t(vKeys) + vOverlay.vTileN - 1) / vOverlay.vTileN;
  const uint16_t * vMaskData   = vMask != nullptr ? static_cast<const uint16_t *>(vMask->data) : nullptr;
  const uint64_t   vMaskStride = vMask != nullptr ? vMask->nb[1] / sizeof(uint16_t) : 0;
  const float *    vSinksData  = vSinks != nullptr ? static_cast<const float *>(vSinks->data) : nullptr;
  const uint64_t   vKStride    = vK->nb[1] / sizeof(uint16_t);  // halves from a key to the next
  const uint64_t   vVStride    = vV->nb[1] / sizeof(uint16_t);
  float *          vOut        = static_cast<float *>(pNode->data);

  XdnaGemmTimes vTimes;
  double         vPMs      = 0;
  double         vPFlushMs = 0;
  uint64_t       vCommands = 0;
  {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    if (!fOpenRunner(vShared, pError)) {
      return false;
    }
    XdnaRunner & vRunner = vShared.vRunner;
    vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    if (vMask != nullptr) {
      pBackend.aAttnKinds.resize(size_t(vTokens) * vStrips);
      fXdnaAttnMaskKinds(vOverlay, vMaskData, vMaskStride, vTokens, vKeys, pBackend.aAttnKinds.data(),
                          &vRunner.mWorkers());
    }
    // The work goes by items, a block of tokens of a head of keys each, one after the other: while the NPU multiplies P
    // by V of an item, the host converts the keys, the values and the queries of the next one and sends its S = Q·Kᵀ,
    // which the NPU then computes while the host reads O of the item. The values go into two sets of buffers, one for
    // the head of the item and one for the next; the keys into one, free once the S that reads them is done. S goes
    // into the buffer of C after those of the pieces of P·V, so with 4 pieces (more than 12288 keys), and with
    // GGML_XDNA_NO_OVERLAP or GGML_XDNA_NO_ATTN_OVERLAP, every S waits for the item before it.
    struct XdnaAttnItem {
      uint32_t vHead;        // of keys
      uint32_t vFirstToken;
      uint32_t vCount;
    };
    std::vector<XdnaAttnItem> aItems;
    for (uint32_t g = 0; g < vKvHeads; g++) {
      for (uint32_t vFirstToken = 0; vFirstToken < vTokens; vFirstToken += vBlockTokens) {
        aItems.push_back({g, vFirstToken, std::min(vBlockTokens, vTokens - vFirstToken)});
      }
    }
    const bool vOverlap = !vShared.vSettings.vNoOverlap && !vShared.vSettings.vNoAttnOverlap && vUsed < cXdnaRunnerSlotsC;
    // the keys and the values of the head of an item, converted once for all its blocks of tokens
    auto fLoadHead = [&](uint32_t pHead) {
      const uint16_t * vKeysOf =
          reinterpret_cast<const uint16_t *>(static_cast<const char *>(vK->data) + pHead * vK->nb[2]);
      const uint16_t * vValuesOf =
          reinterpret_cast<const uint16_t *>(static_cast<const char *>(vV->data) + pHead * vV->nb[2]);
      if (!vRunner.mLoadAttnB(vKeysOf, vKStride, 1, vKeys, vInputsS, vInputsS, XdnaRounding::cNearest, vShared.vAttnK,
                              vTimes, pError, vSplit ? vDk : 0)) {
        return false;
      }
      for (uint32_t p = 0; p < vUsed; p++) {
        if (!vRunner.mLoadAttnB(vValuesOf + aFirsts[p] * vVStride, 1, vVStride, vDv, aCounts[p], aValid[p],
                                XdnaRounding::cNearest, vShared.aAttnV[pHead % 2][p], vTimes, pError)) {
          return false;
        }
      }
      return true;
    };
    // the queries of an item into the buffer of A 0, and its S = Q·Kᵀ into the buffer of C pSlotC
    auto fSendScores = [&](const XdnaAttnItem & pItem, uint32_t pSlotC, XdnaRunner::XdnaPending & pPending) {
      XdnaAttnRows vRowsOf;
      vRowsOf.vQ = reinterpret_cast<const float *>(static_cast<const char *>(vQ->data) +
                                                   uint64_t(pItem.vFirstToken) * vQ->nb[1] +
                                                   uint64_t(pItem.vHead) * vGroup * vQ->nb[2]);
      vRowsOf.vTokenStride = vQ->nb[1] / sizeof(float);
      vRowsOf.vHeadStride  = vQ->nb[2] / sizeof(float);
      vRowsOf.vTokens      = pItem.vCount;
      vRowsOf.vHeads       = vGroup;
      vRowsOf.vScale       = vScale;
      vRowsOf.vSplit       = vSplit;
      vRowsOf.vDim         = vDk;
      vCommands++;
      return vRunner.mPackAttnQ(0, vRowsOf, vInputsS, XdnaRounding::cNearest, vTimes, pError) &&
             vRunner.mSubmitPacked(vShared.vAttnK, 0, pSlotC, pPending, vTimes, pError);
    };
    const uint32_t            vSlotS = vOverlap ? vUsed : 0;  // the buffer of C of S
    XdnaRunner::XdnaPending vScores;
    if (!fLoadHead(aItems[0].vHead) || !fSendScores(aItems[0], vSlotS, vScores)) {
      return false;
    }
    for (size_t i = 0; i < aItems.size(); i++) {
      const XdnaAttnItem & vItem = aItems[i];
      const uint32_t        vRows = vItem.vCount * vGroup;
      if (!vRunner.mFinishPacked(vScores, nullptr, 0, false, vTimes, pError, false)) {
        return false;
      }
      // P into the rows of the pieces of O = P·V
      for (uint32_t p = 0; p < vUsed; p++) {
        if (!vRunner.mPrepareRowsA(aSlotsA[p], vRows, aCounts[p], pError)) {
          return false;
        }
      }
      XdnaAttnScores vScoresOf;
      vScoresOf.vMask = vMaskData != nullptr ? vMaskData + uint64_t(vItem.vFirstToken) * vMaskStride : nullptr;
      vScoresOf.vMaskStride  = vMaskStride;
      vScoresOf.vKinds       = vMaskData != nullptr ?
                                   pBackend.aAttnKinds.data() + uint64_t(vItem.vFirstToken) * vStrips : nullptr;
      vScoresOf.vKindsStride = uint32_t(vStrips);
      vScoresOf.vSinks       = vSinksData != nullptr ? vSinksData + uint64_t(vItem.vHead) * vGroup : nullptr;
      vScoresOf.vHeads       = vGroup;
      XdnaGemmTimes vPTimes;  // apart: its reading of S is not the reading of O
      if (!vRunner.mPackAttnP(vSlotS, vScoresOf, aSlotsA, aFirsts, vUsed, pBackend.aAttnSums.data(),
                              XdnaRounding::cNearest, vPTimes, pError)) {
        return false;
      }
      vPMs += vPTimes.vUnpackMs;
      vPFlushMs += vPTimes.vFlushMs;
      // O = P·V, every piece in flight at once, added up in their order
      XdnaRunner::XdnaPending aPending[cXdnaRunnerSlotsC];
      for (uint32_t p = 0; p < vUsed; p++) {
        if (!vRunner.mSubmitPacked(vShared.aAttnV[vItem.vHead % 2][p], aSlotsA[p], p, aPending[p], vTimes, pError)) {
          return false;
        }
        vCommands++;
      }
      // the next item while the NPU multiplies these: its keys and values when its head is another, and its S
      const bool vNext = i + 1 < aItems.size();
      if (vNext && vOverlap) {
        if ((aItems[i + 1].vHead != vItem.vHead && !fLoadHead(aItems[i + 1].vHead)) ||
            !fSendScores(aItems[i + 1], vSlotS, vScores)) {
          return false;
        }
      }
      XdnaGemmTarget vTarget;
      vTarget.vData   = pBackend.aAttnO.data();
      vTarget.vStride = vDv;
      vTarget.vFirst  = 0;
      vTarget.vCount  = vDv;
      for (uint32_t p = 0; p < vUsed; p++) {
        if (!vRunner.mFinishPacked(aPending[p], &vTarget, 1, p > 0, vTimes, pError)) {
          return false;
        }
      }
      // O over the sums of the rows of P, into the rows of the result: row r is the token vFirstToken + r / vGroup of
      // the head vHead * vGroup + r % vGroup
      const double vOStart = fNowMs();
      vRunner.mWorkers().mForEach((vRows + 63) / 64, [&](uint32_t pTask) {
        const uint32_t vEnd = std::min(vRows, (pTask + 1) * 64);
        for (uint32_t r = pTask * 64; r < vEnd; r++) {
          const float   vSum  = pBackend.aAttnSums[r];
          const float   vInv  = vSum > 0.0f ? 1.0f / vSum : 0.0f;
          const float * vFrom = pBackend.aAttnO.data() + size_t(r) * vDv;
          float *       vTo   = vOut + (uint64_t(vItem.vFirstToken + r / vGroup) * vHeads +
                                        uint64_t(vItem.vHead) * vGroup + r % vGroup) * vDv;
          for (uint32_t d = 0; d < vDv; d++) {
            vTo[d] = vFrom[d] * vInv;
          }
        }
      });
      vTimes.vUnpackMs += fNowMs() - vOStart;
      if (vNext && !vOverlap) {
        if ((aItems[i + 1].vHead != vItem.vHead && !fLoadHead(aItems[i + 1].vHead)) ||
            !fSendScores(aItems[i + 1], vSlotS, vScores)) {
          return false;
        }
      }
    }
  }
  pBackend.vTotals.vAttns++;
  pBackend.vTotals.vAttnCommands += vCommands;
  pBackend.vTotals.vAttnNpuMs += vTimes.vNpuMs;
  pBackend.vTotals.vAttnPackMs += vTimes.vPackMs;
  pBackend.vTotals.vAttnPMs += vPMs;
  pBackend.vTotals.vAttnUnpackMs += vTimes.vUnpackMs;
  pBackend.vTotals.vAttnFlushMs += vTimes.vFlushMs + vPFlushMs;
  pBackend.vTotals.vAttnMs += fNowMs() - vStart;
  return true;
}

// llama.cpp-xdna: whether the backend takes the layers of linear attention while a token is generated
// (GGML_XDNA_LINEAR): on the NPU, with the chained overlay (version 9, the only one the host takes) and the recurrent
// states in buffers of the driver
bool fLinearOn(const XdnaShared & pShared) {
  const bool vWantLinear =
      pShared.vSettings.vLinear ||
      (pShared.vSettings.vFull >= 2 ||
       (pShared.vSettings.vFull == 1 && !(fMoeActiveForRows(pShared, 1) && !pShared.vSettings.vMoeGemv)));
  return vWantLinear && pShared.vHasChain && !pShared.vSettings.vEmulate && !pShared.vSettings.vNoStateBuffers;
}

// llama.cpp-xdna: whether the memory of pTensor is in a buffer of recurrent states of this backend
bool fInStateBuffer(const ggml_tensor * pTensor) {
  const ggml_tensor * vBase = pTensor;
  while (vBase != nullptr && vBase->view_src != nullptr) {
    vBase = vBase->view_src;
  }
  return vBase != nullptr && vBase->buffer != nullptr && ggml_backend_buffer_get_type(vBase->buffer) == fStateBufferType();
}

bool fLinearTaken(const XdnaShared & pShared, const ggml_tensor * pNode);

// llama.cpp-xdna: state-buffer bookkeeping operations of build_rs and build_recurrent_attn (SCALE by 0 clearing a cell,
// GET_ROWS gathering cells and CPY storing new_state or extra cells into XDNA_State), taken so that recurrent layers do
// not split into CPU subgraphs around the convolution and the recurrence
bool fStateOpTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vNoGdnFold || pShared.vSettings.vNoGdn || pShared.vSettings.vEmulate ||
      pShared.vSettings.vNoStateBuffers || pShared.vOverlay.vGdn != 1 || pNode->type != GGML_TYPE_F32) {
    return false;
  }
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  switch (pNode->op) {
    case GGML_OP_SCALE: {
      if (vA == nullptr || vB != nullptr || vA->type != GGML_TYPE_F32 || !fInStateBuffer(vA) ||
          !ggml_is_contiguous(vA) || !ggml_is_contiguous(pNode) || !ggml_are_same_shape(vA, pNode)) {
        return false;
      }
      float aParams[2] = {};
      memcpy(aParams, pNode->op_params, sizeof(aParams));
      return aParams[0] == 0.0f && aParams[1] == 0.0f;
    }
    case GGML_OP_GET_ROWS:
      return vA != nullptr && vB != nullptr && vA->type == GGML_TYPE_F32 && vB->type == GGML_TYPE_I32 &&
             fInStateBuffer(vA) && vA->nb[0] == sizeof(float) && vA->ne[2] == 1 && vA->ne[3] == 1 &&
             vB->nb[0] == sizeof(int32_t) && vB->ne[1] == 1 && vB->ne[2] == 1 && vB->ne[3] == 1 &&
             pNode->ne[0] == vA->ne[0] && pNode->ne[1] == vB->ne[0] && pNode->ne[2] == 1 && pNode->ne[3] == 1 &&
             ggml_is_contiguous(pNode);
    case GGML_OP_CPY: {
      if (vA == nullptr || vB == nullptr || vA->type != GGML_TYPE_F32 || vB->type != GGML_TYPE_F32 ||
          !fInStateBuffer(vB) || vA->nb[0] != sizeof(float) || vB->nb[0] != sizeof(float) || !ggml_is_contiguous(vA) ||
          vB->ne[3] != 1 || ggml_nelements(vA) != ggml_nelements(pNode) ||
          ggml_nelements(vB) != ggml_nelements(pNode) || vB->ne[0] <= 0 || ggml_nelements(pNode) % vB->ne[0] != 0) {
        return false;
      }
      const ggml_tensor * vRoot = vA->view_src != nullptr ? vA->view_src : vA;
      return vRoot->op != GGML_OP_GATED_DELTA_NET || fGdnTaken(pShared, vRoot) || fLinearTaken(pShared, vRoot);
    }
    default:
      return false;
  }
}

bool fStateOpOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (pNode->data == nullptr || vA == nullptr || vA->data == nullptr || (vB != nullptr && vB->data == nullptr)) {
    pError = "a tensor of a state operation has no memory";
    return false;
  }
  if (pNode->op == GGML_OP_SCALE || pNode->op == GGML_OP_CPY) {
    XdnaShared &                vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.sCleanStates.clear();
  }
  constexpr size_t cParallelBytes = 512u << 10;
  switch (pNode->op) {
    case GGML_OP_SCALE: {
      const size_t  vCount = size_t(ggml_nelements(pNode));
      const float * vSrc   = static_cast<const float *>(vA->data);
      float *       vDst   = static_cast<float *>(pNode->data);
      if (vCount * sizeof(float) >= cParallelBytes) {
        XdnaShared &                vShared = fShared();
        std::lock_guard<std::mutex> vLock(vShared.vMutex);
        vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
        const uint32_t vTasks = std::max(1u, vShared.vRunner.mWorkers().mThreads());
        vShared.vRunner.mWorkers().mForEach(vTasks, [&](uint32_t pTask) {
          const size_t vFirst = vCount * pTask / vTasks;
          const size_t vEnd   = vCount * (pTask + 1) / vTasks;
          for (size_t i = vFirst; i < vEnd; i++) {
            vDst[i] = vSrc[i] * 0.0f;
          }
        });
      } else {
        for (size_t i = 0; i < vCount; i++) {
          vDst[i] = vSrc[i] * 0.0f;
        }
      }
      return true;
    }
    case GGML_OP_GET_ROWS: {
      const int64_t   vRows     = vB->ne[0];
      const size_t    vRowBytes = size_t(vA->ne[0]) * sizeof(float);
      const int32_t * vIndices  = static_cast<const int32_t *>(vB->data);
      const char *    vSrcBase  = static_cast<const char *>(vA->data);
      char *          vDstBase  = static_cast<char *>(pNode->data);
      for (int64_t r = 0; r < vRows; r++) {
        if (vIndices[r] < 0 || vIndices[r] >= vA->ne[1]) {
          pError = "a state row index is out of bounds";
          return false;
        }
      }
      if (size_t(vRows) * vRowBytes >= cParallelBytes) {
        XdnaShared &                vShared = fShared();
        std::lock_guard<std::mutex> vLock(vShared.vMutex);
        vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
        const uint32_t vTasks = std::max(1u, vShared.vRunner.mWorkers().mThreads());
        vShared.vRunner.mWorkers().mForEach(vTasks, [&](uint32_t pTask) {
          for (int64_t r = 0; r < vRows; r++) {
            const size_t vFirst = vRowBytes * pTask / vTasks;
            const size_t vEnd   = vRowBytes * (pTask + 1) / vTasks;
            std::memcpy(vDstBase + size_t(r) * vRowBytes + vFirst,
                        vSrcBase + size_t(vIndices[r]) * vA->nb[1] + vFirst, vEnd - vFirst);
          }
        });
      } else {
        for (int64_t r = 0; r < vRows; r++) {
          std::memcpy(vDstBase + size_t(r) * vRowBytes, vSrcBase + size_t(vIndices[r]) * vA->nb[1], vRowBytes);
        }
      }
      return true;
    }
    case GGML_OP_CPY: {
      const char * vSrc = static_cast<const char *>(vA->data);
      char *       vDst = static_cast<char *>(vB->data);
      if (ggml_is_contiguous(vB)) {
        const size_t vBytes = ggml_nbytes(vB);
        if (vSrc != vDst) {
          if (vBytes >= cParallelBytes) {
            XdnaShared &                vShared = fShared();
            std::lock_guard<std::mutex> vLock(vShared.vMutex);
            vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
            const uint32_t vTasks = std::max(1u, vShared.vRunner.mWorkers().mThreads());
            vShared.vRunner.mWorkers().mForEach(vTasks, [&](uint32_t pTask) {
              const size_t vFirst = vBytes * pTask / vTasks;
              const size_t vEnd   = vBytes * (pTask + 1) / vTasks;
              std::memcpy(vDst + vFirst, vSrc + vFirst, vEnd - vFirst);
            });
          } else {
            std::memcpy(vDst, vSrc, vBytes);
          }
        }
        return true;
      }
      const size_t vRowBytes = size_t(vB->ne[0]) * sizeof(float);
      size_t       vSrcOff   = 0;
      for (int64_t i2 = 0; i2 < vB->ne[2]; i2++) {
        for (int64_t i1 = 0; i1 < vB->ne[1]; i1++) {
          std::memcpy(vDst + size_t(i1) * vB->nb[1] + size_t(i2) * vB->nb[2], vSrc + vSrcOff, vRowBytes);
          vSrcOff += vRowBytes;
        }
      }
      return true;
    }
    default:
      pError = "it is not a state operation";
      return false;
  }
}

bool fShapeIs(const ggml_tensor * pTensor, int64_t p0, int64_t p1, int64_t p2, int64_t p3) {
  return pTensor != nullptr && pTensor->ne[0] == p0 && pTensor->ne[1] == p1 && pTensor->ne[2] == p2 &&
         pTensor->ne[3] == p3;
}

bool fIsUnary(const ggml_tensor * pTensor, enum ggml_unary_op pOp) {
  return pTensor != nullptr && pTensor->op == GGML_OP_UNARY && ggml_get_unary_op(pTensor) == pOp;
}

// llama.cpp-xdna: whether pNode is an operation of the short convolution of Qwen3.5 by tokens in a batch that the
// backend takes (build_conv_state with by_tokens; doc/XDNA2.md, "The short convolution while C is read"): the SSM_CONV
// of the CONCAT of the old states and the tokens (read through a transposed view: the channels of a token one after
// the other), its SILU, and the copy of the last 3 rows of that CONCAT, transposed, into the cell of the cache. Only
// with a batch of GGML_XDNA_MIN_BATCH tokens or more (and 3 or more), 4 taps and channels that are a multiple of 16,
// so that the host computes every value in the lane of a vector, as the CPU backend does, and gets the same floats.
// The backend computes the convolution and its SiLU while it reads the C of the projection (fConvFold), or else on
// the host (fConvOnHost), where the copy always goes.
bool fConvTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vNoConv || pShared.vSettings.vNoFuse || pShared.vSettings.vEmulate ||
      pNode->type != GGML_TYPE_F32) {
    return false;
  }
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (vA == nullptr || vA->type != GGML_TYPE_F32 || (vB != nullptr && vB->type != GGML_TYPE_F32)) {
    return false;
  }
  const auto fBatch = [&](int64_t pTokens) {
    return pTokens >= 3 && uint64_t(pTokens) >= pShared.vSettings.vMinRows;
  };
  switch (pNode->op) {
    case GGML_OP_SSM_CONV: {
      // the input [3 + tokens][channels] through a transposed view (the channels the size of a float apart)
      const int64_t vChannels = pNode->ne[0];
      const int64_t vTokens   = pNode->ne[1];
      return vB != nullptr && fShapeIs(vB, 4, vChannels, 1, 1) && ggml_is_contiguous(vB) &&
             fShapeIs(vA, vTokens + 3, vChannels, 1, 1) && vA->nb[1] == sizeof(float) && vA->nb[0] != sizeof(float) &&
             pNode->ne[2] == 1 && pNode->ne[3] == 1 && ggml_is_contiguous(pNode) && vChannels % 16 == 0 &&
             fBatch(vTokens);
    }
    case GGML_OP_UNARY:
      return ggml_get_unary_op(pNode) == GGML_UNARY_OP_SILU && vA->op == GGML_OP_SSM_CONV &&
             fConvTaken(pShared, vA) && ggml_are_same_shape(vA, pNode) && ggml_is_contiguous(vA) &&
             ggml_is_contiguous(pNode);
    case GGML_OP_CPY: {
      // [3][channels] (the 3 rows of a channel next to each other) from the last rows of such a CONCAT
      const ggml_tensor * vConcat = vA->view_src;
      return vB != nullptr && vConcat != nullptr && vConcat->op == GGML_OP_CONCAT && vConcat->type == GGML_TYPE_F32 &&
             fShapeIs(vA, 3, vConcat->ne[0], 1, 1) && vA->nb[1] == sizeof(float) && ggml_is_contiguous(vB) &&
             ggml_nelements(vB) == ggml_nelements(vA) && vConcat->ne[0] % 16 == 0 && fBatch(vConcat->ne[1] - 3);
    }
    default:
      return false;
  }
}

// llama.cpp-xdna: whether two tensors share memory
bool fOverlap(const ggml_tensor * pA, const ggml_tensor * pB) {
  if (pA == nullptr || pB == nullptr || pA->data == nullptr || pB->data == nullptr) {
    return false;
  }
  const char * vA = static_cast<const char *>(pA->data);
  const char * vB = static_cast<const char *>(pB->data);
  return vA < vB + ggml_nbytes(pB) && vB < vA + ggml_nbytes(pA);
}

// llama.cpp-xdna: whether the short convolution of the CONCAT pGraph->nodes[pConcatIndex], which the multiplication
// pMember of aGroup writes, and its SiLU go with that multiplication, computed while it reads C (fConvStrips of
// xdna-gemm.cpp), and what for (pFold). Only when, in this part of the graph, the operations after the CONCAT up to
// the SiLU are the copy of its states into the cache, the SSM_CONV and the SILU (all of them taken, fConvTaken), and
// nothing else reads the CONCAT or the SSM_CONV; when the multiplication goes in one command of rows and in one piece
// along its inputs, on the overlay of the prompts with C as strips (a row needs the 3 before it, and the whole sums);
// and when the memory of the SILU, which is then written before its turn, is apart from what those nodes write or read
// before it: the other results of the multiplications, the CONCAT and the cell of the cache.
bool fConvFold(const struct ggml_cgraph * pGraph, int pConcatIndex, const std::vector<ggml_tensor *> & aGroup,
               const ggml_tensor * pMember, XdnaConvFold & pFold) {
  const XdnaShared & vShared = fShared();
  const ggml_tensor * vConcat = pGraph->nodes[pConcatIndex];
  const ggml_tensor * vRows   = pMember->src[1];
  if (vShared.vSettings.vNoConv || vShared.vSettings.vNoConvFold || vShared.vOverlay.vRowsC != 1 ||
      vShared.vOverlay.vTileN == 0 ||
      vShared.vOverlay.vTileN > 64 || vConcat->src[0]->ne[1] != 3 || (vConcat->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
      !ggml_is_contiguous(pMember) || vRows == nullptr) {
    return false;
  }
  const ggml_tensor * vCopy = nullptr;
  const ggml_tensor * vConv = nullptr;
  const ggml_tensor * vSilu = nullptr;
  for (int j = pConcatIndex + 1; j < pGraph->n_nodes && vSilu == nullptr; j++) {
    const ggml_tensor * vNode = pGraph->nodes[j];
    if ((vNode->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || fDoesNothing(vNode)) {
      continue;
    }
    if (vNode->op == GGML_OP_CPY && vCopy == nullptr && vConv == nullptr && vNode->src[0]->view_src == vConcat &&
        fConvTaken(vShared, vNode)) {
      vCopy = vNode;
    } else if (vNode->op == GGML_OP_SSM_CONV && vConv == nullptr && vNode->src[0]->view_src == vConcat &&
               fConvTaken(vShared, vNode)) {
      vConv = vNode;
    } else if (vConv != nullptr && vNode->op == GGML_OP_UNARY && vNode->src[0] == vConv &&
               fConvTaken(vShared, vNode)) {
      vSilu = vNode;
    } else {
      return false;
    }
  }
  if (vSilu == nullptr || vSilu->data == nullptr || vConcat->data == nullptr || vConv->src[1]->data == nullptr ||
      vConcat->src[0]->data == nullptr || fUseCount(pGraph, vConv) != 1 ||
      (vConv->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return false;
  }
  const int64_t  vTokens   = vConv->ne[1];
  const int64_t  vChannels = vConv->ne[0];
  const uint64_t vCount    = uint64_t(vRows->ne[1]) * uint64_t(vRows->ne[2]) * uint64_t(vRows->ne[3]);
  const uint64_t vInputs   = uint64_t(pMember->src[0]->ne[0]);
  if (vConcat->ne[0] != vChannels || vConcat->ne[1] != vTokens + 3 || pMember->ne[0] != vChannels ||
      vCount != uint64_t(vTokens) || vCount > cRowsPerCommand || vInputs > vShared.vMaxInputs ||
      fByRows(vShared, vCount, uint32_t(vInputs))) {
    return false;
  }
  // nothing else reads the CONCAT, whose rows between the first 3 and the last 3 are then not written
  for (int j = 0; j < pGraph->n_nodes; j++) {
    const ggml_tensor * vNode = pGraph->nodes[j];
    if (vNode == vCopy || vNode == vConv || fDoesNothing(vNode)) {
      continue;
    }
    for (int s = 0; s < GGML_MAX_SRC; s++) {
      const ggml_tensor * vSource = vNode->src[s];
      if (vSource != nullptr && (vSource == vConcat || vSource->view_src == vConcat)) {
        return false;
      }
    }
  }
  // and the memory of the SILU apart from what is written or read before its turn
  for (const ggml_tensor * vNode : aGroup) {
    if (vNode != pMember && fOverlap(vSilu, vNode)) {
      return false;
    }
  }
  if (fOverlap(vSilu, vConcat) || (vCopy != nullptr && fOverlap(vSilu, vCopy))) {
    return false;
  }
  pFold.vConv     = vConv;
  pFold.vSilu     = vSilu;
  pFold.vConcat   = vConcat;
  pFold.vTokens   = vTokens;
  pFold.vChannels = vChannels;
  return true;
}

// llama.cpp-xdna: the weights of the SSM_CONV pConv by tap, [4][channels], into pBackend.aConvWeights, as the CPU
// backend lays them out in its work buffer
void fConvWeights(XdnaBackend & pBackend, const ggml_tensor * pConv) {
  const int64_t       vChannels = pConv->ne[0];
  const ggml_tensor * vKernel   = pConv->src[1];
  pBackend.aConvWeights.resize(size_t(4 * vChannels));
  for (int64_t c = 0; c < vChannels; c++) {
    const float * vTaps = reinterpret_cast<const float *>(static_cast<const char *>(vKernel->data) + c * vKernel->nb[1]);
    for (int64_t k = 0; k < 4; k++) {
      pBackend.aConvWeights[size_t(k * vChannels + c)] = vTaps[k];
    }
  }
}

// llama.cpp-xdna: for a short convolution that goes with its projection (fConvFold), its weights by tap and its old
// states, [3][channels] (the first operand of the CONCAT, read as fConcatHead reads it), into the backend before the
// multiplication writes anything
void fConvPrepare(XdnaBackend & pBackend, const XdnaConvFold & pFold) {
  const int64_t       vChannels = pFold.vChannels;
  const ggml_tensor * vHead     = pFold.vConcat->src[0];
  fConvWeights(pBackend, pFold.vConv);
  pBackend.aConvHead.resize(size_t(3 * vChannels));
  for (int64_t r = 0; r < 3; r++) {
    for (int64_t c = 0; c < vChannels; c++) {
      pBackend.aConvHead[size_t(r * vChannels + c)] = *reinterpret_cast<const float *>(
          static_cast<const char *>(vHead->data) + c * vHead->nb[0] + r * vHead->nb[1]);
    }
  }
}

// llama.cpp-xdna: an operation of the short convolution of a batch (fConvTaken) that did not go with its projection,
// on the host with the same arithmetic as the CPU backend: the copy of the states, a SILU alone (row by row, as
// ggml_compute_forward_silu_f32), or the SSM_CONV, with the SILU right after it when nothing else reads the
// convolution (as ggml_cpu_try_fuse_ops: the convolution itself is then not written), and pLast is then that SILU
bool fConvOnHost(XdnaBackend & pBackend, const struct ggml_cgraph * pGraph, int pIndex, int & pLast,
                 std::string & pError) {
  ggml_tensor *       vNode = pGraph->nodes[pIndex];
  const ggml_tensor * vA    = vNode->src[0];
  pLast                     = pIndex;
  if (vNode->data == nullptr || vA->data == nullptr || (vNode->src[1] != nullptr && vNode->src[1]->data == nullptr)) {
    pError = "a tensor has no memory";
    return false;
  }
  if (vNode->op == GGML_OP_CPY) {
    // value (k, c) of the source, the row k of the channel c, to k + 3 c of the cell
    const int64_t vChannels = vA->ne[1];
    float *       vTo       = static_cast<float *>(vNode->data);
    const char *  vFrom     = static_cast<const char *>(vA->data);
    for (int64_t c = 0; c < vChannels; c++) {
      for (int64_t k = 0; k < 3; k++) {
        vTo[c * 3 + k] = *reinterpret_cast<const float *>(vFrom + k * vA->nb[0] + c * vA->nb[1]);
      }
    }
    pBackend.vTotals.vConvCopies++;
    return true;
  }
  const auto                  vStart  = std::chrono::steady_clock::now();
  XdnaShared &               vShared = fShared();
  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
  XdnaWorkers & vWorkers  = vShared.vRunner.mWorkers();
  const int64_t  vChannels = vNode->ne[0];
  const int64_t  vTokens   = vNode->ne[1];
  if (vNode->op == GGML_OP_UNARY) {
    vWorkers.mForEach(uint32_t(vTokens), [&](uint32_t pToken) {
      fXdnaSilu(reinterpret_cast<const float *>(static_cast<const char *>(vA->data) + pToken * vA->nb[1]),
                 reinterpret_cast<float *>(static_cast<char *>(vNode->data) + pToken * vNode->nb[1]),
                 uint32_t(vChannels));
    });
    pBackend.vTotals.vConvSilus++;
    pBackend.vTotals.vConvHostMs +=
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vStart).count();
    return true;
  }
  // the SSM_CONV, with its SILU when that is the next operation and nothing else reads the convolution
  ggml_tensor * vOut = vNode;
  int           vNext = pIndex + 1;
  while (vNext < pGraph->n_nodes && ((pGraph->nodes[vNext]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                                     fDoesNothing(pGraph->nodes[vNext]))) {
    vNext++;
  }
  if (vNext < pGraph->n_nodes && pGraph->nodes[vNext]->op == GGML_OP_UNARY && pGraph->nodes[vNext]->src[0] == vNode &&
      fConvTaken(vShared, pGraph->nodes[vNext]) && pGraph->nodes[vNext]->data != nullptr &&
      fUseCount(pGraph, vNode) == 1 && (vNode->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
    vOut  = pGraph->nodes[vNext];
    pLast = vNext;
  }
  fConvWeights(pBackend, vNode);
  const float * vWeights = pBackend.aConvWeights.data();
  const bool    vSilu    = vOut != vNode;
  vWorkers.mForEach(uint32_t(vTokens), [&](uint32_t pToken) {
    // the 4 rows of the window of the token (the 3 before it and its own), the channels next to each other
    const char *  vFrom    = static_cast<const char *>(vA->data) + pToken * vA->nb[0];
    const float * aTaps[4] = {reinterpret_cast<const float *>(vFrom), reinterpret_cast<const float *>(vFrom + vA->nb[0]),
                              reinterpret_cast<const float *>(vFrom + 2 * vA->nb[0]),
                              reinterpret_cast<const float *>(vFrom + 3 * vA->nb[0])};
    fXdnaConvToken(aTaps, vWeights, uint64_t(vChannels),
                    reinterpret_cast<float *>(static_cast<char *>(vOut->data) + pToken * vOut->nb[1]),
                    uint32_t(vChannels), vSilu);
  });
  pBackend.vTotals.vConvsHost++;
  pBackend.vTotals.vConvHostMs +=
      std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - vStart).count();
  return true;
}

// llama.cpp-xdna: whether pNode is an operation of a layer of linear attention of Qwen3.5 4B while a token is
// generated that the backend takes (fLinearOn), by its shape and by what it reads: the copy of the new states of the
// convolution into their cell of the cache, the convolution and its SiLU, the L2 norms of q and k (RMS_NORM and
// SCALE), the gate g (the ADD of the bias ssm_dt, SOFTPLUS and the MUL by ssm_a), beta (SIGMOID), the recurrence of
// one token in its cell of the cache, and the gated norm (RMS_NORM, the MUL by ssm_norm, the SiLU of z and their MUL).
// The recurrence has to update its state in place (ggml_gated_delta_net_set_state_in_place): the probe of llama.cpp
// for the fused recurrence (resolve_fused_ops) builds it on a cache with a cell to clear, so not in place, and would
// take it as not supported if it went to another device than its layer. The backend gives them to the NPU as one
// phase when they make a whole layer (fMatchLinear), and otherwise computes them one by one on the host
// (fLinearOnHost); the copy always goes there.
bool fLinearTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (!fLinearOn(pShared) || pNode->type != GGML_TYPE_F32) {
    return false;
  }
  const int64_t       cDim      = cXdnaChainRecurrenceDim;
  const int64_t       cHeads    = cXdnaChainLinearHeads;
  const int64_t       cKeys     = cXdnaChainLinearKeys;
  const int64_t       cChannels = cXdnaChainLinearChannels;
  const int64_t       cTaps     = cXdnaChainLinearTaps;
  const ggml_tensor * vA        = pNode->src[0];
  const ggml_tensor * vB        = pNode->src[1];
  if (vA == nullptr || vA->type != GGML_TYPE_F32 || (vB != nullptr && vB->type != GGML_TYPE_F32)) {
    return false;
  }
  switch (pNode->op) {
    case GGML_OP_CPY:
      // the last 3 rows of the CONCAT, transposed, into the cell of the cache
      return vB != nullptr && fInStateBuffer(vB) && fShapeIs(vA, cTaps - 1, cChannels, 1, 1) &&
             vA->nb[1] == sizeof(float) && ggml_nelements(vB) == (cTaps - 1) * cChannels && ggml_is_contiguous(vB);
    case GGML_OP_SSM_CONV:
      // of the CONCAT through a transposed view: the 3 old states and the token, channel by channel
      return vB != nullptr && fShapeIs(vA, cTaps, cChannels, 1, 1) && vA->nb[1] == sizeof(float) &&
             fShapeIs(vB, cTaps, cChannels, 1, 1) && ggml_is_contiguous(vB) && fShapeIs(pNode, cChannels, 1, 1, 1) &&
             ggml_is_contiguous(pNode);
    case GGML_OP_UNARY:
      if (!ggml_are_same_shape(vA, pNode) || !ggml_is_contiguous(vA) || !ggml_is_contiguous(pNode)) {
        return false;
      }
      switch (ggml_get_unary_op(pNode)) {
        case GGML_UNARY_OP_SILU:
          // of the convolution, and of z
          return (fShapeIs(pNode, cChannels, 1, 1, 1) && vA->op == GGML_OP_SSM_CONV) ||
                 (fShapeIs(pNode, cDim, cHeads, 1, 1) && fThroughReshapes(vA)->op == GGML_OP_MUL_MAT);
        case GGML_UNARY_OP_SOFTPLUS:
          return fShapeIs(pNode, cHeads, 1, 1, 1) && vA->op == GGML_OP_ADD;
        case GGML_UNARY_OP_SIGMOID:
          return fShapeIs(pNode, 1, cHeads, 1, 1) && fThroughReshapes(vA)->op == GGML_OP_MUL_MAT;
        default:
          return false;
      }
    case GGML_OP_RMS_NORM: {
      // of q or of k, views of the SiLU of the convolution, or of the outputs of the recurrence, a view of it
      const ggml_tensor * vBase = vA->view_src;
      return vA->op == GGML_OP_VIEW && vBase != nullptr && ggml_are_same_shape(vA, pNode) &&
             vA->nb[0] == sizeof(float) && vA->nb[1] == cDim * sizeof(float) && ggml_is_contiguous(pNode) &&
             ((fShapeIs(pNode, cDim, cKeys, 1, 1) && fIsUnary(vBase, GGML_UNARY_OP_SILU)) ||
              (fShapeIs(pNode, cDim, cHeads, 1, 1) && vBase->op == GGML_OP_GATED_DELTA_NET));
    }
    case GGML_OP_SCALE:
      return fShapeIs(pNode, cDim, cKeys, 1, 1) && vA->op == GGML_OP_RMS_NORM && ggml_are_same_shape(vA, pNode) &&
             ggml_is_contiguous(vA) && ggml_is_contiguous(pNode);
    case GGML_OP_ADD:
      // alpha, of its projection, and the bias ssm_dt
      if (vB == nullptr || !fShapeIs(pNode, cHeads, 1, 1, 1) || !ggml_is_contiguous(pNode)) {
        return false;
      }
      for (int vSide = 0; vSide < 2; vSide++) {
        const ggml_tensor * vAlpha = pNode->src[vSide];
        const ggml_tensor * vBias  = pNode->src[1 - vSide];
        if (fThroughReshapes(vAlpha)->op == GGML_OP_MUL_MAT && ggml_nelements(vAlpha) == cHeads &&
            ggml_is_contiguous(vAlpha) && fIsWeight(vBias) && ggml_nelements(vBias) == cHeads &&
            ggml_is_contiguous(vBias)) {
          return true;
        }
      }
      return false;
    case GGML_OP_MUL:
      if (vB == nullptr || !ggml_is_contiguous(vA) || !ggml_is_contiguous(vB) || !ggml_is_contiguous(pNode)) {
        return false;
      }
      for (int vSide = 0; vSide < 2; vSide++) {
        const ggml_tensor * vThis  = pNode->src[vSide];
        const ggml_tensor * vOther = pNode->src[1 - vSide];
        // g: softplus(alpha + ssm_dt) by ssm_a
        if (fShapeIs(pNode, cHeads, 1, 1, 1) && fIsUnary(vThis, GGML_UNARY_OP_SOFTPLUS) && fIsWeight(vOther) &&
            ggml_nelements(vOther) == cHeads) {
          return true;
        }
        // the gated norm: the norm of the outputs by ssm_norm, and that by silu(z)
        if (fShapeIs(pNode, cDim, cHeads, 1, 1) && vThis->op == GGML_OP_RMS_NORM && fIsWeight(vOther) &&
            fShapeIs(vOther, cDim, 1, 1, 1)) {
          return true;
        }
        if (fShapeIs(pNode, cDim, cHeads, 1, 1) && fIsUnary(vThis, GGML_UNARY_OP_SILU) &&
            fShapeIs(vThis, cDim, cHeads, 1, 1) && vOther->op == GGML_OP_MUL && fShapeIs(vOther, cDim, cHeads, 1, 1)) {
          return true;
        }
      }
      return false;
    case GGML_OP_GATED_DELTA_NET: {
      // one token, the state of the 32 heads in its cell of a buffer of states of this backend, updated there
      const ggml_tensor * vK    = pNode->src[1];
      const ggml_tensor * vV    = pNode->src[2];
      const ggml_tensor * vG    = pNode->src[3];
      const ggml_tensor * vBeta = pNode->src[4];
      const ggml_tensor * vS    = pNode->src[5];
      if (vK == nullptr || vV == nullptr || vG == nullptr || vBeta == nullptr || vS == nullptr) {
        return false;
      }
      for (const ggml_tensor * vTensor : {vK, vV, vG, vBeta, vS}) {
        if (vTensor->type != GGML_TYPE_F32) {
          return false;
        }
      }
      return fShapeIs(vA, cDim, cKeys, 1, 1) && fShapeIs(vK, cDim, cKeys, 1, 1) && fShapeIs(vV, cDim, cHeads, 1, 1) &&
             vA->nb[0] == sizeof(float) && vK->nb[0] == sizeof(float) && vV->nb[0] == sizeof(float) &&
             fShapeIs(vG, 1, cHeads, 1, 1) && ggml_is_contiguous(vG) && fShapeIs(vBeta, 1, cHeads, 1, 1) &&
             ggml_is_contiguous(vBeta) && fShapeIs(vS, cDim, cDim, cHeads, 1) && ggml_is_contiguous(vS) &&
             fInStateBuffer(vS) && ggml_get_op_params_i32(pNode, 0) == 1 && ggml_get_op_params_i32(pNode, 1) != 0 &&
             ggml_is_contiguous(pNode);
    }
    default:
      return false;
  }
}

// llama.cpp-xdna: the whole layer of linear attention of the recurrence pGdn (fLinearTaken) when all its operations
// from the convolution to the gated norm are in the part of the graph pGraph (dIndexOf: their indexes in it), as
// build_layer_attn_linear makes them, none of them but the last is an output of the graph and their results have no
// other use than the next of them. False otherwise, with nothing done.
bool fMatchLinear(const struct ggml_cgraph * pGraph, const ggml_tensor * pGdn,
                  const std::unordered_map<const ggml_tensor *, int> & dIndexOf, XdnaLinearBlock & pBlock) {
  const XdnaShared & vShared   = fShared();
  const int64_t       cDim      = cXdnaChainRecurrenceDim;
  const int64_t       cHeads    = cXdnaChainLinearHeads;
  const int64_t       cChannels = cXdnaChainLinearChannels;
  const int64_t       cTaps     = cXdnaChainLinearTaps;
  const size_t        cKeyBytes = size_t(cXdnaChainLinearKeys) * size_t(cDim) * sizeof(float);
  XdnaLinearBlock    vBlock;
  vBlock.vGdn = pGdn;
  // q and k: the SCALE of the RMS_NORM of a view of the SiLU of the convolution; v: a view of it
  vBlock.vScaleQ = pGdn->src[0];
  vBlock.vScaleK = pGdn->src[1];
  if (vBlock.vScaleQ->op != GGML_OP_SCALE || vBlock.vScaleK->op != GGML_OP_SCALE) {
    return false;
  }
  vBlock.vNormQ = vBlock.vScaleQ->src[0];
  vBlock.vNormK = vBlock.vScaleK->src[0];
  if (vBlock.vNormQ->op != GGML_OP_RMS_NORM || vBlock.vNormK->op != GGML_OP_RMS_NORM) {
    return false;
  }
  const ggml_tensor * vViewQ = vBlock.vNormQ->src[0];
  const ggml_tensor * vViewK = vBlock.vNormK->src[0];
  const ggml_tensor * vViewV = pGdn->src[2];
  vBlock.vSilu               = vViewQ->view_src;
  if (vViewQ->op != GGML_OP_VIEW || vViewK->op != GGML_OP_VIEW || vViewV->op != GGML_OP_VIEW ||
      vBlock.vSilu == nullptr || vViewK->view_src != vBlock.vSilu || vViewV->view_src != vBlock.vSilu ||
      vViewQ->view_offs != 0 || vViewK->view_offs != cKeyBytes || vViewV->view_offs != 2 * cKeyBytes ||
      !fIsUnary(vBlock.vSilu, GGML_UNARY_OP_SILU) || vBlock.vSilu->src[0]->op != GGML_OP_SSM_CONV) {
    return false;
  }
  vBlock.vConv = vBlock.vSilu->src[0];
  // the input of the convolution: a transposed view of the whole CONCAT
  const ggml_tensor * vInput  = vBlock.vConv->src[0];
  const ggml_tensor * vConcat = vInput->view_src;
  if (vInput->op != GGML_OP_TRANSPOSE || vConcat == nullptr || vInput->view_offs != 0 || vConcat->data == nullptr ||
      !fShapeIs(vConcat, cChannels, cTaps, 1, 1) || !ggml_is_contiguous(vConcat)) {
    return false;
  }
  vBlock.vInput  = static_cast<const float *>(vConcat->data);
  vBlock.vKernel = vBlock.vConv->src[1];
  // the L2 norms: an RMS_NORM with eps / 128 and the SCALE by 1 / sqrt(128), without a bias, the two alike
  float aScaleQ[2];
  float aScaleK[2];
  float vEpsQ;
  float vEpsK;
  memcpy(aScaleQ, vBlock.vScaleQ->op_params, sizeof(aScaleQ));
  memcpy(aScaleK, vBlock.vScaleK->op_params, sizeof(aScaleK));
  memcpy(&vEpsQ, vBlock.vNormQ->op_params, sizeof(float));
  memcpy(&vEpsK, vBlock.vNormK->op_params, sizeof(float));
  if (aScaleQ[0] != aScaleK[0] || aScaleQ[1] != 0.0f || aScaleK[1] != 0.0f ||
      std::fabs(aScaleQ[0] * std::sqrt(float(cDim)) - 1.0f) > 1e-6f || vEpsQ != vEpsK || !(vEpsQ >= 0.0f)) {
    return false;
  }
  vBlock.vEpsL2 = vEpsQ * float(cDim);
  // g: reshapes of the MUL of the SOFTPLUS of the ADD of alpha and ssm_dt by ssm_a
  vBlock.vGate = fThroughReshapes(pGdn->src[3]);
  if (vBlock.vGate->op != GGML_OP_MUL) {
    return false;
  }
  for (int vSide = 0; vSide < 2; vSide++) {
    if (fIsUnary(vBlock.vGate->src[vSide], GGML_UNARY_OP_SOFTPLUS) && fIsWeight(vBlock.vGate->src[1 - vSide])) {
      vBlock.vSoftplus = vBlock.vGate->src[vSide];
      vBlock.vA        = static_cast<const float *>(vBlock.vGate->src[1 - vSide]->data);
    }
  }
  if (vBlock.vSoftplus == nullptr || vBlock.vSoftplus->src[0]->op != GGML_OP_ADD) {
    return false;
  }
  vBlock.vBiased = vBlock.vSoftplus->src[0];
  for (int vSide = 0; vSide < 2; vSide++) {
    const ggml_tensor * vAlpha = vBlock.vBiased->src[vSide];
    const ggml_tensor * vBias  = vBlock.vBiased->src[1 - vSide];
    if (fThroughReshapes(vAlpha)->op == GGML_OP_MUL_MAT && fIsWeight(vBias)) {
      vBlock.vAlpha = static_cast<const float *>(vAlpha->data);
      vBlock.vDt    = static_cast<const float *>(vBias->data);
    }
  }
  // beta: the SIGMOID of its projection
  vBlock.vBeta = pGdn->src[4];
  if (!fIsUnary(vBlock.vBeta, GGML_UNARY_OP_SIGMOID) || vBlock.vAlpha == nullptr) {
    return false;
  }
  vBlock.vBetaIn = static_cast<const float *>(vBlock.vBeta->src[0]->data);
  // the gated norm, after the recurrence in the part: the RMS_NORM of the view of its outputs, by ssm_norm, by silu(z)
  const auto vAtGdn = dIndexOf.find(pGdn);
  if (vAtGdn == dIndexOf.end()) {
    return false;
  }
  for (int i = vAtGdn->second + 1; i < pGraph->n_nodes && vBlock.vOut == nullptr; i++) {
    const ggml_tensor * vNode = pGraph->nodes[i];
    if (vBlock.vNormO == nullptr) {
      if (vNode->op == GGML_OP_RMS_NORM && vNode->src[0]->op == GGML_OP_VIEW && vNode->src[0]->view_src == pGdn &&
          vNode->src[0]->view_offs == 0 && fShapeIs(vNode, cDim, cHeads, 1, 1)) {
        vBlock.vNormO = vNode;
      }
    } else if (vBlock.vWeighted == nullptr) {
      for (int vSide = 0; vSide < 2 && vNode->op == GGML_OP_MUL; vSide++) {
        if (vNode->src[vSide] == vBlock.vNormO && fIsWeight(vNode->src[1 - vSide]) &&
            fShapeIs(vNode->src[1 - vSide], cDim, 1, 1, 1)) {
          vBlock.vWeighted = vNode;
          vBlock.vNorm     = vNode->src[1 - vSide];
        }
      }
    } else {
      for (int vSide = 0; vSide < 2 && vNode->op == GGML_OP_MUL; vSide++) {
        const ggml_tensor * vOther = vNode->src[1 - vSide];
        if (vNode->src[vSide] == vBlock.vWeighted && fIsUnary(vOther, GGML_UNARY_OP_SILU) &&
            fShapeIs(vOther, cDim, cHeads, 1, 1) && fThroughReshapes(vOther->src[0])->op == GGML_OP_MUL_MAT &&
            ggml_is_contiguous(vOther->src[0])) {
          vBlock.vSiluZ = vOther;
          vBlock.vOut   = vNode;
          vBlock.vZ     = static_cast<const float *>(vOther->src[0]->data);
        }
      }
    }
  }
  if (vBlock.vOut == nullptr || vBlock.vOut->data == nullptr || !ggml_is_contiguous(vBlock.vOut)) {
    return false;
  }
  memcpy(&vBlock.vEpsGated, vBlock.vNormO->op_params, sizeof(float));
  if (!(vBlock.vEpsGated >= 0.0f) || vBlock.vZ == nullptr || vBlock.vBetaIn == nullptr || vBlock.vDt == nullptr ||
      vBlock.vA == nullptr || vBlock.vKernel->data == nullptr || vBlock.vNorm->data == nullptr ||
      pGdn->src[5]->data == nullptr) {
    return false;
  }
  // every operation in the part, its result used only by the next of them (the views of the SiLU, the outputs of the
  // recurrence and g through their views and reshapes) and not an output of the graph
  const struct {
    const ggml_tensor * vNode;
    int                 vUses;
  } aSteps[] = {
    {vBlock.vConv, 1},     {vBlock.vSilu, 3},     {vBlock.vNormQ, 1},   {vBlock.vScaleQ, 1},
    {vBlock.vNormK, 1},    {vBlock.vScaleK, 1},   {vBlock.vBiased, 1},  {vBlock.vSoftplus, 1},
    {vBlock.vGate, 1},     {vBlock.vBeta, 1},     {pGdn, 1},            {vBlock.vNormO, 1},
    {vBlock.vWeighted, 1}, {vBlock.vSiluZ, 1},    {vBlock.vOut, -1},
  };
  for (const auto & vStep : aSteps) {
    if (dIndexOf.find(vStep.vNode) == dIndexOf.end() || (vStep.vNode->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        !fLinearTaken(vShared, vStep.vNode) ||
        (vStep.vUses >= 0 && (fUseCount(pGraph, vStep.vNode) != vStep.vUses ||
                              (vStep.vNode->flags & GGML_TENSOR_FLAG_OUTPUT) != 0))) {
      return false;
    }
  }
  const ggml_tensor * aViews[] = {vViewQ, vViewK, vViewV, pGdn->src[3], vBlock.vNormO->src[0]};
  for (const ggml_tensor * vView : aViews) {
    for (const ggml_tensor * vStep = vView; vStep != nullptr && fDoesNothing(vStep); vStep = vStep->src[0]) {
      if (fUseCount(pGraph, vStep) != 1 || (vStep->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return false;
      }
    }
  }
  pBlock = vBlock;
  return true;
}

// llama.cpp-xdna: the fixed objects of a layer of linear attention, from the weights of its convolution and of its
// gated norm, kept by where they are; with the mutex of pShared held
const XdnaChainWeights * fLinearFixedOf(XdnaShared & pShared, const ggml_tensor * pKernel, const ggml_tensor * pNorm,
                                         std::string & pError) {
  const std::pair<const void *, const void *> vKey(pKernel->data, pNorm->data);
  const auto                                  vFound = pShared.dLinear.find(vKey);
  if (vFound != pShared.dLinear.end()) {
    return &vFound->second;
  }
  XdnaChainWeights vFixed;
  if (!pShared.vRunner.mLoadChainLinear(static_cast<const float *>(pKernel->data),
                                         static_cast<const float *>(pNorm->data), vFixed, pError)) {
    return nullptr;
  }
  return &pShared.dLinear.emplace(vKey, vFixed).first->second;
}

// llama.cpp-xdna: a whole layer of linear attention (fMatchLinear) in one command of one phase of the chained overlay
// (XdnaChainRecurrence::vLinear): e^g and beta on the host, as the CPU backend computes them; the convolution, the
// L2 norms, the recurrence on the state in its cell of the cache and the gated norm on the NPU; its outputs into the
// result of the last MUL of the gated norm. The other operations of the layer are not computed.
bool fLinearAttention(XdnaBackend & pBackend, const XdnaLinearBlock & pBlock, std::string & pError) {
  XdnaShared &  vShared = fShared();
  const double   vStart  = fNowMs();
  const uint32_t cHeads  = cXdnaChainLinearHeads;
  // the ADD, the SOFTPLUS (as unary-ops.cpp), the MUL and the exp of the recurrence; the SIGMOID
  float aDecay[cXdnaChainLinearHeads];
  float aBeta[cXdnaChainLinearHeads];
  for (uint32_t h = 0; h < cHeads; h++) {
    const float vBiased = pBlock.vAlpha[h] + pBlock.vDt[h];
    const float vSoft   = vBiased > 20.0f ? vBiased : log1pf(expf(vBiased));
    aDecay[h]           = expf(vSoft * pBlock.vA[h]);
    aBeta[h]            = 1.0f / (1.0f + expf(-pBlock.vBetaIn[h]));
  }
  XdnaChainLinearIo vInputs;
  vInputs.vInput    = pBlock.vInput;
  vInputs.vStride   = cXdnaChainLinearChannels;
  vInputs.vZ        = pBlock.vZ;
  vInputs.vDecay    = aDecay;
  vInputs.vBeta     = aBeta;
  vInputs.vEpsL2    = pBlock.vEpsL2;
  vInputs.vEpsGated = pBlock.vEpsGated;
  XdnaChainIo vIo;
  vIo.vLinear  = &vInputs;
  vIo.vOutputs = static_cast<float *>(pBlock.vOut->data);
  vIo.vScale   = 1.0f;
  const ggml_tensor *         vState = pBlock.vGdn->src[5];
  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
  const XdnaChainWeights * vFixed = fLinearFixedOf(vShared, pBlock.vKernel, pBlock.vNorm, pError);
  if (vFixed == nullptr) {
    return false;
  }
  // the buffer of the driver with the cell of the states, and where the cell is in it; the last object of the last
  // head reads 7 rows past it, which the buffer has (fStateBufferTypeAllocSize)
  auto vAt = vShared.dStateBuffers.upper_bound(vState->data);
  if (vAt == vShared.dStateBuffers.begin()) {
    pError = "the recurrent states are not in a buffer of the driver";
    return false;
  }
  --vAt;
  const XdnaBuffer & vBuffer = vAt->second;
  const uint64_t      vOffset =
      uint64_t(static_cast<const char *>(vState->data) - static_cast<const char *>(vBuffer.vMap));
  if (vOffset + ggml_nbytes(vState) + 7 * 512 > vBuffer.vSize) {
    pError = "the recurrent states are not in a buffer of the driver";
    return false;
  }
  // The cell out of the cache of the processor: the NPU reads and writes it directly, and the processor may have
  // written it (a recurrence of a prompt, a copy of a state). Measured on 2026/11/16: 0.017 ms for the 2 MB of a cell
  // that is not in the cache. When the NPU itself last updated the cell and no host op touched it since, skip flushing.
  double vFlushMs = 0.0;
  if (vShared.sCleanStates.count(vState->data) == 0) {
    const double vFlushStart = fNowMs();
    fXdnaFlushCache(vState->data, 1, ggml_nbytes(vState), ggml_nbytes(vState));
    vFlushMs = fNowMs() - vFlushStart;
  }
  XdnaChainPhase vPhase;
  vPhase.vRecurrence.vHeads       = cHeads;
  vPhase.vRecurrence.vLinear      = 1;
  vPhase.vRecurrence.vState       = 0;
  vPhase.vRecurrence.vStateOffset = vOffset;
  vPhase.vMatrix                  = 0;
  const XdnaBuffer *       aStates[1]  = {&vBuffer};
  const XdnaChainWeights * aWeights[1] = {vFixed};
  XdnaGemmTimes            vTimes;
  if (!vShared.vRunner.mRunChain(&vPhase, &vIo, 1, aWeights, 1, vTimes, pError, aStates, 1)) {
    return false;
  }
  if (vShared.vSettings.vLinear || vShared.vSettings.vFull >= 1) {
    vShared.sCleanStates.insert(vState->data);
  } else {
    vShared.sCleanStates.erase(vState->data);
  }
  pBackend.vTotals.vCommands++;
  pBackend.vTotals.vCommandsRows++;
  pBackend.vTotals.vLinears++;
  pBackend.vTotals.vLinearNpuMs += vTimes.vNpuMs;
  pBackend.vTotals.vPlanMs      += vTimes.vPlanMs;
  pBackend.vTotals.vPackMs      += vTimes.vPackMs;
  pBackend.vTotals.vNpuMs       += vTimes.vNpuMs;
  pBackend.vTotals.vNpuRowsMs   += vTimes.vNpuMs;
  pBackend.vTotals.vUnpackMs    += vTimes.vUnpackMs;
  pBackend.vTotals.vFlushMs     += vTimes.vFlushMs + vFlushMs;
  pBackend.vTotals.vLinearMs    += fNowMs() - vStart;
  pBackend.vTotals.vTotalMs     += fNowMs() - vStart;
  return true;
}

// llama.cpp-xdna: an operation the backend takes for a layer of linear attention (fLinearTaken) that does not go to
// the NPU with the others (fMatchLinear), on the host as the CPU backend computes it, but for the order of some sums
// and its approximations of the exponential; the copy of the new states of the convolution into the cache always
// goes this way
bool fLinearOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  for (int i = 0; i < GGML_MAX_SRC; i++) {
    if (pNode->src[i] != nullptr && pNode->src[i]->data == nullptr) {
      pError = "a tensor has no memory";
      return false;
    }
  }
  if (pNode->data == nullptr) {
    pError = "a tensor has no memory";
    return false;
  }
  if (pNode->op == GGML_OP_CPY || pNode->op == GGML_OP_GATED_DELTA_NET) {
    XdnaShared &                vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.sCleanStates.clear();
  }
  pBackend.vTotals.vLinearHost++;
  const ggml_tensor * vA  = pNode->src[0];
  const ggml_tensor * vB  = pNode->src[1];
  float *             vTo = static_cast<float *>(pNode->data);
  auto                fAt = [](const ggml_tensor * pTensor, int64_t p0, int64_t p1, int64_t p2, int64_t p3) {
    return reinterpret_cast<const float *>(static_cast<const char *>(pTensor->data) + p0 * pTensor->nb[0] +
                                           p1 * pTensor->nb[1] + p2 * pTensor->nb[2] + p3 * pTensor->nb[3]);
  };
  const int64_t vCount = ggml_nelements(pNode);
  switch (pNode->op) {
    case GGML_OP_CPY: {
      // into the destination one value after the other, in the order of the source: row by row of the CONCAT
      const int64_t vColumns = vA->ne[0];
      for (int64_t i0 = 0; i0 < vColumns; i0++) {
        const float * vRow = fAt(vA, i0, 0, 0, 0);
        for (int64_t i1 = 0; i1 < vA->ne[1]; i1++) {
          vTo[i1 * vColumns + i0] = vRow[i1];
        }
      }
      return true;
    }
    case GGML_OP_SSM_CONV:
      // result[i, t] = the sum over k of x[t + k, i] * w[k, i], in that order
      for (int64_t t = 0; t < pNode->ne[1]; t++) {
        for (int64_t i = 0; i < pNode->ne[0]; i++) {
          float vSum = 0.0f;
          for (int64_t k = 0; k < vB->ne[0]; k++) {
            vSum += *fAt(vA, t + k, i, 0, 0) * *fAt(vB, k, i, 0, 0);
          }
          vTo[t * pNode->ne[0] + i] = vSum;
        }
      }
      return true;
    case GGML_OP_UNARY: {
      const float *           vFrom = static_cast<const float *>(vA->data);
      const enum ggml_unary_op vOp  = ggml_get_unary_op(pNode);
      for (int64_t i = 0; i < vCount; i++) {
        const float vX = vFrom[i];
        if (vOp == GGML_UNARY_OP_SILU) {
          vTo[i] = vX / (1.0f + expf(-vX));
        } else if (vOp == GGML_UNARY_OP_SOFTPLUS) {
          vTo[i] = vX > 20.0f ? vX : log1pf(expf(vX));
        } else {
          vTo[i] = 1.0f / (1.0f + expf(-vX));
        }
      }
      return true;
    }
    case GGML_OP_RMS_NORM: {
      float vEps;
      memcpy(&vEps, pNode->op_params, sizeof(float));
      const int64_t vLength = pNode->ne[0];
      for (int64_t vRow = 0; vRow < ggml_nrows(pNode); vRow++) {
        const float * vFrom = fAt(vA, 0, vRow % vA->ne[1], (vRow / vA->ne[1]) % vA->ne[2],
                                  vRow / (vA->ne[1] * vA->ne[2]));
        double        vSum  = 0;
        for (int64_t i = 0; i < vLength; i++) {
          vSum += double(vFrom[i] * vFrom[i]);
        }
        const float vMean  = float(vSum / double(vLength));
        const float vScale = 1.0f / sqrtf(vMean + vEps);
        for (int64_t i = 0; i < vLength; i++) {
          vTo[vRow * vLength + i] = vFrom[i] * vScale;
        }
      }
      return true;
    }
    case GGML_OP_SCALE: {
      float aParams[2];
      memcpy(aParams, pNode->op_params, sizeof(aParams));
      const float * vFrom = static_cast<const float *>(vA->data);
      for (int64_t i = 0; i < vCount; i++) {
        vTo[i] = aParams[1] == 0.0f ? vFrom[i] * aParams[0] : vFrom[i] * aParams[0] + aParams[1];
      }
      return true;
    }
    case GGML_OP_ADD:
    case GGML_OP_MUL: {
      // the second operand repeats over the first, as ggml_add and ggml_mul allow; both one value after the other
      const float * vFromA = static_cast<const float *>(vA->data);
      const float * vFromB = static_cast<const float *>(vB->data);
      int64_t       n      = 0;
      for (int64_t i3 = 0; i3 < pNode->ne[3]; i3++) {
        for (int64_t i2 = 0; i2 < pNode->ne[2]; i2++) {
          for (int64_t i1 = 0; i1 < pNode->ne[1]; i1++) {
            const int64_t vRowB = ((i3 % vB->ne[3]) * vB->ne[2] + i2 % vB->ne[2]) * vB->ne[1] + i1 % vB->ne[1];
            for (int64_t i0 = 0; i0 < pNode->ne[0]; i0++, n++) {
              const float vOther = vFromB[vRowB * vB->ne[0] + i0 % vB->ne[0]];
              vTo[n]             = pNode->op == GGML_OP_ADD ? vFromA[n] + vOther : vFromA[n] * vOther;
            }
          }
        }
      }
      return true;
    }
    case GGML_OP_GATED_DELTA_NET: {
      // one token, the state updated in its cell (fLinearTaken): row by row of every head, as the CPU backend does it
      const ggml_tensor * vQ      = pNode->src[0];
      const ggml_tensor * vK      = pNode->src[1];
      const ggml_tensor * vV      = pNode->src[2];
      const float *       vG      = static_cast<const float *>(pNode->src[3]->data);
      const float *       vBeta   = static_cast<const float *>(pNode->src[4]->data);
      float *             vState  = static_cast<float *>(pNode->src[5]->data);
      const int64_t       vDim    = vV->ne[0];
      const float         vScale  = 1.0f / sqrtf(float(vDim));
      for (int64_t h = 0; h < vV->ne[1]; h++) {
        const float * vQh    = fAt(vQ, 0, h % vQ->ne[1], 0, 0);
        const float * vKh    = fAt(vK, 0, h % vK->ne[1], 0, 0);
        const float * vVh    = fAt(vV, 0, h, 0, 0);
        const float   vDecay = expf(vG[h]);
        float *       vHead  = vState + h * vDim * vDim;
        for (int64_t j = 0; j < vDim; j++) {
          float * vRow = vHead + j * vDim;
          float   vDot = 0.0f;
          for (int64_t i = 0; i < vDim; i++) {
            vRow[i] *= vDecay;
            vDot += vRow[i] * vKh[i];
          }
          const float vDelta = (vVh[j] - vDot) * vBeta[h];
          float       vOut   = 0.0f;
          for (int64_t i = 0; i < vDim; i++) {
            vRow[i] += vKh[i] * vDelta;
            vOut += vRow[i] * vQh[i];
          }
          vTo[h * vDim + j] = vOut * vScale;
        }
      }
      return true;
    }
    default:
      pError = "it is not an operation of a layer of linear attention";
      return false;
  }
}

// The gate and the up of a feed-forward layer (pNodes, two multiplications of the same rows) and the SwiGLU of their
// results (pGlu), which nothing else uses: on the NPU as two commands, one per matrix (fRangesOfMembers cuts the
// matrix of the two there), whose C the host reads together, writing silu(gate) * up into the result of pGlu without
// writing the two results first (measured on 2026/11/15 with Qwen3.5 4B: of 137 MB of the memory of the host per
// layer and batch of 512, between reading the two results, the SwiGLU of the CPU backend and converting its result,
// to 56). pFused is false, with nothing done, when the matrix is not cut that way, the rows are few enough for the
// overlay for generating tokens, or the backend emulates: the caller then multiplies as usual and computes the SwiGLU
// on its own.
//
// llama.cpp-xdna: pDown is the multiplication of the rows of the SwiGLU right after it (ffn_down), when nothing else
// reads them, or null. The SwiGLU then goes straight into the rows of pDown, converted piece by piece along its inputs
// into the buffers of A from cXdnaRunnerFirstReadyA on (XdnaRunner::mUnpackSwigluRows, the same bytes), and is not
// written as floats; fMulMat takes those rows (XdnaBackend::vPacked). Measured on 2026/11/16 with Qwen3.5 4B, a
// batch of 512: 44 ms reading the gates and the ups with their SwiGLU and 34 ms converting the rows of ffn_down from
// it. GGML_XDNA_NO_PACK_GLU writes the SwiGLU as before.
bool fMulMatGlu(XdnaBackend & pBackend, ggml_tensor * const * pNodes, ggml_tensor * pGlu, const ggml_tensor * pDown,
                bool & pFused, std::string & pError) {
  pFused                      = false;
  XdnaShared &       vShared = fShared();
  const ggml_tensor * vRows   = pNodes[0]->src[1];
  if (vShared.vSettings.vEmulate || pNodes[1]->src[1] != vRows || !fMulMatTaken(vShared, pNodes[0]) ||
      !fMulMatTaken(vShared, pNodes[1]) || vRows->data == nullptr || pGlu->data == nullptr ||
      pNodes[0]->src[0]->ne[1] != pNodes[1]->src[0]->ne[1]) {
    return true;
  }
  const uint32_t vInputs = uint32_t(pNodes[0]->src[0]->ne[0]);
  const uint64_t vCount  = uint64_t(vRows->ne[1]) * uint64_t(vRows->ne[2]) * uint64_t(vRows->ne[3]);
  if (fByRows(vShared, vCount, vInputs) || uint64_t(ggml_nrows(pGlu)) != vCount) {
    return true;
  }
  const std::vector<const ggml_tensor *> aWeights = {pNodes[0]->src[0], pNodes[1]->src[0]};
  const uint32_t                         vGate    = pGlu->src[0] == pNodes[0] ? 0 : 1;  // which one is the gate
  const uint32_t                         vOutputs = uint32_t(aWeights[0]->ne[1]);
  const double                           vStart   = fNowMs();

  // the rows, one after the other; with an RMS_NORM that goes with them, the rows it reads, normalised as they are
  // converted (llama.cpp-xdna)
  const float * vA      = static_cast<const float *>(vRows->data);
  const bool    vNormed = pBackend.vNormed.vActive && vRows == pBackend.vNormed.vRows;
  if (vNormed) {
    vA = pBackend.vNormed.vX;
  } else if (!ggml_is_contiguous(vRows)) {
    pBackend.aRowsA.resize(vCount * vInputs);
    for (uint64_t vRow = 0; vRow < vCount; vRow++) {
      std::memcpy(pBackend.aRowsA.data() + vRow * vInputs, fRowOf(vRows, vRow), vInputs * sizeof(float));
    }
    vA = pBackend.aRowsA.data();
  }

  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
  XdnaMatrix   vTemporary;
  XdnaMatrix * vMatrix = fMatrixOf(vShared, pBackend, aWeights, vTemporary, pError);
  if (vMatrix == nullptr) {
    return false;
  }
  // the matrix cut exactly at the end of the first member: in two pieces, or in four (the two halves of each member,
  // fRangesForOverlap), each half of the outputs read while the NPU multiplies the next one
  const uint32_t vHalves = vMatrix->vRangesN == 4 ? 2 : 1;
  bool           vCut    = vMatrix->vRangesK == 1 && (vMatrix->vRangesN == 2 || vMatrix->vRangesN == 4);
  for (uint32_t vMember = 0; vCut && vMember < 2; vMember++) {
    uint32_t vNext = vMember * vOutputs;
    for (uint32_t vHalf = 0; vCut && vHalf < vHalves; vHalf++) {
      const XdnaPiece & vPiece = vMatrix->aPieces[vMember * vHalves + vHalf];
      vCut  = vPiece.vFirstOutput == vNext && vPiece.vOutputs > 0;
      vNext = vPiece.vFirstOutput + vPiece.vOutputs;
    }
    vCut = vCut && vNext == (vMember + 1) * vOutputs;
  }
  // llama.cpp-xdna: the pieces along the inputs of pDown that the SwiGLU goes into, one buffer of A each, if every half
  // of the gate gives whole tiles of them
  uint32_t    aSlots[cXdnaRunnerSlotsA];
  uint64_t    aFirsts[cXdnaRunnerSlotsA];
  uint32_t    aCounts[cXdnaRunnerSlotsA];
  uint32_t    vPieces   = 0;
  XdnaMatrix * vDownKept = nullptr;
  if (pDown != nullptr && vCut && vCount <= cRowsPerCommand && !vShared.vSettings.vNoPackGlu) {
    XdnaMatrix   vDownTemporary;
    XdnaMatrix * vDown = fMatrixOf(vShared, pBackend, {pDown->src[0]}, vDownTemporary, pError, 1);
    if (vDown == nullptr) {
      if (vMatrix == &vTemporary) {
        fFreeMatrix(vShared, vTemporary);
      }
      return false;
    }
    if (vDown != &vDownTemporary && vDown->vInputs == vOutputs &&
        vDown->vRangesK <= cXdnaRunnerSlotsA - cXdnaRunnerFirstReadyA) {
      for (uint32_t r = 0; r < vDown->vRangesK; r++) {
        const XdnaPiece & vPiece = vDown->aPieces[size_t(r) * vDown->vRangesN];
        aSlots[r]                 = cXdnaRunnerFirstReadyA + r;
        aFirsts[r]                = vPiece.vFirstInput;
        aCounts[r]                = vPiece.vInputs;
      }
      vPieces = vDown->vRangesK;
      for (uint32_t vHalf = 0; vPieces > 0 && vHalf < vHalves; vHalf++) {
        const XdnaPiece & vGatePiece = vMatrix->aPieces[vGate * vHalves + vHalf];
        if (!vShared.vRunner.mSwigluRowsFit(uint32_t(vCount), vGatePiece.vFirstOutput - vGate * vOutputs,
                                            vGatePiece.vOutputs, aFirsts, aCounts, vPieces)) {
          vPieces = 0;
        }
      }
      if (vPieces >= 2 && vHalves == 2 && vDown->vRangesN == 1 && !vShared.vSettings.vNoOverlap) {
        vDownKept = vDown;
      }
    }
    if (vDown == &vDownTemporary) {
      fFreeMatrix(vShared, vDownTemporary);
    }
  }
  bool                    vOk = true;
  XdnaRunner::XdnaPending vEarlySent;
  uint32_t                vEarlyHandle = 0;
  if (vCut) {
    for (uint64_t vFirst = 0; vOk && vFirst < vCount; vFirst += cRowsPerCommand) {
      const uint32_t vRowsNow = uint32_t(std::min<uint64_t>(cRowsPerCommand, vCount - vFirst));
      XdnaGemmTimes vTimes;
      XdnaRowNorm vNorm;
      vNorm.vScales  = vNormed ? pBackend.aScales.data() + vFirst : nullptr;
      vNorm.vWeights = pBackend.vNormed.vWeights;
      vOk = vShared.vRunner.mPackRows(vA + vFirst * vInputs, vRowsNow, vInputs, vInputs, cRounding, vTimes, pError, 0,
                                      vNormed ? &vNorm : nullptr);
      for (uint32_t r = 0; vOk && r < vPieces; r++) {
        vOk = vShared.vRunner.mPrepareRowsA(aSlots[r], vRowsNow, aCounts[r], pError);
      }
      // Every half: its gate and its up go one right after the other, without waiting in between (2026/11/15), into
      // two buffers of C (0 and 1, then 2 and 3), where C stays for the SwiGLU; the SwiGLU of the half before is read
      // meanwhile. Whatever was sent is waited for, in order, even after an error.
      uint32_t vReady           = 0;  // halves whose C are in their buffers and not read yet: 0 or 1
      uint32_t vReadyHalf       = 0;
      uint32_t vFirstReadyPiece = 0;
      auto     fReadHalf        = [&](uint32_t pHalf) -> bool {
        const XdnaPiece & vGatePiece = vMatrix->aPieces[vGate * vHalves + pHalf];
        const uint32_t     vColumn    = vGatePiece.vFirstOutput - vGate * vOutputs;
        const uint32_t     vSlot      = 2 * (pHalf % 2);
        if (vPieces > 0) {
          return vShared.vRunner.mUnpackSwigluRows(vSlot, vSlot + 1, aSlots + vFirstReadyPiece,
                                                   aFirsts + vFirstReadyPiece, vPieces - vFirstReadyPiece, cRounding,
                                                   vTimes, pError, vColumn, vOutputs);
        }
        XdnaGemmTarget    vTarget;
        vTarget.vData   = static_cast<float *>(pGlu->data) + vFirst * vOutputs + vColumn;
        vTarget.vStride = vOutputs;
        vTarget.vFirst  = 0;
        vTarget.vCount  = vGatePiece.vOutputs;
        return vShared.vRunner.mUnpackSwiglu(vSlot, vSlot + 1, vTarget, vTimes, pError, vColumn, vOutputs);
      };
      for (uint32_t vHalf = 0; vOk && vHalf < vHalves; vHalf++) {
        const uint32_t            vSlot = 2 * (vHalf % 2);
        XdnaRunner::XdnaPending vGateSent;
        XdnaRunner::XdnaPending vUpSent;
        vOk = vShared.vRunner.mSubmitPacked(vMatrix->aPieces[vGate * vHalves + vHalf].vWeights, 0, vSlot, vGateSent,
                                            vTimes, pError) &&
              vShared.vRunner.mSubmitPacked(vMatrix->aPieces[(1 - vGate) * vHalves + vHalf].vWeights, 0, vSlot + 1,
                                            vUpSent, vTimes, pError);
        if (vOk && vReady > 0) {
          vOk    = fReadHalf(vReadyHalf);
          vReady = 0;
        }
        std::string vError;
        if (vGateSent.vValid && !vShared.vRunner.mFinishPacked(vGateSent, nullptr, 0, false, vTimes, vError, false) &&
            vOk) {
          vOk    = false;
          pError = vError;
        }
        if (vUpSent.vValid && !vShared.vRunner.mFinishPacked(vUpSent, nullptr, 0, false, vTimes, vError, false) &&
            vOk) {
          vOk    = false;
          pError = vError;
        }
        vReady     = 1;
        vReadyHalf = vHalf;
        pBackend.vTotals.vCommands += 2;
      }
      // llama.cpp-xdna: when the first half already completed piece 0 of ffn_down, flush and submit piece 0 into C[0]
      // right away so the NPU multiplies ffn_down piece 0 while the host unpacks the second half (from C[2..3])
      if (vOk && vDownKept != nullptr && vFirst == 0 &&
          aFirsts[0] + aCounts[0] <= vMatrix->aPieces[vGate * vHalves].vOutputs) {
        vOk = vShared.vRunner.mReadyRowsA(aSlots[0], vTimes, pError) &&
              vShared.vRunner.mSubmitPacked(vDownKept->aPieces[0].vWeights, aSlots[0], 0, vEarlySent, vTimes, pError);
        if (vOk) {
          vFirstReadyPiece = 1;
          vEarlyHandle     = vDownKept->aPieces[0].vWeights.vBuffer.vHandle;
        }
      }
      if (vOk && vReady > 0) {
        vOk = fReadHalf(vReadyHalf);
      }
      for (uint32_t r = vFirstReadyPiece; vOk && r < vPieces; r++) {
        vOk = vShared.vRunner.mReadyRowsA(aSlots[r], vTimes, pError);
      }
      if (!vOk && vEarlySent.vValid) {
        XdnaGemmTimes vIgnoredTimes;
        std::string   vIgnoredError;
        vShared.vRunner.mFinishPacked(vEarlySent, nullptr, 0, false, vIgnoredTimes, vIgnoredError, false);
      }
      pBackend.vTotals.vPlanMs   += vTimes.vPlanMs;
      pBackend.vTotals.vPackMs   += vTimes.vPackMs;
      pBackend.vTotals.vNpuMs    += vTimes.vNpuMs;
      pBackend.vTotals.vUnpackMs += vTimes.vUnpackMs;
      pBackend.vTotals.vFlushMs  += vTimes.vFlushMs;
      pBackend.vTotals.vSwigluUnpackMs += vTimes.vUnpackMs;
    }
  }
  if (vMatrix == &vTemporary) {
    fFreeMatrix(vShared, vTemporary);
  }
  if (!vOk || !vCut) {
    return vOk;
  }
  pBackend.vTotals.vOperations += 2;
  pBackend.vTotals.vRows       += vCount * 2;
  pBackend.vTotals.vGlusFused++;
  if (vPieces > 0) {
    fDrainEarlyPacked(vShared, pBackend);
    pBackend.vPacked.vActive            = true;
    pBackend.vPacked.vRows              = pGlu;
    pBackend.vPacked.vEarlySent         = vEarlySent;
    pBackend.vPacked.vEarlyWeightHandle = vEarlyHandle;
    pBackend.vTotals.vGlusPacked++;
  }
  pBackend.vTotals.vTotalMs    += fNowMs() - vStart;
  pFused = true;
  return true;
}

const char * fBackendName(ggml_backend_t pBackend) {
  GGML_UNUSED(pBackend);
  return cName;
}

void fBackendFree(ggml_backend_t pBackend) {
  XdnaShared &  vShared  = fShared();
  XdnaBackend * vContext = static_cast<XdnaBackend *>(pBackend->context);
  const XdnaTotals & vTotals = vContext->vTotals;
  if (vTotals.vOperations > 0) {
    GGML_LOG_INFO("%s: %llu multiplications of %llu rows in %llu commands, %.1f ms: %s %.1f ms, converting rows "
                  "%.1f ms, reading results %.1f ms (%.1f ms of them flushing the cache), instruction streams "
                  "%.1f ms; %llu matrices of weights converted (%.1f million weights) in %.1f ms; %llu commands "
                  "of the overlay for generating tokens, %.1f ms of NPU; SwiGLU %llu times while reading C, %llu "
                  "on the host (%.1f ms); residual ADD %llu times while reading C, %llu on the host (%.1f ms); CONCAT "
                  "%llu times written by the multiplication, %llu on the host; RMS_NORM %llu times with the "
                  "conversion of the rows, %llu RMS_NORM or MUL on the host; %llu feed-forward blocks in one command "
                  "of the chained overlay, %.1f ms of NPU, %llu of them with the norm and the multiplications of the "
                  "next layer (%llu with the output norm and the logits); GET_ROWS of one row %llu times on the host; "
                  "%llu commands more for rows that did not fit in one\n",
                  cName, (unsigned long long) vTotals.vOperations, (unsigned long long) vTotals.vRows,
                  (unsigned long long) vTotals.vCommands, vTotals.vTotalMs,
                  vShared.vSettings.vEmulate ? "emulation" : "NPU", vTotals.vNpuMs, vTotals.vPackMs,
                  vTotals.vUnpackMs, vTotals.vFlushMs, vTotals.vPlanMs, (unsigned long long) vTotals.vMatrices,
                  double(vTotals.vWeights) / 1e6, vTotals.vWeightsMs, (unsigned long long) vTotals.vCommandsRows,
                  vTotals.vNpuRowsMs, (unsigned long long) vTotals.vGlusFused, (unsigned long long) vTotals.vGlusHost,
                  vTotals.vGluMs, (unsigned long long) vTotals.vAddsFused, (unsigned long long) vTotals.vAddsHost,
                  vTotals.vAddMs, (unsigned long long) vTotals.vConcatsFused,
                  (unsigned long long) vTotals.vConcatsHost, (unsigned long long) vTotals.vNormsFused,
                  (unsigned long long) vTotals.vNormsHost, (unsigned long long) vTotals.vChainsFfn,
                  vTotals.vChainsFfnMs, (unsigned long long) vTotals.vChainsNext,
                  (unsigned long long) vTotals.vChainsLogits, (unsigned long long) vTotals.vRowsHost,
                  (unsigned long long) vTotals.vSplitRows);
  }
  if (vTotals.vDowns > 0 || vTotals.vSwigluUnpackMs > 0) {
    GGML_LOG_INFO("%s: of them, %llu multiplications of the rows of a SwiGLU: converting their rows %.1f ms, NPU %.1f "
                  "ms, reading their results %.1f ms; reading the gates and the ups with their SwiGLU %.1f ms, %llu "
                  "times into the rows of the next multiplication\n",
                  cName, (unsigned long long) vTotals.vDowns, vTotals.vDownPackMs, vTotals.vDownNpuMs,
                  vTotals.vDownUnpackMs, vTotals.vSwigluUnpackMs, (unsigned long long) vTotals.vGlusPacked);
  }
  if (vTotals.vGatedPacked > 0 || vTotals.vGatedHost > 0) {
    GGML_LOG_INFO("%s: %llu gated norms of a batch converted into the rows of the next multiplication, %.1f ms "
                  "(converting %.1f ms, flushing the cache %.1f ms); %llu of their operations on the host\n",
                  cName, (unsigned long long) vTotals.vGatedPacked, vTotals.vGatedMs, vTotals.vGatedPackMs,
                  vTotals.vGatedFlushMs, (unsigned long long) vTotals.vGatedHost);
  }
  if (vTotals.vLinears > 0 || vTotals.vLinearHost > 0) {
    GGML_LOG_INFO("%s: %llu layers of linear attention in one phase of the chained overlay, %.1f ms of NPU, %.1f ms in "
                  "the backend; %llu of their operations on the host\n",
                  cName, (unsigned long long) vTotals.vLinears, vTotals.vLinearNpuMs, vTotals.vLinearMs,
                  (unsigned long long) vTotals.vLinearHost);
  }
  if (vTotals.vGdns > 0) {
    GGML_LOG_INFO("%s: %llu recurrences of a batch of tokens on the NPU: %.1f ms of NPU, writing their inputs %.1f ms, "
                  "reading their results %.1f ms; %llu of them with the L2 norms of their q and k, %llu parts of L2 "
                  "norms and %llu operations of gates on the host\n",
                  cName, (unsigned long long) vTotals.vGdns, vTotals.vGdnNpuMs, vTotals.vGdnPackMs,
                  vTotals.vGdnUnpackMs, (unsigned long long) vTotals.vL2Folded, (unsigned long long) vTotals.vL2Host,
                  (unsigned long long) vTotals.vGateHost);
  }
  if (vTotals.vConvsFused + vTotals.vConvsHost + vTotals.vConvSilus + vTotals.vConvCopies > 0) {
    // llama.cpp-xdna
    GGML_LOG_INFO("%s: %llu short convolutions of a batch with their SiLU computed while the C of their projection was "
                  "read, %llu on the host and %llu SiLUs of them alone (%.1f ms); %llu copies of their states into the "
                  "cache\n",
                  cName, (unsigned long long) vTotals.vConvsFused, (unsigned long long) vTotals.vConvsHost,
                  (unsigned long long) vTotals.vConvSilus, vTotals.vConvHostMs,
                  (unsigned long long) vTotals.vConvCopies);
  }
  if (vShared.vIsOpen && vShared.vRunner.mOverlaySwitches() > 0) {
    // llama.cpp-xdna: of the runner, which the backends of the process share
    GGML_LOG_INFO("%s: the NPU went from one overlay to the other %llu times since it was opened, some 2.5 ms each\n",
                  cName, (unsigned long long) vShared.vRunner.mOverlaySwitches());
  }
  if (vTotals.vAttns > 0) {
    GGML_LOG_INFO("%s: %llu attentions of a batch on the NPU in %llu commands, %.1f ms: NPU %.1f ms, converting Q, K "
                  "and V %.1f ms, P from S %.1f ms, reading O %.1f ms, flushing the cache %.1f ms\n",
                  cName, (unsigned long long) vTotals.vAttns, (unsigned long long) vTotals.vAttnCommands,
                  vTotals.vAttnMs, vTotals.vAttnNpuMs, vTotals.vAttnPackMs, vTotals.vAttnPMs, vTotals.vAttnUnpackMs,
                  vTotals.vAttnFlushMs);
  }
  if (vTotals.vMoeOps > 0 || vTotals.vMoeTailHost > 0 || vTotals.vMoeDenseCmds > 0 || vTotals.vChainsShexp > 0) {
    GGML_LOG_INFO("%s: %llu expert projections (MUL_MAT_ID) in %llu commands, %.1f ms: NPU %.1f ms, converting rows "
                  "%.1f ms, reading results %.1f ms, instruction streams %.1f ms; %llu expert matrices converted in "
                  "%.1f ms; %llu gate+up+SwiGLU fused, %llu weighted reductions fused, %llu shared experts folded "
                  "(%llu tail ops on the host); %llu dense MoE-overlay commands (%.1f ms of NPU); %llu chained "
                  "shared-expert blocks (%.1f ms of NPU)\n",
                  cName, (unsigned long long) vTotals.vMoeOps, (unsigned long long) vTotals.vMoeCommands,
                  vTotals.vMoeMs, vTotals.vMoeNpuMs, vTotals.vMoePackMs, vTotals.vMoeUnpackMs, vTotals.vMoePlanMs,
                  (unsigned long long) vTotals.vMoeMatrices, vTotals.vMoeWeightsMs,
                  (unsigned long long) vTotals.vMoeFusedGlu, (unsigned long long) vTotals.vMoeTailFused,
                  (unsigned long long) vTotals.vMoeFusedShexp, (unsigned long long) vTotals.vMoeTailHost,
                  (unsigned long long) vTotals.vMoeDenseCmds, vTotals.vMoeDenseNpuMs,
                  (unsigned long long) vTotals.vChainsShexp, vTotals.vChainsShexpMs);
  }
  if (vTotals.vFullOpsHost > 0) {
    GGML_LOG_INFO("%s: %llu full-subgraph glue operations executed inside the XDNA backend in %.1f ms\n", cName,
                  (unsigned long long) vTotals.vFullOpsHost, vTotals.vFullOpsMs);
  }
  {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vBackends--;
    fCloseIfIdle(vShared);
  }
  delete vContext;
  delete pBackend;
}

// llama.cpp-xdna: the next node of pGraph after pIndex that computes something, or pGraph->n_nodes
int fNextNode(const struct ggml_cgraph * pGraph, int pIndex) {
  int vNext = pIndex + 1;
  while (vNext < pGraph->n_nodes &&
         ((pGraph->nodes[vNext]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || fDoesNothing(pGraph->nodes[vNext]))) {
    vNext++;
  }
  return vNext;
}

// llama.cpp-xdna: whether pOperand is pProduct, or a reshape of it or a GET_ROWS the backend takes (its only row),
// whose every step only leads to pOperand
bool fIsResult(const struct ggml_cgraph * pGraph, const ggml_tensor * pOperand, const ggml_tensor * pProduct) {
  if (fThroughIdentities(pOperand) != pProduct) {
    return false;
  }
  for (const ggml_tensor * vStep = pOperand; vStep != pProduct; vStep = vStep->src[0]) {
    if (fUseCount(pGraph, vStep) != 1 || (vStep->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
      return false;
    }
  }
  return true;
}

// llama.cpp-xdna: the multiplication of aGroup whose result goes into the CONCAT pConcat as its second operand (or
// through reshapes of it) and nothing else, when the backend takes that CONCAT: its result can then be written
// straight into the CONCAT, after the rows of the first operand. Null otherwise.
const ggml_tensor * fConcatMember(const struct ggml_cgraph * pGraph, const ggml_tensor * pConcat,
                                  const std::vector<ggml_tensor *> & aGroup) {
  if (pConcat == nullptr || pConcat->op != GGML_OP_CONCAT || !fConcatTaken(fShared(), pConcat) ||
      pConcat->data == nullptr || pConcat->src[0]->data == nullptr) {
    return nullptr;
  }
  const ggml_tensor * vProduct = fThroughReshapes(pConcat->src[1]);
  bool vAlone = fUseCount(pGraph, vProduct) == 1 && (vProduct->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                ggml_is_contiguous(vProduct);
  for (const ggml_tensor * vStep = pConcat->src[1]; vStep != vProduct; vStep = vStep->src[0]) {
    vAlone = vAlone && fUseCount(pGraph, vStep) == 1 && (vStep->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
  }
  for (const ggml_tensor * vInGroup : aGroup) {
    if (vAlone && vInGroup == vProduct) {
      return vProduct;
    }
  }
  return nullptr;
}

// llama.cpp-xdna: generating a token with the chained overlay, the feed-forward block of a layer from the
// multiplication at pIndex on, in one command of three phases (XdnaRunner::mRunChain):
//
//   P = W0 x (one row), A1 = P + R, N = RMS_NORM(A1), M = N * w, G = Wg x M and U = Wu x M (in any order),
//   S = SwiGLU(G, U), D = Wd x S, A2 = D + A1
//
// phase 0: W0 with X in floats and the residual R (h = A1, which the cores keep); phase 1: the norm of h, Wg and Wu as
// two segments and their SwiGLU; phase 2: Wd with X from the SwiGLU, plus h. Only A1 and A2 are written: the other
// results have to have no other use (the counts of uses are those of the whole graph).
//
// When the next layer starts right after it (step E of the plan, doc/XDNA2.md):
//
//   N2 = RMS_NORM(A2), M2 = N2 * w2, and the multiplications of M2 that follow (its projections: those of the
//   linear attention or of the attention of Qwen3.5), maybe with the CONCAT of the result of one of them after them
//   (the input of the short convolution, which that result then goes straight into),
//
// they go in the same command as a fourth phase: the norm of A2, from the Y of phase 2, and the matrix of those
// multiplications as they go when they run alone (consecutive tensors of one format as one segment), if it has the
// segments of one phase. N2 and M2 have to have no other use than those multiplications; A2 is written as well.
// GGML_XDNA_NO_CHAIN_NEXT leaves them out.
//
// The last layer of the model (doc/XDNA2.md, "The last layer and the logits"): llama.cpp puts the GET_ROWS of the rows
// it outputs between P and A1, of P and of R, which with one row are those rows as they are (fRowsTaken) and are read
// where they are; after A2 come the output norm and the logits, as N2, M2 and the one multiplication of M2. M2 is
// then an output of the graph: the host writes it from A2 after the command.
//
// pDone is false, with nothing done, when the nodes are not such a block or the overlay cannot take it; when it is
// true, pLast is the index of the last node it computed.
bool fFeedForward(XdnaBackend & pBackend, const struct ggml_cgraph * pGraph, int pIndex, int & pLast, bool & pDone,
                  std::string & pError) {
  pDone                 = false;
  XdnaShared & vShared = fShared();
  int           aAt[9];
  aAt[0] = pIndex;
  // In the last layer the GET_ROWS of the rows it outputs come between W0 and A1 (of P and of R): with one row they
  // are those rows as they are (fRowsTaken), and the command reads them where they are.
  std::vector<const ggml_tensor *> aRowsOut;
  aAt[1] = fNextNode(pGraph, pIndex);
  while (aAt[1] < pGraph->n_nodes && aRowsOut.size() < 2 && fRowsTaken(pGraph->nodes[aAt[1]])) {
    aRowsOut.push_back(pGraph->nodes[aAt[1]]);
    aAt[1] = fNextNode(pGraph, aAt[1]);
  }
  if (aAt[1] >= pGraph->n_nodes) {
    return true;
  }
  for (int i = 2; i < 9; i++) {
    aAt[i] = fNextNode(pGraph, aAt[i - 1]);
    if (aAt[i] >= pGraph->n_nodes) {
      return true;
    }
  }
  ggml_tensor * vP  = pGraph->nodes[aAt[0]];
  ggml_tensor * vA1 = pGraph->nodes[aAt[1]];
  ggml_tensor * vN  = pGraph->nodes[aAt[2]];
  ggml_tensor * vM  = pGraph->nodes[aAt[3]];
  ggml_tensor * vX4 = pGraph->nodes[aAt[4]];
  ggml_tensor * vX5 = pGraph->nodes[aAt[5]];
  ggml_tensor * vS  = pGraph->nodes[aAt[6]];
  ggml_tensor * vD  = pGraph->nodes[aAt[7]];
  ggml_tensor * vA2 = pGraph->nodes[aAt[8]];
  // the operations, in this order
  if (vP->op != GGML_OP_MUL_MAT || vA1->op != GGML_OP_ADD || vN->op != GGML_OP_RMS_NORM || vM->op != GGML_OP_MUL ||
      vX4->op != GGML_OP_MUL_MAT || vX5->op != GGML_OP_MUL_MAT || vS->op != GGML_OP_GLU || vD->op != GGML_OP_MUL_MAT ||
      vA2->op != GGML_OP_ADD) {
    return true;
  }
  // one row, and every multiplication one the backend takes
  const ggml_tensor * vRows = vP->src[1];
  if (vRows->ne[1] != 1 || vRows->ne[2] != 1 || vRows->ne[3] != 1 || !fMulMatTaken(vShared, vP) ||
      !fMulMatTaken(vShared, vX4) || !fMulMatTaken(vShared, vX5) || !fMulMatTaken(vShared, vD) ||
      !ggml_is_contiguous(vP) || !ggml_is_contiguous(vX4) || !ggml_is_contiguous(vX5) || !ggml_is_contiguous(vD)) {
    return true;
  }
  // A1 = P + R
  if (!fAddTaken(vShared, vA1) || fUseCount(pGraph, vP) != 1 || (vP->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return true;
  }
  int vSide = -1;
  for (int vOperand = 0; vOperand < 2; vOperand++) {
    if (fIsResult(pGraph, vA1->src[vOperand], vP)) {
      vSide = vOperand;
    }
  }
  if (vSide < 0) {
    return true;
  }
  const ggml_tensor * vResidual = vA1->src[1 - vSide];
  // every GET_ROWS in between is on the way from P to A1 (whose steps fIsResult checked) or is R, used only by A1:
  // then R is read where its row is
  const ggml_tensor * vResidualRows = vResidual;
  for (const ggml_tensor * vRowsOut : aRowsOut) {
    bool vOnTheWay = false;
    for (const ggml_tensor * vStep = vA1->src[vSide]; vStep != vP; vStep = vStep->src[0]) {
      vOnTheWay = vOnTheWay || vStep == vRowsOut;
    }
    if (vRowsOut == vResidual && fUseCount(pGraph, vResidual) == 1 &&
        (vResidual->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
      vResidualRows = vResidual->src[0];
    } else if (!vOnTheWay) {
      return true;
    }
  }
  // N = RMS_NORM(A1), M = N * w
  const ggml_tensor * vNormWeights = fNormWeightsOf(vM);
  if (vN->src[0] != vA1 || !fRmsNormTaken(vN) || vNormWeights == nullptr || fUseCount(pGraph, vN) != 1 ||
      (vN->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || (vM->src[0] != vN && vM->src[1] != vN)) {
    return true;
  }
  // G and U of M, and S = SwiGLU(G, U): the gate is its first operand
  if (vX4->src[1] != vM || vX5->src[1] != vM || fUseCount(pGraph, vM) != 2 || (vM->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
      !fGluTaken(vS) || vShared.vSettings.vNoGlu) {
    return true;
  }
  ggml_tensor * vG = vS->src[0];
  ggml_tensor * vU = vS->src[1];
  if (!((vG == vX4 && vU == vX5) || (vG == vX5 && vU == vX4)) || fUseCount(pGraph, vG) != 1 ||
      fUseCount(pGraph, vU) != 1 || (vG->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
      (vU->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return true;
  }
  // D of S, and A2 = D + A1
  if (vD->src[1] != vS || fUseCount(pGraph, vS) != 1 || (vS->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
      !fAddTaken(vShared, vA2) || fUseCount(pGraph, vD) != 1 || (vD->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return true;
  }
  if (!((fIsResult(pGraph, vA2->src[0], vD) && vA2->src[1] == vA1) ||
        (fIsResult(pGraph, vA2->src[1], vD) && vA2->src[0] == vA1))) {
    return true;
  }
  // the sizes: R, h and the result of D of the outputs of W0; the gate and the up of the same outputs, which D takes
  const ggml_tensor * vW0 = vP->src[0];
  const ggml_tensor * vWg = vG->src[0];
  const ggml_tensor * vWu = vU->src[0];
  const ggml_tensor * vWd = vD->src[0];
  const int64_t       vEmbedding = vW0->ne[1];
  const int64_t       vHidden    = vWg->ne[1];
  if (ggml_nelements(vResidual) != vEmbedding || ggml_nelements(vA1) != vEmbedding || vWg->ne[0] != vEmbedding ||
      vWu->ne[0] != vEmbedding || vWu->ne[1] != vHidden || vWd->ne[0] != vHidden || vWd->ne[1] != vEmbedding ||
      ggml_nelements(vA2) != vEmbedding || vNormWeights->ne[0] != vEmbedding || vRows->data == nullptr ||
      vResidualRows->data == nullptr || vNormWeights->data == nullptr || vA1->data == nullptr || vA2->data == nullptr ||
      !ggml_is_contiguous(vA1) || !ggml_is_contiguous(vA2)) {
    return true;
  }
  float vEps;
  memcpy(&vEps, vN->op_params, sizeof(float));

  // the next layer: N2 = RMS_NORM(A2) and M2 = N2 * w2 right after A2, and the multiplications of M2 right after
  // them, which have to be all its uses; then maybe the CONCAT
  std::vector<ggml_tensor *> aNext;
  const ggml_tensor *        vNormNext  = nullptr;  // w2
  float                      vEpsNext   = 0;
  ggml_tensor *              vConcat    = nullptr;
  const ggml_tensor *        vMember    = nullptr;  // the multiplication whose result goes into the CONCAT
  ggml_tensor *              vNormedOut = nullptr;  // M2 when it is an output of the graph, written on the host
  int                        vLast      = aAt[8];
  auto fWithoutNext = [&]() {
    aNext.clear();
    vConcat    = nullptr;
    vMember    = nullptr;
    vNormedOut = nullptr;
    vLast      = aAt[8];
  };
  auto fOverlaps = [](const ggml_tensor * pA, const ggml_tensor * pB) {
    const char * vA = static_cast<const char *>(pA->data);
    const char * vB = static_cast<const char *>(pB->data);
    return vA < vB + ggml_nbytes(pB) && vB < vA + ggml_nbytes(pA);
  };
  if (!vShared.vSettings.vNoChainNext) {
    const int           vAtN2     = fNextNode(pGraph, aAt[8]);
    const int           vAtM2     = vAtN2 < pGraph->n_nodes ? fNextNode(pGraph, vAtN2) : pGraph->n_nodes;
    ggml_tensor *       vN2       = vAtN2 < pGraph->n_nodes ? pGraph->nodes[vAtN2] : nullptr;
    ggml_tensor *       vM2       = vAtM2 < pGraph->n_nodes ? pGraph->nodes[vAtM2] : nullptr;
    const ggml_tensor * vWeights2 = vM2 != nullptr ? fNormWeightsOf(vM2) : nullptr;
    // M2 may be an output of the graph: the output norm of the last layer, before the logits, which llama.cpp
    // always marks as one (the embeddings, and the hidden state of the multi-token prediction)
    if (vN2 != nullptr && vN2->op == GGML_OP_RMS_NORM && vN2->src[0] == vA2 && fRmsNormTaken(vN2) &&
        fUseCount(pGraph, vN2) == 1 && (vN2->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && vWeights2 != nullptr &&
        (vM2->src[0] == vN2 || vM2->src[1] == vN2) && vM2->data != nullptr && ggml_is_contiguous(vM2) &&
        vWeights2->data != nullptr && vWeights2->ne[0] == vEmbedding && ggml_nelements(vM2) == vEmbedding) {
      ggml_tensor * vM2Rows = vM2;
      int           vAt     = fNextNode(pGraph, vAtM2);
      if (vAt < pGraph->n_nodes && fRowsTaken(pGraph->nodes[vAt]) && pGraph->nodes[vAt]->src[0] == vM2 &&
          fUseCount(pGraph, vM2) == 1 && (pGraph->nodes[vAt]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
        vM2Rows = pGraph->nodes[vAt];
        vAt     = fNextNode(pGraph, vAt);
      }
      while (vAt < pGraph->n_nodes) {
        ggml_tensor * vNext = pGraph->nodes[vAt];
        if (vNext->op != GGML_OP_MUL_MAT || vNext->src[1] != vM2Rows || !ggml_is_contiguous(vNext) ||
            !fMulMatTaken(vShared, vNext) || vNext->src[0]->data == nullptr || vNext->data == nullptr) {
          break;
        }
        aNext.push_back(vNext);
        vLast = vAt;
        vAt   = fNextNode(pGraph, vAt);
      }
      if (!aNext.empty() && fUseCount(pGraph, vM2Rows) == int(aNext.size())) {
        vNormNext = vWeights2;
        memcpy(&vEpsNext, vN2->op_params, sizeof(float));
        vMember = vAt < pGraph->n_nodes ? fConcatMember(pGraph, pGraph->nodes[vAt], aNext) : nullptr;
        if (vMember != nullptr) {
          vConcat = pGraph->nodes[vAt];
          vLast   = vAt;
        }
        // M2 an output: the host writes it from A2 after the command, so what the fourth phase writes (written
        // after A2) cannot be where A2 is
        if ((vM2->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
          bool vApart = vConcat == nullptr || !fOverlaps(vConcat, vA2);
          for (const ggml_tensor * vNext : aNext) {
            vApart = vApart && !fOverlaps(vNext, vA2);
          }
          if (vApart) {
            vNormedOut = vM2;
          } else {
            fWithoutNext();
          }
        }
      } else {
        fWithoutNext();
      }
    }
  }

  const double vStart = fNowMs();
  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
  // the three matrices, kept for the model: the gate and the up as two segments of one, in that order; and that of
  // the multiplications of the next layer, as they go when they run alone
  XdnaGemvMatrix   aTemporary[4];
  XdnaGemvMatrix * aMatrices[4] = {
    fGemvMatrixOf(vShared, pBackend, {vW0}, aTemporary[0], pError),
    nullptr,
    nullptr,
    nullptr,
  };
  if (aMatrices[0] != nullptr) {
    aMatrices[1] = fGemvMatrixOf(vShared, pBackend, {vWg, vWu}, aTemporary[1], pError);
  }
  if (aMatrices[1] != nullptr) {
    aMatrices[2] = fGemvMatrixOf(vShared, pBackend, {vWd}, aTemporary[2], pError);
  }
  if (aMatrices[2] != nullptr && !aNext.empty()) {
    std::vector<const ggml_tensor *> aWeightsNext;
    for (const ggml_tensor * vNext : aNext) {
      aWeightsNext.push_back(vNext->src[0]);
    }
    aMatrices[3] = fGemvMatrixOf(vShared, pBackend, aWeightsNext, aTemporary[3], pError);
  }
  bool vKept = true;
  for (int m = 0; m < 3; m++) {
    vKept = vKept && aMatrices[m] != nullptr && aMatrices[m] != &aTemporary[m];
    fFreeGemvMatrix(vShared, aTemporary[m]);
  }
  if (aMatrices[0] == nullptr || aMatrices[1] == nullptr || aMatrices[2] == nullptr ||
      (!aNext.empty() && aMatrices[3] == nullptr)) {
    fFreeGemvMatrix(vShared, aTemporary[3]);
    return false;
  }
  // the next layer only with its matrix kept, and in the segments of one phase
  if (!aNext.empty() && (aMatrices[3] == &aTemporary[3] ||
                         aMatrices[3]->vChainWeights.vMatrix.vSegments > vShared.vChainOverlay.vSegments)) {
    fWithoutNext();
  }
  fFreeGemvMatrix(vShared, aTemporary[3]);
  if (!vKept || aMatrices[1]->vChainWeights.vMatrix.vSegments != 2) {
    return true;  // as separate multiplications
  }
  XdnaChainPhase aPhases[4];
  aPhases[0].vPrologue    = XdnaChainPrologue::cFloats;
  aPhases[0].vEpilogue    = XdnaChainEpilogue::cResidual;
  aPhases[0].vMatrix      = 0;
  aPhases[1].vPrologue    = XdnaChainPrologue::cNorm;
  aPhases[1].vFromPhase   = 0;
  aPhases[1].vEpilogue    = XdnaChainEpilogue::cSwiglu;
  aPhases[1].vEps         = vEps;
  aPhases[1].vMatrix      = 1;
  aPhases[1].vSegments    = 2;
  aPhases[2].vPrologue    = XdnaChainPrologue::cFloats;
  aPhases[2].vFromPhase   = 1;
  aPhases[2].vFromSegment = 1;
  aPhases[2].vEpilogue    = XdnaChainEpilogue::cAddH;
  aPhases[2].vMatrix      = 2;
  // phase 3: the norm of A2, from the Y of phase 2, and the multiplications of the next layer
  if (!aNext.empty()) {
    aPhases[3].vPrologue  = XdnaChainPrologue::cNorm;
    aPhases[3].vFromPhase = 2;
    aPhases[3].vEpilogue  = XdnaChainEpilogue::cNone;
    aPhases[3].vEps       = vEpsNext;
    aPhases[3].vMatrix    = 3;
    aPhases[3].vSegments  = aMatrices[3]->vChainWeights.vMatrix.vSegments;
  }
  // whether the overlay can run it, before anything is sent: with the next layer, or else without it
  for (uint32_t vPhases = aNext.empty() ? 3 : 4;; vPhases = 3) {
    XdnaChainMatrix  aShapes[4] = {aMatrices[0]->vChainWeights.vMatrix, aMatrices[1]->vChainWeights.vMatrix,
                                    aMatrices[2]->vChainWeights.vMatrix,
                                    vPhases == 4 ? aMatrices[3]->vChainWeights.vMatrix : XdnaChainMatrix()};
    XdnaChainCommand vCommand;
    std::string       vWhy;
    if (fXdnaChainCommandShape(vShared.vChainOverlay, aPhases, vPhases, aShapes, vPhases, vCommand, vWhy)) {
      break;
    }
    if (vPhases == 3) {
      GGML_LOG_DEBUG("%s: the feed-forward block of %s goes as separate multiplications: %s\n", cName, vA2->name,
                     vWhy.c_str());
      return true;
    }
    GGML_LOG_DEBUG("%s: the feed-forward block of %s goes without the next layer: %s\n", cName, vA2->name,
                   vWhy.c_str());
    fWithoutNext();
  }
  const uint32_t            vPhases    = aNext.empty() ? 3 : 4;
  const XdnaGemmTarget     vTargetH   = {static_cast<float *>(vA1->data), uint64_t(vEmbedding), 0, uint32_t(vEmbedding)};
  const XdnaGemmTarget     vTargetOut = {static_cast<float *>(vA2->data), uint64_t(vEmbedding), 0, uint32_t(vEmbedding)};
  // the results of the multiplications of the next layer, one after the other as in their matrix; the one of the
  // CONCAT into it, after the rows of its first operand
  std::vector<XdnaGemmTarget> aTargetsNext;
  uint32_t                     vFirstNext = 0;
  for (const ggml_tensor * vNext : aNext) {
    const uint32_t  vOutputs = uint32_t(vNext->src[0]->ne[1]);
    XdnaGemmTarget vTarget  = {static_cast<float *>(vNext->data), uint64_t(vOutputs), vFirstNext, vOutputs};
    if (vNext == vMember) {
      vTarget.vData = static_cast<float *>(vConcat->data) + vConcat->src[0]->ne[1] * vConcat->ne[0];
    }
    aTargetsNext.push_back(vTarget);
    vFirstNext += vOutputs;
  }
  XdnaChainIo              aIo[4];
  aIo[0].vX        = static_cast<const float *>(vRows->data);
  aIo[0].vStride   = size_t(vRows->ne[0]);
  aIo[0].vResidual = static_cast<const float *>(vResidualRows->data);
  aIo[0].aTargets  = &vTargetH;
  aIo[0].vTargets  = 1;
  aIo[1].vNorm     = static_cast<const float *>(vNormWeights->data);
  aIo[2].aTargets  = &vTargetOut;
  aIo[2].vTargets  = 1;
  if (vPhases == 4) {
    aIo[3].vNorm    = static_cast<const float *>(vNormNext->data);
    aIo[3].aTargets = aTargetsNext.data();
    aIo[3].vTargets = uint32_t(aTargetsNext.size());
  }
  const XdnaChainWeights * aWeights[4] = {&aMatrices[0]->vChainWeights, &aMatrices[1]->vChainWeights,
                                           &aMatrices[2]->vChainWeights,
                                           vPhases == 4 ? &aMatrices[3]->vChainWeights : nullptr};
  XdnaGemmTimes vTimes;
  if (!vShared.vRunner.mRunChain(aPhases, aIo, vPhases, aWeights, vPhases, vTimes, pError)) {
    return false;
  }
  if (vConcat != nullptr) {
    fConcatHead(vConcat);
    pBackend.vTotals.vConcatsFused++;
  }
  // M2 an output of the graph: from A2, as fRmsNormOnHost and then fMulOnHost compute it (through a copy, as M2 may
  // be where A2 is)
  if (vNormedOut != nullptr) {
    const float *      vFrom  = static_cast<const float *>(vA2->data);
    const float *      vBy    = static_cast<const float *>(vNormNext->data);
    float              vScale = 0;
    std::vector<float> aNormed(static_cast<size_t>(vEmbedding));
    fXdnaRmsScales(vFrom, 1, uint64_t(vEmbedding), uint64_t(vEmbedding), vEpsNext, &vScale);
    for (int64_t i = 0; i < vEmbedding; i++) {
      const float vNormed = vFrom[i] * vScale;
      aNormed[size_t(i)]  = vNormed * vBy[i];
    }
    std::memcpy(vNormedOut->data, aNormed.data(), aNormed.size() * sizeof(float));
    pBackend.vTotals.vNormsHost++;
    pBackend.vTotals.vChainsLogits++;
  }
  pBackend.vTotals.vOperations += 4 + aNext.size();
  pBackend.vTotals.vCommands++;
  pBackend.vTotals.vCommandsRows++;
  pBackend.vTotals.vRows += 4 + aNext.size();
  pBackend.vTotals.vChainsFfn++;
  pBackend.vTotals.vChainsNext  += vPhases == 4 ? 1 : 0;
  pBackend.vTotals.vChainsFfnMs += vTimes.vNpuMs;
  pBackend.vTotals.vPlanMs      += vTimes.vPlanMs;
  pBackend.vTotals.vPackMs      += vTimes.vPackMs;
  pBackend.vTotals.vNpuMs       += vTimes.vNpuMs;
  pBackend.vTotals.vNpuRowsMs   += vTimes.vNpuMs;
  pBackend.vTotals.vUnpackMs    += vTimes.vUnpackMs;
  pBackend.vTotals.vFlushMs     += vTimes.vFlushMs;
  pBackend.vTotals.vTotalMs     += fNowMs() - vStart;
  pLast = vLast;
  pDone = true;
  return true;
}

// llama.cpp-xdna: generating a token on the chained overlay when the routed experts of a layer ran on the processor
// (GGML_XDNA_NO_MOE or GGML_XDNA_MOE_MIN_TOKENS >= 2): chains the shared expert ({gate_shexp, up_shexp} + SwiGLU +
// down_shexp + optional scalar sigmoid gate + ADD with moe_out + ADD with ffn_residual) together with the norm and
// projections of the next layer (or output_norm + output.weight in the last layer) in one command of 2 or 3 phases.
bool fSharedExpertChain(XdnaBackend & pBackend, const struct ggml_cgraph * pGraph, int pIndex, int & pLast,
                        bool & pDone, std::string & pError) {
  pDone                 = false;
  XdnaShared & vShared  = fShared();
  if (vShared.vSettings.vNoMoeDense || vShared.vSettings.vNoChain || vShared.vSettings.vNoFuse ||
      vShared.vSettings.vEmulate || fMoeActiveForRows(vShared, 1)) {
    return true;
  }
  const int vAt1 = fNextNode(pGraph, pIndex);
  const int vAt2 = vAt1 < pGraph->n_nodes ? fNextNode(pGraph, vAt1) : pGraph->n_nodes;
  const int vAt3 = vAt2 < pGraph->n_nodes ? fNextNode(pGraph, vAt2) : pGraph->n_nodes;
  if (vAt3 >= pGraph->n_nodes) {
    return true;
  }
  ggml_tensor * vX0 = pGraph->nodes[pIndex];
  ggml_tensor * vX1 = pGraph->nodes[vAt1];
  ggml_tensor * vS  = pGraph->nodes[vAt2];
  ggml_tensor * vD  = pGraph->nodes[vAt3];
  if (vX0->op != GGML_OP_MUL_MAT || vX1->op != GGML_OP_MUL_MAT || vS->op != GGML_OP_GLU || vD->op != GGML_OP_MUL_MAT ||
      !fIsSharedExpertWeight(vX0->src[0]) || !fIsSharedExpertWeight(vX1->src[0]) ||
      !fIsSharedExpertWeight(vD->src[0])) {
    return true;
  }
  const ggml_tensor * vM = vX0->src[1];
  if (vM == nullptr || vM->ne[1] != 1 || vM->ne[2] != 1 || vM->ne[3] != 1 || vX1->src[1] != vM ||
      !fMulMatTaken(vShared, vX0) || !fMulMatTaken(vShared, vX1) || !fMulMatTaken(vShared, vD) ||
      !ggml_is_contiguous(vM) || !ggml_is_contiguous(vX0) || !ggml_is_contiguous(vX1) || !ggml_is_contiguous(vD) ||
      !fGluTaken(vS) || vShared.vSettings.vNoGlu) {
    return true;
  }
  ggml_tensor * vG = vS->src[0];
  ggml_tensor * vU = vS->src[1];
  if (!((vG == vX0 && vU == vX1) || (vG == vX1 && vU == vX0)) || fUseCount(pGraph, vG) != 1 ||
      fUseCount(pGraph, vU) != 1 || (vG->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
      (vU->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || vD->src[1] != vS || fUseCount(pGraph, vS) != 1 ||
      (vS->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || fUseCount(pGraph, vD) != 1 ||
      (vD->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return true;
  }

  const ggml_tensor * vGateInpWeights = nullptr;
  const ggml_tensor * vShexpBranch    = vD;
  ggml_tensor *       vFfnOut         = nullptr;
  int                 vAtFfnOut       = -1;
  const int           vAt4            = fNextNode(pGraph, vAt3);
  if (vAt4 >= pGraph->n_nodes) {
    return true;
  }
  ggml_tensor * vN4 = pGraph->nodes[vAt4];
  if (vN4->op == GGML_OP_MUL_MAT && fIsSharedGate(vN4->src[0]) && vN4->src[1] == vM && fUseCount(pGraph, vN4) == 1 &&
      (vN4->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && ggml_nelements(vN4) == 1) {
    const int vAt5 = fNextNode(pGraph, vAt4);
    const int vAt6 = vAt5 < pGraph->n_nodes ? fNextNode(pGraph, vAt5) : pGraph->n_nodes;
    const int vAt7 = vAt6 < pGraph->n_nodes ? fNextNode(pGraph, vAt6) : pGraph->n_nodes;
    if (vAt7 >= pGraph->n_nodes) {
      return true;
    }
    ggml_tensor * vSig   = pGraph->nodes[vAt5];
    ggml_tensor * vGated = pGraph->nodes[vAt6];
    ggml_tensor * vSum   = pGraph->nodes[vAt7];
    if (vSig->op != GGML_OP_UNARY || ggml_get_unary_op(vSig) != GGML_UNARY_OP_SIGMOID || vSig->src[0] != vN4 ||
        fUseCount(pGraph, vSig) != 1 || (vSig->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || vGated->op != GGML_OP_MUL ||
        !((vGated->src[0] == vD && vGated->src[1] == vSig) || (vGated->src[0] == vSig && vGated->src[1] == vD)) ||
        fUseCount(pGraph, vGated) != 1 || (vGated->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || vSum->op != GGML_OP_ADD ||
        (vSum->src[0] != vGated && vSum->src[1] != vGated)) {
      return true;
    }
    vGateInpWeights = vN4->src[0];
    vShexpBranch    = vGated;
    vFfnOut         = vSum;
    vAtFfnOut       = vAt7;
  } else if (vN4->op == GGML_OP_ADD && (vN4->src[0] == vD || vN4->src[1] == vD)) {
    vFfnOut   = vN4;
    vAtFfnOut = vAt4;
  } else {
    return true;
  }
  const ggml_tensor * vMoeOut = vFfnOut->src[0] == vShexpBranch ? vFfnOut->src[1] : vFfnOut->src[0];
  const int vAtA2 = fNextNode(pGraph, vAtFfnOut);
  if (vAtA2 >= pGraph->n_nodes || fUseCount(pGraph, vFfnOut) != 1 || (vFfnOut->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return true;
  }
  ggml_tensor * vA2 = pGraph->nodes[vAtA2];
  if (vA2->op != GGML_OP_ADD || !fAddTaken(vShared, vA2) || (vA2->src[0] != vFfnOut && vA2->src[1] != vFfnOut)) {
    return true;
  }
  const ggml_tensor * vResidual  = vA2->src[0] == vFfnOut ? vA2->src[1] : vA2->src[0];
  const ggml_tensor * vWg        = vG->src[0];
  const ggml_tensor * vWu        = vU->src[0];
  const ggml_tensor * vWd        = vD->src[0];
  const int64_t       vEmbedding = vWd->ne[1];
  const int64_t       vHidden    = vWg->ne[1];
  if (vWg->ne[0] != vEmbedding || vWu->ne[0] != vEmbedding || vWu->ne[1] != vHidden || vWd->ne[0] != vHidden ||
      ggml_nelements(vM) != vEmbedding || ggml_nelements(vMoeOut) != vEmbedding ||
      ggml_nelements(vResidual) != vEmbedding || ggml_nelements(vA2) != vEmbedding || !ggml_is_contiguous(vMoeOut) ||
      !ggml_is_contiguous(vResidual) || !ggml_is_contiguous(vA2) || vM->data == nullptr || vMoeOut->data == nullptr ||
      vResidual->data == nullptr || vA2->data == nullptr ||
      (vGateInpWeights != nullptr &&
       (vGateInpWeights->data == nullptr || vGateInpWeights->ne[0] != vEmbedding || vGateInpWeights->ne[1] != 1))) {
    return true;
  }

  float vSharedGate = 1.0f;
  if (vGateInpWeights != nullptr) {
    std::vector<float> aGateW;
    const float *      vGateWPtr = nullptr;
    if (vGateInpWeights->type == GGML_TYPE_F32) {
      vGateWPtr = static_cast<const float *>(vGateInpWeights->data);
    } else {
      aGateW.resize(size_t(vEmbedding));
      ggml_get_type_traits(vGateInpWeights->type)->to_float(vGateInpWeights->data, aGateW.data(), vEmbedding);
      vGateWPtr = aGateW.data();
    }
    const float vLogit = fDotRow(static_cast<const float *>(vM->data), vGateWPtr, uint32_t(vEmbedding));
    vSharedGate        = 1.0f / (1.0f + std::exp(-vLogit));
    if (!(vSharedGate >= 1e-4f && std::isfinite(vSharedGate))) {
      return true;
    }
  }

  std::vector<ggml_tensor *> aNext;
  const ggml_tensor *        vNormNext  = nullptr;
  float                      vEpsNext   = 0;
  ggml_tensor *              vConcat    = nullptr;
  const ggml_tensor *        vMember    = nullptr;
  ggml_tensor *              vNormedOut = nullptr;
  int                        vLast      = vAtA2;
  auto fWithoutNext = [&]() {
    aNext.clear();
    vConcat    = nullptr;
    vMember    = nullptr;
    vNormedOut = nullptr;
    vLast      = vAtA2;
  };
  auto fOverlaps = [](const ggml_tensor * pA, const ggml_tensor * pB) {
    const char * vA = static_cast<const char *>(pA->data);
    const char * vB = static_cast<const char *>(pB->data);
    return vA < vB + ggml_nbytes(pB) && vB < vA + ggml_nbytes(pA);
  };
  if (!vShared.vSettings.vNoChainNext) {
    const int           vAtN2     = fNextNode(pGraph, vAtA2);
    const int           vAtM2     = vAtN2 < pGraph->n_nodes ? fNextNode(pGraph, vAtN2) : pGraph->n_nodes;
    ggml_tensor *       vN2       = vAtN2 < pGraph->n_nodes ? pGraph->nodes[vAtN2] : nullptr;
    ggml_tensor *       vM2       = vAtM2 < pGraph->n_nodes ? pGraph->nodes[vAtM2] : nullptr;
    const ggml_tensor * vWeights2 = vM2 != nullptr ? fNormWeightsOf(vM2) : nullptr;
    if (vN2 != nullptr && vN2->op == GGML_OP_RMS_NORM && vN2->src[0] == vA2 && fRmsNormTaken(vN2) &&
        fUseCount(pGraph, vN2) == 1 && (vN2->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && vWeights2 != nullptr &&
        (vM2->src[0] == vN2 || vM2->src[1] == vN2) && vM2->data != nullptr && ggml_is_contiguous(vM2) &&
        vWeights2->data != nullptr && vWeights2->ne[0] == vEmbedding && ggml_nelements(vM2) == vEmbedding) {
      int vAt = fNextNode(pGraph, vAtM2);
      while (vAt < pGraph->n_nodes) {
        ggml_tensor * vNext = pGraph->nodes[vAt];
        if (vNext->op != GGML_OP_MUL_MAT || vNext->src[1] != vM2 || !ggml_is_contiguous(vNext) ||
            !fMulMatTaken(vShared, vNext) || fIsSharedGate(vNext->src[0]) || vNext->src[0]->data == nullptr ||
            vNext->data == nullptr) {
          break;
        }
        aNext.push_back(vNext);
        vLast = vAt;
        vAt   = fNextNode(pGraph, vAt);
      }
      if (!aNext.empty() && fUseCount(pGraph, vM2) == int(aNext.size())) {
        vNormNext = vWeights2;
        memcpy(&vEpsNext, vN2->op_params, sizeof(float));
        vMember = vAt < pGraph->n_nodes ? fConcatMember(pGraph, pGraph->nodes[vAt], aNext) : nullptr;
        if (vMember != nullptr) {
          vConcat = pGraph->nodes[vAt];
          vLast   = vAt;
        }
        if ((vM2->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
          bool vApart = vConcat == nullptr || !fOverlaps(vConcat, vA2);
          for (const ggml_tensor * vNext : aNext) {
            vApart = vApart && !fOverlaps(vNext, vA2);
          }
          if (vApart) {
            vNormedOut = vM2;
          } else {
            fWithoutNext();
          }
        }
      } else {
        fWithoutNext();
      }
    }
  }

  const double vStart = fNowMs();
  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
  XdnaGemvMatrix   aTemporary[3];
  XdnaGemvMatrix * aMatrices[3] = {
    fGemvMatrixOf(vShared, pBackend, {vWg, vWu}, aTemporary[0], pError),
    nullptr,
    nullptr,
  };
  if (aMatrices[0] != nullptr) {
    aMatrices[1] = fGemvMatrixOf(vShared, pBackend, {vWd}, aTemporary[1], pError);
  }
  if (aMatrices[1] != nullptr && !aNext.empty()) {
    std::vector<const ggml_tensor *> aWeightsNext;
    for (const ggml_tensor * vNext : aNext) {
      aWeightsNext.push_back(vNext->src[0]);
    }
    aMatrices[2] = fGemvMatrixOf(vShared, pBackend, aWeightsNext, aTemporary[2], pError);
  }
  const bool vKept = aMatrices[0] != nullptr && aMatrices[0] != &aTemporary[0] && aMatrices[1] != nullptr &&
                     aMatrices[1] != &aTemporary[1];
  fFreeGemvMatrix(vShared, aTemporary[0]);
  fFreeGemvMatrix(vShared, aTemporary[1]);
  if (aMatrices[0] == nullptr || aMatrices[1] == nullptr || (!aNext.empty() && aMatrices[2] == nullptr)) {
    fFreeGemvMatrix(vShared, aTemporary[2]);
    return false;
  }
  if (!aNext.empty() && (aMatrices[2] == &aTemporary[2] ||
                         aMatrices[2]->vChainWeights.vMatrix.vSegments > vShared.vChainOverlay.vSegments)) {
    fWithoutNext();
  }
  fFreeGemvMatrix(vShared, aTemporary[2]);
  if (!vKept || aMatrices[0]->vChainWeights.vMatrix.vSegments != 2) {
    return true;
  }

  const float        vInvGate = vGateInpWeights != nullptr ? 1.0f / vSharedGate : 1.0f;
  std::vector<float> aCombinedResidual(static_cast<size_t>(vEmbedding));
  const float *      vMoePtr = static_cast<const float *>(vMoeOut->data);
  const float *      vResPtr = static_cast<const float *>(vResidual->data);
  for (int64_t k = 0; k < vEmbedding; k++) {
    aCombinedResidual[size_t(k)] = (vMoePtr[k] + vResPtr[k]) * vInvGate;
  }

  XdnaChainPhase aPhases[3];
  aPhases[0].vPrologue    = XdnaChainPrologue::cFloats;
  aPhases[0].vEpilogue    = XdnaChainEpilogue::cSwiglu;
  aPhases[0].vMatrix      = 0;
  aPhases[0].vSegments    = 2;
  aPhases[1].vPrologue    = XdnaChainPrologue::cFloats;
  aPhases[1].vFromPhase   = 0;
  aPhases[1].vFromSegment = 1;
  aPhases[1].vEpilogue    = XdnaChainEpilogue::cResidual;
  aPhases[1].vMatrix      = 1;
  if (!aNext.empty()) {
    aPhases[2].vPrologue  = XdnaChainPrologue::cNorm;
    aPhases[2].vFromPhase = 1;
    aPhases[2].vEpilogue  = XdnaChainEpilogue::cNone;
    aPhases[2].vEps       = vEpsNext * vInvGate * vInvGate;
    aPhases[2].vMatrix    = 2;
    aPhases[2].vSegments  = aMatrices[2]->vChainWeights.vMatrix.vSegments;
  }
  for (uint32_t vTry = aNext.empty() ? 2 : 3;; vTry = 2) {
    XdnaChainMatrix  aShapes[3] = {aMatrices[0]->vChainWeights.vMatrix, aMatrices[1]->vChainWeights.vMatrix,
                                   vTry == 3 ? aMatrices[2]->vChainWeights.vMatrix : XdnaChainMatrix()};
    XdnaChainCommand vCommand;
    std::string      vWhy;
    if (fXdnaChainCommandShape(vShared.vChainOverlay, aPhases, vTry, aShapes, vTry, vCommand, vWhy)) {
      break;
    }
    if (vTry == 2) {
      return true;
    }
    fWithoutNext();
  }

  const uint32_t     vPhases = aNext.empty() ? 2 : 3;
  std::vector<float> aUnscaledOut(vGateInpWeights != nullptr ? size_t(vEmbedding) : 0);
  const XdnaGemmTarget vTargetOut = {
      vGateInpWeights != nullptr ? aUnscaledOut.data() : static_cast<float *>(vA2->data), uint64_t(vEmbedding), 0,
      uint32_t(vEmbedding)};
  std::vector<XdnaGemmTarget> aTargetsNext;
  uint32_t                    vFirstNext = 0;
  for (const ggml_tensor * vNext : aNext) {
    const uint32_t vOutputs = uint32_t(vNext->src[0]->ne[1]);
    XdnaGemmTarget vTarget  = {static_cast<float *>(vNext->data), uint64_t(vOutputs), vFirstNext, vOutputs};
    if (vNext == vMember) {
      vTarget.vData = static_cast<float *>(vConcat->data) + vConcat->src[0]->ne[1] * vConcat->ne[0];
    }
    aTargetsNext.push_back(vTarget);
    vFirstNext += vOutputs;
  }
  XdnaChainIo aIo[3];
  aIo[0].vX        = static_cast<const float *>(vM->data);
  aIo[0].vStride   = size_t(vM->ne[0]);
  aIo[1].vResidual = aCombinedResidual.data();
  aIo[1].aTargets  = &vTargetOut;
  aIo[1].vTargets  = 1;
  if (vPhases == 3) {
    aIo[2].vNorm    = static_cast<const float *>(vNormNext->data);
    aIo[2].aTargets = aTargetsNext.data();
    aIo[2].vTargets = uint32_t(aTargetsNext.size());
  }
  const XdnaChainWeights * aWeights[3] = {&aMatrices[0]->vChainWeights, &aMatrices[1]->vChainWeights,
                                          vPhases == 3 ? &aMatrices[2]->vChainWeights : nullptr};
  XdnaGemmTimes vTimes;
  if (!vShared.vRunner.mRunChain(aPhases, aIo, vPhases, aWeights, vPhases, vTimes, pError)) {
    return false;
  }
  if (vGateInpWeights != nullptr) {
    float * vOutA2 = static_cast<float *>(vA2->data);
    for (int64_t k = 0; k < vEmbedding; k++) {
      vOutA2[k] = aUnscaledOut[size_t(k)] * vSharedGate;
    }
  }
  if (vConcat != nullptr) {
    fConcatHead(vConcat);
    pBackend.vTotals.vConcatsFused++;
  }
  if (vNormedOut != nullptr) {
    const float *      vFrom  = static_cast<const float *>(vA2->data);
    const float *      vBy    = static_cast<const float *>(vNormNext->data);
    float              vScale = 0;
    std::vector<float> aNormed(static_cast<size_t>(vEmbedding));
    fXdnaRmsScales(vFrom, 1, uint64_t(vEmbedding), uint64_t(vEmbedding), vEpsNext, &vScale);
    for (int64_t i = 0; i < vEmbedding; i++) {
      const float vNormed = vFrom[i] * vScale;
      aNormed[size_t(i)]  = vNormed * vBy[i];
    }
    std::memcpy(vNormedOut->data, aNormed.data(), aNormed.size() * sizeof(float));
    pBackend.vTotals.vNormsHost++;
    pBackend.vTotals.vChainsLogits++;
  }
  pBackend.vTotals.vOperations += 3 + aNext.size();
  pBackend.vTotals.vCommands++;
  pBackend.vTotals.vCommandsRows++;
  pBackend.vTotals.vRows += 3 + aNext.size();
  pBackend.vTotals.vChainsShexp++;
  pBackend.vTotals.vChainsNext    += vPhases == 3 ? 1 : 0;
  pBackend.vTotals.vChainsShexpMs += vTimes.vNpuMs;
  pBackend.vTotals.vPlanMs        += vTimes.vPlanMs;
  pBackend.vTotals.vPackMs        += vTimes.vPackMs;
  pBackend.vTotals.vNpuMs         += vTimes.vNpuMs;
  pBackend.vTotals.vNpuRowsMs     += vTimes.vNpuMs;
  pBackend.vTotals.vUnpackMs      += vTimes.vUnpackMs;
  pBackend.vTotals.vFlushMs       += vTimes.vFlushMs;
  pBackend.vTotals.vTotalMs       += fNowMs() - vStart;
  pLast = vLast;
  pDone = true;
  return true;
}

// llama.cpp-xdna: whether pNode is the SIGMOID or MUL of the output gate of a full-attention layer in a prompt batch
// (qwen35/qwen35moe: gate_sigmoid = SIGMOID(CPY(VIEW(Qcur_full))), attn_gated = MUL(FLASH_ATTN_EXT, gate_sigmoid))
bool fAttnGateTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vNoFuse || pShared.vSettings.vEmulate || pShared.vSettings.vNoAttn || pNode == nullptr ||
      pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode) || pNode->ne[0] < 64 || pNode->ne[2] != 1 ||
      pNode->ne[3] != 1 || uint64_t(pNode->ne[1]) < pShared.vSettings.vMinRows) {
    return false;
  }
  if (pNode->op == GGML_OP_UNARY && ggml_get_unary_op(pNode) == GGML_UNARY_OP_SIGMOID) {
    const ggml_tensor * vSrc = pNode->src[0];
    return vSrc != nullptr && vSrc->op == GGML_OP_CPY && vSrc->type == GGML_TYPE_F32 &&
           ggml_are_same_shape(vSrc, pNode) && ggml_is_contiguous(vSrc);
  }
  if (pNode->op == GGML_OP_MUL) {
    for (int vSide = 0; vSide < 2; vSide++) {
      const ggml_tensor * vAttn = fThroughReshapes(pNode->src[vSide]);
      const ggml_tensor * vSig  = pNode->src[1 - vSide];
      if (vAttn != nullptr && vAttn->op == GGML_OP_FLASH_ATTN_EXT && fAttnTaken(pShared, vAttn) && vSig != nullptr &&
          vSig->op == GGML_OP_UNARY && ggml_are_same_shape(pNode->src[vSide], pNode) &&
          ggml_is_contiguous(pNode->src[vSide]) && fAttnGateTaken(pShared, vSig)) {
        return true;
      }
    }
  }
  return false;
}

bool fAttnGateStep(XdnaBackend & pBackend, const struct ggml_cgraph * pGraph, int pIndex, int & pLastIndex,
                   std::string & pError) {
  pLastIndex             = pIndex;
  ggml_tensor * vNode    = pGraph->nodes[pIndex];
  const int64_t vCount   = ggml_nelements(vNode);
  XdnaShared &  vShared  = fShared();
  constexpr int64_t cChunk = 4096;
  const uint32_t    vTasks = uint32_t(std::max<int64_t>(1, (vCount + cChunk - 1) / cChunk));
  auto fRunParallel = [&](auto && pFn) {
    if (vTasks > 1) {
      std::lock_guard<std::mutex> vLock(vShared.vMutex);
      vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
      vShared.vRunner.mWorkers().mForEach(vTasks, pFn);
    } else {
      pFn(0);
    }
  };
  if (vNode->op == GGML_OP_UNARY) {
    if (vNode->src[0] == nullptr || vNode->src[0]->data == nullptr || vNode->data == nullptr) {
      pError = "a tensor of gate_sigmoid has no memory";
      return false;
    }
    const int vMulAt = fNextNode(pGraph, pIndex);
    if (vMulAt < pGraph->n_nodes && fUseCount(pGraph, vNode) == 1 && (vNode->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
      ggml_tensor * vMul = pGraph->nodes[vMulAt];
      if (vMul->op == GGML_OP_MUL && fAttnGateTaken(vShared, vMul) && vMul->data != nullptr &&
          (vMul->src[0] == vNode || vMul->src[1] == vNode)) {
        const ggml_tensor * vAttnSrc = vMul->src[0] == vNode ? vMul->src[1] : vMul->src[0];
        if (vAttnSrc != nullptr && vAttnSrc->data != nullptr) {
          const float * vGate = static_cast<const float *>(vNode->src[0]->data);
          const float * vAttn = static_cast<const float *>(vAttnSrc->data);
          float *       vOut  = static_cast<float *>(vMul->data);
          fRunParallel([&](uint32_t pTask) {
            const int64_t vStart = int64_t(pTask) * cChunk;
            const int64_t vEnd   = std::min(vCount, vStart + cChunk);
            for (int64_t k = vStart; k < vEnd; k++) {
              vOut[k] = vAttn[k] * (1.0f / (1.0f + std::exp(-vGate[k])));
            }
          });
          pLastIndex = vMulAt;
          return true;
        }
      }
    }
    const float * vGate = static_cast<const float *>(vNode->src[0]->data);
    float *       vOut  = static_cast<float *>(vNode->data);
    fRunParallel([&](uint32_t pTask) {
      const int64_t vStart = int64_t(pTask) * cChunk;
      const int64_t vEnd   = std::min(vCount, vStart + cChunk);
      for (int64_t k = vStart; k < vEnd; k++) {
        vOut[k] = 1.0f / (1.0f + std::exp(-vGate[k]));
      }
    });
    return true;
  }
  if (vNode->src[0] == nullptr || vNode->src[1] == nullptr || vNode->src[0]->data == nullptr ||
      vNode->src[1]->data == nullptr || vNode->data == nullptr) {
    pError = "a tensor of attn_gated has no memory";
    return false;
  }
  const float * vA   = static_cast<const float *>(vNode->src[0]->data);
  const float * vB   = static_cast<const float *>(vNode->src[1]->data);
  float *       vOut = static_cast<float *>(vNode->data);
  fRunParallel([&](uint32_t pTask) {
    const int64_t vStart = int64_t(pTask) * cChunk;
    const int64_t vEnd   = std::min(vCount, vStart + cChunk);
    for (int64_t k = vStart; k < vEnd; k++) {
      vOut[k] = vA[k] * vB[k];
    }
  });
  return true;
}

// llama.cpp-xdna: whether pNode is an operation of the gated norm of a layer of linear attention of Qwen3.5 in a batch
// of a prompt that the backend takes (build_norm_gated in src/models/qwen35.cpp: the RMS_NORM of the outputs of the
// recurrence, the MUL by ssm_norm, the SILU of z and their MUL), heads of 128 for at least the rows a multiplication
// needs to go to the NPU, as the multiplication that reads them (ssm_out). When the four come one after the other right
// before it, they go into the conversion of its rows (fGatedNorm); otherwise they are computed on the host as the CPU
// backend computes them (fGatedOnHost). GGML_XDNA_NO_GATED leaves them to the CPU backend.
bool fGatedTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  const int64_t cDim = cXdnaChainRecurrenceDim;
  if (pShared.vSettings.vNoGated || pShared.vSettings.vNoFuse || pShared.vSettings.vEmulate ||
      pNode->type != GGML_TYPE_F32 || pNode->ne[0] != cDim || pNode->ne[2] < 2 ||
      uint64_t(pNode->ne[2]) < pShared.vSettings.vMinRows || pNode->ne[3] != 1 || !ggml_is_contiguous(pNode)) {
    return false;
  }
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (vA == nullptr || vA->type != GGML_TYPE_F32 || (vB != nullptr && vB->type != GGML_TYPE_F32)) {
    return false;
  }
  switch (pNode->op) {
    case GGML_OP_RMS_NORM:
      // of the outputs of the recurrence: a view of it, its rows one after the other
      return vA->op == GGML_OP_VIEW && vA->view_src != nullptr && vA->view_src->op == GGML_OP_GATED_DELTA_NET &&
             ggml_are_same_shape(vA, pNode) && vA->nb[0] == sizeof(float) && vA->nb[1] == cDim * sizeof(float) &&
             vA->nb[2] == size_t(vA->ne[1]) * vA->nb[1];
    case GGML_OP_UNARY: {
      // silu(z) or sigmoid(z), z the result of a multiplication
      const enum ggml_unary_op vUnary = ggml_get_unary_op(pNode);
      return (vUnary == GGML_UNARY_OP_SILU || vUnary == GGML_UNARY_OP_SIGMOID) && ggml_are_same_shape(vA, pNode) &&
             ggml_is_contiguous(vA) && fThroughReshapes(vA)->op == GGML_OP_MUL_MAT;
    }
    case GGML_OP_MUL:
      if (vB == nullptr || !ggml_is_contiguous(vA) || !ggml_is_contiguous(vB)) {
        return false;
      }
      for (int vSide = 0; vSide < 2; vSide++) {
        const ggml_tensor * vThis  = pNode->src[vSide];
        const ggml_tensor * vOther = pNode->src[1 - vSide];
        // the norm by ssm_norm
        if (vThis->op == GGML_OP_RMS_NORM && ggml_are_same_shape(vThis, pNode) && fGatedTaken(pShared, vThis) &&
            fIsWeight(vOther) && fShapeIs(vOther, cDim, 1, 1, 1)) {
          return true;
        }
        // and that by silu(z) or sigmoid(z)
        if ((fIsUnary(vThis, GGML_UNARY_OP_SILU) || fIsUnary(vThis, GGML_UNARY_OP_SIGMOID)) &&
            ggml_are_same_shape(vThis, pNode) && fGatedTaken(pShared, vThis) && vOther->op == GGML_OP_MUL &&
            ggml_are_same_shape(vOther, pNode) && fGatedTaken(pShared, vOther)) {
          return true;
        }
      }
      return false;
    default:
      return false;
  }
}

// llama.cpp-xdna: an operation of the gated norm (fGatedTaken) on its own, on the host, as the CPU backend computes it:
// RMS_NORM (fXdnaRmsScales, x * scale), the MULs, and SILU with the vectors of the CPU backend (fXdnaSwiglu by 1)
bool fGatedOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (pNode->data == nullptr || vA->data == nullptr || (vB != nullptr && vB->data == nullptr)) {
    pError = "a tensor has no memory";
    return false;
  }
  const uint64_t vLength = uint64_t(pNode->ne[0]);
  const uint64_t vRows   = uint64_t(ggml_nrows(pNode));
  const float *  vFrom   = static_cast<const float *>(vA->data);
  float *        vTo     = static_cast<float *>(pNode->data);
  switch (pNode->op) {
    case GGML_OP_RMS_NORM: {
      float vEps;
      memcpy(&vEps, pNode->op_params, sizeof(float));
      std::vector<float> aScales(vRows);
      fXdnaRmsScales(vFrom, vRows, vLength, vLength, vEps, aScales.data());
      for (uint64_t r = 0; r < vRows; r++) {
        for (uint64_t i = 0; i < vLength; i++) {
          vTo[r * vLength + i] = vFrom[r * vLength + i] * aScales[r];
        }
      }
      break;
    }
    case GGML_OP_UNARY: {
      const bool               vSigmoid = ggml_get_unary_op(pNode) == GGML_UNARY_OP_SIGMOID;
      const std::vector<float> aOnes(vLength, 1.0f);
      for (uint64_t r = 0; r < vRows; r++) {
        if (vSigmoid) {
          fXdnaSigmoidMul(vFrom + r * vLength, aOnes.data(), vTo + r * vLength, uint32_t(vLength), 0, vLength);
        } else {
          fXdnaSwiglu(vFrom + r * vLength, aOnes.data(), vTo + r * vLength, uint32_t(vLength), 0, vLength);
        }
      }
      break;
    }
    case GGML_OP_MUL: {
      // the smaller operand repeats over the rows (the weights of the norm), or both are whole
      const float *  vOther = static_cast<const float *>(vB->data);
      const uint64_t vEvery = uint64_t(ggml_nelements(vB)) == vLength ? vLength : vLength * vRows;
      if (uint64_t(ggml_nelements(vA)) == vLength && uint64_t(ggml_nelements(vB)) != vLength) {
        std::swap(vFrom, vOther);
      }
      for (uint64_t n = 0; n < vLength * vRows; n++) {
        vTo[n] = vFrom[n] * vOther[n % vEvery];
      }
      break;
    }
    default:
      pError = "it is not an operation of the gated norm";
      return false;
  }
  pBackend.vTotals.vGatedHost++;
  return true;
}

// llama.cpp-xdna: the gated norm whose RMS_NORM is at pIndex, when its MUL by ssm_norm, the SILU of z and their MUL
// follow it in this order, each used only by the next and none an output of the graph, and right after them the
// multiplication that reads the last one (ssm_out, through reshapes) on the NPU of the prompts: the norm goes into the
// rows of that multiplication, converted ahead piece by piece along its inputs into the buffers of A from
// cXdnaRunnerFirstReadyA on (XdnaRunner::mPackGatedNormRows, the same bytes), and the four are not computed; fMulMat
// takes those rows (XdnaBackend::vPacked). pDone is false, with nothing done, when they do not make such a run;
// otherwise pLast is the index of the last of the four.
bool fGatedNorm(XdnaBackend & pBackend, const struct ggml_cgraph * pGraph, int pIndex, int & pLast, bool & pDone,
                std::string & pError) {
  pDone                 = false;
  XdnaShared & vShared = fShared();
  const int64_t cDim    = cXdnaChainRecurrenceDim;
  int           aAt[5]  = {pIndex, 0, 0, 0, 0};
  for (int k = 1; k < 5; k++) {
    aAt[k] = fNextNode(pGraph, aAt[k - 1]);
    if (aAt[k] >= pGraph->n_nodes) {
      return true;
    }
  }
  const ggml_tensor * vNorm     = pGraph->nodes[aAt[0]];
  const ggml_tensor * vWeighted = pGraph->nodes[aAt[1]];
  const ggml_tensor * vSiluZ    = pGraph->nodes[aAt[2]];
  const ggml_tensor * vOut      = pGraph->nodes[aAt[3]];
  const ggml_tensor * vMat      = pGraph->nodes[aAt[4]];
  if (vWeighted->op != GGML_OP_MUL || vSiluZ->op != GGML_OP_UNARY || vOut->op != GGML_OP_MUL ||
      vMat->op != GGML_OP_MUL_MAT || !fGatedTaken(vShared, vWeighted) || !fGatedTaken(vShared, vSiluZ) ||
      !fGatedTaken(vShared, vOut) || !(vWeighted->src[0] == vNorm || vWeighted->src[1] == vNorm) ||
      !((vOut->src[0] == vWeighted && vOut->src[1] == vSiluZ) || (vOut->src[0] == vSiluZ && vOut->src[1] == vWeighted))) {
    return true;
  }
  for (const ggml_tensor * vStep : {vNorm, vWeighted, vSiluZ}) {
    if (fUseCount(pGraph, vStep) != 1 || (vStep->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
      return true;
    }
  }
  // the multiplication reads the last one, through reshapes used only on the way
  const ggml_tensor * vRows = vMat->src[1];
  if (fThroughReshapes(vRows) != vOut || !fMulMatTaken(vShared, vMat) || !ggml_is_contiguous(vRows)) {
    return true;
  }
  for (const ggml_tensor * vStep = vRows; vStep != nullptr; vStep = vStep == vOut ? nullptr : vStep->src[0]) {
    if (fUseCount(pGraph, vStep) != 1 || (vStep->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
      return true;
    }
  }
  const uint64_t vCount  = uint64_t(vRows->ne[1]) * uint64_t(vRows->ne[2]) * uint64_t(vRows->ne[3]);
  const uint32_t vInputs = uint32_t(vMat->src[0]->ne[0]);
  if (vInputs != uint64_t(vOut->ne[0] * vOut->ne[1]) || vCount != uint64_t(vOut->ne[2]) ||
      vCount > cRowsPerCommand || fByRows(vShared, vCount, vInputs)) {
    return true;
  }
  const ggml_tensor * vWeights = vWeighted->src[0] == vNorm ? vWeighted->src[1] : vWeighted->src[0];
  const ggml_tensor * vZ       = vSiluZ->src[0];
  const ggml_tensor * vO       = vNorm->src[0];
  if (vWeights->data == nullptr || vZ->data == nullptr || vO->data == nullptr) {
    return true;
  }
  float vEps;
  memcpy(&vEps, vNorm->op_params, sizeof(float));
  const double                vStart = fNowMs();
  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
  XdnaMatrix   vTemporary;
  XdnaMatrix * vMatrix = fMatrixOf(vShared, pBackend, {vMat->src[0]}, vTemporary, pError);
  if (vMatrix == nullptr) {
    return false;
  }
  // every piece along the inputs whole tiles of K of whole heads
  const uint32_t vTileK = vShared.vOverlay.vTileK;
  bool           vFits  = vMatrix != &vTemporary && vMatrix->vRangesK <= cXdnaRunnerSlotsA - cXdnaRunnerFirstReadyA &&
                          vTileK % cDim == 0;
  for (uint32_t r = 0; vFits && r < vMatrix->vRangesK; r++) {
    const XdnaPiece & vPiece = vMatrix->aPieces[size_t(r) * vMatrix->vRangesN];
    vFits = vPiece.vFirstInput % vTileK == 0 && vPiece.vInputs % vTileK == 0;
  }
  if (!vFits) {
    if (vMatrix == &vTemporary) {
      fFreeMatrix(vShared, vTemporary);
    }
    return true;
  }
  XdnaGemmTimes  vTimes;
  const uint64_t vStride  = uint64_t(vInputs);  // a row of the outputs of the recurrence and of z: every head
  const bool     vSigmoid = ggml_get_unary_op(vSiluZ) == GGML_UNARY_OP_SIGMOID;
  for (uint32_t r = 0; r < vMatrix->vRangesK; r++) {
    const XdnaPiece & vPiece = vMatrix->aPieces[size_t(r) * vMatrix->vRangesN];
    if (!vShared.vRunner.mPackGatedNormRows(cXdnaRunnerFirstReadyA + r, uint32_t(vCount), vPiece.vInputs,
                                            vPiece.vFirstInput, static_cast<const float *>(vO->data), vStride,
                                            static_cast<const float *>(vZ->data), vStride,
                                            static_cast<const float *>(vWeights->data), uint32_t(cDim), vEps,
                                            cRounding, vTimes, pError, vSigmoid)) {
      return false;
    }
  }
  pBackend.vPacked.vActive = true;
  pBackend.vPacked.vRows   = vRows;
  pBackend.vTotals.vPackMs += vTimes.vPackMs + vTimes.vFlushMs;
  pBackend.vTotals.vGatedPackMs  += vTimes.vPackMs;
  pBackend.vTotals.vGatedFlushMs += vTimes.vFlushMs;
  pBackend.vTotals.vGatedPacked++;
  pBackend.vTotals.vGatedMs += fNowMs() - vStart;
  pBackend.vTotals.vTotalMs += fNowMs() - vStart;
  pLast = aAt[3];
  pDone = true;
  return true;
}

//
// Mixture of experts (GGML_OP_MUL_MAT_ID)
//

XdnaGemvFormat fMoeFormatOf(const XdnaShared & pShared, const ggml_tensor * pWeights) {
  XdnaGemvFormat vFormat = XdnaGemvFormat::cBfp16;
  if (pWeights->type == GGML_TYPE_Q4_K) {
    vFormat = XdnaGemvFormat::cQ4K;
  } else if (pWeights->type == GGML_TYPE_Q5_K) {
    vFormat = XdnaGemvFormat::cQ5K;
  } else if (pWeights->type == GGML_TYPE_Q6_K) {
    vFormat = XdnaGemvFormat::cQ6K;
  } else if (pWeights->type == GGML_TYPE_Q8_0) {
    vFormat = XdnaGemvFormat::cQ8_0;
  } else if (pWeights->type == GGML_TYPE_Q5_1 || pWeights->type == GGML_TYPE_Q5_0) {
    vFormat = XdnaGemvFormat::cQ5_1;
  }
  const bool    vSupported = pShared.vSettings.vEmulate || pShared.vMoeOverlay.mTakes(vFormat);
  const int64_t vStepK =
      (vFormat == XdnaGemvFormat::cQ8_0 || vFormat == XdnaGemvFormat::cQ5_1) ? 32 : 256;
  if (vFormat != XdnaGemvFormat::cBfp16 && (!vSupported || pWeights->ne[0] % vStepK != 0)) {
    vFormat = XdnaGemvFormat::cBfp16;
  }
  return vFormat;
}

std::string fMoeLayerKey(const char * pName) {
  if (std::strncmp(pName, "blk.", 4) == 0) {
    const char * vDot = std::strchr(pName + 4, '.');
    if (vDot != nullptr) {
      return std::string(pName, size_t(vDot - pName + 1));
    }
  }
  return std::string(pName);
}

int fMoeLayerIndex(const char * pName) {
  if (std::strncmp(pName, "blk.", 4) != 0) {
    return -1;
  }
  char *     vEnd = nullptr;
  const long vIdx = std::strtol(pName + 4, &vEnd, 10);
  if (vEnd == pName + 4 || vEnd == nullptr || *vEnd != '.' || vIdx < 0 || vIdx > 65535) {
    return -1;
  }
  return int(vIdx);
}

bool fPlaceMoeWeights(const XdnaShared & pShared, const ggml_tensor * pWeights, uint64_t pBytesW) {
  if (pShared.vSettings.vMoeWeightsLimit == 0) {
    return true;
  }
  XdnaPlacement &             vPlacement = fPlacement();
  std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
  const std::string vKey = std::string(pWeights->name) + "|" + std::to_string(int(pWeights->type)) + "|" +
                           std::to_string(pWeights->ne[0]) + "|" + std::to_string(pWeights->ne[1]) + "|" +
                           std::to_string(pWeights->ne[2]);
  const auto vAt = vPlacement.dTaken.find(vKey);
  if (vAt != vPlacement.dTaken.end()) {
    return vAt->second;
  }
  // when GGML_XDNA_MOE_MIN_TOKENS > 1 or GGML_XDNA_COPY_WEIGHTS is set, MADV_DONTNEED cannot drop the mapped file
  // pages of the experts, so under the default automatic budget each taken expert tensor counts both its NPU copy and
  // its source pages in RAM; when MADV_DONTNEED drops those pages (default), vMoeWeightsLimit allows up to 76 % of
  // host memory across total NPU buffers so all MoE layers fit on the NPU
  const bool        vKeepsSource  = pShared.vSettings.vMoeMinTokens > 1 || pShared.vSettings.vCopyWeights;
  const uint64_t    vCountedBytes = !pShared.vSettings.vWeightsLimitExplicit && vKeepsSource ? 2 * pBytesW : pBytesW;
  const bool        vNvmeCache    = !pShared.vSettings.vWeightsLimitExplicit && fFlag("GGML_XDNA_NVME_PARTIAL_CACHE");
  const uint64_t    vLimit        = vNvmeCache ? (pShared.vSettings.vMoeWeightsLimit * 84) / 100 :
                                                 pShared.vSettings.vMoeWeightsLimit;
  const uint64_t    vHeadroom     = pShared.vSettings.vWeightsLimitExplicit ?
                                        std::min<uint64_t>(2500000000ull, vLimit / 8) :
                                    vLimit >= 45000000000ull ? 4600000000ull :
                                    vLimit >= 30500000000ull ? 1600000000ull : 1900000000ull;
  const std::string vLayer        = fMoeLayerKey(pWeights->name);
  auto              vLayerAt      = vPlacement.dMoeLayer.find(vLayer);
  if (vLayerAt == vPlacement.dMoeLayer.end()) {
    const int      vMtpFirst      = int(std::strtol(fText("GGML_XDNA_MTP_FIRST_LAYER", "0").c_str(), nullptr, 10));
    const int      vMtpCount      = int(std::strtol(fText("GGML_XDNA_MTP_LAYERS", "0").c_str(), nullptr, 10));
    const int      vLayerIdx      = fMoeLayerIndex(pWeights->name);
    const bool     vHasMtpRes     = vMtpFirst > 0 && vMtpCount > 0;
    const bool     vIsMtpLayer    = vHasMtpRes && vLayerIdx >= vMtpFirst;
    const uint64_t vSoFar         = vPlacement.vPrompt + vPlacement.vMoe +
                                    (vPlacement.vPartial.load() ? vPlacement.vLogits : vPlacement.vChain);
    const uint64_t vLayerGuess    = 3 * vCountedBytes;
    const uint64_t vTensorElems   = uint64_t(pWeights->ne[0]) * uint64_t(pWeights->ne[1]) * uint64_t(pWeights->ne[2]);
    const uint64_t vQ8LayerBytes  = (3ull * vTensorElems * 17ull) / 16ull;
    const uint64_t vMtpLayerUpper = std::max(vLayerGuess, vKeepsSource && !pShared.vSettings.vWeightsLimitExplicit ?
                                                              2 * vQ8LayerBytes :
                                                              vQ8LayerBytes) +
                                    350000000ull;
    const uint64_t vNonMoeSoFar   = vSoFar > vPlacement.vMoe ? vSoFar - vPlacement.vMoe : 0ull;
    const uint64_t vDenseLayers   = vNonMoeSoFar > 2 * vPlacement.vLogits ? vNonMoeSoFar - 2 * vPlacement.vLogits : 0ull;
    const uint64_t vRemDense      = (vHasMtpRes && vLayerIdx >= 0 && vLayerIdx + 1 < vMtpFirst) ?
                                        (vDenseLayers / uint64_t(vLayerIdx + 1)) *
                                            uint64_t(vMtpFirst - (vLayerIdx + 1)) :
                                        0ull;
    const uint64_t vMtpReserve    = (vHasMtpRes && vLayerIdx >= 0 && vLayerIdx < vMtpFirst) ?
                                        vRemDense + uint64_t(vMtpCount) * vMtpLayerUpper :
                                        0ull;
    const uint64_t vEffHeadroom   = vIsMtpLayer ? std::min<uint64_t>(vHeadroom, 400000000ull) :
                                    vHasMtpRes  ? std::min<uint64_t>(vHeadroom, 800000000ull) :
                                                  vHeadroom;
    const bool     vLayerFits     = (vIsMtpLayer || !vPlacement.vFullMoe) && !vPlacement.vFull &&
                                    vLimit > vEffHeadroom && vSoFar + vLayerGuess + vMtpReserve <= vLimit - vEffHeadroom;
    vLayerAt = vPlacement.dMoeLayer.emplace(vLayer, vLayerFits).first;
    if (!vLayerFits && !vIsMtpLayer && !vPlacement.vFullMoe) {
      vPlacement.vFullMoe = true;
      GGML_LOG_INFO("%s: the routed experts from %s on stay with the processor so the copies for the NPU stay within "
                    "%.1f GB while keeping all dense projections%s on the NPU (GGML_XDNA_WEIGHTS_GB sets that memory)\n",
                    cName, pWeights->name, double(vLimit) * 1e-9, vHasMtpRes ? " and the MTP block" : "");
    }
  }
  const bool vTaken = vLayerAt->second;
  if (vTaken) {
    vPlacement.vMoe += vCountedBytes;
  } else {
    vPlacement.vRefused++;
  }
  vPlacement.dTaken[vKey] = vTaken;
  return vTaken;
}

bool fMoeTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pNode == nullptr || pNode->op != GGML_OP_MUL_MAT_ID || pShared.vSettings.vNoMoe ||
      (!pShared.vHasMoe && !pShared.vSettings.vEmulate)) {
    return false;
  }
  if (ggml_get_op_params_i32(pNode, 1) != GGML_HINT_NONE) {
    return false;
  }
  const enum ggml_prec vPrecision = ggml_prec_get_src(pNode, 1);
  if (vPrecision != GGML_PREC_UNDEFINED && vPrecision != GGML_PREC_Q8 && vPrecision != GGML_PREC_Q4) {
    return false;
  }
  const ggml_tensor * vWeights = pNode->src[0];
  const ggml_tensor * vRows    = pNode->src[1];
  const ggml_tensor * vIds     = pNode->src[2];
  if (vWeights == nullptr || vRows == nullptr || vIds == nullptr) {
    return false;
  }
  if (vWeights->op != GGML_OP_NONE || vWeights->view_src != nullptr || vWeights->buffer == nullptr ||
      ggml_backend_buffer_get_type(vWeights->buffer) != fBufferType()) {
    return false;
  }
  const int64_t vInputs  = vWeights->ne[0];
  const int64_t vOutputs = vWeights->ne[1];
  const int64_t vExperts = vWeights->ne[2];
  if (vWeights->ne[3] != 1 || vInputs < 1 || vInputs > cLargestSide || vOutputs < 1 || vOutputs > cLargestSide ||
      vExperts <= 1 || vExperts > 65535) {
    return false;
  }
  if (vWeights->type < 0 || vWeights->type >= GGML_TYPE_COUNT) {
    return false;
  }
  const struct ggml_type_traits * vTraits = ggml_get_type_traits(vWeights->type);
  if (vWeights->type != GGML_TYPE_F32 && vTraits->to_float == nullptr) {
    return false;
  }
  if (vTraits->blck_size < 1 || vInputs % vTraits->blck_size != 0 || vWeights->nb[0] != vTraits->type_size ||
      vWeights->nb[1] != ggml_row_size(vWeights->type, vInputs) ||
      vWeights->nb[2] != vWeights->nb[1] * size_t(vOutputs)) {
    return false;
  }
  if (vIds->type != GGML_TYPE_I32 || vIds->nb[0] != sizeof(int32_t) || vIds->ne[0] < 1 || vIds->ne[1] < 1 ||
      vIds->ne[2] != 1 || vIds->ne[3] != 1) {
    return false;
  }
  const int64_t vUsed   = vIds->ne[0];
  const int64_t vTokens = vIds->ne[1];
  if (vRows->type != GGML_TYPE_F32 || vRows->nb[0] != sizeof(float) || vRows->ne[0] != vInputs ||
      (vRows->ne[1] != 1 && vRows->ne[1] != vUsed) || vRows->ne[2] != vTokens || vRows->ne[3] != 1) {
    return false;
  }
  if (pNode->type != GGML_TYPE_F32 || pNode->nb[0] != sizeof(float) || !ggml_is_contiguous(pNode) ||
      pNode->ne[0] != vOutputs || pNode->ne[1] != vUsed || pNode->ne[2] != vTokens || pNode->ne[3] != 1) {
    return false;
  }
  uint64_t vBytesW = ggml_nbytes(vWeights);
  if (!pShared.vSettings.vEmulate) {
    const XdnaGemvPart vPart = {fMoeFormatOf(pShared, vWeights), uint32_t(vOutputs)};
    XdnaMoeMatrix      vMatrix;
    std::string        vError;
    if (!fXdnaMoeMatrixShape(pShared.vMoeOverlay, uint32_t(vExperts), uint32_t(vInputs), &vPart, 1, false, vMatrix,
                             vError)) {
      return false;
    }
    vBytesW = vMatrix.vBytesW;
  }
  const bool vPlacing = ggml_backend_buffer_get_size(vWeights->buffer) == 0;
  if (!vPlacing && uint64_t(vTokens) < pShared.vSettings.vMoeMinTokens) {
    return false;
  }
  return !vPlacing || fPlaceMoeWeights(pShared, vWeights, vBytesW);
}

const ggml_tensor * fMoeWeightedMulBase(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pNode == nullptr || pNode->op != GGML_OP_MUL || pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode) ||
      pNode->ne[3] != 1) {
    return nullptr;
  }
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (vA == nullptr || vB == nullptr || vA->type != GGML_TYPE_F32 || vB->type != GGML_TYPE_F32 ||
      !ggml_is_contiguous(vA) || !ggml_are_same_shape(vA, pNode) || vB->ne[0] != 1 || vB->ne[1] != pNode->ne[1] ||
      vB->ne[2] != pNode->ne[2] || vB->ne[3] != 1 || vB->nb[0] != sizeof(float)) {
    return nullptr;
  }
  if (vA->op == GGML_OP_MUL_MAT_ID && fMoeTaken(pShared, vA)) {
    return vA;
  }
  if (vA->op == GGML_OP_MUL) {
    return fMoeWeightedMulBase(pShared, vA);
  }
  return nullptr;
}

const ggml_tensor * fMoeExpertSlotView(const XdnaShared & pShared, const ggml_tensor * pView, int64_t & pSlot) {
  if (pView == nullptr || pView->op != GGML_OP_VIEW || pView->src[0] == nullptr || pView->type != GGML_TYPE_F32 ||
      pView->ne[2] != 1 || pView->ne[3] != 1 || pView->nb[0] != sizeof(float)) {
    return nullptr;
  }
  const ggml_tensor * vWeighted = pView->src[0];
  if (fMoeWeightedMulBase(pShared, vWeighted) == nullptr || pView->ne[0] != vWeighted->ne[0] ||
      pView->ne[1] != vWeighted->ne[2] || pView->nb[1] != vWeighted->nb[2] || vWeighted->nb[1] == 0 ||
      pView->view_offs % vWeighted->nb[1] != 0) {
    return nullptr;
  }
  const int64_t vSlot = int64_t(pView->view_offs / vWeighted->nb[1]);
  if (vSlot < 0 || vSlot >= vWeighted->ne[1]) {
    return nullptr;
  }
  pSlot = vSlot;
  return vWeighted;
}

const ggml_tensor * fMoeAmaxChainGlu(const ggml_tensor * pNode) {
  const ggml_tensor * vCur = pNode;
  for (int vStep = 0; vStep < 10 && vCur != nullptr; vStep++) {
    if (vCur->op == GGML_OP_GLU) {
      return vStep > 0 ? vCur : nullptr;
    }
    if (vCur->type != GGML_TYPE_F32 || !ggml_is_contiguous(vCur)) {
      return nullptr;
    }
    if (vCur->op == GGML_OP_UNARY) {
      const enum ggml_unary_op vOp = ggml_get_unary_op(vCur);
      if (vOp != GGML_UNARY_OP_ABS && vOp != GGML_UNARY_OP_CEIL && vOp != GGML_UNARY_OP_EXP) {
        return nullptr;
      }
      vCur = vCur->src[0];
    } else if (vCur->op == GGML_OP_POOL_1D) {
      if (ggml_get_op_params_i32(vCur, 0) != GGML_OP_POOL_MAX || vCur->ne[0] != 1) {
        return nullptr;
      }
      vCur = vCur->src[0];
    } else if (vCur->op == GGML_OP_LOG || vCur->op == GGML_OP_SCALE || vCur->op == GGML_OP_CLAMP) {
      vCur = vCur->src[0];
    } else {
      return nullptr;
    }
  }
  return nullptr;
}

// llama.cpp-xdna: whether pNode belongs to the build_amax_src1_f16 rescaling sequence between the SwiGLU and down_exps
// of Flint on a layer whose routed experts run on xdna2-moe
bool fMoeAmaxTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vNoFuse || pShared.vSettings.vNoMoeDense || pShared.vSettings.vEmulate || pNode == nullptr ||
      pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode)) {
    return false;
  }
  const ggml_tensor * vGlu = nullptr;
  if (pNode->op == GGML_OP_DIV) {
    if (pNode->src[0] == nullptr || pNode->src[1] == nullptr || pNode->src[0]->op != GGML_OP_GLU ||
        !ggml_are_same_shape(pNode->src[0], pNode)) {
      return false;
    }
    vGlu = fMoeAmaxChainGlu(pNode->src[1]);
    if (vGlu != pNode->src[0]) {
      return false;
    }
  } else {
    vGlu = fMoeAmaxChainGlu(pNode);
  }
  return vGlu != nullptr && vGlu->src[0] != nullptr && vGlu->src[0]->op == GGML_OP_MUL_MAT_ID && fGluTaken(vGlu);
}

bool fMoeAmaxOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  if (vA == nullptr || vA->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor of the expert amax scale has no memory";
    return false;
  }
  const float *  vFrom  = static_cast<const float *>(vA->data);
  float *        vTo    = static_cast<float *>(pNode->data);
  const uint64_t vCount = uint64_t(ggml_nelements(pNode));
  if (pNode->op == GGML_OP_UNARY) {
    const enum ggml_unary_op vOp = ggml_get_unary_op(pNode);
    for (uint64_t i = 0; i < vCount; i++) {
      const float vVal = vFrom[i];
      vTo[i]           = vOp == GGML_UNARY_OP_ABS  ? std::fabs(vVal) :
                         vOp == GGML_UNARY_OP_CEIL ? std::ceil(vVal) : std::exp(vVal);
    }
  } else if (pNode->op == GGML_OP_LOG) {
    for (uint64_t i = 0; i < vCount; i++) {
      vTo[i] = std::log(vFrom[i]);
    }
  } else if (pNode->op == GGML_OP_POOL_1D) {
    const uint64_t vLen  = uint64_t(vA->ne[0]);
    const uint64_t vRows = uint64_t(ggml_nrows(vA));
    for (uint64_t r = 0; r < vRows; r++) {
      const float * vRow = vFrom + r * vLen;
      float         vMax = vRow[0];
      for (uint64_t i = 1; i < vLen; i++) {
        vMax = std::max(vMax, vRow[i]);
      }
      vTo[r] = vMax;
    }
  } else if (pNode->op == GGML_OP_SCALE) {
    const float vFactor = ggml_get_op_params_f32(pNode, 0);
    const float vBias   = ggml_get_op_params_f32(pNode, 1);
    for (uint64_t i = 0; i < vCount; i++) {
      vTo[i] = vFrom[i] * vFactor + vBias;
    }
  } else if (pNode->op == GGML_OP_CLAMP) {
    const float vMin = ggml_get_op_params_f32(pNode, 0);
    const float vMax = ggml_get_op_params_f32(pNode, 1);
    for (uint64_t i = 0; i < vCount; i++) {
      vTo[i] = std::min(vMax, std::max(vMin, vFrom[i]));
    }
  } else if (pNode->op == GGML_OP_DIV) {
    const ggml_tensor * vB = pNode->src[1];
    if (vB == nullptr || vB->data == nullptr) {
      pError = "a tensor of the expert amax division has no memory";
      return false;
    }
    const uint64_t vCols = uint64_t(pNode->ne[0]);
    const uint64_t vRows = uint64_t(ggml_nrows(pNode));
    const float *  vDiv  = static_cast<const float *>(vB->data);
    for (uint64_t r = 0; r < vRows; r++) {
      const float vInv = 1.0f / vDiv[r];
      for (uint64_t c = 0; c < vCols; c++) {
        vTo[r * vCols + c] = vFrom[r * vCols + c] * vInv;
      }
    }
  } else {
    pError = "unexpected expert amax operation";
    return false;
  }
  pBackend.vTotals.vMoeTailHost++;
  return true;
}

bool fSharedGateOpOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  if (vA == nullptr || vA->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor of the shared-expert gate has no memory";
    return false;
  }
  float * vTo = static_cast<float *>(pNode->data);
  if (pNode->op == GGML_OP_UNARY) {
    const float *  vFrom  = static_cast<const float *>(vA->data);
    const uint64_t vCount = uint64_t(ggml_nelements(pNode));
    for (uint64_t i = 0; i < vCount; i++) {
      vTo[i] = 1.0f / (1.0f + std::exp(-vFrom[i]));
    }
    return true;
  }
  if (pNode->op == GGML_OP_MUL) {
    const ggml_tensor * vB = pNode->src[1];
    if (vB == nullptr || vB->data == nullptr) {
      pError = "a tensor of the shared-expert gate has no memory";
      return false;
    }
    const ggml_tensor * vDown = vA->ne[0] == 1 && vB->ne[0] > 1 ? vB : vA;
    const ggml_tensor * vGate = vDown == vA ? vB : vA;
    const uint32_t      vCols = uint32_t(pNode->ne[0]);
    const uint32_t      vRows = uint32_t(ggml_nrows(pNode));
    const float *       vSrc  = static_cast<const float *>(vDown->data);
    const float *       vSig  = static_cast<const float *>(vGate->data);
    auto                fRow  = [&](uint32_t pRow) {
      const float   vScale = vSig[pRow];
      const float * vIn    = vSrc + size_t(pRow) * vCols;
      float *       vOut   = vTo + size_t(pRow) * vCols;
      for (uint32_t c = 0; c < vCols; c++) {
        vOut[c] = vIn[c] * vScale;
      }
    };
    if (uint64_t(vRows) * vCols >= 16384) {
      XdnaShared &                vShared = fShared();
      std::lock_guard<std::mutex> vLock(vShared.vMutex);
      vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
      vShared.vRunner.mWorkers().mForEach(vRows, fRow);
    } else {
      for (uint32_t r = 0; r < vRows; r++) {
        fRow(r);
      }
    }
    return true;
  }
  pError = "unexpected shared-expert gate operation";
  return false;
}

bool fMoeTailTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vNoFuse || pNode == nullptr) {
    return false;
  }
  if (pNode->op == GGML_OP_MUL) {
    return fMoeWeightedMulBase(pShared, pNode) != nullptr;
  }
  if (pNode->op == GGML_OP_ADD) {
    if (pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode) || pNode->ne[2] != 1 || pNode->ne[3] != 1) {
      return false;
    }
    int64_t             vRightSlot = -1;
    const ggml_tensor * vWeighted  = fMoeExpertSlotView(pShared, pNode->src[1], vRightSlot);
    if (vWeighted == nullptr || vRightSlot < 1 || pNode->ne[0] != vWeighted->ne[0] || pNode->ne[1] != vWeighted->ne[2]) {
      return false;
    }
    const ggml_tensor * vCur = pNode->src[0];
    for (int64_t vExpected = vRightSlot - 1; vExpected >= 1; vExpected--) {
      if (vCur == nullptr || vCur->op != GGML_OP_ADD || vCur->type != GGML_TYPE_F32 || !ggml_is_contiguous(vCur) ||
          !ggml_are_same_shape(vCur, pNode)) {
        return false;
      }
      int64_t vSlot = -1;
      if (fMoeExpertSlotView(pShared, vCur->src[1], vSlot) != vWeighted || vSlot != vExpected) {
        return false;
      }
      vCur = vCur->src[0];
    }
    int64_t vZeroSlot = -1;
    return fMoeExpertSlotView(pShared, vCur, vZeroSlot) == vWeighted && vZeroSlot == 0;
  }
  return false;
}

bool fMoeTailOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  if (vA == nullptr || vB == nullptr || vA->data == nullptr || vB->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor of the expert tail has no memory";
    return false;
  }
  const uint32_t vCols = uint32_t(pNode->ne[0]);
  if (pNode->op == GGML_OP_MUL) {
    const uint32_t vUsed   = uint32_t(pNode->ne[1]);
    const uint32_t vTokens = uint32_t(pNode->ne[2]);
    const uint32_t vRows   = vUsed * vTokens;
    const float *  vFrom   = static_cast<const float *>(vA->data);
    float *        vTo     = static_cast<float *>(pNode->data);
    auto           fRow    = [&](uint32_t pRow) {
      const uint32_t vSlot  = pRow % vUsed;
      const uint32_t vToken = pRow / vUsed;
      const float    vScale = *reinterpret_cast<const float *>(static_cast<const char *>(vB->data) +
                                                               size_t(vSlot) * vB->nb[1] + size_t(vToken) * vB->nb[2]);
      const float *  vSrc   = vFrom + size_t(pRow) * vCols;
      float *        vDst   = vTo + size_t(pRow) * vCols;
      for (uint32_t i = 0; i < vCols; i++) {
        vDst[i] = vSrc[i] * vScale;
      }
    };
    if (uint64_t(vRows) * vCols >= 16384) {
      XdnaShared &                vShared = fShared();
      std::lock_guard<std::mutex> vLock(vShared.vMutex);
      vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
      vShared.vRunner.mWorkers().mForEach(vRows, fRow);
    } else {
      for (uint32_t r = 0; r < vRows; r++) {
        fRow(r);
      }
    }
    pBackend.vTotals.vMoeTailHost++;
    return true;
  }
  if (pNode->op == GGML_OP_ADD) {
    const uint32_t vTokens = uint32_t(pNode->ne[1]);
    float *        vTo     = static_cast<float *>(pNode->data);
    auto           fTok    = [&](uint32_t pToken) {
      const float * vRowA = reinterpret_cast<const float *>(static_cast<const char *>(vA->data) + size_t(pToken) * vA->nb[1]);
      const float * vRowB = reinterpret_cast<const float *>(static_cast<const char *>(vB->data) + size_t(pToken) * vB->nb[1]);
      float *       vDst  = vTo + size_t(pToken) * vCols;
      for (uint32_t i = 0; i < vCols; i++) {
        vDst[i] = vRowA[i] + vRowB[i];
      }
    };
    if (uint64_t(vTokens) * vCols >= 16384) {
      XdnaShared &                vShared = fShared();
      std::lock_guard<std::mutex> vLock(vShared.vMutex);
      vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
      vShared.vRunner.mWorkers().mForEach(vTokens, fTok);
    } else {
      for (uint32_t t = 0; t < vTokens; t++) {
        fTok(t);
      }
    }
    pBackend.vTotals.vMoeTailHost++;
    return true;
  }
  pError = "unexpected expert tail operation";
  return false;
}

bool fIsReadOnlySharedMapping(const void * pData, size_t pSize) {
  if (pData == nullptr || pSize == 0) {
    return false;
  }
  std::ifstream vMaps("/proc/self/maps");
  if (!vMaps) {
    return false;
  }
  const uintptr_t vStart = reinterpret_cast<uintptr_t>(pData);
  const uintptr_t vEnd   = vStart + pSize;
  if (vEnd < vStart) {
    return false;
  }
  std::string vLine;
  while (std::getline(vMaps, vLine)) {
    unsigned long long vLow  = 0;
    unsigned long long vHigh = 0;
    char               aPerms[8] = {};
    if (std::sscanf(vLine.c_str(), "%llx-%llx %7s", &vLow, &vHigh, aPerms) == 3) {
      if (vStart >= uintptr_t(vLow) && vEnd <= uintptr_t(vHigh)) {
        return std::strcmp(aPerms, "r--s") == 0;
      }
    }
  }
  return false;
}

void fReleaseMappedExpertPages(XdnaShared & pShared, const ggml_tensor * pWeights) {
  if (pWeights == nullptr || pWeights->data == nullptr) {
    return;
  }
  const uintptr_t vStart = reinterpret_cast<uintptr_t>(pWeights->data);
  const uintptr_t vEnd   = vStart + ggml_nbytes(pWeights);
  for (const auto & vEntry : pShared.dForeignMappings) {
    if (!vEntry.second.second) {
      continue;
    }
    const uintptr_t vMapStart = reinterpret_cast<uintptr_t>(vEntry.first);
    const uintptr_t vMapEnd   = vMapStart + vEntry.second.first;
    if (vStart >= vMapStart && vEnd <= vMapEnd) {
      const uintptr_t vFirstPage = (vStart + 4095u) & ~uintptr_t(4095u);
      const uintptr_t vEndPage   = vEnd & ~uintptr_t(4095u);
      if (vEndPage > vFirstPage) {
        (void) ::madvise(reinterpret_cast<void *>(vFirstPage), vEndPage - vFirstPage, MADV_DONTNEED);
      }
      return;
    }
  }
}

uint64_t fMoeSourceRowBytes(XdnaGemvFormat pFormat, uint32_t pInputs, const ggml_tensor * pTensor) {
  if (pFormat == XdnaGemvFormat::cBfp16) {
    return uint64_t(pInputs) * sizeof(float);
  }
  if (pFormat == XdnaGemvFormat::cQ5_1) {
    return uint64_t(pInputs / 32) * 24;
  }
  return uint64_t(pTensor->nb[1]);
}

bool fMoeTensorIsDirect(XdnaGemvFormat pFormat, const ggml_tensor * pTensor) {
  if (pFormat == XdnaGemvFormat::cBfp16) {
    return pTensor->type == GGML_TYPE_F32;
  }
  if (pFormat == XdnaGemvFormat::cQ5_1) {
    return pTensor->type == GGML_TYPE_Q5_1;
  }
  return true;
}

// Converts pRows rows of GGML_TYPE_Q5_0 (22 bytes per 32 values: d, qh, qs) into GGML_TYPE_Q5_1 (24 bytes per 32
// values: d, m = fp16(-16 * d), qh, qs) so xdna2-moe decodes both 5-bit formats with cQ5_1.
void fConvertRowsQ5_0ToQ5_1(const void * pSrc, size_t pSrcStride, uint32_t pRows, uint32_t pInputs, uint8_t * pDst) {
  const uint32_t vBlocks = pInputs / 32;
  for (uint32_t r = 0; r < pRows; r++) {
    const uint8_t * vIn  = static_cast<const uint8_t *>(pSrc) + size_t(r) * pSrcStride;
    uint8_t *       vOut = pDst + size_t(r) * vBlocks * 24;
    for (uint32_t b = 0; b < vBlocks; b++) {
      const uint8_t * vFrom  = vIn + size_t(b) * 22;
      uint8_t *       vTo    = vOut + size_t(b) * 24;
      uint16_t          vHalfD = 0;
      std::memcpy(&vHalfD, vFrom, 2);
      const ggml_fp16_t vHalfM = ggml_fp32_to_fp16(-16.0f * fXdnaHalfToFloat(vHalfD));
      std::memcpy(vTo, &vHalfD, 2);
      std::memcpy(vTo + 2, &vHalfM, 2);
      std::memcpy(vTo + 4, vFrom + 2, 20);
    }
  }
}

XdnaMoeWeights * fEnsureMoeDenseGroup(XdnaShared & pShared, XdnaBackend & pBackend,
                                      const std::vector<const ggml_tensor *> & aWeights, std::string & pError) {
  std::vector<const void *> aKey;
  aKey.reserve(aWeights.size() * 2);
  for (const ggml_tensor * vTensor : aWeights) {
    if (vTensor == nullptr || vTensor->data == nullptr) {
      pError = "a dense weight tensor has no memory";
      return nullptr;
    }
    fPushWeightKey(aKey, vTensor);
  }
  auto vFound = pShared.dMoeMatrices.find(aKey);
  if (vFound != pShared.dMoeMatrices.end()) {
    return &vFound->second;
  }
  if (!fOpenRunner(pShared, pError)) {
    return nullptr;
  }
  const double   vStart       = fNowMs();
  const uint32_t vInputs      = uint32_t(aWeights[0]->ne[0]);
  XdnaGemvFormat vFormat      = fMoeFormatOf(pShared, aWeights[0]);
  uint32_t       vTotalOutput = 0;
  for (const ggml_tensor * vTensor : aWeights) {
    if (fMoeFormatOf(pShared, vTensor) != vFormat) {
      vFormat = XdnaGemvFormat::cBfp16;
    }
    vTotalOutput += uint32_t(vTensor->ne[1]);
  }
  const uint64_t       vRowBytes = fMoeSourceRowBytes(vFormat, vInputs, aWeights[0]);
  XdnaMoeSegmentSource vSource;
  vSource.vFormat       = vFormat;
  vSource.vOutputs      = vTotalOutput;
  vSource.vRowBytes     = vRowBytes;
  const bool vDirect    = aWeights.size() == 1 && fMoeTensorIsDirect(vFormat, aWeights[0]);
  vSource.vNeedsScratch = !vDirect;
  vSource.fExpertRows   = [&aWeights, vFormat, vInputs, vRowBytes, vDirect](uint32_t, uint8_t * pScratch) -> const void * {
    if (vDirect) {
      return aWeights[0]->data;
    }
    uint8_t * vCursor = pScratch;
    for (const ggml_tensor * vTensor : aWeights) {
      const uint32_t vOut = uint32_t(vTensor->ne[1]);
      if (fMoeTensorIsDirect(vFormat, vTensor)) {
        const size_t vBytes = size_t(vOut) * vRowBytes;
        std::memcpy(vCursor, vTensor->data, vBytes);
        vCursor += vBytes;
      } else if (vFormat == XdnaGemvFormat::cQ5_1 && vTensor->type == GGML_TYPE_Q5_0) {
        fConvertRowsQ5_0ToQ5_1(vTensor->data, vTensor->nb[1], vOut, vInputs, vCursor);
        vCursor += size_t(vOut) * vRowBytes;
      } else {
        float *      vFloats = reinterpret_cast<float *>(vCursor);
        const char * vBase   = static_cast<const char *>(vTensor->data);
        for (uint32_t r = 0; r < vOut; r++) {
          fDequantizeWeightRow(vTensor->type, vBase + size_t(r) * vTensor->nb[1], vFloats + size_t(r) * vInputs, vInputs);
        }
        vCursor += size_t(vOut) * vInputs * sizeof(float);
      }
    }
    return pScratch;
  };
  XdnaMoeWeights vLoaded;
  if (!pShared.vRunner.mLoadMoeMatrix(1, vInputs, &vSource, 1, false, vLoaded, pError)) {
    return nullptr;
  }
  pBackend.vTotals.vMoeMatrices++;
  pBackend.vTotals.vMoeWeightsMs += fNowMs() - vStart;
  if (pShared.vSettings.vMoeMinTokens <= 1) {
    fNoteDenseCached(pShared, aWeights, false);
  }
  return &pShared.dMoeMatrices.emplace(std::move(aKey), std::move(vLoaded)).first->second;
}

XdnaMoeWeights * fEnsureMoeDensePair(XdnaShared & pShared, XdnaBackend & pBackend, const ggml_tensor * pGate,
                                     const ggml_tensor * pUp, std::string & pError) {
  if (pGate == nullptr || pUp == nullptr || pGate->data == nullptr || pUp->data == nullptr) {
    pError = "a dense feed-forward tensor has no memory";
    return nullptr;
  }
  std::vector<const void *> aKey   = {pGate->data, pUp->data};
  auto                      vFound = pShared.dMoeMatrices.find(aKey);
  if (vFound != pShared.dMoeMatrices.end()) {
    return &vFound->second;
  }
  if (!fOpenRunner(pShared, pError)) {
    return nullptr;
  }
  const double         vStart      = fNowMs();
  const uint32_t       vInputs     = uint32_t(pGate->ne[0]);
  const ggml_tensor *  aPair[2]    = {pGate, pUp};
  XdnaMoeSegmentSource aSources[2];
  for (int s = 0; s < 2; s++) {
    const ggml_tensor *    vTensor = aPair[s];
    const XdnaGemvFormat   vFormat = fMoeFormatOf(pShared, vTensor);
    const uint32_t         vOut    = uint32_t(vTensor->ne[1]);
    const bool             vDirect = fMoeTensorIsDirect(vFormat, vTensor);
    XdnaMoeSegmentSource & vSource = aSources[s];
    vSource.vFormat                = vFormat;
    vSource.vOutputs               = vOut;
    vSource.vRowBytes              = fMoeSourceRowBytes(vFormat, vInputs, vTensor);
    vSource.vNeedsScratch          = !vDirect;
    vSource.fExpertRows = [vTensor, vFormat, vDirect, vOut, vInputs](uint32_t, uint8_t * pScratch) -> const void * {
      if (vDirect) {
        return vTensor->data;
      }
      if (vFormat == XdnaGemvFormat::cQ5_1 && vTensor->type == GGML_TYPE_Q5_0) {
        fConvertRowsQ5_0ToQ5_1(vTensor->data, vTensor->nb[1], vOut, vInputs, pScratch);
        return pScratch;
      }
      float *      vFloats = reinterpret_cast<float *>(pScratch);
      const char * vBase   = static_cast<const char *>(vTensor->data);
      for (uint32_t r = 0; r < vOut; r++) {
        fDequantizeWeightRow(vTensor->type, vBase + size_t(r) * vTensor->nb[1], vFloats + size_t(r) * vInputs, vInputs);
      }
      return vFloats;
    };
  }
  XdnaMoeWeights vLoaded;
  if (!pShared.vRunner.mLoadMoeMatrix(1, vInputs, aSources, 2, true, vLoaded, pError)) {
    return nullptr;
  }
  pBackend.vTotals.vMoeMatrices++;
  pBackend.vTotals.vMoeWeightsMs += fNowMs() - vStart;
  if (pShared.vSettings.vMoeMinTokens <= 1) {
    fNoteDenseCached(pShared, {pGate, pUp}, false);
  }
  return &pShared.dMoeMatrices.emplace(std::move(aKey), std::move(vLoaded)).first->second;
}

XdnaMoeWeights * fEnsureMoeWithShared(XdnaShared & pShared, XdnaBackend & pBackend,
                                      const std::vector<const ggml_tensor *> & aExps,
                                      const std::vector<const ggml_tensor *> & aShexp, uint32_t pVirtualCount,
                                      bool pSwiglu, std::string & pError) {
  std::vector<const void *> aKey;
  aKey.reserve(aExps.size() + aShexp.size());
  for (const ggml_tensor * vTensor : aExps) {
    if (vTensor == nullptr || vTensor->data == nullptr) {
      pError = "an expert weight tensor has no memory";
      return nullptr;
    }
    aKey.push_back(vTensor->data);
  }
  for (const ggml_tensor * vTensor : aShexp) {
    if (vTensor == nullptr || vTensor->data == nullptr) {
      pError = "a shared-expert weight tensor has no memory";
      return nullptr;
    }
    aKey.push_back(vTensor->data);
  }
  auto vFound = pShared.dMoeMatrices.find(aKey);
  if (vFound != pShared.dMoeMatrices.end()) {
    return &vFound->second;
  }
  if (aShexp.empty()) {
    const uint32_t vBaseExps = uint32_t(aExps[0]->ne[2]);
    for (auto & vEntry : pShared.dMoeMatrices) {
      if (vEntry.first.size() > aKey.size() && vEntry.second.vMatrix.vSegments == aKey.size() &&
          vEntry.second.vMatrix.vSwiglu == pSwiglu && vEntry.second.vMatrix.vExperts > vBaseExps &&
          std::equal(aKey.begin(), aKey.end(), vEntry.first.begin())) {
        return &vEntry.second;
      }
    }
  }
  if (!fOpenRunner(pShared, pError)) {
    return nullptr;
  }
  const double   vStart       = fNowMs();
  const uint32_t vBaseExperts = uint32_t(aExps[0]->ne[2]);
  const uint32_t vTotalExps   = vBaseExperts + pVirtualCount;
  const uint32_t vInputs      = uint32_t(aExps[0]->ne[0]);
  XdnaMoeSegmentSource aSources[cXdnaMoeSegmentsMax];
  for (size_t s = 0; s < aExps.size(); s++) {
    const ggml_tensor *    vTensor      = aExps[s];
    const ggml_tensor *    vShexpTensor = pVirtualCount > 0 ? aShexp[s] : nullptr;
    const XdnaGemvFormat   vFormat      = fMoeFormatOf(pShared, vTensor);
    const uint32_t         vOut         = uint32_t(vTensor->ne[1]);
    const uint64_t         vRowBytes    = fMoeSourceRowBytes(vFormat, vInputs, vTensor);
    const bool             vSliceDown   = pVirtualCount > 1 && !pSwiglu;
    XdnaMoeSegmentSource & vSource      = aSources[s];
    vSource.vFormat                     = vFormat;
    vSource.vOutputs                    = vOut;
    vSource.vRowBytes                   = vRowBytes;
    vSource.vNeedsScratch               = !fMoeTensorIsDirect(vFormat, vTensor) ||
                                          (vShexpTensor != nullptr && !fMoeTensorIsDirect(vFormat, vShexpTensor)) ||
                                          vSliceDown;
    vSource.fExpertRows = [vTensor, vShexpTensor, vBaseExperts, pSwiglu, vFormat, vOut, vInputs, vRowBytes](
                              uint32_t pExpert, uint8_t * pScratch) -> const void * {
      if (pExpert < vBaseExperts) {
        const char * vExpertBase = static_cast<const char *>(vTensor->data) + size_t(pExpert) * vTensor->nb[2];
        if (fMoeTensorIsDirect(vFormat, vTensor)) {
          return vExpertBase;
        }
        if (vFormat == XdnaGemvFormat::cQ5_1 && vTensor->type == GGML_TYPE_Q5_0) {
          fConvertRowsQ5_0ToQ5_1(vExpertBase, vTensor->nb[1], vOut, vInputs, pScratch);
          return pScratch;
        }
        float * vFloats = reinterpret_cast<float *>(pScratch);
        for (uint32_t r = 0; r < vOut; r++) {
          fDequantizeWeightRow(vTensor->type, vExpertBase + size_t(r) * vTensor->nb[1], vFloats + size_t(r) * vInputs,
                               vInputs);
        }
        return vFloats;
      }
      const uint32_t vSub       = pExpert - vBaseExperts;
      const char *   vShexpBase = static_cast<const char *>(vShexpTensor->data);
      if (pSwiglu) {
        const char * vSlice = vShexpBase + size_t(vSub * vOut) * vShexpTensor->nb[1];
        if (fMoeTensorIsDirect(vFormat, vShexpTensor)) {
          return vSlice;
        }
        if (vFormat == XdnaGemvFormat::cQ5_1 && vShexpTensor->type == GGML_TYPE_Q5_0) {
          fConvertRowsQ5_0ToQ5_1(vSlice, vShexpTensor->nb[1], vOut, vInputs, pScratch);
          return pScratch;
        }
        float * vFloats = reinterpret_cast<float *>(pScratch);
        for (uint32_t r = 0; r < vOut; r++) {
          fDequantizeWeightRow(vShexpTensor->type, vSlice + size_t(r) * vShexpTensor->nb[1],
                               vFloats + size_t(r) * vInputs, vInputs);
        }
        return vFloats;
      }
      if (uint64_t(vShexpTensor->ne[0]) == uint64_t(vInputs)) {
        if (fMoeTensorIsDirect(vFormat, vShexpTensor)) {
          return vShexpBase;
        }
        if (vFormat == XdnaGemvFormat::cQ5_1 && vShexpTensor->type == GGML_TYPE_Q5_0) {
          fConvertRowsQ5_0ToQ5_1(vShexpBase, vShexpTensor->nb[1], vOut, vInputs, pScratch);
          return pScratch;
        }
        float * vFloats = reinterpret_cast<float *>(pScratch);
        for (uint32_t r = 0; r < vOut; r++) {
          fDequantizeWeightRow(vShexpTensor->type, vShexpBase + size_t(r) * vShexpTensor->nb[1],
                               vFloats + size_t(r) * vInputs, vInputs);
        }
        return vFloats;
      }
      if (vFormat != XdnaGemvFormat::cBfp16) {
        if (vFormat == XdnaGemvFormat::cQ5_1 && vShexpTensor->type == GGML_TYPE_Q5_0) {
          const size_t vSubSrcOff = size_t(vSub * (vInputs / 32)) * 22;
          fConvertRowsQ5_0ToQ5_1(vShexpBase + vSubSrcOff, vShexpTensor->nb[1], vOut, vInputs, pScratch);
          return pScratch;
        }
        for (uint32_t r = 0; r < vOut; r++) {
          std::memcpy(pScratch + size_t(r) * vRowBytes,
                      vShexpBase + size_t(r) * vShexpTensor->nb[1] + size_t(vSub) * vRowBytes, size_t(vRowBytes));
        }
        return pScratch;
      }
      const uint32_t     vFullInputs = uint32_t(vShexpTensor->ne[0]);
      std::vector<float> aFullRow(vFullInputs);
      float *            vFloats = reinterpret_cast<float *>(pScratch);
      for (uint32_t r = 0; r < vOut; r++) {
        fDequantizeWeightRow(vShexpTensor->type, vShexpBase + size_t(r) * vShexpTensor->nb[1], aFullRow.data(),
                             vFullInputs);
        std::memcpy(vFloats + size_t(r) * vInputs, aFullRow.data() + size_t(vSub) * vInputs,
                    size_t(vInputs) * sizeof(float));
      }
      return vFloats;
    };
  }
  XdnaMoeWeights vLoaded;
  if (!pShared.vRunner.mLoadMoeMatrix(vTotalExps, vInputs, aSources, uint32_t(aExps.size()), pSwiglu, vLoaded, pError)) {
    return nullptr;
  }
  pBackend.vTotals.vMoeMatrices++;
  pBackend.vTotals.vMoeWeightsMs += fNowMs() - vStart;
  if (pShared.vSettings.vMoeMinTokens <= 1) {
    for (const ggml_tensor * vTensor : aExps) {
      fReleaseMappedExpertPages(pShared, vTensor);
    }
    if (!aShexp.empty()) {
      fNoteDenseCached(pShared, aShexp, false);
    }
  }
  return &pShared.dMoeMatrices.emplace(std::move(aKey), std::move(vLoaded)).first->second;
}

XdnaMoeWeights * fEnsureMoeWeights(XdnaShared & pShared, XdnaBackend & pBackend,
                                   const std::vector<const ggml_tensor *> & aTensors, bool pSwiglu,
                                   std::string & pError) {
  return fEnsureMoeWithShared(pShared, pBackend, aTensors, {}, 0, pSwiglu, pError);
}

struct XdnaMoeRouting {
  std::vector<XdnaMoeChunk> aChunks;
  std::vector<uint32_t>     aPairs;    // packed (token * vUsed + slot) for each routed row in chunk order
  std::vector<uint32_t>     aSkipped;  // packed (token * vUsed + slot) where expert id < 0
};

bool fBuildMoeRouting(const ggml_tensor * pIds, uint32_t pExperts, uint32_t pMaxRowsChunk, XdnaMoeRouting & pRouting,
                      std::string & pError) {
  const uint32_t vUsed   = uint32_t(pIds->ne[0]);
  const uint32_t vTokens = uint32_t(pIds->ne[1]);
  pRouting.aChunks.clear();
  pRouting.aPairs.clear();
  pRouting.aSkipped.clear();
  std::vector<uint32_t> aCounts(pExperts, 0);
  uint32_t              vValid = 0;
  for (uint32_t t = 0; t < vTokens; t++) {
    const int32_t * vRowIds = reinterpret_cast<const int32_t *>(static_cast<const char *>(pIds->data) + size_t(t) * pIds->nb[1]);
    for (uint32_t s = 0; s < vUsed; s++) {
      const int32_t vId = vRowIds[s];
      if (vId < 0) {
        pRouting.aSkipped.push_back(t * vUsed + s);
        continue;
      }
      if (uint32_t(vId) >= pExperts) {
        pError = "an expert index in MUL_MAT_ID is out of range";
        return false;
      }
      aCounts[uint32_t(vId)]++;
      vValid++;
    }
  }
  if (vValid == 0) {
    return true;
  }
  std::vector<uint32_t> aOffsets(pExperts + 1, 0);
  for (uint32_t e = 0; e < pExperts; e++) {
    aOffsets[e + 1] = aOffsets[e] + aCounts[e];
    for (uint32_t vLeft = aCounts[e]; vLeft > 0;) {
      const uint32_t vTake = std::min(pMaxRowsChunk, vLeft);
      pRouting.aChunks.push_back({e, vTake});
      vLeft -= vTake;
    }
  }
  pRouting.aPairs.resize(vValid);
  std::vector<uint32_t> aCursor = aOffsets;
  for (uint32_t t = 0; t < vTokens; t++) {
    const int32_t * vRowIds = reinterpret_cast<const int32_t *>(static_cast<const char *>(pIds->data) + size_t(t) * pIds->nb[1]);
    for (uint32_t s = 0; s < vUsed; s++) {
      const int32_t vId = vRowIds[s];
      if (vId >= 0) {
        pRouting.aPairs[aCursor[uint32_t(vId)]++] = t * vUsed + s;
      }
    }
  }
  return true;
}

bool fEmulateMulMatId(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  const ggml_tensor * vWeights = pNode->src[0];
  const ggml_tensor * vRows    = pNode->src[1];
  const ggml_tensor * vIds     = pNode->src[2];
  if (vWeights->data == nullptr || vRows->data == nullptr || vIds->data == nullptr || pNode->data == nullptr) {
    pError = "a tensor of MUL_MAT_ID has no memory";
    return false;
  }
  const double   vStart   = fNowMs();
  const uint32_t vInputs  = uint32_t(vWeights->ne[0]);
  const uint32_t vOutputs = uint32_t(vWeights->ne[1]);
  const uint32_t vExperts = uint32_t(vWeights->ne[2]);
  const uint32_t vUsed    = uint32_t(vIds->ne[0]);
  const uint32_t vTokens  = uint32_t(vIds->ne[1]);
  const uint32_t vBlocksK = (vInputs + cXdnaBfp16Values - 1) / cXdnaBfp16Values;
  const auto *   vTraits  = ggml_get_type_traits(vWeights->type);

  for (uint32_t t = 0; t < vTokens; t++) {
    const int32_t * vRowIds = reinterpret_cast<const int32_t *>(static_cast<const char *>(vIds->data) + size_t(t) * vIds->nb[1]);
    for (uint32_t s = 0; s < vUsed; s++) {
      float *       vDst = static_cast<float *>(pNode->data) + (size_t(t) * vUsed + s) * vOutputs;
      const int32_t vId  = vRowIds[s];
      if (vId < 0) {
        std::memset(vDst, 0, size_t(vOutputs) * sizeof(float));
        continue;
      }
      if (uint32_t(vId) >= vExperts) {
        pError = "an expert index in MUL_MAT_ID is out of range";
        return false;
      }
      const float * vSrcRow = reinterpret_cast<const float *>(
          static_cast<const char *>(vRows->data) + size_t(vRows->ne[1] == 1 ? 0 : s) * vRows->nb[1] + size_t(t) * vRows->nb[2]);
      const uint32_t     vPaddedK = (vBlocksK * cXdnaBfp16Values + 15u) & ~15u;
      std::vector<float> aRoundedX(vPaddedK, 0.0f);
      for (uint32_t b = 0; b < vBlocksK; b++) {
        float   aBlk[cXdnaBfp16Values] = {};
        uint8_t aEnc[cXdnaBfp16Bytes];
        for (uint32_t i = 0; i < cXdnaBfp16Values && b * cXdnaBfp16Values + i < vInputs; i++) {
          aBlk[i] = vSrcRow[b * cXdnaBfp16Values + i];
        }
        if (!fXdnaBfp16Encode(aBlk, aEnc, cRounding)) {
          pError = "an input row of MUL_MAT_ID has a value that is not finite";
          return false;
        }
        fXdnaBfp16Decode(aEnc, aRoundedX.data() + size_t(b) * cXdnaBfp16Values);
      }
      std::vector<float> aRowW(vInputs);
      std::vector<float> aDecodedW(vPaddedK, 0.0f);
      const char *       vExpertBase = static_cast<const char *>(vWeights->data) + size_t(vId) * vWeights->nb[2];
      for (uint32_t n = 0; n < vOutputs; n++) {
        const char * vFromW = vExpertBase + size_t(n) * vWeights->nb[1];
        if (vWeights->type == GGML_TYPE_F32) {
          std::memcpy(aRowW.data(), vFromW, size_t(vInputs) * sizeof(float));
        } else {
          vTraits->to_float(vFromW, aRowW.data(), vInputs);
        }
        std::vector<uint8_t> aEncW(size_t(vBlocksK) * cXdnaBfp16Bytes);
        for (uint32_t b = 0; b < vBlocksK; b++) {
          float aBlk[cXdnaBfp16Values] = {};
          for (uint32_t i = 0; i < cXdnaBfp16Values && b * cXdnaBfp16Values + i < vInputs; i++) {
            aBlk[i] = aRowW[b * cXdnaBfp16Values + i];
          }
          if (!fXdnaBfp16Encode(aBlk, aEncW.data() + size_t(b) * cXdnaBfp16Bytes, cRounding)) {
            pError = "an expert weight is not finite";
            return false;
          }
        }
        fDecodeBlocks(aEncW.data(), vBlocksK, aDecodedW.data());
        vDst[n] = fDot(aRoundedX.data(), aDecodedW.data(), vPaddedK);
      }
    }
  }
  pBackend.vTotals.vMoeOps++;
  pBackend.vTotals.vMoeMs += fNowMs() - vStart;
  pBackend.vTotals.vTotalMs += fNowMs() - vStart;
  return true;
}

struct XdnaMoeTailMatch {
  bool                vMatched   = false;
  int                 vLastIndex = -1;
  ggml_tensor *       vMoeOut    = nullptr;
  const ggml_tensor * vScaleS    = nullptr;  // optional Flint amax scale [1, vUsed, vTokens]
  const ggml_tensor * vWeights   = nullptr;  // expert routing weights [1, vUsed, vTokens]
};

XdnaMoeTailMatch fMatchMoeTail(const XdnaShared & pShared, const struct ggml_cgraph * pGraph, int pDownIndex,
                               const ggml_tensor * pDown) {
  XdnaMoeTailMatch vResult;
  if (pShared.vSettings.vNoFuse || pShared.vSettings.vEmulate || fUseCount(pGraph, pDown) != 1 ||
      (pDown->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return vResult;
  }
  const int64_t vUsed   = pDown->ne[1];
  const int     vMul1At = fNextNode(pGraph, pDownIndex);
  if (vMul1At >= pGraph->n_nodes) {
    return vResult;
  }
  const ggml_tensor * vMul1 = pGraph->nodes[vMul1At];
  if (vMul1->op != GGML_OP_MUL || vMul1->src[0] != pDown || !fMoeTailTaken(pShared, vMul1) ||
      (vMul1->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return vResult;
  }
  const ggml_tensor * vWeighted    = vMul1;
  int                 vWeightedAt  = vMul1At;
  const ggml_tensor * vScaleTensor = nullptr;
  const int           vAfterMul1   = fNextNode(pGraph, vMul1At);
  if (fUseCount(pGraph, vMul1) == 1 && vAfterMul1 < pGraph->n_nodes && pGraph->nodes[vAfterMul1]->op == GGML_OP_MUL &&
      pGraph->nodes[vAfterMul1]->src[0] == vMul1 && fMoeTailTaken(pShared, pGraph->nodes[vAfterMul1]) &&
      (pGraph->nodes[vAfterMul1]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
    vScaleTensor = vMul1->src[1];
    vWeighted    = pGraph->nodes[vAfterMul1];
    vWeightedAt  = vAfterMul1;
  }
  if (vUsed < 2 || fUseCount(pGraph, vWeighted) != int(vUsed)) {
    return vResult;
  }
  int           vCurAt  = vWeightedAt;
  ggml_tensor * vPrevAdd = nullptr;
  for (int64_t s = 1; s < vUsed; s++) {
    const int vAddAt = fNextNode(pGraph, vCurAt);
    if (vAddAt >= pGraph->n_nodes) {
      return vResult;
    }
    ggml_tensor * vAdd = pGraph->nodes[vAddAt];
    if (vAdd->op != GGML_OP_ADD || !fMoeTailTaken(pShared, vAdd)) {
      return vResult;
    }
    int64_t vRightSlot = -1;
    if (fMoeExpertSlotView(pShared, vAdd->src[1], vRightSlot) != vWeighted || vRightSlot != s ||
        fUseCount(pGraph, vAdd->src[1]) != 1 || (vAdd->src[1]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
      return vResult;
    }
    if (s == 1) {
      int64_t vZeroSlot = -1;
      if (fMoeExpertSlotView(pShared, vAdd->src[0], vZeroSlot) != vWeighted || vZeroSlot != 0 ||
          fUseCount(pGraph, vAdd->src[0]) != 1 || (vAdd->src[0]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
        return vResult;
      }
    } else if (vAdd->src[0] != vPrevAdd || fUseCount(pGraph, vPrevAdd) != 1 ||
               (vPrevAdd->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
      return vResult;
    }
    vPrevAdd = vAdd;
    vCurAt   = vAddAt;
  }
  if (vPrevAdd == nullptr || vPrevAdd->data == nullptr || vWeighted->src[1]->data == nullptr ||
      (vScaleTensor != nullptr && vScaleTensor->data == nullptr)) {
    return vResult;
  }
  vResult.vMatched   = true;
  vResult.vLastIndex = vCurAt;
  vResult.vMoeOut    = vPrevAdd;
  vResult.vScaleS    = vScaleTensor;
  vResult.vWeights   = vWeighted->src[1];
  return vResult;
}

bool fTensorsOverlap(const ggml_tensor * pA, const ggml_tensor * pB) {
  if (pA == nullptr || pB == nullptr || pA->data == nullptr || pB->data == nullptr) {
    return false;
  }
  const char * vA = static_cast<const char *>(pA->data);
  const char * vB = static_cast<const char *>(pB->data);
  return vA < vB + ggml_nbytes(pB) && vB < vA + ggml_nbytes(pA);
}

struct XdnaMoeAmaxMatch {
  bool                vMatched   = false;
  int                 vDivIndex  = -1;
  ggml_tensor *       vDivNode   = nullptr;
  const ggml_tensor * vScaleNode = nullptr;
};

XdnaMoeAmaxMatch fMatchMoeAmax(const XdnaShared & pShared, const struct ggml_cgraph * pGraph, int pGluIndex,
                               const ggml_tensor * pGlu) {
  XdnaMoeAmaxMatch vResult;
  if (pShared.vSettings.vNoFuse || pShared.vSettings.vNoMoeDense || pShared.vSettings.vEmulate ||
      fUseCount(pGraph, pGlu) != 2 || (pGlu->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return vResult;
  }
  int aAt[10];
  aAt[0] = fNextNode(pGraph, pGluIndex);
  if (aAt[0] >= pGraph->n_nodes) {
    return vResult;
  }
  for (int k = 1; k < 10; k++) {
    aAt[k] = fNextNode(pGraph, aAt[k - 1]);
    if (aAt[k] >= pGraph->n_nodes) {
      return vResult;
    }
  }
  ggml_tensor * vAbs   = pGraph->nodes[aAt[0]];
  ggml_tensor * vPool  = pGraph->nodes[aAt[1]];
  ggml_tensor * vSc0   = pGraph->nodes[aAt[2]];
  ggml_tensor * vLog   = pGraph->nodes[aAt[3]];
  ggml_tensor * vSc1   = pGraph->nodes[aAt[4]];
  ggml_tensor * vCeil  = pGraph->nodes[aAt[5]];
  ggml_tensor * vClamp = pGraph->nodes[aAt[6]];
  ggml_tensor * vSc2   = pGraph->nodes[aAt[7]];
  ggml_tensor * vExp   = pGraph->nodes[aAt[8]];
  ggml_tensor * vDiv   = pGraph->nodes[aAt[9]];
  if (vAbs->op != GGML_OP_UNARY || ggml_get_unary_op(vAbs) != GGML_UNARY_OP_ABS || vAbs->src[0] != pGlu ||
      vPool->op != GGML_OP_POOL_1D || ggml_get_op_params_i32(vPool, 0) != GGML_OP_POOL_MAX || vPool->src[0] != vAbs ||
      vSc0->op != GGML_OP_SCALE || vSc0->src[0] != vPool || vLog->op != GGML_OP_LOG || vLog->src[0] != vSc0 ||
      vSc1->op != GGML_OP_SCALE || vSc1->src[0] != vLog || vCeil->op != GGML_OP_UNARY ||
      ggml_get_unary_op(vCeil) != GGML_UNARY_OP_CEIL || vCeil->src[0] != vSc1 || vClamp->op != GGML_OP_CLAMP ||
      vClamp->src[0] != vCeil || vSc2->op != GGML_OP_SCALE || vSc2->src[0] != vClamp || vExp->op != GGML_OP_UNARY ||
      ggml_get_unary_op(vExp) != GGML_UNARY_OP_EXP || vExp->src[0] != vSc2 || vDiv->op != GGML_OP_DIV ||
      vDiv->src[0] != pGlu || vDiv->src[1] != vExp) {
    return vResult;
  }
  for (int k = 0; k < 8; k++) {
    if (fUseCount(pGraph, pGraph->nodes[aAt[k]]) != 1 ||
        (pGraph->nodes[aAt[k]]->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
      return vResult;
    }
  }
  if (fUseCount(pGraph, vExp) != 2 || (vExp->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || fUseCount(pGraph, vDiv) != 1 ||
      (vDiv->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return vResult;
  }
  vResult.vMatched   = true;
  vResult.vDivIndex  = aAt[9];
  vResult.vDivNode   = vDiv;
  vResult.vScaleNode = vExp;
  return vResult;
}

struct XdnaMoeShexpMatch {
  bool                vMatched         = false;
  int                 vLastIndex       = -1;
  uint32_t            vVirtualCount    = 0;
  const ggml_tensor * vGateWeights     = nullptr;
  const ggml_tensor * vUpWeights       = nullptr;
  const ggml_tensor * vDownWeights     = nullptr;
  const ggml_tensor * vGateInpWeights  = nullptr;
  const ggml_tensor * vInputRows       = nullptr;
  const ggml_tensor * vResidual        = nullptr;
  ggml_tensor *       vFinalOut        = nullptr;
};

XdnaMoeShexpMatch fMatchMoeSharedExpert(const XdnaShared & pShared, const struct ggml_cgraph * pGraph,
                                        const ggml_tensor * pGateExps, const ggml_tensor * pUpExps,
                                        const ggml_tensor * pDownExps, const XdnaMoeTailMatch & pTail,
                                        uint32_t pTokens) {
  XdnaMoeShexpMatch vResult;
  if (!pTail.vMatched || pShared.vSettings.vNoFuse || pShared.vSettings.vNoMoeDense || pShared.vSettings.vEmulate ||
      pGateExps == nullptr || pUpExps == nullptr || pDownExps == nullptr ||
      fUseCount(pGraph, pTail.vMoeOut) != 1 || (pTail.vMoeOut->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return vResult;
  }
  const int vAt0 = fNextNode(pGraph, pTail.vLastIndex);
  const int vAt1 = vAt0 < pGraph->n_nodes ? fNextNode(pGraph, vAt0) : pGraph->n_nodes;
  const int vAt2 = vAt1 < pGraph->n_nodes ? fNextNode(pGraph, vAt1) : pGraph->n_nodes;
  const int vAt3 = vAt2 < pGraph->n_nodes ? fNextNode(pGraph, vAt2) : pGraph->n_nodes;
  if (vAt3 >= pGraph->n_nodes) {
    return vResult;
  }
  ggml_tensor * vX0 = pGraph->nodes[vAt0];
  ggml_tensor * vX1 = pGraph->nodes[vAt1];
  ggml_tensor * vS  = pGraph->nodes[vAt2];
  ggml_tensor * vD  = pGraph->nodes[vAt3];
  if (vX0->op != GGML_OP_MUL_MAT || vX1->op != GGML_OP_MUL_MAT || vS->op != GGML_OP_GLU || vD->op != GGML_OP_MUL_MAT ||
      !fIsSharedExpertWeight(vX0->src[0]) || !fIsSharedExpertWeight(vX1->src[0]) ||
      !fIsSharedExpertWeight(vD->src[0]) || !fMulMatTaken(pShared, vX0) || !fMulMatTaken(pShared, vX1) ||
      !fMulMatTaken(pShared, vD) || !fGluTaken(vS) || pShared.vSettings.vNoGlu) {
    return vResult;
  }
  const ggml_tensor * vM = vX0->src[1];
  if (vM == nullptr || vX1->src[1] != vM || vM->data == nullptr || !ggml_is_contiguous(vM) ||
      vM->ne[0] != pGateExps->ne[0] || uint64_t(ggml_nrows(vM)) != uint64_t(pTokens)) {
    return vResult;
  }
  ggml_tensor * vG = vS->src[0];
  ggml_tensor * vU = vS->src[1];
  if (!((vG == vX0 && vU == vX1) || (vG == vX1 && vU == vX0)) || fUseCount(pGraph, vG) != 1 ||
      fUseCount(pGraph, vU) != 1 || (vG->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 ||
      (vU->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || vD->src[1] != vS || fUseCount(pGraph, vS) != 1 ||
      (vS->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || fUseCount(pGraph, vD) != 1 ||
      (vD->flags & GGML_TENSOR_FLAG_OUTPUT) != 0) {
    return vResult;
  }

  const int64_t vExpFf   = pGateExps->ne[1];
  const int64_t vShexpFf = vG->src[0]->ne[1];
  if (vExpFf <= 0 || vShexpFf <= 0 || vShexpFf % vExpFf != 0 || vU->src[0]->ne[1] != vShexpFf ||
      vD->src[0]->ne[0] != vShexpFf || vD->src[0]->ne[1] != pDownExps->ne[1]) {
    return vResult;
  }
  const uint32_t vVirtualCount = uint32_t(vShexpFf / vExpFf);
  if (vVirtualCount == 0 || vVirtualCount > 4) {
    return vResult;
  }
  const XdnaGemvFormat vFmtG     = fMoeFormatOf(pShared, pGateExps);
  const XdnaGemvFormat vFmtU     = fMoeFormatOf(pShared, pUpExps);
  const XdnaGemvFormat vFmtD     = fMoeFormatOf(pShared, pDownExps);
  const int64_t        vStepDown = (vFmtD == XdnaGemvFormat::cQ8_0 || vFmtD == XdnaGemvFormat::cQ5_1) ? 32 : 256;
  if (fMoeFormatOf(pShared, vG->src[0]) != vFmtG || fMoeFormatOf(pShared, vU->src[0]) != vFmtU ||
      fMoeFormatOf(pShared, vD->src[0]) != vFmtD ||
      (vVirtualCount > 1 && vFmtD != XdnaGemvFormat::cBfp16 && vExpFf % vStepDown != 0)) {
    return vResult;
  }

  const int vAt4 = fNextNode(pGraph, vAt3);
  if (vAt4 >= pGraph->n_nodes) {
    return vResult;
  }
  const ggml_tensor * vGateInpWeights = nullptr;
  ggml_tensor *       vFfnOut         = nullptr;
  int                 vAtFfnOut       = -1;
  ggml_tensor *       vN4             = pGraph->nodes[vAt4];
  if (vN4->op == GGML_OP_MUL_MAT && fIsSharedGate(vN4->src[0]) && vN4->src[1] == vM && fUseCount(pGraph, vN4) == 1 &&
      (vN4->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && vN4->ne[0] == 1 && uint64_t(ggml_nrows(vN4)) == uint64_t(pTokens) &&
      vN4->src[0]->data != nullptr) {
    const int vAt5 = fNextNode(pGraph, vAt4);
    const int vAt6 = vAt5 < pGraph->n_nodes ? fNextNode(pGraph, vAt5) : pGraph->n_nodes;
    const int vAt7 = vAt6 < pGraph->n_nodes ? fNextNode(pGraph, vAt6) : pGraph->n_nodes;
    if (vAt7 >= pGraph->n_nodes) {
      return vResult;
    }
    ggml_tensor * vSig   = pGraph->nodes[vAt5];
    ggml_tensor * vGated = pGraph->nodes[vAt6];
    ggml_tensor * vSum   = pGraph->nodes[vAt7];
    if (vSig->op != GGML_OP_UNARY || ggml_get_unary_op(vSig) != GGML_UNARY_OP_SIGMOID || vSig->src[0] != vN4 ||
        fUseCount(pGraph, vSig) != 1 || (vSig->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || vGated->op != GGML_OP_MUL ||
        !((vGated->src[0] == vD && vGated->src[1] == vSig) || (vGated->src[0] == vSig && vGated->src[1] == vD)) ||
        fUseCount(pGraph, vGated) != 1 || (vGated->flags & GGML_TENSOR_FLAG_OUTPUT) != 0 || vSum->op != GGML_OP_ADD ||
        !((vSum->src[0] == pTail.vMoeOut && vSum->src[1] == vGated) ||
          (vSum->src[0] == vGated && vSum->src[1] == pTail.vMoeOut))) {
      return vResult;
    }
    vGateInpWeights = vN4->src[0];
    vFfnOut         = vSum;
    vAtFfnOut       = vAt7;
  } else if (vN4->op == GGML_OP_ADD &&
             ((vN4->src[0] == pTail.vMoeOut && vN4->src[1] == vD) ||
              (vN4->src[0] == vD && vN4->src[1] == pTail.vMoeOut))) {
    vFfnOut   = vN4;
    vAtFfnOut = vAt4;
  } else {
    return vResult;
  }
  if (!ggml_is_contiguous(vFfnOut) || !ggml_are_same_shape(vFfnOut, pTail.vMoeOut) || vFfnOut->data == nullptr) {
    return vResult;
  }

  ggml_tensor *       vFinalOut  = vFfnOut;
  int                 vLastIdx   = vAtFfnOut;
  const ggml_tensor * vResidual  = nullptr;
  const int           vAtPost    = fNextNode(pGraph, vAtFfnOut);
  if (fUseCount(pGraph, vFfnOut) == 1 && (vFfnOut->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && vAtPost < pGraph->n_nodes) {
    ggml_tensor * vPost = pGraph->nodes[vAtPost];
    if (vPost->op == GGML_OP_ADD && vPost->type == GGML_TYPE_F32 && ggml_is_contiguous(vPost) &&
        ggml_are_same_shape(vPost, vFfnOut) && vPost->data != nullptr &&
        (vPost->src[0] == vFfnOut || vPost->src[1] == vFfnOut)) {
      const ggml_tensor * vRes = vPost->src[0] == vFfnOut ? vPost->src[1] : vPost->src[0];
      if (vRes != nullptr && vRes->type == GGML_TYPE_F32 && ggml_is_contiguous(vRes) &&
          ggml_are_same_shape(vRes, vPost) && vRes->data != nullptr && !fIsWeight(vRes) &&
          (vPost->data == vRes->data || !fTensorsOverlap(vPost, vRes))) {
        vFinalOut = vPost;
        vLastIdx  = vAtPost;
        vResidual = vRes;
      }
    }
  }

  vResult.vMatched        = true;
  vResult.vLastIndex      = vLastIdx;
  vResult.vVirtualCount   = vVirtualCount;
  vResult.vGateWeights    = vG->src[0];
  vResult.vUpWeights      = vU->src[0];
  vResult.vDownWeights    = vD->src[0];
  vResult.vGateInpWeights = vGateInpWeights;
  vResult.vInputRows      = vM;
  vResult.vResidual       = vResidual;
  vResult.vFinalOut       = vFinalOut;
  return vResult;
}

bool fRunMoeStep(XdnaBackend & pBackend, struct ggml_cgraph * pGraph, int pIndex, int & pLastIndex,
                 std::string & pError) {
  XdnaShared &  vShared = fShared();
  ggml_tensor * vFirst  = pGraph->nodes[pIndex];
  pLastIndex            = pIndex;
  if (vShared.vSettings.vEmulate) {
    return fEmulateMulMatId(pBackend, vFirst, pError);
  }
  const double vTotalStart = fNowMs();

  // Detect {up_exps, gate_exps} + SwiGLU [ + Flint amax ] [ + adjacent down_exps ] [ + weighted moe_out ] [ + shexp ]
  ggml_tensor *    vGateNode = nullptr;
  ggml_tensor *    vUpNode   = nullptr;
  ggml_tensor *    vGluNode  = nullptr;
  ggml_tensor *    vDownNode = nullptr;
  int              vGluIndex  = -1;
  int              vDownIndex = -1;
  XdnaMoeAmaxMatch vAmax;
  if (!vShared.vSettings.vNoFuse && fUseCount(pGraph, vFirst) == 1 && (vFirst->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
    const int vSecondAt = fNextNode(pGraph, pIndex);
    const int vGluAt    = vSecondAt < pGraph->n_nodes ? fNextNode(pGraph, vSecondAt) : pGraph->n_nodes;
    if (vGluAt < pGraph->n_nodes) {
      ggml_tensor * vSecond = pGraph->nodes[vSecondAt];
      ggml_tensor * vGlu    = pGraph->nodes[vGluAt];
      if (vSecond->op == GGML_OP_MUL_MAT_ID && fMoeTaken(vShared, vSecond) && vSecond->src[1] == vFirst->src[1] &&
          vSecond->src[2] == vFirst->src[2] && vSecond->src[0]->ne[0] == vFirst->src[0]->ne[0] &&
          vSecond->src[0]->ne[1] == vFirst->src[0]->ne[1] && vSecond->src[0]->ne[2] == vFirst->src[0]->ne[2] &&
          fUseCount(pGraph, vSecond) == 1 && (vSecond->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
          vGlu->op == GGML_OP_GLU && fGluTaken(vGlu) &&
          ((vGlu->src[0] == vFirst && vGlu->src[1] == vSecond) ||
           (vGlu->src[0] == vSecond && vGlu->src[1] == vFirst))) {
        ggml_tensor * vGateCandidate = vGlu->src[0] == vFirst ? vFirst : vSecond;
        ggml_tensor * vUpCandidate   = vGlu->src[0] == vFirst ? vSecond : vFirst;
        const XdnaGemvPart aFusedParts[2] = {
            {fMoeFormatOf(vShared, vGateCandidate->src[0]), uint32_t(vGateCandidate->src[0]->ne[1])},
            {fMoeFormatOf(vShared, vUpCandidate->src[0]), uint32_t(vUpCandidate->src[0]->ne[1])},
        };
        XdnaMoeMatrix vFusedShape;
        std::string   vShapeError;
        if (fXdnaMoeMatrixShape(vShared.vMoeOverlay, uint32_t(vFirst->src[0]->ne[2]), uint32_t(vFirst->src[0]->ne[0]),
                                aFusedParts, 2, true, vFusedShape, vShapeError)) {
          vGateNode = vGateCandidate;
          vUpNode   = vUpCandidate;
          vGluNode  = vGlu;
          vGluIndex = vGluAt;
          const int vAfterGlu = fNextNode(pGraph, vGluAt);
          if (vAfterGlu < pGraph->n_nodes && fUseCount(pGraph, vGlu) == 1 &&
              (vGlu->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            ggml_tensor * vCandidate = pGraph->nodes[vAfterGlu];
            if (vCandidate->op == GGML_OP_MUL_MAT_ID && fMoeTaken(vShared, vCandidate) && vCandidate->src[1] == vGlu &&
                vCandidate->src[2] == vFirst->src[2] && vCandidate->src[0]->ne[2] == vFirst->src[0]->ne[2]) {
              vDownNode  = vCandidate;
              vDownIndex = vAfterGlu;
            }
          } else {
            XdnaMoeAmaxMatch vTryAmax = fMatchMoeAmax(vShared, pGraph, vGluAt, vGlu);
            if (vTryAmax.vMatched) {
              const int vAfterDiv = fNextNode(pGraph, vTryAmax.vDivIndex);
              if (vAfterDiv < pGraph->n_nodes) {
                ggml_tensor * vCandidate = pGraph->nodes[vAfterDiv];
                if (vCandidate->op == GGML_OP_MUL_MAT_ID && fMoeTaken(vShared, vCandidate) &&
                    vCandidate->src[1] == vTryAmax.vDivNode && vCandidate->src[2] == vFirst->src[2] &&
                    vCandidate->src[0]->ne[2] == vFirst->src[0]->ne[2]) {
                  XdnaMoeTailMatch vCheckTail = fMatchMoeTail(vShared, pGraph, vAfterDiv, vCandidate);
                  if (vCheckTail.vMatched && vCheckTail.vScaleS == vTryAmax.vScaleNode) {
                    vAmax      = vTryAmax;
                    vDownNode  = vCandidate;
                    vDownIndex = vAfterDiv;
                  }
                }
              }
            }
          }
        }
      }
    }
  }

  ggml_tensor *    vTailAnchor = vDownNode != nullptr ? vDownNode : (vGluNode == nullptr ? vFirst : nullptr);
  const int        vTailFrom   = vDownNode != nullptr ? vDownIndex : pIndex;
  XdnaMoeTailMatch vTail       = vTailAnchor != nullptr ? fMatchMoeTail(vShared, pGraph, vTailFrom, vTailAnchor) :
                                                           XdnaMoeTailMatch();
  if (vAmax.vMatched && vTail.vMatched) {
    vTail.vScaleS = nullptr;  // SwiGLU was not divided by amax scale_s, so down_exps is not multiplied by it
  }

  const ggml_tensor * vRowsTensor  = vFirst->src[1];
  const ggml_tensor * vIdsTensor   = vFirst->src[2];
  const uint32_t      vBaseExperts = uint32_t(vFirst->src[0]->ne[2]);
  const uint32_t      vUsed        = uint32_t(vIdsTensor->ne[0]);
  const uint32_t      vTokens      = uint32_t(vIdsTensor->ne[1]);
  XdnaMoeShexpMatch   vShexp       = vDownNode != nullptr ? fMatchMoeSharedExpert(vShared, pGraph, vGateNode->src[0],
                                                                                  vUpNode->src[0], vDownNode->src[0],
                                                                                  vTail, vTokens) :
                                                            XdnaMoeShexpMatch();

  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));

  XdnaMoeWeights * vMainWeights = nullptr;
  XdnaMoeWeights * vDownWeights = nullptr;
  if (vGluNode != nullptr) {
    if (vShexp.vMatched) {
      vMainWeights = fEnsureMoeWithShared(vShared, pBackend, {vGateNode->src[0], vUpNode->src[0]},
                                          {vShexp.vGateWeights, vShexp.vUpWeights}, vShexp.vVirtualCount, true, pError);
    } else {
      vMainWeights = fEnsureMoeWeights(vShared, pBackend, {vGateNode->src[0], vUpNode->src[0]}, true, pError);
    }
  } else {
    vMainWeights = fEnsureMoeWeights(vShared, pBackend, {vFirst->src[0]}, false, pError);
  }
  if (vMainWeights == nullptr) {
    return false;
  }
  if (vDownNode != nullptr) {
    if (vShexp.vMatched) {
      vDownWeights = fEnsureMoeWithShared(vShared, pBackend, {vDownNode->src[0]}, {vShexp.vDownWeights},
                                          vShexp.vVirtualCount, false, pError);
    } else {
      vDownWeights = fEnsureMoeWeights(vShared, pBackend, {vDownNode->src[0]}, false, pError);
    }
    if (vDownWeights == nullptr) {
      return false;
    }
    if (vShexp.vMatched &&
        (vMainWeights->vMatrix.vExperts < vBaseExperts + vShexp.vVirtualCount ||
         vDownWeights->vMatrix.vExperts < vBaseExperts + vShexp.vVirtualCount)) {
      vShexp = XdnaMoeShexpMatch();
    }
  }

  uint32_t vMaxRowsChunk = fXdnaMoeMaxRows(vShared.vMoeOverlay, vMainWeights->vMatrix);
  if (vDownWeights != nullptr) {
    vMaxRowsChunk = std::min(vMaxRowsChunk, fXdnaMoeMaxRows(vShared.vMoeOverlay, vDownWeights->vMatrix));
  }

  XdnaMoeRouting vRouting;
  if (!fBuildMoeRouting(vIdsTensor, vBaseExperts, vMaxRowsChunk, vRouting, pError)) {
    return false;
  }

  const uint32_t vMaxChunks = vShared.vMoeOverlay.vChunksMax;
  if (vShexp.vMatched) {
    const uint32_t vShexpChunks = vShexp.vVirtualCount * ((vTokens + vMaxRowsChunk - 1) / vMaxRowsChunk);
    if (vRouting.aChunks.size() + vShexpChunks > vMaxChunks &&
        (fTensorsOverlap(vShexp.vFinalOut, vRowsTensor) || fTensorsOverlap(vShexp.vFinalOut, vShexp.vInputRows))) {
      vShexp = XdnaMoeShexpMatch();
    }
  }

  ggml_tensor * vFinalDst = vShexp.vMatched  ? vShexp.vFinalOut :
                            vTail.vMatched   ? vTail.vMoeOut :
                            vDownNode != nullptr ? vDownNode :
                            vGluNode != nullptr  ? vGluNode : vFirst;
  if (vRowsTensor->data == nullptr || vIdsTensor->data == nullptr || vFinalDst->data == nullptr) {
    pError = "a tensor of the experts has no memory";
    return false;
  }

  std::vector<float> aSharedGates;
  if (vShexp.vMatched && vShexp.vGateInpWeights != nullptr) {
    aSharedGates.resize(vTokens);
    const uint32_t     vInputs = uint32_t(vShexp.vInputRows->ne[0]);
    std::vector<float> aGateW;
    const float *      vGateW = nullptr;
    if (vShexp.vGateInpWeights->type == GGML_TYPE_F32) {
      vGateW = static_cast<const float *>(vShexp.vGateInpWeights->data);
    } else {
      aGateW.resize(vInputs);
      ggml_get_type_traits(vShexp.vGateInpWeights->type)->to_float(vShexp.vGateInpWeights->data, aGateW.data(),
                                                                   vInputs);
      vGateW = aGateW.data();
    }
    const float * vInPtr = static_cast<const float *>(vShexp.vInputRows->data);
    auto          fTok   = [&](uint32_t pTok) {
      const float vLogit = fDotRow(vInPtr + size_t(pTok) * vInputs, vGateW, vInputs);
      aSharedGates[pTok] = 1.0f / (1.0f + std::exp(-vLogit));
    };
    if (vTokens >= 16) {
      vShared.vRunner.mWorkers().mForEach(vTokens, fTok);
    } else {
      for (uint32_t t = 0; t < vTokens; t++) {
        fTok(t);
      }
    }
  }

  const uint32_t             vValidRows = uint32_t(vRouting.aPairs.size());
  const uint32_t             vOutWidth  = uint32_t(vFinalDst->ne[0]);
  float *                    vOutBase   = static_cast<float *>(vFinalDst->data);
  const float *              vInitSrc   = vShexp.vMatched && vShexp.vResidual != nullptr ?
                                              static_cast<const float *>(vShexp.vResidual->data) :
                                              nullptr;
  std::vector<XdnaMoeRowOut> aRowsOut(vValidRows);
  std::vector<bool>          aTokenSeen(vTokens, vInitSrc != nullptr);
  if (vTail.vMatched) {
    for (uint32_t r = 0; r < vValidRows; r++) {
      const uint32_t vPair  = vRouting.aPairs[r];
      const uint32_t vTok   = vPair / vUsed;
      const uint32_t vSlot  = vPair % vUsed;
      float          vScale = *reinterpret_cast<const float *>(
          static_cast<const char *>(vTail.vWeights->data) + size_t(vSlot) * vTail.vWeights->nb[1] +
          size_t(vTok) * vTail.vWeights->nb[2]);
      if (vTail.vScaleS != nullptr) {
        vScale *= *reinterpret_cast<const float *>(
            static_cast<const char *>(vTail.vScaleS->data) + size_t(vSlot) * vTail.vScaleS->nb[1] +
            size_t(vTok) * vTail.vScaleS->nb[2]);
      }
      aRowsOut[r]      = {vOutBase + size_t(vTok) * vOutWidth, vScale, aTokenSeen[vTok]};
      aTokenSeen[vTok] = true;
    }
  } else {
    for (uint32_t r = 0; r < vValidRows; r++) {
      aRowsOut[r] = {vOutBase + size_t(vRouting.aPairs[r]) * vOutWidth, 1.0f, false};
    }
  }

  const bool vSharedInputRows = vRowsTensor->ne[1] == 1 && vRowsTensor->nb[2] % sizeof(float) == 0 &&
                                (!vShexp.vMatched ||
                                 (vShexp.vInputRows->data == vRowsTensor->data &&
                                  vShexp.vInputRows->nb[1] == vRowsTensor->nb[2]));
  std::vector<uint32_t>      aTokenIndices;
  std::vector<const float *> aDistinctPtrs;
  if (vSharedInputRows) {
    aTokenIndices.resize(vValidRows);
    for (uint32_t r = 0; r < vValidRows; r++) {
      aTokenIndices[r] = vRouting.aPairs[r] / vUsed;
    }
  } else {
    aDistinctPtrs.resize(vValidRows);
    for (uint32_t r = 0; r < vValidRows; r++) {
      const uint32_t vPair = vRouting.aPairs[r];
      const uint32_t vTok  = vPair / vUsed;
      const uint32_t vSlot = vPair % vUsed;
      aDistinctPtrs[r]     = reinterpret_cast<const float *>(
          static_cast<const char *>(vRowsTensor->data) + size_t(vSlot) * vRowsTensor->nb[1] +
          size_t(vTok) * vRowsTensor->nb[2]);
    }
  }

  if (vShexp.vMatched) {
    for (uint32_t vSub = 0; vSub < vShexp.vVirtualCount; vSub++) {
      const uint32_t vExpertId = vBaseExperts + vSub;
      for (uint32_t vTokFirst = 0; vTokFirst < vTokens; vTokFirst += vMaxRowsChunk) {
        const uint32_t vTake = std::min(vMaxRowsChunk, vTokens - vTokFirst);
        vRouting.aChunks.push_back({vExpertId, vTake});
        for (uint32_t m = 0; m < vTake; m++) {
          const uint32_t vTok   = vTokFirst + m;
          const float    vScale = !aSharedGates.empty() ? aSharedGates[vTok] : 1.0f;
          aRowsOut.push_back({vOutBase + size_t(vTok) * vOutWidth, vScale, aTokenSeen[vTok]});
          aTokenSeen[vTok] = true;
          if (vSharedInputRows) {
            aTokenIndices.push_back(vTok);
          } else {
            aDistinctPtrs.push_back(reinterpret_cast<const float *>(
                static_cast<const char *>(vShexp.vInputRows->data) + size_t(vTok) * vShexp.vInputRows->nb[1]));
          }
        }
      }
    }
  }

  XdnaGemmTimes  vTimes;
  uint64_t       vCommandsSent = 0;
  const uint32_t vTotalChunks  = uint32_t(vRouting.aChunks.size());
  uint32_t       vRowOffset    = 0;
  bool           vPreparedDst  = false;
  auto           fPrepareUnseen = [&]() {
    if (vPreparedDst) {
      return;
    }
    vPreparedDst = true;
    if (vTail.vMatched) {
      if (vInitSrc == nullptr) {
        for (uint32_t t = 0; t < vTokens; t++) {
          if (!aTokenSeen[t]) {
            std::memset(vOutBase + size_t(t) * vOutWidth, 0, size_t(vOutWidth) * sizeof(float));
          }
        }
      }
    } else {
      for (uint32_t vPair : vRouting.aSkipped) {
        std::memset(vOutBase + size_t(vPair) * vOutWidth, 0, size_t(vOutWidth) * sizeof(float));
      }
    }
  };
  if (vTotalChunks == 0) {
    fPrepareUnseen();
    if (vInitSrc != nullptr && vInitSrc != vOutBase) {
      std::memcpy(vOutBase, vInitSrc, size_t(vTokens) * vOutWidth * sizeof(float));
    }
  }
  for (uint32_t vChunkFirst = 0; vChunkFirst < vTotalChunks; vChunkFirst += vMaxChunks) {
    const uint32_t vChunkCount = std::min(vMaxChunks, vTotalChunks - vChunkFirst);
    uint32_t       vSubRows    = 0;
    for (uint32_t c = 0; c < vChunkCount; c++) {
      vSubRows += vRouting.aChunks[vChunkFirst + c].vRows;
    }
    const float *         vSharedPtr   = vSharedInputRows ? static_cast<const float *>(vRowsTensor->data) : nullptr;
    const size_t          vSharedStr   = vSharedInputRows ? vRowsTensor->nb[2] / sizeof(float) : 0;
    const uint32_t *      vSubIdx      = vSharedInputRows ? aTokenIndices.data() + vRowOffset : nullptr;
    const float * const * vSubRowPtrs  = vSharedInputRows ? nullptr : aDistinctPtrs.data() + vRowOffset;
    const float *         vPassInitSrc = vChunkFirst == 0 ? vInitSrc : nullptr;
    float *               vPassInitDst = vChunkFirst == 0 && vInitSrc != nullptr ? vOutBase : nullptr;
    const uint32_t        vPassInitR   = vChunkFirst == 0 && vInitSrc != nullptr ? vTokens : 0;
    if (vDownWeights != nullptr) {
      XdnaMoeCommand vKept;
      if (!vShared.vRunner.mRunMoe(*vMainWeights, vRouting.aChunks.data() + vChunkFirst, vChunkCount, vSharedPtr,
                                   vSharedStr, vTokens, vSubIdx, vSubRowPtrs, nullptr, vTimes, pError, &vKept)) {
        return false;
      }
      fPrepareUnseen();
      if (!vShared.vRunner.mRunMoeFromKept(vKept, *vDownWeights, nullptr, aRowsOut.data() + vRowOffset, vTimes, pError,
                                           nullptr, 0, vPassInitSrc, vPassInitDst, vPassInitR, vOutWidth)) {
        return false;
      }
      vCommandsSent += 2;
    } else {
      if (!vShared.vRunner.mRunMoe(*vMainWeights, vRouting.aChunks.data() + vChunkFirst, vChunkCount, vSharedPtr,
                                   vSharedStr, vTokens, vSubIdx, vSubRowPtrs, aRowsOut.data() + vRowOffset, vTimes,
                                   pError)) {
        return false;
      }
      fPrepareUnseen();
      vCommandsSent += 1;
    }
    vRowOffset += vSubRows;
  }

  pBackend.vTotals.vMoeOps += vDownNode != nullptr ? 3 : (vGluNode != nullptr ? 2 : 1);
  pBackend.vTotals.vMoeCommands += vCommandsSent;
  if (vGluNode != nullptr) {
    pBackend.vTotals.vMoeFusedGlu++;
  }
  if (vTail.vMatched) {
    pBackend.vTotals.vMoeTailFused++;
  }
  if (vShexp.vMatched) {
    pBackend.vTotals.vMoeFusedShexp++;
    pBackend.vTotals.vOperations += 3 + (vShexp.vGateInpWeights != nullptr ? 1 : 0);
    pBackend.vTotals.vRows       += uint64_t(vTokens) * (3 + (vShexp.vGateInpWeights != nullptr ? 1 : 0));
  }
  pBackend.vTotals.vMoeNpuMs    += vTimes.vNpuMs;
  pBackend.vTotals.vMoePackMs   += vTimes.vPackMs;
  pBackend.vTotals.vMoeUnpackMs += vTimes.vUnpackMs;
  pBackend.vTotals.vMoePlanMs   += vTimes.vPlanMs;
  const double vElapsed          = fNowMs() - vTotalStart;
  pBackend.vTotals.vMoeMs       += vElapsed;
  pBackend.vTotals.vTotalMs     += vElapsed;

  pLastIndex = vShexp.vMatched  ? vShexp.vLastIndex :
               vTail.vMatched   ? vTail.vLastIndex :
               vDownNode != nullptr ? vDownIndex :
               vGluNode != nullptr  ? vGluIndex : pIndex;
  return true;
}

// llama.cpp-xdna: dense feed-forward block ({gate, up} + SwiGLU + down [+ residual ADD]) of 1 to 8 rows on xdna2-moe
// (such as the leading dense layers of Flint) so the token stays on xdna2-moe with 0 overlay switches
bool fRunDenseMoeFfn(XdnaBackend & pBackend, const struct ggml_cgraph * pGraph, ggml_tensor * const * aPair,
                     const ggml_tensor * pGlu, int pDownIndex, const ggml_tensor * pDown, int & pLastIndex,
                     bool & pDone, std::string & pError) {
  pDone                 = false;
  XdnaShared & vShared  = fShared();
  const ggml_tensor * vRows = aPair[0]->src[1];
  const uint32_t      vInputs = uint32_t(aPair[0]->src[0]->ne[0]);
  const uint64_t      vCount  = uint64_t(ggml_nrows(vRows));
  if (pDown == nullptr || vShared.vSettings.vNoMoeDense || vShared.vSettings.vNoFuse || vShared.vSettings.vEmulate ||
      vShared.vSettings.vMoeGemv || !fByRows(vShared, vCount, vInputs) || !fMoeActiveForRows(vShared, vCount) ||
      !ggml_is_contiguous(vRows) || !ggml_is_contiguous(pDown) || vRows->data == nullptr || pDown->data == nullptr) {
    return true;
  }
  const ggml_tensor * vGateW = pGlu->src[0]->src[0];
  const ggml_tensor * vUpW   = pGlu->src[1]->src[0];
  const ggml_tensor * vDownW = pDown->src[0];
  if (vGateW->ne[1] != vUpW->ne[1] || vDownW->ne[0] != vGateW->ne[1] ||
      !fMoeDenseFits(vShared, vGateW, uint32_t(vCount), true) ||
      !fMoeDenseFits(vShared, vDownW, uint32_t(vCount), false)) {
    return true;
  }
  int                 vLast   = pDownIndex;
  float *             vOutDst = static_cast<float *>(pDown->data);
  const float *       vAlso   = nullptr;
  const int           vAddAt  = fNextNode(pGraph, pDownIndex);
  if (fUseCount(pGraph, pDown) == 1 && (pDown->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && vAddAt < pGraph->n_nodes) {
    ggml_tensor * vAdd = pGraph->nodes[vAddAt];
    if (vAdd->op == GGML_OP_ADD && fAddTaken(vShared, vAdd) && vAdd->data != nullptr &&
        (vAdd->src[0] == pDown || vAdd->src[1] == pDown)) {
      const ggml_tensor * vOther = vAdd->src[0] == pDown ? vAdd->src[1] : vAdd->src[0];
      if (vOther != nullptr && vOther->data != nullptr && ggml_is_contiguous(vOther) &&
          ggml_are_same_shape(vOther, vAdd)) {
        vOutDst = static_cast<float *>(vAdd->data);
        vAlso   = static_cast<const float *>(vOther->data);
        vLast   = vAddAt;
      }
    }
  }

  const double        vStart = fNowMs();
  const float *       vA     = static_cast<const float *>(vRows->data);
  XdnaRowNorm         vNorm;
  const XdnaRowNorm * vByRow = nullptr;
  if (pBackend.vNormed.vActive && vRows == pBackend.vNormed.vRows) {
    vNorm.vScales  = pBackend.aScales.data();
    vNorm.vWeights = pBackend.vNormed.vWeights;
    vA             = pBackend.vNormed.vX;
    vByRow         = &vNorm;
  }

  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
  XdnaMoeWeights * vPairW = fEnsureMoeDensePair(vShared, pBackend, vGateW, vUpW, pError);
  if (vPairW == nullptr) {
    return false;
  }
  XdnaMoeWeights * vDownMoeW = fEnsureMoeDenseGroup(vShared, pBackend, {vDownW}, pError);
  if (vDownMoeW == nullptr) {
    return false;
  }

  const XdnaMoeChunk    vChunk = {0, uint32_t(vCount)};
  std::vector<uint32_t> aIdx(vCount);
  for (uint64_t r = 0; r < vCount; r++) {
    aIdx[r] = uint32_t(r);
  }
  const uint32_t vOutputs = uint32_t(vDownW->ne[1]);
  XdnaGemmTarget vTarget;
  vTarget.vData   = vOutDst;
  vTarget.vStride = vOutputs;
  vTarget.vFirst  = 0;
  vTarget.vCount  = vOutputs;
  vTarget.vAlso   = vAlso;

  XdnaGemmTimes  vTimes;
  XdnaMoeCommand vKept;
  if (!vShared.vRunner.mRunMoe(*vPairW, &vChunk, 1, vA, vInputs, uint32_t(vCount), aIdx.data(), nullptr, nullptr,
                               vTimes, pError, &vKept, vByRow) ||
      !vShared.vRunner.mRunMoeFromKept(vKept, *vDownMoeW, nullptr, nullptr, vTimes, pError, &vTarget, 1)) {
    return false;
  }
  pBackend.vTotals.vOperations   += 3;
  pBackend.vTotals.vRows         += vCount * 3;
  pBackend.vTotals.vCommands     += 2;
  pBackend.vTotals.vCommandsRows += 2;
  pBackend.vTotals.vMoeDenseCmds += 2;
  pBackend.vTotals.vMoeDenseNpuMs += vTimes.vNpuMs;
  pBackend.vTotals.vGlusFused++;
  if (vAlso != nullptr) {
    pBackend.vTotals.vAddsFused++;
  }
  pBackend.vTotals.vNpuMs     += vTimes.vNpuMs;
  pBackend.vTotals.vNpuRowsMs += vTimes.vNpuMs;
  pBackend.vTotals.vPackMs    += vTimes.vPackMs;
  pBackend.vTotals.vUnpackMs  += vTimes.vUnpackMs;
  pBackend.vTotals.vFlushMs   += vTimes.vFlushMs;
  pBackend.vTotals.vPlanMs    += vTimes.vPlanMs;
  pBackend.vTotals.vTotalMs   += fNowMs() - vStart;
  pLastIndex = vLast;
  pDone      = true;
  return true;
}

// llama.cpp-xdna: under GGML_XDNA_FULL=1/2, claim all remaining graph operations so the entire prompt and generation
// graphs execute as a single XDNA subgraph (splits = 1, 0 scheduler copies).
bool fFullOpTaken(const XdnaShared & pShared, const ggml_tensor * pNode) {
  if (pShared.vSettings.vFull <= 0 || pShared.vSettings.vEmulate || pNode == nullptr) {
    return false;
  }
  const ggml_tensor * vA = pNode->src[0];
  const ggml_tensor * vB = pNode->src[1];
  const ggml_tensor * vC = pNode->src[2];
  switch (pNode->op) {
    case GGML_OP_GET_ROWS: {
      if (vA == nullptr || vB == nullptr || vB->type != GGML_TYPE_I32 || pNode->type != GGML_TYPE_F32 ||
          pNode->nb[0] != sizeof(float) || vA->ne[3] != 1 || vB->ne[3] != 1) {
        return false;
      }
      const ggml_tensor * vBase = vA->view_src != nullptr ? vA->view_src : vA;
      if (vBase->buffer != nullptr && ggml_backend_buffer_get_usage(vBase->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS &&
          ggml_backend_buffer_get_size(vBase->buffer) == 0) {
        return false;
      }
      if (vA->type == GGML_TYPE_F32) {
        return vA->nb[0] == sizeof(float);
      }
      if (vA->type == GGML_TYPE_F16 || vA->type == GGML_TYPE_BF16) {
        return vA->nb[0] == sizeof(uint16_t);
      }
      const struct ggml_type_traits * vTraits = ggml_get_type_traits(vA->type);
      return vTraits != nullptr && vTraits->to_float != nullptr && vTraits->blck_size > 0 &&
             vA->ne[0] % vTraits->blck_size == 0 && vA->nb[0] == vTraits->type_size;
    }
    case GGML_OP_DUP:
    case GGML_OP_CONT:
      return vA != nullptr && (vA->type == GGML_TYPE_F32 || vA->type == GGML_TYPE_F16) && pNode->type == vA->type &&
             ggml_is_contiguous(pNode) && ggml_nelements(vA) == ggml_nelements(pNode);
    case GGML_OP_ROPE: {
      if (vA == nullptr || vB == nullptr || (vA->type != GGML_TYPE_F32 && vA->type != GGML_TYPE_F16) ||
          vB->type != GGML_TYPE_I32 || (vC != nullptr && vC->type != GGML_TYPE_F32) || pNode->type != vA->type ||
          vA->nb[0] != ggml_type_size(vA->type) || pNode->nb[0] != vA->nb[0] || !ggml_are_same_shape(vA, pNode)) {
        return false;
      }
      const int32_t vMode = reinterpret_cast<const int32_t *>(pNode->op_params)[2];
      return vMode == GGML_ROPE_TYPE_NORMAL || vMode == GGML_ROPE_TYPE_NEOX || vMode == GGML_ROPE_TYPE_MROPE ||
             vMode == GGML_ROPE_TYPE_IMROPE || vMode == GGML_ROPE_TYPE_VISION;
    }
    case GGML_OP_SET_ROWS:
      return vA != nullptr && vB != nullptr && (vA->type == GGML_TYPE_F32 || vA->type == GGML_TYPE_F16) &&
             (vB->type == GGML_TYPE_I64 || vB->type == GGML_TYPE_I32) &&
             (pNode->type == GGML_TYPE_F16 || pNode->type == GGML_TYPE_F32) &&
             vA->nb[0] == ggml_type_size(vA->type) && pNode->nb[0] == ggml_type_size(pNode->type) &&
             pNode->ne[0] == vA->ne[0] && pNode->ne[2] == vA->ne[2] && pNode->ne[3] == vA->ne[3] &&
             vA->ne[2] % vB->ne[1] == 0 && vA->ne[3] % vB->ne[2] == 0;
    case GGML_OP_RMS_NORM:
      return vA != nullptr && vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_F32 &&
             vA->nb[0] == sizeof(float) && ggml_is_contiguous(pNode) && ggml_are_same_shape(vA, pNode);
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_DIV:
      return vA != nullptr && vB != nullptr && vA->type == GGML_TYPE_F32 && vB->type == GGML_TYPE_F32 &&
             pNode->type == GGML_TYPE_F32 && vA->nb[0] == sizeof(float) && vB->nb[0] == sizeof(float) &&
             ggml_is_contiguous(pNode) && ggml_are_same_shape(vA, pNode) && vB->ne[0] > 0 && vB->ne[1] > 0 &&
             vB->ne[2] > 0 && vB->ne[3] > 0 && pNode->ne[0] % vB->ne[0] == 0 && pNode->ne[1] % vB->ne[1] == 0 &&
             pNode->ne[2] % vB->ne[2] == 0 && pNode->ne[3] % vB->ne[3] == 0;
    case GGML_OP_SCALE:
    case GGML_OP_CLAMP:
      return vA != nullptr && vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_F32 &&
             vA->nb[0] == sizeof(float) && ggml_is_contiguous(pNode) && ggml_are_same_shape(vA, pNode);
    case GGML_OP_UNARY: {
      if (vA == nullptr || vA->type != GGML_TYPE_F32 || pNode->type != GGML_TYPE_F32 || vA->nb[0] != sizeof(float) ||
          !ggml_is_contiguous(pNode) || !ggml_are_same_shape(vA, pNode)) {
        return false;
      }
      switch (ggml_get_unary_op(pNode)) {
        case GGML_UNARY_OP_SILU:
        case GGML_UNARY_OP_SIGMOID:
        case GGML_UNARY_OP_SOFTPLUS:
        case GGML_UNARY_OP_ABS:
        case GGML_UNARY_OP_CEIL:
        case GGML_UNARY_OP_EXP:
        case GGML_UNARY_OP_TANH:
        case GGML_UNARY_OP_RELU:
        case GGML_UNARY_OP_GELU:
        case GGML_UNARY_OP_GELU_ERF:
          return true;
        default:
          return false;
      }
    }
    case GGML_OP_SUM_ROWS:
      return vA != nullptr && vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_F32 &&
             vA->nb[0] == sizeof(float) && pNode->ne[0] == 1 && pNode->ne[1] == vA->ne[1] &&
             pNode->ne[2] == vA->ne[2] && pNode->ne[3] == vA->ne[3] && ggml_is_contiguous(pNode);
    case GGML_OP_SOFT_MAX:
      return vA != nullptr && vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_F32 &&
             vA->nb[0] == sizeof(float) && ggml_is_contiguous(pNode) && ggml_are_same_shape(vA, pNode) &&
             (vB == nullptr || vB->type == GGML_TYPE_F16 || vB->type == GGML_TYPE_F32) &&
             (vC == nullptr || (vC->type == GGML_TYPE_F32 && ggml_is_contiguous(vC)));
    case GGML_OP_ARGSORT:
      return vA != nullptr && vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_I32 &&
             vA->nb[0] == sizeof(float) && pNode->nb[0] == sizeof(int32_t) && ggml_are_same_shape(vA, pNode);
    case GGML_OP_CPY:
      return vA != nullptr && vB != nullptr && ggml_nelements(vA) == ggml_nelements(pNode) &&
             (((vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_F32) ||
               (vA->type == GGML_TYPE_F16 && pNode->type == GGML_TYPE_F16)) ||
              ((vA->type == GGML_TYPE_F32 || vA->type == GGML_TYPE_F16 || vA->type == GGML_TYPE_BF16) &&
               (pNode->type == GGML_TYPE_F32 || pNode->type == GGML_TYPE_F16) && ggml_is_contiguous(vA) &&
               ggml_is_contiguous(pNode)));
    case GGML_OP_SSM_CONV:
      return vA != nullptr && vB != nullptr && vA->type == GGML_TYPE_F32 && vB->type == GGML_TYPE_F32 &&
             pNode->type == GGML_TYPE_F32 && pNode->ne[2] == 1 && pNode->ne[3] == 1 && ggml_is_contiguous(pNode) &&
             vA->ne[0] == pNode->ne[1] + vB->ne[0] - 1 && vA->ne[1] == pNode->ne[0] && vB->ne[1] == pNode->ne[0];
    case GGML_OP_GATED_DELTA_NET: {
      const ggml_tensor * vG     = pNode->src[3];
      const ggml_tensor * vBeta  = pNode->src[4];
      const ggml_tensor * vState = pNode->src[5];
      if (vA == nullptr || vB == nullptr || vC == nullptr || vG == nullptr || vBeta == nullptr || vState == nullptr ||
          pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode)) {
        return false;
      }
      for (const ggml_tensor * vT : {vA, vB, vC, vG, vBeta, vState}) {
        if (vT->type != GGML_TYPE_F32) {
          return false;
        }
      }
      return vA->nb[0] == sizeof(float) && vB->nb[0] == sizeof(float) && vC->nb[0] == sizeof(float) &&
             ggml_is_contiguous(vG) && ggml_is_contiguous(vBeta) && ggml_is_contiguous(vState) &&
             vG->ne[0] == 1 && vBeta->ne[0] == 1 && vC->ne[0] == vA->ne[0] && vB->ne[0] == vA->ne[0] &&
             vA->ne[1] > 0 && vB->ne[1] == vA->ne[1] && vC->ne[1] % vA->ne[1] == 0 &&
             ggml_get_op_params_i32(pNode, 0) == 1;
    }
    case GGML_OP_FLASH_ATTN_EXT: {
      if (pShared.vSettings.vFull >= 2 && fAttnTaken(pShared, pNode)) {
        return true;
      }
      const ggml_tensor * vMask  = pNode->src[3];
      const ggml_tensor * vSinks = pNode->src[4];
      if (vA == nullptr || vB == nullptr || vC == nullptr || pNode->src[5] != nullptr ||
          vA->type != GGML_TYPE_F32 || vB->type != GGML_TYPE_F16 || vC->type != GGML_TYPE_F16 ||
          pNode->type != GGML_TYPE_F32 || !ggml_is_contiguous(pNode)) {
        return false;
      }
      float vScale   = 0.0f;
      float vMaxBias = 0.0f;
      float vSoftCap = 0.0f;
      std::memcpy(&vScale, reinterpret_cast<const float *>(pNode->op_params) + 0, sizeof(float));
      std::memcpy(&vMaxBias, reinterpret_cast<const float *>(pNode->op_params) + 1, sizeof(float));
      std::memcpy(&vSoftCap, reinterpret_cast<const float *>(pNode->op_params) + 2, sizeof(float));
      if (!std::isfinite(vScale) || vMaxBias != 0.0f || vSoftCap != 0.0f) {
        return false;
      }
      return vA->ne[0] == vB->ne[0] && vC->ne[1] == vB->ne[1] && vC->ne[2] == vB->ne[2] && vB->ne[2] > 0 &&
             vA->ne[2] % vB->ne[2] == 0 && vA->ne[3] == 1 && vB->ne[3] == 1 && vC->ne[3] == 1 &&
             vA->nb[0] == sizeof(float) && vB->nb[0] == sizeof(uint16_t) && vC->nb[0] == sizeof(uint16_t) &&
             (vMask == nullptr || (vMask->type == GGML_TYPE_F16 && vMask->nb[0] == sizeof(uint16_t))) &&
             (vSinks == nullptr || (vSinks->type == GGML_TYPE_F32 && vSinks->nb[0] == sizeof(float)));
    }
    default:
      return false;
  }
}

void fFullRopeYarn(float pThetaExtrap, float pFreqScale, const float aCorrDims[2], int64_t pI0, float pExtFactor,
                   float pMscale, float & pCos, float & pSin) {
  const float vThetaInterp = pFreqScale * pThetaExtrap;
  float       vTheta       = vThetaInterp;
  float       vMscale      = pMscale;
  if (pExtFactor != 0.0f) {
    const float vY       = (float(pI0 / 2) - aCorrDims[0]) / std::max(0.001f, aCorrDims[1] - aCorrDims[0]);
    const float vRampMix = (1.0f - std::min(1.0f, std::max(0.0f, vY))) * pExtFactor;
    vTheta               = vThetaInterp * (1.0f - vRampMix) + pThetaExtrap * vRampMix;
    vMscale *= 1.0f + 0.1f * logf(1.0f / pFreqScale);
  }
  pCos = cosf(vTheta) * vMscale;
  pSin = sinf(vTheta) * vMscale;
}

#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
const bool cHasFullAvx2Fma = __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");

__attribute__((target("avx2,fma"))) float fHsum256Avx2(__m256 pVec) {
  const __m128 vLo = _mm256_castps256_ps128(pVec);
  const __m128 vHi = _mm256_extractf128_ps(pVec, 1);
  __m128       vS4 = _mm_add_ps(vLo, vHi);
  vS4              = _mm_hadd_ps(vS4, vS4);
  vS4              = _mm_hadd_ps(vS4, vS4);
  return _mm_cvtss_f32(vS4);
}

__attribute__((target("avx2,fma"))) void fGdnHead128Avx2(float * pHead, const float * pQh, const float * pKh,
                                                         const float * pVh, float pDecay, float pBeta, float pScale,
                                                         float * pOut) {
  const __m256 vDec = _mm256_set1_ps(pDecay);
  for (int64_t j = 0; j < 128; j += 2) {
    float * vRowA  = pHead + j * 128;
    float * vRowB  = vRowA + 128;
    __m256  vDotA0 = _mm256_setzero_ps();
    __m256  vDotA1 = _mm256_setzero_ps();
    __m256  vDotB0 = _mm256_setzero_ps();
    __m256  vDotB1 = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 16) {
      const __m256 vA0 = _mm256_mul_ps(_mm256_loadu_ps(vRowA + i), vDec);
      const __m256 vA1 = _mm256_mul_ps(_mm256_loadu_ps(vRowA + i + 8), vDec);
      const __m256 vB0 = _mm256_mul_ps(_mm256_loadu_ps(vRowB + i), vDec);
      const __m256 vB1 = _mm256_mul_ps(_mm256_loadu_ps(vRowB + i + 8), vDec);
      const __m256 vK0 = _mm256_loadu_ps(pKh + i);
      const __m256 vK1 = _mm256_loadu_ps(pKh + i + 8);
      _mm256_storeu_ps(vRowA + i, vA0);
      _mm256_storeu_ps(vRowA + i + 8, vA1);
      _mm256_storeu_ps(vRowB + i, vB0);
      _mm256_storeu_ps(vRowB + i + 8, vB1);
      vDotA0 = _mm256_fmadd_ps(vA0, vK0, vDotA0);
      vDotA1 = _mm256_fmadd_ps(vA1, vK1, vDotA1);
      vDotB0 = _mm256_fmadd_ps(vB0, vK0, vDotB0);
      vDotB1 = _mm256_fmadd_ps(vB1, vK1, vDotB1);
    }
    const float  vDeltaA = (pVh[j] - fHsum256Avx2(_mm256_add_ps(vDotA0, vDotA1))) * pBeta;
    const float  vDeltaB = (pVh[j + 1] - fHsum256Avx2(_mm256_add_ps(vDotB0, vDotB1))) * pBeta;
    const __m256 vDa     = _mm256_set1_ps(vDeltaA);
    const __m256 vDb     = _mm256_set1_ps(vDeltaB);
    __m256       vOutA0  = _mm256_setzero_ps();
    __m256       vOutA1  = _mm256_setzero_ps();
    __m256       vOutB0  = _mm256_setzero_ps();
    __m256       vOutB1  = _mm256_setzero_ps();
    for (int i = 0; i < 128; i += 16) {
      const __m256 vK0 = _mm256_loadu_ps(pKh + i);
      const __m256 vK1 = _mm256_loadu_ps(pKh + i + 8);
      const __m256 vQ0 = _mm256_loadu_ps(pQh + i);
      const __m256 vQ1 = _mm256_loadu_ps(pQh + i + 8);
      const __m256 vA0 = _mm256_fmadd_ps(vK0, vDa, _mm256_loadu_ps(vRowA + i));
      const __m256 vA1 = _mm256_fmadd_ps(vK1, vDa, _mm256_loadu_ps(vRowA + i + 8));
      const __m256 vB0 = _mm256_fmadd_ps(vK0, vDb, _mm256_loadu_ps(vRowB + i));
      const __m256 vB1 = _mm256_fmadd_ps(vK1, vDb, _mm256_loadu_ps(vRowB + i + 8));
      _mm256_storeu_ps(vRowA + i, vA0);
      _mm256_storeu_ps(vRowA + i + 8, vA1);
      _mm256_storeu_ps(vRowB + i, vB0);
      _mm256_storeu_ps(vRowB + i + 8, vB1);
      vOutA0 = _mm256_fmadd_ps(vA0, vQ0, vOutA0);
      vOutA1 = _mm256_fmadd_ps(vA1, vQ1, vOutA1);
      vOutB0 = _mm256_fmadd_ps(vB0, vQ0, vOutB0);
      vOutB1 = _mm256_fmadd_ps(vB1, vQ1, vOutB1);
    }
    pOut[j]     = fHsum256Avx2(_mm256_add_ps(vOutA0, vOutA1)) * pScale;
    pOut[j + 1] = fHsum256Avx2(_mm256_add_ps(vOutB0, vOutB1)) * pScale;
  }
}

__attribute__((target("avx2,fma"))) void fScaleRowAvx2(const float * pSrc, float * pDst, float pScale, int64_t pCount) {
  const __m256 vSc = _mm256_set1_ps(pScale);
  int64_t      k   = 0;
  for (; k + 8 <= pCount; k += 8) {
    _mm256_storeu_ps(pDst + k, _mm256_mul_ps(_mm256_loadu_ps(pSrc + k), vSc));
  }
  for (; k < pCount; k++) {
    pDst[k] = pSrc[k] * pScale;
  }
}

__attribute__((target("avx2,f16c"))) void fCvtF32ToF16RowAvx2(const float * pSrc, ggml_fp16_t * pDst, int64_t pCount) {
  int64_t k = 0;
  for (; k + 8 <= pCount; k += 8) {
    const __m256  vVec = _mm256_loadu_ps(pSrc + k);
    const __m128i vPh  = _mm256_cvtps_ph(vVec, _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128(reinterpret_cast<__m128i *>(pDst + k), vPh);
  }
  for (; k < pCount; k++) {
    pDst[k] = ggml_fp32_to_fp16(pSrc[k]);
  }
}

__attribute__((target("avx2,f16c"))) float fDotF32F16Avx2(const float * pQ, const ggml_fp16_t * pK, int64_t pCount) {
  __m256 vSumVec = _mm256_setzero_ps();
  for (int64_t d = 0; d < pCount; d += 8) {
    const __m128i vHalf = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pK + d));
    const __m256  vKf   = _mm256_cvtph_ps(vHalf);
    const __m256  vQf   = _mm256_loadu_ps(pQ + d);
    vSumVec             = _mm256_add_ps(vSumVec, _mm256_mul_ps(vQf, vKf));
  }
  const __m128 vLo = _mm256_castps256_ps128(vSumVec);
  const __m128 vHi = _mm256_extractf128_ps(vSumVec, 1);
  __m128       vS4 = _mm_add_ps(vLo, vHi);
  vS4              = _mm_hadd_ps(vS4, vS4);
  vS4              = _mm_hadd_ps(vS4, vS4);
  return _mm_cvtss_f32(vS4);
}

__attribute__((target("avx2,f16c"))) void fMadF16Avx2(float pWeight, const ggml_fp16_t * pV, float * pAcc,
                                                      int64_t pCount) {
  const __m256 vWVec = _mm256_set1_ps(pWeight);
  for (int64_t d = 0; d < pCount; d += 8) {
    const __m128i vHalf = _mm_loadu_si128(reinterpret_cast<const __m128i *>(pV + d));
    const __m256  vVf   = _mm256_cvtph_ps(vHalf);
    const __m256  vCur  = _mm256_loadu_ps(pAcc + d);
    _mm256_storeu_ps(pAcc + d, _mm256_add_ps(vCur, _mm256_mul_ps(vWVec, vVf)));
  }
}
#endif

bool fFullOpOnHost(XdnaBackend & pBackend, ggml_tensor * pNode, std::string & pError) {
  for (int i = 0; i < GGML_MAX_SRC; i++) {
    if (pNode->src[i] != nullptr && pNode->src[i]->data == nullptr) {
      pError = "a tensor of a full-subgraph operation has no memory";
      return false;
    }
  }
  if (pNode->data == nullptr) {
    pError = "the output tensor of a full-subgraph operation has no memory";
    return false;
  }
  const double        vStart = fNowMs();
  XdnaShared &        vShared = fShared();
  const ggml_tensor * vA      = pNode->src[0];
  const ggml_tensor * vB      = pNode->src[1];
  const ggml_tensor * vC      = pNode->src[2];
  if (fInStateBuffer(pNode) || (pNode->op == GGML_OP_GATED_DELTA_NET && fInStateBuffer(pNode->src[5]))) {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.sCleanStates.clear();
  }
  auto fParallel = [&](uint32_t pItems, auto && fWork) {
    if (pItems <= 1 || pBackend.vThreads <= 1) {
      for (uint32_t t = 0; t < pItems; t++) {
        fWork(t);
      }
      return;
    }
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.vRunner.mSetThreads(uint32_t(std::max(1, pBackend.vThreads)));
    vShared.vRunner.mWorkers().mForEach(pItems, fWork);
  };

  switch (pNode->op) {
    case GGML_OP_GET_ROWS: {
      const int64_t vNc   = vA->ne[0];
      const int64_t vNe10 = vB->ne[0];
      const int64_t vNe11 = vB->ne[1];
      const int64_t vNe12 = vB->ne[2];
      const int64_t vRows = vNe10 * vNe11 * vNe12;
      for (int64_t r = 0; r < vRows; r++) {
        const int64_t i12 = r / (vNe11 * vNe10);
        const int64_t i11 = (r - i12 * vNe11 * vNe10) / vNe10;
        const int64_t i10 = r - i12 * vNe11 * vNe10 - i11 * vNe10;
        const int32_t vId = *reinterpret_cast<const int32_t *>(
            static_cast<const char *>(vB->data) + i10 * vB->nb[0] + i11 * vB->nb[1] + i12 * vB->nb[2]);
        if (vId < 0 || vId >= vA->ne[1]) {
          pError = "GET_ROWS index out of bounds";
          return false;
        }
      }
      const uint32_t vTasks = vRows * vNc >= 65536 ? uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) : 1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vR0 = vRows * pTask / vTasks;
        const int64_t vR1 = vRows * (pTask + 1) / vTasks;
        for (int64_t r = vR0; r < vR1; r++) {
          const int64_t i12  = r / (vNe11 * vNe10);
          const int64_t i11  = (r - i12 * vNe11 * vNe10) / vNe10;
          const int64_t i10  = r - i12 * vNe11 * vNe10 - i11 * vNe10;
          const int32_t vId  = *reinterpret_cast<const int32_t *>(
              static_cast<const char *>(vB->data) + i10 * vB->nb[0] + i11 * vB->nb[1] + i12 * vB->nb[2]);
          const char *  vSrc = static_cast<const char *>(vA->data) + size_t(vId) * vA->nb[1] +
                               size_t(i11) * vA->nb[2] + size_t(i12) * vA->nb[3];
          float *       vDst = reinterpret_cast<float *>(
              static_cast<char *>(pNode->data) + size_t(i10) * pNode->nb[1] + size_t(i11) * pNode->nb[2] +
              size_t(i12) * pNode->nb[3]);
          if (vA->type == GGML_TYPE_F16) {
            const ggml_fp16_t * vH = reinterpret_cast<const ggml_fp16_t *>(vSrc);
            for (int64_t k = 0; k < vNc; k++) {
              vDst[k] = ggml_fp16_to_fp32(vH[k]);
            }
          } else {
            fDequantizeWeightRow(vA->type, vSrc, vDst, uint32_t(vNc));
          }
        }
      });
      break;
    }
    case GGML_OP_DUP:
    case GGML_OP_CONT:
    case GGML_OP_CPY: {
      if (vA->type != pNode->type) {
        const int64_t vN = ggml_nelements(pNode);
        if (vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_F16) {
          const float * vF = static_cast<const float *>(vA->data);
          ggml_fp16_t * vH = static_cast<ggml_fp16_t *>(pNode->data);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
          if (cHasDequantAvx2F16c) {
            fCvtF32ToF16RowAvx2(vF, vH, vN);
            break;
          }
#endif
          for (int64_t k = 0; k < vN; k++) {
            vH[k] = ggml_fp32_to_fp16(vF[k]);
          }
        } else if (vA->type == GGML_TYPE_F16 && pNode->type == GGML_TYPE_F32) {
          const ggml_fp16_t * vH = static_cast<const ggml_fp16_t *>(vA->data);
          float *             vF = static_cast<float *>(pNode->data);
          for (int64_t k = 0; k < vN; k++) {
            vF[k] = ggml_fp16_to_fp32(vH[k]);
          }
        } else if (vA->type == GGML_TYPE_BF16 && pNode->type == GGML_TYPE_F32) {
          fDequantizeWeightRow(GGML_TYPE_BF16, static_cast<const char *>(vA->data), static_cast<float *>(pNode->data),
                               uint32_t(vN));
        } else {
          const uint16_t * vBf = static_cast<const uint16_t *>(vA->data);
          ggml_fp16_t *    vH  = static_cast<ggml_fp16_t *>(pNode->data);
          for (int64_t k = 0; k < vN; k++) {
            const uint32_t vBits = uint32_t(vBf[k]) << 16;
            float          vVal  = 0.0f;
            std::memcpy(&vVal, &vBits, sizeof(float));
            vH[k] = ggml_fp32_to_fp16(vVal);
          }
        }
        break;
      }
      const size_t vElemSize = ggml_type_size(pNode->type);
      if (ggml_is_contiguous(vA) && ggml_is_contiguous(pNode)) {
        if (vA->data != pNode->data) {
          std::memcpy(pNode->data, vA->data, size_t(ggml_nelements(pNode)) * vElemSize);
        }
        break;
      }
      if (vA->nb[0] == vElemSize && ggml_is_contiguous(pNode) && vA->ne[0] == pNode->ne[0]) {
        const int64_t  vRows     = vA->ne[1] * vA->ne[2] * vA->ne[3];
        const size_t   vRowBytes = size_t(vA->ne[0]) * vElemSize;
        const uint32_t vTasks    = vRows * vA->ne[0] >= 65536 ?
                                       uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) :
                                       1u;
        fParallel(vTasks, [&](uint32_t pTask) {
          const int64_t vR0 = vRows * pTask / vTasks;
          const int64_t vR1 = vRows * (pTask + 1) / vTasks;
          for (int64_t r = vR0; r < vR1; r++) {
            const int64_t i1 = r % vA->ne[1];
            const int64_t i2 = (r / vA->ne[1]) % vA->ne[2];
            const int64_t i3 = r / (vA->ne[1] * vA->ne[2]);
            std::memcpy(static_cast<char *>(pNode->data) + size_t(r) * vRowBytes,
                        static_cast<const char *>(vA->data) + size_t(i1) * vA->nb[1] + size_t(i2) * vA->nb[2] +
                            size_t(i3) * vA->nb[3],
                        vRowBytes);
          }
        });
        break;
      }
      if (ggml_are_same_shape(vA, pNode)) {
        for (int64_t i3 = 0; i3 < vA->ne[3]; i3++) {
          for (int64_t i2 = 0; i2 < vA->ne[2]; i2++) {
            for (int64_t i1 = 0; i1 < vA->ne[1]; i1++) {
              const char * vSr = static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3];
              char * vDr = static_cast<char *>(pNode->data) + i1 * pNode->nb[1] + i2 * pNode->nb[2] + i3 * pNode->nb[3];
              if (vElemSize == sizeof(uint32_t)) {
                for (int64_t i0 = 0; i0 < vA->ne[0]; i0++) {
                  *reinterpret_cast<uint32_t *>(vDr + i0 * pNode->nb[0]) =
                      *reinterpret_cast<const uint32_t *>(vSr + i0 * vA->nb[0]);
                }
              } else {
                for (int64_t i0 = 0; i0 < vA->ne[0]; i0++) {
                  *reinterpret_cast<uint16_t *>(vDr + i0 * pNode->nb[0]) =
                      *reinterpret_cast<const uint16_t *>(vSr + i0 * vA->nb[0]);
                }
              }
            }
          }
        }
        break;
      }
      if (ggml_is_contiguous(pNode)) {
        char * vDp = static_cast<char *>(pNode->data);
        if (vElemSize == sizeof(uint32_t)) {
          uint32_t * vOut32 = reinterpret_cast<uint32_t *>(vDp);
          for (int64_t i3 = 0; i3 < vA->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < vA->ne[2]; i2++) {
              for (int64_t i1 = 0; i1 < vA->ne[1]; i1++) {
                const char * vSr = static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3];
                for (int64_t i0 = 0; i0 < vA->ne[0]; i0++) {
                  *vOut32++ = *reinterpret_cast<const uint32_t *>(vSr + i0 * vA->nb[0]);
                }
              }
            }
          }
        } else {
          uint16_t * vOut16 = reinterpret_cast<uint16_t *>(vDp);
          for (int64_t i3 = 0; i3 < vA->ne[3]; i3++) {
            for (int64_t i2 = 0; i2 < vA->ne[2]; i2++) {
              for (int64_t i1 = 0; i1 < vA->ne[1]; i1++) {
                const char * vSr = static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3];
                for (int64_t i0 = 0; i0 < vA->ne[0]; i0++) {
                  *vOut16++ = *reinterpret_cast<const uint16_t *>(vSr + i0 * vA->nb[0]);
                }
              }
            }
          }
        }
        break;
      }
      const int64_t vTotal = ggml_nelements(pNode);
      const char *  vSrc0  = static_cast<const char *>(vA->data);
      char *        vDst0  = static_cast<char *>(pNode->data);
      for (int64_t n = 0; n < vTotal; n++) {
        const int64_t a0 = n % vA->ne[0];
        const int64_t a1 = (n / vA->ne[0]) % vA->ne[1];
        const int64_t a2 = (n / (vA->ne[0] * vA->ne[1])) % vA->ne[2];
        const int64_t a3 = n / (vA->ne[0] * vA->ne[1] * vA->ne[2]);
        const int64_t d0 = n % pNode->ne[0];
        const int64_t d1 = (n / pNode->ne[0]) % pNode->ne[1];
        const int64_t d2 = (n / (pNode->ne[0] * pNode->ne[1])) % pNode->ne[2];
        const int64_t d3 = n / (pNode->ne[0] * pNode->ne[1] * pNode->ne[2]);
        std::memcpy(vDst0 + d0 * pNode->nb[0] + d1 * pNode->nb[1] + d2 * pNode->nb[2] + d3 * pNode->nb[3],
                    vSrc0 + a0 * vA->nb[0] + a1 * vA->nb[1] + a2 * vA->nb[2] + a3 * vA->nb[3], vElemSize);
      }
      break;
    }
    case GGML_OP_ROPE: {
      const int32_t * vParams     = reinterpret_cast<const int32_t *>(pNode->op_params);
      const int32_t   vNDims      = vParams[1];
      const int32_t   vMode       = vParams[2];
      const int32_t   vNCtxOrig   = vParams[4];
      const bool      vTruncate   = ggml_rope_get_truncate(pNode);
      const int32_t   vNOffs      = vParams[15];
      float           vFreqBase   = 0.0f;
      float           vFreqScale  = 0.0f;
      float           vExtFactor  = 0.0f;
      float           vAttnFactor = 0.0f;
      float           vBetaFast   = 0.0f;
      float           vBetaSlow   = 0.0f;
      int             aSections[4] = {};
      std::memcpy(&vFreqBase, vParams + 5, sizeof(float));
      std::memcpy(&vFreqScale, vParams + 6, sizeof(float));
      std::memcpy(&vExtFactor, vParams + 7, sizeof(float));
      std::memcpy(&vAttnFactor, vParams + 8, sizeof(float));
      std::memcpy(&vBetaFast, vParams + 9, sizeof(float));
      std::memcpy(&vBetaSlow, vParams + 10, sizeof(float));
      std::memcpy(aSections, vParams + 11, sizeof(aSections));
      const int64_t vNe0 = pNode->ne[0];
      const int64_t vNe1 = pNode->ne[1];
      const int64_t vNe2 = pNode->ne[2];
      const int64_t vNe3 = pNode->ne[3];
      if (vNDims < 0 || vNDims % 2 != 0 || vNOffs < 0 || vNOffs % 2 != 0 || vNOffs + vNDims > vNe0) {
        pError = "invalid RoPE dimensions";
        return false;
      }
      const float   vThetaScale  = powf(vFreqBase, -2.0f / float(std::max(1, vNDims)));
      float         aCorrDims[2] = {};
      ggml_rope_yarn_corr_dims(vNDims, vNCtxOrig, vFreqBase, vBetaFast, vBetaSlow, vTruncate, aCorrDims);
      const bool      vIsImrope    = vMode == GGML_ROPE_TYPE_IMROPE;
      const bool      vMropeUsed   = (vMode & GGML_ROPE_TYPE_MROPE) != 0;
      const bool      vIsVision    = vMode == GGML_ROPE_TYPE_VISION;
      const float *   vFreqFactors = vC != nullptr ? static_cast<const float *>(vC->data) : nullptr;
      const int32_t * vPos         = static_cast<const int32_t *>(vB->data);
      const int64_t   vRotN        = vIsVision ? vNe0 : vNDims;
      const int64_t   vRows        = vNe1 * vNe2 * vNe3;
      const uint32_t  vTasks       = vRows * vNe0 >= 32768 ?
                                         uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) :
                                         1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t      vR0 = vRows * pTask / vTasks;
        const int64_t      vR1 = vRows * (pTask + 1) / vTasks;
        std::vector<float> aCache(static_cast<size_t>(std::max<int64_t>(2, vRotN)));
        int64_t            vLastI2 = -1;
        for (int64_t r = vR0; r < vR1; r++) {
          const int64_t i1 = r % vNe1;
          const int64_t i2 = (r / vNe1) % vNe2;
          const int64_t i3 = r / (vNe1 * vNe2);
          if (vLastI2 != i2) {
            if (!vMropeUsed) {
              float vTheta = float(vPos[i2]);
              for (int64_t i0 = 0; i0 < vRotN; i0 += 2) {
                const float vFf = vFreqFactors != nullptr ? vFreqFactors[i0 / 2] : 1.0f;
                fFullRopeYarn(vTheta / vFf, vFreqScale, aCorrDims, i0, vExtFactor, vAttnFactor, aCache[size_t(i0)],
                              aCache[size_t(i0 + 1)]);
                vTheta *= vThetaScale;
              }
            } else {
              const float vBaseT    = float(vPos[i2]);
              const float vBaseH    = float(vPos[i2 + vNe2]);
              const float vBaseW    = float(vPos[i2 + vNe2 * 2]);
              const float vBaseE    = float(vPos[i2 + vNe2 * 3]);
              float       vThetaT   = vBaseT;
              float       vThetaH   = vBaseH;
              float       vThetaW   = vBaseW;
              float       vThetaE   = vBaseE;
              const int   vSectDims = aSections[0] + aSections[1] + aSections[2] + aSections[3];
              const int   vSecW     = aSections[1] + aSections[0];
              const int   vSecE     = aSections[2] + vSecW;
              for (int64_t i0 = 0; i0 < vRotN; i0 += 2) {
                const float vFf     = vFreqFactors != nullptr ? vFreqFactors[i0 / 2] : 1.0f;
                const int   vSector = int((i0 / 2) % std::max(1, vSectDims));
                if (vIsVision) {
                  if (vSector == 0) {
                    vThetaT = vBaseT;
                  } else if (vSector == aSections[0]) {
                    vThetaH = vBaseH;
                  } else if (vSector == vSecW) {
                    vThetaW = vBaseW;
                  } else if (vSector == vSecE) {
                    vThetaE = vBaseE;
                  }
                }
                float vTheta = vThetaT;
                if (vIsImrope) {
                  if (vSector % 3 == 1 && vSector < 3 * aSections[1]) {
                    vTheta = vThetaH;
                  } else if (vSector % 3 == 2 && vSector < 3 * aSections[2]) {
                    vTheta = vThetaW;
                  } else if (vSector % 3 == 0 && vSector < 3 * aSections[0]) {
                    vTheta = vThetaT;
                  } else {
                    vTheta = vThetaE;
                  }
                } else {
                  if (vSector >= aSections[0] && vSector < vSecW) {
                    vTheta = vThetaH;
                  } else if (vSector >= vSecW && vSector < vSecW + aSections[2]) {
                    vTheta = vThetaW;
                  } else if (vSector >= vSecW + aSections[2]) {
                    vTheta = vThetaE;
                  }
                }
                fFullRopeYarn(vTheta / vFf, vFreqScale, aCorrDims, i0, vExtFactor, vAttnFactor, aCache[size_t(i0)],
                              aCache[size_t(i0 + 1)]);
                vThetaT *= vThetaScale;
                vThetaW *= vThetaScale;
                vThetaH *= vThetaScale;
                vThetaE *= vThetaScale;
              }
            }
            vLastI2 = i2;
          }
          const char * vSrcRow = static_cast<const char *>(vA->data) + i3 * vA->nb[3] + i2 * vA->nb[2] + i1 * vA->nb[1];
          char * vDstRow = static_cast<char *>(pNode->data) + i3 * pNode->nb[3] + i2 * pNode->nb[2] + i1 * pNode->nb[1];
          const int64_t vOffset = vMode == GGML_ROPE_TYPE_NORMAL ? 1 : (vIsVision ? vNDims : vNDims / 2);
          const int64_t vScale  = vMode == GGML_ROPE_TYPE_NORMAL ? 1 : 2;
          const int64_t vBase   = vIsVision ? 0 : vNOffs;
          if (pNode->type == GGML_TYPE_F32) {
            const float * vS = reinterpret_cast<const float *>(vSrcRow);
            float *       vD = reinterpret_cast<float *>(vDstRow);
            for (int64_t i0 = 0; i0 < vRotN; i0 += 2) {
              const int64_t vIc  = vBase + i0 / vScale;
              const float   vCos = aCache[size_t(i0)];
              const float   vSin = aCache[size_t(i0 + 1)];
              const float   vX0  = vS[vIc];
              const float   vX1  = vS[vIc + vOffset];
              vD[vIc]            = vX0 * vCos - vX1 * vSin;
              vD[vIc + vOffset]  = vX0 * vSin + vX1 * vCos;
            }
            if (!vIsVision) {
              for (int64_t i0 = 0; i0 < vNe0; i0 += 2) {
                if (i0 == vNOffs && vNDims > 0) {
                  i0 += vNDims - 2;
                  continue;
                }
                vD[i0]     = vS[i0];
                vD[i0 + 1] = vS[i0 + 1];
              }
            }
          } else {
            const ggml_fp16_t * vS = reinterpret_cast<const ggml_fp16_t *>(vSrcRow);
            ggml_fp16_t *       vD = reinterpret_cast<ggml_fp16_t *>(vDstRow);
            for (int64_t i0 = 0; i0 < vRotN; i0 += 2) {
              const int64_t vIc  = vBase + i0 / vScale;
              const float   vCos = aCache[size_t(i0)];
              const float   vSin = aCache[size_t(i0 + 1)];
              const float   vX0  = ggml_fp16_to_fp32(vS[vIc]);
              const float   vX1  = ggml_fp16_to_fp32(vS[vIc + vOffset]);
              vD[vIc]            = ggml_fp32_to_fp16(vX0 * vCos - vX1 * vSin);
              vD[vIc + vOffset]  = ggml_fp32_to_fp16(vX0 * vSin + vX1 * vCos);
            }
            if (!vIsVision) {
              for (int64_t i0 = 0; i0 < vNe0; i0 += 2) {
                if (i0 == vNOffs && vNDims > 0) {
                  i0 += vNDims - 2;
                  continue;
                }
                vD[i0]     = vS[i0];
                vD[i0 + 1] = vS[i0 + 1];
              }
            }
          }
        }
      });
      break;
    }
    case GGML_OP_SET_ROWS: {
      const int64_t  vNc    = vA->ne[0];
      const int64_t  vNr    = vA->ne[1];
      const uint32_t vTasks = vNr * vNc >= 65536 ? uint32_t(std::min<int64_t>(vNr, std::max(1, pBackend.vThreads))) : 1u;
      for (int64_t i03 = 0; i03 < vA->ne[3]; i03++) {
        for (int64_t i02 = 0; i02 < vA->ne[2]; i02++) {
          for (int64_t i = 0; i < vNr; i++) {
            const char * vIdxPtr = static_cast<const char *>(vB->data) + i * vB->nb[0] +
                                   (i02 % vB->ne[1]) * vB->nb[1] + (i03 % vB->ne[2]) * vB->nb[2];
            const int64_t vRow = vB->type == GGML_TYPE_I64 ? *reinterpret_cast<const int64_t *>(vIdxPtr) :
                                                             int64_t(*reinterpret_cast<const int32_t *>(vIdxPtr));
            if (vRow < 0 || vRow >= pNode->ne[1]) {
              pError = "SET_ROWS index out of bounds";
              return false;
            }
          }
        }
      }
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vR0 = vNr * pTask / vTasks;
        const int64_t vR1 = vNr * (pTask + 1) / vTasks;
        for (int64_t i03 = 0; i03 < vA->ne[3]; i03++) {
          for (int64_t i02 = 0; i02 < vA->ne[2]; i02++) {
            for (int64_t i = vR0; i < vR1; i++) {
              const char * vIdxPtr = static_cast<const char *>(vB->data) + i * vB->nb[0] +
                                     (i02 % vB->ne[1]) * vB->nb[1] + (i03 % vB->ne[2]) * vB->nb[2];
              const int64_t vRow = vB->type == GGML_TYPE_I64 ? *reinterpret_cast<const int64_t *>(vIdxPtr) :
                                                               int64_t(*reinterpret_cast<const int32_t *>(vIdxPtr));
              const char * vSrc = static_cast<const char *>(vA->data) + i * vA->nb[1] + i02 * vA->nb[2] +
                                  i03 * vA->nb[3];
              char * vDst = static_cast<char *>(pNode->data) + vRow * pNode->nb[1] + i02 * pNode->nb[2] +
                            i03 * pNode->nb[3];
              if (vA->type == pNode->type) {
                std::memcpy(vDst, vSrc, size_t(vNc) * ggml_type_size(pNode->type));
              } else if (vA->type == GGML_TYPE_F32 && pNode->type == GGML_TYPE_F16) {
                const float * vF = reinterpret_cast<const float *>(vSrc);
                ggml_fp16_t * vH = reinterpret_cast<ggml_fp16_t *>(vDst);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
                if (cHasDequantAvx2F16c) {
                  fCvtF32ToF16RowAvx2(vF, vH, vNc);
                } else
#endif
                {
                  for (int64_t k = 0; k < vNc; k++) {
                    vH[k] = ggml_fp32_to_fp16(vF[k]);
                  }
                }
              } else {
                const ggml_fp16_t * vH = reinterpret_cast<const ggml_fp16_t *>(vSrc);
                float *             vF = reinterpret_cast<float *>(vDst);
                for (int64_t k = 0; k < vNc; k++) {
                  vF[k] = ggml_fp16_to_fp32(vH[k]);
                }
              }
            }
          }
        }
      });
      break;
    }
    case GGML_OP_RMS_NORM: {
      float vEps = 0.0f;
      std::memcpy(&vEps, pNode->op_params, sizeof(float));
      const int64_t  vLen   = pNode->ne[0];
      const int64_t  vRows  = ggml_nrows(pNode);
      const uint32_t vTasks = vRows * vLen >= 65536 ? uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) : 1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vR0 = vRows * pTask / vTasks;
        const int64_t vR1 = vRows * (pTask + 1) / vTasks;
        for (int64_t r = vR0; r < vR1; r++) {
          const int64_t i1    = r % vA->ne[1];
          const int64_t i2    = (r / vA->ne[1]) % vA->ne[2];
          const int64_t i3    = r / (vA->ne[1] * vA->ne[2]);
          const float * vFrom = reinterpret_cast<const float *>(
              static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3]);
          float * vTo    = static_cast<float *>(pNode->data) + r * vLen;
          float   vScale = 0.0f;
          fXdnaRmsScales(vFrom, 1, uint64_t(vLen), uint64_t(vLen), vEps, &vScale);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
          if (cHasFullAvx2Fma) {
            fScaleRowAvx2(vFrom, vTo, vScale, vLen);
            continue;
          }
#endif
          for (int64_t k = 0; k < vLen; k++) {
            vTo[k] = vFrom[k] * vScale;
          }
        }
      });
      break;
    }
    case GGML_OP_ADD:
    case GGML_OP_MUL:
    case GGML_OP_DIV: {
      const int64_t  vNe0   = pNode->ne[0];
      const int64_t  vRows  = pNode->ne[1] * pNode->ne[2] * pNode->ne[3];
      const uint32_t vTasks = vRows * vNe0 >= 65536 ? uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) : 1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vR0 = vRows * pTask / vTasks;
        const int64_t vR1 = vRows * (pTask + 1) / vTasks;
        for (int64_t r = vR0; r < vR1; r++) {
          const int64_t i1  = r % pNode->ne[1];
          const int64_t i2  = (r / pNode->ne[1]) % pNode->ne[2];
          const int64_t i3  = r / (pNode->ne[1] * pNode->ne[2]);
          const float * vSa = reinterpret_cast<const float *>(
              static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3]);
          const float * vSb = reinterpret_cast<const float *>(
              static_cast<const char *>(vB->data) + (i1 % vB->ne[1]) * vB->nb[1] + (i2 % vB->ne[2]) * vB->nb[2] +
              (i3 % vB->ne[3]) * vB->nb[3]);
          float * vDst = static_cast<float *>(pNode->data) + r * vNe0;
          if (pNode->op == GGML_OP_ADD) {
            for (int64_t i0 = 0; i0 < vNe0; i0++) {
              vDst[i0] = vSa[i0] + vSb[i0 % vB->ne[0]];
            }
          } else if (pNode->op == GGML_OP_MUL) {
            for (int64_t i0 = 0; i0 < vNe0; i0++) {
              vDst[i0] = vSa[i0] * vSb[i0 % vB->ne[0]];
            }
          } else {
            for (int64_t i0 = 0; i0 < vNe0; i0++) {
              vDst[i0] = vSa[i0] / vSb[i0 % vB->ne[0]];
            }
          }
        }
      });
      break;
    }
    case GGML_OP_SCALE: {
      float aParams[2] = {};
      std::memcpy(aParams, pNode->op_params, sizeof(aParams));
      const int64_t vNe0  = pNode->ne[0];
      const int64_t vRows = ggml_nrows(pNode);
      for (int64_t r = 0; r < vRows; r++) {
        const int64_t i1  = r % vA->ne[1];
        const int64_t i2  = (r / vA->ne[1]) % vA->ne[2];
        const int64_t i3  = r / (vA->ne[1] * vA->ne[2]);
        const float * vSa = reinterpret_cast<const float *>(
            static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3]);
        float * vDst = static_cast<float *>(pNode->data) + r * vNe0;
        for (int64_t i0 = 0; i0 < vNe0; i0++) {
          vDst[i0] = aParams[1] == 0.0f ? vSa[i0] * aParams[0] : vSa[i0] * aParams[0] + aParams[1];
        }
      }
      break;
    }
    case GGML_OP_CLAMP: {
      float aParams[2] = {};
      std::memcpy(aParams, pNode->op_params, sizeof(aParams));
      const int64_t vNe0  = pNode->ne[0];
      const int64_t vRows = ggml_nrows(pNode);
      for (int64_t r = 0; r < vRows; r++) {
        const int64_t i1  = r % vA->ne[1];
        const int64_t i2  = (r / vA->ne[1]) % vA->ne[2];
        const int64_t i3  = r / (vA->ne[1] * vA->ne[2]);
        const float * vSa = reinterpret_cast<const float *>(
            static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3]);
        float * vDst = static_cast<float *>(pNode->data) + r * vNe0;
        for (int64_t i0 = 0; i0 < vNe0; i0++) {
          const float vX = vSa[i0];
          vDst[i0]       = vX < aParams[0] ? aParams[0] : (vX > aParams[1] ? aParams[1] : vX);
        }
      }
      break;
    }
    case GGML_OP_UNARY: {
      const enum ggml_unary_op vOp    = ggml_get_unary_op(pNode);
      const int64_t            vNe0   = pNode->ne[0];
      const int64_t            vRows  = ggml_nrows(pNode);
      const uint32_t           vTasks = vRows * vNe0 >= 65536 ?
                                            uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) :
                                            1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vR0 = vRows * pTask / vTasks;
        const int64_t vR1 = vRows * (pTask + 1) / vTasks;
        for (int64_t r = vR0; r < vR1; r++) {
          const int64_t i1  = r % vA->ne[1];
          const int64_t i2  = (r / vA->ne[1]) % vA->ne[2];
          const int64_t i3  = r / (vA->ne[1] * vA->ne[2]);
          const float * vSa = reinterpret_cast<const float *>(
              static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3]);
          float * vDst = static_cast<float *>(pNode->data) + r * vNe0;
          if (vOp == GGML_UNARY_OP_SILU) {
            fXdnaSilu(vSa, vDst, uint32_t(vNe0));
            continue;
          }
          for (int64_t i0 = 0; i0 < vNe0; i0++) {
            const float vX = vSa[i0];
            switch (vOp) {
              case GGML_UNARY_OP_SIGMOID:  vDst[i0] = 1.0f / (1.0f + expf(-vX)); break;
              case GGML_UNARY_OP_SOFTPLUS: vDst[i0] = vX > 20.0f ? vX : log1pf(expf(vX)); break;
              case GGML_UNARY_OP_ABS:      vDst[i0] = std::fabs(vX); break;
              case GGML_UNARY_OP_CEIL:     vDst[i0] = std::ceil(vX); break;
              case GGML_UNARY_OP_EXP:      vDst[i0] = expf(vX); break;
              case GGML_UNARY_OP_TANH:     vDst[i0] = tanhf(vX); break;
              case GGML_UNARY_OP_RELU:     vDst[i0] = vX > 0.0f ? vX : 0.0f; break;
              case GGML_UNARY_OP_GELU: {
                constexpr float cSqrt2Pi = 0.79788456080286535587989211986876f;
                constexpr float cCoef    = 0.044715f;
                vDst[i0]                 = 0.5f * vX * (1.0f + tanhf(cSqrt2Pi * vX * (1.0f + cCoef * vX * vX)));
                break;
              }
              case GGML_UNARY_OP_GELU_ERF: {
                constexpr float cInvSqrt2 = 0.70710678118654752440084436210485f;
                vDst[i0]                  = 0.5f * vX * (1.0f + erff(vX * cInvSqrt2));
                break;
              }
              default: vDst[i0] = vX; break;
            }
          }
        }
      });
      break;
    }
    case GGML_OP_SUM_ROWS: {
      const int64_t vNe0  = vA->ne[0];
      const int64_t vRows = ggml_nrows(vA);
      float *       vDst  = static_cast<float *>(pNode->data);
      for (int64_t r = 0; r < vRows; r++) {
        const int64_t i1  = r % vA->ne[1];
        const int64_t i2  = (r / vA->ne[1]) % vA->ne[2];
        const int64_t i3  = r / (vA->ne[1] * vA->ne[2]);
        const float * vSa = reinterpret_cast<const float *>(
            static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3]);
        double vSum = 0.0;
        for (int64_t i0 = 0; i0 < vNe0; i0++) {
          vSum += double(vSa[i0]);
        }
        vDst[r] = float(vSum);
      }
      break;
    }
    case GGML_OP_SOFT_MAX: {
      float vScale   = 1.0f;
      float vMaxBias = 0.0f;
      std::memcpy(&vScale, reinterpret_cast<const float *>(pNode->op_params) + 0, sizeof(float));
      std::memcpy(&vMaxBias, reinterpret_cast<const float *>(pNode->op_params) + 1, sizeof(float));
      const int64_t  vNe00     = vA->ne[0];
      const int64_t  vNe01     = vA->ne[1];
      const int64_t  vNe02     = vA->ne[2];
      const int64_t  vNe03     = vA->ne[3];
      const uint32_t vHead     = uint32_t(std::max<int64_t>(1, vNe02));
      const uint32_t vHeadLog2 = 1u << uint32_t(std::floor(std::log2(double(vHead))));
      const float    vM0       = powf(2.0f, -vMaxBias / float(vHeadLog2));
      const float    vM1       = powf(2.0f, -(vMaxBias / 2.0f) / float(vHeadLog2));
      const bool     vUseF16   = vB != nullptr && vB->type == GGML_TYPE_F16;
      const float *  vSinks    = vC != nullptr ? static_cast<const float *>(vC->data) : nullptr;
      const int64_t  vRows     = vNe01 * vNe02 * vNe03;
      const uint32_t vTasks    = vRows * vNe00 >= 32768 ?
                                     uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) :
                                     1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t      vR0 = vRows * pTask / vTasks;
        const int64_t      vR1 = vRows * (pTask + 1) / vTasks;
        std::vector<float> aWork(static_cast<size_t>(vNe00));
        for (int64_t r = vR0; r < vR1; r++) {
          const int64_t i01    = r % vNe01;
          const int64_t i02    = (r / vNe01) % vNe02;
          const int64_t i03    = r / (vNe01 * vNe02);
          const float   vSlope = vMaxBias > 0.0f ?
                                     (uint32_t(i02) < vHeadLog2 ? powf(vM0, float(i02 + 1)) :
                                                                  powf(vM1, float(2 * (uint32_t(i02) - vHeadLog2) + 1))) :
                                     1.0f;
          const float * vSp = reinterpret_cast<const float *>(
              static_cast<const char *>(vA->data) + i01 * vA->nb[1] + i02 * vA->nb[2] + i03 * vA->nb[3]);
          float * vDp  = static_cast<float *>(pNode->data) + r * vNe00;
          float   vMax = -INFINITY;
          if (vB != nullptr) {
            const char * vMp = static_cast<const char *>(vB->data) + i01 * vB->nb[1] +
                               (i02 % vB->ne[2]) * vB->nb[2] + (i03 % vB->ne[3]) * vB->nb[3];
            if (vUseF16) {
              const ggml_fp16_t * vMh = reinterpret_cast<const ggml_fp16_t *>(vMp);
              for (int64_t k = 0; k < vNe00; k++) {
                aWork[size_t(k)] = vSp[k] * vScale + vSlope * ggml_fp16_to_fp32(vMh[k]);
                vMax             = std::max(vMax, aWork[size_t(k)]);
              }
            } else {
              const float * vMf = reinterpret_cast<const float *>(vMp);
              for (int64_t k = 0; k < vNe00; k++) {
                aWork[size_t(k)] = vSp[k] * vScale + vSlope * vMf[k];
                vMax             = std::max(vMax, aWork[size_t(k)]);
              }
            }
          } else {
            for (int64_t k = 0; k < vNe00; k++) {
              aWork[size_t(k)] = vSp[k] * vScale;
              vMax             = std::max(vMax, aWork[size_t(k)]);
            }
          }
          if (vSinks != nullptr) {
            vMax = std::max(vMax, vSinks[i02]);
          }
          double vSum = 0.0;
          for (int64_t k = 0; k < vNe00; k++) {
            const float vE = aWork[size_t(k)] == -INFINITY ? 0.0f : expf(aWork[size_t(k)] - vMax);
            vDp[k]         = vE;
            vSum += double(vE);
          }
          if (vSinks != nullptr) {
            vSum += double(expf(vSinks[i02] - vMax));
          }
          const float vInv = vSum > 0.0 ? float(1.0 / vSum) : 0.0f;
          for (int64_t k = 0; k < vNe00; k++) {
            vDp[k] *= vInv;
          }
        }
      });
      break;
    }
    case GGML_OP_ARGSORT: {
      const int64_t         vNe0   = vA->ne[0];
      const int64_t         vRows  = ggml_nrows(vA);
      const ggml_sort_order vOrder = ggml_sort_order(ggml_get_op_params_i32(pNode, 0));
      const uint32_t        vTasks = vRows * vNe0 >= 16384 ?
                                         uint32_t(std::min<int64_t>(vRows, std::max(1, pBackend.vThreads))) :
                                         1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vR0 = vRows * pTask / vTasks;
        const int64_t vR1 = vRows * (pTask + 1) / vTasks;
        for (int64_t r = vR0; r < vR1; r++) {
          const int64_t i1  = r % vA->ne[1];
          const int64_t i2  = (r / vA->ne[1]) % vA->ne[2];
          const int64_t i3  = r / (vA->ne[1] * vA->ne[2]);
          const float * vSp = reinterpret_cast<const float *>(
              static_cast<const char *>(vA->data) + i1 * vA->nb[1] + i2 * vA->nb[2] + i3 * vA->nb[3]);
          int32_t * vDp = reinterpret_cast<int32_t *>(
              static_cast<char *>(pNode->data) + i1 * pNode->nb[1] + i2 * pNode->nb[2] + i3 * pNode->nb[3]);
          for (int64_t k = 0; k < vNe0; k++) {
            vDp[k] = int32_t(k);
          }
          if (vOrder == GGML_SORT_ORDER_ASC) {
            std::sort(vDp, vDp + vNe0, [vSp](int32_t a, int32_t b) { return vSp[a] < vSp[b]; });
          } else {
            std::sort(vDp, vDp + vNe0, [vSp](int32_t a, int32_t b) { return vSp[a] > vSp[b]; });
          }
        }
      });
      break;
    }
    case GGML_OP_SSM_CONV: {
      const int64_t  vChannels = pNode->ne[0];
      const int64_t  vTokens   = pNode->ne[1];
      const int64_t  vTaps     = vB->ne[0];
      const uint32_t vTasks    = vChannels * vTokens >= 32768 ? uint32_t(std::max(1, pBackend.vThreads)) : 1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vC0 = vChannels * pTask / vTasks;
        const int64_t vC1 = vChannels * (pTask + 1) / vTasks;
        for (int64_t t = 0; t < vTokens; t++) {
          float * vTo = static_cast<float *>(pNode->data) + t * vChannels;
          for (int64_t c = vC0; c < vC1; c++) {
            const float * vW   = reinterpret_cast<const float *>(static_cast<const char *>(vB->data) + c * vB->nb[1]);
            float         vSum = 0.0f;
            for (int64_t k = 0; k < vTaps; k++) {
              const float vIn = *reinterpret_cast<const float *>(
                  static_cast<const char *>(vA->data) + (t + k) * vA->nb[0] + c * vA->nb[1]);
              vSum += vIn * vW[k];
            }
            vTo[c] = vSum;
          }
        }
      });
      break;
    }
    case GGML_OP_GATED_DELTA_NET: {
      const ggml_tensor * vQ       = pNode->src[0];
      const ggml_tensor * vK       = pNode->src[1];
      const ggml_tensor * vV       = pNode->src[2];
      const ggml_tensor * vG       = pNode->src[3];
      const ggml_tensor * vBeta    = pNode->src[4];
      ggml_tensor *       vState   = pNode->src[5];
      const int64_t       vDim     = vV->ne[0];
      const int64_t       vHeads   = vV->ne[1];
      const int64_t       vTokens  = vV->ne[2];
      const int64_t       vSeqs    = vV->ne[3];
      const bool          vInPlace = ggml_get_op_params_i32(pNode, 1) != 0;
      float *             vOutBase = static_cast<float *>(pNode->data);
      float *             vStBase  = vInPlace ? static_cast<float *>(vState->data) :
                                                (vOutBase + vDim * vHeads * vTokens * vSeqs);
      if (!vInPlace && vStBase != static_cast<const float *>(vState->data)) {
        std::memcpy(vStBase, vState->data, size_t(vDim * vDim * vHeads * vSeqs) * sizeof(float));
      }
      const float    vScale = 1.0f / sqrtf(float(vDim));
      const int64_t  vTotal = vHeads * vSeqs;
      const uint32_t vTasks = vTotal * vTokens >= 48 ?
                                  uint32_t(std::min<int64_t>(vTotal, std::max(1, pBackend.vThreads))) :
                                  1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t vH0 = vTotal * pTask / vTasks;
        const int64_t vH1 = vTotal * (pTask + 1) / vTasks;
        for (int64_t ir = vH0; ir < vH1; ir++) {
          const int64_t h     = ir % vHeads;
          const int64_t s     = ir / vHeads;
          const int64_t hq    = h % vQ->ne[1];
          const int64_t hk    = h % vK->ne[1];
          const int64_t sq    = s % vQ->ne[3];
          const int64_t sk    = s % vK->ne[3];
          float *       vHead = vStBase + (s * vHeads + h) * vDim * vDim;
          for (int64_t t = 0; t < vTokens; t++) {
            const float * vQh = reinterpret_cast<const float *>(
                static_cast<const char *>(vQ->data) + hq * vQ->nb[1] + t * vQ->nb[2] + sq * vQ->nb[3]);
            const float * vKh = reinterpret_cast<const float *>(
                static_cast<const char *>(vK->data) + hk * vK->nb[1] + t * vK->nb[2] + sk * vK->nb[3]);
            const float * vVh = reinterpret_cast<const float *>(
                static_cast<const char *>(vV->data) + h * vV->nb[1] + t * vV->nb[2] + s * vV->nb[3]);
            const float vGVal  = *reinterpret_cast<const float *>(
                static_cast<const char *>(vG->data) + h * vG->nb[1] + t * vG->nb[2] + s * vG->nb[3]);
            const float vBVal  = *reinterpret_cast<const float *>(
                static_cast<const char *>(vBeta->data) + h * vBeta->nb[1] + t * vBeta->nb[2] + s * vBeta->nb[3]);
            const float vDecay = expf(vGVal);
            float *     vOut   = vOutBase + ((s * vTokens + t) * vHeads + h) * vDim;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (cHasFullAvx2Fma && vDim == 128) {
              fGdnHead128Avx2(vHead, vQh, vKh, vVh, vDecay, vBVal, vScale, vOut);
              continue;
            }
#endif
            for (int64_t j = 0; j < vDim; j++) {
              float * vRow = vHead + j * vDim;
              float   vDot = 0.0f;
              for (int64_t k = 0; k < vDim; k++) {
                vRow[k] *= vDecay;
                vDot += vRow[k] * vKh[k];
              }
              const float vDelta = (vVh[j] - vDot) * vBVal;
              float       vAcc   = 0.0f;
              for (int64_t k = 0; k < vDim; k++) {
                vRow[k] += vKh[k] * vDelta;
                vAcc += vRow[k] * vQh[k];
              }
              vOut[j] = vAcc * vScale;
            }
          }
        }
      });
      break;
    }
    case GGML_OP_FLASH_ATTN_EXT: {
      const ggml_tensor * vQ      = pNode->src[0];
      const ggml_tensor * vK      = pNode->src[1];
      const ggml_tensor * vV      = pNode->src[2];
      const ggml_tensor * vMask   = pNode->src[3];
      const ggml_tensor * vSinks  = pNode->src[4];
      const int64_t       vDk     = vQ->ne[0];
      const int64_t       vDv     = vV->ne[0];
      const int64_t       vTokens = vQ->ne[1];
      const int64_t       vHeads  = vQ->ne[2];
      const int64_t       vKeys   = vK->ne[1];
      const int64_t       vKvH    = vK->ne[2];
      const int64_t       vGqa    = vHeads / vKvH;
      float               vScale  = 1.0f;
      std::memcpy(&vScale, pNode->op_params, sizeof(float));
      const float *  vSinkData = vSinks != nullptr ? static_cast<const float *>(vSinks->data) : nullptr;
      float *        vOutBase  = static_cast<float *>(pNode->data);
      const int64_t  vTotal    = vHeads * vTokens;
      const uint32_t vTasks    = vTotal * vKeys >= 16384 ?
                                     uint32_t(std::min<int64_t>(vTotal, std::max(1, pBackend.vThreads))) :
                                     1u;
      fParallel(vTasks, [&](uint32_t pTask) {
        const int64_t      vI0 = vTotal * pTask / vTasks;
        const int64_t      vI1 = vTotal * (pTask + 1) / vTasks;
        std::vector<float> aAcc(static_cast<size_t>(vDv));
        for (int64_t idx = vI0; idx < vI1; idx++) {
          const int64_t h    = idx % vHeads;
          const int64_t t    = idx / vHeads;
          const int64_t hkv  = h / vGqa;
          const float * vQh  = reinterpret_cast<const float *>(
              static_cast<const char *>(vQ->data) + t * vQ->nb[1] + h * vQ->nb[2]);
          const ggml_fp16_t * vMp = vMask != nullptr ?
                                        reinterpret_cast<const ggml_fp16_t *>(
                                            static_cast<const char *>(vMask->data) + t * vMask->nb[1]) :
                                        nullptr;
          std::fill(aAcc.begin(), aAcc.end(), 0.0f);
          float vMax = vSinkData != nullptr ? vSinkData[h] : -INFINITY;
          float vSum = vSinkData != nullptr ? 1.0f : 0.0f;
          for (int64_t kv = 0; kv < vKeys; kv++) {
            float vMVal = 0.0f;
            if (vMp != nullptr) {
              vMVal = ggml_fp16_to_fp32(vMp[kv]);
              if (vMVal == -INFINITY) {
                continue;
              }
            }
            const ggml_fp16_t * vKh = reinterpret_cast<const ggml_fp16_t *>(
                static_cast<const char *>(vK->data) + kv * vK->nb[1] + hkv * vK->nb[2]);
            float vDot = 0.0f;
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (cHasDequantAvx2F16c && vDk % 8 == 0) {
              vDot = fDotF32F16Avx2(vQh, vKh, vDk);
            } else
#endif
            {
              for (int64_t d = 0; d < vDk; d++) {
                vDot += vQh[d] * ggml_fp16_to_fp32(vKh[d]);
              }
            }
            const float vScore = vDot * vScale + vMVal;
            float       vW     = 0.0f;
            if (vScore > vMax) {
              const float vRescale = vMax == -INFINITY ? 0.0f : expf(vMax - vScore);
              for (int64_t d = 0; d < vDv; d++) {
                aAcc[size_t(d)] *= vRescale;
              }
              vSum = vSum * vRescale + 1.0f;
              vMax = vScore;
              vW   = 1.0f;
            } else {
              vW = expf(vScore - vMax);
              vSum += vW;
            }
            const ggml_fp16_t * vVh = reinterpret_cast<const ggml_fp16_t *>(
                static_cast<const char *>(vV->data) + kv * vV->nb[1] + hkv * vV->nb[2]);
#if defined(__x86_64__) && (defined(__GNUC__) || defined(__clang__))
            if (cHasDequantAvx2F16c && vDv % 8 == 0) {
              fMadF16Avx2(vW, vVh, aAcc.data(), vDv);
            } else
#endif
            {
              for (int64_t d = 0; d < vDv; d++) {
                aAcc[size_t(d)] += vW * ggml_fp16_to_fp32(vVh[d]);
              }
            }
          }
          const float vInv = vSum > 0.0f ? 1.0f / vSum : 0.0f;
          float *     vOut = vOutBase + (t * vHeads + h) * vDv;
          for (int64_t d = 0; d < vDv; d++) {
            vOut[d] = aAcc[size_t(d)] * vInv;
          }
        }
      });
      break;
    }
    default:
      pError = "unsupported full-subgraph operation";
      return false;
  }

  const double vElapsed     = fNowMs() - vStart;
  pBackend.vTotals.vFullOpsHost++;
  pBackend.vTotals.vFullOpsMs += vElapsed;
  return true;
}

enum ggml_status fGraphCompute(ggml_backend_t pBackend, struct ggml_cgraph * pGraph) {
  XdnaBackend *               vContext = static_cast<XdnaBackend *>(pBackend->context);
  std::vector<ggml_tensor *> & aGroup   = vContext->aGroup;
  std::vector<int> aIndexes;  // of the nodes of the group in the graph
  // llama.cpp-xdna: the layers of linear attention of this part of the graph that go to the NPU whole, found by their
  // recurrence of one token (fMatchLinear); each one runs when the graph gets to its last operation, and the others are
  // not computed
  vContext->aLinear.clear();
  vContext->dLinearOf.clear();
  // llama.cpp-xdna: the recurrences of a batch of this part of the graph that take the L2 norms of their q and k into
  // the writing of their inputs (fL2Fold); those norms are then not computed
  vContext->dL2Folds.clear();
  vContext->sL2Folded.clear();
  // llama.cpp-xdna: the short convolutions of a batch that go with their projection (fConvFold); their SSM_CONV and
  // SILU are then not computed
  vContext->sConvDone.clear();
  vContext->vPrepackedGdn = nullptr;
  if (!fShared().vSettings.vNoL2) {
    std::unordered_set<const ggml_tensor *> sInPart;
    for (int i = 0; i < pGraph->n_nodes; i++) {
      const ggml_tensor * vNode = pGraph->nodes[i];
      if (vNode->op != GGML_OP_GATED_DELTA_NET || !fGdnTaken(fShared(), vNode)) {
        continue;
      }
      if (sInPart.empty()) {
        sInPart.insert(pGraph->nodes, pGraph->nodes + pGraph->n_nodes);
      }
      XdnaBackend::XdnaL2Fold vFold;
      if (fL2Fold(pGraph, vNode, sInPart, vFold)) {
        vContext->dL2Folds[vNode] = vFold;
        for (int vSide = 0; vSide < 2; vSide++) {
          vContext->sL2Folded.insert(vNode->src[vSide]);
          vContext->sL2Folded.insert(vNode->src[vSide]->src[0]);
        }
      }
    }
  }
  if (fLinearOn(fShared())) {
    std::unordered_map<const ggml_tensor *, int> dIndexOf;
    for (int i = 0; i < pGraph->n_nodes; i++) {
      const ggml_tensor * vNode = pGraph->nodes[i];
      if (vNode->op != GGML_OP_GATED_DELTA_NET || !fLinearTaken(fShared(), vNode)) {
        continue;
      }
      if (dIndexOf.empty()) {
        for (int j = 0; j < pGraph->n_nodes; j++) {
          dIndexOf[pGraph->nodes[j]] = j;
        }
      }
      XdnaLinearBlock vBlock;
      if (!fMatchLinear(pGraph, vNode, dIndexOf, vBlock)) {
        GGML_LOG_DEBUG("%s: the layer of linear attention of %s goes on the host: its operations are not those of a "
                       "whole layer\n", cName, vNode->name);
        continue;
      }
      const int vIndex = int(vContext->aLinear.size());
      vContext->aLinear.push_back(vBlock);
      for (const ggml_tensor * vPart : {vBlock.vConv, vBlock.vSilu, vBlock.vNormQ, vBlock.vScaleQ, vBlock.vNormK,
                                        vBlock.vScaleK, vBlock.vBiased, vBlock.vSoftplus, vBlock.vGate, vBlock.vBeta,
                                        vBlock.vGdn, vBlock.vNormO, vBlock.vWeighted, vBlock.vSiluZ, vBlock.vOut}) {
        vContext->dLinearOf[vPart] = vIndex;
      }
    }
  }
  for (int i = 0; i < pGraph->n_nodes; i++) {
    struct ggml_tensor * vNode = pGraph->nodes[i];
    if ((vNode->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || fDoesNothing(vNode)) {
      continue;
    }
    std::string vError;
    // the rows of an RMS_NORM that goes with the multiplications are only for the ones right after it
    if (vContext->vNormed.vActive && !(vNode->op == GGML_OP_MUL_MAT && vNode->src[1] == vContext->vNormed.vRows)) {
      vContext->vNormed = XdnaBackend::XdnaNormed();
    }
    // llama.cpp-xdna: and so are the rows of a SwiGLU converted ahead
    if (vContext->vPacked.vActive && !(vNode->op == GGML_OP_MUL_MAT && vNode->src[1] == vContext->vPacked.vRows)) {
      vContext->vPacked = XdnaBackend::XdnaPacked();
    }
    // llama.cpp-xdna: an operation of a layer of linear attention: the whole layer on the NPU at its last operation,
    // or on the host one by one
    if (!vContext->dLinearOf.empty()) {
      const auto vIn = vContext->dLinearOf.find(vNode);
      if (vIn != vContext->dLinearOf.end()) {
        const XdnaLinearBlock & vBlock = vContext->aLinear[size_t(vIn->second)];
        if (vNode == vBlock.vOut && !fLinearAttention(*vContext, vBlock, vError)) {
          GGML_LOG_ERROR("%s: the layer of linear attention of %s failed: %s\n", cName, vBlock.vGdn->name,
                         vError.c_str());
          return GGML_STATUS_FAILED;
        }
        continue;
      }
    }
    if (!vContext->sConvDone.empty() && vContext->sConvDone.count(vNode) != 0) {
      continue;
    }
    if (fShared().vSettings.vFull == 0 && fLinearTaken(fShared(), vNode)) {
      if (!fLinearOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of a layer of linear attention failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    // llama.cpp-xdna: an operation of the short convolution of a batch that did not go with its projection: on the
    // host, the SSM_CONV with its SILU
    if (fConvTaken(fShared(), vNode)) {
      int vLast = i;
      if (!fConvOnHost(*vContext, pGraph, i, vLast, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of a short convolution failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      i = vLast;
      continue;
    }
    // llama.cpp-xdna: an L2 norm of q or k that a recurrence takes in, or else on the host
    if (!vContext->sL2Folded.empty() && vContext->sL2Folded.count(vNode) != 0) {
      continue;
    }
    if (fStateOpTaken(fShared(), vNode)) {
      if (!fStateOpOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the state operation %s (%s) failed: %s\n", cName, vNode->name, ggml_op_name(vNode->op),
                       vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (fGateTaken(fShared(), vNode)) {
      if (!fGateOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of a gate failed: %s\n", cName, vNode->name, ggml_op_name(vNode->op),
                       vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (fL2Taken(fShared(), vNode)) {
      if (!fL2OnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of an L2 norm failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (fMoeAmaxTaken(fShared(), vNode)) {
      if (!fMoeAmaxOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of an expert amax scale failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (fSharedGateOpTaken(fShared(), vNode)) {
      if (!fSharedGateOpOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of a shared-expert gate failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (fAttnGateTaken(fShared(), vNode)) {
      int vLast = i;
      if (!fAttnGateStep(*vContext, pGraph, i, vLast, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of an attention gate failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      i = vLast;
      continue;
    }
    // llama.cpp-xdna: the gated norm of a layer of linear attention in a batch: into the rows of the multiplication
    // right after it, or else on the host
    if (fGatedTaken(fShared(), vNode)) {
      if (vNode->op == GGML_OP_RMS_NORM) {
        int  vLast = i;
        bool vDone = false;
        if (!fGatedNorm(*vContext, pGraph, i, vLast, vDone, vError)) {
          GGML_LOG_ERROR("%s: the gated norm from %s failed: %s\n", cName, vNode->name, vError.c_str());
          return GGML_STATUS_FAILED;
        }
        if (vDone) {
          i = vLast;
          continue;
        }
      }
      if (!fGatedOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the operation %s (%s) of a gated norm failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op == GGML_OP_RMS_NORM && fRmsNormTaken(vNode)) {
      // llama.cpp-xdna: with its MUL and the multiplications that read its result right after them, and nothing
      // else: the scales of its rows now, and the rows normalised as they are converted for those multiplications
      int vMulIndex = i + 1;
      while (vMulIndex < pGraph->n_nodes && ((pGraph->nodes[vMulIndex]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                                            fDoesNothing(pGraph->nodes[vMulIndex]))) {
        vMulIndex++;
      }
      ggml_tensor *       vMul     = vMulIndex < pGraph->n_nodes ? pGraph->nodes[vMulIndex] : nullptr;
      const ggml_tensor * vWeights = vMul != nullptr ? fNormWeightsOf(vMul) : nullptr;
      bool vFuse = vWeights != nullptr && vWeights->data != nullptr && (vMul->src[0] == vNode || vMul->src[1] == vNode) &&
                   vNode->src[0]->data != nullptr && fUseCount(pGraph, vNode) == 1 &&
                   (vNode->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (vMul->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
      if (vFuse) {
        int vUses = 0;  // the multiplications right after, of its result
        for (int j = vMulIndex + 1; j < pGraph->n_nodes; j++) {
          ggml_tensor * vNext = pGraph->nodes[j];
          if ((vNext->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || fDoesNothing(vNext)) {
            continue;
          }
          if (vNext->op != GGML_OP_MUL_MAT || vNext->src[1] != vMul || !ggml_is_contiguous(vNext) ||
              !fMulMatTaken(fShared(), vNext)) {
            break;
          }
          vUses++;
        }
        vFuse = vUses > 0 && fUseCount(pGraph, vMul) == vUses;
      }
      if (vFuse) {
        float vEps;
        memcpy(&vEps, vNode->op_params, sizeof(float));
        fRowScales(*vContext, static_cast<const float *>(vNode->src[0]->data), uint64_t(ggml_nrows(vNode)),
                   uint64_t(vNode->ne[0]), vEps);
        vContext->vNormed.vActive  = true;
        vContext->vNormed.vRows    = vMul;
        vContext->vNormed.vX       = static_cast<const float *>(vNode->src[0]->data);
        vContext->vNormed.vWeights = static_cast<const float *>(vWeights->data);
        vContext->vTotals.vNormsFused++;
        i = vMulIndex;
        continue;
      }
      if (!fRmsNormOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the RMS_NORM %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op == GGML_OP_MUL && (fNormWeightsOf(vNode) != nullptr || fMoeTailTaken(fShared(), vNode))) {
      if (fMoeTailTaken(fShared(), vNode)) {
        if (!fMoeTailOnHost(*vContext, vNode, vError)) {
          GGML_LOG_ERROR("%s: the MUL %s of the experts failed: %s\n", cName, vNode->name, vError.c_str());
          return GGML_STATUS_FAILED;
        }
        continue;
      }
      // the MUL of an RMS_NORM that did not go with the multiplications: on the host
      if (!fMulOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the MUL %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op == GGML_OP_GLU) {
      // a SwiGLU that did not go with its multiplications: on the host, from its two results
      if (!fGluOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the SwiGLU %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op == GGML_OP_CONCAT) {
      // a CONCAT that did not go with its multiplication: on the host, the first operand value by value and then
      // the result of the multiplication, row by row
      const ggml_tensor * vRest = vNode->src[1];
      if (vNode->data == nullptr || vNode->src[0]->data == nullptr || vRest->data == nullptr) {
        GGML_LOG_ERROR("%s: the CONCAT %s has a tensor without memory\n", cName, vNode->name);
        return GGML_STATUS_FAILED;
      }
      fConcatHead(vNode);
      std::memcpy(static_cast<float *>(vNode->data) + vNode->src[0]->ne[1] * vNode->ne[0], vRest->data,
                  size_t(ggml_nelements(vRest)) * sizeof(float));
      vContext->vTotals.vConcatsHost++;
      continue;
    }
    if (vNode->op == GGML_OP_ADD && (fAddTaken(fShared(), vNode) || fMoeTailTaken(fShared(), vNode))) {
      if (fMoeTailTaken(fShared(), vNode)) {
        if (!fMoeTailOnHost(*vContext, vNode, vError)) {
          GGML_LOG_ERROR("%s: the ADD %s of the experts failed: %s\n", cName, vNode->name, vError.c_str());
          return GGML_STATUS_FAILED;
        }
        continue;
      }
      // a residual ADD that did not go with its multiplication: on the host
      if (!fAddOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the ADD %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op == GGML_OP_GET_ROWS && fRowsTaken(vNode)) {
      // llama.cpp-xdna: the only row of a result, that did not go with a command: copied on the host
      if (!fRowsOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the GET_ROWS %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op == GGML_OP_GATED_DELTA_NET && fGdnTaken(fShared(), vNode)) {
      // llama.cpp-xdna: the recurrence of a batch of tokens, in one command of the overlay of the prompts
      int vLast = i;
      if (!fGatedDeltaNet(*vContext, pGraph, i, vLast, vError)) {
        GGML_LOG_ERROR("%s: the recurrence %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      i = vLast;
      continue;
    }
    if (vNode->op == GGML_OP_FLASH_ATTN_EXT && fAttnTaken(fShared(), vNode)) {
      // llama.cpp-xdna: the attention of a batch, as two multiplications for every head of keys on the overlay of the
      // prompts
      if (!fFlashAttention(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the attention %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op == GGML_OP_MUL_MAT_ID) {
      // llama.cpp-xdna: the routed experts of a mixture of experts, with their SwiGLU and weighted sum when adjacent
      int vLast = i;
      if (!fRunMoeStep(*vContext, pGraph, i, vLast, vError)) {
        GGML_LOG_ERROR("%s: the experts %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      i = vLast;
      continue;
    }
    if (vNode->op != GGML_OP_MUL_MAT && fFullOpTaken(fShared(), vNode)) {
      if (!fFullOpOnHost(*vContext, vNode, vError)) {
        GGML_LOG_ERROR("%s: the full-subgraph operation %s (%s) failed: %s\n", cName, vNode->name,
                       ggml_op_name(vNode->op), vError.c_str());
        return GGML_STATUS_FAILED;
      }
      continue;
    }
    if (vNode->op != GGML_OP_MUL_MAT) {
      GGML_LOG_ERROR("%s: the operation %s (%s) is not one of this backend\n", cName, vNode->name,
                     ggml_op_name(vNode->op));
      return GGML_STATUS_FAILED;
    }
    // llama.cpp-xdna: generating a token with the chained overlay, the feed-forward block of a layer that starts
    // with this multiplication, in one command
    if (fShared().vHasChain && !fShared().vSettings.vNoChain && !fShared().vSettings.vNoFuse &&
        !(fMoeActiveForRows(fShared(), 1) && !fShared().vSettings.vMoeGemv)) {
      int  vLast = i;
      bool vDone = false;
      if (!fFeedForward(*vContext, pGraph, i, vLast, vDone, vError)) {
        GGML_LOG_ERROR("%s: the feed-forward block from %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      if (vDone) {
        i = vLast;
        continue;
      }
      if (!fSharedExpertChain(*vContext, pGraph, i, vLast, vDone, vError)) {
        GGML_LOG_ERROR("%s: the shared-expert block from %s failed: %s\n", cName, vNode->name, vError.c_str());
        return GGML_STATUS_FAILED;
      }
      if (vDone) {
        i = vLast;
        continue;
      }
    }
    // The multiplications that follow with the same rows, with nothing to compute between them, go together with
    // this one: the rows are converted once and the NPU gets one command for all of them. None of them can need
    // the result of another: their rows were computed before the first one. The results have to be one row after
    // the other, as they are written where they are.
    aGroup.assign(1, vNode);
    aIndexes.assign(1, i);
    for (int j = i + 1; j < pGraph->n_nodes && ggml_is_contiguous(vNode) && !fIsSharedGate(vNode->src[0]); j++) {
      struct ggml_tensor * vNext = pGraph->nodes[j];
      if ((vNext->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || fDoesNothing(vNext)) {
        continue;
      }
      if (vNext->op != GGML_OP_MUL_MAT || vNext->src[1] != vNode->src[1] || !ggml_is_contiguous(vNext) ||
          fIsSharedGate(vNext->src[0])) {
        break;
      }
      aGroup.push_back(vNext);
      aIndexes.push_back(j);
      i = j;
    }
    // A SwiGLU right after two multiplications, of their results, which nothing else uses (the counts of uses are
    // those of the whole graph, of which this one is a part): computed while their C is read.
    if (aGroup.size() == 2) {
      int vNextIndex = i + 1;
      while (vNextIndex < pGraph->n_nodes && ((pGraph->nodes[vNextIndex]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                                             fDoesNothing(pGraph->nodes[vNextIndex]))) {
        vNextIndex++;
      }
      struct ggml_tensor * vGlu = vNextIndex < pGraph->n_nodes ? pGraph->nodes[vNextIndex] : nullptr;
      const bool vPattern =
          vGlu != nullptr && fGluTaken(vGlu) &&
          ((vGlu->src[0] == aGroup[0] && vGlu->src[1] == aGroup[1]) ||
           (vGlu->src[0] == aGroup[1] && vGlu->src[1] == aGroup[0])) &&
          ggml_node_get_use_count(pGraph, aIndexes[0]) == 1 && ggml_node_get_use_count(pGraph, aIndexes[1]) == 1 &&
          (aGroup[0]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (aGroup[1]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
      if (vPattern) {
        // llama.cpp-xdna: the multiplication of the rows of the SwiGLU right after it, when nothing else reads them
        const ggml_tensor * vDown      = nullptr;
        const int           vDownIndex = fNextNode(pGraph, vNextIndex);
        if (vDownIndex < pGraph->n_nodes) {
          const ggml_tensor * vNext = pGraph->nodes[vDownIndex];
          if (vNext->op == GGML_OP_MUL_MAT && vNext->src[1] == vGlu && fMulMatTaken(fShared(), vNext) &&
              fUseCount(pGraph, vGlu) == 1 && (vGlu->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
              ggml_is_contiguous(vGlu)) {
            vDown = vNext;
          }
        }
        int  vMoeFfnLast = i;
        bool vMoeFfnDone = false;
        if (!fRunDenseMoeFfn(*vContext, pGraph, aGroup.data(), vGlu, vDownIndex, vDown, vMoeFfnLast, vMoeFfnDone,
                             vError)) {
          GGML_LOG_ERROR("%s: the dense feed-forward block %s and %s failed: %s\n", cName, aGroup[0]->name,
                         aGroup[1]->name, vError.c_str());
          return GGML_STATUS_FAILED;
        }
        if (vMoeFfnDone) {
          i = vMoeFfnLast;
          continue;
        }
        bool vFused = false;
        if (!fMulMatGlu(*vContext, aGroup.data(), vGlu, vDown, vFused, vError)) {
          GGML_LOG_ERROR("%s: the multiplications %s and %s with their SwiGLU failed: %s\n", cName, aGroup[0]->name,
                         aGroup[1]->name, vError.c_str());
          return GGML_STATUS_FAILED;
        }
        if (vFused) {
          i = vNextIndex;
          continue;
        }
      }
    }
    // llama.cpp-xdna: a CONCAT right after the multiplications, of the result of one of them (through a reshape) that
    // nothing else uses: that result goes straight into the CONCAT, after the rows of its first operand
    if (!fShared().vSettings.vNoFuse) {
      int vNextIndex = i + 1;
      while (vNextIndex < pGraph->n_nodes && ((pGraph->nodes[vNextIndex]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                                             fDoesNothing(pGraph->nodes[vNextIndex]))) {
        vNextIndex++;
      }
      ggml_tensor *       vConcat = vNextIndex < pGraph->n_nodes ? pGraph->nodes[vNextIndex] : nullptr;
      const ggml_tensor * vMember = fConcatMember(pGraph, vConcat, aGroup);
      if (vMember != nullptr) {
        vContext->vRedirect.vActive = true;
        vContext->vRedirect.vMember = vMember;
        vContext->vRedirect.vData   = static_cast<float *>(vConcat->data) + vConcat->src[0]->ne[1] * vConcat->ne[0];
        // llama.cpp-xdna: and the short convolution of the CONCAT with its SiLU, computed while C is read
        XdnaConvFold        vFold;
        const bool          vConv          = fConvFold(pGraph, vNextIndex, aGroup, vMember, vFold);
        const ggml_tensor * vMatchedGdn    = nullptr;
        if (vConv) {
          fConvPrepare(*vContext, vFold);
          vContext->vRedirect.vConvOn = true;
          vContext->vRedirect.vConv   = vFold;
          if (!fShared().vSettings.vNoGdnFold && !fShared().vSettings.vEmulate && vMember == aGroup[0] &&
              fShared().vOverlay.vTileN == 64 && fUseCount(pGraph, vFold.vSilu) == 3 &&
              (vFold.vSilu->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
            for (int k = vNextIndex + 1; k < pGraph->n_nodes; k++) {
              const ggml_tensor * vAhead = pGraph->nodes[k];
              if ((vAhead->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || fDoesNothing(vAhead)) {
                continue;
              }
              if (vAhead == vFold.vConv || vAhead == vFold.vSilu || fConvTaken(fShared(), vAhead) ||
                  vContext->sL2Folded.count(vAhead) != 0 || fGateTaken(fShared(), vAhead) ||
                  fStateOpTaken(fShared(), vAhead)) {
                continue;
              }
              if (vAhead->op == GGML_OP_GATED_DELTA_NET && fGdnTaken(fShared(), vAhead)) {
                vMatchedGdn = vAhead;
              }
              break;
            }
            if (vMatchedGdn != nullptr) {
              const auto          vL2It   = vContext->dL2Folds.find(vMatchedGdn);
              const bool          vHasL2  = vL2It != vContext->dL2Folds.end();
              const ggml_tensor * vViewQ  = vHasL2 ? vL2It->second.vQ : vMatchedGdn->src[0];
              const ggml_tensor * vViewK  = vHasL2 ? vL2It->second.vK : vMatchedGdn->src[1];
              const ggml_tensor * vViewV  = vMatchedGdn->src[2];
              const uint32_t      vKeys   = uint32_t(vViewQ->ne[1]);
              const uint32_t      vValues = uint32_t(vViewV->ne[1]);
              const uint32_t      vTokens = uint32_t(vViewV->ne[2]);
              const size_t        vRowB   = size_t(vFold.vChannels) * sizeof(float);
              if (vViewQ->op == GGML_OP_VIEW && vViewK->op == GGML_OP_VIEW && vViewV->op == GGML_OP_VIEW &&
                  vViewQ->view_src == vFold.vSilu && vViewK->view_src == vFold.vSilu &&
                  vViewV->view_src == vFold.vSilu && vViewQ->view_offs == 0 &&
                  vViewK->view_offs == size_t(vKeys) * 128 * sizeof(float) &&
                  vViewV->view_offs == size_t(2 * vKeys) * 128 * sizeof(float) &&
                  vViewQ->nb[1] == 128 * sizeof(float) && vViewK->nb[1] == 128 * sizeof(float) &&
                  vViewV->nb[1] == 128 * sizeof(float) && vViewQ->nb[2] == vRowB && vViewK->nb[2] == vRowB &&
                  vViewV->nb[2] == vRowB && vFold.vChannels == int64_t(2 * vKeys + vValues) * 128 &&
                  vFold.vTokens == int64_t(vTokens) && fUseCount(pGraph, vViewQ) == 1 &&
                  fUseCount(pGraph, vViewK) == 1 && fUseCount(pGraph, vViewV) == 1 &&
                  (vViewQ->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 && (vViewK->flags & GGML_TENSOR_FLAG_OUTPUT) == 0 &&
                  (vViewV->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
                XdnaShared &                vShared = fShared();
                std::lock_guard<std::mutex> vLock(vShared.vMutex);
                if (fOpenRunner(vShared, vError)) {
                  XdnaGdnPrepack & vPrep = vContext->vRedirect.vGdnPrepack;
                  XdnaGemmTimes    vPrepTimes;
                  if (vShared.vRunner.mPrepareGdnB(vTokens, vKeys, vValues, vPrep.vB, vPrep.vPartBytesB, vPrep.vParts,
                                                   vPrepTimes, vError)) {
                    vPrep.vTokens                  = vTokens;
                    vPrep.vKeyHeads                = vKeys;
                    vPrep.vValueHeads              = vValues;
                    vPrep.vScale                   = 1.0f / std::sqrt(128.0f);
                    vPrep.vHasL2                   = vHasL2;
                    vPrep.vL2Eps                   = vHasL2 ? vL2It->second.vEps : 0.0f;
                    vPrep.vNormScale               = vHasL2 ? vL2It->second.vFactor : 1.0f;
                    vContext->vRedirect.vGdnPrepackOn = true;
                    vContext->vTotals.vPlanMs      += vPrepTimes.vPlanMs;
                  } else {
                    vError.clear();
                    vMatchedGdn = nullptr;
                  }
                } else {
                  vError.clear();
                  vMatchedGdn = nullptr;
                }
              } else {
                vMatchedGdn = nullptr;
              }
            }
          }
        }
        const bool vPrepacked = vContext->vRedirect.vGdnPrepackOn;
        const bool vOk        = fMulMat(*vContext, aGroup.data(), uint32_t(aGroup.size()), vError);
        vContext->vRedirect   = XdnaBackend::XdnaRedirect();
        if (!vOk) {
          GGML_LOG_ERROR("%s: the multiplication %s with its CONCAT failed: %s\n", cName, vNode->name, vError.c_str());
          return GGML_STATUS_FAILED;
        }
        if (vConv) {
          // the rows of the CONCAT that only the convolution reads are not written: neither the old states nor the
          // tokens before the last 3
          vContext->sConvDone.insert(vFold.vConv);
          vContext->sConvDone.insert(vFold.vSilu);
          vContext->vTotals.vConvsFused++;
          if (vPrepacked) {
            vContext->vPrepackedGdn = vMatchedGdn;
          }
        } else {
          fConcatHead(vConcat);
        }
        vContext->vTotals.vConcatsFused++;
        i = vNextIndex;
        continue;
      }
    }
    // llama.cpp-xdna: a residual ADD right after a lone multiplication on the NPU of the prompts, of its result (or a
    // reshape of it) that nothing else uses: computed while its C is read, into the result of the ADD
    if (aGroup.size() == 1 && !fShared().vSettings.vNoFuse && !fShared().vSettings.vEmulate &&
        ggml_is_contiguous(aGroup[0]) && (aGroup[0]->flags & GGML_TENSOR_FLAG_OUTPUT) == 0) {
      int vNextIndex = i + 1;
      while (vNextIndex < pGraph->n_nodes && ((pGraph->nodes[vNextIndex]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
                                             fDoesNothing(pGraph->nodes[vNextIndex]))) {
        vNextIndex++;
      }
      ggml_tensor *       vAdd   = vNextIndex < pGraph->n_nodes ? pGraph->nodes[vNextIndex] : nullptr;
      const ggml_tensor * vRows  = aGroup[0]->src[1];
      const uint64_t      vCount = uint64_t(vRows->ne[1]) * uint64_t(vRows->ne[2]) * uint64_t(vRows->ne[3]);
      int                 vSide  = -1;  // the operand of the ADD that is the result of the multiplication
      if (vAdd != nullptr && vAdd->op == GGML_OP_ADD && fAddTaken(fShared(), vAdd) && vAdd->data != nullptr &&
          !fIsSharedGate(aGroup[0]->src[0]) &&
          (!fByRows(fShared(), vCount, uint32_t(aGroup[0]->src[0]->ne[0])) ||
           (fMoeActiveForRows(fShared(), vCount) && !fShared().vSettings.vNoMoeDense &&
            !fShared().vSettings.vMoeGemv)) &&
          fUseCount(pGraph, aGroup[0]) == 1) {
        for (int vOperand = 0; vOperand < 2; vOperand++) {
          const ggml_tensor * vSource = vAdd->src[vOperand];
          if (fThroughReshapes(vSource) != aGroup[0]) {
            continue;
          }
          bool vAlone = true;  // every reshape between them used only on the way to the ADD
          for (const ggml_tensor * vStep = vSource; vStep != aGroup[0]; vStep = vStep->src[0]) {
            vAlone = vAlone && fUseCount(pGraph, vStep) == 1 && (vStep->flags & GGML_TENSOR_FLAG_OUTPUT) == 0;
          }
          if (vAlone && vAdd->src[1 - vOperand]->data != nullptr) {
            vSide = vOperand;
          }
        }
      }
      if (vSide >= 0) {
        vContext->vResidual.vActive  = true;
        vContext->vResidual.vDst     = static_cast<float *>(vAdd->data);
        vContext->vResidual.vProduct = static_cast<const float *>(aGroup[0]->data);
        vContext->vResidual.vAlso    = static_cast<const float *>(vAdd->src[1 - vSide]->data);
        const bool vOk = fMulMat(*vContext, aGroup.data(), 1, vError);
        vContext->vResidual = XdnaBackend::XdnaResidual();
        if (!vOk) {
          GGML_LOG_ERROR("%s: the multiplication %s with its ADD failed: %s\n", cName, vNode->name, vError.c_str());
          return GGML_STATUS_FAILED;
        }
        vContext->vTotals.vAddsFused++;
        i = vNextIndex;
        continue;
      }
    }
    if (!fMulMat(*vContext, aGroup.data(), uint32_t(aGroup.size()), vError)) {
      GGML_LOG_ERROR("%s: the multiplication %s failed: %s\n", cName, vNode->name, vError.c_str());
      return GGML_STATUS_FAILED;
    }
  }
  vContext->vNormed = XdnaBackend::XdnaNormed();
  fDrainEarlyPacked(fShared(), *vContext);
  return GGML_STATUS_SUCCESS;
}

// llama.cpp looks for it by name in the registry of every backend
void fSetThreads(ggml_backend_t pBackend, int pThreads) {
  static_cast<XdnaBackend *>(pBackend->context)->vThreads = pThreads;
}

ggml_guid_t fGuid() {
  static ggml_guid vGuid = {0x58, 0x44, 0x4e, 0x41, 0x32, 0x9c, 0x4b, 0x07,
                            0xb1, 0x6e, 0x2d, 0xf3, 0x85, 0x1a, 0xc4, 0x70};
  return &vGuid;
}

//
// Buffers: host memory, as the ones of the processor
//

const char * fBufferTypeName(ggml_backend_buffer_type_t pType) {
  GGML_UNUSED(pType);
  return cName;
}

void * fBufferBase(ggml_backend_buffer_t pBuffer) {
  return pBuffer->context;
}

// the weights that lived in the buffer are gone: so is what the NPU kept of them
void fForgetBuffer(ggml_backend_buffer_t pBuffer) {
  XdnaShared & vShared = fShared();
  // llama.cpp-xdna: the placement of the weights goes with the last buffer of the backend (fPlaceWeights), including
  // after zero-size no_alloc fit-probe passes (common_fit_params) when no live buffer is allocated yet
  {
    XdnaPlacement &             vPlacement = fPlacement();
    std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
    if (pBuffer->size > 0) {
      if (--vPlacement.vBuffers < 0) {
        vPlacement.vBuffers = 0;
      }
    } else if (--vPlacement.vDummyBuffers < 0) {
      vPlacement.vDummyBuffers = 0;
    }
    if (vPlacement.vBuffers == 0 && vPlacement.vDummyBuffers == 0) {
      vPlacement.dTaken.clear();
      vPlacement.dMoeLayer.clear();
      vPlacement.sExpertLayers.clear();
      vPlacement.vPrompt  = 0;
      vPlacement.vChain   = 0;
      vPlacement.vMoe     = 0;
      vPlacement.vLogits  = 0;
      vPlacement.vRefused = 0;
      vPlacement.vFull    = false;
      vPlacement.vFullMoe = false;
      vPlacement.vPartial.store(false);
      vPlacement.vStreamPrompt.store(false);
      vPlacement.vCompPrompt.store(false);
    }
  }
  {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    const uintptr_t vBase = reinterpret_cast<uintptr_t>(pBuffer->context);
    auto            fInBuffer = [vBase, pBuffer](const void * pPtr) {
      const uintptr_t vData = reinterpret_cast<uintptr_t>(pPtr);
      return (vData & cWeightViewKeyTag) == 0 && vData >= vBase && vData - vBase < pBuffer->size;
    };
    for (uint32_t s = 0; s < 2; s++) {
      bool vInside = false;
      for (const void * vWeights : vShared.aStreamKeys[s]) {
        vInside = vInside || fInBuffer(vWeights);
      }
      if (vInside || fPlacement().vBuffers == 0) {
        fFreeMatrix(vShared, vShared.aStreamMatrices[s]);
        vShared.aStreamKeys[s].clear();
      }
    }
    for (auto vAt = vShared.sPromptCached.begin(); vAt != vShared.sPromptCached.end();) {
      if (fInBuffer(*vAt)) {
        vAt = vShared.sPromptCached.erase(vAt);
      } else {
        ++vAt;
      }
    }
    for (auto vAt = vShared.sGemvCached.begin(); vAt != vShared.sGemvCached.end();) {
      if (fInBuffer(*vAt)) {
        vAt = vShared.sGemvCached.erase(vAt);
      } else {
        ++vAt;
      }
    }
    for (auto vAt = vShared.dMatrices.begin(); vAt != vShared.dMatrices.end();) {
      bool vInside = false;
      for (const void * vWeights : vAt->first) {
        vInside = vInside || fInBuffer(vWeights);
      }
      if (vInside) {
        uint64_t vBytes = 0;
        for (const XdnaPiece & vPiece : vAt->second.aPieces) {
          vBytes += vPiece.vWeights.vBuffer.vSize;
        }
        vShared.vCachedPromptBytes = vShared.vCachedPromptBytes > vBytes ? vShared.vCachedPromptBytes - vBytes : 0;
        fFreeMatrix(vShared, vAt->second);
        vAt = vShared.dMatrices.erase(vAt);
      } else {
        ++vAt;
      }
    }
    for (auto vAt = vShared.dGemvMatrices.begin(); vAt != vShared.dGemvMatrices.end();) {
      bool vInside = false;
      for (const void * vWeights : vAt->first) {
        vInside = vInside || fInBuffer(vWeights);
      }
      if (vInside) {
        fFreeGemvMatrix(vShared, vAt->second);
        vAt = vShared.dGemvMatrices.erase(vAt);
      } else {
        ++vAt;
      }
    }
    for (auto vAt = vShared.dMoeMatrices.begin(); vAt != vShared.dMoeMatrices.end();) {
      bool vInside = false;
      for (const void * vWeights : vAt->first) {
        vInside = vInside || fInBuffer(vWeights);
      }
      if (vInside) {
        vShared.vRunner.mFreeMoeWeights(vAt->second);
        vAt = vShared.dMoeMatrices.erase(vAt);
      } else {
        ++vAt;
      }
    }
    // llama.cpp-xdna: the weights of the layers of linear attention live in buffers of the CPU backend, which the
    // backend does not see freed: their fixed objects go with any buffer of weights of the model in this backend
    if (ggml_backend_buffer_get_usage(pBuffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
      for (auto & vEntry : vShared.dLinear) {
        vShared.vRunner.mFreeChainWeights(vEntry.second);
      }
      vShared.dLinear.clear();
    }
    fCloseIfIdle(vShared);
  }
}

void fBufferFree(ggml_backend_buffer_t pBuffer) {
  fForgetBuffer(pBuffer);
  ggml_aligned_free(pBuffer->context, pBuffer->size);
}

// llama.cpp-xdna: a buffer over memory of others (buffer_from_host_ptr: the mapping of the file of a model): what the
// NPU kept of its weights goes, the memory stays
void fBufferFreeForeign(ggml_backend_buffer_t pBuffer) {
  {
    XdnaShared &                vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.dForeignMappings.erase(pBuffer->context);
  }
  fForgetBuffer(pBuffer);
}

void fBufferMemsetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor * pTensor, uint8_t pValue, size_t pOffset,
                         size_t pSize) {
  GGML_UNUSED(pBuffer);
  std::memset(static_cast<char *>(pTensor->data) + pOffset, pValue, pSize);
}

void fBufferSetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor * pTensor, const void * pData, size_t pOffset,
                      size_t pSize) {
  GGML_UNUSED(pBuffer);
  std::memcpy(static_cast<char *>(pTensor->data) + pOffset, pData, pSize);
}

void fBufferGetTensor(ggml_backend_buffer_t pBuffer, const struct ggml_tensor * pTensor, void * pData, size_t pOffset,
                      size_t pSize) {
  GGML_UNUSED(pBuffer);
  std::memcpy(pData, static_cast<const char *>(pTensor->data) + pOffset, pSize);
}

bool fBufferCopyTensor(ggml_backend_buffer_t pBuffer, const struct ggml_tensor * pFrom, struct ggml_tensor * pTo) {
  GGML_UNUSED(pBuffer);
  if (!ggml_backend_buffer_is_host(pFrom->buffer)) {
    return false;
  }
  std::memcpy(pTo->data, pFrom->data, ggml_nbytes(pFrom));
  return true;
}

void fBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  std::memset(pBuffer->context, pValue, pBuffer->size);
}

ggml_backend_buffer_t fBufferTypeAlloc(ggml_backend_buffer_type_t pType, size_t pSize) {
  void * vData = ggml_aligned_malloc(pSize);
  if (vData == nullptr) {
    GGML_LOG_ERROR("%s: failed to allocate a buffer of %zu bytes\n", cName, pSize);
    return nullptr;
  }
  struct ggml_backend_buffer_i vInterface = {};
  vInterface.free_buffer   = fBufferFree;
  vInterface.get_base      = fBufferBase;
  vInterface.memset_tensor = fBufferMemsetTensor;
  vInterface.set_tensor    = fBufferSetTensor;
  vInterface.get_tensor    = fBufferGetTensor;
  vInterface.cpy_tensor    = fBufferCopyTensor;
  vInterface.clear         = fBufferClear;
  {
    // llama.cpp-xdna: counted for the placement of the weights (fForgetBuffer)
    XdnaPlacement &             vPlacement = fPlacement();
    std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
    if (pSize > 0) {
      vPlacement.vBuffers++;
    } else {
      vPlacement.vDummyBuffers++;
    }
  }
  return ggml_backend_buffer_init(pType, vInterface, vData, pSize);
}

size_t fBufferTypeAlignment(ggml_backend_buffer_type_t pType) {
  GGML_UNUSED(pType);
  return TENSOR_ALIGNMENT;
}

// llama.cpp-xdna: the weights of a model straight in the mapping of its file, as the CPU backend keeps them, instead of
// copied into a buffer of this backend: llama.cpp asks for it when the model is mapped (its default) and the device says
// it can (buffer_from_host_ptr). The original weights are then pages of the file, which the system can take back,
// instead of anonymous memory beside the copies the NPU reads: 2.6 GB less with Qwen3.5 4B Q4_K_M. The backend only reads
// them, to convert them. The tensors keep their places in the file, as llama.cpp gives them.
ggml_backend_buffer_t fDeviceBufferFromHostPtr(ggml_backend_dev_t pDevice, void * pData, size_t pSize,
                                               size_t pMaxTensorSize) {
  GGML_UNUSED(pDevice);
  GGML_UNUSED(pMaxTensorSize);
  struct ggml_backend_buffer_i vInterface = {};
  vInterface.free_buffer   = fBufferFreeForeign;
  vInterface.get_base      = fBufferBase;
  vInterface.memset_tensor = fBufferMemsetTensor;
  vInterface.set_tensor    = fBufferSetTensor;
  vInterface.get_tensor    = fBufferGetTensor;
  vInterface.cpy_tensor    = fBufferCopyTensor;
  vInterface.clear         = fBufferClear;
  if (pSize > 0) {
    // llama.cpp-xdna: counted for the placement of the weights (fForgetBuffer)
    {
      XdnaPlacement &            vPlacement = fPlacement();
      std::lock_guard<std::mutex> vLock(vPlacement.vMutex);
      vPlacement.vBuffers++;
    }
    const bool                  vReadOnlyShared = fIsReadOnlySharedMapping(pData, pSize);
    XdnaShared &                vShared         = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.dForeignMappings[pData] = {pSize, vReadOnlyShared};
  }
  return ggml_backend_buffer_init(fBufferType(), vInterface, pData, pSize);
}

bool fBufferTypeIsHost(ggml_backend_buffer_type_t pType) {
  GGML_UNUSED(pType);
  return true;
}

//
// Buffers for recurrent states
//

// llama.cpp-xdna: the recurrent states of the layers of a model that run on the processor, in buffers of the driver
// that the NPU can read and write (llama-memory-recurrent.cpp asks for this type with ggml_backend_dev_get_state_buft).
// They are host memory, like the other buffers of the backend, so the CPU works on them as before. A page apart: the
// cache lines of a state never share a page with those of another tensor.
const size_t cStateAlignment = 4096;

const char * fStateBufferTypeName(ggml_backend_buffer_type_t pType) {
  GGML_UNUSED(pType);
  return "XDNA_State";
}

void fStateBufferFree(ggml_backend_buffer_t pBuffer) {
  XdnaShared &               vShared = fShared();
  std::lock_guard<std::mutex> vLock(vShared.vMutex);
  vShared.sCleanStates.clear();
  auto                        vAt = vShared.dStateBuffers.find(pBuffer->context);
  if (vAt != vShared.dStateBuffers.end()) {
    vShared.vRunner.mFreeShared(vAt->second);
    vShared.dStateBuffers.erase(vAt);
  }
  fCloseIfIdle(vShared);
}

void fStateBufferMemsetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor * pTensor, uint8_t pValue,
                              size_t pOffset, size_t pSize) {
  {
    XdnaShared &                vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.sCleanStates.clear();
  }
  fBufferMemsetTensor(pBuffer, pTensor, pValue, pOffset, pSize);
}

void fStateBufferSetTensor(ggml_backend_buffer_t pBuffer, struct ggml_tensor * pTensor, const void * pData,
                           size_t pOffset, size_t pSize) {
  {
    XdnaShared &                vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.sCleanStates.clear();
  }
  fBufferSetTensor(pBuffer, pTensor, pData, pOffset, pSize);
}

bool fStateBufferCopyTensor(ggml_backend_buffer_t pBuffer, const struct ggml_tensor * pFrom, struct ggml_tensor * pTo) {
  {
    XdnaShared &                vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.sCleanStates.clear();
  }
  return fBufferCopyTensor(pBuffer, pFrom, pTo);
}

void fStateBufferClear(ggml_backend_buffer_t pBuffer, uint8_t pValue) {
  {
    XdnaShared &                vShared = fShared();
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    vShared.sCleanStates.clear();
  }
  fBufferClear(pBuffer, pValue);
}

ggml_backend_buffer_t fStateBufferTypeAlloc(ggml_backend_buffer_type_t pType, size_t pSize) {
  XdnaShared & vShared = fShared();
  XdnaBuffer   vBuffer;
  {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    std::string                 vError = "the NPU is emulated";
    if (vShared.vSettings.vEmulate || !fOpenRunner(vShared, vError) ||
        !vShared.vRunner.mCreateShared(pSize, vBuffer, vError)) {
      GGML_LOG_ERROR("%s: failed to allocate a buffer of %zu bytes for recurrent states: %s\n", cName, pSize,
                     vError.c_str());
      fCloseIfIdle(vShared);
      return nullptr;
    }
    vShared.dStateBuffers[vBuffer.vMap] = vBuffer;
  }
  struct ggml_backend_buffer_i vInterface = {};
  vInterface.free_buffer   = fStateBufferFree;
  vInterface.get_base      = fBufferBase;
  vInterface.memset_tensor = fStateBufferMemsetTensor;
  vInterface.set_tensor    = fStateBufferSetTensor;
  vInterface.get_tensor    = fBufferGetTensor;
  vInterface.cpy_tensor    = fStateBufferCopyTensor;
  vInterface.clear         = fStateBufferClear;
  return ggml_backend_buffer_init(pType, vInterface, vBuffer.vMap, pSize);
}

size_t fStateBufferTypeAlignment(ggml_backend_buffer_type_t pType) {
  GGML_UNUSED(pType);
  return cStateAlignment;
}

// Room after every state: the recurrence of the chained overlay (version 8) reads 7 rows of 512 bytes past the last
// head of a cell (the padding of its last object), and they have to be in the buffer.
size_t fStateBufferTypeAllocSize(ggml_backend_buffer_type_t pType, const struct ggml_tensor * pTensor) {
  GGML_UNUSED(pType);
  return ggml_nbytes(pTensor) + cStateAlignment;
}

//
// Device
//

const char * fDeviceName(ggml_backend_dev_t pDevice) {
  GGML_UNUSED(pDevice);
  return cName;
}

const char * fDeviceDescription(ggml_backend_dev_t pDevice) {
  GGML_UNUSED(pDevice);
  return fShared().vDescription.c_str();
}

// its memory is the one of the host: nothing to report
void fDeviceMemory(ggml_backend_dev_t pDevice, size_t * pFree, size_t * pTotal) {
  GGML_UNUSED(pDevice);
  *pFree  = 0;
  *pTotal = 0;
}

enum ggml_backend_dev_type fDeviceType(ggml_backend_dev_t pDevice) {
  GGML_UNUSED(pDevice);
  return GGML_BACKEND_DEVICE_TYPE_ACCEL;
}

void fDeviceProps(ggml_backend_dev_t pDevice, struct ggml_backend_dev_props * pProps) {
  *pProps             = {};
  pProps->name        = fDeviceName(pDevice);
  pProps->description = fDeviceDescription(pDevice);
  pProps->type        = fDeviceType(pDevice);
  fDeviceMemory(pDevice, &pProps->memory_free, &pProps->memory_total);
  // nothing asynchronous; a model can be mapped, and its weights stay in the mapping (fDeviceBufferFromHostPtr)
  pProps->caps.mmap_support         = true;
  pProps->caps.buffer_from_host_ptr = !fShared().vSettings.vCopyWeights;
}

ggml_backend_t fDeviceInit(ggml_backend_dev_t pDevice, const char * pParams) {
  GGML_UNUSED(pParams);
  XdnaShared & vShared = fShared();
  if (!vShared.vUsable) {
    return nullptr;
  }
  {
    std::lock_guard<std::mutex> vLock(vShared.vMutex);
    std::string vError;
    if (!vShared.vSettings.vEmulate && !fOpenRunner(vShared, vError)) {
      GGML_LOG_ERROR("%s: cannot use the NPU: %s\n", cName, vError.c_str());
      return nullptr;
    }
    vShared.vBackends++;
  }

  struct ggml_backend_i vInterface = {};
  vInterface.get_name      = fBackendName;
  vInterface.free          = fBackendFree;
  vInterface.graph_compute = fGraphCompute;

  ggml_backend_t vBackend = new ggml_backend;
  vBackend->guid    = fGuid();
  vBackend->iface   = vInterface;
  vBackend->device  = pDevice;
  vBackend->context = new XdnaBackend;
  return vBackend;
}

bool fDeviceSupportsOp(ggml_backend_dev_t pDevice, const struct ggml_tensor * pNode) {
  GGML_UNUSED(pDevice);
  switch (pNode->op) {
    case GGML_OP_NONE:
    case GGML_OP_RESHAPE:
    case GGML_OP_VIEW:
    case GGML_OP_PERMUTE:
    case GGML_OP_TRANSPOSE:
      return true;
    case GGML_OP_MUL_MAT:
      return fMulMatTaken(fShared(), pNode);
    case GGML_OP_MUL_MAT_ID:
      return fMoeTaken(fShared(), pNode);
    case GGML_OP_GLU:
      return !fShared().vSettings.vNoGlu && fGluTaken(pNode);
    case GGML_OP_ADD:
      return (!fShared().vSettings.vNoFuse && fAddTaken(fShared(), pNode)) || fLinearTaken(fShared(), pNode) ||
             fGateTaken(fShared(), pNode) || fMoeTailTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_CONCAT:
      return !fShared().vSettings.vNoFuse && fConcatTaken(fShared(), pNode);
    case GGML_OP_GET_ROWS:
      return (!fShared().vSettings.vNoFuse && fRowsTaken(pNode)) || fStateOpTaken(fShared(), pNode) ||
             fFullOpTaken(fShared(), pNode);
    case GGML_OP_RMS_NORM:
      return (!fShared().vSettings.vNoFuse && fRmsNormTaken(pNode)) || fLinearTaken(fShared(), pNode) ||
             fGatedTaken(fShared(), pNode) || fL2Taken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_MUL:
      return (!fShared().vSettings.vNoFuse && fNormWeightsOf(pNode) != nullptr) || fLinearTaken(fShared(), pNode) ||
             fGatedTaken(fShared(), pNode) || fGateTaken(fShared(), pNode) || fMoeTailTaken(fShared(), pNode) ||
             fSharedGateOpTaken(fShared(), pNode) || fAttnGateTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_GATED_DELTA_NET:
      return fGdnTaken(fShared(), pNode) || fLinearTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_FLASH_ATTN_EXT:
      return fAttnTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    // llama.cpp-xdna: the other operations of a layer of linear attention while a token is generated
    case GGML_OP_CPY:
      return fLinearTaken(fShared(), pNode) || fConvTaken(fShared(), pNode) || fStateOpTaken(fShared(), pNode) ||
             fFullOpTaken(fShared(), pNode);
    case GGML_OP_SSM_CONV:
      return fLinearTaken(fShared(), pNode) || fConvTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_SCALE:
      return fLinearTaken(fShared(), pNode) || fL2Taken(fShared(), pNode) || fMoeAmaxTaken(fShared(), pNode) ||
             fStateOpTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_UNARY:
      return fLinearTaken(fShared(), pNode) || fGatedTaken(fShared(), pNode) || fGateTaken(fShared(), pNode) ||
             fConvTaken(fShared(), pNode) || fSharedGateOpTaken(fShared(), pNode) || fMoeAmaxTaken(fShared(), pNode) ||
             fAttnGateTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_LOG:
    case GGML_OP_POOL_1D:
    case GGML_OP_CLAMP:
    case GGML_OP_DIV:
      return fMoeAmaxTaken(fShared(), pNode) || fFullOpTaken(fShared(), pNode);
    case GGML_OP_DUP:
    case GGML_OP_CONT:
    case GGML_OP_ROPE:
    case GGML_OP_SET_ROWS:
    case GGML_OP_SUM_ROWS:
    case GGML_OP_SOFT_MAX:
    case GGML_OP_ARGSORT:
      return fFullOpTaken(fShared(), pNode);
    default:
      return false;
  }
}

// its buffers are host memory, and so are the ones it can work with
bool fDeviceSupportsBuft(ggml_backend_dev_t pDevice, ggml_backend_buffer_type_t pType) {
  GGML_UNUSED(pDevice);
  return ggml_backend_buft_is_host(pType);
}

ggml_backend_buffer_type_t fDeviceBufferType(ggml_backend_dev_t pDevice) {
  GGML_UNUSED(pDevice);
  return fBufferType();
}

// llama.cpp-xdna: the buffer type for the recurrent states of the layers that run on the processor
// (ggml_backend_dev_get_state_buft): buffers of the driver, with the NPU there and not emulated
ggml_backend_buffer_type_t fDeviceStateBuft(ggml_backend_dev_t pDevice) {
  GGML_UNUSED(pDevice);
  const XdnaShared & vShared = fShared();
  if (!vShared.vUsable || vShared.vSettings.vEmulate || vShared.vSettings.vNoStateBuffers) {
    return nullptr;
  }
  return fStateBufferType();
}

ggml_backend_dev_t fDevice() {
  static struct ggml_backend_device vDevice = [] {
    struct ggml_backend_device vNew = {};
    vNew.iface.get_name        = fDeviceName;
    vNew.iface.get_description = fDeviceDescription;
    vNew.iface.get_memory      = fDeviceMemory;
    vNew.iface.get_type        = fDeviceType;
    vNew.iface.get_props       = fDeviceProps;
    vNew.iface.init_backend    = fDeviceInit;
    vNew.iface.get_buffer_type = fDeviceBufferType;
    vNew.iface.supports_op     = fDeviceSupportsOp;
    vNew.iface.supports_buft   = fDeviceSupportsBuft;
    vNew.iface.buffer_from_host_ptr = fDeviceBufferFromHostPtr;  // llama.cpp-xdna
    vNew.reg                   = ggml_backend_xdna_reg();
    return vNew;
  }();
  return &vDevice;
}

ggml_backend_buffer_type_t fBufferType() {
  static struct ggml_backend_buffer_type vType = [] {
    struct ggml_backend_buffer_type vNew = {};
    vNew.iface.get_name      = fBufferTypeName;
    vNew.iface.alloc_buffer  = fBufferTypeAlloc;
    vNew.iface.get_alignment = fBufferTypeAlignment;
    vNew.iface.is_host       = fBufferTypeIsHost;
    vNew.device              = fDevice();
    return vNew;
  }();
  return &vType;
}

ggml_backend_buffer_type_t fStateBufferType() {
  static struct ggml_backend_buffer_type vType = [] {
    struct ggml_backend_buffer_type vNew = {};
    vNew.iface.get_name       = fStateBufferTypeName;
    vNew.iface.alloc_buffer   = fStateBufferTypeAlloc;
    vNew.iface.get_alignment  = fStateBufferTypeAlignment;
    vNew.iface.get_alloc_size = fStateBufferTypeAllocSize;
    vNew.iface.is_host        = fBufferTypeIsHost;
    vNew.device              = fDevice();
    return vNew;
  }();
  return &vType;
}

//
// Registry
//

const char * fRegName(ggml_backend_reg_t pReg) {
  GGML_UNUSED(pReg);
  return cName;
}

// one device when there is an NPU, or its emulation, that the backend can use
size_t fRegDeviceCount(ggml_backend_reg_t pReg) {
  GGML_UNUSED(pReg);
  return fShared().vUsable ? 1 : 0;
}

ggml_backend_dev_t fRegDevice(ggml_backend_reg_t pReg, size_t pIndex) {
  GGML_UNUSED(pReg);
  GGML_ASSERT(pIndex == 0);
  return fDevice();
}

void * fRegProcAddress(ggml_backend_reg_t pReg, const char * pName) {
  GGML_UNUSED(pReg);
  if (std::strcmp(pName, "ggml_backend_set_n_threads") == 0) {
    ggml_backend_set_n_threads_t vFunction = fSetThreads;
    return reinterpret_cast<void *>(vFunction);
  }
  if (std::strcmp(pName, "ggml_backend_dev_get_state_buft") == 0) {
    ggml_backend_dev_get_state_buft_t vFunction = fDeviceStateBuft;
    return reinterpret_cast<void *>(vFunction);
  }
  return nullptr;
}

}  // namespace

ggml_backend_reg_t ggml_backend_xdna_reg(void) {
  static struct ggml_backend_reg vReg = [] {
    struct ggml_backend_reg vNew = {};
    vNew.api_version            = GGML_BACKEND_API_VERSION;
    vNew.iface.get_name         = fRegName;
    vNew.iface.get_device_count = fRegDeviceCount;
    vNew.iface.get_device       = fRegDevice;
    vNew.iface.get_proc_address = fRegProcAddress;
    return vNew;
  }();
  return &vReg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_xdna_reg)
