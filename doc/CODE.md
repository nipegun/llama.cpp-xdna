# CODE.md

This CODE is also available in:\
[en-GB](CODE.en-GB.md), en-US, [es-AR](CODE.es-AR.md), [es-ES](CODE.es-ES.md)

Technical reference for llama.cpp-xdna, written to be read surgically: jump to the section you need. It covers what this fork adds to llama.cpp, above all the XDNA backend for the NPU of AMD Ryzen AI processors; the rest of the code is that of [llama.cpp](https://github.com/ggml-org/llama.cpp). The measurements behind every decision are in [XDNA2.md](XDNA2.md).

## Index

1. [Architecture and design decisions](#1-architecture-and-design-decisions)
2. [Module map](#2-module-map)
3. [Index of key symbols](#3-index-of-key-symbols)
4. [Main flows](#4-main-flows)
5. [Map of entry points](#5-map-of-entry-points)
6. [Impact analysis](#6-impact-analysis)
7. [Extension points](#7-extension-points)

## 1. Architecture and design decisions

### The fork

A fork of llama.cpp with three backends: XDNA (written for this project; today for the NPUs XDNA2), CPU and RPC; the other backends of llama.cpp were removed. It has four model architectures: `qwen35` (Qwen3.5 0.8B, 2B and 4B, a hybrid of linear attention and attention, with a layer of multi-token prediction), `qwen35moe` (the same with a block of experts, Qwen3.6 35B-A3B), `granite` (dense Granite 4.2) and `flint` (Flint 40B-A2.5B, a transformer with an interleaved sliding window, attention sinks and a block of experts). Every change to a file that comes from llama.cpp is marked with a comment `llama.cpp-xdna`, so that it can be kept when code is ported from upstream.

### A backend of its own, without XRT

The backend talks to the `amdxdna` driver of Linux directly, with its ioctls (`ggml/src/ggml-xdna/uapi/amdxdna_accel.h`, copied from Linux 7.3.4): XRT is not needed, and Debian 13 has no XRT that knows the NPU. `XdnaDevice` is that layer: it opens `/dev/accel/accel0`, creates buffers that the host and the NPU share, creates and configures contexts, submits commands and waits for them on the synchronization object of DRM of the context (`WAIT_CMD` does not exist on this NPU).

### The NPU and its two overlays

The NPU (AIE2P) is an array of 8 columns: a shim tile (the interface with memory), a memory tile (512 KB) and 4 cores (64 KB of data and 16 KB of program each) per column, 32 cores in all. What the array does is an *overlay*: a PDI with the configuration of every tile and the programs of the cores, built from a design in IRON (Python, `ggml/src/ggml-xdna/overlays/designs/`) and the kernels of the cores (C++ for the AIE, `ggml/src/ggml-xdna/kernels/`) with mlir-aie and its compiler Peano.

Changing the overlay a command runs on costs some 2.5 ms, so there are only two, and both go into one context as its two compute units:

- **The overlay of the prompts** (`xdna2-gemm-bfp16.*`): matrix multiplication C = A·B with the bands of A kept in the memory tiles (`bfpmm_resident.py`), the kernel in steps of three sets of sums (`kernels/xdna2-mm-bfp16-s3.cc`), C written in bfloat16 by strips of 64 columns, and the recurrence of a gated delta net of a batch in the same buffers (`kernels/xdna2-gdn-prompt.cc`).
- **The chained overlay for generating tokens** (`xdna2-gemv-chain.*`, version 9): 1 to 8 rows by a matrix in phases (`gemv_chain.py`, `kernels/xdna2-gemv-chain.cc`). A command runs up to 4 phases, and every phase takes its X from the Y of the phase before inside the array, with its prologue (the norm) and its epilogue (residual sum, SwiGLU): a whole layer goes in one command.

The host writes the instruction stream of every command (the transfers of the shim tiles, the parameters of the cores, the waits) in C++, from a manifest that the build of the overlay writes (`xdna-overlay.*`, `xdna-chain.*`). The streams are the same, operation by operation, as those of mlir-aie, which was checked; nothing runs Python while a model runs.

### Number formats

- **Block floating point** (bfp16ebs8): 8 values of 8 bits that share an exponent, 9 bytes. The operands of the overlay of the prompts are bfp16, the sums 32-bit floats, and C is rounded to bfloat16. In precision this costs about what Q8_0 costs on the processor.
- **The weights of the file**: for generating tokens the cores decode Q4_K, Q5_K and Q6_K as the file stores them; the other types go in bfp16.
- The weights are converted once, the first time they are used, into buffers of the driver; the file stays mapped (`mmap`), and the buffers of weights of the backend are made over that mapping (`fDeviceBufferFromHostPtr`), so that the original weights are pages of the file, not anonymous memory.

### What the backend takes from a graph

The device is an accelerator (`GGML_BACKEND_DEVICE_TYPE_ACCEL`): llama.cpp puts it before the CPU in every context. Its buffer type is host memory (`is_host`), so that llama.cpp gives it the weights of the multiplications it takes and the CPU can still read them; `fDeviceSupportsOp` says which nodes it takes, and `fGraphCompute` runs a part of a graph, recognizing patterns of nodes to do in one go:

- **A batch of a prompt** (40 tokens or more): the multiplications that read the same rows in one command (`fMulMat`); an RMS norm and its weights applied while the rows are converted; the SwiGLU computed while the gate and the up are read, and written straight as the rows of `ffn_down` (`fMulMatGlu`); the residual sum read with C; the `CONCAT` of the short convolution written by the multiplication, and the short convolution with its SiLU computed while the C of its projection is read (`fConvFold`); the gated norm of the linear attention converted into the rows of `ssm_out` (`fGatedNorm`); the recurrence of the linear attention on the NPU (`fGatedDeltaNet`); the attention as two multiplications for every head of keys, the softmax on the host between them (`fFlashAttention`).
- **The memory of the copies**: when llama.cpp places the weights, a matrix whose copies for the NPU would go past `GGML_XDNA_WEIGHTS_GB` stays with the processor, and the ones after it too (`fPlaceWeights`); a model that does not fit whole then generates on the processor but the logits (`XdnaPlacement::vPartial`).
- **A mixture of experts** (`qwen35moe`): the experts (`MUL_MAT_ID`), their router and their SwiGLU stay with the processor (`fIsRouter`, `fFromExperts`), and once the backend has seen a router, so do the multiplications of a few rows but the logits (`fSawExperts`): the overlay for generating tokens cannot chain a layer around the experts.
- **Generating a token** (1 to 8 rows): every feed-forward block, with the norm and the projections of the next layer, in one command of the chained overlay (`fFeedForward`); the last layer with the output norm and the logits.

### The memory of the host and of the NPU

The NPU is not cache coherent with the processor. What the host wrote leaves the cache before a command reads it; what the NPU wrote is invalidated in the cache before the host reads it; a new buffer leaves the cache before the NPU writes it (the kernel zeroes its pages through the cache, and those zeros would come back over the results); and in a buffer that both write, the host flushes up to the end of the page of what it wrote and one page more (the prefetcher leaves the next line modified). `fXdnaFlushCache` flushes from the process (`clflushopt`), `XdnaDevice::mSyncBuffer` through the driver.

A command that does not end leaves the NPU unusable until the computer is powered off; the runner never sends another command after one (`vBroken`), and the width of a context has to be that of the partition the overlay was built for.

### Changes outside the backend

- `src/llama-context.cpp`, `llama_context::graph_get_cb`: the scheduler moves an operation to an accelerator only from its weights or from the nodes next to it; the callback gives the XDNA backend, on a layer of the processor, the recurrence of a batch, the L2 norms of q and k, the attention and the norms of the residual stream, when it takes them.
- `src/models/qwen35.cpp` and `src/models/delta-net-base.cpp`: the projections of a layer one after the other in the graph, the input of the short convolution by tokens, the short convolution before the recurrent state, the states of the convolution and of the recurrence read and written in their cell of the cache.
- `src/llama-graph.{h,cpp}`, `ggml/include/ggml.h`, `ggml/src/ggml.c`: `ggml_gated_delta_net_set_state_in_place` and `llm_graph_rs_in_place`.
- `src/llama-memory-recurrent.cpp`: the recurrent states in buffers of the driver (`ggml_backend_dev_get_state_buft`).
- The mixtures of experts, which the fork had taken out, ported from upstream for `qwen35moe` only (`src/models/qwen35moe.cpp`): the keys, tensors and parameters of the experts (`src/llama-arch.*`, `src/llama-hparams.*`, `src/llama-model.*`), the test of `MUL_MAT_ID` that places their buffers (`src/llama-model-loader.cpp`), `build_lora_mm_id` and a reduced `build_moe_ffn` (`src/llama-graph.*`).
- `flint` (`src/models/flint.cpp`), ported from upstream with what it needs that the fork had taken out: the KV cache with an interleaved sliding window (`src/llama-kv-cache-iswa.{h,cpp}`, `llm_graph_input_attn_kv_iswa`, `build_attn_inp_kv_iswa` and its `build_attn` in `src/llama-graph.*`, the branch of `llama_model::create_memory`), the attention sinks, the RoPE only in the layers that have it, the global base of the RoPE in its layers with a window (`llama_model::get_rope_freq_base`), the bias of the choice of the experts and the rescaling of the activation of its experts before their down projection (`build_amax_src1_f16`).
- `ggml/src/ggml-cpu/ops.cpp`: faster `SSM_CONV`, `GATED_DELTA_NET` and `RMS_NORM` on the processor, with the same results bit for bit.
- Security: the RPC server validates what a client sends and can ask for a token (HMAC-SHA256); `llama-server` restricts CORS to localhost and checks the `Host` header; the GGUF reader and the loader reject malformed files. See [SECURITY.md](../SECURITY.md).

## 2. Module map

| Module | Path | Responsibility | Depends on | Used by |
|---|---|---|---|---|
| Device layer | `ggml/src/ggml-xdna/xdna-device.{h,cpp}`, `uapi/` | ioctls of `amdxdna`: buffers, contexts, commands, waits, cache flushes | the kernel driver | runner, `llama-xdna-*` |
| Instruction streams | `xdna-insts.{h,cpp}` | Operations of a stream; transfers of the shim tiles (`XdnaShimDma`) | — | gemm, gemv, chain |
| Manifests | `xdna-overlay.{h,cpp}` | What an overlay of the prompts offers (tiles, bands, kernel, `gdn`, `rows`) | — | gemm, runner, backend |
| Multiplication of the prompts | `xdna-gemm.{h,cpp}` | Plans and streams; layouts of A, B and C; conversions (rows, SwiGLU, gated norm, attention) | insts, overlay, bfp16, glu, workers | runner |
| Overlay for generating tokens | `xdna-gemv.{h,cpp}` | The non-chained overlay: shapes, streams, formats of the weights | insts | runner, chain |
| Chained overlay | `xdna-chain.{h,cpp}` | Phases, commands, their streams and the buffer of the host | insts, gemv | runner |
| Recurrence of a prompt | `xdna-gdnp.{h,cpp}` | Inputs and outputs of the recurrence of a batch | bfp16, workers | runner |
| Block floating point | `xdna-bfp16.{h,cpp}` | Encoder (AVX2), decoder, bfloat16 and halves | — | everything that converts |
| Arithmetic of the host | `xdna-glu.{h,cpp}` | SwiGLU, RMS scales, scores and exponentials of the attention, as the CPU backend computes them | bfp16 | gemm, backend |
| Threads of the host | `xdna-workers.{h,cpp}` | Shared loops (OpenMP), the other threads asleep while the NPU works | OpenMP | gemm, runner |
| Runner | `xdna-runner.{h,cpp}` | Opens the NPU with both overlays; converts, sends and reads multiplications, recurrences, chained commands and attention | all of the above | backend, tools |
| Backend of ggml | `ggml-xdna.cpp`, `ggml/include/ggml-xdna.h` | Device, buffer types, `supports_op`, graph compute and its fusions | runner, ggml | llama.cpp |
| Kernels | `ggml/src/ggml-xdna/kernels/*.cc` | Programs of the cores | Peano, AIE API | overlays |
| Overlays | `ggml/src/ggml-xdna/overlays/` | The two overlays built, their designs, `build-overlays.sh` | mlir-aie, Peano | CMake, backend |
| Tools | `tools/xdna-*/` | `llama-xdna-info`, `-run`, `-insts`, `-gemm`, `-gemv`, `-probe` | runner, device | developers |
| Installers | `deploy/install-update-reinstall-{debian,alpine}.sh` | Build and install from the sources | git, cmake, Node.js | users |

## 3. Index of key symbols

Paths under `ggml/src/ggml-xdna/` unless said otherwise.

| Symbol | File:line | What it does |
|---|---|---|
| `fProbe` | `ggml-xdna.cpp:452` | Reads the settings (`GGML_XDNA_*`), the manifests and the NPU when the backend registers |
| `fShared` | `ggml-xdna.cpp:577` | What the backends of a process share: the runner, the converted matrices, the buffers of states |
| `fCloseIfIdle`, `fOpenRunner` | `ggml-xdna.cpp:655`, `:677` | Close the NPU when nothing needs it; open it with both overlays |
| `fMulMatTaken` | `ggml-xdna.cpp:1127` | Whether a `MUL_MAT` goes to the NPU (weights in a buffer of the backend, rows, types) |
| `fPlaceWeights` | `ggml-xdna.cpp:883` | Whether a weight that llama.cpp places goes to the backend within the budget of its copies |
| `fIsRouter`, `fSawExperts` | `ggml-xdna.cpp:838`, `:932` | The router of a mixture of experts, which stays with the processor; whether one has been seen |
| `fConvert`, `fMatrixOf` | `ggml-xdna.cpp:1235`, `:1354` | The weights of one or more tensors converted for the overlay of the prompts, in pieces |
| `fGemvConvert` | `ggml-xdna.cpp:1409` | The same for the overlay for generating tokens |
| `fMultiplyRows` | `ggml-xdna.cpp:1523` | 1 to 8 rows on the overlay for generating tokens |
| `fMultiply`, `fMulMat` | `ggml-xdna.cpp:1673`, `:1872` | A multiplication of many rows, with two commands in flight; a group of multiplications of the same rows |
| `fGdnTaken`, `fGatedDeltaNet` | `ggml-xdna.cpp:2174`, `:2422` | The recurrence of a batch on the NPU |
| `fNormedQueries`, `fAttnTaken`, `fFlashAttention` | `ggml-xdna.cpp:2508`, `:2537`, `:2599` | The attention of a batch: whether to split Q and K, whether it is taken, and how it runs |
| `fConvTaken`, `fConvFold`, `fConvOnHost` | `ggml-xdna.cpp:2864`, `:2921`, `:3031` | The short convolution of a batch: whether it is taken, whether it goes with the reading of C, and on the host |
| `fLinearTaken`, `fLinearAttention` | `ggml-xdna.cpp:3113`, `:3411` | The whole layer of linear attention while generating (`GGML_XDNA_LINEAR`) |
| `fMulMatGlu` | `ggml-xdna.cpp:3656` | The gate and the up with their SwiGLU, into the rows of `ffn_down` |
| `fBackendFree` | `ggml-xdna.cpp:3844` | The report of what the backend did |
| `fFeedForward` | `ggml-xdna.cpp:4005` | A layer of a token in one command of the chained overlay |
| `fGatedNorm` | `ggml-xdna.cpp:4469` | The gated norm into the rows of `ssm_out` |
| `fGraphCompute` | `ggml-xdna.cpp:4568` | Runs a part of a graph and recognizes its patterns |
| `fBufferFree`, `fDeviceBufferFromHostPtr` | `ggml-xdna.cpp:5089`, `:5164` | Buffers of weights; over the mapping of the file |
| `fStateBufferTypeAlloc` | `ggml-xdna.cpp:5216` | Buffers of the driver for recurrent states |
| `fDeviceSupportsOp` | `ggml-xdna.cpp:5320` | The operations the device takes |
| `ggml_backend_xdna_reg` | `ggml-xdna.cpp:5468` | The registry of the backend |
| `XdnaRunner::mOpen` | `xdna-runner.cpp:39` | Context with the two overlays as compute units 0 and 1 |
| `XdnaRunner::mEnsure` | `xdna-runner.cpp:223` | A buffer at least of a size, flushed when it is new |
| `XdnaRunner::mPackRows`, `mPrepare` | `xdna-runner.cpp:311`, `:353` | Rows into a buffer of A; the stream and the command of a multiplication |
| `XdnaRunner::mSubmitPacked`, `mFinishPacked` | `xdna-runner.cpp:505`, `:531` | A multiplication sent without waiting, and its result |
| `XdnaRunner::mUnpackSwigluRows` | `xdna-runner.cpp:655` | SwiGLU from two C into the rows of the next multiplication |
| `XdnaRunner::mPackAttnP` | `xdna-runner.cpp:755` | P of the attention from S into the rows of P·V |
| `XdnaRunner::mLoadChainMatrix`, `mChainMultiplyRows` | `xdna-runner.cpp:1003`, `:1113` | Weights for the chained overlay; rows by them |
| `XdnaRunner::mRunGdn` | `xdna-runner.cpp:1147` | The recurrence of a batch, in up to 4 commands in flight |
| `XdnaRunner::mRunChain` | `xdna-runner.cpp:1270` | A command of phases of the chained overlay |
| `XdnaDevice::mOpen`, `mCreateBuffer` | `xdna-device.cpp:161`, `:350` | The NPU; a buffer of the driver |
| `XdnaDevice::mConfigureContextUnits` | `xdna-device.cpp:495` | Several overlays in one context |
| `XdnaDevice::mSubmit`, `mWait` | `xdna-device.cpp:548`, `:566` | A command; waiting for it |
| `fXdnaFlushCache` | `xdna-device.cpp:101` | Flush from the process |
| `fXdnaGemmPlan`, `fXdnaGdnPlan` | `xdna-gemm.cpp:109`, `:356` | Streams of a multiplication and of the recurrence of a batch |
| `fXdnaGemmShape`, `fXdnaGemmPackA`, `fXdnaGemmPackB` | `xdna-gemm.cpp:717`, `:742`, `:775` | Padded sizes; A and B in the layout of the array |
| `fXdnaGemmPackSwiglu`, `fXdnaGemmPackAttnP` | `xdna-gemm.cpp:1483`, `:1776` | A SwiGLU, or P of the attention, converted into the tiles of A |
| `fConvStrips` | `xdna-gemm.cpp:1004` | The short convolution and its SiLU while C is read |
| `XdnaChainOverlay::mLoad`, `fXdnaChainPlan` | `xdna-chain.cpp:141`, `:574` | Manifest and stream of the chained overlay |
| `fXdnaGemvPackStream` | `xdna-gemv.cpp:502` | Weights of a segment, from floats or from the blocks of ggml |
| `XdnaOverlay::mLoad` | `xdna-overlay.cpp:36` | Manifest of the overlay of the prompts |
| `XdnaShimDma::mPush` | `xdna-insts.cpp:121` | A transfer of a shim channel |
| `fXdnaBfp16EncodeNearest` | `xdna-bfp16.cpp:202` | The fast encoder (AVX2) |
| `fXdnaGdnPack` | `xdna-gdnp.cpp:265` | Inputs of the recurrence of a batch |
| `fXdnaSwiglu` | `xdna-glu.cpp:541` | SwiGLU with the vectors of the CPU backend |
| `fXdnaConvToken`, `fXdnaConvRowsBf16` | `xdna-glu.cpp:614`, `:653` | A token of the short convolution, as the CPU backend computes it |
| `XdnaWorkers::mForEach` | `xdna-workers.cpp:32` | A loop on the threads of the host |
| `llama_context::resolve_fused_ops` | `src/llama-context.cpp:576` | The probes that decide flash attention and the fused recurrence |
| `graph_get_cb`, its blocks `llama.cpp-xdna` | `src/llama-context.cpp:2790` | Operations given to the accelerator on layers of the processor |
| `llama_memory_recurrent_accel_buft` | `src/llama-memory-recurrent.cpp:23` | Buffers of states of the accelerator |
| `llama_model_flint` | `src/models/flint.cpp:10` | Flint: its parameters (checked), its tensors and its graph |
| `llama_kv_cache_iswa` | `src/llama-kv-cache-iswa.h:17` | The KV cache of a model with an interleaved sliding window: one cache for each kind of layer |
| `build_attn_inp_kv_iswa`, `build_amax_src1_f16` | `src/llama-graph.cpp:2034`, `:1178` | The inputs of the attention with that cache; the rescaling of the activation of the experts of Flint |
| `rpc_node_ok`, `rpc_serve_auth` | `ggml/src/ggml-rpc/ggml-rpc.cpp:2566`, `:3714` | What an RPC server computes; its authentication |
| `fHttpHostAllowed` | `tools/server/server-http.cpp:178` | The check of the `Host` header of `llama-server` |

## 4. Main flows

### Loading a model

1. `ggml_backend_xdna_reg` registers the device; `fProbe` reads the settings and the manifests and checks the NPU (`XdnaDevice::mQueryInfo`).
2. llama.cpp gives the weights of the multiplications the backend takes to its buffer type; with `mmap`, `fDeviceBufferFromHostPtr` makes those buffers over the mapping of the file.
3. `fDeviceInit` opens the runner (`fOpenRunner` → `XdnaRunner::mOpen`: a context with both overlays). The recurrent states go to buffers of the driver (`fStateBufferTypeAlloc`).

### A batch of a prompt

1. The scheduler gives the backend its parts of the graph; `fGraphCompute` walks them.
2. An `RMS_NORM` and its `MUL` before multiplications: their scales now (`fRowScales`), the rows normalized while converted.
3. `MUL_MAT`s of the same rows: `fMulMat` → `fMatrixOf` (weights converted the first time, `fConvert`) → `fMultiply`: rows into a buffer of A (`mPackRows`), the command (`mSubmitPacked`), C read into the result (`mFinishPacked` → `fXdnaGemmUnpackC`), with the residual sum or the `CONCAT` when they follow.
4. Gate and up followed by a SwiGLU: `fMulMatGlu`, the SwiGLU into the rows of `ffn_down` (`mUnpackSwigluRows`).
5. Linear attention: the short convolution with its SiLU while the C of `attn_qkv` is read (`fConvFold` → `fConvStrips`; on the host when it cannot go with it, `fConvOnHost`), the recurrence (`fGatedDeltaNet` → `mRunGdn`, with the L2 norms of q and k folded in) and the gated norm into the rows of `ssm_out` (`fGatedNorm`).
6. Attention: `fFlashAttention` (next flow).

### Generating a token

1. `fGraphCompute` finds a `MUL_MAT` of one row that starts a feed-forward block: `fFeedForward` matches `MUL_MAT` → `ADD` → `RMS_NORM` → `MUL` → 2 `MUL_MAT` → `GLU` → `MUL_MAT` → `ADD`, and after it the norm and the projections of the next layer (or the output norm and the logits).
2. The matrices of the chained overlay (`fGemvMatrixOf` → `mLoadChainMatrix`), the phases (`XdnaChainPhase`), one command (`mRunChain` → `fXdnaChainPlan`), and the results read where the graph wants them (`fXdnaChainReadY`).
3. What does not make such a block goes by `fMultiplyRows` → `mChainMultiplyRows`.

### The attention of a batch

`fFlashAttention`: for every head of keys, K and the pieces of V (4,096 keys at most) converted from the F16 cache (`mLoadAttnB`); for every block of rows (the queries of the heads that share it, by tokens), S = Q·Kᵀ in one command (`mPackAttnQ`, `mSubmitPacked`), P = exp(S + mask − max) into the rows of the pieces of P·V (`mPackAttnP` → `fXdnaGemmPackAttnP`), O = P·V with all its pieces in flight, and O over the sums of the rows of P into the result. The heads go one after the other with the NPU and the host overlapped. Without a norm of q and k (`fNormedQueries`), Q and K go in two parts of bfp16.

### The recurrence of a batch

`fGatedDeltaNet` → `XdnaRunner::mRunGdn`: `fXdnaGdnPack` writes the states and the chunks of 16 tokens of a part of the rounds while the NPU computes the parts before (up to 4 commands in flight), and `fXdnaGdnUnpack` reads the outputs and the final states.

## 5. Map of entry points

| Command | Handler | File |
|---|---|---|
| `llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`, `llama-quantize`, `llama-completion`… | Those of llama.cpp | `tools/*/` |
| `ggml-rpc-server` | `ggml_backend_rpc_start_server` | `tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp` |
| `llama-xdna-info [--check-buffers]` | `main` | `tools/xdna-info/xdna-info.cpp` |
| `llama-xdna-run --pdi --insts --in --out [--repeat]` | `main` | `tools/xdna-run/xdna-run.cpp` |
| `llama-xdna-insts gemm \| gemv \| chain \| gdn \| empty \| dump …` | `main` | `tools/xdna-insts/xdna-insts.cpp` |
| `llama-xdna-gemm [pack \| unpack \| gdn] …` | `main` | `tools/xdna-gemm/xdna-gemm.cpp` |
| `llama-xdna-gemv [--ffn \| --gdn \| --la] …` | `main` | `tools/xdna-gemv/xdna-gemv.cpp` |
| `llama-xdna-probe chain \| switch \| units` | `main` | `tools/xdna-probe/xdna-probe.cpp` |
| `build-overlays.sh <venv> <folder>` | `fMain` | `ggml/src/ggml-xdna/overlays/build-overlays.sh` |
| `install-update-reinstall-debian.sh`, `-alpine.sh` | `fMain` | `deploy/` |

The environment variables of the backend are in [MANUAL.md](MANUAL.md#8-settings-of-the-npu-backend); they are read in `fProbe`.

## 6. Impact analysis

- **An overlay** (`overlays/designs/`, `kernels/`): the host writes its streams from its manifest; a new layout or new parameters need the stream generators changed (`fXdnaGemmPlan`, `fXdnaGdnPlan`, `fXdnaChainPlan`) and, for the chained overlay, a new version of its manifest (`xdna-gemv <n>`), which the host checks. A new default overlay has to be copied into `ggml/src/ggml-xdna/overlays/`. Run a new design once first: a command that hangs costs powering the computer off.
- **The encoder** (`xdna-bfp16.cpp`): every result of the NPU depends on it; a change has to give the same bytes (check the logits of a model, byte for byte, before and after).
- **The fusions of `fGraphCompute`**: they depend on the order of the nodes that `src/models/qwen35.cpp` builds and on the parts the scheduler makes. A change of the graph upstream can turn a fusion off without an error: the report of the backend (`-lv 4`) counts every fusion.
- **Writing a result before its turn** (`fConvFold`): the `SILU` of the short convolution is written while the C of the projection is read, before its turn in the graph, which is safe only because its memory is checked apart from everything the nodes in between read or write; a new fusion that writes a result early needs the same check. It also relies on `src/models/qwen35.cpp` putting the convolution into the graph before the recurrent state.
- **`graph_get_cb`**: it gives operations to the accelerator; the probes of `resolve_fused_ops` build flash attention with 1 token and the recurrence with 16, which the backend must keep refusing, or llama.cpp turns those operations off for the whole model.
- **The buffers of the runner** (`cXdnaRunnerSlotsA`, `cXdnaRunnerSlotsC`): the SwiGLU into `ffn_down`, the gated norm and the attention use fixed slots; a command in flight owns its slots of A and C until it is finished.
- **The functions with several versions** (`XDNA_CLONES`, in `ggml-xdna.cpp`): they are built for AVX-512, AVX2 and baseline x86-64, and glibc picks the version when the program starts, with an indirect function (ifunc). musl (Alpine) has no ifunc, so there only the baseline version exists; today only the emulation uses it (`fDot`).
- **The operations of the processor** (`ggml/src/ggml-cpu/ops.cpp`): the changes give the same results bit for bit; after any change, compare byte for byte the logits of the processor alone before and after it (`GGML_XDNA_DISABLE=1 llama-perplexity -c 512 -b 512 --chunks 8 --kl-divergence-base <file>`, then `cmp`).
- **The RPC protocol**: the commands of this fork go after those of upstream; porting new commands of upstream renumbers them, and clients and servers of before and after stop understanding each other.

## 7. Extension points

- **An operation for the backend**: a function that says when it is taken (as `fAttnTaken`), its case in `fDeviceSupportsOp`, its handler in `fGraphCompute`, its counters in `XdnaTotals` and in the report of `fBackendFree`, and a switch `GGML_XDNA_NO_…` in `XdnaSettings`, read in `fProbe` (`fConvTaken`, `fConvFold` and `fConvOnHost` are a small example: an operation taken, its fusion and its version on the host). If the scheduler would not give it to the backend, `graph_get_cb` in `src/llama-context.cpp` can.
- **An overlay**: its design in `overlays/designs/` and its kernels in `kernels/`, its recipe in `overlays/build-overlays.sh` (built and copied by CMake with `-DGGML_XDNA_BUILD_OVERLAYS=ON -DGGML_XDNA_MLIR_AIE_VENV=<virtual environment>`), its manifest read in `xdna-overlay.cpp` or `xdna-chain.cpp`, its stream written by a plan, and a method of the runner that sends it.
- **A model architecture**: ported from upstream (the model in `src/models/`, `src/llama-arch.*`, `src/llama-model.*`, `src/llama-hparams.h`, `src/llama-model-saver.cpp`, its toy model in `tests/test-llama-archs.cpp`); its operations that are not in `rpc_node_ok` are not computed by an RPC server until they are added there; the fusions of the backend recognize the patterns of `qwen35`, and a new one needs its own.
- **A setting**: a field of `XdnaSettings`, read in `fProbe`, and its row in the table of [MANUAL.md](MANUAL.md#8-settings-of-the-npu-backend) and in its translations.
