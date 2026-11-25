# CODE.md

This CODE is also available in:\
en-GB, [en-US](CODE.md), [es-AR](CODE.es-AR.md), [es-ES](CODE.es-ES.md)

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

A fork of llama.cpp with three backends: XDNA (written for this project; today for the NPUs XDNA2), CPU and RPC; the other backends of llama.cpp were removed. It has six model architectures: `qwen35` (Qwen3.5 0.8B, 2B, 4B, 9B and Qwen3.8 27B, a hybrid of linear attention and attention, with a layer of multi-token prediction), `qwen35moe` (the same with a block of experts, Qwen3.6 35B-A3B), `qwen4exp` (Qwen3.8 Flash Next `512x56B`: 4-stream Hyper-Connections, QSA block-sparse attention with `lightning_indexer` + `top_k_mask`, lazy Per-Layer Embeddings and 512 experts of width 640), `granite` (dense Granite 4.2), `flint` (Flint 40B-A2.5B, a transformer with an interleaved sliding window, attention sinks and a block of experts) and `qwen3tts` (Qwen3-TTS 12Hz `Base`, `CustomVoice` and `VoiceDesign` speech synthesis, paired with `libmtmd`). Every change to a file that comes from llama.cpp is marked with a comment `llama.cpp-xdna`, so that it can be kept when code is ported from upstream.

### A backend of its own, without XRT

The backend talks to the `amdxdna` driver of Linux directly, with its ioctls (`ggml/src/ggml-xdna/uapi/amdxdna_accel.h`, copied from Linux 7.3.4): XRT is not needed, and Debian 13 has no XRT that knows the NPU. `XdnaDevice` is that layer: it opens `/dev/accel/accel0`, creates buffers that the host and the NPU share, creates and configures contexts, submits commands and waits for them on the synchronisation object of DRM of the context (`WAIT_CMD` does not exist on this NPU).

### The NPU and its overlays

The NPU (AIE2P) is an array of 8 columns: a shim tile (the interface with memory), a memory tile (512 KB) and 4 cores (64 KB of data and 16 KB of program each) per column, 32 cores in all. What the array does is an *overlay*: a PDI with the configuration of every tile and the programs of the cores, built from a design in IRON (Python, `ggml/src/ggml-xdna/overlays/designs/`) and the kernels of the cores (C++ for the AIE, `ggml/src/ggml-xdna/kernels/`) with mlir-aie and its compiler Peano.

Changing the overlay a command runs on costs some 2.0 to 2.5 ms, so there are three, each in its own `XdnaContext` sharing the device, heap and `SHARE` buffers (`vContext`, `vGemvContext` and `vMoeContext` of `XdnaRunner`):

- **The overlay of the prompts** (`xdna2-gemm-bfp16.*`): matrix multiplication C = A·B with the bands of A kept in the memory tiles (`bfpmm_resident.py`), the kernel in steps of three sets of sums (`kernels/xdna2-mm-bfp16-s3.cc`) decoding `B` from `bfp16`, `Q8_0` or 4-tiles-in-3-objects `Q6_K` (`comp 1`), C written in bfloat16 by strips of 64 columns, and the recurrence of a gated delta net of a batch in the same buffers (`kernels/xdna2-gdn-prompt.cc`, supporting any integer head ratio `1 ≤ value_heads / key_heads ≤ 4`).
- **The chained overlay for generating tokens** (`xdna2-gemv-chain.*`, version 9): 1 to 8 rows by a matrix in phases (`gemv_chain.py`, `kernels/xdna2-gemv-chain.cc`), with weights decoded in the cores from `bfp16`, `Q4_K`, `Q5_K`, `Q6_K` or `Q8_0` (`cQ8_0`) and `cChain = 1,728` floats (`vGateMax = 1,088`). A command runs up to 4 phases, and every phase takes its X from the Y of the phase before inside the array, with its prologue (the norm) and its epilogue (residual sum, SwiGLU): a whole layer goes in one command.
- **The mixture-of-experts overlay** (`xdna2-moe.*`, version 1): up to 1,136 chunks of 1 to 24 rows multiplied by their chosen expert matrices in one command (`moe_experts.py`, `kernels/xdna2-moe.cc`), with weights decoded in the cores from `bfp16`, `Q4_K`, `Q5_K`, `Q5_1` (`cQ5_1`, including `Q5_0` converted losslessly on load), `Q6_K` or `Q8_0` (`cQ8_0`), exact partial-tile tail groups in N (`cParamTailGroups`) and symmetric 32-aligned sub-blocks in K (`cParamTailSubs`), cached 1-expert dense plans (`aMoeDensePlans`) and optional two-segment SwiGLU inside core 0.

The host writes the instruction stream of every command (the transfers of the shim tiles, the parameters of the cores, the waits) in C++, from a manifest that the build of the overlay writes (`xdna-overlay.*`, `xdna-chain.*`, `xdna-moe.*`). The streams are the same, operation by operation, as those of mlir-aie, which was checked; nothing runs Python while a model runs.

### Number formats

- **Block floating point** (bfp16ebs8): 8 values of 8 bits that share an exponent, 9 bytes. The operands of the overlay of the prompts are bfp16, the sums 32-bit floats, and C is rounded to bfloat16. In precision this costs about what Q8_0 costs on the processor.
- **The weights of the file**: for generating tokens and for routed experts, the cores decode `Q4_K`, `Q5_K`, `Q6_K` and `Q8_0` as the file stores them; the other types go in bfp16.
- The weights are converted once, the first time they are used, into buffers of the driver (using L2-resident strip packing `fXdnaGemmPackBRows` and AVX2+F16C dequantizers `fDequantizeQ8_0Avx2` / `fDequantizeQ6KAvx2`); the file stays mapped (`mmap`), the buffers of weights are made over that mapping (`fDeviceBufferFromHostPtr`), and `MADV_DONTNEED` drops mapped 2D dense pages (`fNoteDenseCached`) and 3D expert pages (`fReleaseMappedExpertPages`) once their NPU copies are ready.

### What the backend takes from a graph

The device is an accelerator (`GGML_BACKEND_DEVICE_TYPE_ACCEL`): llama.cpp puts it before the CPU in every context. Its buffer type is host memory (`is_host`), so that llama.cpp gives it the weights of the multiplications it takes and the CPU can still read them; `fDeviceSupportsOp` says which nodes it takes, and `fGraphCompute` runs a part of a graph, recognising patterns of nodes to do in one go:

- **A batch of a prompt** (40 tokens or more): the multiplications that read the same rows in one command (`fMulMat`); an RMS norm and its weights applied while the rows are converted; the SwiGLU computed while the gate and the up are read, split into halves from `vHalf >= 1,024` with early submission of piece 0 of `ffn_down` (`vEarlySent`) and written straight as the rows of `ffn_down` (`fMulMatGlu`); the residual sum read with C; the `CONCAT` of the short convolution written by the multiplication, and the short convolution with its SiLU plus direct `bf16` + L2-norm prepacking of `k`, `q` and `v` computed while the C of `attn_qkv` is read (`fConvFold`, `fConvStrips`, `mPrepareGdnB`); the gated norm of the linear attention converted directly from the NPU recurrence output into the `A` tiles of `ssm_out` (`fGatedNorm`, `fXdnaGdnUnpackGatedToA`); the sigmoid gate of full attention fused on the host (`fAttnGateStep`); the recurrence of the linear attention on the NPU (`fGatedDeltaNet`, ratios `1:1` to `4:1`, with `0` recurrent scheduler splits via `fStateOpTaken`); the attention as two multiplications for every head of keys, the softmax on the host between them (`fFlashAttention`).
- **The memory of the copies**: when `MADV_DONTNEED` drops converted file pages (`GGML_XDNA_WEIGHTS_GB` and `GGML_XDNA_COPY_WEIGHTS` unset), dense models place both copies up to 48 % of host memory (`vDenseChainLimit`), or chain 100 % of generation layers while keeping static `bfp16` prompt layers and streaming only the remainder into two reusable buffers up to 72 % (`vDenseTotalLimit`, `vStreamPrompt`, as on Qwen3.8 27B; `GGML_XDNA_PROMPT_FORMAT=q6k` / `q8_0` keeps a single compressed static prompt copy instead), and mixtures of experts use up to 78 % (`vMoeWeightsLimit`). Above those limits or with `GGML_XDNA_WEIGHTS_GB` set, excess layers stay with the processor (`XdnaPlacement::vPartial`).
- **A mixture of experts** (`qwen35moe`, `flint`): whole layers of routed experts (`MUL_MAT_ID`) go to `xdna2-moe` (`fPlaceMoeWeights`, `fMoeTaken`, `fRunMoeStep`), fusing `{up_exps, gate_exps} + SwiGLU`, absorbing the 10-node `build_amax_src1_f16` rescaling of Flint (`fMatchMoeAmax`) so `down_exps` continues from the kept Y (`mRunMoeFromKept`), folding the shared expert and its scalar sigmoid gate as 1 or 2 virtual experts across all batch sizes including 9 to 39 rows (`fMulMatTaken` `vShexpOnMoe`, `fMatchMoeSharedExpert`, `fEnsureMoeWithShared`, which also reuses loaded `(routed + shared)` `dMoeMatrices` entries when queried without shared experts) and fusing the weighted reduction plus residual `ADD` into the reading of Y (`fMatchMoeTail`, `fXdnaMoeReadY`), while releasing the mapped file pages of the 3D experts (`fReleaseMappedExpertPages`) when `GGML_XDNA_MOE_MIN_TOKENS` is 1. Their discrete router stays with the processor by default (`fIsRouter`), or runs on XDNA under `GGML_XDNA_FULL`; generating tokens (1 to 8 rows), dense attention projections (`fEnsureMoeDenseGroup`, `fXdnaMoeReadYTargets`), leading dense feed-forward blocks (`fRunDenseMoeFfn`) and `output.weight` also run on `xdna2-moe` with 0 overlay switches (`GGML_XDNA_NO_MOE_DENSE=1` leaves dense and shared-expert generation on the processor; when single-token routed experts run on the processor via `GGML_XDNA_MOE_MIN_TOKENS` ≥ 2 or `GGML_XDNA_NO_MOE=1`, `fSharedExpertChain` chains the shared expert, its gate and residuals with the next layer's norm and projections on `xdna2-gemv-chain`, also with 0 switches).
- **Generating a token** (1 to 8 rows): every feed-forward block, with the norm and the projections of the next layer, in one command of the chained overlay (`fFeedForward`); the last layer with the output norm and the logits. Under `GGML_XDNA_FULL=1` (or `2`), `fFullOpTaken` and `fFullOpOnHost` claim all remaining graph operations so the entire prompt and generation graphs execute as a single XDNA subgraph (`splits = 1`).

### The memory of the host and of the NPU

The NPU is not cache coherent with the processor. What the host wrote leaves the cache before a command reads it; what the NPU wrote is invalidated in the cache before the host reads it; a new buffer leaves the cache before the NPU writes it (the kernel zeroes its pages through the cache, and those zeros would come back over the results); and in a buffer that both write, the host flushes up to the end of the page of what it wrote and one page more (the prefetcher leaves the next line modified). `fXdnaFlushCache` flushes from the process (`clflushopt`), `XdnaDevice::mSyncBuffer` through the driver.

A command that does not end leaves the NPU unusable until the computer is powered off; the runner never sends another command after one (`vBroken`), and the width of a context has to be that of the partition the overlay was built for.

### Changes outside the backend

- `src/llama-context.cpp`, `llama_context::graph_get_cb` and `resolve_fused_ops`: the scheduler moves an operation to an accelerator only from its weights or from the nodes next to it; the callback gives the XDNA backend, on a layer of the processor, the recurrence of a batch, the L2 norms of q and k, the attention, the full-attention sigmoid gate, the norms (`norm` and `l_last`) of the residual stream and zero-offset prefix `VIEW`s of weight tensors (such as the `n_vocab_mtp` partial output head `head_part`), when it takes them, while `resolve_fused_ops` accepts fused ops claimed by an `ACCEL` backend on CPU layers (`accel_on_cpu`).
- `src/models/qwen35.cpp`, `src/models/delta-net-base.cpp` and `src/models/granite.cpp`: the projections of a layer one after the other in the graph (and unit `SCALE(1.0)` nodes omitted in `granite.cpp` when `f_logit_scale == 1.0f` or `f_residual_scale == 1.0f`), the input of the short convolution by tokens, the short convolution before the recurrent state, the states of the convolution and of the recurrence read and written in their cell of the cache.
- `src/llama-graph.{h,cpp}`, `ggml/include/ggml.h`, `ggml/src/ggml.c`: `ggml_gated_delta_net_set_state_in_place` and `llm_graph_rs_in_place`.
- `src/llama-memory-recurrent.cpp` and `src/llama-kv-cache.cpp`: the recurrent states and CPU-layer `F16` KV caches (`<= 4 GiB`, including `llama_kv_cache_iswa`) in shared `XDNA_State` buffers of the driver (`ggml_backend_dev_get_state_buft`, `llama_kv_cache_accel_buft`), with clean-state flush tracking (`XdnaShared::sCleanStates`).
- The mixtures of experts, which the fork had taken out, ported from upstream for `qwen35moe` (`src/models/qwen35moe.cpp`, generated from `src/models/qwen35.cpp` by `_/port-tools/make_qwen35moe.py`): the keys, tensors and parameters of the experts (`src/llama-arch.*`, `src/llama-hparams.*`, `src/llama-model.*`), the test of `MUL_MAT_ID` that places their buffers (`src/llama-model-loader.cpp`), `build_lora_mm_id` and a reduced `build_moe_ffn` (`src/llama-graph.*`), with `conv_states` of the next recurrent layer pre-read right after `attn_post_norm` so no processor `GET_ROWS` split lands between `moe_out` and `ffn_gate_shexp`.
- `flint` (`src/models/flint.cpp`), ported from upstream with what it needs that the fork had taken out: the KV cache with an interleaved sliding window (`src/llama-kv-cache-iswa.{h,cpp}`, `llm_graph_input_attn_kv_iswa`, `build_attn_inp_kv_iswa` and its `build_attn` in `src/llama-graph.*`, the branch of `llama_model::create_memory`), the attention sinks, the RoPE only in the layers that have it, the global base of the RoPE in its layers with a window (`llama_model::get_rope_freq_base`), the bias of the choice of the experts and the rescaling of the activation of its experts before their down projection (`build_amax_src1_f16`).
- `qwen4exp` (`src/models/qwen4exp.cpp`, `src/llama-memory-hybrid-idx.{h,cpp}`), ported from upstream with exception-checked hyperparameters, paired projection expansions (`ple_key` + `ple_value`, `wq` + `wk` + `wv` + `index_q_proj` + `index_k_proj`), `SIGMOID` gated linear-attention norm (`fXdnaSigmoidMul`), `kv_only` (`n_outputs == 0`) MTP catch-up plus `n_vocab_mtp` partial output head slicing in `graph_mtp` (backed by `GGML_XDNA_MTP_FIRST_LAYER` / `GGML_XDNA_MTP_LAYERS` MoE budget reservation and `cWeightViewKeyTag` in `ggml-xdna.cpp`), automatic `CPU_REPACK` and whole-file `POSIX_MADV_WILLNEED` suppression for `mmap` models exceeding host RAM (`src/llama-model.cpp`), and active-expert `MADV_WILLNEED` / `POSIX_MADV_RANDOM` handling on oversized-readahead USB block devices (`llama_fd_read_ahead_kb`, `ggml/src/ggml-cpu/ggml-cpu.c`).
- `qwen3tts` (`src/models/qwen3tts.cpp`, `tools/mtmd/`, `tools/tts/tts.cpp`, `common/arg.cpp`, `vendor/miniaudio/`, `_/port-tools/convert_qwen3tts_to_gguf.py`): the 28-layer Talker (`LLM_ARCH_QWEN3TTS`, interleaved M-RoPE, `{Qcur, Kcur, Vcur}` and `output` expanded immediately for `fFeedForward`), plus a Qwen3-TTS-only `libmtmd` (`mtmd_select_device` chooses `XDNA`, `mtmd_gen_audio_get_voice` discovers `<|spk_*|>` voices for `CustomVoice`, `--tts-instruct` prepends `<|im_start|>user\n...\n` for `VoiceDesign` and `1.7B-CustomVoice`, `gen_code_head_2d[0..14]` splits `a.gen.code.head.weight` into standalone 2D tensors so `code_gen` chains on `xdna2-gemv-chain`, and `conv1d_w_2d` / `conv1d_tr_w_2d` give `code2wav` standalone 2D convolution weights on `xdna2-gemm-bfp16`).
- `ggml/src/ggml-cpu/ops.cpp`: faster `SSM_CONV`, `GATED_DELTA_NET` and `RMS_NORM` on the processor, with the same results bit for bit; `tools/llama-bench/llama-bench.cpp` and `examples/speculative-simple/speculative-simple.cpp` accept `--no-xdna` (`GGML_XDNA_DISABLE=1` before backend initialization) for pure CPU baselines.
- Security: the RPC server validates what a client sends and can ask for a token (HMAC-SHA256); `llama-server` restricts CORS to localhost and checks the `Host` header; the GGUF reader and the loader reject malformed files. See [SECURITY.md](../SECURITY.md).

## 2. Module map

| Module | Path | Responsibility | Depends on | Used by |
|---|---|---|---|---|
| Device layer | `ggml/src/ggml-xdna/xdna-device.{h,cpp}`, `uapi/` | ioctls of `amdxdna`: buffers, contexts, commands, waits, cache flushes | the kernel driver | runner, `llama-xdna-*` |
| Instruction streams | `xdna-insts.{h,cpp}` | Operations of a stream; transfers of the shim tiles (`XdnaShimDma`) | — | gemm, gemv, chain |
| Manifests | `xdna-overlay.{h,cpp}` | What an overlay of the prompts offers (tiles, bands, kernel, `gdn`, `rows`) | — | gemm, runner, backend |
| Multiplication of the prompts | `xdna-gemm.{h,cpp}` | Plans and streams; layouts of A, B and C; conversions (rows, SwiGLU, gated norm, attention) | insts, overlay, bfp16, glu, workers | runner |
| Overlay for generating tokens | `xdna-gemv.{h,cpp}` | The non-chained overlay: shapes, streams, formats of the weights (with AVX2 packing for `Q6_K`) | insts | runner, chain, moe |
| Chained overlay | `xdna-chain.{h,cpp}` | Phases, commands, their streams and the buffer of the host | insts, gemv | runner |
| Mixture-of-experts overlay | `xdna-moe.{h,cpp}` | Shapes, expert weight packing, 3-slot shim ring streams (`fXdnaMoePlan`), row-norm encoding, Y-to-X transcoding and parallel weighted/target readout | insts, gemv, bfp16, workers | runner, backend |
| Recurrence of a prompt | `xdna-gdnp.{h,cpp}` | Inputs and outputs of the recurrence of a batch | bfp16, workers | runner |
| Block floating point | `xdna-bfp16.{h,cpp}` | Encoder (AVX2), decoder, bfloat16 and halves | — | everything that converts |
| Arithmetic of the host | `xdna-glu.{h,cpp}` | SwiGLU, RMS scales, scores and exponentials of the attention, as the CPU backend computes them | bfp16 | gemm, backend |
| Threads of the host | `xdna-workers.{h,cpp}` | Shared loops (OpenMP), the other threads asleep while the NPU works | OpenMP | gemm, runner |
| Runner | `xdna-runner.{h,cpp}` | Opens the NPU with the three overlays; converts, sends and reads multiplications, recurrences, chained commands, attention and routed experts | all of the above | backend, tools |
| Backend of ggml | `ggml-xdna.cpp`, `ggml/include/ggml-xdna.h` | Device, buffer types, `supports_op`, graph compute and its fusions | runner, ggml | llama.cpp |
| Kernels | `ggml/src/ggml-xdna/kernels/*.cc` | Programs of the cores | Peano, AIE API | overlays |
| Overlays | `ggml/src/ggml-xdna/overlays/` | The three overlays built, their designs, `build-overlays.sh` | mlir-aie, Peano | CMake, backend |
| Tools | `tools/xdna-*/` | `llama-xdna-info`, `-run`, `-insts`, `-gemm`, `-gemv`, `-probe` | runner, device | developers |
| Speech synthesis (`libmtmd`, `llama-tts`) | `tools/mtmd/`, `tools/tts/`, `vendor/miniaudio/`, `_/port-tools/convert_qwen3tts_to_gguf.py` | `ECAPA-TDNN` speaker encoder (`Base`), `<|spk_*|>` voices (`CustomVoice`), `--tts-instruct` conditioning (`VoiceDesign` / `1.7B-CustomVoice`), 16-step `code_gen` predictor, `code2wav` 24 kHz DAC vocoder and GGUF converter | llama, ggml, backend | `llama-tts` |
| Installers | `deploy/install-update-reinstall-{debian,alpine}.sh` | Build and install from the sources | git, cmake, Node.js | users |

## 3. Index of key symbols

Paths under `ggml/src/ggml-xdna/` unless said otherwise.

| Symbol | File:line | What it does |
|---|---|---|
| `fProbe` | `ggml-xdna.cpp:522` | Reads the settings (`GGML_XDNA_*`), the manifests and the NPU when the backend registers |
| `fShared` | `ggml-xdna.cpp:698` | What the backends of a process share: the runner, the converted matrices, the buffers of states |
| `fCloseIfIdle`, `fOpenRunner` | `ggml-xdna.cpp:776`, `:798` | Close the NPU when nothing needs it; open it with both overlays |
| `fMulMatTaken` | `ggml-xdna.cpp:1514` | Whether a `MUL_MAT` goes to the NPU (weights in a buffer of the backend, rows, types) |
| `fNoteDenseCached` | `ggml-xdna.cpp:1693` | Drops mapped 2D file pages with `MADV_DONTNEED` once both NPU copies exist |
| `fPlaceWeights`, `fPlaceMoeWeights` | `ggml-xdna.cpp:1110`, `:6366` | Whether a dense weight or a whole MoE layer goes to the backend within the weight budget (`vWeightsLimit` / `vMoeWeightsLimit`) |
| `fIsRouter`, `fSawExperts` | `ggml-xdna.cpp:1021`, `:1107` | The router of a mixture of experts, which stays with the processor; whether one has been seen |
| `fMoeTaken`, `fRunMoeStep` | `ggml-xdna.cpp:6441`, `:7632` | Whether a `MUL_MAT_ID` goes to `xdna2-moe`, and how `{gate, up} + SwiGLU + down`, shared experts and the weighted tail run |
| `fMatchMoeSharedExpert`, `fEnsureMoeWithShared` | `ggml-xdna.cpp:7494`, `:7041` | Folds a layer's shared expert and its scalar gate as virtual experts into the `xdna2-moe` commands |
| `fEnsureMoeDensePair`, `fRunDenseMoeFfn` | `ggml-xdna.cpp:6985`, `:7991` | Dense feed-forward blocks of 1 to 8 rows on `xdna2-moe` without overlay switches |
| `fSharedExpertChain` | `ggml-xdna.cpp:5660` | Chains a shared-expert block and its residuals with the next layer's norm and projections on `xdna2-gemv-chain` |
| `fConvert`, `fMatrixOf` | `ggml-xdna.cpp:1860`, `:2086` | The weights of one or more tensors converted for the overlay of the prompts, in pieces |
| `fGemvConvert` | `ggml-xdna.cpp:2181` | The same for the overlay for generating tokens |
| `fMultiplyRows` | `ggml-xdna.cpp:2299` | 1 to 8 rows on the overlay for generating tokens |
| `fMultiply`, `fMulMat` | `ggml-xdna.cpp:2458`, `:2663` | A multiplication of many rows, with two commands in flight; a group of multiplications of the same rows |
| `fGdnTaken`, `fGatedDeltaNet` | `ggml-xdna.cpp:3045`, `:3299` | The recurrence of a batch on the NPU |
| `fNormedQueries`, `fAttnTaken`, `fFlashAttention` | `ggml-xdna.cpp:3578`, `:3607`, `:3675` | The attention of a batch: whether to split Q and K, whether it is taken, and how it runs |
| `fConvTaken`, `fConvFold`, `fConvOnHost` | `ggml-xdna.cpp:4099`, `:4156`, `:4266` | The short convolution of a batch: whether it is taken, whether it goes with the reading of C, and on the host |
| `fLinearTaken`, `fLinearAttention` | `ggml-xdna.cpp:3932`, `:4646` | The whole layer of linear attention while generating (`GGML_XDNA_LINEAR`) |
| `fMulMatGlu` | `ggml-xdna.cpp:4904` | The gate and the up with their SwiGLU, into the rows of `ffn_down` |
| `fBackendFree` | `ggml-xdna.cpp:5119` | The report of what the backend did |
| `fFeedForward` | `ggml-xdna.cpp:5298` | A layer of a token in one command of the chained overlay |
| `fAttnGateStep` | `ggml-xdna.cpp:6025` | Fuses the full-attention `SIGMOID + MUL` output gate on the host |
| `fGatedNorm` | `ggml-xdna.cpp:6218` | The gated norm into the rows of `ssm_out` |
| `fFullOpTaken`, `fFullOpOnHost` | `ggml-xdna.cpp:8096`, `:8390` | Single-subgraph glue operations (`GGML_XDNA_FULL=1` / `2`, `splits = 1`) |
| `fGraphCompute` | `ggml-xdna.cpp:9312` | Runs a part of a graph and recognises its patterns |
| `fBufferFree`, `fDeviceBufferFromHostPtr` | `ggml-xdna.cpp:10059`, `:10143` | Buffers of weights; over the mapping of the file |
| `fStateBufferTypeAlloc` | `ggml-xdna.cpp:10240` | Buffers of the driver for recurrent states |
| `fDeviceSupportsOp` | `ggml-xdna.cpp:10344` | The operations the device takes |
| `ggml_backend_xdna_reg` | `ggml-xdna.cpp:10513` | The registry of the backend |
| `XdnaRunner::mOpen`, `mSubmitGemvAndWait` | `xdna-runner.cpp:40`, `:273` | Opens `vContext`, `vGemvContext` and `vMoeContext`; submits and waits on `vGemvContext` |
| `XdnaRunner::mEnsure`, `mLoadWeightsRows` | `xdna-runner.cpp:304`, `:405` | A buffer at least of a size, flushed when new; L2-resident row-by-row prompt weight packing |
| `XdnaRunner::mPackRows`, `mPrepare` | `xdna-runner.cpp:525`, `:567` | Rows into a buffer of A; the stream and the command of a multiplication |
| `XdnaRunner::mSubmitPacked`, `mFinishPacked` | `xdna-runner.cpp:719`, `:745` | A multiplication sent without waiting, and its result |
| `XdnaRunner::mUnpackSwigluRows` | `xdna-runner.cpp:899` | SwiGLU from two C into the rows of the next multiplication |
| `XdnaRunner::mPackAttnP` | `xdna-runner.cpp:999` | P of the attention from S into the rows of P·V |
| `XdnaRunner::mLoadChainMatrix`, `mChainMultiplyRows` | `xdna-runner.cpp:1241`, `:1355` | Weights for the chained overlay; rows by them |
| `XdnaRunner::mRunGdn` | `xdna-runner.cpp:1448` | The recurrence of a batch, in up to 4 commands in flight |
| `XdnaRunner::mRunChain` | `xdna-runner.cpp:1576` | A command of phases of the chained overlay |
| `XdnaRunner::mLoadMoeMatrix`, `mRunMoe`, `mRunMoeFromKept` | `xdna-runner.cpp:1783`, `:1985`, `:2103` | Expert weights and commands on `xdna2-moe`, including direct Y-to-X continuation |
| `XdnaDevice::mOpen`, `mCreateBuffer` | `xdna-device.cpp:171`, `:360` | The NPU; a buffer of the driver |
| `XdnaDevice::mConfigureContextUnits` | `xdna-device.cpp:505` | Several overlays in one context |
| `XdnaDevice::mSubmit`, `mWait` | `xdna-device.cpp:558`, `:576` | A command; waiting for it |
| `fXdnaFlushCache` | `xdna-device.cpp:110` | Flush from the process |
| `fXdnaGemmPlan`, `fXdnaGdnPlan` | `xdna-gemm.cpp:110`, `:395` | Streams of a multiplication and of the recurrence of a batch |
| `fXdnaGemmShape`, `fXdnaGemmPackA`, `fXdnaGemmPackB`, `fXdnaGemmPackBRows` | `xdna-gemm.cpp:766`, `:798`, `:831`, `:880` | Padded sizes; A and B in the layout of the array (whole or strip-by-strip from row callbacks) |
| `fXdnaGemmPackSwiglu`, `fXdnaGemmPackAttnP` | `xdna-gemm.cpp:2002`, `:2299` | A SwiGLU, or P of the attention, converted into the tiles of A |
| `fConvStrips` | `xdna-gemm.cpp:1453` | The short convolution and its SiLU while C is read |
| `XdnaChainOverlay::mLoad`, `fXdnaChainPlan` | `xdna-chain.cpp:149`, `:583` | Manifest and stream of the chained overlay |
| `XdnaMoeOverlay::mLoad`, `fXdnaMoePlan`, `fXdnaMoeReadYTargets` | `xdna-moe.cpp:187`, `:476`, `:828` | Manifest, 3-slot shim ring stream and multi-member target unpacking of the mixture-of-experts overlay |
| `fXdnaGemvPackStream` | `xdna-gemv.cpp:706` | Weights of a segment, from floats or from the blocks of ggml |
| `XdnaOverlay::mLoad` | `xdna-overlay.cpp:36` | Manifest of the overlay of the prompts |
| `XdnaShimDma::mPush` | `xdna-insts.cpp:136` | A transfer of a shim channel |
| `fXdnaBfp16EncodeNearest` | `xdna-bfp16.cpp:202` | The fast encoder (AVX2) |
| `fXdnaGdnPack` | `xdna-gdnp.cpp:440` | Inputs of the recurrence of a batch |
| `fXdnaSwiglu` | `xdna-glu.cpp:595` | SwiGLU with the vectors of the CPU backend |
| `fXdnaConvToken`, `fXdnaConvRowsBf16` | `xdna-glu.cpp:692`, `:731` | A token of the short convolution, as the CPU backend computes it |
| `XdnaWorkers::mForEach` | `xdna-workers.cpp:32` | A loop on the threads of the host |
| `llama_context::resolve_fused_ops` | `src/llama-context.cpp:611` | The probes that decide flash attention and the fused recurrence |
| `graph_get_cb`, its blocks `llama.cpp-xdna` | `src/llama-context.cpp:2850` | Operations given to the accelerator on layers of the processor |
| `llama_memory_recurrent_accel_buft` | `src/llama-memory-recurrent.cpp:23` | Buffers of states of the accelerator |
| `llama_kv_cache_accel_buft` | `src/llama-kv-cache.cpp:125` | `XDNA_State` (`BO_SHARE`) buffers for CPU-layer KV caches |
| `llama_model_flint` | `src/models/flint.cpp:10` | Flint: its parameters (checked), its tensors and its graph |
| `llama_model_qwen4exp`, `llama_memory_hybrid_idx` | `src/models/qwen4exp.cpp:26`, `src/llama-memory-hybrid-idx.h:18` | Qwen3.8 Flash Next (`qwen4exp`): Hyper-Connections, QSA indexer cache, lazy PLE and 512-expert graph |
| `llama_model_qwen3tts` | `src/models/qwen3tts.cpp:7` | Qwen3-TTS Talker: its parameters (checked), its tensors and its chained graph |
| `mtmd_select_device`, `mtmd_graph_qwen3tts_gen::build` | `tools/mtmd/mtmd.cpp:154`, `tools/mtmd/models/qwen3tts-gen.cpp:654` | Selects `XDNA` for `libmtmd` (with `<|spk_*|>` voices and `--tts-instruct` in `mtmd.cpp`); builds `code_gen` (2D codebook heads) and `code2wav` (2D causal/transposed conv weights) |
| `llama_kv_cache_iswa` | `src/llama-kv-cache-iswa.h:17` | The KV cache of a model with an interleaved sliding window: one cache for each kind of layer |
| `build_attn_inp_kv_iswa`, `build_amax_src1_f16` | `src/llama-graph.cpp:2058`, `:1178` | The inputs of the attention with that cache; the rescaling of the activation of the experts of Flint |
| `rpc_node_ok`, `rpc_serve_auth` | `ggml/src/ggml-rpc/ggml-rpc.cpp:2566`, `:3714` | What an RPC server computes; its authentication |
| `fHttpHostAllowed` | `tools/server/server-http.cpp:178` | The check of the `Host` header of `llama-server` |

## 4. Main flows

### Loading a model

1. `ggml_backend_xdna_reg` registers the device; `fProbe` reads the settings and the manifests and checks the NPU (`XdnaDevice::mQueryInfo`).
2. llama.cpp gives the weights of the multiplications the backend takes to its buffer type; with `mmap`, `fDeviceBufferFromHostPtr` makes those buffers over the mapping of the file.
3. `fDeviceInit` opens the runner (`fOpenRunner` → `XdnaRunner::mOpen`: three contexts `vContext`, `vGemvContext` and `vMoeContext` sharing the device and heap). The recurrent states go to buffers of the driver (`fStateBufferTypeAlloc`).

### A batch of a prompt

1. The scheduler gives the backend its parts of the graph; `fGraphCompute` walks them.
2. An `RMS_NORM` and its `MUL` before multiplications: their scales now (`fRowScales`), the rows normalised while converted.
3. `MUL_MAT`s of the same rows: `fMulMat` → `fMatrixOf` (weights converted the first time, `fConvert`) → `fMultiply`: rows into a buffer of A (`mPackRows`), the command (`mSubmitPacked`), C read into the result (`mFinishPacked` → `fXdnaGemmUnpackC`), with the residual sum or the `CONCAT` when they follow.
4. Gate and up followed by a SwiGLU: `fMulMatGlu`, the SwiGLU into the rows of `ffn_down` (`mUnpackSwigluRows`).
5. Linear attention: the short convolution with its SiLU while the C of `attn_qkv` is read (`fConvFold` → `fConvStrips`; on the host when it cannot go with it, `fConvOnHost`), the recurrence (`fGatedDeltaNet` → `mRunGdn`, with the L2 norms of q and k folded in) and the gated norm into the rows of `ssm_out` (`fGatedNorm`).
6. Attention: `fFlashAttention` (next flow).

### Generating a token

1. `fGraphCompute` finds a `MUL_MAT` of one row that starts a dense feed-forward block (`fFeedForward` matches `MUL_MAT` → `ADD` → `RMS_NORM` → `MUL` → 2 `MUL_MAT` → `GLU` → `MUL_MAT` → `ADD`, and after it the norm and the projections of the next layer, or — also across a 1-row `GET_ROWS` and a zero-offset prefix `VIEW` `head_part` (`cWeightViewKeyTag`) in MTP draft graphs — the output norm and the logits) or, when single-token routed experts run on the processor, a shared-expert block (`fSharedExpertChain` matches `{gate_shexp, up_shexp} + GLU + down_shexp [+ gate] + ADD + ADD` plus the next layer's norm and projections).
2. The matrices of the chained overlay (`fGemvMatrixOf` → `mLoadChainMatrix`), the phases (`XdnaChainPhase`), one command (`mRunChain` → `fXdnaChainPlan`), and the results read where the graph wants them (`fXdnaChainReadY`).
3. What does not make such a block goes by `fMultiplyRows` → `mChainMultiplyRows`, or on `xdna2-moe` (`fEnsureMoeDenseGroup`, `fRunDenseMoeFfn`) when routed experts run there at 1 to 8 rows so the token never switches overlays.

### The attention of a batch

`fFlashAttention`: for every head of keys, K and the pieces of V (4,096 keys at most) converted from the F16 cache (`mLoadAttnB`); for every block of rows (the queries of the heads that share it, by tokens), S = Q·Kᵀ in one command (`mPackAttnQ`, `mSubmitPacked`), P = exp(S + mask − max) into the rows of the pieces of P·V (`mPackAttnP` → `fXdnaGemmPackAttnP`, with exp(sink − max) added to the sum of every row when the layer has attention sinks), O = P·V with all its pieces in flight, and O over the sums of the rows of P into the result. The heads go one after the other with the NPU and the host overlapped. Without a norm of q and k (`fNormedQueries`), Q and K go in two parts of bfp16.

### The recurrence of a batch

`fGatedDeltaNet` → `XdnaRunner::mRunGdn`: `fXdnaGdnPack` writes the states and the chunks of 16 tokens of a part of the rounds while the NPU computes the parts before (up to 4 commands in flight), and `fXdnaGdnUnpack` reads the outputs and the final states.

### Routed experts (`MUL_MAT_ID`)

`fRunMoeStep`: groups `(token, slot)` pairs by expert into chunks of 1 to 24 rows (`fBuildMoeRouting`), folds any adjacent shared expert (`ffn_*_shexp` and its scalar gate `ffn_gate_inp_shexp`) as 1 or 2 virtual experts (`fMatchMoeSharedExpert`, `fEnsureMoeWithShared` → `mLoadMoeMatrix`, releasing read-only file pages of the 3D experts with `MADV_DONTNEED` when `GGML_XDNA_MOE_MIN_TOKENS` is 1), runs `{gate_exps, up_exps} + SwiGLU` in one command (`mRunMoe`), absorbs the optional 10-node `build_amax_src1_f16` rescaling of Flint (`fMatchMoeAmax`) so `down_exps` continues directly from the kept Y (`mRunMoeFromKept` → `fXdnaMoeTranscodeYToX`), and writes the weighted token reduction plus residual `ADD` while reading Y (`fMatchMoeTail`, `fXdnaMoeReadY`).

### Speech synthesis (`Qwen3-TTS`)

`llama-tts` (`tools/tts/tts.cpp`): on `Base` models, `mtmd_helper_encode_speech` decodes the reference voice (`miniaudio`), extracts Slaney mel features and runs `ECAPA-TDNN` (`qwen3tts-spkenc.cpp`) in `F32` on the processor, whereas `CustomVoice` looks up a built-in `<|spk_*|>` embedding (`mtmd_gen_audio_get_voice`, `--tts-voice`) and `VoiceDesign` / `1.7B-CustomVoice` prepend `<|im_start|>user\n{instruct}<|im_end|>\n` (`--tts-instruct`); for each 12.5 Hz frame, `llama_decode` runs the 28-layer Talker (`src/models/qwen3tts.cpp`) and `mtmd_generate_audio_tokens` runs 16 single-token steps of the 5-layer `code_gen` transformer (`gen_code_head_2d[0..14]`), both chained on `xdna2-gemv-chain` with 0 overlay switches; every 72 frames, `mtmd_decode_audio_codes` switches once to `xdna2-gemm-bfp16` to run `code2wav` (`conv1d_w_2d`, `conv1d_tr_w_2d`) into 24 kHz PCM.

## 5. Map of entry points

| Command | Handler | File |
|---|---|---|
| `llama-cli`, `llama-server`, `llama-tts`, `llama-bench`, `llama-perplexity`, `llama-quantize`, `llama-completion`… | Those of llama.cpp | `tools/*/` |
| `ggml-rpc-server` | `ggml_backend_rpc_start_server` | `tools/rpc/rpc-server.cpp`, `ggml/src/ggml-rpc/ggml-rpc.cpp` |
| `llama-xdna-info [--check-buffers]` | `main` | `tools/xdna-info/xdna-info.cpp` |
| `llama-xdna-run --pdi --insts --in --out [--repeat]` | `main` | `tools/xdna-run/xdna-run.cpp` |
| `llama-xdna-insts gemm \| gemv \| chain \| moe \| gdn \| empty \| dump …` | `main` | `tools/xdna-insts/xdna-insts.cpp` |
| `llama-xdna-gemm [pack \| unpack \| gdn] …` | `main` | `tools/xdna-gemm/xdna-gemm.cpp` |
| `llama-xdna-gemv [--ffn \| --gdn \| --la \| --moe] …` | `main` | `tools/xdna-gemv/xdna-gemv.cpp` |
| `llama-xdna-probe chain \| switch \| units` | `main` | `tools/xdna-probe/xdna-probe.cpp` |
| `build-overlays.sh <venv> <folder>` | `fMain` | `ggml/src/ggml-xdna/overlays/build-overlays.sh` |
| `install-update-reinstall-debian.sh`, `-alpine.sh` | `fMain` | `deploy/` |

The environment variables of the backend are in [MANUAL.en-GB.md](MANUAL.en-GB.md#8-settings-of-the-npu-backend); they are read in `fProbe`.

## 6. Impact analysis

- **An overlay** (`overlays/designs/`, `kernels/`): the host writes its streams from its manifest; a new layout or new parameters need the stream generators changed (`fXdnaGemmPlan`, `fXdnaGdnPlan`, `fXdnaChainPlan`, `fXdnaMoePlan`) and, for the chained or mixture-of-experts overlays, a new version of its manifest (`xdna2-gemv <n>`, `xdna2-moe <n>`), which the host checks (`fXdnaChainKey` hashes only geometry and operation kinds, never runtime scalars such as `vEps`, so instruction buffers are reused instead of filling the 64 MiB `DEV_HEAP`). Both `xdna2-gemv-chain` (`vGemvContext`) and `xdna2-moe` (`vMoeContext`) must stay in their own `XdnaContext`s: sharing `vContext` with `xdna2-gemm-bfp16` timed out after `Q5_K`/`Q6_K` commands. A new default overlay has to be copied into `ggml/src/ggml-xdna/overlays/`. Run a new design once first: a command that hangs costs powering the computer off.
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
- **An overlay**: its design in `overlays/designs/` and its kernels in `kernels/`, its recipe in `overlays/build-overlays.sh` (built and copied by CMake with `-DGGML_XDNA_BUILD_OVERLAYS=ON -DGGML_XDNA_MLIR_AIE_VENV=<virtual environment>`), its manifest read in `xdna-overlay.cpp`, `xdna-chain.cpp` or `xdna-moe.cpp`, its stream written by a plan, and a method of the runner that sends it.
- **A model architecture**: ported from upstream (the model in `src/models/`, `src/llama-arch.*`, `src/llama-model.*`, `src/llama-hparams.h`, `src/llama-model-saver.cpp`, its toy model in `tests/test-llama-archs.cpp`); its operations that are not in `rpc_node_ok` are not computed by an RPC server until they are added there; the fusions of the backend recognise the patterns of `qwen35`, and a new one needs its own.
- **A setting**: a field of `XdnaSettings`, read in `fProbe`, and its row in the table of [MANUAL.en-GB.md](MANUAL.en-GB.md#8-settings-of-the-npu-backend) and in its translations.
