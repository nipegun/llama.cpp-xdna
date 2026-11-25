# Manual

This MANUAL is also available in:\
en-GB, [en-US](MANUAL.md), [es-AR](MANUAL.es-AR.md), [es-ES](MANUAL.es-ES.md)

How to install llama.cpp-xdna and use it, with the NPU of an AMD Ryzen AI processor or without it.

## Index

1. [What it does](#1-what-it-does)
2. [What it needs](#2-what-it-needs)
3. [Install](#3-install)
4. [Models](#4-models)
5. [Running it](#5-running-it)
6. [What runs on the NPU](#6-what-runs-on-the-npu)
7. [Checking that the NPU works](#7-checking-that-the-npu-works)
8. [Settings of the NPU backend](#8-settings-of-the-npu-backend)
9. [Speed](#9-speed)
10. [Several computers](#10-several-computers)
11. [The web server](#11-the-web-server)
12. [When something goes wrong](#12-when-something-goes-wrong)
13. [Tools for developers](#13-tools-for-developers)

## 1. What it does

llama.cpp-xdna is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp): the same programs (`llama-cli`, `llama-server`, `llama-tts`, `llama-bench`, `llama-perplexity`, `llama-quantize`…), the same GGUF files, and three backends:

- **XDNA**, written for this project, for the NPU of AMD Ryzen AI processors. Today it runs on the generation of NPUs that AMD calls XDNA2 (8 columns of 4 cores); it is named so that it can take XDNA1 and XDNA3 too. It talks to the `amdxdna` driver of Linux directly (XRT is not needed).
- **CPU**, with one library for every generation of x86 processors; the best one for the computer is loaded when a program starts.
- **RPC**, to spread a model over several computers.

The other backends of llama.cpp were taken out.

With an NPU, the XDNA backend reads the prompts and generates the text and speech on it (see [What runs on the NPU](#6-what-runs-on-the-npu)), and the processor does the rest. Without an NPU, or without its driver, the backend stays out and everything runs on the processor, as in llama.cpp.

## 2. What it needs

To use the NPU:

- **An AMD Ryzen AI processor with an XDNA2 NPU** (Strix Point, Krackan Point…). It was developed on a Ryzen AI 7 PRO 350 (Krackan Point, NPU `1022:17f0` revision `0x20`).
- **Linux 6.14 or later with the `amdxdna` driver** (`CONFIG_DRM_ACCEL_AMDXDNA`). Debian 13 comes with Linux 6.12, which does not have it: it needs a newer kernel. The NPU shows up as `/dev/accel/accel0`.
- **The firmware of the NPU, version 1.1.2.64 or later**, from [linux-firmware](https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/tree/amdnpu). With the driver of Linux 7.x and the firmware 1.0.0.63 that Debian 13 brings, every command of the NPU ends in `abort`. Install it as `npu_7.sbin` in the folder of your NPU (for this one, `/lib/firmware/amdnpu/17f0_10/npu_7.sbin`; the driver looks for that name first) and reboot.
- **Access to `/dev/accel/accel0`**: it belongs to the group `render`. Add your user to it (`usermod -aG render <user>`) and log in again.
- **Memory**: the NPU keeps its own copies of the weights of a model, beside the file (at most 40 % of the memory of the computer when `GGML_XDNA_WEIGHTS_GB` or `GGML_XDNA_COPY_WEIGHTS` is set, or automatically up to 48 % / 72 % on dense models and 76–78 % on mixtures of experts when `MADV_DONTNEED` drops the mapped file pages of converted 2D weights and 3D experts; a model whose copies do not fit goes part on the NPU and part on the processor): in block floating point for prompts (1.125 bytes a weight, or optionally compressed `Q6_K` 4-in-3 at 0.84375 bytes a weight / `Q8_0` via `GGML_XDNA_PROMPT_FORMAT`) and in the format of the file for generating text and for routed experts (`Q4_K`, `Q5_K`, `Q5_1`/`Q5_0`, `Q6_K` and `Q8_0`). Qwen3.5 4B Q8_0 (4.3 GiB) and Qwen3.5 9B Q8_0 (9.1 GiB) keep both copies on the NPU; Qwen3.8 27B Q6_K (20.9 GiB) and Q8_0 (27.1 GiB) keep 100 % of their 64 layers in chained generation and stream only the prompt layers that exceed the 72 % budget (or keep all 750 prompt matrices static in `45.0 GB` with `GGML_XDNA_PROMPT_FORMAT=q6k`); Qwen3.6 35B-A3B up to Q8_0 (35.2 GiB) and Flint 40B-A2.5B up to Q8_0 (39.4 GiB) keep 100 % of their expert layers on the NPU (replacing their file pages in memory); and models larger than host memory such as Qwen3.8 Flash Next (`qwen4exp`, 105.4 GiB `Q4_K_XL` on 61.5 GiB of RAM) keep 100 % of their 2D dense projections plus as many MoE layers as fit on the NPU (23/48 without MTP, or 17/48 plus the `blk.48` MTP layer with `--mtp 1`) while streaming the remaining experts directly from `mmap` without `CPU_REPACK` duplication. Both recurrent states and `F16` KV caches up to 4 GiB live in shared driver buffers (`XDNA_State`). The memory that the BIOS reserves for the integrated GPU is not available to the NPU.

To build it: Debian 13 or Alpine 3.21 or later, an Internet connection, some 5 GB of disk and, for the web UI of `llama-server`, some 4 GB of free memory.

## 3. Install

### With the installer

As `root`:

```bash
# Debian 13
curl -fsSL https://raw.githubusercontent.com/nipegun/llama.cpp-xdna/main/deploy/install-update-reinstall-debian.sh \
  | bash -s -- --install

# Alpine Linux 3.21 or later
wget -qO- https://raw.githubusercontent.com/nipegun/llama.cpp-xdna/main/deploy/install-update-reinstall-alpine.sh \
  | sh -s -- --install
```

The installer installs the build dependencies, downloads the latest sources, builds them with the backends CPU, RPC and XDNA and the web UI, and installs them in `/usr/local`: the programs, the backends (`libggml-*.so`) and the overlays of the NPU in `/usr/local/bin`, the libraries in `/usr/local/lib`. It logs what it does in `/root/llama.cpp-xdna-install.log`.

| Option | What it does |
|---|---|
| `--install`, `--update` | Build the latest sources and install them (over the installation before, if there is one) |
| `--reinstall` | The same, removing the files of the installation before first |
| `--uninstall` | Remove the files it installed |
| `--no-ui` | Without the web UI of `llama-server` (it then serves only its API, and Node.js is not needed) |
| `--jobs <n>` | Parallel jobs of the build (by default, the processors of the computer) |
| `--source <folder>` | Build a local copy of the sources instead of downloading them |

On Debian, the web UI needs Node.js 22, and Debian 13 has 20: the installer downloads the official Node.js 22 from nodejs.org into the folder of the build, checks it against its SHA-256 sum and removes it with the build. On Alpine it uses the Node.js of Alpine, which is 22 from Alpine 3.21 on.

### By hand

As root on Debian 13:

```bash
# 1. The build dependencies, and those of the Python tools
apt-get -y update
apt-get -y install git build-essential cmake libssl-dev curl xz-utils python3-numpy python3-yaml python3-tqdm

# 2. The sources
git clone https://github.com/nipegun/llama.cpp-xdna.git
cd llama.cpp-xdna

# 3. Node.js 22 for the web UI, checked, and only for this build
curl -fsSLO https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt
NODE_TAR=$(grep -o 'node-v22[^ ]*-linux-x64\.tar\.xz' SHASUMS256.txt)
curl -fsSLO "https://nodejs.org/dist/latest-v22.x/$NODE_TAR"
grep " $NODE_TAR\$" SHASUMS256.txt | sha256sum -c -
tar -xJf "$NODE_TAR"
export PATH="$PWD/${NODE_TAR%.tar.xz}/bin:$PATH"

# 4. Build
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON \
  -DGGML_RPC=ON -DGGML_XDNA=ON -DLLAMA_BUILD_UI=ON -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_TESTS_INSTALL=OFF
NODE_OPTIONS=--max-old-space-size=4096 cmake --build build -j "$(nproc)"

# 5. Install (programs, backends and overlays in /usr/local/bin, libraries in /usr/local/lib)
cmake --install build
ldconfig
```

- Without the web UI, skip step 3 and configure with `-DLLAMA_BUILD_UI=OFF`. The web UI needs some 3.2 GB of memory to build: `NODE_OPTIONS=--max-old-space-size=4096` lets Node.js use up to 4 GB.
- `-DGGML_XDNA=ON` builds the XDNA backend, its development tools, and puts the three overlays of the NPU (`xdna2-gemm-bfp16.*`, `xdna2-gemv-chain.*` and `xdna2-moe.*`) next to the programs. They come built in `ggml/src/ggml-xdna/overlays/`; to build them again from their designs, see [CODE.en-GB.md](CODE.en-GB.md#7-extension-points).
- Instead of installing it, the whole `build/bin/` folder (programs, libraries, backends and overlays) can be copied to another computer.
- The programs load the backends only from their own folder (`/usr/local/bin`, or `build/bin/`), never from the folder they are started from.
- `python3-numpy`, `python3-yaml` and `python3-tqdm` are only for the Python tools of `gguf-py` (`gguf_dump.py`, `gguf_set_metadata.py`…), which read and edit GGUF files.
- The tests (`test-backend-ops`…) are not installed; they run from `build/bin/`.

## 4. Models

Six architectures:

- **`qwen35`**: Qwen3.5 0.8B, 2B, 4B, 9B and 27B (Qwen3.8 27B), with their layer of multi-token prediction (MTP), text only.
- **`qwen35moe`**: Qwen3.5 and 3.6 with a mixture of experts (Qwen3.6 35B-A3B: 256 experts, 8 of them used by every token), text only. Its routed experts run on the NPU up to the memory budget (`GGML_XDNA_WEIGHTS_GB`), and their router runs on the processor.
- **`qwen4exp`**: Qwen3.8 Flash Next (`512x56B` / `179.55B`: 48 hybrid `3:1` linear-attention and QSA block-sparse attention layers plus 1 MTP layer, 4-stream Hyper-Connections, lazy n-gram hash Per-Layer Embeddings on `blk.1`, and 512 experts of hidden width 640 with 10 routed + 1 shared expert per token), text only.
- **`granite`**: the dense models of IBM Granite 4.2 (Granite 4.2 3B, `ibm-granite/granite-4.2-3b-GGUF`).
- **`flint`**: Flint 40B-A2.5B, with layers with a sliding window of 513 tokens and global layers, attention sinks and a mixture of experts (256 experts, 6 of them used by every token), text only. Its routed experts run on the NPU up to the memory budget, and their router runs on the processor.
- **`qwen3tts`**: Qwen3-TTS 12Hz speech synthesis at 24 kHz (`llama-tts`), paired with its audio projector (`mmproj-Qwen3-TTS-12Hz-*.gguf`), in all three variants: **`Base`** (`0.6B` and `1.7B`, zero-shot voice cloning from an audio sample with `--tts-speaker-file`), **`CustomVoice`** (`0.6B` and `1.7B`, 9 built-in speakers — `aiden`, `dylan`, `eric`, `ono_anna`, `ryan`, `serena`, `sohee`, `uncle_fu`, `vivian` — selected with `--tts-voice`, plus emotion/style instructions with `--tts-instruct` on `1.7B`), and **`VoiceDesign`** (`1.7B`, creating new voices from natural-language descriptions with `--tts-instruct`).

Formats: Q8_0, Q6_K, Q5_K_M, Q5_1, Q5_0 and Q4_K_M / Q4_K_XL, and also BF16, F16 and F32. For generating text, the NPU decodes `Q4_K`, `Q5_K`, `Q6_K` and `Q8_0` weights directly in its cores, and for routed experts (`xdna2-moe`) it also decodes `Q5_1` (and `Q5_0` converted losslessly to `Q5_1`) with exact partial-tile shapes; every other type goes in block floating point (8 bits and an exponent every 8 values), which costs about the precision of Q8_0. The KV cache has to be F16 (the default) for the attention to go to the NPU.

## 5. Running it

The programs are those of llama.cpp. For example:

```bash
# chat in the terminal
llama-cli -m Qwen3.5-4B-Q4_K_M.gguf

# a server with its web UI on http://127.0.0.1:8080
llama-server -m Qwen3.5-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080

# speech synthesis at 24 kHz: clone a voice (Base), create one from text (VoiceDesign), or pick a preset voice (CustomVoice)
llama-tts -m Qwen3-TTS-12Hz-1.7B-Base-Q4_K_M.gguf --mmproj mmproj-Qwen3-TTS-12Hz-1.7B-Base-Q8_0.gguf \
  --tts-speaker-file voice-ref.wav --tts-lang es -c 2048 -p "Hola, esta voz sale de la NPU." -o cloned.wav
llama-tts -m Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q4_K_M.gguf --mmproj mmproj-Qwen3-TTS-12Hz-1.7B-VoiceDesign-Q8_0.gguf \
  --tts-instruct "Voz masculina grave, cálida y serena de narrador de documentales." \
  --tts-lang es -c 2048 -p "Hola, esta voz ha sido diseñada desde cero en la NPU." -o designed.wav
llama-tts -m Qwen3-TTS-12Hz-1.7B-CustomVoice-Q4_K_M.gguf --mmproj mmproj-Qwen3-TTS-12Hz-1.7B-CustomVoice-Q8_0.gguf \
  --tts-voice serena --tts-instruct "Habla con entusiasmo y cercanía." \
  --tts-lang es -c 2048 -p "Hola, hablo con una voz integrada de Qwen3-TTS." -o custom.wav

# speed of the prompts and of the generation
llama-bench -m Qwen3.5-4B-Q4_K_M.gguf -p 512 -n 64
```

Nothing has to be said for the NPU to be used: the XDNA backend comes first when the NPU, its driver and the overlays are there. Flash attention is on by default (`-fa auto`), which the attention on the NPU needs. `-t 8` sets the threads of the processor (the host part of the backend uses them too).

The first time a model is used, the backend converts its weights for the NPU (a few seconds with a 4B). The weights of the file are not copied: they stay in its mapping (`mmap`), as on the processor.

## 6. What runs on the NPU

- **Prompts** (batches of 40 tokens or more):
  - the multiplications of the model by its weights, with the norms, the SwiGLU, the residual sums and the short convolution of the linear attention of Qwen3.5 done while their rows are converted or their results read;
  - the attention (`FLASH_ATTN_EXT`), as two multiplications for every head of keys; with models without a norm of q and k (Granite), from 1,024 keys of context;
  - the recurrence of the layers of linear attention of Qwen3.5, in batches of 128 tokens or more.
- **With a mixture of experts** (`qwen35moe`, `qwen4exp`, `flint`): all layers of routed experts (`MUL_MAT_ID`) that fit in the automatic budget (40/40 on Qwen3.6 35B-A3B, 47/47 on Flint 40B-A2.5B, and 23/48 — or 17/48 plus the `blk.48` MTP layer when `--mtp` is enabled — on Qwen3.8 Flash Next `Q4_K_XL` with 61.5 GiB of RAM), together with their SwiGLU, shared experts, weighted reduction and residual sum, run on the mixture-of-experts overlay (`xdna2-moe`), and while generating text so do the attention, Hyper-Connection and PLE input projections, leading dense feed-forward blocks and logits (including the `n_vocab_mtp` partial head slice) with 0 overlay switches (or on the chained overlay when single-token routed experts run on the processor via `GGML_XDNA_MOE_MIN_TOKENS=2`); their discrete router and attention output projection stay on the processor.
- **Generating text** (1 to 8 tokens at a time): every layer in one command of the chained overlay, with its multiplications, its norms, its SwiGLU and its residual sums; the last one with the logits.
- **Speech synthesis** (`qwen3tts` + `mmproj`): the 28-layer Talker and the 16-step 5-layer `code_gen` multi-codebook predictor (together with its 15 codebook heads) chain on `xdna2-gemv-chain` with 0 overlay switches between frames, and the `code2wav` vocoder runs its 8-layer pre-transformer, ConvNeXt pointwise projections and all multi-channel causal and transposed DAC convolutions on `xdna2-gemm-bfp16` (1 overlay switch per 72-frame chunk); on `Base` models, the `ECAPA-TDNN` speaker encoder runs once in exact F32 on the processor.
- **On the processor**: batches of 9 to 39 tokens (it is faster for them), the recurrence and the convolution of the linear attention while text is generated, the RoPE, the norms of q and k, the attention of a token being generated, the copies to the `XDNA_State` cache, and choosing the next token (`GGML_XDNA_FULL=1` moves the entire computation graph into a single XDNA subgraph with `splits = 1`, and `GGML_XDNA_FULL=2` also runs single-token attention on the prompt overlay).

## 7. Checking that the NPU works

```bash
# what the driver says about the NPU (columns, firmware, clocks); it runs nothing on it
llama-xdna-info

# the messages of the backend: why it does not use the NPU, or what it did, when it is freed
llama-cli -m model.gguf -p "Hello" -n 16 -no-cnv -lv 4
llama-bench -m model.gguf -p 512 -n 0 -v
```

Without an NPU or with something missing, the backend says why with `-lv 4` (for example `XDNA: the NPU is not used: cannot read the overlay ...`). When it is freed, it reports what it did: the multiplications and commands of the NPU and their times, the attentions, the recurrences, the routed experts, and how many times it changed between its overlays.

## 8. Settings of the NPU backend

Environment variables. The usual ones:

| Variable | What it does |
|---|---|
| `GGML_XDNA_DISABLE=1` | The backend does not offer the NPU: everything runs on the processor |
| `GGML_XDNA_DEVICE` | The device node of the NPU (`/dev/accel/accel0` by default) |
| `GGML_XDNA_OVERLAY_DIR` | The folder of the overlays (by default, the folder of the program; never the current folder) |
| `GGML_XDNA_MIN_BATCH` | Tokens a batch needs for its multiplications and its attention to go to the NPU (40 by default) |
| `GGML_XDNA_COPY_WEIGHTS=1` | Copy the weights of the model into buffers of the backend instead of leaving them in the mapping of the file |
| `GGML_XDNA_NO_ATTN=1` | The attention of the prompts on the processor |
| `GGML_XDNA_ATTN_SPLIT=0` or `1` | Q and K of the attention in one or in two parts of block floating point (by default, two when q and k are not normalised) |
| `GGML_XDNA_NO_GDN=1` | The recurrence of the prompts on the processor |
| `GGML_XDNA_NO_CHAIN=1` | Generate with the overlay for generating text that is not chained, when it is there |
| `GGML_XDNA_NO_GEMV=1` | Generate on the processor |
| `GGML_XDNA_LINEAR=1` | The whole layer of linear attention on the NPU while text is generated (it works, within 6.2 % of the processor thanks to clean-state flush skipping) |
| `GGML_XDNA_PROMPT_FORMAT` | Format of prompt weights on the NPU: `auto` (default: `bfp16`, or `q6k` 4-in-3 under a tight `GGML_XDNA_WEIGHTS_GB` cap), `bfp16`, `q6k` (25 % smaller static buffers) or `q8_0` |
| `GGML_XDNA_FULL=1` or `2` | Claim all graph operations so prompts and generation run as a single XDNA subgraph (`splits = 1`); `2` also dispatches 1-token `FLASH_ATTN_EXT` to `xdna2-gemm-bfp16` |
| `GGML_XDNA_NO_STATE_BUFFERS=1` | The recurrent states and KV caches in memory of the processor instead of `XDNA_State` buffers of the driver |
| `GGML_XDNA_WEIGHTS_GB` | The memory, in GB, that the copies of the weights for the NPU may take (40 % of the memory of the computer by default beside a kept mapping, or automatically up to 48 % / 72 % on dense models and 76–78 % on mixtures of experts when `MADV_DONTNEED` drops converted file pages). A model whose copies exceed those budgets goes part on the NPU and part on the processor |
| `GGML_XDNA_NO_MOE=1` | With a mixture of experts, keep the routed experts (`MUL_MAT_ID`) on the processor |
| `GGML_XDNA_MOE_MIN_TOKENS` | Minimum tokens in a batch for the routed experts to run on the NPU (1 by default; 2 runs single-token routed experts on the processor and chains the shared expert on `xdna2-gemv-chain`) |
| `GGML_XDNA_NO_MOE_DENSE=1` | With a mixture of experts, keep the dense projections and shared experts of generated tokens on the processor |
| `GGML_XDNA_MOE_ATTN_OUT=1` | With a mixture of experts, also run `ssm_out` and `attn_output` of MoE layers on the NPU when generating |
| `GGML_XDNA_MOE_GEMV=1` | With a mixture of experts, send every dense multiplication of generating text through `xdna2-gemv-chain` |
| `GGML_XDNA_EMULATE=1` | The processor does what the NPU would do, with its number formats; it does not open the NPU |

To compare (each one turns one improvement off): `GGML_XDNA_NO_FUSE`, `GGML_XDNA_NO_GLU`, `GGML_XDNA_NO_OVERLAP`, `GGML_XDNA_NO_PACK_GLU`, `GGML_XDNA_NO_GATED`, `GGML_XDNA_NO_GDN_FOLD`, `GGML_XDNA_NO_L2`, `GGML_XDNA_NO_PARK`, `GGML_XDNA_NO_CHAIN_FFN`, `GGML_XDNA_NO_CHAIN_NEXT`, `GGML_XDNA_NO_EARLY_DESCRIPTORS`, `GGML_XDNA_NO_USER_FLUSH`, `GGML_XDNA_NO_FILLS_FIRST`, `GGML_XDNA_NO_ATTN_OVERLAP`, `GGML_XDNA_NO_CONV`, `GGML_XDNA_NO_CONV_FOLD`, and `GGML_XDNA_GDN_PARTS` (1 to 4 commands for a recurrence). What each one does is in [XDNA2.md](XDNA2.md).

## 9. Speed

Measured with `llama-bench -fa 1` on a Ryzen AI 7 PRO 350 (8 threads), tokens per second. The laptop changes its speed by several points from one hour to another, so only measures taken one after the other compare well.

Qwen3.5 4B, generating (`-n 64`):

| File | NPU | Processor alone |
|---|---|---|
| Q4_K_M | 21.3 | 15.1 |
| Q5_K_M | 19.1 | 12.9 |
| Q6_K | 18.0 | 12.3 |
| Q8_0 | 14.9 | 10.8 |

Dense and mixture-of-experts models in `Q8_0`, `Q6_K` and `Q4_K_XL` (`llama-bench -p 512 -n 64 -r 3 --delay 3 -t 8 -fa 1` for `pp512` / `tg64`, and `llama-speculative-simple -n 64 --temp 0 -t 8 -fa on --mtp 1 --runs 5 --prompt-cycle` for `--mtp 1` on models with an MTP layer):

| Model | Prompt of 512 (`pp512`), NPU / processor | Generating (`tg64`), NPU / processor | Generating (`--mtp 1`), NPU / processor |
|---|---|---|---|
| Qwen3.5 0.8B `Q8_0` | 4,459.1 / 720.2 (6.19×) | 65.57 / 58.35 (1.12×) | 61.34 / 64.68 (0.95×) |
| Qwen3.5 2B `Q8_0` | 2,947.6 / 272.8 (10.81×) | 31.30 / 23.36 (1.34×) | 36.71 / 27.93 (1.31×) |
| Granite 4.2 3B `Q8_0` | 1,367.3 / 115.7 (11.82×) | 18.53 / 13.54 (1.37×) | — |
| Qwen3.5 4B `Q8_0` | 1,650.8 / 100.7 (16.40×) | 14.90 / 10.82 (1.38×) | 18.55 / 14.09 (1.32×) |
| Qwen3.5 9B `Q8_0` | 1,190.3 / 51.5 (23.12×) | 7.42 / 5.61 (1.32×) | 12.55 / 7.96 (1.58×) |
| Qwen3.8 27B `Q6_K` (64/64 chained, hybrid static+streamed prompt) | 210.2 / 36.2 (5.81×) | 3.02 / 2.15 (1.40×) | 4.25 / 3.27 (1.30×) |
| Qwen3.6 35B-A3B `Q6_K` (40/40 MoE layers) | 392.4 / 182.0 (2.16×) | 18.09 / 15.73 (1.15×) | 18.05 / 16.93 (1.07×) |
| Flint 40B-A2.5B `Q6_K` (47/47 MoE + 2 dense layers) | 311.1 / 168.5 (1.85×) | 19.12 / 18.92 (1.01×) | — |
| Qwen3.8 Flash Next `Q4_K_XL` (105.4 GiB on NVMe, 61.5 GiB RAM, 23/48 MoE) | 63.53 / 61.13 (1.04×) | 5.34 / 5.18 (1.03×) | 5.78 / 4.78 (1.21×) |

With context in the cache (`-d`), Qwen3.5 4B Q4_K_M (`pp512`), the attention on the NPU and on the processor:

| Context | Attention on the processor | Attention on the NPU |
|---|---|---|
| none | 1,089 | 1,149 |
| 1,024 tokens | 793 | 1,118 |
| 2,048 tokens | 696 | 1,048 |
| 8,192 tokens | 294 | 829 |

Qwen3-TTS 12Hz (`0.6B-Base`, `1.7B-Base`, `0.6B-CustomVoice`, `1.7B-CustomVoice` and `1.7B-VoiceDesign`): in `llama-bench` with native `Q8_0` (2026/11/20), the Talker reads a prompt of 512 tokens at 2,385 to 2,870 (`0.6B`, 3.9 to 4.7× the processor's 615.2) and 2,086 to 2,097 (`1.7B`, 9.6× the processor's 218.1), and generates at 94.2 to 94.5 (`0.6B` `Q8_0` vs 89.0, up to 105.2 in `Q4_K_M` and 1.89× in `BF16`) and 42.3 to 42.5 (`1.7B` `Q8_0` vs 30.1, 1.41×; up to 57.9 in `Q4_K_M` and 2.55× in `BF16`). End to end in `llama-tts` (72 frames = 5.76 s of 24 kHz speech), `0.6B Q8_0` synthesizes at `1.33` to `1.35×` real-time (`4.15–4.27 s`), `1.7B-Base Q8_0` runs faster than real-time at `1.06×` (`5.45 s` vs `6.91 s` before native `Q8_0`), and `1.7B-CustomVoice`/`VoiceDesign Q8_0` run at `0.98×` cold one-shot (`5.86–5.90 s`, `1.07` to `1.08×` steady-state post-prompt).

Prompts of 40 tokens or more go faster on the NPU (with 64, Qwen3.5 4B goes at 265 against 170). The integrated GPU of the same processor (Radeon 860M, with the Vulkan backend of upstream llama.cpp) read a prompt of 512 tokens of the 4B at some 340 tokens per second and generated at up to 23.9 (measured on 2026/11/15).

## 10. Several computers

The RPC backend spreads a model over several computers: start one `ggml-rpc-server` on every extra computer and add it with `--rpc`.

Start the server with a token in `GGML_RPC_TOKEN`, and give the same token to the client: the server then serves only the clients that know it, and the token itself never travels over the network. The traffic is not encrypted, so anyone who can see it can read the prompts and the answers: use the RPC server only on a trusted network, or through an SSH tunnel. For example:

```bash
# on the other computer (create the token once, for example with: openssl rand -hex 32)
GGML_RPC_TOKEN='<token>' ggml-rpc-server -H 0.0.0.0 -p 50052

# on the client
GGML_RPC_TOKEN='<token>' llama-server -m model.gguf --rpc 192.168.1.20:50052 -ngl 99
```

Without `GGML_RPC_TOKEN`, any client that can reach the server can use it, so bind it to `127.0.0.1` (the default) unless the network is trusted.

The layers are split between the devices (`-ts` sets the proportion), so a model that does not fit in one computer can fit in several, but every token goes through all of them one after the other, and the slowest one sets the speed.

With Qwen3.5, `-sm tensor` (experimental) splits every layer between the servers instead, so that they work at the same time, and the servers add up their partial results between themselves. Start a `ggml-rpc-server` for every device, the local one too, and use only them:

```bash
GGML_RPC_COMM_ENDPOINT=127.0.0.1:51052 ggml-rpc-server -H 127.0.0.1 -p 50052 &
GGML_RPC_COMM_ENDPOINT=127.0.0.1:51052 ggml-rpc-server -H 127.0.0.1 -p 50053 &
llama-server -m model.gguf --rpc 127.0.0.1:50052,127.0.0.1:50053 -dev RPC0,RPC1 -sm tensor -ngl 99
```

Server-to-server communication is disabled unless `GGML_RPC_COMM_ENDPOINT` is set on every participating server to the same rank-zero IPv4 address and port. Use the first RPC endpoint in the client list, with its port plus 1000 (`51052` above). The address must be a literal IPv4, reachable from the other servers and bindable on rank zero; DNS names and wildcard addresses are rejected. Rank zero listens only on that address, and the other ranks connect only to that endpoint. Ordinary layer splitting does not need this setting. Granite does not support `-sm tensor`.

The client asks every server which operations its device can run, and runs the others on another device.

## 11. The web server

`llama-server` lets only the pages of `localhost` and its own web UI use its API from a browser (`--cors-origins localhost`; the official llama.cpp allows every web site): to use it from a web application on another address, add that address with `--cors-origins`. It also validates the HTTP `Host` header before serving any route: localhost, the local socket address (including a LAN IP) and the names given with `--host` are allowed. For a public DNS name, or a reverse proxy that keeps it, add `--allowed-hosts llama.example.com` (names only, separated by commas). The CORS settings do not turn the check of `Host` off. More in [SECURITY.md](../SECURITY.md).

## 12. When something goes wrong

- **The backend does not use the NPU**: run the program with `-lv 4` and read the line `XDNA: the NPU is not used: ...`. Usually: no `/dev/accel/accel0` (no driver, or a kernel older than 6.14), no permission on it (the group `render`), or no overlays next to the program (or in `GGML_XDNA_OVERLAY_DIR`).
- **Every command ends in `abort`**: the firmware of the NPU is too old for the driver (see [What it needs](#2-what-it-needs)).
- **The NPU hangs**: a command that never ends leaves the NPU unusable (its queries give `EINVAL`, creating a context gives `ENODEV`) until the computer is powered off and on again. Do not unload the driver then (`rmmod amdxdna`): with the NPU in that state it blocks forever inside the kernel, and the computer then does not shut down either (the power button has to be held). `llama-xdna-run --pdi <overlay> --insts <stream> --repeat 0` checks the NPU without running anything: it only creates a context. The backend never sends a command again after one that did not end.
- **Out of memory**: the copies of the weights for the NPU go in buffers of the driver (see [What it needs](#2-what-it-needs)). A smaller format, a smaller model or less memory reserved for the integrated GPU in the BIOS help.
- **To compare with the processor**: `GGML_XDNA_DISABLE=1` (or `--no-xdna` in `llama-bench` and `llama-speculative-simple`) runs everything on it.

## 13. Tools for developers

Built with `-DGGML_XDNA=ON`:

| Tool | What it does |
|---|---|
| `llama-xdna-info` | What the driver reports about the NPU; `--check-buffers` creates and syncs a buffer of every type. It runs nothing on the NPU |
| `llama-xdna-run` | Runs a design compiled with mlir-aie (`--pdi`, `--insts`, `--in`, `--out`) and times it |
| `llama-xdna-insts` | Writes and prints the instruction streams that go to the NPU with a command; it does not touch the NPU |
| `llama-xdna-gemm` | Multiplies random matrices on the overlay of the prompts and compares the results with the processor; `gdn`, the recurrence of a prompt |
| `llama-xdna-gemv` | The same with the overlay for generating text (`--ffn`, `--gdn`, `--la`) and the mixture-of-experts overlay (`--moe`) |
| `llama-xdna-probe` | What a command, a chain of commands and a change of overlay cost |

How the backend is built is in [CODE.en-GB.md](CODE.en-GB.md), and the technical notes on the NPU, with every measurement, in [XDNA2.md](XDNA2.md).
