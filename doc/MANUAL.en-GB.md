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

llama.cpp-xdna is a fork of [llama.cpp](https://github.com/ggml-org/llama.cpp): the same programs (`llama-cli`, `llama-server`, `llama-bench`, `llama-perplexity`, `llama-quantize`…), the same GGUF files, and three backends:

- **XDNA**, written for this project, for the NPU of AMD Ryzen AI processors. Today it runs on the generation of NPUs that AMD calls XDNA2 (8 columns of 4 cores); it is named so that it can take XDNA1 and XDNA3 too. It talks to the `amdxdna` driver of Linux directly (XRT is not needed).
- **CPU**, with one library for every generation of x86 processors; the best one for the computer is loaded when a program starts.
- **RPC**, to spread a model over several computers.

The other backends of llama.cpp were taken out.

With an NPU, the XDNA backend reads the prompts and generates the text on it (see [What runs on the NPU](#6-what-runs-on-the-npu)), and the processor does the rest. Without an NPU, or without its driver, the backend stays out and everything runs on the processor, as in llama.cpp.

## 2. What it needs

To use the NPU:

- **An AMD Ryzen AI processor with an XDNA2 NPU** (Strix Point, Krackan Point…). It was developed on a Ryzen AI 7 PRO 350 (Krackan Point, NPU `1022:17f0` revision `0x20`).
- **Linux 6.14 or later with the `amdxdna` driver** (`CONFIG_DRM_ACCEL_AMDXDNA`). Debian 13 comes with Linux 6.12, which does not have it: it needs a newer kernel. The NPU shows up as `/dev/accel/accel0`.
- **The firmware of the NPU, version 1.1.2.64 or later**, from [linux-firmware](https://git.kernel.org/pub/scm/linux/kernel/git/firmware/linux-firmware.git/tree/amdnpu). With the driver of Linux 7.x and the firmware 1.0.0.63 that Debian 13 brings, every command of the NPU ends in `abort`. Install it as `npu_7.sbin` in the folder of your NPU (for this one, `/lib/firmware/amdnpu/17f0_10/npu_7.sbin`; the driver looks for that name first) and reboot.
- **Access to `/dev/accel/accel0`**: it belongs to the group `render`. Add your user to it (`usermod -aG render <user>`) and log in again.
- **Memory**: the NPU keeps its own copies of the weights of a model, beside the file (at most 40 % of the memory of the computer, `GGML_XDNA_WEIGHTS_GB`; a model whose copies do not fit goes part on the NPU and part on the processor): in block floating point for prompts (1.125 bytes a weight) and in the format of the file for generating text (the same size as the file for Q4_K, Q5_K and Q6_K). Qwen3.5 4B Q4_K_M (2.6 GB) takes some 7 GB of buffers of the driver. The memory that the BIOS reserves for the integrated GPU is not available to the NPU.

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
- `-DGGML_XDNA=ON` builds the XDNA backend, its development tools, and puts the two overlays of the NPU (`xdna2-gemm-bfp16.*` and `xdna2-gemv-chain.*`) next to the programs. They come built in `ggml/src/ggml-xdna/overlays/`; to build them again from their designs, see [CODE.en-GB.md](CODE.en-GB.md#7-extension-points).
- Instead of installing it, the whole `build/bin/` folder (programs, libraries, backends and overlays) can be copied to another computer.
- The programs load the backends only from their own folder (`/usr/local/bin`, or `build/bin/`), never from the folder they are started from.
- `python3-numpy`, `python3-yaml` and `python3-tqdm` are only for the Python tools of `gguf-py` (`gguf_dump.py`, `gguf_set_metadata.py`…), which read and edit GGUF files.
- The tests (`test-backend-ops`…) are not installed; they run from `build/bin/`.

## 4. Models

Four architectures:

- **`qwen35`**: Qwen3.5 0.8B, 2B, 4B and 27B, with their layer of multi-token prediction (MTP), text only.
- **`qwen35moe`**: Qwen3.5 and 3.6 with a mixture of experts (Qwen3.6 35B-A3B: 256 experts, 8 of them used by every token), text only. Its experts and their router run on the processor.
- **`granite`**: the dense models of IBM Granite 4.2 (Granite 4.2 3B, `ibm-granite/granite-4.2-3b-GGUF`).
- **`flint`**: Flint 40B-A2.5B, with layers with a sliding window of 513 tokens and global layers, attention sinks and a mixture of experts (256 experts, 6 of them used by every token), text only. Its experts, their router and its attention run on the processor.

Formats: Q8_0, Q6_K, Q5_K_M and Q4_K_M, and also BF16, F16 and F32. For generating text the NPU reads the Q4_K, Q5_K and Q6_K weights as the file stores them; every other type goes in block floating point (8 bits and an exponent every 8 values), which costs about the precision of Q8_0. The KV cache has to be F16 (the default) for the attention to go to the NPU.

## 5. Running it

The programs are those of llama.cpp. For example:

```bash
# chat in the terminal
llama-cli -m Qwen3.5-4B-Q4_K_M.gguf

# a server with its web UI on http://127.0.0.1:8080
llama-server -m Qwen3.5-4B-Q4_K_M.gguf --host 127.0.0.1 --port 8080

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
- **With a mixture of experts** (`qwen35moe`, `flint`): the experts and their router stay on the processor, and so does generating text, but the logits. The attention of Flint (with sinks) stays on the processor too.
- **Generating text** (1 to 8 tokens at a time): every layer in one command of the chained overlay, with its multiplications, its norms, its SwiGLU and its residual sums; the last one with the logits.
- **On the processor**: batches of 9 to 39 tokens (it is faster for them), the recurrence and the convolution of the linear attention while text is generated, the RoPE, the norms of q and k, the attention of a token being generated, the copies to the cache, and choosing the next token.

## 7. Checking that the NPU works

```bash
# what the driver says about the NPU (columns, firmware, clocks); it runs nothing on it
llama-xdna-info

# the messages of the backend: why it does not use the NPU, or what it did, when it is freed
llama-cli -m model.gguf -p "Hello" -n 16 -no-cnv -lv 4
llama-bench -m model.gguf -p 512 -n 0 -v
```

Without an NPU or with something missing, the backend says why with `-lv 4` (for example `XDNA: the NPU is not used: cannot read the overlay ...`). When it is freed, it reports what it did: the multiplications and commands of the NPU and their times, the attentions, the recurrences, and how many times it changed between its two overlays.

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
| `GGML_XDNA_LINEAR=1` | The whole layer of linear attention on the NPU while text is generated (it works, but it is slower than the processor) |
| `GGML_XDNA_NO_STATE_BUFFERS=1` | The recurrent states in memory of the processor instead of buffers of the driver |
| `GGML_XDNA_WEIGHTS_GB` | The memory, in GB, that the copies of the weights for the NPU may take (40 % of the memory of the computer by default). A model whose copies do not fit goes part on the NPU and part on the processor, and then generates on the processor |
| `GGML_XDNA_MOE_GEMV=1` | With a mixture of experts, generating text on the NPU too (slower than the processor today) |
| `GGML_XDNA_EMULATE=1` | The processor does what the NPU would do, with its number formats; it does not open the NPU |

To compare (each one turns one improvement off): `GGML_XDNA_NO_FUSE`, `GGML_XDNA_NO_GLU`, `GGML_XDNA_NO_OVERLAP`, `GGML_XDNA_NO_PACK_GLU`, `GGML_XDNA_NO_GATED`, `GGML_XDNA_NO_L2`, `GGML_XDNA_NO_PARK`, `GGML_XDNA_NO_CHAIN_FFN`, `GGML_XDNA_NO_CHAIN_NEXT`, `GGML_XDNA_NO_EARLY_DESCRIPTORS`, `GGML_XDNA_NO_USER_FLUSH`, `GGML_XDNA_NO_FILLS_FIRST`, `GGML_XDNA_NO_ATTN_OVERLAP`, `GGML_XDNA_NO_CONV`, `GGML_XDNA_NO_CONV_FOLD`, and `GGML_XDNA_GDN_PARTS` (1 to 4 commands for a recurrence). What each one does is in [XDNA2.md](XDNA2.md).

## 9. Speed

Measured with `llama-bench -fa 1` on a Ryzen AI 7 PRO 350 (8 threads), tokens per second. The laptop changes its speed by several points from one hour to another, so only measures taken one after the other compare well.

Qwen3.5 4B, generating (`-n 64`), measured one after the other on 2026/11/16:

| File | NPU | Processor alone |
|---|---|---|
| Q4_K_M | 21.3 | 15.1 |
| Q5_K_M | 19.1 | 12.9 |
| Q6_K | 18.0 | 12.3 |
| Q8_0 | 13.9 | 9.4 |

Qwen3.5 4B Q4_K_M, a prompt of 512 tokens: some 1,210 with the NPU, against some 150 with the processor alone. With context in the cache (`-d`), the attention on the NPU and on the processor:

| Context | Attention on the processor | Attention on the NPU |
|---|---|---|
| none | 1,089 | 1,149 |
| 1,024 tokens | 793 | 1,118 |
| 2,048 tokens | 696 | 1,048 |
| 8,192 tokens | 294 | 829 |

Qwen3.8 27B Q6_K (22.4 GB, two thirds of it on the NPU): a prompt of 512 tokens, 80.3 against 36.2 with the processor alone; generating, 2.23 against 2.15.

Qwen3.6 35B-A3B Q6_K (a mixture of experts, 29.2 GB): a prompt of 512 tokens, 283 against 168 with the processor alone; generating, 16.4 against 15.7.

Flint 40B-A2.5B Q6_K (a mixture of experts, 30.5 GiB): a prompt of 512 tokens, 223 to 226 against 165 to 166 with the processor alone; generating, 19.0 to 19.4 against 18.4 to 18.8.

Granite 4.2 3B Q5_K_M: a prompt of 512 tokens, 990 against 212 with the processor alone (582 against 421 with the attention on the processor with 1,024 tokens of context); generating, 18.7 against 16.5.

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
- **To compare with the processor**: `GGML_XDNA_DISABLE=1` runs everything on it.

## 13. Tools for developers

Built with `-DGGML_XDNA=ON`:

| Tool | What it does |
|---|---|
| `llama-xdna-info` | What the driver reports about the NPU; `--check-buffers` creates and syncs a buffer of every type. It runs nothing on the NPU |
| `llama-xdna-run` | Runs a design compiled with mlir-aie (`--pdi`, `--insts`, `--in`, `--out`) and times it |
| `llama-xdna-insts` | Writes and prints the instruction streams that go to the NPU with a command; it does not touch the NPU |
| `llama-xdna-gemm` | Multiplies random matrices on the overlay of the prompts and compares the results with the processor; `gdn`, the recurrence of a prompt |
| `llama-xdna-gemv` | The same with the overlay for generating text; `--ffn`, `--gdn`, `--la`, the commands of a layer |
| `llama-xdna-probe` | What a command, a chain of commands and a change of overlay cost |

How the backend is built is in [CODE.en-GB.md](CODE.en-GB.md), and the technical notes on the NPU, with every measurement, in [XDNA2.md](XDNA2.md).
