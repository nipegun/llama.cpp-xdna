#!/bin/bash

# Builds the two overlays of the XDNA backend from their designs (designs/) and their kernels (../kernels/), with
# mlir-aie 1.4.5 and Peano (llvm-aie) installed in a Python 3.13 virtual environment: the overlay of the prompts
# (xdna2-gemm-bfp16.pdi and .manifest) and the chained overlay for generating tokens (xdna2-gemv-chain.pdi and
# .manifest), into the output folder, with what the builds leave in its subfolder work/. It runs nothing on the NPU.
# The overlays this folder keeps were built so, byte for byte (2026/11/16). See doc/CODE.md and doc/XDNA2.md.
#
# Use: bash build-overlays.sh <virtual environment with mlir_aie and llvm-aie> <output folder>

set -euo pipefail

cDesigns="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/designs"

fCleanup() {
  :
}

trap fCleanup EXIT

fMain() {
  if [ "$#" -ne 2 ]; then
    echo "use: bash build-overlays.sh <virtual environment with mlir_aie and llvm-aie> <output folder>"
    return 1
  fi
  local vVenv
  vVenv="$(cd "$1" && pwd)" || return 1
  mkdir -p "$2" || return 1
  local vOut
  vOut="$(cd "$2" && pwd)" || return 1
  local vSite
  vSite="$(ls -d "$vVenv"/lib/python3*/site-packages 2>/dev/null | head -n 1)"
  if [ -z "$vSite" ] || [ ! -d "$vSite/mlir_aie" ] || [ ! -d "$vSite/llvm-aie" ]; then
    echo "no mlir_aie and llvm-aie in $vVenv"
    return 1
  fi
  export MLIR_AIE_INSTALL_DIR="$vSite/mlir_aie"
  export PEANO_INSTALL_DIR="$vSite/llvm-aie"
  export NPU2=1
  export PATH="$vVenv/bin:$MLIR_AIE_INSTALL_DIR/bin:$PATH"
  export PYTHONPATH="$MLIR_AIE_INSTALL_DIR/python"
  export LD_LIBRARY_PATH="$MLIR_AIE_INSTALL_DIR/lib"
  # the cache of mlir-aie, which otherwise goes to ~/.npu
  export NPU_CACHE_HOME="$vOut/work/npu-cache"
  local vPrompt="$vOut/work/prompt"
  local vBanks="$vOut/work/prompt-banks"
  local vChain="$vOut/work/chain"
  rm -rf "$vPrompt" "$vBanks" "$vChain"
  mkdir -p "$vPrompt" "$vBanks/prj" "$vChain" || return 1

  # The overlay of the prompts: the bands of A kept in the memory tiles, the kernel in steps of three, C in bfloat16 by
  # strips, and the recurrence of a gated delta net of a batch, for K and N up to 4096 (any size by its manifest)
  echo "building the overlay of the prompts"
  if ! (cd "$vPrompt" && python3 "$cDesigns/bfpmm_resident.py" --dev npu2 -M 4096 -K 4096 -N 4096 --b-depth 8 \
    --c-type bf16 --kernel s3 --gdn 1 --c-rows 1 --xclbin-path final.xclbin --insts-path insts.bin > build.log 2>&1); then
    echo "failed: see $vPrompt/build.log"
    return 1
  fi
  # its buffers moved to other banks of the memory of every core, and the overlay built again from the MLIR with them
  cp "$vPrompt"/final.prj/xdna2_*.o "$vBanks/prj/" || return 1
  if ! python3 "$cDesigns/core_banks.py" "$vPrompt/final.prj/input_with_addresses.mlir" "$vBanks/banks.mlir"; then
    echo "failed: core_banks.py"
    return 1
  fi
  if ! (cd "$vBanks" && aiecc --unified --peano="$PEANO_INSTALL_DIR" --get-npu-insts --npu-insts-name=insts.bin \
    --get-xclbin --xclbin-name=final.xclbin --fold-ddr-addr-offset=false --tmpdir=prj --output-dir=. banks.mlir \
    > build.log 2>&1); then
    echo "failed: see $vBanks/build.log"
    return 1
  fi
  cp "$vBanks/banks.mlir" "$vBanks/prj/input_with_addresses.mlir" || return 1
  if ! python3 "$cDesigns/overlay_manifest.py" "$vBanks/prj" columns 64 128 64 > "$vOut/xdna2-gemm-bfp16.manifest"; then
    echo "failed: overlay_manifest.py"
    return 1
  fi
  cp "$vBanks/prj/main.pdi" "$vOut/xdna2-gemm-bfp16.pdi" || return 1

  # The chained overlay for generating tokens, version 9 (with the phase of a whole layer of linear attention)
  echo "building the chained overlay"
  if ! (cd "$vChain" && python3 "$cDesigns/gemv_chain.py" --dev npu2 --test la --xclbin-path final.xclbin \
    --insts-path insts.bin > build.log 2>&1); then
    echo "failed: see $vChain/build.log"
    return 1
  fi
  if ! python3 "$cDesigns/gemv_chain_manifest.py" "$vChain/final.prj" "$vOut/xdna2-gemv-chain.manifest"; then
    echo "failed: gemv_chain_manifest.py"
    return 1
  fi
  cp "$vChain/final.prj/main.pdi" "$vOut/xdna2-gemv-chain.pdi" || return 1
  echo "built in $vOut"
  return 0
}

if ! fMain "$@"; then
  echo "error"
  exit 1
fi
