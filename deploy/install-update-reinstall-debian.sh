#!/bin/bash

# llama.cpp-xdna - installer for Debian 13 (trixie).
#
# Builds llama.cpp-xdna from its sources (the latest of its GitHub repository, or a local copy with --source) with
# the backends CPU, RPC and XDNA and the web UI of llama-server, and installs it: the programs, the backends
# (libggml-*.so) and the overlays of the NPU in /usr/local/bin, the libraries in /usr/local/lib. Running it again builds
# the latest sources and installs them over the old ones; --reinstall removes the old ones first. Run it as root.
#
# Use:
#   curl -fsSL https://raw.githubusercontent.com/nipegun/llama.cpp-xdna/main/deploy/install-update-reinstall-debian.sh \
#     | bash -s -- --install
#   bash install-update-reinstall-debian.sh --install | --update | --reinstall | --uninstall [options]
#
# Options:
#   --no-ui           without the web UI of llama-server (then it serves only its API; Node.js is not needed)
#   --jobs <n>        parallel jobs of the build (default: the processors of the computer)
#   --source <folder> build from a local copy of the sources instead of downloading them
#
# What it does is logged in /root/llama.cpp-xdna-install.log.

set -euo pipefail

cRepository="https://github.com/nipegun/llama.cpp-xdna.git"
cPrefix="/usr/local"
cLog="/root/llama.cpp-xdna-install.log"
cManifest="$cPrefix/share/llama.cpp-xdna/install_manifest.txt"
vWork=""

fCleanup() {
  if [ -n "$vWork" ] && [ -d "$vWork" ]; then
    rm -rf "$vWork"
  fi
}

trap fCleanup EXIT

fSay() {
  echo "  $*"
  echo "$(date -u +%Y-%m-%d@%H-%M-%S.%3N) $*" >> "$cLog"
}

fUsage() {
  echo "use: bash install-update-reinstall-debian.sh --install | --update | --reinstall | --uninstall"
  echo "       [--no-ui] [--jobs <n>] [--source <folder>]"
}

# the files of the installation before this one, as cmake --install wrote them
fUninstall() {
  if [ ! -f "$cManifest" ]; then
    fSay "there is no installation of llama.cpp-xdna to remove"
    return 0
  fi
  fSay "removing the files of the installation before"
  local vFile
  while IFS= read -r vFile; do
    if [ -n "$vFile" ] && [ -f "$vFile" ]; then
      rm -f "$vFile"
    fi
  done < "$cManifest"
  rm -f "$cManifest"
  ldconfig
  return 0
}

fMain() {
  local vAction=""
  local vUi="ON"
  local vJobs
  vJobs="$(nproc)"
  local vSource=""
  while [ "$#" -gt 0 ]; do
    case "$1" in
      --install|--update|--reinstall|--uninstall) vAction="$1" ;;
      --no-ui) vUi="OFF" ;;
      --jobs) shift; vJobs="${1:-}" ;;
      --source) shift; vSource="${1:-}" ;;
      *) fUsage; return 1 ;;
    esac
    shift
  done
  if [ -z "$vAction" ]; then
    fUsage
    return 1
  fi
  if [ "$(id -u)" -ne 0 ]; then
    echo "run it as root"
    return 1
  fi
  touch "$cLog"
  chmod 600 "$cLog"
  fSay "llama.cpp-xdna: $vAction"

  if [ "$vAction" = "--uninstall" ]; then
    fUninstall
    return $?
  fi
  if [ "$vAction" = "--reinstall" ]; then
    fUninstall || return 1
  fi

  fSay "installing the dependencies of the build"
  if ! (apt-get -y update && apt-get -y install git build-essential cmake libssl-dev curl xz-utils ca-certificates \
    python3-numpy python3-yaml python3-tqdm) >> "$cLog" 2>&1; then
    echo "apt-get failed: see $cLog"
    return 1
  fi

  vWork="$(mktemp -d /tmp/llama.cpp-xdna-build.XXXXXX)"
  if [ -n "$vSource" ]; then
    fSay "copying the sources from $vSource"
    cp -a "$vSource" "$vWork/src" || return 1
  else
    fSay "downloading the sources from $cRepository"
    if ! git clone --depth 1 "$cRepository" "$vWork/src" >> "$cLog" 2>&1; then
      echo "git clone failed: see $cLog"
      return 1
    fi
  fi

  if [ "$vUi" = "ON" ]; then
    # The web UI is built with npm and needs Node.js 22 or later; Debian 13 has 20. The official Node.js 22 of
    # nodejs.org, checked against its SHA-256 sum, goes into the folder of the build and is removed with it.
    fSay "downloading Node.js 22 for the web UI"
    local vNode
    if ! (cd "$vWork" && curl -fsSLO https://nodejs.org/dist/latest-v22.x/SHASUMS256.txt) >> "$cLog" 2>&1; then
      echo "the download of Node.js failed: see $cLog"
      return 1
    fi
    vNode="$(grep -o 'node-v22[^ ]*-linux-x64\.tar\.xz' "$vWork/SHASUMS256.txt" | head -n 1)"
    if [ -z "$vNode" ] || ! (cd "$vWork" && curl -fsSLO "https://nodejs.org/dist/latest-v22.x/$vNode" &&
      grep " $vNode\$" SHASUMS256.txt | sha256sum -c - && tar -xJf "$vNode") >> "$cLog" 2>&1; then
      echo "Node.js 22 could not be downloaded or checked: see $cLog"
      return 1
    fi
    export PATH="$vWork/${vNode%.tar.xz}/bin:$PATH"
  fi

  fSay "building (this takes a while)"
  if ! (cd "$vWork/src" && cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX="$cPrefix" \
    -DGGML_NATIVE=OFF -DGGML_BACKEND_DL=ON -DGGML_CPU_ALL_VARIANTS=ON -DGGML_RPC=ON -DGGML_XDNA=ON \
    -DLLAMA_BUILD_UI="$vUi" -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_TESTS_INSTALL=OFF) >> "$cLog" 2>&1; then
    echo "cmake failed: see $cLog"
    return 1
  fi
  if ! (cd "$vWork/src" && NODE_OPTIONS=--max-old-space-size=4096 cmake --build build -j "$vJobs") >> "$cLog" 2>&1; then
    echo "the build failed: see $cLog"
    return 1
  fi

  fSay "installing in $cPrefix"
  if ! (cd "$vWork/src" && cmake --install build) >> "$cLog" 2>&1; then
    echo "cmake --install failed: see $cLog"
    return 1
  fi
  mkdir -p "$(dirname "$cManifest")" || return 1
  cp "$vWork/src/build/install_manifest.txt" "$cManifest" || return 1
  ldconfig
  fSay "installed: llama-cli, llama-server, llama-bench and the rest are in $cPrefix/bin"
  return 0
}

if ! fMain "$@"; then
  echo "error: llama.cpp-xdna was not installed (see $cLog)"
  exit 1
fi
