#!/bin/sh

# llama.cpp-xdna - installer for Alpine Linux.
#
# WHY THIS ONE IS NOT A BASH SCRIPT, when everything else in this project is: a fresh Alpine has no bash, so with
# #!/bin/bash the one-line install could not start at all. This file is POSIX shell, which runs under the ash of
# BusyBox, under dash and under bash: no arrays, no [[ ]], no ${var//x/y}. `local` is not POSIX, but the three have it.
#
# Builds llama.cpp-xdna from its sources (the latest of its GitHub repository, or a local copy with --source) with
# the backends CPU, RPC and XDNA and the web UI of llama-server, and installs it: the programs, the backends
# (libggml-*.so) and the overlays of the NPU in /usr/local/bin, the libraries in /usr/local/lib. Running it again builds
# the latest sources and installs them over the old ones; --reinstall removes the old ones first. Run it as root.
#
# Use:
#   wget -qO- https://raw.githubusercontent.com/nipegun/llama.cpp-xdna/main/deploy/install-update-reinstall-alpine.sh \
#     | sh -s -- --install
#   sh install-update-reinstall-alpine.sh --install | --update | --reinstall | --uninstall [options]
#
# Options:
#   --no-ui           without the web UI of llama-server (then it serves only its API; Node.js is not needed)
#   --jobs <n>        parallel jobs of the build (default: the processors of the computer)
#   --source <folder> build from a local copy of the sources instead of downloading them
#
# The web UI needs Node.js 22 or later, which Alpine 3.21 and later have (the official builds of nodejs.org are for
# glibc). What it does is logged in /root/llama.cpp-xdna-install.log.

set -eu

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
  echo "use: sh install-update-reinstall-alpine.sh --install | --update | --reinstall | --uninstall"
  echo "       [--no-ui] [--jobs <n>] [--source <folder>]"
}

# the files of the installation before this one, as cmake --install wrote them
fUninstall() {
  if [ ! -f "$cManifest" ]; then
    fSay "there is no installation of llama.cpp-xdna to remove"
    return 0
  fi
  fSay "removing the files of the installation before"
  while IFS= read -r vFile; do
    if [ -n "$vFile" ] && [ -f "$vFile" ]; then
      rm -f "$vFile"
    fi
  done < "$cManifest"
  rm -f "$cManifest"
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
  local vPackages="git build-base cmake openssl-dev curl xz linux-headers ca-certificates python3 py3-numpy py3-yaml py3-tqdm"
  if [ "$vUi" = "ON" ]; then
    vPackages="$vPackages nodejs npm"
  fi
  # shellcheck disable=SC2086
  if ! apk add --no-cache $vPackages >> "$cLog" 2>&1; then
    echo "apk failed: see $cLog"
    return 1
  fi
  if [ "$vUi" = "ON" ]; then
    local vMajor
    vMajor="$(node --version 2>/dev/null | sed -E 's/^v([0-9]+).*/\1/')"
    if [ -z "$vMajor" ] || [ "$vMajor" -lt 22 ]; then
      echo "the web UI needs Node.js 22 or later, and this Alpine has $(node --version 2>/dev/null): use Alpine 3.21 or later, or --no-ui"
      return 1
    fi
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
  fSay "installed: llama-cli, llama-server, llama-bench and the rest are in $cPrefix/bin"
  return 0
}

if ! fMain "$@"; then
  echo "error: llama.cpp-xdna was not installed (see $cLog)"
  exit 1
fi
