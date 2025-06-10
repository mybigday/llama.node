#!/bin/bash

set -e

function run_as_root() {
  if [ $UID -ne 0 ]; then
    sudo -E $@
  else
    $@
  fi
}

export DEBIAN_FRONTEND=noninteractive

ARCH=${ARCH:-$(uname -m)}
TARGET=${TARGET:-"default"}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -a|--arch) ARCH="$2"; shift ;;
    -t|--target) TARGET="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

run_as_root apt-get update
run_as_root apt-get install -qy lsb-release wget llvm clang lld cmake ninja-build libomp-dev ccache

if [ "$TARGET" == "vulkan" ] && ! command -v glslc &> /dev/null; then
  if ! run_as_root apt-get install -qy shaderc; then
    # build from source
    # SPIRV-Tools
    git clone https://github.com/KhronosGroup/SPIRV-Tools.git --depth 1 --branch vulkan-sdk-1.4.313
    pushd SPIRV-Tools
    git clone https://github.com/KhronosGroup/SPIRV-Headers.git --depth 1 --branch vulkan-sdk-1.4.313 external/spirv-headers
    cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DSPIRV_SKIP_TESTS=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    cmake --build build
    run_as_root cmake --install build
    popd
    # shaderc
    git clone https://github.com/google/shaderc.git --depth 1
    pushd shaderc
    cmake -B build -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/usr \
      -DSHADERC_SKIP_TESTS=ON \
      -DSHADERC_SKIP_EXAMPLES=ON \
      -DCMAKE_C_COMPILER_LAUNCHER=ccache \
      -DCMAKE_CXX_COMPILER_LAUNCHER=ccache
    cmake --build build
    run_as_root cmake --install build
    popd
  fi
fi
