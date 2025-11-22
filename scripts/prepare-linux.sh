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

OPENCL_VERSION=${OPENCL_VERSION:-"2024.10.24"}
HEXAGON_SDK_VERSION=${HEXAGON_SDK_VERSION:-"6.4.0.2"}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -a|--arch) ARCH="$2"; shift ;;
    -t|--target) TARGET="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

run_as_root apt-get update

if [ $TARGET == "snapdragon" ]; then
  run_as_root apt-get install -qy lsb-release wget llvm clang lld cmake ninja-build libomp-dev ccache unzip gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu

  if [ ! -f "externals/opencl-arm64/lib/libOpenCL.so" ]; then
    mkdir -p externals/opencl-arm64
    git clone --depth 1 --branch v$OPENCL_VERSION https://github.com/KhronosGroup/OpenCL-Headers.git externals/OpenCL-Headers
    git clone --depth 1 --branch v$OPENCL_VERSION https://github.com/KhronosGroup/OpenCL-ICD-Loader.git externals/OpenCL-ICD-Loader
    
    # Calculate absolute paths before changing directory
    TOOLCHAIN_FILE="$(realpath cmake/aarch64-linux-gnu.toolchain.cmake)"
    HEADERS_DIR="$(realpath externals/OpenCL-Headers)"
    INSTALL_DIR="$(realpath externals/opencl-arm64)"
    
    cd externals/OpenCL-ICD-Loader
    cmake -S . -B build-arm64 \
      -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN_FILE" \
      -DOPENCL_ICD_LOADER_HEADERS_DIR="$HEADERS_DIR" \
      -DCMAKE_INSTALL_PREFIX="$INSTALL_DIR" \
      -DCMAKE_BUILD_TYPE=Release
    cmake --build build-arm64
    cmake --install build-arm64
    cd ../..
  fi
  
  # Download and extract Hexagon SDK
  if [ ! -d "externals/Hexagon_SDK" ]; then
    echo "Downloading Hexagon SDK..."
    mkdir -p externals
    wget -O externals/Hexagon_SDK_lnx.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/$HEXAGON_SDK_VERSION/Hexagon_SDK_lnx.zip
    echo "Extracting Hexagon SDK..."
    unzip -q externals/Hexagon_SDK_lnx.zip -d externals/Hexagon_SDK
  fi
  
  source externals/Hexagon_SDK/Hexagon_SDK/$HEXAGON_SDK_VERSION/setup_sdk_env.source
else
  run_as_root apt-get install -qy lsb-release wget llvm clang lld cmake ninja-build libomp-dev ccache
fi
