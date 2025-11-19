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

if [ $TARGET == "snapdragon" ]; then
  run_as_root apt-get install -qy lsb-release wget llvm clang lld cmake ninja-build libomp-dev ccache ocl-icd-opencl-dev opencl-headers clinfo unzip
  
  # Install cross-compilation toolchain for aarch64 if on x86_64
  if [ $(uname -m) == "x86_64" ] && [ $ARCH == "arm64" ]; then
    echo "Installing aarch64 cross-compilation toolchain..."
    run_as_root apt-get install -qy gcc-aarch64-linux-gnu g++-aarch64-linux-gnu binutils-aarch64-linux-gnu
    
    # Enable arm64 architecture for OpenCL libraries only
    echo "Installing aarch64 OpenCL libraries..."
    run_as_root dpkg --add-architecture arm64
    run_as_root apt-get update || true  # Continue even if some repos fail
    run_as_root apt-get install -qy ocl-icd-libopencl1:arm64 || {
      echo "Warning: Could not install arm64 OpenCL library from repos, building from source..."
      # Build OpenCL ICD Loader for arm64
      if [ ! -f "externals/opencl-arm64/lib/libOpenCL.so" ]; then
        mkdir -p externals/opencl-arm64
        git clone --depth 1 --branch v2024.05.08 https://github.com/KhronosGroup/OpenCL-Headers.git externals/OpenCL-Headers
        git clone --depth 1 --branch v2024.05.08 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git externals/OpenCL-ICD-Loader
        
        cd externals/OpenCL-ICD-Loader
        cmake -S . -B build-arm64 \
          -DCMAKE_TOOLCHAIN_FILE="$(realpath ../../cmake/aarch64-linux-gnu.toolchain.cmake)" \
          -DOPENCL_ICD_LOADER_HEADERS_DIR="$(realpath ../OpenCL-Headers)" \
          -DCMAKE_INSTALL_PREFIX="$(realpath ../opencl-arm64)" \
          -DCMAKE_BUILD_TYPE=Release
        cmake --build build-arm64
        cmake --install build-arm64
        cd ../..
      fi
    }
  fi
  
  # Download and extract Hexagon SDK
  if [ ! -d "externals/Hexagon_SDK" ]; then
    echo "Downloading Hexagon SDK..."
    mkdir -p externals
    wget -O externals/Hexagon_SDK_lnx.zip https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Linux/Debian/6.4.0.2/Hexagon_SDK_lnx.zip
    echo "Extracting Hexagon SDK..."
    unzip -q externals/Hexagon_SDK_lnx.zip -d externals/Hexagon_SDK
  fi
  
  source externals/Hexagon_SDK/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
else
  run_as_root apt-get install -qy lsb-release wget llvm clang lld cmake ninja-build libomp-dev ccache
fi
