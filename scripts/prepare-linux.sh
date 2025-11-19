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
  
  # Install cross-compilation sysroot for aarch64 if on x86_64
  if [ $(uname -m) == "x86_64" ] && [ $ARCH == "arm64" ]; then
    echo "Installing aarch64 sysroot for cross-compilation..."
    run_as_root dpkg --add-architecture arm64
    run_as_root apt-get update
    run_as_root apt-get install -qy libc6-dev:arm64
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
