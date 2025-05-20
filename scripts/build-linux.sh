#!/bin/bash

set -e

# General

ARCH=${ARCH:-${1:-$(uname -m)}}
EXTRA_CMAKE_FLAGS=${EXTRA_CMAKE_FLAGS:-""}
VARIANT_PREFIX=${VARIANT_PREFIX:-""}

if [ $ARCH == "x86_64" ]; then
  yarn clean && yarn build-native -C
  yarn clean && yarn build-native -C --CDLLAMA_VULKAN=1 --CDVARIANT=vulkan

  # Check CUDA is available
  if [ -f /usr/local/cuda/bin/nvcc ]; then
    yarn clean && yarn build-native -C \
      --CDLLAMA_CUDA=1 \
      --CDVARIANT=cuda \
      --CDCMAKE_CUDA_ARCHITECTURES=89 # > GeForce RTX 40 series
  else
    echo "CUDA is not available, skipping CUDA build"
  fi
else
  if which aarch64-unknown-linux-gnu2.27-gcc > /dev/null && [ -z "$EXTRA_CMAKE_FLAGS" ]; then
    echo "Using obggcc toolchain"
    VARIANT_PREFIX="legacy-" \
    EXTRA_CMAKE_FLAGS="--CDCMAKE_TOOLCHAIN_FILE=$(realpath 'cmake/aarch64-obggcc-toolchain.cmake')" \
      $0
    exit 0
  fi

  if [ $(uname -m) != "aarch64" ]; then
    EXTRA_CMAKE_FLAGS="--CDCMAKE_TOOLCHAIN_FILE=$(realpath 'cmake/aarch64-clang-toolchain.cmake')"
  fi

  yarn clean && yarn build-native -C -a aarch64 --CDGGML_NATIVE=OFF --CDVARIANT=${VARIANT_PREFIX%"-"} $EXTRA_CMAKE_FLAGS

  if [ -z "$VARIANT_PREFIX" ]; then
    yarn clean && yarn build-native -C -a aarch64 --CDGGML_NATIVE=OFF --CDLLAMA_VULKAN=1 --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')" --CDVARIANT=vulkan $EXTRA_CMAKE_FLAGS
  fi

  # Check CUDA is available
  if [ -f /usr/local/cuda/bin/nvcc ]; then
    yarn clean && yarn build-native -C -a aarch64 \
      --CDLLAMA_CUDA=1 \
      --CDVARIANT=${VARIANT_PREFIX}cuda \
      --CDGGML_NATIVE=OFF \
      $EXTRA_CMAKE_FLAGS \
      --CDCMAKE_CUDA_ARCHITECTURES=87 # > Jetson Orin series
  else
    echo "CUDA is not available, skipping CUDA build"
  fi
fi
