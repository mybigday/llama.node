#!/bin/bash

set -e

# General

ARCH=${ARCH:-${1:-$(uname -m)}}

if [ $ARCH == "x86_64" ]; then
  yarn clean && yarn build-native
  yarn clean && yarn build-native --CDLLAMA_VULKAN=1 --CDVARIANT=vulkan

  # Check CUDA is available
  if [ -f /usr/local/cuda/bin/nvcc ]; then
    yarn clean && yarn build-native \
      --CDLLAMA_CUDA=1 \
      --CDVARIANT=cuda \
      --CDCMAKE_CUDA_ARCHITECTURES=89 # > GeForce RTX 40 series
  else
    echo "CUDA is not available, skipping CUDA build"
  fi
else
  yarn clean && yarn build-native
  yarn clean && yarn build-native --CDLLAMA_VULKAN=1 --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')" --CDVARIANT=vulkan

  # Check CUDA is available
  if [ -f /usr/local/cuda/bin/nvcc ]; then
    yarn clean && yarn build-native \
      --CDLLAMA_CUDA=1 \
      --CDVARIANT=cuda \
      --CDCMAKE_CUDA_ARCHITECTURES=87 # > Jetson Orin series
  else
    echo "CUDA is not available, skipping CUDA build"
  fi
fi
