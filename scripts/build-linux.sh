#!/bin/bash

set -e

# General

ARCH=${ARCH:-${1:-$(uname -m)}}

if [ $ARCH == "x86_64" ]; then
  npx cmake-js rebuild -C --CDTO_PACKAGE=ON
  npx cmake-js rebuild -C --CDTO_PACKAGE=ON --CDLLAMA_VULKAN=1 --CDVARIANT=vulkan

  # Check CUDA is available
  if [ -f /usr/local/cuda/bin/nvcc ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDLLAMA_CUDA=1 \
      --CDVARIANT=cuda \
      --CDCMAKE_CUDA_ARCHITECTURES=89 # > GeForce RTX 40 series
  else
    echo "CUDA is not available, skipping CUDA build"
  fi
else
  npx cmake-js rebuild -C --CDTO_PACKAGE=ON --CDGGML_NATIVE=OFF
  npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
    --CDGGML_NATIVE=OFF \
    --CDLLAMA_VULKAN=1 \
    --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')" \
    --CDVARIANT=vulkan

  # Check CUDA is available
  if [ -f /usr/local/cuda/bin/nvcc ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDLLAMA_CUDA=1 \
      --CDVARIANT=cuda \
      --CDGGML_NATIVE=OFF \
      --CDCMAKE_CUDA_ARCHITECTURES=87 # > Jetson Orin series
  else
    echo "CUDA is not available, skipping CUDA build"
  fi
fi
