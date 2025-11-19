#!/bin/bash

set -e

# General

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


if [ $ARCH == "x86_64" ]; then
  # default
  if [ $TARGET == "all" ] || [ $TARGET == "default" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON --CDCLANG_USE_GOMP=ON
  fi

  # vulkan
  if [ $TARGET == "all" ] || [ $TARGET == "vulkan" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON  --CDCLANG_USE_GOMP=ON --CDGGML_VULKAN=1 --CDVARIANT=vulkan
  fi

  # cuda
  if [ $TARGET == "all" ] || [ $TARGET == "cuda" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_CUDA=1 \
      --CDVARIANT=cuda \
      --CDCMAKE_CUDA_ARCHITECTURES=89 # > GeForce RTX 40 series
  fi
else
  # default
  if [ $TARGET == "all" ] || [ $TARGET == "default" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON  --CDCLANG_USE_GOMP=ON --CDGGML_NATIVE=OFF
  fi

  # vulkan
  if [ $TARGET == "all" ] || [ $TARGET == "vulkan" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_NATIVE=OFF \
      --CDGGML_VULKAN=1 \
      --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')" \
      --CDVARIANT=vulkan
  fi

  # cuda
  if [ $TARGET == "all" ] || [ $TARGET == "cuda" ]; then
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_CUDA=1 \
      --CDVARIANT=cuda \
      --CDGGML_NATIVE=OFF \
      --CDCMAKE_CUDA_ARCHITECTURES=87 # > Jetson Orin series
  fi

  # snapdragon
  if [ $TARGET == "all" ] || [ $TARGET == "snapdragon" ]; then
    source externals/Hexagon_SDK/Hexagon_SDK/6.4.0.2/setup_sdk_env.source
    npx cmake-js rebuild -C --CDTO_PACKAGE=ON \
      --CDCLANG_USE_GOMP=ON \
      --CDGGML_NATIVE=OFF \
      --CDGGML_OPENMP=0 \
      --CDGGML_OPENCL=1 \
      --CDGGML_HEXAGON=1 \
      --CDHEXAGON_SDK_ROOT="$HEXAGON_SDK_ROOT" \
      --CDPREBUILT_LIB_DIR=UbuntuARM_aarch64 \
      --CDVARIANT=snapdragon
  fi
fi
