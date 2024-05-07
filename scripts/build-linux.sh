#!/bin/bash

set -e

# General

yarn clean && yarn build-native -a x86_64 --CDLLAMA_VULKAN=1
yarn clean && yarn build-native -a aarch64 --cc aarch64-linux-gnu-gcc --cxx aarch64-linux-gnu-g++ \
  --CDLLAMA_VULKAN=1 --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')"
