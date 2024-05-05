#!/bin/bash

set -e

# General

yarn clean && yarn build-native -a x86_64 --CDLLAMA_VULKAN=1
yarn clean && yarn build-native -a aarch64 --cc aarch64-linux-gnu-gcc --cxx aarch64-linux-gnu-g++

# build with CLBlast
yarn clean && yarn build-native -a aarch64 --cc aarch64-linux-gnu-gcc --cxx aarch64-linux-gnu-g++ \
  --CDVARIANT=opencl --CDLLAMA_CLBLAST=ON --CDCMAKE_PREFIX_PATH="$(realpath 'externals/SDK/lib/cmake/CLBlast')"
