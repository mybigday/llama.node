#!/bin/bash

set -e

# General

ARCH=${ARCH:-${1:-$(uname -m)}}

if [ $ARCH == "x86_64" ]; then
  yarn clean && yarn build-native
  yarn clean && yarn build-native --CDLLAMA_VULKAN=1 --CDVARIANT=vulkan
else
  yarn clean && yarn build-native
  yarn clean && yarn build-native --CDLLAMA_VULKAN=1 --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')" --CDVARIANT=vulkan
fi
