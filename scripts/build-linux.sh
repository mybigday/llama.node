#!/bin/bash

set -e

# General

if [ $(uname -m) == "x86_64" ]; then
  yarn clean && yarn build-native --CDLLAMA_VULKAN=1
else
  yarn clean && yarn build-native --CDLLAMA_VULKAN=1 --CDVULKAN_SDK="$(realpath 'externals/arm64-Vulkan-SDK')"
fi
