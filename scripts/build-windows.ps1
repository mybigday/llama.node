$ErrorActionPreference='Stop'

# General

yarn clean ; yarn build-native -C -a x86_64
yarn clean ; yarn build-native -C -a arm64

# Vulkan, might crash on some scenario

yarn clean ; yarn build-native -C -a x86_64 --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-x64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
yarn clean ; yarn build-native -C -a arm64 --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-arm64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
