$ErrorActionPreference='Stop'

# General

yarn clean ; yarn build-native -a x86_64 -C
yarn clean ; yarn build-native -a arm64 -C

# Vulkan, might crash on some scenario

yarn clean ; yarn build-native -a x86_64 -C --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-x64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
yarn clean ; yarn build-native -a arm64 -C --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-arm64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
