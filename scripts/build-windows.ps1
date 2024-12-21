$ErrorActionPreference='Stop'

# General

yarn clean ; yarn build-native -a x86_64
yarn clean ; yarn build-native -a arm64 --CDCMAKE_C_COMPILER=clang --CDCMAKE_CXX_COMPILER=clang++ 

# Vulkan, might crash on some scenario

yarn clean ; yarn build-native -a x86_64 --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-x64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
yarn clean ; yarn build-native -a arm64 --CDCMAKE_C_COMPILER=clang --CDCMAKE_CXX_COMPILER=clang++ --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-arm64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
