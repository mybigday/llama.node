$ErrorActionPreference='Stop'

$x64Args = "--CDCMAKE_TOOLCHAIN_FILE=cmake/x86_64-windows-msvc-clang.toolchain.cmake"

$arm64Args = "--CDCMAKE_TOOLCHAIN_FILE=cmake/arm64-windows-msvc-clang.toolchain.cmake"

# General

yarn clean ; yarn build-native $x64Args -a x86_64
yarn clean ; yarn build-native $arm64Args -a arm64

# Vulkan, might crash on some scenario

yarn clean ; yarn build-native $x64Args -a x86_64 --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-x64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
yarn clean ; yarn build-native $arm64Args -a arm64 --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-arm64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
