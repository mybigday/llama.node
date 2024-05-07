$ErrorActionPreference='Stop'

# General

yarn clean ; yarn build-native -a x86_64 --CDCMAKE_PREFIX_PATH=(Resolve-Path 'externals/win32-x64/Vulkan-SDK')
yarn clean ; yarn build-native -a arm64 --CDCMAKE_PREFIX_PATH=(Resolve-Path 'externals/win32-arm64/Vulkan-SDK')
