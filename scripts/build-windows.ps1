$ErrorActionPreference='Stop'

# General

yarn clean ; yarn build-native -a x86_64
yarn clean ; yarn build-native -a arm64

# build with CLBlast
$X64_CLBlast=Resolve-Path 'externals/win32-x64/SDK/lib/cmake/CLBlast'
$ARM64_CLBlast=Resolve-Path 'externals/win32-arm64/SDK/lib/cmake/CLBlast'

yarn clean ; yarn build-native -a x86_64 --CDVARIANT=opencl --CDLLAMA_CLBLAST=ON --CDCMAKE_PREFIX_PATH=$X64_CLBlast
yarn clean ; yarn build-native -a arm64 --CDVARIANT=opencl --CDLLAMA_CLBLAST=ON --CDCMAKE_PREFIX_PATH=$ARM64_CLBlast
