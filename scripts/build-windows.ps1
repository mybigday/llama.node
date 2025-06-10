param (
  [string]$target = "all",
  [string]$toolchain = "clang-cl"
)

$ErrorActionPreference='Stop'

if ($toolchain -eq "clang-cl") {
  $x86Args = "--CDCMAKE_TOOLCHAIN_FILE=cmake/x86_64-windows-msvc-clang.toolchain.cmake"
  $arm64Args = "--CDCMAKE_TOOLCHAIN_FILE=cmake/arm64-windows-msvc-clang.toolchain.cmake"
} else {
  $x86Args = "--CDCMAKE_TOOLCHAIN_FILE=cmake/x86_64-w64-mingw32-clang.toolchain.cmake"
  $arm64Args = "--CDCMAKE_TOOLCHAIN_FILE=cmake/aarch64-w64-mingw32-clang.toolchain.cmake"
}

# General

if ($target -eq "all" -or $target -eq "x86_64") {
  npx cmake-js rebuild -C -a x86_64 --CDTO_PACKAGE=ON $x86Args
}
if ($target -eq "all" -or $target -eq "arm64") {
  npx cmake-js rebuild -C -a arm64 --CDTO_PACKAGE=ON $arm64Args
}

# Vulkan, might crash on some scenario

if ($target -eq "all" -or $target -eq "x86_64") {
  npx cmake-js rebuild -C -a x86_64 --CDTO_PACKAGE=ON $x86Args `
    --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-x64/Vulkan-SDK')" `
    --CDVARIANT=vulkan `
    --CDGGML_VULKAN=1
}
if ($target -eq "all" -or $target -eq "arm64") {
  npx cmake-js rebuild -C -a arm64 --CDTO_PACKAGE=ON $arm64Args `
    --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-arm64/Vulkan-SDK')" `
    --CDVARIANT=vulkan `
    --CDGGML_VULKAN=1
}

# CUDA

if ($target -eq "all" -or $target -eq "x86_64" -and (Get-Command -Name "nvcc" -ErrorAction SilentlyContinue)) {
  npx cmake-js rebuild -C -a x86_64 --CDTO_PACKAGE=ON $x86Args `
    --CDVARIANT=cuda `
    --CDGGML_CUDA=1 `
    --CDGGML_CUDA_F16=1 `
    --CDCMAKE_CUDA_ARCHITECTURES="75;80;86;89;120" # See: https://developer.nvidia.com/cuda-gpus
}
