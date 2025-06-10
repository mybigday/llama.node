param (
  [string]$arch = "native",
  [string]$target = "default",
  [string]$toolchain = "clang-cl"
)

$ErrorActionPreference='Stop'

$nativeArch = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture

if ($arch -eq "native") {
  if ($nativeArch -eq "Arm64") {
    $arch = "arm64"
  } else {
    $arch = "x64"
  }
}

$cmakeArgs = New-Object System.Collections.ArrayList
$cmakeArgs.add("--CDTO_PACKAGE=ON")

if ($toolchain -eq "clang-cl") {
  if ($arch -eq "x64") {
    $cmakeArgs.add("--CDCMAKE_TOOLCHAIN_FILE=$(Resolve-Path cmake/x86_64-windows-msvc-clang.toolchain.cmake)")
  } elseif ($arch -eq "arm64") {
    $cmakeArgs.add("--CDCMAKE_TOOLCHAIN_FILE=$(Resolve-Path cmake/arm64-windows-msvc-clang.toolchain.cmake)")
  }
} elseif ($toolchain -eq "mingw-clang") {
  if ((Get-Command aarch64-w64-mingw32-clang -ErrorAction SilentlyContinue) -eq $null) {
    throw "mingw32-clang toolchain is not available"
  }
  $cmakeArgs.add("-G")
  $cmakeArgs.add("Ninja")
  if ($arch -eq "x64") {
    $cmakeArgs.add("--CDCMAKE_TOOLCHAIN_FILE=$(Resolve-Path cmake/x86_64-w64-mingw32-clang.toolchain.cmake)")
  } elseif ($arch -eq "arm64") {
    $cmakeArgs.add("--CDCMAKE_TOOLCHAIN_FILE=$(Resolve-Path cmake/aarch64-w64-mingw32-clang.toolchain.cmake)")
  }
  if ($nativeArch -eq "Arm64") {
    $cmakeArgs.add("--CDGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=$(Resolve-Path cmake/aarch64-w64-mingw32-clang.toolchain.cmake)")
  } else {
    $cmakeArgs.add("--CDGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=$(Resolve-Path cmake/x86_64-w64-mingw32-clang.toolchain.cmake)")
  }
} else {
  throw "Unknown toolchain: $toolchain"
}

# General

if ($target -eq "all" -or $target -eq "default") {
  npx cmake-js rebuild -C -a $arch $cmakeArgs
}

# Vulkan, might crash on some scenario

if ($target -eq "all" -or $target -eq "vulkan") {
  npx cmake-js rebuild -C -a $arch $cmakeArgs `
    --CDVULKAN_SDK="$env:VULKAN_SDK" `
    --CDVARIANT=vulkan `
    --CDGGML_VULKAN=1
}

# CUDA

if ($target -eq "all" -or $target -eq "cuda") {
  npx cmake-js rebuild -C -a $arch $cmakeArgs `
    --CDVARIANT=cuda `
    --CDCMAKE_CUDA_COMPILER_LAUNCHER=ccache `
    --CDGGML_CUDA=1 `
    --CDGGML_CUDA_F16=1 `
    --CDCMAKE_CUDA_ARCHITECTURES="86;89;120" # See: https://developer.nvidia.com/cuda-gpus
}
