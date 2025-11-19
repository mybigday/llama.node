param (
  [string]$arch = "native",
  [string]$target = "default",
  [string]$toolchain = "clang-cl"
)

$ErrorActionPreference='Stop'

# Enable parallel compilation
$env:CMAKE_BUILD_PARALLEL_LEVEL = [Environment]::ProcessorCount

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
    if ($arch -eq "x64" -and $target -eq "vulkan") {
      $cmakeArgs.add("--CDVulkan_LIBRARY:FILEPATH=$(Resolve-Path $env:VULKAN_SDK/Lib-X64/vulkan-1.lib)")
      $cmakeArgs.add("--CDVulkan_INCLUDE_DIR:PATH=$(Resolve-Path $env:VULKAN_SDK/Include)")
    }
  } else {
    $cmakeArgs.add("--CDGGML_VULKAN_SHADERS_GEN_TOOLCHAIN=$(Resolve-Path cmake/x86_64-w64-mingw32-clang.toolchain.cmake)")
    if ($arch -eq "arm64" -and $target -eq "vulkan") {
      $cmakeArgs.add("--CDVulkan_LIBRARY:FILEPATH=$(Resolve-Path $env:VULKAN_SDK/Lib-ARM64/vulkan-1.lib)")
      $cmakeArgs.add("--CDVulkan_INCLUDE_DIR:PATH=$(Resolve-Path $env:VULKAN_SDK/Include)")
    }
  }
} elseif ($toolchain -ne "cl") {
  throw "Unknown toolchain: $toolchain"
}

# General

if ($target -eq "all" -or $target -eq "default") {
  npx cmake-js rebuild -C -a $arch $cmakeArgs
  if ($LASTEXITCODE -ne 0) {
    throw "build failed"
  }
}

# Vulkan, might crash on some scenario

if ($target -eq "all" -or $target -eq "vulkan") {
  npx cmake-js rebuild -C -a $arch $cmakeArgs `
    --CDVULKAN_SDK="$env:VULKAN_SDK" `
    --CDVARIANT=vulkan `
    --CDGGML_VULKAN=1
  if ($LASTEXITCODE -ne 0) {
    throw "build failed"
  }
}

# CUDA

if ($target -eq "all" -or $target -eq "cuda") {
  npx cmake-js rebuild -C -a $arch $cmakeArgs `
    --CDVARIANT=cuda `
    --CDGGML_CUDA=1 `
    --CDGGML_CUDA_F16=1 `
    --CDCMAKE_CUDA_ARCHITECTURES="86;89;120" # See: https://developer.nvidia.com/cuda-gpus
  if ($LASTEXITCODE -ne 0) {
    throw "build failed"
  }
}

# Qualcomm

if ($target -eq "all" -or $target -eq "qualcomm") {
  npx cmake-js rebuild -C -a $arch $cmakeArgs `
    --CDVARIANT=qualcomm `
    --CDGGML_OPENCL=1 `
    --CDGGML_HEXAGON=1 `
    --CDHEXAGON_SDK_ROOT="$(Resolve-Path 'externals/Hexagon_SDK/Hexagon_SDK/6.4.0.2')"
  if ($LASTEXITCODE -ne 0) {
    throw "build failed"
  }
}
