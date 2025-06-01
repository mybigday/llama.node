$ErrorActionPreference='Stop'

param (
  [string]$target = "all"
)

function build-windows {
  param (
    $arch,
    $args = @()
  )

  if ($arch -eq "x86_64") {
    $prefix = "x86_64-w64-mingw32"
  } elseif ($arch -eq "arm64") {
    $prefix = "aarch64-w64-mingw32"
  }

  if (Get-Command "clang" -ErrorAction SilentlyContinue) {
    $args += "-DCMAKE_TOOLCHAIN_FILE=cmake/${arch}-windows-clang.toolchain.cmake"
  } elseif (Get-Command "${prefix}-clang" -ErrorAction SilentlyContinue) {
    $args += "-DCMAKE_TOOLCHAIN_FILE=cmake/${prefix}-clang.toolchain.cmake"
  }

  yarn clean ; yarn build-native -C -a $arch @args
}

# General

if ($target -eq "all" -or $target -eq "x86_64") {
  build-windows x86_64
}
if ($target -eq "all" -or $target -eq "arm64") {
  build-windows arm64
}

# Vulkan, might crash on some scenario

if ($target -eq "all" -or $target -eq "x86_64") {
  build-windows x86_64 --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-x64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
}
if ($target -eq "all" -or $target -eq "arm64") {
  build-windows arm64 --CDVULKAN_SDK="$(Resolve-Path 'externals/win32-arm64/Vulkan-SDK')" --CDVARIANT=vulkan --CDLLAMA_VULKAN=1
}
