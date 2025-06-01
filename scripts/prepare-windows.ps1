$ErrorActionPreference='Stop'

param (
  [string]$target = "all"
)

function DelIfExist {
  param([string]$path)
  if (Test-Path $path) {
    Remove-Item $path -Recurse -Force
  }
}

New-Item -ItemType Directory -Force -Path "externals" | Out-Null

cd "externals"

if (-not (Test-Path "win32-x64/Vulkan-SDK") -or -not (Test-Path "win32-arm64/Vulkan-SDK")) {
  if (-not (Test-Path "Vulkan-Headers-source")) {
    git clone "https://github.com/KhronosGroup/Vulkan-Headers.git" "Vulkan-Headers-source"
    cd "Vulkan-Headers-source"
    git checkout sdk-1.3.261
    cd ".."
  }
  if (-not (Test-Path "Vulkan-Loader-source")) {
    git clone "https://github.com/KhronosGroup/Vulkan-Loader.git" "Vulkan-Loader-source"
    cd "Vulkan-Loader-source"
    git checkout sdk-1.3.261
    cd ".."
  }
  if (-not (Test-Path "Vulkan-Hpp-source")) {
    git clone "https://github.com/KhronosGroup/Vulkan-Hpp.git" "Vulkan-Hpp-source"
    cd "Vulkan-Hpp-source"
    git checkout v1.3.261
    git submodule update --init --recursive
    cd ".."
  }
}

# x64

if (-not (Test-Path "win32-x64/Vulkan-SDK") -and ($target -eq "all" -or $target -eq "x64")) {
  New-Item -ItemType Directory -Force -Path "win32-x64/Vulkan-SDK" | Out-Null
  $VULKAN_ROOT = (Resolve-Path "win32-x64/Vulkan-SDK").Path
  # build from source
  cd "Vulkan-Headers-source"
  DelIfExist "build"
  cmake -B build -G "Visual Studio 17 2022" -A x64
  cmake --build build --config Release
  cmake --install build --prefix $VULKAN_ROOT
  cd "../Vulkan-Loader-source"
  DelIfExist "build"
  cmake -B build -G "Visual Studio 17 2022" -DVULKAN_HEADERS_INSTALL_DIR="$VULKAN_ROOT" -DUSE_MASM=OFF -A x64
  cmake --build build --config Release
  cmake --install build --prefix $VULKAN_ROOT
  cd "../Vulkan-Hpp-source"
  DelIfExist "build"
  cmake -B build -G "Visual Studio 17 2022" -DVULKAN_HPP_INSTALL=ON -DVULKAN_HPP_RUN_GENERATOR=ON
  cmake --build build --config Release
  cmake --install build --prefix $VULKAN_ROOT
  cd ".."
}

# ARM64

if (-not (Test-Path "win32-arm64/Vulkan-SDK") -and ($target -eq "all" -or $target -eq "arm64")) {
  New-Item -ItemType Directory -Force -Path "win32-arm64/Vulkan-SDK" | Out-Null
  $VULKAN_ROOT = (Resolve-Path "win32-arm64/Vulkan-SDK").Path
  # build from source
  cd "Vulkan-Headers-source"
  DelIfExist "build"
  cmake -B build -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $VULKAN_ROOT
  cd "../Vulkan-Loader-source"
  DelIfExist "build"
  cmake -B build -G "Visual Studio 17 2022" -DVULKAN_HEADERS_INSTALL_DIR="$VULKAN_ROOT" -DUSE_MASM=OFF -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $VULKAN_ROOT
  cd "../Vulkan-Hpp-source"
  DelIfExist "build"
  cmake -B build -G "Visual Studio 17 2022" -DVULKAN_HPP_INSTALL=ON -DVULKAN_HPP_RUN_GENERATOR=ON
  cmake --build build --config Release
  cmake --install build --prefix $VULKAN_ROOT
  cd ".."
}
