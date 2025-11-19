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

# install CUDA
if (($arch -eq "all" -or $arch -eq "x64") -and ($target -eq "all" -or $target -eq "cuda")) {
  if (-Not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc.exe not found, please install CUDA"
  }
}

if (-Not (Get-Command ccache -ErrorAction SilentlyContinue)) {
  choco install ccache -y
}

# Configure ccache for optimal performance
if (Get-Command ccache -ErrorAction SilentlyContinue) {
  ccache --set-config=max_size=5G
  ccache --set-config=compression=true
  ccache --set-config=compiler_check=content
  ccache -z  # Zero statistics
}

if ($toolchain -eq "mingw-clang") {
  $version = "20250528"
  if ($nativeArch -eq "X64") {
    $name = "llvm-mingw-${version}-ucrt-x86_64"
  } elseif ($nativeArch -eq "Arm64") {
    $name = "llvm-mingw-${version}-ucrt-aarch64"
  }
  Invoke-WebRequest -Uri "https://github.com/mstorsjo/llvm-mingw/releases/download/${version}/${name}.zip" -OutFile "llvm-mingw.zip"
  Expand-Archive -Path "llvm-mingw.zip" -DestinationPath .
  $env:Path += ";$(Resolve-Path $name\bin)"
  if ($env:GITHUB_PATH -ne $null) {
    Add-Content -Path $env:GITHUB_PATH -Value "$(Resolve-Path $name\bin)"
  }

  choco install ninja -y
}

if ($target -eq "snapdragon") {
  # Download Hexagon SDK
  $sdkPath = "externals/Hexagon_SDK"
  if (-Not (Test-Path $sdkPath)) {
    Write-Host "Downloading Hexagon SDK..."
    New-Item -ItemType Directory -Force -Path "externals" | Out-Null
    Invoke-WebRequest -Uri "https://softwarecenter.qualcomm.com/api/download/software/sdks/Hexagon_SDK/Windows/6.4.0.2/Hexagon_SDK_WinNT.zip" -OutFile "externals/Hexagon_SDK_WinNT.zip"
    Write-Host "Extracting Hexagon SDK..."
    Expand-Archive -Path "externals/Hexagon_SDK_WinNT.zip" -DestinationPath "externals/Hexagon_SDK" -Force
  }

  . "externals/Hexagon_SDK/Hexagon_SDK/6.4.0.2/setup_sdk_env.ps1"

  # Download OpenCL SDK
  $openclPath = "externals/OpenCL-SDK"
  if (-Not (Test-Path $openclPath)) {
    Write-Host "Downloading OpenCL SDK..."
    New-Item -ItemType Directory -Force -Path "externals" | Out-Null
    New-Item -ItemType Directory -Force -Path "externals/OpenCL-SDK" | Out-Null
    
    # Clone OpenCL-Headers
    git clone --depth 1 --branch v2024.10.24 https://github.com/KhronosGroup/OpenCL-Headers.git externals/OpenCL-Headers
    
    # Clone OpenCL-ICD-Loader
    git clone --depth 1 --branch v2024.10.24 https://github.com/KhronosGroup/OpenCL-ICD-Loader.git externals/OpenCL-ICD-Loader
    
    # Build OpenCL-ICD-Loader
    Write-Host "Building OpenCL ICD Loader..."
    cmake -S externals/OpenCL-ICD-Loader -B externals/OpenCL-ICD-Loader/build `
      -DOPENCL_ICD_LOADER_HEADERS_DIR="$(Resolve-Path 'externals/OpenCL-Headers')" `
      -DCMAKE_INSTALL_PREFIX="$(Resolve-Path 'externals/OpenCL-SDK')"
    cmake --build externals/OpenCL-ICD-Loader/build --config Release
    cmake --install externals/OpenCL-ICD-Loader/build --config Release
  }

  $env:OpenCL_INCLUDE_DIR = "$(Resolve-Path 'externals/OpenCL-Headers')"
  $env:OpenCL_LIBRARY = "$(Resolve-Path 'externals/OpenCL-SDK/lib/OpenCL.lib')"
  if ($env:GITHUB_ENV -ne $null) {
    Add-Content -Path $env:GITHUB_ENV -Value "OpenCL_INCLUDE_DIR=$env:OpenCL_INCLUDE_DIR"
    Add-Content -Path $env:GITHUB_ENV -Value "OpenCL_LIBRARY=$env:OpenCL_LIBRARY"
  }
}

