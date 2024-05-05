$ErrorActionPreference='Stop'

function DelIfExist {
  param([string]$path)
  if (Test-Path $path) {
    Remove-Item $path -Recurse -Force
  }
}

New-Item -ItemType Directory -Force -Path "externals" | Out-Null

cd "externals"

if (-not (Test-Path "OpenCL-SDK-source")) {
  git clone "https://github.com/KhronosGroup/OpenCL-SDK.git" "OpenCL-SDK-source" --recursive
  cd "OpenCL-SDK-source"
  git checkout v2023.12.14
  cd ".."
}

if (-not (Test-Path "CLBlast-source")) {
  git clone "https://github.com/CNugteren/CLBlast.git" "CLBlast-source" --recursive
  cd "CLBlast-source"
  git checkout 1.6.2
  cd ".."
}

# x64

if (-not (Test-Path "win32-x64/SDK")) {
  New-Item -ItemType Directory -Force -Path "win32-x64/SDK" | Out-Null
  $SDK_ROOT = (Resolve-Path "win32-x64/SDK").Path
  # build from source
  cd "OpenCL-SDK-source"
  DelIfExist "build"
  cmake -B build -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -G "Visual Studio 17 2022"
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  DelIfExist "build"
  cmake -B build -DBUILD_SHARED_LIBS=ON -DOVERRIDE_MSVC_FLAGS_TO_MT=OFF -DTUNERS=OFF -DOPENCL_ROOT="$SDK_ROOT" -G "Visual Studio 17 2022"
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  cd ".."
}

# ARM64

if (-not (Test-Path "win32-arm64/SDK")) {
  New-Item -ItemType Directory -Force -Path "win32-arm64/SDK" | Out-Null
  $SDK_ROOT = (Resolve-Path "win32-arm64/SDK").Path
  # build from source
  cd "OpenCL-SDK-source"
  DelIfExist "build"
  cmake -B build -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  cd "../CLBlast-source"
  DelIfExist "build"
  cmake -B build -DBUILD_SHARED_LIBS=ON -DOVERRIDE_MSVC_FLAGS_TO_MT=OFF -DTUNERS=OFF -DOPENCL_ROOT="$SDK_ROOT" -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  cd ".."
}
