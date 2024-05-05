$ErrorActionPreference='Stop'

New-Item -ItemType Directory -Force -Path "externals/win32-x64/SDK" | Out-Null
New-Item -ItemType Directory -Force -Path "externals/win32-arm64" | Out-Null

# x64

cd "externals/win32-x64"

# OpenCL-SDK
if (-not (Test-Path "OpenCL-SDK.zip")) {
  Invoke-WebRequest -Uri "https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2023.12.14/OpenCL-SDK-v2023.12.14-Win-x64.zip" -OutFile "OpenCL-SDK.zip"
  7z x OpenCL-SDK.zip
  Copy-Item -Recurse -Force OpenCL-SDK-*/* SDK
  Remove-Item -Recurse -Force OpenCL-SDK-*
}

# CLBlast
if (-not (Test-Path "CLBlast.zip")) {
  Invoke-WebRequest -Uri "https://github.com/CNugteren/CLBlast/releases/download/1.6.2/CLBlast-1.6.2-windows-x64.zip" -OutFile "CLBlast.zip"
  7z x CLBlast.zip
  7z x CLBlast*.7z
  Copy-Item -Recurse -Force CLBlast-*/* SDK
  Remove-Item -Recurse -Force CLBlast-*
}

cd ".."

# ARM64

cd "win32-arm64"

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

if (-not (Test-Path "SDK")) {
  New-Item -ItemType Directory -Force -Path "SDK" | Out-Null
  $SDK_ROOT = (Resolve-Path "SDK").Path
  # build from source
  cd "OpenCL-SDK-source"
  cmake -B build -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  cd "../CLBlast-source"
  cmake -B build -DBUILD_SHARED_LIBS=ON -DOVERRIDE_MSVC_FLAGS_TO_MT=OFF -DTUNERS=OFF -DOPENCL_ROOT="$SDK_ROOT" -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
}
