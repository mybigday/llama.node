#!/bin/bash

set -e

mkdir externals || true
cd externals

mkdir win32-x64 || true
cd win32-x64

mkdir SDK || true

if [ ! -f OpenCL-SDK.zip ]; then
  curl -L https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2023.12.14/OpenCL-SDK-v2023.12.14-Win-x64.zip -o OpenCL-SDK.zip
  7z x OpenCL-SDK.zip
  cp -r OpenCL-SDK-*/* SDK
  rm -rf OpenCL-SDK-*
fi

if [ ! -f CLBlast.zip ]; then
  curl -L https://github.com/CNugteren/CLBlast/releases/download/1.6.2/CLBlast-1.6.2-windows-x64.zip -o CLBlast.zip
  7z x CLBlast.zip
  7z x CLBlast*.7z
  cp -r CLBlast-*/* SDK
  rm -rf CLBlast-*
fi

cd ..

if [ ! -d OpenCL-SDK-source ]; then
  # clone KhronosGroup/OpenCL-SDK tag v2023.12.14
  git clone https://github.com/KhronosGroup/OpenCL-SDK.git OpenCL-SDK-source --recursive
  cd OpenCL-SDK-source
  git checkout v2023.12.14
  cd ..
fi

if [ ! -d CLBlast-source ]; then
  # clone CNugteren/CLBlast tag 1.6.2
  git clone https://github.com/CNugteren/CLBlast.git CLBlast-source --recursive
  cd CLBlast-source
  git checkout 1.6.2
  cd ..
fi

mkdir win32-arm64 || true
cd win32-arm64

SDK_ROOT=$(python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "SDK")

if [ ! -d SDK ]; then
  # build from source
  cd ../OpenCL-SDK-source
  cmake -B build -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  cd ../CLBlast-source
  cmake -B build -DBUILD_SHARED_LIBS=ON -DOVERRIDE_MSVC_FLAGS_TO_MT=OFF -DTUNERS=OFF -DOPENCL_ROOT="$SDK_ROOT" -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
fi
