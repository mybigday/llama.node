#!/bin/bash

pushd externals

pushd win32-x64

if [ ! -f OpenCL-SDK.zip ]; then
  curl -L https://github.com/KhronosGroup/OpenCL-SDK/releases/download/v2023.12.14/OpenCL-SDK-v2023.12.14-Win-x64.zip -o OpenCL-SDK.zip
  7z x OpenCL-SDK-v2023.12.14-Win-x64.zip -oSDK
fi

if [ ! -f CLBlast.zip ]; then
  curl -L https://github.com/CNugteren/CLBlast/releases/download/1.6.2/CLBlast-1.6.2-windows-x64.zip -o CLBlast.zip
  7z x CLBlast.zip
  7z x CLBlast*.7z
  mv CLBlast-1.6.2-windows-x64/* SDK
fi

popd

if [ ! -d OpenCL-SDK-source ]; then
  # clone KhronosGroup/OpenCL-SDK tag v2023.12.14
  git clone https://github.com/KhronosGroup/OpenCL-SDK.git OpenCL-SDK-source --recursive
  pushd OpenCL-SDK-source
  git checkout v2023.12.14
  popd
fi

if [ ! -d CLBlast-source ]; then
  # clone CNugteren/CLBlast tag 1.6.2
  git clone https://github.com/CNugteren/CLBlast.git CLBlast-source --recursive
  pushd CLBlast-source
  git checkout 1.6.2
  popd
fi

pushd win32-arm64

if which cygpath; then
  SDK_ROOT=$(cygpath -w SDK)
else
  SDK_ROOT=$(realpath SDK)
fi

if [ ! -d SDK ]; then
  # build from source
  pushd ../OpenCL-SDK-source
  cmake -B build -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  popd
  pushd ../CLBlast-source
  cmake -B build -DBUILD_SHARED_LIBS=ON -DOVERRIDE_MSVC_FLAGS_TO_MT=OFF -DTUNERS=OFF -DOPENCL_ROOT="$SDK_ROOT" -G "Visual Studio 17 2022" -A arm64
  cmake --build build --config Release
  cmake --install build --prefix $SDK_ROOT
  popd
fi

popd
