#!/bin/bash

set -e

# wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
# sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.280-jammy.list https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-jammy.list

# sudo apt-get update
# sudo apt-get install -qy \
#   binutils-aarch64-linux-gnu \
#   gcc-aarch64-linux-gnu \
#   g++-aarch64-linux-gnu \
#   vulkan-sdk

# Install SDK for arm64 by building from source

mkdir externals || true
cd externals

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

SDK_ROOT=$(realpath ./SDK)

if [ ! -d SDK ]; then
  # build from source
  export CC=aarch64-linux-gnu-gcc
  export CXX=aarch64-linux-gnu-g++
  cd OpenCL-SDK-source
  rm -rf build
  cmake -B build -DBUILD_DOCS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_TESTING=OFF -DOPENCL_SDK_BUILD_SAMPLES=OFF -DOPENCL_SDK_TEST_SAMPLES=OFF -DCMAKE_INSTALL_PREFIX=$SDK_ROOT
  cmake --build build --config Release
  cmake --install build
  cd ../CLBlast-source
  rm -rf build
  cmake -B build -DBUILD_SHARED_LIBS=ON -DOVERRIDE_MSVC_FLAGS_TO_MT=OFF -DTUNERS=OFF -DOPENCL_ROOT="$SDK_ROOT" -DCMAKE_INSTALL_PREFIX=$SDK_ROOT
  cmake --build build --config Release
  cmake --install build
fi
