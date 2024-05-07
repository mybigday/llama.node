#!/bin/bash

set -e

wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.280-jammy.list https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-jammy.list

sudo apt-get update
sudo apt-get install -qy \
  binutils-aarch64-linux-gnu \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  vulkan-sdk

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

# build Vulkan SDK from source for arm64

if [ ! -d arm64-Vulkan-SDK ]; then
  VULKAN_ROOT=$(realpath ./arm64-Vulkan-SDK)
  if [ ! -d Vulkan-Headers-source ]; then
    git clone "https://github.com/KhronosGroup/Vulkan-Headers.git" "Vulkan-Headers-source"
    cd "Vulkan-Headers-source"
    git checkout "sdk-1.3.261"
    CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ cmake -B build
    cmake --build build --config Release
    cmake --install build --prefix $VULKAN_ROOT
    cd ..
  fi
  if [ ! -d Vulkan-Loader-source ]; then
    git clone "https://github.com/KhronosGroup/Vulkan-Loader.git" "Vulkan-Loader-source"
    cd "Vulkan-Loader-source"
    git checkout "sdk-1.3.261"
    CC=aarch64-linux-gnu-gcc CXX=aarch64-linux-gnu-g++ cmake -B build -DVULKAN_HEADERS_INSTALL_DIR="$VULKAN_ROOT" -DUSE_MASM=OFF
    cmake --build build --config Release
    cmake --install build --prefix $VULKAN_ROOT
    cd ".."
  fi
  if [ ! -d Vulkan-Hpp-source ]; then
    git clone "https://github.com/KhronosGroup/Vulkan-Hpp.git" "Vulkan-Hpp-source"
    cd "Vulkan-Hpp-source"
    git checkout "v1.3.261"
    git submodule update --init --recursive
    cmake -B build -DVULKAN_HPP_INSTALL=ON -DVULKAN_HPP_RUN_GENERATOR=ON
    cmake --build build --config Release
    cmake --install build --prefix $VULKAN_ROOT
    cd ".."
  fi
fi
