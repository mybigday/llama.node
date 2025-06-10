#!/bin/bash

set -e

function run_as_root() {
  if [ $UID -ne 0 ]; then
    sudo -E $@
  else
    $@
  fi
}

export DEBIAN_FRONTEND=noninteractive

ARCH=${ARCH:-$(uname -m)}
TARGET=${TARGET:-"default"}

while [[ "$#" -gt 0 ]]; do
  case $1 in
    -a|--arch) ARCH="$2"; shift ;;
    -t|--target) TARGET="$2"; shift ;;
    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

run_as_root apt-get update
run_as_root apt-get install -qy lsb-release wget

if [ $ARCH == "x86_64" ]; then
  if [ $TARGET == "vulkan" ]; then
    DISTRO=$(lsb_release -c -s)
    wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | run_as_root tee /etc/apt/trusted.gpg.d/lunarg.asc
    run_as_root wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.280-$DISTRO.list https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-$DISTRO.list
    run_as_root apt-get update
    run_as_root apt-get install -qy vulkan-sdk cmake pkg-config build-essential  libx11-xcb-dev libxkbcommon-dev libwayland-dev libxrandr-dev
  fi
elif [ $ARCH == "arm64" ]; then
  if [ $TARGET == "vulkan" ]; then
    run_as_root apt-get update
    run_as_root apt-get install -qy curl gnupg2

    if ! command -v node &> /dev/null; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | run_as_root bash -
      run_as_root apt-get install -qy nodejs
      npm install -g yarn
    fi

    run_as_root apt-get install -qy git make cmake pkg-config build-essential libx11-xcb-dev libxkbcommon-dev libwayland-dev libxrandr-dev

    mkdir externals || true
    cd externals

    if [ ! -d OpenCL-SDK-source ]; then
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
        cmake -B build
        cmake --build build --config Release
        cmake --install build --prefix $VULKAN_ROOT
        cd ..
      fi
      if [ ! -d Vulkan-Loader-source ]; then
        git clone "https://github.com/KhronosGroup/Vulkan-Loader.git" "Vulkan-Loader-source"
        cd "Vulkan-Loader-source"
        git checkout "sdk-1.3.261"
        cmake -B build -DVULKAN_HEADERS_INSTALL_DIR="$VULKAN_ROOT" -DUSE_MASM=OFF
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
  fi
fi
