#!/bin/bash

set -e

# add vulkan-sdk source
wget -qO- https://packages.lunarg.com/lunarg-signing-key-pub.asc | sudo tee /etc/apt/trusted.gpg.d/lunarg.asc
sudo wget -qO /etc/apt/sources.list.d/lunarg-vulkan-1.3.280-jammy.list https://packages.lunarg.com/vulkan/1.3.280/lunarg-vulkan-1.3.280-jammy.list

# add arm64 dependencies
sudo dpkg --add-architecture arm64

sudo apt-get update
sudo apt-get install -qy \
  binutils-aarch64-linux-gnu \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  vulkan-sdk \
  libvulkan-dev:arm64
