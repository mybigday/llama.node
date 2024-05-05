#!/bin/bash

set -e

sudo apt-get update
sudo apt-get install -qy \
  binutils-aarch64-linux-gnu \
  gcc-aarch64-linux-gnu \
  g++-aarch64-linux-gnu \
  vulkan-sdk \
  libvulkan-dev:arm64


