#!/bin/bash

set -e

function resolve_path {
  python -c "import os,sys; print(os.path.abspath(sys.argv[1]))" "$1"
}

# General

yarn clean && yarn build-native -a x86_64
yarn clean && yarn build-native -a arm64

# build with CLBlast

yarn clean && yarn build-native -a x86_64 --CDVARIANT=opencl --CDLLAMA_CLBLAST=ON --CDCMAKE_PREFIX_PATH="$(resolve_path 'externals/win32-x64/SDK/lib/cmake/CLBlast')"
yarn clean && yarn build-native -a arm64 --CDVARIANT=opencl --CDLLAMA_CLBLAST=ON --CDCMAKE_PREFIX_PATH="$(resolve_path 'externals/win32-arm64/SDK/lib/cmake/CLBlast')"
