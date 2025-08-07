#!/bin/bash

# Get version info
cd src/llama.cpp
BUILD_NUMBER=$(git rev-list --count HEAD)
BUILD_COMMIT=$(git rev-parse --short HEAD)

# Put to ../version.ts
echo "export const BUILD_NUMBER = '$BUILD_NUMBER';" > ../../lib/version.ts
echo "export const BUILD_COMMIT = '$BUILD_COMMIT';" >> ../../lib/version.ts
