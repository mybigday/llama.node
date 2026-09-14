#!/bin/bash

# Regenerate lib/version.ts from the llama.cpp build pinned by llama.rn
# (src/llama.rn/vendor/VERSIONS). Run after bumping the llama.rn submodule.

set -e

cd "$(dirname "$0")/.."

# shellcheck source=src/llama.rn/vendor/VERSIONS
source src/llama.rn/vendor/VERSIONS

BUILD_NUMBER="${LLAMA_CPP_REF#b}"
BUILD_COMMIT="${LLAMA_CPP_COMMIT:0:7}"

echo "export const BUILD_NUMBER = '$BUILD_NUMBER'" > lib/version.ts
echo "export const BUILD_COMMIT = '$BUILD_COMMIT'" >> lib/version.ts
echo "lib/version.ts: build $BUILD_NUMBER ($BUILD_COMMIT)"
