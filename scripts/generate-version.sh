#!/bin/bash

# Regenerate lib/version.ts from the llama.cpp build pinned by llama.rn.
# Run after bumping the llama.rn submodule.
#
# llama.cpp refs are no longer always bNNNN tags (e.g. v0.8.0), so take the
# build number llama.rn's sync-vendor.sh counted from the pinned commit
# (src/llama.rn/src/version.ts) instead of parsing vendor/VERSIONS.

set -e

cd "$(dirname "$0")/.."

cp src/llama.rn/src/version.ts lib/version.ts

BUILD_NUMBER="$(sed -n "s/^export const BUILD_NUMBER = '\(.*\)'$/\1/p" lib/version.ts)"
BUILD_COMMIT="$(sed -n "s/^export const BUILD_COMMIT = '\(.*\)'$/\1/p" lib/version.ts)"
echo "lib/version.ts: build $BUILD_NUMBER ($BUILD_COMMIT)"
