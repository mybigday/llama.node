#!/usr/bin/env bash

# Regenerate scripts/llama.cpp.patch from the working-tree changes in
# src/llama.rn/vendor/llama.cpp.
#
# llama.rn vendors llama.cpp with its own patches already applied (see
# src/llama.rn/vendor/README.md). This patch carries only the llama.node
# specific changes on top of that tree, so it must be regenerated against a
# clean llama.rn checkout: edit the vendored files in place, then run this.
#
# Usage: ./scripts/regenerate-patch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_RN_DIR="$PROJECT_ROOT/src/llama.rn"
PATCH_FILE="$SCRIPT_DIR/llama.cpp.patch"

echo "Regenerating llama.cpp.patch..."

if [ ! -d "$LLAMA_RN_DIR/vendor/llama.cpp" ]; then
  echo "Error: llama.rn submodule not found at $LLAMA_RN_DIR"
  exit 1
fi

cd "$LLAMA_RN_DIR"

if git diff --quiet -- vendor/llama.cpp; then
  echo "No changes found in src/llama.rn/vendor/llama.cpp"
  exit 0
fi

# Paths are relative to the llama.node root so CMake can `git apply` from there
git diff --src-prefix=a/src/llama.rn/ --dst-prefix=b/src/llama.rn/ -- vendor/llama.cpp > "$PATCH_FILE"

echo "Patch regenerated successfully at $PATCH_FILE"
