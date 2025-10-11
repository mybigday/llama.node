#!/usr/bin/env bash

# Regenerate llama.cpp.patch from current changes in the llama.cpp submodule
# Usage: ./scripts/regenerate-patch.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LLAMA_CPP_DIR="$PROJECT_ROOT/src/llama.cpp"
PATCH_FILE="$SCRIPT_DIR/llama.cpp.patch"

echo "Regenerating llama.cpp.patch..."

# Check if llama.cpp submodule exists
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  echo "Error: llama.cpp submodule not found at $LLAMA_CPP_DIR"
  exit 1
fi

# Change to llama.cpp directory
cd "$LLAMA_CPP_DIR"

# Check if there are changes
if git diff --quiet; then
  echo "No changes found in llama.cpp submodule"
  exit 0
fi

# Generate patch with correct path prefix
git diff --src-prefix=a/src/llama.cpp/ --dst-prefix=b/src/llama.cpp/ > "$PATCH_FILE"

echo "Patch regenerated successfully at $PATCH_FILE"
