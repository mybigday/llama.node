#!/bin/bash

# Script to copy llama.rn source files and remove lm_ and LM_ prefixes

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Starting copy of llama.rn source files...${NC}"

# Define source and destination directories
SRC_DIR="src/llama.rn/cpp"
DEST_DIR="src/rn-llama"

mkdir -p "$DEST_DIR"
rm -rf "$DEST_DIR"/*

# Files to copy from llama.rn/cpp
FILES_TO_COPY=(
    "anyascii.h"
    "anyascii.c"
    "rn-llama.h"
    "rn-llama.cpp" 
    "rn-completion.h"
    "rn-completion.cpp"
    "rn-mtmd.hpp"
    "rn-tts.h"
    "rn-tts.cpp"
)

# Initialize and update llama.rn submodule if needed
echo -e "${YELLOW}Ensuring llama.rn submodule is initialized...${NC}"
if [ ! -f "$SRC_DIR/rn-llama.h" ]; then
    echo "Initializing llama.rn submodule..."
    git submodule init src/llama.rn
    git submodule update --recursive src/llama.rn
fi

# Copy files and remove lm_ and LM_ prefixes
for file in "${FILES_TO_COPY[@]}"; do
    if [ -f "$SRC_DIR/$file" ]; then
        echo -e "${YELLOW}Copying and processing $file...${NC}"
        
        # Copy the file and process it with sed to remove lm_ and LM_ prefixes
        sed -e 's/lm_ggml/ggml/g' -e 's/LM_GGML/GGML/g' "$SRC_DIR/$file" > "$DEST_DIR/$file"
        
        echo -e "${GREEN}✓ $file processed and copied to $DEST_DIR/$file${NC}"
    else
        echo -e "${RED}✗ Source file $SRC_DIR/$file not found!${NC}"
        exit 1
    fi
done

echo -e "${GREEN}All llama.rn source files copied and processed successfully!${NC}"
echo -e "${YELLOW}Note: lm_ and LM_ prefixes have been removed from all copied files.${NC}"