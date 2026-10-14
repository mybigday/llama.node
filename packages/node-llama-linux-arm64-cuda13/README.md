# @fugood/node-llama-linux-arm64-cuda13

Native module for llama.node targeting linux-arm64-cuda13.

This package contains the pre-compiled native module for the specified platform and architecture with CUDA 13 support (Jetson Thor / JetPack 7, DGX Spark / GB10).

## Installation

This package is not installed automatically. Install it next to `@fugood/llama.node` on CUDA 13 devices:

```bash
npm install @fugood/node-llama-linux-arm64-cuda13
```

`lib_variant: 'cuda'` prefers this package when it is installed and finds a CUDA device, and falls back to `@fugood/node-llama-linux-arm64-cuda` (CUDA 12, Jetson Orin) otherwise.

## Platform Support

- **OS**: linux
- **Architecture**: arm64
- **Variant**: cuda13

## Usage

This package is not meant to be used directly. It is consumed by the main `@fugood/llama.node` package.
