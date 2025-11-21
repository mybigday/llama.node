# @fugood/node-llama-linux-arm64-snapdragon

Native module for llama.node targeting linux-arm64-snapdragon.

This package contains the pre-compiled native module for the specified platform and architecture with Qualcomm Snapdragon OpenCL and Hexagon support.

## Installation

This package is typically installed automatically as a dependency of `@fugood/llama.node`.

## Platform Support

- **OS**: linux
- **Architecture**: arm64
- **Variant**: snapdragon
- **Backends**: OpenCL, Hexagon NPU

## Usage

This package is not meant to be used directly. It is consumed by the main `@fugood/llama.node` package.

## Best Practice

This config for best practice.

```js
{
    n_gpu_layers: 99,
    devices: ['HTP0'], // All available HTP devices
    flash_attn_type: 'on',
    n_threads: 6,
    batch_size: 128,
    cache_type_k: 'q8_0',
    cache_type_v: 'q8_0',
    cpu_mask: '0xfc',
    cpu_strict: true,
}
```
