# llama.node

[![CI](https://github.com/mybigday/llama.node/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/mybigday/llama.node/actions/workflows/ci.yml)
[![NPM Version](https://img.shields.io/npm/v/%40fugood%2Fllama.node)](https://www.npmjs.com/package/@fugood/llama.node)
![NPM Downloads](https://img.shields.io/npm/dw/%40fugood%2Fllama.node)

An another Node binding of [llama.cpp](https://github.com/ggerganov/llama.cpp) to make same API with [llama.rn](https://github.com/mybigday/llama.rn) as much as possible.

- [llama.cpp](https://github.com/ggerganov/llama.cpp): Inference of [LLaMA](https://arxiv.org/abs/2302.13971) model in pure C/C++
- [llama.rn](https://github.com/mybigday/llama.rn): React Native binding of llama.cpp

## Platform Support

- macOS
  - arm64: CPU and Metal GPU acceleration
  - x86_64: CPU only
- Windows (x86_64 and arm64)
  - CPU
  - GPU acceleration via Vulkan
- Linux (x86_64 and arm64)
  - CPU
  - GPU acceleration via Vulkan
  - GPU acceleration via CUDA

## Installation

```sh
npm install @fugood/llama.node
```

## Usage

```js
import { loadModel } from '@fugood/llama.node'

// Initial a Llama context with the model (may take a while)
const context = await loadModel({
  model: 'path/to/gguf/model',
  use_mlock: true,
  n_ctx: 2048,
  n_gpu_layers: 1, // > 0: enable GPU
  // embedding: true, // use embedding
  // lib_variant: 'opencl', // Change backend
})

// Do completion
const { text } = await context.completion(
  {
    prompt: 'This is a conversation between user and llama, a friendly chatbot. respond in simple markdown.\n\nUser: Hello!\nLlama:',
    n_predict: 100,
    stop: ['</s>', 'Llama:', 'User:'],
    // n_threads: 4,
  },
  (data) => {
    // This is a partial completion callback
    const { token } = data
  },
)
console.log('Result:', text)
```

## Lib Variants

- [x] `default`: General usage, not support GPU except macOS (Metal)
- [x] `vulkan`: Support GPU Vulkan (Windows/Linux), but some scenario might unstable

## License

MIT

---

<p align="center">
  <a href="https://bricks.tools">
    <img width="90px" src="https://avatars.githubusercontent.com/u/17320237?s=200&v=4">
  </a>
  <p align="center">
    Built and maintained by <a href="https://bricks.tools">BRICKS</a>.
  </p>
</p>
