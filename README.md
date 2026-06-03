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
  - GPU acceleration via CUDA (x86_64)
- Linux (x86_64 and arm64)
  - CPU
  - GPU acceleration via Vulkan
  - GPU acceleration via CUDA
- Web
  - WebAssembly CPU
  - GPU acceleration via WebGPU

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
  n_ctx: 2048,
  n_gpu_layers: 99, // > 0: enable GPU
  // lib_variant: 'vulkan', // Change backend
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

## WebAssembly

Browser builds are published as the optional `@fugood/node-llama-wasm` package.
Import `@fugood/llama.node` in browser bundles; the package browser entry
re-exports the WASM runtime while preserving the same high-level `loadModel()` /
context wrapper shape where browser constraints allow:

```js
import { loadModel } from '@fugood/llama.node'

const context = await loadModel({
  model: 'https://example.com/model.gguf',
  n_ctx: 2048,
})

const tokens = await context.tokenize('Hello from the browser')
const state = await context.saveSession()
await context.loadSession(new Blob([state]))
```

Build the package with Emscripten:

```sh
npm run build-wasm
npm run build-wasm-docker
npm run build-wasm -- --webgpu
npm run serve-wasm-test
```

CPU and WebGPU builds use separate directories under `build-wasm/`, so switching
variants does not invalidate the other build. Fresh builds use Ninja when it is
available, `JOBS=8` can cap parallelism, and installing `ccache` enables compiler
launcher caching automatically. Emscripten's system-library cache is kept in
`build-wasm/emcache` unless `EM_CACHE` is already set. The Docker helper selects
`emscripten/emsdk:4.0.14-arm64` on arm64 hosts such as Apple Silicon Macs, and
uses `emscripten/emsdk:4.0.13` on amd64 hosts. Override with `EMSCRIPTEN_IMAGE`
or `EMSCRIPTEN_PLATFORM` if you need a specific container image.

`loadModel()` runs the WASM runtime in a dedicated Web Worker by default so
model loading, tokenization, completion, state, embeddings, rerank, multimodal
staging, and benchmarks do not block the browser UI thread. On isolated pages
with `SharedArrayBuffer`, the CPU path selects the pthread build and defaults
`n_threads` to `min(4, navigator.hardwareConcurrency)`. Pass
`wasm: { threads: false }` to force the single-thread artifact, or
`wasm: { maxThreads: 8 }` / `n_threads` to tune CPU thread count. Pass
`wasm: { worker: false }` only when you intentionally need direct access to the
Emscripten module on the current thread.

Model strings are fetched as URLs by default. `saveSession()` returns an
`ArrayBuffer`, and `loadSession()` accepts a URL, `Blob`, `ArrayBuffer`, or typed
array. URL downloads are saved in browser Cache Storage by default so repeated
model/session/media URL loads can reuse the previous bytes; pass
`wasm: { cacheDownloads: false }` to force a fresh fetch, `wasm.cacheName` to use
a separate cache, or call `clearWasmDownloadCache()` to clear the default cache.
The `loadModel()` progress callback still receives the numeric percentage first,
and also receives an optional detail object with `source: 'network' | 'cache' |
'memory' | 'buffer'` so UI code can distinguish real downloads from cache hits.
Browser `ArrayBuffer` limits still apply, so split GGUF files at or above the 2
GB limit into smaller shards before loading. WebGPU is opt-in via `n_gpu_layers`
and requires both `navigator.gpu` and WebAssembly JSPI.

The browser package includes a serial `context.parallel` queue for API
compatibility. It preserves the parallel API shape for web callers, but it runs
one request at a time until the native slot-manager path is ported to WASM.

## Lib Variants

- [x] `default`: General usage, not support GPU except macOS (Metal)
- [x] `vulkan`: Support GPU Vulkan (Windows/Linux), but some scenario might unstable
- [x] `cuda`: Support GPU CUDA (Windows/Linux), but only for limited capability
  > Linux: (x86_64: 8.9, arm64: 8.7)
  > Windows: x86_64 - 12.0
- [x] `wasm`: Browser package with WebAssembly CPU and optional WebGPU

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
