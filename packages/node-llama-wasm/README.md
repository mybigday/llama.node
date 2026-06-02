# @fugood/node-llama-wasm

WebAssembly package for llama.node browser inference.

This package exposes the same high-level `loadModel()` and context methods used
by `@fugood/llama.node`, with browser-specific I/O:

- `model` strings are fetched as URLs by default.
- `saveSession()` returns an `ArrayBuffer`.
- `loadSession()` accepts a URL, `Blob`, `ArrayBuffer`, or typed array.
- `initMultimodal()` accepts an mmproj URL, `Blob`, `ArrayBuffer`, typed array,
  or preloaded MEMFS path. Image/audio URL media in `messages` or `media_paths`
  is staged into the virtual filesystem before inference.
- WebGPU can be opted into with `n_gpu_layers` when the WASM binary is built with `GGML_WEBGPU=ON`,
  `navigator.gpu` is available, and the browser exposes WebAssembly JSPI
  (`WebAssembly.promising` and `WebAssembly.Suspending`).
- The distributed build uses WebAssembly Memory64, matching current wllama
  constraints. Browsers without Memory64 support are not supported.
- `loadModel()` uses a dedicated Web Worker by default so WASM work does not
  block the browser UI thread. On isolated pages with `SharedArrayBuffer`, the
  CPU path selects the pthread artifact and defaults `n_threads` to
  `min(4, navigator.hardwareConcurrency)`. Use `wasm: { threads: false }` for
  the single-thread fallback, or set `n_threads` / `wasm.maxThreads` to tune CPU
  threading. Use `wasm: { worker: false }` only for direct Emscripten-module
  debugging or integration code that must run on the current thread.

Large model files at or above the browser WebAssembly `ArrayBuffer` limit are
rejected. Split large GGUF files into shards, preferably 512 MB or smaller.

```js
import { isWebGpuSupported, loadModel } from '@fugood/node-llama-wasm'

const context = await loadModel({
  model: 'https://huggingface.co/Durlabh/gemma-270m-q4-k-m-gguf/resolve/main/gemma3-270m-it-q4_k_m.gguf',
  n_ctx: 2048,
  n_gpu_layers: isWebGpuSupported() ? 99999 : 0,
})

const tokens = await context.tokenize('Hello')
const text = await context.detokenize(tokens.tokens)
const state = await context.saveSession()
await context.loadSession(new Blob([state]))
const result = await context.completion({ prompt: text, n_predict: 32 })
```

Build from the repository root:

```sh
npm run build-wasm
npm run build-wasm -- --webgpu
npm run serve-wasm-test
```

The build script keeps CPU and WebGPU artifacts in separate build directories,
uses Ninja on fresh build dirs when available, respects `JOBS`, and enables
`ccache` automatically when installed. It also stores Emscripten's system-library
cache in `build-wasm/emcache` unless `EM_CACHE` is already set.
