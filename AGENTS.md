## Project Overview

llama.node is a Node.js binding for llama.cpp, designed to maintain API compatibility with llama.rn (the React Native binding). This project enables local LLM inference in Node.js with support for:

- Multiple platforms: macOS (arm64/x64), Windows (x64/arm64), Linux (x64/arm64)
- GPU acceleration: Metal (macOS), Vulkan, CUDA
- Web platform support: WebAssembly CPU and optional WebGPU via the optional `@fugood/node-llama-wasm` package; browser consumers can import `@fugood/llama.node`
- Multimodal support: vision and audio processing
- Text-to-speech (TTS) with vocoder models
- LoRA adapters for model fine-tuning
- Session save/load for context persistence
- Embeddings and reranking capabilities

## Architecture

### Native Binding Layer (C++)

The codebase uses N-API to bridge Node.js and llama.cpp:

- **src/addons.cc**: Entry point that registers the LlamaContext class
- **src/LlamaContext.cpp/h**: Main context class exposing all llama.cpp functionality
- **src/rn-llama/**: Shared implementation from llama.rn submodule (copied during build)
  - `rn-llama.cpp/h`: Core llama context wrapper
  - `rn-completion.cpp/h`: Completion logic
  - `rn-mtmd.hpp`: Multimodal (vision/audio) support
  - `rn-tts.cpp/h`: Text-to-speech functionality

### Worker Pattern

All async operations use N-API AsyncWorker to avoid blocking the event loop:

- **DisposeWorker**: Cleanup/release of llama context
- **LlamaCompletionWorker**: Text generation with streaming callbacks
- **TokenizeWorker/DetokenizeWorker**: Token conversion
- **EmbeddingWorker**: Generate embeddings
- **RerankWorker**: Document reranking
- **LoadSessionWorker/SaveSessionWorker**: Context state persistence
- **DecodeAudioTokenWorker**: TTS audio token decoding

### TypeScript API Layer (lib/)

- **lib/binding.ts**: TypeScript types for the native module
- **lib/index.ts**: High-level API wrapper with `LlamaContextWrapper` class
  - Handles multimodal content transformation (images, audio)
  - Manages Jinja template formatting for chat
  - Provides logging infrastructure
  - Module caching and variant loading

### WebAssembly API Layer (packages/node-llama-wasm/)

- **packages/node-llama-wasm/index.js**: Browser-facing API compatible with the high-level llama.node wrapper where web constraints allow
- **packages/node-llama-wasm/worker.js**: Dedicated Web Worker runtime used by default to keep all heavy WASM calls off the browser UI thread
- **packages/node-llama-wasm/wasm/**: Generated WASM artifacts are written here during build and are ignored except for `.gitkeep`
- **lib/browser.mjs**: Root package browser entry that re-exports `@fugood/node-llama-wasm`, so browser apps can keep importing `@fugood/llama.node`
- **src/wasm/llama-node-wasm.cpp**: C ABI bridge that exposes llama.node actions to Emscripten
- **src/wasm/CMakeLists.txt**: Emscripten build configuration for CPU, CPU pthreads, and WebGPU artifacts

The web package supports URL model loading by default, session save as `ArrayBuffer`, session load from URL/Blob/ArrayBuffer/typed array, and browser model-size checks for the 2 GB WebAssembly ArrayBuffer limit.

### Build System

- **CMakeLists.txt**: Builds native addon, links llama.cpp and rn-llama sources
- **src/llama.cpp/**: llama.cpp submodule (upstream library)
- **src/llama.rn/**: llama.rn submodule (shared React Native implementation)
- **src/wasm/**: Emscripten-specific WASM build target
- **scripts/build-wasm-package.js**: Builds WASM artifacts into `packages/node-llama-wasm/wasm/`
- Platform-specific prebuilt packages: `@fugood/node-llama-{platform}-{arch}[-variant]`
- Browser package: `@fugood/node-llama-wasm` (listed as an optional dependency of `@fugood/llama.node`)

The build process applies patches and copies rn-llama sources with prefix transformations:
- `lm_ggml` → `ggml`
- `LM_GGML` → `GGML`

## Common Development Commands

### Building

```bash
# Bootstrap: copy rn-llama sources and install dependencies
npm run copy-rn-llama-source && npm install --omit=optional

# Build native addon (requires cmake-js)
npm run build-native

# Build TypeScript
npm run build-js

# Build WASM CPU artifact
npm run build-wasm

# Build WASM in Docker (uses native arm64 emsdk image on Apple Silicon)
npm run build-wasm-docker

# Build WASM WebGPU artifact
npm run build-wasm -- --webgpu

# Serve the browser WASM test page
npm run serve-wasm-test

# Type checking
npm run typecheck

# Clean build artifacts
npm run clean
```

### Testing

```bash
# Run all tests (downloads models first if needed)
npm test

# Run specific test file
npx jest test/index.test.ts

# Run tests with specific pattern
npx jest --testNamePattern="tokenize"
```

### Package Management

```bash
# Build packages for distribution
npm run build-native  # with TO_PACKAGE=1 and VARIANT set
npm run build-wasm    # run inside an activated Emscripten SDK
npm run build-wasm-docker  # Docker helper; arm64 hosts use emscripten/emsdk:4.0.14-arm64

# Publish if version changed
npm run publish-if-need
```

## Key Implementation Details

### Model Loading

Models are loaded through `loadModel()` which:
1. Loads the appropriate native module variant (default/vulkan/cuda)
2. Creates a native `LlamaContext` with the specified options
3. Wraps it in `LlamaContextWrapper` for high-level API
4. Returns the wrapper with methods for completion, embedding, etc.

For the web package, `loadModel()`:
1. Fetches model URLs into the Emscripten virtual filesystem by default, reusing browser Cache Storage for repeated URL downloads when available
2. Chooses CPU, CPU pthreads, or WebGPU artifacts from `packages/node-llama-wasm/wasm/`
3. Runs the runtime in `worker.js` by default so API calls do not block the UI thread
4. Uses `n_gpu_layers > 0` to opt into WebGPU
5. Uses the threaded CPU artifact when `SharedArrayBuffer` is available and the page is `crossOriginIsolated`
6. Defaults CPU `n_threads` to `min(4, navigator.hardwareConcurrency)` unless explicitly overridden

### Completion Flow

1. User calls `context.completion(options, callback)`
2. Options are parsed, including:
   - Prompt or messages (formatted with chat template if needed)
   - Sampling parameters (temperature, top_k, etc.)
   - Response format (text/json_schema/json_object)
   - Grammar constraints, tool definitions
3. `LlamaCompletionWorker` is queued with an async task
4. During generation, streaming callbacks fire for each token
5. Final result includes text, timings, and metadata

### Multimodal Support

Images and audio are handled via the MTMD (multimodal) wrapper:
- `initMultimodal({ path, use_gpu })` loads mmproj file
- Messages with image_url or input_audio content parts are transformed
- Media paths are replaced with `<__media__>` markers
- The MTMD system processes media before text generation

### Chat Templating

Two template engines are supported:
- **llama-chat**: Classic llama.cpp chat formatting
- **Jinja**: Full Jinja2 template support via llama.cpp's jinja engine
  - Supports tool calling, reasoning, response formats
  - Automatically handles grammar generation for JSON schemas

### Logging

Native llama.cpp logs can be captured:
```javascript
await toggleNativeLog(true)
const listener = addNativeLogListener((level, text) => {
  console.log(`[${level}] ${text}`)
})
```

Uses ThreadSafeFunction to safely callback from C++ threads.

### GPU Acceleration

Controlled via `n_gpu_layers` option:
- macOS: Metal is default (when n_gpu_layers > 0)
- Windows/Linux: Requires vulkan or cuda variant
- Set `lib_variant` to 'vulkan' or 'cuda' when loading model
- Web: WebGPU is selected with `n_gpu_layers > 0` in `@fugood/node-llama-wasm`; it requires `navigator.gpu` and WebAssembly JSPI support

### Session Management

Save/load KV cache state for context persistence:
```javascript
await context.saveSession('session.bin')
await context.loadSession('session.bin')
```

Note: Vulkan backend does not support session save/load.

In the web package, `saveSession()` returns an `ArrayBuffer`. `loadSession()` accepts a URL, `Blob`, `ArrayBuffer`, or typed array. URL downloads for models, sessions, mmproj files, and media are saved in browser Cache Storage by default. Use `wasm: { cacheDownloads: false }` to force network fetches, `wasm.cacheName` to isolate a cache, or `clearWasmDownloadCache()` to clear the default cache. The `loadModel()` progress callback receives the numeric percentage plus an optional detail object with `source: 'network' | 'cache' | 'memory' | 'buffer'`; use it to avoid labeling cache hits as downloads.

### WASM Threading and UI Responsiveness

The web package defaults to a dedicated Worker runtime. All heavy methods should stay off the main thread, including:
- `loadModel()`
- `completion()`
- `tokenize()` / `detokenize()`
- `saveSession()` / `loadSession()`
- `embedding()` / `rerank()`
- `initMultimodal()` media/projector staging
- `bench()`

On isolated pages with `SharedArrayBuffer`, CPU uses the pthread artifact and `n_threads` defaults to a conservative capped value. Use:
- `wasm: { threads: false }` to force the single-thread CPU artifact
- `wasm: { maxThreads: 8 }` to raise the automatic thread cap
- `wasm: { cacheDownloads: false }` to bypass browser Cache Storage for URL sources
- `n_threads` to set an explicit thread count
- `wasm: { worker: false }` only for direct Emscripten/module debugging

## Testing Considerations

- Test models are cached in `test/*.gguf` (downloaded on first test run)
- Models used: tiny-random-llama, flan-t5-small, bge-small-en, Qwen3-0.6B
- Tests verify: completion, streaming, tokenization, embeddings, multimodal, TTS
- Use `vocab_only: true` for fast model info/tokenization tests without loading full model
- Browser WASM test page: `test/web/llama-node-wasm.html`
- Serve WASM tests with `npm run serve-wasm-test` and open `http://localhost:8088/test/web/llama-node-wasm.html`
- The WASM HTML test should validate Gemma 3 270M via URL for load, UI responsiveness, tokenize/detokenize, session save/load, completion, and tool-call formatting/parsing
- For CPU/WebGPU perf checks, compare `context.bench()` with `n_gpu_layers: 0` versus `n_gpu_layers > 0`; CPU should use reasonable `n_threads` on isolated pages

## Debugging

### Common Issues

1. **Build failures**: Ensure submodules are initialized (`git submodule update --init --recursive`)
2. **Missing native module**: Run `npm run copy-rn-llama-source` then rebuild
3. **GPU not working**: Check `lib_variant` matches your hardware (vulkan/cuda)
4. **Session load fails**: Vulkan doesn't support sessions; use default or cuda variant
5. **Context shifting**: Disable with `ctx_shift: false` if using multimodal
6. **WASM threads not used**: Ensure the page is served with COOP/COEP headers so `crossOriginIsolated === true`; `scripts/serve-wasm-test.js` sets these headers
7. **WASM pthread artifact hangs**: Check that `mainScriptUrlOrBlob` points to `llama-node.threads.js` when loading the threaded artifact
8. **WebGPU unavailable**: Confirm the browser supports `navigator.gpu`, WebAssembly JSPI, and that the WebGPU artifact was built with `npm run build-wasm -- --webgpu`
9. **Model too large in browser**: Split GGUF files at or above the 2 GB WebAssembly ArrayBuffer limit and load the first split URL
10. **Unexpected stale web model/session URL bytes**: Call `clearWasmDownloadCache()` from `@fugood/node-llama-wasm`, use a new `wasm.cacheName`, or pass `wasm: { cacheDownloads: false }` while debugging.
11. **Slow Docker WASM builds on Apple Silicon**: Use `npm run build-wasm-docker`; it selects `emscripten/emsdk:4.0.14-arm64` instead of running the amd64 image under emulation. Override with `EMSCRIPTEN_IMAGE` or `EMSCRIPTEN_PLATFORM` when needed.

### Debug Build

```bash
CMAKE_BUILD_TYPE=Debug npm run build-native
```

### Useful File Locations

- Native addon source: `src/*.cpp`, `src/*.h`
- Shared rn-llama code: `src/rn-llama/` (generated, don't edit directly)
- TypeScript API: `lib/index.ts`, `lib/binding.ts`
- Tests: `test/*.test.ts`
- Build scripts: `scripts/*.js`
- CMake config: `CMakeLists.txt`
- WASM bridge: `src/wasm/llama-node-wasm.cpp`
- WASM package: `packages/node-llama-wasm/`
- WASM browser test: `test/web/llama-node-wasm.html`

## Relationship with llama.rn

This project shares core implementation with llama.rn to maintain API compatibility:
- rn-llama sources are copied from llama.rn submodule during build
- APIs should match llama.rn as closely as possible
- Divergences only where platform differences require it (e.g., Node.js vs React Native)

## Platform-Specific Notes

### macOS
- Metal GPU acceleration available on arm64
- x64 is CPU only
- Default variant includes Metal support

### Windows
- CPU, Vulkan, and CUDA (x64 only) variants available
- Requires Visual Studio build tools

### Linux
- CPU, Vulkan, and CUDA variants available
- CUDA support: x86_64 (capability 8.9), arm64 (capability 8.7)

### Web
- CPU WASM is available in `@fugood/node-llama-wasm`
- CPU pthreads require `SharedArrayBuffer` and `crossOriginIsolated`
- WebGPU is opt-in with `n_gpu_layers > 0`
- The browser package preserves the Node API shape where possible but uses URLs, Blobs, ArrayBuffers, and the Emscripten virtual filesystem instead of native file paths
