# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

llama.node is a Node.js binding for llama.cpp, designed to maintain API compatibility with llama.rn (the React Native binding). This project enables local LLM inference in Node.js with support for:

- Multiple platforms: macOS (arm64/x64), Windows (x64/arm64), Linux (x64/arm64)
- GPU acceleration: Metal (macOS), Vulkan, CUDA
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

### Build System

- **CMakeLists.txt**: Builds native addon, links llama.cpp and rn-llama sources
- **src/llama.cpp/**: llama.cpp submodule (upstream library)
- **src/llama.rn/**: llama.rn submodule (shared React Native implementation)
- Platform-specific prebuilt packages: `@fugood/node-llama-{platform}-{arch}[-variant]`

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

# Type checking
npm run typecheck

# Clean build artifacts
npm run clean

# Regenerate llama.cpp.patch from current submodule changes
./scripts/regenerate-patch.sh
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

### Session Management

Save/load KV cache state for context persistence:
```javascript
await context.saveSession('session.bin')
await context.loadSession('session.bin')
```

Note: Vulkan backend does not support session save/load.

## Testing Considerations

- Test models are cached in `test/*.gguf` (downloaded on first test run)
- Models used: tiny-random-llama, flan-t5-small, bge-small-en, Qwen3-0.6B
- Tests verify: completion, streaming, tokenization, embeddings, multimodal, TTS
- Use `vocab_only: true` for fast model info/tokenization tests without loading full model

## Debugging

### Common Issues

1. **Build failures**: Ensure submodules are initialized (`git submodule update --init --recursive`)
2. **Missing native module**: Run `npm run copy-rn-llama-source` then rebuild
3. **GPU not working**: Check `lib_variant` matches your hardware (vulkan/cuda)
4. **Session load fails**: Vulkan doesn't support sessions; use default or cuda variant
5. **Context shifting**: Disable with `ctx_shift: false` if using multimodal

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
