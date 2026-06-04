const MAX_WASM_MODEL_BYTES = 2 * 1024 * 1024 * 1024

export const WASM_MODEL_SIZE_LIMIT = MAX_WASM_MODEL_BYTES
export const WASM_DOWNLOAD_CACHE_NAME = 'llama-node-wasm-downloads-v1'

export const WASM_CONFIG_PATHS = {
  js: new URL('./wasm/llama-node.js', import.meta.url).href,
  wasm: new URL('./wasm/llama-node.wasm', import.meta.url).href,
  threadsJs: new URL('./wasm/llama-node.threads.js', import.meta.url).href,
  threadsWasm: new URL('./wasm/llama-node.threads.wasm', import.meta.url).href,
  webgpuJs: new URL('./wasm/llama-node.webgpu.js', import.meta.url).href,
  webgpuWasm: new URL('./wasm/llama-node.webgpu.wasm', import.meta.url).href,
  worker: new URL('./worker.js', import.meta.url).href,
}

const mods = new WeakMap()
const logListeners = []
let logEnabled = false
const defaultModulePromises = new Map()
const downloadPromises = new Map()
let nextFileId = 0
let nextParallelRequestId = 1
let nextWorkerRequestId = 1

const modelTooLargeMessage =
  'Model file is too large for the WebAssembly ArrayBuffer limit (2GB). Split the GGUF into smaller shards, preferably <=512MB each, and load the first split URL.'
const MTMD_DEFAULT_MEDIA_MARKER = '<__media__>'
const ENVIRONMENT_IS_WORKER =
  typeof DedicatedWorkerGlobalScope !== 'undefined' &&
  typeof self !== 'undefined' &&
  self instanceof DedicatedWorkerGlobalScope

const unsupported = (feature) => {
  throw new Error(`${feature} is not available in the llama.node WASM package yet`)
}

const writeConsoleLog = (text) => {
  if (typeof console !== 'undefined' && typeof console.log === 'function') {
    console.log(text)
  }
}

const emitNativeLog = (level, text) => {
  if (!logEnabled) return
  for (const listener of logListeners.slice()) {
    listener(level, text)
  }
}

export const isWebGpuSupported = () =>
  typeof navigator !== 'undefined' &&
  !!navigator.gpu &&
  typeof WebAssembly !== 'undefined' &&
  typeof WebAssembly.promising === 'function' &&
  typeof WebAssembly.Suspending === 'function'

export const isWasmWorkerSupported = () =>
  typeof Worker !== 'undefined' && !ENVIRONMENT_IS_WORKER

export const isWasmThreadsSupported = () =>
  typeof SharedArrayBuffer !== 'undefined' &&
  typeof Atomics !== 'undefined' &&
  typeof crossOriginIsolated !== 'undefined' &&
  crossOriginIsolated

export const isWasmDownloadCacheSupported = () =>
  typeof caches !== 'undefined' &&
  typeof Request !== 'undefined' &&
  typeof Response !== 'undefined'

export const clearWasmDownloadCache = async (
  cacheName = WASM_DOWNLOAD_CACHE_NAME,
) => {
  if (!isWasmDownloadCacheSupported()) return false
  return caches.delete(cacheName)
}

const defaultCpuThreadCount = (options = {}) => {
  const requestedMax = Number(options.wasm?.maxThreads || 4)
  const maxThreads =
    Number.isFinite(requestedMax) && requestedMax > 0
      ? Math.floor(requestedMax)
      : 4
  const hardwareThreads =
    typeof navigator !== 'undefined' && navigator.hardwareConcurrency
      ? navigator.hardwareConcurrency
      : maxThreads
  return Math.max(1, Math.min(maxThreads, hardwareThreads))
}

const resolveRuntimeOptions = (options = {}) => {
  const useWebGpu = (options.n_gpu_layers ?? 0) > 0
  const requestedThreads = options.wasm?.threads
  const useThreads =
    !useWebGpu &&
    requestedThreads !== false &&
    isWasmThreadsSupported()
  const nThreads =
    options.n_threads ??
    (useThreads ? defaultCpuThreadCount(options) : 1)
  return {
    useWebGpu,
    useThreads,
    nThreads,
  }
}

const isArrayBufferLike = (value) =>
  value instanceof ArrayBuffer ||
  (ArrayBuffer.isView(value) && value.buffer instanceof ArrayBuffer)

const toUint8Array = async (source) => {
  if (source instanceof Uint8Array) return source
  if (ArrayBuffer.isView(source)) {
    return new Uint8Array(source.buffer, source.byteOffset, source.byteLength)
  }
  if (source instanceof ArrayBuffer) return new Uint8Array(source)
  if (typeof Blob !== 'undefined' && source instanceof Blob) {
    assertSupportedSize(source.size)
    return new Uint8Array(await source.arrayBuffer())
  }
  throw new TypeError('Expected URL, Blob, ArrayBuffer, or typed array')
}

const assertSupportedSize = (size) => {
  if (Number.isFinite(size) && size >= MAX_WASM_MODEL_BYTES) {
    throw new Error(modelTooLargeMessage)
  }
}

const sanitizeFileName = (name, fallback) => {
  const clean = String(name || fallback)
    .split(/[?#]/)[0]
    .split('/')
    .pop()
    ?.replace(/[^a-zA-Z0-9._-]/g, '_')
  return clean || fallback
}

const ensureDir = (FS, dir) => {
  const parts = dir.split('/').filter(Boolean)
  let current = ''
  for (const part of parts) {
    current += `/${part}`
    try {
      FS.mkdir(current)
    } catch (error) {
      if (error?.code !== 'EEXIST' && !String(error).includes('File exists')) {
        try {
          FS.stat(current)
        } catch {
          throw error
        }
      }
    }
  }
}

const writeVirtualFile = (mod, path, bytes) => {
  const FS = mod.FS
  ensureDir(FS, path.slice(0, path.lastIndexOf('/')) || '/')
  try {
    FS.unlink(path)
  } catch {
    // no existing file
  }
  FS.writeFile(path, bytes)
  return path
}

const readVirtualFileAsArrayBuffer = (mod, path) => {
  const bytes = mod.FS.readFile(path)
  return bytes.buffer.slice(bytes.byteOffset, bytes.byteOffset + bytes.byteLength)
}

const virtualFileExists = (mod, path) => {
  if (typeof path !== 'string' || !path.startsWith('/')) return false
  try {
    mod.FS.stat(path)
    return true
  } catch {
    return false
  }
}

const resolveDownloadCacheKey = (url, options = {}) => {
  if (options.cacheDownloads === false || !isWasmDownloadCacheSupported()) {
    return ''
  }
  try {
    const base = typeof location !== 'undefined' ? location.href : import.meta.url
    const resolved = new URL(url, base)
    if (resolved.protocol !== 'http:' && resolved.protocol !== 'https:') {
      return ''
    }
    return resolved.href
  } catch {
    return ''
  }
}

const getDownloadCacheName = (options = {}) =>
  options.cacheName || WASM_DOWNLOAD_CACHE_NAME

const reportProgress = (onProgress, progress, detail = {}) => {
  onProgress?.(progress, { progress, ...detail })
}

const readResponseBytes = async (
  response,
  url,
  onProgress,
  progressDetail = {},
) => {
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`)
  }

  const total = Number(response.headers.get('content-length') || 0)
  assertSupportedSize(total)
  let lastProgress = -1
  const emitProgress = (progress, loaded = 0) => {
    const rounded = Math.max(0, Math.min(100, Math.round(progress)))
    if (rounded === lastProgress && rounded !== 100) return
    lastProgress = rounded
    reportProgress(onProgress, rounded, {
      url,
      bytesLoaded: loaded,
      bytesTotal: total,
      ...progressDetail,
    })
  }

  if (!response.body?.getReader) {
    const bytes = new Uint8Array(await response.arrayBuffer())
    assertSupportedSize(bytes.byteLength)
    emitProgress(100, bytes.byteLength)
    return bytes
  }

  const reader = response.body.getReader()
  const chunks = []
  let loaded = 0

  while (true) {
    const { done, value } = await reader.read()
    if (done) break
    chunks.push(value)
    loaded += value.byteLength
    assertSupportedSize(loaded)
    if (total > 0) emitProgress((loaded / total) * 100, loaded)
  }

  const bytes = new Uint8Array(loaded)
  let offset = 0
  for (const chunk of chunks) {
    bytes.set(chunk, offset)
    offset += chunk.byteLength
  }
  emitProgress(100, loaded)
  return bytes
}

const readCachedResponseBytes = async (response, url, onProgress) => {
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`)
  }

  const total = Number(response.headers.get('content-length') || 0)
  assertSupportedSize(total)
  const bytes = new Uint8Array(await response.arrayBuffer())
  assertSupportedSize(bytes.byteLength)
  reportProgress(onProgress, 100, {
    source: 'cache',
    cached: true,
    url,
    bytesLoaded: bytes.byteLength,
    bytesTotal: total || bytes.byteLength,
  })
  return bytes
}

const cacheResponse = async (cacheName, cacheKey, response) => {
  try {
    const cache = await caches.open(cacheName)
    await cache.put(cacheKey, response)
  } catch {
    // Download caching is best-effort; quota or policy failures should not fail loads.
  }
}

const fetchBytesFromNetwork = async (
  url,
  onProgress,
  cacheName,
  cacheKey,
) => {
  const response = await fetch(url)
  if (!response.ok) {
    throw new Error(`Failed to fetch ${url}: HTTP ${response.status}`)
  }

  const total = Number(response.headers.get('content-length') || 0)
  assertSupportedSize(total)

  const cacheWrite =
    cacheName && cacheKey
      ? cacheResponse(cacheName, cacheKey, response.clone())
      : undefined
  const bytes = await readResponseBytes(response, url, onProgress, {
    source: 'network',
    cached: false,
  })
  await cacheWrite
  return bytes
}

const fetchCachedBytes = async (url, onProgress, options = {}) => {
  const cacheKey = resolveDownloadCacheKey(url, options)
  if (!cacheKey) return fetchBytesFromNetwork(url, onProgress)

  const cacheName = getDownloadCacheName(options)
  const promiseKey = `${cacheName}\n${cacheKey}`
  const existing = downloadPromises.get(promiseKey)
  if (existing) {
    const bytes = await existing
    reportProgress(onProgress, 100, {
      source: 'memory',
      cached: true,
      url,
      bytesLoaded: bytes.byteLength,
      bytesTotal: bytes.byteLength,
    })
    return bytes
  }

  const promise = (async () => {
    try {
      const cache = await caches.open(cacheName)
      const cached = await cache.match(cacheKey)
      if (cached) return readCachedResponseBytes(cached, url, onProgress)
    } catch (error) {
      if (error?.message === modelTooLargeMessage) throw error
    }
    return fetchBytesFromNetwork(url, onProgress, cacheName, cacheKey)
  })()

  downloadPromises.set(promiseKey, promise)
  try {
    return await promise
  } finally {
    downloadPromises.delete(promiseKey)
  }
}

const resolveSourceBytes = async (source, onProgress, downloadOptions = {}) => {
  if (typeof source === 'string') {
    return fetchCachedBytes(source, onProgress, downloadOptions)
  }
  const bytes = await toUint8Array(source)
  assertSupportedSize(bytes.byteLength)
  reportProgress(onProgress, 100, {
    source: 'buffer',
    cached: false,
    bytesLoaded: bytes.byteLength,
    bytesTotal: bytes.byteLength,
  })
  return bytes
}

const writeModelSources = async (
  mod,
  model,
  onProgress,
  downloadOptions = {},
) => {
  const sources = Array.isArray(model) ? model : [model]
  if (sources.length === 0) throw new Error('Model is required')

  const paths = []
  for (let i = 0; i < sources.length; i++) {
    const source = sources[i]
    const bytes = await resolveSourceBytes(
      source,
      (progress, detail = {}) => {
        const base = (i / sources.length) * 100
        const aggregateProgress = Math.round(base + progress / sources.length)
        reportProgress(onProgress, aggregateProgress, {
          ...detail,
          progress: aggregateProgress,
          sourceProgress: progress,
          sourceIndex: i,
          sourceCount: sources.length,
        })
      },
      downloadOptions,
    )
    const fileName =
      typeof source === 'string'
        ? sanitizeFileName(source, `model-${i}.gguf`)
        : sanitizeFileName(source?.name, `model-${nextFileId++}.gguf`)
    paths.push(writeVirtualFile(mod, `/models/${fileName}`, bytes))
  }
  return paths
}

const writeSessionSource = async (mod, source, downloadOptions = {}) => {
  const bytes =
    typeof source === 'string'
      ? await fetchCachedBytes(source, undefined, downloadOptions)
      : await toUint8Array(source)
  const path = `/sessions/session-${nextFileId++}.bin`
  writeVirtualFile(mod, path, bytes)
  return path
}

const isDataMediaUrl = (source) =>
  typeof source === 'string' && /^data:(image|audio)\//.test(source)

const writeVirtualSource = async (
  mod,
  source,
  dir,
  fallbackName,
  downloadOptions = {},
) => {
  if (typeof source === 'string') {
    const maybePath = source.replace(/^file:\/\//, '')
    if (virtualFileExists(mod, maybePath)) return maybePath
    if (isDataMediaUrl(source)) return source
  }

  const bytes = await resolveSourceBytes(source, undefined, downloadOptions)
  const fileName =
    typeof source === 'string'
      ? sanitizeFileName(source, fallbackName)
      : sanitizeFileName(source?.name, fallbackName)
  return writeVirtualFile(mod, `${dir}/${fileName}`, bytes)
}

const writeProjectorSource = (mod, source, downloadOptions = {}) =>
  writeVirtualSource(
    mod,
    source,
    '/mmproj',
    `mmproj-${nextFileId++}.gguf`,
    downloadOptions,
  )

const writeMediaSources = async (
  mod,
  mediaPaths = [],
  downloadOptions = {},
) => {
  const sources = Array.isArray(mediaPaths) ? mediaPaths : [mediaPaths]
  const paths = []
  for (const source of sources.filter(Boolean)) {
    paths.push(
      await writeVirtualSource(
        mod,
        source,
        '/media',
        `media-${nextFileId++}`,
        downloadOptions,
      ),
    )
  }
  return paths
}

const formatMediaChat = (messages) => {
  if (!messages) return { messages, has_media: false, media_paths: [] }

  const mediaPaths = []
  const transformed = messages.map((message) => {
    if (!Array.isArray(message.content)) return message

    return {
      ...message,
      content: message.content.map((part) => {
        if (part?.type === 'image_url') {
          const path = part.image_url?.url || part.url || ''
          if (path) mediaPaths.push(path)
          return { type: 'text', text: MTMD_DEFAULT_MEDIA_MARKER }
        }

        if (part?.type === 'input_audio') {
          const audio = part.input_audio
          if (!audio) throw new Error('input_audio is required')
          if (audio.format && audio.format !== 'wav' && audio.format !== 'mp3') {
            throw new Error(`Unsupported audio format: ${audio.format}`)
          }
          const path = audio.url || audio.data || ''
          if (path) mediaPaths.push(path)
          return { type: 'text', text: MTMD_DEFAULT_MEDIA_MARKER }
        }

        return part
      }),
    }
  })

  return {
    messages: transformed,
    has_media: mediaPaths.length > 0,
    media_paths: mediaPaths,
  }
}

const parseActionResult = (raw) => {
  const result = JSON.parse(raw)
  if (!result.success) {
    throw new Error(result.error || 'llama.node WASM action failed')
  }
  delete result.success
  return result
}

const callAction = (mod, action, payload = {}, options) =>
  mod.ccall(
    'llama_node_wasm_action',
    'string',
    ['string', 'string'],
    [action, JSON.stringify(payload)],
    options,
  )

const callSyncAction = (mod, action, payload = {}) =>
  mod.ccall(
    mod._llama_node_wasm_action_sync
      ? 'llama_node_wasm_action_sync'
      : 'llama_node_wasm_action',
    'string',
    ['string', 'string'],
    [action, JSON.stringify(payload)],
  )

const createAction = (mod) => (action, payload = {}) =>
  parseActionResult(callSyncAction(mod, action, payload))

const createAsyncAction = (mod) => async (action, payload = {}) =>
  parseActionResult(await callAction(mod, action, payload, { async: true }))

const startLlamaModule = async (mod, webgpu) => {
  const startResult = webgpu
    ? await mod.ccall('llama_node_wasm_start', 'string', [], [], { async: true })
    : mod.ccall('llama_node_wasm_start', 'string', [], [])
  parseActionResult(startResult)
}

export const initLlama = async (options = {}) => {
  const paths = options.webgpu
    ? {
        js: options.jsPath || WASM_CONFIG_PATHS.webgpuJs,
        wasm: options.wasmPath || WASM_CONFIG_PATHS.webgpuWasm,
      }
    : options.threads
      ? {
          js: options.jsPath || WASM_CONFIG_PATHS.threadsJs,
          wasm: options.wasmPath || WASM_CONFIG_PATHS.threadsWasm,
        }
    : {
        js: options.jsPath || WASM_CONFIG_PATHS.js,
        wasm: options.wasmPath || WASM_CONFIG_PATHS.wasm,
      }
  if (options.module) {
    await startLlamaModule(options.module, options.webgpu)
    return options.module
  }
  if (options.moduleFactory) {
    const mod = await options.moduleFactory(options.moduleOptions || {})
    await startLlamaModule(mod, options.webgpu)
    return mod
  }

  const cacheKey = options.webgpu ? 'webgpu' : options.threads ? 'cpu-threads' : 'cpu'
  if (!defaultModulePromises.has(cacheKey)) {
    defaultModulePromises.set(cacheKey, (async () => {
      const factory = (await import(paths.js)).default
      const moduleOptions = {
        ...(options.moduleOptions || {}),
        locateFile: (file) => {
          if (file.endsWith('.wasm')) return paths.wasm
          return file
        },
      }
      const userPrint = moduleOptions.print
      const userPrintErr = moduleOptions.printErr
      moduleOptions.print = (text) => {
        const logText = String(text)
        emitNativeLog('info', logText)
        if (typeof userPrint === 'function') {
          userPrint(text)
        } else {
          writeConsoleLog(text)
        }
      }
      moduleOptions.printErr = (text) => {
        const logText = String(text)
        emitNativeLog('info', logText)
        if (typeof userPrintErr === 'function') {
          userPrintErr(text)
        } else {
          writeConsoleLog(text)
        }
      }
      if (options.threads && !moduleOptions.mainScriptUrlOrBlob) {
        moduleOptions.mainScriptUrlOrBlob = paths.js
      }
      const mod = await factory(moduleOptions)
      await startLlamaModule(mod, options.webgpu)
      return mod
    })())
  }

  return defaultModulePromises.get(cacheKey)
}

const shouldUseWorkerRuntime = (options = {}) => {
  const wasm = options.wasm || {}
  if (wasm.worker === false) return false
  if (wasm.module || wasm.moduleFactory) return false
  return isWasmWorkerSupported()
}

const serializeError = (error) => ({
  message: error?.message || String(error),
  stack: error?.stack,
  name: error?.name,
})

const deserializeError = (value) => {
  const error = new Error(value?.message || 'llama.node WASM worker failed')
  if (value?.name) error.name = value.name
  if (value?.stack) error.stack = value.stack
  return error
}

const transferablesForValue = (value) => {
  const transfers = []
  const visit = (item) => {
    if (!item) return
    if (item instanceof ArrayBuffer) {
      transfers.push(item)
      return
    }
    if (ArrayBuffer.isView(item)) {
      transfers.push(item.buffer)
      return
    }
    if (Array.isArray(item)) {
      for (const child of item) visit(child)
      return
    }
    if (typeof item === 'object' && typeof Blob !== 'undefined' && item instanceof Blob) {
      return
    }
    if (typeof item === 'object') {
      for (const child of Object.values(item)) visit(child)
    }
  }
  visit(value)
  return [...new Set(transfers)]
}

const createWorker = (options = {}) => {
  const workerPath = options.wasm?.workerPath || WASM_CONFIG_PATHS.worker
  return new Worker(workerPath, {
    type: 'module',
    name: 'llama.node-wasm',
  })
}

class LlamaWorkerClient {
  constructor(options = {}) {
    this.worker = createWorker(options)
    this.requests = new Map()
    this.worker.addEventListener('message', (event) => {
      const message = event.data || {}
      const request = this.requests.get(message.id)
      if (!request) return

      if (message.type === 'progress') {
        request.onProgress?.(message.progress, message.detail)
        return
      }
      if (message.type === 'token') {
        request.onToken?.(message.token)
        return
      }

      this.requests.delete(message.id)
      if (message.type === 'error') {
        request.reject(deserializeError(message.error))
      } else {
        request.resolve(message.result)
      }
    })
    this.worker.addEventListener('error', (event) => {
      const error = new Error(event.message || 'llama.node WASM worker error')
      for (const request of this.requests.values()) request.reject(error)
      this.requests.clear()
    })
  }

  request(method, args = [], options = {}) {
    const id = nextWorkerRequestId++
    return new Promise((resolve, reject) => {
      this.requests.set(id, {
        resolve,
        reject,
        onProgress: options.onProgress,
        onToken: options.onToken,
      })
      const payload = { id, type: 'request', method, args }
      this.worker.postMessage(payload, transferablesForValue(args))
    })
  }

  terminate() {
    this.worker.terminate()
    for (const request of this.requests.values()) {
      request.reject(new Error('llama.node WASM worker terminated'))
    }
    this.requests.clear()
  }
}

const nowMs = () =>
  typeof performance !== 'undefined' && performance.now
    ? performance.now()
    : Date.now()
const defer =
  typeof queueMicrotask === 'function'
    ? queueMicrotask
    : (callback) => Promise.resolve().then(callback)

export class LlamaParallelAPI {
  constructor(context) {
    this.context = context
    this.enabled = false
    this.nParallel = 1
    this.queue = []
    this.active = null
    this.subscribers = new Map()
    this.nextSubscriberId = 1
  }

  async enable(config = {}) {
    this.enabled = true
    this.nParallel = 1
    this.notifyStatus()
    return true
  }

  disable() {
    for (const request of this.queue.splice(0)) {
      this.rejectRequest(request, new Error('Parallel mode disabled'))
    }
    if (this.active) {
      this.active.cancelled = true
      this.rejectRequest(this.active, new Error('Parallel mode disabled'))
    }
    this.enabled = false
    this.notifyStatus()
  }

  async configure(config = {}) {
    return this.enable(config)
  }

  isEnabled() {
    return this.enabled
  }

  getStatus() {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    const requests = []
    if (this.active && !this.active.cancelled) {
      requests.push(this.statusForRequest(this.active))
    }
    for (const request of this.queue) {
      if (!request.cancelled) requests.push(this.statusForRequest(request))
    }

    return {
      n_parallel: this.nParallel,
      active_slots: this.active && !this.active.cancelled ? 1 : 0,
      queued_requests: this.queue.filter((request) => !request.cancelled).length,
      requests,
    }
  }

  subscribeStatus() {
    return this.subscribeToStatus(...arguments)
  }

  subscribeToStatus(callback) {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }
    const subscriberId = this.nextSubscriberId++
    this.subscribers.set(subscriberId, callback)
    callback(this.getStatus())
    return {
      remove: () => {
        this.subscribers.delete(subscriberId)
      },
    }
  }

  async completion(options, onToken) {
    const queued = this.enqueue('completion', async (request) => {
      const started = nowMs()
      let firstTokenAt = 0
      const result = await this.context.completion(options, (token) => {
        if (request.cancelled) return
        if (!firstTokenAt) {
          firstTokenAt = nowMs()
          request.state = 'generating'
          request.prompt_ms = firstTokenAt - started
        }
        request.tokens_generated += 1
        request.generation_ms = nowMs() - firstTokenAt
        request.tokens_per_second =
          request.generation_ms > 0
            ? (request.tokens_generated / request.generation_ms) * 1000
            : 0
        this.notifyStatus()
        onToken?.(request.requestId, { requestId: request.requestId, ...token })
      })

      const ended = nowMs()
      if (!firstTokenAt) {
        request.prompt_ms = ended - started
        request.generation_ms = 0
      } else {
        request.generation_ms = ended - firstTokenAt
      }
      request.tokens_per_second =
        request.generation_ms > 0
          ? (request.tokens_generated / request.generation_ms) * 1000
          : 0

      return {
        requestId: request.requestId,
        ...result,
        stopped_limit: result.stopped_limited,
        stopped_word: result.stopped_words,
      }
    })
    return queued
  }

  async embedding(text, params) {
    const { requestId, promise } = this.enqueue('embedding', () =>
      this.context.embedding(text, params),
    )
    return { requestId, promise }
  }

  async rerank(query, documents, params) {
    const { requestId, promise } = this.enqueue('rerank', () =>
      this.context.rerank(query, documents, params),
    )
    return { requestId, promise }
  }

  cancelRequest(requestId) {
    const queuedIndex = this.queue.findIndex(
      (request) => request.requestId === requestId,
    )
    if (queuedIndex >= 0) {
      const [request] = this.queue.splice(queuedIndex, 1)
      request.cancelled = true
      this.rejectRequest(request, new Error('Request cancelled'))
      this.notifyStatus()
      return
    }

    if (this.active?.requestId === requestId) {
      this.active.cancelled = true
      this.rejectRequest(this.active, new Error('Request cancelled'))
      this.notifyStatus()
    }
  }

  enqueue(type, run) {
    if (!this.enabled) {
      throw new Error('Parallel mode is not enabled. Call enable() first.')
    }

    const request = {
      requestId: nextParallelRequestId++,
      type,
      state: 'queued',
      prompt_length: 0,
      tokens_generated: 0,
      prompt_ms: 0,
      generation_ms: 0,
      tokens_per_second: 0,
      cancelled: false,
      settled: false,
      run,
      resolve: null,
      reject: null,
    }

    const promise = new Promise((resolve, reject) => {
      request.resolve = resolve
      request.reject = reject
    })

    this.queue.push(request)
    this.notifyStatus()
    this.pump()

    return {
      requestId: request.requestId,
      promise,
      stop: () => this.cancelRequest(request.requestId),
    }
  }

  pump() {
    if (!this.enabled || this.active || this.queue.length === 0) return

    const request = this.queue.shift()
    if (request.cancelled) {
      this.pump()
      return
    }

    this.active = request
    request.state = request.type === 'completion' ? 'processing_prompt' : 'generating'
    this.notifyStatus()

    defer(async () => {
      try {
        const result = await request.run(request)
        request.state = 'done'
        if (!request.cancelled) this.resolveRequest(request, result)
      } catch (error) {
        this.rejectRequest(request, error)
      } finally {
        if (this.active === request) this.active = null
        this.notifyStatus()
        this.pump()
      }
    })
  }

  resolveRequest(request, value) {
    if (request.settled) return
    request.settled = true
    request.resolve(value)
  }

  rejectRequest(request, error) {
    if (request.settled) return
    request.settled = true
    request.reject(error)
  }

  statusForRequest(request) {
    return {
      request_id: request.requestId,
      type: request.type,
      state: request.state,
      prompt_length: request.prompt_length,
      tokens_generated: request.tokens_generated,
      prompt_ms: request.prompt_ms,
      generation_ms: request.generation_ms,
      tokens_per_second: request.tokens_per_second,
    }
  }

  notifyStatus() {
    if (!this.enabled || this.subscribers.size === 0) return
    let status
    try {
      status = this.getStatus()
    } catch {
      return
    }
    for (const callback of this.subscribers.values()) {
      try {
        callback(status)
      } catch {
        // Subscriber errors should not interrupt queued inference.
      }
    }
  }
}

export class LlamaContextWrapper {
  constructor(mod, loadResult, loadOptions = {}, downloadOptions = {}) {
    this.isWorkerRuntime = false
    this.mod = mod
    this.action = createAction(mod)
    this.actionAsync = createAsyncAction(mod)
    this.modelInfo = loadResult.modelInfo
    this.systemInfo = loadResult.systemInfo || ''
    this.loadOptions = loadOptions
    this.downloadOptions = downloadOptions
    this.parallel = new LlamaParallelAPI(this)
    mods.set(this, mod)
  }

  getSystemInfo() {
    return this.systemInfo
  }

  getModelInfo() {
    return this.modelInfo
  }

  getUsedDevices() {
    const devices = ['CPU']
    if ((this.loadOptions.n_gpu_layers ?? 0) > 0) devices.unshift('WebGPU')
    return devices
  }

  isJinjaSupported() {
    const jinja = this.modelInfo?.chatTemplates?.jinja
    return !!jinja?.toolUse || !!jinja?.default
  }

  isLlamaChatSupported() {
    return !!this.modelInfo?.chatTemplates?.llamaChat
  }

  getFormattedChat(messages, template, params = {}) {
    const media = formatMediaChat(messages)
    const forcePureContent = params.force_pure_content ?? false
    const hasChatTemplate = !!this.modelInfo?.metadata?.['tokenizer.chat_template']
    const useJinja =
      (forcePureContent ? hasChatTemplate : this.isJinjaSupported()) &&
      (params.jinja ?? true)
    const result = this.action('formatted_chat', {
      messages: media.messages,
      template,
      params: { ...params, jinja: useJinja },
    })
    if (result.type === 'llama-chat') {
      return {
        type: 'llama-chat',
        prompt: result.prompt,
        has_media: media.has_media,
        media_paths: media.media_paths,
      }
    }
    delete result.type
    return {
      type: 'jinja',
      has_media: media.has_media,
      media_paths: media.media_paths,
      ...result,
    }
  }

  async getFormattedChatAsync(messages, template, params = {}) {
    return this.getFormattedChat(messages, template, params)
  }

  async completion(options, callback) {
    const completionOptions = { ...options }
    if (completionOptions.messages) {
      const media = formatMediaChat(completionOptions.messages)
      completionOptions.messages = media.messages
      if (media.has_media) {
        completionOptions.media_paths = [
          ...(Array.isArray(completionOptions.media_paths)
            ? completionOptions.media_paths
            : completionOptions.media_paths
              ? [completionOptions.media_paths]
              : []),
          ...media.media_paths,
        ]
      }
    }
    if (completionOptions.media_paths) {
      completionOptions.media_paths = await writeMediaSources(
        this.mod,
        completionOptions.media_paths,
        this.downloadOptions,
      )
    }
    if (completionOptions.messages && completionOptions.jinja == null) {
      const forcePureContent = completionOptions.force_pure_content ?? false
      const hasChatTemplate = !!this.modelInfo?.metadata?.['tokenizer.chat_template']
      completionOptions.jinja =
        forcePureContent ? hasChatTemplate : this.isJinjaSupported()
    }

    const result = await this.actionAsync('completion', completionOptions)
    const tokens = result.tokens || []
    const probabilities = result.completion_probabilities || []
    delete result.tokens

    if (callback) {
      let accumulated = ''
      for (let i = 0; i < tokens.length; i++) {
        const token = tokens[i]
        accumulated += token
        const event = { token, accumulated_text: accumulated }
        if (probabilities[i]) {
          event.completion_probabilities = [probabilities[i]]
        }
        callback(event)
      }
    }

    return result
  }

  stopCompletion() {
    unsupported('stopCompletion')
  }

  async tokenize(text, { media_paths } = {}) {
    const stagedMediaPaths = media_paths
      ? await writeMediaSources(this.mod, media_paths, this.downloadOptions)
      : media_paths
    const result = this.action('tokenize', { text, media_paths: stagedMediaPaths })
    return {
      ...result,
      tokens: Int32Array.from(result.tokens || []),
    }
  }

  async detokenize(tokens) {
    return this.action('detokenize', { tokens: Array.from(tokens) }).text
  }

  async embedding(text, params = {}) {
    return this.actionAsync('embedding', { text, ...params })
  }

  async rerank(query, documents, params = {}) {
    const result = await this.actionAsync('rerank', { query, documents, ...params })
    return (result.results || [])
      .map((item) => ({
        index: item.index,
        score: item.score,
        document: documents[item.index],
      }))
      .sort((a, b) => b.score - a.score)
  }

  async saveSession() {
    const path = `/sessions/session-${nextFileId++}.bin`
    ensureDir(this.mod.FS, '/sessions')
    await this.actionAsync('save_session', { path })
    const buffer = readVirtualFileAsArrayBuffer(this.mod, path)
    try {
      this.mod.FS.unlink(path)
    } catch {
      // already removed
    }
    return buffer
  }

  async loadSession(source) {
    const path = await writeSessionSource(this.mod, source, this.downloadOptions)
    try {
      await this.actionAsync('load_session', { path })
    } finally {
      try {
        this.mod.FS.unlink(path)
      } catch {
        // keep going
      }
    }
  }

  async release() {
    await this.actionAsync('release')
  }

  applyLoraAdapters(adapters) {
    this.action('apply_lora_adapters', { adapters })
  }
  removeLoraAdapters() {
    this.action('remove_lora_adapters')
  }
  getLoadedLoraAdapters() {
    return this.action('get_loaded_lora_adapters').lora_adapters || []
  }
  async initMultimodal(options) {
    if (!options?.path) throw new Error('mmproj path is required')
    const path = await writeProjectorSource(
      this.mod,
      options?.path,
      this.downloadOptions,
    )
    const result = await this.actionAsync('init_multimodal', {
      ...options,
      path,
      use_gpu: options?.use_gpu ?? (this.loadOptions.n_gpu_layers ?? 0) > 0,
    })
    return !!result.support
  }
  isMultimodalEnabled() {
    return !!this.action('is_multimodal_enabled').enabled
  }
  releaseMultimodal() {
    this.action('release_multimodal')
  }
  getMultimodalSupport() {
    return this.action('get_multimodal_support').support || {
      vision: false,
      audio: false,
    }
  }
  initVocoder() {
    unsupported('TTS vocoder')
  }
  releaseVocoder() {}
  isVocoderEnabled() {
    return false
  }
  getFormattedAudioCompletion() {
    unsupported('TTS audio completion')
  }
  getAudioCompletionGuideTokens() {
    unsupported('TTS audio completion')
  }
  decodeAudioTokens() {
    unsupported('TTS audio decoding')
  }
  clearCache(clearData = false) {
    this.action('clear_cache', { clear_data: clearData })
  }
  async bench(pp, tg, pl, nr) {
    return this.actionAsync('bench', { pp, tg, pl, nr })
  }
}

const messageText = (message) => {
  if (typeof message?.content === 'string') return message.content
  if (!Array.isArray(message?.content)) return ''
  return message.content
    .map((part) => {
      if (typeof part?.text === 'string') return part.text
      if (part?.type === 'image_url' || part?.type === 'input_audio') {
        return MTMD_DEFAULT_MEDIA_MARKER
      }
      return ''
    })
    .join('')
}

const formatChatLocally = (modelInfo, messages, template, params = {}) => {
  const media = formatMediaChat(messages)
  const chatTemplate = template || modelInfo?.metadata?.['tokenizer.chat_template'] || ''
  const useGemmaTurns = chatTemplate.includes('<start_of_turn>')
  const prompt = (media.messages || [])
    .map((message) => {
      const role = message.role === 'assistant' ? 'model' : message.role || 'user'
      const content = messageText(message)
      return useGemmaTurns
        ? `<start_of_turn>${role}\n${content}<end_of_turn>\n`
        : `${role}: ${content}\n`
    })
    .join('')
  const generationPrompt =
    params.add_generation_prompt === false
      ? ''
      : useGemmaTurns
        ? '<start_of_turn>model\n'
        : 'assistant: '

  return {
    type: 'jinja',
    has_media: media.has_media,
    media_paths: media.media_paths,
    prompt: prompt + generationPrompt,
    chat_format: useGemmaTurns ? 2 : 0,
    grammar: '',
    grammar_lazy: false,
    grammar_triggers: [],
    generation_prompt: generationPrompt,
    thinking_forced_open: false,
    thinking_start_tag: '',
    thinking_end_tag: '',
    preserved_tokens: [],
    additional_stops: [],
    chat_parser: '{}',
  }
}

class LlamaWorkerContextWrapper {
  constructor(client, loadResult, loadOptions = {}) {
    this.isWorkerRuntime = true
    this.client = client
    this.modelInfo = loadResult.modelInfo
    this.systemInfo = loadResult.systemInfo || ''
    this.loadOptions = loadOptions
    this.parallel = new LlamaParallelAPI(this)
    this.queue = Promise.resolve()
    this.loadedLoraAdapters = []
    this.multimodalEnabled = false
    this.multimodalSupport = { vision: false, audio: false }
  }

  enqueueWorker(method, args = [], options = {}) {
    const task = this.queue.then(() => this.client.request(method, args, options))
    this.queue = task.catch(() => {})
    return task
  }

  queueMutation(method, args = [], update) {
    this.enqueueWorker(method, args)
      .then((result) => update?.(result))
      .catch((error) => {
        this.lastMutationError = error
      })
  }

  async takeQueuedMutationError() {
    await this.queue
    if (this.lastMutationError) {
      const error = this.lastMutationError
      this.lastMutationError = undefined
      throw error
    }
  }

  getSystemInfo() {
    return this.systemInfo
  }

  getModelInfo() {
    return this.modelInfo
  }

  getUsedDevices() {
    const devices = ['CPU']
    if ((this.loadOptions.n_gpu_layers ?? 0) > 0) devices.unshift('WebGPU')
    return devices
  }

  isJinjaSupported() {
    const jinja = this.modelInfo?.chatTemplates?.jinja
    return !!jinja?.toolUse || !!jinja?.default
  }

  isLlamaChatSupported() {
    return !!this.modelInfo?.chatTemplates?.llamaChat
  }

  getFormattedChat(messages, template, params = {}) {
    return formatChatLocally(this.modelInfo, messages, template, params)
  }

  async getFormattedChatAsync(messages, template, params = {}) {
    await this.takeQueuedMutationError()
    const media = formatMediaChat(messages)
    const forcePureContent = params.force_pure_content ?? false
    const hasChatTemplate = !!this.modelInfo?.metadata?.['tokenizer.chat_template']
    const useJinja =
      (forcePureContent ? hasChatTemplate : this.isJinjaSupported()) &&
      (params.jinja ?? true)
    const result = await this.enqueueWorker('getFormattedChat', [
      media.messages,
      template,
      { ...params, jinja: useJinja },
    ])
    if (result.type === 'llama-chat') {
      return {
        type: 'llama-chat',
        prompt: result.prompt,
        has_media: media.has_media,
        media_paths: media.media_paths,
      }
    }
    delete result.type
    return {
      type: 'jinja',
      has_media: media.has_media,
      media_paths: media.media_paths,
      ...result,
    }
  }

  async completion(options, callback) {
    await this.takeQueuedMutationError()
    const completionOptions = { ...options }
    if (completionOptions.messages) {
      const media = formatMediaChat(completionOptions.messages)
      completionOptions.messages = media.messages
      if (media.has_media) {
        completionOptions.media_paths = [
          ...(Array.isArray(completionOptions.media_paths)
            ? completionOptions.media_paths
            : completionOptions.media_paths
              ? [completionOptions.media_paths]
              : []),
          ...media.media_paths,
        ]
      }
    }
    if (completionOptions.messages && completionOptions.jinja == null) {
      const forcePureContent = completionOptions.force_pure_content ?? false
      const hasChatTemplate = !!this.modelInfo?.metadata?.['tokenizer.chat_template']
      completionOptions.jinja =
        forcePureContent ? hasChatTemplate : this.isJinjaSupported()
    }
    return this.enqueueWorker('completion', [completionOptions], {
      onToken: callback,
    })
  }

  stopCompletion() {
    unsupported('stopCompletion')
  }

  async tokenize(text, params = {}) {
    await this.takeQueuedMutationError()
    const result = await this.enqueueWorker('tokenize', [text, params])
    return {
      ...result,
      tokens: Int32Array.from(result.tokens || []),
    }
  }

  async detokenize(tokens) {
    await this.takeQueuedMutationError()
    return this.enqueueWorker('detokenize', [Array.from(tokens)])
  }

  async embedding(text, params = {}) {
    await this.takeQueuedMutationError()
    return this.enqueueWorker('embedding', [text, params])
  }

  async rerank(query, documents, params = {}) {
    await this.takeQueuedMutationError()
    return this.enqueueWorker('rerank', [query, documents, params])
  }

  async saveSession() {
    await this.takeQueuedMutationError()
    return this.enqueueWorker('saveSession')
  }

  async loadSession(source) {
    await this.takeQueuedMutationError()
    await this.enqueueWorker('loadSession', [source])
  }

  async release() {
    try {
      await this.enqueueWorker('release')
    } finally {
      this.client.terminate()
    }
  }

  applyLoraAdapters(adapters) {
    this.queueMutation('applyLoraAdapters', [adapters], (result) => {
      this.loadedLoraAdapters = result || []
    })
  }

  removeLoraAdapters() {
    this.queueMutation('removeLoraAdapters', [], (result) => {
      this.loadedLoraAdapters = result || []
    })
  }

  getLoadedLoraAdapters() {
    return this.loadedLoraAdapters
  }

  async initMultimodal(options) {
    await this.takeQueuedMutationError()
    const support = await this.enqueueWorker('initMultimodal', [options])
    this.multimodalEnabled = !!support
    this.multimodalSupport = support || { vision: false, audio: false }
    return this.multimodalEnabled
  }

  isMultimodalEnabled() {
    return this.multimodalEnabled
  }

  releaseMultimodal() {
    this.multimodalEnabled = false
    this.multimodalSupport = { vision: false, audio: false }
    this.queueMutation('releaseMultimodal')
  }

  getMultimodalSupport() {
    return this.multimodalSupport
  }

  initVocoder() {
    unsupported('TTS vocoder')
  }

  releaseVocoder() {}

  isVocoderEnabled() {
    return false
  }

  getFormattedAudioCompletion() {
    unsupported('TTS audio completion')
  }

  getAudioCompletionGuideTokens() {
    unsupported('TTS audio completion')
  }

  decodeAudioTokens() {
    unsupported('TTS audio decoding')
  }

  clearCache(clearData = false) {
    this.queueMutation('clearCache', [clearData])
  }

  async bench(pp, tg, pl, nr) {
    await this.takeQueuedMutationError()
    return this.enqueueWorker('bench', [pp, tg, pl, nr])
  }
}

export const loadModel = async (options, onProgress) => {
  const runtime = resolveRuntimeOptions(options)
  const useWebGpu = runtime.useWebGpu
  const hasWebGpuSupport = isWebGpuSupported()
  if (useWebGpu && !hasWebGpuSupport) {
    throw new Error(
      'WebGPU requires navigator.gpu plus WebAssembly JSPI support (WebAssembly.promising and WebAssembly.Suspending). Disable WebGPU or use a browser with JSPI enabled.',
    )
  }

  if (shouldUseWorkerRuntime(options)) {
    const client = new LlamaWorkerClient(options)
    const loadOptions = {
      ...options,
      n_threads: runtime.nThreads,
      n_gpu_layers: options.n_gpu_layers ?? 0,
      wasm: {
        ...(options.wasm || {}),
        threads: runtime.useThreads,
      },
    }
    const result = await client.request('loadModel', [loadOptions], {
      onProgress,
    })
    return new LlamaWorkerContextWrapper(client, result, loadOptions)
  }

  const wasmOptions = options.wasm || {}
  const mod = await initLlama({
    ...wasmOptions,
    webgpu: useWebGpu,
    threads: runtime.useThreads,
  })
  const modelPaths = await writeModelSources(
    mod,
    options.model,
    onProgress,
    wasmOptions,
  )
  const loadOptions = {
    ...options,
    model: modelPaths[0],
    n_threads: runtime.nThreads,
    n_gpu_layers: options.n_gpu_layers ?? 0,
  }
  if (loadOptions.flash_attn == null && loadOptions.flash_attn_type == null) {
    loadOptions.flash_attn_type = 'off'
  }
  delete loadOptions.lib_variant
  delete loadOptions.wasm

  const result = await createAsyncAction(mod)('load', loadOptions)
  return new LlamaContextWrapper(mod, result, loadOptions, wasmOptions)
}

export const loadLlamaModelInfo = async (model, options = {}) => {
  const context = await loadModel({ ...options, model, vocab_only: true })
  const metadata = context.getModelInfo().metadata || {}
  await context.release()
  return metadata
}

export const getBackendDevicesInfo = async () => [
  {
    name: 'CPU',
    description: 'WebAssembly CPU backend',
    memoryFree: 0,
    memoryTotal: 0,
    type: 'CPU',
    deviceName: 'CPU',
  },
  ...(isWebGpuSupported()
    ? [
        {
          name: 'WebGPU',
          description: 'Browser WebGPU backend',
          memoryFree: 0,
          memoryTotal: 0,
          type: 'GPU',
          deviceName: 'WebGPU',
        },
      ]
    : []),
]

export const toggleNativeLog = async (enable) => {
  logEnabled = enable
}

export function addNativeLogListener(listener) {
  logListeners.push(listener)
  return {
    remove: () => {
      const index = logListeners.indexOf(listener)
      if (index >= 0) logListeners.splice(index, 1)
    },
  }
}

export const isLibVariantAvailable = async (variant) =>
  !variant || variant === 'default' || variant === 'wasm'

export const BuildInfo = {
  number: 'wasm',
  commit: 'wasm',
}

export const createWasmLlama = initLlama
export const isNativeLogEnabled = () => logEnabled

export default {
  initLlama,
  loadModel,
  loadLlamaModelInfo,
  getBackendDevicesInfo,
  toggleNativeLog,
  addNativeLogListener,
  clearWasmDownloadCache,
  isLibVariantAvailable,
  isWebGpuSupported,
  isWasmWorkerSupported,
  isWasmThreadsSupported,
  isWasmDownloadCacheSupported,
  BuildInfo,
}
