export type LibVariant = 'default' | 'wasm'

export type MessagePart = {
  type: string
  text?: string
  image_url?: { url?: string }
  input_audio?: { format: string; data?: string; url?: string }
} & Record<string, any>

export type ChatMessage = {
  role: string
  content?: string | MessagePart[]
}

export type CompletionResponseFormat = {
  type: 'text' | 'json_object' | 'json_schema'
  json_schema?: {
    strict?: boolean
    schema: Record<string, any>
  }
  schema?: Record<string, any>
}

export type Tool = {
  type: 'function'
  function: {
    name: string
    description: string
    parameters: Record<string, any>
  }
}

export type ToolCall = {
  type: 'function'
  function: {
    name: string
    arguments: string
  }
  id?: string
}

export type LlamaModelSource =
  | string
  | Blob
  | ArrayBuffer
  | ArrayBufferView

export type LlamaMediaSource = LlamaModelSource

export type LlamaKvCacheType =
  | 'f16'
  | 'f32'
  | 'q8_0'
  | 'q4_0'
  | 'q4_1'
  | 'iq4_nl'
  | 'q5_0'
  | 'q5_1'

export type LlamaDownloadProgress = {
  progress: number
  source?: 'network' | 'cache' | 'memory' | 'buffer'
  cached?: boolean
  url?: string
  bytesLoaded?: number
  bytesTotal?: number
  sourceProgress?: number
  sourceIndex?: number
  sourceCount?: number
}

export type LlamaModelOptions = {
  model: LlamaModelSource | LlamaModelSource[]
  model_draft?: LlamaModelSource
  draft_model?: LlamaModelSource
  chat_template?: string
  embedding?: boolean
  embd_normalize?: number
  pooling_type?: 'none' | 'mean' | 'cls' | 'last' | 'rank'
  n_ctx?: number
  n_batch?: number
  n_ubatch?: number
  n_parallel?: number
  n_threads?: number
  n_gpu_layers?: number
  flash_attn_type?: 'auto' | 'on' | 'off'
  flash_attn?: boolean
  cache_type_k?: LlamaKvCacheType
  cache_type_v?: LlamaKvCacheType
  ctx_shift?: boolean
  kv_unified?: boolean
  swa_full?: boolean
  /**
   * Memory budget (MiB) for the cross-turn KV prefix cache on recurrent/hybrid
   * models. 0 disables it; no-op on pure-attention models. Default 160.
   */
  state_cache_budget_mb?: number
  /**
   * Max snapshots to keep (secondary cap; the byte budget is primary).
   * 0 = no count cap. Default 8.
   */
  state_cache_max_checkpoints?: number
  use_mmap?: boolean
  vocab_only?: boolean
  rope_freq_base?: number
  rope_freq_scale?: number
  speculative?: LlamaSpeculativeOptions | SpeculativeType | boolean
  spec_type?: SpeculativeType | SpeculativeType[]
  spec_draft_n_max?: number
  spec_draft_n_min?: number
  spec_draft_p_min?: number
  spec_draft_p_split?: number
  spec_draft_n_gpu_layers?: number
  spec_draft_cache_type_k?: LlamaKvCacheType
  spec_draft_cache_type_v?: LlamaKvCacheType
  lib_variant?: LibVariant
  wasm?: {
    jsPath?: string
    wasmPath?: string
    webgpu?: boolean
    worker?: boolean
    workerPath?: string
    threads?: boolean
    maxThreads?: number
    cacheDownloads?: boolean
    cacheName?: string
    module?: any
    moduleFactory?: (options?: Record<string, any>) => Promise<any>
    moduleOptions?: Record<string, any>
  }
}

export type LlamaCompletionOptions = {
  messages?: ChatMessage[]
  jinja?: boolean
  reasoning_format?: 'none' | 'auto' | 'deepseek'
  chat_template?: string
  force_pure_content?: boolean
  response_format?: CompletionResponseFormat
  tools?: Tool[]
  parallel_tool_calls?: boolean
  tool_choice?: string
  now?: string | number
  chat_template_kwargs?: Record<string, string | boolean | number>
  enable_thinking?: boolean
  thinking_forced_open?: boolean
  thinking_budget_tokens?: number
  thinking_budget_message?: string
  thinking_start_tag?: string
  thinking_end_tag?: string
  add_generation_prompt?: boolean
  generation_prompt?: string
  chat_parser?: string
  prompt?: string
  prefill_text?: string
  media_paths?: LlamaMediaSource | LlamaMediaSource[]
  temperature?: number
  top_k?: number
  top_p?: number
  min_p?: number
  mirostat?: number
  mirostat_tau?: number
  mirostat_eta?: number
  penalty_last_n?: number
  penalty_repeat?: number
  penalty_freq?: number
  penalty_present?: number
  typ_p?: number
  typical_p?: number
  xtc_threshold?: number
  xtc_probability?: number
  dry_multiplier?: number
  dry_base?: number
  dry_allowed_length?: number
  dry_penalty_last_n?: number
  dry_sequence_breakers?: string[]
  top_n_sigma?: number
  n_predict?: number
  max_length?: number
  max_tokens?: number
  seed?: number
  stop?: string[]
  grammar?: string
  grammar_lazy?: boolean
  grammar_triggers?: { type: number; value: string; token?: number }[]
  preserved_tokens?: string[]
  json_schema?: string
  logit_bias?: number[][]
  ignore_eos?: boolean
  n_probs?: number
  speculative?: LlamaSpeculativeOptions | SpeculativeType | boolean
  spec_type?: SpeculativeType | SpeculativeType[]
  spec_draft_n_max?: number
  spec_draft_n_min?: number
  spec_draft_p_min?: number
  spec_draft_p_split?: number
}

export type SpeculativeType = 'none' | 'draft-mtp' | 'mtp'

export type LlamaSpeculativeOptions = {
  enabled?: boolean
  type?: SpeculativeType
  types?: SpeculativeType[]
  n_max?: number
  n_min?: number
  p_min?: number
  p_split?: number
  draft?: {
    model?: LlamaModelSource
    path?: LlamaModelSource
    model_draft?: LlamaModelSource
    draft_model?: LlamaModelSource
    n_max?: number
    n_min?: number
    p_min?: number
    p_split?: number
    n_gpu_layers?: number
    cache_type_k?: LlamaKvCacheType
    cache_type_v?: LlamaKvCacheType
  }
}

export type TokenProbability = {
  tok_str: string
  prob: number
}

export type CompletionProbability = {
  content: string
  probs: TokenProbability[]
}

export type LlamaCompletionToken = {
  token: string
  content?: string
  reasoning_content?: string
  tool_calls?: ToolCall[]
  accumulated_text?: string
  completion_probabilities?: CompletionProbability[]
}

export type LlamaCompletionResult = {
  chat_format: number
  tokens_evaluated: number
  tokens_predicted: number
  draft_tokens?: number
  draft_tokens_accepted?: number
  text: string
  content?: string
  reasoning_content?: string
  tool_calls?: ToolCall[]
  completion_probabilities?: CompletionProbability[]
  truncated: boolean
  context_full: boolean
  interrupted: boolean
  stopped_eos: boolean
  stopped_words: boolean
  stopping_word: string
  stopped_limited: boolean
  timings?: Record<string, number>
}

export type LlamaParallelCompletionOptions = LlamaCompletionOptions & {
  load_state_path?: string
  save_state_path?: string
  save_prompt_state_path?: string
  load_state_size?: number
  save_state_size?: number
}

export type ParallelRequestStatus = {
  request_id: number
  type: 'completion' | 'embedding' | 'rerank'
  state: 'queued' | 'processing_prompt' | 'generating' | 'done'
  prompt_length: number
  tokens_generated: number
  prompt_ms: number
  generation_ms: number
  tokens_per_second: number
}

export type ParallelStatus = {
  n_parallel: number
  active_slots: number
  queued_requests: number
  requests: ParallelRequestStatus[]
}

export type LlamaParallelCompletionResult = LlamaCompletionResult & {
  requestId: number
  stopped_limit?: boolean
  stopped_word?: boolean
}

export type TokenizeResult = {
  tokens: Int32Array
  has_media: boolean
  bitmap_hashes?: string[]
  chunk_pos?: number[]
  chunk_pos_media?: number[]
}

export type ModelInfo = {
  desc: string
  nEmbd: number
  nParams: number
  size: number
  is_recurrent: boolean
  is_hybrid: boolean
  chatTemplates: Record<string, any>
  metadata: Record<string, string>
  isChatTemplateSupported: boolean
}

export type FormattedChatResult =
  | {
      type: 'llama-chat'
      prompt: string
      has_media: boolean
      media_paths?: string[]
    }
  | ({
      type: 'jinja'
      has_media: boolean
      media_paths?: string[]
    } & Record<string, any>)

export type LoraAdapter = {
  path: string
  scaled: number
}

export type BenchResult = {
  nKvMax: number
  nBatch: number
  nUBatch: number
  flashAttn: number
  isPpShared: boolean
  nGpuLayers: number
  nThreads: number
  nThreadsBatch: number
  pp: number
  tg: number
  pl: number
  nKv: number
  tPp: number
  speedPp: number
  tTg: number
  speedTg: number
  t: number
  speed: number
}

export declare class LlamaParallelAPI {
  enable(config?: { n_parallel?: number; n_batch?: number }): Promise<boolean>
  disable(): void
  configure(config?: { n_parallel?: number; n_batch?: number }): Promise<boolean>
  completion(
    options: LlamaParallelCompletionOptions,
    onToken?: (
      requestId: number,
      token: LlamaCompletionToken & { requestId: number },
    ) => void,
  ): Promise<{
    requestId: number
    promise: Promise<LlamaParallelCompletionResult>
    stop: () => void
  }>
  embedding(
    text: string,
    params?: { embd_normalize?: number },
  ): Promise<{
    requestId: number
    promise: Promise<{ embedding: number[] }>
  }>
  rerank(
    query: string,
    documents: string[],
    params?: Record<string, any>,
  ): Promise<{
    requestId: number
    promise: Promise<Array<{ index: number; score: number; document: string }>>
  }>
  cancelRequest(requestId: number): void
  isEnabled(): boolean
  getStatus(): ParallelStatus
  subscribeToStatus(callback: (status: ParallelStatus) => void): {
    remove: () => void
  }
  subscribeStatus(callback: (status: ParallelStatus) => void): { remove: () => void }
}

export declare class LlamaContextWrapper {
  isWorkerRuntime: boolean
  parallel: LlamaParallelAPI
  getSystemInfo(): string
  getModelInfo(): ModelInfo
  getUsedDevices(): string[]
  isJinjaSupported(): boolean
  isLlamaChatSupported(): boolean
  getFormattedChat(
    messages: ChatMessage[],
    template?: string,
    params?: Record<string, any>,
  ): FormattedChatResult
  getFormattedChatAsync(
    messages: ChatMessage[],
    template?: string,
    params?: Record<string, any>,
  ): Promise<FormattedChatResult>
  completion(
    options: LlamaCompletionOptions,
    callback?: (token: LlamaCompletionToken) => void,
  ): Promise<LlamaCompletionResult>
  stopCompletion(): void
  tokenize(
    text: string,
    params?: { media_paths?: LlamaMediaSource | LlamaMediaSource[] },
  ): Promise<TokenizeResult>
  detokenize(tokens: number[] | Int32Array): Promise<string>
  embedding(
    text: string,
    params?: { embd_normalize?: number },
  ): Promise<{ embedding: number[] }>
  rerank(
    query: string,
    documents: string[],
    params?: Record<string, any>,
  ): Promise<Array<{ index: number; score: number; document: string }>>
  saveSession(): Promise<ArrayBuffer>
  loadSession(source: string | Blob | ArrayBuffer | ArrayBufferView): Promise<void>
  release(): Promise<void>
  applyLoraAdapters(adapters: LoraAdapter[]): void
  removeLoraAdapters(): void
  getLoadedLoraAdapters(): LoraAdapter[]
  initMultimodal(options: {
    path: LlamaModelSource
    use_gpu?: boolean
    image_min_tokens?: number
    image_max_tokens?: number
  }): Promise<boolean>
  isMultimodalEnabled(): boolean
  releaseMultimodal(): void
  getMultimodalSupport(): {
    vision: boolean
    audio: boolean
  }
  clearCache(clearData?: boolean): void
  bench(pp: number, tg: number, pl: number, nr: number): Promise<BenchResult>
}

export declare const WASM_MODEL_SIZE_LIMIT: number
export declare const WASM_DOWNLOAD_CACHE_NAME: string
export declare const WASM_CONFIG_PATHS: {
  js: string
  wasm: string
  threadsJs: string
  threadsWasm: string
  webgpuJs: string
  webgpuWasm: string
  worker: string
}
export declare function isWebGpuSupported(): boolean
export declare function isWasmWorkerSupported(): boolean
export declare function isWasmThreadsSupported(): boolean
export declare function isWasmDownloadCacheSupported(): boolean
export declare function clearWasmDownloadCache(cacheName?: string): Promise<boolean>

export declare function initLlama(options?: Record<string, any>): Promise<any>
export declare function loadModel(
  options: LlamaModelOptions,
  onProgress?: (progress: number, detail?: LlamaDownloadProgress) => void,
): Promise<LlamaContextWrapper>
export declare function loadLlamaModelInfo(
  model: LlamaModelSource | LlamaModelSource[],
  options?: Partial<LlamaModelOptions>,
): Promise<Record<string, string>>
export declare function getBackendDevicesInfo(): Promise<any[]>
export declare function toggleNativeLog(enable: boolean): Promise<void>
export declare function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void }
export declare function isLibVariantAvailable(variant?: LibVariant): Promise<boolean>
export declare const BuildInfo: { number: string; commit: string }
export declare const createWasmLlama: typeof initLlama
export declare function isNativeLogEnabled(): boolean

declare const defaultExport: {
  initLlama: typeof initLlama
  loadModel: typeof loadModel
  loadLlamaModelInfo: typeof loadLlamaModelInfo
  getBackendDevicesInfo: typeof getBackendDevicesInfo
  toggleNativeLog: typeof toggleNativeLog
  addNativeLogListener: typeof addNativeLogListener
  clearWasmDownloadCache: typeof clearWasmDownloadCache
  isLibVariantAvailable: typeof isLibVariantAvailable
  isNativeLogEnabled: typeof isNativeLogEnabled
  isWebGpuSupported: typeof isWebGpuSupported
  isWasmWorkerSupported: typeof isWasmWorkerSupported
  isWasmThreadsSupported: typeof isWasmThreadsSupported
  isWasmDownloadCacheSupported: typeof isWasmDownloadCacheSupported
  BuildInfo: typeof BuildInfo
}

export default defaultExport
