import path from 'path'

export type MessagePart = {
  type: string
  text?: string
  image_url?: {
    url?: string
  }
  input_audio?: {
    format: string
    data?: string
    url?: string
  }
}

export type ChatMessage = {
  role: string
  content?: string | MessagePart[]
}

export type LlamaModelOptions = {
  model: string
  chat_template?: string
  embedding?: boolean
  embd_normalize?: number
  pooling_type?: 'none' | 'mean' | 'cls' | 'last' | 'rank'
  n_ctx?: number
  n_batch?: number
  n_ubatch?: number
  /**
   * CPU affinity mask
   * Example: '0xfc'
   */
  cpu_mask?: string
  cpu_strict?: boolean
  /**
   * Number of parallel sequences to support (sets n_seq_max).
   * This determines the maximum number of parallel slots that can be used.
   * Default: 8
   */
  n_parallel?: number
  n_threads?: number
  n_gpu_layers?: number
  flash_attn_type?: 'auto' | 'on' | 'off'
  flash_attn?: boolean // Deprecated: use flash_attn_type instead
  cache_type_k?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
  cache_type_v?:
    | 'f16'
    | 'f32'
    | 'q8_0'
    | 'q4_0'
    | 'q4_1'
    | 'iq4_nl'
    | 'q5_0'
    | 'q5_1'
  /**
   * Enable context shifting to handle prompts larger than context size
   */
  ctx_shift?: boolean
  /**
   * Use a unified buffer across the input sequences when computing the attention.
   * Try to disable when n_seq_max > 1 for improved performance when the sequences do not share a large prefix.
   */
  kv_unified?: boolean
  /**
   * Use full-size SWA cache (https://github.com/ggml-org/llama.cpp/pull/13194#issuecomment-2868343055)
   */
  swa_full?: boolean
  /**
   * Number of layers to keep MoE weights on CPU
   */
  n_cpu_moe?: number
  /**
   * List of device names to use for offloading
   * Device names can be obtained from getBackendDevicesInfo()
   * Example: ['Metal', 'BLAS', 'CPU']
   */
  devices?: string[]
  use_mlock?: boolean
  use_mmap?: boolean
  vocab_only?: boolean
  lora?: string
  lora_scaled?: number
  lora_list?: { path: string; scaled: number }[]
  /**
   * RoPE base frequency, use 0 to use model default (recommended)
   */
  rope_freq_base?: number
  /**
   * RoPE frequency scaling factor, use 0 to use model default (recommended)
   */
  rope_freq_scale?: number
}

export type CompletionResponseFormat = {
  type: 'text' | 'json_object' | 'json_schema'
  json_schema?: {
    strict?: boolean
    schema: Record<string, any>
  }
  schema?: Record<string, any> // for json_object type
}

export type LlamaCompletionOptions = {
  messages?: ChatMessage[]
  jinja?: boolean
  reasoning_format?: string
  chat_template?: string
  response_format?: CompletionResponseFormat
  tools?: Tool[]
  parallel_tool_calls?: boolean
  tool_choice?: string
  enable_thinking?: boolean
  thinking_forced_open?: boolean
  prompt?: string
  /**
   * Text to prefill the response with.
   * This text will be added to the beginning of the generated response.
   */
  prefill_text?: string
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
  /**
   * Path(s) to media file(s) to process before generating text.
   * When provided, the media will be processed and added to the context.
   * Requires multimodal support to be enabled via initMultimodal.
   * Supports both file paths and base64 data URLs.
   */
  media_paths?: string | string[]
  /**
   * Guide tokens to use for audio completion.
   * Help prevent hallucinations by forcing the TTS to use the correct words.
   */
  guide_tokens?: number[] | Int32Array
  /**
   * Number of top token probabilities to return for each generated token.
   * When > 0, completion_probabilities will be included in streaming callbacks and final result.
   */
  n_probs?: number
}

/**
 * Parameters for parallel completion requests (queueCompletion).
 * Extends LlamaCompletionOptions with parallel-mode specific options.
 */
export type LlamaParallelCompletionOptions = LlamaCompletionOptions & {
  /**
   * File path to load session state from before processing.
   * This allows you to resume from a previously saved completion state.
   * Use with `save_state_path` to enable conversation continuity across requests.
   * Example: `'/path/to/session.bin'` or `'file:///path/to/session.bin'`
   */
  load_state_path?: string

  /**
   * File path to save session state to after completion.
   * The session state will be saved to this file path when the completion finishes.
   * You can then pass this path to `load_state_path` in a subsequent request to resume.
   * Example: `'/path/to/session.bin'` or `'file:///path/to/session.bin'`
   */
  save_state_path?: string

  /**
   * Number of tokens to save when saving session state.
   * If not specified or <= 0, all tokens will be saved.
   * Use this to limit the size of saved session files.
   * Example: `512` to save only the last 512 tokens
   */
  save_state_size?: number
}

export type TokenProbability = {
  tok_str: string
  prob: number
}

export type CompletionProbability = {
  content: string
  probs: TokenProbability[]
}

export type LlamaCompletionResult = {
  text: string
  reasoning_content?: string
  content?: string
  chat_format: number
  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
  context_full: boolean
  interrupted: boolean
  audio_tokens?: Array<number>
  completion_probabilities?: CompletionProbability[]
  timings: {
    prompt_n: number
    prompt_ms: number
    prompt_per_token_ms: number
    prompt_per_second: number
    predicted_n: number
    predicted_ms: number
    predicted_per_token_ms: number
    predicted_per_second: number
  }
}

export type LlamaCompletionToken = {
  token: string
  content?: string
  reasoning_content?: string
  tool_calls?: ToolCall[]
  accumulated_text?: string
  completion_probabilities?: CompletionProbability[]
}

/**
 * Result from a parallel completion request (queueCompletion callback).
 * Extends the basic completion result with per-slot timing information.
 */
export type LlamaParallelCompletionResult = {
  requestId: number
  text: string
  reasoning_content?: string
  content?: string
  tool_calls?: ToolCall[]
  chat_format: number
  stopped_eos: boolean
  stopped_limit: boolean
  stopped_word: boolean
  context_full: boolean
  tokens_evaluated: number
  tokens_predicted: number
  timings: {
    cache_n: number
    prompt_n: number
    prompt_ms: number
    prompt_per_token_ms: number
    prompt_per_second: number
    predicted_n: number
    predicted_ms: number
    predicted_per_token_ms: number
    predicted_per_second: number
  }
}

export type TokenizeResult = {
  tokens: Int32Array
  has_media: boolean
  bitmap_hashes: string[]
  chunk_pos: number[]
  chunk_pos_media: number[]
}

export type EmbeddingResult = {
  embedding: Float32Array
}

export type RerankParams = {
  normalize?: number
}

export type RerankResult = {
  score: number
  index: number
}

export type BackendDeviceInfo = {
  backend: string
  type: string
  deviceName: string
  maxMemorySize: number
  metadata?: Record<string, any>
}

export type ModelInfo = {
  desc: string
  nEmbd: number
  nParams: number
  size: number
  chatTemplates: {
    llamaChat: boolean
    minja: {
      default: boolean
      defaultCaps: {
        tools: boolean
        toolCalls: boolean
        toolResponses: boolean
        systemRole: boolean
        parallelToolCalls: boolean
        toolCallId: boolean
      }
      toolUse: boolean
      toolUseCaps?: {
        tools: boolean
        toolCalls: boolean
        toolResponses: boolean
        systemRole: boolean
        parallelToolCalls: boolean
        toolCallId: boolean
      }
    }
  }
  metadata: Record<string, string>
  isChatTemplateSupported: boolean
}

export type GGUFModelInfo = {
  version?: number
  alignment?: number
  data_offset?: number
  [key: string]: string | number | undefined
}

export type FormattedChatResult = {
  type: 'jinja' | 'llama-chat'
  prompt: string
  has_media: boolean
  media_paths?: Array<string>
}

export type JinjaFormattedChatResult = {
  prompt: string
  chat_format: number
  grammar: string
  grammar_lazy: boolean
  grammar_triggers: Array<{
    type: number
    value: string
    token: number
  }>
  thinking_forced_open: boolean
  preserved_tokens: string[]
  additional_stops: string[]
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

export interface LlamaContext {
  new (
    options: LlamaModelOptions,
    onProgress?: (progress: number) => void,
  ): LlamaContext
  getSystemInfo(): string
  getModelInfo(): ModelInfo
  getUsedDevices(): string[]
  getFormattedChat(
    messages: ChatMessage[],
    chat_template?: string,
    params?: {
      jinja?: boolean
      response_format?: CompletionResponseFormat
      tools?: Tool[]
      parallel_tool_calls?: boolean
      tool_choice?: string
      enable_thinking?: boolean
      add_generation_prompt?: boolean
      now?: string | number
      chat_template_kwargs?: Record<string, string>
    },
  ): JinjaFormattedChatResult | string
  completion(
    options: LlamaCompletionOptions,
    callback?: (token: LlamaCompletionToken) => void,
  ): Promise<LlamaCompletionResult>
  stopCompletion(): void
  tokenize(text: string, media_paths?: string[]): Promise<TokenizeResult>
  detokenize(tokens: number[]): Promise<string>
  embedding(
    text: string,
    params?: { embd_normalize?: number },
  ): Promise<EmbeddingResult>
  rerank(
    query: string,
    documents: string[],
    params?: RerankParams,
  ): Promise<RerankResult[]>
  saveSession(path: string): Promise<void>
  loadSession(path: string): Promise<void>
  release(): Promise<void>
  applyLoraAdapters(adapters: { path: string; scaled: number }[]): void
  removeLoraAdapters(): void
  getLoadedLoraAdapters(): { path: string; scaled: number }[]
  /**
   * Initialize multimodal support with a mmproj file
   * @param options Object containing path and optional use_gpu flag
   * @returns boolean indicating if initialization was successful
   */
  initMultimodal(options: { path: string; use_gpu?: boolean }): boolean

  /**
   * Check if multimodal support is enabled
   * @returns boolean indicating if multimodal is enabled
   */
  isMultimodalEnabled(): boolean

  /**
   * Get multimodal support capabilities
   * @returns Object with vision and audio support
   */
  getMultimodalSupport(): {
    vision: boolean
    audio: boolean
  }

  /**
   * Release multimodal support
   */
  releaseMultimodal(): void

  /**
   * Load a vocoder model
   * @param options Object containing path and optional n_batch
   * @returns boolean indicating if loading was successful
   */
  initVocoder(options: { path: string; n_batch?: number }): boolean

  /**
   * Unload the vocoder model
   */
  releaseVocoder(): void

  /**
   * Check if the vocoder model is enabled
   * @returns boolean indicating if the vocoder model is enabled
   */
  isVocoderEnabled(): boolean

  /**
   * Get the formatted prompt for audio completion
   * @param speaker Speaker name or null
   * @param text Text to complete
   * @returns Formatted audio completion
   */
  getFormattedAudioCompletion(
    speaker: string | null,
    text: string,
  ): {
    prompt: string
    grammar?: string
  }

  /**
   * Get guide tokens for audio completion
   * @param text Text to complete
   * @returns Guide tokens
   */
  getAudioCompletionGuideTokens(text: string): Int32Array

  /**
   * Decode audio tokens to audio data
   * @param tokens Tokens to decode
   * @returns Promise resolving to decoded audio tokens
   */
  decodeAudioTokens(tokens: number[] | Int32Array): Promise<Float32Array>

  // Parallel decoding methods

  /**
   * Enable parallel decoding mode
   * @param params Configuration for parallel mode
   * @returns boolean indicating if successful
   */
  enableParallelMode(params: { n_parallel?: number; n_batch?: number }): boolean

  /**
   * Disable parallel decoding mode
   */
  disableParallelMode(): void

  /**
   * Queue a completion request for parallel processing
   * @param options Completion options with parallel-specific state management
   * @param callback Optional callback that receives tokens during generation and final result
   * @returns Object with requestId
   */
  queueCompletion(
    options: LlamaParallelCompletionOptions,
    callback?: (error: any, result: LlamaParallelCompletionResult) => void,
  ): { requestId: number }

  /**
   * Queue an embedding request for parallel processing
   * @param text Text to embed
   * @param params Optional embedding parameters
   * @param callback Optional result callback
   * @returns Object with requestId
   */
  queueEmbedding(
    text: string,
    params?: { embd_normalize?: number },
    callback?: (error: any, result: any) => void,
  ): { requestId: number }

  /**
   * Queue a rerank request for parallel processing
   * @param query Query text
   * @param documents Documents to rank
   * @param params Optional rerank parameters
   * @param callback Optional result callback
   * @returns Object with requestId
   */
  queueRerank(
    query: string,
    documents: string[],
    params?: RerankParams,
    callback?: (error: any, result: any) => void,
  ): { requestId: number }

  /**
   * Cancel a queued request
   * @param requestId Request ID to cancel
   */
  cancelRequest(requestId: number): void

  /**
   * Clear the KV and recurrent caches.
   * This is faster than recreating the context and useful for preventing
   * cache contamination between chat sessions.
   * @param clearData If true, also clears the cache data (default: false)
   */
  clearCache(clearData?: boolean): void

  // static
  loadModelInfo(path: string, skip: string[]): Promise<GGUFModelInfo>
  toggleNativeLog(
    enable: boolean,
    callback: (level: string, text: string) => void,
  ): void
  /**
   * Get information about available backend devices
   * @returns Array of backend device information
   */
  getBackendDevicesInfo(): BackendDeviceInfo[]
}

export interface Module {
  LlamaContext: LlamaContext
}

export type LibVariant = 'default' | 'vulkan' | 'cuda' | 'snapdragon'

const getPlatformPackageName = (variant?: LibVariant): string => {
  const platform = process.platform
  const arch = process.arch
  const variantSuffix = variant && variant !== 'default' ? `-${variant}` : ''
  return `@fugood/node-llama-${platform}-${arch}${variantSuffix}`
}

const loadPlatformPackage = async (
  packageName: string,
): Promise<Module | null> => {
  try {
    return (await import(packageName)) as Module
  } catch (error) {
    return null
  }
}

export const loadModule = async (variant?: LibVariant): Promise<Module> => {
  const packageName = getPlatformPackageName(variant)

  // Set ADSP_LIBRARY_PATH for load HTP libs
  if (variant === 'snapdragon') {
    const adspLibraryPath = process.env.ADSP_LIBRARY_PATH
    if (!adspLibraryPath) {
      try {
        process.env.ADSP_LIBRARY_PATH = path.dirname(
          require.resolve(packageName),
        )
      } catch {
        /* no-op */
      }
    }
    const nDev = process.env.GGML_HEXAGON_NDEV
    if (!nDev) process.env.GGML_HEXAGON_NDEV = '16'
  }

  let module = await loadPlatformPackage(packageName)
  if (module) {
    return module
  }

  module = await loadPlatformPackage(getPlatformPackageName())
  if (module) {
    console.warn(
      `Not found package for variant "${variant}", fallback to default`,
    )
    return module
  }

  console.warn(`Not found package for your platform, fallback to local build`)
  // @ts-ignore
  return (await import('../build/Release/index.node')) as Module
}

export const isLibVariantAvailable = async (
  variant?: LibVariant,
): Promise<boolean> => {
  if (variant && variant !== 'default') {
    const module = await loadPlatformPackage(getPlatformPackageName(variant))
    return module != null
  }

  const defaultModule = await loadPlatformPackage(getPlatformPackageName())
  if (defaultModule) return true

  try {
    // @ts-ignore
    await import('../build/Release/index.node')
    return true
  } catch (error) {
    return false
  }
}
