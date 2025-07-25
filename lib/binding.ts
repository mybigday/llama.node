export type MessagePart = {
  type: string,
  text?: string,
  image_url?: {
    url?: string
  },
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
  n_threads?: number
  n_gpu_layers?: number
  flash_attn?: boolean
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
  use_mlock?: boolean
  use_mmap?: boolean
  vocab_only?: boolean
  lora?: string
  lora_scaled?: number
  lora_list?: { path: string; scaled: number }[]
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
  n_predict?: number
  max_length?: number
  max_tokens?: number
  seed?: number
  stop?: string[]
  grammar?: string
  grammar_lazy?: boolean
  grammar_triggers?: { type: number; value: string; token?: number }[]
  preserved_tokens?: string[]
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
}

export type LlamaCompletionResult = {
  text: string
  reasoning_content?: string
  content?: string
  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
  context_full: boolean
  audio_tokens?: Array<number>
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
  grammea_lazy: boolean
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
  new (options: LlamaModelOptions): LlamaContext
  getSystemInfo(): string
  getModelInfo(): ModelInfo
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
    },
  ): JinjaFormattedChatResult | string
  completion(
    options: LlamaCompletionOptions,
    callback?: (token: LlamaCompletionToken) => void,
  ): Promise<LlamaCompletionResult>
  stopCompletion(): void
  tokenize(text: string, media_paths?: string[]): Promise<TokenizeResult>
  detokenize(tokens: number[]): Promise<string>
  embedding(text: string): Promise<EmbeddingResult>
  rerank(query: string, documents: string[], params?: RerankParams): Promise<RerankResult[]>
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
  initVocoder(options: { path: string, n_batch?: number }): boolean

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
  getFormattedAudioCompletion(speaker: string|null, text: string): {
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
  decodeAudioTokens(tokens: number[]|Int32Array): Promise<Float32Array>

  // static
  loadModelInfo(path: string, skip: string[]): Promise<GGUFModelInfo>
  toggleNativeLog(
    enable: boolean,
    callback: (level: string, text: string) => void,
  ): void
}

export interface Module {
  LlamaContext: LlamaContext
}

export type LibVariant = 'default' | 'vulkan' | 'cuda'

const getPlatformPackageName = (variant?: LibVariant): string => {
  const platform = process.platform
  const arch = process.arch
  const variantSuffix = variant && variant !== 'default' ? `-${variant}` : ''
  return `@fugood/node-llama-${platform}-${arch}${variantSuffix}`
}

const loadPlatformPackage = async (packageName: string): Promise<Module | null> => {
  try {
    return await import(packageName) as Module
  } catch (error) {
    return null
  }
}

export const loadModule = async (variant?: LibVariant): Promise<Module> => {
  let module = await loadPlatformPackage(getPlatformPackageName(variant))
  if (module) {
    return module
  }

  module = await loadPlatformPackage(getPlatformPackageName())
  if (module) {
    console.warn(`Not found package for variant "${variant}", fallback to default`)
    return module
  }

  console.warn(`Not found package for your platform, fallback to local build`)
  // @ts-ignore
  return (await import('../build/Release/index.node')) as Module
}
