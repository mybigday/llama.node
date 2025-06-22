import * as path from 'path'


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
    schema: object
  }
  schema?: object // for json_object type
}

export type LlamaCompletionOptions = {
  messages?: ChatMessage[]
  jinja?: boolean
  reasoning_format?: string
  chat_template?: string
  response_format?: CompletionResponseFormat
  tools?: object
  parallel_tool_calls?: boolean
  tool_choice?: string
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
  grammar_triggers?: { type: number; word: string; at_start: boolean }[]
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
  guide_tokens?: Int32Array
}

export type LlamaCompletionResult = {
  text: string
  reasoning_content?: string
  content?: string
  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
  context_full: boolean
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

export interface LlamaContext {
  new (options: LlamaModelOptions): LlamaContext
  getSystemInfo(): string
  getModelInfo(): object
  getFormattedChat(
    messages: ChatMessage[],
    chat_template?: string,
    params?: {
      jinja?: boolean
      response_format?: CompletionResponseFormat
      tools?: object
      parallel_tool_calls?: object
      tool_choice?: string
    },
  ): object | string
  completion(
    options: LlamaCompletionOptions,
    callback?: (token: LlamaCompletionToken) => void,
  ): Promise<LlamaCompletionResult>
  stopCompletion(): void
  tokenize(text: string, media_paths?: string[]): Promise<TokenizeResult>
  detokenize(tokens: number[]): Promise<string>
  embedding(text: string): Promise<EmbeddingResult>
  saveSession(path: string): Promise<void>
  loadSession(path: string): Promise<void>
  release(): Promise<void>
  applyLoraAdapters(adapters: { path: string; scaled: number }[]): void
  removeLoraAdapters(adapters: { path: string }[]): void
  getLoadedLoraAdapters(): { path: string; scaled: number }[]
  /**
   * Initialize multimodal support with a mmproj file
   * @param mmproj_path Path to the multimodal projector file
   * @returns Promise resolving to true if initialization was successful
   */
  initMultimodal(options: { path: string; use_gpu?: boolean }): Promise<boolean>

  /**
   * Check if multimodal support is enabled
   * @returns Promise resolving to true if multimodal is enabled
   */
  isMultimodalEnabled(): Promise<boolean>

  /**
   * Get multimodal support capabilities
   * @returns Promise resolving to an object with vision and audio support
   */
  getMultimodalSupport(): Promise<{
    vision: boolean
    audio: boolean
  }>

  /**
   * Release multimodal support
   */
  releaseMultimodal(): Promise<void>

  /**
   * Load a vocoder model
   * @param path Path to the vocoder model
   * @returns Promise resolving to true if loading was successful
   */
  initVocoder(options: { path: string }): Promise<boolean>

  /**
   * Unload the vocoder model
   * @returns Promise resolving to true if unloading was successful
   */
  releaseVocoder(): Promise<void>

  /**
   * Check if the vocoder model is enabled
   * @returns Promise resolving to true if the vocoder model is enabled
   */
  isVocoderEnabled(): boolean

  /**
   * Get the formatted prompt for audio completion
   * @param speaker Speaker name or null
   * @param text Text to complete
   * @returns Formatted audio completion
   */
  getFormattedAudioCompletion(speaker: string|null, text: string): string

  /**
   * Get guide tokens for audio completion
   * @param text Text to complete
   * @returns Guide tokens
   */
  getAudioCompletionGuideTokens(text: string): Int32Array

  /**
   * Decode audio tokens to audio data
   * @param tokens Tokens to decode
   * @returns Decoded audio tokens
   */
  decodeAudioTokens(tokens: Int32Array): Promise<Float32Array>

  // static
  loadModelInfo(path: string, skip: string[]): Promise<Object>
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
