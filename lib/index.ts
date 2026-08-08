import { loadModule, LlamaModelOptions } from './binding'
import type {
  Module,
  LlamaContext,
  LibVariant,
  ChatMessage,
  LlamaCompletionOptions,
  LlamaCompletionToken,
  LlamaCompletionResult,
  TokenizeResult,
  EmbeddingResult,
  RerankParams,
  RerankResult,
  CompletionResponseFormat,
  ModelInfo,
  JinjaFormattedChatResult,
  Tool,
  GGUFModelInfo,
  BenchResult,
} from './binding'
import { BUILD_NUMBER, BUILD_COMMIT } from './version'
import { LlamaParallelAPI } from './parallel'
import { formatMediaChat } from './utils'
import type { TTSCapabilities } from './tts'
import type { SpeakerPayload } from './tts-voices'
import { lookupVoice, listVoices, listLanguages } from './tts-voices'

export * from './binding'
export { LlamaParallelAPI }
export type { TTSCapabilities } from './tts'
export type {
  OuteTTSWord,
  OuteTTSSpeaker,
  NeuTTSSpeaker,
  SpeakerPayload,
} from './tts-voices'
export {
  lookupVoice as getTTSVoice,
  listVoices as listTTSVoices,
  listLanguages as listTTSLanguages,
} from './tts-voices'

/**
 * Native-backed voice-clone speaker handle. Created via
 * `context.createSpeaker()`; the JS side holds only the numeric registry id.
 */
export class LlamaSpeaker {
  readonly id: number

  readonly family: string

  rows: number

  baked: boolean

  private ctx: LlamaContext

  constructor(
    ctx: LlamaContext,
    handle: { id: number; family: string; rows: number; baked: boolean },
  ) {
    this.ctx = ctx
    this.id = handle.id
    this.family = handle.family
    this.rows = handle.rows
    this.baked = handle.baked
  }

  /** Run the speaker-encode path now instead of lazily on first use. */
  bake(): void {
    const result = this.ctx.bakeSpeaker(this.id)
    this.rows = result.rows
    this.baked = result.baked
  }

  /** Remove the speaker from the native registry. */
  release(): void {
    this.ctx.releaseSpeaker(this.id)
  }
}

export interface LlamaModelOptionsExtended extends LlamaModelOptions {
  lib_variant?: LibVariant
}

const mods: { [key: string]: Module } = {}

const logListeners: Array<(level: string, text: string) => void> = []

const logCallback = (level: string, text: string) => {
  logListeners.forEach((listener) => listener(level, text))
}

let logEnabled = false

const refreshNativeLogSetup = () => {
  Object.entries(mods).forEach(([, mod]) => {
    mod.LlamaContext.toggleNativeLog(logEnabled, logCallback)
  })
}

export const toggleNativeLog = async (enable: boolean) => {
  logEnabled = enable
  refreshNativeLogSetup()
}

export const isNativeLogEnabled = () => logEnabled

export function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void } {
  logListeners.push(listener)
  return {
    remove: () => {
      const index = logListeners.indexOf(listener)
      if (index >= 0) {
        logListeners.splice(index, 1)
      }
    },
  }
}

const getJsonSchema = (responseFormat?: CompletionResponseFormat) => {
  if (responseFormat?.type === 'json_schema') {
    return responseFormat.json_schema?.schema
  }
  if (responseFormat?.type === 'json_object') {
    return responseFormat.schema || {}
  }
  return null
}

export type FormattedChatResult =
  | {
      type: 'llama-chat'
      prompt: string
      has_media: boolean
      media_paths?: Array<string>
    }
  | ({
      type: 'jinja'
      has_media: boolean
      media_paths?: Array<string>
    } & JinjaFormattedChatResult)

class LlamaContextWrapper {
  ctx: LlamaContext
  parallel: LlamaParallelAPI

  constructor(nativeCtx: LlamaContext) {
    this.ctx = nativeCtx
    this.parallel = new LlamaParallelAPI(nativeCtx)
  }

  getSystemInfo(): string {
    return this.ctx.getSystemInfo()
  }

  getModelInfo(): ModelInfo {
    return this.ctx.getModelInfo()
  }

  getUsedDevices(): string[] {
    return this.ctx.getUsedDevices()
  }

  isJinjaSupported(): boolean {
    const { jinja } = this.ctx.getModelInfo().chatTemplates
    return !!jinja?.toolUse || !!jinja?.default
  }

  isLlamaChatSupported(): boolean {
    return !!this.ctx.getModelInfo().chatTemplates.llamaChat
  }

  getFormattedChat(
    messages: ChatMessage[],
    template?: string,
    params?: {
      jinja?: boolean
      response_format?: CompletionResponseFormat
      tools?: Tool[]
      parallel_tool_calls?: boolean
      tool_choice?: string
      enable_thinking?: boolean
      reasoning_format?: 'none' | 'auto' | 'deepseek'
      add_generation_prompt?: boolean
      now?: string | number
      chat_template_kwargs?: Record<string, string>
      force_pure_content?: boolean
    },
  ): FormattedChatResult {
    const { messages: chat, has_media, media_paths } = formatMediaChat(messages)

    const forcePureContent = params?.force_pure_content ?? false
    // When force_pure_content is set, accept any model that has a chat_template
    // string in its metadata without requiring template validation to pass.
    const hasChatTemplate = !!(this.ctx.getModelInfo().metadata?.['tokenizer.chat_template'])
    const useJinja =
      (forcePureContent ? hasChatTemplate : this.isJinjaSupported()) &&
      (params?.jinja ?? true)
    let tmpl
    if (template) tmpl = template // Force replace if provided

    const result = this.ctx.getFormattedChat(chat!, tmpl, {
      jinja: useJinja,
      response_format: params?.response_format,
      tools: params?.tools,
      parallel_tool_calls: params?.parallel_tool_calls,
      tool_choice: params?.tool_choice,
      enable_thinking: params?.enable_thinking ?? true,
      reasoning_format: params?.reasoning_format ?? 'none',
      add_generation_prompt: params?.add_generation_prompt,
      now: params?.now,
      force_pure_content: forcePureContent,
      chat_template_kwargs: params?.chat_template_kwargs
        ? Object.entries(params.chat_template_kwargs).reduce(
            (acc, [key, value]) => {
              acc[key] = JSON.stringify(value) // Each value is a stringified JSON object
              return acc
            },
            {} as Record<string, any>,
          )
        : undefined,
    })

    if (!useJinja) {
      return {
        type: 'llama-chat',
        prompt: result as string,
        has_media,
        media_paths,
      }
    }
    const jinjaResult = result as JinjaFormattedChatResult
    return {
      type: 'jinja',
      has_media,
      media_paths,
      ...jinjaResult,
    }
  }

  completion(
    options: LlamaCompletionOptions & { speaker?: LlamaSpeaker },
    callback?: (token: LlamaCompletionToken) => void,
  ): Promise<LlamaCompletionResult> {
    const { messages, media_paths = options.media_paths } = formatMediaChat(
      options.messages,
    )
    const { speaker, ...rest } = options
    return this.ctx.completion(
      {
        ...rest,
        ...(speaker instanceof LlamaSpeaker ? { speakerId: speaker.id } : {}),
        messages,
        media_paths: options.media_paths || media_paths,
      },
      callback || (() => {}),
    )
  }

  stopCompletion(): void {
    return this.ctx.stopCompletion()
  }

  tokenize(
    text: string,
    { media_paths }: { media_paths?: string[] } = {},
  ): Promise<TokenizeResult> {
    return this.ctx.tokenize(text, media_paths)
  }

  detokenize(tokens: number[]): Promise<string> {
    return this.ctx.detokenize(tokens)
  }

  embedding(
    text: string,
    params?: { embd_normalize?: number },
  ): Promise<EmbeddingResult> {
    return this.ctx.embedding(text, params)
  }

  rerank(
    query: string,
    documents: string[],
    params?: RerankParams,
  ): Promise<Array<RerankResult & { document: string }>> {
    return this.ctx
      .rerank(query, documents, params)
      .then((results: RerankResult[]) => {
        // Sort by score descending and add document text for convenience
        return results
          .map((result: RerankResult) => ({
            ...result,
            document: documents[result.index],
          }))
          .sort(
            (
              a: RerankResult & { document: string },
              b: RerankResult & { document: string },
            ) => b.score - a.score,
          )
      })
  }

  saveSession(path: string): Promise<void> {
    return this.ctx.saveSession(path)
  }

  loadSession(path: string): Promise<void> {
    return this.ctx.loadSession(path)
  }

  release(): Promise<void> {
    return this.ctx.release()
  }

  applyLoraAdapters(adapters: { path: string; scaled: number }[]): void {
    return this.ctx.applyLoraAdapters(adapters)
  }

  removeLoraAdapters(): void {
    this.ctx.removeLoraAdapters()
  }

  getLoadedLoraAdapters(): { path: string; scaled: number }[] {
    return this.ctx.getLoadedLoraAdapters()
  }

  initMultimodal(options: {
    path: string
    use_gpu?: boolean
    image_min_tokens?: number
    image_max_tokens?: number
  }): boolean {
    return this.ctx.initMultimodal(options)
  }

  isMultimodalEnabled(): boolean {
    return this.ctx.isMultimodalEnabled()
  }

  releaseMultimodal(): void {
    this.ctx.releaseMultimodal()
  }

  getMultimodalSupport(): {
    vision: boolean
    audio: boolean
  } {
    return this.ctx.getMultimodalSupport()
  }

  initVocoder(options: {
    path: string
    n_batch?: number
    use_gpu?: boolean
  }): boolean {
    return this.ctx.initVocoder(options)
  }

  releaseVocoder(): void {
    this.ctx.releaseVocoder()
  }

  isVocoderEnabled(): boolean {
    return this.ctx.isVocoderEnabled()
  }

  getTTSCapabilities(): TTSCapabilities {
    return this.ctx.getTTSCapabilities()
  }

  getAudioSampleRate(): number {
    return this.ctx.getAudioSampleRate()
  }

  /**
   * Build the formatted prompt (plus grammar / flow metadata) for a TTS
   * generation. Mirrors llama.rn's API:
   * - `speaker` may be a `LlamaSpeaker` (voice clone), a structured
   *   `SpeakerPayload` object, a built-in voice name, or omitted for the
   *   family default voice.
   * - `phonemizer` is invoked when the model requires phoneme input
   *   (e.g. NeuTTS); it may be async.
   */
  async getFormattedAudioCompletion(options: {
    prompt: string
    speaker?: string | LlamaSpeaker | SpeakerPayload
    phonemizer?: (text: string, language: string) => string | Promise<string>
    language?: string
  }): Promise<{
    prompt: string
    grammar?: string
    embedding: boolean
    flow: 'tokens' | 'continuous_embd' | ''
  }> {
    const caps = this.getTTSCapabilities()
    const language = options.language ?? caps.defaultLanguage ?? 'en-us'

    let text = options.prompt
    if (caps.requiresPhonemes && options.phonemizer) {
      text = await options.phonemizer(text, language)
    }

    const { speaker } = options
    if (speaker instanceof LlamaSpeaker) {
      return this.ctx.getFormattedAudioCompletion('', text, speaker.id)
    }

    let payload: SpeakerPayload | null = null
    if (speaker && typeof speaker === 'object') {
      payload = speaker
    } else {
      const name = typeof speaker === 'string' ? speaker : 'default'
      payload = lookupVoice(caps.family, name, language)
      if (typeof speaker === 'string' && !payload) {
        throw new Error(
          `Unknown built-in voice '${name}' for ${caps.family} (${language})`,
        )
      }
    }

    if (
      payload &&
      caps.requiresPhonemes &&
      options.phonemizer &&
      typeof (payload as any).ref_text === 'string' &&
      typeof (payload as any).ref_phones !== 'string'
    ) {
      payload = {
        ...payload,
        ref_phones: await options.phonemizer((payload as any).ref_text, language),
      }
    }

    return this.ctx.getFormattedAudioCompletion(
      payload ? JSON.stringify(payload) : '',
      text,
    )
  }

  /**
   * Register a voice-clone speaker from raw reference audio.
   * The speaker is encoded lazily on first use unless `bake` is set.
   */
  createSpeaker(config: {
    refAudio: Float32Array | number[]
    refAudioSampleRate: number
    refText?: string
    emotion?: number
    bake?: boolean
  }): LlamaSpeaker {
    const options: {
      pcm: Float32Array | number[]
      sample_rate: number
      ref_text?: string
      emotion?: number
      bake?: boolean
    } = {
      pcm: config.refAudio,
      sample_rate: config.refAudioSampleRate,
    }
    if (config.refText != null) options.ref_text = config.refText
    if (config.emotion != null) options.emotion = config.emotion
    if (config.bake != null) options.bake = config.bake
    const handle = this.ctx.createSpeaker(options)
    return new LlamaSpeaker(this.ctx, handle)
  }

  decodeAudioTokens(tokens: number[] | Int32Array): Promise<Float32Array> {
    return this.ctx.decodeAudioTokens(tokens)
  }

  decodeAudioEmbeddings(
    embeddings: number[] | Float32Array,
    embeddingDim: number,
  ): Promise<Float32Array> {
    return this.ctx.decodeAudioEmbeddings(embeddings, embeddingDim)
  }

  /**
   * Clear the KV and recurrent caches.
   * This is faster than recreating the context and useful for preventing
   * cache contamination between chat sessions.
   * @param clearData If true, also clears the cache data (default: false)
   */
  clearCache(clearData?: boolean): void {
    this.ctx.clearCache(clearData)
  }

  /**
   * Run a benchmark to measure model performance
   * @param pp Number of tokens to process for prompt processing benchmark
   * @param tg Number of tokens to generate for text generation benchmark
   * @param pl Parallel level (number of sequences)
   * @param nr Number of repetitions
   * @returns Benchmark results including timing and speed metrics
   */
  async bench(pp: number, tg: number, pl: number, nr: number): Promise<BenchResult> {
    return this.ctx.bench(pp, tg, pl, nr)
  }
}

export const loadModel = async (
  options: LlamaModelOptionsExtended,
  onProgress?: (progress: number) => void,
): Promise<LlamaContextWrapper> => {
  const variant = options.lib_variant ?? 'default'
  mods[variant] ??= await loadModule(options.lib_variant)
  refreshNativeLogSetup()

  const { devices } = options
  let filteredDevs: Array<string> = []
  if (Array.isArray(devices)) {
    filteredDevs = [...devices]

    // Handle HTP* to use all HTP devices on Hexagon
    if (variant === 'snapdragon' && devices.includes('HTP*')) {
      const backendDevices = await getBackendDevicesInfo(variant)
      const htpDevices = backendDevices
        .filter((d) => d.deviceName.startsWith('HTP'))
        .map((d) => d.deviceName)
      filteredDevs = filteredDevs.reduce((acc, dev) => {
        if (dev.startsWith('HTP*')) {
          acc.push(...htpDevices)
        } else if (!dev.startsWith('HTP')) {
          acc.push(dev)
        }
        return acc
      }, [] as Array<string>)
    }
  }

  const nativeCtx = new mods[variant].LlamaContext(
    {
      ...options,
      devices: filteredDevs.length > 0 ? filteredDevs : undefined,
    },
    onProgress,
  )
  return new LlamaContextWrapper(nativeCtx)
}

export const initLlama = loadModule

const modelInfoSkip = [
  // Large fields
  'tokenizer.ggml.tokens',
  'tokenizer.ggml.token_type',
  'tokenizer.ggml.merges',
  'tokenizer.ggml.scores',
]

export const loadLlamaModelInfo = async (
  path: string,
): Promise<GGUFModelInfo> => {
  const variant = 'default'
  mods[variant] ??= await loadModule(variant)
  refreshNativeLogSetup()
  return mods[variant].LlamaContext.loadModelInfo(path, modelInfoSkip)
}

export const getBackendDevicesInfo = async (
  variant: LibVariant = 'default',
): Promise<import('./binding').BackendDeviceInfo[]> => {
  mods[variant] ??= await loadModule(variant)
  refreshNativeLogSetup()
  const jsonString = mods[variant].LlamaContext.getBackendDevicesInfo()
  return JSON.parse(jsonString as any)
}

export const BuildInfo = {
  number: BUILD_NUMBER,
  commit: BUILD_COMMIT,
}
