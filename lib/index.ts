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
  DecisionQuestions,
  DecisionRequest,
  DecisionResult,
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
import type {
  TTSCapabilities,
  TTSCompletionParams,
  TTSSpeakerInfo,
} from './tts'

export * from './binding'
export { LlamaParallelAPI }
export type {
  TTSCapabilities,
  TTSCompletionParams,
  TTSSpeakerInfo,
} from './tts'

/**
 * Native-backed TTS speaker handle: a voice cloned from reference audio
 * (`context.createSpeaker()`) or loaded from a speaker file
 * (`context.loadSpeaker()`). The JS side holds only the numeric registry id.
 */
export class LlamaSpeaker {
  readonly id: number

  readonly family: string

  private ctx: LlamaContext

  constructor(ctx: LlamaContext, handle: { id: number; family: string }) {
    this.ctx = ctx
    this.id = handle.id
    this.family = handle.family
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
    return responseFormat.schema || { type: 'object' }
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

  /**
   * Run a completion. With a vocoder, `speaker` is the voice that fills the
   * prompt's voice markers: a `LlamaSpeaker` or a builtin voice name; the
   * model's default when omitted.
   */
  completion(
    options: LlamaCompletionOptions & { speaker?: string | LlamaSpeaker },
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
        ...(typeof speaker === 'string' ? { voice: speaker } : {}),
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

  /**
   * Answer typed questions about a state with a decision model
   * (see `getModelInfo().decision`), in the shape of the TypeSafe
   * `/v1/systemone` API. Each answer is read from one forward pass, no token
   * is generated. It runs on the sequence of `completion()`, whose cached
   * prompt is dropped: the next completion evaluates its prompt from the start.
   *
   * Rejects if the request is invalid, if the model is not a decision
   * model of a supported type, or if parallel mode is enabled (use
   * `parallel.decide()` then).
   */
  decide<const Q extends DecisionQuestions>(
    request: DecisionRequest<Q>,
  ): Promise<DecisionResult<Q>> {
    return this.ctx.decide(request) as Promise<DecisionResult<Q>>
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

  /**
   * Release the context synchronously. Use this in `process.on('exit')`
   * handlers, where `release()` cannot be awaited.
   */
  releaseSync(): void {
    this.ctx.releaseSync()
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

  /**
   * Attach a codec / vocoder GGUF to this context, enabling the TTS API.
   * Throws when one is already attached: `releaseVocoder()` first.
   *
   * **Experimental:** the TTS API may change without a major version bump.
   */
  initVocoder(options: {
    path: string
    n_batch?: number
    /**
     * Defaults to true when the main context was created with
     * `n_gpu_layers > 0`, false otherwise.
     */
    use_gpu?: boolean
    /**
     * Codec GGUF metadata overrides as llama.cpp-style `KEY=TYPE:VALUE`
     * (TYPE: int | float | bool | str), e.g. fewer flow-matching steps:
     * `['cosyvoice.flow.n_timesteps=int:6']`.
     */
    kv_overrides?: string[]
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
   * Arm a text-to-speech request and report how to run it.
   *
   * The flow is `getFormattedAudioCompletion` → `completion()` →
   * `decodeAudioTokens` / `decodeAudioLatents`. Like `getFormattedChat`, this
   * returns what the request's format needs as `completionParams`: the
   * prompt, the grammar and the `logit_bias` token mask, `embedding` and the
   * model's minimum length (`n_min_predict`). Sampling (the backbone GGUF's
   * `general.sampling.*` unless set) and `n_predict` are the `completion()`'s;
   * its `prompt` must be `completionParams.prompt`. The result carries what
   * it generated: `audio_tokens` (codec codes) or `audio_latents`.
   * Long text is the caller's to split: one request per piece.
   *
   * - `prompt` — text to speak. Phonemized if `phonemizer` is supplied.
   * - `speaker` — a builtin voice name of the codec (`builtinVoices`), a
   *   `LlamaSpeaker`, or `undefined` for the model's default voice. Pass the
   *   same one to `completion()`.
   * - `phonemizer` — optional `(text, language) => string | Promise<string>`,
   *   for models with `requiresPhonemes`.
   * - `language` — phonemizer hint; defaults to capabilities.defaultLanguage.
   * - `instruct` — how to speak (e.g. "Speak cheerfully"), for models with
   *   `canTakeInstruct`; others reject it.
   */
  async getFormattedAudioCompletion(options: {
    prompt: string
    speaker?: string | LlamaSpeaker
    phonemizer?: (text: string, language: string) => string | Promise<string>
    language?: string
    instruct?: string
  }): Promise<{
    prompt: string
    grammar?: string
    embedding: boolean
    flow: 'tokens' | 'continuous_embd' | ''
    completionParams: TTSCompletionParams
  }> {
    const caps = this.getTTSCapabilities()
    const language = options.language ?? (caps.defaultLanguage || 'en-us')

    let text = options.prompt
    if (caps.requiresPhonemes && options.phonemizer) {
      text = await options.phonemizer(text, language)
    }

    const { speaker } = options
    if (speaker instanceof LlamaSpeaker) {
      return this.ctx.getFormattedAudioCompletion(
        '',
        text,
        speaker.id,
        options.instruct,
      )
    }
    if (speaker !== undefined && typeof speaker !== 'string') {
      throw new TypeError(
        'speaker: pass a builtin voice name (capabilities.builtinVoices) or a LlamaSpeaker',
      )
    }
    if (
      speaker !== undefined &&
      !caps.builtinVoices.some((v) => v.name === speaker)
    ) {
      throw new Error(
        `Unknown voice '${speaker}'; builtin voices: ${
          caps.builtinVoices.map((v) => v.name).join(', ') || 'none'
        }`,
      )
    }
    return this.ctx.getFormattedAudioCompletion(
      speaker ? JSON.stringify({ codec_voice: speaker }) : '',
      text,
      -1,
      options.instruct,
    )
  }

  /**
   * Clone a voice from a reference clip (mono, any sample rate) and its
   * transcript `refText`, for models with `canSpeakerFromAudio`. All of the
   * voice's conditioning is computed here, once.
   */
  createSpeaker(config: {
    refAudio: Float32Array | number[]
    refAudioSampleRate: number
    refText?: string
    emotion?: number
  }): LlamaSpeaker {
    const options: {
      pcm: Float32Array | number[]
      sample_rate: number
      ref_text?: string
      emotion?: number
    } = {
      pcm: config.refAudio,
      sample_rate: config.refAudioSampleRate,
    }
    if (config.refText != null) options.ref_text = config.refText
    if (config.emotion != null) options.emotion = config.emotion
    const handle = this.ctx.createSpeaker(options)
    return new LlamaSpeaker(this.ctx, handle)
  }

  /**
   * Write a speaker's voice (all of its conditioning) to a speaker file of
   * this model. `options.name` names it inside the file (default: codec.cpp's
   * name). Reload it with `loadSpeaker`.
   */
  async saveSpeaker(
    speaker: LlamaSpeaker,
    path: string,
    options?: { name?: string },
  ): Promise<void> {
    this.ctx.saveSpeaker(speaker.id, path, options?.name)
  }

  /**
   * Load a speaker from a speaker file (written by `saveSpeaker` or codec.cpp's
   * `tts-cli --save-speaker`). `options.name` picks a speaker inside the file
   * (default: the file's default speaker). Files of another model are refused.
   */
  async loadSpeaker(
    path: string,
    options?: { name?: string },
  ): Promise<LlamaSpeaker> {
    const handle = this.ctx.loadSpeaker(path, options?.name)
    return new LlamaSpeaker(this.ctx, handle)
  }

  /** Speakers stored in a speaker file. */
  async listSpeakers(path: string): Promise<TTSSpeakerInfo[]> {
    return this.ctx.listSpeakers(path)
  }

  /**
   * PCM (at `getAudioSampleRate()`) of a text-to-speech completion's
   * `audio_tokens`, with the voice of the last text-to-speech completion.
   */
  decodeAudioTokens(tokens: number[] | Int32Array): Promise<Float32Array> {
    return this.ctx.decodeAudioTokens(tokens)
  }

  /**
   * PCM of a text-to-speech completion's `audio_latents` (latent models such
   * as Soprano / BlueMagpie), like `decodeAudioTokens`.
   */
  decodeAudioLatents(latents: number[] | Float32Array): Promise<Float32Array> {
    return this.ctx.decodeAudioLatents(latents)
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

// Release every live context when the process exits, including process.exit()
// and uncaught exceptions, where GC finalizers do not run. Backend buffers that
// outlive the process teardown abort the process on some backends (Metal).
const exitCleanupRegistered = new WeakSet<Module>()
const registerExitCleanup = (mod: Module) => {
  if (typeof process?.once !== 'function' || exitCleanupRegistered.has(mod)) return
  exitCleanupRegistered.add(mod)
  process.once('exit', () => {
    try {
      mod.LlamaContext.releaseAllSync()
    } catch {
      // nothing useful can be done this late
    }
  })
}

export const loadModel = async (
  options: LlamaModelOptionsExtended,
  onProgress?: (progress: number) => void,
): Promise<LlamaContextWrapper> => {
  const variant = options.lib_variant ?? 'default'
  mods[variant] ??= await loadModule(options.lib_variant)
  registerExitCleanup(mods[variant])
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
