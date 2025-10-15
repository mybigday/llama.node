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
} from './binding'
import { BUILD_NUMBER, BUILD_COMMIT } from './version'
import { LlamaParallelAPI } from './parallel'

export * from './binding'
export { LlamaParallelAPI }

export const MTMD_DEFAULT_MEDIA_MARKER = '<__media__>'

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

export function addNativeLogListener(
  listener: (level: string, text: string) => void,
): { remove: () => void } {
  logListeners.push(listener)
  return {
    remove: () => {
      logListeners.splice(logListeners.indexOf(listener), 1)
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

export type FormattedChatResult = {
  type: 'jinja' | 'llama-chat'
  prompt: string
  has_media: boolean
  media_paths?: Array<string>
}

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

  isJinjaSupported(): boolean {
    const { minja } = this.ctx.getModelInfo().chatTemplates
    return !!minja?.toolUse || !!minja?.default
  }

  isLlamaChatSupported(): boolean {
    return !!this.ctx.getModelInfo().chatTemplates.llamaChat
  }

  _formatMediaChat(messages: ChatMessage[] | undefined): {
    messages: ChatMessage[] | undefined
    has_media: boolean
    media_paths?: string[]
  } {
    if (!messages)
      return {
        messages,
        has_media: false,
      }
    const mediaPaths: string[] = []
    return {
      messages: messages.map((msg) => {
        if (Array.isArray(msg.content)) {
          const content = msg.content.map((part) => {
            // Handle multimodal content
            if (part.type === 'image_url') {
              let path = part.image_url?.url || ''
              mediaPaths.push(path)
              return {
                type: 'text',
                text: MTMD_DEFAULT_MEDIA_MARKER,
              }
            } else if (part.type === 'input_audio') {
              const { input_audio: audio } = part
              if (!audio) throw new Error('input_audio is required')

              const { format } = audio
              if (format != 'wav' && format != 'mp3') {
                throw new Error(`Unsupported audio format: ${format}`)
              }
              if (audio.url) {
                const path = audio.url.replace(/file:\/\//, '')
                mediaPaths.push(path)
              } else if (audio.data) {
                mediaPaths.push(audio.data)
              }
              return {
                type: 'text',
                text: MTMD_DEFAULT_MEDIA_MARKER,
              }
            }
            return part
          })

          return {
            ...msg,
            content,
          }
        }
        return msg
      }),
      has_media: mediaPaths.length > 0,
      media_paths: mediaPaths,
    }
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
      add_generation_prompt?: boolean
      now?: string | number
      chat_template_kwargs?: Record<string, string>
    },
  ): FormattedChatResult {
    const {
      messages: chat,
      has_media,
      media_paths,
    } = this._formatMediaChat(messages)

    const useJinja = this.isJinjaSupported() && params?.jinja
    let tmpl
    if (template) tmpl = template // Force replace if provided

    const result = this.ctx.getFormattedChat(chat!, tmpl, {
      jinja: useJinja,
      response_format: params?.response_format,
      tools: params?.tools,
      parallel_tool_calls: params?.parallel_tool_calls,
      tool_choice: params?.tool_choice,
      enable_thinking: params?.enable_thinking ?? true,
      add_generation_prompt: params?.add_generation_prompt,
      now: params?.now,
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
    options: LlamaCompletionOptions,
    callback?: (token: LlamaCompletionToken) => void,
  ): Promise<LlamaCompletionResult> {
    const { messages, media_paths = options.media_paths } =
      this._formatMediaChat(options.messages)
    return this.ctx.completion(
      {
        ...options,
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

  embedding(text: string, params?: { embd_normalize?: number }): Promise<EmbeddingResult> {
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

  initMultimodal(options: { path: string; use_gpu?: boolean }): boolean {
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

  initVocoder(options: { path: string; n_batch?: number }): boolean {
    return this.ctx.initVocoder(options)
  }

  releaseVocoder(): void {
    this.ctx.releaseVocoder()
  }

  isVocoderEnabled(): boolean {
    return this.ctx.isVocoderEnabled()
  }

  getFormattedAudioCompletion(
    speaker: string | null,
    text: string,
  ): {
    prompt: string
    grammar?: string
  } {
    return this.ctx.getFormattedAudioCompletion(speaker, text)
  }

  getAudioCompletionGuideTokens(text: string): Int32Array {
    return this.ctx.getAudioCompletionGuideTokens(text)
  }

  decodeAudioTokens(tokens: number[] | Int32Array): Promise<Float32Array> {
    return this.ctx.decodeAudioTokens(tokens)
  }
}

export const loadModel = async (
  options: LlamaModelOptionsExtended,
): Promise<LlamaContextWrapper> => {
  const variant = options.lib_variant ?? 'default'
  mods[variant] ??= await loadModule(options.lib_variant)
  refreshNativeLogSetup()

  const nativeCtx = new mods[variant].LlamaContext(options)
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

export const BuildInfo = {
  number: BUILD_NUMBER,
  commit: BUILD_COMMIT,
}
