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
  CompletionResponseFormat,
} from './binding'

export * from './binding'

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
  has_image: boolean
  image_paths?: Array<string>
}

class LlamaContextWrapper {
  ctx: any

  constructor(nativeCtx: any) {
    this.ctx = nativeCtx
  }

  getSystemInfo(): string {
    return this.ctx.getSystemInfo()
  }

  getModelInfo(): object {
    return this.ctx.getModelInfo()
  }

  isJinjaSupported(): boolean {
    const { minja } = this.ctx.getModelInfo().chatTemplates
    return !!minja?.toolUse || !!minja?.default
  }

  isLlamaChatSupported(): boolean {
    return !!this.ctx.getModelInfo().chatTemplates.llamaChat
  }

  _formatImageChat(messages: ChatMessage[] | undefined): {
    messages: ChatMessage[] | undefined
    has_image: boolean
    image_paths?: string[]
  } {
    if (!messages)
      return {
        messages,
        has_image: false,
      }
    const imagePaths: string[] = []
    return {
      messages: messages.map((msg) => {
        if (Array.isArray(msg.content)) {
          const content = msg.content.map((part) => {
            // Handle multimodal content
            if (part.type === 'image_url') {
              let path = part.image_url?.url || ''
              imagePaths.push(path)
              return {
                type: 'text',
                text: '<__image__>',
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
      has_image: imagePaths.length > 0,
      image_paths: imagePaths,
    }
  }

  getFormattedChat(
    messages: ChatMessage[],
    template?: string,
    params?: {
      jinja?: boolean
      response_format?: CompletionResponseFormat
      tools?: object
      parallel_tool_calls?: object
      tool_choice?: string
    },
  ): FormattedChatResult {
    const {
      messages: chat,
      has_image,
      image_paths,
    } = this._formatImageChat(messages)

    const useJinja = this.isJinjaSupported() && params?.jinja
    let tmpl
    if (template) tmpl = template // Force replace if provided
    const jsonSchema = getJsonSchema(params?.response_format)

    const result = this.ctx.getFormattedChat(chat, tmpl, {
      jinja: useJinja,
      json_schema: jsonSchema,
      tools: params?.tools,
      parallel_tool_calls: params?.parallel_tool_calls,
      tool_choice: params?.tool_choice,
    })

    if (!useJinja) {
      return {
        type: 'llama-chat',
        prompt: result as string,
        has_image,
        image_paths,
      }
    }
    const jinjaResult = result
    jinjaResult.type = 'jinja'
    jinjaResult.has_image = has_image
    jinjaResult.image_paths = image_paths
    return jinjaResult
  }

  completion(
    options: LlamaCompletionOptions,
    callback?: (token: LlamaCompletionToken) => void,
  ): Promise<LlamaCompletionResult> {
    const { messages, image_paths = options.image_paths } =
      this._formatImageChat(options.messages)
    return this.ctx.completion({
      ...options,
      messages,
      image_paths: options.image_paths || image_paths,
    }, callback || (() => {}))
  }

  stopCompletion(): void {
    return this.ctx.stopCompletion()
  }

  tokenize(text: string, { image_paths }: { image_paths?: string[] } = {}): Promise<TokenizeResult> {
    return this.ctx.tokenize(text, image_paths)
  }

  detokenize(tokens: number[]): Promise<string> {
    return this.ctx.detokenize(tokens)
  }

  embedding(text: string): Promise<EmbeddingResult> {
    return this.ctx.embedding(text)
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

  removeLoraAdapters(adapters: { path: string }[]): void {
    return this.ctx.removeLoraAdapters(adapters)
  }

  getLoadedLoraAdapters(): { path: string; scaled: number }[] {
    return this.ctx.getLoadedLoraAdapters()
  }

  initMultimodal(options: {
    path: string
    use_gpu?: boolean
  }): Promise<boolean> {
    return this.ctx.initMultimodal(options)
  }

  isMultimodalEnabled(): Promise<boolean> {
    return this.ctx.isMultimodalEnabled()
  }

  releaseMultimodal(): Promise<void> {
    return this.ctx.releaseMultimodal()
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

export const loadLlamaModelInfo = async (path: string): Promise<Object> => {
  const variant = 'default'
  mods[variant] ??= await loadModule(variant)
  refreshNativeLogSetup()
  return mods[variant].LlamaContext.loadModelInfo(path, modelInfoSkip)
}
