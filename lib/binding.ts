import * as path from 'path'

export type ChatMessage = {
  role: string
  text: string
}

export type LlamaModelOptions = {
  model: string
  embedding?: boolean
  embd_normalize?: number
  pooling_type?: number
  n_ctx?: number
  n_batch?: number
  n_threads?: number
  n_gpu_layers?: number
  use_mlock?: boolean
  use_mmap?: boolean
  vocab_only?: boolean
}

export type LlamaCompletionOptions = {
  messages?: ChatMessage[]
  prompt?: string
  n_samples?: number
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
}

export type LlamaCompletionResult = {
  text: string
  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
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
}

export type EmbeddingResult = {
  embedding: Float32Array
}

export interface LlamaContext {
  new (options: LlamaModelOptions): LlamaContext
  getSystemInfo(): string
  getModelInfo(): object
  getFormattedChat(messages: ChatMessage[]): string
  completion(options: LlamaCompletionOptions, callback?: (token: LlamaCompletionToken) => void): Promise<LlamaCompletionResult>
  stopCompletion(): void
  tokenize(text: string): Promise<TokenizeResult>
  detokenize(tokens: number[]): Promise<string>
  embedding(text: string): Promise<EmbeddingResult>
  saveSession(path: string): Promise<void>
  loadSession(path: string): Promise<void>
  release(): Promise<void>
}

export interface Module {
  LlamaContext: LlamaContext
}

export type LibVariant = 'default' | 'vulkan'

const setupEnv = (variant?: string) => {
  const postfix = variant ? `-${variant}` : ''
  const binPath = path.resolve(__dirname, `../bin/${process.platform}${postfix}/${process.arch}/`)
  const systemPathEnv = process.env.PATH ?? process.env.Path ?? ''
  if (!systemPathEnv.includes(binPath)) {
    if (process.platform === 'win32') {
      process.env.Path = `${binPath};${systemPathEnv}`
    } else {
      process.env.PATH = `${binPath}:${systemPathEnv}`
    }
  }
}

export const loadModule = async (variant?: LibVariant): Promise<Module> => {
  try {
    if (variant && variant !== 'default') {
      setupEnv(variant)
      return await import(`../bin/${process.platform}-${variant}/${process.arch}/llama-node.node`) as Module
    }
  } catch {} // ignore errors and try the common path
  setupEnv()
  return await import(`../bin/${process.platform}/${process.arch}/llama-node.node`) as Module
}