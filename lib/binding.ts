export type LlamaModelOptions = {
  model: string
  embedding?: boolean
  n_ctx?: number
  n_batch?: number
  n_threads?: number
  n_gpu_layers?: number
  use_mlock?: boolean
  use_mmap?: boolean
}

export type LlamaCompletionOptions = {
  prompt: string
  n_samples?: number
  temperature?: number
  top_k?: number
  top_p?: number
  repetition_penalty?: number
  n_predict?: number
  max_length?: number
  max_tokens?: number
  seed?: number
  stop?: string[]
}

export type LlamaCompletionResult = {
  text: string
  tokens_predicted: number
  tokens_evaluated: number
  truncated: boolean
}

export interface LlamaContext {
  new (options: LlamaModelOptions): LlamaContext
  getSystemInfo(): string
  completion(options: LlamaCompletionOptions, onToken?: (token: string) => void): Promise<LlamaCompletionResult>
  stopCompletion(): void
  saveSession(path: string): Promise<void>
  loadSession(path: string): Promise<void>
}

export interface Module {
  LlamaContext: LlamaContext
}

export const loadModule = (variant?: string): Module => {
  try {
    if (variant) {
      return require(`../bin/${process.platform}/${process.arch}-${variant}/llama-node.node`) as Module
    }
  } catch {} // ignore errors and try the common path
  return require(`../bin/${process.platform}/${process.arch}/llama-node.node`) as Module
}