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

export type LlamaCompletionToken = {
  token: string
}

export interface LlamaContext {
  new (options: LlamaModelOptions): LlamaContext
  getSystemInfo(): string
  completion(options: LlamaCompletionOptions, callback?: (token: LlamaCompletionToken) => void): Promise<LlamaCompletionResult>
  stopCompletion(): void
  saveSession(path: string): Promise<void>
  loadSession(path: string): Promise<void>
  release(): Promise<void>
}

export interface Module {
  LlamaContext: LlamaContext
}

export const loadModule = async (variant?: string): Promise<Module> => {
  try {
    if (variant) {
      return await import(`../bin/${process.platform}-${variant}/${process.arch}/llama-node.node`) as Module
    }
  } catch {} // ignore errors and try the common path
  return await import(`../bin/${process.platform}/${process.arch}/llama-node.node`) as Module
}