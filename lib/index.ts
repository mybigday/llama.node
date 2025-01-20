import { loadModule, LlamaModelOptions } from './binding'
import type { Module, LlamaContext, LibVariant } from './binding'

export * from './binding'

export interface LlamaModelOptionsExtended extends LlamaModelOptions {
  lib_variant?: LibVariant
}

const mods: { [key: string]: Module } = {}

export const loadModel = async (options: LlamaModelOptionsExtended): Promise<LlamaContext> => {
  const variant = options.lib_variant ?? 'default'
  mods[variant] ??= await loadModule(options.lib_variant)
  return new mods[variant].LlamaContext(options)
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
  return mods[variant].LlamaContext.loadModelInfo(path, modelInfoSkip)
}
