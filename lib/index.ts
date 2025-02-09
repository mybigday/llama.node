import { loadModule, LlamaModelOptions } from './binding'
import type { Module, LlamaContext, LibVariant } from './binding'

export * from './binding'

export interface LlamaModelOptionsExtended extends LlamaModelOptions {
  lib_variant?: LibVariant
}

const mods: { [key: string]: Module } = {}

export const loadModel = async (
  options: LlamaModelOptionsExtended,
): Promise<LlamaContext> => {
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

const logListeners: Array<(level: string, text: string) => void> = []

const logCallback = (level: string, text: string) => {
  logListeners.forEach((listener) => listener(level, text))
}

export const toggleNativeLog = async (
  enable: boolean,
  options?: {
    variant?: LibVariant
  },
) => {
  const v = options?.variant ?? 'default'
  mods[v] ??= await loadModule(v)
  return mods[v].LlamaContext.toggleNativeLog(enable, logCallback)
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
