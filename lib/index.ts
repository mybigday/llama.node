import { loadModule, LlamaModelOptions } from './binding'
import type { Module, LlamaContext } from './binding'

export * from './binding'

export interface LlamaModelOptionsExtended extends LlamaModelOptions {
  lib_variant?: string
}

const mods: { [key: string]: Module } = {}

export const loadModel = async (options: LlamaModelOptionsExtended): Promise<LlamaContext> => {
  const variant = options.lib_variant ?? 'default'
  mods[variant] ??= await loadModule(options.lib_variant)
  return new mods[variant].LlamaContext(options)
}
