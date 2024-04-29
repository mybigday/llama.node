import { loadModule, LlamaModelOptions } from './binding'
import type { Module, LlamaContext } from './binding'

export * from './binding'

export interface LlamaModelOptionsExtended extends LlamaModelOptions {
  lib_variant?: string
}

let module: Module | null = null

export const loadModel = async (options: LlamaModelOptionsExtended): Promise<LlamaContext> => {
    module ??= await loadModule(options.lib_variant)
    return new module.LlamaContext(options)
}
