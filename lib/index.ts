import { loadModule, LlamaModelOptions } from './binding'
import type { LlamaContext } from './binding'

export * from './binding'

export interface LlamaModelOptionsExtended extends LlamaModelOptions {
  lib_variant?: string
}

export const loadModel = (options: LlamaModelOptionsExtended): LlamaContext => {
    const { LlamaContext } = loadModule(options.lib_variant)
    return new LlamaContext(options)
}
