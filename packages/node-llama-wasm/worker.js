import {
  addNativeLogListener,
  loadModel as loadModelDirect,
  toggleNativeLog as toggleNativeLogDirect,
} from './index.js'

let context = null

addNativeLogListener((level, text) => {
  self.postMessage({ type: 'nativeLog', level, text })
})

const serializeError = (error) => ({
  message: error?.message || String(error),
  stack: error?.stack,
  name: error?.name,
})

const transferForResult = (result) => {
  if (result instanceof ArrayBuffer) return [result]
  if (ArrayBuffer.isView(result)) return [result.buffer]
  return []
}

const postResult = (id, result) => {
  self.postMessage({ id, type: 'response', result }, transferForResult(result))
}

const requireContext = () => {
  if (!context) throw new Error('Model not loaded')
  return context
}

const setNativeLogEnabled = async (enabled) => {
  await toggleNativeLogDirect(!!enabled)
}

const loadModel = async (options, id) => {
  const wasm = {
    ...(options.wasm || {}),
    worker: false,
  }
  const nativeLogEnabled = !!wasm.nativeLogEnabled
  delete wasm.workerPath
  delete wasm.nativeLogEnabled
  await setNativeLogEnabled(nativeLogEnabled)
  context = await loadModelDirect(
    {
      ...options,
      wasm,
    },
    (progress, detail) =>
      self.postMessage({ id, type: 'progress', progress, detail }),
  )
  return {
    modelInfo: context.getModelInfo(),
    systemInfo: context.getSystemInfo(),
  }
}

const callMethod = async (id, method, args) => {
  switch (method) {
    case 'loadModel':
      return loadModel(args[0], id)
    case 'toggleNativeLog':
      await setNativeLogEnabled(args[0])
      return undefined
    case 'getFormattedChat':
      return requireContext().getFormattedChat(args[0], args[1], args[2])
    case 'completion':
      return requireContext().completion(args[0], (token) => {
        self.postMessage({ id, type: 'token', token })
      })
    case 'tokenize':
      return requireContext().tokenize(args[0], args[1])
    case 'detokenize':
      return requireContext().detokenize(args[0])
    case 'embedding':
      return requireContext().embedding(args[0], args[1])
    case 'rerank':
      return requireContext().rerank(args[0], args[1], args[2])
    case 'saveSession':
      return requireContext().saveSession()
    case 'loadSession':
      await requireContext().loadSession(args[0])
      return undefined
    case 'release':
      await requireContext().release()
      context = null
      return undefined
    case 'applyLoraAdapters':
      requireContext().applyLoraAdapters(args[0])
      return requireContext().getLoadedLoraAdapters()
    case 'removeLoraAdapters':
      requireContext().removeLoraAdapters()
      return requireContext().getLoadedLoraAdapters()
    case 'initMultimodal': {
      const enabled = await requireContext().initMultimodal(args[0])
      return enabled ? requireContext().getMultimodalSupport() : false
    }
    case 'releaseMultimodal':
      requireContext().releaseMultimodal()
      return undefined
    case 'clearCache':
      requireContext().clearCache(args[0])
      return undefined
    case 'bench':
      return requireContext().bench(args[0], args[1], args[2], args[3])
    default:
      throw new Error(`Unknown llama.node WASM worker method: ${method}`)
  }
}

self.addEventListener('message', async (event) => {
  const message = event.data || {}
  if (message.type === 'control') {
    if (message.method === 'toggleNativeLog') {
      await setNativeLogEnabled(message.args?.[0])
    }
    return
  }
  if (message.type !== 'request') return

  try {
    const result = await callMethod(message.id, message.method, message.args || [])
    postResult(message.id, result)
  } catch (error) {
    self.postMessage({
      id: message.id,
      type: 'error',
      error: serializeError(error),
    })
  }
})
