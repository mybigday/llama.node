import { loadModel } from '../lib/index.js'

const libVariant = process.env.LLAMA_LIB_VARIANT || 'default'

const model = await loadModel(
  {
    n_ctx: 8192,
    n_gpu_layers: 99,
    model: import.meta
      .resolve('./nvidia_Nemotron-3-Nano-30B-A3B-Q4_0.gguf')
      .replace('file://', ''),
    use_mlock: true,
    use_mmap: true,
    flash_attn_type: 'auto',
    cache_type_k: 'q8_0',
    cache_type_v: 'q8_0',
    n_parallel: 2,
    lib_variant: libVariant,
  },
  (progress) => {
    if (progress % 10 === 0) console.log(`Loading ${progress}%`)
  },
)

const tools = [
  {
    type: 'function',
    function: {
      name: 'get_current_time',
      description: 'Get the current time',
      parameters: {},
      required: [],
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_current_date',
      description: 'Get the current date',
      parameters: {},
      required: [],
    },
  },
  {
    type: 'function',
    function: {
      name: 'get_current_weather',
      description: 'Get the current weather',
      parameters: {},
      required: [],
    },
  },
]

await model.parallel.enable({
  n_parallel: 2,
  n_batch: 512,
})

const results = await Promise.all(
  [
    'What is the current time?',
    'What is the current date?',
    'What is the current weather?',
    'What is the current weather and time?',
  ].map(async (message) => {
    const { promise } = await model.parallel.completion(
      {
        reasoning_format: 'auto',
        messages: [
          {
            role: 'system',
            content: 'You are a helpful assistant.',
          },
          {
            role: 'user',
            content: message,
          },
        ],
        tools,
        tool_choice: 'auto',
        n_predict: -1,
        jinja: true
      },
      (requestId, data) => {
        console.log(`Request ${requestId}:`, data.token)
      },
    )
    return promise
  }),
)

console.log(results)

await model.release()
