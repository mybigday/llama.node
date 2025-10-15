import { loadModel } from '../lib/index.js'

const model = await loadModel({
  n_ctx: 8192,
  n_gpu_layers: 99,
  model: import.meta.resolve('./gpt-oss-20b-mxfp4.gguf').replace('file://', ''),
  use_mlock: true,
  use_mmap: true,
  flash_attn_type: 'auto',
  cache_type_k: 'q8_0',
  cache_type_v: 'q8_0',
})

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
  }
]

const result = await model.completion({
  reasoning_format: 'auto',
  messages: [
    {
      role: 'system',
      content: 'You are a helpful assistant.',
    },
    {
      role: 'user',
      content: 'What is the current time?',
    },
  ],
  tools,
  tool_choice: 'auto',
  n_predict: -1,
  jinja: true,
}, (data) => {
  process.stdout.write(data.token)
})

console.log(result)
