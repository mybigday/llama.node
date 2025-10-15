import { loadModel } from '../lib/index.js'

const model = await loadModel({
  n_gpu_layers: 99,
  model: import.meta.resolve('./gpt-oss-20b-mxfp4.gguf').replace('file://', ''),
  n_parallel: 4,
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
  },
]

await model.parallel.enable({
  n_parallel: 4,
  n_batch: 512,
})

const results = await Promise.all(
  [
    'What is the current time?',
    'What is the current date?',
    'What is the current weather?',
  ].map(async (message) => {
    const { promise } = model.parallel.completion(
      {
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
        jinja: true,
      },
      (requestId, data) => {
        console.log(`Request ${requestId}:`, data.token)
      },
    )
    await promise
  }
))

console.log(results)