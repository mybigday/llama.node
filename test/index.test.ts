import path from 'path'
import waitForExpect from 'wait-for-expect'
import { loadModel, loadLlamaModelInfo, toggleNativeLog, addNativeLogListener } from '../lib'

const filterCompletionResult = (result: any) => {
  return {
    ...result,
    content: '<removed>',
    timings: `Timings: (${Object.keys(result.timings).length}) keys`,
  }
}

test('basic completion', async () => {
  let tokens = ''
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
  })
  const info = model.getModelInfo()
  expect(info).toMatchSnapshot('model info')
  const result = await model.completion(
    {
      prompt: 'My name is Merve and my favorite',
      temperature: 0,
      n_predict: 10,
      seed: 0,
    },
    (data) => {
      expect(data).toMatchObject({ token: expect.any(String) })
      tokens += data.token
    },
  )
  expect(filterCompletionResult(result)).toMatchSnapshot()
  await waitForExpect(() => {
    expect(tokens).toBe(result.text)
    expect(result.content).toBe(result.text)
  })
  await model.saveSession(path.resolve(__dirname, './tmp.sess'))
  await model.loadSession(path.resolve(__dirname, './tmp.sess'))
  await model.release()
})

test('completion with response_format', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
  })
  expect(
    filterCompletionResult(
      await model.completion({
        temperature: 0,
        max_length: 100,
        n_predict: 5,
        seed: 0,
        prompt: 'My name is Merve and my favorite',
        response_format: {
          type: 'json_schema',
          json_schema: {
            schema: {
              type: 'object',
              properties: {
                name: { type: 'string' },
                age: { type: 'number' },
              },
              required: ['name', 'age'],
            },
            strict: true,
          },
        },
      }),
    ),
  ).toMatchSnapshot('json_schema')
  expect(
    filterCompletionResult(
      await model.completion({
        temperature: 0,
        max_length: 100,
        n_predict: 5,
        seed: 0,
        prompt: 'My name is Merve and my favorite',
        response_format: {
          type: 'json_object',
        },
      }),
    ),
  ).toMatchSnapshot('json_object')
})

test('completion with tools', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
  })
  expect(
    filterCompletionResult(
      await model.completion({
        n_predict: 10,
        temperature: 0,
        max_length: 100,
        seed: 0,
        jinja: true,
        messages: [
          {
            role: 'user',
            content: 'My name is Merve and my favorite',
          },
        ],
        tools: [
          {
            type: 'function',
            function: {
              name: 'ipython',
              description:
                'Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.',
              parameters: {
                type: 'object',
                properties: {
                  code: {
                    type: 'string',
                    description: 'The code to run in the ipython interpreter.',
                  },
                },
                required: ['code'],
              },
            },
          },
        ],
        tool_choice: 'auto',
      }),
    ),
  ).toMatchSnapshot()
})

test('works fine with vocab_only', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
    vocab_only: true,
  })
  expect(model.getModelInfo()).toMatchSnapshot('model info')
  expect(await model.tokenize('Once upon a time')).toMatchSnapshot('tokenize')
  expect(
    await model.completion({ prompt: 'Once upon a time' }),
  ).toMatchSnapshot('empty result')
})

test('tokeneize & detokenize & getFormattedChat', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
  })
  {
    const result = await model.tokenize('Once upon a time')
    expect(result).toMatchSnapshot()
  }
  {
    const result = await model.detokenize([123, 123, 123])
    expect(result).toMatchSnapshot()
  }
  {
    const result = model.getFormattedChat([
      { role: 'user', content: 'Hello' },
      { role: 'bot', content: 'Hi' },
    ])
    expect(result).toMatchSnapshot()
  }
  {
    const result = model.getFormattedChat(
      [
        { role: 'user', content: 'Hello' },
        { role: 'bot', content: 'Hi' },
      ],
      '',
      {
        jinja: true,
        tools: [
          {
            type: 'function',
            function: {
              name: 'ipython',
              description:
                'Runs code in an ipython interpreter and returns the result of the execution after 60 seconds.',
              parameters: {
                type: 'object',
                properties: {
                  code: {
                    type: 'string',
                    description: 'The code to run in the ipython interpreter.',
                  },
                },
                required: ['code'],
              },
            },
          },
        ],
        tool_choice: 'auto',
      },
    )
    expect(result).toMatchSnapshot()
  }
  await model.release()
})

test('embedding', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './bge-small-en.gguf'),
    embedding: true,
    n_gpu_layers: 0,
  })
  const result = await model.embedding('Once upon a time')
  const normalized: number[] = []
  for (let i = 0; i < result.embedding.length; i++) {
    // normalize float to the same between Linux & macOS
    normalized[i] = Math.round(result.embedding[i] * 100000) / 100000
  }
  expect(normalized).toMatchSnapshot('Embedding (normalized)')
  await model.release()
})

test('loadModelInfo', async () => {
  const result = await loadLlamaModelInfo(
    path.resolve(__dirname, './tiny-random-llama.gguf'),
  )
  expect(result).toMatchSnapshot()
})

test('toggleNativeLog', async () => {
  const logs: string[] = []
  await toggleNativeLog(true)
  const listener = addNativeLogListener((level, text) => {
    logs.push(`${level}: ${text}`)
  })
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
    n_gpu_layers: 0,
  })
  await model.release()
  listener.remove()
  expect(logs.length > 10).toBe(true)
  await toggleNativeLog(false)
})

test('ctx_shift and context_full', async () => {
  // Load model with ctx_shift disabled to test context_full flag
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
    n_ctx: 20, // Small context size to easily trigger context_full
    ctx_shift: false, // Disable context shifting
  })
  
  // Generate a completion that should fill the context
  const result1 = await model.completion({
    prompt: 'Testing context full with a fairly long prompt to ensure we trigger the context_full flag',
    n_predict: 100, // Try to generate a lot of tokens
  })
  
  // The context_full flag should be set to true
  expect(result1.context_full).toBe(true)
  expect(result1.truncated).toBe(false)
  
  // Now test with ctx_shift enabled
  const model2 = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
    n_ctx: 20, // Small context size
    ctx_shift: true, // Enable context shifting
  })
  
  // Generate a completion that should use context shifting
  const result2 = await model2.completion({
    prompt: 'Testing context shifting with a fairly long prompt that should continue generating',
    n_predict: 20, // Try to generate a lot of tokens
  })
  
  // The context_full flag should be false since context shifting is enabled
  expect(result2.context_full).toBe(false)
  // The truncated flag should be true because context was shifted
  expect(result2.truncated).toBe(true)
  
  // Clean up
  await model.release()
  await model2.release()
})
