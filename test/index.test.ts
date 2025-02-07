import path from 'path'
import waitForExpect from 'wait-for-expect'
import { loadModel, loadLlamaModelInfo } from '../lib'

it('works fine', async () => {
  let tokens = ''
  const model = await loadModel({ model: path.resolve(__dirname, './tiny-random-llama.gguf') })
  const info = model.getModelInfo()
  expect(info).toMatchSnapshot('model info')
  const result = await model.completion({
    prompt: 'My name is Merve and my favorite',
    n_samples: 1,
    temperature: 0,
    max_length: 100,
    n_predict: 10,
    seed: 0,
  }, (data) => {
    expect(data).toMatchObject({ token: expect.any(String) })
    tokens += data.token
  })
  expect({
    ...result,
    timings: `Timings: (${Object.keys(result.timings).length}) keys`
  }).toMatchSnapshot()
  await waitForExpect(() => {
    expect(tokens).toBe(result.text)
  })
  await model.saveSession(path.resolve(__dirname, './tmp.sess'))
  await model.loadSession(path.resolve(__dirname, './tmp.sess'))
  await model.release()
})

it('works fine with vocab_only', async () => {
  const model = await loadModel({ model: path.resolve(__dirname, './tiny-random-llama.gguf'), vocab_only: true })
  expect(model.getModelInfo()).toMatchSnapshot('model info')
  expect(await model.tokenize('Once upon a time')).toMatchSnapshot('tokenize')
  expect(await model.completion({ prompt: 'Once upon a time' })).toMatchSnapshot('empty result')
})

it('tokeneize', async () => {
  const model = await loadModel({ model: path.resolve(__dirname, './tiny-random-llama.gguf') })
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
      "",
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
      })
    expect(result).toMatchSnapshot()
  }
  await model.release()
})

it('embedding', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './bge-small-en.gguf'),
    embedding: true,
    n_gpu_layers: 0,
  })
  const result = await model.embedding('Once upon a time')
  const normalized: number[] = []
  for (let i = 0; i < result.embedding.length; i++) {
    // normalize float to the same between Linux & macOS
    normalized[i] = Math.round(result.embedding[i] * 1000000) / 1000000
  }
  expect(normalized).toMatchSnapshot('Embedding (normalized)')
  await model.release()
})

it('loadModelInfo', async () => {
  const result = await loadLlamaModelInfo(path.resolve(__dirname, './tiny-random-llama.gguf'))
  expect(result).toMatchSnapshot()
})
