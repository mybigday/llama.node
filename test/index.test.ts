import path from 'path'
import waitForExpect from 'wait-for-expect'
import { loadModel } from '../lib'

it('work fine', async () => {
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

it('work fine with vocab_only', async () => {
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
      { role: 'user', text: 'Hello' },
      { role: 'bot', text: 'Hi' },
    ])
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
  expect(result).toMatchSnapshot()
  await model.release()
})
