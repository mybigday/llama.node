import path from 'path'
import { loadModel } from '../lib'

it('work fine', async () => {
  const model = loadModel({ model: path.resolve(__dirname, './tiny-random-llama.gguf') })
  const result = await model.completion({
    prompt: 'My name is Merve and my favorite',
    n_samples: 1,
    temperature: 0,
    max_length: 100,
    n_predict: 10,
    seed: 0,
  }, (data) => {
    expect(data).toMatchSnapshot('partial')
  })
  expect(result).toMatchSnapshot()
})
