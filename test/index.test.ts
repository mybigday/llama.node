import path from 'path'
import waitForExpect from 'wait-for-expect'
import { loadModel } from '../lib'

it('work fine', async () => {
  let tokens = ''
  const model = await loadModel({ model: path.resolve(__dirname, './tiny-random-llama.gguf') })
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
  expect(result).toMatchSnapshot()
  await waitForExpect(() => {
    expect(tokens).toBe(result.text)
  })
  await model.release()
})
