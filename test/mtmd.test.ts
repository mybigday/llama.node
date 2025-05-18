import path from 'path'
import { loadModel } from '../lib'

jest.setTimeout(60000)

test('multimodal with images', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './SmolVLM-256M-Instruct-Q8_0.gguf'),
    // model: path.resolve(__dirname, './ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf'),
    n_gpu_layers: 0,
  })

  // Initialize multimodal support
  const mmproj_path = path.resolve(
    __dirname,
    './mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',
  )
  // const mmproj_path = path.resolve(__dirname, './ggml-org_gemma-3-4b-it-GGUF_mmproj-model-f16.gguf')
  await model.initMultimodal(mmproj_path)

  // Test with multiple images
  const result = await model.completion({
    messages: [
      {
        role: 'user',
        content: 'What is the content of this image? <__image__>',
      },
    ],
    temperature: 0.2,
    max_length: 1024,
    n_predict: 150,
    seed: 42,
    image_paths: [path.resolve(__dirname, './test-1.jpeg')],
  })

  expect(result.text.length).toBeGreaterThan(0)
  expect(result).toMatchObject({
    tokens_predicted: expect.any(Number),
    tokens_evaluated: expect.any(Number),
    text: " A newspaper with the headline 'men walk on moon' has pictures of astronauts and a flag.<end_of_utterance>",
  })

  await model.release()
})
