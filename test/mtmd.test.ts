import path from 'path'
import { loadModel } from '../lib'

jest.setTimeout(60000)

test('multimodal with images', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './SmolVLM-256M-Instruct-Q8_0.gguf'),
    // model: path.resolve(__dirname, './ggml-org_gemma-3-4b-it-GGUF_gemma-3-4b-it-Q4_K_M.gguf'),
    n_gpu_layers: 0,
    n_ctx: 512,
  })

  // Initialize multimodal support
  const mmproj_path = path.resolve(
    __dirname,
    './mmproj-SmolVLM-256M-Instruct-Q8_0.gguf',
  )
  // const mmproj_path = path.resolve(__dirname, './ggml-org_gemma-3-4b-it-GGUF_mmproj-model-f16.gguf')
  await model.initMultimodal({ path: mmproj_path, use_gpu: false })

  // Test with multiple images
  const result = await model.completion({
    messages: [
      {
        role: 'system',
        content: 'You are a helpful assistant that can answer questions and help with tasks.',
      },
      {
        role: 'user',
        content: [
          {
            type: 'text', 
            text: 'What is the content of this image?',
          },
          {
            type: 'image_url',
            image_url: {
              url: path.resolve(__dirname, './test-1.jpeg'),
            },
          },
        ],
      },
    ],
    temperature: 0,
    n_predict: 100,
    seed: 0,
  })

  expect(result.text.length).toBeGreaterThan(0)
  expect(result.content).toBe(result.text)
  expect(result).toMatchObject({
    tokens_predicted: expect.any(Number),
    tokens_evaluated: expect.any(Number),
    text: expect.stringMatching(/ A newspaper with/),
  })

  console.log(result)

  expect(await model.isMultimodalEnabled()).toBe(true)
  await model.releaseMultimodal()
  expect(await model.isMultimodalEnabled()).toBe(false)

  await model.release()
})
