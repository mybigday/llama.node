import path from 'path'
import fs from 'fs'
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

  const formatted = model.getFormattedChat(
    [
      {
        role: 'system',
        content:
          'You are a helpful assistant that can answer questions and help with tasks.',
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
    undefined,
    { jinja: false },
  )

  expect(
    await model.tokenize(formatted.prompt, {
      media_paths: formatted.media_paths,
    }),
  ).toMatchSnapshot()

  expect(formatted.media_paths).toHaveLength(1)
  expect({
    ...formatted,
    media_paths: ['<img-path>'],
  }).toMatchSnapshot()

  // Test with multiple images
  const result = await model.completion({
    ...formatted,
    temperature: 0,
    n_predict: 100,
    seed: 0,
  })

  expect(result.text.length).toBeGreaterThan(0)
  expect(result.content).toBe(result.text)
  expect(result).toMatchObject({
    tokens_predicted: expect.any(Number),
    tokens_evaluated: expect.any(Number),
    text: expect.stringMatching(/[A-Za-z0-9\s\.\:\-\,\!\?]+$/),
  })

  console.log(result)

  expect(await model.isMultimodalEnabled()).toBe(true)
  
  // Test multimodal support capabilities
  const support = await model.getMultimodalSupport()
  expect(support).toHaveProperty('vision', true)
  expect(support).toHaveProperty('audio', false)
  
  await model.releaseMultimodal()
  expect(await model.isMultimodalEnabled()).toBe(false)

  await model.release()
})

const modelPath = path.resolve(__dirname, './Llama-3.2-1B-Instruct-Q4_K_M.gguf');

;(fs.existsSync(modelPath) ? test : test.skip)('multimodal with audio', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './Llama-3.2-1B-Instruct-Q4_K_M.gguf'),
    n_gpu_layers: 0,
    n_ctx: 512,
  })

  const mmproj_path = path.resolve(__dirname, './mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf')

  await model.initMultimodal({ path: mmproj_path, use_gpu: false })

  const formatted = model.getFormattedChat([
    {
      role: 'system',
      content: 'You are a helpful assistant that can answer questions and help with tasks.',
    },
    {
      role: 'user',
      content: [
        {
          type: 'text',
          text: 'What is the content of this audio?',
        },
        {
          type: 'input_audio',
          input_audio: {
            url: path.resolve(__dirname, './jfk.wav'),
            format: 'wav',
          }
        }
      ]
    }
  ])

  expect(formatted.media_paths).toHaveLength(1)
  expect({
    ...formatted,
    media_paths: ['<audio-path>'],
  }).toMatchSnapshot()


  // Test with multiple images
  const result = await model.completion({
    ...formatted,
    temperature: 0,
    n_predict: 100,
    seed: 0,
  })

  expect(result.text.length).toBeGreaterThan(0)
  expect(result.content).toBe(result.text)
  expect(result).toMatchObject({
    tokens_predicted: expect.any(Number),
    tokens_evaluated: expect.any(Number),
    text: expect.stringMatching(/The content of this audio is a/),
  })

  console.log(result)

  expect(await model.isMultimodalEnabled()).toBe(true)
  
  // Test multimodal support capabilities
  const support = await model.getMultimodalSupport()
  expect(support).toHaveProperty('vision', false)
  expect(support).toHaveProperty('audio', true)

  await model.releaseMultimodal()
  expect(await model.isMultimodalEnabled()).toBe(false)

  await model.release()
})