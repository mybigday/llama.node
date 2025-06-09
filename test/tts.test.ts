import path from 'path'
import waitForExpect from 'wait-for-expect'
import { loadModel, loadLlamaModelInfo, toggleNativeLog, addNativeLogListener } from '../lib'

test('TTS', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './tiny-random-llama.gguf'),
    chat_template: 'outetts-0.3',
  })
  await model.initVocoder({ path: path.resolve(__dirname, './WavTokenizer.gguf') })
  expect(model.isVocoderEnabled()).toBe(true)
  const formatted = model.getFormattedAudioCompletion(null, 'Hello, world! 中文 にほんご 한국어')
  expect(formatted).toMatchSnapshot('formatted')
  const tokens = model.getAudioCompletionGuideTokens('Hello, world!')
  expect(tokens).toMatchSnapshot('guide tokens')
  const audio = await model.decodeAudioTokens(new Int32Array([151672, 151673, 151674]))
  expect(audio.length).toBeGreaterThan(0)
  await model.releaseVocoder()
  expect(model.isVocoderEnabled()).toBe(false)
  await model.release()
})
