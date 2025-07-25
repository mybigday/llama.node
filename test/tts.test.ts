import path from 'path'
import { loadModel } from '../lib'

const modelPath = path.resolve(__dirname, './OuteTTS-0.3-500M-Q4_K_M.gguf');

(modelPath ? test : test.skip)('TTS', async () => {
  const model = await loadModel({
    model: modelPath,
    n_ctx: 8192,
    n_batch: 8192,
    n_ubatch: 128,
    n_threads: 4,
    n_gpu_layers: 99,
    ctx_shift: false,
  });
  await model.initVocoder({
    path: path.resolve(__dirname, './WavTokenizer.gguf'),
    n_batch: 2048,
  });
  expect(model.isVocoderEnabled()).toBe(true);
  const { prompt, grammar } = model.getFormattedAudioCompletion(null, 'Hello, world');
  expect(prompt).toBeDefined();
  expect(grammar).toBeDefined();
  const tokens = model.getAudioCompletionGuideTokens('Hello, world!');
  expect(tokens).toBeInstanceOf(Int32Array);
  const result = await model.completion({
    prompt,
    temperature: 0.2,
    penalty_repeat: 1.2,
    penalty_last_n: 64,
    guide_tokens: tokens,
    top_k: 40,
    top_p: 0.9,
    min_p: 0.05,
    mirostat_tau: 5,
    mirostat_eta: 0.1,
    mirostat: 0,
    grammar,
    n_predict: 3840,
    stop: ['<|audio_end|>', '<|im_end|>'],
  });
  const { audio_tokens: audioTokens } = result;
  console.log(result);
  const audio = await model.decodeAudioTokens(audioTokens!);
  expect(audio.length).toBeGreaterThan(0);
  await model.releaseVocoder();
  expect(model.isVocoderEnabled()).toBe(false);
  await model.release();
}, 1e8);
