import path from 'path'
import fs from 'fs'
import * as wav from 'node-wav'
import { loadModel } from '../lib'
import speaker from './speaker.json'

const modelPath = path.resolve(__dirname, './OuteTTS-0.3-500M-Q4_K_M.gguf');
const vocoderPath = path.resolve(__dirname, './WavTokenizer.gguf');

(modelPath && vocoderPath ? test : test.skip)('TTS', async () => {
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
    path: vocoderPath,
    n_batch: 2048,
  });
  expect(model.isVocoderEnabled()).toBe(true);
  const text = 'Hello, my name is John Doe';
  const { prompt, grammar } = model.getFormattedAudioCompletion(JSON.stringify(speaker), text);
  expect(prompt).toBeDefined();
  expect(grammar).toBeDefined();
  const tokens = model.getAudioCompletionGuideTokens(text);
  expect(tokens).toBeDefined();
  const result = await model.completion({
    prompt,
    temperature: 0.4,
    penalty_repeat: 1.1,
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
  fs.writeFileSync(
    path.resolve(__dirname, './tts-result.wav'),
    wav.encode([audio], { sampleRate: 24000, bitDepth: 16 }),
  );
  await model.releaseVocoder();
  expect(model.isVocoderEnabled()).toBe(false);
  await model.release();
}, 1e8);
