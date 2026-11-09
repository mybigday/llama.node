import path from 'path'
import fs from 'fs'
import * as wav from 'node-wav'
import { loadModel } from '../lib'

// Smallest codec.cpp TTS pair (~281MB total) — see scripts/download-test-models.js
const modelPath = path.resolve(__dirname, './Soprano-1.1-80M.F16.gguf')
const vocoderPath = path.resolve(__dirname, './Soprano-codec-F32.gguf')

const testIf = fs.existsSync(modelPath) && fs.existsSync(vocoderPath) ? test : test.skip

testIf(
  'TTS (codec.cpp)',
  async () => {
    const model = await loadModel({
      model: modelPath,
      n_ctx: 4096,
      n_batch: 4096,
      n_threads: 4,
      n_gpu_layers: 99,
      ctx_shift: false,
    })
    expect(model.initVocoder({ path: vocoderPath, n_batch: 4096 })).toBe(true)
    expect(model.isVocoderEnabled()).toBe(true)

    const caps = model.getTTSCapabilities()
    expect(caps.family).toBe('soprano')
    expect(typeof caps.type).toBe('number')
    expect(Array.isArray(caps.builtinVoices)).toBe(true)
    expect(typeof caps.needsVoice).toBe('boolean')

    const sampleRate = model.getAudioSampleRate()
    expect(sampleRate).toBeGreaterThan(0)

    const text = 'Hello, my name is John Doe.'
    const formatted = await model.getFormattedAudioCompletion({ prompt: text })
    expect(formatted.prompt).toBeTruthy()
    expect(formatted.completionParams.prompt).toBe(formatted.prompt)

    await expect(
      model.getFormattedAudioCompletion({ prompt: text, speaker: 'no-such-voice' }),
    ).rejects.toThrow(/Unknown voice/)

    // completionParams carries the prompt, grammar / logit_bias mask,
    // embedding and n_min_predict; sampling defaults to the GGUF's.
    const result = await model.completion({
      ...formatted.completionParams,
      n_predict: 512,
    })

    let audio: Float32Array
    if (result.audio_latents?.length) {
      audio = await model.decodeAudioLatents(result.audio_latents)
    } else {
      expect(result.audio_tokens?.length).toBeGreaterThan(0)
      audio = await model.decodeAudioTokens(result.audio_tokens!)
    }
    expect(audio.length).toBeGreaterThan(0)

    fs.writeFileSync(
      path.resolve(__dirname, './tts-result.wav'),
      wav.encode([audio], { sampleRate, bitDepth: 16 }),
    )

    model.releaseVocoder()
    expect(model.isVocoderEnabled()).toBe(false)
    await model.release()
  },
  1e8,
)
