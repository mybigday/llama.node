// Shared TTS runner used by the per-model example scripts.
// Mirrors llama.rn's canonical flow:
//   loadModel → initVocoder → getTTSCapabilities → getFormattedAudioCompletion
//   → completion({ ...completionParams, speaker }) → decodeAudioTokens /
//   decodeAudioLatents → WAV file.
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { loadModel } from '../../lib/index.js'
import { TTS_MODELS, getLocalPath } from './models.mjs'

const MODELS_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'models')

// Minimal mono 16-bit WAV encoder to avoid an extra dependency.
const encodeWav = (samples, sampleRate) => {
  const buf = Buffer.alloc(44 + samples.length * 2)
  buf.write('RIFF', 0)
  buf.writeUInt32LE(36 + samples.length * 2, 4)
  buf.write('WAVE', 8)
  buf.write('fmt ', 12)
  buf.writeUInt32LE(16, 16)
  buf.writeUInt16LE(1, 20) // PCM
  buf.writeUInt16LE(1, 22) // mono
  buf.writeUInt32LE(sampleRate, 24)
  buf.writeUInt32LE(sampleRate * 2, 28)
  buf.writeUInt16LE(2, 32)
  buf.writeUInt16LE(16, 34)
  buf.write('data', 36)
  buf.writeUInt32LE(samples.length * 2, 40)
  for (let i = 0; i < samples.length; i++) {
    const s = Math.max(-1, Math.min(1, samples[i]))
    buf.writeInt16LE(Math.round(s * 0x7fff), 44 + i * 2)
  }
  return buf
}

export const runTTS = async (modelKey, options = {}) => {
  const def = TTS_MODELS[modelKey]
  if (!def) throw new Error(`Unknown TTS model key: ${modelKey}`)

  const modelPath = getLocalPath(MODELS_DIR, def.model.repo, def.model.filename)
  const vocoderPath = getLocalPath(MODELS_DIR, def.vocoder.repo, def.vocoder.filename)
  if (!fs.existsSync(modelPath) || !fs.existsSync(vocoderPath)) {
    console.error(`Model files for '${modelKey}' not found. Download first:`)
    console.error(`  node examples/tts/download-models.mjs ${modelKey}`)
    process.exit(1)
  }

  const text =
    options.text ||
    process.argv[2] ||
    'Hello! This is a text to speech demo running locally with llama dot node.'

  console.log(`Loading ${def.name}...`)
  const context = await loadModel({
    model: modelPath,
    n_ctx: 4096,
    n_batch: 4096,
    n_gpu_layers: 99,
    ctx_shift: false,
    ...def.contextParams,
    ...options.contextParams,
  })

  console.log('Loading vocoder/codec...')
  context.initVocoder({ path: vocoderPath, n_batch: 4096 })

  const caps = context.getTTSCapabilities()
  console.log('TTS capabilities:', caps)

  // A voice cloned from a reference clip (models with canSpeakerFromAudio),
  // else a builtin voice name, else the model's default voice.
  let speaker = options.speaker
  if (options.refAudio) {
    speaker = context.createSpeaker(options.refAudio)
  }

  const { completionParams } = await context.getFormattedAudioCompletion({
    prompt: text,
    speaker,
    phonemizer: options.phonemizer,
    instruct: options.instruct,
  })

  // completionParams holds what the model's format needs (prompt, grammar /
  // logit_bias mask, embedding, n_min_predict); sampling defaults to the
  // backbone GGUF's general.sampling.*.
  console.log('Generating audio...')
  const startTime = Date.now()
  const result = await context.completion({
    ...completionParams,
    speaker,
    n_predict: options.n_predict || 4096,
    ...options.completionParams,
  })

  const sampleRate = context.getAudioSampleRate()
  let audio
  if (result.audio_latents?.length) {
    console.log(`Decoding ${result.audio_latents.length} latent values...`)
    audio = await context.decodeAudioLatents(result.audio_latents)
  } else if (result.audio_tokens?.length) {
    console.log(`Decoding ${result.audio_tokens.length} audio tokens...`)
    audio = await context.decodeAudioTokens(result.audio_tokens)
  } else {
    throw new Error('No audio produced (no audio_tokens or audio_latents in result)')
  }

  const outPath =
    options.output || path.join(path.dirname(fileURLToPath(import.meta.url)), `${modelKey}.wav`)
  fs.writeFileSync(outPath, encodeWav(audio, sampleRate))
  const seconds = (audio.length / sampleRate).toFixed(2)
  console.log(
    `\n✓ Wrote ${outPath} (${seconds}s @ ${sampleRate}Hz, generated in ${((Date.now() - startTime) / 1000).toFixed(1)}s)`,
  )

  if (speaker && typeof speaker === 'object') speaker.release()
  context.releaseVocoder()
  await context.release()
}
