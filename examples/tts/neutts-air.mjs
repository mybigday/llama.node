// NeuTTS Nano TTS example. NeuTTS is phoneme-based, so a phonemizer hook is
// required — install the optional `phonemize` package first:
//   npm install phonemize --no-save
//   node examples/tts/download-models.mjs neutts-air
//   node examples/tts/neutts-air.mjs "Text to speak"
import { runTTS } from './run-tts.mjs'

// Load via createRequire — phonemize's ESM build imports JSON without
// import attributes, which newer Node versions reject.
import { createRequire } from 'node:module'

let toIPA
try {
  ;({ toIPA } = createRequire(import.meta.url)('phonemize'))
} catch {
  console.error('NeuTTS requires the `phonemize` package: npm install phonemize --no-save')
  process.exit(1)
}

// Same IPA cleanups as the llama.rn example app.
const phonemizer = (text, language) =>
  toIPA(text, { anyAscii: true, language })
    .replace(/ɫ/g, 'l')
    .replace(/oʊ/g, 'əʊ')
    .replace(/ˈ\b/g, '')

await runTTS('neutts-air', {
  phonemizer,
  // Built-in reference voices: 'default' (jo), 'dave', 'jo'
  speaker: 'default',
})
