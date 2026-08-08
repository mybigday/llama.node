// Download TTS model pairs (backbone + codec/vocoder GGUF) from HuggingFace.
//
// Usage:
//   node examples/tts/download-models.mjs <model-key>[ <model-key> ...]
//   node examples/tts/download-models.mjs all
//   node examples/tts/download-models.mjs            # list available models
import fs from 'node:fs'
import path from 'node:path'
import { fileURLToPath } from 'node:url'
import { TTS_MODELS, getModelDownloadUrl, getLocalPath } from './models.mjs'

const MODELS_DIR = path.join(path.dirname(fileURLToPath(import.meta.url)), 'models')

const listModels = () => {
  console.log('Available TTS models:\n')
  for (const [key, def] of Object.entries(TTS_MODELS)) {
    console.log(`  ${key.padEnd(20)} ${def.name}`)
    console.log(`  ${''.padEnd(20)} model: ${def.model.size}, vocoder: ${def.vocoder.size}\n`)
  }
  console.log('Usage: node examples/tts/download-models.mjs <model-key>|all')
}

const downloadFile = async (url, outputPath) => {
  if (fs.existsSync(outputPath)) {
    console.log(`✓ ${path.basename(outputPath)} already exists, skipping`)
    return
  }
  fs.mkdirSync(path.dirname(outputPath), { recursive: true })
  console.log(`Downloading ${url}`)
  const res = await fetch(url)
  if (!res.ok) throw new Error(`HTTP ${res.status} for ${url}`)
  const total = Number(res.headers.get('content-length')) || 0
  const tmpPath = `${outputPath}.download`
  const file = fs.createWriteStream(tmpPath)
  let received = 0
  let lastLog = 0
  for await (const chunk of res.body) {
    received += chunk.length
    if (total && Date.now() - lastLog > 2000) {
      lastLog = Date.now()
      process.stdout.write(`\r  ${((received / total) * 100).toFixed(1)}% of ${(total / 1e6).toFixed(0)}MB`)
    }
    if (!file.write(chunk)) {
      await new Promise((resolve) => file.once('drain', resolve))
    }
  }
  await new Promise((resolve, reject) => {
    file.end(() => resolve())
    file.on('error', reject)
  })
  process.stdout.write('\n')
  fs.renameSync(tmpPath, outputPath)
  console.log(`✓ Saved to ${outputPath}`)
}

const keys = process.argv.slice(2)
if (keys.length === 0) {
  listModels()
  process.exit(0)
}

const targets = keys.includes('all') ? Object.keys(TTS_MODELS) : keys
for (const key of targets) {
  const def = TTS_MODELS[key]
  if (!def) {
    console.error(`Unknown model key: ${key}`)
    listModels()
    process.exit(1)
  }
  console.log(`\n=== ${def.name} ===`)
  await downloadFile(
    getModelDownloadUrl(def.model.repo, def.model.filename),
    getLocalPath(MODELS_DIR, def.model.repo, def.model.filename),
  )
  await downloadFile(
    getModelDownloadUrl(def.vocoder.repo, def.vocoder.filename),
    getLocalPath(MODELS_DIR, def.vocoder.repo, def.vocoder.filename),
  )
}
console.log('\nDone.')
