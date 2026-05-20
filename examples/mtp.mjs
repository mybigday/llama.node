import fs from 'fs'
import { loadModel } from '../lib/index.js'

const modelPath = process.argv[2] || process.env.MODEL_PATH

if (!modelPath) {
  console.error('Usage: node examples/mtp.mjs <model_path>')
  console.error('  or set MODEL_PATH=/path/to/model.gguf')
  console.error('')
  console.error('Examples:')
  console.error('  node examples/mtp.mjs ./models/Qwen3.6-27B-MTP-Q8_0.gguf')
  console.error('  MTP_SPECULATIVE=0 node examples/mtp.mjs ./models/Qwen3.6-27B-Q8_0.gguf')
  process.exit(1)
}

if (!fs.existsSync(modelPath)) {
  console.error(`Model not found: ${modelPath}`)
  process.exit(1)
}

const libVariant = process.env.LLAMA_LIB_VARIANT || 'default'
const speculativeEnabled = !['0', 'false', 'off', 'no'].includes(
  (process.env.MTP_SPECULATIVE || '1').toLowerCase(),
)
const draftTokens = Number.parseInt(process.env.MTP_DRAFT_N || '3', 10)
const nPredict = Number.parseInt(process.env.MTP_MAX_TOKENS || '128', 10)
const nCtx = Number.parseInt(process.env.MTP_N_CTX || '4096', 10)
const nGpuLayers = Number.parseInt(process.env.MTP_N_GPU_LAYERS || '99', 10)
const cacheTypeK = process.env.MTP_CACHE_TYPE_K || 'q8_0'
const cacheTypeV = process.env.MTP_CACHE_TYPE_V || 'q8_0'
const prompt =
  process.argv.slice(3).join(' ') ||
  process.env.MTP_PROMPT ||
  'Write a concise TypeScript function that groups an array of objects by a key.'

console.log(`Loading model: ${modelPath}`)
console.log(`Library variant: ${libVariant}`)
console.log(`Speculative decoding: ${speculativeEnabled ? 'enabled' : 'disabled'}`)
if (speculativeEnabled) {
  console.log(`MTP draft tokens: ${draftTokens}`)
  console.log('Set MTP_SPECULATIVE=0 to disable speculative decoding.')
}

if (!speculativeEnabled) {
  console.log('Set MTP_SPECULATIVE=1 to enable draft-mtp speculative decoding.')
}

const context = await loadModel(
  {
    model: modelPath,
    n_ctx: nCtx,
    n_batch: 1024,
    n_gpu_layers: nGpuLayers,
    flash_attn_type: 'auto',
    cache_type_k: cacheTypeK,
    cache_type_v: cacheTypeV,
    ...(speculativeEnabled
      ? {
          speculative: {
            type: 'draft-mtp',
            n_max: draftTokens,
          },
        }
      : {}),
    lib_variant: libVariant,
  },
  (progress) => {
    if (progress % 10 === 0) process.stdout.write(`\rLoading ${progress}%`)
  },
)

console.log('\n')

const started = performance.now()
const result = await context.completion(
  {
    messages: [
      {
        role: 'user',
        content: prompt,
      },
    ],
    chat_template_kwargs: {
      preserve_thinking: true,
    },
    n_predict: nPredict,
    temperature: 0.6,
    top_k: 20,
    top_p: 0.95,
    speculative: speculativeEnabled
      ? {
          type: 'draft-mtp',
          n_max: draftTokens,
        }
      : false,
  },
  (data) => {
    process.stdout.write(data.token)
  },
)
const elapsedSeconds = (performance.now() - started) / 1000
const draftTotal = result.draft_tokens || 0
const draftAccepted = result.draft_tokens_accepted || 0

const acceptRate = draftTotal > 0 ? draftAccepted / draftTotal : 0

console.log('\n')
console.log('MTP metrics:')
console.log(`  predicted: ${result.tokens_predicted}`)
console.log(`  drafted: ${draftTotal}`)
console.log(`  accepted: ${draftAccepted}`)
console.log(`  accept_rate: ${acceptRate.toFixed(3)}`)
console.log(`  wall_seconds: ${elapsedSeconds.toFixed(2)}`)
console.log(
  `  tokens_per_second: ${(result.tokens_predicted / elapsedSeconds).toFixed(2)}`,
)

await context.release()
