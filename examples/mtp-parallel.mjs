import fs from 'fs'
import { loadModel } from '../lib/index.js'

const modelPath = process.argv[2] || process.env.MODEL_PATH

if (!modelPath) {
  console.error('Usage: node examples/mtp-parallel.mjs <model_path> [prompt ...]')
  console.error('  or set MODEL_PATH=/path/to/model.gguf')
  console.error('')
  console.error('Examples:')
  console.error('  node examples/mtp-parallel.mjs ./models/Qwen3.6-27B-MTP-Q8_0.gguf')
  console.error('  MTP_SPECULATIVE=0 node examples/mtp-parallel.mjs ./models/Qwen3.6-27B-Q8_0.gguf')
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
const nBatch = Number.parseInt(process.env.MTP_N_BATCH || '1024', 10)
const nParallel = Number.parseInt(process.env.MTP_PARALLEL || '2', 10)
const nGpuLayers = Number.parseInt(process.env.MTP_N_GPU_LAYERS || '99', 10)
const cacheTypeK = process.env.MTP_CACHE_TYPE_K || 'q8_0'
const cacheTypeV = process.env.MTP_CACHE_TYPE_V || 'q8_0'

const cliPrompt = process.argv.slice(3).join(' ').trim()
const envPrompts = (process.env.MTP_PROMPTS || '')
  .split('|||')
  .map((prompt) => prompt.trim())
  .filter(Boolean)
const prompts =
  cliPrompt.length > 0
    ? [cliPrompt]
    : envPrompts.length > 0
      ? envPrompts
      : [
          'Write a concise TypeScript function that groups an array of objects by a key.',
          'Explain why speculative decoding can improve local inference throughput.',
          'Create a short checklist for validating a Node.js native addon release.',
          'Summarize the tradeoffs of cache quantization in two paragraphs.',
        ]

const formatMs = (value) =>
  Number.isFinite(value) ? `${value.toFixed(2)} ms` : 'n/a'
const formatRate = (value) =>
  Number.isFinite(value) ? `${value.toFixed(2)} tok/s` : 'n/a'
const formatMsPerToken = (value) =>
  Number.isFinite(value) ? `${value.toFixed(2)} ms/tok` : 'n/a'

console.log(`Loading model: ${modelPath}`)
console.log(`Library variant: ${libVariant}`)
console.log(`Parallel slots: ${nParallel}`)
console.log(`Speculative decoding: ${speculativeEnabled ? 'enabled' : 'disabled'}`)
if (speculativeEnabled) {
  console.log(`MTP draft tokens: ${draftTokens}`)
  console.log('Set MTP_SPECULATIVE=0 to disable speculative decoding.')
} else {
  console.log('Set MTP_SPECULATIVE=1 to enable draft-mtp speculative decoding.')
}

const context = await loadModel(
  {
    model: modelPath,
    n_ctx: nCtx,
    n_batch: nBatch,
    n_parallel: nParallel,
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
try {
  await context.parallel.enable({
    n_parallel: nParallel,
    n_batch: nBatch,
  })

  const started = performance.now()
  const requests = await Promise.all(
    prompts.map(async (prompt, index) => {
      const requestStarted = performance.now()
      const { requestId, promise } = await context.parallel.completion(
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
        (_requestId, tokenEvent) => {
          console.log(`[${index + 1}] ${JSON.stringify(tokenEvent)}`)
        },
      )

      const result = await promise
      const wallMs = performance.now() - requestStarted
      return { index, prompt, requestId, result, wallMs }
    }),
  )
  const elapsedSeconds = (performance.now() - started) / 1000

  let totalPredicted = 0
  let totalDrafted = 0
  let totalAccepted = 0

  for (const { index, prompt, requestId, result, wallMs } of requests) {
    const draftTotal = result.draft_tokens || 0
    const draftAccepted = result.draft_tokens_accepted || 0
    const timings = result.timings || {}

    totalPredicted += result.tokens_predicted || 0
    totalDrafted += draftTotal
    totalAccepted += draftAccepted

    console.log(`\n[${index + 1}] request ${requestId}`)
    console.log(`Prompt: ${prompt}`)
    console.log(result.text.trim())
    console.log(
      `tokens=${result.tokens_predicted} drafted=${draftTotal} accepted=${draftAccepted}`,
    )
    console.log(
      `timings: wall=${formatMs(wallMs)} prompt=${timings.prompt_n || 0} tok in ${formatMs(timings.prompt_ms)} (${formatRate(timings.prompt_per_second)}, ${formatMsPerToken(timings.prompt_per_token_ms)})`,
    )
    console.log(
      `         predicted=${timings.predicted_n || 0} tok in ${formatMs(timings.predicted_ms)} (${formatRate(timings.predicted_per_second)}, ${formatMsPerToken(timings.predicted_per_token_ms)}) cache=${timings.cache_n || 0} tok`,
    )
  }

  const acceptRate = totalDrafted > 0 ? totalAccepted / totalDrafted : 0

  console.log('\nMTP parallel metrics:')
  console.log(`  requests: ${requests.length}`)
  console.log(`  predicted: ${totalPredicted}`)
  console.log(`  drafted: ${totalDrafted}`)
  console.log(`  accepted: ${totalAccepted}`)
  console.log(`  accept_rate: ${acceptRate.toFixed(3)}`)
  console.log(`  wall_seconds: ${elapsedSeconds.toFixed(2)}`)
  console.log(`  tokens_per_second: ${(totalPredicted / elapsedSeconds).toFixed(2)}`)
} finally {
  await context.release()
}
