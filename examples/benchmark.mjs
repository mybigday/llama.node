import { loadModel } from '../lib/index.js'

const modelPath = process.argv[2] || process.env.MODEL_PATH
if (!modelPath) {
  console.error('Usage: node benchmark.mjs <model_path>')
  console.error('  or set MODEL_PATH environment variable')
  process.exit(1)
}

const libVariant = process.env.LLAMA_LIB_VARIANT || 'default'

console.log(`Loading model: ${modelPath}`)
console.log(`Library variant: ${libVariant}\n`)

const context = await loadModel(
  {
    model: modelPath,
    n_ctx: 2048,
    n_gpu_layers: 99,
    lib_variant: libVariant,
  },
  (progress) => {
    if (progress % 10 === 0) process.stdout.write(`\rLoading ${progress}%`)
  },
)

console.log('\n')

// Warmup run
console.log('Warming up...')
await context.bench(8, 4, 1, 1)

// Main benchmark parameters
const pp = 512 // prompt processing tokens
const tg = 128 // text generation tokens
const pl = 1 // parallel level
const nr = 3 // number of repetitions

console.log('Running benchmark...\n')
const result = await context.bench(pp, tg, pl, nr)

// Print configuration
console.log('Configuration:')
console.log(`  n_kv_max=${result.nKvMax}, n_batch=${result.nBatch}, n_ubatch=${result.nUBatch}`)
console.log(`  flash_attn=${result.flashAttn}, is_pp_shared=${result.isPpShared}`)
console.log(`  n_gpu_layers=${result.nGpuLayers}, n_threads=${result.nThreads}, n_threads_batch=${result.nThreadsBatch}`)
console.log()

// Print results table header
console.log('Results:')
console.log(
  '|    pp |    tg |  pl |  n_kv |    t_pp |   speed_pp |    t_tg |   speed_tg |       t |    speed |',
)
console.log(
  '|-------|-------|-----|-------|---------|------------|---------|------------|---------|----------|',
)

// Print results
console.log(
  `| ${String(result.pp).padStart(5)} ` +
    `| ${String(result.tg).padStart(5)} ` +
    `| ${String(result.pl).padStart(3)} ` +
    `| ${String(result.nKv).padStart(5)} ` +
    `| ${result.tPp.toFixed(3).padStart(7)} ` +
    `| ${result.speedPp.toFixed(2).padStart(10)} ` +
    `| ${result.tTg.toFixed(3).padStart(7)} ` +
    `| ${result.speedTg.toFixed(2).padStart(10)} ` +
    `| ${result.t.toFixed(3).padStart(7)} ` +
    `| ${result.speed.toFixed(2).padStart(8)} |`,
)

console.log()
console.log('Legend:')
console.log('  pp: prompt processing tokens')
console.log('  tg: text generation tokens')
console.log('  pl: parallel level')
console.log('  n_kv: KV cache used')
console.log('  t_pp: time for prompt processing (ms)')
console.log('  speed_pp: prompt processing speed (tokens/sec)')
console.log('  t_tg: time for text generation (ms)')
console.log('  speed_tg: text generation speed (tokens/sec)')
console.log('  t: total time (ms)')
console.log('  speed: overall speed (tokens/sec)')

await context.release()
