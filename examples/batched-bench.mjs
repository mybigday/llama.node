import { loadModel } from '../lib/index.js'

// Parse arguments
const args = process.argv.slice(2)
const isQuick = args.includes('--quick')
const modelPath = args.find((a) => !a.startsWith('--')) || process.env.MODEL_PATH

if (!modelPath) {
  console.error('Usage: node batched-bench.mjs [--quick] <model_path>')
  console.error('  or set MODEL_PATH environment variable')
  console.error('')
  console.error('Options:')
  console.error('  --quick    Run quick benchmark with fewer iterations')
  process.exit(1)
}

// Benchmark presets from llama.cpp/scripts/bench-models.sh
const presets = {
  default: {
    pp: [512, 4096, 8192],
    tg: 32,
    pl: [1, 2, 4, 8, 16, 32],
  },
  quick: {
    pp: [512, 4096],
    tg: 32,
    pl: [1, 2, 4],
  },
}

const preset = isQuick ? presets.quick : presets.default
const libVariant = process.env.LLAMA_LIB_VARIANT || 'default'

console.log(`Loading model: ${modelPath}`)
console.log(`Library variant: ${libVariant}`)
console.log(`Preset: ${isQuick ? 'quick' : 'default'}\n`)

// Calculate required context size and parallel level
const maxPp = Math.max(...preset.pp)
const maxPl = Math.max(...preset.pl)
const requiredCtx = maxPl * (maxPp + preset.tg)

const context = await loadModel(
  {
    model: modelPath,
    n_ctx: requiredCtx,
    n_parallel: maxPl,
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

console.log('Running batched benchmark...\n')

// Print table header
console.log(
  '|    pp |    tg |  pl |  n_kv |    t_pp |   speed_pp |    t_tg |   speed_tg |       t |    speed |',
)
console.log(
  '|-------|-------|-----|-------|---------|------------|---------|------------|---------|----------|',
)

// Run benchmarks for all combinations
const results = []
for (const pp of preset.pp) {
  for (const pl of preset.pl) {
    const result = await context.bench(pp, preset.tg, pl, 1)
    results.push(result)

    // Print result row
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
  }
}

// Print configuration (from last result)
const lastResult = results[results.length - 1]
console.log()
console.log('Configuration:')
console.log(
  `  n_kv_max=${lastResult.nKvMax}, n_batch=${lastResult.nBatch}, n_ubatch=${lastResult.nUBatch}`,
)
console.log(`  flash_attn=${lastResult.flashAttn}, is_pp_shared=${lastResult.isPpShared}`)
console.log(
  `  n_gpu_layers=${lastResult.nGpuLayers}, n_threads=${lastResult.nThreads}, n_threads_batch=${lastResult.nThreadsBatch}`,
)

console.log()
console.log('Legend:')
console.log('  pp: prompt processing tokens')
console.log('  tg: text generation tokens')
console.log('  pl: parallel level')
console.log('  n_kv: KV cache used')
console.log('  t_pp: time for prompt processing (s)')
console.log('  speed_pp: prompt processing speed (tokens/sec)')
console.log('  t_tg: time for text generation (s)')
console.log('  speed_tg: text generation speed (tokens/sec)')
console.log('  t: total time (s)')
console.log('  speed: overall speed (tokens/sec)')

await context.release()
