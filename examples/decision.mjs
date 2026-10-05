// Typed decisions (System One): a decision model answers `choice` / `score` /
// `noul` questions about a state with calibrated probabilities, one forward
// pass per answer, no generated tokens. The request and the response follow
// the TypeSafe `/v1/systemone` API that llama-server serves.
//
// Models: https://huggingface.co/blog/ggml-org/decision-models-in-llamacpp
//   e.g. ggml-org/Julia-1-GGUF (Julia-1-Q8_0.gguf, 168MB), ggml-org/Laya-GGUF,
//        ggml-org/Kev-0.8B-GGUF, ggml-org/lev-GGUF
//
// Usage:
//   node examples/decision.mjs <model_path> [--mmproj <path> --image <path>]
//   node examples/decision.mjs ./test/Julia-1-Q8_0.gguf
//
// `--image` needs a model whose `getModelInfo().decision.imageInput` is true,
// and its projector (`--mmproj`).

import fs from 'fs'
import { loadModel } from '../lib/index.js'

const args = process.argv.slice(2)
const option = (name) => {
  const i = args.indexOf(name)
  return i >= 0 ? args.splice(i, 2)[1] : undefined
}
const mmprojPath = option('--mmproj')
const imagePath = option('--image')
const modelPath = args[0] || process.env.MODEL_PATH

if (!modelPath) {
  console.error(
    'Usage: node examples/decision.mjs <model_path> [--mmproj <path> --image <path>]',
  )
  console.error('  or set MODEL_PATH=/path/to/model.gguf')
  process.exit(1)
}

for (const file of [modelPath, mmprojPath, imagePath].filter(Boolean)) {
  if (!fs.existsSync(file)) {
    console.error(`File not found: ${file}`)
    process.exit(1)
  }
}

const N_PARALLEL = 4

const context = await loadModel({
  model: modelPath,
  lib_variant: process.env.LLAMA_LIB_VARIANT || 'default',
  n_ctx: 4096,
  n_gpu_layers: 99,
  n_parallel: N_PARALLEL,
  // a `system_one` model with a classification head (readout `rank_head`)
  // is read through RANK pooling
  ...(process.env.RANK_POOLING ? { pooling_type: 'rank', embedding: true } : {}),
})

const { decision } = context.getModelInfo()
if (!decision) {
  console.error('This model is not a decision model (no decision metadata)')
  await context.release()
  process.exit(1)
}
console.log('Decision model:', decision)
if (decision.type === 'unknown') {
  console.error(`This build cannot serve the model: ${decision.error}`)
  await context.release()
  process.exit(1)
}

const questions = {
  is_ordering: {
    type: 'noul',
    instructions: 'The customer is placing an order',
  },
  ordered_product: {
    type: 'choice',
    instructions: 'Which product did the customer order first?',
    criteria: {
      'fries-m': 'Fries M (medium fries)',
      'cola-l': 'Cola L (large cola)',
      burger: 'Mos Burger',
      none: 'Nothing ordered',
    },
  },
  mood: {
    type: 'score',
    instructions: 'How impatient does the customer sound?',
    criteria: ['Calm', 'Neutral', 'Impatient'],
  },
  wants_drink: {
    type: 'noul',
    instructions: 'The customer wants a drink',
    criteria: { true: 'a drink is mentioned', false: 'no drink' },
  },
}

const pct = (p) => `${(p * 100).toFixed(1)}%`

const printAnswers = (answers) => {
  for (const [id, answer] of Object.entries(answers)) {
    if (answer.type === 'noul') {
      console.log(`  ${id} (noul): ${pct(answer.noul)} true`)
    } else if (answer.type === 'choice') {
      const ranked = Object.entries(answer.probabilities)
        .sort((a, b) => b[1] - a[1])
        .map(([key, p]) => `${key} ${pct(p)}`)
        .join(', ')
      console.log(
        `  ${id} (choice): ${answer.choice}` +
          ` (confidence ${answer.confidence.toFixed(2)}) [${ranked}]`,
      )
    } else {
      const level = answer.legend[String(Math.round(answer.score))]
      console.log(
        `  ${id} (score): ${answer.score.toFixed(2)} ~ ${level}` +
          ` (confidence ${answer.confidence.toFixed(2)})`,
      )
    }
  }
}

// 1. One request, answered on the context's own sequence
const request = {
  state: {
    utterance:
      'uh can I get a medium fries and, actually make that a large cola too',
    stage: 'menu',
    cart: [],
  },
  questions,
}

let t0 = performance.now()
const result = await context.decide(request)
console.log(
  `\ndecide(): ${result.usage.input_tokens} prompt tokens,` +
    ` ${(performance.now() - t0).toFixed(0)}ms`,
)
printAnswers(result.answers)

// 2. With an image (`images` takes file paths or data URLs)
if (imagePath) {
  if (!decision.imageInput) {
    console.warn('\nThe model takes no image input, skipping --image')
  } else if (!mmprojPath) {
    console.warn('\n--image needs --mmproj, skipping it')
  } else {
    await context.initMultimodal({ path: mmprojPath, use_gpu: true })
    t0 = performance.now()
    const withImage = await context.decide({
      state: 'What is in the picture?',
      images: [imagePath],
      questions: {
        has_food: { type: 'noul', instructions: 'The picture shows food' },
        scene: {
          type: 'choice',
          instructions: 'Where was the picture taken?',
          criteria: {
            indoor: 'Indoors',
            outdoor: 'Outdoors',
            unclear: null,
          },
        },
      },
    })
    console.log(
      `\ndecide() with an image: ${withImage.usage.input_tokens} prompt tokens,` +
        ` ${(performance.now() - t0).toFixed(0)}ms`,
    )
    printAnswers(withImage.answers)
  }
}

// 3. Several states at once, in parallel slots
const utterances = [
  'one burger please, no rush',
  'I just want to look at the menu for now',
  'LARGE COLA. NOW.',
  'can I get fries, medium, and a cola',
]

await context.parallel.enable({ n_parallel: N_PARALLEL })
t0 = performance.now()
const queued = await Promise.all(
  utterances.map((utterance) =>
    context.parallel.decide({
      state: { utterance, stage: 'menu', cart: [] },
      questions,
    }),
  ),
)
const results = await Promise.all(queued.map(({ promise }) => promise))
console.log(
  `\nparallel.decide(): ${utterances.length} requests,` +
    ` ${(performance.now() - t0).toFixed(0)}ms`,
)
results.forEach(({ answers }, i) => {
  console.log(`\n"${utterances[i]}"`)
  printAnswers(answers)
})
context.parallel.disable()

await context.release()
