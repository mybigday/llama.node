import { loadModel } from '../lib/index.js'
import * as fs from 'fs'
import * as path from 'path'
import { fileURLToPath } from 'url'

const libVariant = process.env.LLAMA_LIB_VARIANT || 'default'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

// Helper to generate a simple hash for a question (for state file naming)
const hashString = (str) => {
  let hash = 0
  for (let i = 0; i < str.length; i++) {
    const char = str.charCodeAt(i)
    hash = (hash << 5) - hash + char
    hash &= hash // Convert to 32bit integer
  }
  return Math.abs(hash).toString(36)
}

// Helper to get state file path for a question (per model)
const getStatePath = (modelPath, prompt) => {
  // Extract model filename without extension
  const modelFilename = path.basename(modelPath, path.extname(modelPath))
  const questionHash = hashString(prompt.trim().toLowerCase())
  const stateDir = path.join(__dirname, 'state')

  // Create state directory if it doesn't exist
  if (!fs.existsSync(stateDir)) {
    fs.mkdirSync(stateDir, { recursive: true })
  }

  return path.join(stateDir, `state_${modelFilename}_${questionHash}.bin`)
}

// Example prompts to demonstrate state caching
const EXAMPLE_PROMPTS = [
  'What is the capital of France?',
  'Explain quantum computing in simple terms.',
  'Write a haiku about coding.',
  'What are the primary colors?',
  'What is the meaning of life?',
  'What is art?',
  // Repeat some questions to demonstrate state loading
  'What is the capital of France?', // Should load state from first request
  'Explain quantum computing in simple terms.', // Should load state from second request
]

const SYSTEM_PROMPT =
  'You are a helpful AI assistant. Be concise and direct in your responses.'

console.log('Loading model...')
const model = await loadModel(
  {
    model: import.meta
      .resolve('./Nemotron-3-Nano-30B-A3B-Q4_0.gguf')
      .replace('file://', ''),
    n_ctx: 8192,
    n_gpu_layers: 99,
    use_mlock: true,
    use_mmap: true,
    flash_attn_type: 'auto',
    cache_type_k: 'q8_0',
    cache_type_v: 'q8_0',
    n_parallel: 8,
    lib_variant: libVariant,
  },
  (progress) => {
    if (progress % 10 === 0) console.log(`Loading ${progress}%`)
  },
)

const modelInfo = model.getModelInfo()
const usePromptState = modelInfo.is_recurrent || modelInfo.is_hybrid

console.log('Enabling parallel mode...')
await model.parallel.enable({
  n_parallel: 4,
  n_batch: 512,
})

console.log('\n=== Parallel Completion with State Management Demo ===\n')
console.log(
  'This demo shows how to use load_state_path and prompt caching to reuse state',
)
console.log(
  'question-specific state. When a question is asked multiple times, the',
)
console.log(
  'state from the first request is reused, significantly improving performance.\n',
)
console.log(
  `Model recurrent/hybrid: ${usePromptState ? 'yes' : 'no'} (using ${usePromptState ? 'save_prompt_state_path' : 'save_state_path'})\n`,
)

// Pre-tokenize all prompts SEQUENTIALLY to avoid lock contention
console.log('Pre-tokenizing all prompts...')
const modelPath = import.meta
  .resolve('./gpt-oss-20b-mxfp4.gguf')
  .replace('file://', '')
const preTokenizedPrompts = []

const t0_total = Date.now()
for (let i = 0; i < EXAMPLE_PROMPTS.length; i++) {
  const prompt = EXAMPLE_PROMPTS[i]
  const statePath = getStatePath(modelPath, prompt)
  const stateFileExists = fs.existsSync(statePath)
  const loadStatePath = stateFileExists ? statePath : undefined

  // Build messages
  const messages = [
    { role: 'system', content: SYSTEM_PROMPT },
    { role: 'user', content: prompt },
  ]

  const t0_formatChat = Date.now()
  // Format chat to get the formatted prompt for tokenization
  const formattedChat = model.getFormattedChat(messages, undefined)
  const t1_formatChat = Date.now()
  console.log(`  → Format chat time: ${t1_formatChat - t0_formatChat}ms`)

  // Tokenize the formatted prompt to get question token count
  let questionTokenCount = 0
  try {
    const t0 = Date.now()
    const tokenizeResult = await model.tokenize(formattedChat.prompt)
    const t1 = Date.now()
    questionTokenCount = tokenizeResult.tokens.length
    console.log(
      `  [${i + 1}/${EXAMPLE_PROMPTS.length}] Tokenized "${prompt.substring(0, 40)}..." → ${questionTokenCount} tokens (${t1 - t0}ms)`,
    )
  } catch (error) {
    console.error(
      `  [${i + 1}/${EXAMPLE_PROMPTS.length}] ✗ Error tokenizing prompt:`,
      error,
    )
    questionTokenCount = 0
  }

  preTokenizedPrompts.push({
    prompt,
    messages,
    statePath,
    loadStatePath,
    questionTokenCount,
  })
}
const t1_total = Date.now()
console.log(`✓ Pre-tokenization complete in ${t1_total - t0_total}ms\n`)

// Now process all requests in parallel using pre-computed tokenization results
const requests = preTokenizedPrompts.map(
  async (
    { prompt, messages, statePath, loadStatePath, questionTokenCount },
    i,
  ) => {
    const startTime = Date.now()
    console.log(
      `\n[${i + 1}/${EXAMPLE_PROMPTS.length}] Processing: "${prompt}"`,
    )

    try {
      if (loadStatePath) {
        console.log(
          `  ✓ Loading existing state from: ${path.basename(statePath)}`,
        )
      } else {
        console.log(
          `  ○ No existing state, will save to: ${path.basename(statePath)}`,
        )
      }

      console.log(`  → Question tokens: ${questionTokenCount}`)

      // Queue the completion with state management
      const request = await model.parallel.completion(
        {
          messages,
          reasoning_format: 'auto',
          n_predict: 1024,
          temperature: 0.7,
          // State management parameters
          load_state_path: loadStatePath,
          save_prompt_state_path:
            usePromptState && questionTokenCount > 0 ? statePath : undefined,
          save_state_path:
            !usePromptState && questionTokenCount > 0 ? statePath : undefined,
          save_state_size:
            !usePromptState && questionTokenCount > 0
              ? questionTokenCount
              : undefined,
        },
        (_requestId, data) => {
          // Stream tokens (optional)
          if (data.token) {
            process.stdout.write(data.token)
          }
        },
      )

      // Wait for completion
      const result = await request.promise
      const duration = ((Date.now() - startTime) / 1000).toFixed(2)

      console.log(`\n  ✓ Completed in ${duration}s`)
      console.log(
        `  → Response: ${result.text.substring(0, 100)}${result.text.length > 100 ? '...' : ''}`,
      )

      if (questionTokenCount > 0) {
        console.log(`  ✓ State saved to: ${path.basename(statePath)}`)
      }

      return result
    } catch (error) {
      console.error(`\n  ✗ Error:`, error.message)
      return null
    }
  },
)

// Wait for all requests to complete
const results = await Promise.all(requests)

console.log('\n\n=== Demo Complete ===')
console.log(`Total requests: ${EXAMPLE_PROMPTS.length}`)
console.log(`Successful: ${results.filter((r) => r !== null).length}`)
console.log(`Failed: ${results.filter((r) => r === null).length}`)
console.log('\nState files saved to: examples/state/')
console.log('To clear saved state files, run:')
console.log('  rm -rf examples/state/')

await model.release()
