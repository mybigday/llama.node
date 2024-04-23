import { loadModel } from './index'
import type { LlamaContext } from './index'

/*
Format:
<|system|>
{system_message}</s>
<|user|>
{prompt}</s>
<|assistant|>
*/

// const model = loadModel({ model: 'llama-160m.Q8_0.gguf' })
let model: LlamaContext | null = loadModel({ model: 'tinyllama-1.1b-chat-v1.0.Q6_K.gguf' })

const completion = async (text: string, seed: number) => {
  const time = performance.now()
  const result = await model!.completion({
    prompt: `<|system|>\nYou are a chatbot.\n<|user|>\n${text}\n<|assistant|>\n`,
    n_samples: 1,
    temperature: 0,
    max_length: 100,
    n_predict: 128,
    seed,
    stop: ['\n', '<|assistant|>']
  }, () => {})
  const tokens = result.tokens_evaluated
  const diff = performance.now() - time
  console.log(result)
  console.log(`Tokens: ${tokens}, Time: ${diff}ms, Speed: ${tokens / (diff) * 1000} tokens/s`)
}
model.loadSession('session.bin').then(() =>
  Promise.all([
    completion('A'.repeat(10), 1),
    completion('B'.repeat(10), 2),
    completion('A'.repeat(10), 1),
    completion('B'.repeat(10), 2),
  ])
).then(() =>
  model!.saveSession('session.bin').then(console.log)
).catch(console.error)
