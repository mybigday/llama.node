import path from 'path'
import waitForExpect from 'wait-for-expect'
import { loadModel, loadLlamaModelInfo, toggleNativeLog, addNativeLogListener } from '../lib'

const OUTETTS_GRAMMAR = `\
root       ::= NL? content* audioEnd NL eos?

content ::= wordAudioBlock | emotionBlock

wordAudioBlock ::= punctuation? WORD punctuation? codeBlock space NL

codeBlock ::= TIME CODE{1,144}

emotionBlock ::= emotionStart TEXT emotionEnd space NL

TEXT ::= [A-Za-z0-9 .,?!]+

bos    ::= "<|im_start|>"
eos      ::= "<|im_end|>"
emotionStart ::= "<|emotion_start|>"
emotionEnd ::= "<|emotion_end|>"
audioEnd   ::= "<|audio_end|>"
space      ::= "<|space|>"

WORD       ::= [A-Za-z]+
NL         ::= "\\n"

TIME  ::= "<|t_" DECIMAL "|>"

CODE    ::= "<|" DIGITS "|>"

DIGITS     ::= [0-9]+
DECIMAL    ::= [0-9]+ "." [0-9]+

punctuation ::= "<|" [a-z_]+ "|>"
`

test.skip('TTS', async () => {
  const model = await loadModel({
    model: path.resolve(__dirname, './OuteTTS-0.3-500M-Q4_K_M.gguf'),
    n_ctx: 8192,
    n_batch: 8192,
    n_ubatch: 128,
    n_threads: 4,
    n_gpu_layers: 99,
    ctx_shift: false,
  })
  await model.initVocoder({ path: path.resolve(__dirname, './WavTokenizer.gguf'), n_batch: 2048 })
  expect(model.isVocoderEnabled()).toBe(true)
  const prompt = model.getFormattedAudioCompletion(null, 'Hello, world')
  expect(prompt).toMatchSnapshot('formatted')
  const tokens = model.getAudioCompletionGuideTokens('Hello, world!')
  expect(tokens).toMatchSnapshot('guide tokens')
  const result = await model.completion({
    prompt,
    temperature: 0.2,
    penalty_repeat: 1.2,
    penalty_last_n: 64,
    guide_tokens: tokens,
    top_k: 40,
    top_p: 0.9,
    min_p: 0.05,
    mirostat_tau: 5,
    mirostat_eta: 0.1,
    mirostat: 0,
    grammar: OUTETTS_GRAMMAR,
    n_predict: 3840,
    stop: ['<|audio_end|>', '<|im_end|>'],
  })
  const { audio_tokens: audioTokens } = result
  console.log(result)
  const audio = await model.decodeAudioTokens(audioTokens!)
  expect(audio.length).toBeGreaterThan(0)
  await model.releaseVocoder()
  expect(model.isVocoderEnabled()).toBe(false)
  await model.release()
}, 1e8)
