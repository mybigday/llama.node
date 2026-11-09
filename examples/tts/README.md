# TTS Examples (Experimental)

Text-to-speech examples using the codec.cpp-backed TTS integration.
Each script covers one supported model family and writes a `<model-key>.wav`
next to the script.

## Usage

```bash
# List available models (with sizes)
node examples/tts/download-models.mjs

# Download a model pair (backbone + codec), then run its example
node examples/tts/download-models.mjs soprano
node examples/tts/soprano.mjs "Hello from llama.node!"

# Or grab everything (~10GB)
node examples/tts/download-models.mjs all
```

## Models

| Script | Model | Size (model + codec) | Notes |
|---|---|---|---|
| `outetts-0.3.mjs` | OuteTTS 0.3 500M + WavTokenizer | 403MB + 178MB | Builtin voices in the codec |
| `outetts-1.0.mjs` | OuteTTS 1.0 0.6B + DAC | 402MB + 148MB | |
| `soprano.mjs` | Soprano-1.1 80M | 160MB + 121MB | Smallest pair (used in CI tests) |
| `neutts-nano.mjs` | NeuTTS Nano + NeuCodec | 210MB + 342MB | Needs `npm i phonemize --no-save` |
| `csm.mjs` | CSM-1B (Sesame) + Mimi | 770MB + 803MB | |
| `qwen3-tts.mjs` | Qwen3-TTS 0.6B + 12Hz codec | 257MB + 565MB | |
| `moss-tts-realtime.mjs` | MOSS-TTS-Realtime + MOSS-Audio | 1.8GB + 1.8GB | Streaming interleave |
| `moss-ttsd.mjs` | MOSS-TTSD v0.5 + XY-Tokenizer | 1.8GB + 1.1GB | `[S1]`/`[S2]` dialogue format |
| `chatterbox.mjs` | Chatterbox Multilingual T3 + S3G | 289MB + 178MB | 23 languages, voice cloning |
| `chatterbox-turbo.mjs` | Chatterbox Turbo T3 + MeanFlow S3G | 259MB + 217MB | Voice cloning |
| `breezyvoice.mjs` | BreezyVoice (CosyVoice v1) + flow/HiFT | 161MB + 316MB | Instruct, voice cloning |
| `bluemagpie.mjs` | BlueMagpie-TTS Barbet-1B + AudioVAE | 661MB + 1.04GB | Continuous-latent flow |

NeuTTS Air is also in the catalog (`neutts-air`) — download it and pass the key
to `run-tts.mjs`-based scripts the same way as `neutts-nano`.

## Flow

`getFormattedAudioCompletion()` returns `completionParams` — the prompt, the
grammar / `logit_bias` token mask, `embedding` and the model's minimum length
(`n_min_predict`). Pass them to `completion()` as-is; sampling defaults to the
backbone GGUF's `general.sampling.*`. The result carries `audio_tokens` (codec
codes, for `decodeAudioTokens`) or `audio_latents` (for `decodeAudioLatents`).

```js
const { completionParams } = await context.getFormattedAudioCompletion({ prompt })
const result = await context.completion({ ...completionParams, n_predict: 4096 })
const audio = result.audio_latents?.length
  ? await context.decodeAudioLatents(result.audio_latents)
  : await context.decodeAudioTokens(result.audio_tokens)
```

Builtin voices of the codec are listed in `getTTSCapabilities().builtinVoices`;
pass a voice's `name` as `speaker` (or set `TTS_VOICE` for the NeuTTS scripts).

## Voice cloning

Families with `canSpeakerFromAudio` (Chatterbox, Qwen3-TTS Base, CosyVoice,
...) accept a reference clip:

```js
const speaker = context.createSpeaker({
  refAudio: pcmFloat32,          // mono float32 PCM, any sample rate
  refAudioSampleRate: 24000,
  refText: 'Transcript of the reference clip.', // when caps.usesRefText
})
const { completionParams } = await context.getFormattedAudioCompletion({ prompt, speaker })
const result = await context.completion({ ...completionParams, speaker })

// Keep the voice for later: write it to a speaker file and load it back.
await context.saveSpeaker(speaker, 'my-voice.gguf')
const reloaded = await context.loadSpeaker('my-voice.gguf')
speaker.release()
```
