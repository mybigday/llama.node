// TTS model catalog — mirrors llama.rn's example/src/utils/constants.ts.
// Each entry pairs a backbone (text→audio-token LM) with a codec/vocoder GGUF.
export const TTS_MODELS = {
  'outetts-0.3': {
    name: 'OuteTTS 0.3 500M (Q4_K_M) + WavTokenizer (FP16)',
    model: {
      repo: 'BricksDisplay/OuteTTS-0.3-500M-GGUF',
      filename: 'OuteTTS-0.3-500M-Q4_K_M.gguf',
      size: '403MB',
    },
    // OuteAI's WavTokenizer with the builtin voices baked for this backbone's
    // tokenizer.
    vocoder: {
      repo: 'BricksDisplay/OuteTTS-0.3-500M-GGUF',
      filename: 'codec-f16.gguf',
      size: '178MB',
    },
  },
  'outetts-1.0': {
    name: 'OuteTTS 1.0 0.6B (Q4_K_M) + DAC (FP16)',
    model: {
      repo: 'BricksDisplay/OuteTTS-1.0-0.6B-GGUF',
      filename: 'OuteTTS-1.0-0.6B-Q4_K_M.gguf',
      size: '402MB',
    },
    vocoder: {
      repo: 'BricksDisplay/OuteTTS-1.0-0.6B-GGUF',
      filename: 'codec-f16.gguf',
      size: '148MB',
    },
  },
  soprano: {
    name: 'Soprano-1.1 80M (F16) + codec (FP32)',
    model: {
      repo: 'BricksDisplay/Soprano-1.1-80M-GGUF',
      filename: 'Soprano-1.1-80M-soprano-tok.F16.gguf',
      size: '160MB',
    },
    vocoder: {
      repo: 'BricksDisplay/Soprano-1.1-80M-GGUF',
      filename: 'codec-F32.gguf',
      size: '121MB',
    },
  },
  'neutts-nano': {
    name: 'NeuTTS Nano (Q4_K_M) + NeuCodec (Q8_0)',
    model: {
      repo: 'BricksDisplay/NeuTTS-Nano-GGUF',
      filename: 'neutts-nano-q4_k_m.gguf',
      size: '210MB',
    },
    vocoder: {
      repo: 'BricksDisplay/NeuTTS-Nano-GGUF',
      filename: 'codec-q8_0.gguf',
      size: '342MB',
    },
  },
  'neutts-air': {
    name: 'NeuTTS Air (Q4_K_M) + NeuCodec (Q8_0)',
    model: {
      repo: 'BricksDisplay/NeuTTS-Air-GGUF',
      filename: 'neutts-air-q4_k_m.gguf',
      size: '596MB',
    },
    vocoder: {
      repo: 'BricksDisplay/NeuTTS-Air-GGUF',
      filename: 'codec-q8_0.gguf',
      size: '342MB',
    },
  },
  csm: {
    name: 'CSM-1B Sesame (Q4_K_M) + Mimi + codec_lm (Q8_0)',
    model: {
      repo: 'BricksDisplay/CSM-1B-GGUF',
      filename: 'csm-1b-q4_k_m.gguf',
      size: '770MB',
    },
    vocoder: {
      repo: 'BricksDisplay/CSM-1B-GGUF',
      filename: 'codec-q8_0.gguf',
      size: '803MB',
    },
  },
  'qwen3-tts': {
    name: 'Qwen3-TTS 0.6B (Q4_K_M) + 12Hz codec + codec_lm (Q8_0)',
    model: {
      repo: 'BricksDisplay/Qwen3-TTS-12Hz-0.6B-GGUF',
      filename: 'qwen3-tts-0.6b-q4_k_m.gguf',
      size: '257MB',
    },
    vocoder: {
      repo: 'BricksDisplay/Qwen3-TTS-12Hz-0.6B-GGUF',
      filename: 'codec-q8_0.gguf',
      size: '565MB',
    },
  },
  'moss-tts-realtime': {
    name: 'MOSS-TTS-Realtime (Q8_0) + MOSS-Audio + codec_lm (Q5_K_M)',
    model: {
      repo: 'BricksDisplay/MOSS-TTS-Realtime-GGUF',
      filename: 'moss-tts-realtime-q8_0.gguf',
      size: '1.8GB',
    },
    vocoder: {
      repo: 'BricksDisplay/MOSS-TTS-Realtime-GGUF',
      filename: 'codec-q5_k_m.gguf',
      size: '1.8GB',
    },
  },
  'moss-ttsd': {
    name: 'MOSS-TTSD v0.5 (Q8_0) + XY-Tokenizer + codec_lm (Q5_K_M)',
    model: {
      repo: 'BricksDisplay/MOSS-TTSD-v0.5-GGUF',
      filename: 'moss-ttsd-v0.5-q8_0.gguf',
      size: '1.8GB',
    },
    vocoder: {
      repo: 'BricksDisplay/MOSS-TTSD-v0.5-GGUF',
      filename: 'codec-q5_k_m.gguf',
      size: '1.1GB',
    },
  },
  chatterbox: {
    name: 'Chatterbox Multilingual T3 (Q4_K_M) + S3G + codec_lm (Q4_K_M)',
    model: {
      repo: 'BricksDisplay/Chatterbox-Multilingual-TTS-GGUF',
      filename: 'chatterbox-mtl-q4_k_m.gguf',
      size: '289MB',
    },
    vocoder: {
      repo: 'BricksDisplay/Chatterbox-Multilingual-TTS-GGUF',
      filename: 'codec-q4_k_m.gguf',
      size: '178MB',
    },
  },
  'chatterbox-turbo': {
    name: 'Chatterbox Turbo T3 (Q4_K_M) + MeanFlow S3G + codec_lm (Q4_K_M)',
    model: {
      repo: 'BricksDisplay/Chatterbox-Turbo-GGUF',
      filename: 'chatterbox-turbo-t3-q4_k_m.gguf',
      size: '259MB',
    },
    vocoder: {
      repo: 'BricksDisplay/Chatterbox-Turbo-GGUF',
      filename: 'chatterbox-turbo-codec-q4_k_m.gguf',
      size: '217MB',
    },
  },
  // CosyVoice v1 family; the speech LM's relative-position attention needs an
  // F16 KV cache (no flash attention).
  breezyvoice: {
    name: 'BreezyVoice LM (Q4_K_M) + CosyVoice flow/HiFT + codec_lm (Q5_K_M)',
    model: {
      repo: 'BricksDisplay/BreezyVoice-GGUF',
      filename: 'breezyvoice-lm-q4_k_m.gguf',
      size: '161MB',
    },
    vocoder: {
      repo: 'BricksDisplay/BreezyVoice-GGUF',
      filename: 'breezyvoice-codec-q5_k_m.gguf',
      size: '316MB',
    },
    contextParams: { flash_attn_type: 'off' },
  },
  bluemagpie: {
    name: 'BlueMagpie-TTS Barbet-1B (Q4_K_M) + AudioVAE (Q8_0) + codec_lm',
    model: {
      repo: 'BricksDisplay/BlueMagpie-TTS-GGUF',
      filename: 'BlueMagpie-Barbet-1B-q4_k_m.gguf',
      size: '661MB',
    },
    vocoder: {
      repo: 'BricksDisplay/BlueMagpie-TTS-GGUF',
      filename: 'BlueMagpie-AudioVAE-q8_0.gguf',
      size: '1.04GB',
    },
  },
}

export const getModelDownloadUrl = (repo, filename) =>
  `https://huggingface.co/${repo}/resolve/main/${filename}?download=true`

// Files are stored per-repo (repo "/" → "--") because several repos use the
// same codec filename (codec-q8_0.gguf) with different contents.
export const getLocalPath = (modelsDir, repo, filename) =>
  `${modelsDir}/${repo.replace(/\//g, '--')}/${filename}`
