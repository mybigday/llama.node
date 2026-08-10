// MOSS-TTSD v0.5 TTS example (dialogue model).
// Prompts use the [S1]/[S2] speaker-turn format.
//   1. node examples/tts/download-models.mjs moss-ttsd
//   2. node examples/tts/moss-ttsd.mjs "[S1] Hello there! [S2] Hi, nice to meet you."
import { runTTS } from './run-tts.mjs'

await runTTS('moss-ttsd', {
  text:
    process.argv[2] ||
    '[S1] Hello! This is a text to speech demo running locally with llama dot node.',
})
