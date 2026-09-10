import path from 'path'
import { spawnSync } from 'child_process'

// Each case runs in a child process: the failure happens during process teardown,
// after the model was loaded without being released.
const runChild = (body: string) => {
  const script = `
    const { loadModel } = require(${JSON.stringify(path.resolve(__dirname, '../lib'))})
    ;(async () => {
      const ctx = await loadModel({
        model: ${JSON.stringify(path.resolve(__dirname, './tiny-random-llama.gguf'))},
        n_gpu_layers: 99,
        n_ctx: 256,
      })
      ${body}
    })()
  `
  return spawnSync(process.execPath, ['-e', script], { encoding: 'utf8', timeout: 60000 })
}

const expectCleanTeardown = (result: ReturnType<typeof runChild>, status: number) => {
  expect(result.signal).toBeNull()
  expect(result.stderr).not.toContain('GGML_ASSERT')
  expect(result.status).toBe(status)
}

test('process.exit() with a loaded model', () => {
  expectCleanTeardown(runChild('process.exit(0)'), 0)
})

test('uncaught exception with a loaded model', () => {
  expectCleanTeardown(runChild('throw new Error("boom")'), 1)
})

test('process.exit() during completion', () => {
  expectCleanTeardown(
    runChild(`
      let exited = false
      ctx.completion({ prompt: 'Hello', n_predict: 100 }, (t) => {
        if (!exited && t.token) { exited = true; process.exit(0) }
      })
    `),
    0,
  )
})

test('releaseSync() in an exit handler', () => {
  expectCleanTeardown(runChild("process.on('exit', () => ctx.releaseSync()); process.exit(0)"), 0)
})
