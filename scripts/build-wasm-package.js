#!/usr/bin/env node

const fs = require('fs')
const os = require('os')
const path = require('path')
const { execFileSync } = require('child_process')

const rootDir = path.resolve(__dirname, '..')
const buildRoot = path.join(rootDir, 'build-wasm')
const packageDir = path.join(rootDir, 'packages', 'node-llama-wasm')
const packageWasmDir = path.join(packageDir, 'wasm')
const emCacheDir = process.env.EM_CACHE
  ? path.resolve(rootDir, process.env.EM_CACHE)
  : path.join(buildRoot, 'emcache')

const boolEnv = (name, defaultValue) => {
  const value = process.env[name]
  if (value == null || value === '') return defaultValue
  return !/^(0|false|off|no)$/i.test(value)
}

const hasArg = (name) => process.argv.includes(name)

const webgpu = hasArg('--webgpu')
  ? true
  : hasArg('--no-webgpu')
    ? false
    : boolEnv('WASM_WEBGPU', false)
const threads = hasArg('--threads')
  ? true
  : hasArg('--no-threads')
    ? false
    : boolEnv('WASM_THREADS', false)
const buildVariant = `${webgpu ? 'webgpu' : 'cpu'}${threads ? '-threads' : ''}`
const buildDir = process.env.WASM_BUILD_DIR
  ? path.resolve(rootDir, process.env.WASM_BUILD_DIR)
  : path.join(buildRoot, buildVariant)
const requestedGenerator = process.env.WASM_CMAKE_GENERATOR
const requestedCcache = boolEnv('WASM_CCACHE', true)

const dawnTag = process.env.EMDAWN_TAG || 'v20260317.182325'

const run = (cmd, args, options = {}) => {
  console.log(`$ ${cmd} ${args.join(' ')}`)
  execFileSync(cmd, args, {
    cwd: rootDir,
    env: {
      ...process.env,
      EM_CACHE: emCacheDir,
    },
    stdio: 'inherit',
    ...options,
  })
}

const normalizeEmscriptenMemoryGrow = (js) =>
  // WebAssembly.Memory.grow() takes a numeric page delta even for memory64 builds.
  js.replaceAll('wasmMemory.grow(BigInt(pages))', 'wasmMemory.grow(pages)')

const commandExists = (cmd) => {
  try {
    execFileSync(process.platform === 'win32' ? 'where' : 'which', [cmd], {
      stdio: 'ignore',
    })
    return true
  } catch {
    return false
  }
}

const readCmakeCacheValue = (buildDir, key) => {
  const cachePath = path.join(buildDir, 'CMakeCache.txt')
  if (!fs.existsSync(cachePath)) return null

  const cache = fs.readFileSync(cachePath, 'utf8')
  const escapedKey = key.replace(/[.*+?^${}()|[\]\\]/g, '\\$&')
  const match = cache.match(new RegExp(`^${escapedKey}:[^=]*=(.*)$`, 'm'))
  return match?.[1] ?? null
}

const preferredGenerator = () => {
  if (requestedGenerator) return requestedGenerator
  return commandExists('ninja') ? 'Ninja' : null
}

const generatorArgs = () => {
  const generator = preferredGenerator()
  if (!generator) return []

  const existing = readCmakeCacheValue(buildDir, 'CMAKE_GENERATOR')
  if (!existing) return ['-G', generator]
  if (existing === generator) return ['-G', generator]

  if (requestedGenerator) {
    throw new Error(
      `Build directory ${buildDir} already uses CMake generator "${existing}". ` +
        `Set WASM_BUILD_DIR to a new path or remove the directory before using "${generator}".`,
    )
  }

  console.log(
    `Keeping existing CMake generator "${existing}" for ${buildDir}. ` +
      `Remove the directory to reconfigure with "${generator}".`,
  )
  return []
}

const ccacheArgs = () => {
  if (!requestedCcache || !commandExists('ccache')) return []
  return [
    '-DCMAKE_C_COMPILER_LAUNCHER=ccache',
    '-DCMAKE_CXX_COMPILER_LAUNCHER=ccache',
  ]
}

const ensureEmscripten = () => {
  if (!commandExists('emcmake')) {
    throw new Error(
      'emcmake was not found. Install and activate Emscripten SDK before running npm run build-wasm.',
    )
  }
}

const ensureDawn = () => {
  if (!webgpu) return null
  if (process.env.EMDAWNWEBGPU_DIR) return process.env.EMDAWNWEBGPU_DIR

  const dawnRoot = path.join(buildRoot, 'emdawn')
  const dawnDir = path.join(dawnRoot, 'emdawnwebgpu_pkg')
  if (fs.existsSync(path.join(dawnDir, 'emdawnwebgpu.port.py'))) {
    return dawnDir
  }

  fs.mkdirSync(dawnRoot, { recursive: true })
  const pkg = `emdawnwebgpu_pkg-${dawnTag}.zip`
  const zipPath = path.join(dawnRoot, pkg)
  const url = `https://github.com/google/dawn/releases/download/${dawnTag}/${pkg}`

  run('curl', ['-L', '-o', zipPath, url])
  run('python3', [
    '-c',
    [
      'import zipfile, sys',
      'zipfile.ZipFile(sys.argv[1], "r").extractall(sys.argv[2])',
    ].join('; '),
    zipPath,
    dawnRoot,
  ])

  return dawnDir
}

const main = () => {
  ensureEmscripten()
  const dawnDir = ensureDawn()

  fs.mkdirSync(buildRoot, { recursive: true })
  fs.mkdirSync(buildDir, { recursive: true })
  fs.mkdirSync(emCacheDir, { recursive: true })
  fs.mkdirSync(packageWasmDir, { recursive: true })

  const cmakeArgs = [
    'cmake',
    ...generatorArgs(),
    '-S',
    path.join(rootDir, 'src', 'wasm'),
    '-B',
    buildDir,
    '-DCMAKE_BUILD_TYPE=Release',
    `-DLLAMA_NODE_WASM_WEBGPU=${webgpu ? 'ON' : 'OFF'}`,
    `-DLLAMA_NODE_WASM_THREADS=${threads ? 'ON' : 'OFF'}`,
    ...ccacheArgs(),
  ]

  if (dawnDir) {
    cmakeArgs.push(`-DEMDAWNWEBGPU_DIR=${dawnDir}`)
  }

  run('emcmake', cmakeArgs)
  run('cmake', [
    '--build',
    buildDir,
    '--target',
    'llama-node',
    '--parallel',
    String(process.env.JOBS || os.cpus().length),
  ])

  const jsSrc = path.join(buildDir, 'llama-node.js')
  const wasmSrc = path.join(buildDir, 'llama-node.wasm')
  const suffixParts = []
  if (webgpu) suffixParts.push('webgpu')
  if (threads) suffixParts.push('threads')
  const suffix = suffixParts.length > 0 ? `.${suffixParts.join('.')}` : ''
  const jsDest = path.join(packageWasmDir, `llama-node${suffix}.js`)
  const wasmDest = path.join(packageWasmDir, `llama-node${suffix}.wasm`)

  if (!fs.existsSync(jsSrc) || !fs.existsSync(wasmSrc)) {
    throw new Error('WASM build did not produce llama-node.js and llama-node.wasm')
  }

  fs.writeFileSync(jsDest, normalizeEmscriptenMemoryGrow(fs.readFileSync(jsSrc, 'utf8')))
  fs.copyFileSync(wasmSrc, wasmDest)

  const jsSize = fs.statSync(jsDest).size
  const wasmSize = fs.statSync(wasmDest).size
  console.log(`Wrote ${jsDest} (${jsSize} bytes)`)
  console.log(`Wrote ${wasmDest} (${wasmSize} bytes)`)
}

main()
