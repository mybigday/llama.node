#!/usr/bin/env node

const os = require('os')
const path = require('path')
const { execFileSync } = require('child_process')

const rootDir = path.resolve(__dirname, '..')
const forwardedArgs = process.argv.slice(2)
const isArm64Host = os.arch() === 'arm64'

const image =
  process.env.EMSCRIPTEN_IMAGE ||
  (isArm64Host ? 'emscripten/emsdk:4.0.14-arm64' : 'emscripten/emsdk:4.0.13')
const platform =
  process.env.EMSCRIPTEN_PLATFORM ||
  (isArm64Host ? 'linux/arm64' : 'linux/amd64')

const readCommand = (cmd, args) =>
  execFileSync(cmd, args, {
    encoding: 'utf8',
    stdio: ['ignore', 'pipe', 'ignore'],
  }).trim()

const hostUid =
  process.env.HOST_UID ||
  (typeof process.getuid === 'function'
    ? String(process.getuid())
    : readCommand('id', ['-u']))
const hostGid =
  process.env.HOST_GID ||
  (typeof process.getgid === 'function'
    ? String(process.getgid())
    : readCommand('id', ['-g']))

const envNames = [
  'EMDAWN_TAG',
  'EM_CACHE',
  'JOBS',
  'WASM_BUILD_DIR',
  'WASM_CCACHE',
  'WASM_CMAKE_GENERATOR',
  'WASM_THREADS',
  'WASM_WEBGPU',
]

const dockerArgs = [
  'run',
  '--rm',
  '--platform',
  platform,
  '-e',
  'HOME=/tmp',
  '-e',
  `HOST_UID=${hostUid}`,
  '-e',
  `HOST_GID=${hostGid}`,
]

for (const name of envNames) {
  if (process.env[name] != null) dockerArgs.push('-e', name)
}

dockerArgs.push(
  '-v',
  `${rootDir}:/src`,
  '-w',
  '/src',
  image,
  'bash',
  '-lc',
  [
    'source /emsdk/emsdk_env.sh',
    'node scripts/build-wasm-package.js "$@"',
    'chown -R "$HOST_UID:$HOST_GID" build-wasm packages/node-llama-wasm/wasm',
  ].join(' && '),
  'build-wasm-package.js',
  ...forwardedArgs,
)

console.log(`Using ${image} (${platform})`)
execFileSync('docker', dockerArgs, {
  cwd: rootDir,
  stdio: 'inherit',
})
