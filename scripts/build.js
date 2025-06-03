const fs = require('fs');
const path = require('path');

const validAccelerators = process.platform === 'darwin' ? [] : ['vulkan', 'cuda'];

let isBuildFromSource = process.env.npm_config_build_from_source === 'true';

let accelerator = process.env.npm_config_accelerator || '';

const checkPaths = [
  path.resolve(
    __dirname,
    `../node-llama-${process.platform}-${process.arch}${accelerator ? `-${accelerator}` : ''}`
  ),
  path.resolve(__dirname, `../build/Release/index.node`),
];

if (!isBuildFromSource && !checkPaths.some(path => fs.existsSync(path))) {
  console.warn('Not found prebuild package, please build from source');
  isBuildFromSource = true;
}

if (accelerator && !validAccelerators.includes(accelerator)) {
  throw new Error(`Invalid accelerator: ${accelerator}`);
}

if (isBuildFromSource) {
  console.log('Build from source is enabled');
}

let BuildSystem;
try {
  ({ BuildSystem } = require('cmake-js'));
} catch (error) {
  console.error('cmake-js is not installed, please install it');
  process.exit(1);
}

const buildSystem = new BuildSystem({
  directory: path.resolve(__dirname, '../'),
  arch: process.arch,
  preferClang: true,
  out: path.resolve(__dirname, '../build'),
  extraCMakeArgs: [accelerator && `--CDVARIANT=${accelerator}`].filter(Boolean),
});

buildSystem.build();
