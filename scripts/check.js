const path = require('path');

const validAccelerators = process.platform === 'darwin' ? [] : ['vulkan', 'cuda'];

const accelerator = process.env.npm_config_accelerator || '';

if (process.env.npm_config_build_from_source) {
  console.log('Build from source is enabled');
} else {
  process.exit(0);
}

if (accelerator && !validAccelerators.includes(accelerator)) {
  console.error(`Invalid accelerator: ${accelerator}`);
  process.exit(1);
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
