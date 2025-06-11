const { execSync } = require('child_process');
const { readFileSync } = require('fs');
const path = require('path');

const packageDir = process.argv[2] || '.';
const shell = process.platform === 'win32' ? 'cmd' : undefined;

const packageJson = JSON.parse(readFileSync(path.resolve(packageDir, 'package.json'), 'utf8'));
const localVersion = packageJson.version;
const registry = packageJson.publishConfig?.registry || 'https://registry.npmjs.org';
let remoteVersion = '';

if (process.env.NPM_TOKEN) {
  const registryHost = new URL(registry).hostname;
  execSync(`npm config set //${registryHost}/:_authToken=${process.env.NPM_TOKEN}`, {
    cwd: packageDir,
    stdio: 'inherit',
    shell,
  });
}

try {
  remoteVersion = execSync(`npm view ${packageJson.name} version`, {
    cwd: packageDir,
    stdio: 'pipe',
    shell,
  }).toString().trim();
} catch {}

if (remoteVersion !== localVersion) {
  console.log(`${packageJson.name}@${localVersion} is not published, publishing...`);
  execSync(`npm publish`, {
    cwd: packageDir,
    stdio: 'inherit',
    shell,
  });
}
