const fs = require('fs');
const path = require('path');
const mainPackage = require('../package.json');

const version = mainPackage.version;

const packagesDir = 'packages';
if (!fs.existsSync(packagesDir)) {
  console.log('No packages directory found, skipping');
  process.exit(0);
}

const packages = fs.readdirSync(packagesDir);

packages.forEach(packageName => {
  const packageDir = path.join(packagesDir, packageName);
  const packageJsonPath = path.join(packageDir, 'package.json');

  if (fs.existsSync(packageJsonPath)) {
    const packageJson = JSON.parse(fs.readFileSync(packageJsonPath, 'utf8'));
    packageJson.version = version;

    fs.writeFileSync(packageJsonPath, JSON.stringify(packageJson, null, 2));
    console.log(`Updated version for ${packageName} to ${version}`);
  }
});

const updatedMainPackage = { ...mainPackage };
if (updatedMainPackage.optionalDependencies) {
  Object.keys(updatedMainPackage.optionalDependencies).forEach(dep => {
    updatedMainPackage.optionalDependencies[dep] = version;
  });

  fs.writeFileSync('package.json', JSON.stringify(updatedMainPackage, null, 2));
  console.log('Updated optionalDependencies versions in main package.json');
}
