#!/usr/bin/env node

const fs = require('fs');
const path = require('path');
const { execSync } = require('child_process');

// Colors for output
const colors = {
    red: '\x1b[31m',
    green: '\x1b[32m',
    yellow: '\x1b[33m',
    reset: '\x1b[0m'
};

function log(message, color = 'reset') {
    console.log(`${colors[color]}${message}${colors.reset}`);
}

log('Starting copy of llama.rn source files...', 'green');

// Define source and destination directories
const LLAMA_RN_DIR = 'src/llama.rn';
const SRC_DIR = 'src/llama.rn/cpp';
const DEST_DIR = 'src/rn-llama';
const LLAMA_RN_PATCH_FILE = 'scripts/llama.rn-mtp.patch';

// Files to copy from llama.rn/cpp
const FILES_TO_COPY = [
    'anyascii.h',
    'anyascii.c',
    'rn-llama.h',
    'rn-llama.cpp',
    'rn-completion.h',
    'rn-completion.cpp',
    'rn-mtmd.hpp',
    'rn-tts.h',
    'rn-tts.cpp',
    // Parallel decoding support
    'rn-common.hpp',
    'rn-slot.h',
    'rn-slot.cpp',
    'rn-slot-manager.h',
    'rn-slot-manager.cpp'
];

try {
    // Create destination directory
    if (!fs.existsSync(DEST_DIR)) {
        fs.mkdirSync(DEST_DIR, { recursive: true });
    }

    // Clear destination directory
    if (fs.existsSync(DEST_DIR)) {
        const files = fs.readdirSync(DEST_DIR);
        files.forEach(file => {
            fs.unlinkSync(path.join(DEST_DIR, file));
        });
    }

    // Initialize and update llama.rn submodule if needed
    log('Ensuring llama.rn submodule is initialized...', 'yellow');
    const headerPath = path.join(SRC_DIR, 'rn-llama.h');

    if (!fs.existsSync(headerPath)) {
        log('Initializing llama.rn submodule...');
        execSync('git submodule init src/llama.rn', { stdio: 'inherit' });
        execSync('git submodule update --recursive src/llama.rn', { stdio: 'inherit' });
    }

    if (fs.existsSync(LLAMA_RN_PATCH_FILE)) {
        const patchFile = path.resolve(LLAMA_RN_PATCH_FILE);
        const completionSourcePath = path.join(SRC_DIR, 'rn-completion.cpp');
        const completionSource = fs.readFileSync(completionSourcePath, 'utf8');
        const applyCommand = `git -C ${LLAMA_RN_DIR} apply --whitespace=nowarn ${patchFile}`;
        const checkCommand = `git -C ${LLAMA_RN_DIR} apply --check --whitespace=nowarn ${patchFile}`;

        if (completionSource.includes('COMMON_SPECULATIVE_TYPE_DRAFT_MTP')) {
            log(`${LLAMA_RN_PATCH_FILE} already applied`, 'yellow');
        } else {
            try {
                execSync(checkCommand, { stdio: 'ignore' });
                log(`Applying ${LLAMA_RN_PATCH_FILE}...`, 'yellow');
                execSync(applyCommand, { stdio: 'inherit' });
                log(`✓ ${LLAMA_RN_PATCH_FILE} applied`, 'green');
            } catch {
                log(`${LLAMA_RN_PATCH_FILE} does not match the current llama.rn checkout`, 'red');
                process.exit(1);
            }
        }
    }

    // Copy files and remove lm_ and LM_ prefixes
    FILES_TO_COPY.forEach(file => {
        const srcPath = path.join(SRC_DIR, file);
        const destPath = path.join(DEST_DIR, file);

        if (fs.existsSync(srcPath)) {
            log(`Copying and processing ${file}...`, 'yellow');

            // Read the file and process it to remove lm_ and LM_ prefixes
            let content = fs.readFileSync(srcPath, 'utf8');
            content = content.replace(/lm_ggml/g, 'ggml');
            content = content.replace(/LM_GGML/g, 'GGML');
            content = content.replace(
                /mtmd_decode_use_non_causal\(mtmd_ctx\)/g,
                'mtmd_decode_use_non_causal(mtmd_ctx, nullptr)'
            );

            // Write the processed content to destination
            fs.writeFileSync(destPath, content);

            log(`✓ ${file} processed and copied to ${destPath}`, 'green');
        } else {
            log(`✗ Source file ${srcPath} not found!`, 'red');
            process.exit(1);
        }
    });

    log('All llama.rn source files copied and processed successfully!', 'green');
    log('Note: lm_ggml and LM_GGML prefixes have been removed from all copied files.', 'yellow');

} catch (error) {
    log(`Error: ${error.message}`, 'red');
    process.exit(1);
}
