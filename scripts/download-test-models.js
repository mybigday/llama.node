#!/usr/bin/env node

const fs = require('fs')
const path = require('path')
const https = require('https')
const { createWriteStream } = require('fs')
const { mkdir } = require('fs/promises')

// Ensure test directory exists
const ensureDir = async (dir) => {
  try {
    await mkdir(dir, { recursive: true })
  } catch (error) {
    if (error.code !== 'EEXIST') throw error
  }
}

// Download file function
const downloadFile = (url, outputPath) => {
  return new Promise((resolve, reject) => {
    const requestUrl = (currentUrl) => {
      https
        .get(currentUrl, (response) => {
          // Handle redirects (301, 302, 303, 307, 308)
          if ([301, 302, 303, 307, 308].includes(response.statusCode)) {
            const location = response.headers.location
            if (!location) {
              reject(
                new Error(
                  `Redirect (${response.statusCode}) without Location header`,
                ),
              )
              return
            }
            console.log(`Following redirect to: ${location}`)
            requestUrl(location)
            return
          }

          if (response.statusCode !== 200) {
            reject(new Error(`Failed to download: ${response.statusCode}`))
            return
          }

          const file = createWriteStream(outputPath)
          response.pipe(file)

          file.on('finish', () => {
            file.close()
            console.log(`Downloaded: ${path.basename(outputPath)}`)
            resolve()
          })

          file.on('error', (err) => {
            fs.unlink(outputPath, () => {})
            reject(err)
          })
        })
        .on('error', (err) => {
          reject(err)
        })
    }

    requestUrl(url)
  })
}

// Main function
async function main() {
  const testDir = path.join(__dirname, '../test')
  await ensureDir(testDir)

  const files = [
    {
      path: path.join(testDir, 'bge-small-en.gguf'),
      url: 'https://huggingface.co/ggml-org/bge-small-en-v1.5-Q8_0-GGUF/resolve/main/bge-small-en-v1.5-q8_0.gguf?download=true',
    },
    {
      path: path.join(testDir, 'tiny-random-llama.gguf'),
      url: 'https://huggingface.co/tensorblock/tiny-random-llama-GGUF/resolve/main/tiny-random-llama-Q4_0.gguf?download=true',
    },
    {
      path: path.join(testDir, 'Qwen3-0.6B-Q6_K.gguf'),
      url: 'https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q6_K.gguf?download=true',
    },
    {
      path: path.join(testDir, 'SmolVLM-256M-Instruct-Q8_0.gguf'),
      url: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/SmolVLM-256M-Instruct-Q8_0.gguf?download=true',
    },
    {
      path: path.join(testDir, 'mmproj-SmolVLM-256M-Instruct-Q8_0.gguf'),
      url: 'https://huggingface.co/ggml-org/SmolVLM-256M-Instruct-GGUF/resolve/main/mmproj-SmolVLM-256M-Instruct-Q8_0.gguf?download=true',
    },
    // Uncomment to test with audio
    // {
    //   path: path.join(testDir, 'Llama-3.2-1B-Instruct-Q4_K_M.gguf'),
    //   url: 'https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/Llama-3.2-1B-Instruct-Q4_K_M.gguf?download=true',
    // },
    // {
    //   path: path.join(testDir, 'mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf'),
    //   url: 'https://huggingface.co/ggml-org/ultravox-v0_5-llama-3_2-1b-GGUF/resolve/main/mmproj-ultravox-v0_5-llama-3_2-1b-f16.gguf?download=true'
    // }
    // // TTS
    // {
    //   path: path.join(testDir, 'OuteTTS-0.3-500M-Q4_K_M.gguf'),
    //   url: 'https://huggingface.co/OuteAI/OuteTTS-0.3-500M-GGUF/resolve/main/OuteTTS-0.3-500M-Q4_K_M.gguf?download=true',
    // },
    // {
    //   path: path.join(testDir, 'WavTokenizer.gguf'),
    //   url: 'https://huggingface.co/ggml-org/WavTokenizer/resolve/main/WavTokenizer-Large-75-Q5_1.gguf?download=true',
    // },
  ]

  for (const file of files) {
    if (!fs.existsSync(file.path)) {
      console.log(`Downloading ${path.basename(file.path)}...`)
      try {
        await downloadFile(file.url, file.path)
      } catch (error) {
        console.error(
          `Error downloading ${path.basename(file.path)}:`,
          error.message,
        )
        process.exit(1)
      }
    } else {
      console.log(`File already exists: ${path.basename(file.path)}`)
    }
  }
  process.exit(0)
}

main().catch(console.error)
