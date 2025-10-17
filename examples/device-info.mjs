import { getBackendDevicesInfo } from '../lib/index.js'

// Get variant from command line args (default, vulkan, cuda)
const variant = process.argv[2] || 'default'

console.log(`Querying backend devices for variant: ${variant}\n`)

// Get information about available backend devices
const devices = await getBackendDevicesInfo(variant)

console.log(`Found ${devices.length} backend device(s):\n`)

devices.forEach((device, index) => {
  console.log(`Device ${index + 1}:`)
  console.log(`  Backend: ${device.backend}`)
  console.log(`  Type: ${device.type}`)
  console.log(`  Name: ${device.deviceName}`)
  console.log(`  Memory: ${(device.maxMemorySize / (1024 ** 3)).toFixed(2)} GB`)

  if (device.metadata) {
    console.log(`  Features:`)
    Object.entries(device.metadata)
      .filter(([_, value]) => typeof value === 'boolean' && value)
      .forEach(([key]) => console.log(`    - ${key}`))
  }
  console.log()
})

console.log('Usage: node examples/device-info.mjs [variant]')
console.log('  variant: default, vulkan, or cuda')
