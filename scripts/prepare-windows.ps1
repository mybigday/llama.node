param (
  [string]$arch = "native"
)

if ($arch -eq "native") {
  if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq "Arm64") {
    $arch = "arm64"
  } else {
    $arch = "x64"
  }
}

$ErrorActionPreference='Stop'

# check processor architecture
$Arch = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture

if ($Arch -eq "X64") {
  $VULKAN_URL = "https://sdk.lunarg.com/sdk/download/1.4.313.1/windows/vulkansdk-windows-X64-1.4.313.1.exe"
} else {
  $VULKAN_URL = "https://sdk.lunarg.com/sdk/download/1.4.313.1/windows/vulkansdk-windows-ARM64-1.4.313.1.exe"
}

$VULKAN_COMPONENTS = "com.lunarg.vulkan.sdl2 com.lunarg.vulkan.glm com.lunarg.vulkan.volk com.lunarg.vulkan.vma com.lunarg.vulkan.debug"

if ($Arch -eq "X64" -and ($arch -eq "all" -or $arch -eq "arm64")) {
  $VULKAN_COMPONENTS += " com.lunarg.vulkan.arm64"
}

if ($Arch -eq "Arm64" -and ($arch -eq "all" -or $arch -eq "x64")) {
  $VULKAN_COMPONENTS += " com.lunarg.vulkan.x64"
}

# install vulkan sdk
if ($env:VULKAN_SDK -eq $null) {
  Invoke-WebRequest $VULKAN_URL -OutFile "vulkansdk.exe"
  .\vulkansdk.exe --accept-licenses --default-answer --confirm-command install $VULKAN_COMPONENTS
  rm vulkansdk.exe

  $env:VK_SDK_PATH = "C:\VulkanSDK\1.4.313.1"
  $env:VULKAN_SDK = "C:\VulkanSDK\1.4.313.1"
  $env:PATH = "C:\VulkanSDK\1.4.313.1\Bin;$env:PATH"

  if ($env:GITHUB_ENV -ne $null) {
    Add-Content -Path $env:GITHUB_ENV -Value "VULKAN_SDK=$env:VULKAN_SDK" -Append
    Add-Content -Path $env:GITHUB_ENV -Value "VK_SDK_PATH=$env:VK_SDK_PATH" -Append
  }
}

# install CUDA
if (($arch -eq "all" -or $arch -eq "x64") -and $env:CUDA_PATH -eq $null) {
  choco install cuda --version=12.9.1.576 -y
  $env:PATH += ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin'
  $env:PATH += ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\libnvvp'
  $env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
  $env:CUDA_PATH_V12_9 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'

  if ($env:GITHUB_ENV -ne $null) {
    Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH=$env:CUDA_PATH" -Append
    Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH_V12_9=$env:CUDA_PATH_V12_9" -Append
  }
}

if ($env:GITHUB_ENV -ne $null) {
  Add-Content -Path $env:GITHUB_ENV -Value "PATH=$env:PATH" -Append
}
