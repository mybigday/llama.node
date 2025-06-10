param (
  [string]$arch = "native",
  [string]$target = "default"
)

if ($arch -eq "native") {
  if ([System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture -eq "Arm64") {
    $arch = "arm64"
  } else {
    $arch = "x64"
  }
}

$ErrorActionPreference='Stop'

# install CUDA
if (($arch -eq "all" -or $arch -eq "x64") -and $env:CUDA_PATH -eq $null -and ($target -eq "all" -or $target -eq "cuda")) {
  choco install cuda --version=12.9.1.576 -y
  $env:PATH += ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin'
  $env:PATH += ';C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\libnvvp'
  $env:CUDA_PATH = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'
  $env:CUDA_PATH_V12_9 = 'C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9'

  if ($env:GITHUB_ENV -ne $null) {
    Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH=$env:CUDA_PATH"
    Add-Content -Path $env:GITHUB_ENV -Value "CUDA_PATH_V12_9=$env:CUDA_PATH_V12_9"
  }
}

if ($env:GITHUB_ENV -ne $null) {
  Add-Content -Path $env:GITHUB_ENV -Value "PATH=$env:PATH"
}

if (Get-Command ccache -ErrorAction SilentlyContinue) {
  choco install ccache -y
}
