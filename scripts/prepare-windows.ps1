param (
  [string]$arch = "native",
  [string]$target = "default",
  [string]$toolchain = "clang-cl"
)

$ErrorActionPreference='Stop'

$nativeArch = [System.Runtime.InteropServices.RuntimeInformation]::ProcessArchitecture

if ($arch -eq "native") {
  if ($nativeArch -eq "Arm64") {
    $arch = "arm64"
  } else {
    $arch = "x64"
  }
}

# install CUDA
if (($arch -eq "all" -or $arch -eq "x64") -and $env:CUDA_PATH -eq $null -and ($target -eq "all" -or $target -eq "cuda")) {
  choco install cuda --version=12.9.1.576 -y

  if ($env:ChocolateyInstall -eq $null) {
    $env:ChocolateyInstall = Convert-Path "$((Get-Command choco).Path)\..\.."
  }
  Import-Module "$env:ChocolateyInstall\helpers\chocolateyProfile.psm1"
  refreshenv
}

if (Get-Command ccache -ErrorAction SilentlyContinue) {
  choco install ccache -y
}

if ($toolchain -eq "mingw-clang") {
  $version = "20250528"
  if ($nativeArch -eq "X64") {
    $name = "llvm-mingw-${version}-ucrt-x86_64"
  } elseif ($nativeArch -eq "Arm64") {
    $name = "llvm-mingw-${version}-ucrt-aarch64"
  }
  Invoke-WebRequest -Uri "https://github.com/mstorsjo/llvm-mingw/releases/download/${version}/${name}.zip" -OutFile "llvm-mingw.zip"
  Expand-Archive -Path "llvm-mingw.zip" -DestinationPath .
  $env:Path += ";$(Resolve-Path $name\bin)"

  choco install ninja -y
}

if ($env:GITHUB_ENV -ne $null) {
  Add-Content -Path $env:GITHUB_ENV -Value "PATH=$env:Path"
}
