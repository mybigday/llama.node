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
if (($arch -eq "all" -or $arch -eq "x64") -and ($target -eq "all" -or $target -eq "cuda")) {
  if (-Not (Get-Command nvcc -ErrorAction SilentlyContinue)) {
    throw "nvcc.exe not found, please install CUDA"
  }
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
