param(
    [ValidateSet('cpu', 'cuda')]
    [string]$Device = 'cpu'
)

$ErrorActionPreference = 'Stop'

function Invoke-Checked {
    param([string]$Program, [string[]]$Arguments)
    & $Program @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code ${LASTEXITCODE}: $Program $($Arguments -join ' ')"
    }
}

if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    throw 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/'
}

$RepoRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepoRoot '.venv-win\Scripts\python.exe'
$Requirements = if ($Device -eq 'cuda') { 'requirements/pc-cu121.txt' } else { 'requirements/pc-cpu.txt' }

Push-Location $RepoRoot
try {
    Invoke-Checked -Program 'uv' -Arguments @('python', 'install', '3.10')
    if (-not (Test-Path $Python)) {
        Invoke-Checked -Program 'uv' -Arguments @('venv', '--python', '3.10', '--seed', '.venv-win')
    }
    Invoke-Checked -Program $Python -Arguments @('-c', "import sys; assert sys.version_info[:2] == (3, 10), 'Use a fresh Python 3.10 environment'; print(sys.executable)")
    Invoke-Checked -Program $Python -Arguments @('-m', 'pip', 'install', '-r', $Requirements, '-e', '.')
    Invoke-Checked -Program $Python -Arguments @('-m', 'pip', 'check')
    Invoke-Checked -Program $Python -Arguments @('-c', "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())")
    Write-Host "Ready. Example: .\.venv-win\Scripts\python.exe read.py pretrained=parseq --images demo_images/coco-1166773.jpg --device $Device"
}
finally {
    Pop-Location
}
