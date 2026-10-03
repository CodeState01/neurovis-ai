param([string]$Python = 'python')
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        & $Python -m venv .venv
        if ($LASTEXITCODE -ne 0) { throw 'Não foi possível criar o ambiente Python.' }
    }
    & '.venv\Scripts\python.exe' -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar PyTorch. Use Python 3.12.' }
    & '.venv\Scripts\python.exe' -m pip install -r student/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar o treinamento.' }
    Write-Host 'Treinamento instalado. Inicie o Neurovis e clique em Pedir ao professor.'
} finally { Pop-Location }
