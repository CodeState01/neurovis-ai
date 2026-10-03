$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
Push-Location -LiteralPath $projectRoot
try {
    $py = Get-Command py -ErrorAction SilentlyContinue
    if (-not (Test-Path -LiteralPath '.venv\Scripts\python.exe')) {
        if ($py) { & py -3.12 -m venv .venv } else { & python -m venv .venv }
        if ($LASTEXITCODE -ne 0) { throw 'Instale Python 3.12 e execute este instalador novamente.' }
    }
    & '.venv\Scripts\python.exe' -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar o mecanismo de IA. Confira a internet e o Python 3.12.' }
    & '.venv\Scripts\python.exe' -m pip install -r student/requirements.txt
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao instalar as dependências do aplicativo.' }
    & ollama --version *> $null
    if ($LASTEXITCODE -ne 0) { throw 'Instale o Ollama em https://ollama.com/download/windows e tente novamente.' }
    & ollama pull qwen3.5:4b
    if ($LASTEXITCODE -ne 0) { throw 'Falha ao baixar o professor Qwen pelo Ollama.' }
    & '.venv\Scripts\python.exe' -c "import tkinter, torch, PIL; print('Python', __import__('sys').version.split()[0], '| Torch CUDA:', torch.cuda.is_available(), '| Tk:', tkinter.TkVersion, '| Pillow:', PIL.__version__)"
    Write-Host 'Instalação concluída. Abra Iniciar-Neurovis.cmd.' -ForegroundColor Green
} finally { Pop-Location }
