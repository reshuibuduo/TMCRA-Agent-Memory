param(
    [ValidateSet('byok', 'local-model')]
    [string]$Mode = 'byok',
    [ValidateSet('compact-zh', 'balanced-multilingual', 'enhanced-multilingual')]
    [string]$Embedding = 'balanced-multilingual',
    [ValidateSet('auto', 'cpu', 'cuda', 'mps')]
    [string]$EmbeddingDevice = 'auto',
    [ValidateSet('auto', 'cpu', 'cu128', 'skip')]
    [string]$Torch = 'auto',
    [string]$Provider = '',
    [string]$BaseUrl = '',
    [string]$Model = '',
    [string]$ApiKeyEnv = '',
    [string]$Python = 'python',
    [string]$GenerationRuntimeExecutable = '',
    [switch]$AcceptLargeModel,
    [switch]$SkipGenerationProbe,
    [switch]$NonInteractive
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Runtime = Join-Path $Repo 'runtime'
$LocalRoot = Join-Path $Repo '.tmcra'
$Venv = Join-Path $LocalRoot 'venv'
$ConfigRoot = Join-Path $LocalRoot 'config'
$Config = Join-Path $ConfigRoot 'runtime/local-runtime.json'
$Py = Join-Path $Venv 'Scripts/python.exe'

function Ask([string]$Prompt, [string]$Default = '') {
    if ($NonInteractive) {
        if ($Default) { return $Default }
        throw "Missing required non-interactive value: $Prompt"
    }
    $value = Read-Host ($Prompt + $(if ($Default) { " [$Default]" } else { '' }))
    if (-not $value) { return $Default }
    return $value
}

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true, Position = 0)]
        [string]$FilePath,
        [Parameter(Position = 1, ValueFromRemainingArguments = $true)]
        [string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

function Protect-OwnerDirectory([string]$Path) {
    New-Item -ItemType Directory -Force -Path $Path | Out-Null
    $identity = (& whoami.exe).Trim()
    if ($LASTEXITCODE -ne 0 -or -not $identity -or $identity.Contains("`n") -or $identity.Contains("`r")) {
        throw 'Could not resolve the current Windows security principal.'
    }
    Invoke-Native 'icacls.exe' $Path '/inheritance:r' '/grant:r' "${identity}:(OI)(CI)F" '/Q'
}

Invoke-Native $Python '-c' "import sys; assert sys.version_info[:2] == (3, 12), 'TMCRA local requires Python 3.12'"
Protect-OwnerDirectory $LocalRoot
if (-not (Test-Path $Py)) { Invoke-Native $Python '-m' 'venv' $Venv }
Invoke-Native $Py '-m' 'pip' 'install' '--upgrade' 'pip>=26.1.2' 'wheel'

if ($Torch -eq 'auto') {
    $Torch = if (Get-Command nvidia-smi -ErrorAction SilentlyContinue) { 'cu128' } else { 'cpu' }
}
if ($Torch -eq 'cu128') {
    Invoke-Native $Py '-m' 'pip' 'install' 'torch==2.11.0+cu128' '--extra-index-url' 'https://download.pytorch.org/whl/cu128'
} elseif ($Torch -eq 'cpu') {
    Invoke-Native $Py '-m' 'pip' 'install' 'torch==2.11.0+cpu' '--index-url' 'https://download.pytorch.org/whl/cpu'
} else {
    Invoke-Native $Py '-c' 'import torch; print(torch.__version__)'
}
Invoke-Native $Py '-m' 'pip' 'install' '-e' $Runtime

if (-not (Get-Command git -ErrorAction SilentlyContinue)) {
    throw 'Git with Git LFS is required. Clone this repository with Git before running the installer.'
}
Invoke-Native 'git' 'lfs' 'version'
Invoke-Native 'git' '-C' $Repo 'lfs' 'pull' '--include=models/tmcra_v4_longmemeval_s500_20260715/**'

if ($Mode -eq 'byok') {
    if (-not $Provider) { $Provider = Ask 'OpenAI-compatible provider name' 'openai-compatible' }
    if (-not $BaseUrl) { $BaseUrl = Ask 'Credential-free /v1 base URL' 'https://api.deepseek.com/v1' }
    if (-not $Model) { $Model = Ask 'Model id' 'deepseek-chat' }
    $ByokKeyFile = Join-Path $ConfigRoot 'runtime/secrets/byok-api.key'
    Invoke-Native $Py '-m' 'tmcra_local' 'configure' '--embedding' $Embedding '--embedding-device' $EmbeddingDevice `
        --llm-policy byok --byok-provider $Provider --byok-base-url $BaseUrl `
        --byok-model $Model --byok-api-key-file $ByokKeyFile --config-root $ConfigRoot
    if ($ApiKeyEnv) {
        $source = Get-Item -LiteralPath ("Env:" + $ApiKeyEnv) -ErrorAction SilentlyContinue
        if ($null -eq $source -or -not $source.Value) {
            throw "The environment variable named by -ApiKeyEnv is empty: $ApiKeyEnv"
        }
        $plain = [string]$source.Value
    } else {
        if ($NonInteractive) { throw 'Non-interactive BYOK install requires -ApiKeyEnv.' }
        $secure = Read-Host 'BYOK API key (stored only in .tmcra/config/runtime/secrets)' -AsSecureString
        $plain = [System.Net.NetworkCredential]::new('', $secure).Password
    }
    try {
        $env:TMCRA_INSTALL_API_KEY = $plain
        Invoke-Native $Py '-m' 'tmcra_local' 'set-key' '--config' $Config '--from-env' 'TMCRA_INSTALL_API_KEY'
    } finally {
        Remove-Item Env:TMCRA_INSTALL_API_KEY -ErrorAction SilentlyContinue
        $plain = $null
    }
} else {
    if (-not $GenerationRuntimeExecutable) {
        throw 'Local model mode requires -GenerationRuntimeExecutable pointing to a llama-server executable.'
    }
    $GenerationRuntimeExecutable = (Resolve-Path -LiteralPath $GenerationRuntimeExecutable).Path
    if (-not (Test-Path -LiteralPath $GenerationRuntimeExecutable -PathType Leaf)) {
        throw 'The supplied llama-server executable does not exist.'
    }
    if (-not $AcceptLargeModel) {
        if ($NonInteractive) { throw 'Local model mode requires -AcceptLargeModel.' }
        $confirm = Read-Host 'Download the suggested 12.74 GiB Qwen3.6 model for 32K context? [y/N]'
        if ($confirm -notin @('y', 'Y', 'yes', 'YES')) { throw 'Local model download cancelled.' }
    }
    Invoke-Native $Py '-m' 'tmcra_local' 'configure' '--embedding' $Embedding '--embedding-device' $EmbeddingDevice `
        --llm-policy local-model --generation-profile recommended-qwen36 `
        --generation-runtime-executable $GenerationRuntimeExecutable --config-root $ConfigRoot
    Invoke-Native $Py '-m' 'tmcra_local' 'download-model' '--generation' 'recommended-qwen36' '--models-root' (Join-Path $ConfigRoot 'models') '--execute'
}

Invoke-Native $Py '-m' 'tmcra_local' 'download-model' '--embedding' $Embedding '--models-root' (Join-Path $ConfigRoot 'models') '--execute'
if ($SkipGenerationProbe) {
    Invoke-Native $Py '-m' 'tmcra_local' 'doctor' '--config' $Config '--probe-models' '--json'
} else {
    Invoke-Native $Py '-m' 'tmcra_local' 'doctor' '--config' $Config '--probe-models' '--probe-generation' '--json'
}
Invoke-Native $Py '-m' 'tmcra_local' 'token' '--config' $Config
Write-Host "TMCRA local install is ready. Start with:"
Write-Host "  powershell -ExecutionPolicy Bypass -File `"$Repo\scripts\start-local.ps1`""
