[CmdletBinding()]
param(
    [string]$RuntimeConfig = '',
    [string]$IntegrationConfig = '',
    [string]$IntegrationStateDir = '',
    [string]$Profile = 'web',
    [string]$PackageDirectory = '',
    [string]$NodePath = 'node',
    [string]$NpmPath = 'npm',
    [string]$DshPath = 'dsh',
    [switch]$SkipTests,
    [switch]$SkipDshInstall
)

$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$PackageRoot = Join-Path $Repo 'integrations\deepseek-harness-local'
if (-not $RuntimeConfig) {
    $RuntimeConfig = Join-Path $Repo '.tmcra\config\runtime\local-runtime.json'
}
$RuntimeConfig = (Resolve-Path -LiteralPath $RuntimeConfig).Path
if (-not $PackageDirectory) {
    $PackageDirectory = Join-Path $HOME '.tmcra\packages'
}
$PackageDirectory = [System.IO.Path]::GetFullPath($PackageDirectory)
if ($PackageDirectory -match '[^\x20-\x7E]' -or $PackageDirectory -match '\s') {
    throw 'DeepSeek Harness preview can mis-handle package paths containing spaces or non-ASCII characters. Use -PackageDirectory with a short ASCII-only path.'
}
New-Item -ItemType Directory -Force -Path $PackageDirectory | Out-Null

function Invoke-Native {
    param(
        [Parameter(Mandatory = $true, Position = 0)][string]$FilePath,
        [Parameter(Position = 1, ValueFromRemainingArguments = $true)][string[]]$Arguments
    )
    & $FilePath @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Command failed with exit code $LASTEXITCODE`: $FilePath $($Arguments -join ' ')"
    }
}

Invoke-Native $NodePath '--version'
$configureArguments = @(
    (Join-Path $Repo 'integrations\local-agent-hooks\scripts\configure.mjs'),
    '--runtime-config',
    $RuntimeConfig
)
if ($IntegrationConfig) { $configureArguments += @('--output', $IntegrationConfig) }
if ($IntegrationStateDir) { $configureArguments += @('--state-dir', $IntegrationStateDir) }
Invoke-Native $NodePath @configureArguments

Push-Location $PackageRoot
try {
    Invoke-Native $NpmPath 'ci' '--ignore-scripts' '--no-audit' '--no-fund'
    Invoke-Native $NpmPath 'run' 'typecheck'
    if (-not $SkipTests) { Invoke-Native $NpmPath 'test' }
    Invoke-Native $NpmPath 'run' 'build'
    Invoke-Native $NpmPath 'pack' '--pack-destination' $PackageDirectory
    $package = Get-Content -Raw -LiteralPath (Join-Path $PackageRoot 'package.json') | ConvertFrom-Json
    $fileName = "$($package.name)-$($package.version).tgz"
    $tarball = Join-Path $PackageDirectory $fileName
    if (-not (Test-Path -LiteralPath $tarball -PathType Leaf)) {
        throw 'npm did not create the expected DeepSeek Harness plugin tarball.'
    }
}
finally {
    Pop-Location
}

if ($SkipDshInstall) {
    Write-Host "DeepSeek Harness package verified at: $tarball"
} else {
    Invoke-Native $DshPath 'plugin' '--profile' $Profile 'add' $tarball
    Write-Host "TMCRA owner-local memory was added to DeepSeek Harness profile '$Profile'."
    Write-Host 'Start the local TMCRA API before starting Harness.'
}
