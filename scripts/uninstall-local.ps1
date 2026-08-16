param([switch]$PurgeData)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Target = [IO.Path]::GetFullPath((Join-Path $Repo '.tmcra'))
$Expected = [IO.Path]::GetFullPath((Join-Path $Repo '.tmcra'))
if ($Target -ne $Expected -or -not $Target.StartsWith($Repo + [IO.Path]::DirectorySeparatorChar, [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Refusing to remove a path outside this repository.'
}
if (-not (Test-Path -LiteralPath $Target)) {
    Write-Host 'TMCRA local data is already absent.'
    exit 0
}
if ($PurgeData) {
    Remove-Item -LiteralPath $Target -Recurse -Force
    Write-Host 'Removed runtime, models, credentials, and local memory databases.'
} else {
    $Venv = Join-Path $Target 'venv'
    if (Test-Path -LiteralPath $Venv) { Remove-Item -LiteralPath $Venv -Recurse -Force }
    Write-Host "Removed the Python environment. Memory, models, config, and credentials remain in: $Target"
    Write-Host 'Run again with -PurgeData only when you intend to erase all local TMCRA data.'
}
