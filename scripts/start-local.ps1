param([switch]$AccessLog)
$ErrorActionPreference = 'Stop'
$Repo = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
$Py = Join-Path $Repo '.tmcra/venv/Scripts/python.exe'
$Config = Join-Path $Repo '.tmcra/config/runtime/local-runtime.json'
if (-not (Test-Path $Py)) { throw 'Run scripts/install-local.ps1 first.' }
if (-not (Test-Path $Config)) { throw 'Local runtime config is missing; reinstall or run tmcra-local configure.' }
$arguments = @('-m', 'tmcra_local', 'start', '--config', $Config)
if ($AccessLog) { $arguments += '--access-log' }
& $Py @arguments
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
