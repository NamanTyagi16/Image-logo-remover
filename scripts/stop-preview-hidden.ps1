$ErrorActionPreference = "SilentlyContinue"

$projectRoot = Split-Path -Parent $PSScriptRoot
$previewDir = Join-Path $projectRoot ".preview"
$pidFile = Join-Path $previewDir "server.pid"

if (Test-Path $pidFile) {
    $serverPid = (Get-Content $pidFile -Raw).Trim()
    if ($serverPid) {
        Stop-Process -Id $serverPid -Force
    }
    Remove-Item -LiteralPath $pidFile -Force
}
