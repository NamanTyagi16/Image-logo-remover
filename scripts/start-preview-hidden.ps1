param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"

$projectRoot = Split-Path -Parent $PSScriptRoot
$previewDir = Join-Path $projectRoot ".preview"
$pidFile = Join-Path $previewDir "server.pid"
$stdoutLog = Join-Path $previewDir "server.out.log"
$stderrLog = Join-Path $previewDir "server.err.log"
$url = "http://127.0.0.1:8000/"
$pythonw = "C:\Users\neuzb\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\pythonw.exe"
$python = "C:\Users\neuzb\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe"
$pythonExe = if (Test-Path $python) { $python } else { $pythonw }
$projectPython = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (Test-Path $projectPython) { $pythonExe = $projectPython }

New-Item -ItemType Directory -Force -Path $previewDir | Out-Null

function Test-PreviewReady {
    try {
        $response = Invoke-RestMethod -Uri ($url + "api/health") -TimeoutSec 2
        return $response.app -eq "LogoErase" -and $response.status -eq "ok"
    } catch {
        return $false
    }
}

if (-not (Test-PreviewReady)) {
    $listener = netstat -ano -p tcp | Select-String '127\.0\.0\.1:8000\s+\S+\s+LISTENING'
    if (-not $listener) {
        if (-not (Test-Path $pythonExe)) {
            throw "Python runtime is missing: $pythonExe"
        }
        $server = Start-Process `
            -FilePath $pythonExe `
            -ArgumentList "backend\app.py" `
            -WorkingDirectory $projectRoot `
            -WindowStyle Hidden `
            -RedirectStandardOutput $stdoutLog `
            -RedirectStandardError $stderrLog `
            -PassThru

        Set-Content -Path $pidFile -Value $server.Id
    }

    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Date) -lt $deadline) {
        if (Test-PreviewReady) {
            break
        }
        if ($server -and $server.HasExited) {
            throw "LogoErase could not start. See $stderrLog"
        }
        Start-Sleep -Milliseconds 700
    }
}

if (-not (Test-PreviewReady)) {
    throw "LogoErase did not become ready. See $stderrLog; port 8000 may be occupied."
}
if (-not $NoBrowser) { Start-Process $url }
