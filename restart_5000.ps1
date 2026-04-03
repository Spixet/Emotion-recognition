$ErrorActionPreference = "Stop"

function Stop-Port5000Listeners {
    $listeners = Get-NetTCPConnection -LocalPort 5000 -State Listen -ErrorAction SilentlyContinue
    if (-not $listeners) {
        Write-Host "[INFO] No listener found on port 5000."
        return
    }

    $targetPids = $listeners | Select-Object -ExpandProperty OwningProcess -Unique
    foreach ($targetPid in $targetPids) {
        if (-not $targetPid -or $targetPid -eq $PID) {
            continue
        }

        try {
            $proc = Get-Process -Id $targetPid -ErrorAction Stop
            Write-Host "[INFO] Stopping PID $targetPid ($($proc.ProcessName)) using port 5000..."
            Stop-Process -Id $targetPid -Force -ErrorAction Stop
        } catch {
            Write-Host "[WARN] Failed to stop PID ${targetPid}: $($_.Exception.Message)"
        }
    }
}

function Start-Eternix {
    $projectRoot = if ($PSScriptRoot) { $PSScriptRoot } else { (Get-Location).Path }
    $pythonPath = Join-Path $projectRoot "venv\Scripts\python.exe"
    $appPath = Join-Path $projectRoot "app.py"

    if (-not (Test-Path $pythonPath)) {
        throw "Python executable not found at $pythonPath"
    }
    if (-not (Test-Path $appPath)) {
        throw "app.py not found at $appPath"
    }

    Write-Host "[START] Launching Eternix on http://localhost:5000 ..."
    & $pythonPath $appPath
}

Stop-Port5000Listeners
Start-Sleep -Milliseconds 400
Start-Eternix
