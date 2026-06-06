param()

$root = Split-Path -Parent $MyInvocation.MyCommand.Path
$envFile = Join-Path $root "webgenta" ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([A-Z_][A-Z0-9_]*)=(.+)$') {
            Set-Item -Path "Env:\$($Matches[1])" -Value $Matches[2]
        }
    }
    Write-Host "Loaded environment from webgenta\.env" -ForegroundColor Green
} else {
    Write-Warning "webgenta\.env not found"
}

Start-Job -ScriptBlock {
    Start-Sleep -Seconds 5
    Start-Process "http://localhost:8766/"
} | Out-Null

Write-Host ""
Write-Host "Starting comic server on http://localhost:8766/" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

$frontend = Join-Path $root "frontend"
Set-Location $frontend
python model_server.py
