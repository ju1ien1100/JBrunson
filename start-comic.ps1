param()

$root     = Split-Path -Parent $MyInvocation.MyCommand.Path
$webgenta = Join-Path $root "webgenta"
$frontend = Join-Path $root "frontend"
$envFile  = Join-Path $webgenta ".env"

if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match '^([A-Z_][A-Z0-9_]*)=(.+)$') {
            Set-Item -Path "Env:\$($Matches[1])" -Value $Matches[2]
        }
    }
    Write-Host "Loaded $envFile" -ForegroundColor Green
} else {
    Write-Warning "webgenta\.env not found"
}

Write-Host ""
Write-Host "Starting servers..." -ForegroundColor Cyan
Write-Host "  ws_server.py  ->  ws://localhost:8765" -ForegroundColor DarkGray
Write-Host "  model_server  ->  http://localhost:8766" -ForegroundColor DarkGray
Write-Host ""
Write-Host "Open: http://localhost:5173/comic.html" -ForegroundColor Yellow
Write-Host "Press Ctrl+C to stop." -ForegroundColor DarkGray
Write-Host ""

$wsArgs    = "-NoExit -Command `"Set-Location '$webgenta'; python ws_server.py --modal`""
$modelArgs = "-NoExit -Command `"Set-Location '$frontend'; python model_server.py`""

$wsJob    = Start-Process powershell -ArgumentList $wsArgs    -PassThru
$modelJob = Start-Process powershell -ArgumentList $modelArgs -PassThru

Start-Job -ScriptBlock { Start-Sleep 5; Start-Process "http://localhost:5173/comic.html" } | Out-Null

Write-Host "Both servers started. Close this window or press any key to stop them." -ForegroundColor Cyan
$null = $Host.UI.RawUI.ReadKey("NoEcho,IncludeKeyDown")

if ($wsJob    -and -not $wsJob.HasExited)    { Stop-Process -Id $wsJob.Id    -Force -ErrorAction SilentlyContinue }
if ($modelJob -and -not $modelJob.HasExited) { Stop-Process -Id $modelJob.Id -Force -ErrorAction SilentlyContinue }
