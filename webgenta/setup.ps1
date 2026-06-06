# setup.ps1 — One-shot setup for Webgenta (MRT2 browser demo)
# Run from C:\Users\<you>\JBrunson with the venv NOT yet activated:
#   powershell -ExecutionPolicy Bypass -File webgenta\setup.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot   # JBrunson directory

Write-Host "`n=== Webgenta Setup ===" -ForegroundColor Cyan

# ── 1. Long paths ────────────────────────────────────────────────────────────
Write-Host "`n[1/6] Checking Windows long path support..." -ForegroundColor Yellow
$longPaths = (Get-ItemProperty "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem").LongPathsEnabled
if ($longPaths -ne 1) {
    Write-Host "      Enabling long paths (requires admin)..." -ForegroundColor Yellow
    try {
        Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1
        Write-Host "      Long paths enabled." -ForegroundColor Green
    } catch {
        Write-Warning "Could not enable long paths (not running as admin). If you hit path errors, run this script as Administrator."
    }
} else {
    Write-Host "      Already enabled." -ForegroundColor Green
}

# ── 2. Clone magenta-realtime ────────────────────────────────────────────────
Write-Host "`n[2/6] Checking magenta-realtime repo..." -ForegroundColor Yellow
$MrtDir = Join-Path $Root "magenta-realtime"
if (-not (Test-Path $MrtDir)) {
    Write-Host "      Cloning magenta-realtime..." -ForegroundColor Yellow
    git clone https://github.com/google-deepmind/magenta-realtime $MrtDir
} else {
    Write-Host "      Already present at $MrtDir" -ForegroundColor Green
}

# ── 3. Python venv ───────────────────────────────────────────────────────────
Write-Host "`n[3/6] Setting up Python venv..." -ForegroundColor Yellow
$VenvDir = Join-Path $Root ".venv"
if (-not (Test-Path $VenvDir)) {
    python -m venv $VenvDir
    Write-Host "      Venv created." -ForegroundColor Green
} else {
    Write-Host "      Already exists." -ForegroundColor Green
}

$pip = Join-Path $VenvDir "Scripts\pip.exe"
$python = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "`n[4/6] Installing Python dependencies..." -ForegroundColor Yellow
& $pip install --upgrade pip --quiet
& $pip install -e "$MrtDir[jax]"
& $pip install -e "$MrtDir\magenta_rt\_vendor\sequence-layers"
& $pip install -r (Join-Path $PSScriptRoot "requirements.txt")

# ── 4. Download model checkpoints ───────────────────────────────────────────
Write-Host "`n[5/6] Downloading model checkpoints from HuggingFace..." -ForegroundColor Yellow
$DownloadScript = Join-Path $PSScriptRoot "download_model.py"

@'
import sys, shutil, pathlib
try:
    from huggingface_hub import hf_hub_download
except ImportError:
    print("Installing huggingface_hub...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "huggingface_hub", "-q"])
    from huggingface_hub import hf_hub_download

repo = "google/magenta-realtime-2"
base = pathlib.Path.home() / "Documents/Magenta/magenta-rt-v2"

files = [
    ("checkpoints/mrt2_small.safetensors", "checkpoints/mrt2_small.safetensors"),
    ("resources/musiccoca/text_encoder.tflite", "resources/musiccoca/text_encoder.tflite"),
    ("resources/musiccoca/vocab.txt",            "resources/musiccoca/vocab.txt"),
    ("resources/musiccoca/config.json",           "resources/musiccoca/config.json"),
    ("resources/spectrostream/config.json",       "resources/spectrostream/config.json"),
]

for repo_path, local_rel in files:
    dst = base / local_rel
    if dst.exists():
        print(f"  already present: {local_rel}")
        continue
    print(f"  downloading {repo_path}...")
    src = hf_hub_download(repo, repo_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    print(f"  -> {dst}")

print("Checkpoint download complete.")
'@ | Set-Content $DownloadScript -Encoding utf8

& $python $DownloadScript
Remove-Item $DownloadScript

# ── 5. Web dependencies ──────────────────────────────────────────────────────
Write-Host "`n[6/6] Installing web dependencies (npm)..." -ForegroundColor Yellow
$WebDir = Join-Path $PSScriptRoot "web"
Push-Location $WebDir
npm install
Pop-Location

# ── Done ─────────────────────────────────────────────────────────────────────
Write-Host "`n=== Setup complete! ===" -ForegroundColor Cyan
Write-Host @"

To run:

  Terminal 1 (Python server):
    cd $Root
    .\.venv\Scripts\Activate.ps1
    python -u webgenta\server.py --model mrt2_small

  Terminal 2 (Vite dev server):
    cd $Root\webgenta\web
    npm run dev

  Then open http://localhost:5173 in Chrome or Edge.
"@ -ForegroundColor White
