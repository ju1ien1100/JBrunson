# Webgenta — MRT2 Browser Streaming Demo

Browser app that streams AI-generated music from Google DeepMind's
[Magenta RealTime 2](https://github.com/magenta/magenta-realtime) model via WebSocket.
Type a style prompt, hit Connect, and hear the model generate music conditioned on
a built-in "Mary Had a Little Lamb" MIDI sequence.

---

## Architecture

```
Browser (Vite + TypeScript)
  ├── index.html / main.ts  — UI, WebSocket client, Web MIDI
  ├── public/mrt2-worklet.js — AudioWorklet ring-buffer player
  └── AudioContext @ 48kHz → speakers

Python server  (server.py)
  ├── Loads MagentaRT2System (JAX, CPU)
  ├── Renders Mary Had a Little Lamb offline in per-note batches
  └── Streams pre-rendered float32 PCM over WebSocket
```

Current mode: **offline render → stream**.
The server renders a full pass of the MARY sequence (all notes as batched
`generate()` calls), then streams the audio. Loops continuously.
This avoids real-time CPU pressure and produces stutter-free playback.

---

## Prerequisites

| Tool | Version |
|------|---------|
| Python | 3.10+ |
| Node.js | 18+ |
| Git | any |

Windows-specific requirement: **long paths must be enabled**.
Run once as Administrator:
```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" -Name LongPathsEnabled -Value 1
```

---

## One-Time Setup

Use the provided setup script (see `setup.ps1`) or follow these steps manually.

### 1. Clone magenta-realtime

```powershell
cd C:\Users\<you>\JBrunson
git clone https://github.com/google-deepmind/magenta-realtime magenta-realtime
```

### 2. Create and activate a Python venv

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3. Install magenta-realtime with JAX (CPU)

```powershell
pip install -e "magenta-realtime[jax]"
pip install -e "magenta-realtime\magenta_rt\_vendor\sequence-layers"
```

### 4. Download model checkpoints

```python
# Run once in Python (venv active):
from huggingface_hub import hf_hub_download
import shutil, pathlib

repo = "google/magenta-realtime-2"
base = pathlib.Path.home() / "Documents/Magenta/magenta-rt-v2"

# Checkpoint
src = hf_hub_download(repo, "checkpoints/mrt2_small.safetensors")
dst = base / "checkpoints/mrt2_small.safetensors"
dst.parent.mkdir(parents=True, exist_ok=True)
shutil.copy(src, dst)

# MusicCoCa resources
for f in ["text_encoder.tflite", "vocab.txt", "config.json"]:
    src = hf_hub_download(repo, f"resources/musiccoca/{f}")
    dst = base / "resources/musiccoca" / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)

# SpectroStream resources
for f in ["config.json"]:
    src = hf_hub_download(repo, f"resources/spectrostream/{f}")
    dst = base / "resources/spectrostream" / f
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
```

### 5. Install webgenta Python deps

```powershell
pip install -r webgenta\requirements.txt
```

### 6. Install web dependencies

```powershell
cd webgenta\web
npm install
cd ..\..
```

---

## Running

**Terminal 1 — Python server:**
```powershell
cd C:\Users\<you>\JBrunson
.\.venv\Scripts\Activate.ps1
python -u webgenta\server.py --model mrt2_small
```

Wait for:
```
Model ready! Starting WebSocket server...
Listening on ws://0.0.0.0:8765 — open the browser app to connect
Rendering loop 1...
```
Rendering takes ~30–60 s on CPU. Audio starts streaming once done.

**Terminal 2 — Vite dev server:**
```powershell
cd C:\Users\<you>\JBrunson\webgenta\web
npm run dev
```

Open `http://localhost:5173` in Chrome or Edge.
Enter a style prompt (e.g. "jazzy piano"), click **Connect**.

---

## Known Issues & Limitations

- **CPU-only inference is slow** — rendering one MARY loop (~20 s of audio) takes
  30–60 s on a laptop CPU. A CUDA GPU reduces this to ~2–3 s.
- **No real-time MIDI yet** — the server currently only plays the built-in MARY demo.
  Real-time note input is wired on the client but the server's render loop would need
  to be restructured to react to live MIDI events.
- **Prompt changes reset model state** — changing the style prompt mid-session restarts
  the model state, which may cause a brief musical discontinuity.

---

## Planned Improvements

### Short-term (hackathon)
- [ ] **GPU support** — pass `--model mrt2_base` and run on a CUDA machine for
  real-time generation. The JAX backend auto-detects CUDA; just remove
  `JAX_PLATFORMS=cpu` from server.py.
- [ ] **Live MIDI input** — restructure generate loop to accept incoming note events
  and blend them into the conditioning vector in real-time.
- [ ] **Prompt UI** — style blending slider (interpolate between two style embeddings).

### Real-time path (post-hackathon)
- [ ] **GPU server** — move Python inference to a cloud GPU (Lambda Labs, Modal, etc.)
  and connect the browser over WSS.
- [ ] **Speculative generation** — generate 2–3 frames ahead in parallel to absorb
  network jitter while keeping note latency low.
- [ ] **Adaptive buffer** — worklet reports fill level back to server; server speeds up
  or slows down to maintain a target latency of ~200 ms.
- [ ] **MIDI keyboard** — Web MIDI selector already in the UI; server-side handler
  just needs to be re-enabled.
- [ ] **Drum conditioning** — MIDI channel 10 events already parsed client-side;
  expose a drum toggle in the UI.
- [ ] **Style interpolation** — allow blending between two prompts over time using
  `np.lerp` on the style embeddings.
- [ ] **Export** — "Save as WAV" button that accumulates rendered PCM and triggers a
  browser download.
