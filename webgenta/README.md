# Webgenta

WebSocket orchestration server for AI-generated music, built on top of
[Magenta RealTime 2](https://github.com/magenta/magenta-realtime) (MRT2).

Type a style prompt in the browser, hit Connect, and hear the model generate
music conditioned on a MIDI sequence — streamed in real time from a GPU in the cloud.

---

## Architecture

```
Browser (Vite + TypeScript)
  ├── main.ts          — WebSocket client, Web MIDI forwarding
  ├── public/mrt2-worklet.js — AudioWorklet ring-buffer player
  └── AudioContext @ 48kHz → speakers

ws_server.py  — WebSocket orchestration server
  ├── magenta service  → Modal.com T4 GPU (MRT2 JAX inference)
  ├── suno service     → Suno API  (vocal sound effects)
  └── stability service → Stability.ai (background tracks)

Modal.com  (modal_magenta.py)
  ├── MagentaInference class  — GPU container, stays warm 5 min
  ├── embed_style()  — text → 768-d style embedding
  └── render()       — notes sequence → float32 PCM (stereo, 48kHz)
```

### Message protocol

All WebSocket messages are JSON with a `service` field:

```json
// Client → Server
{ "service": "magenta", "action": "start",   "prompt": "jazzy piano" }
{ "service": "magenta", "action": "stop" }
{ "service": "magenta", "action": "prompt",  "text": "new style" }
{ "service": "magenta", "action": "note_on", "pitch": 64, "velocity": 80 }
{ "service": "magenta", "action": "note_off","pitch": 64 }
{ "service": "magenta", "action": "drum",    "velocity": 100 }

// Server → Client (JSON)
{ "service": "magenta", "event": "status", "state": "embedding" }
{ "service": "magenta", "event": "ready",  "loop": 1, "duration_s": 42.3 }
{ "service": "magenta", "event": "stopped" }
{ "service": "magenta", "event": "error",  "message": "..." }

// Server → Client (binary)
<float32 PCM, stereo interleaved, 48kHz>
```

---

## Prerequisites

| Tool | Version | Notes |
|------|---------|-------|
| Python | 3.11+ | |
| Node.js | 18+ | |
| Modal account | free tier ok | modal.com |
| Git | any | |

**Windows only** — enable long paths once as Administrator:
```powershell
Set-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Control\FileSystem" `
  -Name LongPathsEnabled -Value 1
```

---

## Setup

### 1. Clone repos

```powershell
cd C:\Users\<you>\JBrunson
git clone https://github.com/magenta/magenta-realtime magenta-realtime
git clone https://github.com/yourorg/webgenta webgenta   # or use the existing folder
```

### 2. Python environment

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1

pip install -e "magenta-realtime[jax]"
pip install -e "magenta-realtime\magenta_rt\_vendor\sequence-layers"
pip install -r webgenta\requirements.txt
```

### 3. Modal setup (GPU inference)

```powershell
pip install modal
python -m modal setup    # opens browser for OAuth
```

Upload model weights to Modal volume (one time — ~900 MB):

```powershell
modal volume put magenta-weights `
  "C:\Users\<you>\Documents\Magenta\magenta-rt-v2" /magenta-rt-v2
```

> If you don't have the weights locally yet, run `mrt models init` inside the
> magenta-realtime venv to download them first.

Deploy the inference app:

```powershell
modal deploy webgenta\modal_magenta.py
```

Verify the GPU model loads:

```powershell
modal run webgenta\modal_magenta.py   # runs smoke_test → prints embedding shape
```

### 4. API keys (optional services)

```powershell
copy webgenta\.env.example webgenta\.env
# Edit webgenta\.env and fill in:
#   SUNO_API_KEY=sk_live_...
#   STABILITY_API_KEY=...
```

### 5. Frontend dependencies

```powershell
cd webgenta\web
npm install
cd ..\..
```

---

## Running

**Terminal 1 — WebSocket server (Modal GPU backend):**
```powershell
.\.venv\Scripts\Activate.ps1
python webgenta\ws_server.py --modal
```

Flags:
- `--modal` — use Modal T4 GPU for Magenta (default: local CPU)
- `--no-magenta` — skip MRT2, run Suno + Stability only
- `--port 8765` — change port (default 8765)
- `--model mrt2_base` — use the larger model (local mode only)

**Terminal 2 — Vite dev server:**
```powershell
cd webgenta\web
npm run dev
```

Open `http://localhost:5173` in Chrome or Edge.
Enter a style prompt (e.g. `"ambient lo-fi piano"`), click **Connect**.

First request cold-starts the Modal container (~15 s). Subsequent requests
within 5 minutes use the warm container (~2–3 s render for a full loop).

---

## Development

### Run without Modal (local CPU, slow)

```powershell
python webgenta\ws_server.py
```

Renders on CPU — expect ~60 s per loop for `mrt2_small`. Good for testing
the WebSocket protocol without burning Modal credits.

### Run without Magenta (API services only)

```powershell
python webgenta\ws_server.py --no-magenta
```

Starts the server with only Suno and Stability services registered.
Useful for testing those integrations without loading any ML model.

### Re-deploy after code changes

```powershell
modal deploy webgenta\modal_magenta.py
```

Image is cached — only changed layers rebuild. Usually takes 5–10 s if only
Python code changed.

### Add a new service

1. Create `webgenta/services/myservice.py` with a class that has
   `async def handle(self, ws, message) -> None`
2. Register it in `ws_server.py`:
   ```python
   from services.myservice import MyService
   services["myservice"] = MyService()
   ```
3. Send messages from the browser with `{ "service": "myservice", ... }`

---

## Project structure

```
JBrunson/
├── magenta-realtime/          Google DeepMind MRT2 repo (cloned)
└── webgenta/
    ├── ws_server.py           Entry point — WebSocket router
    ├── modal_magenta.py       Modal GPU deployment (MRT2 inference)
    ├── requirements.txt       Python dependencies
    ├── .env.example           API key template
    ├── services/
    │   ├── __init__.py        msg() helper
    │   ├── magenta.py         MRT2 music generation service
    │   ├── suno.py            Suno API vocal SFX service
    │   └── stability.py       Stability.ai background audio service
    └── web/
        ├── index.html         UI
        ├── main.ts            WebSocket client + Web MIDI
        ├── public/
        │   └── mrt2-worklet.js  AudioWorklet ring-buffer player
        ├── package.json
        └── vite.config.ts
```

---

## Troubleshooting

**`ExecutionError: Function has not been hydrated`**
The server is trying to call Modal without a deployed app.
Run `modal deploy webgenta\modal_magenta.py` first.

**`AsyncUsageWarning` / crash on connect**
Modal `.remote()` called from async context. All Modal calls must use
`.remote.aio()`. Check `services/magenta.py`.

**No audio in browser**
- Check browser console for AudioWorklet errors
- Make sure `AudioContext` sampleRate is 48000 (matches server output)
- Chrome requires a user gesture before creating AudioContext — clicking Connect is enough

**Modal container cold-starting every time**
`scaledown_window=300` keeps the container warm for 5 minutes after the last
request. If you're seeing cold starts, the container idled out. First request
always pays the ~15 s warmup cost.

**`JAX_PLATFORMS=cpu` warning on Windows**
Expected — set in `ws_server.py` before any JAX import to prevent crashes
on machines without CUDA. Modal containers ignore it (they have CUDA JAX).
