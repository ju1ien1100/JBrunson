# Immersive Comic Audio — panel-by-panel soundtrack

Turn a comic / manga / webtoon PDF into an immersive reader: upload a page, and
it detects the individual **panels**, generates a soundtrack **per panel**, then
plays the page back **panel by panel** — each panel fades in as its audio plays,
then advances to the next.

The main app is **`model_server.py`** (a web app at `http://localhost:8766`).

---

## How it works

```
 Upload 1 page ─▶ detect panels ─▶ for each panel: analyze ─▶ generate 3 audio layers
 (in the browser)   (vision)          (mood + dialogue)        (background · melody · voice)
                                                                          │
 Reader: blank page ─▶ reveal panel 1 + play its audio ─▶ panel 2 ─▶ … ◀──┘
```

Each panel gets up to three audio layers:

| Layer | Source | Needs |
|---|---|---|
| **Voice** (spoken dialogue) | Suno | `SUNO_API_KEY` |
| **Background** (ambient bed) | Stable Audio (on Modal GPU) | Modal access + deployed app |
| **Melody** (mood tune) | Magenta MRT2 (on Modal GPU) | Modal access + deployed app |

Panel detection and per-panel analysis (mood + dialogue) use Claude vision and
need an `ANTHROPIC_API_KEY`.

> **Minimum to see it work:** an `ANTHROPIC_API_KEY` (detection + analysis) and a
> `SUNO_API_KEY` (voice). Without Modal, the background and melody layers are
> simply silent — panels still reveal and play their voice. With Modal access,
> all three layers play.

---

## Setup

```bash
# from the repo root
python -m venv .venv && source .venv/bin/activate   # or your own venv
cd JBrunson/frontend
pip install -r requirements.txt
```

Provide API keys (never commit them). Either export them in your shell:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export SUNO_API_KEY="..."
```

…or copy `../webgenta/.env.example` to `../webgenta/.env` and fill it in —
`model_server.py` loads that file automatically. (`.env` is gitignored.)

For the **background + melody** layers you also need Modal access and the two
GPU apps deployed (see `../webgenta/modal_magenta.py` and
`../webgenta/modal_stability.py`):

```bash
pip install modal && python -m modal setup
modal deploy ../webgenta/modal_magenta.py
modal deploy ../webgenta/modal_stability.py
```

---

## Run it

```bash
python model_server.py          # no arguments
# → serves http://localhost:8766
```

Open **http://localhost:8766** and:

1. Drop in a PDF.
2. Pick a **single page** (panels are detected on that one page — kept to one
   page to keep generation cost down).
3. Hit **Generate Soundtrack** and wait for the panels to analyze + generate.
4. The reader opens on a **blank page**, then reveals each panel in turn while
   its audio plays.

### Reader timing
- A panel lasts **as long as its spoken dialogue, plus a 3-second tail**.
- A panel with **no dialogue** shows for a **default 5 seconds**.
- Background/melody play underneath but never extend a panel — the length is
  driven by the dialogue, not the music.
- A **Replay** button restarts the sequence at the end.

---

## Getting content to feed it

`data/download_webtoon.py` downloads one webtoon chapter from webtoons.com as a
PDF (panels included), ready to upload above:

```bash
python ../data/download_webtoon.py --title-no 95 --episode 2
python ../data/download_webtoon.py --name "Tower of God" --chapter 2
# → saved to data/files/<title>_ep<N>.pdf  (data/files/ is gitignored)
```

Use it only for personal/offline testing of content you're allowed to access —
it won't bypass logins, paywalls, or age gates.

---

## Extra tools (optional)

**`pdf_reader.py`** — an older *batch* pipeline (whole pages, not panels). It can
analyze pages to a `pages.json` and render audio via Modal, or stream pages to a
WebSocket server. Run `python pdf_reader.py --help` for the modes.

**`midi_library.py`** — the per-mood MIDI motifs, reusable on their own:

```python
from midi_library import MELODY_LIBRARY, melody_for_mood, export_midi
export_midi(MELODY_LIBRARY["action"], "action.mid")   # writes a real .mid file
```
```bash
python midi_library.py --out midi_out            # export every mood as .mid
```

Moods: `calm`, `tense`, `action`, `sad`, `mysterious`, `triumphant`, `neutral`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError` | `pip install -r requirements.txt` |
| Analysis/detection fails with an auth error | set `ANTHROPIC_API_KEY` (shell export or `webgenta/.env`) |
| No voice on any panel | set `SUNO_API_KEY`; note some dialogue can be flagged by Suno's content moderation |
| Background + melody always silent | expected without Modal — deploy the Modal apps and authenticate to enable them |
| A panel's voice won't play in the browser | the server transcodes Suno audio to MP3; confirm `imageio-ffmpeg` is installed |
| Can't reach the page | confirm `python model_server.py` is running and you're on `http://localhost:8766` |
