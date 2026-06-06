# Immersive Audio — PDF → mood → Magenta pipeline

Turns a comic/manga/book PDF into per-page background music. Each page is read
by a vision model that picks a **mood** + a **music style prompt**; that drives
a curated MIDI melody + Magenta (MRT2) to render an audio clip per page.

It runs as **two stages** so the API key and the GPU don't need to live on the
same machine:

```
 Stage 1  (--analyze)                      Stage 2  (--modal)
 ┌─────────────────────────┐  pages.json   ┌──────────────────────────────┐
 │ Claude vision reads each │ ───────────▶ │ mood → MIDI motif → Modal GPU │
 │ page → mood + prompt     │              │ (Magenta) → one WAV per page  │
 └─────────────────────────┘              └──────────────────────────────┘
   needs ANTHROPIC_API_KEY                   needs Modal auth + deployed app
```

`pages.json` is the handoff file between the two stages.

---

## Setup (both stages)

```bash
# from the repo root
source ../music_hack/bin/activate        # or your own venv
cd JBrunson/frontend
pip install -r requirements.txt
```

`requirements.txt` is grouped by mode: `pymupdf`/`websockets` (core),
`anthropic` (stage 1), `modal`+`numpy` (stage 2).

---

## Stage 1 — analyze (needs `ANTHROPIC_API_KEY`)

Reads page images + text and writes `pages.json` (mood + style prompt per page).

```bash
export ANTHROPIC_API_KEY="sk-ant-..."        # in your shell only — never commit it

# first page only (cheapest smoke test)
python pdf_reader.py Jujutsu_Kaisen.pdf --analyze --pages 1

# specific pages (1-based): single, list, range, or mix
python pdf_reader.py Jujutsu_Kaisen.pdf --analyze --page-range 7-9
python pdf_reader.py Jujutsu_Kaisen.pdf --analyze --page-range 1,3,5-7

# add pages to an existing pages.json instead of overwriting it
python pdf_reader.py Jujutsu_Kaisen.pdf --analyze --page-range 7-9 --append

# whole PDF
python pdf_reader.py Jujutsu_Kaisen.pdf --analyze
```

Options: `--pages N` (first N), `--page-range SPEC` (specific pages, overrides
`--pages`), `--append` (merge into existing `pages.json`), `--out FILE`
(default `pages.json`), `--vision-model` (default `claude-opus-4-8`).

---

## Stage 2 — render on Modal (needs Modal auth) ← teammate runs this

Reads `pages.json`, maps each page's mood to a MIDI motif, calls the deployed
`webgenta-magenta` Modal app, and writes one WAV per page.

```bash
# one-time: Modal auth + deployed app (see ../webgenta/modal_magenta.py)
#   pip install modal && python -m modal setup
#   modal run ../webgenta/modal_magenta.py::download_checkpoints
#   modal deploy ../webgenta/modal_magenta.py

# render everything in pages.json
python pdf_reader.py --modal --pages-json pages.json --out test_output

# render specific pages only (each page = one GPU render = cost/time)
python pdf_reader.py --modal --pages-json pages.json --page-range 7-9 --out test_output
```

Output: `test_output/page_07_tense.wav`, etc. (named `page_<n>_<mood>.wav`).
Options: `--page-range SPEC`, `--pages N`, `--app` / `--cls` (Modal app/class
names, default `webgenta-magenta` / `MagentaInference`).

---

## `pages.json` format

```json
{
  "pdf": "Jujutsu_Kaisen.pdf",
  "pages": [
    {
      "page_number": 1,
      "total_pages": 11,
      "mood": "tense",
      "style_prompt": "dark ominous orchestral strings, building tension",
      "reason": "why the model chose this mood"
    }
  ]
}
```

Moods: `calm`, `tense`, `action`, `sad`, `mysterious`, `triumphant`, `neutral`.
You can hand-edit `mood`/`style_prompt` before stage 2 to tweak the result.

---

## Reusable MIDI library (`midi_library.py`)

The per-mood melodies live in `midi_library.py` and can be reused independently:

```python
from midi_library import MELODY_LIBRARY, melody_for_mood, export_midi
melody_for_mood("tense")                 # MRT2 render() note segments
export_midi(MELODY_LIBRARY["action"], "action.mid")   # real .mid file
```

```bash
python midi_library.py --out midi_out          # export all moods as .mid
python midi_library.py --out midi_out --mood sad
```

---

## Third mode: live WebSocket streaming (unrelated to the two stages)

`python pdf_reader.py comic.pdf` streams pages to a local WebSocket server
(`model_server.py` on `:8765`) — used for the interactive/streaming path, not
the Modal render.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| `ModuleNotFoundError: anthropic` | `pip install -r requirements.txt` (stage 1) |
| `ModuleNotFoundError: modal` / `numpy` | `pip install -r requirements.txt` (stage 2) |
| stage 1 auth error | `export ANTHROPIC_API_KEY=...` in the same shell |
| stage 2 can't find the app | confirm `modal app list` shows `webgenta-magenta` and you're in the right workspace |
| client deadlocks in WS mode | the server always replies (even on error) — check `model_server.py` is running |
