<div align="center">

<img src="COSMIC_Presentation/COMIC_secondcover.png" alt="C.O.S.M.I.C — COmic Soundtrack MusIc Composer" width="640" />

# C.O.S.M.I.C

### **CO**mic **S**oundtrack **M**us**I**c **C**omposer

*Upload a manga or webtoon — and hear it scored, panel by panel, in real time.*

🏆 **1st place — Suno Challenge**  ·  🥉 **3rd place — Google DeepMind Challenge**
Music Technology Hackathon by [Music Hackspace](https://www.musichackspace.org/), hosted at **Berklee College of Music** — out of 100+ developers, researchers, and musicians.

</div>

---

## What is it?

C.O.S.M.I.C turns a silent comic into an immersive audio-visual experience. Drop in a manga / webtoon / comic PDF, pick a page, and the app reads it the way you do — one panel at a time — generating an original soundtrack for each panel on the fly.

It doesn't play *one* sound. For every panel, **three AI models compose together**:

| Layer | What it does | Powered by |
|---|---|---|
| 🎚️ **Background** | The ambient bed — sets the scene and the mood of what's happening | **Stable Audio** (Stability AI) |
| 🎹 **Melody** | A playful mood melody on top — surprising lines you've never heard before | **Magenta RT** (Google DeepMind) |
| 🎤 **Voice** | Brings the characters' dialogue to life with real voices | **Suno** |

A page opens blank, then each panel **fades in as its music plays** and hands off to the next — so the soundtrack stays in step with your reading.

### Play along, in key 🎼

You're not just listening. **Pause on any panel, play the on-screen piano, and keep composing from there with Magenta RT.** The orchestration layer keeps everything — background, melody, and your own keystrokes — locked to the **same key, scale, and tempo**, so whatever you play always sounds in sync with the generated audio.

---

## How it works

A single server orchestrates the whole experience:

```
  Upload PDF ─▶ render page ─▶ detect panels ─▶ "read" each panel with Claude
                                (kumiko CV +      (mood · dialogue · key/scale/tempo)
                                 Claude fallback)            │
                                                             ▼
                          fire 3 generators in parallel, per panel
              ┌──────────────────────┬──────────────────────┬──────────────────────┐
        Stable Audio (background)   Magenta RT (melody)   Suno (voice)
              └──────────────────────┴───────────┬──────────┴──────────────────────┘
                                                  ▼
                          mix + stream to the browser over WebSocket
                          (cloud GPU → client in ~500 ms)
```

- **Accessible by design.** Limited by local hardware, we moved the heavy models to **cloud GPUs (Modal)** and stream responses back to the browser — getting end-to-end latency down to **~500 ms**, low enough that the music feels live.
- **Vision-driven.** Each panel is analyzed with **Claude** to infer mood, dialogue, and a musical key / scale / tempo, which condition all three generators so the layers agree musically.
- **In sync.** Generated audio is transposed and tempo-matched through the orchestration layer — and so is the live MIDI piano at the bottom of the reader.

---

## Tech stack

- **AI / audio:** Magenta RealTime 2 (Google DeepMind), Stable Audio (Stability AI), Suno, Claude (vision analysis)
- **Infra:** [Modal](https://modal.com) (warm T4 GPUs), WebSocket streaming
- **Panel detection:** [kumiko](https://github.com/njean42/kumiko) (computer vision) with a Claude grid-based fallback
- **Backend:** Python · aiohttp · PyMuPDF (page rendering)
- **Frontend:** Vite + TypeScript · Web Audio `AudioWorklet` · Web MIDI

---

## Quick start

**Prerequisites:** Python 3.11+, and API keys for the services you want to use.

```bash
# 1. install dependencies
cd frontend
pip install -r requirements.txt

# 2. add your keys (never commit them)
cp ../webgenta/.env.example ../webgenta/.env
#   ANTHROPIC_API_KEY=...   ← panel detection + analysis
#   SUNO_API_KEY=...        ← character voices
#   STABILITY_API_KEY=...   ← background audio
```

**Run the comic reader:**

```bash
# from the repo root
bash start.sh            # macOS / Linux
#  or on Windows:  .\start.ps1
```

Then open **http://localhost:8766**, drop in a PDF, pick a page, and hit **Generate Soundtrack**.

> **Minimum to see it work:** `ANTHROPIC_API_KEY` (detection + analysis) + `SUNO_API_KEY` (voice). The **background** and **melody** layers run on Modal GPUs — without Modal access they're simply silent while the rest still plays. See [`webgenta/README.md`](webgenta/README.md) for the full Modal + MRT2 setup and the live play-along studio.

**Full live experience (comic reader + Magenta RT play-along studio):**

```powershell
.\start-comic.ps1        # launches both servers, opens http://localhost:5173/comic.html
```

---

## Repository structure

```
JBrunson/
├── frontend/                  C.O.S.M.I.C comic reader (the main app)
│   ├── model_server.py        HTTP + WebSocket server · orchestrates the pipeline
│   ├── pdf_reader.py          Batch pipeline (whole-page analysis) — optional
│   ├── midi_library.py        Per-mood MIDI motifs + key/tempo transforms
│   └── requirements.txt
│
├── webgenta/                  Real-time Magenta RT studio + service layer
│   ├── ws_server.py           WebSocket router
│   ├── modal_magenta.py       Modal GPU deployment (MRT2 inference)
│   ├── modal_stability.py     Modal GPU deployment (Stable Audio)
│   ├── services/              magenta · suno · stability integrations
│   └── web/                   Vite + TypeScript frontend (comic.html, main.ts, worklet)
│
├── data/
│   └── download_webtoon.py    Fetch a webtoon chapter as a PDF to feed the reader
│
└── COSMIC_Presentation/       Hackathon deck, speaker notes, and cover art
```

---

## The hackathon

Built in **48 hours by a team of two** at the Music Technology Hackathon ("Build the Future of Creative Tools") by Music Hackspace, hosted at Berklee College of Music.

A huge thank-you to the **Google DeepMind Magenta mentors** — and especially **Ilaria Manco**, whose guidance on Magenta's parameters was a big reason the real-time generation came together — to the **jury** who pushed every team to think past the weekend, and to **Music Hackspace** and **Berklee** for creating a room where musicians led and engineers followed.

---

## License & use

Built for a hackathon and shared for learning. Third-party models and APIs (Magenta, Stable Audio, Suno, Claude) are governed by their own terms. The webtoon downloader is for personal/offline testing of content you're allowed to access only — it does not bypass logins, paywalls, or age gates.
