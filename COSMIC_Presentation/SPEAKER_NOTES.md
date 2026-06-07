# C.O.S.M.I.C — Speaker Notes & Demo Script

**COmic Soundtrack MusIc Composer** — a system that turns a silent comic / manga / webtoon
page into an immersive, panel-by-panel audio experience.

Open `COSMIC_deck.html` in any browser. Navigate with **← / →** or **Space**, press **F** for
fullscreen, click the dots to jump. Works fully offline (no internet needed for the slides).

---

## ~4-minute talk track

**1 · Title (15s)**
> "We're C.O.S.M.I.C — the COmic Soundtrack MusIc Composer. We give silent comics a soundtrack,
> one panel at a time."

**2 · Problem (25s)**
> Hundreds of millions read manga and webtoons, but the medium is purely visual. Film and games
> get scores, ambience and voice acting — comics get nothing, and hand-producing audio per page
> doesn't scale. Existing tools treat a whole document as one blob and ignore the panel — the real
> unit of pacing.

**3 · Solution (25s)**
> Drop in a page. C.O.S.M.I.C **detects** the panels and reading order, **understands** each panel
> with vision, and **composes** a three-layer soundtrack — then plays it back like a motion comic.

**4 · Experience (25s)**
> The reader opens on a blank page, then each panel fades in as its audio plays and advances on its
> own. Timing is driven by the dialogue — a panel lasts as long as its spoken line plus a tail.
> Crucially, the first panel unlocks in seconds while the rest generate in the background.

**5 · Pipeline (30s)** — the architecture slide
> One async Python server orchestrates the whole thing: render the page, detect & crop panels,
> analyze each with Claude, then fire three generators in parallel — background, melody, voice —
> mix and stream them to the browser over a WebSocket.

**6 · Panel detection (25s)**
> The panel is our unit, so detection has to be bulletproof. We use kumiko — classic computer
> vision — first; it's instant and free. When borders disappear on splash or dark pages, we fall
> back to Claude, which maps panels onto a 10×10 grid. We always infer reading direction too —
> right-to-left for manga.

**7 · Scene understanding (25s)**
> A single Claude vision call returns a structured brief per panel: a music prompt, one of seven
> moods, the verbatim dialogue in reading order, and a matching voice. It's a typed schema, so every
> field drops straight into generation — no fragile parsing.

**8 · Three layers (25s)**
> Three models, three layers: Stable Audio 3 for the ambient bed, Magenta RT2 for a mood melody,
> and Suno for the spoken dialogue. They're independent — if a layer isn't available, the panel still
> reveals and plays whatever's ready. Graceful, never blocking.

**9 · Mood engine (20s)** — optional deep dive
> The bridge from "this feels tense" to actual notes is a hand-crafted motif library: seven moods,
> each a frame-accurate note sequence. It even exports real MIDI files — Ableton-ready.

**10 · Performance (20s)**
> Three models could mean a long wait. We run them on warm Modal T4 GPUs with cached embeddings and
> a persistent weight volume, overlap every job, and unlock the reader on the first finished panel.

**11 · Tech stack (15s)** — point, don't read.

**12 · Hard problems (20s)**
> Reading order, borderless panels, dialogue coming out *sung* instead of spoken, and latency — all
> real problems we hit and solved.

**13 · Roadmap (15s)**
> Full chapters with crossfades, onomatopoeia → SFX, per-character voices, and a hosted one-click app.

**14 · Close (10s)**
> "Detect, understand, compose. We gave comics a soundtrack. Let's hear it."

---

## Live demo checklist (if showing the running app)

1. Start Modal-backed GPUs deployed: `modal deploy webgenta/modal_magenta.py` and
   `modal deploy webgenta/modal_stability.py`.
2. Set keys in `webgenta/.env`: `ANTHROPIC_API_KEY`, `SUNO_API_KEY`.
3. Run `python frontend/model_server.py` → open `http://localhost:8766`.
4. Use a prepared page (e.g. `frontend/Jujutsu_Kaisen.pdf` or a `data/files/*.pdf`).
5. Upload → pick one page → Generate → let panel 1 reveal and play.
6. **Fallback if live fails:** narrate from these slides; the pipeline diagram (slide 5)
   carries the story on its own.

---

## One-liner for judges
> "C.O.S.M.I.C reads a comic the way you do — panel by panel, in the right order — and scores each
> panel on the fly with ambient music, a mood melody, and voiced dialogue."
