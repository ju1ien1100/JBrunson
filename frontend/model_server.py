"""Comic book audio server.

Serves HTTP + WebSocket on port 8766. No CLI arguments needed.

Routes:
  GET  /        → comic.html (the reader UI)
  POST /peek    → multipart: pdf → returns {total_pages}  (no processing)
  POST /upload  → multipart: pdf, page_from, page_to
                  Renders pages, runs Claude analysis, fires audio generation.
                  Returns {total_pages, selected_pages, status}.
  GET  /ws      → WebSocket: progress events + reader protocol

WebSocket events (server → browser):
  {"type": "status",   "status": "idle|analyzing|generating|ready|error"}
  {"type": "analyzed", "pdf_page": N, "mood": "..."}
  {"type": "progress", "pdf_page": N, "track": "stable|magenta|suno", "done": true}
  {"type": "page",     "page_number": N, "total": T, "image_b64": "..."}
  {"type": "track_header", "track": "background|melody|voice", "mime": "..."}
  <binary WAV/MP3 bytes>
  {"type": "tracks_end"}

WebSocket messages (browser → server):
  {"type": "get_page", "page": N}   — N is 1-based reader index into selected pages
"""

from __future__ import annotations

import asyncio
import base64
import io
import uuid
import json
import os
import sys
import tempfile
import wave
from pathlib import Path
from typing import Literal

import aiohttp
import fitz  # PyMuPDF
import numpy as np
from aiohttp import web

sys.path.insert(0, str(Path(__file__).resolve().parent))
from midi_library import melody_for_mood  # noqa: E402

# Load .env so API keys are available regardless of how the script was launched
_ENV_FILE = Path(__file__).parent.parent / "webgenta" / ".env"
if _ENV_FILE.exists():
    for _line in _ENV_FILE.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _, _v = _line.partition("=")
            os.environ.setdefault(_k.strip(), _v.strip())

PORT = 8766
COMIC_HTML = Path(__file__).parent.parent / "webgenta" / "web" / "comic.html"
MAGENTA_SAMPLE_RATE = 48000
MAGENTA_CHANNELS = 2
STABLE_AUDIO_DURATION = 30
SUNO_BASE = "https://api.suno.com"
SUNO_POLL_INTERVAL = 3.0
SUNO_POLL_TIMEOUT = 180.0
ANIME_VOICE_STYLE = (
    "spoken word, solo, voice only, no instruments, no background music, no beat, "
    "spoken word, anime voice, clear speech, dry vocal"
)
# Suno preset voice IDs (only these three are supported)
VOICE_FEMALE  = "5b915c6d-8d96-416c-9755-eba65868cfef"  # female voice
VOICE_KID     = "c036ce3a-55e4-4690-9b8d-4516b37a96d5"  # weird kid voice
VOICE_MALE    = "27f5465b-73c3-4134-b11e-70b0bd571c6c"  # low male voice
VOICE_DEFAULT = VOICE_FEMALE
VISION_MODEL = "claude-haiku-4-5"
PANEL_VISION_MODEL = "claude-sonnet-4-6"  # higher-capability model for panel boundary detection
FIRST_BATCH = 3  # pages needed before reader unlocks; rest generate in background
VISION_SYSTEM = (
    "You score a single comic/manga/book page for a three-layer audio engine. "
    "Look at the art and any text, then produce all five fields:\n"
    "READING ORDER: First infer the layout direction. Manga and Japanese comics "
    "are read RIGHT-TO-LEFT, top-to-bottom — start at the top-right panel/bubble "
    "and move leftward, then down. Western comics are read left-to-right. Use the "
    "art, panel layout, and language to decide, and transcribe panels and speech "
    "bubbles in that correct reading order.\n"
    "stable_audio_prompt: Rich music description for a background track (genre, "
    "instrumentation, mood). Under 200 characters.\n"
    "magenta_mood: Single mood from the allowed set for the melody overlay.\n"
    "suno_lyrics: Dialogue extracted verbatim from speech bubbles, in reading "
    "order, newline-separated. Preserve the original language. Empty string if no "
    "readable dialogue.\n"
    "suno_voice_id: Choose the voice that best matches the dominant speaking "
    "character on this panel. Options:\n"
    f"  {VOICE_FEMALE} — female voice (default for women, girls, feminine characters)\n"
    f"  {VOICE_KID}    — weird kid voice (children, comic-relief, young/quirky characters)\n"
    f"  {VOICE_MALE}   — low male voice (men, older characters, deep/authoritative voices)\n"
    "Pick whichever fits the speaker. If there is no dialogue, still return a voice_id.\n"
    "reason: Brief justification."
)
MOODS = ("calm", "tense", "action", "sad", "mysterious", "triumphant", "neutral")

PANEL_SYSTEM = (
    "You are a comic/manga panel detector. Look at one page image and find every "
    "distinct panel (a framed/bordered illustration cell, or for vertical webtoons, "
    "each block separated by gaps). Read manga/Japanese RIGHT-TO-LEFT top-to-bottom, "
    "Western comics left-to-right, webtoon strips top-to-bottom. Return panel_count "
    "and one box per panel IN READING ORDER. Boxes are [x0,y0,x1,y1] as fractions of "
    "width/height in [0,1]. Ignore margins, gutters, logos, and page numbers."
)


def _detect_panels(page_png: bytes) -> list[list[float]]:
    """Claude-vision panel detection -> list of [x0,y0,x1,y1] boxes in reading order.

    Falls back to a single whole-page 'panel' if nothing is detected.
    """
    import anthropic
    from pydantic import BaseModel

    class _Panel(BaseModel):
        order: int
        box: list[float]

    class _Panels(BaseModel):
        panel_count: int
        panels: list[_Panel]

    img_b64 = base64.b64encode(page_png).decode("ascii")
    client = anthropic.Anthropic()
    resp = client.messages.parse(
        model=PANEL_VISION_MODEL,
        max_tokens=2048,
        system=PANEL_SYSTEM,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {"type": "base64", "media_type": "image/png", "data": img_b64}},
                {"type": "text", "text": "Detect and count the panels on this page."},
            ],
        }],
        output_format=_Panels,
    )
    ordered = sorted(resp.parsed_output.panels, key=lambda p: p.order)
    boxes = []
    for p in ordered:
        x0, y0, x1, y1 = p.box
        # clamp + guard against degenerate boxes
        x0, y0 = max(0.0, min(x0, 1.0)), max(0.0, min(y0, 1.0))
        x1, y1 = max(0.0, min(x1, 1.0)), max(0.0, min(y1, 1.0))
        if x1 - x0 > 0.02 and y1 - y0 > 0.02:
            boxes.append([x0, y0, x1, y1])
    return boxes or [[0.0, 0.0, 1.0, 1.0]]


def _crop_png(page_png: bytes, box: list[float]) -> bytes:
    """Crop a normalized box out of a rendered page PNG, returning PNG bytes."""
    src = fitz.open("png", page_png)
    pg = src[0]
    r = pg.rect
    x0, y0, x1, y1 = box
    clip = fitz.Rect(x0 * r.width, y0 * r.height, x1 * r.width, y1 * r.height)
    out = pg.get_pixmap(clip=clip).tobytes("png")
    src.close()
    return out

# ── Modal handles (set at startup if available) ───────────────────────────────
_stable_cls = None
_magenta_cls = None

# ── Global state (reset on each /upload) ─────────────────────────────────────
_state: dict = {
    "status": "idle",
    "total_pdf_pages": 0,
    "pdf_page_list": [],        # sorted PDF page numbers (subset selected by user)
    "page_images": {},          # pdf_page_num -> PNG bytes
    "page_data": {},            # pdf_page_num -> analysis dict
    "stable_wavs": {},
    "magenta_wavs": {},
    "suno_wavs": {},
    "gen_tasks": [],
    "pipeline_task": None,      # the pipeline Task itself
    "progress": {},             # unit -> {analyzed, stable, magenta, suno}
    "error": None,
    # Panel mode: the single uploaded page + its detected panels. Panels are the
    # generation units, keyed 1..K in page_images / page_data / progress.
    "full_page_png": b"",
    "panel_boxes": [],
}

# Live WebSocket connections (for progress broadcasts)
_ws_clients: set[web.WebSocketResponse] = set()


# ── PCM → WAV ─────────────────────────────────────────────────────────────────

def _pcm32_to_wav(pcm_bytes: bytes) -> bytes:
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    pcm_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(MAGENTA_CHANNELS)
        w.setsampwidth(2)
        w.setframerate(MAGENTA_SAMPLE_RATE)
        w.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


def _to_mp3(audio_bytes: bytes) -> bytes | None:
    """Transcode arbitrary audio bytes to MP3 (browser-playable).

    Returns None if imageio_ffmpeg is not installed (voice track is skipped).
    Blocking — call via run_in_executor from async code.
    """
    import subprocess

    try:
        import imageio_ffmpeg
    except ImportError:
        return None

    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    proc = subprocess.run(
        [ffmpeg, "-loglevel", "error", "-i", "pipe:0",
         "-f", "mp3", "-c:a", "libmp3lame", "-b:a", "128k", "pipe:1"],
        input=audio_bytes, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    if proc.returncode != 0 or not proc.stdout:
        raise RuntimeError(f"ffmpeg transcode failed: {proc.stderr.decode()[:300]}")
    return proc.stdout


# ── Progress broadcast ────────────────────────────────────────────────────────

async def _broadcast(msg: dict) -> None:
    dead = set()
    for ws in _ws_clients:
        try:
            await ws.send_json(msg)
        except Exception:
            dead.add(ws)
    _ws_clients.difference_update(dead)


# ── Audio generation tasks ────────────────────────────────────────────────────

async def _gen_stable(pdf_n: int) -> None:
    if _stable_cls is None:
        _state["stable_wavs"][pdf_n] = b""
    else:
        prompt = _state["page_data"][pdf_n].get("stable_audio_prompt", "ambient music")
        print(f"[p{pdf_n}] stable: {prompt[:70]!r}…", flush=True)
        try:
            wav = await _stable_cls.generate.remote.aio(prompt, STABLE_AUDIO_DURATION)
            _state["stable_wavs"][pdf_n] = wav
            print(f"[p{pdf_n}] stable done ({len(wav)//1024} KB)", flush=True)
        except Exception as e:
            print(f"[p{pdf_n}] stable ERROR: {e}", flush=True)
            _state["stable_wavs"][pdf_n] = b""
    _state["progress"][pdf_n]["stable"] = True
    await _broadcast({"type": "progress", "pdf_page": pdf_n, "track": "stable", "done": True})


async def _gen_magenta(pdf_n: int) -> None:
    if _magenta_cls is None:
        _state["magenta_wavs"][pdf_n] = b""
    else:
        mood = _state["page_data"][pdf_n].get("magenta_mood", "neutral")
        style = _state["page_data"][pdf_n].get("stable_audio_prompt", "ambient music")
        print(f"[p{pdf_n}] magenta: mood={mood!r}", flush=True)
        session_id = f"panel_{pdf_n}_{uuid.uuid4().hex[:8]}"
        try:
            notes_segs = melody_for_mood(mood)
            await _magenta_cls.begin_session.remote.aio(session_id, style)
            pcm = await _magenta_cls.render_melody.remote.aio(session_id, notes_segs)
            _state["magenta_wavs"][pdf_n] = _pcm32_to_wav(pcm)
            print(f"[p{pdf_n}] magenta done ({len(_state['magenta_wavs'][pdf_n])//1024} KB)", flush=True)
        except Exception as e:
            print(f"[p{pdf_n}] magenta ERROR: {e}", flush=True)
            _state["magenta_wavs"][pdf_n] = b""
        finally:
            try:
                await _magenta_cls.end_session.remote.aio(session_id)
            except Exception:
                pass
    _state["progress"][pdf_n]["magenta"] = True
    await _broadcast({"type": "progress", "pdf_page": pdf_n, "track": "magenta", "done": True})


async def _gen_suno(pdf_n: int) -> None:
    lyrics = _state["page_data"][pdf_n].get("suno_lyrics", "").strip()
    api_key = os.environ.get("SUNO_API_KEY", "")

    if not lyrics or not api_key:
        reason = "no dialogue" if not lyrics else "SUNO_API_KEY not set"
        print(f"[p{pdf_n}] suno: skipped ({reason})", flush=True)
        _state["suno_wavs"][pdf_n] = None
        _state["progress"][pdf_n]["suno"] = True
        await _broadcast({"type": "progress", "pdf_page": pdf_n, "track": "suno", "done": True})
        return

    voice_id = _state["page_data"][pdf_n].get("suno_voice_id", VOICE_DEFAULT)
    print(f"[p{pdf_n}] suno: submitting voice track (voice={voice_id[:8]}…)…", flush=True)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"style": ANIME_VOICE_STYLE, "lyrics": lyrics, "voice_id": voice_id}

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(f"{SUNO_BASE}/v0/audio", json=payload, headers=headers) as resp:
                data = await resp.json()
                if resp.status not in (200, 201, 202):
                    print(f"[p{pdf_n}] suno submit error {resp.status}: {data}", flush=True)
                    _state["suno_wavs"][pdf_n] = None
                    _state["progress"][pdf_n]["suno"] = True
                    await _broadcast({"type": "progress", "pdf_page": pdf_n, "track": "suno", "done": True})
                    return

            job_id = data.get("id", "")
            loop = asyncio.get_running_loop()
            deadline = loop.time() + SUNO_POLL_TIMEOUT
            audio_url = None

            while loop.time() < deadline:
                await asyncio.sleep(SUNO_POLL_INTERVAL)
                async with session.get(f"{SUNO_BASE}/v0/audio/{job_id}", headers=headers) as r:
                    sd = await r.json()
                s = sd.get("status", "unknown")
                if s == "complete":
                    audio_url = sd.get("audio_url", "")
                    break
                elif s == "error":
                    print(f"[p{pdf_n}] suno job error: {sd.get('error')}", flush=True)
                    _state["suno_wavs"][pdf_n] = None
                    _state["progress"][pdf_n]["suno"] = True
                    await _broadcast({"type": "progress", "pdf_page": pdf_n, "track": "suno", "done": True})
                    return

            if audio_url:
                async with session.get(audio_url) as r:
                    raw_audio = await r.read()
                try:
                    mp3 = await loop.run_in_executor(None, _to_mp3, raw_audio)
                    if mp3 is None:
                        print(f"[p{pdf_n}] suno: imageio_ffmpeg not installed, skipping voice", flush=True)
                        _state["suno_wavs"][pdf_n] = None
                    else:
                        _state["suno_wavs"][pdf_n] = mp3
                        print(f"[p{pdf_n}] suno done ({len(mp3)//1024} KB MP3)", flush=True)
                except Exception as e:
                    print(f"[p{pdf_n}] suno transcode ERROR: {e}", flush=True)
                    _state["suno_wavs"][pdf_n] = None
            else:
                print(f"[p{pdf_n}] suno: timed out", flush=True)
                _state["suno_wavs"][pdf_n] = None

    except Exception as e:
        print(f"[p{pdf_n}] suno ERROR: {e}", flush=True)
        _state["suno_wavs"][pdf_n] = None

    _state["progress"][pdf_n]["suno"] = True
    await _broadcast({"type": "progress", "pdf_page": pdf_n, "track": "suno", "done": True})


async def _wait_for_page(pdf_n: int) -> None:
    # Wait only for stable audio — melody/voice served if available, skipped if not.
    # This avoids blocking on the cold-start container or slow magenta/suno runs.
    while pdf_n not in _state["stable_wavs"]:
        await asyncio.sleep(0.2)


# ── Analysis + generation pipeline ───────────────────────────────────────────

async def _analyze_and_generate(pdf_page_nums: list[int]) -> None:
    import anthropic
    from pydantic import BaseModel

    _VALID_VOICE_IDS = {VOICE_FEMALE, VOICE_KID, VOICE_MALE}

    class PageMusic(BaseModel):
        stable_audio_prompt: str
        magenta_mood: Literal[
            "calm", "tense", "action", "sad", "mysterious", "triumphant", "neutral"
        ]
        suno_lyrics: str
        suno_voice_id: str
        reason: str

    client = anthropic.Anthropic()

    def _analyze_one(pnum: int) -> dict:
        img_b64 = base64.b64encode(_state["page_images"][pnum]).decode("ascii")
        resp = client.messages.parse(
            model=VISION_MODEL,
            max_tokens=1024,
            system=VISION_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": img_b64},
                    },
                    {"type": "text", "text": f"Allowed moods: {', '.join(MOODS)}."},
                ],
            }],
            output_format=PageMusic,
        )
        pm = resp.parsed_output
        voice_id = pm.suno_voice_id if pm.suno_voice_id in _VALID_VOICE_IDS else VOICE_DEFAULT
        return {
            "page_number": pnum,
            "stable_audio_prompt": pm.stable_audio_prompt.strip() or "ambient background music",
            "magenta_mood": pm.magenta_mood,
            "suno_lyrics": pm.suno_lyrics.strip(),
            "suno_voice_id": voice_id,
            "reason": pm.reason,
        }

    _state["status"] = "analyzing"
    await _broadcast({"type": "status", "status": "analyzing", "total": len(pdf_page_nums)})

    loop = asyncio.get_running_loop()
    first_batch = pdf_page_nums[:FIRST_BATCH]
    reader_unlocked = False

    # Analyze pages one at a time; fire generation tasks immediately after each so
    # Modal GPU work overlaps with the remaining Claude analysis calls.
    for pnum in pdf_page_nums:
        try:
            result = await loop.run_in_executor(None, _analyze_one, pnum)
        except asyncio.CancelledError:
            raise  # re-upload cancelled this task — exit cleanly
        except Exception as exc:
            print(f"[p{pnum}] analysis ERROR: {exc}", flush=True)
            _state["status"] = "error"
            _state["error"] = str(exc)
            await _broadcast({"type": "status", "status": "error", "message": str(exc)})
            return
        _state["page_data"][pnum] = result
        _state["progress"][pnum]["analyzed"] = True
        _state["pdf_page_list"] = sorted(_state["page_data"].keys())
        print(f"[p{pnum}] analyzed: mood={result['magenta_mood']!r}", flush=True)
        await _broadcast({
            "type": "analyzed",
            "pdf_page": pnum,
            "mood": result["magenta_mood"],
            "has_lyrics": bool(result["suno_lyrics"]),
        })

        # Immediately queue generation for this page
        for coro in (_gen_stable(pnum), _gen_magenta(pnum), _gen_suno(pnum)):
            task = asyncio.create_task(coro)
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            _state["gen_tasks"].append(task)

    _state["status"] = "generating"
    await _broadcast({"type": "status", "status": "generating"})

    # Poll until first FIRST_BATCH pages are fully generated, then unlock the reader.
    # The remaining pages keep generating in the background.
    async def _watch_first_batch() -> None:
        nonlocal reader_unlocked
        while not reader_unlocked:
            await asyncio.sleep(0.5)
            # Unlock as soon as ANY panel has stable audio ready — avoids
            # blocking on the cold-start container (which hits the first call).
            if any(
                _state["progress"].get(n, {}).get("stable")
                for n in pdf_page_nums
            ):
                reader_unlocked = True
                _state["status"] = "reader_ready"
                await _broadcast({"type": "status", "status": "reader_ready"})
                print("Stable audio ready on first panel — reader unlocked.", flush=True)

    watch_task = asyncio.create_task(_watch_first_batch())

    await asyncio.gather(*_state["gen_tasks"], return_exceptions=True)
    watch_task.cancel()

    if not reader_unlocked:
        # Fewer than FIRST_BATCH pages selected — just unlock immediately
        _state["status"] = "reader_ready"
        await _broadcast({"type": "status", "status": "reader_ready"})

    _state["status"] = "ready"
    await _broadcast({"type": "status", "status": "ready"})
    print("All tracks ready.", flush=True)


# ── Panel pipeline (single page → detect panels → per-panel generation) ──────

async def _panel_pipeline(page_png: bytes) -> None:
    """Detect panels on the single uploaded page, crop them, then run the normal
    per-unit analysis + audio generation with each PANEL as a unit."""
    loop = asyncio.get_running_loop()
    _state["status"] = "detecting"
    await _broadcast({"type": "status", "status": "detecting"})

    try:
        boxes = await loop.run_in_executor(None, _detect_panels, page_png)
    except asyncio.CancelledError:
        raise
    except Exception as exc:
        print(f"panel detection ERROR: {exc}", flush=True)
        _state["status"] = "error"
        _state["error"] = str(exc)
        await _broadcast({"type": "status", "status": "error", "message": str(exc)})
        return

    # Crop each panel; panels become units 1..K
    page_images = {}
    for i, box in enumerate(boxes, 1):
        page_images[i] = _crop_png(page_png, box)
    _state["panel_boxes"] = boxes
    _state["page_images"] = page_images
    _state["progress"] = {
        i: {"analyzed": False, "stable": False, "magenta": False, "suno": False}
        for i in range(1, len(boxes) + 1)
    }
    print(f"Detected {len(boxes)} panels.", flush=True)
    await _broadcast({"type": "panels_detected", "count": len(boxes)})

    await _analyze_and_generate(list(range(1, len(boxes) + 1)))


# ── HTTP handlers ─────────────────────────────────────────────────────────────

async def handle_index(request: web.Request) -> web.FileResponse:
    return web.FileResponse(COMIC_HTML)


async def handle_peek(request: web.Request) -> web.Response:
    """Return page count for a PDF without starting any processing."""
    reader = await request.multipart()
    pdf_bytes = None
    async for part in reader:
        if part.name == "pdf":
            pdf_bytes = await part.read()
            break
    if not pdf_bytes:
        return web.json_response({"error": "No PDF"}, status=400)
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = len(doc)
    return web.json_response({"total_pages": total})


async def handle_upload(request: web.Request) -> web.Response:
    """Accept a PDF + a SINGLE page number; detect panels on that page and run
    per-panel generation. (To keep cost down we process exactly one page.)"""
    reader = await request.multipart()
    pdf_bytes = None
    page = 1

    async for part in reader:
        if part.name == "pdf":
            pdf_bytes = await part.read()
        elif part.name in ("page", "page_from"):  # accept either name
            txt = await part.text()
            if txt.strip():
                page = int(txt)

    if not pdf_bytes:
        return web.json_response({"error": "No PDF file received"}, status=400)

    # Render exactly the one requested page
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = len(doc)
        page = max(1, min(page, total))
        full_page_png = doc[page - 1].get_pixmap(dpi=150).tobytes("png")

    # Cancel any previous pipeline + gen tasks
    if _state.get("pipeline_task") and not _state["pipeline_task"].done():
        _state["pipeline_task"].cancel()
    for task in _state.get("gen_tasks", []):
        task.cancel()

    _state.update({
        "status": "detecting",
        "total_pdf_pages": total,
        "pdf_page_list": [],
        "page_images": {},
        "page_data": {},
        "stable_wavs": {},
        "magenta_wavs": {},
        "suno_wavs": {},
        "gen_tasks": [],
        "pipeline_task": None,
        "progress": {},
        "error": None,
        "full_page_png": full_page_png,
        "panel_boxes": [],
    })

    pipeline = asyncio.create_task(_panel_pipeline(full_page_png))
    pipeline.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
    _state["pipeline_task"] = pipeline

    return web.json_response({
        "total_pages": total,
        "page": page,
        "status": "detecting",
    })


async def handle_status(request: web.Request) -> web.Response:
    return web.json_response({
        "status": _state["status"],
        "total_pdf_pages": _state["total_pdf_pages"],
        "pdf_page_list": _state["pdf_page_list"],
        "progress": _state["progress"],
    })


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def handle_ws(request: web.Request) -> web.WebSocketResponse:
    ws = web.WebSocketResponse(max_msg_size=100 * 1024 * 1024)
    await ws.prepare(request)
    _ws_clients.add(ws)

    # Immediately sync current state to newly connected client
    await ws.send_json({
        "type": "status",
        "status": _state["status"],
        "pdf_page_list": _state["pdf_page_list"],
        "total_pdf_pages": _state["total_pdf_pages"],
        "progress": _state["progress"],
    })

    try:
        async for msg in ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    continue

                if data.get("type") == "panel_meta":
                    full = base64.b64encode(_state.get("full_page_png", b"")).decode("ascii")
                    await ws.send_json({
                        "type": "panel_meta",
                        "image_b64": full,
                        "panels": [{"order": i + 1, "box": b}
                                   for i, b in enumerate(_state.get("panel_boxes", []))],
                        "total": len(_state.get("panel_boxes", [])),
                    })
                    continue

                if data.get("type") == "get_page":
                    reader_n = int(data.get("page", 1))
                    pdf_page_list = _state["pdf_page_list"]
                    idx = reader_n - 1

                    if idx < 0 or idx >= len(pdf_page_list):
                        await ws.send_json({
                            "type": "error",
                            "message": f"Page {reader_n} out of range (1–{len(pdf_page_list)})",
                        })
                        continue

                    pdf_n = pdf_page_list[idx]
                    total = len(pdf_page_list)
                    print(f"Browser → reader page {reader_n} (PDF p{pdf_n})", flush=True)

                    await _wait_for_page(pdf_n)

                    img_b64 = base64.b64encode(
                        _state["page_images"].get(pdf_n, b"")
                    ).decode("ascii")
                    await ws.send_json({
                        "type": "page",
                        "page_number": reader_n,
                        "total": total,
                        "image_b64": img_b64,
                    })

                    bg = _state["stable_wavs"].get(pdf_n, b"")
                    if bg:
                        await ws.send_json({
                            "type": "track_header", "track": "background", "mime": "audio/wav"
                        })
                        await ws.send_bytes(bg)

                    mel = _state["magenta_wavs"].get(pdf_n, b"")
                    if mel:
                        await ws.send_json({
                            "type": "track_header", "track": "melody", "mime": "audio/wav"
                        })
                        await ws.send_bytes(mel)

                    voice = _state["suno_wavs"].get(pdf_n)
                    if voice:
                        await ws.send_json({
                            "type": "track_header", "track": "voice", "mime": "audio/mpeg"
                        })
                        await ws.send_bytes(voice)

                    await ws.send_json({"type": "tracks_end"})

            elif msg.type in (aiohttp.WSMsgType.ERROR, aiohttp.WSMsgType.CLOSE):
                break
    finally:
        _ws_clients.discard(ws)

    return ws


# ── CORS middleware ───────────────────────────────────────────────────────────

@web.middleware
async def cors_middleware(request: web.Request, handler):
    if request.method == "OPTIONS":
        resp = web.Response()
    else:
        try:
            resp = await handler(request)
        except web.HTTPException as ex:
            resp = ex
    resp.headers["Access-Control-Allow-Origin"] = "*"
    resp.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    resp.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return resp


async def handle_options(request: web.Request) -> web.Response:
    return web.Response()


# ── App factory ───────────────────────────────────────────────────────────────

def make_app() -> web.Application:
    app = web.Application(client_max_size=200 * 1024 * 1024, middlewares=[cors_middleware])
    app.router.add_get("/", handle_index)
    app.router.add_post("/peek", handle_peek)
    app.router.add_post("/upload", handle_upload)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/ws", handle_ws)
    # Preflight OPTIONS for CORS
    app.router.add_route("OPTIONS", "/peek", handle_options)
    app.router.add_route("OPTIONS", "/upload", handle_options)
    app.router.add_route("OPTIONS", "/status", handle_options)
    return app


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Comic book audio server")
    parser.add_argument("--no-stable", action="store_true",
                        help="Skip Stable Audio 3 (run without Modal GPU)")
    parser.add_argument("--no-magenta", action="store_true",
                        help="Skip Magenta MRT2 (run without Modal GPU)")
    args = parser.parse_args()

    import modal

    if not args.no_stable:
        print("Connecting to Modal — StableAudioInference…", flush=True)
        _StableCls = modal.Cls.from_name("webgenta-stability", "StableAudioInference")
        _stable_cls = _StableCls()
        print("Stable Audio ready.", flush=True)

    if not args.no_magenta:
        print("Connecting to Modal — MagentaInference…", flush=True)
        _MagentaCls = modal.Cls.from_name("webgenta-magenta", "MagentaInference")
        _magenta_cls = _MagentaCls()
        print("Magenta ready.", flush=True)

    print(f"\nComic Reader → http://localhost:{PORT}/\n", flush=True)
    web.run_app(make_app(), host="localhost", port=PORT, print=False)
