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
    "a cappella, voice only, no instruments, no background music, no beat, "
    "spoken word, anime female voice, kawaii, clear speech, dry vocal"
)
ANIME_VOICE_ID = "5b915c6d-8d96-416c-9755-eba65868cfef"
VISION_MODEL = "claude-haiku-4-5"
FIRST_BATCH = 3  # pages needed before reader unlocks; rest generate in background
VISION_SYSTEM = (
    "You score a single comic/manga/book page for a three-layer audio engine. "
    "Look at the art and any text, then produce all four fields:\n"
    "stable_audio_prompt: Rich music description for a background track (genre, "
    "instrumentation, mood). Under 200 characters.\n"
    "magenta_mood: Single mood from the allowed set for the melody overlay.\n"
    "suno_lyrics: Dialogue extracted verbatim from speech bubbles, newline-separated. "
    "Empty string if no readable dialogue.\n"
    "reason: Brief justification."
)
MOODS = ("calm", "tense", "action", "sad", "mysterious", "triumphant", "neutral")

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
    "progress": {},             # pdf_page_num -> {analyzed, stable, magenta, suno}
    "error": None,
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
        try:
            notes = melody_for_mood(mood)
            style_bytes = await _magenta_cls.embed_style.remote.aio(style)
            pcm = await _magenta_cls.render.remote.aio(style_bytes, notes)
            _state["magenta_wavs"][pdf_n] = _pcm32_to_wav(pcm)
            print(f"[p{pdf_n}] magenta done ({len(_state['magenta_wavs'][pdf_n])//1024} KB)", flush=True)
        except Exception as e:
            print(f"[p{pdf_n}] magenta ERROR: {e}", flush=True)
            _state["magenta_wavs"][pdf_n] = b""
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

    print(f"[p{pdf_n}] suno: submitting voice track…", flush=True)
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {"style": ANIME_VOICE_STYLE, "lyrics": lyrics, "voice_id": ANIME_VOICE_ID}

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
                    _state["suno_wavs"][pdf_n] = await r.read()
                print(f"[p{pdf_n}] suno done ({len(_state['suno_wavs'][pdf_n])//1024} KB)", flush=True)
            else:
                print(f"[p{pdf_n}] suno: timed out", flush=True)
                _state["suno_wavs"][pdf_n] = None

    except Exception as e:
        print(f"[p{pdf_n}] suno ERROR: {e}", flush=True)
        _state["suno_wavs"][pdf_n] = None

    _state["progress"][pdf_n]["suno"] = True
    await _broadcast({"type": "progress", "pdf_page": pdf_n, "track": "suno", "done": True})


async def _wait_for_page(pdf_n: int) -> None:
    while (pdf_n not in _state["stable_wavs"]
           or pdf_n not in _state["magenta_wavs"]
           or pdf_n not in _state["suno_wavs"]):
        await asyncio.sleep(0.2)


# ── Analysis + generation pipeline ───────────────────────────────────────────

async def _analyze_and_generate(pdf_page_nums: list[int]) -> None:
    import anthropic
    from pydantic import BaseModel

    class PageMusic(BaseModel):
        stable_audio_prompt: str
        magenta_mood: Literal[
            "calm", "tense", "action", "sad", "mysterious", "triumphant", "neutral"
        ]
        suno_lyrics: str
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
        return {
            "page_number": pnum,
            "stable_audio_prompt": pm.stable_audio_prompt.strip() or "ambient background music",
            "magenta_mood": pm.magenta_mood,
            "suno_lyrics": pm.suno_lyrics.strip(),
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
        result = await loop.run_in_executor(None, _analyze_one, pnum)
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
            if all(
                _state["progress"].get(n, {}).get("stable")
                and _state["progress"].get(n, {}).get("magenta")
                and _state["progress"].get(n, {}).get("suno")
                for n in first_batch
            ):
                reader_unlocked = True
                _state["status"] = "reader_ready"
                await _broadcast({"type": "status", "status": "reader_ready"})
                print(
                    f"First {len(first_batch)} page(s) ready — reader unlocked.",
                    flush=True,
                )

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
    reader = await request.multipart()
    pdf_bytes = None
    page_from = 1
    page_to = None

    async for part in reader:
        if part.name == "pdf":
            pdf_bytes = await part.read()
        elif part.name == "page_from":
            txt = await part.text()
            page_from = int(txt) if txt.strip() else 1
        elif part.name == "page_to":
            txt = await part.text()
            page_to = int(txt) if txt.strip() else None

    if not pdf_bytes:
        return web.json_response({"error": "No PDF file received"}, status=400)

    # Render selected pages
    page_images_tmp: dict[int, bytes] = {}
    total = 0
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        total = len(doc)
        page_to = page_to if page_to is not None else total
        page_from = max(1, min(page_from, total))
        page_to = max(page_from, min(page_to, total))
        for i, pg in enumerate(doc):
            pnum = i + 1
            if page_from <= pnum <= page_to:
                pix = pg.get_pixmap(dpi=150)
                page_images_tmp[pnum] = pix.tobytes("png")

    selected = sorted(page_images_tmp.keys())
    if not selected:
        return web.json_response({"error": "No pages in selected range"}, status=400)

    # Reset global state
    for task in _state.get("gen_tasks", []):
        task.cancel()

    _state.update({
        "status": "analyzing",
        "total_pdf_pages": total,
        "pdf_page_list": [],
        "page_images": page_images_tmp,
        "page_data": {},
        "stable_wavs": {},
        "magenta_wavs": {},
        "suno_wavs": {},
        "gen_tasks": [],
        "progress": {
            pnum: {"analyzed": False, "stable": False, "magenta": False, "suno": False}
            for pnum in selected
        },
        "error": None,
    })

    asyncio.create_task(_analyze_and_generate(selected))

    return web.json_response({
        "total_pages": total,
        "selected_pages": selected,
        "status": "analyzing",
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


# ── App factory ───────────────────────────────────────────────────────────────

def make_app() -> web.Application:
    app = web.Application(client_max_size=200 * 1024 * 1024)  # 200 MB upload limit
    app.router.add_get("/", handle_index)
    app.router.add_post("/peek", handle_peek)
    app.router.add_post("/upload", handle_upload)
    app.router.add_get("/status", handle_status)
    app.router.add_get("/ws", handle_ws)
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
