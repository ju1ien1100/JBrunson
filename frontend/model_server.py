"""Comic book audio server.

Usage:
  python model_server.py <pdf_path> [--pages-json pages.json]

Generates 3 audio tracks per page in parallel at startup using Modal GPU apps,
then serves a browser comic reader over WebSocket on ws://localhost:8766/.

Track sources:
  background  — Stable Audio 3 Small via Modal (webgenta-stability)
  melody      — Magenta MRT2 via Modal (webgenta-magenta) + midi_library moods
  voice       — Suno API (character dialogue from suno_lyrics field)

WS protocol (server → browser, per page request):
  {"type": "page", "page_number": N, "total": T, "image_b64": "..."}
  {"type": "track_header", "track": "background", "mime": "audio/wav"}
  <WAV bytes>
  {"type": "track_header", "track": "melody", "mime": "audio/wav"}
  <WAV bytes>
  {"type": "track_header", "track": "voice", "mime": "audio/mpeg"}   # only if dialogue
  <MP3 bytes>
  {"type": "tracks_end"}

Caching: generated audio is held in dicts keyed by PDF page number.
Revisiting a previous page (Prev button) serves instantly from cache — no
Modal calls are re-issued.

Environment:
  SUNO_API_KEY  — Suno bearer token (optional; voice track silently skipped if absent)
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import os
import sys
import wave
from pathlib import Path

import aiohttp
import fitz  # PyMuPDF
import numpy as np
import websockets
import websockets.exceptions

sys.path.insert(0, str(Path(__file__).resolve().parent))
from midi_library import melody_for_mood  # noqa: E402

HOST = "localhost"
PORT = 8766
MAGENTA_SAMPLE_RATE = 48000
MAGENTA_CHANNELS = 2
STABLE_AUDIO_DURATION = 30  # seconds per background track
SUNO_BASE = "https://api.suno.com"
SUNO_POLL_INTERVAL = 3.0
SUNO_POLL_TIMEOUT = 180.0
ANIME_VOICE_STYLE = (
    "a cappella, voice only, no instruments, no background music, no beat, "
    "spoken word, anime female voice, kawaii, clear speech, dry vocal"
)
ANIME_VOICE_ID = "5b915c6d-8d96-416c-9755-eba65868cfef"

# ── Global state ──────────────────────────────────────────────────────────────

page_images: dict[int, bytes] = {}   # pdf_page_number -> PNG bytes
page_data: dict[int, dict] = {}      # pdf_page_number -> pages.json entry
page_list: list[int] = []            # sorted analyzed pdf_page_numbers
total_pages: int = 0                 # = len(page_list)

# Audio caches keyed by pdf_page_number; present once the task finishes
stable_wavs: dict[int, bytes] = {}   # WAV bytes (b"" on failure)
magenta_wavs: dict[int, bytes] = {}  # WAV bytes (b"" on failure)
suno_wavs: dict[int, bytes | None] = {}  # MP3 bytes, or None if no dialogue

_gen_tasks: list[asyncio.Task] = []  # strong refs — prevents GC cancellation

# Modal class handles (set in __main__ before asyncio.run)
_stable_cls = None
_magenta_cls = None
_skip_suno: bool = False


# ── PCM helpers ───────────────────────────────────────────────────────────────

def _pcm32_to_wav(pcm_bytes: bytes,
                  sample_rate: int = MAGENTA_SAMPLE_RATE,
                  channels: int = MAGENTA_CHANNELS) -> bytes:
    """Convert float32 stereo interleaved PCM bytes → 16-bit WAV bytes."""
    audio = np.frombuffer(pcm_bytes, dtype=np.float32)
    pcm_i16 = (np.clip(audio, -1.0, 1.0) * 32767.0).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm_i16.tobytes())
    return buf.getvalue()


# ── Per-page audio generation ─────────────────────────────────────────────────

async def _gen_stable(n: int) -> None:
    """Generate background track for PDF page n via Stable Audio 3 on Modal."""
    if _stable_cls is None:
        stable_wavs[n] = b""
        return
    prompt = page_data[n].get("stable_audio_prompt", "ambient background music")
    print(f"[page {n}] stable audio: {prompt[:70]!r}…", flush=True)
    try:
        wav = await _stable_cls.generate.remote.aio(prompt, STABLE_AUDIO_DURATION)
        stable_wavs[n] = wav
        print(f"[page {n}] stable audio done ({len(wav) // 1024} KB)", flush=True)
    except Exception as e:
        print(f"[page {n}] stable audio ERROR: {e}", flush=True)
        stable_wavs[n] = b""


async def _gen_magenta(n: int) -> None:
    """Generate melody track for PDF page n via Magenta MRT2 on Modal."""
    if _magenta_cls is None:
        magenta_wavs[n] = b""
        return
    mood = page_data[n].get("magenta_mood", "neutral")
    style_prompt = page_data[n].get("stable_audio_prompt", "ambient music")
    print(f"[page {n}] magenta: mood={mood!r}", flush=True)
    try:
        notes = melody_for_mood(mood)
        style_bytes = await _magenta_cls.embed_style.remote.aio(style_prompt)
        pcm_bytes = await _magenta_cls.render.remote.aio(style_bytes, notes)
        magenta_wavs[n] = _pcm32_to_wav(pcm_bytes)
        print(f"[page {n}] magenta done ({len(magenta_wavs[n]) // 1024} KB)", flush=True)
    except Exception as e:
        print(f"[page {n}] magenta ERROR: {e}", flush=True)
        magenta_wavs[n] = b""


async def _gen_suno(n: int) -> None:
    """Generate voice track for PDF page n via Suno API."""
    if _skip_suno:
        suno_wavs[n] = None
        return
    lyrics = page_data[n].get("suno_lyrics", "").strip()
    if not lyrics:
        suno_wavs[n] = None
        print(f"[page {n}] suno: no dialogue, skipped", flush=True)
        return

    api_key = os.environ.get("SUNO_API_KEY", "")
    if not api_key:
        suno_wavs[n] = None
        print(f"[page {n}] suno: SUNO_API_KEY not set, skipped", flush=True)
        return

    print(f"[page {n}] suno: submitting voice track…", flush=True)
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload = {
        "style": ANIME_VOICE_STYLE,
        "lyrics": lyrics,
        "voice_id": ANIME_VOICE_ID,
    }

    try:
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{SUNO_BASE}/v0/audio", json=payload, headers=headers
            ) as resp:
                data = await resp.json()
                if resp.status not in (200, 201, 202):
                    print(f"[page {n}] suno submit error {resp.status}: {data}", flush=True)
                    suno_wavs[n] = None
                    return

            job_id = data.get("id", "")
            if not job_id:
                print(f"[page {n}] suno: no job id in response", flush=True)
                suno_wavs[n] = None
                return

            loop = asyncio.get_running_loop()
            deadline = loop.time() + SUNO_POLL_TIMEOUT
            audio_url = None
            while loop.time() < deadline:
                await asyncio.sleep(SUNO_POLL_INTERVAL)
                async with session.get(
                    f"{SUNO_BASE}/v0/audio/{job_id}", headers=headers
                ) as resp:
                    status_data = await resp.json()
                state = status_data.get("status", "unknown")
                if state == "complete":
                    audio_url = status_data.get("audio_url", "")
                    break
                elif state == "error":
                    print(f"[page {n}] suno job error: {status_data.get('error')}", flush=True)
                    suno_wavs[n] = None
                    return

            if not audio_url:
                print(f"[page {n}] suno: timed out or no audio_url", flush=True)
                suno_wavs[n] = None
                return

            async with session.get(audio_url) as resp:
                audio_bytes = await resp.read()
            suno_wavs[n] = audio_bytes
            print(f"[page {n}] suno done ({len(audio_bytes) // 1024} KB)", flush=True)

    except Exception as e:
        print(f"[page {n}] suno ERROR: {e}", flush=True)
        suno_wavs[n] = None


async def _wait_for_page(pdf_n: int) -> None:
    """Block until all 3 tracks for pdf_n are present in the cache dicts."""
    while pdf_n not in stable_wavs or pdf_n not in magenta_wavs or pdf_n not in suno_wavs:
        await asyncio.sleep(0.2)


# ── WebSocket handler ─────────────────────────────────────────────────────────

async def reader_handler(ws) -> None:
    print(f"Browser connected: {ws.remote_address}", flush=True)
    try:
        async for raw in ws:
            if not isinstance(raw, str):
                continue
            try:
                msg = json.loads(raw)
            except json.JSONDecodeError:
                continue

            if msg.get("type") == "ready":
                # reader_page is 1-based sequential index into page_list
                reader_n = int(msg.get("page", 1))
                idx = reader_n - 1
                if idx < 0 or idx >= len(page_list):
                    await ws.send(json.dumps({
                        "type": "error",
                        "message": f"Page {reader_n} out of range (1–{len(page_list)})",
                    }))
                    continue

                pdf_n = page_list[idx]
                print(f"Browser requesting reader page {reader_n} (PDF page {pdf_n})", flush=True)

                # Wait until all 3 tracks for this page are ready
                await _wait_for_page(pdf_n)

                # Page image
                img_b64 = base64.b64encode(page_images.get(pdf_n, b"")).decode("ascii")
                await ws.send(json.dumps({
                    "type": "page",
                    "page_number": reader_n,
                    "total": total_pages,
                    "image_b64": img_b64,
                }))

                # Background track
                bg = stable_wavs.get(pdf_n, b"")
                if bg:
                    await ws.send(json.dumps({
                        "type": "track_header", "track": "background", "mime": "audio/wav"
                    }))
                    await ws.send(bg)

                # Melody track
                mel = magenta_wavs.get(pdf_n, b"")
                if mel:
                    await ws.send(json.dumps({
                        "type": "track_header", "track": "melody", "mime": "audio/wav"
                    }))
                    await ws.send(mel)

                # Voice track (only if dialogue exists)
                voice = suno_wavs.get(pdf_n)
                if voice:
                    await ws.send(json.dumps({
                        "type": "track_header", "track": "voice", "mime": "audio/mpeg"
                    }))
                    await ws.send(voice)

                await ws.send(json.dumps({"type": "tracks_end"}))

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        print(f"Browser disconnected: {ws.remote_address}", flush=True)


# ── Startup helpers ───────────────────────────────────────────────────────────

def load_pdf(pdf_path: str) -> None:
    print(f"Rendering PDF: {pdf_path}", flush=True)
    with fitz.open(pdf_path) as doc:
        for i, pg in enumerate(doc):
            pix = pg.get_pixmap(dpi=150)
            page_images[i + 1] = pix.tobytes("png")
    print(f"Rendered {len(page_images)} PDF pages.", flush=True)


def load_pages_json(path: str) -> None:
    global page_list, total_pages
    data = json.loads(Path(path).read_text())
    for p in data.get("pages", []):
        page_data[p["page_number"]] = p
    page_list = sorted(page_data.keys())
    total_pages = len(page_list)
    print(f"Loaded {total_pages} analyzed pages from {path}: {page_list}", flush=True)


async def prefetch_all() -> None:
    """Fire 3 async tasks per page immediately; hold strong refs to prevent GC."""
    print(
        f"Pre-generating audio for {total_pages} pages "
        f"({total_pages * 3} tasks firing in parallel)…",
        flush=True,
    )
    for pdf_n in page_list:
        for coro in (_gen_stable(pdf_n), _gen_magenta(pdf_n), _gen_suno(pdf_n)):
            task = asyncio.create_task(coro)
            task.add_done_callback(lambda t: t.exception() if not t.cancelled() else None)
            _gen_tasks.append(task)


async def main_async(pdf_path: str, pages_json: str) -> None:
    load_pdf(pdf_path)
    load_pages_json(pages_json)
    await prefetch_all()
    print(f"\nComic reader WS on ws://{HOST}:{PORT}/\nOpen comic.html in your browser.", flush=True)
    async with websockets.serve(
        reader_handler, HOST, PORT, max_size=100 * 1024 * 1024
    ):
        await asyncio.Future()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Comic book 3-track audio server")
    parser.add_argument("pdf", help="Path to the PDF file")
    parser.add_argument(
        "--pages-json", default="pages.json",
        help="pages.json produced by pdf_reader.py --analyze (default: pages.json)",
    )
    parser.add_argument("--no-stable", action="store_true",
                        help="Skip Stable Audio 3 background track (no Modal call)")
    parser.add_argument("--no-magenta", action="store_true",
                        help="Skip Magenta MRT2 melody track (no Modal call)")
    parser.add_argument("--no-suno", action="store_true",
                        help="Skip Suno voice track")
    args = parser.parse_args()

    import modal

    if not args.no_stable:
        print("Connecting to Modal — StableAudioInference…", flush=True)
        _StableCls = modal.Cls.from_name("webgenta-stability", "StableAudioInference")
        _stable_cls = _StableCls()
        print("Stable Audio ready (T4 GPU warms on first call).", flush=True)

    if not args.no_magenta:
        print("Connecting to Modal — MagentaInference…", flush=True)
        _MagentaCls = modal.Cls.from_name("webgenta-magenta", "MagentaInference")
        _magenta_cls = _MagentaCls()
        print("Magenta ready (GPU warms on first call).", flush=True)

    if args.no_suno:
        _skip_suno = True

    asyncio.run(main_async(args.pdf, args.pages_json))
