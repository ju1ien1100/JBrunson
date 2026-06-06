"""
PDF reader — three modes:

1. WebSocket client (default): reads a PDF page-by-page, extracts text and
   rendered page images, and streams each page to a WebSocket server.

2. Analyze (--analyze): Claude reads each page (image + text) and decides a
   *mood* and a Magenta *style prompt* per page, written to pages.json.
   Run by whoever has an ANTHROPIC_API_KEY. No Modal needed.

3. Modal ingestion (--modal): reads pages.json, maps each page's mood to a
   curated MIDI motif, calls the deployed Modal `webgenta-magenta` app
   (embed_style -> render), and writes one WAV per page. Run by whoever has
   Modal credentials. No ANTHROPIC_API_KEY needed.

Two-stage handoff: you run --analyze locally (produces pages.json), your
teammate runs --modal against the deployed GPU (consumes pages.json).

Usage:
  # WebSocket streaming (needs model_server.py running on :8765)
  python pdf_reader.py comic.pdf

  # Stage 1 — vision analysis (needs: pip install anthropic; ANTHROPIC_API_KEY)
  python pdf_reader.py comic.pdf --analyze --pages 1 --out pages.json

  # Stage 2 — Modal render (needs: pip install modal numpy; Modal auth; app deployed)
  python pdf_reader.py --modal --pages-json pages.json --out test_output
"""

import argparse
import asyncio
import base64
import json
import sys
import wave
from pathlib import Path

import fitz  # PyMuPDF
import websockets

# Reusable mood -> MIDI motif library + converters (see midi_library.py).
# Insert this file's dir so the import works regardless of the caller's CWD.
sys.path.insert(0, str(Path(__file__).resolve().parent))
from midi_library import MELODY_LIBRARY, melody_for_mood  # noqa: E402

WS_URI = "ws://localhost:8765"

# Max WebSocket frame size (50 MB). A 150-DPI PNG page base64-encoded can be
# several MB, so this bounds peak memory while leaving comfortable headroom.
MAX_FRAME_SIZE = 50 * 1024 * 1024

# ── Magenta / Modal ingestion constants ─────────────────────────────────────
SAMPLE_RATE = 48000
CHANNELS = 2  # Modal render() returns stereo interleaved float32
PROMPT_MAX_CHARS = 200
DEFAULT_PROMPT = "calm ambient background music"

# ── Vision (Claude) constants ───────────────────────────────────────────────
VISION_MODEL = "claude-opus-4-8"
PAGE_TEXT_LIMIT = 2000  # chars of page text sent to the vision model
MOODS = ("calm", "tense", "action", "sad", "mysterious", "triumphant", "neutral")


def extract_pages(pdf_path: str):
    """Yield one dict per page: page number, text, and a PNG image (base64)."""
    # Use a context manager so the file handle is released even if the caller
    # breaks out early or an exception is raised mid-extraction.
    with fitz.open(pdf_path) as doc:
        for page_index in range(len(doc)):
            page = doc[page_index]

            # Extract text (speech bubbles / prose)
            text = page.get_text("text").strip()

            # Render the page to a PNG image (the panel/illustration)
            pix = page.get_pixmap(dpi=150)
            img_bytes = pix.tobytes("png")
            img_b64 = base64.b64encode(img_bytes).decode("ascii")

            yield {
                "type": "page",
                "page_number": page_index + 1,
                "total_pages": len(doc),
                "text": text,
                "image_png_b64": img_b64,
            }


async def send_pdf(pdf_path: str):
    async with websockets.connect(WS_URI, max_size=MAX_FRAME_SIZE) as ws:
        for page in extract_pages(pdf_path):
            await ws.send(json.dumps(page))
            print(f"Sent page {page['page_number']}/{page['total_pages']} "
                  f"({len(page['text'])} chars)")

            # Wait for the model's response for this page
            response = await ws.recv()
            result = json.loads(response)
            if result.get("type") == "page_error":
                print(f"  -> server error on page "
                      f"{result.get('page_number')}: {result.get('error')}")
            else:
                print(f"  -> model: {result.get('summary', result)}")

        # Signal completion
        await ws.send(json.dumps({"type": "done"}))
        print("All pages sent.")


# ── Stage 1: Claude vision analysis -> pages.json ────────────────────────────
_VISION_SYSTEM = (
    "You score a single comic/manga/book page for a background-music engine. "
    "Look at the art and any text, then choose the single mood that best fits "
    "the scene and write a short musical style prompt for a music generator. "
    "The style prompt should describe genre, instrumentation, and feeling "
    "(e.g. 'dark ominous orchestral strings, slow and tense'). Keep it under "
    "20 words. Pick mood from the allowed set only."
)


def analyze_pages(pdf_path: str, max_pages, vision_model: str) -> list:
    """Use Claude vision to assign a mood + style prompt to each page.

    Returns [{page_number, total_pages, mood, style_prompt, reason}].
    Requires `pip install anthropic` and ANTHROPIC_API_KEY in the environment.
    """
    import anthropic
    from pydantic import BaseModel
    from typing import Literal

    class PageMusic(BaseModel):
        mood: Literal["calm", "tense", "action", "sad",
                      "mysterious", "triumphant", "neutral"]
        style_prompt: str
        reason: str

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from env

    results = []
    count = 0
    for page in extract_pages(pdf_path):
        if max_pages is not None and count >= max_pages:
            break
        count += 1

        page_text = page["text"][:PAGE_TEXT_LIMIT] or "(no extractable text)"
        resp = client.messages.parse(
            model=vision_model,
            max_tokens=1024,
            system=_VISION_SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": page["image_png_b64"],
                        },
                    },
                    {
                        "type": "text",
                        "text": (f"Allowed moods: {', '.join(MOODS)}.\n\n"
                                 f"Page text:\n{page_text}"),
                    },
                ],
            }],
            output_format=PageMusic,
        )
        pm = resp.parsed_output
        entry = {
            "page_number": page["page_number"],
            "total_pages": page["total_pages"],
            "mood": pm.mood,
            "style_prompt": pm.style_prompt.strip() or DEFAULT_PROMPT,
            "reason": pm.reason,
        }
        results.append(entry)
        print(f"[page {entry['page_number']}/{entry['total_pages']}] "
              f"mood={entry['mood']!r} prompt={entry['style_prompt']!r}", flush=True)

    return results


def run_analyze(pdf_path: str, max_pages, out_path: str, vision_model: str) -> int:
    pages = analyze_pages(pdf_path, max_pages, vision_model)
    if not pages:
        print("No pages analyzed; nothing written.", file=sys.stderr)
        return 1
    payload = {"pdf": pdf_path, "pages": pages}
    Path(out_path).write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {len(pages)} page(s) of analysis to {Path(out_path).resolve()}")
    return 0


# ── Stage 2: pages.json -> Modal Magenta -> per-page WAV ──────────────────────
def save_wav(path, pcm_f32) -> float:
    """Write interleaved stereo float32 PCM to a 16-bit WAV. Returns duration (s)."""
    import numpy as np

    clipped = np.clip(pcm_f32, -1.0, 1.0)
    pcm_i16 = (clipped * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(CHANNELS)
        w.setsampwidth(2)
        w.setframerate(SAMPLE_RATE)
        w.writeframes(pcm_i16.tobytes())
    frames = len(pcm_f32) // CHANNELS
    return frames / SAMPLE_RATE


def render_from_analysis(pages: list, out_dir, modal_inference) -> list:
    """For each analyzed page, pick a melody by mood and render via Modal.

    `modal_inference` is an instantiated Modal class handle exposing
    `embed_style.remote(prompt)` and `render.remote(style_bytes, notes)`.
    """
    import numpy as np

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    for p in pages:
        mood = p.get("mood", "neutral")
        prompt = (p.get("style_prompt") or DEFAULT_PROMPT)[:PROMPT_MAX_CHARS]
        notes_sequence = melody_for_mood(mood)
        print(f"[page {p['page_number']}] mood={mood!r} prompt={prompt!r}", flush=True)

        style_bytes = modal_inference.embed_style.remote(prompt)
        pcm_bytes = modal_inference.render.remote(style_bytes, notes_sequence)
        audio = np.frombuffer(pcm_bytes, dtype=np.float32)

        wav_path = out_dir / f"page_{p['page_number']:02d}_{mood}.wav"
        duration = save_wav(wav_path, audio)
        print(f"    -> {wav_path}  ({duration:.1f}s audio)", flush=True)
        results.append({"page": p["page_number"], "mood": mood,
                        "wav": str(wav_path), "duration_s": duration})

    return results


def run_modal(pages_json: str, max_pages, out_dir: str, app: str, cls: str) -> int:
    """Read pages.json and render each page via the deployed Modal Magenta app."""
    data = json.loads(Path(pages_json).read_text())
    pages = data.get("pages", [])
    if max_pages is not None:
        pages = pages[:max_pages]
    if not pages:
        print("No pages in pages.json; nothing to do.", file=sys.stderr)
        return 1
    print(f"Loaded {len(pages)} page(s) from {pages_json}", flush=True)

    # Imported here so the other modes / --help work without Modal installed.
    import modal

    print(f"Connecting to Modal app '{app}' class '{cls}'...", flush=True)
    MagentaInference = modal.Cls.from_name(app, cls)
    inf = MagentaInference()

    results = render_from_analysis(pages, out_dir, inf)
    print(f"\nDone. Wrote {len(results)} file(s) to {Path(out_dir).resolve()}")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PDF reader: WebSocket streaming, Claude vision analysis, or Modal Magenta render")
    parser.add_argument("pdf", nargs="?", help="Path to the input PDF (for default/--analyze modes)")
    parser.add_argument("--analyze", action="store_true",
                        help="Stage 1: Claude vision -> per-page mood + style prompt (writes pages.json)")
    parser.add_argument("--modal", action="store_true",
                        help="Stage 2: render each page from pages.json via the deployed Modal Magenta app")
    parser.add_argument("--pages", type=int, default=None,
                        help="Limit to first N pages (--analyze / --modal). Each page = one GPU render in stage 2.")
    parser.add_argument("--out", default=None,
                        help="Output path: pages.json for --analyze (default pages.json), output dir for --modal (default test_output)")
    parser.add_argument("--pages-json", default="pages.json",
                        help="(--modal) Input analysis file from stage 1")
    parser.add_argument("--vision-model", default=VISION_MODEL,
                        help="(--analyze) Claude model for page understanding")
    parser.add_argument("--app", default="webgenta-magenta", help="(--modal) Modal app name")
    parser.add_argument("--cls", default="MagentaInference", help="(--modal) Modal class name")
    args = parser.parse_args()

    if args.analyze:
        if not args.pdf:
            parser.error("--analyze requires a PDF path")
        raise SystemExit(run_analyze(args.pdf, args.pages, args.out or "pages.json", args.vision_model))
    elif args.modal:
        raise SystemExit(run_modal(args.pages_json, args.pages, args.out or "test_output", args.app, args.cls))
    else:
        if not args.pdf:
            parser.error("a PDF path is required for WebSocket streaming mode")
        asyncio.run(send_pdf(args.pdf))
