# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os, sys
os.environ["JAX_PLATFORMS"] = "cpu"  # must be set before JAX is imported
print("TOP OF FILE", flush=True); sys.stdout.flush()

"""WebSocket inference server for MRT2 browser streaming.

Protocol:
  Client → Server (text, first message): text prompt string
  Client → Server (text, subsequent):    JSON MIDI event
      {"type": "noteon",  "pitch": 0-127, "velocity": 0-127}
      {"type": "noteoff", "pitch": 0-127}
      {"type": "drum",    "velocity": 0-127}
      {"type": "prompt",  "text": "..."}
  Server → Client (binary):              raw float32 bytes, stereo interleaved
      3840 floats = 1920 samples × 2 channels @ 48kHz (40ms per frame)

Usage:
  python server.py [--host 0.0.0.0] [--port 8765] [--model mrt2_small] [--no-demo]
"""

import argparse
import asyncio
import json
import logging
import sys
import time

print("importing numpy", flush=True)
import numpy as np
print("importing websockets", flush=True)
import websockets
import websockets.exceptions
print("imports done", flush=True)

print("importing magenta_rt...", flush=True)

try:
    print("  trying venv import...", flush=True)
    from magenta_rt.jax.system import MagentaRT2System
    print("  venv import ok", flush=True)
except ModuleNotFoundError:
    import pathlib
    print("  not in venv, trying path fallback...", flush=True)
    _mrt_root = pathlib.Path(__file__).parent.parent / "magenta-realtime"
    if not _mrt_root.exists():
        raise RuntimeError(
            "magenta_rt not found. Activate the magenta-realtime venv or ensure "
            f"the repo exists at {_mrt_root}"
        )
    sys.path.insert(0, str(_mrt_root))
    print(f"  added {_mrt_root} to path, importing...", flush=True)
    from magenta_rt.jax.system import MagentaRT2System
    print("  fallback import ok", flush=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", stream=sys.stdout)
logger = logging.getLogger(__name__)


# ─── Mary Had a Little Lamb sequencer ────────────────────────────────────────
#
# Each entry: (midi_pitch, duration_in_frames, gap_frames_after)
# 1 frame = 40ms.  Quarter note at ~100 BPM ≈ 15 frames.
# Gap of 2 frames between notes prevents legato smear.

E4, D4, C4, G4 = 64, 62, 60, 67

MARY = [
    # Phrase 1 only: E D C D E E E (7 notes, ~5s audio)
    (E4, 13, 2), (D4, 13, 2), (C4, 13, 2), (D4, 13, 2),
    (E4, 13, 2), (E4, 13, 2), (E4, 26, 4),
]


async def run_demo_sequencer(active_notes: dict, stop_event: asyncio.Event):
    """Loop Mary Had a Little Lamb by updating active_notes in place."""
    while not stop_event.is_set():
        for pitch, hold_frames, gap_frames in MARY:
            if stop_event.is_set():
                return
            # Note on
            active_notes.clear()
            active_notes[pitch] = True   # onset
            # Hold
            for _ in range(hold_frames):
                if stop_event.is_set():
                    return
                await asyncio.sleep(0.04)   # ~1 frame (40ms)
            # Note off + gap
            active_notes.clear()
            for _ in range(gap_frames):
                if stop_event.is_set():
                    return
                await asyncio.sleep(0.04)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _build_notes_vector(active: dict[int, bool]) -> list[int]:
    """Convert active-note dict → 128-int MRT2 conditioning vector."""
    notes = [0] * 128
    for pitch, is_onset in active.items():
        notes[pitch] = 2 if is_onset else 1
    return notes


# ─── Client handler ──────────────────────────────────────────────────────────

def _render_mary(system: MagentaRT2System, style, state):
    """Render one full pass of Mary Had a Little Lamb offline, returns (audio, state).

    Generates in per-note batches (onset + hold + gap) so JAX processes
    contiguous chunks rather than single frames — much faster on CPU.
    """
    chunks = []
    total = len(MARY)
    for i, (pitch, hold_frames, gap_frames) in enumerate(MARY, 1):
        print(f"  note {i}/{total} (pitch={pitch})...", flush=True)

        # Onset frame
        notes = [0] * 128
        notes[pitch] = 2
        wf, state = system.generate(style=style, notes=notes, drums=[-1], frames=1, state=state)
        chunks.append(wf.samples)

        # Continuation frames (hold_frames - 1)
        if hold_frames > 1:
            notes[pitch] = 1
            wf, state = system.generate(style=style, notes=notes, drums=[-1], frames=hold_frames - 1, state=state)
            chunks.append(wf.samples)

        # Gap frames — no active notes
        if gap_frames > 0:
            wf, state = system.generate(style=style, notes=[0]*128, drums=[-1], frames=gap_frames, state=state)
            chunks.append(wf.samples)

    return np.concatenate(chunks, axis=0).astype(np.float32), state


async def handle_client(ws, system: MagentaRT2System, demo: bool):
    remote = ws.remote_address
    logger.info("Client connected: %s", remote)

    try:
        prompt = await asyncio.wait_for(ws.recv(), timeout=30)
    except asyncio.TimeoutError:
        logger.warning("Client %s never sent a prompt, closing", remote)
        return

    logger.info("Prompt from %s: %r", remote, prompt)
    style = system.embed_style(prompt)
    state = None

    try:
        loop_num = 0
        while True:
            loop_num += 1
            t0 = time.perf_counter()
            print(f"Rendering loop {loop_num}...", flush=True)

            # Render the full song offline in a thread so the event loop stays alive
            audio, state = await asyncio.get_event_loop().run_in_executor(
                None, _render_mary, system, style, state
            )

            duration = audio.shape[0] / 48000
            elapsed = time.perf_counter() - t0
            print(f"  Done: {duration:.1f}s audio in {elapsed:.1f}s — streaming to client", flush=True)

            # Stream in 400ms chunks so the client starts playing quickly
            CHUNK = 1920 * 10  # 10 frames × 1920 samples
            for i in range(0, len(audio), CHUNK):
                await ws.send(audio[i : i + CHUNK].tobytes())
                await asyncio.sleep(0)

            # Handle any prompt updates that arrived while rendering
            try:
                while True:
                    raw = ws.recv()
                    event = json.loads(await asyncio.wait_for(raw, timeout=0.001))
                    if event.get("type") == "prompt":
                        new_prompt = event.get("text", "")
                        print(f"Prompt updated: {new_prompt!r}", flush=True)
                        style = system.embed_style(new_prompt)
                        state = None  # reset model state on prompt change
            except (asyncio.TimeoutError, json.JSONDecodeError):
                pass

    except websockets.exceptions.ConnectionClosed:
        pass
    finally:
        logger.info("Client disconnected: %s", remote)


# ─── Entry point ─────────────────────────────────────────────────────────────

async def main(host: str, port: int, system: MagentaRT2System, demo: bool):
    print(f"Listening on ws://{host}:{port} — open the browser app to connect", flush=True)

    async with websockets.serve(
        lambda ws: handle_client(ws, system, demo),
        host,
        port,
        max_size=2**20,
    ):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MRT2 WebSocket inference server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="mrt2_small", choices=["mrt2_small", "mrt2_base"])
    parser.add_argument("--no-demo", action="store_true",
                        help="Disable built-in Mary Had a Little Lamb sequencer")
    args = parser.parse_args()

    # Load model before starting the event loop — avoids Windows ProactorEventLoop deadlock
    print(f"Loading model: {args.model} (this takes ~15s on CPU)...", flush=True)
    system = MagentaRT2System(size=args.model)
    print("Model ready! Starting WebSocket server...", flush=True)

    asyncio.run(main(args.host, args.port, system, demo=not args.no_demo))
