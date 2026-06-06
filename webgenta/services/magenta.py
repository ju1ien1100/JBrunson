"""Magenta RT2 music service.

Generates continuous background music conditioned on a text style prompt
and live MIDI note events. Runs the MRT2-small model via JAX (CPU or GPU).

Inbound actions (from client):
  { "service": "magenta", "action": "start",    "prompt": "jazzy piano" }
  { "service": "magenta", "action": "stop" }
  { "service": "magenta", "action": "prompt",   "text": "new style" }
  { "service": "magenta", "action": "note_on",  "pitch": 0-127, "velocity": 0-127 }
  { "service": "magenta", "action": "note_off", "pitch": 0-127 }
  { "service": "magenta", "action": "drum",     "velocity": 0-127 }

Outbound events (to client):
  { "service": "magenta", "event": "status",  "state": "rendering", "loop": 1, "note": "2/7" }
  { "service": "magenta", "event": "ready" }
  { "service": "magenta", "event": "error",   "message": "..." }
  <binary>  raw float32 PCM, stereo interleaved, 48kHz
            chunk size varies; each server send = one render batch
"""

from __future__ import annotations

import asyncio
import time
from typing import TYPE_CHECKING

import numpy as np

from services import msg

if TYPE_CHECKING:
    import websockets

# ── MIDI demo sequence ────────────────────────────────────────────────────────
# (pitch, hold_frames, gap_frames)  1 frame = 40ms @ 25Hz
E4, D4, C4, G4 = 64, 62, 60, 67
BEEP = 84  # C6 — sharp marker note between phrases

def _phrase(notes):
    """Append a BEEP marker after a phrase."""
    return list(notes) + [(BEEP, 3, 8)]

MARY = [
    *_phrase([
        # "Mary had a little lamb"
        (E4, 13, 2), (D4, 13, 2), (C4, 13, 2), (D4, 13, 2),
        (E4, 13, 2), (E4, 13, 2), (E4, 26, 4),
    ]),
    *_phrase([
        # "little lamb, little lamb"
        (D4, 13, 2), (D4, 13, 2), (D4, 26, 4),
        (E4, 13, 2), (G4, 13, 2), (G4, 26, 4),
    ]),
    *_phrase([
        # "Mary had a little lamb, its"
        (E4, 13, 2), (D4, 13, 2), (C4, 13, 2), (D4, 13, 2),
        (E4, 13, 2), (E4, 13, 2), (E4, 13, 2), (E4, 26, 4),
    ]),
    *_phrase([
        # "fleece was white as snow"
        (D4, 13, 2), (D4, 13, 2), (E4, 13, 2), (D4, 13, 2), (C4, 26, 4),
    ]),
    *_phrase([
        # Repeat phrase 1
        (E4, 13, 2), (D4, 13, 2), (C4, 13, 2), (D4, 13, 2),
        (E4, 13, 2), (E4, 13, 2), (E4, 26, 4),
    ]),
    *_phrase([
        # Repeat phrase 2
        (D4, 13, 2), (D4, 13, 2), (D4, 26, 4),
        (E4, 13, 2), (G4, 13, 2), (G4, 26, 4),
    ]),
    *_phrase([
        # Repeat phrase 3
        (E4, 13, 2), (D4, 13, 2), (C4, 13, 2), (D4, 13, 2),
        (E4, 13, 2), (E4, 13, 2), (E4, 13, 2), (E4, 26, 4),
    ]),
    # Final phrase — no beep, let it breathe
    (D4, 13, 2), (D4, 13, 2), (E4, 13, 2), (D4, 13, 2), (C4, 26, 8),
]


def _build_notes_vector(active: dict[int, bool]) -> list[int]:
    """Convert {pitch: is_onset} → 128-int MRT2 conditioning vector."""
    notes = [0] * 128
    for pitch, is_onset in active.items():
        notes[pitch] = 2 if is_onset else 1
    return notes


def _mary_notes_sequence() -> list[dict]:
    """Convert MARY into a flat list of note segments for Modal/local rendering."""
    segments = []
    for pitch, hold, gap in MARY:
        onset = [0] * 128
        onset[pitch] = 2
        segments.append({"notes": onset, "drums": [-1], "frames": 1})

        if hold > 1:
            cont = [0] * 128
            cont[pitch] = 1
            segments.append({"notes": cont, "drums": [-1], "frames": hold - 1})

        if gap > 0:
            segments.append({"notes": [0] * 128, "drums": [-1], "frames": gap})

    return segments


def _render_loop_local(system, style, state: object) -> tuple[np.ndarray, object]:
    """Render one full MARY pass locally (CPU/GPU JAX)."""
    chunks = []
    for seg in _mary_notes_sequence():
        wf, state = system.generate(
            style=style, notes=seg["notes"], drums=seg["drums"],
            frames=seg["frames"], state=state,
        )
        chunks.append(wf.samples)
    return np.concatenate(chunks, axis=0).astype(np.float32), state


async def _render_loop_modal(modal_inference, style_bytes: bytes) -> np.ndarray:
    """Render one full MARY pass via Modal GPU (async)."""
    pcm_bytes = await modal_inference.render.remote.aio(style_bytes, _mary_notes_sequence())
    return np.frombuffer(pcm_bytes, dtype=np.float32)


class MagentaService:
    SERVICE = "magenta"

    def __init__(self, system=None, modal_inference=None):
        """Pass either a local JAX system or a Modal inference handle, not both."""
        self._system = system
        self._modal = modal_inference
        self._use_modal = modal_inference is not None

    async def handle(self, ws: "websockets.WebSocketServerProtocol", action: dict) -> None:
        """Dispatch a single inbound action. Called by the router."""
        a = action.get("action")

        if a == "start":
            prompt = action.get("prompt", "background music")
            await self._run(ws, prompt)
        else:
            # Other actions (note_on, note_off, etc.) are handled inside _run's
            # message loop; receiving them outside an active session is a no-op.
            pass

    async def _run(self, ws: "websockets.WebSocketServerProtocol", initial_prompt: str) -> None:
        """Main generation loop for one client session."""
        import websockets.exceptions

        await ws.send(msg(self.SERVICE, "status", state="embedding"))

        if self._use_modal:
            style_bytes = await self._modal.embed_style.remote.aio(initial_prompt)
            style = None  # not needed locally
        else:
            style = self._system.embed_style(initial_prompt)
            style_bytes = None

        state = None
        loop_num = 0
        active_notes: dict[int, bool] = {}

        async def _render_and_stream():
            nonlocal state, loop_num
            while True:
                loop_num += 1
                t0 = time.perf_counter()

                if self._use_modal:
                    audio = await _render_loop_modal(self._modal, style_bytes)
                else:
                    audio, state = await asyncio.get_running_loop().run_in_executor(
                        None, _render_loop_local, self._system, style, state
                    )

                elapsed = time.perf_counter() - t0
                duration = audio.shape[0] / 48000
                backend = "modal/gpu" if self._use_modal else "local/cpu"
                print(f"[magenta] loop {loop_num} ({backend}): {duration:.1f}s in {elapsed:.1f}s", flush=True)
                await ws.send(msg(self.SERVICE, "ready", loop=loop_num, duration_s=round(duration, 2)))

                CHUNK = 1920 * 10
                for i in range(0, len(audio), CHUNK):
                    await ws.send(audio[i: i + CHUNK].tobytes())
                    await asyncio.sleep(0)

        gen_task = asyncio.create_task(_render_and_stream())

        try:
            import websockets.exceptions
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                import json
                try:
                    event = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                if event.get("service") != self.SERVICE:
                    continue

                a = event.get("action")
                if a == "stop":
                    break
                elif a == "prompt":
                    new_prompt = event.get("text", "")
                    print(f"[magenta] prompt update: {new_prompt!r}", flush=True)
                    if self._use_modal:
                        style_bytes = await self._modal.embed_style.remote.aio(new_prompt)
                    else:
                        style = self._system.embed_style(new_prompt)
                    state = None
                elif a == "note_on":
                    pitch = int(event.get("pitch", 0)) & 0x7F
                    active_notes[pitch] = True
                elif a == "note_off":
                    pitch = int(event.get("pitch", 0)) & 0x7F
                    active_notes.pop(pitch, None)
                elif a == "drum":
                    pass  # reserved for future drum conditioning

        except Exception:
            pass
        finally:
            gen_task.cancel()
            await ws.send(msg(self.SERVICE, "stopped"))
