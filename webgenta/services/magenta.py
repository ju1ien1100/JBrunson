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
  { "service": "magenta", "action": "config",
      "strum": bool,            # true = retrigger held notes every frame (arpeggiate/strum/bow)
      "cfg_musiccoca": float,   # style adherence 0-7 (default 5.0)
      "cfg_notes": float,       # note-conditioning adherence 0-7 (default 3.0)
      "cfg_drums": float        # drum-conditioning adherence 0-7 (default 1.0)
  }

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


CHUNK_FRAMES = 25  # frames per render call (~1 second of audio at 25 Hz)

# High CFG adherence per creator recommendations
DEFAULT_CFG_MUSICCOCA = 5.0  # style adherence (range -1..7, default lib=3.0)
DEFAULT_CFG_NOTES = 3.0      # note-conditioning adherence (range -1..7, default lib=1.0)
DEFAULT_CFG_DRUMS = 1.0


def _sustained_notes(notes: list, strum: bool) -> list:
    """Convert onset notes to sustained or keep as onset depending on strum mode.

    strum=True  → keep 2 (onset) every frame so model can retrigger/arpeggiate
    strum=False → convert 2→1 (continuation) after first frame
    """
    if strum:
        return list(notes)
    return [min(v, 1) for v in notes]  # 2→1, 1→1, 0→0


def _render_chunk_local(
    system, style, notes: list, state: object,
    strum: bool = False,
    cfg_musiccoca: float = DEFAULT_CFG_MUSICCOCA,
    cfg_notes: float = DEFAULT_CFG_NOTES,
    cfg_drums: float = DEFAULT_CFG_DRUMS,
) -> tuple[np.ndarray, object]:
    """Render one chunk locally with current note conditioning."""
    sustain = _sustained_notes(notes, strum)

    segments = [{"notes": notes, "frames": 1}]
    if CHUNK_FRAMES > 1:
        segments.append({"notes": sustain, "frames": CHUNK_FRAMES - 1})

    chunks = []
    for seg in segments:
        wf, state = system.generate(
            style=style,
            notes=seg["notes"],
            drums=[-1],
            frames=seg["frames"],
            state=state,
            cfg_musiccoca=cfg_musiccoca,
            cfg_notes=cfg_notes,
            cfg_drums=cfg_drums,
        )
        chunks.append(wf.samples)
    return np.concatenate(chunks, axis=0).astype(np.float32), state


async def _render_chunk_modal(
    modal_inference,
    session_id: str,
    notes: list,
    strum: bool = False,
) -> np.ndarray:
    """Render one chunk via Modal GPU. State lives on the container."""
    pcm_bytes = await modal_inference.render_chunk.remote.aio(
        session_id, notes, CHUNK_FRAMES, strum,
    )
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
        import uuid
        import json
        import websockets.exceptions

        await ws.send(msg(self.SERVICE, "status", state="embedding"))

        # Generation config — updated live via "config" action
        strum = False
        cfg_musiccoca = DEFAULT_CFG_MUSICCOCA
        cfg_notes_val = DEFAULT_CFG_NOTES
        cfg_drums = DEFAULT_CFG_DRUMS

        state = None   # local JAX state (unused in modal path)
        style = None
        session_id = None

        if self._use_modal:
            # Each client gets a unique session ID; state lives on the Modal container.
            session_id = str(uuid.uuid4())
            await self._modal.begin_session.remote.aio(
                session_id, initial_prompt, cfg_musiccoca, cfg_notes_val, cfg_drums,
            )
        else:
            style = self._system.embed_style(initial_prompt)

        loop_num = 0
        active_notes: dict[int, bool] = {}  # pitch → True=onset, False=sustained

        async def _render_and_stream():
            nonlocal state, loop_num
            while True:
                loop_num += 1
                t0 = time.perf_counter()

                notes = _build_notes_vector(active_notes)

                if self._use_modal:
                    audio = await _render_chunk_modal(
                        self._modal, session_id, notes, strum=strum,
                    )
                else:
                    audio, state = await asyncio.get_running_loop().run_in_executor(
                        None, _render_chunk_local, self._system, style, notes, state,
                        strum, cfg_musiccoca, cfg_notes_val, cfg_drums,
                    )

                # Advance onset notes to sustained for next chunk
                for pitch in list(active_notes):
                    if active_notes[pitch]:
                        active_notes[pitch] = False

                elapsed = time.perf_counter() - t0
                duration = audio.shape[0] / 48000
                backend = "modal/gpu" if self._use_modal else "local/cpu"
                print(f"[magenta] chunk {loop_num} ({backend}): {duration:.1f}s in {elapsed:.1f}s", flush=True)
                await ws.send(msg(self.SERVICE, "ready", loop=loop_num, duration_s=round(duration, 2)))

                STREAM_CHUNK = 1920 * 10
                for i in range(0, len(audio), STREAM_CHUNK):
                    await ws.send(audio[i: i + STREAM_CHUNK].tobytes())
                    await asyncio.sleep(0)

        gen_task = asyncio.create_task(_render_and_stream())

        try:
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
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
                        await self._modal.update_session.remote.aio(session_id, prompt=new_prompt)
                    else:
                        # style/state live in the outer _run scope, not the closure
                        style = self._system.embed_style(new_prompt)
                        state = None  # will be picked up by closure via nonlocal
                elif a == "config":
                    if "strum" in event:
                        strum = bool(event["strum"])
                    if "cfg_musiccoca" in event:
                        cfg_musiccoca = float(event["cfg_musiccoca"])
                    if "cfg_notes" in event:
                        cfg_notes_val = float(event["cfg_notes"])
                    if "cfg_drums" in event:
                        cfg_drums = float(event["cfg_drums"])
                    print(
                        f"[magenta] config: strum={strum} cfg_mc={cfg_musiccoca:.1f} "
                        f"cfg_n={cfg_notes_val:.1f} cfg_d={cfg_drums:.1f}",
                        flush=True,
                    )
                    if self._use_modal:
                        await self._modal.update_session.remote.aio(
                            session_id,
                            cfg_musiccoca=cfg_musiccoca,
                            cfg_notes=cfg_notes_val,
                            cfg_drums=cfg_drums,
                        )
                elif a == "note_on":
                    pitch = int(event.get("pitch", 0)) & 0x7F
                    active_notes[pitch] = True
                elif a == "note_off":
                    pitch = int(event.get("pitch", 0)) & 0x7F
                    active_notes.pop(pitch, None)
                elif a == "drum":
                    pass

        except Exception:
            pass
        finally:
            gen_task.cancel()
            if self._use_modal and session_id:
                try:
                    await self._modal.end_session.remote.aio(session_id)
                except Exception:
                    pass
            await ws.send(msg(self.SERVICE, "stopped"))
