"""Stability service — background tracks via Stable Audio 3 Small on Modal GPU.

Inbound actions:
  { "service": "stability", "action": "generate",
    "prompt": "epic orchestral battle music, cinematic",
    "duration": 30   # seconds (default 30, max 180)
  }

Outbound events:
  { "service": "stability", "event": "status",   "state": "generating" }
  { "service": "stability", "event": "complete" }
  { "service": "stability", "event": "error",    "message": "..." }
  <binary>  raw WAV bytes (stereo, 44100 Hz) — sent between status and complete
"""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING

from services import msg

if TYPE_CHECKING:
    import websockets


class StabilityService:
    SERVICE = "stability"

    def __init__(self, modal_inference=None):
        self._modal = modal_inference
        if modal_inference is None:
            print("[stability] No Modal backend — generate requests will error", flush=True)

    async def handle(self, ws: "websockets.WebSocketServerProtocol", action: dict) -> None:
        if action.get("action") == "generate":
            asyncio.create_task(self._generate(ws, action))

    async def _generate(self, ws, action: dict) -> None:
        if self._modal is None:
            await ws.send(msg(self.SERVICE, "error", message="Stability Modal backend not configured — start server with --modal-stability"))
            return

        prompt = action.get("prompt", "ambient background music")
        duration = max(1, min(180, int(action.get("duration", 30))))

        await ws.send(msg(self.SERVICE, "status", state="generating"))
        try:
            wav_bytes = await self._modal.generate.remote.aio(prompt, duration)
        except Exception as e:
            await ws.send(msg(self.SERVICE, "error", message=str(e)))
            return

        # Send raw WAV as a binary frame, then signal completion
        await ws.send(wav_bytes)
        await ws.send(msg(self.SERVICE, "complete"))
