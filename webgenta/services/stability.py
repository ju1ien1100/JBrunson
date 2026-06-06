"""Stability.ai audio service — background tracks.

Uses the Stable Audio API to generate full background music tracks from a text prompt.

Inbound actions:
  { "service": "stability", "action": "generate",
    "prompt": "epic orchestral battle music, cinematic",
    "duration": 30,          # seconds (default 30, max 180)
    "negative_prompt": "vocals, lyrics"  # optional
  }

Outbound events:
  { "service": "stability", "event": "submitted",  "id": "..." }
  { "service": "stability", "event": "status",     "id": "...", "state": "..." }
  { "service": "stability", "event": "complete",   "id": "...", "audio_url": "..." }
  { "service": "stability", "event": "error",      "id": "...", "message": "..." }

Docs: https://platform.stability.ai/docs/api-reference#tag/Generate/paths/~1v2beta~1audio~1stable-audio~1generate/post
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import aiohttp

from services import msg

if TYPE_CHECKING:
    import websockets

STABILITY_BASE = "https://api.stability.ai"
POLL_INTERVAL = 4.0    # seconds between status polls
POLL_TIMEOUT  = 180.0  # give up after 3 minutes


class StabilityService:
    SERVICE = "stability"

    def __init__(self):
        self._api_key = os.environ.get("STABILITY_API_KEY", "")
        if not self._api_key:
            print("[stability] WARNING: STABILITY_API_KEY not set — requests will fail", flush=True)

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Accept": "application/json",
        }

    async def handle(self, ws: "websockets.WebSocketServerProtocol", action: dict) -> None:
        if action.get("action") == "generate":
            asyncio.create_task(self._generate(ws, action))

    async def _generate(self, ws, action: dict) -> None:
        payload = {
            "prompt": action.get("prompt", "ambient background music"),
            "seconds_total": int(action.get("duration", 30)),
        }
        if "negative_prompt" in action:
            payload["negative_prompt"] = action["negative_prompt"]

        async with aiohttp.ClientSession() as session:
            try:
                async with session.post(
                    f"{STABILITY_BASE}/v2beta/audio/stable-audio/generate",
                    json=payload,
                    headers=self._headers,
                ) as resp:
                    data = await resp.json()
                    if resp.status not in (200, 201, 202):
                        await ws.send(msg(self.SERVICE, "error", message=data.get("message", str(data))))
                        return
            except Exception as e:
                await ws.send(msg(self.SERVICE, "error", message=str(e)))
                return

            # Stability returns the result directly or via a generation_id
            if "audio_url" in data:
                await ws.send(msg(self.SERVICE, "complete", audio_url=data["audio_url"]))
                return

            job_id = data.get("id", "")
            if not job_id:
                await ws.send(msg(self.SERVICE, "error", message=f"Stability response missing 'id': {data}"))
                return
            await ws.send(msg(self.SERVICE, "submitted", id=job_id))

            loop = asyncio.get_running_loop()
            deadline = loop.time() + POLL_TIMEOUT
            state = "unknown"
            while loop.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    async with session.get(
                        f"{STABILITY_BASE}/v2beta/audio/stable-audio/generate/{job_id}",
                        headers=self._headers,
                    ) as resp:
                        status_data = await resp.json()
                except Exception as e:
                    await ws.send(msg(self.SERVICE, "error", id=job_id, message=str(e)))
                    return

                state = status_data.get("status", "unknown")
                await ws.send(msg(self.SERVICE, "status", id=job_id, state=state))

                if state in ("complete", "succeeded"):
                    await ws.send(msg(
                        self.SERVICE, "complete",
                        id=job_id,
                        audio_url=status_data.get("audio_url", ""),
                    ))
                    return
                elif state == "error":
                    await ws.send(msg(
                        self.SERVICE, "error",
                        id=job_id,
                        message=status_data.get("message", "unknown error"),
                    ))
                    return

            await ws.send(msg(self.SERVICE, "error", id=job_id,
                              message=f"Timed out after {int(POLL_TIMEOUT)}s — job still in state '{state}'"))
