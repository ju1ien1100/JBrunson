"""Suno API service — sound effects and vocal tracks.

Inbound actions:
  { "service": "suno", "action": "generate",
    "description": "sword clash metal clang",          # simple mode
    "title": "optional title" }

  { "service": "suno", "action": "generate",
    "style": "dreampop, reverb guitars",               # custom mode
    "lyrics": "[Verse]\\nWords...",
    "instrumental": true,
    "voice_id": "5b915c6d-8d96-416c-9755-eba65868cfef"  # optional preset voice
  }

  { "service": "suno", "action": "cover",
    "source_id": "<clip id>",
    "style": "acoustic folk",
    "lyrics": "optional new lyrics" }

  { "service": "suno", "action": "cancel", "id": "<job id>" }

Outbound events:
  { "service": "suno", "event": "submitted",  "id": "..." }
  { "service": "suno", "event": "status",     "id": "...", "state": "queued|streaming|complete|error" }
  { "service": "suno", "event": "complete",   "id": "...", "audio_url": "...", "title": "..." }
  { "service": "suno", "event": "error",      "id": "...", "message": "..." }

Preset voice IDs:
  female:   5b915c6d-8d96-416c-9755-eba65868cfef
  weird kid: c036ce3a-55e4-4690-9b8d-4516b37a96d5
  low male: 27f5465b-73c3-4134-b11e-70b0bd571c6c
"""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

import aiohttp

from services import msg

if TYPE_CHECKING:
    import websockets

SUNO_BASE = "https://api.suno.com"
POLL_INTERVAL = 3.0   # seconds between status polls
POLL_TIMEOUT  = 300.0 # give up after 5 minutes


class SunoService:
    SERVICE = "suno"

    def __init__(self):
        self._api_key = os.environ.get("SUNO_API_KEY", "")
        if not self._api_key:
            print("[suno] WARNING: SUNO_API_KEY not set — requests will fail", flush=True)

    @property
    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }

    async def handle(self, ws: "websockets.WebSocketServerProtocol", action: dict) -> None:
        a = action.get("action")
        if a == "generate":
            asyncio.create_task(self._generate(ws, action))
        elif a == "cover":
            asyncio.create_task(self._cover(ws, action))
        # cancel is fire-and-forget — Suno has no cancel endpoint yet; just ignore

    async def _generate(self, ws, action: dict) -> None:
        payload: dict = {}

        if "description" in action:
            # Simple mode
            payload["description"] = action["description"]
        else:
            # Custom mode
            if "lyrics" in action:
                payload["lyrics"] = action["lyrics"]
            if "style" in action:
                payload["style"] = action["style"]
            if action.get("instrumental"):
                payload["instrumental"] = True

        if "title" in action:
            payload["title"] = action["title"]
        if "voice_id" in action:
            payload["voice_id"] = action["voice_id"]

        await self._submit_and_poll(ws, "/v0/audio", payload)

    async def _cover(self, ws, action: dict) -> None:
        source_id = action.get("source_id", "")
        payload = {}
        if "lyrics" in action:
            payload["lyrics"] = action["lyrics"]
        if "style" in action:
            payload["style"] = action["style"]
        if "voice_id" in action:
            payload["voice_id"] = action["voice_id"]

        await self._submit_and_poll(ws, f"/v0/audio/{source_id}/covers", payload)

    async def _submit_and_poll(self, ws, endpoint: str, payload: dict) -> None:
        async with aiohttp.ClientSession() as session:
            # Submit
            try:
                async with session.post(
                    f"{SUNO_BASE}{endpoint}",
                    json=payload,
                    headers=self._headers,
                ) as resp:
                    data = await resp.json()
                    if resp.status not in (200, 201, 202):
                        await ws.send(msg(self.SERVICE, "error", message=data.get("error", str(data))))
                        return
            except Exception as e:
                await ws.send(msg(self.SERVICE, "error", message=str(e)))
                return

            job_id = data.get("id", "")
            if not job_id:
                await ws.send(msg(self.SERVICE, "error", message=f"Suno response missing 'id': {data}"))
                return
            await ws.send(msg(self.SERVICE, "submitted", id=job_id))

            # Poll until complete, error, or timeout
            loop = asyncio.get_running_loop()
            deadline = loop.time() + POLL_TIMEOUT
            while loop.time() < deadline:
                await asyncio.sleep(POLL_INTERVAL)
                try:
                    async with session.get(
                        f"{SUNO_BASE}/v0/audio/{job_id}",
                        headers=self._headers,
                    ) as resp:
                        status_data = await resp.json()
                except Exception as e:
                    await ws.send(msg(self.SERVICE, "error", id=job_id, message=str(e)))
                    return

                state = status_data.get("status", "unknown")
                await ws.send(msg(self.SERVICE, "status", id=job_id, state=state))

                if state == "complete":
                    await ws.send(msg(
                        self.SERVICE, "complete",
                        id=job_id,
                        audio_url=status_data.get("audio_url", ""),
                        title=status_data.get("title", ""),
                    ))
                    return
                elif state == "error":
                    await ws.send(msg(
                        self.SERVICE, "error",
                        id=job_id,
                        message=status_data.get("error", "unknown error"),
                    ))
                    return

            await ws.send(msg(self.SERVICE, "error", id=job_id,
                              message=f"Timed out after {int(POLL_TIMEOUT)}s — job still in state '{state}'"))
