"""Webgenta WebSocket Orchestration Server

Routes inbound JSON messages to the appropriate service and streams responses back.

Message format (client → server):
  { "service": "magenta" | "suno" | "stability", "action": "...", ...payload }

Binary messages from client are ignored at the router level.

Usage:
  python ws_server.py [--host 0.0.0.0] [--port 8765] [--model mrt2_small] [--no-magenta]

Environment variables (set in .env or shell before running):
  SUNO_API_KEY       — Suno bearer token (sk_live_...)
  STABILITY_API_KEY  — Stability.ai bearer token
"""

import os
import sys

# Must be set before any JAX import
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import argparse
import asyncio
import json
import logging
from pathlib import Path

import numpy as np
import websockets
import websockets.exceptions

# ── Logging ───────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

# ── Load .env if present ──────────────────────────────────────────────────────

_env_file = Path(__file__).parent / ".env"
if _env_file.exists():
    for line in _env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())

# ── MRT2 import ───────────────────────────────────────────────────────────────

def _import_magenta():
    try:
        from magenta_rt.jax.system import MagentaRT2System
        return MagentaRT2System
    except ModuleNotFoundError:
        _root = Path(__file__).parent.parent / "magenta-realtime"
        sys.path.insert(0, str(_root))
        from magenta_rt.jax.system import MagentaRT2System
        return MagentaRT2System

# ── Router ────────────────────────────────────────────────────────────────────

class Router:
    def __init__(self, services: dict):
        self._services = services  # { "magenta": MagentaService, ... }

    async def __call__(self, ws):
        remote = ws.remote_address
        logger.info("Client connected: %s", remote)
        try:
            async for raw in ws:
                if not isinstance(raw, str):
                    continue
                try:
                    message = json.loads(raw)
                except json.JSONDecodeError:
                    logger.warning("Malformed message from %s: %r", remote, raw[:120])
                    continue

                service_name = message.get("service")
                service = self._services.get(service_name)

                if service is None:
                    await ws.send(json.dumps({
                        "service": service_name,
                        "event": "error",
                        "message": f"Unknown service '{service_name}'. Available: {list(self._services)}",
                    }))
                    continue

                try:
                    await service.handle(ws, message)
                except websockets.exceptions.ConnectionClosed:
                    break
                except Exception as e:
                    logger.exception("Error in service '%s'", service_name)
                    await ws.send(json.dumps({
                        "service": service_name,
                        "event": "error",
                        "message": str(e),
                    }))

        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            logger.info("Client disconnected: %s", remote)


# ── Entry point ───────────────────────────────────────────────────────────────

async def main(host: str, port: int, services: dict):
    router = Router(services)
    print(f"\nWebgenta WS server listening on ws://{host}:{port}", flush=True)
    print(f"Active services: {list(services)}", flush=True)
    async with websockets.serve(router, host, port, max_size=2 ** 20):
        await asyncio.Future()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Webgenta WebSocket orchestration server")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--model", default="mrt2_small", choices=["mrt2_small", "mrt2_base"])
    parser.add_argument("--no-magenta", action="store_true", help="Skip MRT2 entirely")
    parser.add_argument("--modal", action="store_true", help="Use Modal GPU for MRT2 instead of local JAX")
    parser.add_argument("--modal-stability", action="store_true", help="Use Modal GPU for Stable Audio 3 background tracks")
    args = parser.parse_args()

    from services.suno import SunoService
    from services.stability import StabilityService
    from services.magenta import MagentaService

    if args.modal_stability:
        print("Connecting to Modal GPU for Stable Audio 3...", flush=True)
        import modal
        StableAudioInference = modal.Cls.from_name("webgenta-stability", "StableAudioInference")
        services = {
            "suno": SunoService(),
            "stability": StabilityService(modal_inference=StableAudioInference()),
        }
        print("Modal Stable Audio 3 ready (GPU warms up on first request).", flush=True)
    else:
        services = {
            "suno": SunoService(),
            "stability": StabilityService(),
        }

    if not args.no_magenta:
        if args.modal:
            print("Connecting to Modal GPU for MRT2...", flush=True)
            import modal
            MagentaInference = modal.Cls.from_name("webgenta-magenta", "MagentaInference")
            modal_inference = MagentaInference()
            services["magenta"] = MagentaService(modal_inference=modal_inference)
            print("Modal MRT2 ready (GPU will warm up on first request).", flush=True)
        else:
            print(f"Loading MRT2 model: {args.model} (takes ~15s on CPU)...", flush=True)
            MagentaRT2System = _import_magenta()
            system = MagentaRT2System(size=args.model)
            print("MRT2 model ready.", flush=True)
            services["magenta"] = MagentaService(system=system)
    else:
        print("Skipping MRT2 (--no-magenta). Suno + Stability only.", flush=True)

    asyncio.run(main(args.host, args.port, services))
