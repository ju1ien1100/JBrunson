"""Base types and message helpers shared across all services."""

from __future__ import annotations
import json
from typing import Any


def msg(service: str, event: str, **kwargs: Any) -> str:
    """Serialize an outbound JSON control message."""
    return json.dumps({"service": service, "event": event, **kwargs})
