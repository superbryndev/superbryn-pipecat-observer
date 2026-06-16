"""
Transport recording adapters.

Pipecat itself has no unified recording API — each transport records audio
differently or not at all. This subpackage isolates that per-transport logic
so the observer stays transport-agnostic.

Supported transports today:
  - `DailyTransport`                       — Daily Cloud recording
  - `FastAPIWebsocketTransport` + Twilio   — Twilio REST recordings

Not supported (no transport-side recording mechanism):
  - `WebsocketServerTransport` — generic WS server
  - `SmallWebRTCTransport`     — direct browser↔agent WebRTC
  - `LocalAudioTransport`      — dev / local mic

Out of scope here:
  - `LiveKitTransport`         — use the dedicated `livekit-evals` package,
    which integrates with LiveKit Agents more deeply than a Pipecat observer
    can.

Public surface:
    get_recording_adapter(transport) -> RecordingAdapter | None
        Sniff the transport instance and return an adapter that knows how to
        start/finalize recording for it. Returns None for unknown transports
        (e.g. raw WebRTC / generic WebSocket) — the observer then falls back
        to whatever `recording_url=...` the caller passed explicitly.

Detection is by class name + module path (not isinstance) so a Pipecat
upgrade that shuffles internal class paths doesn't hard-break us.
"""

from __future__ import annotations

import logging
from typing import Any

from .base import RecordingAdapter
from .daily import DailyRecordingAdapter
from .twilio import TwilioRecordingAdapter

logger = logging.getLogger("superbryn_pipecat_observer.transports")

__all__ = [
    "RecordingAdapter",
    "DailyRecordingAdapter",
    "TwilioRecordingAdapter",
    "get_recording_adapter",
]


def get_recording_adapter(transport: Any) -> RecordingAdapter | None:
    """
    Resolve a recording adapter for a Pipecat transport instance.

    Strings (legacy `transport="daily"`) are ignored — only live transport
    objects get auto-wired. Unknown transports return None so the caller
    can fall back to manual `recording_url=...`.
    """
    if transport is None or isinstance(transport, str):
        return None

    cls_name = type(transport).__name__
    module = (type(transport).__module__ or "").lower()

    try:
        # Daily: `DailyTransport` from `pipecat.transports.services.daily`
        if cls_name == "DailyTransport" or "daily" in module:
            return DailyRecordingAdapter(transport)

        # Twilio rides on top of FastAPIWebsocketTransport with a Twilio
        # serializer. We detect by sniffing for `twilio` anywhere in the
        # transport's class name, module, or attached serializer module.
        if "twilio" in module or "twilio" in cls_name.lower():
            return TwilioRecordingAdapter(transport)
        serializer = getattr(transport, "_serializer", None) or getattr(
            transport, "serializer", None
        )
        if serializer is not None:
            ser_module = (type(serializer).__module__ or "").lower()
            if "twilio" in ser_module or "twilio" in type(serializer).__name__.lower():
                return TwilioRecordingAdapter(transport)

    except Exception as exc:  # noqa: BLE001 — detection must never raise
        logger.debug("transport adapter detection failed: %s", exc)

    return None
