"""
Transport recording adapters.

Pipecat itself has no unified recording API — each transport records audio
differently or not at all. This subpackage isolates that per-transport logic
so the observer stays transport-agnostic.

Supported transports today (native recording auto-fetch):
  - `DailyTransport`                       — Daily Cloud recording
  - `FastAPIWebsocketTransport` + Twilio   — Twilio REST recordings
  - `FastAPIWebsocketTransport` + Plivo    — Plivo REST recordings
  - `FastAPIWebsocketTransport` + Vobiz    — Vobiz REST recordings

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
from .plivo import PlivoRecordingAdapter
from .twilio import TwilioRecordingAdapter
from .vobiz import VobizRecordingAdapter

logger = logging.getLogger("superbryn_pipecat_observer.transports")

__all__ = [
    "RecordingAdapter",
    "DailyRecordingAdapter",
    "PlivoRecordingAdapter",
    "TwilioRecordingAdapter",
    "VobizRecordingAdapter",
    "get_recording_adapter",
]

# Carriers that ride on top of `FastAPIWebsocketTransport` are distinguished
# only by which serializer is attached. We probe the transport's class /
# module first, then the serializer's class / module. First match wins.
_CARRIER_SIGNATURES: tuple[tuple[str, type[RecordingAdapter]], ...] = (
    ("twilio", TwilioRecordingAdapter),
    ("plivo", PlivoRecordingAdapter),
    ("vobiz", VobizRecordingAdapter),
)


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
        # Daily owns its own transport class — direct match.
        if cls_name == "DailyTransport" or "daily" in module:
            return DailyRecordingAdapter(transport)

        # Carriers (Twilio / Plivo / Vobiz) all ride on
        # FastAPIWebsocketTransport and are distinguished by the serializer.
        # Probe transport identifiers first, then the attached serializer.
        for tag, AdapterCls in _CARRIER_SIGNATURES:
            if tag in module or tag in cls_name.lower():
                return AdapterCls(transport)

        # Pipecat's FastAPI/WebSocket transports stash the serializer inside
        # their `params` object (`transport._params.serializer`), not as a
        # direct attribute. We try both layouts so any transport that exposes
        # a serializer — directly or via params — can still be sniffed.
        serializer = (
            getattr(transport, "_serializer", None)
            or getattr(transport, "serializer", None)
        )
        if serializer is None:
            for params_attr in ("_params", "params"):
                params_obj = getattr(transport, params_attr, None)
                if params_obj is None:
                    continue
                serializer = getattr(params_obj, "serializer", None) or getattr(
                    params_obj, "_serializer", None
                )
                if serializer is not None:
                    break

        if serializer is not None:
            ser_module = (type(serializer).__module__ or "").lower()
            ser_cls = type(serializer).__name__.lower()
            for tag, AdapterCls in _CARRIER_SIGNATURES:
                if tag in ser_module or tag in ser_cls:
                    return AdapterCls(transport)

    except Exception as exc:  # noqa: BLE001 — detection must never raise
        logger.debug("transport adapter detection failed: %s", exc)

    return None
