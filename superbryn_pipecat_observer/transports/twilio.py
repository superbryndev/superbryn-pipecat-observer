"""
Twilio recording adapter.

Twilio call recordings are owned by Twilio, not by Pipecat. The flow is:

  1. The customer must enable recording on the Twilio side, either:
       - On call creation: `record="true"` (or `"record-from-answer"`)
       - On TwiML response: `<Record>` verb
     This adapter does NOT start recording — Twilio doesn't expose a mid-call
     "start recording" API that's safe to call from a Pipecat WebSocket
     handler. Configuring recording is a Twilio-side concern.

  2. We capture the `CallSid` from the Twilio Media Streams WebSocket. The
     `start` event sent by Twilio includes `streamSid` and `callSid`. Pipecat's
     Twilio serializer typically exposes one or both as attributes after the
     stream starts. We sniff several common locations.

  3. After the pipeline ends, we poll Twilio's REST API for the call's
     recordings:
        GET /2010-04-01/Accounts/{AccountSid}/Calls/{CallSid}/Recordings.json
     Twilio finalizes recordings asynchronously (typically 5-30 seconds after
     hangup), so we retry with backoff.

  4. The recording media URL is built as:
        https://api.twilio.com{recording.uri.replace('.json', '.mp3')}
     authenticated via HTTP Basic with the Twilio account credentials.

Required env vars:
    TWILIO_ACCOUNT_SID
    TWILIO_AUTH_TOKEN

Optional override:
    TWILIO_CALL_SID — useful if the SID isn't sniffable from the transport
    (some customers route Twilio through their own dispatcher and only have
    the SID in their backend context).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..observer import SuperbrynObserver

logger = logging.getLogger("superbryn_pipecat_observer.transports.twilio")

_RECORDINGS_URL = (
    "https://api.twilio.com/2010-04-01/Accounts/{sid}/Calls/{call_sid}/Recordings.json"
)
_RECORDING_RETRIES = 5
_RECORDING_RETRY_BACKOFF_SEC = 3.0


class TwilioRecordingAdapter:
    """Backfills the Twilio recording URL after the call ends."""

    transport_name = "twilio"

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.account_sid = os.getenv("TWILIO_ACCOUNT_SID", "")
        self.auth_token = os.getenv("TWILIO_AUTH_TOKEN", "")
        self.call_sid: str | None = os.getenv("TWILIO_CALL_SID") or None

        if not (self.account_sid and self.auth_token):
            logger.warning(
                "TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN not set — Twilio recording URL "
                "will not be fetched. Set both env vars to enable auto-wiring.",
            )

    async def start(self, observer: SuperbrynObserver) -> None:
        # Twilio recordings are configured server-side at call creation; we
        # don't toggle them here. The only thing we do at start-of-call is
        # try to sniff the CallSid so we can fetch the recording later.
        sniffed = _sniff_call_sid(self.transport)
        if sniffed:
            self.call_sid = sniffed
            logger.info("Twilio CallSid sniffed from transport: %s", sniffed)
        elif self.call_sid:
            logger.info("Twilio CallSid from env: %s", self.call_sid)
        else:
            logger.info(
                "Twilio CallSid not yet known — will retry sniffing at finalize. "
                'Pass `extra_metadata={"call_sid": ...}` if sniffing fails.',
            )

    async def finalize(self, observer: SuperbrynObserver) -> None:
        # Retry sniff at finalize — by now the transport has seen the Twilio
        # `start` message and the CallSid is more likely to be available.
        if not self.call_sid:
            self.call_sid = _sniff_call_sid(self.transport) or observer.extra_metadata.get(
                "call_sid"
            )

        if not self.call_sid:
            logger.info(
                "Twilio CallSid still unknown at finalize — skipping recording fetch. "
                'Pass `extra_metadata={"call_sid": ...}` to enable.',
            )
            return
        if not (self.account_sid and self.auth_token):
            return

        url = await self._fetch_recording_url(self.call_sid)
        if url:
            observer.recording_url = url
            logger.info("Twilio recording URL attached: %s", url)

    async def _fetch_recording_url(self, call_sid: str) -> str | None:
        """Poll Twilio for the call's recording; return the first ready MP3 URL."""
        try:
            import aiohttp  # lazy
        except ImportError:
            logger.error("aiohttp not installed — cannot fetch Twilio recording URL")
            return None

        endpoint = _RECORDINGS_URL.format(sid=self.account_sid, call_sid=call_sid)
        auth = aiohttp.BasicAuth(self.account_sid, self.auth_token)

        for attempt in range(_RECORDING_RETRIES):
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(
                        endpoint,
                        auth=auth,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            recordings = data.get("recordings") or []
                            for rec in recordings:
                                if rec.get("status") == "completed" and rec.get("uri"):
                                    # Build playable MP3 URL. Twilio's `uri` is
                                    # the resource path with `.json` suffix.
                                    base = rec["uri"].replace(".json", ".mp3")
                                    return f"https://api.twilio.com{base}"
                            logger.debug(
                                "Twilio recordings not ready yet (attempt=%d, count=%d)",
                                attempt + 1,
                                len(recordings),
                            )
                        elif resp.status in (401, 403):
                            logger.error(
                                "Twilio auth failed (%d) — check TWILIO_ACCOUNT_SID / TWILIO_AUTH_TOKEN",
                                resp.status,
                            )
                            return None
                        else:
                            body = await resp.text()
                            logger.warning("Twilio HTTP %d: %s", resp.status, body[:200])
            except Exception as exc:  # noqa: BLE001
                logger.debug("Twilio fetch attempt %d failed: %s", attempt + 1, exc)

            if attempt < _RECORDING_RETRIES - 1:
                await asyncio.sleep(_RECORDING_RETRY_BACKOFF_SEC * (attempt + 1))

        logger.info(
            "Twilio recording not available after %d attempts. CallSid=%s — "
            "Twilio may still be finalizing; backfill manually if needed.",
            _RECORDING_RETRIES,
            call_sid,
        )
        return None


def _sniff_call_sid(transport: Any) -> str | None:
    """
    Best-effort CallSid extraction from a Pipecat Twilio-flavored transport.

    Pipecat doesn't expose this through a stable API, so we walk a few
    well-known locations:

      - Direct attributes: ``call_sid``, ``callSid``, ``_call_sid``
      - The serializer, found at ``transport._serializer``,
        ``transport.serializer``, or ``transport._params.serializer``
        (where ``FastAPIWebsocketTransport`` keeps it).
      - ``transport._call_data["start"]["callSid"]`` if the raw start
        payload is stashed there.
    """
    if transport is None:
        return None

    candidates = ("call_sid", "callSid", "_call_sid")
    for name in candidates:
        val = getattr(transport, name, None)
        if val:
            return str(val)

    serializers: list[Any] = []
    for ser_attr in ("_serializer", "serializer"):
        ser = getattr(transport, ser_attr, None)
        if ser is not None:
            serializers.append(ser)
    params = getattr(transport, "_params", None) or getattr(transport, "params", None)
    if params is not None:
        ser = getattr(params, "serializer", None) or getattr(params, "_serializer", None)
        if ser is not None:
            serializers.append(ser)

    for ser in serializers:
        for name in candidates:
            val = getattr(ser, name, None)
            if val:
                return str(val)

    # Last resort — some Pipecat versions stash the raw `start` event payload.
    raw = getattr(transport, "_call_data", None) or getattr(transport, "_start_data", None)
    if isinstance(raw, dict):
        start = raw.get("start") or raw
        if isinstance(start, dict):
            val = start.get("callSid") or start.get("call_sid")
            if val:
                return str(val)

    return None
