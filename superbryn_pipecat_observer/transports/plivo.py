"""
Plivo recording adapter.

Mirrors the Twilio pattern: the customer enables recording on the Plivo
side (via the `<Record>` verb in their answer XML, or by calling Plivo's
`start_recording` REST API mid-call). This adapter does **not** start
recording — Plivo, like Twilio, treats recording configuration as a
provider-side concern.

Flow:
  1. Sniff `CallUUID` from the Pipecat WebSocket transport. Plivo's Media
     Streams sends `{event: "start", start: {callUuid: "..."}}` over the
     WS, so the underlying serializer typically stashes it on the
     transport / serializer object once received.
  2. After the pipeline ends, GET Plivo's per-call Recording list:
        https://api.plivo.com/v1/Account/{AUTH_ID}/Call/{call_uuid}/Recording/
     Plivo finalizes recordings shortly after hangup; we retry with
     backoff to give the storage layer a few seconds.
  3. Pluck `objects[0].recording_url` (Plivo serves an authenticated MP3
     URL from their S3 bucket — no extra signing needed when fetched via
     the customer's account credentials).

Required env vars:
    PLIVO_AUTH_ID
    PLIVO_AUTH_TOKEN

Optional override:
    PLIVO_CALL_UUID — useful if sniffing fails for a particular Pipecat
    + Plivo serializer combination (e.g. customers running Plivo through
    their own dispatcher and only holding the CallUUID in their backend).
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..observer import SuperbrynObserver

logger = logging.getLogger("superbryn_pipecat_observer.transports.plivo")

_RECORDINGS_URL = "https://api.plivo.com/v1/Account/{auth_id}/Call/{call_uuid}/Recording/"
_RECORDING_RETRIES = 5
_RECORDING_RETRY_BACKOFF_SEC = 3.0


class PlivoRecordingAdapter:
    """Backfills the Plivo recording URL after the call ends."""

    transport_name = "plivo"

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.auth_id = os.getenv("PLIVO_AUTH_ID", "")
        self.auth_token = os.getenv("PLIVO_AUTH_TOKEN", "")
        self.call_uuid: str | None = os.getenv("PLIVO_CALL_UUID") or None

        if not (self.auth_id and self.auth_token):
            logger.warning(
                "PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN not set — Plivo recording URL "
                "will not be fetched. Set both env vars to enable auto-wiring.",
            )

    async def start(self, observer: SuperbrynObserver) -> None:
        # Plivo recordings are configured server-side (in the answer XML or
        # via the start_recording REST call). We don't toggle them here.
        sniffed = _sniff_call_uuid(self.transport)
        if sniffed:
            self.call_uuid = sniffed
            logger.info("Plivo CallUUID sniffed from transport: %s", sniffed)
        elif self.call_uuid:
            logger.info("Plivo CallUUID from env: %s", self.call_uuid)
        else:
            logger.info(
                "Plivo CallUUID not yet known — will retry sniffing at finalize. "
                'Pass `extra_metadata={"call_uuid": ...}` if sniffing fails.',
            )

    async def finalize(self, observer: SuperbrynObserver) -> None:
        if not self.call_uuid:
            self.call_uuid = _sniff_call_uuid(self.transport) or observer.extra_metadata.get(
                "call_uuid"
            )

        if not self.call_uuid:
            logger.info(
                "Plivo CallUUID still unknown at finalize — skipping recording fetch. "
                'Pass `extra_metadata={"call_uuid": ...}` to enable.',
            )
            return
        if not (self.auth_id and self.auth_token):
            return

        url = await self._fetch_recording_url(self.call_uuid)
        if url:
            observer.recording_url = url
            logger.info("Plivo recording URL attached: %s", url)

    async def _fetch_recording_url(self, call_uuid: str) -> str | None:
        """Poll Plivo for the call's recordings; return the first finalized URL."""
        try:
            import aiohttp  # lazy
        except ImportError:
            logger.error("aiohttp not installed — cannot fetch Plivo recording URL")
            return None

        endpoint = _RECORDINGS_URL.format(auth_id=self.auth_id, call_uuid=call_uuid)
        auth = aiohttp.BasicAuth(self.auth_id, self.auth_token)

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
                            objects = data.get("objects") or []
                            for rec in objects:
                                url = rec.get("recording_url")
                                if url:
                                    return str(url)
                            logger.debug(
                                "Plivo recordings not ready yet (attempt=%d, count=%d)",
                                attempt + 1,
                                len(objects),
                            )
                        elif resp.status == 404:
                            # Plivo returns 404 while the recording is still
                            # being written — perfectly normal during the
                            # first few retries after hangup.
                            logger.debug(
                                "Plivo recording 404 (attempt=%d) — likely still finalizing",
                                attempt + 1,
                            )
                        elif resp.status in (401, 403):
                            logger.error(
                                "Plivo auth failed (%d) — check PLIVO_AUTH_ID / PLIVO_AUTH_TOKEN",
                                resp.status,
                            )
                            return None
                        else:
                            body = await resp.text()
                            logger.warning("Plivo HTTP %d: %s", resp.status, body[:200])
            except Exception as exc:  # noqa: BLE001
                logger.debug("Plivo fetch attempt %d failed: %s", attempt + 1, exc)

            if attempt < _RECORDING_RETRIES - 1:
                await asyncio.sleep(_RECORDING_RETRY_BACKOFF_SEC * (attempt + 1))

        logger.info(
            "Plivo recording not available after %d attempts. CallUUID=%s — "
            "Plivo may still be finalizing; backfill manually if needed.",
            _RECORDING_RETRIES,
            call_uuid,
        )
        return None


def _sniff_call_uuid(transport: Any) -> str | None:
    """
    Best-effort CallUUID extraction from a Pipecat Plivo-flavored transport.

    Plivo's WebSocket start event carries `callUuid`. We walk a few
    well-known locations:

      - Direct attributes on the transport.
      - The serializer, found at either ``transport._serializer``,
        ``transport.serializer``, or ``transport._params.serializer``
        (the last is where FastAPIWebsocketTransport keeps it).
      - The raw ``start`` event payload, if the transport happens to
        stash it.

    Pipecat's ``PlivoFrameSerializer`` stores the CallUUID as
    ``_call_id`` rather than ``_call_uuid``, so we walk both naming
    conventions to stay robust across versions.
    """
    if transport is None:
        return None

    candidates = (
        "call_uuid", "callUuid", "_call_uuid",
        "call_id", "callId", "_call_id",
    )

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

    raw = getattr(transport, "_call_data", None) or getattr(transport, "_start_data", None)
    if isinstance(raw, dict):
        start = raw.get("start") or raw
        if isinstance(start, dict):
            val = (
                start.get("callUuid")
                or start.get("call_uuid")
                or start.get("callId")
                or start.get("call_id")
            )
            if val:
                return str(val)

    return None
