"""
Vobiz recording adapter.

Vobiz is an India-focused voice platform. Recording lifecycle:

  1. Auth uses the ``X-Auth-ID`` + ``X-Auth-Token`` header pair.
  2. Recordings are listed via the global ``Recording`` endpoint, filtered
     by ``call_uuid`` — there is no per-call recording resource.

The customer enables recording on the Vobiz side (either by setting
``record=true`` in their answer XML or by POSTing to
``/Call/{call_uuid}/Record/``). This adapter then waits until the call
ends and fetches the resulting URL by polling the Recording endpoint.

Required env vars:
    VOBIZ_AUTH_ID
    VOBIZ_AUTH_TOKEN

Optional override:
    VOBIZ_CALL_UUID — explicit override if the CallUUID can't be sniffed
    from the Pipecat transport (e.g. when Vobiz traffic goes through a
    customer dispatcher).

Vobiz docs: https://docs.vobiz.ai/recording
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..observer import SuperbrynObserver

logger = logging.getLogger("superbryn_pipecat_observer.transports.vobiz")

# Vobiz exposes a global Recording endpoint and supports `?call_uuid=` as a
# filter; that lets us fetch without knowing the recording_id ahead of time.
_RECORDINGS_URL = "https://api.vobiz.ai/api/v1/Account/{auth_id}/Recording/"
_RECORDING_RETRIES = 5
_RECORDING_RETRY_BACKOFF_SEC = 3.0


class VobizRecordingAdapter:
    """Backfills the Vobiz recording URL after the call ends."""

    transport_name = "vobiz"

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.auth_id = os.getenv("VOBIZ_AUTH_ID", "")
        self.auth_token = os.getenv("VOBIZ_AUTH_TOKEN", "")
        self.call_uuid: str | None = os.getenv("VOBIZ_CALL_UUID") or None

        if not (self.auth_id and self.auth_token):
            logger.warning(
                "VOBIZ_AUTH_ID / VOBIZ_AUTH_TOKEN not set — Vobiz recording URL "
                "will not be fetched. Set both env vars to enable auto-wiring.",
            )

    async def start(self, observer: SuperbrynObserver) -> None:
        sniffed = _sniff_call_uuid(self.transport)
        if sniffed:
            self.call_uuid = sniffed
            logger.info("Vobiz CallUUID sniffed from transport: %s", sniffed)
        elif self.call_uuid:
            logger.info("Vobiz CallUUID from env: %s", self.call_uuid)
        else:
            logger.info(
                "Vobiz CallUUID not yet known — will retry sniffing at finalize. "
                'Pass `extra_metadata={"call_uuid": ...}` if sniffing fails.',
            )

    async def finalize(self, observer: SuperbrynObserver) -> None:
        if not self.call_uuid:
            self.call_uuid = _sniff_call_uuid(self.transport) or observer.extra_metadata.get(
                "call_uuid"
            )

        if not self.call_uuid:
            logger.info(
                "Vobiz CallUUID still unknown at finalize — skipping recording fetch. "
                'Pass `extra_metadata={"call_uuid": ...}` to enable.',
            )
            return
        if not (self.auth_id and self.auth_token):
            return

        url = await self._fetch_recording_url(self.call_uuid)
        if url:
            observer.recording_url = url
            logger.info("Vobiz recording URL attached: %s", url)

    async def _fetch_recording_url(self, call_uuid: str) -> str | None:
        """Poll Vobiz for a recording matching this call_uuid."""
        try:
            import aiohttp  # lazy
        except ImportError:
            logger.error("aiohttp not installed — cannot fetch Vobiz recording URL")
            return None

        endpoint = _RECORDINGS_URL.format(auth_id=self.auth_id)
        headers = {
            "X-Auth-ID": self.auth_id,
            "X-Auth-Token": self.auth_token,
            "Content-Type": "application/json",
        }
        params = {"call_uuid": call_uuid}

        for attempt in range(_RECORDING_RETRIES):
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(
                        endpoint,
                        headers=headers,
                        params=params,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            # Vobiz returns matching recordings under `objects`;
                            # fall back to `recordings` or a bare list for
                            # defensiveness across API revisions.
                            objects = (
                                data.get("objects")
                                or data.get("recordings")
                                or (data if isinstance(data, list) else [])
                            )
                            for rec in objects:
                                url = rec.get("recording_url") or rec.get("url")
                                if url:
                                    return str(url)
                            logger.debug(
                                "Vobiz recordings not ready yet (attempt=%d, count=%d)",
                                attempt + 1,
                                len(objects),
                            )
                        elif resp.status == 404:
                            logger.debug(
                                "Vobiz recording 404 (attempt=%d) — likely still finalizing",
                                attempt + 1,
                            )
                        elif resp.status in (401, 403):
                            logger.error(
                                "Vobiz auth failed (%d) — check VOBIZ_AUTH_ID / VOBIZ_AUTH_TOKEN",
                                resp.status,
                            )
                            return None
                        else:
                            body = await resp.text()
                            logger.warning("Vobiz HTTP %d: %s", resp.status, body[:200])
            except Exception as exc:  # noqa: BLE001
                logger.debug("Vobiz fetch attempt %d failed: %s", attempt + 1, exc)

            if attempt < _RECORDING_RETRIES - 1:
                await asyncio.sleep(_RECORDING_RETRY_BACKOFF_SEC * (attempt + 1))

        logger.info(
            "Vobiz recording not available after %d attempts. CallUUID=%s — "
            "Vobiz may still be finalizing; backfill manually if needed.",
            _RECORDING_RETRIES,
            call_uuid,
        )
        return None


def _sniff_call_uuid(transport: Any) -> str | None:
    """
    Best-effort CallUUID extraction from a Pipecat Vobiz transport.

    Vobiz's WebSocket start event announces the CallUUID under the
    ``callUuid`` / ``callId`` key, and Pipecat stores it on the
    serializer once the start event is parsed. We walk every plausible
    location:

      - Direct attributes on the transport.
      - The serializer, found at ``transport._serializer``,
        ``transport.serializer``, or ``transport._params.serializer``
        (the last is where ``FastAPIWebsocketTransport`` keeps it).
      - The raw ``start`` event payload, if the transport happens to
        stash it.

    The attribute is named ``_call_id`` on Pipecat's μ-law 8 kHz WS
    serializer and historically ``_call_uuid`` on some forks, so we
    accept both naming conventions.
    """
    if transport is None:
        return None

    candidates = (
        "call_uuid",
        "callUuid",
        "_call_uuid",
        "call_id",
        "callId",
        "_call_id",
    )

    for name in candidates:
        val = getattr(transport, name, None)
        if val:
            return str(val)

    # Walk every plausible serializer location, including the FastAPI
    # transport's nested `_params.serializer`.
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
