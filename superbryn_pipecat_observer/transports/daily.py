"""
Daily recording adapter.

Daily's cloud recording lives on Daily's servers. The flow is:

  1. Start recording when the pipeline starts:
        await transport.start_recording()

  2. Daily fires `on_recording_started` with a `recordingId` once egress is
     active. We hook the event handler to capture it.

  3. When the pipeline ends, stop the recording cleanly:
        await transport.stop_recording()

  4. Fetch the playback link from Daily's REST API:
        GET https://api.daily.co/v1/recordings/{id}/access-link

     Daily finalizes recordings asynchronously, so the access link may not
     be immediately available. We retry a few times with backoff, then
     fail-open.

Required env var:
    DAILY_API_KEY — Daily API key with `recordings:read` scope. The same
    key the customer used to create their Daily room can be reused.
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..observer import SuperbrynObserver

logger = logging.getLogger("superbryn_pipecat_observer.transports.daily")

_ACCESS_LINK_URL = "https://api.daily.co/v1/recordings/{id}/access-link"
_ACCESS_LINK_RETRIES = 3
_ACCESS_LINK_RETRY_BACKOFF_SEC = 2.0


class DailyRecordingAdapter:
    """Auto-wires Daily cloud recording for a Pipecat `DailyTransport`."""

    transport_name = "daily"

    def __init__(self, transport: Any) -> None:
        self.transport = transport
        self.recording_id: str | None = None
        # Read at construction so customers get a clear warning early if the
        # key is missing, rather than discovering it at finalize time.
        self.api_key = os.getenv("DAILY_API_KEY", "")
        if not self.api_key:
            logger.warning(
                "DAILY_API_KEY not set — Daily recording URL will not be fetched. "
                "Recording will still start, but the URL must be retrieved manually.",
            )

    async def start(self, observer: SuperbrynObserver) -> None:
        # Register the event handler first so we don't miss the event if
        # `start_recording` returns synchronously and Daily fires fast.
        try:
            handler = getattr(self.transport, "event_handler", None)
            if handler is not None:

                @handler("on_recording_started")
                async def _on_started(_transport: Any, recording: Any) -> None:  # noqa: ARG001
                    # Pipecat shapes vary — accept dict, dataclass, or object with attrs.
                    rid = (
                        _get(recording, "recordingId")
                        or _get(recording, "recording_id")
                        or _get(recording, "id")
                    )
                    if rid:
                        self.recording_id = rid
                        logger.info("Daily recording started: id=%s", rid)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Daily on_recording_started hook failed: %s", exc)

        # Best-effort start. If the room already auto-records (room property
        # `enable_recording: "cloud"`), this is a no-op and Daily still fires
        # the event handler above.
        try:
            start = getattr(self.transport, "start_recording", None)
            if start is not None:
                await start()
                logger.info("Daily start_recording invoked")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Daily start_recording failed (continuing): %s", exc)

    async def finalize(self, observer: SuperbrynObserver) -> None:
        # Stop recording cleanly so the file is finalized on Daily's side
        # before we ask for the access link.
        try:
            stop = getattr(self.transport, "stop_recording", None)
            if stop is not None:
                await stop()
                logger.info("Daily stop_recording invoked")
        except Exception as exc:  # noqa: BLE001
            logger.warning("Daily stop_recording failed (continuing): %s", exc)

        if not self.recording_id:
            logger.info("Daily recording_id not captured — skipping access-link fetch")
            return
        if not self.api_key:
            return

        url = await self._fetch_access_link(self.recording_id)
        if url:
            observer.recording_url = url
            logger.info("Daily recording URL attached: %s", url)

    async def _fetch_access_link(self, recording_id: str) -> str | None:
        """Poll Daily's access-link endpoint with a few retries."""
        try:
            import aiohttp  # lazy
        except ImportError:
            logger.error("aiohttp not installed — cannot fetch Daily access link")
            return None

        endpoint = _ACCESS_LINK_URL.format(id=recording_id)
        headers = {"Authorization": f"Bearer {self.api_key}"}

        for attempt in range(_ACCESS_LINK_RETRIES):
            try:
                async with aiohttp.ClientSession() as sess:
                    async with sess.get(
                        endpoint,
                        headers=headers,
                        timeout=aiohttp.ClientTimeout(total=10),
                    ) as resp:
                        if resp.status == 200:
                            data = await resp.json()
                            return data.get("download_link") or data.get("downloadLink")
                        if resp.status in (404, 423):
                            # 404: not finalized yet. 423: locked (still encoding).
                            logger.debug(
                                "Daily access-link not ready (status=%d, attempt=%d)",
                                resp.status,
                                attempt + 1,
                            )
                        else:
                            body = await resp.text()
                            logger.warning(
                                "Daily access-link HTTP %d: %s",
                                resp.status,
                                body[:200],
                            )
                            return None
            except Exception as exc:  # noqa: BLE001
                logger.debug("Daily access-link attempt %d failed: %s", attempt + 1, exc)

            if attempt < _ACCESS_LINK_RETRIES - 1:
                await asyncio.sleep(_ACCESS_LINK_RETRY_BACKOFF_SEC * (attempt + 1))

        logger.info(
            "Daily access-link not available after %d attempts — recording may still be encoding. "
            "Recording id=%s preserved in extra metadata for later backfill.",
            _ACCESS_LINK_RETRIES,
            recording_id,
        )
        return None


def _get(obj: Any, key: str) -> Any:
    """Best-effort attribute/key lookup — handles dicts, dataclasses, objects."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(key)
    return getattr(obj, key, None)
