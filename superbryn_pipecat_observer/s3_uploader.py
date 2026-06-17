"""
Direct-to-S3 audio upload for SuperbrynObserver.

Two-step flow (orchestration is never on the audio data path):

    1. SDK POSTs to ``{api_base_url}/api/recording-upload-url`` with the
       customer's API key. Orchestration mints a **presigned PUT URL**
       (signed with its own ``AWS_ACCESS_KEY_ID``, never disclosed) for
       a single object key under
       ``observability/pipecat/{session_id}-{N}ch.wav``. URL expires in
       15 minutes.

    2. SDK does a plain HTTPS PUT to that URL with the WAV bytes. No
       AWS SDK needed — the signature is embedded in the query string.

This is the same pattern SuperBryn already uses for dashboard audio
uploads (see ``generatePresignedUploadUrl`` in
``orchestration-service-v2/src/lib/s3.ts``), now wired for SDK use.

Failure handling: every step is best-effort. If the presign endpoint is
unreachable or the PUT fails, we log and return ``None`` so the
observer falls back to a JSON-only webhook *without* an audio URL. The
call record still lands; only the recording link is missing on the
dashboard.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger("superbryn_pipecat_observer.s3_uploader")


async def fetch_recording_upload_url(
    api_base_url: str,
    api_key: str,
    session_id: str,
    num_channels: int,
) -> dict[str, Any] | None:
    """Ask orchestration for a presigned PUT URL for this call's WAV.

    Returns the parsed JSON body on 200 — keys: ``upload_url``,
    ``object_key``, ``public_url`` — or ``None`` on any error.
    """
    try:
        import aiohttp  # lazy: matches the import discipline in observer.py
    except ImportError:
        logger.error(
            "SUPERBRYN_PIPECAT_S3_NO_AIOHTTP: install aiohttp to enable direct S3 upload"
        )
        return None

    url = f"{api_base_url.rstrip('/')}/api/recording-upload-url"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    body = {"session_id": session_id, "num_channels": num_channels}

    try:
        timeout = aiohttp.ClientTimeout(total=10)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(url, json=body, headers=headers) as resp:
                if resp.status != 200:
                    text = (await resp.text())[:300]
                    logger.error(
                        "SUPERBRYN_PIPECAT_S3_PRESIGN_HTTP_%d: %s",
                        resp.status,
                        text,
                    )
                    return None
                presign = await resp.json()
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.error("SUPERBRYN_PIPECAT_S3_PRESIGN_FETCH_FAILED: %s", exc, exc_info=True)
        return None

    if not presign.get("upload_url") or not presign.get("public_url"):
        logger.error("SUPERBRYN_PIPECAT_S3_PRESIGN_INCOMPLETE: %s", presign)
        return None
    return presign


async def upload_wav_via_presigned(
    upload_url: str,
    wav_path: Path,
) -> bool:
    """PUT ``wav_path`` to ``upload_url``. Returns True on 200.

    The presigned URL embeds AWS credentials + ContentType in the
    signed query string, so we set ``Content-Type: audio/wav`` to
    match what was signed and stream the file body without any
    additional headers (extra signed headers would invalidate the
    signature).
    """
    try:
        import aiohttp
    except ImportError:
        logger.error(
            "SUPERBRYN_PIPECAT_S3_NO_AIOHTTP: install aiohttp to enable direct S3 upload"
        )
        return False

    try:
        size_bytes = wav_path.stat().st_size
    except OSError as exc:
        logger.error("SUPERBRYN_PIPECAT_S3_STAT_FAILED: %s", exc)
        return False

    # Generous total timeout — at typical voice-call sizes (~38 MB for
    # a 10-min stereo 16kHz WAV) the PUT itself is a few seconds, but
    # we leave headroom for slow customer egress links.
    timeout = aiohttp.ClientTimeout(total=120)
    headers = {"Content-Type": "audio/wav"}

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            with open(wav_path, "rb") as fh:
                async with session.put(upload_url, data=fh, headers=headers) as resp:
                    if resp.status not in (200, 204):
                        text = (await resp.text())[:300]
                        logger.error(
                            "SUPERBRYN_PIPECAT_S3_PUT_HTTP_%d: %s",
                            resp.status,
                            text,
                        )
                        return False
    except Exception as exc:  # noqa: BLE001 — fail-open
        logger.error("SUPERBRYN_PIPECAT_S3_PUT_FAILED: %s", exc, exc_info=True)
        return False

    logger.info(
        "SUPERBRYN_PIPECAT_S3_PUT_OK: bytes=%d",
        size_bytes,
    )
    return True
