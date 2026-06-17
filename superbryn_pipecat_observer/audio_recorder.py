"""
SuperbrynAudioRecorder — in-pipeline audio capture for SuperbrynObserver.

Wraps Pipecat's ``AudioBufferProcessor`` so the observer can ship a
single WAV file to SuperBryn at end of session, regardless of which
transport the customer is on (Daily, Twilio, Plivo, Vobiz, WebRTC, raw
WebSocket — anything Pipecat speaks). This is the *only* recording
path in the SDK: we don't fetch carrier-side recordings, which means
customers never need to share carrier credentials with SuperBryn and
plain-WebRTC / WebSocket transports are covered automatically.

Pipecat's ``AudioBufferProcessor`` captures the raw user + bot PCM as
it flows through the pipeline. We write it out as a WAV to a temp
file, hand the path to the observer, and the observer uploads it
**directly to S3** via a presigned PUT URL fetched from the SuperBryn
API (``POST /api/recording-upload-url``). Orchestration is never on
the audio data path. The temp file is ``os.unlink``'d before the
finalize coroutine returns.

Usage
-----
::

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from superbryn_pipecat_observer import SuperbrynObserver, SuperbrynAudioRecorder

    recorder = SuperbrynAudioRecorder(num_channels=2)  # user-left, bot-right

    pipeline = Pipeline([
        transport.input(),
        stt,
        ctx.user(),
        llm,
        tts,
        recorder.processor,         # insert BEFORE transport.output()
        transport.output(),
        ctx.assistant(),
    ])

    observer = SuperbrynObserver(
        agent_name="my-bot",
        transport=transport,
        audio_recorder=recorder,
    )

    task = PipelineTask(pipeline, observers=[observer],
                        params=PipelineParams(enable_usage_metrics=True))
    observer.attach_to_task(task)

    @transport.event_handler("on_client_connected")
    async def _on_connect(t, c):
        await recorder.start()

Stereo by default
-----------------
We default to ``num_channels=2`` because SuperBryn's downstream STT
pipeline (Deepgram in multichannel mode) gets perfect speaker
separation when user audio is on the left channel and bot audio is on
the right. Mono recordings work too but require AI-based diarization
which is less accurate.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import wave
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover
    from .observer import SuperbrynObserver

logger = logging.getLogger("superbryn_pipecat_observer.audio_recorder")

# Reasonable defaults; pipecat typically negotiates 16 kHz for STT-friendly
# pipelines. The processor adapts to whatever the pipeline actually carries
# at runtime — we only use these as a fallback for the WAV header in the
# (unlikely) case audio data never arrives.
_DEFAULT_SAMPLE_RATE = 16000


class SuperbrynAudioRecorder:
    """In-pipeline audio capture for SuperbrynObserver.

    Owns a Pipecat ``AudioBufferProcessor`` and a temp WAV file. The
    observer reads ``finalize()`` at end-of-session to get the path,
    uploads it to S3 via a presigned PUT, then deletes the file.

    Failure modes:
      - If ``pipecat`` isn't installed, this class still imports cleanly
        and ``processor`` is ``None`` — the observer treats that the
        same as ``audio_recorder=None`` (no recording captured).
      - If the buffer emits no ``on_audio_data`` events before finalize
        (very short calls, transport never produced audio), the WAV
        file is closed empty and finalize returns ``None`` so the
        observer skips the upload step.
    """

    def __init__(
        self,
        *,
        num_channels: int = 2,
        sample_rate: int | None = None,
        buffer_size: int | None = None,
    ) -> None:
        if num_channels not in (1, 2):
            raise ValueError(f"num_channels must be 1 or 2, got {num_channels}")

        self.num_channels = num_channels
        self._sample_rate = sample_rate or _DEFAULT_SAMPLE_RATE
        self._buffer_size = buffer_size

        self._tmp_path: Path | None = None
        self._wave_writer: wave.Wave_write | None = None
        self._bytes_written = 0
        self._lock = asyncio.Lock()
        self._finalized = False

        # Pipecat's ``AudioBufferProcessor`` dispatches ``on_audio_data``
        # via ``asyncio.create_task``, so ``processor.stop_recording()``
        # returns *before* our ``_write_chunk`` runs. We set this Event
        # at the end of each ``_write_chunk`` and wait on it in
        # ``finalize()`` so the WAV is fully flushed before we decide
        # whether any audio was captured.
        self._chunk_written = asyncio.Event()

        self.processor = self._build_processor()

    # ── Pipecat plumbing ─────────────────────────────────────────────────

    def _build_processor(self) -> Any | None:
        """Construct the underlying ``AudioBufferProcessor`` and register
        our ``on_audio_data`` handler. Returns ``None`` when pipecat is
        not importable so this module stays import-safe in test
        environments without the full pipecat install.
        """
        try:
            from pipecat.processors.audio.audio_buffer_processor import (
                AudioBufferProcessor,
            )
        except Exception as exc:  # pragma: no cover - surfaced via warning
            logger.warning(
                "AudioBufferProcessor unavailable (%s) — SuperbrynAudioRecorder will no-op. "
                "Make sure pipecat-ai >= 0.0.50 is installed.",
                exc,
            )
            return None

        kwargs: dict[str, Any] = {
            "num_channels": self.num_channels,
            "sample_rate": self._sample_rate,
        }
        if self._buffer_size is not None:
            kwargs["buffer_size"] = self._buffer_size

        processor = AudioBufferProcessor(**kwargs)

        # Pipecat's event_handler returns a decorator; we use it to register
        # an inline coroutine. The signature is fixed by Pipecat:
        #   (buffer, audio: bytes, sample_rate: int, num_channels: int)
        @processor.event_handler("on_audio_data")
        async def _on_audio_data(  # type: ignore[misc]
            _buffer: Any,
            audio: bytes,
            sample_rate: int,
            num_channels: int,
        ) -> None:
            # Diagnostic so we can tell at a glance whether
            # AudioBufferProcessor is producing chunks vs whether the
            # WAV writer is dropping them.
            logger.info(
                "SUPERBRYN_AUDIO_CHUNK: bytes=%d rate=%d channels=%d",
                len(audio),
                sample_rate,
                num_channels,
            )
            await self._write_chunk(audio, sample_rate, num_channels)

        return processor

    async def start(self) -> None:
        """Start recording. Safe to call multiple times; only the first
        call actually opens the WAV file and tells the processor to begin
        capturing.
        """
        if self.processor is None:
            return

        async with self._lock:
            if self._tmp_path is not None:
                # Already started — no-op so callers can wire this into
                # `on_client_connected` without worrying about reconnects.
                return

            fd, path_str = tempfile.mkstemp(prefix="sb-pipecat-", suffix=".wav")
            os.close(fd)  # we reopen via `wave` below for proper headers
            self._tmp_path = Path(path_str)

        try:
            await self.processor.start_recording()
            logger.info("SuperbrynAudioRecorder: recording started (tmp=%s)", self._tmp_path)
        except Exception as exc:  # noqa: BLE001 — fail-open per observer contract
            logger.warning("SuperbrynAudioRecorder.start failed: %s", exc)

    async def stop(self) -> None:
        """Tell the underlying processor to stop. The observer calls this
        from ``_finalize_session`` so the buffer flushes its final
        ``on_audio_data`` event before we close the WAV.
        """
        if self.processor is None:
            return
        try:
            await self.processor.stop_recording()
        except Exception as exc:  # noqa: BLE001
            logger.debug("SuperbrynAudioRecorder.stop swallowed: %s", exc)

    # ── Capture ──────────────────────────────────────────────────────────

    async def _write_chunk(
        self,
        audio: bytes,
        sample_rate: int,
        num_channels: int,
    ) -> None:
        """Append a PCM chunk to the WAV. Opens the writer lazily on the
        first chunk so we use the actual sample rate / channel count
        negotiated by the pipeline, not whatever defaults the caller
        passed in.
        """
        if not audio:
            return

        async with self._lock:
            if self._tmp_path is None or self._finalized:
                return

            if self._wave_writer is None:
                try:
                    wf = wave.open(str(self._tmp_path), "wb")
                    wf.setnchannels(num_channels)
                    wf.setsampwidth(2)  # AudioBufferProcessor emits 16-bit PCM
                    wf.setframerate(sample_rate)
                    self._wave_writer = wf
                    self._sample_rate = sample_rate
                except Exception as exc:  # noqa: BLE001
                    logger.warning(
                        "SuperbrynAudioRecorder: failed to open WAV writer at %s: %s",
                        self._tmp_path,
                        exc,
                    )
                    return

            try:
                self._wave_writer.writeframes(audio)
                self._bytes_written += len(audio)
            except Exception as exc:  # noqa: BLE001
                logger.warning("SuperbrynAudioRecorder: writeframes failed: %s", exc)

        # Notify any awaiters (``finalize``) that at least one chunk has
        # landed. Done outside the lock so a slow waiter can't block the
        # write path.
        self._chunk_written.set()

    # ── Lifecycle ────────────────────────────────────────────────────────

    async def finalize(self, _observer: SuperbrynObserver | None = None) -> Path | None:
        """Stop capture, flush the WAV, and return the temp file path.

        Returns ``None`` if no audio was ever captured — the observer
        then skips the S3 upload step. The caller (observer) is
        responsible for deleting the file once it's been uploaded.
        """
        if self.processor is None:
            return None

        # Snapshot buffer state so we know whether the processor actually
        # captured audio. Pipecat's ``stop_recording`` resets the buffers
        # after firing ``on_audio_data`` (which is dispatched on a
        # separate task), so we look here *before* calling stop.
        had_buffered_audio = False
        try:
            user_buf = len(getattr(self.processor, "_user_audio_buffer", b""))
            bot_buf = len(getattr(self.processor, "_bot_audio_buffer", b""))
            had_buffered_audio = (user_buf + bot_buf) > 0
            recording = getattr(self.processor, "_recording", None)
            proc_rate = getattr(self.processor, "_sample_rate", None)
            logger.info(
                "SuperbrynAudioRecorder.finalize: recording=%s user_buf=%d bot_buf=%d processor_rate=%s",
                recording,
                user_buf,
                bot_buf,
                proc_rate,
            )
        except Exception as exc:  # noqa: BLE001
            logger.debug("audio recorder buffer-state log failed: %s", exc)

        # Reset before stop so we can detect the post-stop handler firing.
        self._chunk_written.clear()
        await self.stop()

        # Pipecat dispatches ``on_audio_data`` via ``asyncio.create_task``,
        # so ``stop_recording`` returns *before* the WAV writer runs.
        # Wait for the writer to actually flush before reading
        # ``_bytes_written``. If there was no buffered audio, skip the
        # wait — the handler won't fire and we'd just timeout.
        if had_buffered_audio:
            try:
                await asyncio.wait_for(self._chunk_written.wait(), timeout=3.0)
            except asyncio.TimeoutError:
                logger.warning(
                    "SuperbrynAudioRecorder: timed out waiting for on_audio_data to flush "
                    "(buffers had data but writer never fired)"
                )

        async with self._lock:
            if self._finalized:
                return self._tmp_path if (self._tmp_path and self._bytes_written > 0) else None
            self._finalized = True

            if self._wave_writer is not None:
                try:
                    self._wave_writer.close()
                except Exception as exc:  # noqa: BLE001
                    logger.debug("WAV close swallowed: %s", exc)
                self._wave_writer = None

            if self._tmp_path is None:
                return None

            if self._bytes_written == 0:
                # Empty buffer — clean up immediately so we don't leak
                # zero-byte files into /tmp.
                try:
                    self._tmp_path.unlink(missing_ok=True)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("empty WAV cleanup failed: %s", exc)
                logger.info("SuperbrynAudioRecorder: no audio captured; skipping upload")
                return None

            logger.info(
                "SuperbrynAudioRecorder: finalized (path=%s, bytes=%d, sample_rate=%d, channels=%d)",
                self._tmp_path,
                self._bytes_written,
                self._sample_rate,
                self.num_channels,
            )
            return self._tmp_path

    @property
    def sample_rate(self) -> int:
        """Sample rate of the captured audio (updated on first chunk)."""
        return self._sample_rate

    @property
    def bytes_written(self) -> int:
        return self._bytes_written

    def cleanup(self) -> None:
        """Best-effort temp-file removal. Called by the observer after
        a successful upload, and also as a safety net in error paths.
        """
        if self._tmp_path is None:
            return
        try:
            self._tmp_path.unlink(missing_ok=True)
        except Exception as exc:  # noqa: BLE001
            logger.debug("SuperbrynAudioRecorder.cleanup swallowed: %s", exc)
        self._tmp_path = None
