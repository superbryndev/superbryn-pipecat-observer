"""
SuperbrynObserver — drop-in observer for Pipecat pipelines.

Sits alongside the pipeline (passed via `PipelineParams(observers=[...])`),
accumulates per-call state from frames as they flow between processors,
and POSTs a normalized call payload to SuperBryn's ingest endpoint when
the pipeline finishes.

Design mirrors `livekit-evals/WebhookHandler` so both SDKs:
  - Read the same env vars (`SUPERBRYN_API_KEY`, optional `SUPERBRYN_WEBHOOK_URL`)
  - Send the same auth header (`X-API-Key`)
  - Produce the same shape of `call` payload (so the SuperBryn backend
    treats both the same way — see `pipecat.adapter.ts` for the contract)

Fail-open behaviour:
  - Missing API key → observer no-ops (never raises).
  - Failed HTTP POST → logged, never propagated. Telemetry must never crash
    the customer's agent process.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any

from ._provider_detect import detect_provider_from_model
from .config import AGENT_CONFIG, WEBHOOK_CONFIG
from .transports import RecordingAdapter, get_recording_adapter

try:
    from pipecat.observers.base_observer import BaseObserver, FramePushed
except Exception:  # pragma: no cover - import error surfaced clearly to the caller
    BaseObserver = object  # type: ignore[assignment,misc]
    FramePushed = Any  # type: ignore[assignment,misc]

logger = logging.getLogger("superbryn_pipecat_observer")

__version__ = "0.2.2"
_SDK_TAG = f"@superbryn/pipecat-observer@{__version__}"


class SuperbrynObserver(BaseObserver):
    """
    Pipecat observer that reports a normalized call record to SuperBryn at
    end of session.

    Usage:
        from pipecat.pipeline.task import PipelineTask, PipelineParams
        from superbryn_pipecat_observer import SuperbrynObserver

        task = PipelineTask(
            pipeline,
            params=PipelineParams(
                enable_usage_metrics=True,         # required for token/char counts
                observers=[SuperbrynObserver(agent_name="my-bot")],
            ),
        )
    """

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        agent_id: str | None = None,
        api_key: str | None = None,
        webhook_url: str | None = None,
        transport: Any | None = None,
        from_number: str | None = None,
        to_number: str | None = None,
        recording_url: str | None = None,
        stereo_recording_url: str | None = None,
        extra_metadata: dict[str, Any] | None = None,
    ) -> None:
        self.agent_name = agent_name
        self.agent_id = agent_id or AGENT_CONFIG["id"]
        self.api_key = api_key or WEBHOOK_CONFIG["api_key"]
        self.webhook_url = webhook_url or WEBHOOK_CONFIG["url"]

        # `transport` accepts either a plain string label (legacy 0.1.x usage)
        # or a live Pipecat transport object. When an object is passed we try
        # to auto-wire recording for the matching transport (Daily / Twilio).
        # Unknown transports degrade gracefully — we still stamp a `transport`
        # label on the payload and the caller can pass `recording_url=...`
        # explicitly as before. SuperBryn never records audio itself; we only
        # surface URLs produced by the transport.
        self._transport_obj: Any | None = None
        self._recording_adapter: RecordingAdapter | None = None
        if transport is None or isinstance(transport, str):
            self.transport = transport
        else:
            self._transport_obj = transport
            self._recording_adapter = get_recording_adapter(transport)
            if self._recording_adapter is not None:
                self.transport = self._recording_adapter.transport_name
                logger.info(
                    "Auto-wired recording adapter for transport=%s",
                    self.transport,
                )
            else:
                # Stamp something useful so the dashboard shows what was used.
                self.transport = type(transport).__name__.lower().replace("transport", "") or None
                logger.info(
                    "No recording adapter for transport class %s — manual recording_url required",
                    type(transport).__name__,
                )

        self.from_number = from_number
        self.to_number = to_number
        self.recording_url = recording_url
        self.stereo_recording_url = stereo_recording_url
        self.extra_metadata = extra_metadata or {}

        self.session_id = str(uuid.uuid4())
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self._call_start_ms: int | None = None

        self.transcript_turns: list[dict[str, Any]] = []
        self._last_user_end_ms: int | None = None

        self.usage = {
            "llm_provider": None,
            "llm_model": None,
            "llm_input_tokens": 0,
            "llm_output_tokens": 0,
            "stt_provider": None,
            "stt_model": None,
            "stt_duration_seconds": 0.0,
            "tts_provider": None,
            "tts_model": None,
            "tts_voice_id": None,
            "tts_characters": 0,
        }
        self.latencies_ms: list[float] = []
        self.call_end_reason: str | None = None
        self._sent = False

        if not self.api_key:
            logger.warning("SUPERBRYN_API_KEY not configured — SuperbrynObserver will no-op.")

    # ── Pipeline lifecycle ────────────────────────────────────────────────

    async def on_pipeline_started(self) -> None:
        self.started_at = datetime.now(timezone.utc)
        self._call_start_ms = int(self.started_at.timestamp() * 1000)
        logger.info("SUPERBRYN_PIPECAT_CALL_STARTED: session_id=%s", self.session_id)

        if self._recording_adapter is not None:
            try:
                await self._recording_adapter.start(self)
            except Exception as exc:  # noqa: BLE001 — fail-open
                logger.warning(
                    "Recording adapter start failed (continuing without auto-recording): %s",
                    exc,
                )

    async def on_pipeline_finished(self) -> None:
        if self._sent:
            return
        self._sent = True
        self.ended_at = datetime.now(timezone.utc)

        # Transport-native recording (Daily / Twilio). The adapter stamps
        # `recording_url` on the observer when the transport produces one.
        if self._recording_adapter is not None:
            try:
                await self._recording_adapter.finalize(self)
            except Exception as exc:  # noqa: BLE001 — fail-open
                logger.warning(
                    "Recording adapter finalize failed (sending without recording URL): %s",
                    exc,
                )

        await self._send_webhook()

    # ── Frame observation ────────────────────────────────────────────────

    async def on_push_frame(self, data: FramePushed) -> None:  # type: ignore[override]
        """
        Inspect frames as they flow between processors.

        Pipecat's frame names move occasionally — we look up by class name
        instead of importing concrete frame types, so a Pipecat upgrade
        doesn't hard-break the observer when new frames are added.
        """
        try:
            frame = getattr(data, "frame", None)
            if frame is None:
                return

            cls_name = type(frame).__name__

            if cls_name == "TranscriptionFrame":
                self._capture_user_turn(frame)
            elif cls_name == "TextFrame" and self._looks_like_bot_text(frame):
                self._capture_bot_turn(frame)
            elif cls_name in ("LLMFullResponseStartFrame", "BotStartedSpeakingFrame"):
                self._mark_bot_response_start()
            elif cls_name == "MetricsFrame":
                self._capture_metrics(frame)
            elif cls_name == "UserStoppedSpeakingFrame":
                self._mark_user_stop()
            elif cls_name in ("EndFrame", "StopFrame", "CancelFrame"):
                self._capture_end_reason(cls_name)

            # First time we see a service frame, infer provider from module path
            source = getattr(data, "source", None)
            if source is not None:
                self._sniff_provider(source, cls_name)

        except Exception as exc:  # noqa: BLE001 — observer must never raise
            logger.debug("on_push_frame swallowed error: %s", exc)

    # ── Capture helpers ──────────────────────────────────────────────────

    def _now_ms(self) -> int:
        return int(datetime.now(timezone.utc).timestamp() * 1000) - (self._call_start_ms or 0)

    def _capture_user_turn(self, frame: Any) -> None:
        text = (getattr(frame, "text", "") or "").strip()
        if not text:
            return
        now_ms = self._now_ms()
        self.transcript_turns.append(
            {
                "speaker": "user",
                "text": text,
                "start_time_ms": now_ms,
                "end_time_ms": now_ms,
                "confidence": getattr(frame, "confidence", None),
            }
        )
        self._last_user_end_ms = now_ms

    def _capture_bot_turn(self, frame: Any) -> None:
        text = (getattr(frame, "text", "") or "").strip()
        if not text:
            return
        now_ms = self._now_ms()
        # If a bot turn is already open with empty text, fill it; else append.
        for turn in reversed(self.transcript_turns):
            if turn["speaker"] == "agent" and not turn["text"]:
                turn["text"] = text
                turn["end_time_ms"] = now_ms
                return
        self.transcript_turns.append(
            {
                "speaker": "agent",
                "text": text,
                "start_time_ms": now_ms,
                "end_time_ms": now_ms,
                "latency_ms": (
                    now_ms - self._last_user_end_ms if self._last_user_end_ms is not None else None
                ),
            }
        )

    def _mark_bot_response_start(self) -> None:
        now_ms = self._now_ms()
        latency = now_ms - self._last_user_end_ms if self._last_user_end_ms is not None else None
        if latency is not None and latency >= 0:
            self.latencies_ms.append(float(latency))
        # Open a placeholder bot turn — text gets filled by TextFrame.
        self.transcript_turns.append(
            {
                "speaker": "agent",
                "text": "",
                "start_time_ms": now_ms,
                "end_time_ms": None,
                "latency_ms": latency,
            }
        )

    def _mark_user_stop(self) -> None:
        self._last_user_end_ms = self._now_ms()

    def _capture_metrics(self, frame: Any) -> None:
        """
        Pipecat emits MetricsFrame containing a list of typed metric records.
        We pluck LLM token counts and TTS character counts.
        """
        records = getattr(frame, "data", None) or []
        for rec in records:
            cls_name = type(rec).__name__
            if cls_name == "LLMUsageMetricsData":
                tok = getattr(rec, "value", None)
                if tok is not None:
                    self.usage["llm_input_tokens"] += int(getattr(tok, "prompt_tokens", 0) or 0)
                    self.usage["llm_output_tokens"] += int(
                        getattr(tok, "completion_tokens", 0) or 0
                    )
                if getattr(rec, "model", None):
                    self.usage["llm_model"] = rec.model
                    if not self.usage["llm_provider"]:
                        self.usage["llm_provider"] = detect_provider_from_model(rec.model)
            elif cls_name == "TTSUsageMetricsData":
                self.usage["tts_characters"] += int(getattr(rec, "value", 0) or 0)
                if getattr(rec, "model", None):
                    self.usage["tts_model"] = rec.model
                    if not self.usage["tts_provider"]:
                        self.usage["tts_provider"] = detect_provider_from_model(rec.model)

    def _capture_end_reason(self, cls_name: str) -> None:
        mapping = {
            "EndFrame": "completed",
            "StopFrame": "completed",
            "CancelFrame": "cancelled",
        }
        if not self.call_end_reason:
            self.call_end_reason = mapping.get(cls_name, cls_name.lower())

    def _sniff_provider(self, source: Any, frame_cls: str) -> None:
        """
        Best-effort detection of which service produced this frame.
        Looks at the module path of the producing processor.
        """
        try:
            module = type(source).__module__ or ""
            if "stt" in module and not self.usage["stt_provider"]:
                self.usage["stt_provider"] = detect_provider_from_model(module)
                self.usage["stt_model"] = getattr(source, "model_name", None) or getattr(
                    source, "model", None
                )
            elif "llm" in module and not self.usage["llm_provider"]:
                self.usage["llm_provider"] = detect_provider_from_model(module)
                if not self.usage["llm_model"]:
                    self.usage["llm_model"] = getattr(source, "model_name", None) or getattr(
                        source, "model", None
                    )
            elif "tts" in module and not self.usage["tts_provider"]:
                self.usage["tts_provider"] = detect_provider_from_model(module)
                if not self.usage["tts_model"]:
                    self.usage["tts_model"] = getattr(source, "model_name", None) or getattr(
                        source, "model", None
                    )
                if not self.usage["tts_voice_id"]:
                    self.usage["tts_voice_id"] = getattr(source, "voice_id", None) or getattr(
                        source, "voice", None
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider sniff failed: %s", exc)

    # ── Payload + send ───────────────────────────────────────────────────

    def _build_payload(self) -> dict[str, Any]:
        duration_seconds = 0.0
        if self.started_at and self.ended_at:
            duration_seconds = (self.ended_at - self.started_at).total_seconds()

        avg_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None
        p95_latency = None
        if self.latencies_ms:
            sorted_lat = sorted(self.latencies_ms)
            idx = max(0, int(len(sorted_lat) * 0.95) - 1)
            p95_latency = sorted_lat[idx]

        # Drop empty turns — matches livekit-evals' filter.
        turns_with_text = [t for t in self.transcript_turns if t.get("text", "").strip()]

        return {
            "event": "call.completed",
            "sdk_version": _SDK_TAG,
            "call": {
                "id": self.session_id,
                "session_id": self.session_id,
                "started_at": self.started_at.isoformat() if self.started_at else None,
                "ended_at": self.ended_at.isoformat() if self.ended_at else None,
                "duration_seconds": duration_seconds,
                "call_end_reason": self.call_end_reason or "completed",
                "from_number": self.from_number,
                "to_number": self.to_number,
                "transcript": {"turns": turns_with_text},
                "recording_url": self.recording_url,
                "stereo_recording_url": self.stereo_recording_url,
                "metadata": {
                    "agent_id": self.agent_id,
                    "agent_name": self.agent_name,
                    "transport": self.transport,
                    "llm_provider": self.usage["llm_provider"],
                    "llm_model": self.usage["llm_model"],
                    "stt_provider": self.usage["stt_provider"],
                    "stt_model": self.usage["stt_model"],
                    "tts_provider": self.usage["tts_provider"],
                    "tts_model": self.usage["tts_model"],
                    "tts_voice_id": self.usage["tts_voice_id"],
                    "pipeline_version": _SDK_TAG,
                    **self.extra_metadata,
                },
                "usage": {
                    "llm_input_tokens": self.usage["llm_input_tokens"],
                    "llm_output_tokens": self.usage["llm_output_tokens"],
                    "stt_duration_seconds": self.usage["stt_duration_seconds"],
                    "tts_characters": self.usage["tts_characters"],
                },
                "latency": {
                    "avg_ms": avg_latency,
                    "p95_ms": p95_latency,
                },
            },
        }

    async def _send_webhook(self) -> None:
        if not self.api_key:
            logger.info("SUPERBRYN_PIPECAT_SKIPPED: no API key configured")
            return
        if not self.webhook_url:
            logger.warning("SUPERBRYN_PIPECAT_NO_URL: webhook URL not configured")
            return

        payload = self._build_payload()
        logger.info(
            "SUPERBRYN_PIPECAT_SENDING: session_id=%s url=%s",
            self.session_id,
            self.webhook_url,
        )

        try:
            import aiohttp  # lazy: optional at import time, required at send time
        except ImportError:
            logger.error(
                "SUPERBRYN_PIPECAT_MISSING_AIOHTTP: install aiohttp to enable webhook delivery"
            )
            return

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers={
                        "Content-Type": "application/json",
                        "X-API-Key": self.api_key,
                    },
                    timeout=aiohttp.ClientTimeout(total=30),
                ) as resp:
                    body = await resp.text()
                    if resp.status == 200:
                        logger.info("SUPERBRYN_PIPECAT_SENT: %s", body[:200])
                    elif resp.status in (401, 403):
                        logger.error(
                            "SUPERBRYN_PIPECAT_AUTH_FAILED (%d): check SUPERBRYN_API_KEY",
                            resp.status,
                        )
                    else:
                        logger.error("SUPERBRYN_PIPECAT_HTTP_%d: %s", resp.status, body[:200])
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.error("SUPERBRYN_PIPECAT_ERROR: %s", exc, exc_info=True)

    # ── Helpers ──────────────────────────────────────────────────────────

    def _looks_like_bot_text(self, frame: Any) -> bool:
        """
        A `TextFrame` can come from many places. We treat it as a bot turn
        only if it appears between LLM-response-start and the next user
        transcription — i.e. there's an open bot turn waiting for text.
        """
        return any(t["speaker"] == "agent" and not t["text"] for t in self.transcript_turns)
