"""
SuperbrynObserver — drop-in observer for Pipecat pipelines.

Sits alongside the pipeline (passed to `PipelineTask(observers=[...])`),
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

import asyncio
import json
import logging
import os
import re
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from ._provider_detect import detect_provider_from_model
from .audio_recorder import SuperbrynAudioRecorder
from .config import AGENT_CONFIG, WEBHOOK_CONFIG
from .s3_uploader import fetch_recording_upload_url, upload_wav_via_presigned

try:
    from pipecat.observers.base_observer import BaseObserver, FramePushed
except Exception:  # pragma: no cover - import error surfaced clearly to the caller
    BaseObserver = object  # type: ignore[assignment,misc]
    FramePushed = Any  # type: ignore[assignment,misc]

logger = logging.getLogger("superbryn_pipecat_observer")

__version__ = "0.6.5"
_SDK_TAG = f"@superbryn/pipecat-observer@{__version__}"

# Frame class names that signal the call/pipeline is wrapping up. Pipecat 1.3
# pushes one of these through `on_push_frame` when a session ends; we use
# them as the trigger for our finalize step instead of the (now removed)
# observer-level `on_pipeline_finished` lifecycle hook.
_TERMINAL_FRAME_NAMES = ("EndFrame", "StopFrame", "CancelFrame")

# Frame class names that carry assistant-side spoken/streamed text we want
# to capture as a bot turn. Matched by ``type(frame).__name__`` so we don't
# have to import the concrete classes (Pipecat renames them between
# minor releases). Includes:
#   - "TextFrame"           — legacy Pipecat <= 1.2 LLM streaming output.
#   - "LLMTextFrame"        — Pipecat 1.3+ subclass for LLM streamed text.
#   - "LLMFullResponseFrame"— Pipecat 1.3+ end-of-response aggregated text
#                             (some providers emit this instead of streaming).
#   - "TTSTextFrame"        — pipelines that push text through TTS before
#                             the LLM aggregator sees it.
_BOT_TEXT_FRAME_NAMES = (
    "TextFrame",
    "LLMTextFrame",
    "LLMFullResponseFrame",
    "TTSTextFrame",
)

# Matches Anthropic-style prompt-engineered tool calls that arrive as plain
# text inside an `LLMTextFrame` (i.e. the LLM was *told* to emit `<tool_use>`
# blocks via the system prompt, instead of having real Pipecat tools
# registered via `register_function`). We extract these in `_close_bot_turn`
# so they land on `call.tool_calls` as structured records — and we strip the
# XML out of the transcript turn text so the dashboard transcript reads
# cleanly instead of leaking JSON at the user.
#
#   <tool_use>
#   {"tool": "get_cart", "items": [...]}
#   </tool_use>
#
# The pattern is non-greedy on the JSON body so a single turn containing
# multiple `<tool_use>` blocks parses as multiple calls instead of one
# spanning the entire turn. DOTALL lets the inner JSON span newlines.
_TOOL_USE_PATTERN = re.compile(r"<tool_use>\s*(\{.*?\})\s*</tool_use>", re.DOTALL)


def _coerce_tool_payload(value: Any) -> Any:
    """Normalize a tool call's `arguments` / `result` for JSON transport.

    Pipecat hands these through as `Any` — they might be already-parsed
    dicts/lists from the LLM, raw JSON strings, or arbitrary Python
    objects returned by the tool. We keep dicts/lists/primitives intact
    (they survive `json.dumps` unchanged), attempt to parse strings that
    look like JSON, and `str()`-coerce anything exotic so the webhook
    payload stays serializable.
    """
    if value is None:
        return None
    if isinstance(value, (dict, list, int, float, bool)):
        return value
    if isinstance(value, str):
        s = value.strip()
        if s.startswith(("{", "[")):
            try:
                return json.loads(s)
            except (ValueError, TypeError):
                return value
        return value
    return str(value)


class SuperbrynObserver(BaseObserver):
    """Pipecat observer that reports a normalized call record to SuperBryn
    at end of session.

    Two integration shapes — pick one:

    1) **Single-step (recommended).** The observer owns the
       ``AudioBufferProcessor``, the ``PipelineTask``, and the
       lifecycle handlers. Customers don't touch any of that
       wiring::

           observer = SuperbrynObserver(
               agent_id=AGENT_ID,
               api_key=os.getenv("SUPERBRYN_API_KEY"),
           )

           # Production (captures audio):
           task = observer.observe_and_create_task(
               pipeline, context,
               runner_args=runner_args,
               transport=transport,
           )

           # UAT / simulation (transcript-only, no audio):
           task = observer.track_and_create_task(
               pipeline, context,
               runner_args=runner_args,
               transport=transport,
           )

           await PipelineRunner().run(task)

    2) **Advanced / manual.** Build the ``SuperbrynAudioRecorder``
       and ``PipelineTask`` yourself, then attach the observer.
       Useful when you need a custom processor order or
       ``PipelineParams``::

           recorder = SuperbrynAudioRecorder(num_channels=2)
           pipeline = Pipeline([..., transport.output(), recorder.processor, ...])
           observer = SuperbrynObserver(
               agent_id=AGENT_ID,
               audio_recorder=recorder,
               transport=transport,
           )
           task = PipelineTask(
               pipeline,
               observers=[observer],
               params=PipelineParams(enable_usage_metrics=True),
           )
           observer.register_task_handlers(task, transport=transport)
    """

    def __init__(
        self,
        *,
        agent_name: str | None = None,
        agent_id: str | None = None,
        api_key: str | None = None,
        webhook_url: str | None = None,
        api_base_url: str | None = None,
        transport: Any | None = None,
        from_number: str | None = None,
        to_number: str | None = None,
        recording_url: str | None = None,
        stereo_recording_url: str | None = None,
        audio_recorder: SuperbrynAudioRecorder | None = None,
        session_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        extra_metadata: dict[str, Any] | None = None,
        enabled: bool = True,
        capture_logs: bool = True,
        max_log_records: int = 1000,
    ) -> None:
        # Pipecat 1.3's BaseObserver.__init__ wires internal state (e.g.
        # `_event_tasks`) that the pipeline relies on during cleanup. Without
        # this call, end-of-pipeline triggers an AttributeError. We guard the
        # call because in environments where pipecat isn't installed the
        # `BaseObserver` import fallback uses plain `object`, whose
        # `__init__` rejects keyword args.
        try:
            super().__init__()
        except TypeError:
            pass

        self.agent_name = agent_name
        self.agent_id = agent_id or AGENT_CONFIG["id"]
        self.api_key = api_key or WEBHOOK_CONFIG["api_key"]
        self.webhook_url = webhook_url or WEBHOOK_CONFIG["url"]
        # API root used for the presigned-URL broker
        # (``/api/recording-upload-url``). Derived from the webhook URL by
        # default; override only when the broker lives on a different
        # host than the webhook ingest.
        self.api_base_url = api_base_url or WEBHOOK_CONFIG.get("api_base_url") or ""

        # `transport` is no longer used for recording (the in-pipeline
        # `AudioBufferProcessor` captures audio regardless of carrier),
        # so we only keep it for two narrow purposes:
        #   • The string label that lands on `metadata.transport` in
        #     the webhook payload (e.g. "daily", "twilio").
        #   • A reference to the live transport object so
        #     `_find_output_transport_index` can match by identity
        #     when inserting the recorder into the pipeline.
        # Accepts either a plain string (legacy 0.1.x usage) or a
        # Pipecat transport instance. Unknown shapes degrade to None.
        self._transport_obj: Any | None = None
        if transport is None or isinstance(transport, str):
            self.transport = transport
        else:
            self._transport_obj = transport
            self.transport = type(transport).__name__.lower().replace("transport", "") or None

        self.from_number = from_number
        self.to_number = to_number
        self.recording_url = recording_url
        self.stereo_recording_url = stereo_recording_url
        # In-pipeline audio capture (Pipecat's AudioBufferProcessor). When
        # set, the observer flushes the WAV at end-of-session, requests a
        # presigned PUT URL from `{api_base_url}/api/recording-upload-url`,
        # PUTs the WAV directly to S3 from this process, stamps
        # `stereo_recording_url` on the JSON payload, and then deletes
        # the temp file. See `audio_recorder.py` + `s3_uploader.py`.
        self.audio_recorder = audio_recorder

        # Mutable metadata bag — `custom_metadata` is the documented name
        # (parity with the Cekura SDK surface); `extra_metadata` remains
        # accepted for back-compat with 0.3.x callers. Both end up in the
        # same dict and can be updated mid-session via
        # `set_custom_metadata()`. Values flow into `metadata.*` on the
        # outgoing webhook payload.
        merged_metadata: dict[str, Any] = {}
        if extra_metadata:
            merged_metadata.update(extra_metadata)
        if custom_metadata:
            merged_metadata.update(custom_metadata)
        self.extra_metadata = merged_metadata

        # Session ID precedence: explicit constructor arg → uuid4(). The
        # high-level `observe_pipeline` / `track_pipeline` entrypoints
        # can override this later (e.g. with `runner_args.session_id`)
        # before the pipeline runs.
        self.session_id = session_id or str(uuid.uuid4())
        self.started_at: datetime | None = None
        self.ended_at: datetime | None = None
        self._call_start_ms: int | None = None

        self.transcript_turns: list[dict[str, Any]] = []
        self._last_user_end_ms: int | None = None

        # VAD-derived fallback for `stt_duration_seconds`. We track the
        # start of every user-speaking segment and accumulate total
        # speech ms here. At payload-build time we use this only if the
        # STT service never emitted `STTUsageMetricsData` (e.g. provider
        # has metrics disabled, or the call ended before a flush).
        self._user_speech_start_ms: int | None = None
        self._vad_user_speech_ms: int = 0

        # Streaming-LLM bot-turn accumulator. Set True when we open a
        # placeholder turn on the first BotStartedSpeakingFrame /
        # LLMFullResponseStartFrame and reset when the matching stopped /
        # end frame fires. Prevents the same conceptual response from
        # producing multiple turns when the start/stop frames propagate
        # across several pipeline links.
        self._bot_turn_open: bool = False

        # Tool / function-call invocations made by the LLM during the call.
        # Pipecat emits `FunctionCallInProgressFrame` when the LLM decides to
        # invoke a tool and `FunctionCallResultFrame` once the tool returns.
        # We pair them by `tool_call_id` and ship the merged record on
        # `call.tool_calls` so the SuperBryn analysis worker can stitch it
        # onto the right assistant turn (same contract as VAPI / Retell /
        # Traces — see `obs-analysis.worker.ts::buildTranscriptTurnsWithToolCalls`).
        self.tool_calls: list[dict[str, Any]] = []
        self._tool_calls_by_id: dict[str, dict[str, Any]] = {}

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

        # Deferred-upload / consent gate (Cekura parity). When set, the
        # SDK still captures audio + transcript locally but holds the S3
        # upload + webhook until the host process explicitly calls
        # `start_audio_upload()`. `abort()` flips this into a discard
        # state so the temp WAV is deleted and no webhook is sent.
        self._defer_upload = False
        self._deferred_release: asyncio.Event | None = None  # set on consent
        self._aborted = False
        self._sent_track_only = False  # for the track-mode auto-suppress flag

        # Captured WAV path waiting to be uploaded to S3. Set in
        # `_finalize_session` after `audio_recorder.finalize()`, read by
        # `_send_webhook` which does the direct S3 PUT, and unlinked in
        # the finally block of `_finalize_session` so the temp file
        # lives only for the duration of the upload.
        self._pending_audio_path: Path | None = None

        # Session log capture (Cekura `capture_logs=True`).
        # When enabled, we attach a logging.Handler to the root logger from
        # `on_pipeline_started` and detach it during finalize. Records are
        # buffered in `self._captured_logs` (capped at `max_log_records`
        # to bound memory + payload size).
        self._capture_logs = capture_logs
        self._max_log_records = max(0, int(max_log_records))
        self._captured_logs: list[dict[str, Any]] = []
        self._log_handler: logging.Handler | None = None
        self._prev_root_log_level: int | None = None

        # Runtime kill-switch. The explicit `enabled` arg wins; the env var
        # gives ops a way to disable telemetry across a fleet without code
        # changes. Disabled observers no-op every public method except the
        # constructor, so an integration can be flipped off in production
        # without any pipeline edits.
        env_disabled = os.getenv("SUPERBRYN_OBSERVER_ENABLED", "").strip().lower() in (
            "0",
            "false",
            "no",
            "off",
        )
        self._enabled = enabled and not env_disabled

        if not self._enabled:
            logger.info(
                "SuperbrynObserver disabled (enabled=%s, env_override=%s)", enabled, env_disabled
            )
        elif not self.api_key:
            logger.warning("SUPERBRYN_API_KEY not configured — SuperbrynObserver will no-op.")

    # ── Pipeline lifecycle ────────────────────────────────────────────────

    async def on_pipeline_started(self) -> None:
        if not self._enabled:
            return

        self.started_at = datetime.now(UTC)
        self._call_start_ms = int(self.started_at.timestamp() * 1000)
        logger.info("SUPERBRYN_PIPECAT_CALL_STARTED: session_id=%s", self.session_id)

        # Attach session log capture (Cekura parity). Done at start, not
        # in the constructor, so the buffer covers exactly the call window
        # — startup / shutdown noise stays out of the dashboard.
        self._attach_log_handler()

        # Auto-start the in-pipeline audio recorder. Doing this here (instead
        # of in `transport.event_handler("on_client_connected")`) means
        # customers don't have to wire a separate start call — `observe_*`
        # entry-points become true one-liners.
        if self.audio_recorder is not None:
            try:
                await self.audio_recorder.start()
            except Exception as exc:  # noqa: BLE001 — fail-open
                logger.warning("audio recorder auto-start failed: %s", exc)

    async def on_pipeline_finished(self) -> None:
        """Legacy Pipecat lifecycle hook.

        Older Pipecat releases (< 1.3) called this on observers directly.
        Pipecat 1.3 removed it from `BaseObserver` and only fires it as a
        task-level event handler, so observers no longer get the hook
        automatically. We keep this method for backward compatibility — it
        simply delegates to the version-agnostic `_finalize_session()`,
        which is also driven from `on_push_frame` on Pipecat 1.3+.
        """
        await self._finalize_session()

    # ── High-level entrypoints (recommended) ─────────────────────────────
    #
    # `monitor_*` is production (records audio + ships to SuperBryn).
    # `simulate_*` is UAT / scenario runs (transcript + metrics only, no
    # audio, no billing). Both wrap pipeline mutation + `PipelineTask`
    # construction + lifecycle handler registration into a single call,
    # hiding the AudioBufferProcessor + observer wiring.
    #
    # Production:
    #     task = observer.monitor_and_create_task(pipeline, context,
    #                                             runner_args=runner_args,
    #                                             transport=transport)
    #
    # UAT / simulation (no audio):
    #     task = observer.simulate_and_create_task(pipeline, context,
    #                                              runner_args=runner_args,
    #                                              transport=transport)
    #
    # The earlier `observe_*` / `track_*` names are kept as deprecated
    # aliases for one minor release (removal target: 0.8.0). They emit a
    # ``DeprecationWarning`` and delegate to the new names.

    def monitor_pipeline(
        self,
        pipeline: Any,
        _context: Any = None,
        *,
        runner_args: Any | None = None,
        session_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        num_channels: int = 2,
        defer_upload: bool = False,
    ) -> Any:
        """Production mode: insert audio capture into a Pipecat ``Pipeline``
        and return the rebuilt pipeline.

        The ``AudioBufferProcessor`` is placed immediately *before* the
        transport's output processor — see ``_rebuild_pipeline_with``
        for why "before" rather than "after". When the recorder was
        already supplied to the constructor we reuse it; otherwise a
        stereo (user-left, bot-right) recorder is created on the fly.

        ``_context`` is accepted for forward compatibility (aggregator
        introspection for channel assignment) but is currently unused —
        we default to stereo with the convention `left=user`, `right=bot`.
        """
        if not self._enabled:
            return pipeline

        self._apply_runtime_overrides(runner_args, session_id, custom_metadata)
        self._defer_upload = bool(defer_upload)
        if self._defer_upload:
            self._deferred_release = asyncio.Event()

        if self.audio_recorder is None:
            self.audio_recorder = SuperbrynAudioRecorder(num_channels=num_channels)

        if self.audio_recorder.processor is None:
            # Recorder couldn't build its processor (pipecat missing or
            # incompatible). Skip the rebuild — observer still captures
            # transcripts via on_push_frame.
            logger.warning(
                "monitor_pipeline: AudioBufferProcessor unavailable; falling back to simulate mode"
            )
            self.audio_recorder = None
            return pipeline

        return self._rebuild_pipeline_with(pipeline, self.audio_recorder.processor)

    def simulate_pipeline(
        self,
        pipeline: Any,
        _context: Any = None,
        *,
        runner_args: Any | None = None,
        session_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
    ) -> Any:
        """UAT / simulation mode: transcript + metrics only, no audio.

        Returns the pipeline unmodified — the observer still hooks
        ``on_push_frame`` to capture transcript turns, metrics, and end
        reasons, but no ``AudioBufferProcessor`` is inserted and no S3
        upload runs.
        """
        if not self._enabled:
            return pipeline

        self._apply_runtime_overrides(runner_args, session_id, custom_metadata)

        # Simulate mode explicitly disables any pre-wired recorder so users
        # can flip between monitor/simulate without rebuilding the observer.
        # The wire-format `mode` value stays "track" for backend compatibility.
        self.audio_recorder = None
        self.extra_metadata.setdefault("mode", "track")
        return pipeline

    def monitor_and_create_task(
        self,
        pipeline: Any,
        context: Any = None,
        *,
        runner_args: Any | None = None,
        transport: Any | None = None,
        session_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        num_channels: int = 2,
        defer_upload: bool = False,
        **pipeline_task_kwargs: Any,
    ) -> Any:
        """Single-step production setup: wrap pipeline, build ``PipelineTask``,
        register handlers, return the task.

        Equivalent to::

            pipeline = observer.monitor_pipeline(pipeline, context,
                                                 runner_args=runner_args,
                                                 session_id=session_id,
                                                 custom_metadata=custom_metadata,
                                                 num_channels=num_channels)
            task = PipelineTask(pipeline, observers=[observer, ...],
                                params=PipelineParams(enable_usage_metrics=True),
                                **pipeline_task_kwargs)
            task = observer.register_task_handlers(task, transport=transport)

        Customers who need finer control should call the steps
        individually. Extra ``PipelineTask`` kwargs are forwarded
        as-is.
        """
        pipeline = self.monitor_pipeline(
            pipeline,
            context,
            runner_args=runner_args,
            session_id=session_id,
            custom_metadata=custom_metadata,
            num_channels=num_channels,
            defer_upload=defer_upload,
        )
        task = self._build_pipeline_task(pipeline, pipeline_task_kwargs)
        return self.register_task_handlers(task, transport=transport)

    def simulate_and_create_task(
        self,
        pipeline: Any,
        context: Any = None,
        *,
        runner_args: Any | None = None,
        transport: Any | None = None,
        session_id: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        **pipeline_task_kwargs: Any,
    ) -> Any:
        """Single-step UAT setup. See :meth:`monitor_and_create_task`."""
        pipeline = self.simulate_pipeline(
            pipeline,
            context,
            runner_args=runner_args,
            session_id=session_id,
            custom_metadata=custom_metadata,
        )
        task = self._build_pipeline_task(pipeline, pipeline_task_kwargs)
        return self.register_task_handlers(task, transport=transport)

    # ── Deprecated aliases (target removal: 0.8.0) ───────────────────────

    def observe_pipeline(self, *args: Any, **kwargs: Any) -> Any:
        """Deprecated: use :meth:`monitor_pipeline` instead."""
        warnings.warn(
            "SuperbrynObserver.observe_pipeline is deprecated; "
            "use monitor_pipeline instead. Will be removed in 0.8.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.monitor_pipeline(*args, **kwargs)

    def track_pipeline(self, *args: Any, **kwargs: Any) -> Any:
        """Deprecated: use :meth:`simulate_pipeline` instead."""
        warnings.warn(
            "SuperbrynObserver.track_pipeline is deprecated; "
            "use simulate_pipeline instead. Will be removed in 0.8.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.simulate_pipeline(*args, **kwargs)

    def observe_and_create_task(self, *args: Any, **kwargs: Any) -> Any:
        """Deprecated: use :meth:`monitor_and_create_task` instead."""
        warnings.warn(
            "SuperbrynObserver.observe_and_create_task is deprecated; "
            "use monitor_and_create_task instead. Will be removed in 0.8.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.monitor_and_create_task(*args, **kwargs)

    def track_and_create_task(self, *args: Any, **kwargs: Any) -> Any:
        """Deprecated: use :meth:`simulate_and_create_task` instead."""
        warnings.warn(
            "SuperbrynObserver.track_and_create_task is deprecated; "
            "use simulate_and_create_task instead. Will be removed in 0.8.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.simulate_and_create_task(*args, **kwargs)

    def register_task_handlers(self, task: Any, *, transport: Any | None = None) -> Any:
        """Register the observer's finalize hook on a Pipecat 1.3+ ``PipelineTask``.

        Pipecat 1.3 fires ``on_pipeline_finished`` only as a task-level event
        handler — observers no longer receive it. The ``on_push_frame``
        terminal-frame fallback can race with ``task.cancel()`` teardown,
        so the recording fetch + webhook never run on a forced disconnect.

        Call this once after constructing the ``PipelineTask`` to guarantee
        the finalize step runs and is awaited by Pipecat's cleanup. The
        single-step ``observe_and_create_task`` / ``track_and_create_task``
        wrappers do this for you.

        When ``transport`` is provided, we also try to hook the transport's
        client-disconnect events so a hard hangup still triggers finalize
        even if Pipecat's terminal frames are swallowed during teardown.

        Returns the task so callers can chain.
        """
        register = getattr(task, "event_handler", None)
        if register is not None:

            @register("on_pipeline_finished")  # type: ignore[misc]
            async def _on_finished(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
                # Pipecat 1.3 fires this as ``handler(task, frame)``; older
                # versions fire it as ``handler(task)``. Accept any signature.
                await self._finalize_session()

        if transport is not None:
            self._register_transport_disconnect(transport)

        return task

    # ── Backward-compatible alias ────────────────────────────────────────

    def attach_to_task(self, task: Any) -> None:
        """Deprecated alias for :meth:`register_task_handlers`.

        Kept for 0.3.x callers. New code should use
        ``register_task_handlers(task, transport=transport)`` so transport
        disconnect events are also wired up.
        """
        self.register_task_handlers(task)

    # ── Deferred upload / consent API (Cekura-parity) ────────────────────
    #
    # Flow:
    #   1. Customer constructs the SDK with `defer_upload=True`.
    #   2. Audio is captured locally as usual (AudioBufferProcessor still
    #      runs, transcript turns + metrics still accumulate).
    #   3. The pipeline reaches a terminal frame OR is cancelled.
    #   4. `_finalize_session` awaits `_deferred_release.set()` before
    #      doing the S3 upload + JSON webhook. While waiting it shields
    #      itself from asyncio cancellation so a `task.cancel()` doesn't
    #      orphan the upload.
    #   5. Customer calls `await observer.start_audio_upload()` from
    #      their consent handler → release fires → upload + webhook run.
    #   6. Customer calls `await observer.abort()` instead → release
    #      fires with `_aborted=True` → finalize cleans up + suppresses
    #      the webhook entirely.

    async def start_audio_upload(self) -> None:
        """Release the consent gate and let the buffered audio + webhook
        be sent. No-op when `defer_upload` was not enabled or the
        session has already been finalized.
        """
        if not self._defer_upload or self._deferred_release is None:
            logger.debug("start_audio_upload called without defer_upload=True; ignoring")
            return
        self._aborted = False
        self._deferred_release.set()
        logger.info("SUPERBRYN_PIPECAT_CONSENT_GRANTED: session_id=%s", self.session_id)

    async def abort(self) -> None:
        """Stop capture, discard any buffered audio, and suppress the
        webhook for this session. Safe to call at any time before the
        session has already been finalized. After abort, the SDK is
        a no-op for the remainder of the call.
        """
        self._aborted = True

        # Stop the recorder + delete the temp WAV before we let
        # `_finalize_session` see the flag — keeps customer audio off
        # disk as quickly as possible.
        if self.audio_recorder is not None:
            try:
                await self.audio_recorder.stop()
            except Exception as exc:  # noqa: BLE001
                logger.debug("audio_recorder.stop during abort swallowed: %s", exc)

        # `_cleanup_pending_audio` also calls `audio_recorder.cleanup()`,
        # so this single call covers both the "finalize already produced
        # a temp WAV" race and the simple "recorder still buffering"
        # case in one place.
        self._cleanup_pending_audio()

        # Release the deferred-upload gate if one is in play so
        # `_finalize_session` unblocks (and short-circuits on
        # `_aborted=True`).
        if self._deferred_release is not None:
            self._deferred_release.set()

        # Detach the log handler immediately so further log records
        # aren't buffered for a session we've explicitly disowned.
        self._detach_log_handler()

        logger.info("SUPERBRYN_PIPECAT_ABORTED: session_id=%s", self.session_id)

    # ── Mutable metadata API (Cekura-parity) ─────────────────────────────

    def set_custom_metadata(self, metadata: dict[str, Any]) -> None:
        """Merge ``metadata`` into the bag that will land on the outgoing
        webhook's ``call.metadata`` block. Mutable until ``_finalize_session``
        actually sends the payload; calls made after that point silently
        no-op so cleanup handlers don't crash on a missed deadline.
        """
        if not metadata:
            return
        if self._sent:
            logger.debug("set_custom_metadata called after webhook send — ignoring")
            return
        self.extra_metadata.update(metadata)

    def get_custom_metadata(self) -> dict[str, Any]:
        """Return a shallow copy of the current custom-metadata bag."""
        return dict(self.extra_metadata)

    # ── Session log capture (Cekura `capture_logs=True` parity) ──────────

    def _attach_log_handler(self) -> None:
        """Attach a buffering handler to the root logger so INFO+ records
        from anywhere in the agent process get captured into
        ``self._captured_logs``. Bounded by ``_max_log_records`` so a
        chatty agent can't blow up the webhook payload.
        """
        if not self._capture_logs or self._log_handler is not None or self._max_log_records == 0:
            return

        observer = self

        class _SessionLogHandler(logging.Handler):
            def emit(self_inner, record: logging.LogRecord) -> None:  # noqa: N805
                # Avoid recursion: drop log records from our own logger
                # name so emitting a log inside the handler doesn't
                # produce another handler call.
                if record.name.startswith("superbryn_pipecat_observer"):
                    return
                if len(observer._captured_logs) >= observer._max_log_records:
                    return
                try:
                    message = record.getMessage()
                except Exception:  # noqa: BLE001
                    message = str(record.msg)
                observer._captured_logs.append(
                    {
                        "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(),
                        "level": record.levelname,
                        "name": record.name,
                        "message": message,
                    }
                )

        handler = _SessionLogHandler(level=logging.INFO)
        # Plain message-only formatter — we already serialise the level
        # + logger name into the record dict ourselves.
        handler.setFormatter(logging.Formatter("%(message)s"))
        root = logging.getLogger()
        root.addHandler(handler)
        # Root logger's effective level filters records BEFORE handlers
        # see them. If the host process leaves the root at WARNING (the
        # Python default) we'd silently miss every INFO record we
        # promised to capture. Save the previous level and lower the
        # floor to INFO; we restore it in `_detach_log_handler`.
        if root.level == logging.NOTSET or root.level > logging.INFO:
            self._prev_root_log_level = root.level
            root.setLevel(logging.INFO)
        self._log_handler = handler

    def _detach_log_handler(self) -> None:
        if self._log_handler is None:
            return
        root = logging.getLogger()
        try:
            root.removeHandler(self._log_handler)
        except Exception as exc:  # noqa: BLE001
            logger.debug("log handler detach swallowed: %s", exc)
        self._log_handler = None
        if self._prev_root_log_level is not None:
            try:
                root.setLevel(self._prev_root_log_level)
            except Exception as exc:  # noqa: BLE001
                logger.debug("log level restore swallowed: %s", exc)
            self._prev_root_log_level = None

    # ── Internal helpers for the high-level entrypoints ──────────────────

    def _apply_runtime_overrides(
        self,
        runner_args: Any | None,
        session_id: str | None,
        custom_metadata: dict[str, Any] | None,
    ) -> None:
        """Resolve session_id + metadata supplied to ``observe_pipeline`` /
        ``track_pipeline``. Precedence:

            explicit session_id > runner_args.session_id > current self.session_id
        """
        resolved_sid = session_id or self._extract_session_id_from_runner_args(runner_args)
        if resolved_sid:
            self.session_id = resolved_sid
        if custom_metadata:
            self.extra_metadata.update(custom_metadata)

    @staticmethod
    def _extract_session_id_from_runner_args(runner_args: Any | None) -> str | None:
        """Pipecat 1.3's ``RunnerArguments`` carries an optional ``session_id``.
        We sniff with ``getattr`` instead of importing the type so different
        Pipecat versions don't break this path.
        """
        if runner_args is None:
            return None
        sid = getattr(runner_args, "session_id", None)
        if sid:
            return str(sid)
        return None

    def _rebuild_pipeline_with(self, pipeline: Any, inserted_processor: Any) -> Any:
        """Return a new ``Pipeline`` with ``inserted_processor`` placed right
        *before* the transport's output processor.

        Why before, not after?
        ``BaseOutputTransport.process_frame`` in pipecat 1.3+ routes
        ``OutputAudioRawFrame`` straight to its media sender (write-to-wire)
        and never calls ``push_frame`` on it; ``InputAudioRawFrame`` arriving
        at the output transport is similarly sunk. So an
        ``AudioBufferProcessor`` placed *after* ``transport.output()`` sees
        neither side of the conversation and records nothing.

        Placing it *before* the output transport works because at that
        point both:
          • ``OutputAudioRawFrame`` from the TTS service is still flowing
            downstream (will be consumed by the output transport on the
            next hop).
          • ``InputAudioRawFrame`` from the input transport is still
            flowing downstream (STT services default
            ``audio_passthrough=True`` and intermediate LLM / aggregator
            services pass unknown frames through unchanged).

        Pipecat's ``Pipeline`` is immutable post-construction
        (``_link_processors`` binds the chain in ``__init__``), so we must
        rebuild rather than mutate. We strip the auto-injected
        ``PipelineSource`` / ``PipelineSink`` (always at index 0 and -1 of
        ``processors``) before rebuilding.
        """
        try:
            from pipecat.pipeline.pipeline import Pipeline
        except Exception as exc:  # pragma: no cover
            logger.warning("Pipeline rebuild skipped (pipecat not importable): %s", exc)
            return pipeline

        try:
            raw = list(pipeline.processors)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Pipeline rebuild skipped (no .processors accessor): %s", exc)
            return pipeline

        # Strip source/sink — they get re-added by Pipeline.__init__.
        user_processors = [
            p for p in raw if type(p).__name__ not in ("PipelineSource", "PipelineSink")
        ]

        output_idx = self._find_output_transport_index(user_processors)
        if output_idx is None:
            # Couldn't sniff the output transport — drop the recorder at the
            # end as a safe fallback. The customer can still place it
            # explicitly by skipping `observe_pipeline` and constructing the
            # pipeline themselves.
            logger.warning(
                "observe_pipeline: could not locate transport.output() — "
                "inserting recorder at end of pipeline (may miss bot audio)"
            )
            insert_at = len(user_processors)
        else:
            insert_at = output_idx  # insert BEFORE the output transport

        new_processors = list(user_processors)
        new_processors.insert(insert_at, inserted_processor)
        return Pipeline(new_processors)

    def _find_output_transport_index(self, processors: list[Any]) -> int | None:
        """Locate the transport's output processor in the pipeline.

        Strategy:
          1. If the observer was constructed with a live ``transport``
             object, ask it for ``.output()`` and match by identity.
          2. Otherwise sniff for a processor whose class name contains
             ``Output`` and ``Transport`` (e.g. ``DailyOutputTransport``,
             ``FastAPIWebsocketOutputTransport``).
        """
        if self._transport_obj is not None:
            try:
                output = self._transport_obj.output()
            except Exception:  # noqa: BLE001
                output = None
            if output is not None:
                for idx, p in enumerate(processors):
                    if p is output:
                        return idx

        for idx, p in enumerate(processors):
            name = type(p).__name__
            if "Output" in name and "Transport" in name:
                return idx
        return None

    def _build_pipeline_task(
        self,
        pipeline: Any,
        pipeline_task_kwargs: dict[str, Any],
    ) -> Any:
        """Construct a ``PipelineTask`` with this observer attached. Honors
        any user-supplied ``observers`` / ``params`` in ``pipeline_task_kwargs``.
        """
        try:
            from pipecat.pipeline.task import PipelineParams, PipelineTask
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "pipecat is required for observe_and_create_task / track_and_create_task — "
                "install pipecat-ai>=0.0.50"
            ) from exc

        user_observers = pipeline_task_kwargs.pop("observers", None) or []
        observers = [self, *user_observers]

        params = pipeline_task_kwargs.pop("params", None)
        if params is None:
            params = PipelineParams(enable_usage_metrics=True)

        return PipelineTask(pipeline, observers=observers, params=params, **pipeline_task_kwargs)

    def _register_transport_disconnect(self, transport: Any) -> None:
        """Hook the transport's disconnect event so a hard hangup still
        triggers ``_finalize_session``. The exact event name varies across
        Pipecat transports — we register against every candidate and let
        the transport raise on the ones it doesn't know.
        """
        register = getattr(transport, "event_handler", None)
        if register is None:
            return

        candidate_events = (
            "on_client_disconnected",  # WebRTC / FastAPI WS / Twilio / Plivo / Vobiz
            "on_participant_left",  # Daily / LiveKit-style
            "on_disconnect",  # legacy
        )

        for event_name in candidate_events:
            try:

                @register(event_name)  # type: ignore[misc]
                async def _on_disconnect(*_args: Any, **_kwargs: Any) -> None:  # noqa: ANN401
                    await self._finalize_session()
            except Exception as exc:  # noqa: BLE001
                logger.debug("transport.event_handler(%s) refused: %s", event_name, exc)

    async def _finalize_session(self) -> None:
        """Build the call payload and ship it to SuperBryn.

        Idempotent: only runs once per observer instance, guarded by the
        ``_sent`` flag. Safe to call from any number of triggers (legacy
        observer lifecycle, terminal-frame detection in `on_push_frame`,
        explicit shutdown hooks, …).

        Cancellation-safe: when triggered from ``on_push_frame`` during
        pipeline teardown (CancelFrame), the asyncio Task driving
        ``on_push_frame`` may be cancelled mid-await. ``asyncio.shield()``
        schedules the webhook POST as an independent Task that runs to
        completion even if the caller's Task is cancelled. We suppress the
        resulting ``CancelledError`` so the caller returns cleanly.

        Recording strategy: the in-pipeline ``AudioBufferProcessor``
        (transport-agnostic) flushes its buffer to a temp WAV. The SDK
        then asks orchestration for a presigned PUT URL via
        ``POST /api/recording-upload-url`` and uploads the WAV
        **directly to S3** from this process — orchestration never
        sees the audio bytes. The resulting public URL is stamped
        onto the JSON call payload before the webhook fires. Two
        outbound HTTP calls per session (presign + S3 PUT) instead of
        a multi-megabyte multipart body.

        The temp WAV is deleted in a ``finally`` block whether the
        upload succeeded or failed, so a crash mid-upload still leaves
        nothing in ``/tmp``.
        """
        if not self._enabled:
            return
        if self._sent:
            return
        self._sent = True
        self.ended_at = datetime.now(UTC)

        # Always stop log capture as part of finalize — even when deferred
        # uploads make us wait below, the log set is what the user already
        # captured during the call window.
        self._detach_log_handler()

        # Deferred-upload gate. When `defer_upload=True` was set on
        # `observe_pipeline`, the host process must call either
        # `start_audio_upload()` or `abort()` before any data leaves
        # this process. If neither call happens before the session
        # tears down, default behaviour is to DISCARD (matches Cekura's
        # compliance contract: "If neither method is called before the
        # session ends, all data is automatically discarded.").
        if self._defer_upload and self._deferred_release is not None:
            try:
                # Bound the wait so a stuck consent handler can't pin a
                # PipelineRunner forever. 30 s mirrors the audio-upload
                # timeout we already enforce on the S3 PUT step.
                await asyncio.wait_for(self._deferred_release.wait(), timeout=30.0)
            except TimeoutError:
                logger.warning(
                    "SUPERBRYN_PIPECAT_CONSENT_TIMEOUT: no start_audio_upload()/abort() "
                    "within 30s — discarding session %s",
                    self.session_id,
                )
                self._aborted = True
            except asyncio.CancelledError:
                logger.debug("deferred wait cancelled during finalize teardown")
                self._aborted = True

        if self._aborted:
            # Clean up local state without sending the webhook. Recorder
            # has already been stopped + cleaned by `abort()` in the
            # happy path; this is a belt-and-braces sweep for the
            # timeout path above (idempotent).
            self._cleanup_pending_audio()
            logger.info("SUPERBRYN_PIPECAT_SESSION_DISCARDED: session_id=%s", self.session_id)
            return

        # 1) Flush the in-pipeline audio buffer to a temp WAV. We don't
        #    upload here — the file path is stashed on
        #    `_pending_audio_path` and uploaded directly to S3 from
        #    `_send_webhook` via a presigned PUT URL fetched from
        #    `/api/recording-upload-url`.
        if self.audio_recorder is not None:
            try:
                wav_path = await asyncio.shield(self.audio_recorder.finalize(self))
            except asyncio.CancelledError:
                logger.debug(
                    "_finalize_session: audio finalize cancelled; webhook will fire without audio"
                )
                wav_path = None
            except Exception as exc:  # noqa: BLE001
                logger.warning("audio finalize failed (continuing without recording): %s", exc)
                wav_path = None
            self._pending_audio_path = wav_path

        # 2) Fire the webhook. If a WAV was captured, _send_webhook
        # first uploads it directly to S3 via a presigned PUT URL and
        # stamps the resulting URL onto the payload, then POSTs the
        # JSON payload. The temp WAV is deleted in the `finally`
        # block below before this function returns.
        try:
            await asyncio.shield(self._send_webhook())
        except asyncio.CancelledError:
            logger.debug("_finalize_session cancelled; webhook task is running independently")
        finally:
            self._cleanup_pending_audio()

    def _cleanup_pending_audio(self) -> None:
        """Delete the temp WAV captured for this session, if any.

        Idempotent — safe to call from `abort()`, the `finally` block of
        `_finalize_session`, and any teardown path. We never want
        customer audio sitting on disk past the lifetime of the upload.
        """
        wav_path = self._pending_audio_path
        self._pending_audio_path = None
        if wav_path is not None:
            try:
                wav_path.unlink(missing_ok=True)
            except Exception as exc:  # noqa: BLE001
                logger.debug("temp WAV cleanup failed: %s", exc)
        if self.audio_recorder is not None:
            try:
                self.audio_recorder.cleanup()
            except Exception as exc:  # noqa: BLE001
                logger.debug("audio_recorder.cleanup swallowed: %s", exc)

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
            source = getattr(data, "source", None)
            source_module = type(source).__module__ if source is not None else ""

            if cls_name == "TranscriptionFrame":
                # ``TranscriptionFrame`` traverses every downstream
                # processor (STT → user aggregator → LLM → …). Pipecat
                # fires ``on_push_frame`` once per link, so without this
                # source filter we'd register the same user turn 4-5
                # times. Capture only on the first hop (from the STT
                # service itself).
                if "pipecat.services." in source_module and "stt" in source_module:
                    self._capture_user_turn(frame)
            elif cls_name in _BOT_TEXT_FRAME_NAMES and self._looks_like_bot_text(frame):
                # Same dedupe problem for bot text. ``LLMTextFrame`` (and
                # subclasses) flow from the LLM service through TTS,
                # AudioBufferProcessor, output transport, and assistant
                # aggregator — each link fires ``on_push_frame``. Capture
                # only on the producer-side hop so a single LLM token
                # becomes a single appended chunk in the turn buffer
                # (not 3-4 duplicate appends).
                if "pipecat.services." in source_module and (
                    "llm" in source_module or "tts" in source_module
                ):
                    self._capture_bot_turn(frame)
            elif cls_name in ("LLMFullResponseStartFrame", "BotStartedSpeakingFrame"):
                self._mark_bot_response_start()
            elif cls_name in ("LLMFullResponseEndFrame", "BotStoppedSpeakingFrame"):
                # Close the open bot turn so the next response starts a
                # fresh turn instead of appending to the previous one.
                self._close_bot_turn()
            elif cls_name == "MetricsFrame":
                self._capture_metrics(frame)
            elif cls_name == "UserStartedSpeakingFrame":
                self._mark_user_start()
            elif cls_name == "UserStoppedSpeakingFrame":
                self._mark_user_stop()
            elif cls_name == "FunctionCallInProgressFrame":
                self._capture_tool_call_start(frame)
            elif cls_name == "FunctionCallResultFrame":
                self._capture_tool_call_result(frame)
            elif cls_name in _TERMINAL_FRAME_NAMES:
                self._capture_end_reason(cls_name)
                # Pipecat 1.3 stopped fanning out `on_pipeline_finished` to
                # observers — so a terminal frame in the stream is now our
                # primary signal that the call is over. `_finalize_session`
                # is idempotent, so older Pipecat versions that also call
                # `on_pipeline_finished` won't double-send.
                await self._finalize_session()

            # First time we see a service frame, infer provider from module path
            if source is not None:
                self._sniff_provider(source, cls_name)

        except Exception as exc:  # noqa: BLE001 — observer must never raise
            logger.debug("on_push_frame swallowed error: %s", exc)

    # ── Capture helpers ──────────────────────────────────────────────────

    def _now_ms(self) -> int:
        return int(datetime.now(UTC).timestamp() * 1000) - (self._call_start_ms or 0)

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
        # Anthropic / OpenAI LLMs stream their response as many
        # ``LLMTextFrame`` chunks ("Hello", ",", " I'm", " here", ...).
        # We want one logical bot turn per response, so we append text
        # into the most recent open agent turn until it's closed by
        # ``BotStoppedSpeakingFrame`` / ``LLMFullResponseEndFrame``
        # (handled in ``_close_bot_turn``).
        for turn in reversed(self.transcript_turns):
            if turn["speaker"] == "agent" and turn.get("_open"):
                existing = turn["text"]
                if existing:
                    # Join with a space unless the next chunk already
                    # starts with whitespace or punctuation that wouldn't
                    # take a leading space.
                    joiner = "" if text[:1] in (" ", ".", ",", "!", "?", ";", ":") else " "
                    turn["text"] = f"{existing}{joiner}{text}"
                else:
                    turn["text"] = text
                turn["end_time_ms"] = now_ms
                return
            if turn["speaker"] == "agent":
                # Found a closed agent turn — stop searching, fall
                # through and open a new one. Stops us from mutating
                # an older closed turn if the start-frame fired late.
                break
        self.transcript_turns.append(
            {
                "speaker": "agent",
                "text": text,
                "start_time_ms": now_ms,
                "end_time_ms": now_ms,
                "latency_ms": (
                    now_ms - self._last_user_end_ms if self._last_user_end_ms is not None else None
                ),
                "_open": True,
            }
        )

    def _close_bot_turn(self) -> None:
        """Mark the most recent open agent turn as closed.

        Called when ``BotStoppedSpeakingFrame`` or
        ``LLMFullResponseEndFrame`` fires. After this, the next
        ``LLMTextFrame`` will start a brand-new turn instead of
        appending to the previous one. Idempotent so the repeated
        propagation of those frames across pipeline links doesn't
        accidentally re-close an already-closed turn.

        Before flipping the open flag we run the closed turn through
        ``_extract_prompt_tool_calls`` so any ``<tool_use>...</tool_use>``
        blocks the LLM emitted as plain text get lifted onto
        ``self.tool_calls`` and stripped from the transcript text. We do
        this here (rather than per-chunk in ``_capture_bot_turn``) so a
        single XML block streamed across multiple ``LLMTextFrame`` chunks
        is reassembled into the turn first.
        """
        if not self._bot_turn_open:
            return
        self._bot_turn_open = False
        for turn in reversed(self.transcript_turns):
            if turn["speaker"] == "agent" and turn.get("_open"):
                self._extract_prompt_tool_calls(turn)
                turn["_open"] = False
                return

    def _extract_prompt_tool_calls(self, turn: dict[str, Any]) -> None:
        """Lift Anthropic-style prompt tool calls out of a closed agent turn.

        Searches ``turn["text"]`` for ``<tool_use>{...}</tool_use>`` blocks,
        records each as a structured entry on ``self.tool_calls`` matching
        the existing real-tool record shape (``function_name``,
        ``arguments``, ``result``, ``timestamp_ms``, ``tool_call_id``),
        and replaces the XML in the turn text with empty so the dashboard
        transcript reads as natural speech.

        A few intentional design points:
          * ``result`` is ``None`` because nothing actually executed —
            the LLM was just role-playing a tool call via the prompt. The
            host application is responsible for noticing the call and
            doing real work elsewhere.
          * ``arguments`` keeps the *entire* parsed JSON body (including
            the ``tool`` / ``name`` discriminator). Downstream consumers
            already have ``function_name``; keeping the full body avoids
            losing information when a custom prompt uses non-standard
            keys.
          * Malformed JSON inside a block is logged and left in place so
            we don't silently lose data the operator might want to see.
        """
        text = turn.get("text") or ""
        if not text or "<tool_use>" not in text:
            return

        timestamp_ms = turn.get("start_time_ms")
        new_text_parts: list[str] = []
        cursor = 0
        added = False

        for match in _TOOL_USE_PATTERN.finditer(text):
            body = match.group(1).strip()
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                logger.warning(
                    "SUPERBRYN_PIPECAT_TOOL_USE_PARSE_FAILED: %s; leaving XML in transcript",
                    exc,
                )
                continue

            if not isinstance(parsed, dict):
                logger.warning(
                    "SUPERBRYN_PIPECAT_TOOL_USE_NOT_OBJECT: parsed=%r; leaving XML in transcript",
                    parsed,
                )
                continue

            function_name = (
                parsed.get("tool") or parsed.get("name") or parsed.get("function_name") or "unknown"
            )

            self.tool_calls.append(
                {
                    "tool_call_id": str(uuid.uuid4()),
                    "function_name": str(function_name),
                    "arguments": parsed,
                    "result": None,
                    "timestamp_ms": timestamp_ms,
                    "source": "prompt_tool_use",
                }
            )
            added = True

            new_text_parts.append(text[cursor : match.start()])
            cursor = match.end()

        if not added:
            return  # nothing matched cleanly — leave the turn alone

        new_text_parts.append(text[cursor:])
        cleaned = re.sub(r"\s+", " ", "".join(new_text_parts)).strip()
        turn["text"] = cleaned

    def _mark_bot_response_start(self) -> None:
        # ``BotStartedSpeakingFrame`` / ``LLMFullResponseStartFrame``
        # propagate to every downstream processor, so ``on_push_frame``
        # sees them multiple times per actual start. We track an open-
        # turn flag so each logical bot response gets exactly one
        # placeholder turn.
        if self._bot_turn_open:
            return
        self._bot_turn_open = True
        now_ms = self._now_ms()
        latency = now_ms - self._last_user_end_ms if self._last_user_end_ms is not None else None
        if latency is not None and latency >= 0:
            self.latencies_ms.append(float(latency))
        self.transcript_turns.append(
            {
                "speaker": "agent",
                "text": "",
                "start_time_ms": now_ms,
                "end_time_ms": None,
                "latency_ms": latency,
                "_open": True,
            }
        )

    def _mark_user_start(self) -> None:
        """Stash the VAD-derived start of a user speech segment.

        Used as a fallback signal for ``stt_duration_seconds`` when the
        STT service didn't emit ``STTUsageMetricsData`` (e.g. the call
        was cancelled mid-utterance). Recorded as wall-time-since-call-
        start so we can compute a segment length on the matching
        ``UserStoppedSpeakingFrame``.
        """
        if self._user_speech_start_ms is None:
            self._user_speech_start_ms = self._now_ms()

    def _mark_user_stop(self) -> None:
        now_ms = self._now_ms()
        self._last_user_end_ms = now_ms
        # If we paired with a UserStartedSpeakingFrame, accumulate the
        # segment length into our VAD-derived total. Only used at
        # payload-build time if STTUsageMetricsData never fired.
        if self._user_speech_start_ms is not None:
            seg_ms = max(0, now_ms - self._user_speech_start_ms)
            self._vad_user_speech_ms += seg_ms
            self._user_speech_start_ms = None

    def _capture_tool_call_start(self, frame: Any) -> None:
        """Stash the start of a function-call invocation by `tool_call_id`.

        Pipecat's `FunctionCallInProgressFrame` fires the moment the LLM
        emits a tool call, before the tool returns. We open a record now
        so we have an accurate `timestamp_ms` even if the tool runs for
        a long time.
        """
        tool_call_id = getattr(frame, "tool_call_id", None)
        name = getattr(frame, "function_name", None)
        if not tool_call_id or not name:
            return
        record: dict[str, Any] = {
            "tool_call_id": tool_call_id,
            "function_name": name,
            "arguments": _coerce_tool_payload(getattr(frame, "arguments", None)),
            "timestamp_ms": self._now_ms(),
        }
        self._tool_calls_by_id[tool_call_id] = record
        self.tool_calls.append(record)

    def _capture_tool_call_result(self, frame: Any) -> None:
        """Attach the tool's return value to the record opened above.

        We match on `tool_call_id` so concurrent / interleaved tool calls
        stay associated correctly. If we somehow see a result without a
        preceding start frame (e.g. observer attached mid-call), we
        append a fresh record so the data isn't lost.
        """
        tool_call_id = getattr(frame, "tool_call_id", None)
        if not tool_call_id:
            return
        record = self._tool_calls_by_id.get(tool_call_id)
        if record is None:
            record = {
                "tool_call_id": tool_call_id,
                "function_name": getattr(frame, "function_name", None),
                "arguments": _coerce_tool_payload(getattr(frame, "arguments", None)),
                "timestamp_ms": self._now_ms(),
            }
            self._tool_calls_by_id[tool_call_id] = record
            self.tool_calls.append(record)
        record["result"] = _coerce_tool_payload(getattr(frame, "result", None))

    def _capture_metrics(self, frame: Any) -> None:
        """
        Pipecat emits MetricsFrame containing a list of typed metric records.
        We pluck LLM token counts, STT audio seconds, and TTS character counts.
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
            elif cls_name == "STTUsageMetricsData":
                # Pipecat STT services emit `value` as seconds of audio
                # processed since the last report. Sum them so we
                # correctly track total speech-to-text duration even
                # across reconnects.
                self.usage["stt_duration_seconds"] += float(getattr(rec, "value", 0) or 0)
                if getattr(rec, "model", None):
                    self.usage["stt_model"] = rec.model
                    if not self.usage["stt_provider"]:
                        self.usage["stt_provider"] = detect_provider_from_model(rec.model)
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

    # Known Pipecat service module → provider name. Mapped explicitly
    # instead of going through `detect_provider_from_model` (which expects
    # actual model names like "claude-3-5-sonnet" or "nova-3", not module
    # paths). Extend as new providers land in pipecat.services.*.
    _MODULE_PROVIDER_MAP: ClassVar[dict[str, str]] = {
        "anthropic": "anthropic",
        "openai": "openai",
        "google": "google",
        "gemini": "google",
        "azure": "azure",
        "groq": "groq",
        "together": "together",
        "deepgram": "deepgram",
        "assemblyai": "assemblyai",
        "cartesia": "cartesia",
        "elevenlabs": "elevenlabs",
        "rime": "rime",
        "playht": "playht",
    }

    @classmethod
    def _provider_from_module(cls, module: str) -> str | None:
        """Match a Pipecat service module path against the known-provider map.

        Pipecat groups services as `pipecat.services.<provider>.<role>`
        (e.g. `pipecat.services.anthropic.llm`). We look for a known
        provider keyword anywhere in the module string and return the
        first hit so subpackage layout changes don't break detection.
        """
        for needle, name in cls._MODULE_PROVIDER_MAP.items():
            if needle in module:
                return name
        return None

    @staticmethod
    def _read_attr_chain(source: Any, *names: str) -> Any:
        """Read the first non-None attribute from a list of candidates.

        Pipecat services keep their model / voice on different attribute
        names across versions and providers (`model`, `model_name`,
        `_model`, `settings.model`, `settings.voice`, `_voice_id`, …).
        This walker tries each path in order and returns the first hit
        without raising on missing intermediates.
        """
        for name in names:
            cur: Any = source
            for part in name.split("."):
                cur = getattr(cur, part, None)
                if cur is None:
                    break
            if cur not in (None, ""):
                return cur
        return None

    def _sniff_provider(self, source: Any, frame_cls: str) -> None:
        """
        Best-effort detection of which service produced this frame.
        Looks at the module path of the producing processor for the
        provider, and walks a small set of common attribute paths to
        find the model / voice id.

        We restrict to processors under ``pipecat.services.*`` because
        aggregators (e.g. ``pipecat.processors.aggregators.llm_response_universal``)
        also have ``llm`` / ``stt`` in their module path but aren't
        provider-backed — sniffing them once would lock in
        ``provider="unknown"`` and ignore the real service frame later.
        """
        try:
            module = type(source).__module__ or ""
            # Only services carry provider/model info.
            if "pipecat.services." not in module:
                return
            provider = self._provider_from_module(module)

            if "stt" in module:
                if not self.usage["stt_provider"] and provider:
                    self.usage["stt_provider"] = provider
                if not self.usage["stt_model"]:
                    self.usage["stt_model"] = self._read_attr_chain(
                        source,
                        "model_name",
                        "model",
                        "_model",
                        "_settings.model",
                        "settings.model",
                    )
            elif "llm" in module:
                if not self.usage["llm_provider"] and provider:
                    self.usage["llm_provider"] = provider
                if not self.usage["llm_model"]:
                    self.usage["llm_model"] = self._read_attr_chain(
                        source,
                        "model_name",
                        "model",
                        "_model",
                        "_settings.model",
                        "settings.model",
                    )
            elif "tts" in module:
                if not self.usage["tts_provider"] and provider:
                    self.usage["tts_provider"] = provider
                if not self.usage["tts_model"]:
                    self.usage["tts_model"] = self._read_attr_chain(
                        source,
                        "model_name",
                        "model",
                        "_model",
                        "_settings.model",
                        "settings.model",
                    )
                if not self.usage["tts_voice_id"]:
                    self.usage["tts_voice_id"] = self._read_attr_chain(
                        source,
                        "voice_id",
                        "voice",
                        "_voice_id",
                        "_voice",
                        "_settings.voice_id",
                        "_settings.voice",
                        "settings.voice_id",
                        "settings.voice",
                    )
        except Exception as exc:  # noqa: BLE001
            logger.debug("provider sniff failed: %s", exc)

    # ── Payload + send ───────────────────────────────────────────────────

    def _compute_audio_duration_seconds(self) -> float | None:
        """Compute the duration of the captured WAV from its byte size.

        Returns ``None`` when no recorder is wired, the WAV file is
        missing, the file is empty, or we don't have a sample rate to
        divide by — callers fall back to wall-clock in those cases.
        Uses the file size (minus the 44-byte RIFF header is *not*
        subtracted intentionally: the rounding noise is < 1 ms at
        16 kHz and keeps this consistent with how the audio meta
        block was computed historically).
        """
        recorder = self.audio_recorder
        wav_path = self._pending_audio_path
        if recorder is None or wav_path is None:
            return None
        if not recorder.sample_rate or not recorder.num_channels:
            return None
        try:
            size_bytes = wav_path.stat().st_size
        except OSError:
            return None
        if not size_bytes:
            return None
        # 16-bit PCM → 2 bytes per sample per channel.
        samples_per_channel = size_bytes / (2 * recorder.num_channels)
        return samples_per_channel / float(recorder.sample_rate)

    def _build_payload(self) -> dict[str, Any]:
        wallclock_duration_seconds = 0.0
        if self.started_at and self.ended_at:
            wallclock_duration_seconds = (self.ended_at - self.started_at).total_seconds()

        # Prefer the recorded audio's duration over wall-clock when a
        # recording exists. Pipecat's ``AudioBufferProcessor`` records
        # bot-track content at the rate TTS delivers it (Cartesia,
        # ElevenLabs, etc. stream audio faster than realtime over
        # websocket telephony) and then back-fills silence between turns
        # using wall-clock gaps. The net effect is that the WAV file
        # ends up a few seconds longer than ``ended_at − started_at``,
        # which is confusing on the dashboard because the *playable
        # recording* is what users perceive as the call. Reporting the
        # audio's duration here keeps "what plays back" and "how long
        # the call lasted" in sync.
        audio_duration_seconds = self._compute_audio_duration_seconds()
        duration_seconds = (
            audio_duration_seconds
            if audio_duration_seconds is not None
            else wallclock_duration_seconds
        )

        avg_latency = sum(self.latencies_ms) / len(self.latencies_ms) if self.latencies_ms else None
        p95_latency = None
        if self.latencies_ms:
            sorted_lat = sorted(self.latencies_ms)
            idx = max(0, int(len(sorted_lat) * 0.95) - 1)
            p95_latency = sorted_lat[idx]

        # Belt-and-braces: extract prompt-style `<tool_use>` blocks from any
        # agent turn that's still flagged open at payload-build time. That
        # happens when the call ends mid-LLM-response — `_close_bot_turn`
        # never fires, so the turn would otherwise ship with raw XML in its
        # text. `_extract_prompt_tool_calls` is idempotent on already-clean
        # turns, so this is safe to run on every agent turn.
        for t in self.transcript_turns:
            if t.get("speaker") == "agent" and t.get("_open"):
                self._extract_prompt_tool_calls(t)

        # Drop empty turns — matches livekit-evals' filter. Also strip
        # the internal ``_open`` marker (used by the streaming-LLM bot
        # turn accumulator) so it doesn't leak into the wire format.
        turns_with_text = [
            {k: v for k, v in t.items() if k != "_open"}
            for t in self.transcript_turns
            if t.get("text", "").strip()
        ]

        # Wire-format `mode` defaults to "observe" (production) so older
        # callers and the manual/advanced flow keep their existing
        # semantics. `simulate_pipeline` already sets
        # `extra_metadata["mode"] = "track"`; we don't override it here.
        # The string values "observe"/"track" are kept for backend
        # compatibility even though the public method names are now
        # monitor_*/simulate_*.
        metadata: dict[str, Any] = {
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
            "mode": "observe",
            **self.extra_metadata,
        }

        call_body: dict[str, Any] = {
            # `id` is the canonical call identifier across all SuperBryn
            # integrations (VAPI / Retell / Traces / …). At the SDK level
            # we still call it `session_id` in Python because that's what
            # Pipecat itself uses for a pipeline session — but on the wire
            # one field is enough; the adapter maps `call.id` →
            # `NormalizedCall.provider_call_id`.
            "id": self.session_id,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "duration_seconds": duration_seconds,
            "call_end_reason": self.call_end_reason or "completed",
            "from_number": self.from_number,
            "to_number": self.to_number,
            "transcript": {"turns": turns_with_text},
            "recording_url": self.recording_url,
            "stereo_recording_url": self.stereo_recording_url,
            "metadata": metadata,
            "usage": {
                "llm_input_tokens": self.usage["llm_input_tokens"],
                "llm_output_tokens": self.usage["llm_output_tokens"],
                # Prefer the STT service's own usage metrics when the
                # provider reports them; otherwise fall back to the
                # VAD-derived total so the dashboard still has a
                # reasonable speech-duration figure on aborted calls.
                "stt_duration_seconds": (
                    self.usage["stt_duration_seconds"]
                    if self.usage["stt_duration_seconds"] > 0
                    else round(self._vad_user_speech_ms / 1000.0, 3)
                ),
                "tts_characters": self.usage["tts_characters"],
            },
            "latency": {
                "avg_ms": avg_latency,
                "p95_ms": p95_latency,
            },
        }

        # Attach tool / function-call invocations. Shape matches the
        # `ExtraToolCall` contract in `obs-analysis.worker.ts`:
        # `{ function_name, arguments, result, timestamp_ms, tool_call_id }`.
        # The backend adapter lifts this onto `extra.tool_calls` and the
        # analysis worker stitches each call onto the nearest preceding
        # assistant turn by `timestamp_ms`.
        if self.tool_calls:
            call_body["tool_calls"] = list(self.tool_calls)

        # Attach session logs when there's anything to send. Done after
        # the dict literal to keep small / empty-log payloads tidy.
        if self._captured_logs:
            call_body["logs"] = list(self._captured_logs)

        return {
            "event": "call.completed",
            "sdk_version": _SDK_TAG,
            "call": call_body,
        }

    async def _send_webhook(self, *, label: str = "") -> None:
        """Ship the call payload to SuperBryn.

        Two-step delivery — orchestration never sees the audio bytes:

          1. **S3 upload (optional).** If a WAV was captured this
             session, fetch a presigned PUT URL from
             ``{api_base_url}/api/recording-upload-url`` and PUT the
             file straight to S3. The resulting public URL is stamped
             onto ``call.stereo_recording_url`` (or ``call.recording_url``
             for mono) before the JSON webhook fires.

          2. **JSON webhook.** POST the (now URL-bearing) payload to
             ``self.webhook_url`` as ``application/json``.

        Either step may fail; both fail open. A failed S3 upload still
        results in a JSON-only webhook (no recording URL), so the
        dashboard row lands without audio rather than dropping the call
        on the floor entirely.
        """
        if not self.api_key:
            logger.info("SUPERBRYN_PIPECAT_SKIPPED: no API key configured")
            return
        if not self.webhook_url:
            logger.warning("SUPERBRYN_PIPECAT_NO_URL: webhook URL not configured")
            return

        payload = self._build_payload()

        # ── Step 1: direct-to-S3 audio upload ─────────────────────────
        wav_path = self._pending_audio_path
        if wav_path is not None:
            await self._upload_audio_to_s3(payload, wav_path)

        # ── Step 2: JSON-only webhook ─────────────────────────────────
        try:
            import aiohttp  # lazy: optional at import time, required at send time
        except ImportError:
            logger.error(
                "SUPERBRYN_PIPECAT_MISSING_AIOHTTP: install aiohttp to enable webhook delivery"
            )
            return

        logger.info(
            "SUPERBRYN_PIPECAT_SENDING: session_id=%s url=%s",
            self.session_id,
            self.webhook_url,
        )
        # Dump the outbound payload for debugging. Kept on a single line so
        # log grep tooling can find it via the `SUPERBRYN_PIPECAT_PAYLOAD:`
        # prefix and so it lines up with the matching receive-side log on
        # the orchestration service.
        try:
            logger.info(
                "SUPERBRYN_PIPECAT_PAYLOAD: session_id=%s payload=%s",
                self.session_id,
                json.dumps(payload, default=str),
            )
        except Exception as exc:  # noqa: BLE001 — never let logging break delivery
            logger.warning("SUPERBRYN_PIPECAT_PAYLOAD_LOG_FAILED: %s", exc)

        # 30s is generous for a JSON-only POST (a few hundred KB at
        # most). The audio bytes have already gone direct to S3 above,
        # so the webhook itself no longer carries them.
        timeout = aiohttp.ClientTimeout(total=30)
        headers = {
            "X-API-Key": self.api_key,
            "Content-Type": "application/json",
        }

        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    self.webhook_url,
                    json=payload,
                    headers=headers,
                ) as resp:
                    await self._handle_webhook_response(resp)
        except Exception as exc:  # noqa: BLE001 — fail-open
            logger.error("SUPERBRYN_PIPECAT_ERROR: %s", exc, exc_info=True)

    async def _upload_audio_to_s3(
        self,
        payload: dict[str, Any],
        wav_path: Path,
    ) -> None:
        """Fetch a presigned PUT URL and upload the WAV directly to S3.

        Stamps the resulting public URL onto ``payload['call']`` so the
        JSON webhook arrives with the recording link already populated.
        Any failure short-circuits to a JSON-only webhook without an
        audio URL — the call record still lands.
        """
        try:
            size_bytes = wav_path.stat().st_size
        except OSError as exc:
            logger.warning("SUPERBRYN_PIPECAT_S3_STAT_FAILED: %s", exc)
            return
        if size_bytes == 0:
            logger.info("audio recorder produced an empty WAV; sending JSON-only")
            return

        if not self.api_base_url:
            logger.error(
                "SUPERBRYN_PIPECAT_S3_NO_API_BASE_URL: cannot fetch presigned upload URL; "
                "set SUPERBRYN_API_BASE_URL or pass api_base_url=... to SuperbrynObserver"
            )
            return

        num_channels = self.audio_recorder.num_channels if self.audio_recorder is not None else 2

        presign = await fetch_recording_upload_url(
            self.api_base_url,
            self.api_key,
            self.session_id,
            num_channels,
        )
        if presign is None:
            return  # fetch_recording_upload_url already logged the cause

        uploaded = await upload_wav_via_presigned(presign["upload_url"], wav_path)
        if not uploaded:
            return  # upload_wav_via_presigned already logged the cause

        call = payload.setdefault("call", {})
        if not isinstance(call, dict):
            logger.warning(
                "SUPERBRYN_PIPECAT_S3_PAYLOAD_SHAPE: 'call' is not a dict; "
                "skipping recording-URL stamp"
            )
            return

        public_url = presign["public_url"]
        is_mono = num_channels == 1
        if is_mono:
            call["recording_url"] = public_url
            call["mono_audio"] = True
        else:
            call["stereo_recording_url"] = public_url

        logger.info(
            "SUPERBRYN_PIPECAT_S3_UPLOADED: session_id=%s bytes=%d url=%s",
            self.session_id,
            size_bytes,
            public_url,
        )

    @staticmethod
    async def _handle_webhook_response(resp: Any) -> None:
        body = await resp.text()
        if resp.status == 200:
            logger.info("SUPERBRYN_PIPECAT_SENT: %s", body[:200])
        elif resp.status in (401, 403):
            logger.error(
                "SUPERBRYN_PIPECAT_AUTH_FAILED (%d): check SUPERBRYN_API_KEY",
                resp.status,
            )
        elif resp.status == 413:
            logger.error(
                "SUPERBRYN_PIPECAT_PAYLOAD_TOO_LARGE: JSON payload exceeded the 20MB "
                "backend cap (likely from a very large transcript / log payload); "
                "consider trimming `extra_metadata` or disabling `capture_logs`"
            )
        else:
            logger.error("SUPERBRYN_PIPECAT_HTTP_%d: %s", resp.status, body[:200])

    # ── Helpers ──────────────────────────────────────────────────────────

    def _looks_like_bot_text(self, frame: Any) -> bool:
        """
        A `TextFrame` can come from many places. We treat it as a bot turn
        only if it appears between LLM-response-start and the next user
        transcription — i.e. there's an open bot turn waiting for text.
        """
        return any(t["speaker"] == "agent" and not t["text"] for t in self.transcript_turns)
