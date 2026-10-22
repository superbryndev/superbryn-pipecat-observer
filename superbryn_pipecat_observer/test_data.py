"""Eval mode: send the agent's own record of each SuperBryn test call; other calls send nothing."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import re
import time
import uuid
from collections import deque
from collections.abc import Mapping
from datetime import UTC, datetime
from typing import Any
from urllib.parse import parse_qsl

import aiohttp

try:
    from pipecat.adapters.schemas.tools_schema import ToolsSchema
    from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
    from pipecat.frames.frames import (
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        CancelFrame,
        EndFrame,
        ErrorFrame,
        FunctionCallCancelFrame,
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        InputAudioRawFrame,
        InterruptionFrame,
        LLMContextFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        MetricsFrame,
        ServiceMetadataFrame,
        SpeechControlParamsFrame,
        StartFrame,
        StopFrame,
        TranscriptionFrame,
        TTSTextFrame,
        UserMuteStartedFrame,
        UserMuteStoppedFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
    )
    from pipecat.metrics.metrics import (
        LLMUsageMetricsData,
        TTFBMetricsData,
        TTSUsageMetricsData,
        TurnMetricsData,
    )
    from pipecat.observers.base_observer import BaseObserver, FramePushed
    from pipecat.processors.frame_processor import FrameDirection
    from pipecat.services.llm_service import LLMService
    from pipecat.services.stt_service import SegmentedSTTService, STTService
    from pipecat.services.tts_service import TTSService
    from pipecat.transports.base_input import BaseInputTransport
    from pipecat.transports.base_output import BaseOutputTransport
    from pipecat.utils.string import TextPartForConcatenation, concatenate_aggregated_text

    _TERMINAL = (EndFrame, StopFrame, CancelFrame)
    _SETUP = (StartFrame, EndFrame, StopFrame, CancelFrame, ServiceMetadataFrame)
    _TRACKED = (
        VADUserStartedSpeakingFrame,
        VADUserStoppedSpeakingFrame,
        UserStartedSpeakingFrame,
        UserStoppedSpeakingFrame,
        UserMuteStartedFrame,
        UserMuteStoppedFrame,
        BotStartedSpeakingFrame,
        BotStoppedSpeakingFrame,
        InterruptionFrame,
        TranscriptionFrame,
        LLMTextFrame,
        LLMFullResponseStartFrame,
        FunctionCallInProgressFrame,
        FunctionCallResultFrame,
        FunctionCallCancelFrame,
        MetricsFrame,
        LLMContextFrame,
        SpeechControlParamsFrame,
    )
    _IMPORT_ERROR: Exception | None = None
except Exception as e:  # pragma: no cover - pipecat-ai is a hard dependency
    BaseObserver = object  # type: ignore[assignment,misc]
    _IMPORT_ERROR = e

# Only in some Pipecat versions: STT usage (1.4 reports none) and the user turn start strategies that set how the
# caller interrupts.
try:
    from pipecat.metrics.metrics import STTUsageMetricsData
except ImportError:  # pragma: no cover - depends on the Pipecat version
    STTUsageMetricsData = None  # type: ignore[assignment,misc]
try:
    from pipecat.turns.user_start import MinWordsUserTurnStartStrategy, VADUserTurnStartStrategy
except ImportError:  # pragma: no cover - depends on the Pipecat version
    MinWordsUserTurnStartStrategy = VADUserTurnStartStrategy = None  # type: ignore[assignment,misc]
try:
    from pipecat.turns.user_stop import ExternalUserTurnCompletionStopStrategy
except ImportError:  # pragma: no cover - depends on the Pipecat version
    ExternalUserTurnCompletionStopStrategy = None  # type: ignore[assignment,misc]

logger = logging.getLogger("superbryn_pipecat_observer.test_data")

_DEFAULT_BASE_URL = "https://api.superbryn.com"
_AGENT_DATA_PATH = "/public-api/v1/test-calls/agent-data"
_CALLER_NUMBERS_PATH = "/public-api/v1/test-calls/caller-numbers"
_TEST_ID_KEY = "superbryn_call_id"
_TEST_ID_HEADER_SUFFIXES = ("x-superbryn-call-id", "superbryn_call_id")
_SCHEMA_VERSION = 1
_ATTEMPTS = 3
_BACKOFF_S = (1.0, 2.0)
_STARTED_TIMEOUT_S = 5.0
_ENDED_TIMEOUT_S = 10.0
_NUMBERS_TTL_S = 3600.0
_NUMBERS_RETRY_S = 60.0
_DRAIN_TIMEOUT_S = 2.0
_FINISH_BUDGET_S = 12.0  # all of on_finished: call.started, the drain and call.ended
_DEDUPE_WINDOW = 2000
_MAX_TURNS = 5000
_MAX_TEXT = 20000
_MAX_TOOL_CALLS = 1000
_MAX_TOOLS = 200
_MAX_LABEL = 200
_MAX_DESCRIPTION = 4000
_MAX_LLM_REQUESTS = 2000
_MAX_ERRORS = 500
_MAX_ERROR_MESSAGE = 1000
# how LLMService reports a tool handler that raised
_TOOL_ERROR = re.compile(r"^Error executing function call \[(.+?)\]")
# LLM services whose prompt_tokens leave out cache reads and writes (Anthropic and Bedrock usage).
_EXCLUSIVE_INPUT_LLMS = {
    ("pipecat.services.anthropic.llm", "AnthropicLLMService"),
    ("pipecat.services.aws.llm", "AWSBedrockLLMService"),
}

# Speech-to-speech services that mark a response started only when its first output arrives (Nova Sonic with its
# reply's first text, Gemini Live with its first model turn), so no request's start can be told: none are sent.
_OUTPUT_STAMPED_LLMS = {
    ("pipecat.services.aws.nova_sonic.llm", "AWSNovaSonicLLMService"),
    ("pipecat.services.google.gemini_live.llm", "GeminiLiveLLMService"),
}

# base URL -> (monotonic expiry, caller numbers or None after a failed fetch)
_caller_numbers_cache: dict[str, tuple[float, list[str] | None]] = {}
_warned: set[str] = set()
_now = time.time


def attach_test_data(
    task: Any,
    runner_args: Any = None,
    *,
    api_key: str | None = None,
    agent_id: str | None = None,
    base_url: str | None = None,
    superbryn_call_id: str | None = None,
    from_number: str | None = None,
    to_number: str | None = None,
    call_id: str | None = None,
    call_data: Any = None,
) -> None:
    """Report this call to SuperBryn if it is a SuperBryn test call; returns at once, never raises."""
    try:
        if _IMPORT_ERROR is not None:
            _warn_once(
                "import", "SUPERBRYN_TEST_DATA_DISABLED: pipecat import failed: %s", _IMPORT_ERROR
            )
            return
        key = api_key or os.getenv("SUPERBRYN_TEST_API_KEY") or os.getenv("SUPERBRYN_API_KEY")
        if not key:
            _warn_once(
                "key", "SUPERBRYN_TEST_DATA_DISABLED: no API key, set SUPERBRYN_TEST_API_KEY"
            )
            return
        sent_agent = agent_id or os.getenv("SUPERBRYN_AGENT_ID")
        agent = _agent_uuid(sent_agent)
        if not agent:
            _warn_once(
                "agent",
                "SUPERBRYN_TEST_DATA_DISABLED: %s",
                f"agent_id {sent_agent!r} is not a SuperBryn agent ID (a UUID)"
                if sent_agent
                else "no agent ID, set SUPERBRYN_AGENT_ID",
            )
            return
        call_data = call_data if call_data is not None else _call_data(runner_args)
        dialin = _field(_field(runner_args, "body"), "dialin_settings")
        found = _drop_placeholders(
            {
                "superbryn_call_id": _text(superbryn_call_id)
                or _runner_test_id(runner_args, call_data),
                "from": _text(from_number)
                or _first_text(call_data, "from_number", "from")
                or _first_text(dialin, "From"),
                "to": _text(to_number)
                or _first_text(call_data, "to_number", "to")
                or _first_text(dialin, "To"),
            }
        )
        if not any(found.values()):
            _warn_once(
                "unrecognisable",
                "SUPERBRYN_TEST_DATA: a call with no superbryn_call_id and no phone numbers is "
                "never reported; pass them (or call_data=) to attach_test_data if the bot has them",
                level=logging.INFO,
            )
        observer = _TestDataObserver(
            api_key=key,
            base_url=(base_url or os.getenv("SUPERBRYN_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/"),
            agent_id=agent,
            test_id=found["superbryn_call_id"],
            from_number=found["from"],
            to_number=found["to"],
            call_id=_text(call_id)
            or _first_text(call_data, "call_id")
            or _first_text(dialin, "call_id")
            or _first_text(runner_args, "session_id")
            or str(uuid.uuid4()),
            final_sink=_final_sink(task),
        )
        task.add_observer(observer)
        task.event_handler("on_pipeline_finished")(observer.on_finished)
        _warn_if_metrics_off(task)
    except Exception as e:
        logger.warning("SUPERBRYN_TEST_DATA_ERROR: could not attach: %s", e)


class _TestDataObserver(BaseObserver):
    """Captures one call and sends call.started / call.ended when it is a SuperBryn test call."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        agent_id: str,
        test_id: str | None,
        from_number: str | None,
        to_number: str | None,
        call_id: str,
        final_sink: Any,
    ) -> None:
        super().__init__(name="SuperbrynTestDataObserver")
        self._api_key = api_key
        self._base_url = base_url
        self._agent_id = agent_id
        self._test_id = _cut(test_id, _MAX_LABEL) if test_id else None
        self._from = _cut(from_number, _MAX_LABEL) if from_number else None
        self._to = _cut(to_number, _MAX_LABEL) if to_number else None
        self._call_id = _cut(call_id, _MAX_LABEL)
        self._capture = _Capture(final_sink)
        self._capturing = True
        self._capture_failed = False
        self._start_fields: dict[str, Any] = {}
        self._start_task: asyncio.Task | None = None
        self._first_push = True
        # recognised by its number (or a test ID): call.started is on its way
        self._recognised = False
        self._finished = False

    async def on_push_frame(self, data: FramePushed) -> None:
        if not self._capturing:
            return
        try:
            if self._first_push:
                self._first_push = False
                if self._start_task is None and not isinstance(data.frame, StartFrame):
                    self._attached_late()
            self._capture.on_frame(data)
        except Exception as e:
            if not self._capture_failed:
                self._capture_failed = True
                logger.warning("SUPERBRYN_TEST_DATA_ERROR: frame capture failed: %s", e)

    async def on_pipeline_started(self) -> None:
        try:
            self._begin()
        except Exception as e:
            logger.warning("SUPERBRYN_TEST_DATA_ERROR: could not start: %s", e)

    def _attached_late(self) -> None:
        """The StartFrame went by before attach, so on_pipeline_started never comes: start now."""
        _warn_once(
            "late",
            "SUPERBRYN_TEST_DATA: attach_test_data ran after the task started, so the call is "
            "timed from the first frame seen; attach it before the task runs",
        )
        self._begin()

    async def on_finished(self, _task: Any, frame: Any) -> None:
        """on_pipeline_finished handler: Pipecat awaits it, so it ends within one time budget."""
        if self._finished:
            return
        self._finished = True
        try:
            ended_at = _now()
            self._begin()
            if not self._test_id and not self._recognised and not self._start_task.done():
                self._start_task.cancel()  # the caller-number lookup must not hold the teardown
                logger.debug(
                    "SUPERBRYN_TEST_DATA: call ended before it was recognised, nothing is sent"
                )
                return
            async with asyncio.timeout(_FINISH_BUDGET_S):
                await self._send_ended(frame, ended_at)
        except TimeoutError:
            logger.error(
                "SUPERBRYN_TEST_DATA_FAILED: call.ended not sent within %ss of the call's end",
                _FINISH_BUDGET_S,
            )
        except Exception as e:
            logger.warning("SUPERBRYN_TEST_DATA_ERROR: call.ended not sent: %s", e)

    async def _send_ended(self, frame: Any, ended_at: float) -> None:
        """Wait for call.started and the end frame to drain, then send call.ended."""
        if not await self._start_task:
            return
        try:
            await asyncio.wait_for(self._capture.drained.wait(), _DRAIN_TIMEOUT_S)
        except TimeoutError:
            logger.debug("SUPERBRYN_TEST_DATA: end frame not seen at the sink, sending anyway")
        body = self._ended_body(frame, ended_at)
        url = self._base_url + _AGENT_DATA_PATH
        answer = await _send("POST", url, self._api_key, body, timeout=_ENDED_TIMEOUT_S)
        _log_answer("call.ended", self._call_id, answer)

    def _begin(self) -> None:
        """Fix time zero and start recognition plus call.started in the background, once."""
        if self._start_task is not None:
            return
        if self._capture.t0 is None:
            self._capture.t0 = _now()
        self._start_fields = _compact(
            {
                "superbryn_call_id": self._test_id,
                "agent_id": self._agent_id,
                "from": self._from,
                "to": self._to,
                "started_at": _iso(self._capture.t0),
            }
        )
        self._start_task = asyncio.get_running_loop().create_task(
            self._start(), name="superbryn_test_data_started"
        )

    async def _start(self) -> bool:
        """True when this is a SuperBryn test call and call.ended should follow."""
        try:
            if not self._test_id and not await self._is_superbryn_number():
                logger.debug("SUPERBRYN_TEST_DATA: not a SuperBryn test call, nothing is sent")
                self._capturing = False
                return False
            self._recognised = True
            url = self._base_url + _AGENT_DATA_PATH
            body = self._event("started")
            answer = await _send("POST", url, self._api_key, body, timeout=_STARTED_TIMEOUT_S)
            _log_answer("call.started", self._call_id, answer)
            if answer is not None and answer.get("is_test") is False:
                self._capturing = False
                return False
            return True
        except Exception as e:
            logger.warning("SUPERBRYN_TEST_DATA_ERROR: %s", e)
            return bool(self._test_id)

    async def _is_superbryn_number(self) -> bool:
        numbers = [n for n in (self._from, self._to) if n]
        if not numbers:
            return False
        listed = await _caller_numbers(self._base_url, self._api_key) or []
        return any(_same_number(n, m) for n in numbers for m in listed)

    def _event(self, kind: str, **fields: Any) -> dict[str, Any]:
        """Contract body for call.<kind>; both events carry the same start fields."""
        return {
            "schema_version": _SCHEMA_VERSION,
            "event_id": f"{_cut(f'pc:{self._call_id}', _MAX_LABEL - 9)}:{kind}",
            "event": f"call.{kind}",
            "call_id": self._call_id,
            "source": "pipecat",
            **self._start_fields,
            **fields,
        }

    def _ended_body(self, frame: Any, ended_at: float) -> dict[str, Any]:
        capture = self._capture
        end_t = capture.terminal_t or ended_at
        transcript = _section("transcript", capture.transcript, end_t)
        agent_ids = {t["id"] for t in transcript or () if t["role"] == "agent"}
        fields = {
            "ended_at": _iso(ended_at),
            "ended_reason": _ended_reason(frame, capture.fatal_error),
            "transcript": transcript,
            "tool_calls": _section("tool_calls", capture.tool_calls, transcript or []),
            "diagnostics": _section("diagnostics", capture.diagnostics, agent_ids),
        }
        return self._event("ended", **_compact(fields))


class _Capture:
    """Turns, tool calls, latency cycles, services and usage, from each frame's first push."""

    def __init__(self, final_sink: Any) -> None:
        self.final_sink = final_sink
        self.t0: float | None = None
        self.terminal_t: float | None = None
        self.drained = asyncio.Event()
        self.fatal_error = False
        self._seen: set[int] = set()
        self._seen_order: deque[int] = deque()
        self._user_turns: list[dict[str, Any]] = []
        self._user_sealed = False
        self._vad_seen = False
        self._speech_seen = False
        self._muted = False
        self._agent_turns: list[dict[str, Any]] = []
        self._agent: dict[str, Any] | None = None
        self._pending_tts: list[Any] = []
        self._pending_llm: list[Any] = []
        self._tts_played = False
        self._calls: dict[str, dict[str, Any]] = {}
        self._tools: dict[str, str | None] = {}
        self._cycle: dict[str, Any] | None = None
        self._late_stt: dict[str, Any] | None = None
        self._latencies: list[dict[str, Any]] = []
        self._roles: dict[Any, str | None] = {}
        self._services: dict[str, Any] = {}
        self._metric_models: dict[Any, str] = {}
        self._vad_stop_secs: float | None = None
        self._smart_stop_secs: float | None = None
        self._shortest_wait: float | None = None
        self._turn_analyzer: str | None = None
        self._usage: dict[str, int] = {}
        # one per LLM response: when it started, whether it produced text or a tool call, its tokens
        self._requests: list[dict[str, Any]] = []
        self._errors: list[dict[str, Any]] = []
        self._worked: dict[str, float] = {}  # step -> the last time it produced output
        self._interruptions: dict[str, Any] | None = None
        # True when a stop strategy lets the LLM start before the turn is final (LLM-judged turn completion)
        self._preemptive: bool | None = None
        # each sent line's place on the observer's clock, ms (an untimed caller line's too), set by transcript()
        self._line_at: dict[str, int] = {}
        self._vad_start_secs: float | None = None
        # audio delivered to a streaming STT, for Pipecat versions that report no STT usage
        self._stt_audio_s = 0.0
        self._stt_segmented = False

    def on_frame(self, data: FramePushed) -> None:
        """Record a push: errors either way, TTS text when played, the rest on its first hop."""
        frame, source = data.frame, data.source
        role = self._role(source)
        if role and role not in self._services and not isinstance(frame, _SETUP):
            self._services[role] = source
        if isinstance(frame, ErrorFrame):
            self.fatal_error = self.fatal_error or bool(frame.fatal)
            if self._first_time(frame.id):
                self._error(frame, frame.processor or source, self._time(data))
            return
        # realtime (speech-to-speech) services push the caller's transcript upstream
        if data.direction != FrameDirection.DOWNSTREAM and not isinstance(
            frame, TranscriptionFrame
        ):
            return
        if isinstance(frame, _TERMINAL):
            if self.terminal_t is None:
                self.terminal_t = self._time(data)
            if self.final_sink is None or data.destination is self.final_sink:
                self.drained.set()
            return
        if isinstance(frame, InputAudioRawFrame):
            if isinstance(data.destination, STTService):
                self._stt_audio(data.destination, frame)
            return
        if isinstance(frame, TTSTextFrame):
            if not isinstance(source, BaseOutputTransport):
                return
        elif not isinstance(frame, _TRACKED):
            return
        # a frame broadcast both ways arrives as two frames; count it once
        sibling = getattr(frame, "broadcast_sibling_id", None)
        if sibling is not None and sibling in self._seen:
            return
        if not self._first_time(frame.id):
            return
        t = self._time(data)
        if isinstance(frame, VADUserStartedSpeakingFrame):
            self._user_started(frame)
        elif isinstance(frame, VADUserStoppedSpeakingFrame):
            self._user_stopped(frame)
        elif isinstance(frame, UserStartedSpeakingFrame):
            self._turn_strategies(source)
            self._turn_started(t)
        elif isinstance(frame, UserStoppedSpeakingFrame):
            self._turn_stopped(t)
        elif isinstance(frame, (UserMuteStartedFrame, UserMuteStoppedFrame)):
            self._muted = isinstance(frame, UserMuteStartedFrame)
        elif isinstance(frame, BotStartedSpeakingFrame):
            self._worked["tts"] = t
            self._bot_started(t)
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_stopped(t)
        elif isinstance(frame, InterruptionFrame):
            self._interrupted()
        elif isinstance(frame, TranscriptionFrame):
            self._transcription(frame, t)
        elif isinstance(frame, TTSTextFrame):
            self._worked["tts"] = t
            self._tts_played = True
            self._agent_text(frame, self._agent["tts"] if self._agent else self._pending_tts)
        elif isinstance(frame, LLMTextFrame):
            self._agent_text(frame, self._agent["llm"] if self._agent else self._pending_llm)
            self._worked["llm"] = t
            self._request_output()
        elif isinstance(frame, LLMFullResponseStartFrame):
            if self._agent is None:
                self._pending_llm = []
            if len(self._requests) < _MAX_LLM_REQUESTS:
                self._requests.append({"start": t, "output": False, "tool": False})
        elif isinstance(frame, FunctionCallInProgressFrame):
            self._tool_started(frame, t)
            self._worked["llm"] = t
            self._request_output(tool=True)
        elif isinstance(frame, FunctionCallResultFrame):
            self._tool_result(frame, t)
        elif isinstance(frame, FunctionCallCancelFrame):
            self._tool_cancelled(frame, t)
        elif isinstance(frame, MetricsFrame):
            self._metrics(frame, role, source)
        elif isinstance(frame, LLMContextFrame):
            self._turn_strategies(source)
            self._context_tools(frame.context)
        elif isinstance(frame, SpeechControlParamsFrame):
            self._speech_params(frame)

    def _first_time(self, frame_id: int) -> bool:
        """Bounded deque plus set of frame ids, as Pipecat's UserBotLatencyObserver dedupes."""
        if frame_id in self._seen:
            return False
        if len(self._seen_order) >= _DEDUPE_WINDOW:
            self._seen.discard(self._seen_order.popleft())
        self._seen_order.append(frame_id)
        self._seen.add(frame_id)
        return True

    def _time(self, data: FramePushed) -> float:
        """Wall time of the push: now, minus how long ago the pipeline clock stamped it."""
        now = _now()
        try:
            pushed = data.timestamp
            if pushed and pushed > 0:
                lag = (data.source.get_clock().get_time() - pushed) / 1e9
                if 0 <= lag < 60:
                    return now - lag
        except Exception:
            pass
        return now

    def _role(self, processor: Any) -> str | None:
        try:
            return self._roles[processor]
        except KeyError:
            pass
        except TypeError:
            return None
        if isinstance(processor, STTService):
            role = "stt"
        elif isinstance(processor, LLMService):
            role = "llm"
        elif isinstance(processor, TTSService):
            role = "tts"
        else:
            role = None
        self._roles[processor] = role
        return role

    def _user_started(self, frame: Any) -> None:
        self._vad_seen = True
        self._cycle = None
        self._late_stt = None
        self._open_user_turn(frame.timestamp - _secs(frame.start_secs))

    def _open_user_turn(self, start: float) -> None:
        last = self._user_turns[-1] if self._user_turns else None
        if last is not None and last["vad"] and not self._user_sealed:
            return  # resumed before the agent answered: same turn
        self._new_user_turn(start, vad=True)
        self._user_sealed = False

    def _turn_started(self, t: float) -> None:
        """No VAD frames (server-side turn detection): the service's turn frames time the caller."""
        if not self._vad_seen:
            self._speech_seen = True
            self._open_user_turn(t)

    def _turn_stopped(self, t: float) -> None:
        """UserStoppedSpeakingFrame: the turn is released; ends a turn when there is no VAD."""
        if self._cycle is not None and "wait" not in self._cycle:
            wait = self._cycle["wait"] = max(0.0, t - self._cycle["silence_at"])
            if self._shortest_wait is None or wait < self._shortest_wait:
                self._shortest_wait = wait
        if not self._vad_seen and self._speech_seen and self._user_turns:
            self._user_turns[-1]["end"] = t

    def _user_stopped(self, frame: Any) -> None:
        self._vad_seen = True
        stop_secs = _secs(frame.stop_secs)
        silence_at = frame.timestamp - stop_secs
        if stop_secs > 0:
            self._vad_stop_secs = stop_secs
        if self._user_turns and self._user_turns[-1]["vad"]:
            self._user_turns[-1]["end"] = silence_at
        self._cycle = {"silence_at": silence_at, "stop_secs": stop_secs}
        self._late_stt = None

    def _new_user_turn(self, start: float, *, vad: bool) -> dict[str, Any] | None:
        if len(self._user_turns) >= _MAX_TURNS:
            return None
        turn = {"id": f"u{len(self._user_turns) + 1}", "start": start, "end": None, "vad": vad}
        turn["parts"] = []
        self._user_turns.append(turn)
        return turn

    def _transcription(self, frame: Any, t: float) -> None:
        if not isinstance(frame.text, str) or not frame.text.strip():
            return
        if self._muted:
            return  # a user mute strategy drops it before the agent hears it
        if (self._vad_seen or self._speech_seen) and self._user_turns:
            turn = self._user_turns[-1]
        else:
            turn = self._new_user_turn(t, vad=False)  # untimed: one turn per transcript
        self._worked["stt"] = t
        if turn is not None:
            turn["parts"].append(_part(frame))
            confidence = _confidence(frame.result)
            if confidence is not None:
                turn.setdefault("confidences", []).append(confidence)

    def _bot_started(self, t: float) -> None:
        if self._agent is not None:
            return
        turn = {
            "id": f"a{len(self._agent_turns) + 1}",
            "start": t,
            "end": None,
            "tts": self._pending_tts,
            "llm": self._pending_llm,
            "interrupted": False,
        }
        self._pending_tts, self._pending_llm = [], []
        if len(self._agent_turns) < _MAX_TURNS:
            self._agent_turns.append(turn)
        self._agent = turn
        self._user_sealed = True
        if self._cycle is not None:
            self._close_cycle(t, turn["id"])

    def _bot_stopped(self, t: float) -> None:
        if self._agent is not None:
            self._agent["end"] = t
            self._agent = None

    def _interrupted(self) -> None:
        if self._agent is not None:
            self._agent["interrupted"] = True
        self._pending_tts, self._pending_llm = [], []
        if self._cycle is not None:  # stale LLM/TTS metrics, as UserBotLatencyObserver drops them
            self._cycle.pop("model", None)
            self._cycle.pop("voice", None)

    def _agent_text(self, frame: Any, parts: list[Any]) -> None:
        if isinstance(frame.text, str) and frame.text:
            parts.append(_part(frame))

    def _tool_started(self, frame: Any, t: float) -> None:
        key = frame.tool_call_id or f"frame-{frame.id}"
        if key in self._calls or len(self._calls) >= _MAX_TOOL_CALLS:
            return
        self._calls[key] = {
            "id": frame.tool_call_id,
            "name": frame.function_name,
            "arguments": frame.arguments,
            "start": t,
        }

    def _tool_result(self, frame: Any, t: float) -> None:
        call = self._calls.get(frame.tool_call_id)
        if call is None or "end" in call:
            return
        if getattr(frame.properties, "is_final", True) is False:
            return  # an intermediate update of an async call
        call["end"] = t
        call["result"] = frame.result
        # LLMService's answer when the LLM called a tool that isn't registered
        missing = getattr(LLMService, "MISSING_FUNCTION_CALL_MESSAGE_TEMPLATE", None)
        if isinstance(missing, str) and frame.result == missing.format(function_name=call["name"]):
            call["error"] = True

    def _tool_failed(self, name: str) -> None:
        """The tool handler raised: its latest call still waiting for a result, else its latest call."""
        calls = [call for call in self._calls.values() if call["name"] == name]
        waiting = [call for call in calls if "result" not in call]
        if calls:
            (waiting or calls)[-1]["error"] = True

    def _error(self, frame: Any, processor: Any, t: float) -> None:
        """A pipeline failure, by the step that raised it; a tool handler's failure flags its call instead."""
        message = frame.error if isinstance(frame.error, str) else ""
        tool = _TOOL_ERROR.match(message)
        if tool:
            self._tool_failed(tool.group(1))
            return
        step = self._role(processor)
        if step is None and isinstance(processor, (BaseInputTransport, BaseOutputTransport)):
            step = "telephony"
        if step is not None and len(self._errors) < _MAX_ERRORS:
            fatal = bool(frame.fatal)
            self._errors.append(
                {"step": step, "t": t, "message": _error_message(message), "fatal": fatal}
            )

    def _tool_cancelled(self, frame: Any, t: float) -> None:
        call = self._calls.get(frame.tool_call_id)
        if call is not None:
            call.pop("result", None)
            call.setdefault("end", t)

    def _metrics(self, frame: Any, role: str | None, source: Any) -> None:
        for data in frame.data:
            if isinstance(data, TurnMetricsData):
                self._turn_metrics(data)
            elif isinstance(data, TTFBMetricsData):
                self._ttfb(role, data.value)
            elif isinstance(data, LLMUsageMetricsData):
                self._llm_usage(source, data.value)
            elif isinstance(data, TTSUsageMetricsData):
                self._add_usage("tts_characters", _count(data.value))
            elif STTUsageMetricsData is not None and isinstance(data, STTUsageMetricsData):
                self._stt_usage(data.value)
            if role and isinstance(data.model, str) and data.model.strip():
                self._metric_models[source] = data.model

    def _turn_metrics(self, data: Any) -> None:
        name = data.model or data.processor
        if isinstance(name, str) and name.strip():
            self._turn_analyzer = name
        e2e = data.e2e_processing_time_ms
        if self._cycle is not None and data.is_complete and _is_num(e2e) and e2e >= 0:
            self._cycle.setdefault("endpoint", e2e)

    def _ttfb(self, role: str | None, seconds: Any) -> None:
        if not _is_num(seconds) or seconds <= 0:
            return
        if role == "llm":
            self._request_output()
        ms = seconds * 1000
        if role == "stt":
            # STT TTFB may land up to stt_ttfb_timeout after the reply already started.
            if self._cycle is not None:
                self._cycle.setdefault("transcriber", ms)
            elif self._late_stt is not None:
                self._late_stt.setdefault("transcriberLatency", _round(ms))
        elif role in ("llm", "tts") and self._cycle is not None:
            self._cycle.setdefault("model" if role == "llm" else "voice", ms)

    def _llm_usage(self, source: Any, usage: Any) -> None:
        prompt = _count(usage.prompt_tokens)
        read = _count(usage.cache_read_input_tokens)
        written = _count(usage.cache_creation_input_tokens)
        exclusive = any(
            (cls.__module__, cls.__qualname__) in _EXCLUSIVE_INPUT_LLMS
            for cls in type(source).__mro__
        )
        given = prompt + read + written if exclusive else prompt
        produced = _count(usage.completion_tokens)
        cached = read if usage.cache_read_input_tokens is not None else None
        self._add_usage("llm_input_tokens", given)
        self._add_usage("llm_output_tokens", produced)
        if cached is not None:
            self._add_usage("llm_cached_input_tokens", cached)
        # usage closes a response, so it belongs to the latest one
        if self._requests and "tokens" not in self._requests[-1] and (given or produced):
            tokens = {
                "input_tokens": given,
                "cached_input_tokens": cached,
                "output_tokens": produced,
            }
            self._requests[-1]["tokens"] = _compact(tokens)

    def _request_output(self, *, tool: bool = False) -> None:
        """The latest LLM response produced something: text, a first token, or a tool call."""
        if self._requests:
            self._requests[-1]["output"] = True
            self._requests[-1]["tool"] = self._requests[-1]["tool"] or tool

    def _stt_audio(self, stt: Any, frame: Any) -> None:
        """Audio a streaming STT sends on: every frame delivered to it while it isn't muted. A segmented STT sends
        only speech segments, so its audio isn't counted."""
        if isinstance(stt, SegmentedSTTService):
            self._stt_segmented = True
        elif getattr(stt, "_muted", False) is not True and frame.sample_rate and frame.num_channels:
            self._stt_audio_s += len(frame.audio) / (frame.sample_rate * frame.num_channels * 2)

    def _stt_usage(self, value: Any) -> None:
        """Audio the STT processed: seconds, or an STTUsage with audio_seconds (newer Pipecat)."""
        seconds = value if _is_num(value) else _field(value, "audio_seconds")
        if _is_num(seconds) and seconds >= 0:
            self._add_usage("stt_audio_ms", _round(seconds * 1000))

    def _add_usage(self, key: str, value: int) -> None:
        self._usage[key] = self._usage.get(key, 0) + value

    def _context_tools(self, context: Any) -> None:
        tools = getattr(context, "tools", None)
        if not isinstance(tools, ToolsSchema):
            return
        for schema in tools.standard_tools:
            name = schema.name.strip() if isinstance(schema.name, str) else ""
            if not name or len(self._tools) >= _MAX_TOOLS:
                continue
            name = _cut(name, _MAX_LABEL)
            if name not in self._tools:
                text = schema.description.strip() if isinstance(schema.description, str) else ""
                self._tools[name] = _cut(text, _MAX_DESCRIPTION) or None

    def _speech_params(self, frame: Any) -> None:
        stop = getattr(frame.vad_params, "stop_secs", None)
        if _is_num(stop) and stop > 0:
            self._vad_stop_secs = stop
        start = getattr(frame.vad_params, "start_secs", None)
        if _is_num(start) and start >= 0:
            self._vad_start_secs = start
        turn = frame.turn_params
        if isinstance(turn, SmartTurnParams) and _is_num(turn.stop_secs) and turn.stop_secs > 0:
            self._smart_stop_secs = turn.stop_secs

    def _turn_strategies(self, source: Any) -> None:
        """How the caller interrupts, from the user turn start strategies the user aggregator (or turn processor)
        holds: on when a strategy interrupts the bot, the min-words strategy's words, VAD's start time. And whether
        the LLM starts before the turn is final: a stop strategy that leaves completion to an external judge (the
        LLM's own markers) runs inference on each deferred detector's trigger, and an incomplete turn drops it."""
        controller = getattr(source, "_user_turn_controller", None)
        stop = getattr(getattr(controller, "_user_turn_strategies", None), "stop", None)
        if self._preemptive is None and isinstance(stop, list) and stop:
            judged = ExternalUserTurnCompletionStopStrategy
            self._preemptive = judged is not None and any(isinstance(s, judged) for s in stop)
        if self._interruptions is not None or MinWordsUserTurnStartStrategy is None:
            return
        strategies = getattr(getattr(controller, "_user_turn_strategies", None), "start", None)
        if not isinstance(strategies, list) or not strategies:
            return
        flags = [getattr(s, "_enable_interruptions", None) for s in strategies]
        flags = [flag for flag in flags if isinstance(flag, bool)]
        words = [s._min_words for s in strategies if isinstance(s, MinWordsUserTurnStartStrategy)]
        self._interruptions = {
            "enabled": any(flags) if flags else None,
            "min_words": next((w for w in words if isinstance(w, int) and w >= 0), None),
            "vad": any(isinstance(s, VADUserTurnStartStrategy) for s in strategies),
        }

    def _close_cycle(self, bot_started: float, turn_id: str) -> None:
        cycle, self._cycle = self._cycle, None
        entry: dict[str, Any] = {
            "_turn": turn_id,
            "turnLatency": _round(max(0.0, bot_started - cycle["silence_at"]) * 1000),
        }
        stop_ms = cycle["stop_secs"] * 1000
        # the caller's silence to the turn's release, as UserBotLatencyObserver's user_turn_secs
        if "wait" in cycle:
            entry["endpointingLatency"] = _round(cycle["wait"] * 1000)
        # else 1.4's e2e starts at VAD's stop decision, so the VAD silence is added back
        elif "endpoint" in cycle:
            entry["endpointingLatency"] = _round(stop_ms + cycle["endpoint"])
        elif stop_ms > 0:
            entry["endpointingLatency"] = _round(stop_ms)
        for key, name in (
            ("transcriber", "transcriberLatency"),
            ("model", "modelLatency"),
            ("voice", "voiceLatency"),
        ):
            if key in cycle:
                entry[name] = _round(cycle[key])
        if len(self._latencies) < _MAX_TURNS:
            self._latencies.append(entry)
        self._late_stt = entry

    def _ms(self, t: float) -> int:
        return max(0, _round((t - (self.t0 or t)) * 1000))

    def transcript(self, end_t: float) -> list[dict[str, Any]]:
        """User and agent turns with text, by start time. A caller turn neither VAD nor the service's turn frames
        timed (a speech-to-speech model that reports none, such as Nova Sonic or Gemini Live) has only when its
        transcript arrived, after the caller finished: it carries no times, and that arrival only orders it."""
        timed: list[tuple[int, dict[str, Any]]] = []
        self._line_at.clear()
        for user in self._user_turns:
            text = _cut(concatenate_aggregated_text(user["parts"]), _MAX_TEXT)
            if not text:
                continue
            turn = {"id": user["id"], "role": "user", "text": text}
            at = self._line_at[user["id"]] = self._ms(user["start"])
            if user["vad"]:
                turn["start_ms"] = at
            if user.get("confidences"):
                turn["confidence"] = round(sum(user["confidences"]) / len(user["confidences"]), 4)
            end = user["end"] if user["end"] is not None else (end_t if user["vad"] else None)
            if end is not None and user["vad"]:
                turn["end_ms"] = max(at, self._ms(end))
            timed.append((at, turn))
        for agent in self._agent_turns:
            text = concatenate_aggregated_text(agent["tts"])
            # LLM text only when the call has no played TTS text at all, never for a cut-off turn
            if not text and not self._tts_played and not agent["interrupted"]:
                text = concatenate_aggregated_text(agent["llm"])
            text = _cut(text, _MAX_TEXT)
            if not text:
                continue
            start = self._line_at[agent["id"]] = self._ms(agent["start"])
            end = agent["end"] if agent["end"] is not None else end_t
            timed.append(
                (
                    start,
                    {
                        "id": agent["id"],
                        "role": "agent",
                        "text": text,
                        "start_ms": start,
                        "end_ms": max(start, self._ms(end)),
                        "interrupted": agent["interrupted"],
                    },
                )
            )
        timed.sort(key=lambda entry: entry[0])
        return [turn for _, turn in timed][:_MAX_TURNS]

    def tool_calls(self, transcript: list[dict[str, Any]]) -> list[dict[str, Any]]:
        calls: list[dict[str, Any]] = []
        for call in self._calls.values():
            name = call["name"].strip() if isinstance(call["name"], str) else ""
            if not name:
                continue
            start = self._ms(call["start"])
            item = {"name": _cut(name, _MAX_LABEL), "start_ms": start}
            if isinstance(call["id"], str) and call["id"]:
                item["id"] = _cut(call["id"], _MAX_LABEL)
            item["arguments"] = _json_value(call["arguments"])
            item["result"] = _json_value(call.get("result"))
            if "end" in call:
                item["end_ms"] = max(start, self._ms(call["end"]))
            # raised, or not a registered tool; a result otherwise means it ran
            if call.get("error"):
                item["is_error"] = True
            elif call.get("result") is not None:
                item["is_error"] = False
            item["turn_id"] = _tool_turn_id(start, transcript, self._line_at)
            calls.append(_compact(item))
        return calls[:_MAX_TOOL_CALLS]

    def llm_requests(self) -> list[dict[str, Any]]:
        """Each LLM response and what became of it. A bot turn that spoke LLM text belongs to the last response
        with output started before it: spoken, or cut off when the caller cut in. None from a service whose
        response starts are stamped at its first output (_OUTPUT_STAMPED_LLMS)."""
        llm = self._services.get("llm")
        if llm is not None and any(
            (cls.__module__, cls.__qualname__) in _OUTPUT_STAMPED_LLMS for cls in type(llm).__mro__
        ):
            return []
        heard: dict[int, bool] = {}  # response index -> cut off
        for turn in self._agent_turns:
            if not turn["llm"]:
                continue  # scripted speech (a TTSSpeakFrame), not an LLM answer
            owner = next(
                (
                    i
                    for i in range(len(self._requests) - 1, -1, -1)
                    if self._requests[i]["output"] and self._requests[i]["start"] <= turn["start"]
                ),
                None,
            )
            if owner is not None:
                heard[owner] = heard.get(owner, False) or turn["interrupted"]
        requests = []
        for i, request in enumerate(self._requests):
            if i in heard:
                outcome = "cut_off" if heard[i] else "spoken"
            elif request["tool"]:
                outcome = "tool_call"
            else:
                outcome = "not_spoken" if request["output"] else "stopped"
            entry = {"start_ms": self._ms(request["start"]), "outcome": outcome}
            requests.append({**entry, **request.get("tokens", {})})
        return requests

    def diagnostics(self, agent_ids: set[str]) -> dict[str, Any] | None:
        """Each section alone; one that raises is left out with a warning."""
        configuration = {
            "stt": _section("configuration.stt", self._service_config, "stt"),
            "llm": _section("configuration.llm", self._service_config, "llm"),
            "tts": _section("configuration.tts", self._service_config, "tts"),
            "turn_detection": _section("configuration.turn_detection", self._turn_detection),
            "interruptions": _section("configuration.interruptions", self._interruption_settings),
            "tools": _section("configuration.tools", self._tool_list),
        }
        diagnostics = {
            "pipeline": _section("pipeline", self._pipeline),
            "configuration": {k: v for k, v in configuration.items() if v} or None,
            "turn_latencies": _section("turn_latencies", self._turn_latencies, agent_ids),
            "usage": _section("usage", self._usage_totals),
            "llm_requests": _section("llm_requests", self.llm_requests),
            "errors": _section("errors", self.errors),
        }
        return {k: v for k, v in diagnostics.items() if v} or None

    def _service_config(self, role: str) -> dict[str, str] | None:
        service = self._services.get(role)
        if service is None:
            return None
        settings = getattr(service, "_settings", None)
        fields = {
            "provider": _label(_provider(service)),
            "model": _label(_setting(settings, "model"))
            or _label(self._metric_models.get(service)),
        }
        if role == "tts":
            fields["voice"] = _label(_setting(settings, "voice")) or _label(
                _setting(settings, "voice_id")
            )
        return _compact(fields) or None

    def _turn_detection(self) -> dict[str, Any] | None:
        vad, smart = self._vad_stop_secs, self._smart_stop_secs
        # a smart-turn analyzer starts its silence timeout at VAD's stop decision
        most = vad + smart if vad and smart else None
        # the shortest measured wait (silence to release) covers the stop strategy's own wait
        least = self._shortest_wait if self._shortest_wait is not None else vad
        if least is not None and most is not None:
            least = min(least, most)
        realtime = self._pipeline() == "realtime"
        fields = {
            # with no VAD or analyzer, a speech-to-speech service takes the turns itself
            "type": _label(self._turn_analyzer)
            or ("vad" if self._vad_seen else "realtime_llm" if realtime else None),
            "min_end_of_turn_ms": _round(least * 1000) if least is not None else None,
            "max_end_of_turn_ms": _round(most * 1000) if most else None,
            # a speech-to-speech model answers when it decides the caller finished: no early drafts to speak of
            "preemptive_generation": None if realtime else self._preemptive,
        }
        return _compact(fields) or None

    def _usage_totals(self) -> dict[str, int]:
        """The usage totals; STT time measured from its audio when the STT reported none (Pipecat 1.4)."""
        usage = dict(self._usage)
        if "stt_audio_ms" not in usage and self._stt_audio_s > 0 and not self._stt_segmented:
            usage["stt_audio_ms"] = _round(self._stt_audio_s * 1000)
        return usage

    def _interruption_settings(self) -> dict[str, Any] | None:
        found = self._interruptions
        if found is None:
            return None
        start = self._vad_start_secs if found["vad"] else None
        fields = {
            "enabled": found["enabled"],
            "min_duration_ms": _round(start * 1000) if start is not None else None,
            "min_words": found["min_words"],
        }
        return _compact(fields) or None

    def errors(self) -> list[dict[str, Any]]:
        """Each failure: recovered when its step worked again later; a fatal one never did, and a transport's
        recovery is unknown."""
        found = []
        for error in self._errors:
            step, fatal = error["step"], error["fatal"]
            if fatal or step == "telephony":
                recovered = False if fatal else None
            else:
                recovered = self._worked.get(step, -math.inf) > error["t"]
            item = {
                "step": step,
                "side": "caller" if step == "stt" else None,
                "start_ms": self._ms(error["t"]),
                "message": error["message"],
                "recovered": recovered,
            }
            found.append(_compact(item))
        return found

    def _tool_list(self) -> list[dict[str, str]]:
        return [_compact({"name": n, "description": d}) for n, d in self._tools.items()]

    def _pipeline(self) -> str | None:
        roles = set(self._services)
        if {"stt", "llm", "tts"} <= roles:
            return "stt_llm_tts"
        return "realtime" if roles == {"llm"} else None

    def _turn_latencies(self, agent_ids: set[str]) -> list[dict[str, Any]]:
        entries = []
        for entry in self._latencies[:_MAX_TURNS]:
            item = {k: v for k, v in entry.items() if not k.startswith("_")}
            if entry["_turn"] in agent_ids:
                item["turn_id"] = entry["_turn"]
            entries.append(item)
        return entries


async def _http_request(
    method: str, url: str, headers: dict[str, str], data: bytes | None
) -> tuple[int, Any]:
    """One request with aiohttp; returns (status, parsed JSON or None). Tests replace it."""
    async with aiohttp.ClientSession() as session:
        async with session.request(method, url, data=data, headers=headers) as response:
            try:
                answer = await response.json(content_type=None)
            except ValueError:
                answer = None
            return response.status, answer


async def _send(
    method: str, url: str, api_key: str, body: dict[str, Any] | None, *, timeout: float
) -> dict[str, Any] | None:
    """Up to 3 attempts with the same body; the JSON object of a 2xx answer, else None."""
    headers = {"X-API-Key": api_key}
    data = None
    if body is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(body, allow_nan=False).encode()
    for attempt in range(_ATTEMPTS):
        if attempt:
            await asyncio.sleep(_BACKOFF_S[attempt - 1])
        try:
            status, answer = await asyncio.wait_for(
                _http_request(method, url, headers, data), timeout
            )
        except Exception as e:
            logger.warning(
                "SUPERBRYN_TEST_DATA_RETRY: %s %s attempt %d: %r", method, url, attempt + 1, e
            )
            continue
        if 200 <= status < 300:
            return answer if isinstance(answer, dict) else {}
        if status in (408, 429) or status >= 500:
            logger.warning(
                "SUPERBRYN_TEST_DATA_RETRY: %s %s attempt %d -> %s",
                method,
                url,
                attempt + 1,
                status,
            )
            continue
        hint = " (needs an org API key with test_data:write)" if status in (401, 403) else ""
        logger.error(
            "SUPERBRYN_TEST_DATA_REJECTED: %s %s -> %s%s %s", method, url, status, hint, answer
        )
        return None
    logger.error(
        "SUPERBRYN_TEST_DATA_FAILED: %s %s gave up after %d attempts", method, url, _ATTEMPTS
    )
    return None


async def _caller_numbers(base_url: str, api_key: str) -> list[str] | None:
    """SuperBryn's test numbers: cached an hour per base URL, a failure remembered for a minute."""
    hit = _caller_numbers_cache.get(base_url)
    if hit is not None and time.monotonic() < hit[0]:
        return hit[1]
    url = base_url + _CALLER_NUMBERS_PATH
    answer = await _send("GET", url, api_key, None, timeout=_STARTED_TIMEOUT_S)
    numbers = answer.get("numbers") if answer else None
    if not isinstance(numbers, list):
        _caller_numbers_cache[base_url] = (time.monotonic() + _NUMBERS_RETRY_S, None)
        return None
    numbers = [n for n in numbers if isinstance(n, str)]
    _caller_numbers_cache[base_url] = (time.monotonic() + _NUMBERS_TTL_S, numbers)
    return numbers


def _same_number(a: str, b: str) -> bool:
    """Equal digits, or equal last 10 digits when both have at least 10 (as the server matches)."""
    da, db = "".join(filter(str.isdigit, a)), "".join(filter(str.isdigit, b))
    if not da or not db:
        return False
    return da == db or (len(da) >= 10 and len(db) >= 10 and da[-10:] == db[-10:])


def _log_answer(event: str, call_id: str, answer: dict[str, Any] | None) -> None:
    if answer is not None:
        logger.info(
            "SUPERBRYN_TEST_DATA_SENT: %s for call %s: status=%s is_test=%s",
            event,
            call_id,
            answer.get("status"),
            answer.get("is_test"),
        )


def _runner_test_id(runner_args: Any, call_data: Any) -> str | None:
    """Test ID from the start body, Daily dial-in SIP headers, or telephony custom parameters."""
    body = _field(runner_args, "body")
    test_id = _first_text(body, _TEST_ID_KEY)
    if test_id:
        return test_id
    headers = _field(_field(body, "dialin_settings"), "sip_headers")
    if isinstance(headers, Mapping):
        for key, value in headers.items():
            if (
                isinstance(key, str)
                and key.lower().endswith(_TEST_ID_HEADER_SUFFIXES)
                and _text(value)
            ):
                return _text(value)
    for params in (_field(call_data, "body"), _params(_field(call_data, "custom_parameters"))):
        test_id = _first_text(params, _TEST_ID_KEY)
        if test_id:
            return test_id
    return None


def _call_data(runner_args: Any) -> Any:
    """runner_args.call_data, else the handshake parse Pipecat cached on the websocket."""
    call_data = _field(runner_args, "call_data")
    if call_data is None:
        parsed = getattr(_field(runner_args, "websocket"), "_pipecat_parsed_telephony", None)
        if isinstance(parsed, tuple) and len(parsed) == 2:
            call_data = parsed[1]
    return call_data


def _drop_placeholders(found: dict[str, str | None]) -> dict[str, str | None]:
    """A value still holding a template placeholder such as {From} is ignored, with one warning."""
    unfilled = {k: v for k, v in found.items() if v and "{" in v}
    if unfilled:
        _warn_once(
            "placeholder",
            "SUPERBRYN_TEST_DATA: ignored unfilled template placeholders %s",
            unfilled,
        )
    return {k: None if k in unfilled else v for k, v in found.items()}


def _params(value: Any) -> Any:
    """Custom parameters as a mapping: a dict, a JSON object string or a query string."""
    if not isinstance(value, str):
        return value
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else None
    except ValueError:
        return dict(parse_qsl(value))


def _field(obj: Any, key: str) -> Any:
    """obj[key] for a mapping, else the attribute (pydantic models included); None when missing."""
    if obj is None:
        return None
    try:
        return obj.get(key) if isinstance(obj, Mapping) else getattr(obj, key, None)
    except Exception:
        return None


def _first_text(obj: Any, *keys: str) -> str | None:
    return next(filter(None, (_text(_field(obj, key)) for key in keys)), None)


def _text(value: Any) -> str | None:
    return (value.strip() or None) if isinstance(value, str) else None


def _final_sink(task: Any) -> Any:
    """The worker's pipeline sink; the end frame's push into it is the last one observed."""
    try:
        return task.pipeline.processors[-1]
    except Exception:
        return None


def _warn_if_metrics_off(task: Any) -> None:
    params = getattr(task, "params", None)
    off = [
        n for n in ("enable_metrics", "enable_usage_metrics") if getattr(params, n, True) is False
    ]
    if off:
        _warn_once(
            "metrics",
            "SUPERBRYN_TEST_DATA: %s off in PipelineParams, so test calls carry no latency or usage",
            " and ".join(off),
        )


def _warn_once(key: str, message: str, *args: Any, level: int = logging.WARNING) -> None:
    if key not in _warned:
        _warned.add(key)
        logger.log(level, message, *args)


def _agent_uuid(value: str | None) -> str | None:
    """The agent ID when the server's z.uuid() accepts it (RFC 4122 variant, version 1-8)."""
    if not value:
        return None
    try:
        parsed = uuid.UUID(value.strip())
        if parsed.variant == uuid.RFC_4122 and 1 <= (parsed.version or 0) <= 8:
            return str(parsed)
    except ValueError:
        pass
    return None


def _ended_reason(frame: Any, fatal_error: bool) -> str | None:
    if fatal_error:
        return "error"
    for cls, reason in ((EndFrame, "ended"), (CancelFrame, "cancelled"), (StopFrame, "stopped")):
        if isinstance(frame, cls):
            return reason
    return None


def _tool_turn_id(
    start_ms: int, transcript: list[dict[str, Any]], at: dict[str, int]
) -> str | None:
    """The agent turn that started last before the call with no user turn between, else the next one. at holds each
    line's place on the observer's clock, an untimed caller line's included."""
    before = None
    for turn in transcript:
        if at[turn["id"]] > start_ms:
            break
        before = turn
    if before is not None and before["role"] == "agent":
        return before["id"]
    return next(
        (t["id"] for t in transcript if t["role"] == "agent" and at[t["id"]] > start_ms), None
    )


def _provider(service: Any) -> str | None:
    """The package under pipecat.services of the service's class or its nearest Pipecat base."""
    for cls in type(service).__mro__:
        parts = cls.__module__.split(".")
        if len(parts) >= 4 and parts[0] == "pipecat" and parts[1] == "services":
            return parts[2]
    return None


def _setting(settings: Any, name: str) -> Any:
    return settings.get(name) if isinstance(settings, Mapping) else getattr(settings, name, None)


def _section(name: str, build: Any, *args: Any) -> Any:
    try:
        return build(*args)
    except Exception as e:
        logger.warning("SUPERBRYN_TEST_DATA_ERROR: %s left out of call.ended: %s", name, e)
        return None


def _part(frame: Any) -> Any:
    return TextPartForConcatenation(frame.text, bool(frame.includes_inter_frame_spaces))


def _json_value(value: Any) -> Any:
    """JSON-safe copy of a dict, list, str or number; str() of anything else; None stays unset."""
    if value is None:
        return None
    if isinstance(value, (dict, list, str, int, float)):
        try:
            return json.loads(json.dumps(value, default=str, allow_nan=False))
        except (TypeError, ValueError, RecursionError):
            pass
    try:
        return str(value)
    except Exception:
        return None


def _label(value: Any) -> str | None:
    """A provider, model, voice or detector name: non-empty, not "unknown", at most 200 units."""
    text = value.strip() if isinstance(value, str) else ""
    return _cut(text, _MAX_LABEL) if text and text.lower() != "unknown" else None


def _cut(text: str, limit: int) -> str:
    """At most limit UTF-16 code units, as zod's .max() counts; a split surrogate pair is dropped."""
    if len(text) * 2 <= limit:
        return text
    return text.encode("utf-16-le", "surrogatepass")[: limit * 2].decode("utf-16-le", "ignore")


def _error_message(text: str) -> str | None:
    """The failure's text with links and long ids (keys, signatures) masked, cut to the server's limit."""
    text = re.sub(r"[a-z][a-z0-9+.-]*://\S+", "<url>", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", re.sub(r"[A-Za-z0-9+/_-]{32,}", "***", text)).strip()
    return _cut(text, _MAX_ERROR_MESSAGE) or None


def _confidence(result: Any) -> float | None:
    """The STT's confidence in a final transcript, from the provider's raw result: Deepgram (channel.alternatives),
    Google (alternatives), Gladia (data.utterance), AWS Transcribe (word items) and Soniox (tokens) carry one."""
    for value in (
        _field(_first(_field(_field(result, "channel"), "alternatives")), "confidence"),
        _field(_first(_field(result, "alternatives")), "confidence"),
        _field(_field(_field(result, "data"), "utterance"), "confidence"),
    ):
        if _unit(value):
            return value
    words = _field(_first(_field(result, "Alternatives")), "Items")
    if words is None and isinstance(result, list):
        words = result
    scores = [
        s
        for w in words or ()
        for s in (_field(w, "Confidence"), _field(w, "confidence"))
        if _unit(s)
    ]
    return sum(scores) / len(scores) if scores else None


def _unit(value: Any) -> bool:
    """A confidence: a number above 0 and at most 1 (0 is how some providers say there is none)."""
    return _is_num(value) and 0 < value <= 1


def _first(items: Any) -> Any:
    """The first element of a list-like value, else None."""
    if items is None or isinstance(items, (str, bytes, Mapping)):
        return None
    try:
        return items[0]
    except (IndexError, KeyError, TypeError):
        return None


def _is_num(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


def _secs(value: Any) -> float:
    return float(value) if _is_num(value) and value > 0 else 0.0


def _count(value: Any) -> int:
    return int(value) if _is_num(value) and value >= 0 else 0


def _round(value: float) -> int:
    return int(round(value))


def _compact(fields: dict[str, Any]) -> dict[str, Any]:
    """Unset fields are left out: the contract wants them absent, not null."""
    return {k: v for k, v in fields.items() if v is not None}


def _iso(t: float) -> str:
    return datetime.fromtimestamp(t, UTC).isoformat(timespec="milliseconds").replace("+00:00", "Z")
