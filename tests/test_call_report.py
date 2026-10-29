"""attach() (call_report.py) driven with real Pipecat frames, metrics and service base classes."""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import math
import os
import tempfile
import time
import warnings
import wave
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from loguru import logger as loguru_logger
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    EndWorkerFrame,
    ErrorFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InputAudioRawFrame,
    InputDTMFFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    OutputAudioRawFrame,
    SpeechControlParamsFrame,
    StartFrame,
    StopFrame,
    SystemFrame,
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
    LLMTokenUsage,
    LLMUsageMetricsData,
    SmartTurnMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)
from pipecat.observers.base_observer import BaseObserver, FramePushed
from pipecat.pipeline.worker import PipelineParams
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings, STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService, STTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_input import BaseInputTransport
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy, VADUserTurnStartStrategy
from pydantic import BaseModel, ConfigDict, Field

import superbryn_pipecat_observer.call_report as cr
from superbryn_pipecat_observer import attach

T0 = 1_800_000_000.0  # 2027-01-15T08:00:00Z
AGENT_ID = "4347a69f-0c1e-4b6a-9f1e-2b3c4d5e6f70"
UP = FrameDirection.UPSTREAM
LOGGER = "superbryn_pipecat_observer.call_report"


# ── fakes: real Pipecat base classes, so the isinstance checks are real ──


class FakeSTT(STTService):
    async def run_stt(self, audio):
        yield None


class FakeSegmentedSTT(SegmentedSTTService):
    async def run_stt(self, audio):
        yield None


class FakeLLM(LLMService):
    pass


class FakeAnthropicLLM(LLMService):
    pass


class FakeTTS(TTSService):
    async def run_tts(self, text, context_id):
        yield None


class FakeOutput(BaseOutputTransport):
    pass


FakeSTT.__module__ = "pipecat.services.deepgram.stt"
FakeLLM.__module__ = "pipecat.services.openai.llm"
FakeTTS.__module__ = "pipecat.services.cartesia.tts"
FakeAnthropicLLM.__module__ = "pipecat.services.anthropic.llm"
FakeAnthropicLLM.__qualname__ = "AnthropicLLMService"


def stt_service(model: str | None = "nova-3") -> FakeSTT:
    return FakeSTT(settings=STTSettings(model=model, language=None))


def llm_service(model: str | None = "gpt-4o-mini", cls: type = FakeLLM) -> LLMService:
    return cls(settings=LLMSettings(model=model))


def tts_service(model: str | None = "sonic-2", voice: str | None = "voice-1") -> FakeTTS:
    return FakeTTS(settings=TTSSettings(model=model, voice=voice, language=None))


class CallDataModel(BaseModel):
    """Same shape as pipecat.runner.types.CallData (not importable here without fastapi)."""

    model_config = ConfigDict(populate_by_name=True, extra="allow")
    stream_id: str | None = None
    call_id: str | None = None
    from_number: str | None = Field(default=None, alias="from")
    to_number: str | None = Field(default=None, alias="to")
    body: dict = Field(default_factory=dict)


class DialinSettingsModel(BaseModel):
    call_id: str
    call_domain: str
    To: str | None = None
    From: str | None = None
    sip_headers: dict[str, str] | None = None


class DialinRequestModel(BaseModel):
    dialin_settings: DialinSettingsModel
    daily_api_key: str
    daily_api_url: str


class BodyModel(BaseModel):
    superbryn_call_id: str | None = None


class FakeHTTP:
    """Stands in for _http_request: records each request, answers from a script."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.replies: list[Any] = []
        self.numbers: list[str] = ["+14155550100"]

    async def __call__(self, method, url, headers, data):
        body = json.loads(data) if data else None
        self.requests.append(
            {"method": method, "url": url, "headers": headers, "body": body, "raw": data}
        )
        if self.replies:
            reply = self.replies.pop(0)
        elif method == "GET":
            reply = (200, {"numbers": self.numbers})
        else:
            reply = (200, {"status": "linked", "is_test": True})
        if isinstance(reply, BaseException):
            raise reply
        return reply

    def events(self, kind: str | None = None) -> list[dict[str, Any]]:
        return [
            r["body"]
            for r in self.requests
            if r["method"] == "POST" and (kind is None or r["body"]["event"] == f"call.{kind}")
        ]

    def gets(self) -> list[dict[str, Any]]:
        return [r for r in self.requests if r["method"] == "GET"]


class FakeTask:
    """The three things attach uses on a PipelineTask."""

    def __init__(self, *, metrics: bool = True) -> None:
        self.params = PipelineParams(enable_metrics=metrics, enable_usage_metrics=metrics)
        self.observers: list[Any] = []
        self.handlers: dict[str, list[Any]] = {}

    def add_observer(self, observer) -> None:
        self.observers.append(observer)

    def event_handler(self, name):
        def register(handler):
            self.handlers.setdefault(name, []).append(handler)
            return handler

        return register


class Clock:
    t = T0


class Call:
    """Drives one attached observer the way PipelineTask does; `at` is seconds after the start."""

    def __init__(self, clock: Clock, runner_args: Any = None, **kwargs: Any) -> None:
        self.clock = clock
        self.task = FakeTask()
        kwargs.setdefault("api_key", "sk_test")
        kwargs.setdefault("agent_id", AGENT_ID)
        # SuperBryn test calls unless a test says otherwise: other calls send nothing
        kwargs.setdefault("production", False)
        self.report = attach(self.task, runner_args, **kwargs)
        self.observer = self.task.observers[0] if self.task.observers else None

    async def start(self) -> Call:
        self.clock.t = T0
        await self.observer.on_pipeline_started()
        await self.observer._start_task
        return self

    async def push(
        self, frame, *, at=None, source=None, direction=FrameDirection.DOWNSTREAM, destination=None
    ):
        if at is not None:
            self.clock.t = T0 + at
        data = FramePushed(
            source=source, destination=destination, frame=frame, direction=direction, timestamp=0
        )
        await self.observer.on_push_frame(data)

    async def end(self, frame=None, *, at=None) -> None:
        frame = frame or EndFrame()
        await self.push(frame, at=at)
        for handler in self.task.handlers["on_pipeline_finished"]:
            await handler(self.task, frame)


@pytest.fixture(autouse=True)
def _isolated(monkeypatch, tmp_path):
    for name in (
        "SUPERBRYN_TEST_API_KEY",
        "SUPERBRYN_API_KEY",
        "SUPERBRYN_AGENT_ID",
        "SUPERBRYN_BASE_URL",
        "SUPERBRYN_OBSERVER_ENABLED",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cr, "_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))  # this test's caller-number file only
    cr._caller_numbers_cache.clear()
    cr._warned.clear()
    cr._attached_tasks.clear()


@pytest.fixture(autouse=True)
def pushes(monkeypatch) -> list[dict[str, Any]]:
    """push_prompt stand-in: the prompt pushes made, never sent."""
    made: list[dict[str, Any]] = []

    async def push(**kwargs: Any) -> bool:
        made.append(kwargs)
        return True

    monkeypatch.setattr(cr, "push_prompt", push)
    return made


@pytest.fixture(autouse=True)
def http(monkeypatch) -> FakeHTTP:
    fake = FakeHTTP()
    monkeypatch.setattr(cr, "_http_request", fake)
    return fake


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(cr, "_now", lambda: c.t)
    return c


def vad_start(at: float, start_secs: float = 0.2) -> VADUserStartedSpeakingFrame:
    return VADUserStartedSpeakingFrame(start_secs=start_secs, timestamp=T0 + at)


def vad_stop(at: float, stop_secs: float = 0.8) -> VADUserStoppedSpeakingFrame:
    return VADUserStoppedSpeakingFrame(stop_secs=stop_secs, timestamp=T0 + at)


def transcription(text: str) -> TranscriptionFrame:
    return TranscriptionFrame(text, "user-1", "2026-09-28T00:00:00.000Z")


def tts_text(text: str) -> TTSTextFrame:
    return TTSTextFrame(text, aggregated_by="sentence")


def metrics(*data) -> MetricsFrame:
    return MetricsFrame(data=list(data))


def ttfb(seconds: float, model: str | None = None) -> MetricsFrame:
    return metrics(TTFBMetricsData(processor="svc", value=seconds, model=model))


def turn_metrics(e2e_ms: float, *, complete: bool = True) -> MetricsFrame:
    return metrics(
        TurnMetricsData(
            processor="BaseSmartTurn",
            is_complete=complete,
            probability=0.9,
            e2e_processing_time_ms=e2e_ms,
        )
    )


def tool_start(call_id: str, name: str = "lookup", arguments: Any = None):
    return FunctionCallInProgressFrame(
        function_name=name, tool_call_id=call_id, arguments=arguments or {"q": call_id}
    )


def tool_result(call_id: str, result: Any, name: str = "lookup", *, final: bool = True):
    return FunctionCallResultFrame(
        function_name=name,
        tool_call_id=call_id,
        arguments={},
        result=result,
        properties=FunctionCallResultProperties(is_final=final),
    )


def ended(http: FakeHTTP) -> dict[str, Any]:
    [event] = http.events("ended")
    return event


def texts(body: dict[str, Any]) -> list[tuple[str, str, str]]:
    return [(t["id"], t["role"], t["text"]) for t in body["transcript"]]


def messages(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


async def user_turn(call: Call, text: str, *, start: float, stop: float, source=None) -> None:
    """VAD from `start` to `stop` (after start_secs / stop_secs), then the transcript."""
    await call.push(vad_start(start + 0.2), at=start + 0.2)
    await call.push(vad_stop(stop + 0.8), at=stop + 0.8)
    await call.push(transcription(text), at=stop + 0.9, source=source)


async def agent_turn(call: Call, text: str, *, start: float, stop: float, out=None) -> None:
    out = out or FakeOutput(TransportParams())
    await call.push(BotStartedSpeakingFrame(), at=start, source=out)
    await call.push(tts_text(text), at=start + 0.1, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=stop, source=out)


# ── recognition ──────────────────────────────────────────────────────────


async def test_explicit_test_id_sends_started_and_ended(clock, http):
    call = await Call(clock, superbryn_call_id="sb-1", call_id="c-1").start()
    assert [e["event"] for e in http.events()] == ["call.started"]
    await call.end(at=5)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]
    assert http.gets() == []
    assert http.events("started")[0]["testing"]["superbryn_call_id"] == "sb-1"


@pytest.mark.parametrize(
    "body",
    [{"superbryn_call_id": " sb-body "}, BodyModel(superbryn_call_id="sb-body")],
    ids=["dict", "pydantic"],
)
async def test_test_id_from_start_body(clock, http, body):
    await Call(clock, SimpleNamespace(body=body)).start()
    assert http.events("started")[0]["testing"]["superbryn_call_id"] == "sb-body"


@pytest.mark.parametrize(
    "header", ["X-SuperBryn-Call-Id", "sip.h.x-superbryn-call-id", "SUPERBRYN_CALL_ID"]
)
@pytest.mark.parametrize("as_model", [False, True], ids=["dict", "pydantic"])
async def test_test_id_from_daily_dialin_sip_headers(clock, http, header, as_model):
    settings = {
        "call_id": "daily-call",
        "call_domain": "domain",
        "From": "+14155550999",
        "To": "+14155550123",
        "sip_headers": {"X-Other": "x", header: "sb-sip"},
    }
    body = {"dialin_settings": settings, "daily_api_key": "secret", "daily_api_url": "u"}
    if as_model:
        body = DialinRequestModel.model_validate(body)
    await Call(clock, SimpleNamespace(body=body)).start()
    started = http.events("started")[0]
    assert started["testing"]["superbryn_call_id"] == "sb-sip"
    assert (started["from"], started["to"], started["call_id"]) == (
        "+14155550999",
        "+14155550123",
        "daily-call",
    )
    assert "secret" not in http.requests[0]["raw"].decode()


@pytest.mark.parametrize(
    "call_data",
    [
        CallDataModel(call_id="CA1", body={"superbryn_call_id": "sb-tw"}),
        CallDataModel(call_id="CA1", custom_parameters={"superbryn_call_id": "sb-tw"}),
        CallDataModel(call_id="CA1", custom_parameters='{"superbryn_call_id": "sb-tw"}'),
        CallDataModel(call_id="CA1", custom_parameters="a=1&superbryn_call_id=sb-tw"),
        {"call_id": "CA1", "body": {"superbryn_call_id": "sb-tw"}},
    ],
    ids=["twilio-body", "exotel-dict", "exotel-json", "exotel-query", "plain-dict"],
)
async def test_test_id_from_telephony_call_data(clock, http, call_data):
    await Call(clock, SimpleNamespace(call_data=call_data)).start()
    started = http.events("started")[0]
    assert (started["testing"]["superbryn_call_id"], started["call_id"]) == ("sb-tw", "CA1")


async def test_handshake_the_bot_parsed_itself(clock, http):
    handshake = CallDataModel(
        call_id="CA1",
        body={"superbryn_call_id": "sbc_123"},
        **{"from": "+14155550100", "to": "+16505550199"},
    )

    class WebSocket(dict):  # Starlette's WebSocket is a Mapping over the ASGI scope
        pass

    # parse_telephony_websocket caches its result on the websocket; only create_transport sets
    # runner_args.call_data
    websocket = WebSocket(type="websocket")
    websocket._pipecat_parsed_telephony = ("twilio", handshake)
    await Call(clock, SimpleNamespace(websocket=websocket, call_data=None)).start()
    started = http.events("started")[0]
    assert (started["testing"]["superbryn_call_id"], started["call_id"]) == ("sbc_123", "CA1")
    assert (started["from"], started["to"]) == ("+14155550100", "+16505550199")


async def test_call_data_argument_comes_first(clock, http):
    handshake = {"call_id": "CA2", "body": {"superbryn_call_id": "sb-arg"}}
    args = SimpleNamespace(call_data=CallDataModel(call_id="CA1"))
    await Call(clock, args, call_data=handshake).start()
    started = http.events("started")[0]
    assert (started["testing"]["superbryn_call_id"], started["call_id"]) == ("sb-arg", "CA2")


async def test_nothing_to_recognise_is_logged_once_at_info(clock, http, caplog):
    with caplog.at_level(logging.INFO, logger=LOGGER):
        for _ in range(2):
            await Call(clock, SimpleNamespace(call_data=CallDataModel(call_id="CA1"))).start()
    [record] = [r for r in caplog.records if "no phone numbers" in r.getMessage()]
    assert record.levelno == logging.INFO
    assert http.requests == []


async def test_unfilled_template_placeholders_are_ignored(clock, http, caplog):
    handshake = CallDataModel(
        call_id="CA1",
        body={"superbryn_call_id": "{SipHeader_X-SuperBryn-Call-Id}"},
        **{"from": "{From}", "to": "+14155550100"},  # the dialled number was filled in
    )
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(2):
            await Call(clock, SimpleNamespace(call_data=handshake)).start()
    started = http.events("started")[0]
    assert not {"superbryn_call_id", "from"} & started.keys() and started["to"] == "+14155550100"
    assert messages(caplog, "placeholders") == [
        "SUPERBRYN_CALL_REPORT: ignored unfilled template placeholders "
        "{'superbryn_call_id': '{SipHeader_X-SuperBryn-Call-Id}', 'from': '{From}'}"
    ]


async def test_caller_number_fallback_matches_and_caches(clock, http):
    http.numbers = ["+1 (415) 555-0100", "+919876543210"]
    args = SimpleNamespace(call_data=CallDataModel(**{"from": "14155550100", "to": "+15550001"}))
    await Call(clock, args).start()
    started = http.events("started")[0]
    assert "testing" not in started and started["from"] == "14155550100"
    # last 10 digits match; the numbers come from the per-base-URL cache
    await Call(clock, SimpleNamespace(call_data=CallDataModel(**{"from": "09876543210"}))).start()
    assert len(http.events("started")) == 2
    [get] = http.gets()
    assert get["url"] == "https://api.superbryn.com/public-api/v1/test-calls/caller-numbers"
    assert get["headers"] == {"X-API-Key": "sk_test"}
    expiry, _ = cr._caller_numbers_cache[cr._numbers_key("https://api.superbryn.com", "sk_test")]
    assert 3590 < expiry - time.monotonic() <= 3600


async def test_dialled_superbryn_number_also_counts(clock, http):
    await Call(clock, from_number="+15550009999", to_number="+1 415 555 0100").start()
    assert len(http.events("started")) == 1


async def test_not_a_test_call_sends_nothing(clock, http):
    call = await Call(clock, from_number="+15550009999").start()
    await user_turn(call, "hello", start=1, stop=2)
    await call.end(at=3)
    assert http.events() == [] and len(http.gets()) == 1
    assert call.observer._capture._user_turns == []  # capture stops once ruled out


async def test_no_test_id_and_no_numbers_needs_no_lookup(clock, http):
    call = await Call(clock).start()
    await call.end(at=1)
    assert http.requests == []


BASE = "https://api.superbryn.com"


def numbers_left() -> float:
    """Seconds until this host's caller-number file expires."""
    with open(cr._numbers_file(BASE, "sk_test"), encoding="utf-8") as f:
        return json.load(f)["expires_at"] - time.time()


def new_process() -> None:
    """A new process per call: its memory starts empty, the host's file stays."""
    cr._caller_numbers_cache.clear()


async def test_caller_numbers_are_shared_with_a_new_process_on_the_host(clock, http):
    await Call(clock, from_number="+14155550100").start()
    new_process()
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 1 and len(http.events("started")) == 2
    assert cr._numbers_file(BASE, "sk_test") != cr._numbers_file(BASE, "another agent's key")


async def test_an_outage_is_remembered_for_a_minute_and_a_refused_key_for_an_hour(clock, http):
    http.replies = [(500, None)] * 3
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 3 and http.events() == []
    new_process()
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 3
    assert 50 < numbers_left() <= 60
    os.remove(cr._numbers_file(BASE, "sk_test"))  # the minute is over
    new_process()
    http.replies = [(403, {"error": {"code": "insufficient_scope"}})]
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 4 and http.events() == []
    assert 3590 < numbers_left() <= 3600
    new_process()
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 4


@pytest.mark.parametrize(
    ("args", "kwargs", "expected"),
    [
        (SimpleNamespace(call_data={"call_id": "cd"}, session_id="s"), {"call_id": "x"}, "x"),
        (SimpleNamespace(call_data={"call_id": "cd"}, session_id="s"), {}, "cd"),
        (
            SimpleNamespace(
                body={"dialin_settings": {"call_id": "dial", "call_domain": "d"}}, session_id="s"
            ),
            {},
            "dial",
        ),
        (SimpleNamespace(session_id="sess"), {}, "sess"),
    ],
    ids=["explicit", "call_data", "dialin", "session_id"],
)
async def test_call_id_order(clock, http, args, kwargs, expected):
    await Call(clock, args, superbryn_call_id="sb", **kwargs).start()
    assert http.events("started")[0]["call_id"] == expected


async def test_call_id_defaults_to_one_uuid4(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.end(at=1)
    started = http.events("started")[0]
    assert started["call_id"] == ended(http)["call_id"] and len(started["call_id"]) == 36


# ── event bodies ─────────────────────────────────────────────────────────


async def test_event_bodies(clock, http):
    call = await Call(
        clock,
        superbryn_call_id="sb-1",
        call_id="c-1",
        agent_id=AGENT_ID.upper(),
        from_number="+14155550100",
        to_number="+14155550123",
        base_url="https://staging.example.com/",
    ).start()
    start_fields = {
        "schema_version": 1,
        "call_id": "c-1",
        "surface": "testing",
        "source": "pipecat",
        "testing": {"superbryn_call_id": "sb-1"},
        "agent_id": AGENT_ID,
        "from": "+14155550100",
        "to": "+14155550123",
        "started_at": "2027-01-15T08:00:00.000Z",
    }
    assert http.events("started")[0] == {
        "event_id": "pc:c-1:started",
        "event": "call.started",
        **start_fields,
    }
    await call.end(at=12.5)
    body = ended(http)
    assert {k: body[k] for k in start_fields} == start_fields
    assert (body["event_id"], body["event"]) == ("pc:c-1:ended", "call.ended")
    assert (body["ended_at"], body["ended_reason"]) == ("2027-01-15T08:00:12.500Z", "ended")
    assert body["transcript"] == [] and body["tool_calls"] == []
    assert body["diagnostics"] == {"errors": []}  # nothing else to report, but no failure either
    assert set(body["context"]) == {"environment"} and "custom_data" not in body
    request = http.requests[-1]
    assert request["url"] == "https://staging.example.com/public-api/v1/calls/agent-data"
    assert request["headers"] == {"X-API-Key": "sk_test", "Content-Type": "application/json"}


async def test_unset_fields_are_absent_not_null(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.end()
    for body in http.events():
        assert None not in body.values()
        assert not {"from", "to"} & body.keys()


# ── dedupe ───────────────────────────────────────────────────────────────


async def test_frame_seen_at_three_hops_counts_once(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    tts = tts_service()
    frames = [
        (vad_start(1.2), None),
        (vad_stop(2.8), None),
        (transcription("only once"), None),
        (metrics(TTSUsageMetricsData(processor="t", value=42)), tts),
    ]
    for frame, source in frames:
        for hop in range(3):
            await call.push(frame, at=3 + hop / 10, source=source if hop == 0 else None)
        await call.push(frame, direction=UP)
    await agent_turn(call, "fine", start=4, stop=5)
    await call.end(at=6)
    body = ended(http)
    assert texts(body) == [("u1", "user", "only once"), ("a1", "agent", "fine")]
    assert body["diagnostics"]["usage"] == {"tts_characters": 42}
    assert len(body["diagnostics"]["turn_latencies"]) == 1


async def test_realtime_caller_transcript_pushed_upstream_counts_once(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    llm = llm_service(model="gpt-realtime")
    heard = transcription("where is my order")
    await call.push(vad_start(1.2), at=1.2)
    await call.push(vad_stop(2.8), at=2.8)
    await call.push(heard, at=2.9, source=llm, direction=UP)  # realtime services push it up
    await call.push(heard, at=2.95, direction=UP)  # the next processor up forwards it
    await agent_turn(call, "It shipped yesterday.", start=3.5, stop=4.5)
    await call.end(at=5)
    body = ended(http)
    assert texts(body) == [
        ("u1", "user", "where is my order"),
        ("a1", "agent", "It shipped yesterday."),
    ]
    assert body["diagnostics"]["pipeline"] == "realtime"


async def test_server_side_turns_time_the_caller_without_vad(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    llm, out = llm_service(model="gpt-realtime"), FakeOutput(TransportParams())
    await call.push(UserStartedSpeakingFrame(), at=1, source=llm)  # the service's own VAD
    await call.push(UserStoppedSpeakingFrame(), at=2.5, source=llm)
    await call.push(BotStartedSpeakingFrame(), at=3, source=out)
    await call.push(tts_text("It shipped yesterday."), at=3.1, source=out)
    await call.push(transcription("where is my order"), at=3.3, source=llm, direction=UP)
    await call.push(BotStoppedSpeakingFrame(), at=4, source=out)
    await call.push(UserStartedSpeakingFrame(), at=5, source=llm)
    await call.push(transcription("thanks"), at=5.6, source=llm, direction=UP)
    await call.end(at=6)
    assert ended(http)["transcript"] == [
        {"id": "u1", "role": "user", "text": "where is my order", "start_ms": 1000, "end_ms": 2500},
        {
            "id": "a1",
            "role": "agent",
            "text": "It shipped yesterday.",
            "start_ms": 3000,
            "end_ms": 4000,
            "interrupted": False,
        },
        {"id": "u2", "role": "user", "text": "thanks", "start_ms": 5000, "end_ms": 6000},
    ]


async def test_vad_times_the_caller_when_both_are_seen(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(vad_start(1.2), at=1.2)
    await call.push(UserStartedSpeakingFrame(), at=1.5)  # the aggregator's turn start, later
    await call.push(vad_stop(2.8), at=2.8)
    await call.push(UserStoppedSpeakingFrame(), at=3.5)
    await call.push(transcription("hello"), at=3.6)
    await call.end(at=4)
    [user] = ended(http)["transcript"]
    assert (user["start_ms"], user["end_ms"]) == (1000, 2000)


@pytest.mark.parametrize("upstream_first", [False, True], ids=["down-first", "up-first"])
async def test_transcript_broadcast_both_ways_counts_once(clock, http, upstream_first):
    call = await Call(clock, superbryn_call_id="sb").start()
    down, up = transcription("hello"), transcription("hello")
    down.broadcast_sibling_id, up.broadcast_sibling_id = up.id, down.id
    pushes = [(down, FrameDirection.DOWNSTREAM), (up, UP), (down, FrameDirection.DOWNSTREAM)]
    for frame, direction in reversed(pushes) if upstream_first else pushes:
        await call.push(frame, at=1, direction=direction)
    await call.end(at=2)
    assert texts(ended(http)) == [("u1", "user", "hello")]


# ── transcript ───────────────────────────────────────────────────────────


async def test_user_turn_uses_vad_offsets(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(vad_start(1.2, start_secs=0.2), at=1.2)
    await call.push(vad_stop(2.8, stop_secs=0.8), at=2.8)
    await call.push(transcription("hello"), at=2.9)
    await call.end(at=4)
    assert ended(http)["transcript"] == [
        {"id": "u1", "role": "user", "text": "hello", "start_ms": 1000, "end_ms": 2000}
    ]


async def test_user_pause_before_the_reply_stays_one_turn(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await user_turn(call, "I need", start=1, stop=2)
    await user_turn(call, "a refund", start=3, stop=4)
    await agent_turn(call, "Sure.", start=5.5, stop=6)
    await call.end(at=7)
    user = ended(http)["transcript"][0]
    assert (user["text"], user["start_ms"], user["end_ms"]) == ("I need a refund", 1000, 4000)


async def test_late_transcript_joins_the_turn_before_the_next_speech(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await user_turn(call, "first part", start=1, stop=2)
    await agent_turn(call, "Okay.", start=3, stop=4)
    await call.push(transcription("late part"), at=4.5)
    await user_turn(call, "next question", start=6, stop=7)
    await call.end(at=9)
    assert texts(ended(http)) == [
        ("u1", "user", "first part late part"),
        ("a1", "agent", "Okay."),
        ("u2", "user", "next question"),
    ]


async def test_without_vad_each_final_transcript_is_a_turn(clock, http):
    # nothing timed the caller: a transcript's arrival comes after they finished, so the lines carry no times
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(transcription("hi"), at=1)
    await call.push(transcription("are you there"), at=3)
    await call.end(at=4)
    assert ended(http)["transcript"] == [
        {"id": "u1", "role": "user", "text": "hi"},
        {"id": "u2", "role": "user", "text": "are you there"},
    ]


async def test_agent_text_is_what_the_output_transport_played(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out, tts = FakeOutput(TransportParams()), tts_service()
    played, cut_off = tts_text("Hello!"), tts_text("Here is the rest.")
    await call.push(LLMTextFrame("Hello! Here is the rest."), at=0.5)
    await call.push(played, at=0.6, source=tts)
    await call.push(BotStartedSpeakingFrame(), at=1, source=out)
    await call.push(played, at=1.1, source=out)
    await call.push(cut_off, at=1.2, source=tts)  # synthesized, never played
    await call.push(InterruptionFrame(), at=1.5)
    await call.push(BotStoppedSpeakingFrame(), at=1.6, source=out)
    await call.end(at=3)
    assert ended(http)["transcript"] == [
        {
            "id": "a1",
            "role": "agent",
            "text": "Hello!",
            "start_ms": 1000,
            "end_ms": 1600,
            "interrupted": True,
        }
    ]


async def test_word_frames_join_and_early_text_goes_to_the_next_turn(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await call.push(tts_text("Hi"), at=0.9, source=out)  # played just before BotStarted is seen
    await call.push(BotStartedSpeakingFrame(), at=1, source=out)
    await call.push(tts_text("there."), at=1.2, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=2, source=out)
    await call.end(at=3)
    assert texts(ended(http)) == [("a1", "agent", "Hi there.")]


async def test_agent_text_falls_back_to_llm_text(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await call.push(LLMTextFrame("stale, never spoken"), at=0.2)
    await call.push(LLMFullResponseStartFrame(), at=0.4)
    await call.push(LLMTextFrame("Hello "), at=0.5)
    await call.push(BotStartedSpeakingFrame(), at=1, source=out)
    await call.push(LLMTextFrame("world."), at=1.1)
    await call.push(BotStoppedSpeakingFrame(), at=2, source=out)
    await call.end(at=3)
    assert texts(ended(http)) == [("a1", "agent", "Hello world.")]


async def test_a_turn_cut_off_before_any_played_text_gets_no_llm_text(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await call.push(LLMFullResponseStartFrame(), at=1)
    await call.push(LLMTextFrame("Sure, I can help. Your order 4471 ships Tuesday."), at=1.1)
    await call.push(BotStartedSpeakingFrame(), at=1.5, source=out)
    await call.push(InterruptionFrame(), at=1.8)  # cut off in its first (sentence-level) sentence
    await call.push(BotStoppedSpeakingFrame(), at=1.85, source=out)
    await call.push(LLMFullResponseStartFrame(), at=3)
    await call.push(LLMTextFrame("Go ahead."), at=3.1)
    await call.push(BotStartedSpeakingFrame(), at=3.5, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=4, source=out)
    await call.end(at=5)
    assert texts(ended(http)) == [("a2", "agent", "Go ahead.")]


async def test_no_llm_text_once_the_output_transport_played_text(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await agent_turn(call, "Hello.", start=0.5, stop=1, out=out)
    await call.push(LLMFullResponseStartFrame(), at=2)
    await call.push(LLMTextFrame("Your order 4471 ships Tuesday."), at=2.1)
    await call.push(BotStartedSpeakingFrame(), at=2.5, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=3, source=out)  # none of it reported played
    await call.end(at=4)
    assert texts(ended(http)) == [("a1", "agent", "Hello.")]


async def test_speech_the_agent_never_heard_is_left_out(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await user_turn(call, "check my order", start=1, stop=2)
    await call.push(BotStartedSpeakingFrame(), at=3, source=out)
    for direction in (FrameDirection.DOWNSTREAM, UP):  # the aggregator broadcasts it
        await call.push(UserMuteStartedFrame(), at=3, direction=direction)
    await call.push(tts_text("Let me check."), at=3.1, source=out)
    await call.push(transcription("it's order 4471"), at=3.8)  # muted: the aggregator drops it
    await call.push(BotStoppedSpeakingFrame(), at=4, source=out)
    await call.push(UserMuteStoppedFrame(), at=4)
    await user_turn(call, "hello?", start=5, stop=6)
    await call.end(at=7)
    assert texts(ended(http)) == [
        ("u1", "user", "check my order"),
        ("a1", "agent", "Let me check."),
        ("u2", "user", "hello?"),
    ]


async def test_interrupted_only_while_the_bot_speaks(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await call.push(BotStartedSpeakingFrame(), at=1, source=out)
    await call.push(tts_text("Let me tell you"), at=1.1, source=out)
    await call.push(InterruptionFrame(), at=1.5)
    await call.push(BotStoppedSpeakingFrame(), at=1.6, source=out)
    await call.push(InterruptionFrame(), at=2)  # a user turn start while the bot is quiet
    await agent_turn(call, "Go ahead.", start=3, stop=4, out=out)
    await call.end(at=5)
    assert [t["interrupted"] for t in ended(http)["transcript"]] == [True, False]


async def test_turns_without_text_are_skipped(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await call.push(vad_start(0.7), at=0.7)  # noise: VAD fired, no transcript
    await call.push(vad_stop(1.3), at=1.3)
    await call.push(BotStartedSpeakingFrame(), at=1.5, source=out)
    await call.push(tts_text("   "), at=1.6, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=2, source=out)
    await user_turn(call, "hello", start=3, stop=4)
    await agent_turn(call, "Hi.", start=5, stop=6, out=out)
    await call.end(at=7)
    body = ended(http)
    assert texts(body) == [("u2", "user", "hello"), ("a2", "agent", "Hi.")]
    first, second = body["diagnostics"]["turn_latencies"]
    assert "turn_id" not in first and second["turn_id"] == "a2"


async def test_open_turns_close_when_the_call_is_cancelled(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await call.push(BotStartedSpeakingFrame(), at=1, source=out)
    await call.push(tts_text("Goodbye"), at=1.1, source=out)
    await call.end(CancelFrame(), at=2.5)
    body = ended(http)
    assert body["ended_reason"] == "cancelled"
    assert body["transcript"][0]["end_ms"] == 2500


# ── tool calls ───────────────────────────────────────────────────────────


async def test_tool_call_turn_ids(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out = FakeOutput(TransportParams())
    await user_turn(call, "check my order", start=1, stop=2)
    await call.push(BotStartedSpeakingFrame(), at=3, source=out)
    await call.push(tts_text("Let me check."), at=3.1, source=out)
    await call.push(tool_start("t1"), at=3.5)
    await call.push(BotStoppedSpeakingFrame(), at=4, source=out)
    await call.push(tool_result("t1", {"status": "shipped"}), at=4.2)
    await user_turn(call, "and the refund", start=5, stop=6)
    await call.push(tool_start("t2"), at=7)
    await agent_turn(call, "Refund is done.", start=8, stop=9, out=out)
    await user_turn(call, "bye", start=10, stop=11)
    await call.push(tool_start("t3"), at=12)
    await call.end(at=13)
    calls = ended(http)["tool_calls"]
    # t1 runs during a1; t2 follows a user turn, so the next reply; t3 has no reply after it
    assert [(c["id"], c.get("turn_id")) for c in calls] == [
        ("t1", "a1"),
        ("t2", "a2"),
        ("t3", None),
    ]
    assert calls[0] == {
        "id": "t1",
        "name": "lookup",
        "arguments": {"q": "t1"},
        "result": {"status": "shipped"},
        "is_error": False,
        "start_ms": 3500,
        "end_ms": 4200,
        "turn_id": "a1",
    }
    assert "end_ms" not in calls[2] and "result" not in calls[2]


async def test_tool_call_before_the_reply_starts_takes_the_reply(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await user_turn(call, "weather?", start=1, stop=2)
    await call.push(tool_start("t1", "weather"), at=3)
    await call.push(tool_result("t1", "sunny", "weather"), at=3.2)
    await agent_turn(call, "It is sunny.", start=3.5, stop=4.5)
    await call.end(at=5)
    assert ended(http)["tool_calls"][0]["turn_id"] == "a1"


async def test_cancelled_tool_call_keeps_no_result(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(tool_start("t1"), at=1)
    await call.push(tool_result("t1", {"partial": True}, final=False), at=1.5)
    await call.push(FunctionCallCancelFrame(function_name="lookup", tool_call_id="t1"), at=2)
    await call.push(tool_result("t1", {"late": True}), at=3)
    await call.end(at=4)
    [tool] = ended(http)["tool_calls"]
    assert "result" not in tool
    assert (tool["start_ms"], tool["end_ms"]) == (1000, 2000)


async def test_tool_values_are_kept_as_json_or_stringified(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    when = datetime(2026, 9, 28, tzinfo=UTC)
    await call.push(tool_start("t1", arguments={"when": when, "n": 2}), at=1)
    await call.push(tool_result("t1", SimpleNamespace(x=1)), at=2)
    await call.push(tool_start("t2", arguments={"bad": math.nan}), at=3)
    await call.push(tool_result("t2", [1, "two", None]), at=4)
    await call.end(at=5)
    first, second = ended(http)["tool_calls"]
    assert first["arguments"] == {"when": "2026-09-28 00:00:00+00:00", "n": 2}
    assert first["result"] == "namespace(x=1)"
    assert second["arguments"] == "{'bad': nan}"
    assert second["result"] == [1, "two", None]


# ── latency ──────────────────────────────────────────────────────────────


async def test_latency_entry_with_all_five_values(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    stt, llm, tts = stt_service(), llm_service(), tts_service()
    await call.push(vad_start(1.2), at=1.2)
    await call.push(vad_stop(2.2, stop_secs=0.2), at=2.2)
    await call.push(turn_metrics(40), at=2.25)
    await call.push(ttfb(0.3), at=2.3, source=stt)
    await call.push(transcription("hi"), at=2.3, source=stt)
    await call.push(ttfb(0.45), at=2.8, source=llm)
    await call.push(ttfb(0.9), at=2.9, source=llm)  # a second LLM request: the first counts
    await call.push(ttfb(0.12), at=3.0, source=tts)
    await agent_turn(call, "Hello.", start=3.1, stop=4)
    await call.end(at=5)
    assert ended(http)["diagnostics"]["turn_latencies"] == [
        {
            "turn_id": "a1",
            "turn_ms": 1100,
            "endpointing_ms": 240,
            "transcriber_ms": 300,
            "model_ms": 450,
            "voice_ms": 120,
        }
    ]


async def test_late_stt_ttfb_attaches_to_the_last_reply(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    stt = stt_service()
    await user_turn(call, "one", start=1, stop=2)
    await agent_turn(call, "Reply one.", start=3, stop=4)
    await call.push(ttfb(1.9), at=4.5, source=stt)  # reported after stt_ttfb_timeout
    await call.push(ttfb(1.0), at=4.6, source=stt)
    await user_turn(call, "two", start=5, stop=6)
    await agent_turn(call, "Reply two.", start=7, stop=8)
    await call.push(vad_start(8.7), at=8.7)
    await call.push(vad_stop(9.8), at=9.8)
    await call.push(ttfb(1.5), at=10, source=stt)
    await call.end(at=12)
    latencies = ended(http)["diagnostics"]["turn_latencies"]
    assert [e.get("transcriber_ms") for e in latencies] == [1900, None]


async def test_interruption_drops_stale_llm_and_tts_metrics(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    llm, tts = llm_service(), tts_service()
    await user_turn(call, "hello", start=1, stop=2)
    await call.push(ttfb(0.2), at=2.9, source=llm)  # from the response being cancelled
    await call.push(ttfb(0.1), at=2.95, source=tts)
    await call.push(InterruptionFrame(), at=3)
    await call.push(ttfb(0.6), at=3.6, source=llm)
    await agent_turn(call, "Hi.", start=4, stop=5)
    await call.end(at=6)
    [entry] = ended(http)["diagnostics"]["turn_latencies"]
    assert entry["model_ms"] == 600 and "voice_ms" not in entry
    assert entry["turn_ms"] == 2000


async def test_user_resuming_resets_the_cycle(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(vad_start(1.2), at=1.2)
    await call.push(vad_stop(2.8), at=2.8)
    await call.push(ttfb(0.3), at=2.9, source=llm_service())
    await user_turn(call, "long sentence", start=3, stop=4)  # the user goes on: silence at 4.0
    await agent_turn(call, "Sure.", start=5, stop=6)
    await call.end(at=7)
    [entry] = ended(http)["diagnostics"]["turn_latencies"]
    assert entry == {"turn_id": "a1", "turn_ms": 1000, "endpointing_ms": 800}


async def test_no_entry_for_the_greeting_or_a_reply_after_a_tool_call(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await agent_turn(call, "Welcome!", start=0.5, stop=1.5)
    await user_turn(call, "order status", start=2, stop=3)
    await agent_turn(call, "Checking.", start=4, stop=5)
    await call.push(tool_start("t1"), at=5.5)
    await call.push(tool_result("t1", "ok"), at=6)
    await agent_turn(call, "It shipped.", start=6.5, stop=7.5)
    await call.end(at=8)
    latencies = ended(http)["diagnostics"]["turn_latencies"]
    assert [e["turn_id"] for e in latencies] == ["a2"]


async def test_endpointing_uses_turn_metrics_else_vad_stop_secs(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(vad_start(1.2), at=1.2)
    await call.push(vad_stop(2.2, stop_secs=0.2), at=2.2)
    await call.push(turn_metrics(35, complete=False), at=2.25)
    await call.push(transcription("one"), at=2.3)
    await agent_turn(call, "A.", start=3, stop=4)
    await call.push(vad_start(5.2), at=5.2)
    await call.push(vad_stop(6.2, stop_secs=0.2), at=6.2)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        smart = SmartTurnMetricsData(
            processor="BaseSmartTurn",
            is_complete=True,
            probability=0.8,
            e2e_processing_time_ms=55.4,
        )
    await call.push(metrics(smart), at=6.3)
    await call.push(transcription("two"), at=6.3)
    await agent_turn(call, "B.", start=7, stop=8)
    await call.end(at=9)
    latencies = ended(http)["diagnostics"]["turn_latencies"]
    assert [e["endpointing_ms"] for e in latencies] == [200, 255]


async def released_turn(
    call: Call, text: str, *, silence: float, wait: float, reply: float, analysis=None
) -> None:
    """VAD stop 0.2 s after `silence`, the turn released `wait` after it, the reply at `reply`."""
    await call.push(vad_start(silence - 0.8), at=silence - 0.8)
    await call.push(vad_stop(silence + 0.2, stop_secs=0.2), at=silence + 0.2)
    if analysis is not None:
        await call.push(analysis, at=silence + 0.24)
    await call.push(transcription(text), at=silence + 0.25)
    await call.push(UserStoppedSpeakingFrame(), at=silence + wait)
    await agent_turn(call, "Okay.", start=reply, stop=reply + 0.5)


async def test_endpointing_runs_to_the_turn_release(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    # a speech-timeout stop strategy waits 0.6 s after VAD's stop, then the turn is released
    await released_turn(call, "what is my balance", silence=2, wait=0.8, reply=3.3)
    await released_turn(call, "and my limit", silence=6, wait=0.85, reply=7.2)
    await call.push(UserStoppedSpeakingFrame(), at=8)  # no VAD stop before it: not a wait
    await call.end(at=9)
    diagnostics = ended(http)["diagnostics"]
    latencies = diagnostics["turn_latencies"]
    assert [(e["endpointing_ms"], e["turn_ms"]) for e in latencies] == [
        (800, 1300),
        (850, 1200),
    ]
    assert diagnostics["configuration"]["turn_detection"] == {
        "type": "vad",
        "min_end_of_turn_ms": 800,
    }


@pytest.mark.parametrize(
    ("waits", "least"), [((1.23, 0.25), 250), ((1.23,), 1200)], ids=["measured", "capped"]
)
async def test_smart_turn_timeout_counts_in_endpointing(clock, http, waits, least):
    call = await Call(clock, superbryn_call_id="sb").start()
    params = SpeechControlParamsFrame(
        vad_params=VADParams(stop_secs=0.2), turn_params=SmartTurnParams(stop_secs=1.0)
    )
    await call.push(params)
    for i, wait in enumerate(waits):
        silence = 2 + i * 4
        # "incomplete" at VAD's stop, then the analyzer's 1 s silence timeout ends the turn
        analysis = turn_metrics(35, complete=wait < 1)
        await released_turn(
            call, "um", silence=silence, wait=wait, reply=silence + 1.5, analysis=analysis
        )
    await call.end(at=12)
    diagnostics = ended(http)["diagnostics"]
    assert [e["endpointing_ms"] for e in diagnostics["turn_latencies"]] == [
        round(w * 1000) for w in waits
    ]
    assert diagnostics["configuration"]["turn_detection"] == {
        "type": "BaseSmartTurn",
        "min_end_of_turn_ms": least,
        "max_end_of_turn_ms": 1200,
    }


def tone(seconds: float = 0.02, rate: int = 16000) -> InputAudioRawFrame:
    """The caller talking: a 440 Hz tone, about -23 dBFS."""
    t = np.arange(int(rate * seconds)) / rate
    pcm = (3000 * np.sin(2 * np.pi * 440 * t)).astype("<i2").tobytes()
    return InputAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1)


async def caller_audio(call: Call, *, start: float, speech: float, silence: float) -> None:
    """The caller's line in 20 ms frames: quiet, `speech` seconds of tone from `start`, then `silence`."""
    loud = [False] * 10 + [True] * round(speech / 0.02) + [False] * round(silence / 0.02)
    for i, is_loud in enumerate(loud):
        await call.push(tone() if is_loud else audio(0.02), at=start + (i - 9) * 0.02)


async def test_speech_to_speech_turn_latency_from_the_callers_audio(clock, http):
    """No VAD frames: the caller's audio times the wait; the service's turn release, the endpointing."""
    llm, out = llm_service(model="gpt-realtime"), FakeOutput(TransportParams())
    call = await Call(clock, superbryn_call_id="sb").start()
    await caller_audio(call, start=1.0, speech=1.2, silence=0.3)  # speech ends at 2.2
    await call.push(UserStoppedSpeakingFrame(), at=2.9, source=llm)  # the server's release
    await call.push(BotStartedSpeakingFrame(), at=3.0, source=out)
    await call.push(tts_text("It shipped yesterday."), at=3.1, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=4.0, source=out)
    await call.end(at=5)
    assert ended(http)["diagnostics"]["turn_latencies"] == [
        {"turn_ms": 800, "endpointing_ms": 700, "turn_id": "a1"}
    ]


async def test_reply_inside_the_audio_hangover_gets_a_turn_latency(clock, http):
    """A model that answers before the detector's hangover ends (Nova Sonic and Gemini Live send no turn frames)."""
    out = FakeOutput(TransportParams())
    call = await Call(clock, superbryn_call_id="sb").start()
    await caller_audio(call, start=1.0, speech=0.6, silence=0.1)  # ends at 1.6, quiet to 1.7
    await call.push(BotStartedSpeakingFrame(), at=1.75, source=out)
    await call.push(tts_text("Sure."), at=1.8, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=2.5, source=out)
    await call.end(at=3)
    assert ended(http)["diagnostics"]["turn_latencies"] == [{"turn_ms": 150, "turn_id": "a1"}]


async def test_vad_times_the_turn_when_both_vad_and_audio_are_seen(clock, http):
    out = FakeOutput(TransportParams())
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(vad_start(1.2), at=1.2)
    await caller_audio(call, start=1.2, speech=1.0, silence=0.5)  # tone to 2.2
    await call.push(vad_stop(3.0, stop_secs=0.5), at=3.0)
    await agent_turn(call, "Sure.", start=3.2, stop=4.0, out=out)
    await call.end(at=5)
    [entry] = ended(http)["diagnostics"]["turn_latencies"]
    assert entry["turn_ms"] == 700  # from VAD's stop (3.0 - 0.5), not the audio's end


# ── configuration, tools, pipeline, usage ────────────────────────────────


async def test_configuration_from_observed_services(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    stt, llm, tts = stt_service(), llm_service(model="unknown"), tts_service()
    unused = stt_service(model="whisper")
    await call.push(StartFrame(), source=unused)  # lifecycle frames don't count
    await call.push(transcription("hi"), at=1, source=stt)
    await call.push(ttfb(0.4, model="claude-x"), at=1.1, source=llm)
    await call.push(tts_text("hi"), at=1.2, source=tts)
    await call.end(at=2)
    diagnostics = ended(http)["diagnostics"]
    assert diagnostics["configuration"]["stt"] == {"provider": "deepgram", "model": "nova-3"}
    assert diagnostics["configuration"]["llm"] == {"provider": "openai", "model": "claude-x"}
    assert diagnostics["configuration"]["tts"] == {
        "provider": "cartesia",
        "model": "sonic-2",
        "voice": "voice-1",
    }
    assert diagnostics["pipeline"] == "stt_llm_tts"


async def test_customer_subclass_reports_the_pipecat_provider(clock, http):
    class MyLLM(FakeLLM):
        pass

    MyLLM.__module__ = "bot.services"
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(LLMTextFrame("x"), source=MyLLM(settings=LLMSettings(model="m" * 300)))
    await call.end(at=1)
    diagnostics = ended(http)["diagnostics"]
    assert diagnostics["configuration"]["llm"] == {"provider": "openai", "model": "m" * 200}
    assert diagnostics["pipeline"] == "realtime"


@pytest.mark.parametrize(
    ("roles", "expected"),
    [(("stt", "llm", "tts"), "stt_llm_tts"), (("llm",), "realtime"), (("llm", "tts"), None)],
)
async def test_pipeline_kind(clock, http, roles, expected):
    services = {"stt": stt_service, "llm": llm_service, "tts": tts_service}
    call = await Call(clock, superbryn_call_id="sb").start()
    for role in roles:
        await call.push(InterruptionFrame(), source=services[role]())
    await call.end(at=1)
    assert ended(http)["diagnostics"].get("pipeline") == expected


async def test_smart_turn_detection_configuration(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    params = SpeechControlParamsFrame(
        vad_params=VADParams(stop_secs=0.2), turn_params=SmartTurnParams(stop_secs=3.0)
    )
    await call.push(params)
    await call.push(turn_metrics(30), at=1)
    await call.end(at=2)
    assert ended(http)["diagnostics"]["configuration"]["turn_detection"] == {
        "type": "BaseSmartTurn",
        "min_end_of_turn_ms": 200,
        "max_end_of_turn_ms": 3200,
    }


async def test_vad_only_turn_detection(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await user_turn(call, "hi", start=1, stop=2)
    await call.end(at=3)
    assert ended(http)["diagnostics"]["configuration"]["turn_detection"] == {
        "type": "vad",
        "min_end_of_turn_ms": 800,
    }


async def test_tools_from_llm_context(clock, http):
    def schema(name, description="") -> FunctionSchema:
        return FunctionSchema(name=name, description=description, properties={}, required=[])

    first = ToolsSchema(standard_tools=[schema("lookup", "Find an order"), schema("")])
    later = ToolsSchema(
        standard_tools=[
            schema("lookup", "changed"),
            schema("refund"),
            schema("n" * 250, "d" * 5000),
        ]
    )
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(LLMContextFrame(context=LLMContext(tools=first)), at=1)
    await call.push(LLMContextFrame(context=LLMContext()), at=1.5)
    await call.push(LLMContextFrame(context=LLMContext(tools=later)), at=2)
    await call.end(at=3)
    assert ended(http)["diagnostics"]["configuration"]["tools"] == [
        {"name": "lookup", "description": "Find an order"},
        {"name": "refund"},
        {"name": "n" * 200, "description": "d" * 4000},
    ]


async def test_tools_are_capped_at_200(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    schemas = ToolsSchema(standard_tools=[FunctionSchema(f"f{i}", "", {}, []) for i in range(250)])
    await call.push(LLMContextFrame(context=LLMContext(tools=schemas)))
    await call.end(at=1)
    assert len(ended(http)["diagnostics"]["configuration"]["tools"]) == 200


async def test_usage_counts_cached_input_per_service_kind(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    openai, anthropic, tts = llm_service(), llm_service(cls=FakeAnthropicLLM), tts_service()
    # OpenAI-style prompt_tokens include cached tokens; Anthropic's leave reads and writes out
    inclusive = LLMTokenUsage(
        prompt_tokens=100, completion_tokens=20, total_tokens=120, cache_read_input_tokens=40
    )
    exclusive = LLMTokenUsage(
        prompt_tokens=10,
        completion_tokens=5,
        total_tokens=15,
        cache_read_input_tokens=80,
        cache_creation_input_tokens=5,
    )
    await call.push(metrics(LLMUsageMetricsData(processor="o", value=inclusive)), source=openai)
    await call.push(metrics(LLMUsageMetricsData(processor="a", value=exclusive)), source=anthropic)
    await call.push(metrics(TTSUsageMetricsData(processor="t", value=17)), source=tts)
    await call.push(metrics(TTSUsageMetricsData(processor="t", value=8)), source=tts)
    await call.end(at=1)
    assert ended(http)["diagnostics"]["usage"] == {
        "llm_input_tokens": 100 + (10 + 80 + 5),
        "llm_cached_input_tokens": 40 + 80,
        "llm_output_tokens": 25,
        "tts_characters": 25,
    }


async def test_usage_without_cache_info_leaves_cached_tokens_out(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    usage = LLMTokenUsage(prompt_tokens=7, completion_tokens=3, total_tokens=10)
    await call.push(metrics(LLMUsageMetricsData(processor="o", value=usage)), source=llm_service())
    await call.end(at=1)
    assert ended(http)["diagnostics"]["usage"] == {"llm_input_tokens": 7, "llm_output_tokens": 3}


def usage_frame(given: int, produced: int) -> MetricsFrame:
    tokens = LLMTokenUsage(
        prompt_tokens=given, completion_tokens=produced, total_tokens=given + produced
    )
    return metrics(LLMUsageMetricsData(processor="o", value=tokens))


async def test_each_llm_response_gets_what_became_of_its_answer(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    out, llm = FakeOutput(TransportParams()), llm_service()
    # only a tool call; the tool's scripted "One moment." is not its answer
    await call.push(LLMFullResponseStartFrame(), at=1.0, source=llm)
    await call.push(tool_start("call_1"), at=1.3)
    await call.push(usage_frame(900, 20), at=1.4, source=llm)
    await agent_turn(call, "One moment.", start=1.5, stop=2.0, out=out)
    await call.push(tool_result("call_1", {"status": "shipped"}), at=2.1)
    # the answer, heard in full
    await call.push(LLMFullResponseStartFrame(), at=2.2, source=llm)
    await call.push(LLMTextFrame("Your order ships Tuesday."), at=2.4, source=llm)
    await call.push(usage_frame(950, 12), at=2.5, source=llm)
    await agent_turn(call, "Your order ships Tuesday.", start=2.8, stop=4.0, out=out)
    # thrown away before it played
    await call.push(LLMFullResponseStartFrame(), at=5.0, source=llm)
    await call.push(LLMTextFrame("Sure,"), at=5.3, source=llm)
    await call.push(InterruptionFrame(), at=5.4)
    # nothing came back
    await call.push(LLMFullResponseStartFrame(), at=6.0, source=llm)
    await call.push(InterruptionFrame(), at=6.1)
    # played, then the caller cut in
    await call.push(LLMFullResponseStartFrame(), at=7.0, source=llm)
    await call.push(LLMTextFrame("Let me read the whole policy to you."), at=7.2, source=llm)
    await call.push(BotStartedSpeakingFrame(), at=7.5, source=out)
    await call.push(tts_text("Let me read"), at=7.6, source=out)
    await call.push(InterruptionFrame(), at=8.0)
    await call.push(BotStoppedSpeakingFrame(), at=8.05, source=out)
    await call.end(at=9)
    assert ended(http)["diagnostics"]["llm_requests"] == [
        {"start_ms": 1000, "outcome": "tool_call", "input_tokens": 900, "output_tokens": 20},
        {"start_ms": 2200, "outcome": "spoken", "input_tokens": 950, "output_tokens": 12},
        {"start_ms": 5000, "outcome": "not_spoken"},
        {"start_ms": 6000, "outcome": "stopped"},
        {"start_ms": 7000, "outcome": "cut_off"},
    ]


async def test_a_first_token_alone_counts_as_output(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    llm = llm_service()
    await call.push(LLMFullResponseStartFrame(), at=1.0, source=llm)
    await call.push(ttfb(0.4), at=1.4, source=llm)
    await call.push(InterruptionFrame(), at=1.5)
    await call.end(at=2)
    assert ended(http)["diagnostics"]["llm_requests"] == [
        {"start_ms": 1000, "outcome": "not_spoken"}
    ]


async def test_pipeline_failures_are_sent_as_errors(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    stt, llm, tts = stt_service(), llm_service(), tts_service()
    transport = FakeOutput(TransportParams())
    leaked = (
        "socket closed: wss://api.deepgram.com/v1/listen?token=abcdefghijklmnopqrstuvwxyz0123456789"
    )

    async def fail(message, processor, at, fatal=False):
        frame = ErrorFrame(error=message, fatal=fatal, processor=processor)
        await call.push(frame, at=at, direction=UP, source=processor)

    await fail(leaked, stt, 1.0)
    await call.push(transcription("hello again"), at=2.0, source=stt)  # the STT came back
    await fail("rate limited", llm, 3.0)  # and the LLM never answered again
    await fail("voice not found", tts, 4.0, fatal=True)
    await fail("websocket closed", transport, 5.0)
    await fail(
        "Error executing function call [lookup]: boom", llm, 6.0
    )  # a tool's, not the pipeline's
    await call.end(at=7)
    assert ended(http)["diagnostics"]["errors"] == [
        {
            "step": "stt",
            "side": "caller",
            "start_ms": 1000,
            "message": "socket closed: <url>",
            "recovered": True,
        },
        {"step": "llm", "start_ms": 3000, "message": "rate limited", "recovered": False},
        {"step": "tts", "start_ms": 4000, "message": "voice not found", "recovered": False},
        {"step": "telephony", "start_ms": 5000, "message": "websocket closed"},
    ]


async def test_a_tool_that_raised_or_does_not_exist_is_an_error(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    llm = llm_service()
    await call.push(tool_start("c1", name="lookup"), at=1.0)
    failed = ErrorFrame(error="Error executing function call [lookup]: timeout", processor=llm)
    await call.push(failed, at=1.5, direction=UP, source=llm)
    await call.push(tool_start("c2", name="refund"), at=2.0)
    missing = "The function `refund` is not currently available."
    await call.push(tool_result("c2", missing, name="refund"), at=2.1)
    await call.push(tool_start("c3", name="lookup"), at=3.0)
    await call.push(tool_result("c3", {"status": "shipped"}), at=3.4)
    await call.end(at=4)
    body = ended(http)
    assert [(c["id"], c.get("is_error")) for c in body["tool_calls"]] == [
        ("c1", True),
        ("c2", True),
        ("c3", False),
    ]
    assert (
        body["diagnostics"]["errors"] == []
    )  # a tool that failed is the tool call's error, not the pipeline's


def audio(seconds: float, rate: int = 16000) -> InputAudioRawFrame:
    return InputAudioRawFrame(
        audio=b"\x00\x00" * int(rate * seconds), sample_rate=rate, num_channels=1
    )


async def test_stt_audio_time_is_measured_when_pipecat_reports_none(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    stt = stt_service()
    for _ in range(25):
        await call.push(audio(0.1), destination=stt)
    await call.push(audio(1.0, rate=8000), destination=stt)
    stt._muted = True
    await call.push(audio(1.0), destination=stt)  # muted: the STT drops it
    await call.push(audio(1.0), destination=None)  # a hop that doesn't reach the STT
    await call.end(at=5)
    assert ended(http)["diagnostics"]["usage"] == {"stt_audio_ms": 3500}


async def test_a_segmented_stt_gets_no_measured_audio_time(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    stt = FakeSegmentedSTT(settings=STTSettings(model="whisper-1", language=None))
    await call.push(audio(1.0), destination=stt)  # only speech segments reach the provider
    await call.end(at=2)
    assert "stt_audio_ms" not in (ended(http).get("diagnostics") or {}).get("usage", {})


async def test_stt_audio_time_when_pipecat_reports_it(clock, http, monkeypatch):
    class STTUsage:  # stands in for newer Pipecat's STTUsageMetricsData
        def __init__(self, value):
            self.value, self.model, self.processor = value, None, "stt"

    monkeypatch.setattr(cr, "STTUsageMetricsData", STTUsage)
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(audio(3.0), destination=stt_service())  # measured, but the report wins
    # seconds, as older versions send
    await call.push(metrics(STTUsage(12.5)), source=stt_service())
    await call.push(metrics(STTUsage(SimpleNamespace(audio_seconds=7.25))), source=stt_service())
    await call.end(at=1)
    assert ended(http)["diagnostics"]["usage"] == {"stt_audio_ms": 19750}


async def test_interruption_settings_from_the_user_turn_start_strategies(clock, http):
    strategies = [VADUserTurnStartStrategy(), MinWordsUserTurnStartStrategy(min_words=3)]
    controller = SimpleNamespace(_user_turn_strategies=SimpleNamespace(start=strategies))
    aggregator = SimpleNamespace(_user_turn_controller=controller)
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(SpeechControlParamsFrame(vad_params=VADParams(start_secs=0.3)), at=0.1)
    await call.push(LLMContextFrame(context=LLMContext()), at=1, source=aggregator)
    await call.end(at=2)
    assert ended(http)["diagnostics"]["configuration"]["interruptions"] == {
        "enabled": True,
        "min_duration_ms": 300,
        "min_words": 3,
    }


def _alternative(confidence):
    return SimpleNamespace(transcript="hi there", confidence=confidence)


@pytest.mark.parametrize(
    ("result", "expected"),
    [
        (
            SimpleNamespace(channel=SimpleNamespace(alternatives=[_alternative(0.91)])),
            0.91,
        ),  # Deepgram
        (SimpleNamespace(alternatives=[_alternative(0.84)]), 0.84),  # Google
        ({"data": {"utterance": {"text": "hi there", "confidence": 0.77}}}, 0.77),  # Gladia
        (
            {
                "Alternatives": [
                    {
                        "Transcript": "hi there.",
                        "Items": [
                            {"Content": "hi", "Confidence": 0.9},
                            {"Content": "there", "Confidence": 0.7},
                            {"Content": ".", "Type": "punctuation"},
                        ],
                    }
                ]
            },
            0.8,
        ),  # AWS Transcribe
        ([{"text": "hi", "confidence": 0.6}, {"text": " there", "confidence": 1.0}], 0.8),  # Soniox
        ({"text": "hi there"}, None),  # a provider that sends none
    ],
)
async def test_transcript_confidence_from_the_providers_result(clock, http, result, expected):
    call = await Call(clock, superbryn_call_id="sb").start()
    frame = TranscriptionFrame("hi there", "user-1", "2026-09-28T00:00:00.000Z", result=result)
    await call.push(frame, at=1)
    await call.end(at=2)
    [turn] = ended(http)["transcript"]
    assert turn.get("confidence") == expected


# ── ended_reason ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("frame", "reason"),
    [(EndFrame(), "ended"), (CancelFrame(), "cancelled"), (StopFrame(), "stopped")],
)
async def test_ended_reason_from_the_finish_frame(clock, http, frame, reason):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(ErrorFrame("recoverable"), direction=UP)
    await call.end(frame, at=1)
    assert ended(http)["ended_reason"] == reason


async def test_fatal_error_seen_upstream_means_error(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(ErrorFrame("stt died", fatal=True), direction=UP)
    await call.end(CancelFrame(), at=1)
    assert ended(http)["ended_reason"] == "error"


TASK_SOURCE = SimpleNamespace(
    name="task source"
)  # the worker's pipeline source, where its policy applies


@pytest.mark.parametrize("policy", ["end", "cancel"])
async def test_an_error_from_a_processor_that_can_no_longer_work_means_error_when_the_policy_ends_the_call(
    clock, http, monkeypatch, policy
):
    monkeypatch.setattr(
        FakeTask, "_processor_unusable_policy", SimpleNamespace(value=policy), raising=False
    )
    monkeypatch.setattr(cr, "_task_source", lambda task: TASK_SOURCE)
    call = await Call(clock, superbryn_call_id="sb").start()
    broken = SimpleNamespace(
        is_usable=False
    )  # Pipecat 1.8+: push_error(..., force_treat_as_permanent=True)
    error = ErrorFrame("tts key rejected", processor=broken)
    await call.push(error, direction=UP)  # its first hop: not yet at the task
    await call.push(
        error, direction=UP, destination=TASK_SOURCE
    )  # reaches the task: its policy ends the call
    await call.end(CancelFrame(), at=1)
    body = ended(http)
    assert (body["ended_reason"], body["ended_by"]) == ("error", "system")


async def test_a_failover_under_an_ending_policy_is_not_the_calls_end(clock, http, monkeypatch):
    monkeypatch.setattr(
        FakeTask, "_processor_unusable_policy", SimpleNamespace(value="end"), raising=False
    )
    monkeypatch.setattr(cr, "_task_source", lambda task: TASK_SOURCE)
    call = await Call(clock, superbryn_call_id="sb").start()
    primary = SimpleNamespace(
        is_usable=False
    )  # a ServiceSwitcher's primary; the switcher keeps its error back
    await call.push(ErrorFrame("llm key rejected", processor=primary), direction=UP)
    await call.end(at=1)
    body = ended(http)
    assert body["ended_reason"] != "error" and body.get("ended_by") != "system"


async def test_an_unusable_processor_under_the_default_policy_is_not_the_calls_end(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    broken = SimpleNamespace(
        is_usable=False
    )  # e.g. a ServiceSwitcher's primary; the switcher fails over
    await call.push(ErrorFrame("llm key rejected", processor=broken), direction=UP)
    await call.end(at=1)
    body = ended(http)
    assert body["ended_reason"] != "error" and body.get("ended_by") != "system"


# ── limits ───────────────────────────────────────────────────────────────


async def test_text_is_cut_to_20000_utf16_units_without_a_lone_surrogate(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await agent_turn(call, "a" * 19999 + "😀" + "b" * 10, start=1, stop=2)
    await call.end(at=3)
    assert ended(http)["transcript"][0]["text"] == "a" * 19999


async def test_ids_and_names_are_clamped_to_200(clock, http):
    call = await Call(clock, superbryn_call_id="s" * 300, call_id="c" * 300).start()
    await call.push(tool_start("i" * 300, name="n" * 300), at=1)
    await call.push(tool_start("t2", name="  "), at=2)
    await call.end(at=3)
    for body in http.events():
        assert len(body["call_id"]) == 200 and len(body["testing"]["superbryn_call_id"]) == 200
        assert len(body["event_id"]) <= 200 and body["event_id"].startswith("pc:ccc")
    [tool] = ended(http)["tool_calls"]
    assert len(tool["id"]) == 200 and len(tool["name"]) == 200


async def test_turn_latency_and_tool_caps(clock, http, monkeypatch):
    monkeypatch.setattr(cr, "_MAX_TURNS", 3)
    monkeypatch.setattr(cr, "_MAX_TOOL_CALLS", 2)
    call = await Call(clock, superbryn_call_id="sb").start()
    for i in range(4):
        await user_turn(call, f"u{i}", start=i * 10 + 1, stop=i * 10 + 2)
        await agent_turn(call, f"a{i}", start=i * 10 + 3, stop=i * 10 + 4)
        await call.push(tool_start(f"t{i}"), at=i * 10 + 5)
    await call.end(at=50)
    body = ended(http)
    assert len(body["transcript"]) == 3 and len(body["tool_calls"]) == 2
    assert len(body["diagnostics"]["turn_latencies"]) == 3


async def test_a_failing_section_is_left_out(clock, http, monkeypatch, caplog):
    def broken(self, agent_ids):
        raise RuntimeError("boom")

    monkeypatch.setattr(cr._Capture, "_turn_latencies", broken)
    call = await Call(clock, superbryn_call_id="sb").start()
    await user_turn(call, "hi", start=1, stop=2)
    await agent_turn(call, "Hello.", start=3, stop=4)
    await call.push(metrics(TTSUsageMetricsData(processor="t", value=6)), source=tts_service())
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await call.end(at=5)
    body = ended(http)
    assert "turn_latencies" not in body["diagnostics"]
    assert body["diagnostics"]["usage"] == {"tts_characters": 6}
    assert len(body["transcript"]) == 2
    assert messages(caplog, "left out") == [
        "SUPERBRYN_CALL_REPORT_ERROR: turn_latencies left out of call.ended: boom"
    ]


# ── sending ──────────────────────────────────────────────────────────────


async def test_retries_reuse_the_same_body(clock, http):
    http.replies = [(503, None), TimeoutError(), (200, {"status": "linked", "is_test": True})]
    await Call(clock, superbryn_call_id="sb").start()
    posts = [r for r in http.requests if r["method"] == "POST"]
    assert len(posts) == 3 and len({r["raw"] for r in posts}) == 1


@pytest.mark.parametrize("status", [408, 429, 500])
async def test_retryable_statuses(clock, http, status):
    http.replies = [(status, None), (200, {"is_test": True})]
    await Call(clock, superbryn_call_id="sb").start()
    assert len(http.events("started")) == 2


async def test_client_errors_are_not_retried(clock, http, caplog):
    http.replies = [(200, {"status": "linked", "is_test": True}), (400, {"error": "invalid_body"})]
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        call = await Call(clock, superbryn_call_id="sb").start()
        await call.end(at=1)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]
    [rejected] = messages(caplog, "REJECTED")
    assert "400" in rejected and "test_data:write" not in rejected


async def test_a_refused_call_started_sends_no_call_ended(clock, http, caplog):
    http.replies = [(401, {"error": "unauthorized"})]
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        call = await Call(clock, superbryn_call_id="sb").start()
        await call.end(at=1)
    assert [e["event"] for e in http.events()] == ["call.started"]  # it would be refused too
    [rejected] = messages(caplog, "REJECTED")
    assert (
        "a SuperBryn test call needs a key with test_data:write; check the key is valid" in rejected
    )


async def test_a_missing_agent_id_names_the_fix(clock, http, caplog):
    http.replies = [(400, {"code": "agent_id_required"})]
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        await Call(clock, agent_id=None, superbryn_call_id="sb").start()
    [rejected] = messages(caplog, "REJECTED")
    assert "set SUPERBRYN_AGENT_ID" in rejected


async def test_gives_up_after_three_attempts_and_still_sends_ended(clock, http):
    http.replies = [ConnectionError("down")] * 3
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.end(at=1)
    assert [e["event"] for e in http.events()] == ["call.started"] * 3 + ["call.ended"]


async def test_timeouts_per_event(clock, http, monkeypatch):
    timeouts: list[tuple[str, float]] = []
    real_send = cr._send

    async def spy(method, url, api_key, body, *, timeout, **kwargs):
        timeouts.append((body["event"] if body else method, timeout))
        return await real_send(method, url, api_key, body, timeout=timeout, **kwargs)

    monkeypatch.setattr(cr, "_send", spy)
    call = await Call(clock, from_number="+14155550100").start()
    await call.end(at=1)
    assert timeouts == [("GET", 5.0), ("call.started", 5.0), ("call.ended", 10.0)]


async def test_a_slow_attempt_times_out_and_is_retried(clock, http, monkeypatch):
    monkeypatch.setattr(cr, "_STARTED_TIMEOUT_S", 0.01)
    attempts = []

    async def slow(method, url, headers, data):
        attempts.append(method)
        await asyncio.sleep(1)

    monkeypatch.setattr(cr, "_http_request", slow)
    await Call(clock, superbryn_call_id="sb").start()
    assert attempts == ["POST"] * 3


async def test_a_call_ending_during_the_number_lookup_sends_nothing_at_once(
    clock, http, monkeypatch
):
    async def hang(method, url, headers, data):
        http.requests.append({"method": method})
        await asyncio.sleep(3600)

    monkeypatch.setattr(cr, "_http_request", hang)
    call = Call(clock, from_number="+14155550100")
    await call.observer.on_pipeline_started()
    await asyncio.sleep(0)  # the caller-number lookup is under way
    began = time.monotonic()
    await call.end(at=2)
    assert time.monotonic() - began < 0.5
    with pytest.raises(asyncio.CancelledError):
        await call.observer._start_task
    assert http.requests == [{"method": "GET"}]


async def test_a_number_matched_call_still_sending_call_started_keeps_its_record(
    clock, http, monkeypatch
):
    async def slow_started(method, url, headers, data):
        if data and json.loads(data)["event"] == "call.started":
            await asyncio.sleep(0.3)
        return await http(method, url, headers, data)

    monkeypatch.setattr(cr, "_http_request", slow_started)
    call = Call(clock, from_number="+14155550100")
    await call.observer.on_pipeline_started()
    await asyncio.sleep(0.05)  # recognised by its number; call.started is on its way
    await call.end(at=2)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]


@pytest.mark.parametrize("hangs", ["call.started", "call.ended"])
async def test_the_finish_handler_gives_up_after_one_budget(
    clock, http, monkeypatch, caplog, hangs
):
    monkeypatch.setattr(cr, "_FINISH_BUDGET_S", 0.2)

    async def answer(method, url, headers, data):
        http.requests.append({"method": method, "body": json.loads(data)})
        if http.requests[-1]["body"]["event"] == hangs:
            await asyncio.sleep(3600)
        return 200, {"status": "linked", "is_test": True}

    monkeypatch.setattr(cr, "_http_request", answer)
    call = Call(clock, superbryn_call_id="sb", call_id="c-1")
    await call.observer.on_pipeline_started()
    began = time.monotonic()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        await call.end(at=2)
    assert 0.2 <= time.monotonic() - began < 1
    assert messages(caplog, "not sent within") == [
        "SUPERBRYN_CALL_REPORT_FAILED: call.ended for call c-1 not sent within 0.2s of the call's end"
    ]
    assert call.observer._start_task.done()
    assert [e["event"] for e in http.events()][-1] == hangs


async def test_is_test_false_skips_call_ended(clock, http):
    http.replies = [(200, {"status": "ignored", "is_test": False})]
    call = await Call(clock, superbryn_call_id="sb").start()
    await user_turn(call, "hi", start=1, stop=2)
    await call.end(at=3)
    assert [e["event"] for e in http.events()] == ["call.started"]


async def test_finish_before_start_still_reports(clock, http):
    call = Call(clock, superbryn_call_id="sb")
    await call.end(CancelFrame(), at=0)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]


async def test_attach_after_the_task_started_times_the_call_from_its_first_frame(
    clock, http, caplog
):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        call = Call(clock, superbryn_call_id="sb")  # on_pipeline_started never comes
        await user_turn(call, "hello", start=1, stop=2)
        await call.observer._start_task
        assert [e["event"] for e in http.events()] == ["call.started"]
        await agent_turn(call, "Hi there.", start=3, stop=4)
        await call.end(at=60)
    body = ended(http)
    assert (body["started_at"], body["ended_at"]) == (
        "2027-01-15T08:00:01.200Z",
        "2027-01-15T08:01:00.000Z",
    )
    assert [(t["start_ms"], t["end_ms"]) for t in body["transcript"]] == [(0, 800), (1800, 2800)]
    assert len(messages(caplog, "after the task started")) == 1


async def test_finish_handler_runs_once(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.end(at=1)
    await call.end(CancelFrame(), at=2)
    assert len(http.events("ended")) == 1


# ── configuration from env, and guards ──────────────────────────────────


async def test_no_key_warns_once_and_does_nothing(clock, http, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        first = Call(clock, api_key=None, superbryn_call_id="sb")
        second = Call(clock, api_key=None, superbryn_call_id="sb")
    assert first.observer is None and second.observer is None
    assert first.task.handlers == {}
    assert len(messages(caplog, "no API key")) == 1
    assert http.requests == []


async def test_env_configuration(clock, http, monkeypatch):
    monkeypatch.setenv("SUPERBRYN_API_KEY", "sk_general")
    monkeypatch.setenv("SUPERBRYN_TEST_API_KEY", "sk_test_data")  # 0.11's eval key: not read
    monkeypatch.setenv("SUPERBRYN_BASE_URL", "http://localhost:3000/")
    monkeypatch.setenv("SUPERBRYN_AGENT_ID", AGENT_ID)
    await Call(clock, api_key=None, agent_id=None, superbryn_call_id="sb").start()
    request = http.requests[0]
    assert request["headers"]["X-API-Key"] == "sk_general"
    assert request["url"] == "http://localhost:3000/public-api/v1/calls/agent-data"
    assert request["body"]["agent_id"] == AGENT_ID


@pytest.mark.parametrize("value", ["false", "0", " OFF "])
async def test_superbryn_observer_enabled_off_sends_nothing(clock, http, monkeypatch, value):
    monkeypatch.setenv("SUPERBRYN_OBSERVER_ENABLED", value)
    call = Call(clock, superbryn_call_id="sb", production=True)
    assert call.observer is None and call.task.handlers == {} and http.requests == []


@pytest.mark.parametrize(
    "agent_id",
    [
        None,
        "pipecat-agent",
        "4347a69f-0c1e-0b6a-9f1e-2b3c4d5e6f70",
        "4347a69f-0c1e-4b6a-cf1e-2b3c4d5e6f70",
    ],
    ids=["missing", "name", "version-0", "wrong-variant"],
)
async def test_an_agent_id_that_is_not_a_uuid_is_left_out(clock, http, caplog, agent_id):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(2):
            call = await Call(clock, agent_id=agent_id, superbryn_call_id="sb").start()
            await call.end(at=1)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"] * 2
    assert all("agent_id" not in e for e in http.events())
    warned = messages(caplog, "is not a SuperBryn agent ID")
    assert len(warned) == (1 if agent_id else 0)


async def test_metrics_off_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for metrics_on in (False, False, True):
            attach(FakeTask(metrics=metrics_on), api_key="k", agent_id=AGENT_ID)
    [message] = messages(caplog, "PipelineParams")
    assert "enable_metrics and enable_usage_metrics" in message


async def test_attach_never_raises(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        attach(object(), SimpleNamespace(body=12, call_data="x"), api_key="k", agent_id=AGENT_ID)
    assert len(messages(caplog, "could not attach")) == 1


async def test_capture_errors_never_reach_the_pipeline(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    broken = VADUserStartedSpeakingFrame()
    broken.timestamp = "not a number"
    await call.push(broken)
    await call.push(object())  # not a frame at all
    await call.observer.on_push_frame(None)
    await user_turn(call, "still works", start=1, stop=2)
    await call.end(at=3)
    assert texts(ended(http)) == [("u1", "user", "still works")]


async def test_push_time_is_corrected_for_observer_lag(clock, http):
    class PipelineClock:
        def get_time(self):
            return 5_500_000_000  # the pipeline clock reads 5.5 s when the observer runs

    out = FakeOutput(TransportParams())
    # Stubbed at the public method: Pipecat 1.4 kept the clock in `_clock`, newer releases in `processor_setup`.
    out.get_clock = PipelineClock
    call = await Call(clock, superbryn_call_id="sb").start()
    clock.t = T0 + 6.0  # the observer runs late
    pushes = [
        (BotStartedSpeakingFrame(), 4_000_000_000),
        (tts_text("Hello."), 4_100_000_000),
        (BotStoppedSpeakingFrame(), 5_000_000_000),
    ]
    for frame, pushed_ns in pushes:
        data = FramePushed(
            source=out,
            destination=None,
            frame=frame,
            direction=FrameDirection.DOWNSTREAM,
            timestamp=pushed_ns,
        )
        await call.observer.on_push_frame(data)
    await call.end(at=7)
    turn = ended(http)["transcript"][0]
    assert (turn["start_ms"], turn["end_ms"]) == (4500, 5500)


# ── a real PipelineTask run ──────────────────────────────────────────────


class PassThroughOutput(BaseOutputTransport):
    """An output transport that plays nothing and forwards every frame as it arrives."""

    async def process_frame(self, frame, direction):
        await FrameProcessor.process_frame(self, frame, direction)
        await self.push_frame(frame, direction)


async def test_hooks_fire_in_a_real_pipeline_task(http, monkeypatch):
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask
    from pipecat.workers.runner import WorkerRunner

    monkeypatch.setattr(
        cr, "_DRAIN_TIMEOUT_S", 60
    )  # only the EndFrame reaching the sink ends the wait
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(
            Pipeline([PassThroughOutput(TransportParams(), enable_direct_mode=True)]),
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
    attach(
        task, SimpleNamespace(body={"superbryn_call_id": "sb-real"}), api_key="k", agent_id=AGENT_ID
    )

    async def converse():
        await asyncio.sleep(0.1)
        now = time.time()
        for frame in (
            VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=now),
            VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=now + 0.05),
            transcription("hello bot"),
            BotStartedSpeakingFrame(),
            tts_text("Hello human."),
            BotStoppedSpeakingFrame(),
        ):
            await task.queue_frame(frame)
            await asyncio.sleep(0.02)
        await task.queue_frame(EndFrame())

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    await asyncio.wait_for(asyncio.gather(runner.run(), converse()), 20)

    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]
    body = ended(http)
    assert (body["testing"]["superbryn_call_id"], body["ended_reason"]) == ("sb-real", "ended")
    assert texts(body) == [("u1", "user", "hello bot"), ("a1", "agent", "Hello human.")]
    [entry] = body["diagnostics"]["turn_latencies"]
    assert entry["turn_id"] == "a1" and entry["turn_ms"] >= 0


def real_task(*processors: FrameProcessor, attached: bool = True) -> Any:
    """A real PipelineTask over the processors and a pass-through output, test data attached."""
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(
            Pipeline([*processors, PassThroughOutput(TransportParams(), enable_direct_mode=True)]),
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
    if attached:
        args = SimpleNamespace(body={"superbryn_call_id": "sb-real"})
        attach(task, args, api_key="k", agent_id=AGENT_ID)
    return task


async def run(task: Any, script: Any, timeout: float = 20) -> None:
    from pipecat.workers.runner import WorkerRunner

    runner = WorkerRunner(handle_sigint=False)
    await runner.add_workers(task)
    await asyncio.wait_for(asyncio.gather(runner.run(), script()), timeout)


class Forward(FrameProcessor):
    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        await self.push_frame(frame, direction)


class TakesCallerTranscript(FrameProcessor):
    """Stands in for the user aggregator, which takes a realtime service's upstream transcript."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if not (isinstance(frame, TranscriptionFrame) and direction == UP):
            await self.push_frame(frame, direction)


@dataclass
class CallerSaid(SystemFrame):
    """Test trigger standing in for a realtime service's input transcription event."""

    text: str = ""


class FakeRealtimeLLM(LLMService):
    """Pushes the caller's transcript upstream, as Pipecat 1.4's realtime services do."""

    async def process_frame(self, frame, direction):
        await super().process_frame(frame, direction)
        if isinstance(frame, CallerSaid):
            await self.push_frame(TranscriptionFrame(frame.text, "", "t"), UP)
        else:
            await self.push_frame(frame, direction)


FakeRealtimeLLM.__module__ = "pipecat.services.openai.realtime.llm"


async def test_realtime_caller_line_in_a_real_pipeline(http, monkeypatch):
    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 60)
    llm = FakeRealtimeLLM(settings=LLMSettings(model="gpt-realtime"))
    task = real_task(TakesCallerTranscript(), Forward(), llm)

    async def script():
        await asyncio.sleep(0.1)
        for frame in (
            UserStartedSpeakingFrame(),
            UserStoppedSpeakingFrame(),
            BotStartedSpeakingFrame(),
            tts_text("It shipped yesterday."),
            CallerSaid(text="where is my order"),  # transcribed after the reply started
            BotStoppedSpeakingFrame(),
        ):
            await task.queue_frame(frame)
            await asyncio.sleep(0.02)
        await task.queue_frame(EndFrame())

    await run(task, script)
    body = ended(http)
    assert texts(body) == [
        ("u1", "user", "where is my order"),
        ("a1", "agent", "It shipped yesterday."),
    ]
    assert body["diagnostics"]["pipeline"] == "realtime"


class Pushed(BaseObserver):
    """The watched frame types pushed downstream so far."""

    def __init__(self, *types: type) -> None:
        super().__init__()
        self.types, self.seen = types, set()

    async def on_push_frame(self, data: FramePushed) -> None:
        if data.direction == FrameDirection.DOWNSTREAM and isinstance(data.frame, self.types):
            self.seen.add(type(data.frame))


async def test_endpointing_with_a_real_speech_timeout_strategy(http, monkeypatch):
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregator,
        LLMUserAggregatorParams,
    )
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
        SpeechTimeoutUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 60)
    stop = SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.3)
    params = LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(stop=[stop]))
    task = real_task(LLMUserAggregator(LLMContext(), params=params))
    probe = Pushed(UserStoppedSpeakingFrame)
    task.add_observer(probe)

    async def script():
        await asyncio.sleep(0.1)
        await task.queue_frame(VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=time.time()))
        await asyncio.sleep(0.2)
        await task.queue_frame(transcription("what is my balance"))
        await task.queue_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time()))
        while UserStoppedSpeakingFrame not in probe.seen:  # the strategy waits 0.3 s more
            await asyncio.sleep(0.01)
        for frame in (BotStartedSpeakingFrame(), tts_text("Forty."), BotStoppedSpeakingFrame()):
            await task.queue_frame(frame)
            await asyncio.sleep(0.02)
        await task.queue_frame(EndFrame())

    await run(task, script)
    diagnostics = ended(http)["diagnostics"]
    [entry] = diagnostics["turn_latencies"]
    # 0.2 s of VAD silence plus the strategy's 0.3 s wait; the VAD-only estimate said 200
    assert 500 <= entry["endpointing_ms"] <= entry["turn_ms"]
    detection = diagnostics["configuration"]["turn_detection"]
    # the speech-timeout strategy starts the LLM only as the turn ends: no early drafts
    assert detection == {
        "type": "vad",
        "min_end_of_turn_ms": entry["endpointing_ms"],
        "preemptive_generation": False,
    }


async def test_llm_judged_turn_completion_drafts_replies_early_in_a_real_pipeline(
    http, monkeypatch
):
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregator,
        LLMUserAggregatorParams,
    )
    from pipecat.turns.user_stop import (
        LLMTurnCompletionUserTurnStopStrategy,
        SpeechTimeoutUserTurnStopStrategy,
        deferred,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 60)
    # the detector starts the LLM; the LLM's own marker says whether the turn was complete (else it is dropped)
    stop = [
        deferred(SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.3)),
        LLMTurnCompletionUserTurnStopStrategy(),
    ]
    params = LLMUserAggregatorParams(user_turn_strategies=UserTurnStrategies(stop=stop))
    task = real_task(LLMUserAggregator(LLMContext(), params=params))

    async def script():
        await asyncio.sleep(0.1)
        await task.queue_frame(VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=time.time()))
        await asyncio.sleep(0.2)
        await task.queue_frame(transcription("what is my balance"))
        await task.queue_frame(VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time()))
        await asyncio.sleep(0.5)
        await task.queue_frame(EndFrame())

    await run(task, script)
    assert (
        ended(http)["diagnostics"]["configuration"]["turn_detection"]["preemptive_generation"]
        is True
    )


# Pipecat's speech-to-speech services that report no caller turns and stamp a response at its first output.
OUTPUT_STAMPED = [
    (
        "pipecat.services.aws.nova_sonic.llm",
        "AWSNovaSonicLLMService",
        {"secret_access_key": "x", "access_key_id": "x", "region": "us-east-1"},
    ),
    ("pipecat.services.google.gemini_live.llm", "GeminiLiveLLMService", {"api_key": "x"}),
]


@pytest.mark.parametrize(
    ("module", "name", "kwargs"), OUTPUT_STAMPED, ids=["nova-sonic", "gemini-live"]
)
async def test_output_stamped_speech_to_speech_sends_no_requests_or_caller_times(
    clock, http, module, name, kwargs
):
    # the real class; Pipecat raises ImportError without its aws / google extra
    cls = getattr(pytest.importorskip(module, exc_type=ImportError), name)
    assert (cls.__module__, cls.__qualname__) in cr._OUTPUT_STAMPED_LLMS
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        llm = cls(**kwargs)
    out = FakeOutput(TransportParams())
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(
        LLMFullResponseStartFrame(), at=2.9, source=llm
    )  # the reply's first output, not its request
    await call.push(
        transcription("where is my order"), at=2.9, source=llm, direction=UP
    )  # sent as the reply starts
    await call.push(BotStartedSpeakingFrame(), at=3.2, source=out)
    await call.push(tts_text("It shipped yesterday."), at=3.3, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=4.5, source=out)
    await call.end(at=5)
    body = ended(http)
    assert body["transcript"][0] == {"id": "u1", "role": "user", "text": "where is my order"}
    assert "llm_requests" not in body["diagnostics"]
    assert body["diagnostics"]["configuration"]["turn_detection"] == {"type": "realtime_llm"}


async def test_a_realtime_service_that_reports_turns_keeps_its_requests_and_caller_times(
    clock, http
):
    llm, out = llm_service(model="gpt-realtime"), FakeOutput(TransportParams())
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(
        UserStartedSpeakingFrame(), at=1, source=llm
    )  # OpenAI Realtime's own speech detection
    await call.push(UserStoppedSpeakingFrame(), at=2.5, source=llm)
    await call.push(
        LLMFullResponseStartFrame(), at=2.7, source=llm
    )  # its response item, before any audio
    await call.push(transcription("where is my order"), at=2.9, source=llm, direction=UP)
    await call.push(BotStartedSpeakingFrame(), at=3.2, source=out)
    await call.push(LLMTextFrame("It shipped yesterday."), at=3.2, source=llm)
    await call.push(tts_text("It shipped yesterday."), at=3.3, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=4.5, source=out)
    await call.end(at=5)
    body = ended(http)
    assert (body["transcript"][0]["start_ms"], body["transcript"][0]["end_ms"]) == (1000, 2500)
    assert body["diagnostics"]["llm_requests"] == [{"start_ms": 2700, "outcome": "spoken"}]


async def test_muted_speech_in_a_real_pipeline(http, monkeypatch):
    from pipecat.processors.aggregators.llm_response_universal import (
        LLMUserAggregator,
        LLMUserAggregatorParams,
    )
    from pipecat.turns.user_mute.always_user_mute_strategy import AlwaysUserMuteStrategy
    from pipecat.turns.user_stop.speech_timeout_user_turn_stop_strategy import (
        SpeechTimeoutUserTurnStopStrategy,
    )
    from pipecat.turns.user_turn_strategies import UserTurnStrategies

    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 60)
    params = LLMUserAggregatorParams(
        user_turn_strategies=UserTurnStrategies(
            stop=[SpeechTimeoutUserTurnStopStrategy(user_speech_timeout=0.05)]
        ),
        user_mute_strategies=[AlwaysUserMuteStrategy()],  # muted while the bot speaks
    )
    task = real_task(LLMUserAggregator(LLMContext(), params=params))

    async def script():
        await asyncio.sleep(0.1)
        for make in (
            lambda: VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=time.time()),
            lambda: transcription("check my order"),
            lambda: VADUserStoppedSpeakingFrame(stop_secs=0.2, timestamp=time.time()),
            BotStartedSpeakingFrame,
            lambda: tts_text("Let me check."),
            lambda: transcription("it's order 4471"),  # said over the agent: dropped
            BotStoppedSpeakingFrame,
            lambda: VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=time.time() + 0.2),
            lambda: transcription("hello?"),
        ):
            await task.queue_frame(make())
            await asyncio.sleep(0.05)
        await task.queue_frame(EndFrame())

    await run(task, script)
    assert texts(ended(http)) == [
        ("u1", "user", "check my order"),
        ("a1", "agent", "Let me check."),
        ("u2", "user", "hello?"),
    ]


@pytest.mark.parametrize("late", [False, True], ids=["before-run", "after-start"])
async def test_attach_time_in_a_real_pipeline(http, monkeypatch, caplog, late):
    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 60)
    task = real_task(attached=False)

    def attach_report():
        args = SimpleNamespace(body={"superbryn_call_id": "sb-real"})
        attach(task, args, api_key="k", agent_id=AGENT_ID)

    async def script():
        await asyncio.sleep(0.2)
        if late:
            attach_report()  # e.g. in on_client_connected, once the pipeline is running
        for frame in (BotStartedSpeakingFrame(), tts_text("Hello."), BotStoppedSpeakingFrame()):
            await task.queue_frame(frame)
            await asyncio.sleep(0.05)
        await task.queue_frame(EndFrame())

    if not late:
        attach_report()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await run(task, script)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]
    body = ended(http)
    [turn] = body["transcript"]
    assert body["started_at"] < body["ended_at"] and 0 < turn["end_ms"] < 1000
    assert len(messages(caplog, "after the task started")) == int(late)


# ── production calls (surface "observability") ────────────────────────────


class FakeRecorder:
    """SuperbrynAudioRecorder stand-in: what was asked of it, and a WAV file to hand over."""

    num_channels = 2

    def __init__(self, path) -> None:
        self.path, self.events = path, []

    async def start(self) -> None:
        self.events.append("start")

    async def stop(self) -> None:
        self.events.append("stop")

    async def finalize(self):
        self.events.append("finalize")
        return self.path

    def cleanup(self) -> None:
        self.events.append("cleanup")


@pytest.fixture
def recording(monkeypatch, tmp_path) -> SimpleNamespace:
    """The recorder attach() inserts, and the presign + upload, stubbed: no pipeline, no S3."""
    rec = SimpleNamespace(recorders=[], presigned=[], uploaded=[], upload_ok=True)

    def insert(pipeline):
        path = tmp_path / f"call-{len(rec.recorders)}.wav"
        path.write_bytes(b"RIFF")
        rec.recorders.append(FakeRecorder(path))
        return rec.recorders[-1]

    async def presign(base_url, api_key, session_id, num_channels):
        rec.presigned.append((base_url, api_key, session_id, num_channels))
        return {
            "upload_url": f"https://s3.test/put/{session_id}",
            "public_url": f"https://s3.test/observability/pipecat/{session_id}-2ch.wav",
        }

    async def upload(url, path):
        rec.uploaded.append((url, path))
        return rec.upload_ok

    monkeypatch.setattr(cr, "_insert_recorder", insert)
    monkeypatch.setattr(cr, "fetch_recording_upload_url", presign)
    monkeypatch.setattr(cr, "upload_wav_via_presigned", upload)
    return rec


def runner(**fields: Any) -> SimpleNamespace:
    return SimpleNamespace(session_id="sess-1", **fields)


async def test_a_production_call_is_reported_with_its_recording(clock, http, recording):
    call = await Call(clock, runner(), production=True).start()
    started = http.events("started")[0]
    assert started == {
        "schema_version": 1,
        "event_id": "pc:sess-1:started",
        "event": "call.started",
        "surface": "observability",
        "source": "pipecat",
        "call_id": "sess-1",
        "agent_id": AGENT_ID,
        "started_at": "2027-01-15T08:00:00.000Z",
        "observability": {"mode": "observe"},
    }
    await user_turn(call, "hi", start=1, stop=2)
    await agent_turn(call, "Hello.", start=3, stop=4)
    await call.end(at=5)
    body = ended(http)
    assert body["event_id"] == "pc:sess-1:ended" and "testing" not in body
    url = "https://s3.test/observability/pipecat/sess-1-2ch.wav"
    assert body["observability"] == {
        "mode": "observe",
        "recording_url": url,
        "stereo_recording_url": url,
        "recording_url_source": "superbryn_s3",
    }
    [recorder] = recording.recorders
    assert recorder.events == ["start", "finalize", "stop", "cleanup"]
    assert recording.presigned == [("https://api.superbryn.com", "sk_test", "sess-1", 2)]
    assert texts(body) == [("u1", "user", "hi"), ("a1", "agent", "Hello.")]


async def test_call_ids_per_surface(clock, http, recording):
    """A test call keeps the telephony call's ID (Custom cURL matches it); a production call the runner's
    session ID, as 0.11 named Monitor calls; call_id= names both."""
    twilio = CallDataModel(call_id="CA1", **{"from": "+15550001111", "to": "+15550002222"})
    await Call(clock, runner(call_data=twilio), production=True).start()
    await Call(clock, runner(call_data=twilio), production=True, superbryn_call_id="sb").start()
    await Call(clock, runner(call_data=twilio), production=True, call_id="mine").start()
    assert [(e["surface"], e["call_id"]) for e in http.events("started")] == [
        ("observability", "sess-1"),
        ("testing", "CA1"),
        ("observability", "mine"),
    ]
    # without a session ID, a production call gets a random one
    await Call(clock, production=True).start()
    assert len(http.events("started")[-1]["call_id"]) == 36


async def test_production_false_sends_nothing_and_records_nothing(clock, http, recording):
    call = await Call(clock, runner(), production=False).start()
    await user_turn(call, "hi", start=1, stop=2)
    await call.end(at=3)
    assert http.requests == [] and recording.recorders == []
    assert call.observer._logs is None


async def test_record_false_sends_no_recording(clock, http, recording):
    call = await Call(clock, runner(), production=True, record=False).start()
    await call.end(at=2)
    assert ended(http)["observability"] == {"mode": "observe"}
    assert recording.recorders == [] and recording.uploaded == []


async def test_an_upload_that_fails_leaves_the_call_unrecorded_not_unreported(
    clock, http, recording
):
    recording.upload_ok = False
    call = await Call(clock, runner(), production=True).start()
    await call.end(at=2)
    assert ended(http)["observability"] == {"mode": "observe"}
    assert recording.recorders[0].events[-1] == "cleanup"


@pytest.mark.parametrize("when", ["before-start", "during-call"])
async def test_your_own_recording_is_sent_instead(clock, http, recording, when):
    call = Call(clock, runner(), production=True)
    if when == "before-start":
        call.report.set_recording_url(" https://my.bucket/call-1.mp3 ")
    await call.start()
    if when == "during-call":
        call.report.set_recording_url("https://my.bucket/call-1.mp3")
    await call.end(at=2)
    assert ended(http)["observability"] == {
        "mode": "observe",
        "recording_url": "https://my.bucket/call-1.mp3",
        "recording_url_source": "external",
    }
    [recorder] = recording.recorders
    assert "start" in recorder.events if when == "during-call" else "start" not in recorder.events
    assert recording.uploaded == [] and recorder.events[-1] == "cleanup"


async def test_a_test_call_found_by_its_number_is_not_recorded(clock, http, recording):
    call = await Call(clock, runner(), production=True, from_number="+14155550100").start()
    assert http.events("started")[0]["surface"] == "testing"
    assert recording.recorders[0].events == ["start", "stop", "cleanup"]  # dropped once recognised
    logging.getLogger("agent").info("during the test call")
    await call.end(at=2)
    body = ended(http)
    assert "observability" not in body and "logs" not in body["diagnostics"]
    assert recording.uploaded == []


async def test_a_call_known_to_be_a_test_is_never_recorded(clock, http, recording):
    await Call(clock, runner(), production=True, superbryn_call_id="sb").start()
    assert recording.recorders == []


async def test_a_production_call_ending_during_the_number_lookup_waits_for_it(
    clock, http, monkeypatch, recording
):
    async def slow_numbers(method, url, headers, data):
        if method == "GET":
            await asyncio.sleep(0.2)
        return await http(method, url, headers, data)

    monkeypatch.setattr(cr, "_http_request", slow_numbers)
    http.numbers = []
    call = Call(clock, runner(), production=True, from_number="+15550001111")
    await call.observer.on_pipeline_started()
    await call.end(at=1)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]


async def test_simulation_is_a_track_run_without_recording_or_prompt(
    clock, http, recording, pushes
):
    context = LLMContext(messages=[{"role": "system", "content": "Draft prompt."}])
    call = await Call(
        clock, runner(), production=True, simulation=True, prompt="Draft {x}", prompt_version="v2"
    ).start()
    await call.push(LLMContextFrame(context=context), at=1)
    await call.end(at=2)
    body = ended(http)
    assert http.events("started")[0]["observability"] == {"mode": "track"}
    assert body["observability"] == {"mode": "track"}
    assert recording.recorders == [] and pushes == []
    assert not {"system_prompt", "prompt_ref"} & body["context"].keys()


async def test_prompt_sync_names_the_prompt_and_pushes_it_once_confirmed(clock, http, pushes):
    context = LLMContext(messages=[{"role": "system", "content": "You are Acme's agent."}])
    call = await Call(
        clock, runner(), production=True, record=False, prompt="You are {who}.", prompt_version=13
    ).start()
    await call.push(LLMContextFrame(context=context), at=1)
    await call.end(at=2)
    [push] = pushes
    assert (push["template"], push["version"], push["api_key"]) == (
        "You are {who}.",
        "13",
        "sk_test",
    )
    assert push["api_base_url"] == "https://api.superbryn.com"
    context_block = ended(http)["context"]
    assert context_block["prompt_ref"] == {
        "hash": "sha256:" + __import__("hashlib").sha256(b"You are {who}.").hexdigest(),
        "version": "13",
    }
    assert context_block["system_prompt"] == "You are Acme's agent."


async def test_no_prompt_push_for_a_refused_call(clock, http, pushes):
    http.replies = [(403, {"error": "forbidden"})]
    await Call(clock, runner(), production=True, record=False, prompt="P").start()
    assert pushes == []


async def test_custom_data_is_sent_as_given(clock, http, caplog):
    call = Call(
        clock,
        runner(),
        production=True,
        record=False,
        custom_data={"workflow_id": 4471, "tenant": "acme", "tags": ["a", "b"]},
    )
    await call.start()
    call.report.update_custom_data({"booked": True, "at": datetime(2027, 1, 15, tzinfo=UTC)})
    call.report.set_call_end_reason(" transfer_to_human ")
    await call.end(at=2)
    body = ended(http)
    assert body["custom_data"] == {
        "workflow_id": 4471,
        "tenant": "acme",
        "tags": ["a", "b"],
        "booked": True,
        "at": "2027-01-15 00:00:00+00:00",
    }
    assert body["ended_reason"] == "transfer_to_human"
    assert "custom_data" not in http.events("started")[0]


async def test_custom_data_that_cannot_be_json_is_left_out(clock, http, caplog):
    call = await Call(clock, superbryn_call_id="sb", custom_data={"score": math.nan}).start()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await call.end(at=1)
    assert "custom_data" not in ended(http)
    assert len(messages(caplog, "custom_data left out")) == 1


async def test_custom_data_that_is_not_a_dict_is_left_out(clock, http, caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        call = await Call(clock, superbryn_call_id="sb", custom_data=[1, 2]).start()
    await call.end(at=1)
    assert "custom_data" not in ended(http)
    assert len(messages(caplog, "set_custom_data takes a dict, got list")) == 1


@pytest.fixture
def app_logs_info():
    """The app's own logging config lets its INFO lines through, as an app that logs them has."""
    agent = logging.getLogger("agent")
    before = agent.level
    agent.setLevel(logging.INFO)
    yield
    agent.setLevel(before)


async def test_session_logs_go_with_a_production_call(clock, http, app_logs_info):
    call = await Call(clock, runner(), production=True, record=False).start()
    logging.getLogger("agent.tools").info("lookup_order called via logging")
    loguru_logger.info("lookup_order called via loguru")
    logging.getLogger("superbryn_pipecat_observer.call_report").info("our own line")
    logging.getLogger("agent").debug("below INFO")
    await call.end(at=2)
    logging.getLogger("agent").info("after the call")
    logs = ended(http)["diagnostics"]["logs"]
    assert [(r["level"], r["message"]) for r in logs] == [
        ("INFO", "lookup_order called via logging"),
        ("INFO", "lookup_order called via loguru"),
    ]
    assert logs[0]["name"] == "agent.tools" and logs[0]["ts"].endswith("+00:00")
    assert call.observer._logs._handler is None  # detached


async def test_capture_logs_false_sends_none(clock, http):
    call = await Call(clock, runner(), production=True, record=False, capture_logs=False).start()
    logging.getLogger("agent").info("not sent")
    await call.end(at=2)
    assert "logs" not in ended(http)["diagnostics"]


async def test_session_logs_are_this_calls_only_and_leave_the_log_levels_alone(
    clock, http, app_logs_info
):
    root_level = logging.getLogger().level
    call = await Call(clock, runner(), production=True, record=False).start()
    assert logging.getLogger().level == root_level

    async def another_call():
        logging.getLogger("agent").info("another caller's line")
        loguru_logger.info("another caller's loguru line")

    await asyncio.create_task(another_call(), context=contextvars.Context())
    logging.getLogger("agent").info("this caller's line " + "x" * 3000)
    await call.end(at=2)
    [line] = ended(http)["diagnostics"]["logs"]
    assert line["message"].startswith("this caller's line") and len(line["message"]) == 2000


async def test_session_logs_are_capped(clock, http, monkeypatch, app_logs_info):
    monkeypatch.setattr(cr, "_MAX_LOGS", 3)
    call = await Call(clock, runner(), production=True, record=False).start()
    for i in range(5):
        logging.getLogger("agent").info("line %d", i)
    await call.end(at=2)
    assert [r["message"] for r in ended(http)["diagnostics"]["logs"]] == [
        "line 0",
        "line 1",
        "line 2",
    ]


class RTVIProcessor(FrameProcessor):
    """Named as Pipecat's: it passes on a web client's disconnect-bot as an EndWorkerFrame."""


@pytest.mark.parametrize(
    ("signal", "want"),
    [
        ("tool", "agent"),
        ("rtvi", "caller"),
        ("app", None),
        ("hang-up", "caller"),
        ("fatal", "system"),
        ("none", None),
    ],
)
async def test_ended_by(clock, http, signal, want):
    call = await Call(clock, superbryn_call_id="sb").start()
    if signal == "tool":
        await call.push(EndWorkerFrame(), at=1, source=llm_service())
    elif signal == "rtvi":
        await call.push(EndWorkerFrame(), at=1, source=RTVIProcessor())
    elif signal == "app":  # the app queued it itself: no processor pushed it
        await call.push(EndWorkerFrame(), at=1, source=None)
    elif signal == "hang-up":
        call.observer._capture.caller_left(object(), object())
    elif signal == "fatal":
        await call.push(ErrorFrame("boom", fatal=True), at=1, source=llm_service())
    await call.end(at=2)
    assert ended(http).get("ended_by") == want


async def test_a_hang_up_after_the_end_frame_is_not_the_callers(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(EndFrame(), at=1)
    call.observer._capture.caller_left()  # the transport closing after the agent ended the call
    await call.end(at=2)
    assert "ended_by" not in ended(http)


async def test_caller_keypresses(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(InputDTMFFrame(button=KeypadEntry.ONE), at=1.5)
    await call.push(InputDTMFFrame(button=KeypadEntry.POUND), at=2)
    await call.end(at=3)
    assert ended(http)["diagnostics"]["dtmf"] == [
        {"at_ms": 1500, "digit": "1"},
        {"at_ms": 2000, "digit": "#"},
    ]


async def test_prompted_tool_calls_are_lifted_out_of_the_text(clock, http):
    llm, out = llm_service(), FakeOutput(TransportParams())
    call = await Call(clock, superbryn_call_id="sb").start()
    await user_turn(call, "my cart please", start=1, stop=2)
    await call.push(LLMFullResponseStartFrame(), at=3, source=llm)
    reply = 'Let me look. <tool_use>{"tool": "get_cart", "items": [1]}</tool_use> <tool_use>{bad</tool_use>'
    for part in (reply[:20], reply[20:]):  # streamed in pieces
        await call.push(LLMTextFrame(part), at=3.1, source=llm)
    await call.push(BotStartedSpeakingFrame(), at=3.2, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=4, source=out)
    await call.end(at=5)
    body = ended(http)
    assert body["transcript"][1]["text"] == "Let me look. <tool_use>{bad</tool_use>"
    assert body["tool_calls"] == [
        {
            "name": "get_cart",
            "arguments": {"tool": "get_cart", "items": [1]},
            "start_ms": 3000,
            "turn_id": "a1",
        }
    ]


async def test_daily_dial_in_names_the_direction_and_room(clock, http):
    settings = {
        "call_id": "daily-call",
        "call_domain": "d",
        "From": "+15550001111",
        "To": "+15550002222",
    }
    body = {"dialin_settings": settings}
    args = runner(body=body, room_url="https://acme.daily.co/room-1")
    call = await Call(clock, args, production=True, record=False).start()
    await call.end(at=1)
    started, done = http.events("started")[0], ended(http)
    assert started["observability"] == {"direction": "inbound", "mode": "observe"}
    assert (started["call_id"], started["from"], started["to"]) == (
        "sess-1",
        "+15550001111",
        "+15550002222",
    )
    assert done["context"]["room_url"] == "https://acme.daily.co/room-1"
    dialout = runner(body={"dialout_settings": [{"phoneNumber": "+15550003333"}]})
    await Call(clock, dialout, production=True, record=False).start()
    assert http.events("started")[-1]["observability"]["direction"] == "outbound"


async def test_environment_names_the_versions(clock, http):
    import platform

    from superbryn_pipecat_observer import __version__

    call = await Call(clock, superbryn_call_id="sb").start()
    await call.end(at=1)
    environment = ended(http)["context"]["environment"]
    assert environment["pipecat_observer_version"] == __version__
    assert environment["pipeline_version"] == f"@superbryn/pipecat-observer@{__version__}"
    assert environment["python_version"] == platform.python_version()
    assert environment["pipecat_version"].count(".") >= 2


def test_metrics_left_unset_are_switched_on():
    params = PipelineParams(enable_metrics=True)  # usage left unset: tokens would read 0
    task = FakeTask()
    task.params = params
    attach(task, api_key="k")
    assert params.enable_usage_metrics is True and params.enable_metrics is True


async def test_attach_twice_on_one_task_uses_the_first(clock, http, caplog):
    call = Call(clock, superbryn_call_id="sb")
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        attach(call.task, api_key="k", superbryn_call_id="other")
    assert len(call.task.observers) == 1
    assert len(messages(caplog, "attach already ran")) == 1


# ── the recorder in a real pipeline ──────────────────────────────────────


class PhoneTransport(BaseTransport):
    """A transport with the hang-up event network transports have; input and output keep it as theirs."""

    def __init__(self) -> None:
        super().__init__()
        self._register_event_handler("on_client_disconnected")
        self.inp = BaseInputTransport(TransportParams())
        self.out = PassThroughOutput(TransportParams(), enable_direct_mode=True)
        self.inp._transport = self.out._transport = self

    def input(self):
        return self.inp

    def output(self):
        return self.out


def pcm(seconds: float, rate: int = 16000) -> bytes:
    samples = (np.sin(np.arange(int(seconds * rate)) * 0.2) * 8000).astype(np.int16)
    return samples.tobytes()


async def test_the_recorder_records_both_sides_in_a_real_pipeline(http, monkeypatch, tmp_path):
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask
    from pipecat.processors.audio.audio_buffer_processor import AudioBufferProcessor

    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 60)
    uploads: dict[str, bytes] = {}

    async def presign(base_url, api_key, session_id, num_channels):
        return {
            "upload_url": "https://s3.test/put",
            "public_url": f"https://s3.test/{session_id}.wav",
        }

    async def upload(url, path):
        uploads[url] = path.read_bytes()
        return True

    monkeypatch.setattr(cr, "fetch_recording_upload_url", presign)
    monkeypatch.setattr(cr, "upload_wav_via_presigned", upload)
    transport = PhoneTransport()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(
            Pipeline([Forward(), transport.output()]),
            params=PipelineParams(audio_in_sample_rate=16000, audio_out_sample_rate=16000),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
    report = attach(task, SimpleNamespace(session_id="sess-real"), api_key="k")
    inner = task.pipeline.processors[1]
    names = [type(p).__name__ for p in inner.processors]
    assert names[-3:] == ["AudioBufferProcessor", "PassThroughOutput", "PipelineSink"]
    assert isinstance(inner.processors[-3], AudioBufferProcessor)
    handlers = transport._event_handlers["on_client_disconnected"].handlers
    assert len(handlers) == 1  # the hang-up hook
    assert report is not None

    async def script():
        await asyncio.sleep(0.1)
        await task.queue_frame(
            InputAudioRawFrame(audio=pcm(0.5), sample_rate=16000, num_channels=1)
        )
        await asyncio.sleep(0.05)
        await task.queue_frame(
            OutputAudioRawFrame(audio=pcm(0.4), sample_rate=16000, num_channels=1)
        )
        await asyncio.sleep(0.05)
        await transport._call_event_handler("on_client_disconnected", None)
        await asyncio.sleep(0.05)
        await task.queue_frame(EndFrame())

    await run(task, script)
    body = ended(http)
    assert (body["surface"], body["call_id"], body["ended_by"]) == (
        "observability",
        "sess-real",
        "caller",
    )
    assert body["observability"]["stereo_recording_url"] == "https://s3.test/sess-real.wav"
    assert body["context"]["environment"]["transport"] == "phone"
    [wav_bytes] = uploads.values()
    with wave.open(__import__("io").BytesIO(wav_bytes)) as w:
        frames = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16).reshape(-1, 2)
        assert (w.getnchannels(), w.getframerate()) == (2, 16000)
    left, right = (np.abs(frames[:, i]).max() for i in (0, 1))
    assert left > 1000 and right > 1000  # the caller left, the agent right
    assert not list(tmp_path.iterdir())  # nothing kept here; the recorder's temp file is gone too


class RoomTransport(PhoneTransport):
    """A transport like LiveKit's: no on_client_disconnected, only on_participant_left."""

    def __init__(self) -> None:
        super().__init__()
        del self._event_handlers["on_client_disconnected"]
        self._register_event_handler("on_participant_left")


async def test_a_participant_leaving_a_room_is_the_caller_hanging_up(http):
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask

    transport = RoomTransport()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(
            Pipeline([Forward(), transport.output()]),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
    attach(task, SimpleNamespace(session_id="sess-room"), api_key="k", record=False)
    assert len(transport._event_handlers["on_participant_left"].handlers) == 1

    async def script():
        await asyncio.sleep(0.05)
        await transport._call_event_handler("on_participant_left", "PA_caller", "disconnected")
        await asyncio.sleep(0.05)
        await task.queue_frame(EndFrame())

    await run(task, script)
    assert ended(http)["ended_by"] == "caller"


async def test_no_recorder_once_the_pipeline_runs(http, monkeypatch, caplog):
    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 60)
    task = real_task(attached=False)

    async def script():
        await asyncio.sleep(0.2)
        with caplog.at_level(logging.WARNING, logger=LOGGER):
            attach(task, SimpleNamespace(session_id="late"), api_key="k")
        await task.queue_frame(EndFrame())

    await run(task, script)
    assert "AudioBufferProcessor" not in [
        type(p).__name__ for p in task.pipeline.processors[1].processors
    ]
    assert len(messages(caplog, "calls are not recorded")) == 1
    assert ended(http)["observability"] == {"mode": "observe"}


# ── final-review fixes ───────────────────────────────────────────────────


async def test_preparing_the_recording_shares_the_budget_and_call_ended_still_goes(
    clock, http, monkeypatch, recording
):
    monkeypatch.setattr(cr, "_FINISH_BUDGET_S", 0.2)

    async def stuck(*args):
        await asyncio.sleep(3600)  # SuperBryn's API takes the request and never answers

    monkeypatch.setattr(cr, "fetch_recording_upload_url", stuck)
    call = await Call(clock, runner(), production=True).start()
    began = time.monotonic()
    await call.end(at=2)
    assert time.monotonic() - began < 1
    assert "stereo_recording_url" not in ended(http)["observability"]


class ClientOnlyTransport(BaseTransport):
    """A transport like SmallWebRTC's: its input and output keep only the client, whose callbacks are the
    transport's own methods."""

    def __init__(self) -> None:
        super().__init__()
        self._register_event_handler("on_client_disconnected")
        client = SimpleNamespace(
            _callbacks=SimpleNamespace(on_client_disconnected=self._disconnected)
        )
        self.inp = BaseInputTransport(TransportParams())
        self.out = PassThroughOutput(TransportParams(), enable_direct_mode=True)
        self.inp._client = self.out._client = client

    async def _disconnected(self, *args: Any) -> None:
        await self._call_event_handler("on_client_disconnected", *args)

    def input(self):
        return self.inp

    def output(self):
        return self.out


async def test_a_hang_up_is_heard_on_a_transport_reached_through_its_client(http):
    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineTask

    transport = ClientOnlyTransport()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(
            Pipeline([Forward(), transport.output()]),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
    attach(task, SimpleNamespace(session_id="sess-webrtc"), api_key="k", record=False)
    assert len(transport._event_handlers["on_client_disconnected"].handlers) == 1

    async def script():
        await asyncio.sleep(0.05)
        await transport._disconnected(None)
        await asyncio.sleep(0.05)
        await task.queue_frame(EndFrame())

    await run(task, script)
    assert ended(http)["ended_by"] == "caller"


def test_the_handle_never_raises_on_values_it_cannot_take(caplog):
    report = cr.CallReport()
    report.set_custom_data({"team_id": "t-42"})
    report.set_custom_data(None)
    report.update_custom_data(["not", "a", "dict"])
    report.set_call_end_reason(5)
    report.set_recording_url(5)
    assert report._custom_data == {"team_id": "t-42"} and report._external_recording is None
    assert messages(caplog, "update_custom_data takes a dict, got list")
    report.set_call_end_reason("x" * 300)
    assert len(report._end_reason) == 200


async def test_the_recorders_cleanup_closes_a_file_it_never_uploaded():
    from superbryn_pipecat_observer.audio_recorder import SuperbrynAudioRecorder

    recorder = SuperbrynAudioRecorder()
    await recorder.start()
    await recorder._write_chunk(b"\x00\x01" * 160, 16000, 1)
    path = recorder._tmp_path
    assert recorder._wave_writer is not None and path.exists()
    recorder.cleanup()
    assert recorder._wave_writer is None and not path.exists()


# ── second review ────────────────────────────────────────────────────────


async def test_a_call_that_does_not_capture_logs_clears_the_capture_it_inherits(
    clock, http, app_logs_info
):
    a = await Call(clock, runner(), production=True, record=False).start()
    Call(
        clock, runner(), production=True, record=False, capture_logs=False
    )  # attached in the same context
    logging.getLogger("agent").info("the second call's line")
    assert a.observer._logs.records == []


def test_with_no_logging_config_warnings_still_reach_stderr_while_a_call_captures(
    monkeypatch, capsys
):
    monkeypatch.setattr(logging.getLogger(), "handlers", [])
    capture = cr._LogCapture()
    capture.start()
    try:
        logging.getLogger("httpx").warning("still printed")
        logging.getLogger("httpx").info("below the last resort")
    finally:
        capture.stop()
    err = capsys.readouterr().err
    assert "still printed" in err and "below the last resort" not in err


def test_custom_data_takes_any_mapping_and_attach_never_raises_on_what_it_is_given(caplog):
    from collections import UserDict
    from types import MappingProxyType

    report = cr.CallReport()
    report.set_custom_data(MappingProxyType({"team_id": "t-42"}))
    report.update_custom_data(UserDict({"enterprise_id": "e-1"}))
    assert report._custom_data == {"team_id": "t-42", "enterprise_id": "e-1"}

    class Ambiguous:  # like a numpy array: asking whether it is empty raises
        def __bool__(self):
            raise ValueError("truth value is ambiguous")

    handle = attach(FakeTask(), runner(), api_key="k", custom_data=Ambiguous())
    assert handle._custom_data == {}
    assert messages(caplog, "set_custom_data takes a dict, got Ambiguous")
    handle.set_call_end_reason(5)
    assert messages(caplog, "set_call_end_reason takes text, got int")


async def test_a_slow_prompt_push_never_holds_call_ended(clock, http, monkeypatch):
    monkeypatch.setattr(cr, "_FINISH_BUDGET_S", 0.3)

    async def slow_push(**kwargs):
        await asyncio.sleep(3600)

    monkeypatch.setattr(cr, "push_prompt", slow_push)
    call = Call(clock, superbryn_call_id="sb", prompt="You are Ava.")
    await call.observer.on_pipeline_started()
    began = time.monotonic()
    await call.end(at=2)
    assert time.monotonic() - began < 1
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]
