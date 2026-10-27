"""Eval mode (test_data.py) driven with real Pipecat frames, metrics and service base classes."""

from __future__ import annotations

import asyncio
import json
import logging
import math
import time
import warnings
from dataclasses import dataclass
from datetime import UTC, datetime
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.turn.smart_turn.base_smart_turn import SmartTurnParams
from pipecat.audio.vad.vad_analyzer import VADParams
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    CancelFrame,
    EndFrame,
    ErrorFrame,
    FunctionCallCancelFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    FunctionCallResultProperties,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
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
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import TransportParams
from pipecat.turns.user_start import MinWordsUserTurnStartStrategy, VADUserTurnStartStrategy
from pydantic import BaseModel, ConfigDict, Field

import superbryn_pipecat_observer.test_data as td
from superbryn_pipecat_observer import attach_test_data

T0 = 1_800_000_000.0  # 2027-01-15T08:00:00Z
AGENT_ID = "4347a69f-0c1e-4b6a-9f1e-2b3c4d5e6f70"
UP = FrameDirection.UPSTREAM
LOGGER = "superbryn_pipecat_observer.test_data"


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
    """The three things attach_test_data uses on a PipelineTask."""

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
        attach_test_data(self.task, runner_args, **kwargs)
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
def _isolated(monkeypatch):
    for name in (
        "SUPERBRYN_TEST_API_KEY",
        "SUPERBRYN_API_KEY",
        "SUPERBRYN_AGENT_ID",
        "SUPERBRYN_BASE_URL",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(td, "_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(td, "_DRAIN_TIMEOUT_S", 0.05)
    td._caller_numbers_cache.clear()
    td._warned.clear()


@pytest.fixture(autouse=True)
def http(monkeypatch) -> FakeHTTP:
    fake = FakeHTTP()
    monkeypatch.setattr(td, "_http_request", fake)
    return fake


@pytest.fixture
def clock(monkeypatch) -> Clock:
    c = Clock()
    monkeypatch.setattr(td, "_now", lambda: c.t)
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
    assert http.events("started")[0]["superbryn_call_id"] == "sb-1"


@pytest.mark.parametrize(
    "body",
    [{"superbryn_call_id": " sb-body "}, BodyModel(superbryn_call_id="sb-body")],
    ids=["dict", "pydantic"],
)
async def test_test_id_from_start_body(clock, http, body):
    await Call(clock, SimpleNamespace(body=body)).start()
    assert http.events("started")[0]["superbryn_call_id"] == "sb-body"


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
    assert started["superbryn_call_id"] == "sb-sip"
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
    assert (started["superbryn_call_id"], started["call_id"]) == ("sb-tw", "CA1")


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
    assert (started["superbryn_call_id"], started["call_id"]) == ("sbc_123", "CA1")
    assert (started["from"], started["to"]) == ("+14155550100", "+16505550199")


async def test_call_data_argument_comes_first(clock, http):
    handshake = {"call_id": "CA2", "body": {"superbryn_call_id": "sb-arg"}}
    args = SimpleNamespace(call_data=CallDataModel(call_id="CA1"))
    await Call(clock, args, call_data=handshake).start()
    started = http.events("started")[0]
    assert (started["superbryn_call_id"], started["call_id"]) == ("sb-arg", "CA2")


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
        "SUPERBRYN_TEST_DATA: ignored unfilled template placeholders "
        "{'superbryn_call_id': '{SipHeader_X-SuperBryn-Call-Id}', 'from': '{From}'}"
    ]


async def test_caller_number_fallback_matches_and_caches(clock, http):
    http.numbers = ["+1 (415) 555-0100", "+919876543210"]
    args = SimpleNamespace(call_data=CallDataModel(**{"from": "14155550100", "to": "+15550001"}))
    await Call(clock, args).start()
    started = http.events("started")[0]
    assert "superbryn_call_id" not in started and started["from"] == "14155550100"
    # last 10 digits match; the numbers come from the per-base-URL cache
    await Call(clock, SimpleNamespace(call_data=CallDataModel(**{"from": "09876543210"}))).start()
    assert len(http.events("started")) == 2
    [get] = http.gets()
    assert get["url"] == "https://api.superbryn.com/public-api/v1/test-calls/caller-numbers"
    assert get["headers"] == {"X-API-Key": "sk_test"}
    expiry, _ = td._caller_numbers_cache["https://api.superbryn.com"]
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


async def test_caller_numbers_failure_is_remembered_for_a_minute(clock, http):
    http.replies = [(500, None)] * 3
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 3 and http.events() == []
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 3
    base = "https://api.superbryn.com"
    expiry, numbers = td._caller_numbers_cache[base]
    assert numbers is None and 50 < expiry - time.monotonic() <= 60
    td._caller_numbers_cache[base] = (time.monotonic() - 1, None)  # the minute is over
    await Call(clock, from_number="+14155550100").start()
    assert len(http.gets()) == 4 and len(http.events("started")) == 1


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
        "source": "pipecat",
        "superbryn_call_id": "sb-1",
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
    request = http.requests[-1]
    assert request["url"] == "https://staging.example.com/public-api/v1/test-calls/agent-data"
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
            "turnLatency": 1100,
            "endpointingLatency": 240,
            "transcriberLatency": 300,
            "modelLatency": 450,
            "voiceLatency": 120,
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
    assert [e.get("transcriberLatency") for e in latencies] == [1900, None]


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
    assert entry["modelLatency"] == 600 and "voiceLatency" not in entry
    assert entry["turnLatency"] == 2000


async def test_user_resuming_resets_the_cycle(clock, http):
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(vad_start(1.2), at=1.2)
    await call.push(vad_stop(2.8), at=2.8)
    await call.push(ttfb(0.3), at=2.9, source=llm_service())
    await user_turn(call, "long sentence", start=3, stop=4)  # the user goes on: silence at 4.0
    await agent_turn(call, "Sure.", start=5, stop=6)
    await call.end(at=7)
    [entry] = ended(http)["diagnostics"]["turn_latencies"]
    assert entry == {"turn_id": "a1", "turnLatency": 1000, "endpointingLatency": 800}


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
    assert [e["endpointingLatency"] for e in latencies] == [200, 255]


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
    assert [(e["endpointingLatency"], e["turnLatency"]) for e in latencies] == [
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
    assert [e["endpointingLatency"] for e in diagnostics["turn_latencies"]] == [
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
        {"turnLatency": 800, "endpointingLatency": 700, "turn_id": "a1"}
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
    assert ended(http)["diagnostics"]["turn_latencies"] == [{"turnLatency": 150, "turn_id": "a1"}]


async def test_vad_times_the_turn_when_both_vad_and_audio_are_seen(clock, http):
    out = FakeOutput(TransportParams())
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.push(vad_start(1.2), at=1.2)
    await caller_audio(call, start=1.2, speech=1.0, silence=0.5)  # tone to 2.2
    await call.push(vad_stop(3.0, stop_secs=0.5), at=3.0)
    await agent_turn(call, "Sure.", start=3.2, stop=4.0, out=out)
    await call.end(at=5)
    [entry] = ended(http)["diagnostics"]["turn_latencies"]
    assert entry["turnLatency"] == 700  # from VAD's stop (3.0 - 0.5), not the audio's end


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

    monkeypatch.setattr(td, "STTUsageMetricsData", STTUsage)
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
        assert len(body["call_id"]) == 200 and len(body["superbryn_call_id"]) == 200
        assert len(body["event_id"]) <= 200 and body["event_id"].startswith("pc:ccc")
    [tool] = ended(http)["tool_calls"]
    assert len(tool["id"]) == 200 and len(tool["name"]) == 200


async def test_turn_latency_and_tool_caps(clock, http, monkeypatch):
    monkeypatch.setattr(td, "_MAX_TURNS", 3)
    monkeypatch.setattr(td, "_MAX_TOOL_CALLS", 2)
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

    monkeypatch.setattr(td._Capture, "_turn_latencies", broken)
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
        "SUPERBRYN_TEST_DATA_ERROR: turn_latencies left out of call.ended: boom"
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
    http.replies = [(400, {"error": "invalid_body"}), (401, {"error": "unauthorized"})]
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        call = await Call(clock, superbryn_call_id="sb").start()
        await call.end(at=1)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]
    rejected = messages(caplog, "REJECTED")
    assert len(rejected) == 2 and "test_data:write" not in rejected[0]
    assert "needs an org API key with test_data:write" in rejected[1]


async def test_gives_up_after_three_attempts_and_still_sends_ended(clock, http):
    http.replies = [ConnectionError("down")] * 3
    call = await Call(clock, superbryn_call_id="sb").start()
    await call.end(at=1)
    assert [e["event"] for e in http.events()] == ["call.started"] * 3 + ["call.ended"]


async def test_timeouts_per_event(clock, http, monkeypatch):
    timeouts: list[tuple[str, float]] = []
    real_send = td._send

    async def spy(method, url, api_key, body, *, timeout):
        timeouts.append((body["event"] if body else method, timeout))
        return await real_send(method, url, api_key, body, timeout=timeout)

    monkeypatch.setattr(td, "_send", spy)
    call = await Call(clock, from_number="+14155550100").start()
    await call.end(at=1)
    assert timeouts == [("GET", 5.0), ("call.started", 5.0), ("call.ended", 10.0)]


async def test_a_slow_attempt_times_out_and_is_retried(clock, http, monkeypatch):
    monkeypatch.setattr(td, "_STARTED_TIMEOUT_S", 0.01)
    attempts = []

    async def slow(method, url, headers, data):
        attempts.append(method)
        await asyncio.sleep(1)

    monkeypatch.setattr(td, "_http_request", slow)
    await Call(clock, superbryn_call_id="sb").start()
    assert attempts == ["POST"] * 3


async def test_a_call_ending_during_the_number_lookup_sends_nothing_at_once(
    clock, http, monkeypatch
):
    async def hang(method, url, headers, data):
        http.requests.append({"method": method})
        await asyncio.sleep(3600)

    monkeypatch.setattr(td, "_http_request", hang)
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

    monkeypatch.setattr(td, "_http_request", slow_started)
    call = Call(clock, from_number="+14155550100")
    await call.observer.on_pipeline_started()
    await asyncio.sleep(0.05)  # recognised by its number; call.started is on its way
    await call.end(at=2)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]


@pytest.mark.parametrize("hangs", ["call.started", "call.ended"])
async def test_the_finish_handler_gives_up_after_one_budget(
    clock, http, monkeypatch, caplog, hangs
):
    monkeypatch.setattr(td, "_FINISH_BUDGET_S", 0.2)

    async def answer(method, url, headers, data):
        http.requests.append({"method": method, "body": json.loads(data)})
        if http.requests[-1]["body"]["event"] == hangs:
            await asyncio.sleep(3600)
        return 200, {"status": "linked", "is_test": True}

    monkeypatch.setattr(td, "_http_request", answer)
    call = Call(clock, superbryn_call_id="sb")
    await call.observer.on_pipeline_started()
    began = time.monotonic()
    with caplog.at_level(logging.ERROR, logger=LOGGER):
        await call.end(at=2)
    assert 0.2 <= time.monotonic() - began < 1
    assert messages(caplog, "not sent within") == [
        "SUPERBRYN_TEST_DATA_FAILED: call.ended not sent within 0.2s of the call's end"
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
    monkeypatch.setenv("SUPERBRYN_TEST_API_KEY", "sk_test_data")
    monkeypatch.setenv("SUPERBRYN_BASE_URL", "http://localhost:3000/")
    monkeypatch.setenv("SUPERBRYN_AGENT_ID", AGENT_ID)
    await Call(clock, api_key=None, agent_id=None, superbryn_call_id="sb").start()
    request = http.requests[0]
    assert request["headers"]["X-API-Key"] == "sk_test_data"
    assert request["url"] == "http://localhost:3000/public-api/v1/test-calls/agent-data"
    assert request["body"]["agent_id"] == AGENT_ID


async def test_api_key_falls_back_to_superbryn_api_key(clock, http, monkeypatch):
    monkeypatch.setenv("SUPERBRYN_API_KEY", "sk_general")
    await Call(clock, api_key=None, superbryn_call_id="sb").start()
    assert http.requests[0]["headers"]["X-API-Key"] == "sk_general"


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
async def test_no_valid_agent_id_sends_nothing(clock, http, caplog, agent_id):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for _ in range(2):
            assert Call(clock, agent_id=agent_id, superbryn_call_id="sb").observer is None
    assert http.requests == []
    assert len(messages(caplog, "SUPERBRYN_TEST_DATA_DISABLED")) == 1
    reason = (
        f"{agent_id!r} is not a SuperBryn agent ID"
        if agent_id
        else "no agent ID, set SUPERBRYN_AGENT_ID"
    )
    assert len(messages(caplog, reason)) == 1


async def test_metrics_off_warns_once(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        for metrics_on in (False, False, True):
            attach_test_data(FakeTask(metrics=metrics_on), api_key="k", agent_id=AGENT_ID)
    [message] = messages(caplog, "PipelineParams")
    assert "enable_metrics and enable_usage_metrics" in message


async def test_attach_never_raises(caplog):
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        attach_test_data(
            object(), SimpleNamespace(body=12, call_data="x"), api_key="k", agent_id=AGENT_ID
        )
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
        td, "_DRAIN_TIMEOUT_S", 60
    )  # only the EndFrame reaching the sink ends the wait
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(
            Pipeline([PassThroughOutput(TransportParams(), enable_direct_mode=True)]),
            params=PipelineParams(enable_metrics=True, enable_usage_metrics=True),
            cancel_on_idle_timeout=False,
            enable_rtvi=False,
        )
    attach_test_data(
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
    assert (body["superbryn_call_id"], body["ended_reason"]) == ("sb-real", "ended")
    assert texts(body) == [("u1", "user", "hello bot"), ("a1", "agent", "Hello human.")]
    [entry] = body["diagnostics"]["turn_latencies"]
    assert entry["turn_id"] == "a1" and entry["turnLatency"] >= 0


def real_task(*processors: FrameProcessor, attach: bool = True) -> Any:
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
    if attach:
        args = SimpleNamespace(body={"superbryn_call_id": "sb-real"})
        attach_test_data(task, args, api_key="k", agent_id=AGENT_ID)
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
    monkeypatch.setattr(td, "_DRAIN_TIMEOUT_S", 60)
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

    monkeypatch.setattr(td, "_DRAIN_TIMEOUT_S", 60)
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
    assert 500 <= entry["endpointingLatency"] <= entry["turnLatency"]
    detection = diagnostics["configuration"]["turn_detection"]
    # the speech-timeout strategy starts the LLM only as the turn ends: no early drafts
    assert detection == {
        "type": "vad",
        "min_end_of_turn_ms": entry["endpointingLatency"],
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

    monkeypatch.setattr(td, "_DRAIN_TIMEOUT_S", 60)
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
    assert (cls.__module__, cls.__qualname__) in td._OUTPUT_STAMPED_LLMS
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

    monkeypatch.setattr(td, "_DRAIN_TIMEOUT_S", 60)
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
    monkeypatch.setattr(td, "_DRAIN_TIMEOUT_S", 60)
    task = real_task(attach=False)

    def attach():
        args = SimpleNamespace(body={"superbryn_call_id": "sb-real"})
        attach_test_data(task, args, api_key="k", agent_id=AGENT_ID)

    async def script():
        await asyncio.sleep(0.2)
        if late:
            attach()  # e.g. in on_client_connected, once the pipeline is running
        for frame in (BotStartedSpeakingFrame(), tts_text("Hello."), BotStoppedSpeakingFrame()):
            await task.queue_frame(frame)
            await asyncio.sleep(0.05)
        await task.queue_frame(EndFrame())

    if not late:
        attach()
    with caplog.at_level(logging.WARNING, logger=LOGGER):
        await run(task, script)
    assert [e["event"] for e in http.events()] == ["call.started", "call.ended"]
    body = ended(http)
    [turn] = body["transcript"]
    assert body["started_at"] < body["ended_at"] and 0 < turn["end_ms"] < 1000
    assert len(messages(caplog, "after the task started")) == int(late)
