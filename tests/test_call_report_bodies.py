"""Whole calls through attach(): one scripted Pipecat call reported as a SuperBryn test call and as a production
call, on one fake clock, with no network. Set SUPERBRYN_PIPECAT_DUMP_DIR to also write each body to
<dir>/<surface>_<event>.json."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
from pipecat.adapters.schemas.function_schema import FunctionSchema
from pipecat.adapters.schemas.tools_schema import ToolsSchema
from pipecat.audio.dtmf.types import KeypadEntry
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    EndWorkerFrame,
    ErrorFrame,
    InputDTMFFrame,
    LLMContextFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.processors.aggregators.llm_context import LLMContext
from pipecat.transports.base_transport import TransportParams

import superbryn_pipecat_observer.call_report as cr
from tests.test_call_report import (
    AGENT_ID,
    Call,
    CallDataModel,
    Clock,
    FakeHTTP,
    FakeOutput,
    FakeRecorder,
    llm_service,
    metrics,
    stt_service,
    tool_result,
    tool_start,
    transcription,
    ttfb,
    tts_service,
    tts_text,
    vad_start,
    vad_stop,
)

PROMPT = "You are {agent_name}, a support agent for Acme."
SYSTEM_PROMPT = "You are Ava, a support agent for Acme."
CALLER, AGENT_LINE = "+14155550123", "+16505550199"
CUSTOM_DATA = {"workflow_id": "wf_inbound_support", "tenant_id": 7, "agent_name": "support-bot"}
# Every top-level key the server's call report reads; anything else would be dropped.
TOP_LEVEL = {
    "schema_version",
    "event_id",
    "event",
    "surface",
    "source",
    "call_id",
    "agent_id",
    "from",
    "to",
    "started_at",
    "ended_at",
    "ended_reason",
    "ended_by",
    "transcript",
    "tool_calls",
    "diagnostics",
    "context",
    "custom_data",
    "testing",
    "observability",
}
# 0.11's call.completed body and package-side numbers: none may come back anywhere in a body.
REMOVED = {
    "call",
    "metadata",
    "sdk_version",
    "speech_stats",
    "latency",
    "duration_seconds",
    "call_end_reason",
    "mono_audio",
    "turnLatency",
    "endpointingLatency",
    "transcriberLatency",
    "modelLatency",
    "voiceLatency",
    "speaker",
    "start_time_ms",
    "end_time_ms",
    "latency_ms",
    "function_name",
    "tool_call_id",
    "timestamp_ms",
    "avg_ms",
    "p95_ms",
}


@pytest.fixture(autouse=True)
def offline(monkeypatch, tmp_path) -> SimpleNamespace:
    """The API, the prompt push and the recording's upload, stubbed."""
    for name in ("SUPERBRYN_API_KEY", "SUPERBRYN_AGENT_ID", "SUPERBRYN_BASE_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cr, "_BACKOFF_S", (0.0, 0.0))
    monkeypatch.setattr(cr, "_DRAIN_TIMEOUT_S", 0.05)
    monkeypatch.setattr("tempfile.tempdir", str(tmp_path))  # this test's caller-number file only
    cr._caller_numbers_cache.clear()
    cr._warned.clear()
    cr._attached_tasks.clear()
    world = SimpleNamespace(http=FakeHTTP(), pushes=[], recorders=[])
    world.http.numbers = ["+18005550100"]
    clock = Clock()
    monkeypatch.setattr(cr, "_now", lambda: clock.t)
    monkeypatch.setattr(cr, "_http_request", world.http)

    async def push(**kwargs: Any) -> bool:
        world.pushes.append(kwargs)
        return True

    def insert(_pipeline: Any) -> FakeRecorder:
        (tmp_path / "call.wav").write_bytes(b"RIFF")
        world.recorders.append(FakeRecorder(tmp_path / "call.wav"))
        return world.recorders[-1]

    async def presign(base_url: str, api_key: str, session_id: str, num_channels: int) -> dict:
        key = f"observability/pipecat/proj_42/{session_id}-{num_channels}ch.wav"
        return {
            "upload_url": f"https://bucket.s3.test/{key}?X-Amz-Signature=x",
            "public_url": f"https://bucket.s3.test/{key}",
        }

    async def upload(_url: str, _path: Path) -> bool:
        return True

    monkeypatch.setattr(cr, "push_prompt", push)
    monkeypatch.setattr(cr, "_insert_recorder", insert)
    # the fake task has no pipeline to read it from; a Twilio bot runs FastAPIWebsocketTransport
    monkeypatch.setattr(cr, "_transport_label", lambda _pipeline: "fastapiwebsocket")
    monkeypatch.setattr(cr, "fetch_recording_upload_url", presign)
    monkeypatch.setattr(cr, "upload_wav_via_presigned", upload)
    world.clock = clock
    return world


async def script(call: Call) -> None:
    """A support call: the greeting, an order lookup through a tool, a keypress, a goodbye, and the agent hanging
    up through an end-call tool. The STT drops once and comes back."""
    stt, llm, tts, out = stt_service(), llm_service(), tts_service(), FakeOutput(TransportParams())
    tools = ToolsSchema(
        standard_tools=[
            FunctionSchema(
                name="lookup_order",
                description="Look up an order by its number.",
                properties={"order_id": {"type": "string"}},
                required=["order_id"],
            )
        ]
    )
    context = LLMContext(messages=[{"role": "system", "content": SYSTEM_PROMPT}], tools=tools)
    usage = LLMTokenUsage(
        prompt_tokens=412, completion_tokens=18, total_tokens=430, cache_read_input_tokens=256
    )

    await call.push(LLMContextFrame(context=context), at=0.4)
    await call.push(LLMFullResponseStartFrame(), at=0.5, source=llm)
    await call.push(LLMTextFrame("Hello, thanks for calling Acme."), at=0.9, source=llm)
    await call.push(metrics(LLMUsageMetricsData(processor="llm", value=usage)), at=1.0, source=llm)
    await call.push(BotStartedSpeakingFrame(), at=1.2, source=out)
    await call.push(tts_text("Hello, thanks for calling Acme."), at=1.3, source=out)
    await call.push(metrics(TTSUsageMetricsData(processor="tts", value=31)), at=1.3, source=tts)
    await call.push(BotStoppedSpeakingFrame(), at=3.0, source=out)

    tools_log = logging.getLogger("agent.tools")
    tools_log.setLevel(logging.INFO)  # the app's own config lets its INFO lines through
    tools_log.info("caller asked about an order")
    tools_log.setLevel(logging.NOTSET)
    await call.push(vad_start(4.0), at=4.0)
    await call.push(vad_stop(6.0), at=6.0)
    await call.push(ttfb(0.21), at=6.1, source=stt)
    await call.push(transcription("I need help with order 884291."), at=6.2, source=stt)
    await call.push(LLMContextFrame(context=context), at=6.3)
    await call.push(LLMFullResponseStartFrame(), at=6.3, source=llm)
    await call.push(ttfb(0.38), at=6.7, source=llm)
    await call.push(
        tool_start("call_1", "lookup_order", {"order_id": "884291"}), at=6.8, source=llm
    )
    await call.push(
        tool_result("call_1", {"status": "shipped", "eta": "Friday"}, "lookup_order"), at=7.1
    )
    await call.push(LLMFullResponseStartFrame(), at=7.2, source=llm)
    await call.push(LLMTextFrame("Your order 884291 has shipped."), at=7.5, source=llm)
    await call.push(metrics(LLMUsageMetricsData(processor="llm", value=usage)), at=7.6, source=llm)
    await call.push(ttfb(0.19), at=7.7, source=tts)
    await call.push(BotStartedSpeakingFrame(), at=7.9, source=out)
    await call.push(tts_text("Your order 884291 has shipped."), at=8.0, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=9.6, source=out)

    await call.push(ErrorFrame("Deepgram connection closed", fatal=False), at=10.0, source=stt)
    await call.push(InputDTMFFrame(button=KeypadEntry.ONE), at=10.5)
    await call.push(vad_start(11.0), at=11.0)
    await call.push(vad_stop(12.0), at=12.0)
    await call.push(transcription("Great, thanks. Bye."), at=12.2, source=stt)
    await call.push(LLMFullResponseStartFrame(), at=12.3, source=llm)
    await call.push(LLMTextFrame("You're welcome, goodbye!"), at=12.6, source=llm)
    await call.push(BotStartedSpeakingFrame(), at=12.9, source=out)
    await call.push(tts_text("You're welcome, goodbye!"), at=13.0, source=out)
    await call.push(BotStoppedSpeakingFrame(), at=14.2, source=out)
    await call.push(EndWorkerFrame(), at=14.4, source=llm)  # the end-call tool hangs up


async def run_call(world: SimpleNamespace, surface: str) -> tuple[dict, dict]:
    """One whole call: (call.started, call.ended); each is written to SUPERBRYN_PIPECAT_DUMP_DIR when set."""
    caller = "+18005550100" if surface == "testing" else CALLER  # SuperBryn's number calls a test
    twilio = CallDataModel(
        call_id="CA9f8e7d6c5b4a39281706f5e4d3c2b1a0",
        **{"from": caller, "to": AGENT_LINE},
    )
    runner_args = SimpleNamespace(
        session_id="5b1f9c2e-8d47-4a6b-9e3f-1c2d3e4f5a6b", call_data=twilio
    )
    call = Call(
        world.clock,
        runner_args,
        api_key="sbryn_dummy",
        agent_id=AGENT_ID,
        base_url="https://api.superbryn.test",
        production=True,
        prompt=PROMPT,
        prompt_version="v3",
        custom_data=CUSTOM_DATA,
    )
    await call.start()
    call.report.update_custom_data({"lead_source": "website"})
    await script(call)
    await call.end(EndFrame(), at=14.6)

    posts = [r["body"] for r in world.http.requests if r["method"] == "POST"]
    assert [b["event"] for b in posts] == ["call.started", "call.ended"]
    out = os.getenv("SUPERBRYN_PIPECAT_DUMP_DIR")
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        for body in posts:
            name = f"{surface}_{body['event'].split('.')[1]}.json"
            (Path(out) / name).write_text(json.dumps(body, indent=2, ensure_ascii=False) + "\n")
    return posts[0], posts[1]


def keys_anywhere(value: Any):
    if isinstance(value, dict):
        for key, item in value.items():
            yield key
            if key != "custom_data":
                yield from keys_anywhere(item)
    elif isinstance(value, list):
        for item in value:
            yield from keys_anywhere(item)


@pytest.mark.parametrize("surface", ["testing", "observability"])
async def test_whole_call_bodies(offline, surface):
    started, ended = await run_call(offline, surface)
    call_id = (
        "CA9f8e7d6c5b4a39281706f5e4d3c2b1a0"
        if surface == "testing"
        else "5b1f9c2e-8d47-4a6b-9e3f-1c2d3e4f5a6b"
    )

    for body in (started, ended):
        assert set(body) <= TOP_LEVEL
        assert REMOVED.isdisjoint(keys_anywhere(body))
        assert (body["surface"], body["source"], body["schema_version"]) == (surface, "pipecat", 1)
        assert (body["call_id"], body["agent_id"]) == (call_id, AGENT_ID)
        assert body["started_at"] == "2027-01-15T08:00:00.000Z"
        json.dumps(body, allow_nan=False)
    assert (started["event_id"], ended["event_id"]) == (
        f"pc:{call_id}:started",
        f"pc:{call_id}:ended",
    )
    assert {"transcript", "diagnostics", "context", "custom_data", "ended_at"}.isdisjoint(started)

    other = "observability" if surface == "testing" else "testing"
    assert other not in started and other not in ended
    diagnostics = ended["diagnostics"]
    if surface == "testing":
        assert "testing" not in started  # found by SuperBryn's number: no test ID to send
        assert offline.recorders[0].events == [
            "start",
            "stop",
            "cleanup",
        ]  # dropped, never uploaded
        assert "logs" not in diagnostics
    else:
        assert started["observability"] == {"mode": "observe"}
        url = f"https://bucket.s3.test/observability/pipecat/proj_42/{call_id}-2ch.wav"
        assert ended["observability"] == {
            "mode": "observe",
            "recording_url": url,
            "stereo_recording_url": url,
            "recording_url_source": "superbryn_s3",
        }
        assert [r["message"] for r in diagnostics["logs"]] == ["caller asked about an order"]

    assert (ended["ended_reason"], ended["ended_by"]) == ("ended", "agent")
    assert ended["ended_at"] == "2027-01-15T08:00:14.600Z"
    assert ended["custom_data"] == {**CUSTOM_DATA, "lead_source": "website"}
    context = ended["context"]
    assert context["system_prompt"] == SYSTEM_PROMPT
    assert context["prompt_ref"] == {"hash": cr.prompt_ref(PROMPT)["hash"], "version": "v3"}
    assert set(context["environment"]) == {
        "pipecat_observer_version",
        "pipecat_version",
        "python_version",
        "transport",
        "pipeline_version",
    }
    assert [p["template"] for p in offline.pushes] == [PROMPT]

    assert [(t["id"], t["role"]) for t in ended["transcript"]] == [
        ("a1", "agent"),
        ("u1", "user"),
        ("a2", "agent"),
        ("u2", "user"),
        ("a3", "agent"),
    ]
    assert ended["tool_calls"] == [
        {
            "name": "lookup_order",
            "start_ms": 6800,
            "id": "call_1",
            "arguments": {"order_id": "884291"},
            "result": {"status": "shipped", "eta": "Friday"},
            "end_ms": 7100,
            "is_error": False,
            "turn_id": "a2",
        }
    ]
    assert diagnostics["turn_latencies"] == [
        {
            "turn_id": "a2",
            "turn_ms": 2700,
            "endpointing_ms": 800,
            "transcriber_ms": 210,
            "model_ms": 380,
            "voice_ms": 190,
        },
        {"turn_id": "a3", "turn_ms": 1700, "endpointing_ms": 800},
    ]
    assert diagnostics["llm_requests"] == [
        {
            "start_ms": 500,
            "outcome": "spoken",
            "input_tokens": 412,
            "cached_input_tokens": 256,
            "output_tokens": 18,
        },
        {"start_ms": 6300, "outcome": "tool_call"},
        {
            "start_ms": 7200,
            "outcome": "spoken",
            "input_tokens": 412,
            "cached_input_tokens": 256,
            "output_tokens": 18,
        },
        {"start_ms": 12300, "outcome": "spoken"},
    ]
    assert diagnostics["errors"] == [
        {
            "step": "stt",
            "side": "caller",
            "start_ms": 10000,
            "message": "Deepgram connection closed",
            "recovered": True,
        }
    ]
    assert diagnostics["dtmf"] == [{"at_ms": 10500, "digit": "1"}]
    assert diagnostics["usage"] == {
        "llm_input_tokens": 824,
        "llm_output_tokens": 36,
        "llm_cached_input_tokens": 512,
        "tts_characters": 31,
    }
    assert diagnostics["configuration"]["tools"] == [
        {"name": "lookup_order", "description": "Look up an order by its number."}
    ]


async def test_production_false_sends_nothing_for_a_production_call(offline):
    twilio = CallDataModel(call_id="CA1", **{"from": CALLER, "to": AGENT_LINE})
    call = Call(
        offline.clock,
        SimpleNamespace(call_data=twilio),
        api_key="k",
        production=False,
        prompt=PROMPT,
    )
    await call.start()
    await script(call)
    await call.end(at=15)
    assert [r["method"] for r in offline.http.requests] == ["GET"]  # the caller-number check alone
    assert offline.recorders == [] and offline.pushes == []


async def test_record_false_sends_no_recording(offline):
    call = Call(
        offline.clock, SimpleNamespace(session_id="s-1"), api_key="k", production=True, record=False
    )
    await call.start()
    await script(call)
    await call.end(at=15)
    ended = [r["body"] for r in offline.http.requests if r["method"] == "POST"][-1]
    assert ended["observability"] == {"mode": "observe"} and offline.recorders == []
