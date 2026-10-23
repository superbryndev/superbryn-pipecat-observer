"""Observer capture: each frame counted once, turns from played audio, metrics switched on."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import pytest
from loguru import logger as loguru_logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TTSTextFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage, LLMUsageMetricsData, TTSUsageMetricsData
from pipecat.pipeline.task import PipelineParams
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.text.base_text_aggregator import AggregationType

from superbryn_pipecat_observer.observer import SuperbrynObserver


@dataclass
class Pushed:
    frame: object
    source: object = None
    destination: object = None
    direction: object = FrameDirection.DOWNSTREAM
    timestamp: int = 0


class BaseOutputTransport:  # matched by class name, like Pipecat's
    pass


class FakeOutputTransport(BaseOutputTransport):
    pass


def service(module: str):
    return type("FakeService", (), {"__module__": module})()


LLM = service("pipecat.services.openai.llm")
OUT = FakeOutputTransport()


@pytest.fixture
async def obs():
    observer = SuperbrynObserver(agent_id="a", api_key="k", capture_logs=False)
    await observer.on_pipeline_started()
    return observer


async def hops(observer, frame, source=None, times=4):
    """Push one frame the way Pipecat does: once per pipeline link."""
    for _ in range(times):
        await observer.on_push_frame(Pushed(frame=frame, source=source))


def turns(observer, speaker="agent"):
    return [
        t
        for t in observer._build_payload()["call"]["transcript"]["turns"]
        if t["speaker"] == speaker
    ]


async def test_usage_is_counted_once_per_frame_not_per_hop(obs):
    usage = MetricsFrame(
        data=[
            LLMUsageMetricsData(
                processor="llm",
                model="m",
                value=LLMTokenUsage(prompt_tokens=100, completion_tokens=10, total_tokens=110),
            ),
            TTSUsageMetricsData(processor="tts", model="m", value=42),
        ]
    )
    await hops(obs, usage, LLM)
    call = obs._build_payload()["call"]
    assert call["usage"]["llm_input_tokens"] == 100
    assert call["usage"]["llm_output_tokens"] == 10
    assert call["usage"]["tts_characters"] == 42


async def test_tool_call_is_one_record_with_its_result(obs):
    args = {"order_id": "884291"}
    start = FunctionCallInProgressFrame(
        function_name="lookup_order", tool_call_id="c1", arguments=args
    )
    result = FunctionCallResultFrame(
        function_name="lookup_order", tool_call_id="c1", arguments=args, result={"ok": True}
    )
    await hops(obs, start, LLM)
    await hops(obs, result, LLM)
    calls = obs._build_payload()["call"]["tool_calls"]
    assert len(calls) == 1
    assert calls[0]["result"] == {"ok": True}


async def test_interruption_counted_once_and_upstream_twin_skipped(obs):
    await hops(obs, BotStartedSpeakingFrame(), OUT)
    down, up = InterruptionFrame(), InterruptionFrame()
    down.broadcast_sibling_id, up.broadcast_sibling_id = up.id, down.id
    await hops(obs, down)
    await obs.on_push_frame(Pushed(frame=up, direction=FrameDirection.UPSTREAM))
    section = obs._build_payload()["call"]["interruptions"]
    assert section["interruption_count"] == 1
    assert section["bot_turns_interrupted"] == 1


async def test_cut_off_reply_keeps_only_the_words_played(obs):
    await hops(obs, LLMFullResponseStartFrame(), LLM)
    for token in ["Sure", ",", " I", " can", " help", ".", " Could", " you", " repeat", "?"]:
        await hops(obs, LLMTextFrame(text=token), LLM)
    await hops(obs, LLMFullResponseEndFrame(), LLM)
    await hops(obs, BotStartedSpeakingFrame(), OUT)
    for word in ["Sure,", "I", "can"]:
        await hops(obs, TTSTextFrame(text=word, aggregated_by=AggregationType.WORD), OUT)
    await hops(obs, InterruptionFrame())
    await hops(obs, BotStoppedSpeakingFrame(), OUT)
    agent = turns(obs)
    assert [t["text"] for t in agent] == ["Sure, I can"]
    assert agent[0]["interrupted"] is True


async def test_tts_text_before_playback_is_not_counted(obs):
    """The TTS service pushes text before the transport plays it; only the played hop counts."""
    await hops(obs, BotStartedSpeakingFrame(), OUT)
    word = TTSTextFrame(text="Hello", aggregated_by=AggregationType.WORD)
    await obs.on_push_frame(Pushed(frame=word, source=service("pipecat.services.cartesia.tts")))
    await obs.on_push_frame(Pushed(frame=word, source=OUT))
    await obs.on_push_frame(Pushed(frame=word, source=OUT))
    await hops(obs, BotStoppedSpeakingFrame(), OUT)
    assert [t["text"] for t in turns(obs)] == ["Hello"]


async def test_reply_latency_runs_from_end_of_speech_to_bot_audio(obs):
    await hops(obs, VADUserStartedSpeakingFrame(start_secs=0.2))
    await asyncio.sleep(0.6)  # the caller talks, then 0.5s of silence ends the turn
    stopped = VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=time.time())
    await hops(obs, stopped)
    await hops(obs, LLMFullResponseStartFrame(), LLM)  # the LLM starting is not the reply
    await hops(obs, BotStartedSpeakingFrame(), OUT)
    await hops(obs, TTSTextFrame(text="Hi", aggregated_by=AggregationType.WORD), OUT)
    await hops(obs, BotStoppedSpeakingFrame(), OUT)
    latency = turns(obs)[0]["latency_ms"]
    # 0.5s of silence before the VAD stop, then the bot answered at once
    assert 450 <= latency <= 700
    assert obs.latencies_ms == [latency]


async def test_terminal_frame_waits_for_task_hook(obs):
    obs._finish_hooked = True
    await hops(obs, EndFrame())
    assert obs._sent is False
    assert obs.call_end_reason == "completed"


def test_usage_metrics_switched_on_for_customer_params():
    """PipelineParams(enable_metrics=True) leaves usage off; tokens used to read 0."""
    params = PipelineParams(enable_metrics=True)
    task = type("Task", (), {"params": params})()
    SuperbrynObserver._ensure_metrics(task)
    assert params.enable_usage_metrics is True
    assert params.enable_metrics is True


def test_metrics_turned_off_on_purpose_stay_off(caplog):
    params = PipelineParams(enable_usage_metrics=False)
    task = type("Task", (), {"params": params})()
    with caplog.at_level(logging.WARNING):
        SuperbrynObserver._ensure_metrics(task)
    assert params.enable_usage_metrics is False
    assert params.enable_metrics is True
    assert "SUPERBRYN_PIPECAT_METRICS_OFF" in caplog.text


def test_transport_label_set_from_task_setup():
    class FakeWebsocketTransport:
        def output(self):
            return None

    observer = SuperbrynObserver(agent_id="a", api_key="k")
    task = type("Task", (), {"params": PipelineParams()})()
    observer.register_task_handlers(task, transport=FakeWebsocketTransport())
    assert observer._build_payload()["call"]["metadata"]["transport"] == "fakewebsocket"


async def test_loguru_and_logging_lines_are_both_captured():
    observer = SuperbrynObserver(agent_id="a", api_key="k")
    await observer.on_pipeline_started()
    loguru_logger.info("from loguru")
    logging.getLogger("agent").info("from logging")
    observer._detach_log_handler()
    loguru_logger.info("after the call")
    messages = [entry["message"] for entry in observer._captured_logs]
    assert "from loguru" in messages
    assert "from logging" in messages
    assert "after the call" not in messages
