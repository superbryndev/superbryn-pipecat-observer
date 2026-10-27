"""Observer capture: each frame counted once, turns from played audio, metrics switched on."""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass

import numpy as np
import pytest
from loguru import logger as loguru_logger
from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    FunctionCallInProgressFrame,
    FunctionCallResultFrame,
    InputAudioRawFrame,
    InterruptionFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    MetricsFrame,
    TranscriptionFrame,
    TTSTextFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
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


class STTService:  # matched by class name, like Pipecat's
    pass


class FakeSTT(STTService):
    pass


def service(module: str):
    return type("FakeService", (), {"__module__": module})()


LLM = service("pipecat.services.openai.llm")
REALTIME = service("pipecat.services.openai.realtime.llm")
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


class Clock:
    """The observer's call clock (ms), set by the test."""

    def __init__(self, observer) -> None:
        self.ms = 0
        observer._now_ms = lambda: self.ms


def vad_at(observer, ms):
    """A VAD frame's own timestamp, ``ms`` into the call."""
    return (observer._call_start_ms + ms) / 1000


def caller_frame(loud: bool, rate: int = 16000) -> InputAudioRawFrame:
    """20 ms of the caller's line: a 440 Hz tone (about -23 dBFS) or silence."""
    t = np.arange(rate // 50) / rate
    pcm = (3000 * np.sin(2 * np.pi * 440 * t) * loud).astype("<i2").tobytes()
    return InputAudioRawFrame(audio=pcm, sample_rate=rate, num_channels=1)


async def caller_audio(observer, clock, *, start, speech, silence):
    """The caller's line in 20 ms frames: quiet, ``speech`` ms of tone from ``start``, then ``silence`` ms."""
    loud = [False] * 10 + [True] * (speech // 20) + [False] * (silence // 20)
    for i, is_loud in enumerate(loud):
        clock.ms = start + (i - 9) * 20
        await hops(observer, caller_frame(is_loud), times=2)


async def at(clock, ms, observer, frame, source=None, **kwargs):
    clock.ms = ms
    if kwargs:
        await observer.on_push_frame(Pushed(frame=frame, source=source, **kwargs))
    else:
        await hops(observer, frame, source)


def heard(text):
    return TranscriptionFrame(text, "user-1", "2026-10-27T00:00:00.000Z")


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


async def test_turn_start_after_the_last_vad_stop_keeps_the_reply_latency(obs):
    """A line with a pause: the aggregator's turn start lands after VAD's last stop and used to wipe the latency."""
    clock = Clock(obs)
    await at(
        clock, 1200, obs, VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=vad_at(obs, 1200))
    )
    await at(
        clock, 2000, obs, VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=vad_at(obs, 2000))
    )
    await at(
        clock, 2300, obs, VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=vad_at(obs, 2300))
    )
    await at(
        clock, 3500, obs, VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=vad_at(obs, 3500))
    )
    await at(clock, 3600, obs, UserStartedSpeakingFrame())  # the user aggregator's turn start, late
    await at(clock, 3700, obs, UserStoppedSpeakingFrame())
    await at(clock, 4000, obs, BotStartedSpeakingFrame(), OUT)
    await at(
        clock, 4100, obs, TTSTextFrame(text="Sure.", aggregated_by=AggregationType.SENTENCE), OUT
    )
    await at(clock, 4500, obs, BotStoppedSpeakingFrame(), OUT)
    assert turns(obs)[0]["latency_ms"] == 1000  # from VAD's last stop (3500 - 500)
    assert obs.latencies_ms == [1000.0]


async def test_finals_of_one_caller_turn_make_one_user_turn(obs):
    """One user turn from the caller's start to their last stop, however many finals the STT sent."""
    clock = Clock(obs)
    await at(
        clock, 1200, obs, VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=vad_at(obs, 1200))
    )
    await at(clock, 1600, obs, heard("Hi."))
    await at(
        clock, 2000, obs, VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=vad_at(obs, 2000))
    )
    await at(
        clock, 2400, obs, VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=vad_at(obs, 2400))
    )
    await at(clock, 3000, obs, heard(" I need help with my recent order."))  # a leading space
    await at(
        clock, 3800, obs, VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=vad_at(obs, 3800))
    )
    await at(clock, 4000, obs, BotStartedSpeakingFrame(), OUT)
    await at(
        clock, 4100, obs, TTSTextFrame(text="Sure.", aggregated_by=AggregationType.SENTENCE), OUT
    )
    await at(clock, 4500, obs, BotStoppedSpeakingFrame(), OUT)
    assert turns(obs, "user") == [
        {
            "speaker": "user",
            "text": "Hi. I need help with my recent order.",
            "start_time_ms": 1000,
            "end_time_ms": 3300,
            "confidence": None,
        }
    ]


async def test_untimed_finals_join_until_the_agent_speaks(obs):
    clock = Clock(obs)
    await at(clock, 1000, obs, heard("Great."))
    await at(clock, 1800, obs, heard("Thanks for your help."))
    await at(clock, 2500, obs, BotStartedSpeakingFrame(), OUT)
    await at(
        clock, 2600, obs, TTSTextFrame(text="Bye.", aggregated_by=AggregationType.SENTENCE), OUT
    )
    await at(clock, 3000, obs, BotStoppedSpeakingFrame(), OUT)
    await at(clock, 4000, obs, heard("Bye."))
    users = [(t["text"], t["start_time_ms"], t["end_time_ms"]) for t in turns(obs, "user")]
    assert users == [("Great. Thanks for your help.", 1000, 1800), ("Bye.", 4000, 4000)]


async def test_server_turn_frames_time_the_caller_without_vad_or_audio(obs):
    clock = Clock(obs)
    await at(clock, 1000, obs, UserStartedSpeakingFrame(), REALTIME)
    await at(clock, 2500, obs, UserStoppedSpeakingFrame(), REALTIME)
    await at(clock, 3000, obs, BotStartedSpeakingFrame(), OUT)
    up = FrameDirection.UPSTREAM
    await at(
        clock, 3300, obs, heard("where is my order"), REALTIME, direction=up
    )  # after the reply started
    await at(clock, 4000, obs, BotStoppedSpeakingFrame(), OUT)
    [user] = turns(obs, "user")
    assert (user["text"], user["start_time_ms"], user["end_time_ms"]) == (
        "where is my order",
        1000,
        2500,
    )
    assert obs.latencies_ms == [500.0]


async def test_speech_to_speech_reply_latency_from_the_callers_audio(obs):
    """No VAD frames: the caller's audio times the reply, not the service's turn release a second later."""
    clock = Clock(obs)
    await caller_audio(obs, clock, start=1000, speech=1200, silence=300)  # speech ends at 2200
    await at(clock, 1400, obs, UserStartedSpeakingFrame(), REALTIME)  # the server's own detection
    await at(clock, 2900, obs, UserStoppedSpeakingFrame(), REALTIME)  # its turn release
    await at(clock, 3000, obs, BotStartedSpeakingFrame(), OUT)
    up = FrameDirection.UPSTREAM
    await at(clock, 3300, obs, heard("where is my order"), REALTIME, direction=up)
    await at(clock, 4000, obs, BotStoppedSpeakingFrame(), OUT)
    [user] = turns(obs, "user")
    assert (user["text"], user["start_time_ms"], user["end_time_ms"]) == (
        "where is my order",
        1000,
        2200,
    )
    assert obs.latencies_ms == [800.0]


async def test_reply_inside_the_audio_hangover_is_timed_from_the_speech_end(obs):
    clock = Clock(obs)
    await caller_audio(
        obs, clock, start=1000, speech=600, silence=100
    )  # ends at 1600, quiet to 1700
    await at(clock, 1750, obs, BotStartedSpeakingFrame(), OUT)
    assert obs.latencies_ms == [150.0]


async def test_a_click_on_the_line_is_not_caller_speech(obs):
    clock = Clock(obs)
    await caller_audio(obs, clock, start=1000, speech=40, silence=400)
    await at(clock, 1500, obs, BotStartedSpeakingFrame(), OUT)
    assert obs.latencies_ms == []
    assert obs.transcript_turns == []


async def test_vad_times_the_reply_when_both_vad_and_audio_are_seen(obs):
    clock = Clock(obs)
    await at(
        clock, 1200, obs, VADUserStartedSpeakingFrame(start_secs=0.2, timestamp=vad_at(obs, 1200))
    )
    await caller_audio(obs, clock, start=1200, speech=1000, silence=500)  # tone to 2200
    await at(
        clock, 3000, obs, VADUserStoppedSpeakingFrame(stop_secs=0.5, timestamp=vad_at(obs, 3000))
    )
    await at(clock, 3200, obs, BotStartedSpeakingFrame(), OUT)
    assert obs.latencies_ms == [700.0]  # from VAD's stop (3000 - 500), not the audio's end


async def test_no_stt_time_without_an_stt_service(obs):
    """A speech-to-speech pipeline has no STT: its turn frames are not STT time."""
    await hops(obs, UserStartedSpeakingFrame(), REALTIME)
    await asyncio.sleep(0.3)
    await hops(obs, UserStoppedSpeakingFrame(), REALTIME)
    up = FrameDirection.UPSTREAM
    await obs.on_push_frame(Pushed(frame=heard("hi"), source=REALTIME, direction=up))
    assert obs._build_payload()["call"]["usage"]["stt_duration_seconds"] == 0


async def test_stt_time_falls_back_to_turn_frames_with_an_stt_service(obs):
    await hops(obs, UserStartedSpeakingFrame())
    await asyncio.sleep(0.3)
    await hops(obs, UserStoppedSpeakingFrame())
    await hops(obs, heard("hi"), FakeSTT())
    assert 0.25 <= obs._build_payload()["call"]["usage"]["stt_duration_seconds"] <= 0.6
