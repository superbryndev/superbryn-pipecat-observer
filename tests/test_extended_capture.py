"""Extended capture: metric-record isolation, new payload sections, opt-out."""

from __future__ import annotations

from dataclasses import dataclass

import pytest
from pipecat.frames.frames import (
    ClientConnectedFrame,
    ErrorFrame,
    InputDTMFFrame,
    InterruptionFrame,
    KeypadEntry,
    MetricsFrame,
    STTMetadataFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (
    LLMTokenUsage,
    LLMUsageMetricsData,
    STTUsage,
    STTUsageMetricsData,
    TTFBMetricsData,
    TTSUsageMetricsData,
    TurnMetricsData,
)

from superbryn_pipecat_observer.observer import SuperbrynObserver


@dataclass
class FakePushed:
    frame: object
    source: object = None
    destination: object = None
    direction: object = None
    timestamp: int = 0


def make_observer(**kwargs):
    return SuperbrynObserver(agent_id="test-agent", api_key="test-key", **kwargs)


async def push(observer, frame, source=None):
    await observer.on_push_frame(FakePushed(frame=frame, source=source))


@pytest.mark.asyncio
async def test_stt_usage_object_is_read_and_does_not_drop_later_records():
    """Pipecat >= 1.6 wraps STT usage in an object; a float() on it used to
    raise and abort the whole record loop, silently losing LLM/TTS usage."""
    obs = make_observer()
    await push(
        obs,
        MetricsFrame(
            data=[
                STTUsageMetricsData(
                    processor="stt", model="nova-3", value=STTUsage(audio_seconds=4.25)
                ),
                LLMUsageMetricsData(
                    processor="llm",
                    model="claude-sonnet-4",
                    value=LLMTokenUsage(prompt_tokens=100, completion_tokens=20, total_tokens=120),
                ),
                TTSUsageMetricsData(processor="tts", model="sonic", value=42),
            ]
        ),
    )
    assert obs.usage["stt_duration_seconds"] == 4.25
    assert obs.usage["llm_input_tokens"] == 100
    assert obs.usage["tts_characters"] == 42


@pytest.mark.asyncio
async def test_legacy_float_stt_usage_still_supported():
    """Pipecat < 1.6 delivered a bare float. Built by duck-typing because the
    installed model rejects it — that older shape is exactly what we tolerate."""
    obs = make_observer()
    legacy_record = type("STTUsageMetricsData", (), {})()
    legacy_record.processor = "stt"
    legacy_record.model = "nova-3"
    legacy_record.value = 2.5

    obs._capture_metric_record(legacy_record)

    assert obs.usage["stt_duration_seconds"] == 2.5


@pytest.mark.asyncio
async def test_unparseable_metric_record_does_not_break_the_batch():
    obs = make_observer()

    class Exploding:
        processor = "bad"

        def __getattribute__(self, name):
            if name == "value":
                raise RuntimeError("boom")
            return super().__getattribute__(name)

    await push(
        obs,
        MetricsFrame(
            data=[Exploding(), TTSUsageMetricsData(processor="tts", model="sonic", value=7)]
        ),
    )
    assert obs.usage["tts_characters"] == 7


@pytest.mark.asyncio
async def test_extended_sections_present_with_expected_values():
    obs = make_observer()
    await push(obs, ClientConnectedFrame())
    await push(obs, STTMetadataFrame(service_name="deepgram", ttfs_p99_latency=0.42))
    await push(obs, VADUserStartedSpeakingFrame(start_secs=0.2))
    await push(obs, VADUserStoppedSpeakingFrame(stop_secs=0.8))
    await push(obs, InterruptionFrame())
    await push(obs, ErrorFrame(error="stt websocket closed", fatal=False))
    await push(obs, InputDTMFFrame(button=KeypadEntry.ONE))
    await push(
        obs,
        MetricsFrame(
            data=[
                TurnMetricsData(
                    processor="turn",
                    is_complete=True,
                    probability=0.9,
                    e2e_processing_time_ms=300.0,
                ),
                TTFBMetricsData(processor="llm", model="m", value=0.5),
            ]
        ),
    )

    call = obs._build_payload()["call"]

    assert call["turn_detection"]["prediction_count"] == 1
    assert call["turn_detection"]["avg_e2e_processing_ms"] == 300.0
    assert call["latency"]["avg_ttfb_ms"] == 500.0
    assert call["latency"]["by_service"]["llm"]["avg_ttfb_ms"] == 500.0
    assert call["interruptions"]["interruption_count"] == 1
    assert call["errors"][0]["message"] == "stt websocket closed"
    assert call["errors"][0]["fatal"] is False
    assert call["sip"]["dtmf"][0]["digit"] == "1"
    assert call["vad"]["segment_count"] == 1
    assert call["vad"]["params"] == {"start_secs": 0.2, "stop_secs": 0.8}
    assert call["pipeline"]["service_metadata"]["deepgram"]["ttfs_p99_ms"] == 420.0
    assert call["environment"]["pipecat_version"]
    assert any(e["type"] == "ClientConnectedFrame" for e in call["connection"])


@pytest.mark.asyncio
async def test_llm_token_detail_is_accumulated():
    obs = make_observer()
    await push(
        obs,
        MetricsFrame(
            data=[
                LLMUsageMetricsData(
                    processor="llm",
                    model="claude-sonnet-4",
                    value=LLMTokenUsage(
                        prompt_tokens=10,
                        completion_tokens=5,
                        total_tokens=15,
                        cache_read_input_tokens=8,
                        reasoning_tokens=3,
                        input_audio_tokens=2,
                    ),
                )
            ]
        ),
    )
    usage = obs._build_payload()["call"]["usage"]
    assert usage["llm_cache_read_input_tokens"] == 8
    assert usage["llm_reasoning_tokens"] == 3
    assert usage["llm_input_audio_tokens"] == 2
    assert usage["llm_total_tokens"] == 15


@pytest.mark.asyncio
async def test_extended_capture_off_emits_legacy_payload_only():
    obs = make_observer(extended_capture=False)
    await push(obs, InterruptionFrame())
    await push(
        obs,
        MetricsFrame(
            data=[
                STTUsageMetricsData(processor="stt", value=STTUsage(audio_seconds=2.0)),
                TTFBMetricsData(processor="llm", model="m", value=0.5),
            ]
        ),
    )

    call = obs._build_payload()["call"]

    for section in (
        "turn_detection",
        "speech_stats",
        "interruptions",
        "errors",
        "connection",
        "sip",
        "vad",
        "pipeline",
        "environment",
    ):
        assert section not in call
    assert "avg_ttfb_ms" not in call["latency"]
    assert "llm_reasoning_tokens" not in call["usage"]
    # the STT usage fix is not gated behind extended capture
    assert call["usage"]["stt_duration_seconds"] == 2.0


@pytest.mark.asyncio
async def test_a_broken_section_builder_does_not_drop_the_payload():
    obs = make_observer()
    obs.turn_events = "not-a-list"  # force _build_turn_detection_section to raise

    call = obs._build_payload()["call"]

    assert call["turn_detection"] is None
    assert call["interruptions"]["interruption_count"] == 0
    assert "transcript" in call
