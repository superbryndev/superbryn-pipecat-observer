"""
Smoke tests for superbryn-pipecat-observer.

Run with:  pytest test_import.py -v   (or just `python test_import.py`)

Validates:
  - Package imports without pipecat installed (graceful degradation)
  - Payload builder produces the canonical SuperBryn shape
  - Provider detection covers the major LLM / STT / TTS names
  - Observer no-ops fail-open when no API key is set
  - Transport arg behaviour: string labels and transport objects both
    produce a metadata label, neither pulls in a per-carrier adapter
    (those were removed in 0.5.0 — audio capture is now in-pipeline,
    transport-agnostic).
"""

import asyncio
import os


def test_package_imports() -> None:
    from superbryn_pipecat_observer import SuperbrynObserver, __version__

    assert SuperbrynObserver is not None
    assert isinstance(__version__, str) and __version__


def test_payload_shape_minimal() -> None:
    from superbryn_pipecat_observer import SuperbrynObserver

    obs = SuperbrynObserver(agent_name="test-bot", api_key="sb_test")
    payload = obs._build_payload()

    assert payload["event"] == "call.completed"
    assert payload["sdk_version"].startswith("@superbryn/pipecat-observer@")
    assert payload["call"]["id"]
    assert payload["call"]["metadata"]["agent_name"] == "test-bot"
    assert "transcript" in payload["call"]
    assert "usage" in payload["call"]
    assert "latency" in payload["call"]


def test_provider_detect() -> None:
    from superbryn_pipecat_observer._provider_detect import detect_provider_from_model

    assert detect_provider_from_model("gpt-4o-mini") == "openai"
    assert detect_provider_from_model("claude-3-5-sonnet") == "anthropic"
    assert detect_provider_from_model("nova-3") == "deepgram"
    assert detect_provider_from_model("sonic-english") == "cartesia"
    assert detect_provider_from_model("eleven_turbo_v2_5") == "elevenlabs"
    assert detect_provider_from_model("unknown-model") == "unknown"


def test_observer_no_api_key_is_noop() -> None:
    from superbryn_pipecat_observer import SuperbrynObserver

    os.environ.pop("SUPERBRYN_API_KEY", None)
    obs = SuperbrynObserver(agent_name="no-key", api_key="")

    asyncio.run(obs._send_webhook())


def test_transport_arg_string_label() -> None:
    """A string `transport=` lands on the payload metadata unchanged."""
    from superbryn_pipecat_observer import SuperbrynObserver

    obs = SuperbrynObserver(agent_name="t", api_key="x", transport="vobiz")
    assert obs.transport == "vobiz"
    assert obs._transport_obj is None
    assert obs._build_payload()["call"]["metadata"]["transport"] == "vobiz"


def test_transport_arg_object_derives_label() -> None:
    """An object `transport=` keeps a reference (for pipeline output lookup)
    and produces a normalized lowercase label on the payload."""
    from superbryn_pipecat_observer import SuperbrynObserver

    class DailyTransport:
        pass

    t = DailyTransport()
    obs = SuperbrynObserver(agent_name="t", api_key="x", transport=t)
    assert obs.transport == "daily"
    assert obs._transport_obj is t


def test_no_legacy_recording_adapter_state() -> None:
    """The per-carrier polling flow was removed in 0.5.0 — neither the
    submodule nor the observer fields should exist any more."""
    import importlib

    from superbryn_pipecat_observer import SuperbrynObserver

    obs = SuperbrynObserver(agent_name="t", api_key="x", transport="daily")
    assert not hasattr(obs, "_recording_adapter")
    assert not hasattr(obs, "_backfill_recording")

    try:
        importlib.import_module("superbryn_pipecat_observer.transports")
    except ImportError:
        return
    raise AssertionError(
        "superbryn_pipecat_observer.transports should not be importable — "
        "per-carrier recording adapters were removed in 0.5.0."
    )


def test_payload_includes_stereo_recording_url() -> None:
    from superbryn_pipecat_observer import SuperbrynObserver

    obs = SuperbrynObserver(
        agent_name="t",
        api_key="x",
        recording_url="https://r/mono.mp3",
        stereo_recording_url="https://r/stereo.mp3",
    )
    payload = obs._build_payload()
    assert payload["call"]["recording_url"] == "https://r/mono.mp3"
    assert payload["call"]["stereo_recording_url"] == "https://r/stereo.mp3"


def test_observer_rejects_record_audio_kwarg() -> None:
    """`record_audio` / `recording_storage` were removed; passing them must raise TypeError."""
    from superbryn_pipecat_observer import SuperbrynObserver

    for kwargs in (
        {"record_audio": True},
        {"recording_storage": {"type": "s3"}},
        {"recording_config": object()},
    ):
        try:
            SuperbrynObserver(agent_name="t", api_key="x", **kwargs)
        except TypeError:
            continue
        raise AssertionError(f"SuperbrynObserver should reject {kwargs}")


# ── Bot-turn capture regressions ─────────────────────────────────────────
#
# These reproduce two production bugs seen with the Anthropic+Cartesia
# pipeline in `pipecat-agent`:
#   1. Streaming LLM tokens after the first were silently dropped
#      because `_looks_like_bot_text` gated on "agent turn with empty
#      text", which becomes False the moment the first token lands.
#   2. When Cartesia (a word-timestamp TTS) emits `TTSTextFrame` per
#      spoken word *and* the LLM streams `LLMTextFrame`s for the same
#      response, both were captured with the fix from (1) — duplicating
#      the whole utterance in the transcript.
#
# The observer must now:
#   * Capture every streamed token of a single bot response.
#   * Lock a turn to the first text source (LLM or TTS) and drop the
#     other, so the same content is never counted twice.
#   * Still capture a `TTSSpeakFrame`-injected greeting (LLM bypassed)
#     via the TTS's word-level `TTSTextFrame`s.


def _make_frame_pushed(frame, source):
    """Build a fake `FramePushed` payload that mirrors what Pipecat delivers."""
    from dataclasses import make_dataclass

    FramePushed = make_dataclass("FramePushed", ["frame", "source"])
    return FramePushed(frame=frame, source=source)


def _make_service(module_path):
    """Build a fake service whose `__module__` matches a real Pipecat plugin."""
    return type(f"Fake_{module_path.replace('.', '_')}", (), {"__module__": module_path})()


def test_streaming_llm_tokens_are_all_captured() -> None:
    """Regression: an LLM response that streams as multiple `LLMTextFrame`s
    must end up as a single agent turn containing the full concatenated text.
    Previously only the first token was captured."""
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
    )

    from superbryn_pipecat_observer import SuperbrynObserver

    obs = SuperbrynObserver(agent_name="stream", api_key="sb_test")
    llm = _make_service("pipecat.services.anthropic.llm")

    async def drive() -> None:
        await obs.on_push_frame(_make_frame_pushed(LLMFullResponseStartFrame(), llm))
        for token in ["Hello", " there", ",", " how", " can", " I", " help", "?"]:
            await obs.on_push_frame(_make_frame_pushed(LLMTextFrame(text=token), llm))
        await obs.on_push_frame(_make_frame_pushed(LLMFullResponseEndFrame(), llm))

    asyncio.run(drive())

    agent_turns = [t for t in obs.transcript_turns if t["speaker"] == "agent"]
    assert len(agent_turns) == 1
    assert agent_turns[0]["text"] == "Hello there, how can I help?"


def test_llm_stream_and_tts_word_frames_do_not_double_count() -> None:
    """Regression: a word-timestamp TTS (Cartesia) emits `TTSTextFrame` per
    spoken word for content the LLM already streamed. The observer must
    capture from exactly one source (LLM here) and drop the other."""
    from pipecat.frames.frames import (
        LLMFullResponseEndFrame,
        LLMFullResponseStartFrame,
        LLMTextFrame,
        TTSTextFrame,
    )
    from pipecat.utils.text.base_text_aggregator import AggregationType

    from superbryn_pipecat_observer import SuperbrynObserver

    obs = SuperbrynObserver(agent_name="dup", api_key="sb_test")
    llm = _make_service("pipecat.services.anthropic.llm")
    tts = _make_service("pipecat.services.cartesia.tts")

    async def drive() -> None:
        await obs.on_push_frame(_make_frame_pushed(LLMFullResponseStartFrame(), llm))
        for token in ["Hello", " there"]:
            await obs.on_push_frame(_make_frame_pushed(LLMTextFrame(text=token), llm))
        for word in ["Hello", "there"]:
            frame = TTSTextFrame(text=word, aggregated_by=AggregationType.TOKEN)
            await obs.on_push_frame(_make_frame_pushed(frame, tts))
        await obs.on_push_frame(_make_frame_pushed(LLMFullResponseEndFrame(), llm))

    asyncio.run(drive())

    agent_turns = [t for t in obs.transcript_turns if t["speaker"] == "agent"]
    assert len(agent_turns) == 1
    assert agent_turns[0]["text"] == "Hello there"


def test_tts_speak_greeting_is_captured_via_tts_words() -> None:
    """Regression: a `TTSSpeakFrame` injection (pre-baked greeting) has no
    LLM stream. The observer must open the bot turn on `TTSSpeakFrame`
    early enough that the TTS service's word-level `TTSTextFrame`s land
    inside it — the greeting text was previously missing entirely."""
    from pipecat.frames.frames import (
        BotStoppedSpeakingFrame,
        TTSSpeakFrame,
        TTSTextFrame,
    )
    from pipecat.utils.text.base_text_aggregator import AggregationType

    from superbryn_pipecat_observer import SuperbrynObserver

    obs = SuperbrynObserver(agent_name="greet", api_key="sb_test")
    tts = _make_service("pipecat.services.cartesia.tts")
    pipeline_source = _make_service("pipecat.pipeline.pipeline")

    async def drive() -> None:
        greeting = "Hi, welcome to the cafe."
        greeting_frame = TTSSpeakFrame(text=greeting)
        await obs.on_push_frame(_make_frame_pushed(greeting_frame, pipeline_source))
        await obs.on_push_frame(_make_frame_pushed(greeting_frame, pipeline_source))
        for word in ["Hi,", "welcome", "to", "the", "cafe."]:
            frame = TTSTextFrame(text=word, aggregated_by=AggregationType.TOKEN)
            await obs.on_push_frame(_make_frame_pushed(frame, tts))
        await obs.on_push_frame(_make_frame_pushed(BotStoppedSpeakingFrame(), tts))

    asyncio.run(drive())

    agent_turns = [t for t in obs.transcript_turns if t["speaker"] == "agent"]
    assert len(agent_turns) == 1, f"expected 1 agent turn, got {len(agent_turns)}"
    assert agent_turns[0]["text"] == "Hi, welcome to the cafe."


if __name__ == "__main__":
    test_package_imports()
    test_payload_shape_minimal()
    test_provider_detect()
    test_observer_no_api_key_is_noop()
    test_transport_arg_string_label()
    test_transport_arg_object_derives_label()
    test_no_legacy_recording_adapter_state()
    test_payload_includes_stereo_recording_url()
    test_observer_rejects_record_audio_kwarg()
    test_streaming_llm_tokens_are_all_captured()
    test_llm_stream_and_tts_word_frames_do_not_double_count()
    test_tts_speak_greeting_is_captured_via_tts_words()
    print("All smoke tests passed.")
