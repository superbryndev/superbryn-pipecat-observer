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
    assert payload["call"]["session_id"]
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
    print("All smoke tests passed.")
