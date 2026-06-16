"""
Smoke tests for superbryn-pipecat-observer.

Run with:  pytest test_import.py -v   (or just `python test_import.py`)

Validates:
  - Package imports without pipecat installed (graceful degradation)
  - Payload builder produces the canonical SuperBryn shape
  - Provider detection covers the major LLM / STT / TTS names
  - Observer no-ops fail-open when no API key is set
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

    # Should not raise even when no API key is present.
    asyncio.run(obs._send_webhook())


def test_transport_adapter_detection() -> None:
    """Class-name-based detection picks the right adapter per transport."""
    from superbryn_pipecat_observer.transports import (
        DailyRecordingAdapter,
        TwilioRecordingAdapter,
        get_recording_adapter,
    )

    class DailyTransport:  # noqa: D401 — minimal stub
        pass

    class _TwilioFrameSerializer:
        __module__ = "pipecat.serializers.twilio"

    class FastAPIWebsocketTransport:
        def __init__(self) -> None:
            self._serializer = _TwilioFrameSerializer()

    class RawWebRTCTransport:
        pass

    class SmallWebRTCTransport:
        pass

    class WebsocketServerTransport:
        pass

    assert isinstance(get_recording_adapter(DailyTransport()), DailyRecordingAdapter)
    assert isinstance(get_recording_adapter(FastAPIWebsocketTransport()), TwilioRecordingAdapter)
    # Transports with no recording API → no adapter (caller passes recording_url manually).
    assert get_recording_adapter(RawWebRTCTransport()) is None
    assert get_recording_adapter(SmallWebRTCTransport()) is None
    assert get_recording_adapter(WebsocketServerTransport()) is None
    # Legacy string label must not produce an adapter.
    assert get_recording_adapter("daily") is None
    assert get_recording_adapter(None) is None


def test_observer_accepts_transport_object() -> None:
    """Passing a transport object wires the adapter; string label still works."""
    from superbryn_pipecat_observer import SuperbrynObserver

    class DailyTransport:
        pass

    obs_obj = SuperbrynObserver(agent_name="t", api_key="x", transport=DailyTransport())
    assert obs_obj.transport == "daily"
    assert obs_obj._recording_adapter is not None
    assert obs_obj._recording_adapter.transport_name == "daily"

    # Backwards-compat: string label still works, no adapter.
    obs_str = SuperbrynObserver(agent_name="t", api_key="x", transport="custom")
    assert obs_str.transport == "custom"
    assert obs_str._recording_adapter is None


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


def test_lifecycle_fail_open_on_adapter_error() -> None:
    """If the adapter raises, the observer still sends the webhook."""
    from superbryn_pipecat_observer import SuperbrynObserver

    class BrokenAdapter:
        transport_name = "broken"
        transport = None

        async def start(self, _observer):  # noqa: D401, ANN001
            raise RuntimeError("boom on start")

        async def finalize(self, _observer):  # noqa: D401, ANN001
            raise RuntimeError("boom on finalize")

    obs = SuperbrynObserver(agent_name="t", api_key="")
    obs._recording_adapter = BrokenAdapter()

    # Neither hook should propagate the adapter exception.
    asyncio.run(obs.on_pipeline_started())
    asyncio.run(obs.on_pipeline_finished())


def test_no_recording_module_exposed() -> None:
    """The in-pipeline recording subpackage was removed in favor of transport-native URLs only."""
    import importlib

    try:
        importlib.import_module("superbryn_pipecat_observer.recording")
    except ImportError:
        return
    raise AssertionError(
        "superbryn_pipecat_observer.recording should not be importable — "
        "in-pipeline recording was removed."
    )


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
    test_transport_adapter_detection()
    test_observer_accepts_transport_object()
    test_payload_includes_stereo_recording_url()
    test_lifecycle_fail_open_on_adapter_error()
    test_no_recording_module_exposed()
    test_observer_rejects_record_audio_kwarg()
    print("All smoke tests passed.")
