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


if __name__ == "__main__":
    test_package_imports()
    test_payload_shape_minimal()
    test_provider_detect()
    test_observer_no_api_key_is_noop()
    print("All smoke tests passed.")
