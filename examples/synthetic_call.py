"""
Synthetic end-to-end test for SuperbrynObserver.

Simulates a realistic Pipecat call by pushing scripted frames through the
observer in the same order Pipecat would emit them, then intercepts the
outgoing webhook and pretty-prints the exact payload SuperBryn would
receive.

Run:
    python examples/synthetic_call.py

No real STT / LLM / TTS credentials needed. Nothing is sent over the
network — the HTTP send is monkey-patched to capture the payload.
"""

from __future__ import annotations

import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any

# Make sure the observer thinks it's configured (otherwise it no-ops)
os.environ.setdefault("SUPERBRYN_API_KEY", "sb_test_dummy_key")
os.environ.setdefault("AGENT_ID", "synthetic-support-bot")

from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    EndFrame,
    LLMFullResponseStartFrame,
    MetricsFrame,
    TextFrame,
    TranscriptionFrame,
    UserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import (  # noqa: E402
    LLMTokenUsage,
    LLMUsageMetricsData,
    TTSUsageMetricsData,
)

from superbryn_pipecat_observer import SuperbrynObserver  # noqa: E402


# A fake FramePushed payload. The real Pipecat class is a dataclass with
# (source, destination, frame, direction, timestamp) — we only need
# `source` and `frame` for what the observer reads.
@dataclass
class FakeFramePushed:
    frame: Any
    source: Any = None
    destination: Any = None
    direction: Any = None
    timestamp: int = 0


# Fake "service" used to test provider detection from module path.
@dataclass
class FakeService:
    model_name: str | None = None
    voice_id: str | None = None
    # `__module__` is what the SDK actually sniffs
    _module: str = ""

    def __post_init__(self) -> None:
        type(self).__module__ = self._module or type(self).__module__


def make_service(module_path: str, **kwargs: Any) -> FakeService:
    """Build a fake service whose __module__ matches a real Pipecat plugin."""
    cls = type(
        f"FakeService_{module_path.replace('.', '_')}",
        (FakeService,),
        {"__module__": module_path},
    )
    return cls(**kwargs)


async def push(observer: SuperbrynObserver, source: Any, frame: Any) -> None:
    """Helper — push a single frame through the observer."""
    await observer.on_push_frame(FakeFramePushed(frame=frame, source=source))


async def main() -> None:
    captured: dict[str, Any] = {}

    # Build the observer with metadata we can verify on the other side
    observer = SuperbrynObserver(
        agent_name="synthetic-support-bot",
        transport="websocket",
        from_number="+15551234567",
        to_number="+15557654321",
        recording_url="https://example.com/recordings/synthetic.mp3",
        extra_metadata={
            "campaign": "summer-promo",
            "tenant_id": "acme-corp",
        },
    )

    # Intercept the webhook send so nothing actually leaves the machine.
    async def fake_send(self_: SuperbrynObserver) -> None:
        payload = self_._build_payload()
        captured["payload"] = payload
        captured["webhook_url"] = self_.webhook_url
        captured["api_key_present"] = bool(self_.api_key)

    SuperbrynObserver._send_webhook = fake_send  # type: ignore[method-assign]

    # Fake services for provider sniffing — module paths match real Pipecat plugins
    stt = make_service("pipecat.services.deepgram.stt", model_name="nova-3")
    llm = make_service("pipecat.services.openai.llm", model_name="gpt-4o-mini")
    tts = make_service(
        "pipecat.services.cartesia.tts",
        model_name="sonic-english",
        voice_id="bf991597-6c13-47e4-8411-91ec2de5c466",
    )

    # ───────── Call lifecycle ─────────
    await observer.on_pipeline_started()

    # Provider sniffing happens on any frame whose `source` is a service.
    # Push a synthetic neutral frame through each service first so providers
    # get tagged before any usage metrics arrive.
    await push(observer, stt, TextFrame(text=""))
    await push(observer, llm, TextFrame(text=""))
    await push(observer, tts, TextFrame(text=""))

    # ── Turn 1: user says hello ──
    await asyncio.sleep(0.05)
    await push(
        observer,
        stt,
        TranscriptionFrame(
            text="Hi, I need help with my recent order.",
            user_id="caller-1",
            timestamp="2026-06-15T12:00:05.000Z",
        ),
    )
    await push(observer, stt, UserStoppedSpeakingFrame())

    # ── Bot responds ──
    await asyncio.sleep(0.12)  # response delay
    await push(observer, llm, LLMFullResponseStartFrame())
    await push(observer, llm, BotStartedSpeakingFrame())
    await push(
        observer,
        llm,
        TextFrame(text="Sure! I can look that up. What's your order number?"),
    )
    await push(
        observer,
        llm,
        MetricsFrame(
            data=[
                LLMUsageMetricsData(
                    processor="llm",
                    model="gpt-4o-mini",
                    value=LLMTokenUsage(
                        prompt_tokens=180,
                        completion_tokens=24,
                        total_tokens=204,
                    ),
                ),
            ]
        ),
    )
    await push(
        observer,
        tts,
        MetricsFrame(
            data=[
                TTSUsageMetricsData(
                    processor="tts",
                    model="sonic-english",
                    value=52,
                )
            ]
        ),
    )

    # ── Turn 2: user gives order number ──
    await asyncio.sleep(0.08)
    await push(
        observer,
        stt,
        TranscriptionFrame(
            text="It's order number 8 8 4 2 9 1.",
            user_id="caller-1",
            timestamp="2026-06-15T12:00:09.000Z",
        ),
    )
    await push(observer, stt, UserStoppedSpeakingFrame())

    # ── Bot responds ──
    await asyncio.sleep(0.18)
    await push(observer, llm, LLMFullResponseStartFrame())
    await push(observer, llm, BotStartedSpeakingFrame())
    await push(
        observer,
        llm,
        TextFrame(text="Got it. Your order shipped yesterday and arrives Wednesday."),
    )
    await push(
        observer,
        llm,
        MetricsFrame(
            data=[
                LLMUsageMetricsData(
                    processor="llm",
                    model="gpt-4o-mini",
                    value=LLMTokenUsage(
                        prompt_tokens=210,
                        completion_tokens=18,
                        total_tokens=228,
                    ),
                )
            ]
        ),
    )
    await push(
        observer,
        tts,
        MetricsFrame(
            data=[
                TTSUsageMetricsData(
                    processor="tts",
                    model="sonic-english",
                    value=64,
                )
            ]
        ),
    )

    # ── Turn 3: user says thanks ──
    await asyncio.sleep(0.06)
    await push(
        observer,
        stt,
        TranscriptionFrame(
            text="Perfect, thanks!",
            user_id="caller-1",
            timestamp="2026-06-15T12:00:13.000Z",
        ),
    )
    await push(observer, stt, UserStoppedSpeakingFrame())

    # ── Final bot reply ──
    await asyncio.sleep(0.15)
    await push(observer, llm, LLMFullResponseStartFrame())
    await push(observer, llm, BotStartedSpeakingFrame())
    await push(observer, llm, TextFrame(text="You're welcome! Have a great day."))
    await push(
        observer,
        llm,
        MetricsFrame(
            data=[
                LLMUsageMetricsData(
                    processor="llm",
                    model="gpt-4o-mini",
                    value=LLMTokenUsage(
                        prompt_tokens=60,
                        completion_tokens=10,
                        total_tokens=70,
                    ),
                )
            ]
        ),
    )
    await push(
        observer,
        tts,
        MetricsFrame(
            data=[
                TTSUsageMetricsData(
                    processor="tts",
                    model="sonic-english",
                    value=30,
                )
            ]
        ),
    )

    # ── End of call ──
    await push(observer, llm, EndFrame())
    await observer.on_pipeline_finished()

    # ───────── Output ─────────
    print("=" * 70)
    print("SuperBryn Pipecat Observer — synthetic call complete")
    print("=" * 70)
    print(f"Webhook URL:     {captured['webhook_url']}")
    print(f"API key set:     {captured['api_key_present']}")
    print("X-API-Key header: SUPERBRYN_API_KEY (sb_test_dummy_key)")
    print()
    print("Payload that would be POSTed to SuperBryn:")
    print("-" * 70)
    print(json.dumps(captured["payload"], indent=2, default=str))
    print("-" * 70)

    # Quick sanity assertions so the script self-verifies
    call = captured["payload"]["call"]
    assert captured["payload"]["event"] == "call.completed"
    assert call["from_number"] == "+15551234567"
    assert call["to_number"] == "+15557654321"
    assert call["recording_url"] == "https://example.com/recordings/synthetic.mp3"
    assert call["metadata"]["agent_id"] == "synthetic-support-bot"
    assert call["metadata"]["agent_name"] == "synthetic-support-bot"
    assert call["metadata"]["transport"] == "websocket"
    assert call["metadata"]["campaign"] == "summer-promo"
    assert call["metadata"]["tenant_id"] == "acme-corp"
    assert call["metadata"]["llm_provider"] == "openai"
    assert call["metadata"]["llm_model"] == "gpt-4o-mini"
    assert call["metadata"]["stt_provider"] == "deepgram"
    assert call["metadata"]["tts_provider"] == "cartesia"
    assert call["usage"]["llm_input_tokens"] == 180 + 210 + 60
    assert call["usage"]["llm_output_tokens"] == 24 + 18 + 10
    assert call["usage"]["tts_characters"] == 52 + 64 + 30
    assert call["call_end_reason"] == "completed"
    assert len(call["transcript"]["turns"]) == 6  # 3 user + 3 bot
    print()
    print("✓ All payload assertions passed")
    print()


if __name__ == "__main__":
    asyncio.run(main())
