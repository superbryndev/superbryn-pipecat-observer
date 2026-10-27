"""Provider names from model strings: Bedrock / Nova Sonic, Deepgram Nova and Cartesia Sonic kept apart."""

from __future__ import annotations

import pytest

from superbryn_pipecat_observer._provider_detect import detect_provider_from_model


@pytest.mark.parametrize(
    ("model", "provider"),
    [
        ("amazon.nova-2-5-sonic", "aws"),
        ("amazon.nova-sonic-v1:0", "aws"),
        ("amazon.nova-2-sonic-v1:0", "aws"),
        ("nova-sonic", "aws"),
        ("us.amazon.nova-pro-v1:0", "aws"),
        ("bedrock/amazon.titan-text", "aws"),
        ("polly-neural", "aws"),
        ("nova-2", "deepgram"),
        ("nova-3", "deepgram"),
        ("nova-3-general", "deepgram"),
        ("aura-2-thalia-en", "deepgram"),
        ("sonic-2", "cartesia"),
        ("sonic-english", "cartesia"),
        ("sonic-turbo", "cartesia"),
        ("claude-sonnet-4-5", "anthropic"),
        ("gpt-realtime", "openai"),
        ("gemini-2.5-flash-native-audio", "google"),
    ],
)
def test_provider_from_model(model, provider):
    assert detect_provider_from_model(model) == provider
