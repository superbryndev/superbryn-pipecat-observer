"""Manifest extraction: direct services, wrappers, switchers, fallbacks, secrets, overrides."""

from __future__ import annotations

import json

import pytest

from superbryn_pipecat_observer.config_sync import (
    _LANGUAGE_ATTRS,
    _MAX_TOKENS_ATTRS,
    _MODEL_ATTRS,
    _TEMPERATURE_ATTRS,
    _VOICE_ATTRS,
    build_manifest_from_pipeline,
)

from .conftest import (
    FakeContext,
    FakeContextAggregator,
    FakeParallelPipeline,
    FakePipeline,
    FakeServiceSwitcher,
    FakeWrapper,
    make_llm,
    make_stt,
    make_tts,
)

# ── direct services ──────────────────────────────────────────────────────


def test_direct_services_full_pipeline():
    pipeline = FakePipeline([make_stt(), make_llm(), make_tts()])
    manifest = build_manifest_from_pipeline(pipeline)

    assert manifest["llm"] == {
        "provider": "openai",
        "model": "gpt-4o-mini",
        "temperature": 0.4,
        "max_tokens": 512,
    }
    assert manifest["stt"] == {"provider": "deepgram", "model": "nova-3", "language": "en"}
    assert manifest["tts"] == {
        "provider": "elevenlabs",
        "model": "eleven_turbo_v2",
        "voice_id": "voice-abc",
    }
    assert manifest["voice"] == {"provider": "elevenlabs", "voice_id": "voice-abc"}
    assert manifest["source"] == "pipecat"


def test_empty_pipeline_omits_blocks():
    manifest = build_manifest_from_pipeline(FakePipeline([]))
    for key in ("llm", "stt", "tts", "voice"):
        assert key not in manifest


# ── nesting, wrappers, switchers ─────────────────────────────────────────


def test_services_inside_nested_pipeline_are_found():
    inner = FakePipeline([make_llm()])
    outer = FakePipeline([make_stt(), inner, make_tts()])
    manifest = build_manifest_from_pipeline(outer)
    assert manifest["llm"]["provider"] == "openai"
    assert manifest["stt"]["provider"] == "deepgram"


def test_services_inside_parallel_pipeline_branches_are_found():
    parallel = FakeParallelPipeline([make_llm()], [make_tts()])
    pipeline = FakePipeline([make_stt(), parallel])
    manifest = build_manifest_from_pipeline(pipeline)
    assert manifest["llm"]["provider"] == "openai"
    assert manifest["tts"]["provider"] == "elevenlabs"


def test_custom_wrapper_is_unwrapped():
    wrapped = FakeWrapper(make_tts(provider="cartesia", model="sonic-2"), attr="_tts")
    manifest = build_manifest_from_pipeline(FakePipeline([wrapped]))
    assert manifest["tts"]["provider"] == "cartesia"
    assert manifest["voice"]["voice_id"] == "voice-abc"


def test_cyclic_structures_terminate():
    a = FakeWrapper(None)
    b = FakeWrapper(a)
    a._service = b  # wrapper cycle
    pipeline = FakePipeline([a])
    pipeline._processors.append(pipeline)  # pipeline cycle
    manifest = build_manifest_from_pipeline(pipeline)  # must not hang or raise
    assert manifest["source"] == "pipecat"


def test_llm_switcher_primary_and_fallback():
    switcher = FakeServiceSwitcher(
        [make_llm(provider="openai"), make_llm(provider="anthropic", model="claude-sonnet-4-5")]
    )
    manifest = build_manifest_from_pipeline(FakePipeline([make_stt(), switcher, make_tts()]))

    assert manifest["llm"]["provider"] == "openai"
    assert manifest["llm"]["fallback"]["provider"] == "anthropic"
    assert manifest["llm"]["fallback"]["model"] == "claude-sonnet-4-5"


def test_tts_switcher_fallback_reaches_voice_block():
    switcher = FakeServiceSwitcher(
        [
            make_tts(provider="elevenlabs", voice="v-primary"),
            make_tts(provider="cartesia", model="sonic-2", voice="v-fallback"),
        ]
    )
    manifest = build_manifest_from_pipeline(FakePipeline([switcher]))

    assert manifest["tts"]["voice_id"] == "v-primary"
    assert manifest["tts"]["fallback"] == {
        "provider": "cartesia",
        "model": "sonic-2",
        "voice_id": "v-fallback",
    }
    assert manifest["voice"] == {
        "provider": "elevenlabs",
        "voice_id": "v-primary",
        "fallback": {"provider": "cartesia", "voice_id": "v-fallback"},
    }


def test_single_member_switcher_has_no_fallback():
    switcher = FakeServiceSwitcher([make_stt()])
    manifest = build_manifest_from_pipeline(FakePipeline([switcher]))
    assert manifest["stt"]["provider"] == "deepgram"
    assert "fallback" not in manifest["stt"]


# ── secret non-extraction ────────────────────────────────────────────────

SECRET = "sk-live-EXTREMELY-SECRET-VALUE"


def test_secrets_never_reach_the_manifest():
    llm = make_llm(api_key=SECRET, _api_key=SECRET)
    llm._settings["api_key"] = SECRET
    stt = make_stt(_credentials=SECRET)
    tts = make_tts(token=SECRET)
    pipeline = FakePipeline([stt, llm, tts])

    manifest = build_manifest_from_pipeline(pipeline)
    assert SECRET not in json.dumps(manifest)


def test_attribute_allow_lists_contain_no_credential_names():
    forbidden = {
        "key",
        "apikey",
        "token",
        "secret",
        "password",
        "credential",
        "credentials",
        "auth",
    }
    for attrs in (
        _MODEL_ATTRS,
        _VOICE_ATTRS,
        _LANGUAGE_ATTRS,
        _TEMPERATURE_ATTRS,
        _MAX_TOKENS_ATTRS,
    ):
        for candidate in attrs:
            segments = candidate.lower().replace(".", "_").split("_")
            assert not (set(segments) & forbidden), f"{candidate!r} looks credential-shaped"


# ── context: prompt + tools ──────────────────────────────────────────────


def _pipeline_with_context(context):
    return FakePipeline([make_llm(), FakeContextAggregator(context)])


def test_prompt_and_tools_from_context():
    context = FakeContext(
        messages=[{"role": "system", "content": "You are a helpful support agent."}],
        tools=[
            {
                "type": "function",
                "function": {
                    "name": "book_slot",
                    "description": "Book an appointment slot",
                    "parameters": {"type": "object", "properties": {"slot": {"type": "string"}}},
                },
            }
        ],
    )
    manifest = build_manifest_from_pipeline(_pipeline_with_context(context))

    assert manifest["config"]["behavior"]["prompt"] == "You are a helpful support agent."
    tool = manifest["config"]["tools"][0]
    assert tool["name"] == "book_slot"
    assert tool["schema"]["properties"]["slot"]["type"] == "string"


# ── override precedence ──────────────────────────────────────────────────


def test_behavior_override_beats_context_prompt():
    context = FakeContext(messages=[{"role": "system", "content": "runtime prompt"}])
    manifest = build_manifest_from_pipeline(
        _pipeline_with_context(context), behavior={"prompt": "explicit override"}
    )
    assert manifest["config"]["behavior"]["prompt"] == "explicit override"


def test_tools_override_beats_context_tools():
    context = FakeContext(tools=[{"type": "function", "function": {"name": "runtime_tool"}}])
    manifest = build_manifest_from_pipeline(
        _pipeline_with_context(context), tools=[{"name": "explicit_tool"}]
    )
    assert manifest["config"]["tools"] == [{"name": "explicit_tool"}]


def test_extraction_failure_degrades_not_raises():
    class Exploding:
        def __getattr__(self, name):
            raise RuntimeError("boom")

    manifest = build_manifest_from_pipeline(Exploding())
    assert manifest["source"] == "pipecat"


# ── server-schema fixtures ───────────────────────────────────────────────
#
# Mirrors ManifestConfig in the orchestration service's agent-sync.types.ts.
# If these fail, the server schema and the client TypedDicts have drifted —
# update both together.

CANONICAL_IDENTITY_FIELDS = {
    "name",
    "type",
    "agent_modality",
    "description",
    "pain_point",
    "gender",
    "age",
    "dob",
}
CANONICAL_BEHAVIOR_FIELDS = {"prompt", "flow"}
CANONICAL_TOOL_FIELDS = {"name", "description", "schema", "server"}
CANONICAL_LANGUAGE_FIELDS = {"primary_language", "additional_languages"}
CANONICAL_TELEPHONY_FIELDS = {"phone_number", "ivr_config"}


def test_typeddicts_match_canonical_schema():
    from superbryn_pipecat_observer.config_sync import (
        BehaviorConfig,
        IdentityConfig,
        LanguageConfig,
        TelephonyConfig,
        ToolConfig,
    )

    assert set(IdentityConfig.__annotations__) == CANONICAL_IDENTITY_FIELDS
    assert set(BehaviorConfig.__annotations__) == CANONICAL_BEHAVIOR_FIELDS
    assert set(ToolConfig.__annotations__) == CANONICAL_TOOL_FIELDS
    assert set(LanguageConfig.__annotations__) == CANONICAL_LANGUAGE_FIELDS
    assert set(TelephonyConfig.__annotations__) == CANONICAL_TELEPHONY_FIELDS


def test_unknown_override_keys_raise_locally():
    pipeline = FakePipeline([])
    with pytest.raises(ValueError, match="unknown identity field"):
        build_manifest_from_pipeline(pipeline, identity={"nickname": "Bob"})
    with pytest.raises(ValueError, match="unknown tools"):
        build_manifest_from_pipeline(pipeline, tools=[{"name": "t", "endpoint": "x"}])
    with pytest.raises(TypeError):
        build_manifest_from_pipeline(pipeline, identity="not-a-dict")
