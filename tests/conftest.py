"""Shared fakes for manifest-extraction tests.

Pipecat services are simulated by setting ``__module__`` on fake classes to
``pipecat.services.<provider>.<role>`` — that is what extraction keys on, so
tests don't need live service instances (which require API keys).
"""

from __future__ import annotations

from types import SimpleNamespace


def make_service(provider: str, role: str, **attrs):
    cls = type(f"Fake{role.upper()}", (), {})
    cls.__module__ = f"pipecat.services.{provider}.{role}"
    obj = cls()
    for key, value in attrs.items():
        setattr(obj, key, value)
    return obj


def make_llm(provider: str = "openai", model: str = "gpt-4o-mini", **attrs):
    return make_service(
        provider,
        "llm",
        model_name=model,
        _settings={"temperature": 0.4, "max_tokens": 512},
        **attrs,
    )


def make_stt(provider: str = "deepgram", model: str = "nova-3", language: str = "en", **attrs):
    return make_service(
        provider, "stt", model_name=model, _settings={"language": language}, **attrs
    )


def make_tts(
    provider: str = "elevenlabs", model: str = "eleven_turbo_v2", voice: str = "voice-abc", **attrs
):
    return make_service(provider, "tts", model_name=model, _voice_id=voice, **attrs)


class FakePipeline:
    """Mimics pipecat.pipeline.Pipeline — children in ``_processors``."""

    def __init__(self, processors) -> None:
        self._processors = list(processors)


class FakeParallelPipeline:
    """Mimics ParallelPipeline — branches in ``_pipelines``, not ``_processors``."""

    def __init__(self, *branches) -> None:
        self._pipelines = [FakePipeline(branch) for branch in branches]


class FakeServiceSwitcher(FakeParallelPipeline):
    """Mimics ServiceSwitcher/LLMSwitcher — members in ``_services``, branches
    wrapping each service as Filter → Service → Filter inside a Pipeline."""

    def __init__(self, services) -> None:
        filt = SimpleNamespace()
        super().__init__(*[[filt, service, filt] for service in services])
        self._services = list(services)


class FakeWrapper:
    """Mimics a customer wrapper class holding the real service."""

    def __init__(self, inner, attr: str = "_service") -> None:
        setattr(self, attr, inner)


class FakeContext:
    """Mimics an LLMContext with messages and OpenAI-style tools."""

    def __init__(self, messages=None, tools=None) -> None:
        self._messages = messages or []
        self.tools = tools

    def get_messages(self):
        return self._messages


class FakeContextAggregator:
    def __init__(self, context) -> None:
        self._context = context
