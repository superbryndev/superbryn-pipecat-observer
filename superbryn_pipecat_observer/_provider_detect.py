"""
Provider detection from model / module / voice strings.

Lifted from livekit-evals so both SDKs stamp identical provider names in the
normalized call payload (lets the SuperBryn analytics roll up
"OpenAI calls across all sources" consistently).
"""


def detect_provider_from_model(model_name: str | None) -> str:
    """Best-effort provider tag from a model / voice / module string."""
    if not model_name:
        return "unknown"

    s = model_name.lower()

    # ── LLMs ────────────────────────────────────────────────────────────────
    if any(x in s for x in ("gpt", "openai", "whisper", "tts-1", "o1-", "o3-")):
        return "openai"
    if any(x in s for x in ("claude", "anthropic")):
        return "anthropic"
    if any(x in s for x in ("gemini", "palm", "bard", "gemma")):
        return "google"
    if any(x in s for x in ("llama", "meta-llama")):
        return "meta"
    if any(x in s for x in ("mistral", "mixtral")):
        return "mistral"
    if any(x in s for x in ("cohere", "command")):
        return "cohere"
    if "groq" in s:
        return "groq"
    if any(x in s for x in ("together", "togethercomputer")):
        return "together"
    if "perplexity" in s or "pplx" in s:
        return "perplexity"

    # ── TTS ─────────────────────────────────────────────────────────────────
    if any(x in s for x in ("eleven", "elevenlabs")):
        return "elevenlabs"
    if any(x in s for x in ("cartesia", "sonic")):
        return "cartesia"
    if any(x in s for x in ("playht", "play.ht", "play-ht")):
        return "playht"
    if "rime" in s:
        return "rime"
    if any(x in s for x in ("saarika", "sarvam", "bulbul")):
        return "sarvam"
    if "azure" in s:
        return "azure"
    if any(x in s for x in ("polly", "amazon")):
        return "aws"

    # ── STT ─────────────────────────────────────────────────────────────────
    if any(x in s for x in ("deepgram", "nova", "aura")):
        return "deepgram"
    if any(x in s for x in ("assemblyai", "assembly")):
        return "assemblyai"
    if "speechmatics" in s:
        return "speechmatics"
    if "gladia" in s:
        return "gladia"

    # ── Realtime / transport ────────────────────────────────────────────────
    if "daily" in s:
        return "daily"
    if "twilio" in s:
        return "twilio"
    if "livekit" in s:
        return "livekit"

    return "unknown"


def detect_provider_from_module(module_path: str | None) -> str:
    """
    Extract provider name from a Pipecat service's __module__.

    Pipecat services live under `pipecat.services.<provider>.<llm|stt|tts>`,
    e.g. `pipecat.services.deepgram.stt` → "deepgram".
    """
    if not module_path:
        return "unknown"
    parts = module_path.split(".")
    if "pipecat" in parts and "services" in parts:
        try:
            idx = parts.index("services")
            if idx + 1 < len(parts):
                return parts[idx + 1]
        except ValueError:
            pass
    return parts[-1] if parts else "unknown"
