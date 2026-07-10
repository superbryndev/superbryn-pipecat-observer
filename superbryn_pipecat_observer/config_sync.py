"""
Agent config sync for superbryn-pipecat-observer (opt-in).

Builds a SuperBryn AgentSyncManifest from a running Pipecat pipeline and
pushes it to ``POST {api_base}/public-api/v1/agents/me/sync``. Nothing here
runs unless the customer explicitly calls ``sync_config`` /
``async_sync_config`` — attaching the observer alone never syncs.

Requires an **agent-scoped** API key (created against a single agent in the
SuperBryn dashboard); org-scoped keys are rejected by the endpoint. The
pushed manifest lands as a pending draft that a human approves in the
review UI — syncing never changes the live agent directly.

Extractors read public attributes only. API keys and other secrets held by
Pipecat service objects are never read or transmitted.
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Literal, TypedDict

from ._provider_detect import detect_provider_from_model
from .config import WEBHOOK_CONFIG

logger = logging.getLogger("superbryn.pipecat.sync")

SYNC_PATH = "/public-api/v1/agents/me/sync"


# ── config section shapes ────────────────────────────────────────────────
# These mirror the canonical AgentSyncManifest schema field-for-field. The
# sync endpoint validates strictly (unknown keys → HTTP 400), so the same
# key sets are enforced client-side in build_manifest_from_pipeline and a
# typo fails at the call site instead of at deploy time.


class IdentityConfig(TypedDict, total=False):
    name: str
    type: Literal["inbound", "outbound"]
    agent_modality: Literal["voice", "chat"]
    description: str
    pain_point: str | None
    gender: str | None
    age: int
    dob: str  # ISO date, e.g. "1990-01-31"


class BehaviorConfig(TypedDict, total=False):
    prompt: str | None
    flow: dict[str, Any] | None


class ToolServerConfig(TypedDict, total=False):
    type: Literal["http", "mcp", "native"]
    url: str


class ToolConfig(TypedDict, total=False):
    name: str
    description: str
    schema: Any
    server: ToolServerConfig


class AdditionalLanguage(TypedDict, total=False):
    code: str
    priority: int


class LanguageConfig(TypedDict, total=False):
    primary_language: str | None
    additional_languages: list[AdditionalLanguage]


class IvrConfig(TypedDict, total=False):
    enabled: bool
    number: str


class TelephonyConfig(TypedDict, total=False):
    phone_number: str | None
    ivr_config: IvrConfig | None


def _check_keys(section: str, value: Any, allowed: frozenset[str]) -> None:
    """Reject unknown keys locally — the server schema is strict and would 400."""
    if value is None:
        return
    if not isinstance(value, dict):
        raise TypeError(f"{section} must be a dict, got {type(value).__name__}")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise ValueError(
            f"unknown {section} field(s) {unknown}; allowed fields: {sorted(allowed)}. "
            "The SuperBryn sync endpoint rejects unknown keys."
        )


_IDENTITY_KEYS = frozenset(IdentityConfig.__annotations__)
_BEHAVIOR_KEYS = frozenset(BehaviorConfig.__annotations__)
_TOOL_KEYS = frozenset(ToolConfig.__annotations__)
_TOOL_SERVER_KEYS = frozenset(ToolServerConfig.__annotations__)
_LANGUAGE_KEYS = frozenset(LanguageConfig.__annotations__)
_ADDITIONAL_LANGUAGE_KEYS = frozenset(AdditionalLanguage.__annotations__)
_IVR_KEYS = frozenset(IvrConfig.__annotations__)
_TELEPHONY_KEYS = frozenset(TelephonyConfig.__annotations__)


def _validate_config_sections(
    identity: Any,
    behavior: Any,
    tools: Any,
    language: Any,
    telephony: Any,
) -> None:
    _check_keys("identity", identity, _IDENTITY_KEYS)
    _check_keys("behavior", behavior, _BEHAVIOR_KEYS)
    if tools is not None:
        for i, tool in enumerate(tools):
            _check_keys(f"tools[{i}]", tool, _TOOL_KEYS)
            if isinstance(tool, dict):
                _check_keys(f"tools[{i}].server", tool.get("server"), _TOOL_SERVER_KEYS)
    if language is not None:
        _check_keys("language", language, _LANGUAGE_KEYS)
        if isinstance(language, dict):
            for i, entry in enumerate(language.get("additional_languages") or []):
                _check_keys(f"language.additional_languages[{i}]", entry, _ADDITIONAL_LANGUAGE_KEYS)
    if telephony is not None:
        _check_keys("telephony", telephony, _TELEPHONY_KEYS)
        if isinstance(telephony, dict):
            _check_keys("telephony.ivr_config", telephony.get("ivr_config"), _IVR_KEYS)


# Keep in lock-step with SuperbrynObserver._MODULE_PROVIDER_MAP.
_MODULE_PROVIDER_MAP: dict[str, str] = {
    "anthropic": "anthropic",
    "openai": "openai",
    "google": "google",
    "gemini": "google",
    "azure": "azure",
    "groq": "groq",
    "together": "together",
    "deepgram": "deepgram",
    "assemblyai": "assemblyai",
    "cartesia": "cartesia",
    "elevenlabs": "elevenlabs",
    "rime": "rime",
    "playht": "playht",
}

_MODEL_ATTRS = ("model_name", "model", "_model", "_settings.model", "settings.model")
_VOICE_ATTRS = (
    "voice_id",
    "voice",
    "_voice_id",
    "_voice",
    "_settings.voice_id",
    "_settings.voice",
    "settings.voice_id",
    "settings.voice",
)
_TEMPERATURE_ATTRS = (
    "temperature",
    "_temperature",
    "_settings.temperature",
    "settings.temperature",
)
_MAX_TOKENS_ATTRS = (
    "max_tokens",
    "_max_tokens",
    "_settings.max_tokens",
    "settings.max_tokens",
)
_LANGUAGE_ATTRS = (
    "language",
    "_language",
    "_settings.language",
    "settings.language",
)


def _read_attr_chain(source: Any, *names: str) -> Any:
    """First non-empty attribute along dotted candidate paths (never raises).

    Handles both attribute access and dict lookups so pipecat services that
    keep ``_settings`` as a plain dict are covered too.
    """
    for name in names:
        cur: Any = source
        for part in name.split("."):
            if isinstance(cur, dict):
                cur = cur.get(part)
            else:
                cur = getattr(cur, part, None)
            if cur is None:
                break
        if cur not in (None, ""):
            return cur
    return None


def _as_str(value: Any) -> str | None:
    """Coerce plain strings and enums (e.g. pipecat Language) to str."""
    if isinstance(value, str):
        return value or None
    inner = getattr(value, "value", None)
    if isinstance(inner, str):
        return inner or None
    return None


def _provider_from_module(module: str) -> str | None:
    for needle, name in _MODULE_PROVIDER_MAP.items():
        if needle in module:
            return name
    return None


def _extract_service(processor: Any, role: str) -> dict[str, Any] | None:
    """Extract a {provider, model[, voice_id]} block from a pipecat service."""
    module = type(processor).__module__ or ""
    if "pipecat.services." not in module or role not in module:
        return None

    block: dict[str, Any] = {}
    model = _read_attr_chain(processor, *_MODEL_ATTRS)
    provider = _provider_from_module(module)
    if not provider and isinstance(model, str):
        detected = detect_provider_from_model(model)
        provider = detected if detected != "unknown" else None

    if provider:
        block["provider"] = provider
    if isinstance(model, str) and model:
        block["model"] = model

    if role == "llm":
        temperature = _read_attr_chain(processor, *_TEMPERATURE_ATTRS)
        if isinstance(temperature, (int, float)) and not isinstance(temperature, bool):
            block["temperature"] = temperature
        max_tokens = _read_attr_chain(processor, *_MAX_TOKENS_ATTRS)
        if isinstance(max_tokens, int) and not isinstance(max_tokens, bool):
            block["max_tokens"] = max_tokens

    if role == "stt":
        language = _as_str(_read_attr_chain(processor, *_LANGUAGE_ATTRS))
        if language:
            block["language"] = language

    if role == "tts":
        voice = _as_str(_read_attr_chain(processor, *_VOICE_ATTRS))
        if voice:
            block["voice_id"] = voice

    return block or None


def _walk_processors(pipeline: Any) -> list[Any]:
    """Flatten a Pipecat pipeline into a processor list (best effort)."""
    processors = getattr(pipeline, "_processors", None) or getattr(pipeline, "processors", None)
    if not processors:
        return []
    flat: list[Any] = []
    for p in processors:
        flat.append(p)
        nested = getattr(p, "_processors", None)
        if nested:
            flat.extend(nested)
    return flat


def _find_llm_context(pipeline: Any) -> Any:
    """Locate the LLMContext held by a context-aggregator processor, if any."""
    for processor in _walk_processors(pipeline):
        for attr in ("context", "_context"):
            context = getattr(processor, attr, None)
            if context is not None and (
                hasattr(context, "get_messages") or hasattr(context, "messages")
            ):
                return context
    return None


def _extract_prompt_from_context(context: Any) -> str | None:
    """The system prompt from the context's message list, if present."""
    try:
        messages = context.get_messages()
    except Exception:  # noqa: BLE001
        messages = getattr(context, "messages", None)
    if not messages:
        return None
    for message in messages:
        role = message.get("role") if isinstance(message, dict) else getattr(message, "role", None)
        if role != "system":
            continue
        content = (
            message.get("content")
            if isinstance(message, dict)
            else getattr(message, "content", None)
        )
        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):  # multi-part content — join the text parts
            texts = [
                part.get("text")
                for part in content
                if isinstance(part, dict) and isinstance(part.get("text"), str)
            ]
            joined = "\n".join(t for t in texts if t)
            if joined.strip():
                return joined
    return None


def _extract_tools_from_context(context: Any) -> list[ToolConfig] | None:
    """Manifest tool entries from the context's tools (ToolsSchema or OpenAI-style list)."""
    tools_obj = getattr(context, "tools", None)
    entries: list[ToolConfig] = []

    standard = getattr(tools_obj, "standard_tools", None)
    if standard:
        for schema in standard:
            entry: ToolConfig = {}
            name = getattr(schema, "name", None)
            if isinstance(name, str) and name:
                entry["name"] = name
            description = getattr(schema, "description", None)
            if isinstance(description, str) and description:
                entry["description"] = description
            to_dict = getattr(schema, "to_default_dict", None)
            if callable(to_dict):
                try:
                    parameters = to_dict().get("parameters")
                    if parameters is not None:
                        entry["schema"] = parameters
                except Exception:  # noqa: BLE001
                    pass
            if entry:
                entries.append(entry)
    elif isinstance(tools_obj, list):
        for tool in tools_obj:
            if not isinstance(tool, dict):
                continue
            fn = tool.get("function") if tool.get("type") == "function" else tool
            if not isinstance(fn, dict):
                continue
            entry = {}
            if isinstance(fn.get("name"), str) and fn["name"]:
                entry["name"] = fn["name"]
            if isinstance(fn.get("description"), str) and fn["description"]:
                entry["description"] = fn["description"]
            if fn.get("parameters") is not None:
                entry["schema"] = fn["parameters"]
            if entry:
                entries.append(entry)

    return entries or None


def build_manifest_from_pipeline(
    pipeline: Any,
    *,
    source: str = "pipecat",
    identity: IdentityConfig | None = None,
    behavior: BehaviorConfig | None = None,
    tools: list[ToolConfig] | None = None,
    language: LanguageConfig | None = None,
    telephony: TelephonyConfig | None = None,
    policy_guardrails: str | None = None,
    additional_details: str | None = None,
    concurrency_calls: int | None = None,
    scan_root: str | None = None,
) -> dict[str, Any]:
    """Build an AgentSyncManifest dict from a Pipecat pipeline.

    Walks the pipeline's processors and fills the top-level ``llm`` /
    ``stt`` / ``tts`` / ``voice`` blocks from whatever services it finds.
    Also reads the LLM context (via the context aggregators): the system
    message becomes ``config.behavior.prompt`` and the advertised function
    schemas become ``config.tools`` — unless explicit ``behavior`` /
    ``tools`` overrides are given, which always win.

    Everything the pipeline genuinely can't know (identity, telephony,
    guardrails, concurrency, ...) is supplied through the keyword
    overrides — or discovered by a static source scan when ``scan_root``
    (a file or project directory) is given. Precedence per section:
    explicit keyword override > runtime extraction > source scan. See
    :mod:`superbryn_pipecat_observer.codescan` for what the scan looks
    for (``agent_name=``, ``phone_number=``, ``POLICY_GUARDRAILS = ...``,
    ``concurrency_calls=``, ...).

    Override shapes mirror the canonical manifest schema exactly (see the
    TypedDicts at the top of this module):

    - ``identity``: ``name``, ``type`` ("inbound"/"outbound"),
      ``agent_modality`` ("voice"/"chat"), ``description``, ``pain_point``,
      ``gender``, ``age``, ``dob``
    - ``behavior``: ``prompt``, ``flow``
    - ``tools``: list of ``{name, description, schema, server}`` where
      ``server`` is ``{type: http|mcp|native, url}``
    - ``language``: ``primary_language``, ``additional_languages``
      (list of ``{code, priority}``)
    - ``telephony``: ``phone_number``, ``ivr_config``
      (``{enabled, number}``)

    Unknown keys in these overrides raise ``ValueError`` here — the sync
    endpoint validates strictly, so this surfaces typos at the call site
    instead of as an HTTP 400.

    All manifest fields are optional — an empty manifest is valid, so
    extraction failures degrade to a sparser manifest, never an error.
    """
    _validate_config_sections(identity, behavior, tools, language, telephony)

    manifest: dict[str, Any] = {"source": source}

    try:
        for processor in _walk_processors(pipeline):
            for role in ("llm", "stt", "tts"):
                if role in manifest:
                    continue
                block = _extract_service(processor, role)
                if block:
                    manifest[role] = block
    except Exception as exc:  # noqa: BLE001 — never break the customer's agent
        logger.debug("pipeline extraction failed: %s", exc)

    if behavior is None or tools is None:
        try:
            context = _find_llm_context(pipeline)
            if context is not None:
                if behavior is None:
                    prompt = _extract_prompt_from_context(context)
                    if prompt:
                        behavior = {"prompt": prompt}
                if tools is None:
                    tools = _extract_tools_from_context(context)
        except Exception as exc:  # noqa: BLE001 — never break the customer's agent
            logger.debug("context extraction failed: %s", exc)

    if scan_root is not None:
        try:
            from .codescan import scan_source_config

            scanned = scan_source_config(scan_root)
            if identity is None and "identity" in scanned:
                identity = scanned["identity"]
            if behavior is None and "behavior" in scanned:
                behavior = scanned["behavior"]
            if telephony is None and "telephony" in scanned:
                telephony = scanned["telephony"]
            if policy_guardrails is None and "policy_guardrails" in scanned:
                policy_guardrails = scanned["policy_guardrails"]
            if additional_details is None and "additional_details" in scanned:
                additional_details = scanned["additional_details"]
            if concurrency_calls is None and "concurrency_calls" in scanned:
                concurrency_calls = scanned["concurrency_calls"]
        except Exception as exc:  # noqa: BLE001 — never break the customer's agent
            logger.debug("source scan failed: %s", exc)

    tts_block = manifest.get("tts")
    if isinstance(tts_block, dict) and tts_block.get("voice_id"):
        manifest["voice"] = {
            k: v
            for k, v in (
                ("provider", tts_block.get("provider")),
                ("voice_id", tts_block["voice_id"]),
            )
            if v
        }

    config: dict[str, Any] = {}
    if identity is not None:
        config["identity"] = identity
    if behavior is not None:
        config["behavior"] = behavior
    if tools is not None:
        config["tools"] = tools
    if language is not None:
        config["language"] = language
    if telephony is not None:
        config["telephony"] = telephony
    if policy_guardrails is not None:
        config["policy_guardrails"] = policy_guardrails
    if additional_details is not None:
        config["additional_details"] = additional_details
    if concurrency_calls is not None:
        config["concurrency_calls"] = concurrency_calls
    if config:
        manifest["config"] = config

    return manifest


def _resolve_endpoint(base_url: str | None) -> str:
    base = (base_url or WEBHOOK_CONFIG.get("api_base_url") or "").rstrip("/")
    if not base:
        raise ValueError(
            "SuperBryn API base URL is not configured — set SUPERBRYN_API_BASE_URL "
            "or pass base_url= to sync_config()"
        )
    return base + SYNC_PATH


def sync_manifest(
    manifest: dict[str, Any],
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
) -> dict[str, Any]:
    """Push a prebuilt manifest to SuperBryn (blocking).

    Returns the parsed JSON response: ``{status: "noop", ...}`` for a
    hash-identical sync, or the accepted-draft body with ``agent_row_id``,
    ``approval_status``, ``verification_status``, ``hash`` and
    ``change_types``. Raises on transport errors and non-2xx responses so
    deploy pipelines fail loudly.
    """
    key = api_key or WEBHOOK_CONFIG.get("api_key") or ""
    if not key:
        raise ValueError("SuperBryn API key missing — set SUPERBRYN_API_KEY or pass api_key=")

    url = _resolve_endpoint(base_url)
    body = json.dumps(manifest).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-API-Key": key,
            "User-Agent": "superbryn-pipecat-observer",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        logger.error("SUPERBRYN_SYNC_FAILED: HTTP %s — %s", exc.code, detail)
        raise
    logger.info("SUPERBRYN_SYNC_OK: %s", payload.get("status") or payload.get("approval_status"))
    return payload


def sync_config(
    pipeline: Any,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    **manifest_overrides: Any,
) -> dict[str, Any]:
    """One-liner: build a manifest from the pipeline and push it.

    ``manifest_overrides`` are forwarded to
    :func:`build_manifest_from_pipeline` (``identity=``, ``behavior=``,
    ``policy_guardrails=``, ...).
    """
    manifest = build_manifest_from_pipeline(pipeline, **manifest_overrides)
    return sync_manifest(manifest, api_key=api_key, base_url=base_url, timeout=timeout)


async def async_sync_config(
    pipeline: Any,
    *,
    api_key: str | None = None,
    base_url: str | None = None,
    timeout: float = 30.0,
    **manifest_overrides: Any,
) -> dict[str, Any]:
    """Async variant of :func:`sync_config` using aiohttp."""
    import aiohttp  # lazy — matches the observer's send path

    key = api_key or WEBHOOK_CONFIG.get("api_key") or ""
    if not key:
        raise ValueError("SuperBryn API key missing — set SUPERBRYN_API_KEY or pass api_key=")

    manifest = build_manifest_from_pipeline(pipeline, **manifest_overrides)
    url = _resolve_endpoint(base_url)
    headers = {
        "Content-Type": "application/json",
        "X-API-Key": key,
        "User-Agent": "superbryn-pipecat-observer",
    }
    client_timeout = aiohttp.ClientTimeout(total=timeout)
    async with aiohttp.ClientSession(timeout=client_timeout) as session:
        async with session.post(url, json=manifest, headers=headers) as response:
            payload = await response.json()
            if response.status >= 400:
                logger.error("SUPERBRYN_SYNC_FAILED: HTTP %s — %s", response.status, payload)
                response.raise_for_status()
    logger.info("SUPERBRYN_SYNC_OK: %s", payload.get("status") or payload.get("approval_status"))
    return payload
