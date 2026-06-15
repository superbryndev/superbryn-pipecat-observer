"""
Configuration for superbryn-pipecat-observer.

Defaults are wired to the SuperBryn ingest endpoint. Every value may be
overridden via environment variables — useful for staging, on-prem, or
local development against a different backend.

Mirrors the env-var contract used by `livekit-evals` so customers running
both integrations see a single consistent set of variables.
"""

import os

_DEFAULT_WEBHOOK_URL = "https://api.superbryn.com/webhooks/obs/pipecat"

WEBHOOK_CONFIG = {
    "url": os.getenv("SUPERBRYN_WEBHOOK_URL") or os.getenv("WEBHOOK_URL") or _DEFAULT_WEBHOOK_URL,
    "api_key": os.getenv("SUPERBRYN_API_KEY", ""),
}

_DEFAULT_AGENT_ID = "pipecat-agent"
_DEFAULT_VERSION_ID = "v1"

AGENT_CONFIG = {
    "id": os.getenv("AGENT_ID", _DEFAULT_AGENT_ID),
    "version_id": os.getenv("VERSION_ID", _DEFAULT_VERSION_ID),
}


def is_configured() -> bool:
    """Return True iff we have both a webhook URL and an API key to send with."""
    return bool(WEBHOOK_CONFIG["url"] and WEBHOOK_CONFIG["api_key"])
