"""
Configuration for superbryn-pipecat-observer.

Defaults are wired to the SuperBryn ingest endpoint. Every value may be
overridden via environment variables — useful for staging, on-prem, or
local development against a different backend.

Mirrors the env-var contract used by `livekit-evals` so customers running
both integrations see a single consistent set of variables.
"""

import os
from urllib.parse import urlsplit, urlunsplit

_DEFAULT_WEBHOOK_URL = "https://api.superbryn.com/webhooks/obs/pipecat"


def _derive_api_base_url(webhook_url: str) -> str:
    """Best-effort derivation of the SuperBryn API root from the webhook URL.

    The S3-credential broker lives at ``{api_base_url}/api/recording-credentials``
    on the same host as the webhook ingest endpoint. Customers can override
    explicitly via ``SUPERBRYN_API_BASE_URL`` for split-host deployments.
    """
    try:
        parts = urlsplit(webhook_url)
        if not parts.scheme or not parts.netloc:
            return ""
        return urlunsplit((parts.scheme, parts.netloc, "", "", ""))
    except Exception:
        return ""


WEBHOOK_CONFIG = {
    "url": os.getenv("SUPERBRYN_WEBHOOK_URL") or os.getenv("WEBHOOK_URL") or _DEFAULT_WEBHOOK_URL,
    "api_key": os.getenv("SUPERBRYN_API_KEY", ""),
}

WEBHOOK_CONFIG["api_base_url"] = os.getenv("SUPERBRYN_API_BASE_URL") or _derive_api_base_url(
    WEBHOOK_CONFIG["url"]
)

_DEFAULT_AGENT_ID = "pipecat-agent"
_DEFAULT_VERSION_ID = "v1"

AGENT_CONFIG = {
    "id": os.getenv("AGENT_ID", _DEFAULT_AGENT_ID),
    "version_id": os.getenv("VERSION_ID", _DEFAULT_VERSION_ID),
}


def is_configured() -> bool:
    """Return True iff we have both a webhook URL and an API key to send with."""
    return bool(WEBHOOK_CONFIG["url"] and WEBHOOK_CONFIG["api_key"])
