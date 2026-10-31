"""
superbryn-pipecat-observer: one call, attach(task, runner_args), reports each call of a Pipecat agent to SuperBryn.
See README.md for usage.
"""

from .call_report import CallReport, attach
from .config_sync import (
    BehaviorConfig,
    IdentityConfig,
    LanguageConfig,
    TelephonyConfig,
    ToolConfig,
    async_sync_config,
    build_manifest_from_pipeline,
    sync_config,
    sync_manifest,
)
from .prompt_sync import prompt_hash, prompt_ref, push_prompt

__version__ = "0.12.0"
__all__ = [
    "BehaviorConfig",
    "CallReport",
    "IdentityConfig",
    "LanguageConfig",
    "TelephonyConfig",
    "ToolConfig",
    "__version__",
    "async_sync_config",
    "attach",
    "build_manifest_from_pipeline",
    "prompt_hash",
    "prompt_ref",
    "push_prompt",
    "sync_config",
    "sync_manifest",
]
