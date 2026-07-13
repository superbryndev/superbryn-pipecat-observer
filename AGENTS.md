# superbryn-pipecat-observer — Agent Guide

Public PyPI package (`superbryn-pipecat-observer`) that observes Pipecat
agent sessions (call payloads, transcripts, usage/latency metrics, audio
recording) and optionally syncs agent config to SuperBryn as a reviewable
draft. Python >= 3.11.

## Layout

```text
superbryn_pipecat_observer/
  observer.py          # SuperbrynObserver — frame observation + webhook delivery
  audio_recorder.py    # recording via presigned S3 upload
  config_sync.py       # opt-in agent config sync (manifest build + push)
  _provider_detect.py  # provider name detection from model/module strings
  config.py            # env-driven defaults (SUPERBRYN_API_KEY, ...)
tests/                 # pytest suite — no network, fakes instead of live pipelines
test_import.py         # root-level smoke script (also run in CI)
```

## Non-negotiable invariants

- **No secrets are ever extracted or transmitted.** Config-sync extraction
  reads a fixed allow-list of configuration attributes (which includes
  private fields like `_settings` / `_model` / `_voice_id`); credential
  attributes (api_key, token, secret) must never be added to any candidate
  list. `tests/` locks this down.
- **Nothing syncs implicitly.** Config sync only runs when the customer
  calls `sync_config` / `async_sync_config` explicitly. Attaching the
  observer must never trigger a sync.
- **Extraction failures degrade, never raise.** A sparser manifest is fine;
  breaking the customer's agent is not.
- **No project-wide source scanning.** Static scanning of customer source
  trees (`scan_root`) was removed in 0.8.0 (confidentiality risk); do not
  reintroduce it. Missing sections are supplied via explicit overrides.
- **Composite components must be walked, wrappers unwrapped.** The pipeline
  walk recurses through `Pipeline` / `ParallelPipeline` / `ServiceSwitcher` /
  `LLMSwitcher` (cycle-safe) and `_unwrap_service` descends custom wrappers;
  switcher fallbacks are represented in the manifest's `fallback` sub-blocks.

## Versioning and publishing

- `pyproject.toml` is the version source of truth; `.bumpversion.cfg` keeps
  it, `observer.py`'s `__version__`, and the README badge in lockstep
  (`bump2version`).
- The publish workflow (`.github/workflows/publish.yml`) publishes exactly
  the declared version on merge to `main`, skips if already on PyPI, and
  requires a matching `## [X.Y.Z]` section in `CHANGELOG.md`. Never
  reintroduce commit-message-driven auto-bumps.

## Development

```bash
pip install -e ".[dev]"
pytest -q                     # runs tests/ and test_import.py; no network needed
ruff format && ruff check superbryn_pipecat_observer tests test_import.py
```

- The manifest schema mirrors the orchestration service's
  `AgentSyncManifest` (strict server-side validation). If the schema changes
  upstream, update the TypedDicts in `config_sync.py` and the tests together.
- Update `CHANGELOG.md` for any behavior change.
