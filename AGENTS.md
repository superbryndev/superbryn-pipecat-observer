# superbryn-pipecat-observer — Agent Guide

Public PyPI package (`superbryn-pipecat-observer`) that reports each call of
a Pipecat agent to SuperBryn with one call, `attach(task, runner_args)`:
one report per call (`POST /public-api/v1/calls/agent-data`), surface
`testing` for a SuperBryn test call and `observability` (with an in-pipeline
stereo recording) for every other call. It also syncs agent config to
SuperBryn as a reviewable draft, on request. Python >= 3.11.

## Layout

```text
superbryn_pipecat_observer/
  call_report.py       # attach() + CallReport: capture, recognition, the call report (both surfaces)
  audio_recorder.py    # the stereo recorder attach() puts before the output transport
  s3_uploader.py       # the recording's presigned S3 upload
  prompt_sync.py       # prompt_ref + the prompt push
  config_sync.py       # opt-in agent config sync (manifest build + push)
  _provider_detect.py  # provider name detection from model strings
  _caller_speech.py    # caller speech from raw audio, for pipelines with no VAD frames
  config.py            # env defaults for config sync (SUPERBRYN_API_KEY, SUPERBRYN_BASE_URL)
tests/                 # pytest suite — no network (conftest refuses it), fakes and real pipelines
test_import.py         # root-level smoke script (also run in CI)
```

## Non-negotiable invariants

- **No secrets are ever extracted or transmitted.** Config-sync extraction
  reads a fixed allow-list of configuration attributes (which includes
  private fields like `_settings` / `_model` / `_voice_id`); credential
  attributes (api_key, token, secret) must never be added to any candidate
  list. `tests/` locks this down.
- **Nothing syncs implicitly.** Config sync only runs when the customer
  calls `sync_config` / `async_sync_config` explicitly. `attach()` must never
  trigger a config sync (its prompt push runs only with `prompt=`).
- **The report never breaks a call.** `attach()` never raises, sends in the
  background, and every body section is built on its own (one that fails is
  left out). The body follows the server contract
  (`orchestration-service-v2/src/modules/agent-data-ingest/agent-data-ingest.schema.ts`):
  unknown top-level keys are dropped there, so customer keys go in
  `custom_data`, as given. No package-side averages: raw records only.
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
  it, `__init__.py`'s `__version__`, and the README badge in lockstep
  (`bump2version`).
- The publish workflow (`.github/workflows/publish.yml`) publishes exactly
  the declared version on merge to `main`, skips if already on PyPI, and
  requires a matching `## [X.Y.Z]` section in `CHANGELOG.md`. Never
  reintroduce commit-message-driven auto-bumps.

## Development

```bash
pip install -e ".[dev]"
pytest -q                     # runs tests/ and test_import.py; no network needed
ruff format && ruff check superbryn_pipecat_observer tests test_import.py examples/pipeline_check
python examples/pipeline_check/run.py   # real Pipecat pipeline, every body field checked, ~2 min; CI runs it
```

- `examples/pipeline_check/` runs a scripted call through a real Pipecat pipeline
  (aggregators, function calls, output transport, the recorder attach() inserts)
  with stand-in STT / LLM / TTS, and checks every report field against what
  actually happened, as a production call and as a test call. A capture change
  that passes the unit tests but miscounts in a real pipeline (once per hop, say)
  fails here.
- `SUPERBRYN_PIPECAT_DUMP_DIR=<dir> pytest tests/test_call_report_bodies.py` writes
  whole-call bodies for both surfaces, for checking against the server.
- `examples/pipeline_check/real_services.py` runs the same check against real
  Deepgram, OpenAI and Cartesia (keys needed, a few cents a run, not in CI).
  Run it before a release that changes capture: `--env-file <.env>`, and once
  with `--barge-in`.

- The manifest schema mirrors the orchestration service's
  `AgentSyncManifest` (strict server-side validation). If the schema changes
  upstream, update the TypedDicts in `config_sync.py` and the tests together.
- Update `CHANGELOG.md` for any behavior change.
