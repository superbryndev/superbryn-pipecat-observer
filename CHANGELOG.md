# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.8.0] - 2026-07-13

### Added
- Config-sync extraction now handles wrapped and composite pipeline components. The pipeline walk recurses through nested `Pipeline` / `ParallelPipeline` / `ServiceSwitcher` / `LLMSwitcher` structures (cycle-safe) instead of stopping one level deep, and custom wrapper classes holding the real service in an inner attribute (`_service`, `_llm`, `_inner`, ...) are unwrapped before extraction. Agents using switchers previously produced manifests with **no** `llm`/`stt`/`tts`/`voice` blocks at all.
- Fallback representation: for `ServiceSwitcher` / `LLMSwitcher`, the primary (initially active) member fills the role block and the next member is reported in the manifest's `fallback` sub-block (`llm`/`stt`/`tts`, and the derived `voice` block).
- Test suite (`tests/`) covering manifest extraction (direct, wrapped, switchers, fallbacks), secret non-extraction, override precedence, schema validation, HTTP error handling (blocking and async), and package exports. The CI pytest step now fails the build (previously `continue-on-error: true`).

### Changed
- Publishing is no longer commit-message driven. The publish workflow publishes exactly the version declared in `pyproject.toml` (skipping if already on PyPI) and refuses to publish without a matching `CHANGELOG.md` section.
- Documentation no longer claims extraction reads "public attributes only". Extraction reads a fixed allow-list of configuration attributes that includes private fields (`_settings`, `_model`, `_voice_id`, ...); credential attributes are never in that list. Docs now say so explicitly.

### Removed
- **Breaking:** static source scanning (`scan_root=` on `sync_config` / `build_manifest_from_pipeline`, and the `scan_source_config` export). Walking a whole project tree and uploading the longest prompt-like string risked syncing unrelated or confidential source content. Sections the runtime can't expose are supplied via explicit keyword overrides instead.

## [0.7.1] - 2026-07-09

### Added
- Agent config sync (opt-in): `build_manifest_from_pipeline`, `sync_manifest`, `sync_config`, `async_sync_config`, pushing an AgentSyncManifest to `POST {api_base}/public-api/v1/agents/me/sync` with an agent-scoped API key. Included the now-removed `scan_source_config` static scanner.

## [0.6.x and earlier]

Versions before 0.7.1 predate this changelog. They cover the core observer
(call payload delivery, transcripts, usage/latency metrics, audio recording
via presigned S3 upload, log capture). See the git history for details.

[Unreleased]: https://github.com/superbryndev/superbryn-pipecat-observer/compare/v0.8.0...HEAD
[0.8.0]: https://github.com/superbryndev/superbryn-pipecat-observer/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/superbryndev/superbryn-pipecat-observer/releases/tag/v0.7.1
