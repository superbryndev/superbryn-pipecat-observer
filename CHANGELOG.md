# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed
- **STT duration was silently zero on Pipecat >= 1.6, and took LLM/TTS usage down with it.** `STTUsageMetricsData.value` became an `STTUsage` object (`.audio_seconds`) where it used to be a bare float; the observer's `float(...)` coercion raised `TypeError` inside the `MetricsFrame` record loop, which aborted the loop — so any LLM or TTS record positioned after the STT record in the same frame was dropped too, and the whole failure was swallowed by `on_push_frame`'s catch-all. Usage now reads either shape, and each record is parsed in isolation so one unparseable record can't discard the rest of the batch. This fix applies regardless of `extended_capture`.

### Added
- **Extended capture** (`extended_capture=True`, on by default): the observer now extracts the maximum telemetry Pipecat's frame stream exposes and emits it as **additive** sections on the call payload. All existing fields are unchanged; set `extended_capture=False` for the exact legacy payload. Section names deliberately mirror `livekit-evals` so both SDKs land the same shape server-side.
  - `call.turn_detection` — turn-completion predictions from `TurnMetricsData`: average/max end-to-end processing time (VAD speech-to-silence → turn complete), prediction count, incomplete predictions, average confidence, and a bounded event list (≤200). This is Pipecat's end-of-turn signal and the biggest previously-invisible slice of response latency.
  - `call.latency` (new keys) — real per-service timings from the metrics frames the observer previously ignored entirely: `avg/p95_ttfb_ms`, `avg_ttfa_ms` plus `avg_tts_leading_silence_ms` (how much of TTS latency is silence padding rather than service response), `avg_processing_ms`, `avg_text_aggregation_ms`, `avg_turn_detection_ms`, and a `by_service` breakdown keyed by processor name. The existing `avg_ms` / `p95_ms` (user-stop → bot-start response delay) are unchanged.
  - `call.usage` (new keys) — the full `LLMTokenUsage` breakdown: `llm_total_tokens`, `llm_cache_read_input_tokens`, `llm_cache_creation_input_tokens`, `llm_reasoning_tokens`, `llm_input_audio_tokens`, `llm_output_audio_tokens`, `llm_cache_read_input_audio_tokens`.
  - `call.speech_stats` — user/agent talk seconds, turn counts, average turn lengths, agent talk ratio, approximate silence, longest gap, avg/max response delay. User talk time uses VAD segments when observed (Pipecat gives user turns a single STT timestamp).
  - `call.interruptions` — `InterruptionFrame` count, how many landed while the bot was speaking, and user idle timeouts.
  - `call.errors` — `ErrorFrame` / `FatalErrorFrame` capture: message (truncated to 500 chars), `fatal` flag, originating processor, exception type. Capped at 50. Previously error frames were not observed at all.
  - `call.connection` — transport lifecycle timeline (`ClientConnectedFrame`, `BotConnectedFrame`, `OutputTransportReadyFrame`, audio-streaming start, disconnect) with offsets from call start, for setup-timing and drop analysis.
  - `call.sip` — `{attributes, dtmf}`; DTMF keypresses with timestamps and direction. `attributes` is present but empty (Pipecat exposes no SIP attribute bag) so the backend has one path across SDKs.
  - `call.vad` — VAD speech segments (start/end/duration), segment count, total speech seconds, average segment length, and the observed `start_secs` / `stop_secs` VAD params.
  - `call.pipeline` — services seen, service metadata broadcast at pipeline start (`STTMetadataFrame.ttfs_p99_latency`, `LLMServiceMetadataFrame.is_realtime_service`), a frame-type histogram, and LLM thought/reasoning counts (count + character total only — reasoning text is never shipped).
  - `call.environment` — observer / pipecat / Python versions, so per-version capture gaps are auditable server-side.
- Version tolerance: frames and metric records are matched by class name (never imported), so a Pipecat release that renames or adds frames yields fewer fields instead of breaking the observer. Verified against Pipecat 1.7.0.
- Fail-soft guarantees: every extended section builder is exception-isolated (a broken section serializes as `null` rather than dropping the webhook), all event lists are bounded, and the extended payload is JSON-sanitized before POST.

### Notes
- No WebRTC network statistics (jitter, RTT, packet loss) are included: unlike the LiveKit SDK, a Pipecat observer has no access to transport-level stats. `call.connection` captures the transport lifecycle events that *are* observable.

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
