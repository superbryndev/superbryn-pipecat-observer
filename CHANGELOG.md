# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.11.0] - 2026-10-23

Found by running a scripted call through a real Pipecat 1.6 pipeline (`examples/local_harness/`) and checked against production payloads.

### Fixed
- **Observer: every frame counted once per pipeline hop.** Pipecat calls an observer once for each link a frame crosses, and the observer handled every call. Each frame is now handled on its first push only (by frame id), and the upstream twin of a frame broadcast both ways is skipped. What this fixes:
  - **Tool calls:** one record per call, not one per hop (production stored 13 per call).
  - **Usage:** LLM tokens and TTS characters are no longer multiplied by the number of hops (3–6× too high).
  - **Interruptions:** `interruption_count` and `bot_turns_interrupted` count each barge-in once (production showed 210 for 14 caller turns).
  - **Pipeline section:** `frame_counts` are real frame counts.
- **Observer: usage read 0 when the customer passed their own `PipelineParams`.** Pipecat sends usage (and service timings) only when asked. `monitor_and_create_task`, `simulate_and_create_task` and `register_task_handlers` now switch on `enable_usage_metrics` and `enable_metrics` unless the customer set one to `False`. In that case a `SUPERBRYN_PIPECAT_METRICS_OFF` warning is logged.
- **Observer: agent turns duplicated or carrying words never said.** An agent turn is now one stretch of bot audio (`BotStartedSpeakingFrame` → `BotStoppedSpeakingFrame`), and its text is what the output transport played (`TTSTextFrame`). What this fixes:
  - A short reply no longer appears twice, once from the LLM stream and once from TTS.
  - A reply the caller cut off keeps only the words heard and carries `interrupted: true`.
  - Pipelines that play no TTS text fall back to the LLM's streamed text.
- **Observer: numbers split apart.** LLM tokens are joined the way Pipecat joins them, so "884" + "291" reads "884291", not "884 291".
- **Observer: reply latency read far too low.** It ran from the last pipeline hop of the caller's turn to the LLM *starting*, so it was often below the LLM's own first-token time. It now runs from the caller's end of speech to the bot's first audio:
  - **With VAD:** the end of speech is the VAD stop time minus its `stop_secs`.
  - **Without VAD:** the turn's release.
  - `latency.avg_ms` / `p95_ms` hold one value per reply.
- **Observer: `metadata.transport` was empty** when the transport was passed to `monitor_and_create_task` (the README setup) instead of the constructor.
- **Observer: Pipecat and agent logs were missing.** Pipecat and most Pipecat agents log through loguru, which the session log capture never saw. A loguru sink now feeds the same buffer as `logging`.
- **Observer: the last reply's text could be cut.** With `register_task_handlers` (all `*_and_create_task` setups), the payload is now sent from `on_pipeline_finished`, after the pipeline drains, not on the first sight of the terminal frame.

### Changed
- Prompt-engineered `<tool_use>` blocks are lifted from each whole LLM response rather than from a transcript turn.

### Security
- Dev lock (`uv.lock`, not shipped in the package): aiohttp 3.14.5, certifi 2026.10.19, and Pillow 13.0.0 through a uv override, since pipecat-ai caps Pillow below 13. This clears the Dependabot alerts that have a fix. nltk (pulled in by pipecat-ai) has none yet.

### Added
- `examples/local_harness/`: a scripted call through a real Pipecat pipeline (aggregators, function calling, output transport, recorder) with stand-in STT / LLM / TTS. It attaches both the observer and eval mode, and checks every payload field against what happened. There are five scenarios: sentence TTS, word-timed TTS, customer `PipelineParams`, barge-in, and a slow LLM. CI runs it on every pull request.
- Agent transcript turns carry `interrupted`.

## [0.10.1] - 2026-10-22

### Fixed
- **Eval mode** (`test_data.py`): a call with no failures sends `errors: []` instead of leaving `errors` out, which SuperBryn read as not sent. Its error metrics were skipped on every clean call instead of reading 0. The observer sees every `ErrorFrame`, so an empty list means none happened. A tool handler that raised is still that tool call's `is_error`, not a pipeline error.
- **Eval mode** (`test_data.py`): a caller line nothing timed now carries no `start_ms`. That happens with a speech-to-speech service that reports no caller turns and no local VAD (Nova Sonic, Gemini Live), or a pipeline with neither VAD nor turn frames. It used to carry when its transcript arrived, after the caller finished. Nova Sonic sends that transcript only as its reply starts. The line still keeps its place in the transcript, and tool calls still link to the right agent turn.

### Changed
- **Eval mode:** Nova Sonic and Gemini Live send no `llm_requests`. Both mark a response started only when its first output arrives (Nova with its reply's first text, Gemini Live with its first model turn), so no request's start or outcome can be told. OpenAI Realtime keeps its requests: it marks a response when the server creates it, before any audio.

### Added
- **Eval mode:** `configuration.turn_detection.preemptive_generation`.
  - `true` when a user turn stop strategy leaves completion to an external judge (`ExternalUserTurnCompletionStopStrategy`, such as `LLMTurnCompletionUserTurnStopStrategy`). The LLM then starts on a detector's trigger before the turn is final, and drops the answer if the turn wasn't complete.
  - `false` with the other strategies, which start it as the turn ends.
  - Not sent for speech-to-speech pipelines.
  - SuperBryn uses it to name why replies were dropped: early drafts, or the end-of-turn wait.
- **Eval mode:** `configuration.turn_detection.type` is `realtime_llm` for a speech-to-speech pipeline with no VAD or turn analyzer, whose service takes the turns itself.

## [0.10.0] - Unreleased

### Fixed
- **Eval mode, found in review before release** (`test_data.py`):
  - A slow or unreachable SuperBryn API no longer holds the pipeline's teardown (up to ~51 s for a test call, ~18 s for an ordinary call). The end-of-call step (waiting for `call.started`, the end frame and `call.ended` with its retries) has one 12 s budget, and a call without a test ID that ends while the SuperBryn-number lookup is still running cancels it and sends nothing.
  - Realtime (speech-to-speech) pipelines sent no caller turns: their services push the caller's transcript upstream, which was dropped. It is now captured once however many processors it passes, and without VAD frames the service's own `UserStartedSpeakingFrame` / `UserStoppedSpeakingFrame` time the caller's turns, so a transcript that completes after the reply started stays before it.
  - `endpointingLatency` left out the stop strategy's own wait (speech timeout, smart-turn timeout) and read 200 ms for turns that took 0.8-3.2 s to end. It now runs from the user's silence to the turn's release (`UserStoppedSpeakingFrame`), falling back to the old estimate when there is none; `turn_detection.min_end_of_turn_ms` is the shortest such wait.
  - An agent turn cut off before any of it was played reported the LLM's whole answer. LLM text now stands in only when the pipeline never reports played TTS text, and never for an interrupted turn.
  - With a user mute strategy, what the user said while muted (never heard by the agent) was appended to the previous user turn; it is now left out.
  - Attaching after the task started timed the whole call from its end (every turn at 0 ms). The call is now timed from the first frame seen, `call.started` is sent then, and a warning is logged.
  - A Twilio bot that parses the handshake itself (`parse_telephony_websocket` instead of `create_transport`) was never recognised. The handshake Pipecat caches on the websocket is now read, and `call_data=` passes one explicitly; a call with neither a test ID nor phone numbers logs one `INFO` line.
  - A test ID or number that still holds an unfilled template placeholder (such as `{From}` pasted into TwiML) is ignored, with a warning, instead of being sent as the test ID.

### Added
- **Eval mode (testing data).** New `attach_test_data(task, runner_args)` reports SuperBryn test calls to `POST /public-api/v1/test-calls/agent-data`: `call.started` when the pipeline starts, and `call.ended` when it finishes, with the transcript, tool calls and diagnostics (per-reply latency, STT/LLM/TTS configuration, turn detection, tools, token and character usage, every LLM response with its start, its tokens and what became of it: spoken, cut off, not spoken, stopped or only a tool call, every error frame by pipeline step, interruption settings from the user turn start strategies, the STT's confidence for Deepgram, Google, Gladia, AWS Transcribe and Soniox, a tool call's `is_error` when its handler raised or the tool isn't registered, and STT audio time: reported by newer Pipecat, measured on 1.4 from the audio a streaming STT receives). Every other call sends nothing.
  - A call is a test call when it carries a SuperBryn test ID (the `superbryn_call_id=` argument, `superbryn_call_id` in the start body, a Daily dial-in SIP header ending in `x-superbryn-call-id`, or a telephony custom parameter), or when its caller or dialled number is one of SuperBryn's numbers (`GET /public-api/v1/test-calls/caller-numbers`, cached for an hour).
  - Needs an org API key with `test_data:write` (`SUPERBRYN_TEST_API_KEY`, else `SUPERBRYN_API_KEY`) and the SuperBryn agent ID (`agent_id=` or `SUPERBRYN_AGENT_ID`); without either, nothing is sent and one line is logged. Up to 3 attempts per event; `call.ended` is skipped when the server answers `is_test: false`.
  - Independent of `SuperbrynObserver`: its own module (`test_data.py`), frame capture and HTTP client. Built against the Pipecat 1.4 frame, metric and service classes.

## [0.9.0] - 2026-10-01

### Fixed
- **STT duration was silently zero on Pipecat >= 1.6, and took LLM/TTS usage down with it.** `STTUsageMetricsData.value` became an `STTUsage` object (`.audio_seconds`) where it used to be a bare float; the observer's `float(...)` coercion raised `TypeError` inside the `MetricsFrame` record loop, which aborted the loop — so any LLM or TTS record positioned after the STT record in the same frame was dropped too, and the whole failure was swallowed by `on_push_frame`'s catch-all. Usage now reads either shape, and each record is parsed in isolation so one unparseable record can't discard the rest of the batch. This fix applies regardless of `extended_capture`.

### Added
- **Prompt sync.** `SuperbrynObserver(prompt=..., prompt_version=...)`: each production call carries `metadata.prompt_ref` (`sha256` of the template plus your label) and the template is pushed once per process per version to `POST /public-api/v1/prompts` (needs the `obs:write` scope; fails open). Without `prompt=`, the system prompt from the pipeline's LLM context is sent as `metadata.system_prompt`. Simulation runs never sync. `prompt_ref` / `prompt_hash` / `push_prompt` are exported for custom setups.
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

[Unreleased]: https://github.com/superbryndev/superbryn-pipecat-observer/compare/v0.9.0...HEAD
[0.9.0]: https://github.com/superbryndev/superbryn-pipecat-observer/compare/v0.8.0...v0.9.0
[0.8.0]: https://github.com/superbryndev/superbryn-pipecat-observer/compare/v0.7.1...v0.8.0
[0.7.1]: https://github.com/superbryndev/superbryn-pipecat-observer/releases/tag/v0.7.1
