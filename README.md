# SuperBryn Pipecat Observer

[![PyPI version](https://img.shields.io/badge/pypi-v0.12.0-orange)](https://pypi.org/project/superbryn-pipecat-observer/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pipecat](https://img.shields.io/badge/pipecat-compatible-purple.svg)](https://github.com/pipecat-ai/pipecat)

**One call in your Pipecat bot reports every call to SuperBryn.**

`attach(task, runner_args)` sends your agent's own record of each call: the transcript, tool calls, each reply's
latency parts, the models and settings that ran, token, audio and character usage, every LLM request and what became
of it, and every STT, LLM, TTS or transport failure. SuperBryn does every calculation on its side. It works on any
transport Pipecat speaks (Daily, Twilio, Plivo, Vobiz, Exotel, WebRTC, WebSocket) with no carrier credentials.

- **SuperBryn test calls** are reported for testing (surface `testing`): the agent's record sits next to the test.
- **Every other call** is a production call (surface `observability`): it is recorded in stereo and appears in
  Monitor. `production=False` turns production reporting off.

Each call is reported once, as `call.started` when the pipeline starts and `call.ended` when it finishes. Everything
runs in the background: it adds no latency and never raises into your pipeline.

## Install

```bash
pip install -U "superbryn-pipecat-observer>=0.12.0"
```

## Setup

```bash
export SUPERBRYN_API_KEY=sbryn_...                               # one key with test_data:write and obs:write
export SUPERBRYN_AGENT_ID=3f2b8c1e-5d4a-4e6f-9a7b-1c2d3e4f5a6b  # the agent's SuperBryn ID (not needed with an agent key)
```

Add one call to your bot, once per call, after building the `PipelineTask` and **before** it runs:

```python
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from superbryn_pipecat_observer import attach


async def bot(runner_args):
    ...
    task = PipelineTask(pipeline, params=PipelineParams())
    report = attach(task, runner_args)
    await PipelineRunner().run(task)
```

That's all: `attach` adds its own observer, its own `on_pipeline_finished` hook and, for production calls, the
recorder. Pass `runner_args` (what Pipecat's runner hands your bot) so test calls are recognised and the call is
named as before; a bot with no runner arguments passes what it knows as keywords instead (below).

### Options

| Option | Environment variable | What it does |
|--------|----------------------|--------------|
| `api_key` | `SUPERBRYN_API_KEY` | A SuperBryn API key: test calls need `test_data:write` and prompt versions `obs:write`; production calls take any valid key (the setup screens' keys hold both). Without one nothing is sent. |
| `agent_id` | `SUPERBRYN_AGENT_ID` | The agent's SuperBryn ID, a UUID. Test calls need it unless the key is the agent's own key. Any other value is left out, with one warning. |
| `production` | | `True` by default. `False` reports SuperBryn test calls only: production calls send nothing and are not recorded. |
| `record` | | `True` by default. `False` reports production calls without SuperBryn's recording (for example for compliance). |
| `simulation` | | `False` by default. `True` marks production calls as your own UAT / simulation runs (`observability.mode: "track"`): SuperBryn keeps their transcript without analysis or billing, and they are not recorded and name no prompt. |
| `capture_logs` | | `True` by default: a production call carries the INFO+ log lines its own code wrote (loguru, and `logging` at the levels your config lets through), up to 1000, each cut to 2,000 characters. Other calls in the same process are left out (call `attach()` in the task that runs the pipeline), and your logging config is not changed. `False` sends none. |
| `custom_data` | | Your own keys for this call, a dict sent as given (see below). |
| `prompt`, `prompt_version` | | The prompt template your agent runs, and your label for this version (see Prompt sync). |
| `base_url` | `SUPERBRYN_BASE_URL` | The SuperBryn API origin, `https://api.superbryn.com` by default. |
| `superbryn_call_id`, `from_number`, `to_number`, `call_id`, `call_data` | | What your bot already knows about the call, when `runner_args` doesn't carry it: SuperBryn's test ID, the caller and the dialled number, your own ID for the call, and a parsed telephony handshake (`parse_telephony_websocket`). |
| | `SUPERBRYN_OBSERVER_ENABLED` | `false`, `0`, `no` or `off` turns `attach` off for the whole process, as in 0.11. |

`SUPERBRYN_TEST_API_KEY`, `SUPERBRYN_WEBHOOK_URL`, `WEBHOOK_URL`, `AGENT_ID` and `VERSION_ID`, which 0.11 read, are
not read any more.

### The handle

`attach` returns the call's handle. Use it for anything known only during the call, until the call ends:

```python
report.set_custom_data({"ticket_id": "TKT-9001"})          # replaces custom_data
report.update_custom_data({"appointment_booked": True})    # merges into it; other keys are kept
report.set_call_end_reason("transfer_to_human")            # sent as ended_reason instead of the end frame's
report.set_recording_url("https://my-bucket/.../call.mp3")  # production calls: your own recording (see below)
```

### Custom data

`custom_data` is sent at the top of `call.ended` exactly as you give it: the same key names and the same value types
(a value JSON can't hold, such as a `datetime`, is sent as its text). SuperBryn's pre-ingestion filters and profiles
match these key names, so keys like `workflow_id` or `tenant_id` route production calls to the right source and
profile. Set it in `attach(custom_data=...)` for what you know at the start, and with `set_custom_data()` or
`update_custom_data()` for what you learn during the call.

### Prompt sync

Pass the prompt template your agent runs (placeholders unfilled) so SuperBryn knows exactly which prompt each call
ran on:

```python
report = attach(task, runner_args, prompt=PROMPT_TEMPLATE, prompt_version="v13")
```

Each call carries `context.prompt_ref = {"hash": "sha256:...", "version": "v13"}`, and the template's text is pushed
to SuperBryn once per host per version (`POST /public-api/v1/prompts`, which needs `obs:write`), with the tools in
the pipeline's LLM context. A prompt change you deploy shows up in the agent's settings in SuperBryn. Every call also
sends the system message of its LLM context as `context.system_prompt`, which SuperBryn matches as text when no
`prompt=` is given. Any stack can compute the hash: `sha256` over the template's UTF-8 bytes, written `sha256:<hex>`
(`superbryn_pipecat_observer.prompt_ref(template, version)`).

## How a SuperBryn test call is recognised

In this order:

1. `superbryn_call_id=` is passed;
2. `runner_args.body` has `superbryn_call_id` (webhook, Custom cURL and Pipecat Cloud starts);
3. a Daily dial-in SIP header ends in `x-superbryn-call-id`;
4. a telephony WebSocket carries a `superbryn_call_id` custom parameter (Twilio stream parameters, Exotel custom
   parameters), read from `call_data=`, else `runner_args.call_data`, else the handshake your bot parsed itself with
   `parse_telephony_websocket`;
5. otherwise, the caller or the dialled number is one of SuperBryn's test numbers (fetched with your key, any valid
   SuperBryn API key, and cached for an hour in a temp file every process on the host shares; a refused key is
   asked again after an hour, an unanswered fetch after a minute). A test call's data is stored only with
   `test_data:write`: with a Monitor-only key, SuperBryn's test calls are recognised and kept out of Monitor.

A value that still holds a template placeholder such as `{From}` is ignored, with a warning. A call SuperBryn says is
not one of its tests is not reported again as a production call.

A test call's `call_id` is the telephony call's own ID (`call_data.call_id`, the Daily dial-in `call_id`), which
SuperBryn's Custom cURL trigger matches; a production call's is `runner_args.session_id`, the ID 0.11 gave Monitor
calls. `call_id=` names both.

## Production calls

**Recording.** Unless `record=False`, a production call is recorded in stereo, the caller on the left channel and the
agent on the right: `attach` puts Pipecat's `AudioBufferProcessor` right before your output transport. When the call
ends, the WAV goes straight to SuperBryn's storage through a short-lived presigned URL fetched with your key, and the
temp file is deleted. No S3 settings are needed. A recording that can't be made or uploaded leaves the call
unrecorded, never unreported. A call recognised as a SuperBryn test is never recorded. `attach` must run before the
task does: once the pipeline runs, no recorder can be added (one warning).

**Your own recording.** Pass its URL (public or pre-signed) with `report.set_recording_url(url)`; set before the
task runs, SuperBryn's recording is not started. It is sent with `recording_url_source: "external"`.

`observability` also carries `mode` (`observe`, or `track` with `simulation=True`) and, on Daily dial-in and dial-out
calls, `direction`.

## What a report contains

`POST {base_url}/public-api/v1/calls/agent-data`, with the key in `X-API-Key`. Every event repeats the call's identity:
`schema_version`, `event_id` (`pc:<call_id>:started` or `:ended`, the same on a retry), `event`, `surface`,
`source: "pipecat"`, `call_id`, `agent_id`, `from`, `to`, and `started_at`, when the pipeline started. Every
`start_ms`, `end_ms` and `at_ms` counts from `started_at`. `call.ended` adds:

| Field | What it holds |
|-------|---------------|
| `ended_at`, `ended_reason`, `ended_by` | When the pipeline finished; why (`ended`, `cancelled`, `stopped`, `error`, or yours); and who ended it when that is known: `agent` (a processor such as an end-call tool asked the pipeline to end), `caller` (the transport's client disconnected first) or `system` (a fatal error). |
| `transcript[]` | Each line: `id`, `role`, `text`, `start_ms`, `end_ms`; caller lines `confidence`; agent lines `interrupted`. Agent text is what the output transport played. |
| `tool_calls[]` | `id`, `name`, `arguments`, `result`, `is_error`, `start_ms`, `end_ms`, and `turn_id`, the agent line it fed; `<tool_use>{...}</tool_use>` blocks an LLM was prompted to write are lifted out of the text as tool calls. |
| `diagnostics` | `pipeline`; `configuration` (STT, LLM and TTS with the voice, turn detection, interruptions, tools); `turn_latencies[]` per reply (`transcriber_ms`, `endpointing_ms`, `model_ms`, `voice_ms`, `turn_ms`); `usage`; `llm_requests[]` with each one's outcome; `errors[]`; `dtmf[]`; and, on production calls, `logs[]` (`ts`, `level`, `name`, `message`). |
| `context` | `system_prompt`, `prompt_ref`, `room_url` (the Daily room), and `environment` (this package's, Pipecat's and Python's versions, the transport, and 0.11's `pipeline_version` tag). |
| `custom_data` | Your keys, as given. |
| `testing` | Test calls: `superbryn_call_id`, when the call carried one. |
| `observability` | Production calls: `mode` and `direction`, and `recording_url`, `stereo_recording_url`, `recording_url_source` on `call.ended`. |

No audio is sent in the body, and the package computes no averages: SuperBryn works them out from these records.
Latency and usage need Pipecat's metrics: `attach` switches on `enable_metrics` and `enable_usage_metrics` when your
`PipelineParams` leave them unset, and logs one warning when you turned one off.

## Agent Config Sync (opt-in)

Push your agent's configuration to SuperBryn as a reviewable draft. Requires an **agent-scoped** API key; nothing syncs unless you call it explicitly:

```python
from superbryn_pipecat_observer import sync_config

sync_config(
    pipeline,
    api_key="sk_agent_...",           # or SUPERBRYN_API_KEY
    identity={"name": "Support Agent", "type": "inbound", "agent_modality": "voice"},
    behavior={"prompt": open("prompt.txt").read()},
    policy_guardrails=open("guardrails.md").read(),
)
```

The helper walks the pipeline's processors — recursing into nested pipelines, `ParallelPipeline` branches, and `ServiceSwitcher` / `LLMSwitcher` members, and unwrapping custom wrapper classes — to fill the `llm` / `stt` / `tts` / `voice` blocks automatically. For switchers, the primary (initially active) service fills the block and the next member is reported in its `fallback` sub-block. It also reads the LLM context to fill `behavior.prompt` (the system message) and `tools` (the advertised function schemas) — explicit `behavior=` / `tools=` overrides always win. Extraction reads a fixed allow-list of configuration attributes (including private fields like `_settings`, `_model`, `_voice_id` where Pipecat services store their settings); credential attributes such as provider API keys are never part of that list and are never read or transmitted. The manifest lands as a pending draft in the SuperBryn dashboard for review; it never changes the live agent directly. Use `async_sync_config(...)` inside a running event loop, or `build_manifest_from_pipeline(...)` + `sync_manifest(...)` to inspect/modify the manifest before pushing. The API origin is `base_url=`, else `SUPERBRYN_BASE_URL`, else `SUPERBRYN_API_BASE_URL`, else `https://api.superbryn.com`.

Override sections accept exactly the fields of the canonical manifest schema (unknown keys raise `ValueError` locally — the endpoint rejects them anyway):

| Keyword | Fields |
|---|---|
| `identity` | `name`, `type` (`inbound`/`outbound`), `agent_modality` (`voice`/`chat`), `description`, `pain_point`, `gender`, `age`, `dob` |
| `behavior` | `prompt`, `flow` |
| `tools` | list of `{name, description, schema, server: {type, url}}` |
| `language` | `primary_language`, `additional_languages: [{code, priority}]` |
| `telephony` | `phone_number`, `ivr_config: {enabled, number}` |

Plus top-level strings/ints: `policy_guardrails`, `additional_details`, `concurrency_calls`. Sections the pipeline can't expose (identity, telephony, guardrails, ...) are supplied through these explicit overrides.

Docs: https://docs.superbryn.com/advanced/agent-sync

## Logs

Log lines (logger `superbryn_pipecat_observer.call_report`) start with `SUPERBRYN_CALL_REPORT_`: `SENT` when SuperBryn
accepted an event (with any optional field it dropped), `DISABLED` without an API key or with
`SUPERBRYN_OBSERVER_ENABLED` off, `WARNING` for an agent ID that isn't a UUID or caller numbers the key can't read,
`RETRY`, `REJECTED` (with what to change) and `FAILED` for delivery problems, and `ERROR` for anything else. None of
them affects the call. When the pipeline finishes, the work before the recording's upload (`call.started`, the end
frame, finishing the file and getting its upload URL) gets at most 12 seconds, and `call.ended` after it 12 more; the upload itself can take up to 2 minutes for a long call, after the caller has gone.

## Upgrading from 0.11

0.12.0 replaces both 0.11 entry points (`SuperbrynObserver` with `monitor_and_create_task` /
`simulate_and_create_task` for production calls, and `attach_test_data` for SuperBryn test calls) with `attach`. It
posts a new body to a new endpoint, so pin the new version and change the setup in the same deploy.

**Before (0.11):**

```python
from pipecat.pipeline.runner import PipelineRunner
from superbryn_pipecat_observer import SuperbrynObserver, attach_test_data


async def bot(runner_args):
    ...
    observer = SuperbrynObserver(
        agent_id="support-bot",
        agent_name="Support Bot",
        custom_metadata={"workflow_id": "wf_inbound", "tenant_id": 42},
        prompt=PROMPT, prompt_version="v13",
    )
    task = observer.monitor_and_create_task(
        pipeline, context, runner_args=runner_args, transport=transport, idle_timeout_secs=120
    )
    attach_test_data(task, runner_args)
    ...
    observer.set_custom_metadata({"booked": True})
    await PipelineRunner().run(task)
```

**After (0.12):**

```python
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from superbryn_pipecat_observer import attach


async def bot(runner_args):
    ...
    task = PipelineTask(pipeline, params=PipelineParams(), idle_timeout_secs=120)
    report = attach(
        task, runner_args,
        custom_data={"workflow_id": "wf_inbound", "tenant_id": 42,
                     "agent_id": "support-bot", "agent_name": "Support Bot"},
        prompt=PROMPT, prompt_version="v13",
    )
    ...
    report.update_custom_data({"booked": True})
    await PipelineRunner().run(task)
```

Environment: `SUPERBRYN_API_KEY` is one key with both `test_data:write` and `obs:write` (it replaces
`SUPERBRYN_TEST_API_KEY`); `SUPERBRYN_AGENT_ID` is the agent's SuperBryn ID, a UUID (a free-form `AGENT_ID` is not
read; drop it when you use the agent's own key); `SUPERBRYN_BASE_URL` is the API origin (it replaces
`SUPERBRYN_WEBHOOK_URL` and `WEBHOOK_URL`; config sync still falls back to `SUPERBRYN_API_BASE_URL`).

What else changed:

- You build the `PipelineTask` yourself (with the `PipelineTask` keywords you passed to `monitor_and_create_task`),
  then call `attach`; it adds the recorder before the output transport, as `monitor_*` did. Remove a
  `SuperbrynAudioRecorder` you placed in the pipeline yourself, and the observer from `observers=[...]`.
  `simulate_and_create_task` is `attach(..., simulation=True)`. Recording is always stereo.
- `custom_metadata`, `extra_metadata` and `set_custom_metadata()` are `custom_data` and `update_custom_data()`, sent
  in the body's `custom_data` (0.11 sent them in `call.metadata`). `agent_name`, and an `agent_id` that is not a
  SuperBryn UUID, are not options any more: keep them in `custom_data` under the same names if your filters use them.
- `session_id=` is `call_id=`; `recording_url=` / `stereo_recording_url=` are `report.set_recording_url(url)`;
  `capture_logs` and `prompt` / `prompt_version` are unchanged. `transport`, `audio_recorder`, `extended_capture`,
  `max_log_records`, `first_message`, `sync_prompt`, `enabled`, `num_channels` and `defer_upload` (with
  `start_audio_upload()` and `abort()`) are gone.
- The package no longer sends averages or counts (`latency.avg_ms` / `p95_ms`, `speech_stats`, turn-detection,
  interruption, VAD, connection and pipeline summaries, `duration_seconds`); SuperBryn computes them from the raw
  records. Turn latency keys are snake_case (`transcriber_ms`, `endpointing_ms`, `model_ms`, `voice_ms`, `turn_ms`).
- `ended_reason` is `ended` / `cancelled` / `stopped` / `error` (0.11's production calls said `completed` /
  `cancelled`).

**A prompt for your coding agent** (copy and paste):

```text
Upgrade this Pipecat bot from superbryn-pipecat-observer 0.11 to 0.12.0 (pip: "superbryn-pipecat-observer>=0.12.0").

1. Pin superbryn-pipecat-observer>=0.12.0 in the project's dependencies.
2. Remove every use of the old API and its imports: SuperbrynObserver(...), SuperbrynAudioRecorder(...) (and its
   recorder.processor inside the Pipeline([...])), observer.monitor_and_create_task(...),
   observer.simulate_and_create_task(...), observer.monitor_pipeline(...), observer.simulate_pipeline(...), the
   deprecated observe_* / track_* methods, observer.register_task_handlers(...), observer.attach_to_task(...), the
   observer inside PipelineTask(observers=[...]), and attach_test_data(task, ...).
3. Where monitor_and_create_task / simulate_and_create_task built the task, build it yourself:
   task = PipelineTask(pipeline, params=<the params= passed before, else PipelineParams()>, <every other keyword
   passed before>). Then add one call right after it and BEFORE the task runs:
   report = attach(task, runner_args, ...)   (from superbryn_pipecat_observer import attach)
   Pass runner_args when the bot has them (whatever was passed as runner_args= before). If attach_test_data was
   given superbryn_call_id=, from_number=, to_number=, call_id= or call_data=, pass them to attach the same way.
4. Map the old options onto attach:
   - custom_metadata= and extra_metadata= -> custom_data= (one dict). Keep every key name and every value type
     exactly as it is: SuperBryn's filters and profiles match on them (for example workflow_id).
   - agent_name= -> add "agent_name": <same value> to custom_data. agent_id= -> if it is the agent's SuperBryn ID
     (a UUID), pass agent_id= (or set SUPERBRYN_AGENT_ID); otherwise add "agent_id": <same value> to custom_data.
   - session_id= -> call_id=.  prompt= / prompt_version= / capture_logs= -> the same names.
   - simulate_and_create_task / simulate_pipeline -> attach(..., simulation=True).
   - recording_url= / stereo_recording_url= -> report.set_recording_url(url), called right after attach.
   - api_key= -> api_key=. webhook_url= / api_base_url= -> base_url= with the URL's origin only
     (https://host, no path).
   - Drop transport=, audio_recorder=, extended_capture=, max_log_records=, first_message=, sync_prompt=,
     num_channels=. If sync_prompt=False was set, do not pass prompt=. If enabled=False was set, do not call
     attach in that case.
5. Replace observer.set_custom_metadata(...) with report.update_custom_data(...) (same argument), and
   observer.get_custom_metadata() with the bot's own copy of the dict. If the observer is passed to other functions
   or tools, pass `report` instead.
6. If the bot uses defer_upload=True, observer.start_audio_upload() or observer.abort(), STOP and ask me: 0.12 has no
   consent gate; the closest is attach(..., record=False).
7. Environment and deployment config: put SuperBryn's Agent data key (it holds both
   test_data:write and obs:write) in SUPERBRYN_API_KEY and remove SUPERBRYN_TEST_API_KEY; replace a free-form AGENT_ID with SUPERBRYN_AGENT_ID set to the agent's SuperBryn
   agent ID (a UUID), or remove it when the key is the agent's own key; replace SUPERBRYN_WEBHOOK_URL / WEBHOOK_URL /
   SUPERBRYN_API_BASE_URL with SUPERBRYN_BASE_URL set to the API origin (https://api.superbryn.com unless told
   otherwise); remove VERSION_ID if only this package used it. Keep SUPERBRYN_OBSERVER_ENABLED as it is.
8. Do not change anything else in the bot. Show the diff when done.
```

## Links

- [Pipecat documentation](https://docs.pipecat.ai/)
- [GitHub repository](https://github.com/superbryndev/superbryn-pipecat-observer)
- [Issue tracker](https://github.com/superbryndev/superbryn-pipecat-observer/issues)
- [Get an API key](https://app.superbryn.com/api-keys)

## License

MIT, see [LICENSE](LICENSE).

Made by [SuperBryn](https://www.superbryn.com). Support: support@superbryn.com
