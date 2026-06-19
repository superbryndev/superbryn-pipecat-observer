# SuperBryn Pipecat Observer

[![PyPI version](https://img.shields.io/pypi/v/superbryn-pipecat-observer.svg?cacheSeconds=300)](https://pypi.org/project/superbryn-pipecat-observer/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Pipecat](https://img.shields.io/badge/pipecat-compatible-purple.svg)](https://github.com/pipecat-ai/pipecat)

Drop-in observer for [Pipecat](https://github.com/pipecat-ai/pipecat) voice AI agents. Captures transcript, audio, usage, and latency, then ships everything to SuperBryn at end of session — on any transport (Daily, Twilio, Plivo, Vobiz, WebRTC, WebSocket, …).

## Features

- **Drop-in integration** — wrap your pipeline with one call, every call shows up in SuperBryn.
- **Transport-agnostic** — same code works on Daily, Twilio, Plivo, Vobiz, WebRTC, and FastAPI WebSocket. No carrier credentials needed.
- **Precise transcripts** — speaker turns with timestamps, confidence scores, and per-turn bot latency.
- **Usage metrics** — LLM tokens (in / out), STT seconds, TTS characters, captured automatically from Pipecat's `MetricsFrame`.
- **Latency tracking** — average and p95 response time between user stop and bot start.
- **Provider auto-detection** — LLM / STT / TTS provider, model, and voice ID inferred from the service modules in your pipeline.
- **Tool-call capture** — both real Pipecat tools (`FunctionCallInProgressFrame` / `FunctionCallResultFrame`) and Anthropic-style `<tool_use>` prompt tools.
- **UAT / simulation mode** — `simulate_and_create_task` suppresses billing for non-production runs.
- **Session log capture** — buffer INFO+ logs from the call window and attach them to the payload.
- **Custom metadata** — attach any key/value bag to the call record; mutable mid-session.
- **Fail-open** — if anything in telemetry fails, your pipeline keeps running. Telemetry never crashes a call.
- **Frame-version tolerant** — frames are matched by class name, so a Pipecat upgrade doesn't hard-break the observer.

## Prerequisites

- Python **3.11+**
- An active Pipecat pipeline (`pipecat-ai >= 0.0.50` recommended for the 1.3+ observer lifecycle)
- `aiohttp >= 3.9.0` (installed as a transitive dependency)
- A SuperBryn API key from [https://app.superbryn.com/api-keys](https://app.superbryn.com/api-keys)

## Install

```bash
pip install superbryn-pipecat-observer
```

## Quick Start

1. Set your API key in the agent process environment:

   ```bash
   export SUPERBRYN_API_KEY="sb_..."
   ```

2. Wrap your pipeline with the observer:

   ```python
   from pipecat.pipeline.runner import PipelineRunner
   from superbryn_pipecat_observer import SuperbrynObserver

   observer = SuperbrynObserver(
       agent_id="your-agent-id",
       agent_name="support-bot",
   )

   task = observer.monitor_and_create_task(
       pipeline,
       transport=transport,
   )

   await PipelineRunner().run(task)
   ```

3. Run a call. The session shows up in SuperBryn within seconds of the call ending.

That's it. The same code works on every transport — no carrier credentials, no per-carrier wiring.

For UAT / simulation calls that shouldn't be billed or analyzed, use `simulate_and_create_task` instead — same signature.

> **Note:** earlier releases exposed these as `observe_and_create_task` /
> `track_and_create_task`. Those names still work but emit a
> `DeprecationWarning` and will be removed in `0.8.0`.

## What Gets Tracked

### Transcript
- Speaker turns (`user` / `agent`)
- Per-turn `start_time_ms` / `end_time_ms` (relative to call start)
- Per-turn `confidence` (when the STT provider reports it)
- Per-bot-turn `latency_ms` (gap between user stop and bot response start)
- Streaming LLM chunks (`LLMTextFrame`) are merged into a single turn per response
- Empty turns are filtered out before send

### Usage Metrics
Captured from Pipecat's `MetricsFrame` (requires `enable_usage_metrics=True`, set automatically by `monitor_and_create_task`):
- **LLM** — input tokens, output tokens, model, provider
- **TTS** — character count, model, provider, voice ID
- **STT** — seconds of audio processed, model, provider (with a VAD-derived fallback when the provider doesn't report)

### Latency
- Per-turn response delay (user stop → bot speak start)
- Aggregated **average** and **p95** across the call

### Session Metadata
- Agent ID, agent name, transport label (e.g. `daily`, `twilio`, `vobiz`)
- LLM / STT / TTS provider, model, voice ID
- Pipeline / SDK version tag
- Optional `from_number`, `to_number`, `recording_url`
- `mode` (`observe` or `track`)
- Any `custom_metadata` you pass in (mutable mid-session via `observer.set_custom_metadata({...})`)
- Call end reason (`completed` / `cancelled`)

### Tool Calls
- Real Pipecat tools — paired from `FunctionCallInProgressFrame` + `FunctionCallResultFrame` by `tool_call_id`
- Anthropic-style prompt tool calls — `<tool_use>{...}</tool_use>` blocks extracted from bot turns and lifted onto `call.tool_calls` (XML is stripped from the visible transcript)
- Each record carries `function_name`, `arguments`, `result`, `timestamp_ms`, and `tool_call_id`

### Session Logs (opt-in)
- INFO+ records from the agent process during the call window
- Bounded by `max_log_records` (default 1000) so a chatty agent can't blow up the payload
- Disable with `capture_logs=False`

## How It Works

1. **Pipeline observation.** The observer is registered via `PipelineTask(observers=[...])` — it runs **alongside** your pipeline, not inside it.
2. **Frame inspection.** `on_push_frame` watches every frame flowing between processors and aggregates by class name, so a Pipecat upgrade doesn't break it.
3. **Auto-detection.** The observer inspects each service's module path (`pipecat.services.<provider>.<role>`) to tag LLM / STT / TTS provider, model, and voice ID automatically.
4. **End-of-session delivery.** When the call ends (`EndFrame` / `CancelFrame` / transport disconnect), the SDK builds a normalized call payload and POSTs it to SuperBryn.

## Webhook Payload

```json
{
  "event": "call.completed",
  "sdk_version": "@superbryn/pipecat-observer@0.6.6",
  "call": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "started_at": "2026-06-15T12:00:00.000+00:00",
    "ended_at": "2026-06-15T12:05:30.000+00:00",
    "duration_seconds": 330.0,
    "call_end_reason": "completed",
    "from_number": "+15551234567",
    "to_number": "+15557654321",
    "transcript": {
      "turns": [
        {
          "speaker": "user",
          "text": "Hello, how are you?",
          "start_time_ms": 5000,
          "end_time_ms": 5000,
          "confidence": 0.98
        },
        {
          "speaker": "agent",
          "text": "I'm doing great, thanks for asking!",
          "start_time_ms": 7000,
          "end_time_ms": 11000,
          "latency_ms": 2000
        }
      ]
    },
    "metadata": {
      "agent_id": "support-bot",
      "agent_name": "support-bot",
      "transport": "daily",
      "llm_provider": "openai",
      "llm_model": "gpt-4o-mini",
      "stt_provider": "deepgram",
      "stt_model": "nova-3",
      "tts_provider": "cartesia",
      "tts_model": "sonic-english",
      "tts_voice_id": "...",
      "pipeline_version": "@superbryn/pipecat-observer@0.6.6",
      "mode": "observe"
    },
    "usage": {
      "llm_input_tokens": 1250,
      "llm_output_tokens": 850,
      "stt_duration_seconds": 45.2,
      "tts_characters": 1200
    },
    "latency": {
      "avg_ms": 750.5,
      "p95_ms": 1240.0
    },
    "tool_calls": [
      {
        "tool_call_id": "call_abc123",
        "function_name": "get_cart",
        "arguments": { "user_id": "u_42" },
        "result": { "items": [] },
        "timestamp_ms": 8200
      }
    ]
  }
}
```

`tool_calls` is included only when the LLM invoked one or more tools.

## Troubleshooting

### Enable debug logs

```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("superbryn_pipecat_observer").setLevel(logging.DEBUG)
```

### Log messages to look for

| Log prefix | Meaning |
|---|---|
| `SUPERBRYN_PIPECAT_CALL_STARTED` | Observer initialized for this session |
| `SUPERBRYN_PIPECAT_SENDING` | Webhook about to be sent |
| `SUPERBRYN_PIPECAT_PAYLOAD` | Full outbound payload (debug aid) |
| `SUPERBRYN_PIPECAT_SENT` | Webhook delivered successfully (HTTP 200) |
| `SUPERBRYN_PIPECAT_SKIPPED` | No API key configured — observer no-ops |
| `SUPERBRYN_PIPECAT_AUTH_FAILED` | Invalid or revoked API key (401/403) |
| `SUPERBRYN_PIPECAT_PAYLOAD_TOO_LARGE` | JSON exceeded the 20 MB backend cap |
| `SUPERBRYN_PIPECAT_HTTP_*` | Non-2xx response from the SuperBryn server |
| `SUPERBRYN_PIPECAT_ERROR` | Network / exception during delivery |
| `SUPERBRYN_PIPECAT_MISSING_AIOHTTP` | `aiohttp` not installed in the runtime |
| `SUPERBRYN_PIPECAT_SESSION_DISCARDED` | Session aborted — no webhook sent |

### Common errors

| Error | Cause | Fix |
|---|---|---|
| `SUPERBRYN_API_KEY not configured` | Missing API key | Set `SUPERBRYN_API_KEY` in your environment |
| `SUPERBRYN_PIPECAT_AUTH_FAILED (401)` | Invalid API key | Verify the key in SuperBryn → API keys |
| `SUPERBRYN_PIPECAT_AUTH_FAILED (403)` | Expired / disabled key | Generate a new API key |
| `SUPERBRYN_PIPECAT_MISSING_AIOHTTP` | `aiohttp` missing | `pip install aiohttp>=3.9.0` |
| `SUPERBRYN_PIPECAT_PAYLOAD_TOO_LARGE` | Transcript / logs too large | Trim `custom_metadata` or pass `capture_logs=False` |

### Missing usage metrics

Pipecat only emits `MetricsFrame`s when usage metrics are enabled. `monitor_and_create_task` / `simulate_and_create_task` set this for you. If you build your own `PipelineTask`, pass `enable_usage_metrics=True`:

```python
PipelineTask(
    pipeline,
    observers=[observer],
    params=PipelineParams(enable_usage_metrics=True),
)
```

Without it you'll still get the transcript, latency, and provider detection — but `llm_input_tokens`, `llm_output_tokens`, and `tts_characters` will be `0`.

## Links

- [Pipecat documentation](https://docs.pipecat.ai/)
- [GitHub repository](https://github.com/superbryndev/superbryn-pipecat-observer)
- [Issue tracker](https://github.com/superbryndev/superbryn-pipecat-observer/issues)
- [Get an API key](https://app.superbryn.com/api-keys)

## Support

- Email: support@superbryn.com
- GitHub issues: [Report a bug](https://github.com/superbryndev/superbryn-pipecat-observer/issues)

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

Made with ❤️ by [SuperBryn](https://www.superbryn.com)
