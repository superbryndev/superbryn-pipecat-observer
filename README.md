# SuperBryn Pipecat Observer

[![PyPI version](https://badge.fury.io/py/superbryn-pipecat-observer.svg)](https://badge.fury.io/py/superbryn-pipecat-observer)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Track and evaluate your [Pipecat](https://github.com/pipecat-ai/pipecat) voice AI agents with just 2 lines of code.**

Automatically capture transcripts, usage metrics, latency data, and session analytics from your Pipecat pipelines. Perfect for monitoring, debugging, and optimizing your voice AI applications.

## ✨ Features

- 🎯 **2-Line Integration** - Add to any Pipecat agent in seconds
- 📝 **Precise Transcripts** - Speaker turns with timestamps from `TranscriptionFrame` and `TextFrame` events
- 📊 **Usage Metrics** - Track LLM tokens, TTS character counts, STT duration
- ⚡ **Latency Tracking** - Response time between user speech end and bot response start (avg + p95)
- 🔍 **Auto-Detection** - Automatically extracts models, providers, and voice IDs from your services
- 📞 **Telephony Ready** - Pass `from_number` / `to_number` / `transport` for SIP / Daily / Twilio / WebRTC calls
- 🎥 **Recording URLs** - Forward egress recording links if your transport produces them
- 🛡️ **Fail-Open** - Never crashes your pipeline if telemetry delivery fails
- 🔐 **Secure** - API key authentication with HTTPS webhook delivery
- 🔄 **Frame-Version Tolerant** - Detects frames by class name so a Pipecat upgrade won't hard-break the observer

## 🚀 Quick Start

### Prerequisites

1. **Get your API key** from [https://app.superbryn.com/api-keys](https://app.superbryn.com/api-keys)
2. **Set environment variable:**
   ```bash
   export SUPERBRYN_API_KEY=your_api_key_here
   ```

### Installation

```bash
pip install superbryn-pipecat-observer
```

### Integration (2 Lines)

Add these lines to your Pipecat agent:

```python
from pipecat.pipeline.task import PipelineTask, PipelineParams
from superbryn_pipecat_observer import SuperbrynObserver  # 1. Import

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        enable_usage_metrics=True,                                  # required for token / TTS char counts
        observers=[SuperbrynObserver(agent_name="support-bot")],    # 2. Add observer
    ),
)
```

**That's it!** 🎉 Every completed call shows up in your SuperBryn **Monitor → Calls** view within seconds of session end.

> ⚠️ **Important:** `enable_usage_metrics=True` must be set on `PipelineParams` for LLM token counts and TTS character counts to be captured.

## 📖 Full Example

Here's a complete working example:

```python
import os

from pipecat.pipeline.pipeline import Pipeline
from pipecat.pipeline.runner import PipelineRunner
from pipecat.pipeline.task import PipelineParams, PipelineTask
from pipecat.services.cartesia.tts import CartesiaTTSService
from pipecat.services.deepgram.stt import DeepgramSTTService
from pipecat.services.openai.llm import OpenAILLMService
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport

from superbryn_pipecat_observer import SuperbrynObserver


async def main(transport: FastAPIWebsocketTransport) -> None:
    stt = DeepgramSTTService(api_key=os.environ["DEEPGRAM_API_KEY"])
    llm = OpenAILLMService(api_key=os.environ["OPENAI_API_KEY"], model="gpt-4o-mini")
    tts = CartesiaTTSService(
        api_key=os.environ["CARTESIA_API_KEY"],
        voice_id="your-voice-id",
    )

    pipeline = Pipeline([
        transport.input(),
        stt,
        llm,
        tts,
        transport.output(),
    ])

    # Drop in the observer — that's the whole integration
    task = PipelineTask(
        pipeline,
        params=PipelineParams(
            enable_usage_metrics=True,
            observers=[
                SuperbrynObserver(
                    agent_name="support-bot",
                    transport="websocket",
                ),
            ],
        ),
    )

    await PipelineRunner().run(task)
```

## 🔧 Configuration

### Environment Variables

| Variable | Required | Description | Default |
|----------|----------|-------------|---------|
| `SUPERBRYN_API_KEY` | ✅ Yes | API key for webhook authentication | - |
| `AGENT_ID` | ⚪ Optional | Unique agent identifier stamped on every call | `"pipecat-agent"` |
| `VERSION_ID` | ⚪ Optional | Agent version identifier | `"v1"` |

### Setting Environment Variables

**Linux/Mac:**
```bash
export SUPERBRYN_API_KEY=your_api_key_here
```

**Windows (CMD):**
```cmd
set SUPERBRYN_API_KEY=your_api_key_here
```

**Windows (PowerShell):**
```powershell
$env:SUPERBRYN_API_KEY="your_api_key_here"
```

**Docker:**
```bash
docker run -e SUPERBRYN_API_KEY=your_api_key_here ...
```

**.env file:**
```env
SUPERBRYN_API_KEY=your_api_key_here
AGENT_ID=customer-support-bot
VERSION_ID=v1.2.0
```

## 📊 What Gets Tracked

### Transcript Data
- Speaker turns (`user` / `agent`)
- Per-turn `start_time_ms` / `end_time_ms` (relative to call start)
- Per-turn confidence score (when STT provides it)
- Per-bot-turn `latency_ms` (gap between user stop and bot response start)
- Empty turns are filtered out before send

### Usage Metrics (`MetricsFrame`)
Captured automatically when `enable_usage_metrics=True`:
- **LLM:** input tokens, output tokens, model, provider (from `LLMUsageMetricsData`)
- **TTS:** character count, model, provider, voice ID (from `TTSUsageMetricsData`)
- **STT:** model, provider (audio duration is reserved for future use — Pipecat does not currently emit an STT usage metric)

### Latency Metrics
- Per-turn response delay (user stop → bot speak start)
- Aggregated **average** and **p95** across the call
- Surfaced under `call.latency` in the payload

### Session Metadata
- Agent ID, agent name, transport (e.g. `daily`, `twilio`, `webrtc`)
- LLM / STT / TTS provider, model, voice ID
- Pipeline / SDK version tag
- Optional `recording_url`, `from_number`, `to_number`
- Any `extra_metadata` you pass in
- Call end reason (`completed` / `cancelled`)

## 🔍 How It Works

1. **Pipeline Observation** - Registered via `PipelineParams(observers=[...])` — runs **alongside** your pipeline, not inside it
2. **Frame Inspection** - `on_push_frame` watches every frame flowing between processors and aggregates by class name (so a Pipecat upgrade doesn't break it)
3. **Auto-Detection** - Inspects the module path of each service (`*.stt.*`, `*.llm.*`, `*.tts.*`) to tag providers automatically
4. **Webhook Delivery** - When `on_pipeline_finished` fires, builds a normalized call payload and POSTs it to SuperBryn

### Webhook Payload Format

```json
{
  "event": "call.completed",
  "sdk_version": "@superbryn/pipecat-observer@0.1.0",
  "call": {
    "id": "550e8400-e29b-41d4-a716-446655440000",
    "session_id": "550e8400-e29b-41d4-a716-446655440000",
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
          "confidence": null
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
    "recording_url": "https://...",
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
      "pipeline_version": "@superbryn/pipecat-observer@0.1.0"
    },
    "usage": {
      "llm_input_tokens": 1250,
      "llm_output_tokens": 850,
      "stt_duration_seconds": 0.0,
      "tts_characters": 1200
    },
    "latency": {
      "avg_ms": 750.5,
      "p95_ms": 1240.0
    }
  }
}
```

#### Field availability notes

- **`transcript.turns[].confidence`** — present in the schema for forward compatibility, but Pipecat's stable `TranscriptionFrame` (as of 1.3.0) does not carry a confidence score, so this field is `null` in practice today.
- **`usage.stt_duration_seconds`** — placeholder; Pipecat 1.3.0 does not emit an STT usage metric (no `STTUsageMetricsData` exists upstream), so this field is always `0.0`. It will start populating automatically once Pipecat ships STT usage metrics.
- **`recording_url`, `from_number`, `to_number`, `transport`** — only set if you pass them into `SuperbrynObserver(...)`. Otherwise `null`.

## 🛠️ Advanced Usage

### Custom API Key

Pass the API key directly instead of using the environment variable:

```python
SuperbrynObserver(
    agent_name="support-bot",
    api_key="sb_live_...",
)
```

### Telephony / Transport Metadata

```python
SuperbrynObserver(
    agent_name="support-bot",
    transport="twilio",            # "daily" | "twilio" | "webrtc" | "websocket" | ...
    from_number="+15551234567",
    to_number="+15557654321",
    recording_url="https://...",   # if your transport produces one
)
```

### Custom Metadata

Attach any key/value pairs you want to see on the call record:

```python
SuperbrynObserver(
    agent_name="support-bot",
    extra_metadata={
        "campaign": "summer-promo",
        "tenant_id": "acme-corp",
        "experiment": "prompt-v3",
    },
)
```

### Explicit Agent ID

Override the `AGENT_ID` env var per-observer:

```python
SuperbrynObserver(
    agent_name="support-bot",
    agent_id="prod-support-v3",
)
```

## 🐛 Troubleshooting

### Webhook Not Sending

**Check API Key:**
```bash
echo $SUPERBRYN_API_KEY
```

**Enable Debug Logging:**
```python
import logging
logging.basicConfig(level=logging.DEBUG)
logging.getLogger("superbryn_pipecat_observer").setLevel(logging.DEBUG)
```

**Look for these log messages:**
- `SUPERBRYN_PIPECAT_CALL_STARTED` - Observer initialized for this session
- `SUPERBRYN_PIPECAT_SENDING` - Webhook about to be sent
- `SUPERBRYN_PIPECAT_SENT` - Webhook delivered successfully
- `SUPERBRYN_PIPECAT_AUTH_FAILED` - Invalid or revoked API key (401/403)
- `SUPERBRYN_PIPECAT_HTTP_*` - Non-2xx response from server
- `SUPERBRYN_PIPECAT_ERROR` - Network / exception during delivery
- `SUPERBRYN_PIPECAT_SKIPPED` - No API key configured (observer no-ops)
- `SUPERBRYN_PIPECAT_MISSING_AIOHTTP` - `aiohttp` not installed in the runtime

### Common Errors

| Error | Cause | Solution |
|-------|-------|----------|
| `SUPERBRYN_API_KEY not configured` | Missing API key | Set `SUPERBRYN_API_KEY` environment variable |
| `SUPERBRYN_PIPECAT_AUTH_FAILED (401)` | Invalid API key | Verify the key in SuperBryn Monitor → API keys |
| `SUPERBRYN_PIPECAT_AUTH_FAILED (403)` | Expired / disabled key | Generate a new API key |
| `SUPERBRYN_PIPECAT_MISSING_AIOHTTP` | `aiohttp` missing | `pip install aiohttp>=3.9.0` |

### Missing Usage Metrics

Pipecat only emits `MetricsFrame`s when usage metrics are enabled at the task level:

```python
PipelineParams(enable_usage_metrics=True, observers=[...])
```

Without this flag, you'll still get the transcript, latency, and provider detection — but `llm_input_tokens`, `llm_output_tokens`, and `tts_characters` will be `0`.

### Provider Detection Issues

The observer auto-detects providers from the producing service's module path and model name. Supported providers include:

**LLM:** OpenAI, Anthropic, Google (Gemini), Meta (Llama), Mistral, Cohere, Perplexity, Groq, Together AI, Replicate, Hugging Face

**STT:** Deepgram, AssemblyAI, Rev.ai, Speechmatics, Gladia, OpenAI Whisper

**TTS:** ElevenLabs, Cartesia, PlayHT, Resemble AI, Murf, WellSaid Labs, Speechify, Sarvam, Azure, AWS Polly, Google Cloud

**Transport:** Daily, Twilio, LiveKit, Vonage

If your provider isn't detected, it will show as `"unknown"` but the call still tracks normally.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [Pipecat Documentation](https://docs.pipecat.ai/)
- [GitHub Repository](https://github.com/superbryndev/superbryn-pipecat-observer)
- [Issue Tracker](https://github.com/superbryndev/superbryn-pipecat-observer/issues)
- [Get API Key](https://app.superbryn.com/api-keys)

## 💡 Support

- 📧 Email: support@superbryn.com
- 💬 GitHub Issues: [Report a bug](https://github.com/superbryndev/superbryn-pipecat-observer/issues)
- 📚 Documentation: [README](https://github.com/superbryndev/superbryn-pipecat-observer#readme)

---

Made with ❤️ by [SuperBryn](https://www.superbryn.com)
