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
- 🎥 **Auto-Recording (Daily / Twilio)** - Pass the transport object and the observer pulls the recording URL from the transport's own API. See [Automatic Recording](#automatic-recording-daily--twilio)
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

### Integration

Add the observer to your `PipelineTask`. Pick the path that matches your transport — the observer auto-fetches the recording URL for Daily and Twilio, otherwise you pass it through.

**Option A — Daily / Twilio (auto-fetch recording URL):**

```python
from pipecat.pipeline.task import PipelineTask, PipelineParams
from superbryn_pipecat_observer import SuperbrynObserver

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        enable_usage_metrics=True,                       # required for token / TTS char counts
        observers=[
            SuperbrynObserver(
                agent_name="support-bot",
                transport=daily_transport,               # or twilio_transport
            ),
        ],
    ),
)
# requires DAILY_API_KEY (Daily) or TWILIO_ACCOUNT_SID + TWILIO_AUTH_TOKEN (Twilio)
```

**Option B — Other transport (Plivo / Telnyx / WebSocket / SmallWebRTC / video avatars / …):**

SuperBryn does not record audio itself. Record on your side and pass the URL through:

```python
from pipecat.pipeline.task import PipelineTask, PipelineParams
from superbryn_pipecat_observer import SuperbrynObserver

task = PipelineTask(
    pipeline,
    params=PipelineParams(
        enable_usage_metrics=True,                       # required for token / TTS char counts
        observers=[
            SuperbrynObserver(
                agent_name="support-bot",
                transport="plivo",                       # string label
                recording_url="https://your-bucket/recordings/abc.mp3",
            ),
        ],
    ),
)
```

**That's it!** 🎉 Every completed call shows up in your SuperBryn **Monitor → Calls** view within seconds of session end.

> ⚠️ **Important:** `enable_usage_metrics=True` must be set on `PipelineParams` for LLM token counts and TTS character counts to be captured.
>
> Recording is best-effort. If the auto-fetch fails (missing env var, transport not finalized, network error) the call record still ships — just without `recording_url`.

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
| `DAILY_API_KEY` | ⚪ Optional | Required only if you pass a `DailyTransport` and want auto-recording | - |
| `TWILIO_ACCOUNT_SID` | ⚪ Optional | Required only for Twilio auto-recording | - |
| `TWILIO_AUTH_TOKEN` | ⚪ Optional | Required only for Twilio auto-recording | - |
| `TWILIO_CALL_SID` | ⚪ Optional | Override if Twilio CallSid auto-detection fails | - |

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
  "sdk_version": "@superbryn/pipecat-observer@0.2.0",
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
      "pipeline_version": "@superbryn/pipecat-observer@0.2.0"
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

### Telephony Metadata

Attach phone numbers (and optionally a recording URL fallback) to any call:

```python
SuperbrynObserver(
    agent_name="support-bot",
    transport=twilio_transport,        # see Integration above for the two transport paths
    from_number="+15551234567",
    to_number="+15557654321",
)
```

These show up on the call record so you can filter / segment by direction, area code, or DID.

### Pipecat Transports — Compatibility Matrix

| Transport | Pipecat class | Typical use case | Recording option |
|-----------|---------------|------------------|------------------|
| **Daily** | `DailyTransport` | Voice/video calls via Daily Cloud (WebRTC) — most popular all-in-one | ✅ Native (Daily Cloud) |
| **Twilio** | `FastAPIWebsocketTransport` + Twilio serializer | Phone calls via Twilio Media Streams — most common for telephony | ✅ Native (Twilio REST) |
| **Plivo / Telnyx / Exotel / Vonage** | `FastAPIWebsocketTransport` + respective serializer | Other carrier telephony — same shape as Twilio | ⚠️ Manual (pass `recording_url=...` from carrier REST) |
| **Generic WebSocket** | `WebsocketServerTransport` | Custom apps with their own WS clients (Unity, mobile, etc.) | ❌ No recording (transport has no recording API) |
| **SmallWebRTC** | `SmallWebRTCTransport` | Direct browser↔agent WebRTC (OpenAI Realtime-style demos) | ❌ No recording (transport has no recording API) |
| **Tavus / HeyGen / Simli** | Video-avatar transports | Talking-head video agents | ⚠️ Manual (pass `recording_url=...` from avatar service) |
| **Local audio** | `LocalAudioTransport` | Dev/testing with mic & speakers | ❌ Not applicable |

> **"Native"** = the observer calls the transport's own recording API (Daily
> Cloud, Twilio REST) and stamps the URL it returns. SuperBryn does **not**
> record audio itself — your transport owns the audio data; we only surface
> the URL it produces.

> 💡 **Using LiveKit?** Don't use Pipecat + LiveKit + this package. Use
> [`livekit-evals`](https://github.com/superbryndev/livekit-evals) directly on
> top of LiveKit Agents — it integrates more deeply and produces stereo
> recordings out of the box.

### Automatic Recording (Daily / Twilio)

For Daily and Twilio, pass the **transport object itself** (instead of a string
label) and the observer wires up recording for you — no manual URL plumbing:

```python
observer = SuperbrynObserver(
    agent_name="support-bot",
    transport=daily_transport,   # ← live transport object, not a string
)
```

The observer detects the transport type at runtime and dispatches to the
matching adapter. Detection is by class name + module path, so a Pipecat
upgrade doesn't break it.

Pick your transport below for a copy-paste setup.

---

#### Daily setup

**1. Install:**

```bash
pip install superbryn-pipecat-observer
```

**2. Set env vars:**

```bash
export SUPERBRYN_API_KEY=sb_live_...
export DAILY_API_KEY=...        # same key you use to create Daily rooms
```

**3. Wire the observer:**

```python
from pipecat.transports.services.daily import DailyTransport
from superbryn_pipecat_observer import SuperbrynObserver

daily_transport = DailyTransport(...)

observer = SuperbrynObserver(
    agent_name="support-bot",
    transport=daily_transport,   # auto-records to Daily cloud + fetches URL
)
```

> Daily finalizes recordings asynchronously. If the access link isn't ready
> within ~6 seconds after the call ends, the observer ships the call without
> a URL and logs the recording id so you can backfill later.

---

#### Twilio setup

**1. Install:**

```bash
pip install superbryn-pipecat-observer
```

**2. Set env vars:**

```bash
export SUPERBRYN_API_KEY=sb_live_...
export TWILIO_ACCOUNT_SID=AC...
export TWILIO_AUTH_TOKEN=...
```

**3. Enable recording on the Twilio side** (this SDK does **not** start
recording — Twilio doesn't expose a safe mid-call API for that):

```python
# When creating the call
client.calls.create(
    url="https://your.app/voice/twiml",
    to="+15557654321",
    from_="+15551234567",
    record=True,                     # or "record-from-answer"
)
```

**4. Wire the observer:**

```python
from pipecat.transports.network.fastapi_websocket import FastAPIWebsocketTransport
from superbryn_pipecat_observer import SuperbrynObserver

twilio_transport = FastAPIWebsocketTransport(...)   # with the Twilio serializer

observer = SuperbrynObserver(
    agent_name="support-bot",
    transport=twilio_transport,
)
```

**CallSid hint:** Pipecat's Twilio integration doesn't always expose `CallSid`
on a stable attribute. The adapter sniffs several common locations; if
auto-detection fails, pass it explicitly:

```python
SuperbrynObserver(
    agent_name="support-bot",
    transport=twilio_transport,
    extra_metadata={"call_sid": "CA..."},   # or set TWILIO_CALL_SID env var
)
```

---

#### Other transports (Plivo / Telnyx / WebSocket / SmallWebRTC / video avatars / …)

For everything that isn't Daily or Twilio, SuperBryn does not record audio.
Your transport (or carrier) owns the audio data — you record it there and
pass the URL through:

```python
SuperbrynObserver(
    agent_name="support-bot",
    transport="plivo",                                  # or "telnyx", "websocket", etc.
    recording_url="https://your-bucket/recordings/abc.mp3",
)
```

Pick the option that fits your transport:

- **Carrier-managed (Plivo / Telnyx / Exotel / Vonage)** — enable recording
  on the carrier when you place the call, then fetch the URL from their REST
  API after the call ends and pass it in via `recording_url=...`
- **WebRTC / WebSocket / SmallWebRTC** — there is no recording API. If you
  need audio, capture it on your own side (e.g. browser `MediaRecorder` or
  server-side ffmpeg piping the WS payload to S3) and pass the URL through
- **Video avatars (Tavus / HeyGen / Simli)** — use the avatar service's
  built-in recording / replay feature and pass that URL through

> **Why doesn't the SDK record for you?** Audio recording involves PII,
> storage cost, and retention/compliance policy that should live with your
> infrastructure, not ours. We keep the observer focused on transcripts,
> metrics, latency, and usage — the things only the pipeline can produce.
> Want first-class adapters for more carriers? Open an issue.

---

#### Fallback behavior

If the auto-recording adapter fails to fetch a URL (missing env vars, network
error, transport not yet finalized), the observer ships the call record
without a `recording_url`. **Recording wiring never crashes your agent.**

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

**Transport label (stamped on the call):** auto-derived from the transport object class name; common values include `daily`, `twilio`, `websocket`, `smallwebrtc`, `plivo`, `telnyx`. Pass a string explicitly if you want a custom label.

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
