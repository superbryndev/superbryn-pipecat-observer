# superbryn-pipecat-observer

Drop-in [Pipecat](https://github.com/pipecat-ai/pipecat) observer that automatically reports call transcripts, metrics, and usage to [SuperBryn](https://superbryn.com) at the end of each session.

Set an env var, add one observer to your pipeline, done.

## Install

```bash
pip install superbryn-pipecat-observer
```

## Quick start

1. Generate a SuperBryn API key in **Monitor → Add Source → Pipecat → Generate API key**.
2. Export it:

   ```bash
   export SUPERBRYN_API_KEY="sb_live_..."
   ```

3. Add the observer to your Pipecat `PipelineTask`:

   ```python
   from pipecat.pipeline.task import PipelineTask, PipelineParams
   from superbryn_pipecat_observer import SuperbrynObserver

   task = PipelineTask(
       pipeline,
       params=PipelineParams(
           enable_usage_metrics=True,          # required for token / TTS char counts
           observers=[SuperbrynObserver(agent_name="support-bot")],
       ),
   )
   ```

That's it. Every completed call shows up in your SuperBryn **Monitor → Calls** view within seconds of session end.

## Environment variables

| Variable | Required | Description |
|---|---|---|
| `SUPERBRYN_API_KEY` | Yes | API key from SuperBryn Monitor |
| `SUPERBRYN_WEBHOOK_URL` | No | Override the ingest URL (defaults to SuperBryn production) |
| `AGENT_ID` | No | Identifier stamped on each call's metadata. Defaults to `pipecat-agent`. |

## What gets captured

| Field | Source |
|---|---|
| Transcript turns + timestamps | `TranscriptionFrame`, `TextFrame`, `LLMFullResponseStartFrame`, `BotStartedSpeakingFrame` |
| LLM provider, model, token counts | `MetricsFrame` → `LLMUsageMetricsData` (requires `enable_usage_metrics=True`) |
| TTS provider, model, voice, characters | `MetricsFrame` → `TTSUsageMetricsData` |
| STT provider, model | Inferred from the producing service's module |
| Response latency (avg, p95) | Time between `UserStoppedSpeakingFrame` and bot response start |
| Call end reason | `EndFrame` / `StopFrame` / `CancelFrame` |

## Constructor options

```python
SuperbrynObserver(
    agent_name="support-bot",          # shown in SuperBryn UI
    agent_id="prod-support-v3",        # optional explicit ID; else AGENT_ID env / default
    transport="daily",                 # e.g. "daily", "twilio", "webrtc"
    from_number="+1...",               # for telephony calls
    to_number="+1...",
    recording_url="https://...",       # if your transport produces one
    extra_metadata={"campaign": "...", "tenant_id": "..."},
    api_key="sb_live_...",             # override env
    webhook_url="https://...",         # override env, useful for staging
)
```

## Fail-open by design

The observer never raises into your pipeline. If the API key is missing, the network fails, or any frame-handler errors, the call still completes normally — only the SuperBryn delivery is skipped (and logged).

## License

MIT — see [LICENSE](LICENSE).
