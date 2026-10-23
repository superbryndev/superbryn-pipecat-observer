"""Compares what each SDK path sent against what actually happened."""

from __future__ import annotations

import io
import statistics
import wave
from dataclasses import dataclass, field
from typing import Any

from catcher import Caught
from scripted_services import GroundTruth

try:  # newer Pipecat reports STT usage itself
    from pipecat.metrics.metrics import STTUsageMetricsData  # noqa: F401

    STT_USAGE_REPORTED = True
except ImportError:
    STT_USAGE_REPORTED = False

# Reported reply latency vs caller-measured end of speech → bot audio start
LATENCY_TOLERANCE_MS = 250


@dataclass
class Check:
    area: str
    name: str
    ok: bool
    detail: str = ""


@dataclass
class Expected:
    api_key: str
    agent_id: str
    agent_name: str
    custom_metadata: dict[str, Any]
    # What the bot actually got out, greeting first; the interrupted reply cut to what was heard
    agent_lines: list[str]
    unspoken: list[str]
    interruptions: int
    usage_metrics_requested: bool
    log_lines: dict[str, str] = field(default_factory=dict)
    # Index in agent_lines of the cut-off reply: any non-empty start of that line is right
    cut_off: int | None = None


PROVIDERS = {
    "stt": ("deepgram", "nova-3"),
    "llm": ("openai", "gpt-4o-mini"),
    "tts": ("cartesia", "sonic-2"),
}
VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"


def run_checks(
    caught: Caught,
    truth: GroundTruth,
    want: Expected,
    *,
    session_id: str,
    wall_seconds: float,
) -> list[Check]:
    checks: list[Check] = []

    def check(area: str, name: str, ok: Any, detail: Any = "") -> None:
        checks.append(Check(area, name, bool(ok), str(detail)))

    check("setup", "scripted call ran without anomalies", not truth.anomalies, truth.anomalies)
    check(
        "setup",
        "caller interrupted the bot as scripted",
        truth.interruptions == want.interruptions,
        truth.interruptions,
    )
    _obs_checks(check, caught, truth, want, session_id, wall_seconds)
    _evals_checks(check, caught, truth, want)
    return checks


# ── Obs: SuperbrynObserver → /webhooks/obs/pipecat ──────────────────────────


def _obs_checks(check, caught, truth, want, session_id, wall_seconds) -> None:
    a = "obs"
    check(a, "exactly one webhook", len(caught.webhooks) == 1, len(caught.webhooks))
    if not caught.webhooks:
        return
    payload, headers = caught.webhooks[0], caught.webhook_headers[0]
    call = payload.get("call") or {}
    meta = call.get("metadata") or {}
    usage = call.get("usage") or {}
    turns = (call.get("transcript") or {}).get("turns") or []

    check(a, "X-API-Key header sent", headers.get("X-API-Key") == want.api_key)
    check(a, "event is call.completed", payload.get("event") == "call.completed")
    check(a, "call id is the session id", call.get("id") == session_id)
    duration = call.get("duration_seconds") or 0
    check(
        a,
        "duration within 2s of wall clock",
        abs(duration - wall_seconds) <= 2,
        f"{duration:.2f}s vs {wall_seconds:.2f}s",
    )
    check(a, "end reason completed", call.get("call_end_reason") == "completed")

    # Transcript
    user_texts = [t["text"] for t in turns if t.get("speaker") == "user"]
    agent_texts = [t["text"] for t in turns if t.get("speaker") == "agent"]
    check(
        a,
        "user turns = caller lines",
        user_texts == truth.user_lines,
        _diff(truth.user_lines, user_texts),
    )
    check(
        a,
        "agent turns = what the bot said",
        _same_lines(agent_texts, want),
        _diff(want.agent_lines, agent_texts),
    )
    check(a, "no duplicated agent text", not _repeats(agent_texts), _repeats(agent_texts))
    check(
        a,
        "numbers kept whole (884291)",
        all("884 291" not in t for t in agent_texts),
        [t for t in agent_texts if "884 291" in t],
    )
    for line in want.unspoken:
        check(a, "unheard words left out", all(line not in t for t in agent_texts), line)
    starts = [t.get("start_time_ms") for t in turns]
    check(
        a,
        "turn start times increase",
        all(x <= y for x, y in zip(starts, starts[1:], strict=False)),
        starts,
    )

    # Latency
    first_replies = _first_reply_after_each_user(turns, "speaker", "latency_ms")
    _latency_check(check, a, first_replies, truth.perceived_latency_ms)
    lat = call.get("latency") or {}
    valid = [v for v in first_replies if v is not None]
    if valid:
        check(
            a,
            "avg_ms = mean of reply latencies",
            lat.get("avg_ms") is not None and abs(lat["avg_ms"] - statistics.mean(valid)) < 1,
            f"{lat.get('avg_ms')} vs {statistics.mean(valid):.1f}",
        )

    # Usage
    _usage_check(
        check, a, "LLM input tokens", usage.get("llm_input_tokens"), truth.llm_prompt_tokens, want
    )
    _usage_check(
        check,
        a,
        "LLM output tokens",
        usage.get("llm_output_tokens"),
        truth.llm_completion_tokens,
        want,
    )
    _usage_check(
        check,
        a,
        "TTS characters",
        usage.get("tts_characters"),
        sum(map(len, truth.tts_texts)),
        want,
    )
    stt = usage.get("stt_duration_seconds") or 0
    if STT_USAGE_REPORTED:
        # Pipecat measures the audio it hands the STT, pre-speech buffer included
        check(
            a,
            "STT seconds = audio sent to the STT",
            abs(stt - truth.stt_audio_seconds) <= 0.1,
            f"{stt} vs {truth.stt_audio_seconds:.2f}",
        )
    else:
        # Older Pipecat reports none; the observer falls back to the caller's turns
        check(
            a,
            "STT seconds within 0.6s of caller turns",
            abs(stt - truth.user_turn_seconds) <= 0.6,
            f"{stt} vs {truth.user_turn_seconds:.2f}",
        )

    # Metadata
    expected_meta = {
        "agent_id": want.agent_id,
        "agent_name": want.agent_name,
        "transport": "localmock",
        "llm_provider": PROVIDERS["llm"][0],
        "llm_model": PROVIDERS["llm"][1],
        "stt_provider": PROVIDERS["stt"][0],
        "stt_model": PROVIDERS["stt"][1],
        "tts_provider": PROVIDERS["tts"][0],
        "tts_model": PROVIDERS["tts"][1],
        "tts_voice_id": VOICE_ID,
        "mode": "observe",
        **want.custom_metadata,
    }
    wrong = {k: meta.get(k) for k, v in expected_meta.items() if meta.get(k) != v}
    check(
        a,
        "metadata fields (agent, providers, models, voice, custom)",
        not {k for k in wrong if k != "transport"},
        wrong,
    )
    check(a, "metadata.transport set", meta.get("transport") == "localmock", meta.get("transport"))
    check(a, "system prompt captured", bool(meta.get("system_prompt")))

    # Tools
    sent = call.get("tool_calls") or []
    check(
        a,
        "one record per tool call",
        len(sent) == len(truth.tool_calls),
        f"{len(sent)} vs {len(truth.tool_calls)}",
    )
    _tool_checks(check, a, sent, truth, id_key="tool_call_id", name_key="function_name")

    # Recording
    check(
        a,
        "one WAV uploaded for this session",
        [r.get("session_id") for r in caught.presign_requests] == [session_id]
        and len(caught.uploads) == 1,
    )
    url = call.get("stereo_recording_url")
    check(
        a,
        "stereo_recording_url points at the upload",
        bool(url) and url.rsplit("/", 1)[-1] in caught.uploads,
        url,
    )
    for wav_bytes in list(caught.uploads.values())[:1]:
        _wav_checks(check, a, wav_bytes, duration, truth)

    # Extended sections
    timings = lat.get("by_service") or {}
    check(
        a,
        "per-service TTFB present",
        {"llm", "stt", "tts"} <= {_kind(k) for k, v in timings.items() if "avg_ttfb_ms" in v},
        sorted(timings),
    )
    speech = call.get("speech_stats") or {}
    check(
        a,
        "speech_stats turn counts",
        speech.get("user_turn_count") == len(truth.user_lines)
        and speech.get("agent_turn_count") == len(want.agent_lines),
        f"user {speech.get('user_turn_count')}, agent {speech.get('agent_turn_count')}",
    )
    check(
        a,
        "one VAD segment per caller line",
        (call.get("vad") or {}).get("segment_count") == len(truth.user_lines),
        (call.get("vad") or {}).get("segment_count"),
    )
    interruptions = call.get("interruptions") or {}
    check(
        a,
        "interruption count",
        interruptions.get("interruption_count") == want.interruptions,
        interruptions,
    )
    check(
        a,
        "interrupted bot turns",
        interruptions.get("bot_turns_interrupted") == want.interruptions,
        interruptions,
    )
    check(a, "no pipeline errors", not call.get("errors"), call.get("errors"))
    messages = [e.get("message") for e in call.get("logs") or []]
    for source, line in want.log_lines.items():
        check(a, f"agent log via {source} captured", line in messages, f"{len(messages)} lines")


# ── Evals: attach_test_data → /public-api/v1/test-calls/agent-data ─────────


def _evals_checks(check, caught, truth, want) -> None:
    a = "evals"
    kinds = [e.get("event") for e in caught.test_events]
    check(a, "call.started then call.ended", kinds == ["call.started", "call.ended"], kinds)
    ended = next((e for e in caught.test_events if e.get("event") == "call.ended"), None)
    if ended is None:
        return
    diag = ended.get("diagnostics") or {}
    turns = ended.get("transcript") or []
    check(a, "agent_id sent", ended.get("agent_id") == want.agent_id, ended.get("agent_id"))
    check(a, "ended_reason set", bool(ended.get("ended_reason")), ended.get("ended_reason"))

    # Transcript
    user_texts = [t["text"] for t in turns if t.get("role") == "user"]
    agent_texts = [t["text"] for t in turns if t.get("role") == "agent"]
    check(
        a,
        "user turns = caller lines",
        user_texts == truth.user_lines,
        _diff(truth.user_lines, user_texts),
    )
    check(
        a,
        "agent turns = what the bot said",
        _same_lines(agent_texts, want),
        _diff(want.agent_lines, agent_texts),
    )
    check(a, "no duplicated agent text", not _repeats(agent_texts), _repeats(agent_texts))
    check(a, "numbers kept whole (884291)", all("884 291" not in t for t in agent_texts))
    for line in want.unspoken:
        check(a, "unheard words left out", all(line not in t for t in agent_texts), line)
    flagged = sum(1 for t in turns if t.get("role") == "agent" and t.get("interrupted"))
    check(a, "interrupted turns flagged", flagged == want.interruptions, flagged)
    check(
        a,
        "every turn has start <= end",
        all(
            t.get("start_ms") is not None
            and t.get("end_ms") is not None
            and t["start_ms"] <= t["end_ms"]
            for t in turns
        ),
    )

    # Latency
    latencies = diag.get("turn_latencies") or []
    reported = [e.get("turnLatency") for e in latencies]
    check(
        a,
        "one turn latency per caller line",
        len(reported) == len(truth.perceived_latency_ms),
        reported,
    )
    if len(reported) == len(truth.perceived_latency_ms):
        _latency_check(check, a, reported, truth.perceived_latency_ms)
    parts = ("endpointingLatency", "transcriberLatency", "modelLatency", "voiceLatency")
    check(
        a,
        "latency broken into endpointing/STT/LLM/TTS",
        latencies and all(all(p in e for p in parts) for e in latencies),
        [sorted(k for k in e if k in parts) for e in latencies],
    )

    # Usage
    usage = diag.get("usage") or {}
    _usage_check(
        check, a, "LLM input tokens", usage.get("llm_input_tokens"), truth.llm_prompt_tokens, want
    )
    _usage_check(
        check,
        a,
        "LLM output tokens",
        usage.get("llm_output_tokens"),
        truth.llm_completion_tokens,
        want,
    )
    _usage_check(
        check,
        a,
        "TTS characters",
        usage.get("tts_characters"),
        sum(map(len, truth.tts_texts)),
        want,
    )

    # Configuration
    config = diag.get("configuration") or {}
    for role, (provider, model) in PROVIDERS.items():
        got = config.get(role) or {}
        check(
            a,
            f"{role} provider/model",
            got.get("provider") == provider and got.get("model") == model,
            got,
        )
    check(
        a,
        "tts voice",
        (config.get("tts") or {}).get("voice") == VOICE_ID,
        (config.get("tts") or {}).get("voice"),
    )
    check(
        a,
        "tools listed",
        [t.get("name") for t in config.get("tools") or []] == ["lookup_order"],
        config.get("tools"),
    )
    check(a, "pipeline type", diag.get("pipeline") == "stt_llm_tts", diag.get("pipeline"))

    # Tools + LLM requests
    sent = ended.get("tool_calls") or []
    check(
        a,
        "one record per tool call",
        len(sent) == len(truth.tool_calls),
        f"{len(sent)} vs {len(truth.tool_calls)}",
    )
    _tool_checks(check, a, sent, truth, id_key="id", name_key="name")
    agent_ids = {t["id"] for t in turns if t.get("role") == "agent"}
    check(
        a,
        "tool call tied to an agent turn",
        all(c.get("turn_id") in agent_ids for c in sent),
        [c.get("turn_id") for c in sent],
    )
    requests = diag.get("llm_requests") or []
    outcomes = [r.get("outcome") for r in requests]
    expected_outcomes = [
        "cut_off" if want.interruptions else "spoken",
        "tool_call",
        "spoken",
        "spoken",
    ]
    check(a, "LLM requests and outcomes", outcomes == expected_outcomes, outcomes)
    check(a, "no errors", diag.get("errors") == [], diag.get("errors"))


# ── Shared helpers ──────────────────────────────────────────────────────────


def _usage_check(check, area: str, name: str, got: Any, truth_value: int, want: Expected) -> None:
    if want.usage_metrics_requested:
        check(area, name, got == truth_value, f"{got} vs {truth_value}")
    else:
        # Customer turned on timing metrics only; usage is still what dashboards bill on
        check(
            area,
            f"{name} (customer's own PipelineParams)",
            got == truth_value,
            f"{got} vs {truth_value}",
        )


def _tool_checks(
    check, area: str, sent: list[dict], truth: GroundTruth, *, id_key: str, name_key: str
) -> None:
    for want in truth.tool_calls:
        same = [t for t in sent if t.get(id_key) == want["tool_call_id"]]
        got = next((t for t in same if t.get("result") is not None), same[0] if same else None)
        name = want["function_name"]
        check(area, f"{name} captured with name", got is not None and got.get(name_key) == name)
        if got is not None:
            check(
                area,
                f"{name} arguments",
                got.get("arguments") == want["arguments"],
                got.get("arguments"),
            )
            check(
                area, f"{name} result", got.get("result") == want.get("result"), got.get("result")
            )


def _first_reply_after_each_user(turns: list[dict], role_key: str, latency_key: str) -> list[Any]:
    out = []
    for i, t in enumerate(turns):
        if t.get(role_key) == "user":
            out.append(
                next(
                    (n.get(latency_key) for n in turns[i + 1 :] if n.get(role_key) == "agent"), None
                )
            )
    return out


def _latency_check(check, area: str, reported: list[Any], perceived: list[float]) -> None:
    if len(reported) != len(perceived) or None in reported:
        check(area, "a reply latency for every caller line", False, f"{reported} vs {perceived}")
        return
    gaps = [round(r - p) for r, p in zip(reported, perceived, strict=True)]
    check(
        area,
        f"reply latency within {LATENCY_TOLERANCE_MS}ms of what the caller waited",
        all(abs(g) <= LATENCY_TOLERANCE_MS for g in gaps),
        f"reported {reported} vs waited {perceived} (diff {gaps})",
    )


def _wav_checks(check, area: str, wav_bytes: bytes, duration: float, truth: GroundTruth) -> None:
    with wave.open(io.BytesIO(wav_bytes)) as w:
        channels, rate, frames = w.getnchannels(), w.getframerate(), w.readframes(w.getnframes())
    seconds = len(frames) / (2 * channels * rate)
    check(area, "WAV is stereo", channels == 2, channels)
    check(
        area,
        "WAV length matches call duration",
        abs(seconds - duration) <= 1.0,
        f"{seconds:.2f}s vs {duration:.2f}s",
    )
    if channels != 2:
        return
    samples = memoryview(frames).cast("h")
    left = sum(1 for i in range(0, len(samples), 2) if abs(samples[i]) > 1000) / rate
    right = sum(1 for i in range(1, len(samples), 2) if abs(samples[i]) > 1000) / rate
    # Tone peaks pass the threshold ~70% of the time, so expect at least half the audio
    check(
        area,
        "left channel carries the caller",
        left >= 0.5 * truth.user_audio_seconds,
        f"{left:.2f}s vs {truth.user_audio_seconds:.2f}s",
    )
    check(area, "right channel carries the bot", right >= 0.3, f"{right:.2f}s loud")


def _same_lines(got: list[str], want: Expected) -> bool:
    if len(got) != len(want.agent_lines):
        return False
    for i, (g, w) in enumerate(zip(got, want.agent_lines, strict=True)):
        if i == want.cut_off:
            if not g or not w.startswith(g.rstrip(".")):
                return False
        elif g != w:
            return False
    return True


def _repeats(texts: list[str]) -> list[str]:
    """Agent turns whose text repeats the end of the turn before."""
    return [b for a, b in zip(texts, texts[1:], strict=False) if b and a.endswith(b)]


def _kind(processor: str) -> str:
    name = processor.lower()
    return next((k for k in ("stt", "llm", "tts") if k in name), name)


def _diff(want: list[str], got: list[str]) -> str:
    if want == got:
        return ""
    lines = []
    for i in range(max(len(want), len(got))):
        w = want[i] if i < len(want) else "—"
        g = got[i] if i < len(got) else "—"
        if w != g:
            lines.append(f"#{i + 1} want {w!r} got {g!r}")
    return "; ".join(lines)


def print_report(checks: list[Check]) -> None:
    area = None
    for c in checks:
        if c.ok:
            continue
        if c.area != area:
            area = c.area
            print(f"  {area.upper()}")
        print(f"    [FAIL] {c.name}  — {c.detail}")
    failed = sum(1 for c in checks if not c.ok)
    print(f"  {len(checks) - failed}/{len(checks)} checks passed")


def print_summary(results: dict[str, list[Check]]) -> None:
    names = list(results)
    rows: dict[tuple[str, str], dict[str, str]] = {}
    for scenario, checks in results.items():
        for c in checks:
            rows.setdefault((c.area, c.name), {})[scenario] = "ok" if c.ok else "FAIL"
    failing = {k: v for k, v in rows.items() if "FAIL" in v.values()}
    print("\n=== Failing checks by scenario (blank = check not run)")
    print("  " + " | ".join(n[:10].ljust(10) for n in names) + " | check")
    for (area, name), marks in failing.items():
        cells = " | ".join(marks.get(n, "").ljust(10) for n in names)
        print(f"  {cells} | {area}: {name}")
