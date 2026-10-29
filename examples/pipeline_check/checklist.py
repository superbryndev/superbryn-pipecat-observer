"""Compares what attach() sent against what actually happened."""

from __future__ import annotations

import io
import wave
from dataclasses import dataclass, field
from datetime import datetime
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
    surface: str  # "observability" (a production call) or "testing" (a SuperBryn test call)
    call_id: str
    custom_data: dict[str, Any]
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
    ended = _report_checks(check, caught, truth, want, wall_seconds)
    if ended is not None and want.surface == "observability":
        _production_checks(check, caught, truth, want, ended)
    elif ended is not None:
        check("test", "test ID sent", ended.get("testing") == {"superbryn_call_id": want.call_id})
        check("test", "no recording", not caught.presign_requests and "observability" not in ended)
        check("test", "no session logs", "logs" not in (ended.get("diagnostics") or {}))
    return checks


# ── attach() → /public-api/v1/calls/agent-data ─────────────────────────────


def _report_checks(check, caught, truth, want, wall_seconds) -> dict[str, Any] | None:
    a = "report"
    kinds = [e.get("event") for e in caught.events]
    check(a, "call.started then call.ended", kinds == ["call.started", "call.ended"], kinds)
    ended = next((e for e in caught.events if e.get("event") == "call.ended"), None)
    if ended is None:
        return None
    check(a, "X-API-Key header sent", caught.event_headers[-1].get("X-API-Key") == want.api_key)
    check(
        a,
        "surface, source and call id",
        (ended.get("surface"), ended.get("source"), ended.get("call_id"))
        == (want.surface, "pipecat", want.call_id),
        (ended.get("surface"), ended.get("source"), ended.get("call_id")),
    )
    check(a, "agent_id sent", ended.get("agent_id") == want.agent_id, ended.get("agent_id"))
    check(a, "ended_reason set", ended.get("ended_reason") == "ended", ended.get("ended_reason"))
    seconds = (_time(ended["ended_at"]) - _time(ended["started_at"])).total_seconds()
    check(
        a,
        "duration within 2s of wall clock",
        abs(seconds - wall_seconds) <= 2,
        f"{seconds:.2f}s vs {wall_seconds:.2f}s",
    )
    diag = ended.get("diagnostics") or {}
    turns = ended.get("transcript") or []

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
    starts = [t.get("start_ms") for t in turns]
    check(
        a,
        "turn start times increase",
        all(x <= y for x, y in zip(starts, starts[1:], strict=False)),
        starts,
    )

    # Latency: raw per-reply parts; SuperBryn works out averages
    latencies = diag.get("turn_latencies") or []
    reported = [e.get("turn_ms") for e in latencies]
    check(
        a,
        "one turn latency per caller line",
        len(reported) == len(truth.perceived_latency_ms),
        reported,
    )
    if len(reported) == len(truth.perceived_latency_ms):
        _latency_check(check, a, reported, truth.perceived_latency_ms)
    parts = ("endpointing_ms", "transcriber_ms", "model_ms", "voice_ms")
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
    if STT_USAGE_REPORTED:
        # Pipecat measures the audio it hands the STT, pre-speech buffer included
        stt = (usage.get("stt_audio_ms") or 0) / 1000
        check(
            a,
            "STT seconds = audio sent to the STT",
            abs(stt - truth.stt_audio_seconds) <= 0.1,
            f"{stt} vs {truth.stt_audio_seconds:.2f}",
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

    # Context and custom data
    context = ended.get("context") or {}
    check(a, "system prompt captured", bool(context.get("system_prompt")))
    environment = context.get("environment") or {}
    check(a, "transport named", environment.get("transport") == "localmock", environment)
    check(
        a,
        "custom_data sent as given",
        ended.get("custom_data") == want.custom_data,
        ended.get("custom_data"),
    )
    return ended


def _production_checks(check, caught, truth, want, ended) -> None:
    a = "production"
    observability = ended.get("observability") or {}
    check(a, "mode observe", observability.get("mode") == "observe", observability)
    check(
        a,
        "one WAV uploaded for this call",
        [r.get("session_id") for r in caught.presign_requests] == [want.call_id]
        and len(caught.uploads) == 1,
    )
    url = observability.get("stereo_recording_url")
    check(
        a,
        "stereo_recording_url points at the upload",
        bool(url) and url.rsplit("/", 1)[-1] in caught.uploads,
        url,
    )
    check(
        a,
        "recording_url_source superbryn_s3",
        observability.get("recording_url_source") == "superbryn_s3"
        and observability.get("recording_url") == url,
    )
    seconds = (_time(ended["ended_at"]) - _time(ended["started_at"])).total_seconds()
    for wav_bytes in list(caught.uploads.values())[:1]:
        _wav_checks(check, a, wav_bytes, seconds, truth)
    messages = [e.get("message") for e in (ended.get("diagnostics") or {}).get("logs") or []]
    for source, line in want.log_lines.items():
        check(a, f"agent log via {source} captured", line in messages, f"{len(messages)} lines")


def _time(stamp: str) -> datetime:
    return datetime.fromisoformat(stamp.replace("Z", "+00:00"))


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
    # The recorder sits before the output transport, as 0.11's did: TTS audio a barge-in cut off is
    # recorded too (about 1.5 s here), so the WAV runs a little past the call's wall-clock length
    check(
        area,
        "WAV length within 2s of the call's",
        abs(seconds - duration) <= 2.0,
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
