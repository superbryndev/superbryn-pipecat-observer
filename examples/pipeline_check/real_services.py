"""
Real-services check: Deepgram, OpenAI and Cartesia behind the SDK, on this machine.

The scripted pipeline check (run.py) proves capture logic with stand-in services. This
run swaps in the real ones: real streaming STT, real token streaming and usage,
real word-timed TTS over a websocket, and Silero VAD deciding turns. The caller
is recorded audio. Each service is wrapped to record what it reported to
Pipecat (tokens, TTS text, STT seconds, tool calls); the call report attach()
sends for the production call is checked against those records.

Needs DEEPGRAM_API_KEY, OPENAI_API_KEY, CARTESIA_API_KEY (environment or
--env-file) and pipecat-ai[deepgram,openai,cartesia,silero]. Not run in CI:
it costs a few cents a run.

    python examples/pipeline_check/real_services.py --env-file ../.env
    python examples/pipeline_check/real_services.py --barge-in --dump out/
"""

from __future__ import annotations

import argparse
import asyncio
import io
import json
import logging
import os
import re
import subprocess
import sys
import tempfile
import time
import wave
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent))

from catcher import Catcher  # noqa: E402
from checklist import Check, print_report  # noqa: E402
from loguru import logger as loguru_logger  # noqa: E402
from pipecat.adapters.schemas.function_schema import FunctionSchema  # noqa: E402
from pipecat.adapters.schemas.tools_schema import ToolsSchema  # noqa: E402
from pipecat.audio.vad.silero import SileroVADAnalyzer  # noqa: E402
from pipecat.frames.frames import (  # noqa: E402
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    EndFrame,
    Frame,
    InputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    TTSSpeakFrame,
)
from pipecat.pipeline.pipeline import Pipeline  # noqa: E402
from pipecat.pipeline.runner import PipelineRunner  # noqa: E402
from pipecat.pipeline.task import PipelineParams, PipelineTask  # noqa: E402
from pipecat.processors.aggregators.llm_context import LLMContext  # noqa: E402
from pipecat.processors.aggregators.llm_response_universal import (  # noqa: E402
    LLMContextAggregatorPair,
    LLMUserAggregatorParams,
)
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor  # noqa: E402
from pipecat.services.cartesia.tts import CartesiaTTSService  # noqa: E402
from pipecat.services.deepgram.stt import DeepgramSTTService  # noqa: E402
from pipecat.services.llm_service import FunctionCallParams  # noqa: E402
from pipecat.services.openai.llm import OpenAILLMService  # noqa: E402
from pipecat.turns.user_stop import SpeechTimeoutUserTurnStopStrategy  # noqa: E402
from pipecat.turns.user_turn_strategies import UserTurnStrategies  # noqa: E402
from scripted_services import LocalMockTransport  # noqa: E402

from superbryn_pipecat_observer import attach  # noqa: E402

API_KEY = "sb_pipeline_check"
AGENT_ID = "6f1c2a8e-3b4d-4e5f-8a9b-0c1d2e3f4a5b"
LLM_MODEL = "gpt-4.1-mini"
VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"
RATE = 16000
CHUNK_SECONDS = 0.02
LATENCY_TOLERANCE_MS = 250

GREETING = "Hello, thanks for calling Acme support. How can I help you today?"
CALLER_LINES = [
    "Hi, I need help with my recent order.",
    "It's order number 884291.",
    "Great, thanks for your help.",
]
SYSTEM_PROMPT = (
    "You are Acme's phone support agent. Answer in one or two short sentences. "
    "If you don't have an order number yet, ask for it, mentioning it is on the "
    "confirmation email. When the caller gives one, call lookup_order with it, then "
    "read back the order number digits and tell them the status and delivery day. "
    "When they thank you, say goodbye."
)
ORDER_RESULT = {"status": "shipped", "eta": "Friday"}
LOG_LINE = "lookup_order ran"


@dataclass
class Truth:
    """What the real services reported to Pipecat, recorded at the service."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    tts_texts: list[str] = field(default_factory=list)
    stt_seconds: float = 0.0
    stt_reported: bool = False
    transcripts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    perceived_latency_ms: list[float] = field(default_factory=list)
    barge_ins: int = 0
    anomalies: list[str] = field(default_factory=list)


TRUTH = Truth()


class TappedDeepgramSTT(DeepgramSTTService):
    async def start_stt_usage_metrics(self, usage: Any) -> None:
        TRUTH.stt_reported = True
        TRUTH.stt_seconds += float(getattr(usage, "audio_seconds", usage) or 0)
        await super().start_stt_usage_metrics(usage)

    async def push_frame(self, frame: Frame, direction: FrameDirection = FrameDirection.DOWNSTREAM):
        if isinstance(frame, TranscriptionFrame) and frame.text.strip():
            TRUTH.transcripts.append(frame.text.strip())
        await super().push_frame(frame, direction)


class TappedOpenAILLM(OpenAILLMService):
    async def start_llm_usage_metrics(self, tokens: Any) -> None:
        TRUTH.prompt_tokens += int(tokens.prompt_tokens or 0)
        TRUTH.completion_tokens += int(tokens.completion_tokens or 0)
        await super().start_llm_usage_metrics(tokens)

    async def run_function_calls(self, function_calls: Any) -> None:
        for call in function_calls:
            TRUTH.tool_calls.append(
                {
                    "id": call.tool_call_id,
                    "name": call.function_name,
                    "arguments": dict(call.arguments),
                }
            )
        await super().run_function_calls(function_calls)


class TappedCartesiaTTS(CartesiaTTSService):
    async def start_tts_usage_metrics(self, text: str) -> None:
        TRUTH.tts_texts.append(text)
        await super().start_tts_usage_metrics(text)


# The report names providers from the service's module, as it would the real class
TappedDeepgramSTT.__module__ = DeepgramSTTService.__module__
TappedOpenAILLM.__module__ = OpenAILLMService.__module__
TappedCartesiaTTS.__module__ = CartesiaTTSService.__module__


def record_line(text: str, folder: Path) -> tuple[bytes, float]:
    """The caller's line as 16 kHz PCM (macOS `say`) and where its speech ends, in seconds."""
    path = folder / f"{abs(hash(text))}.wav"
    if not path.exists():
        subprocess.run(
            ["say", "-v", "Samantha", "-o", str(path), "--data-format=LEI16@16000", text],
            check=True,
        )
    with wave.open(str(path)) as w:
        pcm = w.readframes(w.getnframes())
    samples = memoryview(pcm).cast("h")
    last = max((i for i in range(len(samples)) if abs(samples[i]) > 500), default=len(samples))
    return pcm, last / RATE


class RecordedCaller(FrameProcessor):
    """Streams caller audio like a phone line: continuous 20 ms frames, silence between lines."""

    def __init__(self, folder: Path, barge_in: bool, on_finished: Any) -> None:
        super().__init__()
        self._lines = [record_line(t, folder) for t in CALLER_LINES]
        self._barge_in = barge_in
        self._on_finished = on_finished
        self._bot_speaking = False
        self._bot_stopped_at = time.monotonic()
        self._reply_started_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self.create_task(self._run())
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            if self._reply_started_at is None:
                self._reply_started_at = time.monotonic()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped_at = time.monotonic()
        await self.push_frame(frame, direction)

    async def _send(self, pcm: bytes, speech_secs: float | None = None) -> float | None:
        """Push the audio in real time; returns when the chunk holding the last speech went out."""
        step = int(CHUNK_SECONDS * RATE) * 2
        last_speech_byte = int((speech_secs or 0) * RATE) * 2
        speech_end = None
        for i in range(0, len(pcm), step):
            chunk = pcm[i : i + step].ljust(step, b"\0")
            await self.push_frame(InputAudioRawFrame(audio=chunk, sample_rate=RATE, num_channels=1))
            if speech_secs is not None and speech_end is None and i + step >= last_speech_byte:
                speech_end = time.monotonic()
            await asyncio.sleep(CHUNK_SECONDS)
        return speech_end

    async def _silence_until(self, done: Any, what: str, timeout: float = 25) -> None:
        silence = bytes(int(CHUNK_SECONDS * RATE) * 2)
        end = time.monotonic() + timeout
        while not done():
            if time.monotonic() > end:
                TRUTH.anomalies.append(f"timed out waiting for {what}")
                return
            await self._send(silence)

    def _quiet(self) -> bool:
        return not self._bot_speaking and time.monotonic() - self._bot_stopped_at > 1.0

    async def _run(self) -> None:
        await self._silence_until(lambda: self._reply_started_at is not None, "the greeting")
        await self._silence_until(self._quiet, "the greeting to end")
        for i, (pcm, speech_secs) in enumerate(self._lines):
            self._reply_started_at = None
            if self._bot_speaking:
                TRUTH.barge_ins += 1
            speech_end = await self._send(pcm, speech_secs)
            await self._silence_until(lambda: self._reply_started_at is not None, f"reply {i + 1}")
            if self._reply_started_at is not None:
                TRUTH.perceived_latency_ms.append(
                    round((self._reply_started_at - speech_end) * 1000)
                )
            if self._barge_in and i == 0:
                # Talk over the first reply shortly after it starts
                await self._silence_until(
                    lambda: time.monotonic() - self._reply_started_at > 0.8, "barge-in point"
                )
                continue
            await self._silence_until(self._quiet, f"reply {i + 1} to end")
        await self._on_finished()


async def run_call(keys: dict[str, str], barge_in: bool, dump: Path | None) -> list[Check]:
    catcher = Catcher()
    await catcher.start()
    task = None

    async def end_call() -> None:
        await task.queue_frame(EndFrame())

    async def lookup_order(params: FunctionCallParams) -> None:
        loguru_logger.info(LOG_LINE)
        for call in TRUTH.tool_calls:
            if call["id"] == params.tool_call_id:
                call["result"] = ORDER_RESULT
        await params.result_callback(ORDER_RESULT)

    audio_folder = Path(tempfile.gettempdir()) / "sb-real-caller"
    audio_folder.mkdir(exist_ok=True)
    caller = RecordedCaller(audio_folder, barge_in, end_call)
    transport = LocalMockTransport(caller)
    stt = TappedDeepgramSTT(api_key=keys["DEEPGRAM_API_KEY"], sample_rate=RATE)
    llm = TappedOpenAILLM(
        api_key=keys["OPENAI_API_KEY"], settings=OpenAILLMService.Settings(model=LLM_MODEL)
    )
    llm.register_function("lookup_order", lookup_order)
    tts = TappedCartesiaTTS(
        api_key=keys["CARTESIA_API_KEY"], settings=CartesiaTTSService.Settings(voice=VOICE_ID)
    )
    context = LLMContext(
        messages=[{"role": "system", "content": SYSTEM_PROMPT}],
        tools=ToolsSchema(
            standard_tools=[
                FunctionSchema(
                    name="lookup_order",
                    description="Look up an order's status by its order number.",
                    properties={"order_id": {"type": "string"}},
                    required=["order_id"],
                )
            ]
        ),
    )
    aggregators = LLMContextAggregatorPair(
        context,
        user_params=LLMUserAggregatorParams(
            vad_analyzer=SileroVADAnalyzer(),
            user_turn_strategies=UserTurnStrategies(stop=[SpeechTimeoutUserTurnStopStrategy()]),
        ),
    )
    pipeline = Pipeline(
        [
            transport.input(),
            stt,
            aggregators.user(),
            llm,
            tts,
            transport.output(),
            aggregators.assistant(),
        ]
    )
    task = PipelineTask(pipeline, params=PipelineParams())
    attach(task, api_key=API_KEY, agent_id=AGENT_ID, base_url=catcher.base_url, call_id="real-run")
    await task.queue_frames([TTSSpeakFrame(GREETING)])

    try:
        await asyncio.wait_for(PipelineRunner(handle_sigint=False).run(task), timeout=150)
    except TimeoutError:
        TRUTH.anomalies.append("call did not finish within 150s")
    await catcher.stop()

    if dump:
        dump.mkdir(parents=True, exist_ok=True)
        name = "barge_in" if barge_in else "normal"
        out = {"report": catcher.caught.events, "truth": TRUTH.__dict__}
        (dump / f"real_{name}.json").write_text(json.dumps(out, indent=2, default=str))
    return checks(catcher, barge_in, stt, tts)


# ── Checks ───────────────────────────────────────────────────────────────────


def words(text: str) -> list[str]:
    return re.findall(r"[a-z0-9']+", text.lower())


def repeats(texts: list[str]) -> list[str]:
    return [b for a, b in zip(texts, texts[1:], strict=False) if b and a.endswith(b)]


def checks(catcher: Catcher, barge_in: bool, stt: Any, tts: Any) -> list[Check]:
    result: list[Check] = []

    def check(area: str, name: str, ok: Any, detail: Any = "") -> None:
        result.append(Check(area, name, bool(ok), str(detail)))

    caught = catcher.caught
    check("run", "call ran without anomalies", not TRUTH.anomalies, TRUTH.anomalies)
    check("run", "caller barged in as scripted", TRUTH.barge_ins == int(barge_in), TRUTH.barge_ins)
    heard = words(" ".join(TRUTH.transcripts))
    script = words(" ".join(CALLER_LINES))
    overlap = len(set(heard) & set(script)) / max(1, len(set(script)))
    check(
        "run",
        "Deepgram heard the script (>= 70% of words)",
        overlap >= 0.7,
        f"{overlap:.0%}: {TRUTH.transcripts}",
    )
    check("run", "LLM called lookup_order once", len(TRUTH.tool_calls) == 1, TRUTH.tool_calls)

    kinds = [e.get("event") for e in caught.events]
    check("report", "call.started then call.ended", kinds == ["call.started", "call.ended"], kinds)
    ended = next((e for e in caught.events if e.get("event") == "call.ended"), None)
    if ended is None:
        return result
    diag = ended.get("diagnostics") or {}
    usage = diag.get("usage") or {}
    turns = ended.get("transcript") or []
    users = [t["text"] for t in turns if t["role"] == "user"]
    agents = [t["text"] for t in turns if t["role"] == "agent"]
    spoken = words(" ".join(TRUTH.tts_texts))
    check(
        "report", "a production call", ended.get("surface") == "observability", ended.get("surface")
    )

    check(
        "report",
        "user turns = Deepgram's final transcripts",
        words(" ".join(users)) == heard,
        users,
    )
    if barge_in:
        got = words(" ".join(agents))
        check(
            "report",
            "agent text is words sent to TTS, cut-off part dropped",
            set(got) <= set(spoken) and len(got) < len(spoken),
            f"{len(got)} of {len(spoken)} words",
        )
    else:
        check(
            "report",
            "agent text = what Cartesia was asked to say",
            words(" ".join(agents)) == spoken,
            f"{agents} vs {TRUTH.tts_texts}",
        )
    flagged = [t for t in turns if t["role"] == "agent" and t.get("interrupted")]
    check("report", "agent turns flagged interrupted = barge-ins", len(flagged) == TRUTH.barge_ins)
    check("report", "no duplicated agent text", not repeats(agents), repeats(agents))
    numbers = set(re.findall(r"\d{4,}", " ".join(TRUTH.tts_texts)))
    check(
        "report",
        "numbers spoken whole stay whole",
        all(n in " ".join(agents) for n in numbers),
        numbers,
    )

    latencies = diag.get("turn_latencies") or []
    replies = [e.get("turn_ms") for e in latencies]
    if len(replies) == len(TRUTH.perceived_latency_ms) and None not in replies:
        gaps = [round(r - p) for r, p in zip(replies, TRUTH.perceived_latency_ms, strict=True)]
        check(
            "report",
            f"reply latency within {LATENCY_TOLERANCE_MS}ms of what the caller waited",
            all(abs(g) <= LATENCY_TOLERANCE_MS for g in gaps),
            f"reported {replies} vs waited {TRUTH.perceived_latency_ms}",
        )
    else:
        check(
            "report",
            "a reply latency per caller line",
            False,
            f"{replies} vs {TRUTH.perceived_latency_ms}",
        )
    check(
        "report",
        "STT, LLM and TTS time per reply",
        latencies and all({"transcriber_ms", "model_ms", "voice_ms"} <= set(e) for e in latencies),
        latencies,
    )

    check(
        "report",
        "LLM input tokens = OpenAI usage",
        usage.get("llm_input_tokens") == TRUTH.prompt_tokens,
        f"{usage.get('llm_input_tokens')} vs {TRUTH.prompt_tokens}",
    )
    check(
        "report",
        "LLM output tokens = OpenAI usage",
        usage.get("llm_output_tokens") == TRUTH.completion_tokens,
        f"{usage.get('llm_output_tokens')} vs {TRUTH.completion_tokens}",
    )
    tts_chars = sum(map(len, TRUTH.tts_texts))
    check(
        "report",
        "TTS characters = text sent to Cartesia",
        usage.get("tts_characters") == tts_chars,
        f"{usage.get('tts_characters')} vs {tts_chars}",
    )
    if TRUTH.stt_reported:
        stt_seconds = (usage.get("stt_audio_ms") or 0) / 1000
        check(
            "report",
            "STT seconds = Deepgram usage",
            abs(stt_seconds - TRUTH.stt_seconds) < 0.05,
            f"{stt_seconds} vs {TRUTH.stt_seconds:.2f}",
        )

    sent = ended.get("tool_calls") or []
    check(
        "report",
        "one record per tool call",
        len(sent) == len(TRUTH.tool_calls),
        f"{len(sent)} vs {len(TRUTH.tool_calls)}",
    )
    if sent and TRUTH.tool_calls:
        want = TRUTH.tool_calls[0]
        check(
            "report",
            "tool arguments and result",
            sent[0]["arguments"] == want["arguments"]
            and sent[0].get("result") == want.get("result"),
            sent[0],
        )

    config = diag.get("configuration") or {}
    expected = {
        "stt": {"provider": "deepgram", "model": stt._settings.model},
        "llm": {"provider": "openai", "model": LLM_MODEL},
        "tts": {"provider": "cartesia", "model": tts._settings.model, "voice": VOICE_ID},
    }
    got = {
        role: {k: (config.get(role) or {}).get(k) for k in want} for role, want in expected.items()
    }
    check("report", "providers, models, voice", got == expected, got)
    environment = (ended.get("context") or {}).get("environment") or {}
    check("report", "transport named", environment.get("transport") == "localmock", environment)
    check(
        "report",
        "loguru line from the tool captured",
        LOG_LINE in [e["message"] for e in diag.get("logs") or []],
        len(diag.get("logs") or []),
    )
    observability = ended.get("observability") or {}
    check(
        "report",
        "stereo recording uploaded",
        len(caught.uploads) == 1 and observability.get("stereo_recording_url"),
        observability,
    )
    started, finished = (
        datetime.fromisoformat(ended[k].replace("Z", "+00:00")) for k in ("started_at", "ended_at")
    )
    call_seconds = (finished - started).total_seconds()
    for wav_bytes in list(caught.uploads.values())[:1]:
        with wave.open(io.BytesIO(wav_bytes)) as w:
            seconds = w.getnframes() / w.getframerate()
        check(
            "report",
            "recording length matches the call",
            abs(seconds - call_seconds) <= 1.0,
            f"{seconds:.1f}s vs {call_seconds:.1f}s",
        )
    return result


def load_keys(env_file: Path | None) -> dict[str, str]:
    names = ("DEEPGRAM_API_KEY", "OPENAI_API_KEY", "CARTESIA_API_KEY")
    keys = {n: os.environ.get(n, "") for n in names}
    if env_file:
        for line in env_file.read_text().splitlines():
            m = re.match(r"^\s*(\w+)\s*=\s*(.*)$", line)
            if m and m.group(1) in names and not keys[m.group(1)]:
                keys[m.group(1)] = m.group(2).strip().strip("'\"")
    missing = [n for n, v in keys.items() if not v]
    if missing:
        sys.exit(f"missing keys: {', '.join(missing)}")
    return keys


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--env-file", type=Path, help="read the three API keys from this file")
    parser.add_argument("--barge-in", action="store_true", help="caller talks over the first reply")
    parser.add_argument("--dump", type=Path, help="write payloads and service records here")
    parser.add_argument("-v", "--verbose", action="store_true", help="show Pipecat logs")
    args = parser.parse_args()
    if not args.verbose:
        loguru_logger.remove()
        loguru_logger.add(sys.stderr, level="ERROR")
    logging.basicConfig(level=logging.WARNING)
    keys = load_keys(args.env_file)
    print(f"=== real services ({'barge-in' if args.barge_in else 'normal'} call)")
    result = asyncio.run(run_call(keys, args.barge_in, args.dump))
    print_report(result)
    sys.exit(0 if all(c.ok for c in result) else 1)


if __name__ == "__main__":
    main()
