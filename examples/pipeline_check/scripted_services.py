"""Scripted stand-ins for a caller, STT, LLM, TTS and transport.

They subclass Pipecat's real base classes, so the real pipeline machinery
(aggregators, metrics, function calls, output transport) runs unchanged.
Module paths mimic Deepgram / OpenAI / Cartesia so the report's
provider detection is exercised too. Each stand-in records ground truth
on a shared `GroundTruth` for the checklist to compare against.
"""

from __future__ import annotations

import asyncio
import io
import math
import re
import time
import wave
from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import dataclass, field
from typing import Any

from pipecat.frames.frames import (
    BotStartedSpeakingFrame,
    BotStoppedSpeakingFrame,
    Frame,
    FunctionCallFromLLM,
    InputAudioRawFrame,
    LLMContextFrame,
    LLMFullResponseEndFrame,
    LLMFullResponseStartFrame,
    LLMTextFrame,
    OutputAudioRawFrame,
    StartFrame,
    TranscriptionFrame,
    UserStartedSpeakingFrame,
    UserStoppedSpeakingFrame,
    VADUserStartedSpeakingFrame,
    VADUserStoppedSpeakingFrame,
)
from pipecat.metrics.metrics import LLMTokenUsage
from pipecat.processors.frame_processor import FrameDirection, FrameProcessor
from pipecat.services.llm_service import LLMService
from pipecat.services.settings import LLMSettings, STTSettings, TTSSettings
from pipecat.services.stt_service import SegmentedSTTService
from pipecat.services.tts_service import TTSService
from pipecat.transports.base_output import BaseOutputTransport
from pipecat.transports.base_transport import BaseTransport, TransportParams
from pipecat.utils.time import time_now_iso8601

USER_SAMPLE_RATE = 16000
CHUNK_SECONDS = 0.02
VAD_START_SECS = 0.2
VAD_STOP_SECS = 0.2


@dataclass
class GroundTruth:
    """What actually happened in the call, recorded by the stand-ins."""

    user_lines: list[str] = field(default_factory=list)
    user_audio_seconds: float = 0.0
    # Speech plus the silence VAD waits for: what a turn-based STT duration covers
    user_turn_seconds: float = 0.0
    # Audio the STT was handed (segments include the pre-speech buffer): what newer Pipecat reports as STT usage
    stt_audio_seconds: float = 0.0
    # Caller-measured gap: end of caller speech → bot audio start, one per user turn
    perceived_latency_ms: list[float] = field(default_factory=list)
    llm_prompt_tokens: int = 0
    llm_completion_tokens: int = 0
    agent_lines: list[str] = field(default_factory=list)
    tts_texts: list[str] = field(default_factory=list)
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    interruptions: int = 0
    anomalies: list[str] = field(default_factory=list)


def _tone(seconds: float, sample_rate: int, hz: float) -> bytes:
    """16-bit mono sine tone, quiet enough to be obviously synthetic."""
    n = int(seconds * sample_rate)
    out = bytearray()
    for i in range(n):
        v = int(3000 * math.sin(2 * math.pi * hz * i / sample_rate))
        out += v.to_bytes(2, "little", signed=True)
    return bytes(out)


def stream_tokens(text: str) -> list[str]:
    """Split text the way an OpenAI stream does: words with a leading space,
    digit runs in chunks of three, punctuation on its own."""
    tokens: list[str] = []
    for m in re.finditer(r"\s*(\d+|[A-Za-z']+|[^\sA-Za-z\d])", text):
        lead = m.group(0)[: m.start(1) - m.start(0)]
        word = m.group(1)
        if word.isdigit():
            parts = [word[i : i + 3] for i in range(0, len(word), 3)]
            tokens.append(lead + parts[0])
            tokens.extend(parts[1:])
        else:
            tokens.append(lead + word)
    return tokens


class ScriptedCaller(FrameProcessor):
    """Plays the caller like an input transport with VAD: speech, a silence tail,
    then the VAD stop. Lines in `barge_in` start while the previous reply plays."""

    def __init__(
        self,
        lines: list[str],
        truth: GroundTruth,
        on_finished: Callable[[], Awaitable[None]],
        barge_in: set[int] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self._lines = lines
        self._truth = truth
        self._on_finished = on_finished
        self._barge_in = barge_in or set()
        self._bot_started = asyncio.Event()
        self._bot_stopped = asyncio.Event()
        self._bot_speaking = False
        self._reply_started_at: float | None = None

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if isinstance(frame, StartFrame):
            await self.push_frame(frame, direction)
            self.create_task(self._run())
            return
        if isinstance(frame, BotStartedSpeakingFrame):
            self._bot_speaking = True
            self._bot_stopped.clear()
            self._bot_started.set()
            if self._reply_started_at is None:
                self._reply_started_at = time.monotonic()
        elif isinstance(frame, BotStoppedSpeakingFrame):
            self._bot_speaking = False
            self._bot_stopped.set()
        await self.push_frame(frame, direction)

    async def _wait(self, event: asyncio.Event, what: str) -> None:
        try:
            await asyncio.wait_for(event.wait(), timeout=15)
        except TimeoutError:
            self._truth.anomalies.append(f"timed out waiting for {what}")

    async def _wait_until_quiet(self, what: str) -> None:
        """Bot finished and stayed silent for a beat (a reply can pause between sentences)."""
        while True:
            await self._wait(self._bot_stopped, what)
            await asyncio.sleep(0.6)
            if not self._bot_speaking:
                return

    async def _speak(self, line: str) -> float:
        """Speak one line; returns when the caller's speech ended (before the silence tail)."""
        seconds = min(3.0, max(0.8, 0.25 * len(line.split())))
        await self.push_frame(
            VADUserStartedSpeakingFrame(start_secs=VAD_START_SECS, timestamp=time.time())
        )
        await self.push_frame(UserStartedSpeakingFrame())
        if self._bot_speaking:
            # What a VAD turn-start strategy does when the caller talks over the bot
            self._truth.interruptions += 1
            await self.broadcast_interruption()
        step = int(CHUNK_SECONDS * USER_SAMPLE_RATE) * 2
        speech = _tone(seconds, USER_SAMPLE_RATE, 440)
        silence = bytes(int(VAD_STOP_SECS * USER_SAMPLE_RATE) * 2)
        speech_end = 0.0
        for i in range(0, len(speech) + len(silence), step):
            chunk = (speech + silence)[i : i + step]
            await self.push_frame(
                InputAudioRawFrame(audio=chunk, sample_rate=USER_SAMPLE_RATE, num_channels=1)
            )
            await asyncio.sleep(CHUNK_SECONDS)
            if i + step >= len(speech) and not speech_end:
                speech_end = time.monotonic()
        self._truth.user_lines.append(line)
        self._truth.user_audio_seconds += seconds
        self._truth.user_turn_seconds += seconds + VAD_STOP_SECS
        await self.push_frame(
            VADUserStoppedSpeakingFrame(stop_secs=VAD_STOP_SECS, timestamp=time.time())
        )
        await self.push_frame(UserStoppedSpeakingFrame())
        return speech_end

    async def _run(self) -> None:
        await self._wait_until_quiet("the greeting")
        for i, line in enumerate(self._lines):
            if i not in self._barge_in:
                await asyncio.sleep(0.4)
            self._bot_started.clear()
            self._reply_started_at = None
            speech_end = await self._speak(line)
            if i + 1 in self._barge_in:
                # Next line talks over this reply shortly after it starts
                await self._wait(self._bot_started, f"reply to {line!r}")
                await asyncio.sleep(0.5)
            else:
                await self._wait(self._bot_started, f"reply to {line!r}")
                await self._wait_until_quiet(f"reply to {line!r}")
            if self._reply_started_at is not None:
                self._truth.perceived_latency_ms.append(
                    round((self._reply_started_at - speech_end) * 1000)
                )
        await asyncio.sleep(0.5)
        await self._on_finished()


class ScriptedSTTService(SegmentedSTTService):
    """Returns the next scripted line for each VAD segment it receives."""

    def __init__(
        self, lines: list[str], truth: GroundTruth, delay: float = 0.15, **kwargs: Any
    ) -> None:
        super().__init__(settings=STTSettings(model="nova-3", language=None), **kwargs)
        self._lines = list(lines)
        self._truth = truth
        self._delay = delay

    def can_generate_metrics(self) -> bool:
        return True

    async def run_stt(self, audio: bytes) -> AsyncGenerator[Frame | None, None]:
        with wave.open(io.BytesIO(audio)) as segment:
            self._truth.stt_audio_seconds += segment.getnframes() / segment.getframerate()
        await asyncio.sleep(self._delay)
        text = self._lines.pop(0) if self._lines else ""
        yield TranscriptionFrame(text=text, user_id="caller", timestamp=time_now_iso8601())


ScriptedSTTService.__module__ = "pipecat.services.deepgram.stt"


@dataclass
class Say:
    text: str
    prompt_tokens: int
    completion_tokens: int


@dataclass
class CallTool:
    name: str
    arguments: dict[str, Any]
    prompt_tokens: int
    completion_tokens: int


class ScriptedLLMService(LLMService):
    """Answers each context with the next scripted step: speak or call a tool."""

    def __init__(
        self,
        steps: list[Say | CallTool],
        truth: GroundTruth,
        delay: float = 0.2,
        token_delay: float = 0.01,
        **kwargs: Any,
    ) -> None:
        unset = dict.fromkeys(
            (
                "system_instruction",
                "temperature",
                "max_tokens",
                "top_p",
                "top_k",
                "frequency_penalty",
                "presence_penalty",
                "seed",
                "filter_incomplete_user_turns",
                "user_turn_completion_config",
            )
        )
        super().__init__(settings=LLMSettings(model="gpt-4o-mini", **unset), **kwargs)
        self._steps = list(steps)
        self._truth = truth
        self._delay = delay
        self._token_delay = token_delay
        self._calls = 0

    def can_generate_metrics(self) -> bool:
        return True

    async def process_frame(self, frame: Frame, direction: FrameDirection) -> None:
        await super().process_frame(frame, direction)
        if not isinstance(frame, LLMContextFrame):
            await self.push_frame(frame, direction)
            return
        await self.push_frame(LLMFullResponseStartFrame())
        await self.start_processing_metrics()
        try:
            await self._respond(frame)
        finally:
            await self.stop_processing_metrics()
            await self.push_frame(LLMFullResponseEndFrame())

    async def _respond(self, frame: LLMContextFrame) -> None:
        await self.start_ttfb_metrics()
        await asyncio.sleep(self._delay)
        await self.stop_ttfb_metrics()
        if not self._steps:
            self._truth.anomalies.append("LLM ran more times than the script has steps")
            return
        step = self._steps.pop(0)
        if isinstance(step, CallTool):
            self._calls += 1
            call_id = f"call_{self._calls}"
            self._truth.tool_calls.append(
                {"tool_call_id": call_id, "function_name": step.name, "arguments": step.arguments}
            )
            await self._usage(step)
            await self.run_function_calls(
                [
                    FunctionCallFromLLM(
                        function_name=step.name,
                        tool_call_id=call_id,
                        arguments=step.arguments,
                        context=frame.context,
                    )
                ]
            )
            return
        for token in stream_tokens(step.text):
            await self.push_frame(LLMTextFrame(token))
            await asyncio.sleep(self._token_delay)
        self._truth.agent_lines.append(step.text)
        await self._usage(step)

    async def _usage(self, step: Say | CallTool) -> None:
        self._truth.llm_prompt_tokens += step.prompt_tokens
        self._truth.llm_completion_tokens += step.completion_tokens
        await self.start_llm_usage_metrics(
            LLMTokenUsage(
                prompt_tokens=step.prompt_tokens,
                completion_tokens=step.completion_tokens,
                total_tokens=step.prompt_tokens + step.completion_tokens,
            )
        )


ScriptedLLMService.__module__ = "pipecat.services.openai.llm"


class ScriptedTTSService(TTSService):
    """Turns text into a tone whose length tracks the text length. With
    `word_timestamps`, text is sent per word in playback order (Cartesia,
    ElevenLabs); otherwise one text frame per sentence (HTTP TTS)."""

    VOICE_ID = "79a125e8-cd45-4c13-8a67-188112f4dd22"

    def __init__(
        self, truth: GroundTruth, delay: float = 0.25, word_timestamps: bool = False, **kwargs: Any
    ) -> None:
        super().__init__(
            push_start_frame=True,
            push_stop_frames=True,
            push_text_frames=not word_timestamps,
            settings=TTSSettings(model="sonic-2", voice=self.VOICE_ID, language=None),
            **kwargs,
        )
        self._truth = truth
        self._delay = delay
        self._word_timestamps = word_timestamps
        # Word times run from the start of the context, across its sentences (as Cartesia's do)
        self._context_offsets: dict[str, float] = {}

    def can_generate_metrics(self) -> bool:
        return True

    async def run_tts(self, text: str, context_id: str) -> AsyncGenerator[Frame | None, None]:
        self._truth.tts_texts.append(text)
        await self.start_tts_usage_metrics(text)
        await asyncio.sleep(self._delay)
        audio = _tone(0.03 * len(text), self.sample_rate, 660)
        if self._word_timestamps:
            words = text.split()
            seconds = len(audio) / (2 * self.sample_rate)
            offset = self._context_offsets.get(context_id, 0.0)
            self._context_offsets[context_id] = offset + seconds
            await self.add_word_timestamps(
                [(w, offset + i * seconds / len(words)) for i, w in enumerate(words)], context_id
            )

        async def chunks() -> AsyncGenerator[bytes, None]:
            step = int(0.04 * self.sample_rate) * 2
            for i in range(0, len(audio), step):
                yield audio[i : i + step]

        async for frame in self._stream_audio_frames_from_iterator(
            chunks(), in_sample_rate=self.sample_rate, context_id=context_id
        ):
            await self.stop_ttfb_metrics()
            yield frame


ScriptedTTSService.__module__ = "pipecat.services.cartesia.tts"


class LocalMockOutputTransport(BaseOutputTransport):
    """Plays bot audio into nothing at real-time speed."""

    async def start(self, frame: StartFrame) -> None:
        await super().start(frame)
        await self.set_transport_ready(frame)

    async def write_audio_frame(self, frame: OutputAudioRawFrame) -> bool:
        await asyncio.sleep(len(frame.audio) / (2 * frame.sample_rate * frame.num_channels))
        return True


class LocalMockTransport(BaseTransport):
    """Transport whose input is the scripted caller and output is a null sink."""

    def __init__(self, caller: ScriptedCaller) -> None:
        super().__init__()
        self._caller = caller
        self._output = LocalMockOutputTransport(TransportParams(audio_out_enabled=True))
        self._output._transport = self  # as network transports' outputs keep theirs
        # Same event as network transports, so attach()'s hang-up hook attaches
        self._register_event_handler("on_client_disconnected")

    def input(self) -> FrameProcessor:
        return self._caller

    def output(self) -> FrameProcessor:
        return self._output
