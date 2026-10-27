"""The caller-audio energy detector: speech over line noise, later hops of a frame ignored."""

from __future__ import annotations

import numpy as np
from pipecat.frames.frames import InputAudioRawFrame

from superbryn_pipecat_observer._caller_speech import CallerSpeech

RATE = 8000  # a phone line


def frames(noise_rms: float, speech_s: float, seed: int = 0) -> list[InputAudioRawFrame]:
    """1 s of line noise, a tone for ``speech_s`` over it, then 1 s of noise; 20 ms frames."""
    rng = np.random.default_rng(seed)
    n = RATE // 50
    t = np.arange(round(speech_s * RATE)) / RATE
    speech = 3000 * np.sin(2 * np.pi * 440 * t)
    line = np.concatenate([np.zeros(RATE), speech, np.zeros(RATE)])
    pcm = np.clip(line + rng.normal(0, noise_rms, line.size), -32768, 32767).astype("<i2")
    return [
        InputAudioRawFrame(audio=pcm[i : i + n].tobytes(), sample_rate=RATE, num_channels=1)
        for i in range(0, pcm.size, n)
    ]


def run(detector: CallerSpeech, line: list[InputAudioRawFrame], hops: int = 1) -> list[tuple]:
    events = []
    for i, frame in enumerate(line):
        for _ in range(hops):
            event = detector.feed(frame, (i + 1) * 0.02)
            if event:
                events.append(
                    (event, round(detector.start if event == "start" else detector.end, 2))
                )
    return events


def test_speech_over_line_noise_starts_and_ends_on_time():
    for noise in (0, 30, 300):  # silent, quiet and a -40 dBFS line
        assert run(CallerSpeech(), frames(noise, 1.5)) == [("start", 1.0), ("stop", 2.5)]


def test_later_hops_of_a_frame_are_not_heard_again():
    assert run(CallerSpeech(), frames(30, 1.0), hops=3) == [("start", 1.0), ("stop", 2.0)]


def test_a_quiet_stretch_ends_when_the_agent_starts():
    detector = CallerSpeech()
    run(detector, frames(30, 1.0)[:105])  # speech ends at 2.0, then 0.1 s of quiet
    assert detector.pause() == 2.0
    assert detector.pause() is None
