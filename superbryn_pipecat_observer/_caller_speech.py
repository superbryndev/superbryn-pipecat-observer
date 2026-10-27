"""Caller speech from the caller's raw audio: an energy detector for pipelines with no VAD frames."""

from __future__ import annotations

from collections import deque
from typing import Any

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy comes with pipecat-ai
    np = None

# A frame is speech when its RMS clears both a floor (about -46 dBFS) and 4x the line's noise,
# the quietest frame of the last 2 s (speech always dips between words; noise doesn't)
_MIN_RMS = 160.0
_NOISE_RATIO = 4.0
_NOISE_WINDOW_S = 2.0
_ONSET_S = 0.1  # this much unbroken speech starts a stretch, so a click or a pop doesn't
_HANGOVER_S = 0.25  # this much quiet ends it


class CallerSpeech:
    """Fed every input audio frame; tells when a stretch of the caller's speech starts and when it ended."""

    def __init__(self) -> None:
        self.heard = False  # any speech yet
        self.start: float | None = (
            None  # when the latest stretch began (seconds, the caller's clock)
        )
        self.end: float | None = None  # when it ended: the end of its last loud frame
        self._speaking = False
        self._quiet = True  # the latest frame was quiet
        self._loud_since: float | None = None
        self._last_loud = 0.0
        self._levels: deque[tuple[float, float]] = deque()  # (arrival, RMS) over the noise window
        self._last_id = -1

    def feed(self, frame: Any, at: float) -> str | None:
        """A frame that arrived at ``at``: "start" or "stop" when a stretch begins or (after the hangover) ends."""
        frame_id = getattr(frame, "id", None)
        if isinstance(frame_id, int):
            if frame_id <= self._last_id:
                return None  # a later hop of a frame already heard (frame ids only grow)
            self._last_id = frame_id
        loud, seconds = self._loud(frame, at)
        if seconds <= 0:
            return None
        if not loud:
            self._quiet, self._loud_since = True, None
            if self._speaking and at - self._last_loud >= _HANGOVER_S:
                return self._stop()
            return None
        self._quiet, self._last_loud = False, at
        if self._loud_since is None:
            self._loud_since = at - seconds
        if not self._speaking and at - self._loud_since >= _ONSET_S - 1e-6:
            self._speaking = self.heard = True
            self.start, self.end = self._loud_since, None
            return "start"
        return None

    def pause(self) -> float | None:
        """The agent started talking: a stretch already gone quiet ends now, its hangover cut short."""
        if self._speaking and self._quiet:
            self._stop()
            return self.end
        return None

    def _stop(self) -> str:
        self._speaking = False
        self.end = self._last_loud
        return "stop"

    def _loud(self, frame: Any, at: float) -> tuple[bool, float]:
        """Whether a 16-bit PCM frame is louder than the line's noise, and how long it lasts."""
        audio = getattr(frame, "audio", None)
        rate = getattr(frame, "sample_rate", 0) or 0
        channels = getattr(frame, "num_channels", 1) or 1
        if np is None or not audio or rate <= 0:
            return False, 0.0
        samples = np.frombuffer(audio, dtype="<i2", count=len(audio) // 2).astype(np.float32)
        if not samples.size:
            return False, 0.0
        rms = float(np.sqrt(np.mean(samples * samples)))
        seconds = samples.size / (rate * channels)
        self._levels.append((at, rms))
        while abs(at - self._levels[0][0]) > _NOISE_WINDOW_S:  # abs: a clock reset empties it too
            self._levels.popleft()
        noise = min(level for _, level in self._levels)
        return rms >= max(_MIN_RMS, noise * _NOISE_RATIO), seconds
