"""
superbryn-pipecat-observer

Drop-in Pipecat observer that reports a normalized call record to SuperBryn
at end of session. See README.md for usage.
"""

from .audio_recorder import SuperbrynAudioRecorder
from .observer import SuperbrynObserver, __version__

__all__ = ["SuperbrynAudioRecorder", "SuperbrynObserver", "__version__"]
