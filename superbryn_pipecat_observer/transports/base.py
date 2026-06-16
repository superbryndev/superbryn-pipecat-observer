"""
Recording adapter protocol.

Each transport adapter is responsible for two lifecycle hooks:

  - `start(observer)` — called during `on_pipeline_started`. Use this to
    kick off recording (e.g. Daily `start_recording`, LiveKit egress) and
    register any event handlers needed to capture the recording id.

  - `finalize(observer)` — called during `on_pipeline_finished`, before the
    webhook is sent. Use this to stop the recording cleanly and stamp
    `observer.recording_url` (and optionally `observer.stereo_recording_url`)
    with the final URL.

Both hooks must be fail-open: if recording wiring fails for any reason, the
observer still sends the call record without a URL. Never re-raise.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from ..observer import SuperbrynObserver


@runtime_checkable
class RecordingAdapter(Protocol):
    """Per-transport recording lifecycle hook."""

    transport_name: str
    """Canonical transport tag stamped onto `call.metadata.transport`."""

    transport: Any
    """The underlying Pipecat transport instance."""

    async def start(self, observer: SuperbrynObserver) -> None:
        """Start recording (or attach to an in-progress one). Fail-open."""

    async def finalize(self, observer: SuperbrynObserver) -> None:
        """
        Stop recording, fetch the final URL, and assign it to
        `observer.recording_url` / `observer.stereo_recording_url`. Fail-open.
        """
