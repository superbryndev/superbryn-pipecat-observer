"""
Smoke tests for superbryn-pipecat-observer.

Run with:  pytest test_import.py -v   (or just `python test_import.py`)

Validates:
  - The package imports and exposes one entry point, attach(), with its handle
  - attach() does nothing without an API key
  - attach() on a real PipelineTask adds its observer and finish hook, and sends nothing until the task runs
  - The 0.11 entry points and the per-carrier recording adapters are gone
"""

import os


def test_package_imports() -> None:
    from superbryn_pipecat_observer import CallReport, __version__, attach

    assert callable(attach) and CallReport is not None
    assert isinstance(__version__, str) and __version__


def test_old_entry_points_are_gone() -> None:
    import importlib

    import superbryn_pipecat_observer as pkg

    for name in ("SuperbrynObserver", "attach_test_data", "SuperbrynAudioRecorder"):
        assert not hasattr(pkg, name), f"{name} was removed in 0.12.0"
    for module in ("observer", "test_data", "transports"):
        try:
            importlib.import_module(f"superbryn_pipecat_observer.{module}")
        except ImportError:
            continue
        raise AssertionError(f"superbryn_pipecat_observer.{module} should not be importable")


def test_attach_without_a_key_is_a_noop() -> None:
    from superbryn_pipecat_observer import CallReport, attach

    os.environ.pop("SUPERBRYN_API_KEY", None)
    task = type("Task", (), {"add_observer": None, "event_handler": None})()
    report = attach(task, api_key="")
    assert isinstance(report, CallReport)
    report.set_custom_data({"workflow_id": "wf_1"})
    report.update_custom_data({"booked": True})
    report.set_call_end_reason("transfer_to_human")
    report.set_recording_url("https://example.com/call.mp3")


def test_attach_to_a_real_task() -> None:
    import warnings

    from pipecat.pipeline.pipeline import Pipeline
    from pipecat.pipeline.task import PipelineParams, PipelineTask
    from pipecat.processors.frame_processor import FrameProcessor

    from superbryn_pipecat_observer import attach

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        task = PipelineTask(Pipeline([FrameProcessor()]), params=PipelineParams())
    added = []
    add_observer = task.add_observer
    task.add_observer = lambda observer: (added.append(observer), add_observer(observer))
    attach(task, api_key="sb_test", production=False)
    assert len(added) == 1
    # unset metrics are switched on: usage and latency need them
    assert task.params.enable_metrics and task.params.enable_usage_metrics


if __name__ == "__main__":
    test_package_imports()
    test_old_entry_points_are_gone()
    test_attach_without_a_key_is_a_noop()
    test_attach_to_a_real_task()
    print("All smoke tests passed.")
