"""Package surface: exports, version consistency, import safety."""

from __future__ import annotations

import re
import socket
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def test_all_exports_are_importable():
    import superbryn_pipecat_observer as pkg

    for name in pkg.__all__:
        assert hasattr(pkg, name), f"__all__ lists {name} but it is not importable"


def test_source_scanning_is_gone():
    import inspect

    import superbryn_pipecat_observer as pkg

    assert not hasattr(pkg, "scan_source_config")
    assert "scan_source_config" not in pkg.__all__
    assert not (REPO_ROOT / "superbryn_pipecat_observer" / "codescan.py").exists()
    assert "scan_root" not in inspect.signature(pkg.build_manifest_from_pipeline).parameters


def test_version_is_consistent_everywhere():
    import superbryn_pipecat_observer as pkg

    pyproject = (REPO_ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "(.+?)"', pyproject, re.MULTILINE).group(1)
    assert pkg.__version__ == declared

    bumpversion = (REPO_ROOT / ".bumpversion.cfg").read_text()
    tracked = re.search(r"current_version = (.+)", bumpversion).group(1).strip()
    assert tracked == declared

    changelog = (REPO_ROOT / "CHANGELOG.md").read_text()
    assert f"## [{declared}]" in changelog, (
        "CHANGELOG.md must have a section for the declared version"
    )


def test_manifest_build_without_env_or_network(monkeypatch):
    """Building a manifest must work with no env vars and no sockets.

    The sync feature being unused must have zero side effects: no credentials
    required, no network touched.
    """
    monkeypatch.delenv("SUPERBRYN_API_KEY", raising=False)
    monkeypatch.delenv("SUPERBRYN_API_BASE_URL", raising=False)

    def no_network(*args, **kwargs):
        raise AssertionError("network access attempted during manifest build")

    monkeypatch.setattr(socket.socket, "connect", no_network)

    from superbryn_pipecat_observer import build_manifest_from_pipeline

    manifest = build_manifest_from_pipeline(object())
    assert manifest == {"source": "pipecat"}
