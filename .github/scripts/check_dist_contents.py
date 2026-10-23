"""Fail the build if the wheel or sdist carries anything but the observer itself.

The wheel may hold only the `superbryn_pipecat_observer` package and its
`.dist-info`; neither file may hold `examples/` (the local harness lives there)
at any depth, or a path that is absolute or climbs out with `..`.
"""

import posixpath
import re
import sys
import tarfile
import zipfile
from pathlib import Path

PACKAGE = "superbryn_pipecat_observer"
WHEEL_ALLOWED = re.compile(rf"^{re.escape(PACKAGE)}(/|-[^/]+\.dist-info/)")


def parts(name: str) -> list[str]:
    return [p for p in posixpath.normpath(name).split("/") if p not in ("", ".")]


def unsafe(name: str) -> bool:
    return name.startswith("/") or ".." in parts(name)


def main(dist: Path) -> int:
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if not wheels or not sdists:
        print(f"expected a wheel and an sdist in {dist}")
        return 1
    bad = []
    for wheel in wheels:
        with zipfile.ZipFile(wheel) as archive:
            for name in archive.namelist():
                if unsafe(name) or not WHEEL_ALLOWED.match(name):
                    bad.append(f"{wheel.name}: {name}")
    for sdist in sdists:
        with tarfile.open(sdist) as archive:
            for name in archive.getnames():
                if unsafe(name) or "examples" in parts(name):
                    bad.append(f"{sdist.name}: {name}")
    for entry in bad:
        print(f"not allowed in the package: {entry}")
    if not bad:
        print(f"ok: {len(wheels)} wheel(s), {len(sdists)} sdist(s) carry only the observer")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "dist")))
