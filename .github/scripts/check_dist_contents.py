"""Fail the build if the wheel or sdist carries anything but the observer itself.

The wheel may hold only the `superbryn_pipecat_observer` package and its
dist-info; neither file may hold `examples/` (the local harness lives there).
"""

import sys
import tarfile
import zipfile
from pathlib import Path

PACKAGE = "superbryn_pipecat_observer"


def main(dist: Path) -> int:
    wheels, sdists = sorted(dist.glob("*.whl")), sorted(dist.glob("*.tar.gz"))
    if not wheels or not sdists:
        print(f"expected a wheel and an sdist in {dist}")
        return 1
    bad = []
    for wheel in wheels:
        for name in zipfile.ZipFile(wheel).namelist():
            if not name.startswith((f"{PACKAGE}/", f"{PACKAGE}-")):
                bad.append(f"{wheel.name}: {name}")
    for sdist in sdists:
        with tarfile.open(sdist) as archive:
            for name in archive.getnames():
                inner = name.split("/", 1)[1] if "/" in name else ""
                if inner == "examples" or inner.startswith("examples/"):
                    bad.append(f"{sdist.name}: {name}")
    for entry in bad:
        print(f"not allowed in the package: {entry}")
    if not bad:
        print(f"ok: {len(wheels)} wheel(s), {len(sdists)} sdist(s) carry only the observer")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1] if len(sys.argv) > 1 else "dist")))
