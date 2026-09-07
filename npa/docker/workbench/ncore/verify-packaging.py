"""Offline import and CLI-schema check only; does not validate a capture."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys

from ncore.data.v4 import SequenceLoaderV4


def main() -> None:
    import npa

    assert SequenceLoaderV4 and npa
    assert sys.executable == "/opt/venv/bin/python"
    assert importlib.util.find_spec("torch") is None
    import numpy as np
    import pycolmap

    assert pycolmap.SceneManager.INVALID_POINT3D == np.iinfo(np.uint64).max
    root = Path("/opt/ncore/src")
    inventory = json.loads((root / "source-inventory.json").read_text())
    for relative, expected in inventory["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, (
            relative
        )
    assert str(Path(sys.modules[SequenceLoaderV4.__module__].__file__)).startswith(
        "/opt/ncore/src/ncore/"
    )
    command = ["/opt/ncore/bin/colmap-convert"]
    help_text = subprocess.check_output([*command, "--help"], text=True)
    assert "--root-dir" in help_text and "--output-dir" in help_text
    assert "colmap-v4" in help_text
    help_text = subprocess.check_output(
        [*command, "--output-dir", "/tmp/ncore-schema-unused", "colmap-v4", "--help"],
        text=True,
    )
    assert "--include-3d-points" in help_text and "--store-type" in help_text
    subprocess.run(
        ["/opt/venv/bin/npa", "workbench", "nurec", "--help"],
        check=True,
        stdout=subprocess.DEVNULL,
    )
    print(
        json.dumps(
            {
                "status": "packaging-ready",
                "validation": "import-and-cli-schema-only",
                "ncore_revision": inventory["sources"]["ncore"]["revision"],
                "pycolmap_revision": inventory["sources"]["pycolmap"]["revision"],
                "npa_source_sha": Path("/usr/share/doc/npa-ncore/npa-source-sha")
                .read_text()
                .strip(),
                "gpu_required": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
