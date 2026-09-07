"""Image boundary or runtime import/CLI check; neither validates a capture."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import shutil

from npa.workflows.ncore_runtime import (
    RUNTIME_LOCK,
    ensure_runtime,
    read_lock,
    verify_runtime,
    _run,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-only",
        action="store_true",
        help="Offline packaging boundary; never fetch runtime dependencies.",
    )
    args = parser.parse_args()
    for command in ("sh", "sudo", "sshd", "rsync", "service"):
        assert shutil.which(command), f"bootstrap command absent from PATH: {command}"
    lock = read_lock(RUNTIME_LOCK)
    for name in ("torch", "av", "numpy", "rerun", "lancedb", "pyarrow", "ncore"):
        assert importlib.util.find_spec(name) is None, (
            f"unexpected baked dependency: {name}"
        )
    root = Path("/opt/ncore/src")
    inventory = json.loads((root / "source-inventory.json").read_bytes())
    for relative, expected in inventory["files"].items():
        assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == expected, (
            relative
        )
    revision = Path("/usr/share/doc/npa-ncore/npa-source-sha").read_text().strip()
    assert re.fullmatch("[0-9a-f]{40}", revision)
    if not args.image_only:
        runtime = ensure_runtime()
        verify_runtime(runtime)
        _run(
            [
                str(runtime / "bin/python"),
                "-I",
                "-B",
                "-m",
                "npa.workflows.ncore",
                "workbench",
                "nurec",
                "convert-colmap",
                "--help",
            ]
        )
    print(
        json.dumps(
            {
                "status": "packaging-ready",
                "validation": "image-boundary-only"
                if args.image_only
                else "import-and-cli-schema-only",
                "dependency_delivery": "hash-locked-runtime-fetch",
                "locked_distributions": len(lock["artifacts"]),
                "ncore_revision": inventory["sources"]["ncore"]["revision"],
                "pycolmap_revision": inventory["sources"]["pycolmap"]["revision"],
                "npa_source_sha": revision,
                "gpu_required": False,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
