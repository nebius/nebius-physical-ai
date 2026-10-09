"""Run native Lyra reconstruction on an operator-supplied checksummed capture."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

import av


def main() -> int:
    """Reconstruct real RGB input and require a decoded native Gaussian video."""
    source = os.environ.get("NPA_LYRA_INPUT_PATH")
    if not source:
        raise ValueError("Set NPA_LYRA_INPUT_PATH to a prepared Lyra capture bundle")
    with tempfile.TemporaryDirectory(prefix="lyra2-golden-") as scratch:
        output = Path(scratch) / "reconstruction"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "npa.workflows.lyra_stage",
                "--input-path",
                source,
                "--output-path",
                str(output),
            ],
            check=True,
        )
        result = json.loads((output / "reconstruction.json").read_text())
        with av.open(str(output / "gs_trajectory.mp4")) as video:
            frames = sum(1 for _ in video.decode(video=0))
        if frames < 2 or result["views"] < 2:
            raise ValueError("Lyra produced no useful native rendered sequence")
        if not (output / "index.html").is_file():
            raise ValueError("Lyra produced no standalone viewer")
        print(
            json.dumps(
                {
                    "status": "passed",
                    "frames": frames,
                    "gpu": result["gpu"],
                    "views": result["views"],
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
