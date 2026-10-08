"""Materialize related Lyra and Isaac bundles and publish an offline HTML review."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys
import tempfile

from npa.workflows.lerobot_transfer_data import materialize, publish


def main():
    """Build the review only from checksummed, scene-matched stage outputs.

    Args:
        None; reads private artifact locations from the command line.
    Returns:
        None after publishing the completed HTML bundle.
    Raises:
        ValueError: Artifact identity or its checksums do not match.
        subprocess.CalledProcessError: HTML evidence validation fails.
        OSError: Artifact exchange fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("reconstruction", "geometry", "actions", "output"):
        parser.add_argument(f"--{name}-path", required=True)
    args = parser.parse_args()
    with tempfile.TemporaryDirectory(prefix="lyra-review-") as temporary:
        root = Path(temporary)
        inputs = {
            name: materialize(getattr(args, f"{name}_path"), root / name)
            for name in ("reconstruction", "geometry", "actions")
        }
        argv = [sys.executable, "-m", "npa.workflows.lyra_demo"]
        for name, path in inputs.items():
            argv.extend([f"--{name}-path", str(path)])
        output = root / "review"
        argv.extend(["--input-path", str(inputs["reconstruction"])])
        argv.extend(["--output-path", str(output / "index.html")])
        argv.append("--require-complete")
        subprocess.run(argv, check=True)
        publish(output, args.output_path)


if __name__ == "__main__":
    main()
