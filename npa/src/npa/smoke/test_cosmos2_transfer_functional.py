"""Cosmos Transfer 2.5 golden-eval adapter.

The image-owned smoke script generates a repository-authored procedural video,
runs the real multi-step ``examples/inference.py`` path, and validates the MP4
numerically. Keeping this module as a thin adapter makes the manifest's ``module``
and ``command`` fields exercise the same implementation.
"""

from __future__ import annotations

import os
import subprocess
import sys

SMOKE_SCRIPT = os.environ.get(
    "COSMOS_TRANSFER_SMOKE_SCRIPT",
    "/opt/cosmos2-transfer/smoke_functional.sh",
)


def main() -> int:
    return subprocess.run(["bash", SMOKE_SCRIPT], check=False).returncode


if __name__ == "__main__":
    sys.exit(main())
