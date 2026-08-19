"""Start the pinned OpenPI service from a verified read-only runtime cache."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import sys
from typing import Sequence

try:
    from openpi_checkpoint_cache import (
        DEFAULT_CACHE_ROOT,
        fetch_generation_manifest,
        fetch_tokenizer_record,
        verify_runtime_cache,
    )
except ModuleNotFoundError:  # Repository import; image executes the adjacent script.
    from npa.workflows.byof.openpi_checkpoint_cache import (
        DEFAULT_CACHE_ROOT,
        fetch_generation_manifest,
        fetch_tokenizer_record,
        verify_runtime_cache,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-root", default=DEFAULT_CACHE_ROOT)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    if not 1 <= args.port <= 65535:
        parser.error("port must be between 1 and 65535")
    checkpoint = verify_runtime_cache(
        args.cache_root, fetch_generation_manifest(), fetch_tokenizer_record()
    )
    server = Path("/opt/byof/scripts/serve_policy.py")
    os.execv(
        sys.executable,
        [
            sys.executable,
            str(server),
            f"--port={args.port}",
            "policy:checkpoint",
            "--policy.config=pi05_droid_jointpos_polaris",
            f"--policy.dir={checkpoint}",
        ],
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
