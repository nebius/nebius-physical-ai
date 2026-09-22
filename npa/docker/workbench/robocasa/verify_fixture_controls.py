#!/usr/bin/env python3
"""Verify pinned RoboCasa fixture-control bytes from the upstream checkout."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--assets-root", type=Path, required=True)
    parser.add_argument("--upstream-commit", required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == "npa.robocasa.pinned-fixture-controls/v1"
    assert manifest["upstream_commit"] == args.upstream_commit
    files = manifest["files"]
    assert isinstance(files, dict) and files
    for relative, expected in sorted(files.items()):
        path = PurePosixPath(relative)
        assert not path.is_absolute() and ".." not in path.parts
        source = args.assets_root.joinpath(*path.parts)
        assert source.is_file() and not source.is_symlink()
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected


if __name__ == "__main__":
    main()
