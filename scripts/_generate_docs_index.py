#!/usr/bin/env python3
"""Generate a deterministic docs/cli index."""

from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: _generate_docs_index.py <docs-cli-dir>", file=sys.stderr)
        return 2
    root = Path(sys.argv[1])
    pages = sorted(path for path in root.glob("*.md") if path.name != "README.md")

    print("# CLI Reference")
    print()
    print(
        "Generated from `npa --help`. Run `bash scripts/build_docs.sh` after CLI changes."
    )
    print()
    for path in pages:
        title = _title(path)
        print(f"- [{title}]({path.name})")
    return 0


def _title(path: Path) -> str:
    for line in path.read_text().splitlines():
        if line.startswith("# "):
            return line[2:].strip().strip("`")
    return path.stem


if __name__ == "__main__":
    raise SystemExit(main())
