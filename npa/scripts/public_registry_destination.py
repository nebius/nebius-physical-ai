"""Refuse a visibility change that would expose existing private GHCR versions.

The trusted workflow supplies every page from ``gh api --paginate --slurp``.
Untagged versions remain addressable by digest and are never presumed harmless.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


def require_empty_private_destination(pages: object) -> None:
    """Require a complete, empty paginated version inventory.

    Args:
        pages: Parsed output of the authenticated, fully paginated GHCR request.

    Returns:
        None when every page is an empty array.

    Raises:
        ValueError: The page envelope is malformed or contains existing versions.
    """
    if not isinstance(pages, list) or not pages:
        raise ValueError("Private destination version inventory is incomplete")
    if not all(isinstance(page, list) for page in pages):
        raise ValueError("Private destination version inventory is malformed")
    if any(pages):
        raise ValueError("Private destination contains existing versions; refusing a visibility change")


def main() -> int:
    """Check the private inventory without printing version names or metadata.

    Args:
        None; the command line supplies the private JSON file.

    Returns:
        Zero for a verified empty inventory, one for refusal.

    Raises:
        SystemExit: Command-line arguments are invalid.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--versions-json", type=Path, required=True)
    args = parser.parse_args()
    try:
        require_empty_private_destination(json.loads(args.versions_json.read_text()))
    except (OSError, ValueError):
        print("::error::Cannot prove private destination has no existing versions", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
