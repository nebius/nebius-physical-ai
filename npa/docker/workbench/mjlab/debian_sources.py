"""Fetch Debian package sources concurrently through signed snapshot APT metadata."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess

from bootstrap_sources import packages, source_files


def _fetch(root: Path, source: str, version: str) -> dict:
    directory = root / "ubuntu-sources" / source / version
    directory.mkdir(parents=True, exist_ok=False)
    subprocess.run(
        [
            "apt-get",
            "source",
            "--download-only",
            "--only-source",
            f"{source}={version}",
        ],
        cwd=directory,
        check=True,
    )
    return {
        "source": source,
        "version": version,
        "artifacts": source_files(directory, source, version),
    }


def main() -> None:
    """Deliver source for original and installed Debian package versions.

    Args:
        None. The source delivery directory is supplied on the command line.
    Returns:
        None.
    Raises:
        ValueError, OSError, subprocess.CalledProcessError: Source validation or fetch fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    # Separate directories keep each download independent; APT indexes are read only.
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(_fetch, args.root, source, version)
            for source, version in packages(args.root)
        ]
        records = [future.result() for future in futures]
    (args.root / "bootstrap-sources.json").write_text(
        json.dumps(records, sort_keys=True, indent=2) + "\n"
    )
    print(f"Delivered verified source for {len(records)} Debian components")


if __name__ == "__main__":
    main()
