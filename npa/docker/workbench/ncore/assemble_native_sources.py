"""Fetch/verify the native source annex; a cache is read-only and hash checked.

This annex covers the components in native-source-lock.json, not every byte of
the container. Publishing it separately still requires actual recipient access.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request


def sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def assemble(lock_path: Path, output: Path, cache: Path | None = None) -> None:
    lock = json.loads(lock_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    for component in lock["components"]:
        for item in component["artifacts"]:
            name = item["filename"]
            if Path(name).name != name or name in ("", ".", ".."):
                raise ValueError(f"unsafe artifact name: {name}")
            target = output / name
            if target.exists():
                if sha256(target) != item["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {name}")
                continue
            cached = cache / name if cache else None
            temporary = output / (name + ".partial")
            try:
                if cached and cached.exists():
                    shutil.copyfile(cached, temporary)
                else:
                    if not item["url"].startswith("https://"):
                        raise ValueError("source URL must use HTTPS")
                    with urllib.request.urlopen(item["url"]) as source:
                        with temporary.open("wb") as destination:
                            shutil.copyfileobj(source, destination)
                if sha256(temporary) != item["sha256"]:
                    raise ValueError(f"SHA-256 mismatch: {name}")
                temporary.replace(target)
            finally:
                temporary.unlink(missing_ok=True)
    shutil.copyfile(lock_path, output / "native-source-lock.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cache", type=Path)
    args = parser.parse_args()
    assemble(args.lock, args.output, args.cache)
