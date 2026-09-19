"""Remove non-runtime developer payloads from separately installed NVIDIA wheels."""

from __future__ import annotations

import importlib.metadata
import json
import shutil
from pathlib import Path


def _site_root() -> Path:
    distribution = importlib.metadata.distribution("nvidia-cudnn-cu12")
    return Path(distribution.locate_file(""))


def _targets(root: Path) -> list[Path]:
    vendor_root = root / "nvidia"
    targets = list(vendor_root.glob("*/include"))
    targets.extend(vendor_root.rglob("*.a"))
    return sorted(set(targets))


def _remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path)
    else:
        path.unlink()


def main() -> int:
    """Prune separately installed SDK headers/static archives and record the result."""
    root = _site_root()
    targets = _targets(root)
    removed = [str(path.relative_to(root)) for path in targets]
    for path in targets:
        _remove(path)
    remaining = _targets(root)
    if remaining:
        raise RuntimeError(f"developer payload remains: {remaining}")
    destination = Path("/usr/share/doc/npa-openarm/python-vendor-prune.json")
    destination.write_text(
        json.dumps({"schema": "npa.vendor-prune.v1", "removed": removed}, indent=2)
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
