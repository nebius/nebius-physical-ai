#!/usr/bin/env python3
"""Fail unless a built Antioch adapter contains no vendor distribution or config."""

from __future__ import annotations

import argparse
import json
import subprocess


def _run(argv: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(argv, text=True, capture_output=True, check=False)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    args = parser.parse_args()
    probe = r"""
import importlib.metadata as metadata
import json
from pathlib import Path

packages = sorted(
    distribution.metadata.get("Name", "")
    for distribution in metadata.distributions()
    if "antioch" in distribution.metadata.get("Name", "").lower()
)
forbidden = []
for root in (Path("/app"), Path("/etc"), Path("/opt"), Path("/workspace")):
    if not root.exists():
        continue
    for path in root.rglob("*"):
        name = path.name.lower()
        if name in {"auth.json", "machines.json", "credentials.json"} or name.startswith("antioch_sim-"):
            forbidden.append(str(path))
print(json.dumps({"packages": packages, "forbidden_paths": sorted(forbidden)}))
raise SystemExit(1 if packages or forbidden else 0)
"""
    result = _run(
        ["docker", "run", "--rm", "--entrypoint", "python", args.image, "-c", probe]
    )
    if result.returncode not in {0, 1}:
        raise SystemExit("could not inspect built Antioch image")
    try:
        detail = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise SystemExit("built-image probe returned malformed output") from exc
    report = {
        "format": "npa_antioch_payload_scan_v1",
        "image": args.image,
        "scan_complete": True,
        "verdict": "clean" if result.returncode == 0 else "forbidden-payload-detected",
        **detail,
    }
    print(json.dumps(report, sort_keys=True))
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
