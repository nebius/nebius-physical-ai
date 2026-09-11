"""Fail-closed verifier for a running Gymnasium-Robotics candidate image."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys

LOCK_ROOT = Path("/opt/npa/gymnasium-robotics")
SOURCE_ROOT = Path("/usr/share/source/npa-gymnasium-robotics")
FORBIDDEN = (
    Path("/root/.cache"),
    Path("/home/ubuntu/.cache/pip"),
    Path("/var/cache/apt/archives"),
    Path("/root/.docker/config.json"),
    Path("/root/.ssh"),
    Path("/workspace/byof-runs"),
)
EXPECTED_SOURCE = "4d1ebecbc6436806cfbc0e42ebc36f594d05844e"
EXPECTED_ASSET_LOCK = "e22eb62fc690a5e1d1ea931bab950392ca480caf3d51c7f16fd8cb4133d65568"
# Filled only after independent review of exact completed lock bytes. These
# Phase A sentinels make a local status edit insufficient to authorize use.
EXPECTED_COMPLETE_LOCK_SHA256: dict[str, str | None] = {
    "source-lock.json": None,
    "apt-runtime.lock.json": None,
    "corresponding-source.lock.json": None,
    "requirements.lock": None,
}


def _missing(value: object) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, list):
        return not value or any(_missing(item) for item in value)
    if isinstance(value, dict):
        return not value or any(_missing(item) for item in value.values())
    return False


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root: Path = Path("/")) -> dict[str, object]:
    def at(path: Path) -> Path:
        return root / path.relative_to("/")

    for name, expected in EXPECTED_COMPLETE_LOCK_SHA256.items():
        if not isinstance(expected, str) or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise ValueError(f"reviewed complete lock digest is not configured: {name}")
        if _sha256(at(LOCK_ROOT / name)) != expected:
            raise ValueError(f"reviewed complete lock bytes changed: {name}")
    locks = {}
    for name in (
        "source-lock.json",
        "apt-runtime.lock.json",
        "corresponding-source.lock.json",
    ):
        path = at(LOCK_ROOT / name)
        locks[name] = json.loads(path.read_text(encoding="utf-8"))
        if locks[name].get("status") != "complete":
            raise ValueError(f"incomplete image evidence: {name}")
        if _missing(locks[name]):
            raise ValueError(f"complete image evidence contains missing values: {name}")
    source_component = locks["source-lock.json"]["components"][
        "farama_gymnasium_robotics"
    ]
    if source_component.get("commit") != EXPECTED_SOURCE:
        raise ValueError("installed source lock does not match the reviewed commit")
    requirements = at(LOCK_ROOT / "requirements.lock").read_text(encoding="utf-8")
    if "# status: complete" not in requirements or "--hash=sha256:" not in requirements:
        raise ValueError("Python distribution lock is incomplete")
    if not at(SOURCE_ROOT).is_dir() or not any(at(SOURCE_ROOT).iterdir()):
        raise ValueError("corresponding-source annex is absent")
    for path in FORBIDDEN:
        candidate = at(path)
        if candidate.exists() and (not candidate.is_dir() or any(candidate.iterdir())):
            raise ValueError(f"forbidden baked runtime state: {path}")
    if root == Path("/") and os.geteuid() == 0:
        raise ValueError("image verifier must run as the non-root runtime user")
    executable = at(LOCK_ROOT / "capability_smoke.py")
    if not executable.is_file():
        raise ValueError("capability smoke is missing")
    if _sha256(at(LOCK_ROOT / "asset-lock.json")) != EXPECTED_ASSET_LOCK:
        raise ValueError("approved asset lock bytes changed")
    return {
        "schema": "npa.gymnasium-robotics.image-verification.v1",
        "status": "passed",
        "source_lock_sha256": _sha256(at(LOCK_ROOT / "source-lock.json")),
        "capability_smoke_sha256": _sha256(executable),
        "corresponding_source_present": True,
        "forbidden_runtime_state_count": 0,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("/"))
    parser.add_argument("--json", type=Path)
    args = parser.parse_args(argv)
    try:
        result = verify(args.root)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        print(str(error), file=sys.stderr)
        return 1
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.json:
        args.json.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
