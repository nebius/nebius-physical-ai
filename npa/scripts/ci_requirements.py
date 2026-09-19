"""Refresh or check the dependency pins used by CI's Python test environments."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib


_ROOT = Path(__file__).resolve().parents[2]
_UV_VERSION = "0.12.5"
_FINGERPRINT_PREFIX = "# CI dependency inputs SHA256: "


def _fingerprint(root: Path) -> str:
    project = tomllib.loads((root / "npa/pyproject.toml").read_text())["project"]
    inputs = {
        "python": project["requires-python"],
        "dependencies": project["dependencies"],
        "extras": {
            extra: project["optional-dependencies"][extra]
            for extra in ("dev", "adapter", "sonic")
        },
        "constraints": (root / "npa/ci/constraints.in").read_text(),
        "uv": _UV_VERSION,
    }
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _check(root: Path) -> None:
    header = (root / "npa/ci/requirements.txt").read_text().splitlines()[0]
    if header != _FINGERPRINT_PREFIX + _fingerprint(root):
        raise ValueError(
            "CI dependencies changed; run npa/.venv/bin/python "
            "npa/scripts/ci_requirements.py --update"
        )


def _update(root: Path, upgrade: bool) -> None:
    version = subprocess.check_output(["uv", "--version"], text=True).split()[1]
    if version != _UV_VERSION:
        raise ValueError(f"Regenerate CI requirements with uv {_UV_VERSION}")
    command = [
        "uv",
        "pip",
        "compile",
        "npa/pyproject.toml",
        "--extra",
        "dev",
        "--extra",
        "adapter",
        "--extra",
        "sonic",
        "--constraints",
        "npa/ci/constraints.in",
        "--universal",
        "--python-version",
        "3.10",
        "--torch-backend",
        "cpu",
        "--no-annotate",
        "--custom-compile-command",
        "npa/.venv/bin/python npa/scripts/ci_requirements.py --update",
        "--output-file",
        "npa/ci/requirements.txt",
    ]
    if upgrade:
        command.append("--upgrade")
    subprocess.run(command, cwd=root, check=True)
    _validate_linux_wheels(root)
    path = root / "npa/ci/requirements.txt"
    path.write_text(_FINGERPRINT_PREFIX + _fingerprint(root) + "\n" + path.read_text())


def _validate_linux_wheels(root: Path) -> None:
    # Universal resolution checks metadata; explicit targets also check wheel tags.
    with tempfile.TemporaryDirectory(prefix="npa-ci-pins-") as directory:
        for version in ("3.10", "3.12", "3.14"):
            subprocess.run(
                [
                    "uv",
                    "pip",
                    "compile",
                    "npa/pyproject.toml",
                    "--extra",
                    "dev",
                    "--extra",
                    "adapter",
                    "--extra",
                    "sonic",
                    "--constraints",
                    "npa/ci/requirements.txt",
                    "--torch-backend",
                    "cpu",
                    "--python-version",
                    version,
                    "--python-platform",
                    "x86_64-unknown-linux-gnu",
                    "--only-binary",
                    ":all:",
                    "--output-file",
                    str(Path(directory) / f"{version}.txt"),
                ],
                cwd=root,
                check=True,
                stdout=subprocess.DEVNULL,
            )


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--check", action="store_true")
    mode.add_argument("--update", action="store_true")
    parser.add_argument("--upgrade", action="store_true")
    args = parser.parse_args()
    if args.upgrade and not args.update:
        parser.error("--upgrade requires --update")
    if args.update:
        _update(_ROOT, args.upgrade)
    else:
        _check(_ROOT)


if __name__ == "__main__":
    _main()
