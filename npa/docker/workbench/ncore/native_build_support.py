"""Shared paths, checked source extraction, and build receipts."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import tarfile

from assemble_native_sources import sha256

ROOT = Path("/build/ncore-native")
ANNEX = Path("/opt/ncore/native-sources")
WHEELS = ROOT / "wheels"
FFMPEG = Path("/opt/ncore/ffmpeg")
COMMANDS: list[list[str]] = []


def run(*args: str, cwd: Path | None = None, env: dict | None = None) -> None:
    COMMANDS.append(list(args))
    subprocess.run(args, check=True, cwd=cwd, env=env)


def unpack(name: str, version: str) -> Path:
    lock = json.loads((ANNEX / "native-source-lock.json").read_text())
    item = next(
        c for c in lock["components"] if (c["name"], c["version"]) == (name, version)
    )
    archive = ANNEX / item["artifacts"][0]["filename"]
    if sha256(archive) != item["artifacts"][0]["sha256"]:
        raise ValueError(f"SHA-256 mismatch: {archive.name}")
    dest = ROOT / "source" / f"{name}-{version}"
    dest.mkdir(parents=True)
    with tarfile.open(archive) as source:
        source.extractall(dest, filter="data")
    children = list(dest.iterdir())
    if len(children) != 1 or not children[0].is_dir():
        raise ValueError("expected a single source directory")
    return children[0]


def record(component: str) -> None:
    (ANNEX / f"build-commands-{component}.json").write_text(
        json.dumps(COMMANDS, indent=2) + "\n"
    )
    (ANNEX / "wheel-sha256.json").write_text(
        json.dumps(
            {
                str(p.relative_to(WHEELS)): sha256(p)
                for p in sorted(WHEELS.rglob("*.whl"))
            },
            indent=2,
        )
        + "\n"
    )
    (ANNEX / "build-debian-packages.tsv").write_text(
        subprocess.check_output(
            [
                "dpkg-query",
                "-W",
                "-f=${binary:Package}\t${Version}\t${source:Package}\t${source:Version}\n",
            ],
            text=True,
        )
    )
