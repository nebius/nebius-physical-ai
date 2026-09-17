"""Scan the public base-image inventory with bounded local parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import re
import subprocess


_NAME_PATTERN = re.compile(r"[a-z0-9][a-z0-9-]+")
_PATCH_DOCKERFILE = """\
ARG BASE_IMAGE
FROM ${BASE_IMAGE}
USER root
ARG PURGE_LINUX_LIBC_DEV=false
ARG UPGRADE_OS=false
RUN if [ "$UPGRADE_OS" = "true" ]; then \\
      apt-get update \\
      && apt-get upgrade -y --no-install-recommends; \\
    fi \\
    && if [ "$PURGE_LINUX_LIBC_DEV" = "true" ]; then \\
      dpkg --purge --force-depends linux-libc-dev; \\
    fi \\
    && rm -rf /var/lib/apt/lists/*
"""


def load_inventory(path: Path) -> list[dict[str, object]]:
    """Load and validate the digest-pinned base-image inventory.

    Args:
        path: JSON inventory path.
    Returns:
        Validated image entries.
    Raises:
        ValueError: An entry is incomplete, duplicated, or not digest pinned.
    """

    entries = json.loads(path.read_text(encoding="utf-8"))
    names = [str(entry.get("name", "")) for entry in entries]
    if not entries or len(names) != len(set(names)):
        raise ValueError("base-image inventory must contain unique entries")
    for entry, name in zip(entries, names, strict=True):
        image = str(entry.get("image", ""))
        if not _NAME_PATTERN.fullmatch(name) or "@sha256:" not in image:
            raise ValueError(f"invalid or unpinned base-image entry: {name!r}")
        for field in ("purge_linux_libc_dev", "upgrade_os"):
            if not isinstance(entry.get(field), bool):
                raise ValueError(f"{name}: {field} must be boolean")
    return entries


def preparation_command(entry: dict[str, object]) -> list[str] | None:
    """Build the exact Docker command needed to prepare one scan target.

    Args:
        entry: Validated base-image inventory entry.
    Returns:
        Docker command, or ``None`` when the pinned image is scanned directly.
    Raises:
        None.
    """

    purge = str(entry["purge_linux_libc_dev"]).lower()
    upgrade = str(entry["upgrade_os"]).lower()
    if purge == "false" and upgrade == "false":
        return None
    return [
        "docker", "build", "--pull", "--no-cache",
        "--build-arg", f"BASE_IMAGE={entry['image']}",
        "--build-arg", f"PURGE_LINUX_LIBC_DEV={purge}",
        "--build-arg", f"UPGRADE_OS={upgrade}",
        "-t", f"npa-base-scan:{entry['name']}", "-f", "-", ".",
    ]


def prepare_target(entry: dict[str, object]) -> str:
    """Prepare and return the exact image reference Trivy must scan.

    Args:
        entry: Validated base-image inventory entry.
    Returns:
        Original digest or locally patched image tag.
    Raises:
        subprocess.CalledProcessError: Docker cannot construct the target.
    """

    command = preparation_command(entry)
    if command is None:
        return str(entry["image"])
    subprocess.run(command, input=_PATCH_DOCKERFILE, text=True, check=True)
    return f"npa-base-scan:{entry['name']}"


def _trivy_command(target: str, cache: Path, *, sarif: Path | None) -> list[str]:
    command = [
        "trivy", "image", "--cache-dir", str(cache), "--skip-db-update",
        "--severity", "CRITICAL", "--ignore-unfixed", "--vuln-type", "os",
        "--timeout", "2562047h47m16s", "--exit-code", "0" if sarif else "1",
    ]
    if sarif:
        command.extend(["--format", "sarif", "--output", str(sarif)])
    command.append(target)
    return command


def scan_entry(
    entry: dict[str, object], cache: Path, sarif_directory: Path | None
) -> int:
    """Prepare and scan one base image, optionally emitting SARIF.

    Args:
        entry: Validated base-image inventory entry.
        cache: Shared pre-populated Trivy cache.
        sarif_directory: Output directory for non-PR reporting.
    Returns:
        Blocking table-scan exit status.
    Raises:
        subprocess.CalledProcessError: SARIF generation itself fails.
    """

    target = prepare_target(entry)
    blocking = subprocess.run(_trivy_command(target, cache, sarif=None), check=False)
    if sarif_directory is not None:
        report = sarif_directory / f"trivy-{entry['name']}.sarif"
        subprocess.run(_trivy_command(target, cache, sarif=report), check=True)
    return blocking.returncode


def scan_inventory(
    entries: list[dict[str, object]], cache: Path, workers: int,
    sarif_directory: Path | None,
) -> None:
    """Scan all entries with bounded parallelism and fail on any finding.

    Args:
        entries: Validated base-image inventory.
        cache: Shared Trivy cache.
        workers: Maximum concurrent image scans.
        sarif_directory: Optional SARIF output directory.
    Returns:
        None.
    Raises:
        RuntimeError: A blocking scan reports a finding or execution error.
    """

    cache.mkdir(parents=True, exist_ok=True)
    if sarif_directory is not None:
        sarif_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["trivy", "image", "--cache-dir", str(cache), "--download-db-only"],
        check=True,
    )
    with ThreadPoolExecutor(max_workers=workers) as executor:
        results = list(
            executor.map(
                lambda entry: scan_entry(entry, cache, sarif_directory), entries
            )
        )
    if any(result != 0 for result in results):
        raise RuntimeError("one or more base-image security scans failed")


def main() -> int:
    """Run the bounded base-image security scan.

    Args:
        None.
    Returns:
        Process exit status.
    Raises:
        ValueError: Inventory or worker arguments are invalid.
        RuntimeError: Any blocking scan fails.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--sarif-directory", type=Path)
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    entries = load_inventory(arguments.inventory)
    scan_inventory(
        entries, arguments.cache_dir, arguments.workers, arguments.sarif_directory
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
