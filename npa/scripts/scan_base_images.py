"""Scan the public base-image inventory with bounded local parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import tempfile
from uuid import uuid4


_BUILDKIT_IMAGE = (
    "moby/buildkit:v0.33.0@sha256:"
    "6c2fa84a6b61ccd72899dde4239f8d5717f05f9a8ca6f3cad185fb1a95a94de3"
)
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


class _BuilderCleanupError(RuntimeError):
    """Keep the owned builder receipt when its disk cleanup is incomplete."""


def preparation_command(
    entry: dict[str, object], builder: str, archive: Path, context: Path
) -> list[str] | None:
    """Export a prepared base without loading it into the shared image store.

    Args:
        entry: Validated base-image inventory entry.
        builder: Scan-owned isolated Buildx builder.
        archive: Private Docker archive output path.
        context: Empty private build context.
    Returns:
        Docker command, or None for an unmodified digest.
    Raises:
        None.
    """

    if not entry["purge_linux_libc_dev"] and not entry["upgrade_os"]:
        return None
    command = [
        "docker",
        "buildx",
        "build",
        "--builder",
        builder,
        "--pull",
        "--no-cache",
    ]
    for name, value in (
        ("BASE_IMAGE", entry["image"]),
        ("PURGE_LINUX_LIBC_DEV", str(entry["purge_linux_libc_dev"]).lower()),
        ("UPGRADE_OS", str(entry["upgrade_os"]).lower()),
    ):
        command.extend(["--build-arg", f"{name}={value}"])
    command.extend(["--output", f"type=docker,dest={archive}", "-f", "-", str(context)])
    return command


def _remove_builder(builder: str, environment: dict[str, str], root: Path) -> None:
    command = ["docker", "buildx", "rm", "--force", builder]
    try:
        result = subprocess.run(command, env=environment, check=False)
    except OSError as error:
        raise _BuilderCleanupError(
            f"builder cleanup could not execute; retained receipt: {root}"
        ) from error
    if result.returncode:
        raise _BuilderCleanupError(f"builder cleanup failed; retained receipt: {root}")


def _create_builder(builder: str, root: Path) -> dict[str, str]:
    environment = {**os.environ, "BUILDX_CONFIG": str(root / "buildx")}
    (root / "builder.json").write_text(json.dumps({"name": builder}))
    command = ["docker", "buildx", "create", "--name", builder]
    command.extend(
        ["--driver", "docker-container", "--driver-opt", f"image={_BUILDKIT_IMAGE}"]
    )
    subprocess.run(command, env=environment, check=True)
    return environment


def prepare_target(entry: dict[str, object], root: Path) -> str | Path:
    """Free an entry's isolated builder before scanning its archive.

    Args:
        entry: Validated base-image inventory entry.
        root: Entry-private working directory.
    Returns:
        Original digest or private Docker archive.
    Raises:
        subprocess.CalledProcessError: Builder creation or export fails.
        _BuilderCleanupError: Owned builder cleanup fails.
    """

    builder = f"npa-base-scan-{uuid4().hex}"
    archive = root / "image.tar"
    context = root / "context"
    command = preparation_command(entry, builder, archive, context)
    if command is None:
        return str(entry["image"])
    context.mkdir(mode=0o700)
    environment = _create_builder(builder, root)
    try:
        subprocess.run(
            command, input=_PATCH_DOCKERFILE, text=True, env=environment, check=True
        )
    finally:
        _remove_builder(builder, environment, root)
    return archive


@contextmanager
def _entry_workspace(cache: Path):
    root = Path(tempfile.mkdtemp(prefix="entry-", dir=cache))
    retain = False
    try:
        yield root
    except _BuilderCleanupError:
        retain = True
        raise
    finally:
        if not retain:
            shutil.rmtree(root)


def _trivy_command(target: str | Path, cache: Path, *, sarif: Path | None) -> list[str]:
    command = [
        "trivy",
        "image",
        "--cache-dir",
        str(cache),
        "--skip-db-update",
        "--severity",
        "CRITICAL",
        "--ignore-unfixed",
        "--vuln-type",
        "os",
        "--timeout",
        "2562047h47m16s",
        "--exit-code",
        "0" if sarif else "1",
    ]
    if sarif:
        command.extend(["--format", "sarif", "--output", str(sarif)])
    if isinstance(target, Path):
        command.extend(["--input", str(target)])
    else:
        command.extend(["--image-src", "remote", target])
    return command


def scan_entry(
    entry: dict[str, object], cache: Path, sarif_directory: Path | None
) -> int:
    """Prepare and scan one base image, optionally emitting SARIF.

    Args:
        entry: Validated base-image inventory entry.
        cache: Worker-private Trivy cache with a pre-populated database.
        sarif_directory: Output directory for non-PR reporting.
    Returns:
        Blocking table-scan exit status.
    Raises:
        subprocess.CalledProcessError: Preparation or SARIF generation fails.
        _BuilderCleanupError: Owned builder cleanup fails.
    """

    with _entry_workspace(cache) as root:
        target = prepare_target(entry, root)
        entry_cache = _create_worker_cache(cache / "db", root, 0)
        temporary = root / "tmp"
        temporary.mkdir(mode=0o700)
        environment = {**os.environ, "TMPDIR": str(temporary)}
        blocking = subprocess.run(
            _trivy_command(target, entry_cache, sarif=None),
            env=environment,
            check=False,
        )
        if sarif_directory is not None:
            report = sarif_directory / f"trivy-{entry['name']}.sarif"
            subprocess.run(
                _trivy_command(target, entry_cache, sarif=report),
                env=environment,
                check=True,
            )
        return blocking.returncode


def _scan_worker(
    entries: Queue[dict[str, object]], cache: Path, sarif_directory: Path | None
) -> list[int]:
    """Drain shared work with one worker-private Trivy cache.

    Args:
        entries: Shared queue of unclaimed inventory entries.
        cache: Worker-private Trivy cache.
        sarif_directory: Optional SARIF output directory.
    Returns:
        Blocking scan exit statuses.
    Raises:
        subprocess.CalledProcessError: Preparation or SARIF generation fails.
    """

    results = []
    while True:
        try:
            entry = entries.get_nowait()
        except Empty:
            return results
        try:
            results.append(scan_entry(entry, cache, sarif_directory))
        finally:
            entries.task_done()


def _create_worker_cache(database: Path, root: Path, index: int) -> Path:
    """Create a private artifact cache that reuses the read-only database.

    Args:
        database: Pre-populated vulnerability database directory.
        root: Temporary worker-cache root.
        index: Stable worker index.
    Returns:
        Worker-private cache directory.
    Raises:
        OSError: Cache directories or database links cannot be created.
    """

    worker_cache = root / str(index)
    worker_cache.mkdir()
    shutil.copytree(database, worker_cache / "db", copy_function=os.link)
    return worker_cache


@contextmanager
def _inventory_workspace(cache: Path):
    root = Path(tempfile.mkdtemp(prefix="scan-workers-", dir=cache))
    try:
        yield root
    finally:
        if not list(root.glob("*/entry-*/builder.json")):
            shutil.rmtree(root)


def _download_database(cache: Path) -> None:
    subprocess.run(
        ["trivy", "image", "--cache-dir", str(cache), "--download-db-only"],
        check=True,
    )


def scan_inventory(
    entries: list[dict[str, object]],
    cache: Path,
    workers: int,
    sarif_directory: Path | None,
) -> None:
    """Scan all entries with bounded parallelism and fail on any finding.

    Args:
        entries: Validated base-image inventory.
        cache: Trivy database and temporary worker-cache root.
        workers: Maximum concurrent image scans.
        sarif_directory: Optional SARIF output directory.
    Returns:
        None.
    Raises:
        RuntimeError: Any blocking scan fails.
    """
    cache.mkdir(parents=True, exist_ok=True)
    if sarif_directory is not None:
        sarif_directory.mkdir(parents=True, exist_ok=True)
    _download_database(cache)
    worker_count = min(workers, len(entries))
    with _inventory_workspace(cache) as root:
        worker_caches = [
            _create_worker_cache(cache / "db", root, index)
            for index in range(worker_count)
        ]
        queue: Queue[dict[str, object]] = Queue()
        for entry in entries:
            queue.put(entry)
        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            futures = [
                executor.submit(_scan_worker, queue, worker_cache, sarif_directory)
                for worker_cache in worker_caches
            ]
            results = [result for future in futures for result in future.result()]
    if any(result != 0 for result in results):
        raise RuntimeError("one or more base-image security scans failed")


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sarif-directory", type=Path)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument("--matrix", action="store_true")
    selection.add_argument("--entry-name")
    return parser.parse_args()


def main() -> int:
    """Scan an inventory or one validated entry, or emit its complete CI matrix.

    Args:
        None.
    Returns:
        Process exit status.
    Raises:
        ValueError: Inventory, entry name, or worker arguments are invalid.
        RuntimeError: Any blocking scan fails.
    """

    arguments = _arguments()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    entries = load_inventory(arguments.inventory)
    if arguments.matrix:
        print(json.dumps({"entry": [entry["name"] for entry in entries]}))
        return 0
    if arguments.entry_name is not None:
        entries = [entry for entry in entries if entry["name"] == arguments.entry_name]
        if not entries:
            raise ValueError("entry name is absent from the validated inventory")
    scan_inventory(
        entries, arguments.cache_dir, arguments.workers, arguments.sarif_directory
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
