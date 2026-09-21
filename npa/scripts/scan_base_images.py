"""Scan the public base-image inventory with bounded local parallelism."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
from queue import Empty, Queue
import re
import shutil
import subprocess
import tempfile


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
        "docker",
        "build",
        "--pull",
        "--no-cache",
        "--build-arg",
        f"BASE_IMAGE={entry['image']}",
        "--build-arg",
        f"PURGE_LINUX_LIBC_DEV={purge}",
        "--build-arg",
        f"UPGRADE_OS={upgrade}",
        "-t",
        f"npa-base-scan:{entry['name']}",
        "-f",
        "-",
        ".",
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
    command.append(target)
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
        subprocess.CalledProcessError: SARIF generation itself fails.
    """

    target = prepare_target(entry)
    return _scan_target(target, str(entry["name"]), cache, sarif_directory)


def _scan_target(
    target: str, name: str, cache: Path, sarif_directory: Path | None
) -> int:
    """Apply the unchanged blocking scan and optional report to one target."""

    blocking = subprocess.run(_trivy_command(target, cache, sarif=None), check=False)
    if sarif_directory is not None:
        report = sarif_directory / f"trivy-{name}.sarif"
        subprocess.run(_trivy_command(target, cache, sarif=report), check=True)
    return blocking.returncode


def _require_disposable_runner(workers: int) -> None:
    """Refuse destructive cache cleanup outside one local hosted-runner worker."""

    hosted = (
        os.environ.get("GITHUB_ACTIONS") == "true"
        and os.environ.get("RUNNER_ENVIRONMENT") == "github-hosted"
        and os.environ.get("RUNNER_OS") == "Linux"
    )
    local = os.environ.get("DOCKER_HOST", "") in {
        "",
        "unix:///var/run/docker.sock",
    } and os.environ.get("DOCKER_CONTEXT", "") in {"", "default"}
    local = (
        local
        and os.environ.get("BUILDX_BUILDER", "") in {"", "default"}
        and not os.environ.get("BUILDKIT_HOST")
    )
    if not hosted or not local or workers != 1:
        raise ValueError(
            "disposable Docker cleanup requires one local GitHub-hosted worker"
        )
    context = subprocess.run(
        ["docker", "context", "inspect", "--format", "{{.Endpoints.docker.Host}}"],
        capture_output=True,
        text=True,
        check=True,
    )
    if context.stdout.strip() != "unix:///var/run/docker.sock":
        raise ValueError("disposable Docker cleanup requires the local default daemon")


def _prune_build_cache() -> None:
    """Release disposable build cache while retaining the exact loaded image."""

    subprocess.run(
        ["docker", "buildx", "--builder", "default", "prune", "--all", "--force"],
        check=True,
    )


def _scan_disposable_entry(
    entry: dict[str, object], cache: Path, sarif_directory: Path | None
) -> int:
    """Bound hosted-runner storage to one inventory entry and its artifact cache."""

    _require_disposable_runner(1)
    with tempfile.TemporaryDirectory(prefix="entry-", dir=cache) as temp:
        entry_cache = _create_worker_cache(cache / "db", Path(temp), 0)
        try:
            target = prepare_target(entry)
            _prune_build_cache()
            return _scan_target(
                target, str(entry["name"]), entry_cache, sarif_directory
            )
        finally:
            try:
                _prune_build_cache()
            finally:
                subprocess.run(
                    ["docker", "image", "prune", "--all", "--force"], check=True
                )


def _scan_worker(
    entries: Queue[dict[str, object]],
    cache: Path,
    sarif_directory: Path | None,
    disposable_docker: bool = False,
) -> list[int]:
    """Drain shared work with one worker-private Trivy cache.

    Args:
        entries: Shared queue of unclaimed inventory entries.
        cache: Worker-private Trivy cache.
        sarif_directory: Optional SARIF output directory.
        disposable_docker: Reclaim each image on an isolated hosted runner.
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
            scan = _scan_disposable_entry if disposable_docker else scan_entry
            results.append(scan(entry, cache, sarif_directory))
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


def _prepare_scan_storage(cache: Path, sarif_directory: Path | None) -> None:
    """Create scan output directories and download the shared vulnerability database.

    Args:
        cache: Database and temporary worker-cache root.
        sarif_directory: Optional SARIF output directory.
    Returns:
        None.
    Raises:
        OSError: A scan output directory cannot be created.
        subprocess.CalledProcessError: The vulnerability database download fails.
    """

    cache.mkdir(parents=True, exist_ok=True)
    if sarif_directory is not None:
        sarif_directory.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["trivy", "image", "--cache-dir", str(cache), "--download-db-only"],
        check=True,
    )


def _scan_prepared_caches(
    entries: list[dict[str, object]],
    worker_caches: list[Path],
    sarif_directory: Path | None,
    disposable_docker: bool,
) -> list[int]:
    """Scan a shared inventory queue using the prepared isolated caches.

    Args:
        entries: Inventory entries in their original order.
        worker_caches: One prepared cache for each participating worker.
        sarif_directory: Optional SARIF output directory.
        disposable_docker: Enable the guarded disposable-runner cleanup mode.
    Returns:
        Blocking scan exit statuses in worker result order.
    Raises:
        subprocess.CalledProcessError: Preparation, reporting, or cleanup fails.
    """

    queue: Queue[dict[str, object]] = Queue()
    for entry in entries:
        queue.put(entry)
    with ThreadPoolExecutor(max_workers=len(worker_caches)) as executor:
        futures = [
            executor.submit(
                _scan_worker, queue, worker_cache, sarif_directory, disposable_docker
            )
            for worker_cache in worker_caches
        ]
        return [result for future in futures for result in future.result()]


def scan_inventory(
    entries: list[dict[str, object]],
    cache: Path,
    workers: int,
    sarif_directory: Path | None,
    *,
    disposable_docker: bool = False,
) -> None:
    """Scan all entries with bounded parallelism and fail on any finding.

    Args:
        entries: Validated base-image inventory.
        cache: Trivy database and temporary worker-cache root.
        workers: Maximum concurrent image scans.
        sarif_directory: Optional SARIF output directory.
        disposable_docker: Reclaim scan images on one GitHub-hosted worker only.
    Returns:
        None.
    Raises:
        RuntimeError: A blocking scan reports a finding or execution error.
    """

    if disposable_docker:
        _require_disposable_runner(workers)
    _prepare_scan_storage(cache, sarif_directory)
    worker_count = min(workers, len(entries))
    with tempfile.TemporaryDirectory(prefix="scan-workers-", dir=cache) as temp:
        root = Path(temp)
        worker_caches = [
            _create_worker_cache(cache / "db", root, index)
            for index in range(worker_count)
        ]
        results = _scan_prepared_caches(
            entries, worker_caches, sarif_directory, disposable_docker
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
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--sarif-directory", type=Path)
    parser.add_argument(
        "--disposable-docker",
        action="store_true",
        help="Reclaim scan images and build cache on a GitHub-hosted runner; requires --workers 1.",
    )
    arguments = parser.parse_args()
    if arguments.workers < 1:
        raise ValueError("workers must be positive")
    entries = load_inventory(arguments.inventory)
    scan_inventory(
        entries,
        arguments.cache_dir,
        arguments.workers,
        arguments.sarif_directory,
        disposable_docker=arguments.disposable_docker,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
