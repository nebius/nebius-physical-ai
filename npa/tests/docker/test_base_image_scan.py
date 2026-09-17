"""Keep bounded base-image scans aligned with production image preparation."""

import os
from pathlib import Path
import subprocess
import sys
from threading import Barrier, Event, Lock

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
import scan_base_images as scanner  # noqa: E402


INVENTORY = ROOT / "npa/docker/workbench/base-image-security.json"


def _entries() -> list[dict[str, object]]:
    return scanner.load_inventory(INVENTORY)


@pytest.mark.parametrize("entry", _entries(), ids=lambda entry: entry["name"])
def test_scan_target_applies_only_declared_preparation(
    monkeypatch: pytest.MonkeyPatch, entry: dict[str, object]
) -> None:
    """Prepare only inventory entries whose production image does so.

    Args:
        monkeypatch: Isolated subprocess replacement.
        entry: Base-image inventory entry.
    Returns:
        None.
    Raises:
        AssertionError: Preparation or direct-scan routing changes.
    """

    calls = []

    def record(command, **arguments):
        calls.append((command, arguments))
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(scanner.subprocess, "run", record)
    target = scanner.prepare_target(entry)
    command = scanner.preparation_command(entry)
    if command is None:
        assert target == entry["image"] and calls == []
        return
    assert target == f"npa-base-scan:{entry['name']}"
    assert calls == [
        (command, {"input": scanner._PATCH_DOCKERFILE, "text": True, "check": True})
    ]
    assert "FROM ${BASE_IMAGE}" in scanner._PATCH_DOCKERFILE


def test_failed_preparation_never_returns_a_scan_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Fail before scanning when Docker cannot construct the patched base.

    Args:
        monkeypatch: Isolated subprocess failure.
    Returns:
        None.
    Raises:
        AssertionError: A failed preparation is treated as scannable.
    """

    entry = next(item for item in _entries() if item["upgrade_os"] is True)

    def fail(command, **arguments):
        raise subprocess.CalledProcessError(17, command)

    monkeypatch.setattr(scanner.subprocess, "run", fail)
    with pytest.raises(subprocess.CalledProcessError) as error:
        scanner.prepare_target(entry)
    assert error.value.returncode == 17


def _record_package_commands(tmp_path: Path, failed_command: str) -> None:
    for name in ("apt-get", "dpkg", "rm"):
        executable = tmp_path / name
        executable.write_text(
            f"#!{sys.executable}\nimport pathlib, sys\n"
            "with pathlib.Path('commands').open('a') as stream:\n"
            "    stream.write(pathlib.Path(sys.argv[0]).name + ' ' + ' '.join(sys.argv[1:]) + '\\n')\n"
            f"raise SystemExit(23 if {failed_command!r} in [arg.lstrip('-') for arg in sys.argv[1:]] else 0)\n"
        )
        executable.chmod(0o700)


@pytest.mark.parametrize("failed_command", ["update", "upgrade", "purge"])
def test_os_preparation_stops_after_package_manager_failure(
    tmp_path: Path, failed_command: str
) -> None:
    """Keep chained OS remediation fail closed.

    Args:
        tmp_path: Isolated executable directory.
        failed_command: Synthetic command that exits unsuccessfully.
    Returns:
        None.
    Raises:
        AssertionError: A later remediation runs after an earlier failure.
    """

    command = scanner._PATCH_DOCKERFILE.split("RUN ", 1)[1].replace("\\\n", "")
    _record_package_commands(tmp_path, failed_command)
    result = subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PATH": str(tmp_path),
            "UPGRADE_OS": "true",
            "PURGE_LINUX_LIBC_DEV": "true",
        },
        check=False,
    )
    assert result.returncode == 23, result.stderr
    commands = (tmp_path / "commands").read_text().splitlines()
    expected = [
        "apt-get update",
        "apt-get upgrade -y --no-install-recommends",
        "dpkg --purge --force-depends linux-libc-dev",
    ]
    end = ["update", "upgrade", "purge"].index(failed_command) + 1
    assert commands == expected[:end]


def test_python_scan_matches_fiftyones_pinned_and_upgraded_base() -> None:
    """Track the exact base and OS upgrade used by FiftyOne.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Base-image preparation drifts from the Dockerfile.
    """

    dockerfile = (ROOT / "npa/docker/workbench/fiftyone/Dockerfile").read_text()
    base = next(
        line.removeprefix("FROM ")
        for line in dockerfile.splitlines()
        if line.startswith("FROM ")
    )
    entry = next(item for item in _entries() if item["name"] == "python-3-11-slim-trixie")
    assert entry["image"] == base and "@sha256:" in base
    assert entry["upgrade_os"] is True
    for command in ("apt-get update", "apt-get upgrade", "rm -rf /var/lib/apt/lists/*"):
        assert command in dockerfile and command in scanner._PATCH_DOCKERFILE


def test_base_cve_gate_is_blocking_and_os_scoped() -> None:
    """Retain fixed-CRITICAL OS vulnerability enforcement.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: The Trivy command becomes advisory or changes scope.
    """

    command = scanner._trivy_command("synthetic@sha256:digest", Path("cache"), sarif=None)
    assert command[command.index("--exit-code") + 1] == "1"
    assert command[command.index("--severity") + 1] == "CRITICAL"
    assert command[command.index("--vuln-type") + 1] == "os"
    assert "--ignore-unfixed" in command


def _record_parallel_scans(cache: Path, scan_caches: list[Path]):
    """Keep the first two fake scans active together to exercise both workers."""

    first_pair = Barrier(2)
    lock = Lock()

    def record(command, **arguments):
        if "--download-db-only" in command:
            database = cache / "db"
            database.mkdir()
            (database / "trivy.db").write_text("verified database")
        else:
            worker_cache = Path(command[command.index("--cache-dir") + 1])
            database = worker_cache / "db/trivy.db"
            assert database.read_text() == "verified database"
            with lock:
                scan_caches.append(worker_cache)
                scan_index = len(scan_caches)
            if scan_index <= first_pair.parties:
                first_pair.wait(timeout=2)
        return subprocess.CompletedProcess(command, 0)

    return record


def test_parallel_scans_use_worker_private_trivy_caches(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Share the downloaded database without sharing Trivy's mutable cache.

    Args:
        monkeypatch: Isolated subprocess replacement.
        tmp_path: Temporary Trivy cache root.
    Returns:
        None.
    Raises:
        AssertionError: Parallel scans share mutable cache state.
    """

    cache = tmp_path / "trivy"
    scan_caches: list[Path] = []
    monkeypatch.setattr(scanner.subprocess, "run", _record_parallel_scans(cache, scan_caches))
    entries = [
        {
            "name": f"base-{index}",
            "image": f"example.invalid/base@sha256:{index}",
            "purge_linux_libc_dev": False,
            "upgrade_os": False,
        }
        for index in range(3)
    ]
    scanner.scan_inventory(entries, cache, workers=2, sarif_directory=None)

    assert len(scan_caches) == 3
    assert cache not in scan_caches
    assert len(set(scan_caches)) == 2


def test_parallel_scans_dynamically_claim_the_next_entry(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Let an available worker claim work instead of waiting on a static lane.

    Args:
        monkeypatch: Isolated scanner and subprocess replacements.
        tmp_path: Temporary Trivy cache root.
    Returns:
        None.
    Raises:
        AssertionError: The third entry cannot start while the first is blocked.
    """

    cache = tmp_path / "trivy"
    first_pair = Barrier(2)
    third_started = Event()
    lock = Lock()
    entry_caches: dict[str, Path] = {}

    def download(command, **arguments):
        database = cache / "db"
        database.mkdir()
        (database / "trivy.db").write_text("verified database")
        return subprocess.CompletedProcess(command, 0)

    def scan(entry, worker_cache, sarif_directory):
        name = str(entry["name"])
        with lock:
            entry_caches[name] = worker_cache
        if name in {"base-0", "base-1"}:
            first_pair.wait(timeout=2)
        if name == "base-0":
            assert third_started.wait(timeout=2)
        elif name == "base-2":
            third_started.set()
        return 0

    monkeypatch.setattr(scanner.subprocess, "run", download)
    monkeypatch.setattr(scanner, "scan_entry", scan)
    entries = [
        {
            "name": f"base-{index}",
            "image": f"example.invalid/base@sha256:{index}",
            "purge_linux_libc_dev": False,
            "upgrade_os": False,
        }
        for index in range(3)
    ]
    scanner.scan_inventory(entries, cache, workers=2, sarif_directory=None)

    assert entry_caches["base-1"] == entry_caches["base-2"]
    assert entry_caches["base-0"] != entry_caches["base-2"]
