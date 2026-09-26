"""Exercise scan-owned disk reclamation without touching shared Docker state."""

import json
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
import scan_base_images as scanner  # noqa: E402


@pytest.fixture
def cache(tmp_path: Path) -> Path:
    """Provide an existing database alongside unrelated caller-owned files."""
    database = tmp_path / "cache/db"
    database.mkdir(parents=True)
    (database / "trivy.db").write_text("shared verified database")
    (database.parent / "foreign-cache").write_text("keep")
    return database.parent


def _entry(name="base", *, patched=True):
    return {
        "name": name,
        "image": "example.invalid/public@sha256:" + "a" * 64,
        "purge_linux_libc_dev": False,
        "upgrade_os": patched,
    }


class DockerState:
    """Model independent Docker state and reject any broad deletion command."""

    def __init__(self, failure="", scan_status=0):
        self.failure = failure
        self.scan_status = scan_status
        self.builders = {"unrelated-builder"}
        self.calls = []
        self.scan_roots = []

    def __call__(self, command, **arguments):
        self.calls.append(command)
        if command[:3] == ["docker", "buildx", "create"]:
            self._create(command, arguments)
        elif command[:3] == ["docker", "buildx", "build"]:
            self._build(command, arguments)
        elif command[:3] == ["docker", "buildx", "rm"]:
            if self.failure == "cleanup-exec":
                raise OSError("Docker executable is unavailable")
            if self.failure == "cleanup":
                return subprocess.CompletedProcess(command, 19)
            self.builders.remove(command[-1])
        elif command[:2] == ["trivy", "image"]:
            return self._scan(command, arguments)
        else:
            pytest.fail(f"unexpected command: {command}")
        return subprocess.CompletedProcess(command, 0)

    def _create(self, command, arguments):
        if self.failure == "create":
            raise subprocess.CalledProcessError(17, command)
        name = command[command.index("--name") + 1]
        assert name not in self.builders
        self.builders.add(name)
        config = Path(arguments["env"]["BUILDX_CONFIG"])
        config.mkdir()
        (config / "instance").write_text(name)

    def _build(self, command, arguments):
        assert command[command.index("--builder") + 1] in self.builders
        context = Path(command[-1])
        assert list(context.iterdir()) == []
        assert context.stat().st_mode & 0o777 == 0o700
        archive = Path(command[command.index("--output") + 1].split("dest=", 1)[1])
        archive.write_bytes(b"partial archive")
        if self.failure == "build":
            raise subprocess.CalledProcessError(18, command)

    def _scan(self, command, arguments):
        assert self.builders == {"unrelated-builder"}
        cache = Path(command[command.index("--cache-dir") + 1])
        assert (cache / "db/trivy.db").read_text() == "shared verified database"
        temporary = Path(arguments["env"]["TMPDIR"])
        assert cache.parent == temporary.parent
        (cache / "artifact-cache").write_bytes(b"mutable cache")
        (temporary / "layer-export").write_bytes(b"temporary layer")
        self.scan_roots.append(cache.parent)
        if "--output" in command:
            report = Path(command[command.index("--output") + 1])
            report.write_text('{"version":"2.1.0"}')
            if self.failure == "sarif":
                raise subprocess.CalledProcessError(20, command)
        return subprocess.CompletedProcess(command, self.scan_status)


def _assert_foreign_files(cache):
    assert (cache / "foreign-cache").read_text() == "keep"
    assert (cache / "db/trivy.db").read_text() == "shared verified database"


@pytest.mark.parametrize("scan_status", [0, 1, 42])
def test_each_entry_reclaims_archive_and_cache_before_the_next_scan(
    monkeypatch, cache, scan_status
):
    """Preserve scan failures and foreign state across repeated owned cleanups."""
    state = DockerState(scan_status=scan_status)
    monkeypatch.setattr(scanner.subprocess, "run", state)
    for name in ("first", "second"):
        assert scanner.scan_entry(_entry(name), cache, None) == scan_status
        assert not list(cache.glob("entry-*"))
        _assert_foreign_files(cache)
    assert len(set(state.scan_roots)) == 2
    assert state.builders == {"unrelated-builder"}
    assert all(not root.exists() for root in state.scan_roots)
    assert not any("prune" in command for command in state.calls)


@pytest.mark.parametrize("failure", ["create", "build", "sarif"])
def test_failure_reclaims_only_created_resources(monkeypatch, cache, tmp_path, failure):
    """Never delete a pre-existing builder after create failure or retain exports."""
    state = DockerState(failure=failure)
    monkeypatch.setattr(scanner.subprocess, "run", state)
    with pytest.raises(subprocess.CalledProcessError):
        scanner.scan_entry(_entry(), cache, tmp_path)
    assert state.builders == {"unrelated-builder"}
    assert not list(cache.glob("entry-*"))
    if failure == "create":
        assert not any(
            command[:3] == ["docker", "buildx", "rm"] for command in state.calls
        )
    _assert_foreign_files(cache)


@pytest.mark.parametrize("failure", ["cleanup", "cleanup-exec"])
def test_failed_builder_cleanup_fails_closed_and_retains_recovery_receipt(
    monkeypatch, cache, failure
):
    """A failed delete cannot be hidden by a clean scan or lost ownership metadata."""
    state = DockerState(failure=failure)
    monkeypatch.setattr(scanner.subprocess, "run", state)
    with pytest.raises(scanner._BuilderCleanupError) as error:
        scanner.scan_entry(_entry(), cache, None)
    if failure == "cleanup-exec":
        assert isinstance(error.value.__cause__, OSError)
    assert state.scan_roots == []
    receipts = list(cache.glob("entry-*/builder.json"))
    assert len(receipts) == 1
    name = json.loads(receipts[0].read_text())["name"]
    assert name in state.builders and name != "unrelated-builder"
    assert (receipts[0].parent / "buildx/instance").read_text() == name
    assert (receipts[0].parent / "image.tar").exists()
    _assert_foreign_files(cache)


def test_sarif_survives_owned_cleanup_without_overriding_blocking_result(
    monkeypatch, cache, tmp_path
):
    """Keep externally requested reports and the failing table-scan status."""
    state = DockerState(scan_status=1)
    monkeypatch.setattr(scanner.subprocess, "run", state)
    assert scanner.scan_entry(_entry(), cache, tmp_path) == 1
    assert (tmp_path / "trivy-base.sarif").is_file()
    assert not list(cache.glob("entry-*"))


def test_unmodified_base_uses_exact_remote_digest_without_local_builder(
    monkeypatch, cache
):
    """Prevent direct scans from pulling images into a shared local store."""
    state = DockerState()
    monkeypatch.setattr(scanner.subprocess, "run", state)
    entry = _entry(patched=False)
    assert scanner.scan_entry(entry, cache, None) == 0
    assert len(state.calls) == 1
    command = state.calls[0]
    assert command[-1] == entry["image"]
    assert command[command.index("--image-src") + 1] == "remote"
    assert not list(cache.glob("entry-*"))


def test_cleanup_does_not_follow_scanner_created_symlinks(monkeypatch, cache, tmp_path):
    """A cache symlink must not broaden cleanup into unrelated caller files."""
    protected = tmp_path / "foreign-data"
    protected.mkdir()
    (protected / "receipt").write_text("keep")
    state = DockerState()

    def run(command, **arguments):
        if command[:2] == ["trivy", "image"]:
            cache_path = Path(command[command.index("--cache-dir") + 1])
            (cache_path / "outside").symlink_to(protected, target_is_directory=True)
        return state(command, **arguments)

    monkeypatch.setattr(scanner.subprocess, "run", run)
    assert scanner.scan_entry(_entry(), cache, None) == 0
    assert (protected / "receipt").read_text() == "keep"


@pytest.mark.parametrize("failure", ["cleanup", "cleanup-exec"])
def test_inventory_keeps_owned_builder_recovery_metadata(monkeypatch, cache, failure):
    """Outer inventory cleanup cannot erase a failed builder deletion receipt."""
    state = DockerState(failure=failure)

    def run(command, **arguments):
        if "--download-db-only" in command:
            return subprocess.CompletedProcess(command, 0)
        return state(command, **arguments)

    monkeypatch.setattr(scanner.subprocess, "run", run)
    with pytest.raises(scanner._BuilderCleanupError):
        scanner.scan_inventory([_entry()], cache, 1, None)
    receipts = list(cache.glob("scan-workers-*/*/entry-*/builder.json"))
    assert len(receipts) == 1
    assert json.loads(receipts[0].read_text())["name"] in state.builders
    _assert_foreign_files(cache)


def test_matrix_covers_each_validated_entry_without_starting_a_scan(
    monkeypatch, tmp_path, capsys
):
    """Planning emits the whole inventory and performs no Docker or Trivy call."""
    inventory = tmp_path / "inventory.json"
    entries = [_entry("first"), _entry("second", patched=False)]
    inventory.write_text(json.dumps(entries))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan",
            "--inventory",
            str(inventory),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--matrix",
        ],
    )
    monkeypatch.setattr(
        scanner.subprocess,
        "run",
        lambda *args, **kwargs: pytest.fail("planning launched a process"),
    )
    assert scanner.main() == 0
    assert json.loads(capsys.readouterr().out) == {"entry": ["first", "second"]}
    assert not (tmp_path / "cache").exists()


@pytest.mark.parametrize(
    "name,accepted",
    [("first", True), ("omitted", False), ("../first", False), ("", False)],
)
def test_single_entry_selection_fails_on_unknown_names(
    monkeypatch, tmp_path, name, accepted
):
    """A missing or hostile selector cannot silently turn coverage into success."""
    inventory = tmp_path / "inventory.json"
    entries = [_entry("first"), _entry("second", patched=False)]
    inventory.write_text(json.dumps(entries))
    calls = []
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "scan",
            "--inventory",
            str(inventory),
            "--cache-dir",
            str(tmp_path / "cache"),
            "--entry-name",
            name,
        ],
    )
    monkeypatch.setattr(
        scanner, "scan_inventory", lambda selected, *args: calls.append(selected)
    )
    if accepted:
        assert scanner.main() == 0
        assert calls == [[entries[0]]]
    else:
        with pytest.raises(ValueError, match="absent"):
            scanner.main()
        assert calls == []
