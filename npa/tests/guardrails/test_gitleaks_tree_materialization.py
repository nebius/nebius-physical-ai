"""Verify Gitleaks scans exact committed blobs without checkout transformations."""

from __future__ import annotations

import base64
import gzip
import importlib.util
import io
import json
import os
import shutil
import subprocess
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "scripts" / "materialize_gitleaks_tree.py"
SPEC = importlib.util.spec_from_file_location("materialize_gitleaks_tree", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
materializer = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(materializer)
RUNNER = ROOT / "scripts" / "run_gitleaks_tree_scan.py"
RUNNER_SPEC = importlib.util.spec_from_file_location("run_gitleaks_tree_scan", RUNNER)
assert RUNNER_SPEC is not None and RUNNER_SPEC.loader is not None
tree_scan = importlib.util.module_from_spec(RUNNER_SPEC)
RUNNER_SPEC.loader.exec_module(tree_scan)

_CONTROLLED_DECODE_PROBE = (
    "assert '--ignore-gitleaks-allow' in sys.argv\n"
    "assert sys.argv[sys.argv.index('--max-archive-depth') + 1] == '3'\n"
    "decode_depth = sys.argv[sys.argv.index('--max-decode-depth') + 1]\n"
    "assert decode_depth == '5'\n"
    "probe = pathlib.Path(sys.argv[sys.argv.index('--config') + 1]).name "
    "== 'decode-depth-probe.toml'\n"
    "if probe:\n"
    "    tags = ['decode-depth:5'] if pathlib.Path.cwd().name "
    "== 'decode-depth-control' else []\n"
    "    name = 'past-limit.txt' if tags else 'probe.txt'\n"
    "    field = 'Se' + 'cret'\n"
    "    report.write_text(json.dumps([{'RuleID': "
    "'npa-decode-depth-probe-base64', field: 'c2FmZSBib3VuZGFyeS4=', "
    "'Match': 'c2FmZSBib3VuZGFyeS4=', 'Tags': tags, 'File': name}]))\n"
    "    raise SystemExit(1)\n"
)
_CONTROLLED_GITLEAKS_VERSION = (
    "if sys.argv[1:] == ['version']:\n"
    f"    print({tree_scan._GITLEAKS_VERSION!r})\n"
    "    raise SystemExit(0)\n"
)


def _git(repo: Path, *arguments: str) -> str:
    """Run one synthetic-repository Git command."""
    return subprocess.run(
        ["git", "-C", str(repo), *arguments],
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _repo(tmp_path: Path) -> Path:
    """Create an isolated repository with deterministic author identity."""
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-q")
    _git(repo, "config", "user.email", "npa-test@example.invalid")
    _git(repo, "config", "user.name", "NPA synthetic control")
    return repo


def _commit(repo: Path, message: str) -> str:
    """Commit every synthetic path and return the new commit."""
    _git(repo, "add", "--all")
    _git(repo, "commit", "-qm", message)
    return _git(repo, "rev-parse", "HEAD")


def _runner_arguments(
    repo: Path,
    candidate: str,
    base: str,
    scanner: str,
) -> list[str]:
    """Build one exact-identity complete-scan invocation."""
    return [
        "--repo-root",
        str(repo),
        "--revision",
        candidate,
        "--history-base",
        base,
        "--policy-revision",
        candidate,
        "--gitleaks-bin",
        scanner,
    ]


def _corrupt_loose_object(
    repo: Path,
    object_id: str,
    before: bytes,
    after: bytes,
) -> None:
    """Replace bytes in one loose object without changing its object filename."""
    assert len(before) == len(after)
    object_path = repo / ".git" / "objects" / object_id[:2] / object_id[2:]
    decoded = zlib.decompress(object_path.read_bytes())
    assert decoded.count(before) == 1
    object_path.chmod(0o600)
    object_path.write_bytes(zlib.compress(decoded.replace(before, after, 1)))


def _materialize(repo: Path, destination: Path) -> int:
    """Invoke the materializer through its public command boundary."""
    destination.mkdir()
    return materializer.main(
        [
            "--repo-root",
            str(repo),
            "--output-dir",
            str(destination),
            "--revision",
            "HEAD",
        ]
    )


def test_materializes_regular_and_symlink_blob_bytes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Preserve exact bytes for regular, unusual-name, and symlink entries.

    Args:
        tmp_path: Isolated repository and snapshot parent.
        capsys: Captured materializer receipt.
    Returns:
        None.
    Raises:
        AssertionError: A committed blob is omitted or transformed.
    """
    repo = _repo(tmp_path)
    regular = b"api_key = committed-candidate-bytes\n"
    unusual = b"newline and leading-option path\n"
    link_target = "../target with spaces\nand newline"
    (repo / "regular.txt").write_bytes(regular)
    (repo / "--option\nname").write_bytes(unusual)
    os.symlink(link_target, repo / "candidate-link")
    commit = _commit(repo, "candidate blobs")

    destination = tmp_path / "snapshot"
    assert _materialize(repo, destination) == 0
    receipt = json.loads(capsys.readouterr().out)
    assert receipt["commit"] == commit
    assert receipt["files"] == 3
    assert receipt["symlink_blobs"] == 1
    assert (destination / "regular.txt").read_bytes() == regular
    assert (destination / "--option\nname").read_bytes() == unusual
    assert not (destination / "candidate-link").is_symlink()
    assert (destination / "candidate-link").read_bytes() == os.fsencode(link_target)


def test_reads_committed_blobs_not_transformed_worktree_bytes(tmp_path: Path) -> None:
    """Ignore checkout filters and dirty bytes that can conceal a committed secret.

    Args:
        tmp_path: Isolated repository and snapshot parent.
    Returns:
        None.
    Raises:
        AssertionError: The materializer reads the checkout instead of Git objects.
    """
    repo = _repo(tmp_path)
    committed = b'api_key = "Qw7eR9tY2uI4oP6aS8dF0gH1jK3lZ5xC"\n'
    path = repo / "settings.py"
    path.write_bytes(committed)
    _commit(repo, "committed source")
    path.write_bytes(committed.decode().encode("utf-16le"))

    destination = tmp_path / "snapshot"
    assert _materialize(repo, destination) == 0
    assert (destination / "settings.py").read_bytes() == committed


def test_ignores_git_replace_for_tree_and_config_bytes(tmp_path: Path) -> None:
    """Bind materialized source and scanner policy to the requested commit."""
    repo = _repo(tmp_path)
    original_config = b"[extend]\nuseDefault = true\n"
    original_source = b"ORIGINAL\n"
    (repo / ".gitleaks.toml").write_bytes(original_config)
    (repo / "source.txt").write_bytes(original_source)
    original_commit = _commit(repo, "original")
    original_tree = _git(repo, "rev-parse", f"{original_commit}^{{tree}}")

    (repo / ".gitleaks.toml").write_text(
        "[allowlists]\npaths = ['''.*''']\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_bytes(b"REPLACED\n")
    replacement_commit = _commit(repo, "replacement")
    _git(repo, "replace", original_commit, replacement_commit)

    destination = tmp_path / "snapshot"
    destination.mkdir()
    receipt = materializer.materialize(
        repo,
        destination,
        original_commit,
    )
    assert receipt["commit"] == original_commit
    assert receipt["tree"] == original_tree
    assert (destination / "source.txt").read_bytes() == original_source
    assert (
        tree_scan._config_blob(repo, original_tree, ".gitleaks.toml") == original_config
    )


@pytest.mark.parametrize(
    ("kind", "before", "after"),
    [
        ("commit", b"\nsource\n", b"\ntamper\n"),
        ("tree", b"tracked.txt\0", b"mutated.txt\0"),
        ("blob", b"SAFE\n", b"EVIL\n"),
    ],
)
def test_rejects_object_bytes_stored_under_the_wrong_id(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    kind: str,
    before: bytes,
    after: bytes,
) -> None:
    """Recompute commit, tree, and blob IDs instead of trusting object filenames."""
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_bytes(b"SAFE\n")
    commit = _commit(repo, "source")
    object_ids = {
        "commit": commit,
        "tree": _git(repo, "rev-parse", f"{commit}^{{tree}}"),
        "blob": _git(repo, "rev-parse", f"{commit}:tracked.txt"),
    }
    _corrupt_loose_object(repo, object_ids[kind], before, after)

    destination = tmp_path / "snapshot"
    assert _materialize(repo, destination) == 1
    assert "object" in capsys.readouterr().err.lower()


def test_rejects_zero_padded_tree_modes(tmp_path: Path) -> None:
    """Accept only canonical tree encodings that strict Git integrity accepts."""
    repo = _repo(tmp_path)
    empty_tree = subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-t", "tree", "-w", "--stdin"],
        input=b"",
        check=True,
        capture_output=True,
    ).stdout.strip()
    malformed = b"040000 child\0" + bytes.fromhex(empty_tree.decode("ascii"))
    malformed_tree = subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "hash-object",
            "--literally",
            "-t",
            "tree",
            "-w",
            "--stdin",
        ],
        input=malformed,
        check=True,
        capture_output=True,
    ).stdout.strip()

    with pytest.raises(materializer.SnapshotError, match="Unsupported Git tree entry"):
        materializer._tree_entries(repo, malformed_tree, b"sha1")


def test_blob_process_cleanup_reaps_child_and_attempts_every_close() -> None:
    """One cleanup failure must not strand the child or skip later streams."""
    events: list[str] = []

    class CleanupStream:
        def __init__(self, name: str, fail: bool = False) -> None:
            self.name = name
            self.fail = fail
            self.closed = False

        def close(self) -> None:
            events.append(f"close:{self.name}")
            self.closed = True
            if self.fail:
                raise OSError(f"{self.name} close failed")

    class CleanupProcess:
        def __init__(self) -> None:
            self.stdin = CleanupStream("stdin", fail=True)
            self.stdout = CleanupStream("stdout")
            self.stderr = CleanupStream("stderr")

        @staticmethod
        def poll() -> None:
            return None

        @staticmethod
        def kill() -> None:
            events.append("kill")

        @staticmethod
        def wait() -> int:
            events.append("wait")
            return -9

    first = CleanupStream("first", fail=True)
    second = CleanupStream("second")
    errors = materializer._cleanup_blob_process(
        CleanupProcess(),
        (first, second),
    )

    assert len(errors) == 2
    assert events == [
        "close:stdin",
        "kill",
        "wait",
        "close:stdout",
        "close:stderr",
        "close:first",
        "close:second",
    ]
    assert second.closed


def test_history_walk_uses_verified_parent_headers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not let commit-graph caches replace authenticated parent traversal."""
    repo = _repo(tmp_path)
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("intermediate\n", encoding="utf-8")
    intermediate = _commit(repo, "intermediate")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")
    _git(repo, "commit-graph", "write", "--reachable")
    _git(repo, "config", "core.commitGraph", "true")

    original_git = materializer._git

    def reject_cached_traversal(repo_path: Path, *arguments: str) -> bytes:
        assert arguments[0] not in {"merge-base", "rev-list"}
        return original_git(repo_path, *arguments)

    monkeypatch.setattr(materializer, "_git", reject_cached_traversal)
    assert materializer.history_commits(repo, base, candidate) == [
        intermediate,
        candidate,
    ]


def _assert_isolated_scan_wrappers(
    repo: Path,
    destination: Path,
    binary: bytes,
    receipt: dict[str, object],
) -> None:
    """Check raw bytes, inert policy placement, and isolated wrappers."""
    assert receipt["scan_view"] is True
    assert receipt["scan_prefix_bytes"] == len(materializer.SCAN_PREFIX)
    assert receipt["raw_scan_files"] == 2
    assert receipt["wrapper_scan_files"] == 2
    assert receipt["scan_files"] == 4
    assert receipt["scan_bytes"] > receipt["bytes"]
    assert (destination / "artifact.zip").read_bytes() == binary
    policy_blob = _git(repo, "rev-parse", "HEAD:.gitleaks.toml")
    assert not (destination / ".gitleaks.toml").exists()
    assert (destination / ".npa-gitleaks-policy" / f"{policy_blob}.toml").read_text(
        encoding="utf-8"
    ) == "[extend]\nuseDefault = true\n"
    wrappers = list(
        (destination / os.fsdecode(materializer.SCAN_NON_TEST_ROOT)).glob("*.txt")
    )
    assert len(wrappers) == 2
    artifact_wrappers = [
        path
        for path in wrappers
        if b"artifact.zip" in path.read_bytes() and binary in path.read_bytes()
    ]
    assert len(artifact_wrappers) == 1
    assert artifact_wrappers[0].read_bytes().startswith(materializer.SCAN_PREFIX)
    assert not (destination / os.fsdecode(materializer.SCAN_TEST_ROOT)).exists()


def test_scan_view_keeps_binary_bytes_reachable_to_gitleaks(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Prefix and rename scan targets without changing their committed payload."""
    repo = _repo(tmp_path)
    binary = b"PK\x03\x04" + bytes(range(256)) + (b"\0" * 4096)
    (repo / "artifact.zip").write_bytes(binary)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    _commit(repo, "binary source")
    destination = tmp_path / "scan-view"
    destination.mkdir()
    arguments = [
        "--repo-root",
        str(repo),
        "--output-dir",
        str(destination),
        "--revision",
        "HEAD",
        "--scan-view",
    ]

    assert materializer.main(arguments) == 0

    receipt = json.loads(capsys.readouterr().out)
    _assert_isolated_scan_wrappers(repo, destination, binary, receipt)


def test_history_view_deduplicates_blob_bytes_per_allowlist_context(
    tmp_path: Path,
) -> None:
    """Retain one shared blob in both test and non-test policy contexts."""
    repo = _repo(tmp_path)
    shared = b'api_key = "shared historical bytes"\n'
    test_path = repo / "npa" / "tests" / "shared.py"
    test_path.parent.mkdir(parents=True)
    test_path.write_bytes(shared)
    base = _commit(repo, "test use")
    source_path = repo / "npa" / "src" / "shared.py"
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(shared)
    intermediate = _commit(repo, "non-test use")
    destination = tmp_path / "historical"
    destination.mkdir()

    receipt = materializer.materialize_history_scan(
        repo,
        destination,
        [base, intermediate],
    )

    assert receipt["commits"] == [base, intermediate]
    assert receipt["unique_blob_contexts"] == 2
    test_wrappers = (destination / os.fsdecode(materializer.SCAN_TEST_ROOT)).glob(
        "*.txt"
    )
    non_test_wrappers = (
        destination / os.fsdecode(materializer.SCAN_NON_TEST_ROOT)
    ).glob("*.txt")
    assert any(shared in path.read_bytes() for path in test_wrappers)
    assert any(shared in path.read_bytes() for path in non_test_wrappers)


def test_history_view_preserves_compound_archive_basename(tmp_path: Path) -> None:
    """Keep the exact basename required by Gitleaks archive identification."""
    repo = _repo(tmp_path)
    archive_path = repo / "credential.tar.gz"
    with tarfile.open(archive_path, "w:gz") as archive:
        payload = b"historical archive payload\n"
        metadata = tarfile.TarInfo("credential.txt")
        metadata.size = len(payload)
        archive.addfile(metadata, io.BytesIO(payload))
    revision = _commit(repo, "compound archive")
    expected = archive_path.read_bytes()
    destination = tmp_path / "historical"
    destination.mkdir()

    materializer.materialize_history_scan(repo, destination, [revision])

    matches = list(destination.rglob("credential.tar.gz"))
    assert len(matches) == 1
    assert matches[0].read_bytes() == expected


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_scan_view_never_joins_rule_matches_across_unrelated_blobs(
    tmp_path: Path,
) -> None:
    """Keep a BEGIN in one blob from pairing with an END in another blob."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    begin_marker = "-----BEGIN " + "PRIVATE KEY-----\n"
    end_marker = "-----END " + "PRIVATE KEY-----\n"
    (repo / "begin.txt").write_text(
        begin_marker + ("A" * 80) + "\n",
        encoding="utf-8",
    )
    (repo / "end.txt").write_text(
        end_marker,
        encoding="utf-8",
    )
    revision = _commit(repo, "split private-key markers")
    destination = tmp_path / "scan-view"
    destination.mkdir()
    materializer.materialize(repo, destination, revision, scan_view=True)
    ignore = tmp_path / "policy.ignore"
    ignore.write_text("", encoding="utf-8")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    result, findings = tree_scan._run_gitleaks(
        scanner,
        destination,
        repo / ".gitleaks.toml",
        ignore,
        tmp_path / "report.json",
    )

    assert result.returncode == 0
    assert not findings


def test_scan_environment_bounds_gitleaks_parallelism(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Prevent host CPU count from multiplying per-file scanner memory."""
    monkeypatch.setenv("GOMAXPROCS", "999")

    assert tree_scan._scan_environment()["GOMAXPROCS"] == str(
        tree_scan._GITLEAKS_MAX_PROCS
    )


@pytest.mark.parametrize(
    "variable",
    ["BASH_ENV", "ENV", "LD_LIBRARY_PATH", "LD_PRELOAD", "PYTHONPATH", "TMPDIR"],
)
def test_scan_environment_drops_process_injection_variables(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    """Do not let caller-controlled process hooks alter Git or Gitleaks."""
    monkeypatch.setenv(variable, "/attacker-controlled")

    assert variable not in tree_scan._scan_environment()


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_no_git_scan_preserves_relative_policy_path_context(tmp_path: Path) -> None:
    """Keep test allowlists scoped while scanning aggregate paths from their root."""
    source = tmp_path / "source"
    synthetic_identifier = "e" + "00" + ("a" * 12)
    payload = f"account = {synthetic_identifier}\n"
    test_path = source / "npa" / "tests" / "fixture.txt"
    test_path.parent.mkdir(parents=True)
    test_path.write_text(payload, encoding="utf-8")
    non_test_path = source / "src" / "value.txt"
    non_test_path.parent.mkdir()
    non_test_path.write_text(payload, encoding="utf-8")
    ignore = tmp_path / "policy.ignore"
    ignore.write_text("", encoding="utf-8")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    result, findings = tree_scan._run_gitleaks(
        scanner,
        source,
        ROOT / ".gitleaks.toml",
        ignore,
        tmp_path / "report.json",
    )

    matching = [
        finding for finding in findings if finding.get("RuleID") == "nebius-generic-id"
    ]
    assert result.returncode == 1
    assert len(matching) == 1
    assert matching[0]["File"] == "src/value.txt"


@pytest.mark.parametrize("history_override", ["shallow", "grafts"])
def test_runner_rejects_truncated_or_rewritten_history(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    history_override: str,
) -> None:
    """Fail closed when repository metadata can omit or rewrite range commits."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")
    if history_override == "shallow":
        (repo / ".git" / "shallow").write_text(f"{base}\n", encoding="ascii")
    else:
        info = repo / ".git" / "info"
        info.mkdir(exist_ok=True)
        (info / "grafts").write_text(f"{candidate} {base}\n", encoding="ascii")

    assert (
        tree_scan.main(_runner_arguments(repo, candidate, base, "/usr/bin/true")) == 1
    )
    assert history_override in capsys.readouterr().err.lower()


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_complete_tree_runner_rejects_late_secret_after_earlier_binary_blob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep a later text secret reachable after an earlier binary aggregate blob."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    (repo / "a-opaque.bin").write_bytes(b"\0" * 4096)
    (repo / "z-secret.py").write_text(
        f'api_key = "{token}"\n',
        encoding="utf-8",
    )
    candidate = _commit(repo, "binary secret control")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "generic-api-key" in output.err


def _encode_base64_layers(content: bytes, layers: int) -> bytes:
    """Encode test content through an exact number of Base64 layers."""
    for _ in range(layers):
        content = base64.b64encode(content)
    return content


def _nested_zip(content: bytes, depth: int) -> bytes:
    """Wrap test content in nested deflated ZIP archives."""
    name = "credential.txt"
    for level in range(depth):
        encoded = io.BytesIO()
        with zipfile.ZipFile(
            encoded,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(name, content)
        content = encoded.getvalue()
        name = f"level-{level}.zip"
    return content


def _zip_member(name: str, content: bytes) -> bytes:
    """Return one deflated ZIP containing the supplied member."""
    encoded = io.BytesIO()
    with zipfile.ZipFile(
        encoded,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr(name, content)
    return encoded.getvalue()


def _write_encoded_archive_carrier(
    repo: Path,
    carrier: str,
    assignment: str,
) -> bool:
    """Write a ZIP hidden behind a supported text encoding."""
    if carrier not in {
        "encoded_zip",
        "double_encoded_zip",
        "hex_zip",
        "percent_zip",
    }:
        return False
    archive = _zip_member("credential.txt", assignment.encode())
    if carrier == "encoded_zip":
        content = base64.b64encode(archive)
    elif carrier == "double_encoded_zip":
        content = base64.b64encode(base64.b64encode(archive))
    elif carrier == "hex_zip":
        content = archive.hex().encode("ascii")
    else:
        content = "".join(f"%{value:02X}" for value in archive).encode("ascii")
    (repo / "credential.txt").write_bytes(content)
    return True


def _write_composed_archive_carrier(
    repo: Path,
    carrier: str,
    assignment: str,
) -> bool:
    """Write an encoded ZIP inside another canonical archive."""
    if carrier not in {"zip_encoded_zip", "gzip_encoded_zip"}:
        return False
    nested = base64.b64encode(_zip_member("credential.txt", assignment.encode()))
    if carrier == "zip_encoded_zip":
        (repo / "credential.zip").write_bytes(_zip_member("carrier.txt", nested))
    else:
        compressor = zlib.compressobj(wbits=31)
        (repo / "credential.gz").write_bytes(
            compressor.compress(nested) + compressor.flush()
        )
    return True


def _write_archive_carrier(repo: Path, carrier: str, assignment: str) -> bool:
    """Write one archive carrier and report whether the name was handled."""
    if carrier == "deep_zip":
        (repo / "credential.zip").write_bytes(_nested_zip(assignment.encode(), 4))
        return True
    if _write_encoded_archive_carrier(repo, carrier, assignment):
        return True
    if _write_composed_archive_carrier(repo, carrier, assignment):
        return True
    encoded = assignment.encode()
    if carrier in {"zip_six_encoded", "tar_gz_six_encoded"}:
        encoded = _encode_base64_layers(encoded, 6)
    if carrier in {"zip", "zip_six_encoded"}:
        with zipfile.ZipFile(
            repo / "credential.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("credential.txt", encoded)
        return True
    if carrier != "tar_gz_six_encoded":
        return False
    metadata = tarfile.TarInfo("credential.txt")
    metadata.size = len(encoded)
    with tarfile.open(repo / "credential.tar.gz", "w:gz") as archive:
        archive.addfile(metadata, io.BytesIO(encoded))
    return True


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize(
    ("path", "carrier"),
    [
        ("credential.jar", "alias"),
        ("credential.zip", "nested_alias"),
        ("credential.zip", "nested_disguised"),
        ("credential.bin", "disguised"),
        ("secret-name", "secret_alias"),
    ],
)
def test_runner_rejects_unscannable_zip_names(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    path: str,
    carrier: str,
) -> None:
    """Reject ZIP bytes whose names prevent Gitleaks archive traversal."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    assignment = f'api_key = "{token}"\n'.encode()
    payload = _zip_member("credential.txt", assignment)
    if carrier == "nested_alias":
        payload = _zip_member("nested.jar", payload)
    elif carrier == "nested_disguised":
        payload = _zip_member("nested.bin", b"self-extracting-prefix\n" + payload)
    target = path
    if carrier == "secret_alias":
        target = f'api_key = "{token}".jar'
    (repo / target).write_bytes(payload)
    candidate = _commit(repo, "unscannable ZIP carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "archive" in output.err.lower()


def _write_text_carrier(repo: Path, carrier: str, assignment: str) -> None:
    """Write one Unicode, encoded, policy, inline, or path carrier."""
    if carrier == "utf16":
        (repo / "credential.txt").write_bytes(assignment.encode("utf-16le"))
    elif carrier == "delayed_utf16":
        prefix = b"A" * (materializer._COPY_CHUNK_BYTES + 3)
        (repo / "credential.bin").write_bytes(prefix + assignment.encode("utf-16le"))
    elif carrier == "utf32":
        (repo / "credential.txt").write_bytes(assignment.encode("utf-32le"))
    elif carrier == "encoded":
        (repo / "credential.txt").write_bytes(
            _encode_base64_layers(assignment.encode(), 1)
        )
    elif carrier == "six_encoded":
        (repo / "credential.txt").write_bytes(
            _encode_base64_layers(assignment.encode(), 6)
        )
    elif carrier == "inline_allow":
        allowed = assignment.rstrip() + "  # gitleaks:allow\n"
        (repo / "credential.py").write_text(allowed, encoding="utf-8")
    elif carrier == "policy_blob":
        (repo / ".gitleaksignore").write_text(assignment, encoding="utf-8")
    elif carrier == "policy_blob_utf16":
        (repo / ".gitleaksignore").write_bytes(assignment.encode("utf-16le"))
    else:
        (repo / assignment.strip()).write_bytes(b"")


def _assert_carrier_failure(carrier: str, token: str, out: str, err: str) -> None:
    """Check redaction and the expected fail-closed boundary."""
    assert token not in out
    assert token not in err
    if carrier == "deep_zip":
        assert "archive" in err
    elif carrier in {
        "encoded_zip",
        "double_encoded_zip",
        "gzip_encoded_zip",
        "hex_zip",
        "percent_zip",
        "zip_encoded_zip",
    }:
        assert "archive" in err
    elif carrier in {"six_encoded", "tar_gz_six_encoded", "zip_six_encoded"}:
        assert "decodable segment" in err
        assert "decode-depth:5" in err
    else:
        assert "generic-api-key" in err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize(
    "carrier",
    [
        "utf16",
        "delayed_utf16",
        "utf32",
        "zip",
        "deep_zip",
        "encoded_zip",
        "double_encoded_zip",
        "zip_encoded_zip",
        "gzip_encoded_zip",
        "hex_zip",
        "percent_zip",
        "encoded",
        "six_encoded",
        "zip_six_encoded",
        "tar_gz_six_encoded",
        "inline_allow",
        "path",
        "policy_blob",
        "policy_blob_utf16",
    ],
)
def test_runner_rejects_nonplaintext_secret_carriers(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    carrier: str,
) -> None:
    """Cover encoded text, archives, and secret-bearing Git path bytes."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    assignment = f'api_key = "{token}"\n'
    if not _write_archive_carrier(repo, carrier, assignment):
        _write_text_carrier(repo, carrier, assignment)
    candidate = _commit(repo, f"{carrier} secret carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    arguments = _runner_arguments(repo, candidate, base, scanner)
    if carrier.startswith("policy_blob"):
        arguments[arguments.index("--policy-revision") + 1] = base
    assert tree_scan.main(arguments) == 1
    output = capsys.readouterr()
    _assert_carrier_failure(carrier, token, output.out, output.err)


@pytest.mark.parametrize(
    "payload",
    [
        b"\xff\xfeA\x00B",
        b"\xff\xfe\x00\x00A\x00\x00",
    ],
)
def test_scan_view_fails_closed_on_malformed_bom_unicode(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    payload: bytes,
) -> None:
    """Do not silently discard malformed text after a strong Unicode signal."""
    repo = _repo(tmp_path)
    (repo / "credential.bin").write_bytes(payload)
    _commit(repo, "malformed unicode")
    destination = tmp_path / "scan-view"
    destination.mkdir()

    assert (
        materializer.main(
            [
                "--repo-root",
                str(repo),
                "--output-dir",
                str(destination),
                "--revision",
                "HEAD",
                "--scan-view",
            ]
        )
        == 1
    )
    assert "malformed" in capsys.readouterr().err.lower()


def test_runner_requires_an_explicit_policy_revision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Prevent an omitted trust input from silently selecting candidate policy."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    candidate = _commit(repo, "candidate")
    arguments = _runner_arguments(repo, candidate, "", "/usr/bin/true")
    policy_index = arguments.index("--policy-revision")
    del arguments[policy_index : policy_index + 2]

    with pytest.raises(SystemExit, match="2"):
        tree_scan.main(arguments)
    assert "--policy-revision" in capsys.readouterr().err


def test_runner_rejects_candidate_as_its_own_history_base(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep candidate commit metadata inside the native history scan."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    candidate = _commit(repo, "candidate")

    arguments = _runner_arguments(repo, candidate, candidate, "/usr/bin/true")
    assert tree_scan.main(arguments) == 1
    assert "must precede the candidate" in capsys.readouterr().err


def test_scan_revalidates_history_metadata_after_workspace_scan(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a repository that becomes shallow during a long-running scan."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")

    def mutate_history(
        _repo_path: Path,
        _root: Path,
        _plan: object,
        _executable: str,
    ) -> dict[str, object]:
        (repo / ".git" / "shallow").write_text(f"{base}\n", encoding="ascii")
        return {}

    monkeypatch.setattr(tree_scan, "_scan_workspace", mutate_history)
    monkeypatch.setattr(
        tree_scan,
        "_verify_gitleaks_identity",
        lambda _executable: "synthetic-sha256",
    )
    monkeypatch.setattr(
        tree_scan,
        "_stage_gitleaks_executable",
        lambda executable, _root: (executable, "synthetic-sha256"),
    )

    with pytest.raises(tree_scan.materializer.SnapshotError, match="Shallow"):
        tree_scan.scan(
            repo,
            candidate,
            base,
            candidate,
            "/usr/bin/true",
            ".gitleaks.toml",
        )


@pytest.mark.parametrize(
    ("kind", "value", "expected"),
    [
        ("base64", "c2FmZSBib3VuZGFyeS4=", True),
        ("base64", "0000000000000000", False),
        ("hex", "41" * 16, True),
        ("hex", "af" * 16, False),
        ("percent", "%25", True),
        ("percent", "%00", False),
        ("unicode", r"\u0061", True),
    ],
)
def test_decode_boundary_probe_validates_real_codec_matches(
    kind: str,
    value: str,
    expected: bool,
) -> None:
    """Keep the bounded probe aligned with Gitleaks' decoder acceptance."""
    rule_id = f"{tree_scan._DECODE_PROBE_PREFIX}{kind}"
    assert tree_scan._probe_match_decodes(rule_id, value) is expected


def test_inert_encoded_archive_exception_is_exact() -> None:
    """Keep the sole public empty-metadata fixture byte-exact and non-secret."""
    archive = bytes.fromhex(
        "1f8b08000000000000ff621805a360148c5800080000ffff2eafb5ef00040000"
    )
    assert archive[3] == 0
    assert gzip.decompress(archive) == b"\0" * 1024
    assert tree_scan._INERT_ENCODED_ARCHIVE_SHA256 == {
        tree_scan.hashlib.sha256(archive).hexdigest()
    }

    tree_scan._reject_encoded_archives(archive.hex().encode("ascii"))
    mutated = archive[:-1] + bytes([archive[-1] ^ 1])
    with pytest.raises(tree_scan._ScanError, match="Encoded archive"):
        tree_scan._reject_encoded_archives(mutated.hex().encode("ascii"))


def _assert_archive_probe_view(
    destination: Path,
    receipt: dict[str, int],
    encoded: bytes,
    wrapper: Path,
    binary: Path,
    archive: Path,
) -> None:
    """Check compact candidate streams and zero-copy archive links."""
    archive_files = sorted(
        path
        for path in (destination / "decode-input-archives").rglob("*")
        if path.is_file()
    )
    candidate_streams = sorted(destination.glob("decode-input-candidates-*.txt"))
    assert receipt["candidate_bytes"] >= len(encoded) + 2
    assert receipt["candidate_matches"] >= 1
    assert receipt["candidate_streams"] == len(candidate_streams)
    assert receipt["files"] == 2
    assert receipt["archive_bytes"] == archive.stat().st_size
    assert receipt["archive_files"] == 1
    assert receipt["archive_streams"] == 1
    assert receipt["wrapper_files"] == 1
    assert {path.name for path in archive_files} == {"credential.zip"}
    archive_inodes = {path.stat().st_ino for path in archive_files}
    assert archive_inodes == {archive.stat().st_ino}
    assert wrapper.stat().st_ino not in archive_inodes
    assert binary.stat().st_ino not in archive_inodes
    assert any(encoded in path.read_bytes() for path in candidate_streams)


def test_decode_probe_view_links_only_archives_without_memory_heavy_aggregates(
    tmp_path: Path,
) -> None:
    """Keep binary spans visible while delegating selected archives to Gitleaks."""
    source = tmp_path / "source"
    source.mkdir()
    wrapper_root = source / os.fsdecode(materializer.SCAN_NON_TEST_ROOT)
    wrapper_root.mkdir()
    wrapper = wrapper_root / "source-wrapper.txt"
    wrapper.write_bytes(materializer.SCAN_PREFIX + (b"A" * 4096))
    encoded = base64.b64encode(b'api_\x6bey = "\x51w7eR9tY2uI4oP6aS8dF0gH1jK3lZ5xC"\n')
    binary = source / "credential.bin"
    binary.write_bytes(b"\0" * 128 + encoded)
    archive = source / "credential.zip"
    with zipfile.ZipFile(
        archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        bundle.writestr("credential.txt", encoded)
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(source, destination)

    _assert_archive_probe_view(
        destination,
        receipt,
        encoded,
        wrapper,
        binary,
        archive,
    )


def test_decode_probe_groups_candidates_into_complete_scanner_fragments(
    tmp_path: Path,
) -> None:
    """Never let a supported encoded match straddle Gitleaks file fragments."""
    source = tmp_path / "source"
    source.mkdir()
    candidates = [
        base64.b64encode(f"safe candidate {index:08d}.".encode())
        for index in range(4000)
    ]
    (source / "candidates.txt").write_bytes(b"\n".join(candidates))
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(source, destination)

    streams = sorted(destination.glob("decode-input-candidates-*.txt"))
    assert receipt["candidate_streams"] == len(streams)
    assert len(streams) > 1
    assert receipt["max_candidate_bytes"] == max(map(len, candidates))
    for stream in streams:
        content = stream.read_bytes()
        assert len(content) <= tree_scan._PROBE_STREAM_MAX_BYTES
        assert content.startswith(materializer.SCAN_PREFIX)
        assert content.endswith(b"\n\n")


def test_decode_probe_rejects_match_larger_than_one_scanner_fragment(
    tmp_path: Path,
) -> None:
    """Fail closed instead of silently splitting a recursive decode candidate."""
    source = tmp_path / "source"
    source.mkdir()
    oversized = base64.b64encode(
        b"safe boundary 0123456789 " * (tree_scan._PROBE_STREAM_MAX_BYTES // 20)
    )
    assert any(character in oversized for character in b"0123456789+/-_")
    (source / "oversized.txt").write_bytes(oversized)

    with pytest.raises(tree_scan._ScanError, match="complete Gitleaks scan fragment"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


@pytest.mark.parametrize("payload_byte", [b"A", b"0"])
def test_decode_probe_skips_oversized_nondecodable_matches(
    tmp_path: Path,
    payload_byte: bytes,
) -> None:
    """Do not reject a large regex span that the pinned codecs cannot decode."""
    source = tmp_path / "source"
    source.mkdir()
    payload = payload_byte * (tree_scan._PROBE_STREAM_MAX_BYTES + 1)
    (source / "payload.bin").write_bytes(payload)

    receipt = tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")

    assert receipt["candidate_matches"] == 0
    assert receipt["candidate_streams"] == 0


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_rejects_binary_secret_deleted_inside_history_range(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scan intermediate trees in one explicit, bounded commit range."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    (repo / "deleted.zip").write_bytes(
        _zip_member(
            "credential.txt",
            f'\napi_key = "{token}"\n'.encode("ascii"),
        )
    )
    _commit(repo, "add binary secret")
    (repo / "deleted.zip").unlink()
    candidate = _commit(repo, "delete binary secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert any(
        label in output.err for label in ("committed-history", "historical-trees")
    )


def test_runner_rejects_an_empty_history_base(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not expose a known-unusable implicit whole-history scan mode."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    candidate = _commit(repo, "candidate")

    assert (
        tree_scan.main(
            _runner_arguments(repo, candidate, "", "/usr/bin/true"),
        )
        == 1
    )
    assert "History base" in capsys.readouterr().err


def test_runner_rejects_commit_allowlist_for_a_materialized_history_tree(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail closed where a no-Git tree cannot preserve commit provenance."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("intermediate\n", encoding="utf-8")
    intermediate = _commit(repo, "intermediate")
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[allowlists]]\n"
        f"commits = ['''{intermediate}''']\n",
        encoding="utf-8",
    )
    candidate = _commit(repo, "candidate policy")

    assert (
        tree_scan.main(
            _runner_arguments(repo, candidate, base, "/usr/bin/true"),
        )
        == 1
    )
    assert "commit allowlists intersect" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("candidate_mode", "expected_error"),
    [
        ("finding", "status 0 and 1 finding(s)"),
        ("missing", "did not produce a regular JSON report"),
        ("nonzero", "status 2 and 0 finding(s)"),
    ],
)
def test_complete_tree_runner_rejects_inconsistent_scanner_results(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    candidate_mode: str,
    expected_error: str,
) -> None:
    """Reject success-with-findings and success-without-report scanner behavior."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    (repo / "clean.txt").write_text("clean candidate\n", encoding="utf-8")
    candidate = _commit(repo, "clean candidate")
    scanner = tmp_path / "controlled-gitleaks"
    scanner.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        + _CONTROLLED_GITLEAKS_VERSION
        + "report = pathlib.Path(sys.argv[sys.argv.index('--report-path') + 1])\n"
        + _CONTROLLED_DECODE_PROBE
        + "if pathlib.Path.cwd().name == 'binary-control':\n"
        "    report.write_text(json.dumps([{'RuleID': 'generic-api-key', "
        "'Secret': 'REDACTED'}] * 3))\n"
        "    raise SystemExit(1)\n"
        f"mode = {candidate_mode!r}\n"
        "if mode == 'finding':\n"
        "    report.write_text(json.dumps([{'RuleID': 'generic-api-key', "
        "'Secret': 'REDACTED', 'File': 'candidate', 'StartLine': 1}]))\n"
        "elif mode == 'nonzero':\n"
        "    report.write_text('[]')\n"
        "    raise SystemExit(2)\n"
        "raise SystemExit(0)\n",
        encoding="utf-8",
    )
    scanner.chmod(0o700)

    assert tree_scan.main(_runner_arguments(repo, candidate, base, str(scanner))) == 1
    assert expected_error in capsys.readouterr().err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_complete_tree_runner_accepts_clean_binary_blob(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Keep the generated binary control separate from a clean candidate verdict."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    (repo / "opaque.bin").write_bytes(
        bytes(range(256)) * 32 + b"\0clean binary payload"
    )
    candidate = _commit(repo, "clean binary")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 0
    output = capsys.readouterr()
    receipt = json.loads(output.out.splitlines()[-1])
    assert receipt["binary_control"] == "passed"
    assert receipt["history_base"] == base
    assert receipt["history_commits"] == [candidate]
    assert receipt["history_findings"] == 0
    assert receipt["historical_tree_findings"] == 0
    assert receipt["historical_tree_scan"]["commits"] == []
    assert receipt["historical_tree_scan"]["scan_files"] == 0
    assert receipt["tree_findings"] == 0
    assert receipt["gitleaks_findings"] == 0
    assert receipt["gitleaks_version"] == tree_scan._GITLEAKS_VERSION
    assert receipt["gitleaks_binary_sha256"] == tree_scan._executable_sha256(
        Path(scanner)
    )


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_uses_exact_history_and_committed_policy_when_head_is_redirected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a deleted historical secret despite dirty policy and redirected HEAD."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    (repo / "deleted-secret.py").write_text(
        f'api_key = "{token}"\n',
        encoding="utf-8",
    )
    _commit(repo, "historical secret")
    (repo / "deleted-secret.py").unlink()
    candidate = _commit(repo, "remove historical secret")

    (repo / ".gitleaks.toml").write_text(
        "[allowlists]\npaths = ['''.*''']\n",
        encoding="utf-8",
    )
    _git(repo, "update-ref", "HEAD", base)
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "committed-history" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_native_history_scan_ignores_dirty_checkout_gitleaksignore(
    tmp_path: Path,
) -> None:
    """Run native Git history outside the checkout that supplies implicit ignores."""
    repo = _repo(tmp_path)
    config = repo / ".gitleaks.toml"
    config.write_text("[extend]\nuseDefault = true\n", encoding="utf-8")
    (repo / "source.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    (repo / "secret.py").write_text(f'api_key = "{token}"\n', encoding="utf-8")
    candidate = _commit(repo, "secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None
    ignore = tmp_path / "verified.ignore"
    ignore.write_text("", encoding="utf-8")
    isolated = tmp_path / "scanner-work"
    isolated.mkdir()
    commit_range = f"{base}..{candidate}"

    first_result, first_findings = tree_scan._run_gitleaks(
        scanner,
        repo,
        config,
        ignore,
        tmp_path / "first.json",
        history_range=commit_range,
        working_directory=isolated,
    )
    assert first_result.returncode == 1
    assert len(first_findings) == 1
    fingerprint = first_findings[0].get("Fingerprint")
    assert isinstance(fingerprint, str) and fingerprint
    (repo / ".gitleaksignore").write_text(f"{fingerprint}\n", encoding="utf-8")

    suppressed_result, suppressed_findings = tree_scan._run_gitleaks(
        scanner,
        repo,
        config,
        ignore,
        tmp_path / "suppressed.json",
        history_range=commit_range,
    )
    assert suppressed_result.returncode == 0
    assert suppressed_findings == []

    history_source = tree_scan._isolated_history_source(
        repo,
        tmp_path / "history-repository",
    )
    isolated_result, isolated_findings = tree_scan._run_gitleaks(
        scanner,
        history_source,
        config,
        ignore,
        tmp_path / "isolated.json",
        history_range=commit_range,
        working_directory=isolated,
    )
    assert isolated_result.returncode == 1
    assert len(isolated_findings) == 1


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_native_history_scan_never_executes_local_textconv(tmp_path: Path) -> None:
    """Disable external diff and textconv while Gitleaks invokes Git history."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / ".gitattributes").write_text("*.txt diff=pwn\n", encoding="utf-8")
    (repo / "source.txt").write_text("clean candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")

    marker = tmp_path / "textconv-ran"
    helper = tmp_path / "textconv"
    helper.write_text(
        f'#!/bin/sh\n: > "{marker}"\ncat "$1"\n',
        encoding="utf-8",
    )
    helper.chmod(0o700)
    _git(repo, "config", "diff.pwn.textconv", str(helper))
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 0
    assert not marker.exists()


def test_runner_can_bind_policy_to_an_exact_ancestor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep candidate policy bytes from weakening a trusted workflow scan."""
    repo = _repo(tmp_path)
    base_policy = "[extend]\nuseDefault = true\n# base-policy\n"
    candidate_policy = "[extend]\nuseDefault = true\n# candidate-policy\n"
    (repo / ".gitleaks.toml").write_text(base_policy, encoding="utf-8")
    (repo / ".gitleaksignore").write_text("base-ignore\n", encoding="utf-8")
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / ".gitleaks.toml").write_text(candidate_policy, encoding="utf-8")
    (repo / ".gitleaksignore").write_text("candidate-ignore\n", encoding="utf-8")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")

    scanner = tmp_path / "controlled-gitleaks"
    scanner.write_text(
        "#!/usr/bin/env python3\n"
        "import json, pathlib, sys\n"
        + _CONTROLLED_GITLEAKS_VERSION
        + "report = pathlib.Path(sys.argv[sys.argv.index('--report-path') + 1])\n"
        "config = pathlib.Path(sys.argv[sys.argv.index('--config') + 1]).read_text()\n"
        "ignore = pathlib.Path(sys.argv[sys.argv.index('--gitleaks-ignore-path') + 1]).read_text()\n"
        + _CONTROLLED_DECODE_PROBE
        + "if pathlib.Path.cwd().name == 'binary-control':\n"
        "    report.write_text(json.dumps([{'RuleID': 'generic-api-key', "
        "'Secret': 'REDACTED'}] * 3))\n"
        "    raise SystemExit(1)\n"
        f"if config == {base_policy!r} and ignore == 'base-ignore\\n':\n"
        "    report.write_text(json.dumps([{'RuleID': 'generic-api-key', "
        "'Secret': 'REDACTED', 'File': 'source.txt', 'StartLine': 1}]))\n"
        "    raise SystemExit(1)\n"
        "report.write_text('[]')\n",
        encoding="utf-8",
    )
    scanner.chmod(0o700)

    candidate_arguments = _runner_arguments(repo, candidate, base, str(scanner))
    assert tree_scan.main(candidate_arguments) == 0
    capsys.readouterr()
    assert (
        tree_scan.main(
            [
                *candidate_arguments,
                "--policy-revision",
                base,
            ]
        )
        == 1
    )
    assert "committed-history" in capsys.readouterr().err


def test_runner_ignores_path_prepended_scanner(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Execute the explicit scanner even when PATH contains an attacker namesake."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")

    trusted = tmp_path / "trusted-gitleaks"
    trusted.write_text(
        "#!/usr/bin/python3\n"
        "import json, pathlib, sys\n"
        + _CONTROLLED_GITLEAKS_VERSION
        + "report = pathlib.Path(sys.argv[sys.argv.index('--report-path') + 1])\n"
        + _CONTROLLED_DECODE_PROBE
        + "if pathlib.Path.cwd().name == 'binary-control':\n"
        "    report.write_text(json.dumps([{'RuleID': 'generic-api-key', "
        "'Secret': 'REDACTED'}] * 3))\n"
        "    raise SystemExit(1)\n"
        "report.write_text('[]')\n",
        encoding="utf-8",
    )
    trusted.chmod(0o700)
    attacker = tmp_path / "attacker-bin"
    attacker.mkdir()
    marker = tmp_path / "path-scanner-ran"
    for name in ("git", "gitleaks"):
        namesake = attacker / name
        namesake.write_text(
            f"#!/bin/sh\n: > {marker}\nexit 0\n",
            encoding="utf-8",
        )
        namesake.chmod(0o700)
    monkeypatch.setenv("PATH", f"{attacker}:{os.environ['PATH']}")

    assert tree_scan.main(_runner_arguments(repo, candidate, base, str(trusted))) == 0
    assert not marker.exists()
    receipt = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert receipt["commit"] == candidate
    assert receipt["history_base"] == base
    assert receipt["history_commits"] == [candidate]
    assert receipt["history_range"] == f"{base}..{candidate}"


def test_runner_rejects_path_resolved_scanner(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Refuse a scanner command whose bytes could be redirected through PATH."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")

    assert tree_scan.main(_runner_arguments(repo, candidate, base, "gitleaks")) == 1
    assert "absolute path" in capsys.readouterr().err


def test_runner_rejects_unpinned_gitleaks_version(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep archive and decoder assumptions bound to one scanner release."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")
    scanner = tmp_path / "wrong-version-gitleaks"
    scanner.write_text(
        "#!/usr/bin/python3\nprint('8.27.0')\n",
        encoding="utf-8",
    )
    scanner.chmod(0o700)

    assert tree_scan.main(_runner_arguments(repo, candidate, base, str(scanner))) == 1
    assert "exact version 8.28.0" in capsys.readouterr().err


def test_runner_rejects_symbolic_candidate_revision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Require the event-bound full object ID instead of mutable HEAD."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    _commit(repo, "candidate")

    assert tree_scan.main(_runner_arguments(repo, "HEAD", base, "/usr/bin/true")) == 1
    assert "canonical full commit ID" in capsys.readouterr().err


def test_rejects_gitlinks_before_writing_partial_snapshot(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Reject submodule gitlinks whose pinned child bytes are not in the parent tree.

    Args:
        tmp_path: Isolated repository and snapshot parent.
        capsys: Captured fail-closed diagnostic.
    Returns:
        None.
    Raises:
        AssertionError: A gitlink is silently omitted or partially materialized.
    """
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    commit = _commit(repo, "parent source")
    _git(repo, "update-index", "--add", "--cacheinfo", f"160000,{commit},vendor")
    _git(repo, "commit", "-qm", "pin child repository")

    destination = tmp_path / "snapshot"
    assert _materialize(repo, destination) == 1
    assert "Unsupported Git tree entry 'vendor'" in capsys.readouterr().err
    assert list(destination.iterdir()) == []


def test_rejects_nonempty_snapshot_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse output that could retain stale or attacker-controlled bytes.

    Args:
        tmp_path: Isolated repository and snapshot parent.
        capsys: Captured fail-closed diagnostic.
    Returns:
        None.
    Raises:
        AssertionError: Existing output bytes are accepted or overwritten.
    """
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _commit(repo, "source")
    destination = tmp_path / "snapshot"
    destination.mkdir()
    (destination / "stale.txt").write_text("stale\n", encoding="utf-8")

    assert (
        materializer.main(["--repo-root", str(repo), "--output-dir", str(destination)])
        == 1
    )
    assert "Output directory must be empty" in capsys.readouterr().err


def test_rejects_symbolic_link_snapshot_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Refuse a destination alias that could be redirected after validation.

    Args:
        tmp_path: Isolated repository and snapshot parent.
        capsys: Captured fail-closed diagnostic.
    Returns:
        None.
    Raises:
        AssertionError: A symbolic-link output root is accepted.
    """
    repo = _repo(tmp_path)
    (repo / "tracked.txt").write_text("tracked\n", encoding="utf-8")
    _commit(repo, "source")
    destination = tmp_path / "snapshot"
    destination.mkdir()
    alias = tmp_path / "snapshot-alias"
    alias.symlink_to(destination, target_is_directory=True)

    assert (
        materializer.main(["--repo-root", str(repo), "--output-dir", str(alias)]) == 1
    )
    assert "must not be a symbolic link" in capsys.readouterr().err
    assert list(destination.iterdir()) == []
