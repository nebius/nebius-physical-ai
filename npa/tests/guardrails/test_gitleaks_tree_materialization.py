"""Verify Gitleaks scans exact committed blobs without checkout transformations."""

from __future__ import annotations

from collections.abc import Callable
import base64
import bz2
import gzip
import importlib.util
import io
import json
import lzma
import os
import shutil
import stat
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
    "assert sys.argv[sys.argv.index('--max-archive-depth') + 1] in {'0', '3'}\n"
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
_CONTROLLED_SECRET_CONTROLS = (
    "control_name = pathlib.Path.cwd().name\n"
    "if control_name in "
    "{'binary-control', 'fragment-boundary-raw-probe', "
    "'fragment-boundary-archive-probe'}:\n"
    "    files = (['binary-wrapper-control.txt', "
    "'archive-control.zip!payload.txt', 'decode-control.txt'] "
    "if control_name == 'binary-control' else ['carrier.bin'])\n"
    "    report.write_text(json.dumps([{'RuleID': 'generic-api-key', "
    "'Secret': 'REDACTED', 'File': name} for name in files]))\n"
    "    raise SystemExit(1)\n"
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
        base,
        "--gitleaks-bin",
        scanner,
    ]


def _trust_test_scanner(
    monkeypatch: pytest.MonkeyPatch,
    scanner: Path,
) -> None:
    """Bind one synthetic scanner's bytes inside an isolated unit test."""
    platform_identity = (tree_scan.sys.platform, tree_scan.os.uname().machine.lower())
    monkeypatch.setitem(
        tree_scan._TRUSTED_GITLEAKS_SHA256,
        platform_identity,
        tree_scan._executable_sha256(scanner),
    )


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


def test_materializes_authenticated_selected_commit_metadata(tmp_path: Path) -> None:
    """Expose exact selected commit bodies to the filesystem policy scan."""
    repo = _repo(tmp_path)
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    first = _commit(repo, "first selected message")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    second = _commit(repo, "second selected message")
    destination = tmp_path / "commit-metadata"
    destination.mkdir()

    receipt = materializer.materialize_commit_metadata(
        repo,
        destination,
        [first, second],
    )

    assert receipt["commits"] == [first, second]
    assert receipt["files"] == 2
    assert receipt["wrapper_files"] == 2
    assert receipt["bytes"] == sum(
        (destination / f"{commit}.txt").stat().st_size for commit in (first, second)
    )
    wrappers = sorted(
        (destination / os.fsdecode(materializer.SCAN_NON_TEST_ROOT)).glob(
            "commit-*.txt"
        )
    )
    assert len(wrappers) == 2
    assert all(
        wrapper.read_bytes().startswith(materializer.SCAN_PREFIX)
        for wrapper in wrappers
    )
    assert receipt["scan_bytes"] == receipt["bytes"] + sum(
        wrapper.stat().st_size for wrapper in wrappers
    )
    assert (
        (destination / f"{first}.txt")
        .read_bytes()
        .endswith(b"\nfirst selected message\n")
    )
    assert (
        (destination / f"{second}.txt")
        .read_bytes()
        .endswith(b"\nsecond selected message\n")
    )


def test_materializer_bounds_blob_and_commit_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject candidate-controlled source expansion before wrapper duplication."""
    repo = _repo(tmp_path)
    (repo / "first.txt").write_bytes(b"A" * 8)
    (repo / "second.txt").write_bytes(b"B" * 8)
    first = _commit(repo, "first")
    _git(repo, "commit", "--allow-empty", "-qm", "second")
    second = _git(repo, "rev-parse", "HEAD")

    monkeypatch.setattr(materializer, "_MAX_SCAN_BLOB_BYTES", 7)
    monkeypatch.setattr(materializer, "_MAX_SCAN_SOURCE_BYTES", 1024)
    single_blob_destination = tmp_path / "bounded-single-blob"
    single_blob_destination.mkdir()
    with pytest.raises(materializer.SnapshotError, match="bounded scan-source"):
        materializer.materialize(
            repo,
            single_blob_destination,
            second,
            scan_view=True,
        )

    monkeypatch.setattr(materializer, "_MAX_SCAN_BLOB_BYTES", 12)
    monkeypatch.setattr(materializer, "_MAX_SCAN_SOURCE_BYTES", 12)
    blob_destination = tmp_path / "bounded-blobs"
    blob_destination.mkdir()
    with pytest.raises(materializer.SnapshotError, match="aggregate scan-source"):
        materializer.materialize(
            repo,
            blob_destination,
            second,
            scan_view=True,
        )

    monkeypatch.setattr(materializer, "_MAX_SCAN_BLOB_BYTES", 1024)
    monkeypatch.setattr(materializer, "_MAX_SCAN_SOURCE_BYTES", 1024)
    monkeypatch.setattr(materializer, "_MAX_SCAN_VIEW_BYTES", 1)
    view_destination = tmp_path / "bounded-generated-view"
    view_destination.mkdir()
    with pytest.raises(materializer.SnapshotError, match="Generated scan view"):
        materializer.materialize(
            repo,
            view_destination,
            second,
            scan_view=True,
        )
    assert not any(view_destination.iterdir())

    first_bytes = len(materializer.read_verified_commit(repo, first))
    second_bytes = len(materializer.read_verified_commit(repo, second))
    with monkeypatch.context() as bounded:
        bounded.setattr(
            materializer,
            "_MAX_COMMIT_METADATA_BYTES",
            first_bytes - 1,
        )

        def unexpected_body_read(_stream: object, _size: int) -> bytes:
            pytest.fail("oversized commit body was read before its header limit")

        bounded.setattr(
            materializer,
            "_read_exact_object_body",
            unexpected_body_read,
        )
        with pytest.raises(materializer.SnapshotError, match="metadata exceeds"):
            materializer.read_verified_commit(repo, first)

    monkeypatch.setattr(
        materializer,
        "_MAX_COMMIT_METADATA_BYTES",
        first_bytes - 1,
    )
    monkeypatch.setattr(
        materializer,
        "_MAX_COMMIT_METADATA_TOTAL_BYTES",
        1024 * 1024,
    )
    single_metadata_destination = tmp_path / "bounded-single-metadata"
    single_metadata_destination.mkdir()
    with pytest.raises(materializer.SnapshotError, match="metadata exceeds"):
        materializer.materialize_commit_metadata(
            repo,
            single_metadata_destination,
            [first],
        )

    monkeypatch.setattr(materializer, "_MAX_COMMIT_METADATA_BYTES", 1024 * 1024)
    monkeypatch.setattr(
        materializer,
        "_MAX_COMMIT_METADATA_TOTAL_BYTES",
        first_bytes + second_bytes - 1,
    )
    metadata_destination = tmp_path / "bounded-metadata"
    metadata_destination.mkdir()
    with pytest.raises(materializer.SnapshotError, match="aggregate bounded"):
        materializer.materialize_commit_metadata(
            repo,
            metadata_destination,
            [first, second],
        )


def test_materializer_bounds_tree_metadata_and_paths_before_writes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound tree allocation and path amplification independently of blob bytes."""
    repo = _repo(tmp_path)
    (repo / "first").write_bytes(b"")
    (repo / "second").write_bytes(b"")
    commit = _commit(repo, "bounded tree")
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}").encode("ascii")

    with monkeypatch.context() as bounded:
        bounded.setattr(materializer, "_MAX_TREE_OBJECT_BYTES", 1)

        def unexpected_body_read(_stream: object, _size: int) -> bytes:
            pytest.fail("oversized tree body was read before its header limit")

        bounded.setattr(
            materializer,
            "_read_exact_object_body",
            unexpected_body_read,
        )
        with pytest.raises(materializer.SnapshotError, match="tree metadata"):
            materializer._tree_entries(repo, tree, b"sha1")

    monkeypatch.setattr(materializer, "_MAX_TREE_PATHS", 1)
    destination = tmp_path / "bounded-paths"
    destination.mkdir()
    with pytest.raises(materializer.SnapshotError, match="path count"):
        materializer.materialize(repo, destination, commit, scan_view=True)
    assert not any(destination.iterdir())


def test_policy_blob_bound_precedes_body_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an oversized committed policy from its Git header."""
    repo = _repo(tmp_path)
    policy = b"[extend]\nuseDefault = true\n#" + (b"x" * 4096)
    ignore = b"# bounded ignore\n" + (b"y" * 8192)
    (repo / ".gitleaks.toml").write_bytes(policy)
    (repo / ".gitleaksignore").write_bytes(ignore)
    commit = _commit(repo, "oversized policy")
    tree = _git(repo, "rev-parse", f"{commit}^{{tree}}")
    read_body = tree_scan.materializer._read_exact_object_body
    monkeypatch.setattr(tree_scan, "_MAX_POLICY_BLOB_BYTES", len(policy) - 1)

    def reject_policy_body(stream: object, size: int) -> bytes:
        if size in {len(policy), len(ignore)}:
            pytest.fail("oversized policy body was read before its header limit")
        return read_body(stream, size)

    monkeypatch.setattr(
        tree_scan.materializer,
        "_read_exact_object_body",
        reject_policy_body,
    )
    with pytest.raises(
        tree_scan.materializer.SnapshotError,
        match="policy blob exceeds",
    ):
        tree_scan._config_blob(repo, tree, ".gitleaks.toml")
    with pytest.raises(
        tree_scan.materializer.SnapshotError,
        match="policy blob exceeds",
    ):
        tree_scan._optional_config_blob(repo, tree, ".gitleaksignore")


def test_private_scanner_controls_have_a_diversity_floor() -> None:
    """Keep random control entropy from making real scanner checks intermittent."""
    tokens = tree_scan._private_control_tokens(128)

    assert len(tokens) == 128
    assert len(set(tokens)) == 128
    assert all(len(token) == 56 for token in tokens)
    assert all(len(set(token[-24::2])) == 12 for token in tokens)
    assert all(token[-24::2].isalpha() for token in tokens)
    assert all(token[-23::2].isdigit() for token in tokens)


def test_binary_control_requires_three_distinct_carrier_paths(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not let duplicate findings from one carrier satisfy all controls."""
    duplicate = {
        "RuleID": "generic-api-key",
        "Secret": "REDACTED",
        "File": "binary-wrapper-control.txt",
    }

    def duplicate_scan(*_args: object, **_kwargs: object) -> tuple[object, list[dict]]:
        return subprocess.CompletedProcess([], 1, b"", b""), [duplicate] * 3

    monkeypatch.setattr(tree_scan, "_run_gitleaks", duplicate_scan)
    with pytest.raises(tree_scan._ScanError, match="each distinct private"):
        tree_scan._verify_binary_control(
            "/unused/gitleaks",
            tmp_path / "policy.toml",
            tmp_path / "policy.ignore",
            tmp_path,
            tree_scan._ReportBudget(),
        )


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


def test_history_walk_bounds_reachable_commit_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an authenticated graph before unbounded commit accumulation."""
    repo = _repo(tmp_path)
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")
    monkeypatch.setattr(materializer, "_MAX_HISTORY_COMMITS", 1)

    with pytest.raises(materializer.SnapshotError, match="bounded commit count"):
        materializer.reachable_history_commits(repo, candidate)


def test_history_walk_bounds_aggregate_commit_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound aggregate authenticated commit reads before body allocation."""
    repo = _repo(tmp_path)
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")
    monkeypatch.setattr(materializer, "_MAX_HISTORY_COMMIT_BYTES", 1)

    with pytest.raises(materializer.SnapshotError, match="aggregate metadata byte"):
        materializer.reachable_history_commits(repo, candidate)


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
    assert receipt["wrapper_scan_files"] == 4
    assert receipt["scan_files"] == 6
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
    context_wrappers = [
        path
        for path in (
            destination / os.fsdecode(materializer.SCAN_NON_TEST_ROOT) / "path-context"
        ).rglob("*")
        if path.is_file()
    ]
    assert len(context_wrappers) == 2
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


def test_history_view_preserves_each_blob_path_association(tmp_path: Path) -> None:
    """Do not collapse one blob renamed between path-constrained rule scopes."""
    repo = _repo(tmp_path)
    first_path = repo / "a" / "key.pem"
    first_path.parent.mkdir()
    first_path.write_bytes(b"shared historical bytes\n")
    first = _commit(repo, "first logical path")
    second_path = repo / "b" / "key.pem"
    second_path.parent.mkdir()
    first_path.rename(second_path)
    second = _commit(repo, "second logical path")
    destination = tmp_path / "historical"
    destination.mkdir()

    receipt = materializer.materialize_history_scan(
        repo,
        destination,
        [first, second],
    )

    assert receipt["unique_blob_contexts"] == 2
    context_paths = {
        path.relative_to(destination).as_posix()
        for path in destination.rglob("key.pem")
        if "path-context" in path.parts
    }
    assert any(path.endswith("/a/key.pem") for path in context_paths)
    assert any(path.endswith("/b/key.pem") for path in context_paths)


def test_historical_raw_path_recovery_preserves_non_utf8_git_bytes(
    tmp_path: Path,
) -> None:
    """Match raw-history digests for Git paths represented with surrogates."""
    repo = _repo(tmp_path)
    logical_name = os.fsdecode(b"archive-\xff.zip")
    (repo / logical_name).write_bytes(b"not an archive\n")
    revision = _commit(repo, "non-UTF-8 path")
    destination = tmp_path / "historical"
    destination.mkdir()

    materializer.materialize_history_scan(repo, destination, [revision])

    history_files = [
        path
        for path in (destination / ".npa-gitleaks-history").rglob("*")
        if path.is_file()
    ]
    assert len(history_files) == 1
    history_relative = history_files[0].relative_to(destination).as_posix()
    recovered = tree_scan._historical_raw_logical_paths(destination)
    assert recovered[history_relative] == (logical_name,)


def test_history_exact_paths_shard_file_directory_prefix_transitions(
    tmp_path: Path,
) -> None:
    """Keep historical file-to-directory transitions representable exactly."""
    repo = _repo(tmp_path)
    node = repo / "node"
    node.write_text("first version\n", encoding="utf-8")
    first = _commit(repo, "node is a file")
    node.unlink()
    node.mkdir()
    (node / "child").write_text("second version\n", encoding="utf-8")
    second = _commit(repo, "node becomes a directory")
    destination = tmp_path / "historical"
    destination.mkdir()
    exact_root = tmp_path / "historical-exact"

    receipt = materializer.materialize_history_scan(
        repo,
        destination,
        [first, second],
        exact_path_root=exact_root,
    )

    assert receipt["exact_path_shards"] == 2
    exact_files = {
        path.relative_to(exact_root).as_posix()
        for path in exact_root.rglob("*")
        if path.is_file()
    }
    assert exact_files == {"0/node", "1/node/child"}


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

    matches = [
        path
        for path in destination.rglob("credential.tar.gz")
        if ".npa-gitleaks-history" in path.parts
    ]
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
    if carrier not in {
        "zip_encoded_zip",
        "gzip_encoded_zip",
        "zip_prefixed_gzip",
    }:
        return False
    nested = base64.b64encode(_zip_member("credential.txt", assignment.encode()))
    if carrier == "zip_encoded_zip":
        (repo / "credential.zip").write_bytes(_zip_member("carrier.txt", nested))
    elif carrier == "gzip_encoded_zip":
        compressor = zlib.compressobj(wbits=31)
        (repo / "credential.gz").write_bytes(
            compressor.compress(nested) + compressor.flush()
        )
    else:
        compressor = zlib.compressobj(wbits=31)
        nested = compressor.compress(assignment.encode()) + compressor.flush()
        (repo / "credential.zip").write_bytes(
            _zip_member("carrier.bin", b"safe-prefix\n" + nested)
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
    if carrier == "zip_directory_path":
        with zipfile.ZipFile(repo / "credential.zip", "w") as archive:
            archive.writestr(assignment.strip() + "/", b"")
        return True
    if carrier == "zip_directory_payload":
        with zipfile.ZipFile(repo / "credential.zip", "w") as archive:
            archive.writestr("metadata/", assignment.encode())
        return True
    if carrier == "zip_comment":
        compressor = zlib.compressobj(wbits=31)
        with zipfile.ZipFile(repo / "credential.zip", "w") as archive:
            archive.comment = (
                compressor.compress(assignment.encode()) + compressor.flush()
            )
            archive.writestr("safe.txt", b"safe\n")
        return True
    if carrier == "zip_utf16":
        with zipfile.ZipFile(
            repo / "credential.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("credential.txt", assignment.encode("utf-16le"))
        return True
    if carrier == "zip_policy_blob":
        with zipfile.ZipFile(
            repo / "credential.zip",
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr(".gitleaksignore", assignment)
        return True
    if carrier == "tar_pax":
        metadata = tarfile.TarInfo("safe.txt")
        metadata.size = len(b"safe\n")
        metadata.pax_headers = {"comment": assignment}
        with tarfile.open(
            repo / "credential.tar.gz",
            "w:gz",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            archive.addfile(metadata, io.BytesIO(b"safe\n"))
        return True
    if carrier == "tar_trailing_gzip":
        raw_tar = io.BytesIO()
        with tarfile.open(fileobj=raw_tar, mode="w") as archive:
            metadata = tarfile.TarInfo("safe.txt")
            metadata.size = len(b"safe\n")
            archive.addfile(metadata, io.BytesIO(b"safe\n"))
        (repo / "credential.tar.gz").write_bytes(
            gzip.compress(raw_tar.getvalue() + gzip.compress(assignment.encode()))
        )
        return True
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
        ("credential.lzma", "alias"),
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
        "tar_trailing_gzip",
        "zip_comment",
        "zip_directory_payload",
        "zip_encoded_zip",
        "zip_prefixed_gzip",
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
        "zip_comment",
        "zip_directory_path",
        "zip_directory_payload",
        "zip_prefixed_gzip",
        "zip_utf16",
        "zip_policy_blob",
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
        "tar_pax",
        "tar_trailing_gzip",
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


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_blob_wrapper_preserves_policy_path_suffix(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Apply a path-constrained rule to decoded committed blob text."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "pem-only-control"\n'
        'description = "Path-preservation control"\n'
        "regex = '''NPA_PATH_SECRET_[A-Z0-9]{24}'''\n"
        "path = '''^keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_PATH_SECRET_" + ("A1" * 12)
    (repo / "keys").mkdir()
    (repo / "keys/key.pem").write_bytes(token.encode("utf-16le"))
    candidate = _commit(repo, "path-scoped blob secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "pem-only-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_fragment_overlap_preserves_exact_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scan a transformed boundary overlap under its anchored logical path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "anchored-multiline"\n'
        'description = "Anchored transformed fragment control"\n'
        "regex = '''NPA_PATH_KEY\\s*=\\s*"
        "NPA_PATH_SECRET_[A-Z0-9]{24}'''\n"
        "path = '''^keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_PATH_SECRET_" + ("A1" * 12)
    text = "A" * 31914 + "NPA_PATH_KEY =\n\n" + token + "\n"
    key = repo / "keys" / "key.pem"
    key.parent.mkdir()
    key.write_bytes(text.encode("utf-16le"))
    candidate = _commit(repo, "path-scoped boundary secret")
    inspection = tmp_path / "inspection"
    inspection.mkdir()
    materializer.materialize(repo, inspection, candidate, scan_view=True)
    wrapper = next(
        path
        for path in (inspection / os.fsdecode(materializer.SCAN_NON_TEST_ROOT)).glob(
            "*.txt"
        )
        if b"keys/key.pem" in path.read_bytes()
    )
    assert wrapper.read_bytes().index(b"NPA_PATH_SECRET_") == 100_000
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "anchored-multiline" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_exact_view_covers_rule_wider_than_fragment_overlap(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Catch a 45,001/44,999 split that the symmetric overlap cannot contain."""
    repo = _repo(tmp_path)
    config_content = (
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "wide-boundary-control"\n'
        'description = "Wide boundary control"\n'
        "regex = '''BEGIN(?s:.)*END'''\n"
        "path = '''^wide\\.txt$'''\n"
    )
    (repo / ".gitleaks.toml").write_text(config_content, encoding="utf-8")
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    boundary = tree_scan._GITLEAKS_FRAGMENT_BYTES + 1
    left = b"BEGIN" + (b"L" * (45_001 - len(b"BEGIN") - 2)) + b"\n\n"
    payload = (b"A" * (boundary - len(left))) + left + (b"R" * 44_999) + b"END"
    (repo / "wide.txt").write_bytes(payload)
    candidate = _commit(repo, "wide boundary carrier")
    short_read_boundary = (
        boundary
        + tree_scan._GITLEAKS_READER_BUFFER_BYTES
        - 1
        + tree_scan._GITLEAKS_FRAGMENT_PEEK_BYTES
    )
    assert tree_scan._gitleaks_fragment_boundaries(repo / "wide.txt") == [
        boundary,
        short_read_boundary,
    ]
    scanner = shutil.which("gitleaks")
    assert scanner is not None
    direct = tmp_path / "direct"
    direct.mkdir()
    (direct / "wide.txt").write_bytes(payload)
    config = tmp_path / "wide-policy.toml"
    config.write_text(config_content, encoding="utf-8")
    ignore = tmp_path / "wide-policy.ignore"
    ignore.write_text("", encoding="utf-8")
    direct_result, direct_findings = tree_scan._run_gitleaks(
        scanner,
        direct,
        config,
        ignore,
        tmp_path / "wide-direct.json",
    )
    assert direct_result.returncode == 0
    assert direct_findings == []

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert "wide-boundary-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_forced_text_exact_view_ignores_candidate_git_controls(
    tmp_path: Path,
) -> None:
    """Force-add binary paths without honoring candidate ignore or attributes."""
    view = tmp_path / "exact-view"
    view.mkdir()
    (view / ".gitignore").write_text("*\n", encoding="utf-8")
    (view / ".gitattributes").write_text(
        "* filter=attacker -diff\n",
        encoding="utf-8",
    )
    token = "NPA_FORCED_TEXT_" + ("A1" * 12)
    (view / "carrier.zip").write_bytes((b"\0" * 128) + token.encode())
    config_content = (
        "[[rules]]\n"
        'id = "forced-text-control"\n'
        'description = "Forced text control"\n'
        "regex = '''NPA_FORCED_TEXT_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.zip$'''\n"
    ).encode()
    config = tmp_path / "policy.toml"
    config.write_bytes(config_content)
    ignore = tmp_path / "policy.ignore"
    ignore.write_text("", encoding="utf-8")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    result, findings = tree_scan._run_gitleaks_exact_view(
        scanner,
        view,
        config,
        ignore,
        tmp_path / "report.json",
        config_content,
        tree_scan._ReportBudget(),
    )

    assert result.returncode == 1
    assert any(
        finding.get("RuleID") == "forced-text-control"
        and finding.get("File") == "carrier.zip"
        for finding in findings
    )


def test_generated_exact_view_rejects_its_commit_allowlist(tmp_path: Path) -> None:
    """Keep synthetic Git identities from restoring a trusted suppression."""
    first = tmp_path / "first"
    second = tmp_path / "second"
    first.mkdir()
    second.mkdir()
    for view in (first, second):
        (view / "carrier.txt").write_text("same exact view\n", encoding="utf-8")
    commit = tree_scan._prepare_forced_text_repository(
        first,
        b"[extend]\nuseDefault = true\n",
    )
    policy = (
        b"[extend]\n"
        b"useDefault = true\n\n"
        b"[[allowlists]]\n" + f"commits = ['''{commit}''']\n".encode()
    )

    with pytest.raises(
        tree_scan._ScanError,
        match="allowlist suppresses a generated exact-path",
    ):
        tree_scan._prepare_forced_text_repository(second, policy)


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_encoded_probe_preserves_exact_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep a complete boundary-split encoded span under its logical path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "encoded-path-control"\n'
        'description = "Encoded exact-path control"\n'
        "regex = '''NPA_ENCODED_PATH_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_ENCODED_PATH_" + ("A1" * 12)
    encoded = base64.b64encode((b"A" * 30_000) + token.encode())
    (repo / "carrier.txt").write_bytes((b"!" * 110_000) + encoded)
    candidate = _commit(repo, "path-scoped encoded boundary secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "encoded-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_root_ignore_wide_rule_uses_scanner_safe_generated_git_view(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scan full binary root-ignore bytes without activating candidate policy."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "root-ignore-wide-control"\n'
        'description = "Root ignore wide control"\n'
        "regex = '''NPA_ROOT_IGNORE_BEGIN(?s:.)*NPA_ROOT_IGNORE_END'''\n",
        encoding="utf-8",
    )
    (repo / ".gitleaksignore").write_text("", encoding="utf-8")
    (repo / "safe.txt").write_text("policy base\n", encoding="utf-8")
    base = _commit(repo, "trusted policy")
    (repo / ".gitleaksignore").write_bytes(
        b"\0NPA_ROOT_IGNORE_BEGIN" + (b"Z" * 300_000) + b"NPA_ROOT_IGNORE_END"
    )
    candidate = _commit(repo, "binary root ignore carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert "root-ignore-wide-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_fails_closed_for_root_ignore_anchored_rule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not omit an anchored rule for the scanner-control source path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "root-ignore-only-control"\n'
        'description = "Root ignore anchored control"\n'
        "regex = '''NPA_IGNORE_ANCHORED_[A-Z0-9]{24}'''\n"
        "path = '''^\\.gitleaksignore$'''\n",
        encoding="utf-8",
    )
    (repo / ".gitleaksignore").write_text("", encoding="utf-8")
    (repo / "safe.txt").write_text("policy base\n", encoding="utf-8")
    policy = _commit(repo, "trusted policy")
    token = "NPA_IGNORE_ANCHORED_" + ("A1" * 12)
    (repo / ".gitleaksignore").write_text(token + "\n", encoding="utf-8")
    history_base = _commit(repo, "excluded base introduces carrier")
    (repo / "safe.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate unrelated change")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    arguments = _runner_arguments(repo, candidate, history_base, scanner)
    arguments[arguments.index("--policy-revision") + 1] = policy
    assert tree_scan.main(arguments) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "root .gitleaksignore exact path" in output.err


def test_root_ignore_policy_check_rejects_unsupported_re2_path() -> None:
    """Fail closed when Python cannot safely evaluate a valid RE2 path form."""
    policy = (
        b"[[rules]]\n"
        b'id = "root-ignore-posix-control"\n'
        b"regex = '''NPA_IGNORE_ANCHORED_[A-Z0-9]{24}'''\n"
        b"path = '''^[[:punct:]]gitleaksignore$'''\n"
    )

    with pytest.raises(tree_scan._ScanError, match="unsupported custom rule path"):
        tree_scan._policy_path_matches_root_ignore(policy)


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_allows_root_ignore_rule_when_no_source_excludes_that_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not reject an anchored rule when no omitted source has that path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "root-ignore-only-control"\n'
        'description = "Root ignore anchored control"\n'
        "regex = '''NPA_IGNORE_ANCHORED_[A-Z0-9]{24}'''\n"
        "path = '''^\\.gitleaksignore$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("policy base\n", encoding="utf-8")
    policy = _commit(repo, "trusted policy without root ignore")
    (repo / "safe.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    arguments = _runner_arguments(repo, candidate, policy, scanner)
    arguments[arguments.index("--policy-revision") + 1] = policy
    assert tree_scan.main(arguments) == 0
    output = capsys.readouterr()
    assert "root .gitleaksignore exact path" not in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_fails_closed_for_historical_root_ignore_anchored_rule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Account for omitted root scanner-control paths in historical views."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "root-ignore-only-control"\n'
        'description = "Root ignore anchored control"\n'
        "regex = '''NPA_IGNORE_ANCHORED_[A-Z0-9]{24}'''\n"
        "path = '''^\\.gitleaksignore$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("policy base\n", encoding="utf-8")
    policy = _commit(repo, "trusted policy")
    token = "NPA_IGNORE_ANCHORED_" + ("B2" * 12)
    (repo / ".gitleaksignore").write_text(token + "\n", encoding="utf-8")
    _commit(repo, "historical carrier")
    (repo / ".gitleaksignore").unlink()
    (repo / "safe.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate removes carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    arguments = _runner_arguments(repo, candidate, policy, scanner)
    arguments[arguments.index("--policy-revision") + 1] = policy
    assert tree_scan.main(arguments) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "root .gitleaksignore exact path" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_archive_member_named_ignore_preserves_full_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scan an inner ignore-named member without activating it as policy."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "root-ignore-only-control"\n'
        'description = "Root ignore anchored control"\n'
        "regex = '''NPA_IGNORE_ANCHORED_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.zip!\\.gitleaksignore$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("policy base\n", encoding="utf-8")
    policy = _commit(repo, "trusted policy")
    token = "NPA_IGNORE_ANCHORED_" + ("C3" * 12)
    with zipfile.ZipFile(repo / "carrier.zip", "w") as archive:
        archive.writestr(".gitleaksignore", token)
    candidate = _commit(repo, "archived root ignore carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    arguments = _runner_arguments(repo, candidate, policy, scanner)
    arguments[arguments.index("--policy-revision") + 1] = policy
    assert tree_scan.main(arguments) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "root-ignore-only-control" in output.err
    assert "root .gitleaksignore exact path" not in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_fails_closed_for_commit_archive_root_ignore_rule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject an encoded root scanner-control archive in commit metadata."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "root-ignore-only-control"\n'
        'description = "Root ignore anchored control"\n'
        "regex = '''NPA_IGNORE_ANCHORED_[A-Z0-9]{24}'''\n"
        "path = '''^\\.gitleaksignore$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("policy base\n", encoding="utf-8")
    policy = _commit(repo, "trusted policy")
    token = "NPA_IGNORE_ANCHORED_" + ("D4" * 12)
    encoded_archive = io.BytesIO()
    with zipfile.ZipFile(encoded_archive, "w") as archive:
        archive.writestr(".gitleaksignore", token)
    commit_carrier = base64.b64encode(encoded_archive.getvalue()).decode("ascii")
    (repo / "safe.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, f"encoded archive carrier {commit_carrier}")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    arguments = _runner_arguments(repo, candidate, policy, scanner)
    arguments[arguments.index("--policy-revision") + 1] = policy
    assert tree_scan.main(arguments) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "Encoded archive bytes cannot prove complete" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize("archive_suffix", [".tar", ".tar.gz", ".zip"])
def test_archive_member_wrapper_preserves_full_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    archive_suffix: str,
) -> None:
    """Apply a path-constrained rule to decoded archive-member text."""
    repo = _repo(tmp_path)
    archive_name = f"credential{archive_suffix}"
    escaped_archive_name = archive_name.replace(".", "\\.")
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "pem-only-control"\n'
        'description = "Path-preservation control"\n'
        "regex = '''NPA_PATH_SECRET_[A-Z0-9]{24}'''\n"
        f"path = '''^{escaped_archive_name}!keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_PATH_SECRET_" + ("A1" * 12)
    payload = token.encode("utf-16le")
    archive_path = repo / archive_name
    if archive_suffix == ".zip":
        with zipfile.ZipFile(
            archive_path,
            "w",
            compression=zipfile.ZIP_DEFLATED,
        ) as archive:
            archive.writestr("keys/key.pem", payload)
    else:
        mode = "w:gz" if archive_suffix == ".tar.gz" else "w"
        with tarfile.open(archive_path, mode) as archive:
            metadata = tarfile.TarInfo("keys/key.pem")
            metadata.size = len(payload)
            archive.addfile(metadata, io.BytesIO(payload))
    candidate = _commit(repo, "path-scoped archive secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "pem-only-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_historical_archive_member_preserves_full_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep outer/member path semantics in reconstructed historical trees."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "historical-archive-path-control"\n'
        'description = "Historical archive path control"\n'
        "regex = '''NPA_HISTORY_ARCHIVE_[A-Z0-9]{24}'''\n"
        "path = '''^history\\.zip!keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("trusted base\n", encoding="utf-8")
    base = _commit(repo, "trusted policy and base")
    token = "NPA_HISTORY_ARCHIVE_" + ("A1" * 12)
    with zipfile.ZipFile(
        repo / "history.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("keys/key.pem", token.encode("utf-16le"))
    _commit(repo, "historical archive carrier")
    (repo / "history.zip").unlink()
    (repo / "safe.txt").write_text("clean candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate removes historical archive")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "historical-archive-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_archive_container_wrapper_preserves_outer_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Scan transformed outer archive bytes under their exact logical path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "archive-container-path-control"\n'
        'description = "Archive container path control"\n'
        "regex = '''NPA_ARCHIVE_COMMENT_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.zip$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_ARCHIVE_COMMENT_" + ("A1" * 12)
    with zipfile.ZipFile(repo / "carrier.zip", "w") as archive:
        archive.writestr("safe.txt", b"safe\n")
        archive.comment = token.encode("utf-16le")
    candidate = _commit(repo, "path-scoped archive comment")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "archive-container-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize(
    ("suffix", "compress"),
    [
        (".bz2", bz2.compress),
        (".gz", gzip.compress),
        (".xz", lzma.compress),
    ],
)
def test_compressed_stream_wrapper_preserves_outer_policy_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
    compress: Callable[[bytes], bytes],
) -> None:
    """Keep the outer path used by Gitleaks for decompressor-only streams."""
    repo = _repo(tmp_path)
    escaped_suffix = suffix.replace(".", "\\.")
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "compressed-path-control"\n'
        'description = "Compressed-stream path control"\n'
        "regex = '''NPA_COMPRESSED_SECRET_[A-Z0-9]{24}'''\n"
        f"path = '''^credential{escaped_suffix}$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_COMPRESSED_SECRET_" + ("A1" * 12)
    (repo / f"credential{suffix}").write_bytes(compress(token.encode("utf-16le")))
    candidate = _commit(repo, "path-scoped compressed secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "compressed-path-control" in output.err


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


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize("policy_position", ["candidate", "intermediate"])
def test_runner_rejects_candidate_controlled_policy_revision(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    policy_position: str,
) -> None:
    """Do not let candidate policy bytes suppress their own real finding."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("trusted base\n", encoding="utf-8")
    base = _commit(repo, "trusted excluded base")
    token = "NPA_POLICY_SECRET_" + ("A1" * 12)
    suppressing_policy = (
        "[extend]\n"
        "useDefault = true\n\n"
        "[[allowlists]]\n"
        'description = "candidate suppression"\n'
        f"regexes = ['''{token}''']\n"
    )
    (repo / ".gitleaks.toml").write_text(
        suppressing_policy,
        encoding="utf-8",
    )
    if policy_position == "intermediate":
        policy = _commit(repo, "candidate-controlled policy")
    (repo / "secret.txt").write_text(
        f'api_key = "{token}"\n',
        encoding="utf-8",
    )
    candidate = _commit(repo, "candidate secret and policy")
    if policy_position == "candidate":
        policy = candidate
    scanner = shutil.which("gitleaks")
    assert scanner is not None
    ignore = tmp_path / "empty-ignore"
    ignore.write_text("", encoding="utf-8")
    result, findings = tree_scan._run_gitleaks(
        scanner,
        repo,
        repo / ".gitleaks.toml",
        ignore,
        tmp_path / "suppressed-report.json",
    )
    assert result.returncode == 0
    assert findings == []

    arguments = _runner_arguments(repo, candidate, base, scanner)
    arguments[arguments.index("--policy-revision") + 1] = policy
    assert tree_scan.main(arguments) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "Policy revision must equal or precede" in output.err


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


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize("secret_commit", ["candidate", "intermediate"])
def test_runner_scans_selected_commit_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    secret_commit: str,
) -> None:
    """Reject a secret carried only by any selected-range commit message."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    _git(repo, "commit", "--allow-empty", "-qm", f'api_key = "{token}"')
    if secret_commit == "intermediate":
        _git(repo, "commit", "--allow-empty", "-qm", "safe candidate")
    candidate = _git(repo, "rev-parse", "HEAD")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "generic-api-key" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_binary_commit_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep a NUL-bearing commit message reachable through a text wrapper."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    _git(repo, "commit", "--allow-empty", "-qm", "placeholder")
    placeholder = _git(repo, "rev-parse", "HEAD")
    content = materializer.read_verified_commit(repo, placeholder)
    message_start = content.index(b"\n\n") + 2
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    binary_content = (
        content[:message_start]
        + b'\0binary metadata\napi_key = "'
        + token.encode()
        + b'"\n'
    )
    candidate = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "hash-object",
                "-t",
                "commit",
                "-w",
                "--stdin",
            ],
            input=binary_content,
            check=True,
            capture_output=True,
        )
        .stdout.decode("ascii")
        .strip()
    )
    _git(repo, "update-ref", "HEAD", candidate, placeholder)
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "generic-api-key" in output.err


def test_embedded_archive_probe_ignores_invalid_media_signatures(
    tmp_path: Path,
) -> None:
    """Do not classify incidental short compression markers as containers."""
    carrier = tmp_path / "media.bin"
    carrier.write_bytes(
        b"image-prefix"
        + bytes.fromhex("1f8b08b353aa0c56fba7e165")
        + b"video-middle"
        + bytes.fromhex("425a68372dfcdb7ffb6fe197")
    )

    assert not tree_scan._contains_embedded_archive(carrier)


@pytest.mark.parametrize("compress", [gzip.compress, bz2.compress])
def test_embedded_archive_probe_recognizes_decodable_streams(
    tmp_path: Path,
    compress: Callable[[bytes], bytes],
) -> None:
    """Recognize real compressed carriers across the former chunk boundary."""
    carrier = tmp_path / "carrier.bin"
    prefix = b"x" * (tree_scan._ARCHIVE_READ_BYTES - 2)
    carrier.write_bytes(prefix + compress(b"hidden payload"))

    assert tree_scan._contains_embedded_archive(carrier)


def test_embedded_archive_probe_rejects_truncated_output_and_unsupported_streams(
    tmp_path: Path,
) -> None:
    """Fail closed after decoded output and on unsupported container bytes."""
    carrier = tmp_path / "carrier.bin"
    carrier.write_bytes(b"prefix" + gzip.compress(b"hidden payload")[:-8])
    assert tree_scan._contains_embedded_archive(carrier)

    corrupt_checksum = bytearray(gzip.compress(b"x"))
    corrupt_checksum[-8] ^= 1
    carrier.write_bytes(b"prefix" + bytes(corrupt_checksum))
    assert tree_scan._contains_embedded_archive(carrier)

    carrier.write_bytes(b"prefix\x28\xb5\x2f\xfdsuffix")
    assert tree_scan._contains_embedded_archive(carrier)


def test_embedded_archive_probe_fails_closed_at_input_limit(tmp_path: Path) -> None:
    """Reject a plausible gzip header whose optional name exhausts the probe."""
    carrier = tmp_path / "carrier.bin"
    gzip_header_with_name = b"\x1f\x8b\x08\x08" + (b"\0" * 4) + b"\0\xff"
    carrier.write_bytes(
        gzip_header_with_name + (b"x" * tree_scan._PROBE_DECODE_CHUNK_BYTES)
    )

    assert tree_scan._contains_embedded_archive(carrier)


def test_embedded_archive_probe_fails_closed_at_candidate_limit(
    tmp_path: Path,
) -> None:
    """Bound work when a file contains many invalid compression markers."""
    carrier = tmp_path / "carrier.bin"
    invalid_gzip_header = bytes.fromhex("1f8b08b3")
    carrier.write_bytes(
        invalid_gzip_header * (tree_scan._ENCODED_ARCHIVE_MAX_CANDIDATES + 1)
    )

    assert tree_scan._contains_embedded_archive(carrier)


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_rejects_embedded_archive_in_commit_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a compressed carrier hidden after a benign commit-message prefix."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    _git(repo, "commit", "--allow-empty", "-qm", "placeholder")
    placeholder = _git(repo, "rev-parse", "HEAD")
    content = materializer.read_verified_commit(repo, placeholder)
    message_start = content.index(b"\n\n") + 2
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    binary_content = (
        content[:message_start]
        + b"benign commit prefix\n"
        + gzip.compress(f'api_key = "{token}"\n'.encode())
    )
    candidate = (
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "hash-object",
                "-t",
                "commit",
                "-w",
                "--stdin",
            ],
            input=binary_content,
            check=True,
            capture_output=True,
        )
        .stdout.decode("ascii")
        .strip()
    )
    _git(repo, "update-ref", "HEAD", candidate, placeholder)
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "Embedded archive" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_commit_message_secret_across_fragment_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject commit metadata whose key and value cross a scanner fragment."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "commit-path-boundary-control"\n'
        'description = "Commit metadata exact-path overlap control"\n'
        "regex = '''api_key\\s*=\\s*\"[A-Za-z0-9]{32}\"'''\n"
        "path = '''^[0-9a-f]{40}\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    _git(repo, "commit", "--allow-empty", "-qm", "placeholder")
    placeholder = _git(repo, "rev-parse", "HEAD")
    content = materializer.read_verified_commit(repo, placeholder)
    message_start = content.index(b"\n\n") + 2
    assignment = b"api_key =\n\n"
    filler_bytes = tree_scan._GITLEAKS_FRAGMENT_BYTES - message_start - len(assignment)
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    message = (
        (b"!" * (filler_bytes - 1))
        + b"\n"
        + assignment
        + b'"'
        + token.encode()
        + b'"\n'
    )
    message_path = tmp_path / "message.txt"
    message_path.write_bytes(message)
    _git(
        repo,
        "commit",
        "--amend",
        "--allow-empty",
        "-q",
        "-F",
        str(message_path),
    )
    candidate = _git(repo, "rev-parse", "HEAD")
    exact_content = materializer.read_verified_commit(repo, candidate)
    assert (
        exact_content.index(b'"', message_start) == tree_scan._GITLEAKS_FRAGMENT_BYTES
    )
    assert exact_content.index(token.encode()) == tree_scan._GITLEAKS_FRAGMENT_BYTES + 1
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "commit-path-boundary-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_binary_secret_across_gitleaks_fragment_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a binary secret split at the pinned scanner's fragment edge."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    path = b"carrier.bin"
    assignment = b"\napi_key =\n\n"
    wrapper_offset = (
        len(materializer.SCAN_PREFIX) + len(path) + 2 + len(materializer._SCAN_BOUNDARY)
    )
    filler_size = tree_scan._GITLEAKS_FRAGMENT_BYTES - wrapper_offset - len(assignment)
    payload = (
        b"\0" + (b"A" * (filler_size - 1)) + assignment + b'"' + token.encode() + b'"\n'
    )
    (repo / os.fsdecode(path)).write_bytes(payload)
    candidate = _commit(repo, "candidate binary")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "generic-api-key" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_archive_secret_across_gitleaks_fragment_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a secret split at a decompressed archive-member fragment edge."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    assignment = b"\napi_key =\n\n"
    filler_bytes = tree_scan._GITLEAKS_FRAGMENT_BYTES - len(assignment)
    payload = (
        b"\0"
        + (b"!" * (filler_bytes - 1))
        + assignment
        + b'"'
        + token.encode()
        + b'"\n'
    )
    with zipfile.ZipFile(
        repo / "carrier.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("carrier.bin", payload)
    candidate = _commit(repo, "candidate archive")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "generic-api-key" in output.err


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
            base,
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


@pytest.mark.parametrize(
    "diagnostic",
    [
        b'7:00AM WRN could not open root archive error="unexpected EOF"\n',
        b"7:00AM ERR archive extraction failed\n",
        b"7:00AM INF skipping archive: exceeds max archive depth\n",
    ],
)
def test_scanner_diagnostics_fail_closed(diagnostic: bytes) -> None:
    """Do not accept a clean status after scanner coverage diagnostics."""
    result = subprocess.CompletedProcess(
        args=["gitleaks"],
        returncode=0,
        stdout=b"",
        stderr=diagnostic,
    )

    with pytest.raises(tree_scan._ScanError, match="diagnostic"):
        tree_scan._reject_scanner_diagnostics(result, [])


def test_expected_finding_summary_is_not_a_coverage_warning() -> None:
    """Permit the scanner's normal warning when its report contains a leak."""
    result = subprocess.CompletedProcess(
        args=["gitleaks"],
        returncode=1,
        stdout=b"",
        stderr=b"7:00AM WRN leaks found: 1\n",
    )

    tree_scan._reject_scanner_diagnostics(result, [{"RuleID": "control"}])


def test_gitleaks_report_has_a_memory_bound(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a candidate-amplified report before loading it into memory."""
    report = tmp_path / "report.json"
    report.write_text("[]\n", encoding="utf-8")
    monkeypatch.setattr(tree_scan, "_MAX_GITLEAKS_REPORT_BYTES", 2)

    with pytest.raises(tree_scan._ScanError, match="bounded byte"):
        tree_scan._load_findings(report)


def test_gitleaks_reports_have_one_aggregate_invocation_budget(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound cumulative reports across sharded exact-path scanner processes."""
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text("[]", encoding="utf-8")
    second.write_text("[]", encoding="utf-8")
    monkeypatch.setattr(tree_scan, "_MAX_GITLEAKS_REPORT_TOTAL_BYTES", 3)
    budget = tree_scan._ReportBudget()

    assert tree_scan._load_findings(first, budget) == []
    with pytest.raises(tree_scan._ScanError, match="aggregate byte"):
        tree_scan._load_findings(second, budget)


def test_gitleaks_report_is_bounded_while_scanner_runs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply the report ceiling to the child write, not only the later load."""
    scanner = tmp_path / "scanner"
    scanner.write_text(
        "#!/usr/bin/python3\n"
        "import pathlib, sys\n"
        "report = pathlib.Path(sys.argv[sys.argv.index('--report-path') + 1])\n"
        "report.write_bytes(b'[' + (b'{}' * 1000000) + b']')\n",
        encoding="utf-8",
    )
    scanner.chmod(0o700)
    source = tmp_path / "source"
    source.mkdir()
    config = tmp_path / "config.toml"
    config.write_text("", encoding="utf-8")
    ignore = tmp_path / "ignore"
    ignore.write_text("", encoding="utf-8")
    report = tmp_path / "report.json"
    monkeypatch.setattr(tree_scan, "_MAX_GITLEAKS_REPORT_BYTES", 1024)

    with pytest.raises((json.JSONDecodeError, tree_scan._ScanError)):
        tree_scan._run_gitleaks(
            str(scanner),
            source,
            config,
            ignore,
            report,
        )

    assert report.stat().st_size <= tree_scan._MAX_GITLEAKS_REPORT_BYTES


def test_inert_encoded_archive_exception_is_exact(tmp_path: Path) -> None:
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

    prefixed_zip = (b"J" * 9_000) + _zip_member("payload.txt", b"safe\n")
    with pytest.raises(tree_scan._ScanError, match="Encoded archive"):
        tree_scan._reject_encoded_archives(base64.b64encode(prefixed_zip))

    escaped_zip = b"".join(
        f"\\u{value:04x}".encode("ascii")
        for value in _zip_member("payload.txt", b"safe\n")
    )
    with pytest.raises(tree_scan._ScanError, match="Encoded archive"):
        tree_scan._reject_encoded_archives(escaped_zip)

    source = tmp_path / "source"
    source.mkdir()
    delayed_zip = base64.b64encode(
        (b"J" * 110_000) + _zip_member("payload.txt", b"safe\n")
    )
    assert len(delayed_zip) > tree_scan._GITLEAKS_FRAGMENT_BYTES
    (source / "delayed.txt").write_bytes(delayed_zip)
    with pytest.raises(tree_scan._ScanError, match="Encoded archive"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_archive_validation_bounds_member_count(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject archives whose entry count exceeds the explicit budget."""
    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_MEMBERS", 2)
    archive = tmp_path / "many.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        for index in range(3):
            bundle.writestr(f"{index}.txt", b"safe\n")

    with pytest.raises(tree_scan._ScanError, match="member count"):
        tree_scan._validate_archive_file(archive)


def test_archive_validation_bounds_member_and_aggregate_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound both one member and the total decompressed archive payload."""
    member_archive = tmp_path / "member.zip"
    with zipfile.ZipFile(
        member_archive, "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        bundle.writestr("large.txt", b"A" * 17)
    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_MEMBER_BYTES", 16)
    with pytest.raises(tree_scan._ScanError, match="member.*byte"):
        tree_scan._validate_archive_file(member_archive)

    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_MEMBER_BYTES", 128)
    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_TOTAL_BYTES", 120)
    aggregate_archive = tmp_path / "aggregate.zip"
    with zipfile.ZipFile(
        aggregate_archive,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as bundle:
        bundle.writestr("first.txt", b"A" * 72)
        bundle.writestr("second.txt", b"B" * 72)
    with pytest.raises(tree_scan._ScanError, match="aggregate decompressed"):
        tree_scan._validate_archive_file(aggregate_archive)


def test_archive_validation_bounds_compressed_tar_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound decompressed TAR headers before the parser can allocate them."""
    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_MEMBER_BYTES", 8 * 1024)
    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_TOTAL_BYTES", 1024)
    archive = tmp_path / "metadata.tar.gz"
    member = tarfile.TarInfo("a" * 2048)
    member.size = 1
    with tarfile.open(archive, "w:gz", format=tarfile.PAX_FORMAT) as bundle:
        bundle.addfile(member, io.BytesIO(b"x"))

    with pytest.raises(tree_scan._ScanError, match="aggregate structural"):
        tree_scan._validate_archive_file(archive)


def test_probe_bounds_aggregate_bytes_across_archives(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply one decompression budget across every archive in a scan source."""
    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_MEMBER_BYTES", 128)
    monkeypatch.setattr(tree_scan, "_MAX_ARCHIVE_TOTAL_BYTES", 120)
    source = tmp_path / "source"
    source.mkdir()
    for name, content in (("first.zip", b"A" * 72), ("second.zip", b"B" * 72)):
        with zipfile.ZipFile(source / name, "w") as bundle:
            bundle.writestr("payload.txt", content)

    with pytest.raises(tree_scan._ScanError, match="aggregate decompressed"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_archive_validation_rejects_encrypted_and_malformed_zip(
    tmp_path: Path,
) -> None:
    """Fail closed on encrypted, special, or structurally invalid ZIP payloads."""
    encrypted = bytearray(_zip_member("payload.txt", b"safe\n"))
    local = encrypted.index(b"PK\x03\x04")
    central = encrypted.index(b"PK\x01\x02")
    encrypted[local + 6 : local + 8] = (
        int.from_bytes(encrypted[local + 6 : local + 8], "little") | 1
    ).to_bytes(2, "little")
    encrypted[central + 8 : central + 10] = (
        int.from_bytes(encrypted[central + 8 : central + 10], "little") | 1
    ).to_bytes(2, "little")
    encrypted_path = tmp_path / "encrypted.zip"
    encrypted_path.write_bytes(encrypted)
    with pytest.raises(tree_scan._ScanError, match="Encrypted"):
        tree_scan._validate_archive_file(encrypted_path)

    special = tmp_path / "special.zip"
    link = zipfile.ZipInfo("link")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    with zipfile.ZipFile(special, "w") as bundle:
        bundle.writestr(link, b"target")
    with pytest.raises(tree_scan._ScanError, match="Special"):
        tree_scan._validate_archive_file(special)

    malformed = tmp_path / "malformed.zip"
    malformed.write_bytes(b"PK\x03\x04truncated")
    with pytest.raises(tree_scan._ScanError, match="malformed"):
        tree_scan._validate_archive_file(malformed)


def test_archive_validation_rejects_unreferenced_zip_bytes(tmp_path: Path) -> None:
    """Require local records, central records, and EOCD counts to cover all bytes."""
    canonical = _zip_member("payload.txt", b"safe\n")

    prefixed = tmp_path / "prefixed.zip"
    prefixed.write_bytes(gzip.compress(b"hidden\n") + canonical)
    with pytest.raises(
        tree_scan._ScanError,
        match="central directory|unreferenced",
    ):
        tree_scan._validate_archive_file(prefixed)

    concatenated = tmp_path / "concatenated.zip"
    concatenated.write_bytes(_zip_member("hidden.txt", b"hidden\n") + canonical)
    with pytest.raises(
        tree_scan._ScanError,
        match="central directory|unreferenced",
    ):
        tree_scan._validate_archive_file(concatenated)

    zero_count = bytearray(canonical)
    eocd = zero_count.rindex(b"PK\x05\x06")
    zero_count[eocd + 8 : eocd + 20] = b"\0" * 12
    malformed_counts = tmp_path / "zero-count.zip"
    malformed_counts.write_bytes(zero_count)
    with pytest.raises(tree_scan._ScanError, match="central directory"):
        tree_scan._validate_archive_file(malformed_counts)


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
    assert receipt["archive_decompressed_bytes"] == len(encoded)
    assert receipt["archive_files"] == 1
    assert receipt["archive_member_probe_files"] == 3
    assert receipt["archive_member_probe_bytes"] >= 3 * len(encoded)
    assert receipt["archive_exact_path_files"] == 1
    assert receipt["archive_exact_path_shards"] == 1
    assert receipt["archive_members"] == 1
    assert receipt["archive_streams"] == 1
    assert receipt["archive_structural_bytes"] > 0
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


def test_decode_probe_reuses_hardlink_analysis_and_retains_path_association(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Analyze one wrapper inode once without losing its exact logical path."""
    source = tmp_path / "source"
    source.mkdir()
    wrapper_root = source / os.fsdecode(materializer.SCAN_NON_TEST_ROOT)
    wrapper_root.mkdir()
    encoded = base64.b64encode(b"safe boundary.")
    neutral = wrapper_root / "aaa.txt"
    neutral.write_bytes(encoded)
    context = wrapper_root / "path-context" / "blob" / "carrier.txt"
    context.parent.mkdir(parents=True)
    os.link(neutral, context)
    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_MATCHES", 1)

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        tmp_path / "probe",
        (tree_scan.re.compile(r"^carrier\.txt$"),),
    )

    assert receipt["candidate_examined_matches"] == 1
    assert receipt["candidate_exact_path_files"] == 1
    exact_files = [
        path
        for path in (tmp_path / "probe" / "npa-gitleaks-candidate-exact-paths").rglob(
            "*"
        )
        if path.is_file()
    ]
    assert [path.name for path in exact_files] == ["carrier.txt"]


def test_decode_probe_bounds_distinct_matches_across_independent_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Count repeated bytes once while retaining every governed path."""
    source = tmp_path / "source"
    source.mkdir()
    encoded = base64.b64encode(b"safe boundary.")
    for name in ("a.txt", "b.txt"):
        (source / name).write_bytes(encoded)
    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_MATCHES", 1)

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        tmp_path / "probe",
        (tree_scan.re.compile(r"^[ab]\.txt$"),),
    )

    assert receipt["candidate_examined_matches"] == 1
    assert receipt["candidate_exact_path_files"] == 2
    assert receipt["candidate_duplicates"] >= 1
    exact_files = {
        path.name
        for path in (tmp_path / "probe" / "npa-gitleaks-candidate-exact-paths").rglob(
            "*"
        )
        if path.is_file()
    }
    assert exact_files == {"a.txt", "b.txt"}


def test_decode_boundary_view_contains_only_isolated_candidates(
    tmp_path: Path,
) -> None:
    """Exclude fragment windows without omitting any isolated codec stream."""
    source = tmp_path / "probe"
    source.mkdir()
    candidate = source / "decode-input-candidates-0.txt"
    candidate.write_bytes(materializer.SCAN_PREFIX + b"encoded\n\n")
    fragments = source / "npa-gitleaks-fragment-windows"
    fragments.mkdir()
    (fragments / "noise.txt").write_text("encoded", encoding="utf-8")

    boundary = tree_scan._decode_boundary_view(tmp_path, "candidate", source, 1)

    files = [path for path in boundary.rglob("*") if path.is_file()]
    assert [path.name for path in files] == [candidate.name]
    assert files[0].stat().st_ino == candidate.stat().st_ino
    with pytest.raises(tree_scan._ScanError, match="inventory is incomplete"):
        tree_scan._decode_boundary_view(tmp_path, "mismatch", source, 2)


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


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_decode_probe_never_joins_unrelated_encoded_candidates(
    tmp_path: Path,
) -> None:
    """Keep decoded values from separate source files in separate fragments."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "left.txt").write_bytes(base64.b64encode(b"LEFT_CONTROL_0123456789"))
    (source / "right.txt").write_bytes(base64.b64encode(b"RIGHT_CONTROL_0123456789"))
    destination = tmp_path / "probe"
    receipt = tree_scan._prepare_decode_probe_source(source, destination)
    config = tmp_path / "policy.toml"
    config.write_text(
        "[[rules]]\n"
        'id = "candidate-isolation-control"\n'
        'description = "Candidate isolation control"\n'
        "regex = '''LEFT_CONTROL(?s:.)*RIGHT_CONTROL'''\n",
        encoding="utf-8",
    )
    ignore = tmp_path / "policy.ignore"
    ignore.write_text("", encoding="utf-8")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    result, findings = tree_scan._run_gitleaks(
        scanner,
        destination,
        config,
        ignore,
        tmp_path / "report.json",
    )

    assert receipt["candidate_streams"] >= 2
    assert result.returncode == 0
    assert findings == []


def test_decode_probe_bounds_candidate_cardinality_and_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound candidate state and output independently of raw source bytes."""
    source = tmp_path / "source"
    source.mkdir()
    values = [
        base64.b64encode(f"candidate value {index:08d}".encode()) for index in range(3)
    ]
    (source / "matches.txt").write_bytes(b"!".join(values))
    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_MATCHES", 2)
    with pytest.raises(tree_scan._ScanError, match="bounded match count"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "match-probe")

    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_MATCHES", 100)
    monkeypatch.setattr(
        tree_scan,
        "_MAX_ENCODED_CANDIDATE_BYTES",
        len(values[0]) + 1,
    )
    with pytest.raises(tree_scan._ScanError, match="aggregate bounded byte"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "byte-probe")


def test_decode_probe_materializes_fragment_boundary_windows(tmp_path: Path) -> None:
    """Keep overlap around every pinned scanner split in sub-fragment text files."""
    source = tmp_path / "source"
    source.mkdir()
    boundary = tree_scan._GITLEAKS_FRAGMENT_BYTES
    payload = b"A" * (boundary - len(b"LEFT\n\n")) + b"LEFT\n\nRIGHT" + b"B" * boundary
    (source / "carrier.bin").write_bytes(payload)
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(source, destination)

    windows = sorted(
        path
        for path in (destination / "npa-gitleaks-fragment-windows").rglob("*")
        if path.is_file()
    )
    assert receipt["fragment_window_files"] == 2
    assert receipt["fragment_window_bytes"] == sum(
        window.stat().st_size for window in windows
    )
    assert len(windows) == 2
    first = windows[0].read_bytes()
    assert first.startswith(materializer.SCAN_PREFIX)
    assert b"LEFT\n\nRIGHT" in first
    assert len(first) < tree_scan._GITLEAKS_FRAGMENT_BYTES


def test_fragment_boundary_peek_reexamines_trailing_byte(tmp_path: Path) -> None:
    """Match Gitleaks 8.28.0's first peek-loop inspection of buffered data."""
    carrier = tmp_path / "carrier.bin"
    carrier.write_bytes(
        (b"A" * (tree_scan._GITLEAKS_FRAGMENT_BYTES - 1)) + b"\n\nX\n\nY"
    )

    assert tree_scan._gitleaks_fragment_boundaries(carrier) == [
        tree_scan._GITLEAKS_FRAGMENT_BYTES + 1
    ]


def test_fragment_boundaries_preserve_bufio_peek_leftovers(tmp_path: Path) -> None:
    """Model the short read produced by Gitleaks' buffered safe-boundary peek."""
    carrier = tmp_path / "carrier.bin"
    carrier.write_bytes(
        (b"A" * tree_scan._GITLEAKS_FRAGMENT_BYTES)
        + b"\n\n"
        + (b"B" * 4092)
        + b"\n\n"
        + (b"C" * 200_000)
    )

    assert tree_scan._gitleaks_fragment_boundaries(carrier)[:2] == [
        tree_scan._GITLEAKS_FRAGMENT_BYTES + 2,
        tree_scan._GITLEAKS_FRAGMENT_BYTES + 4096,
    ]


def test_decode_probe_materializes_archive_member_fragment_windows(
    tmp_path: Path,
) -> None:
    """Expose overlap around scanner splits in decompressed archive members."""
    source = tmp_path / "source"
    source.mkdir()
    boundary = tree_scan._GITLEAKS_FRAGMENT_BYTES
    payload = b"!" * (boundary - len(b"LEFT\n\n")) + b"LEFT\n\nRIGHT" + b"!" * boundary
    with zipfile.ZipFile(
        source / "carrier.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("carrier.bin", payload)
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(source, destination)

    all_windows = sorted(
        path
        for path in (destination / "npa-gitleaks-fragment-windows").rglob("*")
        if path.is_file()
    )
    exact_windows = sorted(
        path
        for path in (destination / "npa-gitleaks-fragment-exact-paths").rglob("*")
        if path.is_file()
    )
    assert receipt["archive_fragment_window_files"] == len(all_windows)
    assert receipt["archive_fragment_window_bytes"] == sum(
        window.stat().st_size for window in all_windows
    )
    assert len(all_windows) >= 4
    assert any(b"LEFT\n\nRIGHT" in window.read_bytes() for window in all_windows)
    assert all(
        window.read_bytes().startswith(materializer.SCAN_PREFIX)
        for window in all_windows
    )
    assert all(
        window.stat().st_size < tree_scan._GITLEAKS_FRAGMENT_BYTES
        for window in all_windows
    )
    assert receipt["fragment_exact_path_files"] == len(exact_windows)
    assert receipt["fragment_exact_path_bytes"] == sum(
        window.stat().st_size for window in exact_windows
    )
    assert exact_windows
    assert all(window.name == "carrier.zip!carrier.bin" for window in exact_windows)


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


def test_decode_probe_rejects_oversized_archive_member_encoding(
    tmp_path: Path,
) -> None:
    """Apply the encoded-span bound inside decompressed archive members."""
    source = tmp_path / "source"
    source.mkdir()
    oversized = base64.b64encode(
        b"safe boundary 0123456789 " * (tree_scan._PROBE_STREAM_MAX_BYTES // 20)
    )
    with zipfile.ZipFile(source / "oversized.zip", "w") as archive:
        archive.writestr("oversized.txt", oversized)

    with pytest.raises(tree_scan._ScanError, match="complete Gitleaks scan fragment"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_decode_probe_rejects_malformed_archive_member_unicode(
    tmp_path: Path,
) -> None:
    """Do not discard malformed strongly identified Unicode in an archive."""
    source = tmp_path / "source"
    source.mkdir()
    with zipfile.ZipFile(source / "malformed.zip", "w") as archive:
        archive.writestr("malformed.bin", b"\xff\xfeA\x00B")

    with pytest.raises(
        tree_scan.materializer.SnapshotError,
        match="Unicode text is malformed",
    ):
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
        label in output.err
        for label in (
            "committed-history",
            "historical-trees",
            "historical-exact-path",
            "archive-exact-path",
            "historical-probe",
        )
    )


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_preserves_historical_anchored_path_associations(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Detect transformed bytes only while they occupy an anchored logical path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "anchored-history-control"\n'
        'description = "Anchored historical path control"\n'
        "regex = '''NPA_PATH_SECRET_[A-Z0-9]{24}'''\n"
        "path = '''^b/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_PATH_SECRET_" + ("A1" * 12)
    first = repo / "a" / "key.pem"
    first.parent.mkdir()
    first.write_bytes(token.encode("utf-16le"))
    _commit(repo, "first path")
    second = repo / "b" / "key.pem"
    second.parent.mkdir()
    first.rename(second)
    _commit(repo, "anchored path")
    second.unlink()
    candidate = _commit(repo, "delete anchored path")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "anchored-history-control" in output.err


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
    monkeypatch: pytest.MonkeyPatch,
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
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _commit(repo, "candidate")
    monkeypatch.setattr(
        tree_scan,
        "_verified_executable",
        lambda executable: executable,
    )
    monkeypatch.setattr(
        tree_scan,
        "_policy_commit_allowlists",
        lambda _content: {intermediate},
    )

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
    monkeypatch: pytest.MonkeyPatch,
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
        + _CONTROLLED_SECRET_CONTROLS
        + f"mode = {candidate_mode!r}\n"
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
    _trust_test_scanner(monkeypatch, scanner)

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
    assert receipt["fragment_boundary_control"] == "passed"
    assert receipt["history_base"] == base
    assert receipt["history_commits"] == [candidate]
    assert receipt["history_findings"] == 0
    assert 0 < receipt["gitleaks_report_bytes"]
    assert receipt["gitleaks_report_bytes"] < receipt["gitleaks_report_byte_limit"]
    assert receipt["historical_tree_findings"] == 0
    assert receipt["historical_tree_scan"]["commits"] == []
    assert receipt["historical_tree_scan"]["scan_files"] == 0
    assert receipt["tree_findings"] == 0
    assert receipt["commit_metadata_scan"]["commits"] == [candidate]
    assert receipt["commit_metadata_scan"]["files"] == 1
    assert receipt["commit_metadata_findings"] == 0
    assert receipt["candidate_probe_findings"] == 0
    assert receipt["historical_probe_findings"] == 0
    assert receipt["commit_metadata_probe_findings"] == 0
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
    monkeypatch: pytest.MonkeyPatch,
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
        + _CONTROLLED_SECRET_CONTROLS
        + f"if config == {base_policy!r} and ignore == 'base-ignore\\n':\n"
        "    report.write_text(json.dumps([{'RuleID': 'generic-api-key', "
        "'Secret': 'REDACTED', 'File': 'source.txt', 'StartLine': 1}]))\n"
        "    raise SystemExit(1)\n"
        "report.write_text('[]')\n",
        encoding="utf-8",
    )
    scanner.chmod(0o700)
    _trust_test_scanner(monkeypatch, scanner)

    candidate_arguments = _runner_arguments(repo, candidate, base, str(scanner))
    assert tree_scan.main(candidate_arguments) == 1
    assert "committed-history" in capsys.readouterr().err
    candidate_arguments[candidate_arguments.index("--policy-revision") + 1] = candidate
    assert tree_scan.main(candidate_arguments) == 1
    assert "Policy revision must equal or precede" in capsys.readouterr().err


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
        + _CONTROLLED_SECRET_CONTROLS
        + "report.write_text('[]')\n",
        encoding="utf-8",
    )
    trusted.chmod(0o700)
    _trust_test_scanner(monkeypatch, trusted)
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


def test_runner_rejects_self_attested_gitleaks_digest(tmp_path: Path) -> None:
    """Do not trust an arbitrary executable that prints the pinned version."""
    scanner = tmp_path / "self-attested-gitleaks"
    scanner.write_text(
        f"#!/usr/bin/python3\nprint({tree_scan._GITLEAKS_VERSION!r})\n",
        encoding="utf-8",
    )
    scanner.chmod(0o700)

    with pytest.raises(tree_scan._ScanError, match="trusted release digest"):
        tree_scan._verify_gitleaks_identity(str(scanner))


def test_runner_rejects_unpinned_gitleaks_version(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
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
    _trust_test_scanner(monkeypatch, scanner)

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
