"""Verify Gitleaks scans exact committed blobs without checkout transformations."""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable
import base64
import bz2
import gzip
import hashlib
import importlib.util
import io
import json
import lzma
import mmap
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


def _synthetic_signed_commit(repo: Path, parent: str) -> str:
    """Create one inert commit carrying enough SSH signature syntax for Git."""
    _git(repo, "add", "source.txt")
    tree = _git(repo, "write-tree")
    body = (
        f"tree {tree}\n"
        f"parent {parent}\n"
        "author NPA synthetic control <npa-test@example.invalid> 1700000000 +0000\n"
        "committer NPA synthetic control <npa-test@example.invalid> "
        "1700000000 +0000\n"
        "gpgsig -----BEGIN SSH SIGNATURE-----\n"
        " U1NIU0lHAAAAAQ==\n"
        " -----END SSH SIGNATURE-----\n"
        "\n"
        "synthetic signed candidate\n"
    )
    return subprocess.run(
        ["git", "-C", str(repo), "hash-object", "-t", "commit", "-w", "--stdin"],
        input=body,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()


def _configure_signature_helper(repo: Path, root: Path) -> tuple[Path, Path]:
    """Install one observable repository-local SSH signature helper."""
    marker = root / "signature-helper-invoked"
    helper = root / "signature-helper"
    helper.write_text(
        f"#!/bin/sh\nprintf invoked > {marker}\nexit 1\n",
        encoding="utf-8",
    )
    helper.chmod(0o700)
    _git(repo, "config", "log.showSignature", "true")
    _git(repo, "config", "gpg.format", "ssh")
    _git(repo, "config", "gpg.ssh.program", str(helper))
    _git(repo, "config", "gpg.ssh.allowedSignersFile", os.devnull)
    return marker, helper


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
    """Keep deterministic controls diverse and free of compression candidates."""
    tokens = tree_scan._private_control_tokens(128)

    assert tokens == tree_scan._private_control_tokens(128)
    assert len(tokens) == 128
    assert len(set(tokens)) == 128
    assert all(len(token) == 56 for token in tokens)
    assert all(len(set(token[-24::2])) == 12 for token in tokens)
    assert all(token[-24::2].isalpha() for token in tokens)
    assert all(token[-23::2].isdigit() for token in tokens)
    assert all(
        not tuple(tree_scan._embedded_compressed_streams(token)) for token in tokens
    )


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

    assert receipt["exact_path_shards"] == 4
    exact_files = {
        path.relative_to(exact_root).as_posix()
        for path in exact_root.rglob("*")
        if path.is_file()
    }
    assert exact_files == {
        "0/node",
        "1/node/child",
        "2/node",
        "3/node/child",
    }


def test_history_view_preserves_compound_archive_basename(tmp_path: Path) -> None:
    """Keep the exact basename required by Gitleaks archive identification."""
    repo = _repo(tmp_path)
    archive_path = repo / "credential.tar.gz"
    payload = b"historical archive payload\n"
    archive_path.write_bytes(_gzip_compress(_tar_member("credential.txt", payload)))
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
    [
        "BASH_ENV",
        "ENV",
        "HOME",
        "LD_LIBRARY_PATH",
        "LD_PRELOAD",
        "PYTHONPATH",
        "TMPDIR",
        "XDG_CONFIG_HOME",
    ],
)
def test_git_environment_drops_process_injection_variables(
    monkeypatch: pytest.MonkeyPatch,
    variable: str,
) -> None:
    """Do not inherit process hooks into authenticated Git object reads."""
    monkeypatch.setenv(variable, "/attacker-controlled")

    assert variable not in materializer._git_environment()


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


def _encode_percent_layers(content: bytes, layers: int) -> bytes:
    """Encode depth controls without incidental binary-container headers."""
    for _ in range(layers):
        content = b"".join(f"%{byte:02X}".encode() for byte in content)
    return content


def _gzip_compress(content: bytes) -> bytes:
    """Build deterministic gzip bytes for scanner fixtures."""
    return gzip.compress(content, mtime=0)


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


def _tar_member(name: str, content: bytes) -> bytes:
    """Return one canonical uncompressed TAR containing the supplied member."""
    encoded = io.BytesIO()
    with tarfile.open(fileobj=encoded, mode="w") as archive:
        metadata = tarfile.TarInfo(name)
        metadata.size = len(content)
        archive.addfile(metadata, io.BytesIO(content))
    return encoded.getvalue()


def _concatenated_tar_streams(
    archive_suffix: str,
    first_stream: bytes,
    second_stream: bytes,
) -> bytes:
    """Return two raw or independently compressed TAR streams."""
    if archive_suffix == ".tar":
        return first_stream + second_stream
    compressor = {
        ".tar.bz2": bz2.compress,
        ".tar.gz": _gzip_compress,
        ".tar.xz": lzma.compress,
    }[archive_suffix]
    return compressor(first_stream) + compressor(second_stream)


def _legacy_tar_member(
    name: str,
    content: bytes,
    *,
    checksum_encoding: str = "nul",
) -> bytes:
    """Return one checksum-valid V7-style TAR without a ustar magic field."""
    encoded = bytearray(_tar_member(name, content))
    encoded[257:512] = b"\0" * (512 - 257)
    if checksum_encoding == "negative-base256":
        encoded[:100] = b"\x80" * 100
    encoded[148:156] = b" " * 8
    unsigned_checksum, signed_checksum = tarfile.calc_chksums(encoded[:512])
    if checksum_encoding == "nul":
        encoded[148:156] = f"{unsigned_checksum:07o}\0".encode("ascii")
    elif checksum_encoding == "ascii":
        encoded[148:156] = f"{unsigned_checksum:08o}".encode("ascii")
    elif checksum_encoding == "whitespace":
        encoded[148:156] = f"{unsigned_checksum:06o}\n ".encode("ascii")
    elif checksum_encoding == "vertical-tab":
        encoded[148:156] = b"\v" + f"{unsigned_checksum:06o}".encode("ascii") + b"\v"
    elif checksum_encoding == "form-feed":
        encoded[148:156] = b"\f" + f"{unsigned_checksum:06o}".encode("ascii") + b"\f"
    elif checksum_encoding == "separator-whitespace":
        encoded[148:156] = (
            b"\x1c" + f"{unsigned_checksum:06o}".encode("ascii") + b"\x1f"
        )
    elif checksum_encoding == "underscored":
        digits = f"{unsigned_checksum:07o}".encode("ascii")
        encoded[148:156] = digits[:3] + b"_" + digits[3:]
    elif checksum_encoding == "octal-prefix":
        encoded[148:156] = f"0o{unsigned_checksum:06o}".encode("ascii")
    elif checksum_encoding == "nul-arbitrary-tail":
        encoded[148:156] = f"{unsigned_checksum:06o}\0".encode("ascii") + b"\xfe"
    elif checksum_encoding == "signed":
        encoded[148:156] = f"+{unsigned_checksum:06o}\0".encode("ascii")
    elif checksum_encoding == "base256":
        encoded[148:156] = b"\x80" + unsigned_checksum.to_bytes(7, "big")
    elif checksum_encoding == "negative-base256":
        assert signed_checksum < 0
        encoded[148:156] = b"\xff" + (signed_checksum + (256**7)).to_bytes(7, "big")
    else:
        raise ValueError(f"unsupported checksum encoding: {checksum_encoding}")
    return bytes(encoded)


def _write_encoded_archive_carrier(
    repo: Path,
    carrier: str,
    assignment: str,
) -> bool:
    """Write a ZIP hidden behind a supported text encoding."""
    if carrier not in {
        "encoded_zip",
        "double_encoded_zip",
        "encoded_zlib",
        "hex_zip",
        "percent_zip",
    }:
        return False
    if carrier == "encoded_zlib":
        (repo / "credential.txt").write_bytes(
            base64.b64encode(zlib.compress(assignment.encode()))
        )
        return True
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
        "zip_substring_alias",
        "zip_zlib",
    }:
        return False
    if carrier == "zip_substring_alias":
        (repo / "credential.zip").write_bytes(_zip_member("carrier.zip.p12", b""))
        return True
    if carrier == "zip_zlib":
        (repo / "credential.zip").write_bytes(
            _zip_member("nested.bin", zlib.compress(assignment.encode()))
        )
        return True
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
    if carrier == "zlib_disguised":
        (repo / "credential.bin").write_bytes(zlib.compress(assignment.encode()))
        return True
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
    if carrier in {"zip_comment", "zip_comment_zlib"}:
        compressor = zlib.compressobj(wbits=31)
        comment = (
            zlib.compress(assignment.encode())
            if carrier == "zip_comment_zlib"
            else compressor.compress(assignment.encode()) + compressor.flush()
        )
        with zipfile.ZipFile(repo / "credential.zip", "w") as archive:
            archive.comment = comment
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
        encoded_tar = io.BytesIO()
        with tarfile.open(
            fileobj=encoded_tar,
            mode="w",
            format=tarfile.PAX_FORMAT,
        ) as archive:
            archive.addfile(metadata, io.BytesIO(b"safe\n"))
        (repo / "credential.tar.gz").write_bytes(_gzip_compress(encoded_tar.getvalue()))
        return True
    if carrier == "tar_trailing_gzip":
        raw_tar = io.BytesIO()
        with tarfile.open(fileobj=raw_tar, mode="w") as archive:
            metadata = tarfile.TarInfo("safe.txt")
            metadata.size = len(b"safe\n")
            archive.addfile(metadata, io.BytesIO(b"safe\n"))
        (repo / "credential.tar.gz").write_bytes(
            _gzip_compress(raw_tar.getvalue() + _gzip_compress(assignment.encode()))
        )
        return True
    if carrier in {"zip_six_encoded", "tar_gz_six_encoded"}:
        encoded = _encode_percent_layers(encoded, 6)
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
    (repo / "credential.tar.gz").write_bytes(
        _gzip_compress(_tar_member("credential.txt", encoded))
    )
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


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_rejects_archive_substring_alias_for_empty_path_rule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject a name the pinned archive stack may redispatch before path rules."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    (repo / "carrier.zip.p12").write_bytes(b"")
    candidate = _commit(repo, "archive substring path alias")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
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
            _encode_percent_layers(assignment.encode(), 6)
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
        assert "archive" in err.lower()
    elif carrier in {
        "encoded_zip",
        "double_encoded_zip",
        "gzip_encoded_zip",
        "hex_zip",
        "percent_zip",
        "tar_trailing_gzip",
        "zip_comment",
        "zip_comment_zlib",
        "zip_directory_path",
        "zip_directory_payload",
        "zip_encoded_zip",
        "zip_prefixed_gzip",
        "zip_substring_alias",
    }:
        assert "archive" in err.lower()
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
        "zip_comment_zlib",
        "zip_directory_path",
        "zip_directory_payload",
        "zip_prefixed_gzip",
        "zip_utf16",
        "zip_policy_blob",
        "deep_zip",
        "encoded_zip",
        "encoded_zlib",
        "double_encoded_zip",
        "zip_encoded_zip",
        "gzip_encoded_zip",
        "zip_substring_alias",
        "zip_zlib",
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
        "zlib_disguised",
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
def test_exact_path_view_preserves_binary_content_anchors(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep absolute content anchors relative to the original blob bytes."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[[rules]]\n"
        'id = "generic-api-key"\n'
        'description = "Private scanner control"\n'
        "regex = '''(?i)api[_-]?key\\s*=\\s*[\"']?[A-Za-z0-9]{32,}'''\n\n"
        "[[rules]]\n"
        'id = "pdf-start-only"\n'
        'description = "Binary start anchor control"\n'
        "regex = '''^%PDF-1\\.7\\nNPA_PDF_SECRET_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.pdf$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_PDF_SECRET_" + ("A1" * 12)
    (repo / "carrier.pdf").write_bytes(b"%PDF-1.7\n" + token.encode() + b"\n\0%%EOF\n")
    candidate = _commit(repo, "binary anchored secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "pdf-start-only" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize("encoding", ["utf-16le", "utf-32be"])
def test_exact_path_transcoding_preserves_absolute_content_anchor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    encoding: str,
) -> None:
    """Keep source-offset-zero decoded text at exact-path byte zero."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "unicode-start-only"\n'
        'description = "Unicode start anchor control"\n'
        "regex = '''^NPA_UNICODE_START_[A-Z0-9]{24}'''\n"
        "path = '''^keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_UNICODE_START_" + ("A1" * 12)
    key = repo / "keys/key.pem"
    key.parent.mkdir()
    key.write_bytes(token.encode(encoding))
    candidate = _commit(repo, "absolute Unicode anchor")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "unicode-start-only" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_localized_transcoding_preserves_absolute_content_anchor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Keep a decoded byte-zero run anchored when later bytes are binary."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "localized-unicode-start-only"\n'
        'description = "Localized Unicode start anchor control"\n'
        "regex = '''^NPA_LOCAL_START_[A-Z0-9]{24}'''\n"
        "path = '''^keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_LOCAL_START_" + ("A1" * 12)
    key = repo / "keys/key.pem"
    key.parent.mkdir()
    key.write_bytes(token.encode("utf-16le") + (bytes(range(1, 256)) * 8))
    assert materializer._whole_unicode_encoding(key) is None
    candidate = _commit(repo, "localized absolute Unicode anchor")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "localized-unicode-start-only" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize("encoding", ["utf-16le", "utf-32be"])
def test_historical_exact_path_transcoding_preserves_absolute_content_anchor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    encoding: str,
) -> None:
    """Keep an earlier tree's decoded byte-zero text under its exact path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "historical-unicode-start-only"\n'
        'description = "Historical Unicode start anchor control"\n'
        "regex = '''^NPA_HISTORY_START_[A-Z0-9]{24}'''\n"
        "path = '''^keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_HISTORY_START_" + ("A1" * 12)
    key = repo / "keys/key.pem"
    key.parent.mkdir()
    key.write_bytes(token.encode(encoding))
    _commit(repo, "historical absolute Unicode anchor")
    key.unlink()
    candidate = _commit(repo, "delete historical Unicode carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "historical-unicode-start-only" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize("encoding", ["utf-16le", "utf-32be"])
def test_archive_exact_path_transcoding_preserves_absolute_content_anchor(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    encoding: str,
) -> None:
    """Keep decoded archive-member byte zero at exact logical-path byte zero."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "archive-unicode-start-only"\n'
        'description = "Archive Unicode start anchor control"\n'
        "regex = '''^NPA_ARCHIVE_START_[A-Z0-9]{24}'''\n"
        "path = '''^payload\\.tar!keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_ARCHIVE_START_" + ("A1" * 12)
    content = token.encode(encoding)
    member = tarfile.TarInfo("keys/key.pem")
    member.size = len(content)
    with tarfile.open(repo / "payload.tar", "w", format=tarfile.PAX_FORMAT) as archive:
        archive.addfile(member, io.BytesIO(content))
    candidate = _commit(repo, "archive absolute Unicode anchor")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "archive-unicode-start-only" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_archive_member_starting_angle_bracket_retains_exact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Do not interpret a valid archive member name as an internal sentinel."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[[rules]]\n"
        'id = "generic-api-key"\n'
        'description = "Private scanner control"\n'
        "regex = '''(?i)api[_-]?key\\s*=\\s*[\"']?[A-Za-z0-9]{32,}'''\n\n"
        "[[rules]]\n"
        'id = "angle-member-only"\n'
        'description = "Angle archive member path control"\n'
        "regex = '''NPA_ANGLE_SECRET_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.zip!<hidden\\.txt>$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_ANGLE_SECRET_" + ("A1" * 12)
    with zipfile.ZipFile(
        repo / "carrier.zip",
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.writestr("<hidden.txt>", token + "\n")
    candidate = _commit(repo, "angle member secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "angle-member-only" in output.err


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


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_forced_text_exact_history_scans_same_path_root_variants(
    tmp_path: Path,
) -> None:
    """Find every binary source-start value sharing one governed exact path."""
    views = [tmp_path / "first", tmp_path / "second"]
    tokens = [
        "NPA_ROOT_HISTORY_" + ("A1" * 12),
        "NPA_ROOT_HISTORY_" + ("B2" * 12),
    ]
    for view, token in zip(views, tokens, strict=True):
        view.mkdir()
        carrier = view / "nested dir" / "café.zip"
        carrier.parent.mkdir()
        (view / ".gitignore").write_text("*\n", encoding="utf-8")
        (view / ".gitattributes").write_text(
            "* filter=attacker ident -diff\n",
            encoding="utf-8",
        )
        carrier.write_bytes(b"$Id: " + token.encode() + b" $\0binary suffix")
    config_content = (
        "[[rules]]\n"
        'id = "root-history-control"\n'
        'description = "Root history control"\n'
        "regex = '''^\\$Id: NPA_ROOT_HISTORY_[A-Z0-9]{24} \\$'''\n"
        "path = '''^nested dir/café\\.zip$'''\n"
    ).encode()
    config = tmp_path / "policy.toml"
    config.write_bytes(config_content)
    ignore = tmp_path / "policy.ignore"
    ignore.write_text("", encoding="utf-8")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    result, findings = tree_scan._run_gitleaks_exact_views(
        scanner,
        tmp_path / "history",
        views,
        config,
        ignore,
        tmp_path / "report.json",
        config_content,
        tree_scan._ReportBudget(),
    )

    assert result.returncode == 1
    assert len(findings) == 2
    assert {finding.get("RuleID") for finding in findings} == {"root-history-control"}
    assert {finding.get("File") for finding in findings} == {"nested dir/café.zip"}
    assert len({finding.get("Commit") for finding in findings}) == 2
    assert {finding.get("StartLine") for finding in findings} == {1}


def test_exact_path_orchestration_runs_one_generated_history(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use one history scan and forbid per-view filesystem scanner processes."""
    views = [tmp_path / name for name in ("first", "second", "third")]
    config = tmp_path / "policy.toml"
    ignore = tmp_path / "policy.ignore"
    config_content = b"[extend]\nuseDefault = true\n"
    budget = tree_scan._ReportBudget()
    result = subprocess.CompletedProcess(
        args=["gitleaks"],
        returncode=0,
        stdout=b"",
        stderr=b"",
    )
    generated_history_calls: list[Path] = []

    def run_generated_history(
        executable: str,
        repository: Path,
        received_views: list[Path],
        received_config: Path,
        received_ignore: Path,
        report: Path,
        received_content: bytes,
        *,
        report_budget: tree_scan._ReportBudget,
    ) -> tuple[subprocess.CompletedProcess[bytes], list[dict[str, object]]]:
        assert executable == "/trusted/gitleaks"
        assert received_views == views
        assert received_config == config
        assert received_ignore == ignore
        assert report == tmp_path / "work" / "candidate-exact-path-git.json"
        assert received_content == config_content
        assert report_budget is budget
        generated_history_calls.append(repository)
        return result, []

    def reject_extra_scan(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("exact-path orchestration launched a per-view scan")

    monkeypatch.setattr(tree_scan, "_run_gitleaks_exact_views", run_generated_history)
    monkeypatch.setattr(tree_scan, "_run_gitleaks_exact_view", reject_extra_scan)
    monkeypatch.setattr(tree_scan, "_run_gitleaks", reject_extra_scan)

    scans = tree_scan._scan_exact_path_views(
        "/trusted/gitleaks",
        tmp_path / "work",
        views,
        "candidate-exact-path",
        config,
        ignore,
        config_content,
        budget,
    )

    assert generated_history_calls == [
        tmp_path / "work" / "candidate-exact-path-history"
    ]
    assert len(scans) == 1
    assert scans[0][0] is result
    assert scans[0][1] == []


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


def test_generated_exact_history_rejects_any_root_commit_allowlist(
    tmp_path: Path,
) -> None:
    """Reject a policy exception for either independent exact-path root."""
    views = [tmp_path / "first", tmp_path / "second"]
    for index, view in enumerate(views):
        view.mkdir()
        (view / "carrier.txt").write_text(
            f"exact history value {index}\n",
            encoding="utf-8",
        )
    commits = tree_scan._prepare_forced_text_history(
        tmp_path / "first-history",
        views,
        b"[extend]\nuseDefault = true\n",
    )
    assert len(commits) == 2
    policy = (
        b"[extend]\n"
        b"useDefault = true\n\n"
        b"[[allowlists]]\n" + f"commits = ['''{commits[1]}''']\n".encode()
    )

    with pytest.raises(
        tree_scan._ScanError,
        match="allowlist suppresses a generated exact-path",
    ):
        tree_scan._prepare_forced_text_history(
            tmp_path / "second-history",
            views,
            policy,
        )


@pytest.mark.parametrize("bound", ["views", "files", "bytes"])
def test_generated_exact_history_applies_input_bounds(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
) -> None:
    """Bound root count, file population, and streamed bytes before scanning."""
    views = [tmp_path / "first", tmp_path / "second"]
    for view in views:
        view.mkdir()
        (view / "carrier.txt").write_text("bounded content\n", encoding="utf-8")
    if bound == "views":
        monkeypatch.setattr(tree_scan, "_MAX_EXACT_PATH_SHARDS", 1)
        error = "shard limit"
    elif bound == "files":
        (views[0] / "second.txt").write_text("another file\n", encoding="utf-8")
        monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_FILES", 1)
        error = "bounded file count"
    else:
        monkeypatch.setattr(tree_scan, "_MAX_EXACT_PATH_HISTORY_BYTES", 1)
        error = "bounded byte count"

    with pytest.raises(tree_scan._ScanError, match=error):
        tree_scan._prepare_forced_text_history(
            tmp_path / "bounded-history",
            views,
            b"[extend]\nuseDefault = true\n",
        )


def test_exact_path_allocator_advances_repeated_path_monotonically(
    tmp_path: Path,
) -> None:
    """Do not restart a same-path shard search at zero for every projection."""
    allocator = tree_scan._ExactPathAllocator(tmp_path / "exact")
    parts = ("nested", "carrier.bin")

    targets = [allocator.allocate_parts(parts) for _index in range(128)]

    assert [target.parts[-3] for target in targets] == [
        str(index) for index in range(128)
    ]
    assert allocator.next_shards[parts] == 128
    assert allocator.shards == allocator.files == 128

    class PreviouslyExaminedShard(set[tuple[str, ...]]):
        def __contains__(self, value: object) -> bool:
            raise AssertionError(f"revisited occupied shard for {value!r}")

    allocator.full_paths[0] = PreviouslyExaminedShard(allocator.full_paths[0])
    next_target = allocator.allocate_parts(parts)
    assert next_target.parts[-3] == "128"
    assert allocator.next_shards[parts] == 129


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
def test_runner_preserves_pathless_absolute_anchor_for_derived_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Give every derived output a byte-zero view even without a path rule."""
    repo = _repo(tmp_path)
    config = (
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "pathless-derived-anchor-control"\n'
        'description = "Pathless derived absolute-anchor control"\n'
        "regex = '''^NPA_DERIVED_ANCHOR_[A-Z0-9]{24}$'''\n"
    )
    (repo / ".gitleaks.toml").write_text(config, encoding="utf-8")
    assert not tree_scan._policy_has_custom_rule_paths(
        config.encode(),
        content_rules_only=True,
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_DERIVED_ANCHOR_" + ("A1" * 12)
    carrier = zlib.compress(token.encode())
    assert token.encode() not in carrier
    (repo / "carrier.txt").write_bytes(carrier)
    candidate = _commit(repo, "pathless derived absolute-anchor secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None
    original_prepare = tree_scan._prepare_probe_sources
    observed: dict[str, int] = {}

    def capture_probe_sources(
        root: Path,
        plan: tree_scan._ScanPlan,
        sources: tree_scan._MaterializedSources,
    ) -> tree_scan._ProbeSources:
        result = original_prepare(root, plan, sources)
        observed.update(result.candidate_receipt)
        return result

    monkeypatch.setattr(tree_scan, "_prepare_probe_sources", capture_probe_sources)
    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "pathless-derived-anchor-control" in output.err
    assert observed["derived_exact_path_files"] >= 1
    assert observed["derived_exact_path_shards"] >= 1


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


def test_root_ignore_policy_check_conservatively_handles_re2_path() -> None:
    """Avoid reevaluating trusted RE2 expressions with Python's regex engine."""
    policy = (
        b"[[rules]]\n"
        b'id = "root-ignore-posix-control"\n'
        b"regex = '''NPA_IGNORE_ANCHORED_[A-Z0-9]{24}'''\n"
        b"path = '''^[[:punct:]]gitleaksignore$'''\n"
    )

    assert tree_scan._policy_path_matches_root_ignore(policy)


def test_custom_rule_path_association_never_compiles_with_python_re(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Treat every content-path rule conservatively, including backtracking forms."""
    policy = (
        b"[[rules]]\n"
        b'id = "re2-complexity-control"\n'
        b"regex = '''SAFE_[A-Z0-9]{24}'''\n"
        b"path = '''(a+)+$'''\n"
    )

    def reject_compile(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("trusted RE2 path was compiled with Python re")

    monkeypatch.setattr(tree_scan.re, "compile", reject_compile)
    assert tree_scan._policy_has_custom_rule_paths(
        policy,
        content_rules_only=True,
    )


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
    elif archive_suffix == ".tar.gz":
        archive_path.write_bytes(_gzip_compress(_tar_member("keys/key.pem", payload)))
    else:
        with tarfile.open(archive_path, "w") as archive:
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
@pytest.mark.parametrize(
    "archive_suffix",
    [".tar", ".tar.bz2", ".tar.gz", ".tar.xz"],
)
def test_concatenated_tar_preserves_second_stream_path_only_rule(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    archive_suffix: str,
) -> None:
    """Expose a later TAR stream to the pinned path-only PKCS#12 rule."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    first_stream = _tar_member("safe.txt", b"safe\n")
    second_stream = _tar_member(
        "keys/hidden.p12",
        b"HARMLESS_PLACEHOLDER_BYTES\n",
    )
    concatenated = _concatenated_tar_streams(
        archive_suffix,
        first_stream,
        second_stream,
    )
    (repo / f"credential{archive_suffix}").write_bytes(concatenated)
    candidate = _commit(repo, "concatenated TAR path-only carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert "HARMLESS_PLACEHOLDER_BYTES" not in output.out
    assert "HARMLESS_PLACEHOLDER_BYTES" not in output.err
    assert "pkcs12-file" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize(
    "archive_suffix",
    [".tar", ".tar.bz2", ".tar.gz", ".tar.xz"],
)
def test_concatenated_tar_preserves_second_stream_member_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    archive_suffix: str,
) -> None:
    """Do not stop member coverage at the first TAR stream's zero records."""
    repo = _repo(tmp_path)
    archive_name = f"credential{archive_suffix}"
    escaped_archive_name = archive_name.replace(".", "\\.")
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "concatenated-tar-path-control"\n'
        'description = "Concatenated TAR path control"\n'
        "regex = '''NPA_CONCAT_TAR_[A-Z0-9]{24}'''\n"
        f"path = '''^{escaped_archive_name}!keys/key\\.pem$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_CONCAT_TAR_" + ("A1" * 12)
    first_stream = _tar_member("safe.txt", b"safe\n")
    second_stream = _tar_member(
        "keys/key.pem",
        token.encode(),
    )
    concatenated = _concatenated_tar_streams(
        archive_suffix,
        first_stream,
        second_stream,
    )
    (repo / archive_name).write_bytes(concatenated)
    candidate = _commit(repo, "concatenated TAR path-scoped secret")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "concatenated-tar-path-control" in output.err


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


def test_xz_decoder_rejects_oversized_dictionary_before_allocation(
    tmp_path: Path,
) -> None:
    """Bound attacker-controlled XZ dictionary memory as well as output bytes."""
    encoded = bytearray(lzma.compress(b"bounded payload", format=lzma.FORMAT_XZ))
    assert encoded[12:20] == bytes.fromhex("0200210116000000")
    encoded[16] = 40
    encoded[20:24] = (zlib.crc32(encoded[12:20]) & 0xFFFFFFFF).to_bytes(4, "little")
    path = tmp_path / "oversized-dictionary.xz"
    path.write_bytes(encoded)

    with pytest.raises(tree_scan._ScanError, match="xz archive could not be verified"):
        tree_scan._validate_archive_file(path)


def test_memory_bounded_xz_reader_finishes_a_short_read(tmp_path: Path) -> None:
    """Return short final output instead of waiting for requested bytes forever."""
    payload = b"bounded payload"
    path = tmp_path / "short.xz"
    path.write_bytes(lzma.compress(payload, format=lzma.FORMAT_XZ))

    with tree_scan._open_memory_bounded_xz(path, "rb") as stream:
        assert stream.read(len(payload) + 1) == payload
        assert stream.read(1) == b""


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
        + b"\x78\x9cnot-a-zlib-stream"
    )

    assert not tree_scan._contains_embedded_archive(carrier)


def _png_chunk(kind: bytes, content: bytes) -> bytes:
    """Build one checksum-valid PNG chunk."""
    checksum = zlib.crc32(kind + content) & 0xFFFFFFFF
    return (
        len(content).to_bytes(4, "big") + kind + content + checksum.to_bytes(4, "big")
    )


def _png_ihdr(
    *,
    width: int = 1,
    height: int = 1,
    bit_depth: int = 8,
    color_type: int = 0,
    interlace: int = 0,
) -> bytes:
    """Build one PNG image header."""
    return (
        width.to_bytes(4, "big")
        + height.to_bytes(4, "big")
        + bytes((bit_depth, color_type, 0, 0, interlace))
    )


def _png_document(*chunks: bytes, ihdr: bytes | None = None) -> bytes:
    """Frame image chunks inside one checksum-valid PNG document."""
    return (
        tree_scan._PNG_SIGNATURE
        + _png_chunk(b"IHDR", _png_ihdr() if ihdr is None else ihdr)
        + b"".join(chunks)
        + _png_chunk(b"IEND", b"")
    )


def _strict_png(
    image_bytes: bytes = b"\0\0",
    *,
    ancillary: bytes = b"",
    split_idat: bool = False,
) -> bytes:
    """Build one strict 1x1 8-bit grayscale PNG."""
    compressed = zlib.compress(image_bytes)
    if split_idat:
        midpoint = len(compressed) // 2
        idat = _png_chunk(b"IDAT", compressed[:midpoint]) + _png_chunk(
            b"IDAT",
            compressed[midpoint:],
        )
    else:
        idat = _png_chunk(b"IDAT", compressed)
    return _png_document(ancillary, idat)


def _png_chunk_payload_ranges(
    content: bytes, kind: bytes
) -> tuple[tuple[int, int], ...]:
    """Read exact payload ranges from checksum-framed PNG chunks."""
    position = len(tree_scan._PNG_SIGNATURE)
    ranges: list[tuple[int, int]] = []
    while position + 12 <= len(content):
        length = int.from_bytes(content[position : position + 4], "big")
        data_start = position + 8
        data_end = data_start + length
        if content[position + 4 : data_start] == kind:
            ranges.append((data_start, data_end))
        position = data_end + 4
    return tuple(ranges)


def _mutate_png_chunk_payload(content: bytes, kind: bytes, offset: int) -> bytes:
    """Change one ancillary byte while repairing that chunk's CRC."""
    mutated = bytearray(content)
    position = len(tree_scan._PNG_SIGNATURE)
    while position + 12 <= len(content):
        length = int.from_bytes(content[position : position + 4], "big")
        chunk_kind = content[position + 4 : position + 8]
        data_start = position + 8
        data_end = data_start + length
        if chunk_kind == kind:
            assert 0 <= offset < length
            mutated[data_start + offset] ^= 1
            checksum = zlib.crc32(bytes(mutated[position + 4 : data_end]))
            mutated[data_end : data_end + 4] = checksum.to_bytes(4, "big")
            return bytes(mutated)
        position = data_end + 4
    raise AssertionError(f"missing PNG chunk: {kind!r}")


def _output_bearing_malformed_zlib(payload: bytes, mutation: str) -> bytes:
    """Build a zlib stream that emits payload before its terminal failure."""
    if mutation == "deflate-error":
        assert len(payload) <= 0xFFFF
        length = len(payload)
        stored = (
            b"\x00"
            + length.to_bytes(2, "little")
            + ((~length) & 0xFFFF).to_bytes(2, "little")
            + payload
            + b"\x07"
        )
        return b"\x78\x01" + stored + (b"\0" * 4)
    compressed = bytearray(zlib.compress(payload))
    if mutation == "truncated":
        return bytes(compressed[:-1])
    if mutation == "deflate-truncated":
        return bytes(compressed[:-5])
    if mutation == "checksum":
        compressed[-1] ^= 1
        return bytes(compressed)
    raise AssertionError(f"unknown zlib mutation: {mutation}")


def _fdict_zlib(payload: bytes, dictionary: bytes) -> bytes:
    """Build one RFC 1950 FDICT stream with an authenticated DICTID."""
    compressor = zlib.compressobj(
        level=9,
        wbits=zlib.MAX_WBITS,
        zdict=dictionary,
    )
    return compressor.compress(payload) + compressor.flush()


def _dictionary_independent_fdict_zlib(payload: bytes) -> bytes:
    """Set FDICT while keeping fixture output independent of dictionary history."""
    dictionary = b"public-known-dictionary-v1:" + (b"common-prefix-" * 128)
    return _fdict_zlib(payload, dictionary)


def _malformed_dictionary_independent_fdict_zlib(payload: bytes) -> bytes:
    """Corrupt the trailer after deterministic dictionary-independent output."""
    stream = bytearray(_dictionary_independent_fdict_zlib(payload))
    stream[-1] ^= 1
    return bytes(stream)


def _partial_dictionary_required_fdict_zlib(payload: bytes) -> bytes:
    """Emit the payload before requiring unavailable dictionary history."""
    return _fdict_zlib(payload + (b"A" * 100), b"A" * 32_768)


def _nested_zlib(payload: bytes, levels: int) -> bytes:
    """Wrap payload in an exact number of RFC 1950 streams."""
    for _index in range(levels):
        payload = zlib.compress(payload)
    return payload


@pytest.mark.parametrize("split_idat", [False, True])
def test_embedded_archive_probe_excludes_only_strict_png_idat(
    tmp_path: Path,
    split_idat: bool,
) -> None:
    """Treat validated image data as media without weakening other bytes."""
    carrier = tmp_path / "frame.png"
    content = _strict_png(split_idat=split_idat)
    carrier.write_bytes(content)

    ranges = tree_scan._validated_png_idat_ranges(content)
    assert ranges == _png_chunk_payload_ranges(content, b"IDAT")
    for start, end in ranges:
        assert content[start - 4 : start] == b"IDAT"
        assert int.from_bytes(content[start - 8 : start - 4], "big") == end - start
    assert not tree_scan._contains_embedded_archive(carrier)
    tree_scan._validate_archive_file(carrier)


@pytest.mark.parametrize(
    ("ihdr", "expected_size", "needs_palette"),
    [
        (_png_ihdr(bit_depth=1), 2, False),
        (_png_ihdr(bit_depth=16), 3, False),
        (_png_ihdr(bit_depth=16, color_type=2), 7, False),
        (_png_ihdr(bit_depth=4, color_type=3), 2, True),
        (_png_ihdr(bit_depth=16, color_type=4), 5, False),
        (_png_ihdr(bit_depth=16, color_type=6), 9, False),
        (_png_ihdr(width=8, height=8, color_type=6, interlace=1), 271, False),
    ],
)
def test_png_idat_exception_covers_legal_profiles(
    tmp_path: Path,
    ihdr: bytes,
    expected_size: int,
    needs_palette: bool,
) -> None:
    """Validate packed, high-depth, indexed, and Adam7 image layouts."""
    layout = tree_scan._png_image_layout(ihdr)
    assert layout is not None
    total_size, rows = layout
    assert total_size == expected_size
    image = b"".join((b"\0" + (b"\0" * width)) * count for width, count in rows)
    chunks = [_png_chunk(b"PLTE", b"\0\0\0")] if needs_palette else []
    chunks.append(_png_chunk(b"IDAT", zlib.compress(image)))
    content = _png_document(*chunks, ihdr=ihdr)
    carrier = tmp_path / "profile.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is not None
    assert not tree_scan._contains_embedded_archive(carrier)


@pytest.mark.parametrize(
    ("bit_depth", "color_type"),
    [(4, 2), (1, 4), (4, 6), (16, 3)],
)
def test_png_idat_exception_rejects_illegal_color_depths(
    bit_depth: int,
    color_type: int,
) -> None:
    """Reject color-depth pairs excluded by the PNG specification."""
    content = _png_document(
        _png_chunk(b"IDAT", zlib.compress(b"\0\0")),
        ihdr=_png_ihdr(bit_depth=bit_depth, color_type=color_type),
    )
    assert tree_scan._validated_png_idat_ranges(content) is None


@pytest.mark.parametrize(
    "mutation",
    ["prefix", "trailing", "crc", "image-size", "filter"],
)
def test_png_idat_exception_rejects_noncanonical_mutations(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Keep malformed or wrapped PNG streams inside the fail-closed boundary."""
    image_bytes = {
        "image-size": b"\0\0\0",
        "filter": b"\x05\0",
    }.get(mutation, b"\0\0")
    content = bytearray(_strict_png(image_bytes=image_bytes))
    if mutation == "prefix":
        content = bytearray(b"J" + content)
    elif mutation == "trailing":
        content.extend(b"J")
    elif mutation == "crc":
        content[29] ^= 1
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is None
    assert tree_scan._contains_embedded_archive(carrier)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(carrier)


def test_png_idat_exception_rejects_ancillary_zlib_carrier(tmp_path: Path) -> None:
    """Continue inspecting compressed bytes outside the exact IDAT ranges."""
    token_value = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    token = f'api_key = "{token_value}"\n'.encode()
    ancillary = _png_chunk(b"zTXt", b"label\0\0" + zlib.compress(token))
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(_strict_png(ancillary=ancillary))

    assert tree_scan._validated_png_idat_ranges(carrier.read_bytes()) is not None
    assert tree_scan._contains_embedded_archive(carrier)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(carrier)


def test_png_idat_exception_rejects_malformed_ancillary_zlib_carrier(
    tmp_path: Path,
) -> None:
    """Reject raw-DEFLATE output in ordinary strict-PNG ancillary bytes."""
    malformed = _output_bearing_malformed_zlib(b"hidden payload", "checksum")
    ancillary = _png_chunk(b"caBX", b"prefix" + malformed + b"suffix")
    content = _strict_png(ancillary=ancillary)
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is not None
    assert tree_scan._contains_embedded_archive(carrier)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(carrier)


@pytest.mark.parametrize(
    "payload",
    [b"prefixPK\x03\x04suffix", _tar_member("payload.txt", b"safe\n")],
    ids=["fixed-signature", "tar-header"],
)
def test_png_idat_exception_rejects_other_ancillary_containers(
    tmp_path: Path,
    payload: bytes,
) -> None:
    """Inspect fixed signatures and TAR headers outside exact IDAT ranges."""
    ancillary = _png_chunk(b"tEXt", payload)
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(_strict_png(ancillary=ancillary))

    assert tree_scan._validated_png_idat_ranges(carrier.read_bytes()) is not None
    assert tree_scan._contains_embedded_archive(carrier)


@pytest.mark.parametrize(
    ("payload", "excluded_start"),
    [
        (b"PK\x03\x04", 2),
        (_tar_member("payload.txt", b"safe\n"), 256),
    ],
    ids=["fixed-signature", "tar-header"],
)
def test_png_regions_keep_lookahead_across_idat_boundary(
    monkeypatch: pytest.MonkeyPatch,
    payload: bytes,
    excluded_start: int,
) -> None:
    """Detect immediate containers whose allowed start precedes excluded bytes."""
    prefix = b"outside-prefix"
    content = prefix + payload + b"outside-suffix"
    payload_offset = len(prefix)
    monkeypatch.setattr(
        tree_scan,
        "_validated_png_idat_ranges",
        lambda _content: (
            (payload_offset + excluded_start, payload_offset + len(payload)),
        ),
    )

    regions = tuple(tree_scan._embedded_scan_regions(content))
    assert tree_scan._regions_contain_immediate_archive(regions)


def test_png_parser_regions_keep_exact_idat_lookahead() -> None:
    """Bind parser-produced ranges to payload framing and bounded lookahead."""
    content = _strict_png(split_idat=True)
    ranges = tree_scan._validated_png_idat_ranges(content)
    assert ranges == _png_chunk_payload_ranges(content, b"IDAT")
    regions = tuple(tree_scan._embedded_scan_regions(content))
    assert len(regions) == len(ranges) + 1

    position = 0
    for (region, source_offset, start_limit), (start, end) in zip(
        regions[:-1],
        ranges,
        strict=True,
    ):
        expected_end = min(len(content), start + tree_scan._ARCHIVE_PREFIX_BYTES - 1)
        assert source_offset == position
        assert start_limit == start - position
        assert bytes(region) == content[position:expected_end]
        position = end


def test_png_regions_translate_compressed_candidate_after_idat(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probe an allowed candidate by its source offset between excluded ranges."""
    first_prefix = b"N" * 64
    first_excluded = b"X" * 32
    middle = b"M" * 17
    compressed = zlib.compress(b"decoded payload")
    candidate_offset = len(first_prefix) + len(first_excluded) + len(middle)
    content = first_prefix + first_excluded + middle + compressed + b"outside-suffix"
    monkeypatch.setattr(
        tree_scan,
        "_validated_png_idat_ranges",
        lambda _content: (
            (len(first_prefix), len(first_prefix) + len(first_excluded)),
            (candidate_offset + 1, candidate_offset + len(compressed)),
        ),
    )

    regions = tuple(tree_scan._embedded_scan_regions(content))
    assert tree_scan._regions_contain_compressed_archive(content, regions)


def test_png_parser_regions_translate_post_idat_zlib_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Translate a real parser-produced post-IDAT candidate to its source offset."""
    hidden_stream = zlib.compress(b"hidden payload")
    content = _png_document(
        _png_chunk(b"IDAT", zlib.compress(b"\0\0")),
        _png_chunk(b"tEXt", b"label\0" + hidden_stream),
    )
    ranges = tree_scan._validated_png_idat_ranges(content)
    assert ranges == _png_chunk_payload_ranges(content, b"IDAT")
    expected_offset = content.index(hidden_stream, ranges[-1][1])
    observed: list[tuple[str, int]] = []
    original_probe = tree_scan._embedded_stream_requires_rejection

    def _recording_probe(source: bytes, kind: str, offset: int) -> bool:
        observed.append((kind, offset))
        return original_probe(source, kind, offset)

    monkeypatch.setattr(
        tree_scan,
        "_embedded_stream_requires_rejection",
        _recording_probe,
    )
    regions = tuple(tree_scan._embedded_scan_regions(content))
    assert tree_scan._regions_contain_compressed_archive(content, regions)
    assert ("zlib", expected_offset) in observed


def _png_structure_mutation(mutation: str) -> bytes:
    """Build one image whose valid zlib stream has invalid PNG structure."""
    compressed = zlib.compress(b"\0\0")
    idat = _png_chunk(b"IDAT", compressed)
    if mutation == "duplicate-ihdr":
        return _png_document(_png_chunk(b"IHDR", _png_ihdr()), idat)
    if mutation == "unknown-critical":
        return _png_document(_png_chunk(b"ABCD", b""), idat)
    if mutation == "reserved-bit":
        return _png_document(_png_chunk(b"abcD", b""), idat)
    if mutation == "nonconsecutive-idat":
        return _png_document(
            idat,
            _png_chunk(b"tEXt", b"label\0value"),
            _png_chunk(b"IDAT", b""),
        )
    if mutation == "missing-palette":
        return _png_document(idat, ihdr=_png_ihdr(color_type=3))
    raise AssertionError(f"unknown PNG mutation: {mutation}")


def _png_palette_mutation(mutation: str) -> bytes:
    """Build one PNG with an invalid palette placement or shape."""
    idat = _png_chunk(b"IDAT", zlib.compress(b"\0\0"))
    palette = _png_chunk(b"PLTE", b"\0\0\0")
    if mutation == "duplicate":
        return _png_document(palette, palette, idat, ihdr=_png_ihdr(color_type=3))
    if mutation == "late":
        return _png_document(idat, palette, ihdr=_png_ihdr(color_type=3))
    if mutation == "forbidden":
        return _png_document(palette, idat)
    if mutation == "empty":
        empty = _png_chunk(b"PLTE", b"")
        return _png_document(empty, idat, ihdr=_png_ihdr(color_type=3))
    if mutation == "malformed":
        malformed = _png_chunk(b"PLTE", b"\0\0")
        return _png_document(malformed, idat, ihdr=_png_ihdr(color_type=3))
    if mutation == "oversized":
        oversized = _png_chunk(b"PLTE", b"\0" * 771)
        rgb_idat = _png_chunk(b"IDAT", zlib.compress(b"\0\0\0\0"))
        return _png_document(oversized, rgb_idat, ihdr=_png_ihdr(color_type=2))
    if mutation == "indexed-oversized":
        oversized = _png_chunk(b"PLTE", b"\0" * 9)
        header = _png_ihdr(bit_depth=1, color_type=3)
        return _png_document(oversized, idat, ihdr=header)
    raise AssertionError(f"unknown PLTE mutation: {mutation}")


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate-ihdr",
        "unknown-critical",
        "reserved-bit",
        "nonconsecutive-idat",
        "missing-palette",
    ],
)
def test_png_idat_exception_rejects_invalid_chunk_grammar(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Do not exclude image bytes when critical PNG grammar is invalid."""
    content = _png_structure_mutation(mutation)
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is None
    assert tree_scan._contains_embedded_archive(carrier)


@pytest.mark.parametrize(
    "mutation",
    [
        "duplicate",
        "late",
        "forbidden",
        "empty",
        "malformed",
        "oversized",
        "indexed-oversized",
    ],
)
def test_png_idat_exception_rejects_invalid_palette_variants(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Reject duplicate, late, forbidden, malformed, and oversized palettes."""
    content = _png_palette_mutation(mutation)
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is None
    assert tree_scan._contains_embedded_archive(carrier)


def test_png_idat_exception_handles_malformed_zlib_as_unvalidated(
    tmp_path: Path,
) -> None:
    """Return to ordinary discovery instead of leaking a decoder exception."""
    content = _png_document(_png_chunk(b"IDAT", b"\x78\x9c\xff"))
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is None
    assert not tree_scan._zlib_candidate_requires_rejection(b"\x78\x9c\xff", 0)
    assert not tree_scan._contains_embedded_archive(carrier)


@pytest.mark.parametrize(
    "mutation",
    ["truncated", "deflate-truncated", "checksum", "deflate-error"],
)
def test_png_idat_exception_rejects_output_bearing_malformed_zlib(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Fail closed when malformed IDAT emits the complete hidden plaintext."""
    malformed = _output_bearing_malformed_zlib(b"\0\0", mutation)
    content = _png_document(_png_chunk(b"IDAT", malformed))
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is None
    assert tree_scan._contains_embedded_archive(carrier)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(carrier)


@pytest.mark.parametrize("mutation", ["truncated", "trailing", "second-stream"])
def test_png_idat_exception_requires_one_complete_zlib_stream(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Reject exact-output truncation, trailing bytes, and a second stream."""
    compressed = zlib.compress(b"\0\0")
    if mutation == "truncated":
        compressed = compressed[:-1]
    elif mutation == "trailing":
        compressed += b"trailing"
    else:
        compressed += zlib.compress(b"")
    content = _png_document(_png_chunk(b"IDAT", compressed))
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)

    assert tree_scan._validated_png_idat_ranges(content) is None
    assert tree_scan._contains_embedded_archive(carrier)


def test_png_idat_exception_bounds_chunk_allocation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a PNG boundary when its structural work limit is exhausted."""
    content = _strict_png()
    carrier = tmp_path / "frame.png"
    carrier.write_bytes(content)
    monkeypatch.setattr(tree_scan, "_MAX_PNG_CHUNKS", 2)

    assert tree_scan._validated_png_idat_ranges(content) is None
    assert tree_scan._contains_embedded_archive(carrier)


def _classify_raw_zlib_candidate(
    content: bytes,
    offset: int,
) -> tuple[str, bool]:
    """Classify one RFC 1950 start independently of the production predicate."""
    fdict = bool(content[offset + 1] & 0x20)
    compressed_start = offset + (6 if fdict else 2)
    canonical = offset == 0
    if compressed_start > len(content):
        return (
            ("zlib_fdict_malformed", canonical)
            if fdict
            else (
                "zlib_inert",
                False,
            )
        )
    compressed_end = min(
        len(content),
        compressed_start + tree_scan._PROBE_DECODE_CHUNK_BYTES,
    )
    window_bits = 8 + (content[offset] >> 4)
    decoder = zlib.decompressobj(-window_bits)
    output = bytearray()

    def classified_failure(error: zlib.error) -> tuple[str, bool]:
        if not fdict:
            return ("zlib_output_error", True) if output else ("zlib_error", False)
        status = (
            "unresolved"
            if "invalid distance too far back" in str(error).lower()
            else "malformed"
        )
        emitted = "output" if output else "empty"
        return f"zlib_fdict_{status}_{emitted}", canonical or bool(output)

    for position in range(compressed_start, compressed_end):
        pending = content[position : position + 1]
        while pending:
            try:
                decoded = decoder.decompress(pending, max_length=1)
            except zlib.error as error:
                return classified_failure(error)
            output.extend(decoded)
            if len(output) > tree_scan._MAX_ENCODED_DERIVED_OUTPUT_BYTES:
                return "zlib_output_bound", True
            if decoder.eof:
                if not fdict:
                    return (
                        ("zlib_output", True)
                        if output
                        else ("zlib_complete_empty", True)
                    )
                trailer = content[position + 1 : position + 5]
                complete = len(trailer) == 4 and int.from_bytes(
                    trailer, "big"
                ) == zlib.adler32(output)
                emitted = "output" if output else "empty"
                status = "complete" if complete else "malformed"
                return (
                    f"zlib_fdict_{status}_{emitted}",
                    complete or canonical or bool(output),
                )
            remaining = decoder.unconsumed_tail
            if remaining and len(remaining) >= len(pending) and not decoded:
                return "zlib_no_progress", True
            pending = remaining
        while True:
            try:
                decoded = decoder.decompress(b"", max_length=1)
            except zlib.error as error:
                return classified_failure(error)
            if not decoded:
                break
            output.extend(decoded)
            if len(output) > tree_scan._MAX_ENCODED_DERIVED_OUTPUT_BYTES:
                return "zlib_output_bound", True
            if decoder.eof:
                if not fdict:
                    return "zlib_output", True
                trailer = content[position + 1 : position + 5]
                complete = len(trailer) == 4 and int.from_bytes(
                    trailer, "big"
                ) == zlib.adler32(output)
                status = "complete" if complete else "malformed"
                return f"zlib_fdict_{status}_output", True
    if compressed_end < len(content):
        return "zlib_input_bound", True
    emitted = "output" if output else "empty"
    if fdict:
        return f"zlib_fdict_malformed_{emitted}", canonical or bool(output)
    return (f"zlib_incomplete_{emitted}", bool(output))


def _classify_framed_candidate(candidate: bytes, kind: str) -> tuple[str, bool]:
    """Classify one gzip or bzip2 stream independently."""
    if kind == "gzip":
        decoder = zlib.decompressobj(16 + zlib.MAX_WBITS)
    else:
        assert kind == "bzip2"
        decoder = bz2.BZ2Decompressor()
    for position in range(
        0,
        len(candidate),
        tree_scan._EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES,
    ):
        chunk = candidate[
            position : position + tree_scan._EMBEDDED_ARCHIVE_PROBE_CHUNK_BYTES
        ]
        try:
            decoded = decoder.decompress(chunk, max_length=1)
        except (EOFError, OSError, zlib.error):
            return f"{kind}_error", False
        if decoded:
            return f"{kind}_output", True
        if decoder.eof:
            return f"{kind}_complete_empty", True
    bounded = len(candidate) == tree_scan._PROBE_DECODE_CHUNK_BYTES
    return (f"{kind}_bounded" if bounded else f"{kind}_inert"), bounded


def _independent_probe_classification(
    content: bytes,
    kind: str,
    offset: int,
) -> tuple[str, bool]:
    """Dispatch one independent candidate classification."""
    if kind == "zlib":
        return _classify_raw_zlib_candidate(content, offset)
    candidate = content[offset : offset + tree_scan._PROBE_DECODE_CHUNK_BYTES]
    return _classify_framed_candidate(candidate, kind)


def _exact_embedded_media_expectations() -> tuple[
    tuple[Path, int, int, dict[str, int]], ...
]:
    """Return pinned media paths and exact independent probe classifications."""
    return (
        (
            ROOT / "docs/workbench/evidence/lerobot-video-subtasks/lerobot-labeled.mp4",
            8_633,
            1_693,
            {
                "gzip_error": 1,
                "zlib_complete_empty": 2,
                "zlib_error": 3_377,
                "zlib_fdict_malformed_empty": 3_320,
                "zlib_fdict_malformed_output": 135,
                "zlib_fdict_unresolved_empty": 242,
                "zlib_fdict_unresolved_output": 749,
                "zlib_output": 17,
                "zlib_output_error": 790,
            },
        ),
        (
            ROOT
            / "docs/workbench/evidence/lerobot-video-subtasks/lerobot-video-subtasks.rrd",
            8_813,
            1_710,
            {
                "gzip_error": 1,
                "zlib_complete_empty": 3,
                "zlib_error": 3_457,
                "zlib_fdict_malformed_empty": 3_403,
                "zlib_fdict_malformed_output": 136,
                "zlib_fdict_unresolved_empty": 242,
                "zlib_fdict_unresolved_output": 758,
                "zlib_output": 18,
                "zlib_output_error": 795,
            },
        ),
    )


def _outside_png_compressed_candidates(
    content: bytes,
) -> tuple[tuple[str, int], ...]:
    """Return candidates whose starts are outside authenticated IDAT payloads."""
    candidates: list[tuple[str, int]] = []
    for region, source_offset, start_limit in tree_scan._embedded_scan_regions(content):
        candidates.extend(
            (kind, source_offset + offset)
            for kind, offset in tree_scan._embedded_compressed_streams(region)
            if offset < start_limit
        )
    return tuple(candidates)


def _assert_exact_media_fixture(
    path: Path,
    expected_candidates: int,
    expected_rejecting: int,
    expected_classifications: dict[str, int],
    probe: Callable[[bytes, str, int], bool],
) -> bytes:
    """Assert independent and production classifications for one exact file."""
    raw = path.read_bytes()
    candidates = list(tree_scan._embedded_compressed_streams(raw))
    assert len(candidates) == expected_candidates
    classifications: Counter[str] = Counter()
    for kind, offset in candidates:
        classification, independently_rejecting = _independent_probe_classification(
            raw, kind, offset
        )
        classifications[classification] += 1
        assert probe(raw, kind, offset) is independently_rejecting
    assert dict(classifications) == expected_classifications
    assert sum(probe(raw, kind, offset) for kind, offset in candidates) == (
        expected_rejecting
    )
    return raw


def _assert_exact_media_allowlists(
    fixtures: tuple[tuple[Path, int, int, dict[str, int]], ...],
) -> None:
    """Bind each admission to one complete repository-owned payload."""
    architecture = ROOT / "docs/assets/workbench-architecture.png"
    expected = {
        hashlib.sha256(path.read_bytes()).hexdigest()
        for path, _count, _rejecting, _classifications in fixtures
    } | {hashlib.sha256(architecture.read_bytes()).hexdigest()}
    assert tree_scan._INERT_EMBEDDED_MEDIA_SHA256 == expected
    assert tree_scan._INERT_STRICT_PNG_MEDIA_SHA256 == {
        hashlib.sha256(architecture.read_bytes()).hexdigest()
    }


def test_embedded_candidate_limit_exceptions_are_byte_exact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind three media files while still checking every compression candidate."""
    fixtures = _exact_embedded_media_expectations()
    _assert_exact_media_allowlists(fixtures)
    original_probe = tree_scan._embedded_stream_requires_rejection
    observed: list[tuple[str, int]] = []

    def _recording_probe(content: bytes, kind: str, offset: int) -> bool:
        observed.append((kind, offset))
        return original_probe(content, kind, offset)

    monkeypatch.setattr(
        tree_scan, "_embedded_stream_requires_rejection", _recording_probe
    )
    for (
        path,
        expected_candidates,
        expected_rejecting,
        expected_classifications,
    ) in fixtures:
        raw = _assert_exact_media_fixture(
            path,
            expected_candidates,
            expected_rejecting,
            expected_classifications,
            original_probe,
        )
        observed.clear()
        assert not tree_scan._contains_embedded_archive(path)
        assert len(observed) == expected_candidates
        mutated = tmp_path / path.name
        mutated.write_bytes(b"J" + raw)
        assert tree_scan._contains_embedded_archive(mutated)


def test_architecture_png_admission_is_exact_and_probes_ancillary_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Authenticate one public PNG after probing both ancillary candidates."""
    path = ROOT / "docs/assets/workbench-architecture.png"
    raw = path.read_bytes()
    assert hashlib.sha256(raw).hexdigest() in tree_scan._INERT_EMBEDDED_MEDIA_SHA256
    assert tree_scan._validated_png_idat_ranges(raw) is not None
    candidates = _outside_png_compressed_candidates(raw)
    assert candidates == (
        ("zlib", 2633),
        ("zlib", 2652),
        ("zlib", 4785),
        ("zlib", 8813),
        ("zlib", 9561),
        ("zlib", 9666),
        ("zlib", 10004),
    )
    original_probe = tree_scan._embedded_stream_requires_rejection
    assert [original_probe(raw, kind, offset) for kind, offset in candidates] == [
        False,
        False,
        False,
        False,
        False,
        True,
        False,
    ]
    decoder = zlib.decompressobj(-zlib.MAX_WBITS)
    assert decoder.decompress(raw[9668:], max_length=1) == b"\xa8"
    assert tree_scan._inert_embedded_media_approved(raw)

    observed: list[tuple[str, int]] = []

    def _recording_probe(content: bytes, kind: str, offset: int) -> bool:
        observed.append((kind, offset))
        return original_probe(content, kind, offset)

    monkeypatch.setattr(
        tree_scan,
        "_embedded_stream_requires_rejection",
        _recording_probe,
    )
    assert not tree_scan._contains_embedded_archive(path)
    assert observed == list(candidates)


def test_architecture_png_admission_rejects_near_and_changed_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject changed identities, path aliases, prefixes, and invalid parsing."""
    path = ROOT / "docs/assets/workbench-architecture.png"
    raw = path.read_bytes()
    candidates = _outside_png_compressed_candidates(raw)
    changed = _mutate_png_chunk_payload(raw, b"caBX", 0)
    assert tree_scan._validated_png_idat_ranges(changed) is not None
    assert _outside_png_compressed_candidates(changed) == candidates
    assert not tree_scan._inert_embedded_media_approved(changed)
    changed_path = tmp_path / "docs/assets/workbench-architecture.png"
    changed_path.parent.mkdir(parents=True)
    changed_path.write_bytes(changed)
    assert tree_scan._contains_embedded_archive(changed_path)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(changed_path)

    prefixed = tmp_path / "prefixed.png"
    prefixed.write_bytes(b"J" + raw)
    assert tree_scan._contains_embedded_archive(prefixed)

    admitted = hashlib.sha256(raw).hexdigest()
    near_match = admitted[:1] + ("0" * 63)
    assert near_match not in tree_scan._INERT_EMBEDDED_MEDIA_SHA256
    with monkeypatch.context() as partial_hash:
        partial_hash.setattr(tree_scan, "_content_sha256", lambda _content: near_match)
        assert not tree_scan._inert_embedded_media_approved(raw)

    with monkeypatch.context() as strict_png_failure:
        strict_png_failure.setattr(
            tree_scan,
            "_validated_png_idat_ranges",
            lambda _content: None,
        )
        assert not tree_scan._inert_embedded_media_approved(raw)
        assert tree_scan._contains_embedded_archive(path)


class _RecordingMemberWriter:
    """Capture the exact bytes supplied to an archive-member scan view."""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []
        self.finished = False
        self.aborted = False

    def append(self, content: bytes) -> None:
        self.chunks.append(content)

    def finish(self) -> None:
        self.finished = True

    def abort(self) -> None:
        self.aborted = True


class _ChunkedReadStream:
    """Return caller-independent short chunks before the real EOF."""

    def __init__(self, chunks: list[bytes]) -> None:
        self.chunks = iter(chunks)

    def read(self, _size: int = -1) -> bytes:
        return next(self.chunks, b"")


def _proof_bundle_rrd() -> tuple[Path, bytes]:
    """Return the exact public proof bundle and its Rerun member."""
    bundle = ROOT / "docs/workbench/evidence/lerobot-subtasks/proof-bundle.zip"
    with zipfile.ZipFile(bundle) as archive:
        return bundle, archive.read("lerobot-subtasks.rrd")


def _assert_proof_bundle_rrd_candidates(content: bytes) -> None:
    """Bind every proof-member predicate decision to an independent result."""
    candidates = list(tree_scan._embedded_compressed_streams(content))
    assert len(candidates) == 61
    classifications: Counter[str] = Counter()
    for kind, offset in candidates:
        classification, independently_rejecting = _independent_probe_classification(
            content, kind, offset
        )
        classifications[classification] += 1
        assert (
            tree_scan._embedded_stream_requires_rejection(content, kind, offset)
            is independently_rejecting
        )
    assert classifications == {
        "zlib_complete_empty": 1,
        "zlib_error": 28,
        "zlib_fdict_malformed_empty": 28,
        "zlib_fdict_malformed_output": 1,
        "zlib_fdict_unresolved_output": 2,
        "zlib_output": 1,
    }


def _record_zlib_probe_digests(
    monkeypatch: pytest.MonkeyPatch,
) -> list[str]:
    """Record every bounded candidate passed to the production zlib probe."""
    original_probe = tree_scan._zlib_candidate_requires_rejection
    observed: list[str] = []

    def _recording_probe(
        candidate: bytes,
        offset: int,
        **options: bool,
    ) -> bool:
        end = offset + tree_scan._PROBE_DECODE_CHUNK_BYTES
        observed.append(hashlib.sha256(candidate[offset:end]).hexdigest())
        return original_probe(candidate, offset, **options)

    monkeypatch.setattr(
        tree_scan,
        "_zlib_candidate_requires_rejection",
        _recording_probe,
    )
    return observed


def _alias_path_identity(
    monkeypatch: pytest.MonkeyPatch,
    alias: Path,
    target: Path,
) -> None:
    """Present one path as another through resolution and direct equality."""
    original_resolve = Path.resolve
    original_eq = Path.__eq__

    def _spoofed_resolve(path: Path, strict: bool = False) -> Path:
        if str(path) == str(alias):
            return target
        return original_resolve(path, strict=strict)

    def _spoofed_eq(left: Path, right: object) -> bool:
        if {str(left), str(right)} == {str(alias), str(target)}:
            return True
        return original_eq(left, right)

    monkeypatch.setattr(Path, "resolve", _spoofed_resolve)
    monkeypatch.setattr(Path, "__eq__", _spoofed_eq)


def test_exact_proof_bundle_member_uses_same_stream_hash_and_scan_bytes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bind exact bytes while requiring a real derived-output sink for admission."""
    bundle, content = _proof_bundle_rrd()
    identity = hashlib.sha256(content).hexdigest()
    _assert_proof_bundle_rrd_candidates(content)
    writer = _RecordingMemberWriter()
    budget = tree_scan._ArchiveBudget()
    budget.start_member(len(content))
    observed_digests = _record_zlib_probe_digests(monkeypatch)
    prefix, contains_archive, observed_identity = tree_scan._inspect_archive_stream(
        io.BytesIO(content),
        budget,
        writer,
    )
    assert contains_archive
    assert observed_identity == identity
    assert b"".join(writer.chunks) == content
    assert budget.total_bytes == len(content)
    assert writer.finished and not writer.aborted
    expected_digests = {
        hashlib.sha256(
            content[offset : offset + tree_scan._PROBE_DECODE_CHUNK_BYTES]
        ).hexdigest()
        for _kind, offset in tree_scan._embedded_compressed_streams(content)
    }
    assert len(expected_digests) == 61
    assert expected_digests <= set(observed_digests)
    with pytest.raises(tree_scan._ScanError, match="Nested archive member"):
        tree_scan._reject_nested_archive(
            "lerobot-subtasks.rrd",
            prefix,
            contains_archive=True,
        )
    with pytest.raises(tree_scan._ScanError, match="Nested archive member"):
        tree_scan._validate_archive_file(bundle)


def test_archive_stream_probes_each_final_byte() -> None:
    """Probe a nested signature whose last byte is the member's final byte."""
    content = b"A" * (tree_scan._ARCHIVE_READ_BYTES - 4) + b"PK\x03\x04"
    writer = _RecordingMemberWriter()
    budget = tree_scan._ArchiveBudget()
    budget.start_member(len(content))
    _prefix, contains_archive, observed_identity = tree_scan._inspect_archive_stream(
        io.BytesIO(content),
        budget,
        writer,
    )
    assert contains_archive
    assert observed_identity == hashlib.sha256(content).hexdigest()
    assert b"".join(writer.chunks) == content
    assert budget.total_bytes == len(content)


def test_archive_stream_reads_short_chunks_through_real_eof() -> None:
    """Do not treat a short read as EOF before a later nested signature."""
    chunks = [b"A" * 17, b"B" * 19, b"PK\x03\x04"]
    content = b"".join(chunks)
    writer = _RecordingMemberWriter()
    budget = tree_scan._ArchiveBudget()
    budget.start_member(len(content))
    _prefix, contains_archive, observed_identity = tree_scan._inspect_archive_stream(
        _ChunkedReadStream(chunks),
        budget,
        writer,
    )
    assert contains_archive
    assert observed_identity == hashlib.sha256(content).hexdigest()
    assert b"".join(writer.chunks) == content
    assert budget.total_bytes == len(content)


@pytest.mark.parametrize(
    ("suffix", "compressor"),
    [
        (".gz", _gzip_compress),
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
    ],
)
def test_archive_member_media_admission_excludes_standalone_streams(
    tmp_path: Path,
    suffix: str,
    compressor: Callable[[bytes], bytes],
) -> None:
    """Reject standalone streams expanding to the exact admitted member."""
    _bundle, content = _proof_bundle_rrd()
    carrier = tmp_path / f"member{suffix}"
    carrier.write_bytes(compressor(content))
    with pytest.raises(tree_scan._ScanError, match="Nested archive member"):
        tree_scan._validate_archive_file(carrier)


def _change_bytes_preserving_crc32(content: bytes) -> bytes:
    """Change content and solve a four-byte suffix for the original CRC-32."""
    target_crc = zlib.crc32(content)
    patch_offset = len(content) - 4
    baseline = bytearray(content)
    baseline[0] ^= 1
    baseline[patch_offset:] = b"\0" * 4
    baseline_crc = zlib.crc32(baseline)
    basis: dict[int, tuple[int, int]] = {}
    for input_bit in range(32):
        candidate = bytearray(baseline)
        candidate[patch_offset + input_bit // 8] ^= 1 << (input_bit % 8)
        value = zlib.crc32(candidate) ^ baseline_crc
        mask = 1 << input_bit
        for pivot in range(31, -1, -1):
            if not value & (1 << pivot):
                continue
            if pivot in basis:
                value ^= basis[pivot][0]
                mask ^= basis[pivot][1]
            else:
                basis[pivot] = (value, mask)
                break
    value = target_crc ^ baseline_crc
    solution = 0
    for pivot in range(31, -1, -1):
        if value & (1 << pivot):
            value ^= basis[pivot][0]
            solution ^= basis[pivot][1]
    assert value == 0
    baseline[patch_offset:] = solution.to_bytes(4, "little")
    assert zlib.crc32(baseline) == target_crc
    return bytes(baseline)


def test_changed_proof_bundle_member_retaining_candidates_rejects(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject same-path and same-CRC changed bytes retaining both collisions."""
    bundle, content = _proof_bundle_rrd()
    changed = _change_bytes_preserving_crc32(content)
    assert zlib.crc32(changed) == zlib.crc32(content)
    _assert_proof_bundle_rrd_candidates(changed)
    changed_bundle = (
        tmp_path / "docs/workbench/evidence/lerobot-subtasks" / "proof-bundle.zip"
    )
    changed_bundle.parent.mkdir(parents=True)
    with (
        zipfile.ZipFile(bundle) as source,
        zipfile.ZipFile(changed_bundle, "w") as destination,
    ):
        for member in source.infolist():
            member_content = source.read(member)
            if member.filename == "lerobot-subtasks.rrd":
                member_content = changed
            destination.writestr(member, member_content)
    _alias_path_identity(monkeypatch, changed_bundle, bundle.resolve())
    with pytest.raises(tree_scan._ScanError, match="Nested archive member"):
        tree_scan._validate_archive_file(changed_bundle)


@pytest.mark.parametrize(
    ("member", "prefix", "contains_archive", "retain_rfc1950_outputs"),
    [
        ("nested.zip", b"safe", False, True),
        ("nested.zip.txt", b"safe", False, True),
        ("safe.bin", b"PK\x03\x04", False, True),
        ("safe.bin", b"safe", True, True),
        ("<compressed-stream>", b"safe", True, False),
    ],
)
def test_archive_member_output_capability_cannot_waive_structural_rejection(
    member: str,
    prefix: bytes,
    contains_archive: bool,
    retain_rfc1950_outputs: bool,
) -> None:
    """Keep names, prefixes, non-zlib bytes, and no-sink streams fail closed."""
    with pytest.raises(tree_scan._ScanError, match="Nested archive member"):
        tree_scan._reject_nested_archive(
            member,
            prefix,
            contains_archive=contains_archive,
            retain_rfc1950_outputs=retain_rfc1950_outputs,
        )


def _record_path_swap_during_hash(
    monkeypatch: pytest.MonkeyPatch,
    carrier: Path,
    staged: Path,
) -> tuple[Callable[[bytes], str], dict[str, list[int]]]:
    """Replace a path during admission hashing and record mapped identities."""
    original_hash = tree_scan._content_sha256
    original_probe = tree_scan._embedded_stream_requires_rejection
    original_regions = tree_scan._embedded_scan_regions
    observed: dict[str, list[int]] = {"regions": [], "hash": [], "probe": []}

    def _recording_regions(content: bytes):
        assert isinstance(content, mmap.mmap)
        observed["regions"].append(id(content))
        yield from original_regions(content)

    def _replace_during_hash(content: bytes) -> str:
        observed["hash"].append(id(content))
        os.replace(staged, carrier)
        return original_hash(content)

    def _recording_probe(content: bytes, kind: str, offset: int) -> bool:
        observed["probe"].append(id(content))
        return original_probe(content, kind, offset)

    monkeypatch.setattr(tree_scan, "_embedded_scan_regions", _recording_regions)
    monkeypatch.setattr(tree_scan, "_content_sha256", _replace_during_hash)
    monkeypatch.setattr(
        tree_scan,
        "_embedded_stream_requires_rejection",
        _recording_probe,
    )
    return original_hash, observed


def test_candidate_limit_hash_and_probe_share_open_mmap(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Keep admission hashing and stream probing bound across path replacement."""
    invalid = b"\x78\x9cnot-a-zlib-stream"
    original = b"prefix" + invalid + b"middle" + invalid
    replacement = b"prefix" + invalid + b"middle" + zlib.compress(b"hidden")
    carrier = tmp_path / "carrier.bin"
    staged = tmp_path / "replacement.bin"
    carrier.write_bytes(original)
    staged.write_bytes(replacement)
    original_hash, observed = _record_path_swap_during_hash(
        monkeypatch,
        carrier,
        staged,
    )
    monkeypatch.setattr(tree_scan, "_ENCODED_ARCHIVE_MAX_CANDIDATES", 1)
    monkeypatch.setattr(
        tree_scan,
        "_INERT_EMBEDDED_MEDIA_SHA256",
        {hashlib.sha256(original).hexdigest()},
    )

    assert not tree_scan._contains_embedded_archive(carrier)
    assert carrier.read_bytes() == replacement
    assert len(observed["regions"]) == 1
    assert len(observed["hash"]) == 1
    assert set(observed["probe"]) == set(observed["hash"]) == set(observed["regions"])

    monkeypatch.setattr(tree_scan, "_content_sha256", original_hash)
    monkeypatch.setattr(tree_scan, "_ENCODED_ARCHIVE_MAX_CANDIDATES", 10)
    monkeypatch.setattr(
        tree_scan,
        "_INERT_EMBEDDED_MEDIA_SHA256",
        set(),
    )
    assert tree_scan._contains_embedded_archive(carrier)


def test_zlib_header_recognizer_covers_every_rfc1950_header() -> None:
    """Match all CM=8, bounded-window, FCHECK-valid headers including FDICT."""
    valid_headers = {
        bytes((cmf, flg))
        for cmf in range(256)
        for flg in range(256)
        if cmf & 0x0F == 8 and cmf >> 4 <= 7 and ((cmf << 8) | flg) % 31 == 0
    }
    assert len(valid_headers) == 66
    assert sum(bool(header[1] & 0x20) for header in valid_headers) == 34
    recognized_headers = {
        bytes((cmf, flg))
        for cmf in range(256)
        for flg in range(256)
        if tree_scan._zlib_header_plausible(bytes((cmf, flg)), 0)
    }
    assert recognized_headers == valid_headers
    assert tree_scan._zlib_header_plausible(bytes.fromhex("7820"), 0)
    assert not tree_scan._zlib_header_plausible(bytes.fromhex("789d"), 0)


def test_rfc1950_decoder_enforces_advertised_window() -> None:
    """Do not decode a large-distance stream under a smaller CINFO window."""
    prefix = b"".join(
        hashlib.sha256(index.to_bytes(4, "big")).digest() for index in range(16)
    )
    payload = prefix + prefix
    original = zlib.compress(payload, level=9)
    assert tree_scan._bounded_rfc1950_output(original, 0) == payload

    small_window = bytearray(original)
    cmf = 0x08
    flg = (-(cmf << 8)) % 31
    small_window[:2] = bytes((cmf, flg))
    assert tree_scan._zlib_header_plausible(small_window, 0)
    constrained = tree_scan._bounded_rfc1950_result(bytes(small_window), 0)
    assert constrained.strict_rejection
    assert constrained.output != payload


def test_fdict_retains_repository_determined_output_and_authenticates_trailer() -> None:
    """Skip DICTID and retain output that does not actually require dictionary bytes."""
    payload = b"NPA_FDICT_" + (b"Z9" * 12)
    dictionary = b"public-known-dictionary-v1:" + (b"common-prefix-" * 128)
    stream = _fdict_zlib(payload, dictionary)
    assert payload not in stream
    assert payload not in dictionary
    assert stream[1] & 0x20

    result = tree_scan._bounded_rfc1950_result(stream, 0)
    assert result == tree_scan._RFC1950DecodeResult(payload, True, "complete")

    corrupted = bytearray(stream)
    corrupted[-1] ^= 1
    malformed = tree_scan._bounded_rfc1950_result(bytes(corrupted), 0)
    assert malformed == tree_scan._RFC1950DecodeResult(
        payload,
        True,
        "malformed",
    )
    complete_analysis = tree_scan._direct_compressed_outputs(
        b"ordinary-prefix" + stream,
        max_candidates=10,
        max_emitted_bytes=1024,
    )
    assert complete_analysis.fdict_candidates == 1
    assert complete_analysis.fdict_complete_streams == 1
    assert complete_analysis.fdict_malformed_streams == 0
    malformed_analysis = tree_scan._direct_compressed_outputs(
        b"ordinary-prefix" + bytes(corrupted),
        max_candidates=10,
        max_emitted_bytes=1024,
    )
    assert malformed_analysis.fdict_candidates == 1
    assert malformed_analysis.fdict_complete_streams == 0
    assert malformed_analysis.fdict_malformed_streams == 1


def test_fdict_dictionary_required_boundary_is_explicit() -> None:
    """Reject canonical external state while accounting an arbitrary embedded start."""
    dictionary = b"common-prefix-" * 2000
    payload = b"common-prefix-" * 20
    stream = _fdict_zlib(payload, dictionary)
    decoder = zlib.decompressobj(wbits=zlib.MAX_WBITS, zdict=dictionary)
    assert decoder.decompress(stream) + decoder.flush() == payload

    result = tree_scan._bounded_rfc1950_result(stream, 0)
    assert result == tree_scan._RFC1950DecodeResult(None, True, "unresolved")
    assert tree_scan._zlib_candidate_requires_rejection(
        stream,
        0,
        canonical=True,
    )

    with pytest.raises(tree_scan._ScanError, match="external dictionary"):
        tree_scan._direct_compressed_outputs(
            stream,
            max_candidates=10,
            max_emitted_bytes=1024,
        )

    embedded = tree_scan._direct_compressed_outputs(
        b"ordinary-prefix" + stream,
        max_candidates=10,
        max_emitted_bytes=1024,
    )
    assert embedded.fdict_candidates == 1
    assert embedded.fdict_unresolved_streams == 1
    assert embedded.invalid_streams == 1
    assert not embedded.outputs


def test_arbitrary_commit_hash_fdict_header_requires_commit_sink(
    tmp_path: Path,
) -> None:
    """Reject without a sink, but retain deterministic structural-header output."""
    hash_suffix = (
        b"80ab7d33046b342\n"
        b"parent 2294dab5f5c8433f1a19f22c20927a92378be4bb\n"
        b"author NPA synthetic control <npa-test@example.invalid>\n"
    )
    result = tree_scan._bounded_rfc1950_result(hash_suffix, 0)
    assert result.fdict_status == "unresolved"
    assert result.output
    assert not tree_scan._zlib_candidate_requires_rejection(hash_suffix, 0)
    assert tree_scan._zlib_candidate_requires_rejection(
        hash_suffix,
        0,
        canonical=True,
    )

    metadata = tmp_path / "commit-metadata.txt"
    metadata.write_bytes(
        b"tree 51fed86c184a52cb244c3b45f" + hash_suffix + b"\nclean message\n"
    )
    assert tree_scan._contains_embedded_archive(metadata)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(metadata)
    probe = tmp_path / "probe"
    probe.mkdir()
    with tree_scan._CandidateProbeWriter(probe) as candidates:
        tree_scan._process_commit_metadata_compressed_boundaries(
            metadata,
            candidates,
            ("commit-metadata.txt",),
        )
        assert candidates.direct_fdict_unresolved_streams == 1
        assert candidates.derived_outputs >= 1


def test_commit_metadata_retains_rfc1950_stream_across_message_boundary(
    tmp_path: Path,
) -> None:
    """Decode a structural RFC 1950 start even when its bytes contain ``\n\n``."""
    payload = b"NPA_COMMIT_BOUNDARY_" + (b"Z9" * 12)
    stream = bytes.fromhex(
        "78da63600a0af00b708c77f6f7f5f50c8977f20ff573710c8a8c8fb2c40e0148100db2"
    )
    assert zlib.decompress(stream) == (152144).to_bytes(4, "big") + payload
    assert stream.index(b"\n\n") == 4
    assert payload not in stream
    metadata = tmp_path / "commit-metadata.txt"
    metadata.write_bytes(b"tree 51fed86c184a52cb244c3b45f\nx-npa " + stream + b"\n")
    probe = tmp_path / "probe"
    probe.mkdir()

    with tree_scan._CandidateProbeWriter(probe) as candidates:
        tree_scan._process_commit_metadata_compressed_boundaries(
            metadata,
            candidates,
            ("commit-metadata.txt",),
        )
        assert candidates.direct_rfc1950_candidates == 1
        assert candidates.direct_output_occurrences == 1
        assert candidates.direct_invalid_streams == 0
    derived = list(probe.glob("decode-derived-output-*.txt"))
    assert len(derived) == 1
    assert (152144).to_bytes(4, "big") + payload in derived[0].read_bytes()


def test_commit_metadata_direct_probe_excludes_message_only_starts(
    tmp_path: Path,
) -> None:
    """Keep direct header probing bounded to starts before the first separator."""
    metadata = tmp_path / "commit-metadata.txt"
    metadata.write_bytes(
        b"tree 51fed86c184a52cb244c3b45f\n\n"
        b"ordinary message with an invalid candidate \x78\x9c\xff\n"
    )
    probe = tmp_path / "probe"
    probe.mkdir()

    with tree_scan._CandidateProbeWriter(probe) as candidates:
        tree_scan._process_commit_metadata_compressed_boundaries(
            metadata,
            candidates,
            ("commit-metadata.txt",),
        )
        assert candidates.direct_rfc1950_candidates == 0
        assert candidates.direct_output_occurrences == 0
        assert candidates.direct_invalid_streams == 0
    assert list(probe.iterdir()) == []


@pytest.mark.parametrize(
    ("status", "compress"),
    [
        ("malformed", _malformed_dictionary_independent_fdict_zlib),
        ("unresolved", _partial_dictionary_required_fdict_zlib),
    ],
)
def test_commit_metadata_header_retains_partial_fdict_output(
    tmp_path: Path,
    status: str,
    compress: Callable[[bytes], bytes],
) -> None:
    """Scan deterministic header output while keeping message handling strict."""
    payload = b"NPA_FDICT_HEADER_" + (b"Z9" * 12)
    metadata = tmp_path / "commit-metadata.txt"
    metadata.write_bytes(
        b"tree 51fed86c184a52cb244c3b45f\nx-npa "
        + compress(payload)
        + b"\n\nclean message\n"
    )
    probe = tmp_path / "probe"
    probe.mkdir()
    with tree_scan._CandidateProbeWriter(probe) as candidates:
        tree_scan._process_commit_metadata_compressed_boundaries(
            metadata,
            candidates,
            ("commit-metadata.txt",),
        )
        assert getattr(candidates, f"direct_fdict_{status}_streams") == 1
    derived = list(probe.glob("decode-derived-output-*.txt"))
    assert derived
    assert any(payload in path.read_bytes() for path in derived)


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_commit_stream_spanning_message_boundary(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find a secret decoded from a structural stream spanning ``\n\n``."""
    repo = _repo(tmp_path)
    token = "NPA_COMMIT_BOUNDARY_" + ("Z9" * 12)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "commit-compressed-boundary-control"\n'
        'description = "Derived commit boundary control"\n'
        "regex = '''NPA_COMMIT_BOUNDARY_Z9Z9Z9Z9Z9Z9Z9Z9Z9Z9Z9Z9'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    _git(repo, "commit", "--allow-empty", "-qm", "placeholder")
    placeholder = _git(repo, "rev-parse", "HEAD")
    content = materializer.read_verified_commit(repo, placeholder)
    header = content[: content.index(b"\n\n")]
    stream = bytes.fromhex(
        "78da63600a0af00b708c77f6f7f5f50c8977f20ff573710c8a8c8fb2c40e0148100db2"
    )
    assert zlib.decompress(stream) == (152144).to_bytes(4, "big") + token.encode()
    binary_content = header + b"\nx-npa " + stream + b"\n"
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
    assert "commit-compressed-boundary-control" in output.err


def test_encoded_fdict_records_arbitrary_external_state_and_rejects_canonical() -> None:
    """Preserve offset semantics after text decoding instead of global rejection."""
    dictionary = b"common-prefix-" * 2000
    stream = _fdict_zlib(b"common-prefix-" * 20, dictionary)

    analysis = tree_scan._encoded_candidate_analysis(
        base64.b64encode(b"ordinary-prefix" + stream)
    )
    assert analysis.fdict_candidates >= 1
    assert analysis.fdict_unresolved_streams >= 1

    with pytest.raises(tree_scan._ScanError, match="Canonical encoded"):
        tree_scan._encoded_candidate_analysis(base64.b64encode(stream))


@pytest.mark.parametrize(
    "encode",
    [
        lambda content: content.hex().encode(),
        lambda content: base64.b64encode(content.hex().encode()),
    ],
    ids=["hex", "base64-hex"],
)
def test_encoded_fdict_source_offset_is_cached_with_analysis(
    tmp_path: Path,
    encode: Callable[[bytes], bytes],
) -> None:
    """Retain an offset token without authorizing identical canonical bytes."""
    payload = b"NPA_FDICT_SOURCE_OFFSET_" + (b"Z9" * 12)
    candidate = encode(_partial_dictionary_required_fdict_zlib(payload))
    assert payload not in candidate
    probe = tmp_path / "probe"
    probe.mkdir()

    with tree_scan._CandidateProbeWriter(probe) as candidates:
        candidates.append(b"ordinary-prefix\n" + candidate)
        assert candidates.encoded_fdict_unresolved_streams >= 1
        with pytest.raises(tree_scan._ScanError, match="Canonical encoded"):
            candidates.append(candidate)
    derived = list(probe.glob("decode-derived-output-*.txt"))
    assert derived
    assert any(payload in path.read_bytes() for path in derived)


@pytest.mark.parametrize("parent_levels", [1, 2])
def test_recursive_fdict_preserves_original_direct_source_offset(
    parent_levels: int,
) -> None:
    """Do not promote a recursive child reached from an arbitrary source start."""
    payload = b"NPA_RECURSIVE_FDICT_" + (b"Z9" * 12)
    child = _partial_dictionary_required_fdict_zlib(payload)
    parent = _nested_zlib(child, parent_levels)

    arbitrary = tree_scan._direct_compressed_outputs(
        b"ordinary-prefix" + parent,
        max_candidates=100,
        max_emitted_bytes=1024 * 1024,
    )

    assert arbitrary.nested_fdict_unresolved_streams >= 1
    assert arbitrary.nested_max_depth == parent_levels
    assert child in arbitrary.outputs
    assert payload in arbitrary.outputs
    with pytest.raises(tree_scan._ScanError, match="Canonical recursive"):
        tree_scan._direct_compressed_outputs(
            parent,
            max_candidates=100,
            max_emitted_bytes=1024 * 1024,
        )


def test_recursive_fdict_preserves_nonzero_child_offset_in_canonical_parent() -> None:
    """Conjoin child offset with canonical parent provenance before rejection."""
    payload = b"NPA_RECURSIVE_CHILD_OFFSET_" + (b"Z9" * 12)
    child = _partial_dictionary_required_fdict_zlib(payload)
    parent_output = b"ordinary-parent-prefix\n" + child
    parent = zlib.compress(parent_output)

    analysis = tree_scan._direct_compressed_outputs(
        parent,
        max_candidates=100,
        max_emitted_bytes=1024 * 1024,
    )

    assert analysis.nested_fdict_unresolved_streams == 1
    assert parent_output in analysis.outputs
    assert payload in analysis.outputs


def test_recursive_fdict_preserves_encoded_source_offset_and_cache(
    tmp_path: Path,
) -> None:
    """Keep recursive source capability in both encoded analysis and its cache."""
    payload = b"NPA_RECURSIVE_ENCODED_" + (b"Z9" * 12)
    child = _partial_dictionary_required_fdict_zlib(payload)
    parent = zlib.compress(child)
    candidate = base64.b64encode(parent)
    assert payload not in candidate
    probe = tmp_path / "probe"
    probe.mkdir()

    with tree_scan._CandidateProbeWriter(probe) as candidates:
        candidates.append(b"ordinary-prefix\n" + candidate)
        assert candidates.nested_fdict_unresolved_streams == 1
        with pytest.raises(tree_scan._ScanError, match="Canonical recursive"):
            candidates.append(candidate)
    derived = list(probe.glob("decode-derived-output-*.txt"))
    assert any(payload in path.read_bytes() for path in derived)


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_offset_encoded_partial_fdict_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find retained output from a text token at a nonzero source offset."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "fdict-source-offset-control"\n'
        'description = "Encoded FDICT source-offset control"\n'
        "regex = '''^NPA_FDICT_SOURCE_OFFSET_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_FDICT_SOURCE_OFFSET_" + ("Z9" * 12)
    encoded = _partial_dictionary_required_fdict_zlib(token.encode()).hex().encode()
    assert token.encode() not in encoded
    (repo / "carrier.txt").write_bytes(b"sha256 = " + encoded + b"\n")
    candidate = _commit(repo, "offset encoded FDICT carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "fdict-source-offset-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_offset_encoded_recursive_fdict_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find recursive FDICT bytes without promoting their encoded source offset."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "recursive-fdict-source-offset-control"\n'
        'description = "Recursive encoded FDICT source-offset control"\n'
        "regex = '''^NPA_RECURSIVE_FDICT_OFFSET_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_RECURSIVE_FDICT_OFFSET_" + ("Z9" * 12)
    child = _partial_dictionary_required_fdict_zlib(token.encode())
    encoded = zlib.compress(child).hex().encode()
    assert token.encode() not in encoded
    (repo / "carrier.txt").write_bytes(b"sha256 = " + encoded + b"\n")
    candidate = _commit(repo, "offset encoded recursive FDICT carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "recursive-fdict-source-offset-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize("prefix", [b"", b"ordinary-prefix\n"])
def test_runner_scans_dictionary_independent_fdict_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    prefix: bytes,
) -> None:
    """Find a nonliteral secret in FDICT output at canonical and embedded offsets."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "fdict-secret-control"\n'
        'description = "FDICT repository-determined output control"\n'
        "regex = '''NPA_FD[I]CT_[A-Z0-9]{24}'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    secret = b"NPA_FDICT_" + (b"Z9" * 12)
    dictionary = b"public-known-dictionary-v1:" + (b"common-prefix-" * 128)
    carrier = _fdict_zlib(secret, dictionary)
    assert secret not in carrier
    assert secret not in dictionary
    (repo / "carrier.txt").write_bytes(prefix + carrier)
    candidate = _commit(repo, "FDICT output carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert secret.decode() not in output.out
    assert secret.decode() not in output.err
    assert "fdict-secret-control" in output.err


def test_zlib_probe_rejects_output_before_archive_read_boundary() -> None:
    """Reject observed raw output without waiting for the wrapper's next read."""
    payload = b"".join(
        hashlib.sha256(index.to_bytes(4, "big")).digest() for index in range(1536)
    )
    compressed = zlib.compress(payload)
    assert 20 * 1024 < len(compressed) < tree_scan._ARCHIVE_READ_BYTES
    content = b"x" * (tree_scan._ARCHIVE_READ_BYTES - 20 * 1024) + compressed

    _, contains_archive, content_sha256 = tree_scan._inspect_archive_stream(
        io.BytesIO(content),
        tree_scan._ArchiveBudget(),
    )
    assert contains_archive
    assert content_sha256 == hashlib.sha256(content).hexdigest()

    bounded = tree_scan._BoundedArchiveStream(
        io.BytesIO(content),
        tree_scan._ArchiveBudget(),
    )
    with pytest.raises(tree_scan._ScanError, match="Nested archive"):
        bounded.read(tree_scan._ARCHIVE_READ_BYTES)


@pytest.mark.parametrize("rejecting_offset", [0, 2])
def test_zlib_probe_aggregates_every_candidate_result(
    monkeypatch: pytest.MonkeyPatch,
    rejecting_offset: int,
) -> None:
    """Retain a rejection from every plausible candidate in the bounded window."""
    content = b"\x78\x9c\x78\x9c"
    observed_offsets: list[int] = []

    def _controlled_probe(
        _content: bytes,
        offset: int,
        **_options: bool,
    ) -> bool:
        observed_offsets.append(offset)
        return offset == rejecting_offset

    monkeypatch.setattr(
        tree_scan,
        "_zlib_candidate_requires_rejection",
        _controlled_probe,
    )
    assert tree_scan._contains_non_tar_archive_signature(content)
    assert observed_offsets == [0, 2]


def test_archive_name_alias_matches_pinned_substring_dispatch() -> None:
    """Reject nonterminal aliases while retaining canonical compound TAR names."""
    assert tree_scan._has_nonterminal_archive_token("carrier.zip.p12")
    assert tree_scan._has_nonterminal_archive_token("carrier.zip.txt.gz")
    assert not tree_scan._has_nonterminal_archive_token("carrier.zip")
    assert not tree_scan._has_nonterminal_archive_token("carrier.tar.gz")


@pytest.mark.parametrize("compress", [gzip.compress, bz2.compress, zlib.compress])
def test_embedded_archive_probe_recognizes_decodable_streams(
    tmp_path: Path,
    compress: Callable[[bytes], bytes],
) -> None:
    """Recognize real compressed carriers across the former chunk boundary."""
    carrier = tmp_path / "carrier.bin"
    prefix = b"x" * (tree_scan._ARCHIVE_READ_BYTES - 2)
    carrier.write_bytes(prefix + compress(b"hidden payload"))

    assert tree_scan._contains_embedded_archive(carrier)


@pytest.mark.parametrize(
    "mutation",
    ["truncated", "deflate-truncated", "checksum", "deflate-error"],
)
def test_zlib_probe_rejects_output_bearing_malformed_streams(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Reject a malformed RFC 1950 carrier after its first decoded byte."""
    token = b"".join((b"Qw7eR9tY", b"2uI4oP6a", b"S8dF0gH1", b"jK3lZ5xC"))
    payload = b'api_key = "' + token + b'"\n'
    malformed = _output_bearing_malformed_zlib(payload, mutation)
    carrier = tmp_path / "carrier.bin"
    carrier.write_bytes(b"ordinary-prefix" + malformed)

    assert tree_scan._zlib_candidate_requires_rejection(malformed, 0)
    one_byte = _output_bearing_malformed_zlib(b"x", mutation)
    assert tree_scan._zlib_candidate_requires_rejection(one_byte, 0)
    assert tree_scan._contains_embedded_archive(carrier)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(carrier)


@pytest.mark.parametrize(
    "mutation",
    ["truncated", "deflate-truncated", "checksum"],
)
def test_archive_member_rejects_output_bearing_malformed_zlib(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Reject hidden malformed zlib output inside an otherwise valid ZIP."""
    malformed = _output_bearing_malformed_zlib(b"hidden payload", mutation)
    carrier = tmp_path / "carrier.zip"
    carrier.write_bytes(_zip_member("nested.bin", malformed))

    with pytest.raises(tree_scan._ScanError, match="Nested archive"):
        tree_scan._validate_archive_file(carrier)


@pytest.mark.parametrize("status", ["malformed", "unresolved"])
def test_archive_member_rejects_partial_fdict_output(
    tmp_path: Path,
    status: str,
) -> None:
    """Reject deterministic FDICT bytes where no derived-output sink exists."""
    payload = b"NPA_FDICT_PARTIAL_" + (b"Z9" * 12)
    if status == "malformed":
        encoded = _malformed_dictionary_independent_fdict_zlib(payload)
    else:
        encoded = _partial_dictionary_required_fdict_zlib(payload)
    prefix = b"ordinary-prefix"
    content = prefix + encoded
    result = tree_scan._bounded_rfc1950_result(content, len(prefix))
    assert result.fdict_status == status
    assert result.output == payload
    assert not tree_scan._zlib_candidate_requires_rejection(content, len(prefix))
    assert tree_scan._zlib_candidate_requires_rejection(
        content,
        len(prefix),
        reject_partial_fdict_output=True,
    )

    carrier = tmp_path / "carrier.zip"
    carrier.write_bytes(_zip_member("nested.bin", content))
    with pytest.raises(tree_scan._ScanError, match="Nested archive"):
        tree_scan._validate_archive_file(carrier)


@pytest.mark.parametrize("status", ["malformed", "unresolved"])
def test_embedded_archive_probe_rejects_partial_fdict_without_sink(
    tmp_path: Path,
    status: str,
) -> None:
    """Fail closed on deterministic FDICT output outside a retained scan sink."""
    payload = b"NPA_FDICT_STRICT_" + (b"Z9" * 12)
    if status == "malformed":
        encoded = _malformed_dictionary_independent_fdict_zlib(payload)
    else:
        encoded = _partial_dictionary_required_fdict_zlib(payload)
    assert payload not in encoded
    carrier = tmp_path / "carrier.bin"
    carrier.write_bytes(b"ordinary-prefix" + encoded)

    assert tree_scan._contains_embedded_archive(carrier)
    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(carrier)


@pytest.mark.parametrize(
    ("suffix", "wrap"),
    [
        (".zip", lambda content: _zip_member("nested.txt", content)),
        (".gz", gzip.compress),
        (".bz2", bz2.compress),
        (".xz", lzma.compress),
    ],
    ids=["zip", "gzip", "bzip2", "xz"],
)
@pytest.mark.parametrize("status", ["malformed", "unresolved"])
def test_archive_member_probe_retains_partial_fdict_output(
    tmp_path: Path,
    suffix: str,
    wrap: Callable[[bytes], bytes],
    status: str,
) -> None:
    """Retain deterministic member output only through a complete scan sink."""
    payload = b"NPA_FDICT_MEMBER_" + (b"Z9" * 12)
    if status == "malformed":
        encoded = _malformed_dictionary_independent_fdict_zlib(payload)
    else:
        encoded = _partial_dictionary_required_fdict_zlib(payload)
    assert payload not in encoded
    source = tmp_path / "source"
    source.mkdir()
    (source / f"carrier{suffix}").write_bytes(wrap(b"ordinary-prefix" + encoded))
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    exact = [
        path.read_bytes()
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["direct_fdict_candidates"] >= 1
    assert receipt[f"direct_fdict_{status}_streams"] >= 1
    assert receipt["direct_output_occurrences"] >= 1
    assert payload in exact


@pytest.mark.parametrize("status", ["malformed", "unresolved"])
def test_tar_probe_rejects_partial_fdict_before_member_attribution(
    tmp_path: Path,
    status: str,
) -> None:
    """Keep parser-visible TAR structure strict before member attribution."""
    payload = b"NPA_FDICT_TAR_" + (b"Z9" * 12)
    if status == "malformed":
        encoded = _malformed_dictionary_independent_fdict_zlib(payload)
    else:
        encoded = _partial_dictionary_required_fdict_zlib(payload)
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.tar").write_bytes(
        _tar_member("nested.txt", b"ordinary-prefix" + encoded)
    )

    with pytest.raises(tree_scan._ScanError, match="Nested archive bytes"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize(
    ("suffix", "logical_path", "wrap"),
    [
        (
            ".zip",
            "carrier.zip!nested.txt",
            lambda content: _zip_member("nested.txt", content),
        ),
        (".gz", "carrier.gz", gzip.compress),
    ],
    ids=["zip-member", "gzip-stream"],
)
@pytest.mark.parametrize("status", ["malformed", "unresolved"])
def test_runner_scans_partial_fdict_archive_output_at_exact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    suffix: str,
    logical_path: str,
    wrap: Callable[[bytes], bytes],
    status: str,
) -> None:
    """Find retained member output through its exact archive-derived path."""
    repo = _repo(tmp_path)
    escaped_path = logical_path.replace(".", "\\.")
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "fdict-archive-path-control"\n'
        'description = "FDICT archive-derived exact path control"\n'
        "regex = '''^NPA_FDICT_ARCHIVE_[A-Z0-9]{24}'''\n"
        f"path = '''^{escaped_path}$'''\n",
        encoding="utf-8",
    )
    (repo / "safe.txt").write_text("safe base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_FDICT_ARCHIVE_" + ("Z9" * 12)
    payload = token.encode()
    if status == "malformed":
        encoded = _malformed_dictionary_independent_fdict_zlib(payload)
    else:
        encoded = _partial_dictionary_required_fdict_zlib(payload)
    assert payload not in encoded
    (repo / f"carrier{suffix}").write_bytes(wrap(b"ordinary-prefix" + encoded))
    candidate = _commit(repo, "FDICT archive-derived path carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "fdict-archive-path-control" in output.err


def test_archive_stream_retention_capability_requires_real_sink() -> None:
    """Do not let a nominal byte writer waive strict member rejection."""
    writer = _RecordingMemberWriter()
    with pytest.raises(tree_scan._ScanError, match="requires a scan sink"):
        tree_scan._inspect_archive_stream(
            io.BytesIO(zlib.compress(b"payload")),
            tree_scan._ArchiveBudget(),
            writer,
            retain_rfc1950_outputs=True,
        )


def test_embedded_archive_probe_rejects_truncated_output_and_unsupported_streams(
    tmp_path: Path,
) -> None:
    """Fail closed after decoded output and on unsupported container bytes."""
    carrier = tmp_path / "carrier.bin"
    carrier.write_bytes(b"prefix" + _gzip_compress(b"hidden payload")[:-8])
    assert tree_scan._contains_embedded_archive(carrier)

    corrupt_checksum = bytearray(_gzip_compress(b"x"))
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


def test_embedded_archive_probe_fails_closed_at_fdict_input_limit(
    tmp_path: Path,
) -> None:
    """Reject an offset FDICT stream whose DEFLATE input exceeds the probe."""
    payload = b"x" * (tree_scan._PROBE_DECODE_CHUNK_BYTES + 1_024)
    compressor = zlib.compressobj(level=0, wbits=-zlib.MAX_WBITS)
    raw_deflate = compressor.compress(payload) + compressor.flush()
    stream = (
        b"\x78\x20"
        + (0).to_bytes(4, "big")
        + raw_deflate
        + zlib.adler32(payload).to_bytes(4, "big")
    )
    prefix = b"ordinary-prefix"
    content = prefix + stream
    carrier = tmp_path / "carrier.bin"
    carrier.write_bytes(content)

    with pytest.raises(tree_scan._ScanError, match="input exceeds"):
        tree_scan._bounded_rfc1950_result(content, len(prefix))
    assert tree_scan._zlib_candidate_requires_rejection(content, len(prefix))
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
@pytest.mark.parametrize(
    "compress",
    [
        gzip.compress,
        zlib.compress,
        _dictionary_independent_fdict_zlib,
        _malformed_dictionary_independent_fdict_zlib,
        _partial_dictionary_required_fdict_zlib,
    ],
    ids=["gzip", "zlib", "fdict", "fdict-malformed", "fdict-partial-unresolved"],
)
def test_runner_rejects_embedded_archive_in_commit_metadata(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    compress: Callable[[bytes], bytes],
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
        + compress(f'api_key = "{token}"\n'.encode())
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


def _public_inert_encoded_archive_fixtures() -> list[bytes]:
    """Recover each public compressed fixture covered by the exact allowlist."""
    fixtures = [
        bytes.fromhex(
            "1f8b08000000000000ff621805a360148c5800080000ffff2eafb5ef00040000"
        ),
        *(
            base64.b64decode(value, validate=True)
            for value in (
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/p9sAAAAASUVORK5CYII=",
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+a9RkAAAAASUVORK5CYII=",
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42YAAAAASUVORK5CYII=",
                "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jRZkAAAAASUVORK5CYII=",
            )
        ),
        bytes.fromhex(
            "0000000b4944415478da6364f80f00010501012718e3660000000049454e44ae426082"
        ),
    ]
    mcap = (
        Path(__file__).parents[1] / "browser/cypress/fixtures/sim2real_sample.mcap"
    ).read_bytes()
    base64_pattern = dict(tree_scan._ENCODING_CANDIDATES)["base64"]
    for match in base64_pattern.finditer(mcap):
        fixtures.extend(
            decoded
            for decoded in tree_scan._decode_candidate("base64", match.group())
            if decoded.startswith(b"\x89PNG\r\n\x1a\n")
        )
    return list({hashlib.sha256(value).digest(): value for value in fixtures}.values())


def test_inert_encoded_archive_exception_is_exact(tmp_path: Path) -> None:
    """Keep public non-secret compressed fixtures byte-exact."""
    fixtures = _public_inert_encoded_archive_fixtures()
    archive = fixtures[0]
    assert len(fixtures) == 8
    assert archive[3] == 0
    assert gzip.decompress(archive) == b"\0" * 1024
    assert tree_scan._INERT_ENCODED_ARCHIVE_SHA256 == {
        hashlib.sha256(value).hexdigest() for value in fixtures
    }

    for fixture in fixtures:
        tree_scan._reject_encoded_archives(fixture.hex().encode("ascii"))
        with pytest.raises(tree_scan._ScanError, match="Encoded archive"):
            tree_scan._reject_encoded_archives((b"J" + fixture).hex().encode("ascii"))

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


@pytest.mark.parametrize("archive_suffix", [".zip", ".tar"])
def test_archive_validation_rejects_empty_directory_members(
    tmp_path: Path,
    archive_suffix: str,
) -> None:
    """Fail closed when a directory's trailing-slash path cannot be scanned."""
    archive_path = tmp_path / f"directory{archive_suffix}"
    if archive_suffix == ".zip":
        with zipfile.ZipFile(archive_path, "w") as archive:
            archive.writestr("hidden.p12/", b"")
    else:
        with tarfile.open(archive_path, "w") as archive:
            directory = tarfile.TarInfo("hidden.p12/")
            directory.type = tarfile.DIRTYPE
            archive.addfile(directory)

    with pytest.raises(tree_scan._ScanError, match="directory member"):
        tree_scan._validate_archive_file(archive_path)


def test_archive_validation_rejects_regular_tar_name_with_trailing_slash(
    tmp_path: Path,
) -> None:
    """Do not let a regular TAR entry imitate an unrepresentable directory path."""
    archive_path = tmp_path / "regular-slash.tar"
    with tarfile.open(archive_path, "w") as archive:
        member = tarfile.TarInfo("hidden.p12/")
        member.type = tarfile.REGTYPE
        member.size = 4
        archive.addfile(member, io.BytesIO(b"safe"))

    with pytest.raises(tree_scan._ScanError, match="not canonical"):
        tree_scan._validate_archive_file(archive_path)


def test_archive_validation_rejects_disguised_legacy_tar(tmp_path: Path) -> None:
    """Recognize checksum-valid V7 TAR bytes without relying on ustar magic."""
    disguised = tmp_path / "legacy.bin"
    content = _legacy_tar_member("payload.txt", b"safe\n")
    disguised.write_bytes(content)
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        assert archive.getnames() == ["payload.txt"]

    with pytest.raises(tree_scan._ScanError, match="Disguised tar"):
        tree_scan._validate_archive_file(disguised)


@pytest.mark.parametrize(
    "checksum_encoding",
    [
        "nul",
        "ascii",
        "whitespace",
        "vertical-tab",
        "form-feed",
        "separator-whitespace",
        "underscored",
        "octal-prefix",
        "nul-arbitrary-tail",
        "signed",
        "base256",
        "negative-base256",
    ],
)
def test_archive_validation_rejects_prefixed_legacy_tar(
    tmp_path: Path,
    checksum_encoding: str,
) -> None:
    """Recognize a V7 TAR header hidden at a non-block-aligned offset."""
    disguised = tmp_path / "prefixed-legacy.bin"
    content = _legacy_tar_member(
        "keys/hidden.p12",
        b"safe\n",
        checksum_encoding=checksum_encoding,
    )
    with tarfile.open(fileobj=io.BytesIO(content), mode="r:") as archive:
        assert len(archive.getmembers()) == 1
    disguised.write_bytes(b"opaque prefix\n" + content)

    with pytest.raises(tree_scan._ScanError, match="Embedded archive"):
        tree_scan._validate_archive_file(disguised)


@pytest.mark.parametrize(
    "delimiter",
    [9, 10, 11, 12, 13, 28, 29, 30, 31, 32],
    ids=lambda value: f"ascii-{value:02x}",
)
def test_embedded_tar_search_covers_python_ascii_strip_whitespace(
    delimiter: int,
) -> None:
    """Search every ASCII whitespace byte accepted by Python's TAR parser."""
    content = bytearray(_legacy_tar_member("keys/hidden.p12", b"safe\n"))
    content[148:156] = b" " * 8
    checksum, _ = tarfile.calc_chksums(content[:512])
    content[148:156] = (
        bytes([delimiter]) + f"{checksum:06o}".encode("ascii") + bytes([delimiter])
    )

    assert tarfile.nti(content[148:156]) == checksum
    assert tree_scan._tar_header_checksum_valid(content)
    assert tree_scan._contains_tar_header(b"non-aligned prefix" + content)


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_complete_tree_runner_rejects_parser_only_tar_checksum_spelling(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Fail closed on the exact vertical-tab checksum bypass found in review."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    content = _legacy_tar_member(
        "keys/hidden.p12",
        b"safe\n",
        checksum_encoding="vertical-tab",
    )
    (repo / "carrier.bin").write_bytes(b"non-aligned prefix" + content)
    candidate = _commit(repo, "parser-only checksum carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert "Embedded archive bytes" in output.err


def test_embedded_tar_search_rejects_checksum_only_collision() -> None:
    """A matching checksum is not enough to classify arbitrary bytes as TAR."""
    collision = bytearray(b"A" * tree_scan._ARCHIVE_PREFIX_BYTES)
    collision[148:156] = b" " * 8
    checksum = sum(collision)
    collision[148:156] = f"{checksum:06o}\0 ".encode("ascii")
    expected = tarfile.nti(collision[148:156])
    assert expected in set(tarfile.calc_chksums(collision))

    assert not tree_scan._tar_header_checksum_valid(collision)
    assert not tree_scan._contains_tar_header(collision)


def test_embedded_tar_search_skips_impossible_checksum_runs_before_budget(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Do not spend the candidate budget on media-like numeric byte runs."""
    impossible = bytearray(b"A" * tree_scan._ARCHIVE_PREFIX_BYTES)
    impossible[100:108] = b"0000644\0"
    impossible[108:116] = b"0000000\0"
    impossible[116:124] = b"0000000\0"
    impossible[124:136] = b"00000000000\0"
    impossible[136:148] = b"00000000000\0"
    impossible[148:156] = b"0000000\0"
    impossible[156:157] = b"Y"
    monkeypatch.setattr(tree_scan, "_MAX_EMBEDDED_TAR_CHECKSUM_CANDIDATES", 0)

    assert not tree_scan._contains_tar_header(impossible)


def test_embedded_tar_checksum_search_fails_closed_at_work_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound checksum candidates without classifying a normal short run."""
    numeric = b"0" * tree_scan._ARCHIVE_PREFIX_BYTES
    assert not tree_scan._contains_tar_header(numeric)

    monkeypatch.setattr(tree_scan, "_MAX_EMBEDDED_TAR_CHECKSUM_CANDIDATES", 0)
    assert tree_scan._contains_tar_header(numeric)


def test_embedded_tar_checksum_work_is_bounded_across_stream_chunks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Aggregate candidate work instead of resetting the limit per read."""
    numeric = b"0" * tree_scan._ARCHIVE_PREFIX_BYTES
    budget = tree_scan._ArchiveBudget()
    monkeypatch.setattr(tree_scan, "_MAX_EMBEDDED_TAR_CHECKSUM_CANDIDATES", 1)

    assert not tree_scan._contains_tar_header(numeric, budget)
    assert tree_scan._contains_tar_header(numeric, budget)


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
    encoded_tar = io.BytesIO()
    with tarfile.open(
        fileobj=encoded_tar,
        mode="w",
        format=tarfile.PAX_FORMAT,
    ) as bundle:
        bundle.addfile(member, io.BytesIO(b"x"))
    archive.write_bytes(_gzip_compress(encoded_tar.getvalue()))

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
    prefixed.write_bytes(_gzip_compress(b"hidden\n") + canonical)
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
    assert receipt["archive_member_probe_files"] == 4
    assert receipt["archive_member_probe_bytes"] >= 4 * len(encoded)
    assert receipt["archive_exact_path_files"] == 2
    assert receipt["archive_exact_path_shards"] == 2
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


def test_empty_decode_probe_receipt_matches_populated_schema(tmp_path: Path) -> None:
    """Keep zero-work historical receipts schema-identical to real probe receipts."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "ordinary.txt").write_text("safe ordinary text\n", encoding="utf-8")

    populated = tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")
    empty = tree_scan._empty_probe_receipt()

    assert empty.keys() == populated.keys()
    assert all(value == 0 for value in empty.values())
    assert empty["derived_exact_path_identity_reuses"] == 0


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
        preserve_exact_paths=True,
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
        preserve_exact_paths=True,
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


@pytest.mark.parametrize(
    "mutation",
    ["truncated", "deflate-truncated", "checksum", "deflate-error"],
)
def test_decode_probe_materializes_every_malformed_zlib_output(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Retain emitted plaintext in generic and source-start exact-path views."""
    secret = b"NPA_DERIVED_PATH_" + (b"A1" * 12)
    payload = secret + b"\n" + (b"bounded padding " * 32)
    malformed = _output_bearing_malformed_zlib(payload, mutation)
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(base64.b64encode(malformed))
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    derived = sorted(destination.glob("decode-derived-output-*.txt"))
    exact = sorted(
        path
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    )
    assert receipt["derived_streams"] == len(derived) == 1
    assert receipt["derived_outputs"] == 1
    assert receipt["derived_bytes"] == len(payload)
    assert receipt["derived_exact_path_files"] == len(exact) == 1
    assert receipt["derived_exact_path_identity_reuses"] == 1
    assert receipt["derived_exact_path_shards"] == 1
    assert receipt["derived_exact_path_bytes"] == len(payload)
    assert receipt["max_derived_bytes"] == len(payload)
    assert payload in derived[0].read_bytes()
    assert all(path.name == "carrier.txt" for path in exact)
    assert all(path.read_bytes() == payload for path in exact)


def test_decode_probe_rejects_oversized_derived_zlib_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject decompressor output that cannot fit one complete scanner fragment."""
    source = tmp_path / "source"
    source.mkdir()
    payload = b"derived-output-bound"
    (source / "carrier.txt").write_bytes(base64.b64encode(zlib.compress(payload)))
    monkeypatch.setattr(
        tree_scan,
        "_MAX_ENCODED_DERIVED_OUTPUT_BYTES",
        len(payload) - 1,
    )

    with pytest.raises(tree_scan._ScanError, match="derived output exceeds"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_decode_probe_transcodes_derived_unicode_views(tmp_path: Path) -> None:
    """Expose derived UTF-16 in both generic and source-start exact-path views."""
    secret = b"NPA_DERIVED_UNICODE_" + (b"B2" * 12)
    unicode_payload = secret.decode("ascii").encode("utf-16le")
    source = tmp_path / "source"
    source.mkdir()
    encoded = base64.b64encode(zlib.compress(unicode_payload))
    (source / "carrier.txt").write_bytes(encoded)
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    derived = next(destination.glob("decode-derived-output-*.txt")).read_bytes()
    exact = [
        path.read_bytes()
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["derived_transcoded_files"] == 1
    assert receipt["derived_transcoded_bytes"] > len(secret)
    assert secret in derived
    assert sorted(exact, key=len) == [secret, unicode_payload]


def test_decode_probe_projects_malformed_derived_unicode_without_dropping_raw(
    tmp_path: Path,
) -> None:
    """Scan valid text around a malformed code unit while retaining every raw byte."""
    secret = b"NPA_MALFORMED_UNICODE_" + (b"E5" * 12)
    malformed = b"\xff\xfe" + secret.decode("ascii").encode("utf-16le") + b"\x09"
    replacement = secret + b"\xef\xbf\xbd"
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(zlib.compress(malformed))
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    derived = next(destination.glob("decode-derived-output-*.txt")).read_bytes()
    exact = [
        path.read_bytes()
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["derived_transcoded_files"] == 1
    assert receipt["derived_transcoded_bytes"] == len(replacement)
    assert malformed in derived
    assert replacement in derived
    assert sorted(exact, key=len) == [replacement, malformed]


def test_decode_probe_bounds_derived_input_bytes_files_and_no_progress(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply every generated-output budget before a derived view is accepted."""
    payload = b"derived bounded payload"
    encoded = base64.b64encode(zlib.compress(payload))

    byte_source = tmp_path / "byte-source"
    byte_source.mkdir()
    (byte_source / "carrier.txt").write_bytes(encoded)
    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_BYTES", 1)
    with pytest.raises(tree_scan._ScanError, match="aggregate bounded byte"):
        tree_scan._prepare_decode_probe_source(byte_source, tmp_path / "byte-probe")

    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_BYTES", 1024 * 1024)
    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_FILES", 1)
    file_source = tmp_path / "file-source"
    file_source.mkdir()
    (file_source / "carrier.txt").write_bytes(encoded)
    with pytest.raises(tree_scan._ScanError, match="bounded file count"):
        tree_scan._prepare_decode_probe_source(
            file_source,
            tmp_path / "file-probe",
            preserve_exact_paths=True,
        )

    monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_FILES", 100)
    monkeypatch.setattr(tree_scan, "_PROBE_DECODE_CHUNK_BYTES", 2)
    input_source = tmp_path / "input-source"
    input_source.mkdir()
    (input_source / "carrier.txt").write_bytes(encoded)
    with pytest.raises(tree_scan._ScanError, match="input exceeds"):
        tree_scan._prepare_decode_probe_source(
            input_source,
            tmp_path / "input-probe",
        )


def test_derived_zlib_output_rejects_decoder_no_progress(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject a decoder that retains input without emitting or consuming bytes."""

    class _StalledDecoder:
        eof = False
        unconsumed_tail = b""

        def decompress(self, pending: bytes, max_length: int) -> bytes:
            assert max_length == 1
            self.unconsumed_tail = pending
            return b""

    monkeypatch.setattr(
        tree_scan.zlib,
        "decompressobj",
        lambda _window: _StalledDecoder(),
    )
    with pytest.raises(tree_scan._ScanError, match="made no progress"):
        tree_scan._bounded_rfc1950_output(b"\x78\x9cX", 0)


def test_decode_probe_analyzes_duplicate_matches_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Cache by candidate bytes and canonical-source capability, not occurrence."""
    source = tmp_path / "source"
    source.mkdir()
    encoded = base64.b64encode(zlib.compress(b"deduplicated output"))
    (source / "carrier.txt").write_bytes(b"\n".join([encoded] * 1000))
    canonical_modes: list[bool] = []
    original = tree_scan._encoded_candidate_analysis
    expected_canonical = original(encoded)
    expected_offset = original(encoded, canonical_source=False)

    def count_analysis(
        candidate: bytes,
        *,
        max_emitted_bytes: int = tree_scan._MAX_ENCODED_DERIVED_ANALYSIS_BYTES,
        canonical_source: bool = True,
    ) -> tree_scan._EncodedCandidateAnalysis:
        if candidate == encoded:
            canonical_modes.append(canonical_source)
        return original(
            candidate,
            max_emitted_bytes=max_emitted_bytes,
            canonical_source=canonical_source,
        )

    monkeypatch.setattr(tree_scan, "_encoded_candidate_analysis", count_analysis)
    destination = tmp_path / "probe"
    receipt = tree_scan._prepare_decode_probe_source(source, destination)

    assert canonical_modes == [True, False]
    assert receipt["candidate_examined_matches"] == 2
    assert receipt["candidate_streams"] == 0
    assert receipt["derived_analysis_bytes"] == (
        expected_canonical.emitted_bytes + expected_offset.emitted_bytes
    )
    assert receipt["derived_streams"] == len(expected_canonical.outputs)
    assert expected_canonical.outputs == expected_offset.outputs


def test_decode_probe_bounds_distinct_carriers_with_same_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Charge decompression work before identical outputs are interned."""
    payload = b"same derived output" * 64
    source = tmp_path / "source"
    source.mkdir()
    carriers = [
        base64.b64encode(zlib.compress(payload) + index.to_bytes(4, "big"))
        for index in range(3)
    ]
    (source / "carrier.txt").write_bytes(b"\n".join(carriers))
    monkeypatch.setattr(
        tree_scan,
        "_MAX_ENCODED_DERIVED_ANALYSIS_BYTES",
        (2 * len(payload)) + 1,
    )

    with pytest.raises(tree_scan._ScanError, match="derived analysis exceeds"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_decode_probe_bounds_repeated_streams_inside_one_carrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Charge every stream emission before intra-candidate output deduplication."""
    payload = b"repeated stream output" * 64
    source = tmp_path / "source"
    source.mkdir()
    repeated = zlib.compress(payload) * 3
    (source / "carrier.txt").write_bytes(base64.b64encode(repeated))
    monkeypatch.setattr(
        tree_scan,
        "_MAX_ENCODED_DERIVED_ANALYSIS_BYTES",
        (2 * len(payload)) + 1,
    )

    with pytest.raises(tree_scan._ScanError, match="derived analysis exceeds"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_decode_probe_preserves_derived_paths_across_hardlinks(
    tmp_path: Path,
) -> None:
    """Reuse inode analysis without dropping any derived logical-path view."""
    source = tmp_path / "source"
    source.mkdir()
    encoded = base64.b64encode(zlib.compress(b"derived hardlink payload"))
    first = source / "first.txt"
    first.write_bytes(encoded)
    os.link(first, source / "second.txt")
    destination = tmp_path / "probe"
    expected = tree_scan._encoded_candidate_analysis(encoded).outputs

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    exact = [
        path
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["candidate_examined_matches"] == 1
    assert receipt["derived_exact_path_files"] == len(exact) == 2 * len(expected)
    assert receipt["derived_exact_path_identity_reuses"] == 2 * len(expected)
    assert Counter(path.name for path in exact) == {
        "first.txt": len(expected),
        "second.txt": len(expected),
    }
    assert Counter(path.read_bytes() for path in exact) == Counter(
        {output: 2 for output in expected}
    )


@pytest.mark.parametrize(
    "mutation",
    ["truncated", "deflate-truncated", "checksum", "deflate-error"],
)
def test_decode_probe_materializes_direct_malformed_zlib_output(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Retain output from an RFC 1950 stream embedded directly in a raw file."""
    payload = b"NPA_DIRECT_DERIVED_" + (b"C3" * 12)
    malformed = _output_bearing_malformed_zlib(payload, mutation)
    expected = tree_scan._bounded_rfc1950_output(malformed, 0)
    assert expected
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.bin").write_bytes(b"ordinary-prefix\n" + malformed)
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    derived = sorted(destination.glob("decode-derived-output-*.txt"))
    exact = sorted(
        path
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    )
    assert receipt["direct_source_carriers"] == 1
    assert receipt["direct_rfc1950_candidates"] >= 1
    assert receipt["direct_output_occurrences"] >= 1
    assert receipt["direct_emitted_bytes"] == len(expected)
    assert receipt["derived_analysis_bytes"] == len(expected)
    assert receipt["derived_streams"] == len(derived) == 1
    assert receipt["derived_exact_path_files"] == len(exact) == 1
    assert receipt["derived_exact_path_identity_reuses"] == 1
    assert expected in derived[0].read_bytes()
    assert all(path.name == "carrier.bin" for path in exact)
    assert all(path.read_bytes() == expected for path in exact)


def test_decode_probe_accounts_direct_empty_and_invalid_zlib_streams(
    tmp_path: Path,
) -> None:
    """Distinguish a complete empty stream from a marker that emits no bytes."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "empty.bin").write_bytes(zlib.compress(b""))
    (source / "invalid.bin").write_bytes(b"\x78\x9c\xff")

    receipt = tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")

    assert receipt["direct_source_carriers"] == 2
    assert receipt["direct_rfc1950_candidates"] == 2
    assert receipt["direct_complete_empty_streams"] == 1
    assert receipt["direct_invalid_streams"] == 1
    assert receipt["direct_output_occurrences"] == 0
    assert receipt["direct_emitted_bytes"] == 0
    assert receipt["derived_outputs"] == 0


def test_decode_probe_reuses_direct_hardlink_analysis_and_paths(
    tmp_path: Path,
) -> None:
    """Analyze one direct carrier inode once while retaining both exact paths."""
    payload = b"direct hardlink output"
    source = tmp_path / "source"
    source.mkdir()
    first = source / "first.bin"
    content = zlib.compress(payload)
    first.write_bytes(content)
    os.link(first, source / "second.bin")
    destination = tmp_path / "probe"
    expected = tree_scan._direct_compressed_outputs(
        content,
        max_candidates=tree_scan._MAX_DIRECT_COMPRESSED_CANDIDATES,
        max_emitted_bytes=tree_scan._MAX_ENCODED_DERIVED_ANALYSIS_BYTES,
    )

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    exact = [
        path
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["direct_source_carriers"] == 1
    assert receipt["direct_duplicate_carriers"] == 1
    assert receipt["direct_output_occurrences"] == expected.output_occurrences
    assert receipt["direct_emitted_bytes"] == expected.emitted_bytes
    assert (
        receipt["derived_exact_path_files"] == len(exact) == (2 * len(expected.outputs))
    )
    assert receipt["derived_exact_path_identity_reuses"] == 2 * len(expected.outputs)
    assert Counter(path.name for path in exact) == {
        "first.bin": len(expected.outputs),
        "second.bin": len(expected.outputs),
    }
    assert Counter(path.read_bytes() for path in exact) == Counter(
        {output: 2 for output in expected.outputs}
    )
    assert sum(path.read_bytes() == payload for path in exact) == 2


def test_decode_probe_bounds_direct_candidates_and_repeated_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bound direct candidate work and charge output before deduplication."""
    candidate_source = tmp_path / "candidate-source"
    candidate_source.mkdir()
    (candidate_source / "carrier.bin").write_bytes(zlib.compress(b"") * 3)
    monkeypatch.setattr(tree_scan, "_MAX_DIRECT_COMPRESSED_CANDIDATES", 2)
    with pytest.raises(tree_scan._ScanError, match="bounded candidate count"):
        tree_scan._prepare_decode_probe_source(
            candidate_source,
            tmp_path / "candidate-probe",
        )

    monkeypatch.setattr(tree_scan, "_MAX_DIRECT_COMPRESSED_CANDIDATES", 100)
    payload = b"repeated direct output" * 64
    output_source = tmp_path / "output-source"
    output_source.mkdir()
    (output_source / "carrier.bin").write_bytes(zlib.compress(payload) * 3)
    monkeypatch.setattr(
        tree_scan,
        "_MAX_ENCODED_DERIVED_ANALYSIS_BYTES",
        (2 * len(payload)) + 1,
    )
    with pytest.raises(tree_scan._ScanError, match="Direct RFC 1950 derived"):
        tree_scan._prepare_decode_probe_source(
            output_source,
            tmp_path / "output-probe",
        )


def test_decode_probe_allows_diagnosed_prose_after_scanning_output(
    tmp_path: Path,
) -> None:
    """Replace an incidental prose-header rejection with retained derived bytes."""
    prose = (
        b"(SkyPilot CLI, npa source, placeholder bucket) with the fix for each; " * 32
    )
    assert tree_scan._zlib_candidate_requires_rejection(prose, 0)
    source = tmp_path / "source"
    source.mkdir()
    carrier = source / "CHANGELOG.md"
    carrier.write_bytes(prose)
    with pytest.raises(tree_scan._ScanError, match="archive bytes"):
        tree_scan._validate_archive_file(carrier)

    receipt = tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")

    assert receipt["direct_output_occurrences"] >= 1
    assert receipt["direct_emitted_bytes"] > 0
    assert receipt["derived_outputs"] >= 1


@pytest.mark.parametrize(
    "nested",
    [
        _zip_member("payload.txt", b"safe\n"),
        gzip.compress(b"safe\n"),
        bz2.compress(b"safe\n"),
        lzma.compress(b"safe\n"),
        _tar_member("payload.txt", b"safe\n"),
    ],
    ids=["zip", "gzip", "bzip2", "xz", "tar"],
)
def test_decode_probe_rejects_nested_archive_in_direct_output(
    tmp_path: Path,
    nested: bytes,
) -> None:
    """Keep a container emitted by a direct stream fail closed."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.bin").write_bytes(zlib.compress(nested))

    with pytest.raises(tree_scan._ScanError, match="Nested archive"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_decode_probe_materializes_recursive_rfc1950_output(
    tmp_path: Path,
) -> None:
    """Retain a terminal child found only through a second RFC 1950 decode."""
    terminal = b"NPA_RECURSIVE_CHILD_" + (b"R7" * 12)
    parent = zlib.compress(terminal)
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(zlib.compress(parent))
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    derived = [path.read_bytes() for path in destination.glob("decode-derived-*")]
    exact = [
        path.read_bytes()
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["direct_output_occurrences"] == 1
    assert receipt["direct_emitted_bytes"] == len(parent)
    assert receipt["nested_compressed_candidates"] == 1
    assert receipt["nested_rfc1950_candidates"] == 1
    assert receipt["nested_output_occurrences"] == 1
    assert receipt["nested_emitted_bytes"] == len(terminal)
    assert receipt["nested_max_depth"] == 1
    assert receipt["derived_analysis_bytes"] == len(parent) + len(terminal)
    assert receipt["derived_outputs"] == 2
    assert any(parent in content for content in derived)
    assert any(terminal in content for content in derived)
    assert receipt["derived_exact_path_identity_reuses"] == 2
    assert exact.count(parent) == 1
    assert exact.count(terminal) == 1


def test_decode_probe_accounts_repeated_recursive_occurrences_before_dedup(
    tmp_path: Path,
) -> None:
    """Charge all repeated parents and children while storing each identity once."""
    terminal = b"recursive occurrence accounting"
    parent = zlib.compress(terminal)
    carrier = zlib.compress(parent) * 3
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(carrier)

    receipt = tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")

    assert receipt["direct_output_occurrences"] == 3
    assert receipt["direct_emitted_bytes"] == 3 * len(parent)
    assert receipt["nested_rfc1950_candidates"] == 3
    assert receipt["nested_output_occurrences"] == 3
    assert receipt["nested_emitted_bytes"] == 3 * len(terminal)
    assert receipt["derived_outputs"] == 2


def test_decode_probe_accounts_all_recursive_candidates_in_one_parent(
    tmp_path: Path,
) -> None:
    """Inspect every child candidate and distinguish empty and invalid streams."""
    first = b"first recursive child"
    second = b"second recursive child"
    parent = (
        zlib.compress(first)
        + zlib.compress(second)
        + zlib.compress(b"")
        + b"\x78\x9c\xff"
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(zlib.compress(parent))

    receipt = tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")

    # One additional plausible RFC 1950 header occurs inside the compressed
    # fixture bytes and must be charged as an invalid candidate, not skipped.
    assert receipt["nested_compressed_candidates"] == 5
    assert receipt["nested_rfc1950_candidates"] == 5
    assert receipt["nested_output_occurrences"] == 2
    assert receipt["nested_complete_empty_streams"] == 1
    assert receipt["nested_invalid_streams"] == 2
    assert receipt["nested_emitted_bytes"] == len(first) + len(second)
    assert receipt["derived_outputs"] == 3


def test_decode_probe_preserves_malformed_unicode_recursive_child(
    tmp_path: Path,
) -> None:
    """Keep complete raw bytes beside replacement text for a recursive child."""
    secret = b"NPA_RECURSIVE_MALFORMED_" + (b"M8" * 12)
    malformed = b"\xff\xfe" + secret.decode().encode("utf-16le") + b"\x09"
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(_nested_zlib(malformed, 2))
    destination = tmp_path / "probe"

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    exact = [
        path.read_bytes()
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["nested_output_occurrences"] == 1
    assert receipt["nested_emitted_bytes"] == len(malformed)
    assert malformed in exact
    assert secret + b"\xef\xbf\xbd" in exact


def test_decode_probe_replays_recursive_outputs_for_hardlink_paths(
    tmp_path: Path,
) -> None:
    """Reuse immutable analysis while preserving every hardlink association."""
    terminal = b"recursive hardlink terminal"
    source = tmp_path / "source"
    source.mkdir()
    first = source / "first.txt"
    content = _nested_zlib(terminal, 2)
    first.write_bytes(content)
    os.link(first, source / "second.txt")
    destination = tmp_path / "probe"
    expected = tree_scan._direct_compressed_outputs(
        content,
        max_candidates=tree_scan._MAX_DIRECT_COMPRESSED_CANDIDATES,
        max_emitted_bytes=tree_scan._MAX_ENCODED_DERIVED_ANALYSIS_BYTES,
    )

    receipt = tree_scan._prepare_decode_probe_source(
        source,
        destination,
        preserve_exact_paths=True,
    )

    exact = [
        path
        for path in (destination / "npa-gitleaks-derived-exact-paths").rglob("*")
        if path.is_file()
    ]
    assert receipt["direct_source_carriers"] == 1
    assert receipt["direct_duplicate_carriers"] == 1
    assert receipt["nested_output_occurrences"] == expected.nested_output_occurrences
    assert receipt["nested_emitted_bytes"] == expected.nested_emitted_bytes
    assert (
        receipt["derived_exact_path_files"] == len(exact) == (2 * len(expected.outputs))
    )
    assert receipt["derived_exact_path_identity_reuses"] == 2 * len(expected.outputs)
    assert Counter(path.name for path in exact) == {
        "first.txt": len(expected.outputs),
        "second.txt": len(expected.outputs),
    }
    assert Counter(path.read_bytes() for path in exact) == Counter(
        {output: 2 for output in expected.outputs}
    )
    assert sum(path.read_bytes() == terminal for path in exact) == 2


@pytest.mark.parametrize(
    "bound",
    ["candidate", "emitted", "generated", "aggregate", "shard"],
)
def test_decode_probe_applies_shared_bounds_to_recursive_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    bound: str,
) -> None:
    """Fail closed when recursive work crosses any shared resource ceiling."""
    terminal = (
        b"\xff\xfe" + "recursive bounded output".encode("utf-16le")
        if bound == "shard"
        else b"recursive bounded output"
    )
    parent = zlib.compress(terminal)
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(zlib.compress(parent))
    if bound == "candidate":
        monkeypatch.setattr(tree_scan, "_MAX_DIRECT_COMPRESSED_CANDIDATES", 1)
        error = "bounded candidate count"
    elif bound == "emitted":
        monkeypatch.setattr(
            tree_scan,
            "_MAX_ENCODED_DERIVED_ANALYSIS_BYTES",
            len(parent) + len(terminal) - 1,
        )
        error = "Recursive RFC 1950 derived analysis"
    elif bound == "generated":
        monkeypatch.setattr(tree_scan, "_MAX_ENCODED_CANDIDATE_FILES", 1)
        error = "bounded file count"
    elif bound == "aggregate":
        first_payload = (
            len(materializer.SCAN_PREFIX)
            + (2 * len(materializer._SCAN_BOUNDARY))
            + len(parent)
        )
        monkeypatch.setattr(
            tree_scan,
            "_MAX_ENCODED_CANDIDATE_BYTES",
            first_payload,
        )
        error = "aggregate bounded byte limit"
    else:
        monkeypatch.setattr(tree_scan, "_MAX_EXACT_PATH_SHARDS", 2)
        error = "shard limit"

    with pytest.raises(tree_scan._ScanError, match=error):
        tree_scan._prepare_decode_probe_source(
            source,
            tmp_path / "probe",
            preserve_exact_paths=bound == "shard",
        )


def test_decode_probe_rejects_recursive_depth_exhaustion(tmp_path: Path) -> None:
    """Reject an RFC 1950 child at the configured recursive depth ceiling."""
    source = tmp_path / "source"
    source.mkdir()
    levels = tree_scan._MAX_DECODE_DEPTH + 2
    (source / "carrier.txt").write_bytes(_nested_zlib(b"terminal", levels))

    with pytest.raises(tree_scan._ScanError, match="decode-depth bound"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_direct_output_rejects_recursive_identity_cycle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reject an emitted output that returns to an ancestor identity."""
    parent = zlib.compress(b"terminal")
    monkeypatch.setattr(
        tree_scan,
        "_bounded_rfc1950_result",
        lambda _content, _offset: tree_scan._RFC1950DecodeResult(
            parent,
            True,
            None,
        ),
    )

    with pytest.raises(tree_scan._ScanError, match="cycle"):
        tree_scan._recursive_direct_outputs(
            (parent,),
            tree_scan._derived_output_identity(b"distinct source"),
            10,
            100,
        )


def test_decode_probe_rejects_encoded_archive_in_recursive_output(
    tmp_path: Path,
) -> None:
    """Keep text-encoded containers inside a derived node fail closed."""
    encoded_child = base64.b64encode(zlib.compress(b"hidden child"))
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(zlib.compress(encoded_child))

    with pytest.raises(tree_scan._ScanError, match="encoded archive output"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


@pytest.mark.parametrize(
    "nested",
    [
        _zip_member("payload.txt", b"safe\n"),
        gzip.compress(b"safe\n"),
        bz2.compress(b"safe\n"),
        _tar_member("payload.txt", b"safe\n"),
    ],
    ids=["zip", "gzip", "bzip2", "tar"],
)
def test_decode_probe_rejects_nested_archive_output(
    tmp_path: Path,
    nested: bytes,
) -> None:
    """Do not replace a nested container with an incomplete derived view."""
    source = tmp_path / "source"
    source.mkdir()
    encoded = base64.b64encode(zlib.compress(nested))
    (source / "carrier.txt").write_bytes(encoded)

    with pytest.raises(tree_scan._ScanError, match="Nested archive output"):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_decode_probe_recursively_retains_encoded_derived_rfc1950(
    tmp_path: Path,
) -> None:
    """Route an encoded first-level output through shared recursive closure."""
    source = tmp_path / "source"
    source.mkdir()
    terminal = b"second-level terminal"
    (source / "carrier.txt").write_bytes(
        base64.b64encode(zlib.compress(zlib.compress(terminal)))
    )

    receipt = tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")

    assert receipt["derived_outputs"] == 2
    assert receipt["nested_rfc1950_candidates"] == 1
    assert receipt["nested_output_occurrences"] == 1
    assert any(
        terminal in path.read_bytes()
        for path in (tmp_path / "probe").glob("decode-derived-output-*.txt")
    )


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_encoded_derived_recursive_rfc1950_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find a policy secret present only in an encoded second-level child."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "encoded-recursive-rfc1950-control"\n'
        'description = "Encoded recursive RFC 1950 control"\n'
        "regex = '''NPA_ENCODED_RECURS[I]VE_[A-Z0-9]{20}'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    secret = b"NPA_ENCODED_RECURSIVE_" + (b"Z9" * 10)
    carrier = base64.b64encode(zlib.compress(zlib.compress(secret)))
    assert secret not in carrier
    (repo / "carrier.txt").write_bytes(carrier)
    candidate = _commit(repo, "encoded recursive RFC 1950 carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert secret.decode() not in output.out
    assert secret.decode() not in output.err
    assert "encoded-recursive-rfc1950-control" in output.err


@pytest.mark.parametrize("encoding", ["utf-16le", "utf-32be"])
def test_decode_probe_rejects_archive_exposed_by_unicode_projection(
    tmp_path: Path,
    encoding: str,
) -> None:
    """Reapply archive analysis to every generated Unicode projection."""
    nested = base64.b64encode(zlib.compress(b"projection-hidden output"))
    unicode_payload = nested.decode("ascii").encode(encoding)
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(
        base64.b64encode(zlib.compress(unicode_payload))
    )

    with pytest.raises(
        tree_scan._ScanError,
        match="Derived Unicode projection contains nested archive",
    ):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


def test_decode_probe_rejects_archive_exposed_by_malformed_unicode_projection(
    tmp_path: Path,
) -> None:
    """Keep nesting strict when replacement decoding recovers valid encoded text."""
    nested = base64.b64encode(zlib.compress(b"projection-hidden output"))
    malformed = b"\xff\xfe" + nested.decode("ascii").encode("utf-16le") + b"\x09"
    source = tmp_path / "source"
    source.mkdir()
    (source / "carrier.txt").write_bytes(zlib.compress(malformed))

    with pytest.raises(
        tree_scan._ScanError,
        match="Derived Unicode projection contains nested archive",
    ):
        tree_scan._prepare_decode_probe_source(source, tmp_path / "probe")


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
@pytest.mark.parametrize(
    "mutation",
    ["truncated", "deflate-truncated", "checksum", "deflate-error"],
)
def test_runner_scans_malformed_zlib_output_at_exact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    mutation: str,
) -> None:
    """Find emitted malformed-stream bytes only under their governed source path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "derived-path-control"\n'
        'description = "Derived RFC 1950 exact-path control"\n'
        "regex = '''NPA_DERIVED_PATH_[A-Z0-9]{24}$'''\n"
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_DERIVED_PATH_" + ("A1" * 12)
    payload = (b"bounded prefix " * 32) + b"\n" + token.encode()
    malformed = _output_bearing_malformed_zlib(payload, mutation)
    (repo / "carrier.txt").write_bytes(base64.b64encode(malformed))
    candidate = _commit(repo, f"{mutation} derived-output carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "derived-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_direct_zlib_output_at_exact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find direct-stream plaintext only under its governed logical path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "direct-derived-path-control"\n'
        'description = "Direct RFC 1950 exact-path control"\n'
        "regex = '''NPA_DIRECT_PATH_[A-Z0-9]{24}$'''\n"
        # The pinned default policy globally excludes .bin paths.
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_DIRECT_PATH_" + ("D4" * 12)
    malformed = _output_bearing_malformed_zlib(token.encode(), "checksum")
    (repo / "carrier.txt").write_bytes(b"ordinary-prefix\n" + malformed)
    candidate = _commit(repo, "direct derived-output carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "direct-derived-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_recursive_zlib_child_at_exact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find a source-anchored secret visible only in a second-level child."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "recursive-derived-path-control"\n'
        'description = "Recursive RFC 1950 exact-path control"\n'
        "regex = '''^NPA_RECURSIVE_PATH_[A-Z0-9]{24}$'''\n"
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_RECURSIVE_PATH_" + ("F6" * 12)
    (repo / "carrier.txt").write_bytes(_nested_zlib(token.encode(), 2))
    candidate = _commit(repo, "recursive derived-output carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "recursive-derived-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_derived_utf16_at_exact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find a derived Unicode secret only after its exact transformed projection."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "derived-unicode-path-control"\n'
        'description = "Derived Unicode exact-path control"\n'
        "regex = '''^NPA_DERIVED_UNICODE_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_DERIVED_UNICODE_" + ("B2" * 12)
    payload = token.encode("utf-16le")
    (repo / "carrier.txt").write_bytes(base64.b64encode(zlib.compress(payload)))
    candidate = _commit(repo, "derived Unicode carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "derived-unicode-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_scans_malformed_derived_utf16_at_exact_path(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Find valid text projected from malformed derived Unicode at its source path."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\n"
        "useDefault = true\n\n"
        "[[rules]]\n"
        'id = "malformed-derived-unicode-path-control"\n'
        'description = "Malformed derived Unicode exact-path control"\n'
        "regex = '''^NPA_MALFORMED_UNICODE_[A-Z0-9]{24}'''\n"
        "path = '''^carrier\\.txt$'''\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "NPA_MALFORMED_UNICODE_" + ("E5" * 12)
    malformed_unicode = b"\xff\xfe" + token.encode("utf-16le") + b"\x09"
    malformed_zlib = _output_bearing_malformed_zlib(
        malformed_unicode,
        "checksum",
    )
    (repo / "carrier.txt").write_bytes(b"ordinary-prefix\n" + malformed_zlib)
    candidate = _commit(repo, "malformed derived Unicode carrier")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "malformed-derived-unicode-path-control" in output.err


@pytest.mark.skipif(
    shutil.which("gitleaks") is None, reason="gitleaks is not installed"
)
def test_runner_rejects_decode_exhaustion_inside_derived_output(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Reject nested encoded output before downstream scanner decode-depth work."""
    repo = _repo(tmp_path)
    (repo / ".gitleaks.toml").write_text(
        "[extend]\nuseDefault = true\n",
        encoding="utf-8",
    )
    (repo / "clean.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "clean base")
    token = "".join(("Qw7eR9tY", "2uI4oP6a", "S8dF0gH1", "jK3lZ5xC"))
    secret = f'api_key = "{token}"\n'.encode()
    derived = _encode_base64_layers(secret, 6)
    (repo / "carrier.txt").write_bytes(base64.b64encode(zlib.compress(derived)))
    candidate = _commit(repo, "derived decode-depth exhaustion")
    scanner = shutil.which("gitleaks")
    assert scanner is not None

    assert tree_scan.main(_runner_arguments(repo, candidate, base, scanner)) == 1
    output = capsys.readouterr()
    assert token not in output.out
    assert token not in output.err
    assert "Nested encoded archive output" in output.err


def test_decode_boundary_view_contains_only_isolated_candidates(
    tmp_path: Path,
) -> None:
    """Exclude fragment windows without omitting any isolated codec stream."""
    source = tmp_path / "probe"
    source.mkdir()
    candidate = source / "decode-input-candidates-0.txt"
    candidate.write_bytes(materializer.SCAN_PREFIX + b"encoded\n\n")
    derived = source / "decode-derived-output-0.txt"
    derived.write_bytes(materializer.SCAN_PREFIX + b"derived\n\n")
    fragments = source / "npa-gitleaks-fragment-windows"
    fragments.mkdir()
    (fragments / "noise.txt").write_text("encoded", encoding="utf-8")

    boundary = tree_scan._decode_boundary_view(tmp_path, "candidate", source, 2)

    files = [path for path in boundary.rglob("*") if path.is_file()]
    assert {path.name for path in files} == {candidate.name, derived.name}
    assert {path.stat().st_ino for path in files} == {
        candidate.stat().st_ino,
        derived.stat().st_ino,
    }
    with pytest.raises(tree_scan._ScanError, match="inventory is incomplete"):
        tree_scan._decode_boundary_view(tmp_path, "mismatch", source, 1)


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
    assert receipt["native_history_input_bytes_per_scan"] > 0
    assert receipt["native_history_patch_bytes_per_scan"] > 0
    assert receipt["native_history_blob_bytes_upper_bound_per_scan"] > 0
    assert receipt["native_history_diff_metadata_bytes"] > 0
    assert receipt["native_history_input_bytes_per_scan"] == (
        receipt["native_history_patch_bytes_per_scan"]
        + receipt["native_history_blob_bytes_upper_bound_per_scan"]
    )
    assert receipt["native_history_input_bytes"] == (
        receipt["native_history_input_bytes_per_scan"]
        * receipt["native_history_scan_invocations"]
    )
    assert receipt["native_history_scan_invocations"] == 2
    assert (
        receipt["native_history_input_bytes"]
        <= receipt["native_history_input_byte_limit"]
    )
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


def test_native_history_input_preflight_counts_and_bounds_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Charge repeated patches and reopened binary blobs before both scans."""

    def stored_zip(content: bytes) -> bytes:
        encoded = io.BytesIO()
        with zipfile.ZipFile(encoded, "w", compression=zipfile.ZIP_STORED) as archive:
            archive.writestr("payload.bin", content)
        return encoded.getvalue()

    repo = _repo(tmp_path)
    (repo / "carrier.zip").write_bytes(stored_zip(b"A" * 65_536))
    base = _commit(repo, "base")
    commits = []
    for index in range(4):
        payload = (b"B" if index % 2 == 0 else b"A") * 65_536
        (repo / "carrier.zip").write_bytes(stored_zip(payload))
        candidate = _commit(repo, f"alternating payload {index}")
        commits.append(candidate)
    history_range = f"{base}..{candidate}"
    source = tree_scan._isolated_history_source(repo, tmp_path / "history-source")

    measured = tree_scan._measure_native_history_input(
        source,
        history_range,
        commits,
    )
    assert measured.patch_bytes > 0
    assert measured.changed_blob_bytes_upper_bound > 4 * 65_536
    assert measured.diff_metadata_bytes > 0

    monkeypatch.setattr(
        tree_scan,
        "_MAX_NATIVE_HISTORY_INPUT_BYTES",
        2 * measured.bytes_per_scan,
    )
    assert (
        tree_scan._measure_native_history_input(
            source,
            history_range,
            commits,
        )
        == measured
    )

    monkeypatch.setattr(
        tree_scan,
        "_MAX_NATIVE_HISTORY_INPUT_BYTES",
        2 * (measured.bytes_per_scan - 1),
    )
    with pytest.raises(tree_scan._ScanError, match="bounded byte limit"):
        tree_scan._measure_native_history_input(
            source,
            history_range,
            commits,
        )


def test_native_history_input_charges_symlink_archive_blob(
    tmp_path: Path,
) -> None:
    """Charge archive bytes that Gitleaks reopens even under Git mode 120000."""
    repo = _repo(tmp_path)
    (repo / "base.txt").write_text("clean base\n", encoding="utf-8")
    base = _commit(repo, "base")
    encoded = io.BytesIO()
    with zipfile.ZipFile(encoded, "w", compression=zipfile.ZIP_STORED) as archive:
        archive.writestr("payload.txt", b"A" * 65_536)
    blob_path = tmp_path / "symlink-archive.bin"
    blob_path.write_bytes(encoded.getvalue())
    blob = _git(repo, "hash-object", "-w", str(blob_path))
    _git(repo, "read-tree", base)
    _git(repo, "update-index", "--add", "--cacheinfo", f"120000,{blob},carrier.zip")
    tree = _git(repo, "write-tree")
    candidate = _git(
        repo,
        "commit-tree",
        tree,
        "-p",
        base,
        "-m",
        "symlink-mode archive",
    )
    _git(repo, "update-ref", "HEAD", candidate)
    source = tree_scan._isolated_history_source(repo, tmp_path / "history-source")
    metadata = tree_scan._read_native_diff_metadata(source, candidate, 1_000_000)

    assert tree_scan._changed_blob_ids(metadata, b"sha1") == [blob.encode()]
    measured = tree_scan._measure_native_history_input(
        source,
        f"{base}..{candidate}",
        [candidate],
    )
    assert measured.changed_blob_bytes_upper_bound == len(encoded.getvalue())
    assert measured.bytes_per_scan == (measured.patch_bytes + len(encoded.getvalue()))


def test_isolated_history_source_drops_local_signature_program(
    tmp_path: Path,
) -> None:
    """Never execute a caller-controlled verifier from the source Git config."""
    repo = _repo(tmp_path)
    _git(repo, "config", "commit.gpgsign", "false")
    (repo / "source.txt").write_text("base\n", encoding="utf-8")
    base = _commit(repo, "base")
    (repo / "source.txt").write_text("candidate\n", encoding="utf-8")
    candidate = _synthetic_signed_commit(repo, base)
    _git(repo, "update-ref", "HEAD", candidate)
    marker, helper = _configure_signature_helper(repo, tmp_path)
    subprocess.run(
        ["git", "-C", str(repo), "log", "-1", candidate],
        check=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    assert marker.read_text(encoding="utf-8") == "invoked"
    marker.unlink()

    source = tree_scan._isolated_history_source(repo, tmp_path / "history-source")
    assert materializer.git_directory(source) != materializer.git_directory(repo)
    assert str(helper) not in (source / ".git" / "config").read_text(encoding="utf-8")
    measured = tree_scan._measure_native_history_input(
        source,
        f"{base}..{candidate}",
        [candidate],
    )
    assert measured.bytes_per_scan > 0
    assert not marker.exists()


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
    assert receipt["historical_decode_probe"]["derived_exact_path_identity_reuses"] == 0


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
