"""Materialize exact committed Git blobs for filesystem-mode Gitleaks scans."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import json
import mmap
import os
import re
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path
from typing import BinaryIO, Protocol

REGULAR_MODES = {b"100644", b"100755"}
_SYMLINK_MODE = b"120000"
_SUPPORTED_MODES = REGULAR_MODES | {_SYMLINK_MODE}
SCAN_PREFIX = b"\n" * 4096
SCAN_NON_TEST_ROOT = b".npa-gitleaks-scan"
SCAN_TEST_ROOT = b"npa/tests/.npa-gitleaks-scan"
_SCAN_BOUNDARY = b"\n\n\0\0\0\0\n\n"
_COPY_CHUNK_BYTES = 1024 * 1024
_TREE_MODES = {b"40000"}
_OBJECT_HASHES = {b"sha1": "sha1", b"sha256": "sha256"}
_GIT_EXECUTABLE = "/usr/bin/git"
_POLICY_PATHS = {b".gitleaks.toml", b".gitleaksignore"}
_MAX_SCAN_BLOB_BYTES = 512 * 1024 * 1024
_MAX_SCAN_SOURCE_BYTES = 2 * 1024 * 1024 * 1024
_MAX_COMMIT_METADATA_BYTES = 64 * 1024 * 1024
_MAX_COMMIT_METADATA_TOTAL_BYTES = 256 * 1024 * 1024
_MAX_TREE_OBJECT_BYTES = 512 * 1024 * 1024
_MAX_TREE_METADATA_BYTES = 2 * 1024 * 1024 * 1024
_MAX_TREE_PATHS = 100_000
_MAX_TREE_PATH_BYTES = 512 * 1024 * 1024
_MAX_SCAN_VIEW_BYTES = 16 * 1024 * 1024 * 1024
_MAX_EXACT_PATH_SHARDS = 1024
_MAX_HISTORY_COMMITS = 100_000
_MAX_HISTORY_COMMIT_BYTES = 2 * 1024 * 1024 * 1024
_UNICODE_RUNS = (
    (re.compile(rb"(?:[\x09\x0a\x0d\x20-\x7e]\x00){4,}"), "utf-16-le"),
    (re.compile(rb"(?:\x00[\x09\x0a\x0d\x20-\x7e]){4,}"), "utf-16-be"),
    (
        re.compile(rb"(?:[\x09\x0a\x0d\x20-\x7e]\x00\x00\x00){4,}"),
        "utf-32-le",
    ),
    (
        re.compile(rb"(?:\x00\x00\x00[\x09\x0a\x0d\x20-\x7e]){4,}"),
        "utf-32-be",
    ),
)

_Entry = tuple[bytes, bytes, bytes]


class _WriteReceipt:
    """Accumulate byte and file counts while materializing Git blobs."""

    def __init__(self) -> None:
        self.total_bytes = 0
        self.materialized_bytes = 0
        self.scan_bytes = 0
        self.raw_scan_files = 0
        self.wrapper_scan_files = 0
        self.exact_path_wrapper_files = 0
        self.path_bytes = 0
        self.transcoded_files = 0
        self.transcoded_bytes = 0


class _HistoryIndex:
    """Track deduplicated objects and paths across historical trees."""

    def __init__(self) -> None:
        self.commits: list[str] = []
        self.trees: list[str] = []
        self.entries: dict[tuple[bytes, bytes], _Entry] = {}
        self.paths: set[bytes] = set()
        self.source_entries = 0


class _TreeBudget:
    """Bound authenticated tree metadata and expanded logical paths."""

    def __init__(self) -> None:
        self.metadata_bytes = 0
        self.paths = 0
        self.path_bytes = 0

    def remaining_metadata_bytes(self) -> int:
        """Return the bytes still available for one authenticated tree object."""
        remaining = _MAX_TREE_METADATA_BYTES - self.metadata_bytes
        if remaining <= 0:
            raise SnapshotError("Git tree metadata exceeds the aggregate byte limit")
        return min(remaining, _MAX_TREE_OBJECT_BYTES)

    def add_metadata(self, size: int) -> None:
        """Record one verified tree object after its bounded read."""
        self.metadata_bytes += size

    def add_path(self, path: bytes) -> None:
        """Bound expanded path cardinality and bytes before materialization."""
        self.paths += 1
        self.path_bytes += len(path)
        if self.paths > _MAX_TREE_PATHS:
            raise SnapshotError("Git tree exceeds the bounded path count")
        if self.path_bytes > _MAX_TREE_PATH_BYTES:
            raise SnapshotError("Git tree exceeds the aggregate path byte limit")


class _HistoryBudget:
    """Bound authenticated commit cardinality and aggregate metadata reads."""

    def __init__(self) -> None:
        self.commits: set[str] = set()
        self.metadata_bytes = 0

    def remaining_bytes(self, commit: str) -> int:
        """Reserve one distinct commit and return its remaining read allowance."""
        if commit not in self.commits:
            if len(self.commits) >= _MAX_HISTORY_COMMITS:
                raise SnapshotError(
                    "Reachable Git history exceeds the bounded commit count"
                )
            self.commits.add(commit)
        remaining = _MAX_HISTORY_COMMIT_BYTES - self.metadata_bytes
        if remaining <= 0:
            raise SnapshotError(
                "Reachable Git history exceeds the aggregate metadata byte limit"
            )
        return remaining

    def add_metadata(self, size: int) -> None:
        """Account one authenticated commit body after its bounded read."""
        self.metadata_bytes += size
        if self.metadata_bytes > _MAX_HISTORY_COMMIT_BYTES:
            raise SnapshotError(
                "Reachable Git history exceeds the aggregate metadata byte limit"
            )


class _Digest(Protocol):
    """Minimal streaming digest contract used while copying blobs."""

    def update(self, content: bytes) -> None:
        """Add bytes to the digest."""
        ...

    def hexdigest(self) -> str:
        """Return the lowercase hexadecimal digest."""
        ...


class SnapshotError(RuntimeError):
    """Report a source tree that cannot be materialized without omission."""


def _git_environment() -> dict[str, str]:
    """Build an environment without caller-controlled Git indirection."""
    environment = {
        key: value for key, value in os.environ.items() if not key.startswith("GIT_")
    }
    environment["GIT_CONFIG_GLOBAL"] = os.devnull
    environment["GIT_CONFIG_NOSYSTEM"] = "1"
    environment["GIT_CONFIG_COUNT"] = "1"
    environment["GIT_CONFIG_KEY_0"] = "core.commitGraph"
    environment["GIT_CONFIG_VALUE_0"] = "false"
    environment["GIT_NO_REPLACE_OBJECTS"] = "1"
    environment["GIT_OPTIONAL_LOCKS"] = "0"
    return environment


def _git_command(repo: Path, *arguments: str) -> list[str]:
    """Build one replacement-disabled Git command."""
    return [_GIT_EXECUTABLE, "--no-replace-objects", "-C", str(repo), *arguments]


def _git(repo: Path, *arguments: str) -> bytes:
    """Run one read-only Git command and return its stdout."""
    return subprocess.run(
        _git_command(repo, *arguments),
        check=True,
        capture_output=True,
        env=_git_environment(),
    ).stdout


def _object_format(repo: Path) -> bytes:
    """Resolve the repository object format used for identity verification."""
    object_format = _git(repo, "rev-parse", "--show-object-format").strip()
    if object_format not in _OBJECT_HASHES:
        raise SnapshotError("Git returned an unsupported object format")
    return object_format


def git_directory(repo: Path) -> Path:
    """Resolve the repository metadata directory without checkout indirection."""
    raw_path = _git(repo, "rev-parse", "--absolute-git-dir").rstrip(b"\n")
    if not raw_path or b"\n" in raw_path or b"\r" in raw_path:
        raise SnapshotError("Git returned an invalid metadata directory")
    path = Path(os.fsdecode(raw_path)).resolve(strict=True)
    if not path.is_dir():
        raise SnapshotError("Git metadata directory must be a directory")
    return path


def _validate_object_id(object_id: bytes, object_format: bytes) -> None:
    """Require one canonical hexadecimal object ID for the selected format."""
    try:
        raw_id = bytes.fromhex(object_id.decode("ascii"))
    except (UnicodeDecodeError, ValueError) as exc:
        raise SnapshotError("Git returned an invalid object ID") from exc
    expected_size = hashlib.new(_OBJECT_HASHES[object_format]).digest_size
    if len(raw_id) != expected_size or raw_id.hex().encode("ascii") != object_id:
        raise SnapshotError("Git returned an invalid object ID")


def _verified_digest(
    object_format: bytes,
    kind: bytes,
    content: bytes,
) -> bytes:
    """Compute the canonical Git object ID for one complete object body."""
    digest = hashlib.new(_OBJECT_HASHES[object_format])
    digest.update(kind + b" " + str(len(content)).encode("ascii") + b"\0")
    digest.update(content)
    return digest.hexdigest().encode("ascii")


def _read_exact_object_body(stream: BinaryIO, size: int) -> bytes:
    """Read one already bounded Git object body without accepting truncation."""
    content = bytearray()
    remaining = size
    while remaining:
        chunk = stream.read(min(remaining, _COPY_CHUNK_BYTES))
        if not chunk:
            raise SnapshotError("Git returned an incomplete source object")
        content.extend(chunk)
        remaining -= len(chunk)
    return bytes(content)


def _read_verified_object(
    repo: Path,
    object_id: bytes,
    expected_kind: bytes,
    object_format: bytes,
    *,
    max_bytes: int | None = None,
    bounded_error: str = "Git object exceeds the bounded byte limit",
) -> bytes:
    """Read one metadata object and recompute its content-addressed identity."""
    _validate_object_id(object_id, object_format)
    process = subprocess.Popen(
        _git_command(repo, "cat-file", "--batch"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    try:
        if process.stdin is None or process.stdout is None or process.stderr is None:
            raise SnapshotError("Could not open the Git object stream")
        process.stdin.write(object_id + b"\n")
        process.stdin.close()
        header_line = process.stdout.readline(1024)
        if not header_line.endswith(b"\n"):
            raise SnapshotError("Git returned an invalid object header")
        header = header_line.removesuffix(b"\n").split()
        if len(header) != 3:
            raise SnapshotError("Git returned an invalid object header")
        returned_id, kind, raw_size = header
        if returned_id != object_id or kind != expected_kind:
            raise SnapshotError("Git returned an unexpected source object")
        try:
            size = int(raw_size)
        except ValueError as exc:
            raise SnapshotError("Git returned an invalid source object size") from exc
        if size < 0:
            raise SnapshotError("Git returned an invalid source object size")
        if max_bytes is not None and size > max_bytes:
            raise SnapshotError(bounded_error)
        content = _read_exact_object_body(process.stdout, size)
        if process.stdout.read(1) != b"\n" or process.stdout.read(1):
            raise SnapshotError("Git returned an incomplete source object")
        stderr = process.stderr.read()
        status = process.wait()
        if status:
            detail = stderr.decode("utf-8", errors="replace").strip()
            suffix = f": {detail}" if detail else ""
            raise SnapshotError(
                f"Git object stream failed with status {status}{suffix}"
            )
        if _verified_digest(object_format, kind, content) != object_id:
            raise SnapshotError("Git object bytes do not match their object ID")
        return content
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors = _cleanup_blob_process(process, ())
        if cleanup_errors and active_error is None:
            raise SnapshotError("Could not clean up the Git object stream") from (
                cleanup_errors[0]
            )


def resolve_commit(
    repo: Path,
    revision: str,
    *,
    history_budget: _HistoryBudget | None = None,
) -> dict[str, object]:
    """Resolve and cryptographically verify one commit identity."""
    object_format = _object_format(repo)
    commit = (
        _git(
            repo,
            "rev-parse",
            "--verify",
            "--end-of-options",
            f"{revision}^{{commit}}",
        )
        .decode("ascii")
        .strip()
    )
    commit_id = commit.encode("ascii")
    _validate_object_id(commit_id, object_format)
    max_bytes = _MAX_COMMIT_METADATA_BYTES
    bounded_error = "Commit metadata exceeds the bounded byte limit"
    if history_budget is not None:
        remaining = history_budget.remaining_bytes(commit)
        if remaining <= max_bytes:
            bounded_error = (
                "Reachable Git history exceeds the aggregate metadata byte limit"
            )
        max_bytes = min(max_bytes, remaining)
    content = _read_verified_object(
        repo,
        commit_id,
        b"commit",
        object_format,
        max_bytes=max_bytes,
        bounded_error=bounded_error,
    )
    if history_budget is not None:
        history_budget.add_metadata(len(content))
    headers = content.split(b"\n\n", 1)[0].splitlines()
    tree_ids = [
        header.removeprefix(b"tree ")
        for header in headers
        if header.startswith(b"tree ")
    ]
    if len(tree_ids) != 1 or not headers or headers[0] != b"tree " + tree_ids[0]:
        raise SnapshotError("Git commit does not identify one canonical root tree")
    _validate_object_id(tree_ids[0], object_format)
    parent_ids = [
        header.removeprefix(b"parent ")
        for header in headers
        if header.startswith(b"parent ")
    ]
    for parent_id in parent_ids:
        _validate_object_id(parent_id, object_format)
    return {
        "commit": commit,
        "tree": tree_ids[0].decode("ascii"),
        "parents": [parent_id.decode("ascii") for parent_id in parent_ids],
        "object_format": object_format.decode("ascii"),
    }


def read_verified_commit(repo: Path, commit: str) -> bytes:
    """Read one exact commit body after recomputing its content-addressed ID."""
    object_format = _object_format(repo)
    commit_id = commit.encode("ascii")
    _validate_object_id(commit_id, object_format)
    return _read_verified_object(
        repo,
        commit_id,
        b"commit",
        object_format,
        max_bytes=_MAX_COMMIT_METADATA_BYTES,
        bounded_error="Commit metadata exceeds the bounded byte limit",
    )


def _validate_history_metadata(repo: Path) -> bytes:
    """Reject local metadata that can truncate or rewrite traversal."""
    object_format = _object_format(repo)
    if _git(repo, "rev-parse", "--is-shallow-repository").strip() != b"false":
        raise SnapshotError("Shallow Git history cannot prove complete coverage")
    graft_value = _git(repo, "rev-parse", "--git-path", "info/grafts").strip()
    graft_path = Path(os.fsdecode(graft_value))
    if not graft_path.is_absolute():
        graft_path = repo / graft_path
    if os.path.lexists(graft_path):
        raise SnapshotError("Git grafts cannot prove complete history coverage")
    return object_format


def _verified_reachable_history(
    repo: Path,
    candidate: str,
    history_budget: _HistoryBudget | None = None,
) -> list[str]:
    """Walk raw authenticated parent headers without commit-graph traversal."""
    budget = _HistoryBudget() if history_budget is None else history_budget
    object_format = _validate_history_metadata(repo)
    candidate_id = candidate.encode("ascii")
    _validate_object_id(candidate_id, object_format)
    state: dict[str, int] = {}
    order: list[str] = []
    stack: list[tuple[str, bool]] = [(candidate, False)]
    while stack:
        commit, expanded = stack.pop()
        current = state.get(commit, 0)
        if expanded:
            if current != 1:
                raise SnapshotError("Git commit graph traversal became inconsistent")
            state[commit] = 2
            order.append(commit)
            continue
        if current == 2:
            continue
        if current == 1:
            raise SnapshotError("Git returned a cyclic commit graph")
        identity = resolve_commit(repo, commit, history_budget=budget)
        if identity["commit"] != commit:
            raise SnapshotError("Git history did not return canonical commit IDs")
        parents = identity["parents"]
        if not isinstance(parents, list):
            raise SnapshotError("Git commit returned invalid parent metadata")
        state[commit] = 1
        stack.append((commit, True))
        for parent in reversed(parents):
            if not isinstance(parent, str):
                raise SnapshotError("Git commit returned an invalid parent ID")
            if state.get(parent) == 1:
                raise SnapshotError("Git returned a cyclic commit graph")
            stack.append((parent, False))
    if not order or order[-1] != candidate:
        raise SnapshotError("Git history did not terminate at the candidate")
    return order


def history_commits(
    repo: Path,
    base: str,
    candidate: str,
    *,
    history_budget: _HistoryBudget | None = None,
) -> list[str]:
    """Return a verified, complete topological commit range."""
    budget = _HistoryBudget() if history_budget is None else history_budget
    object_format = _validate_history_metadata(repo)
    base_id = base.encode("ascii")
    candidate_id = candidate.encode("ascii")
    _validate_object_id(base_id, object_format)
    _validate_object_id(candidate_id, object_format)
    candidate_history = _verified_reachable_history(repo, candidate, budget)
    candidate_set = set(candidate_history)
    if base not in candidate_set:
        raise SnapshotError("History base must be an ancestor of the candidate")
    base_set = set(_verified_reachable_history(repo, base, budget))
    if candidate == base:
        return []
    commits = [commit for commit in candidate_history if commit not in base_set]
    if not commits or commits[-1] != candidate:
        raise SnapshotError("Git history range did not terminate at the candidate")
    return commits


def reachable_history_commits(repo: Path, candidate: str) -> list[str]:
    """Return every verified commit reachable from one exact candidate."""
    return _verified_reachable_history(repo, candidate)


def _validate_path(path: bytes) -> None:
    """Reject paths that cannot stay below the private snapshot root."""
    if not path or path.startswith(b"/"):
        raise SnapshotError("Git returned an empty or absolute source path")
    if any(part in {b"", b".", b".."} for part in path.split(b"/")):
        raise SnapshotError("Git returned a source path with unsafe components")


def _tree_entries(
    repo: Path,
    tree: bytes,
    object_format: bytes,
    member_cache: dict[bytes, list[_Entry]] | None = None,
    budget: _TreeBudget | None = None,
) -> list[_Entry]:
    """Parse and cryptographically verify every reachable tree object."""
    entries: list[_Entry] = []
    seen: set[bytes] = set()
    cached_members = {} if member_cache is None else member_cache
    tree_budget = _TreeBudget() if budget is None else budget
    digest_size = hashlib.new(_OBJECT_HASHES[object_format]).digest_size

    def members(tree_id: bytes) -> list[_Entry]:
        cached = cached_members.get(tree_id)
        if cached is not None:
            return cached
        content = _read_verified_object(
            repo,
            tree_id,
            b"tree",
            object_format,
            max_bytes=tree_budget.remaining_metadata_bytes(),
            bounded_error="Git tree metadata exceeds the bounded byte limit",
        )
        tree_budget.add_metadata(len(content))
        parsed: list[_Entry] = []
        position = 0
        names: set[bytes] = set()
        previous_sort_key: bytes | None = None
        while position < len(content):
            mode_end = content.find(b" ", position)
            name_end = content.find(b"\0", mode_end + 1)
            object_end = name_end + 1 + digest_size
            if (
                mode_end <= position
                or name_end <= mode_end + 1
                or object_end > len(content)
            ):
                raise SnapshotError("Git returned an invalid tree object")
            mode = content[position:mode_end]
            name = content[mode_end + 1 : name_end]
            if b"/" in name:
                raise SnapshotError("Git returned a tree entry containing '/'")
            if name in names:
                raise SnapshotError("Git returned a duplicate tree entry")
            names.add(name)
            sort_key = name + (b"/" if mode in _TREE_MODES else b"")
            if previous_sort_key is not None and sort_key <= previous_sort_key:
                raise SnapshotError("Git returned a non-canonical tree order")
            previous_sort_key = sort_key
            object_id = content[name_end + 1 : object_end].hex().encode("ascii")
            _validate_object_id(object_id, object_format)
            parsed.append((mode, object_id, name))
            position = object_end
        cached_members[tree_id] = parsed
        return parsed

    def visit(
        tree_id: bytes,
        prefix: bytes,
        ancestors: frozenset[bytes],
    ) -> None:
        if tree_id in ancestors:
            raise SnapshotError("Git returned a cyclic tree graph")
        for mode, object_id, name in members(tree_id):
            path = name if not prefix else prefix + b"/" + name
            _validate_path(path)
            tree_budget.add_path(path)
            if path in seen:
                raise SnapshotError("Git returned a duplicate source path")
            seen.add(path)
            if mode in _TREE_MODES:
                visit(object_id, path, ancestors | {tree_id})
                continue
            if mode not in _SUPPORTED_MODES:
                display = os.fsdecode(path)
                raise SnapshotError(
                    f"Unsupported Git tree entry {display!r}: mode={mode.decode()}"
                )
            entries.append((mode, object_id, path))

    visit(tree, b"", frozenset())
    return entries


def read_optional_verified_blob(
    repo: Path,
    tree: str,
    raw_path: bytes,
    *,
    max_bytes: int | None = None,
    bounded_error: str = "Git blob exceeds the bounded byte limit",
) -> bytes | None:
    """Read an optional regular blob from a fully verified committed tree."""
    _validate_path(raw_path)
    object_format = _object_format(repo)
    tree_id = tree.encode("ascii")
    _validate_object_id(tree_id, object_format)
    matches = [
        (mode, object_id)
        for mode, object_id, path in _tree_entries(repo, tree_id, object_format)
        if path == raw_path
    ]
    if not matches:
        return None
    if len(matches) != 1 or matches[0][0] not in REGULAR_MODES:
        raise SnapshotError("Committed config must identify one regular blob")
    return _read_verified_object(
        repo,
        matches[0][1],
        b"blob",
        object_format,
        max_bytes=max_bytes,
        bounded_error=bounded_error,
    )


def read_verified_blob(
    repo: Path,
    tree: str,
    raw_path: bytes,
    *,
    max_bytes: int | None = None,
    bounded_error: str = "Git blob exceeds the bounded byte limit",
) -> bytes:
    """Read one required regular blob from a fully verified committed tree."""
    content = read_optional_verified_blob(
        repo,
        tree,
        raw_path,
        max_bytes=max_bytes,
        bounded_error=bounded_error,
    )
    if content is None:
        raise SnapshotError("Committed config must identify one regular blob")
    return content


def _target_path(destination: Path, raw_path: bytes) -> Path:
    """Map one validated Git path to its private snapshot destination."""
    parts = (os.fsdecode(part) for part in raw_path.split(b"/"))
    return destination.joinpath(*parts)


def _open_target(target: Path) -> BinaryIO:
    """Create one regular snapshot file without following a destination link."""
    target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(target, flags, 0o600)
    return os.fdopen(descriptor, "wb")


def _copy_exact(
    source: BinaryIO,
    targets: tuple[BinaryIO, ...],
    size: int,
    digest: _Digest,
) -> None:
    """Copy exactly one object body without buffering a complete blob."""
    remaining = size
    while remaining:
        content = source.read(min(remaining, _COPY_CHUNK_BYTES))
        if not content:
            raise SnapshotError("Git returned an incomplete source object")
        for target in targets:
            target.write(content)
        digest.update(content)
        remaining -= len(content)


def _cleanup_blob_process(
    process: subprocess.Popen[bytes],
    streams: tuple[BinaryIO, ...],
) -> list[Exception]:
    """Reap the Git child and attempt every close despite individual failures."""
    errors: list[Exception] = []

    def attempt(operation: Callable[[], object]) -> None:
        try:
            operation()
        except (OSError, ValueError) as exc:
            errors.append(exc)

    if process.stdin is not None and not process.stdin.closed:
        attempt(process.stdin.close)
    if process.poll() is None:
        attempt(process.kill)
    attempt(process.wait)
    if process.stdout is not None:
        attempt(process.stdout.close)
    if process.stderr is not None:
        attempt(process.stderr.close)
    for stream in streams:
        attempt(stream.close)
    return errors


def _whole_unicode_encoding(path: Path) -> str | None:
    """Recognize a complete BOM-marked or strongly interleaved Unicode file."""
    positions = [0, 0, 0, 0]
    size = 0
    with path.open("rb") as stream:
        prefix = stream.read(4)
        if prefix.startswith((codecs.BOM_UTF32_LE, codecs.BOM_UTF32_BE)):
            return "utf-32"
        if prefix.startswith((codecs.BOM_UTF16_LE, codecs.BOM_UTF16_BE)):
            return "utf-16"
        stream.seek(0)
        while content := stream.read(_COPY_CHUNK_BYTES):
            for slot in range(4):
                start = (slot - size) % 4
                positions[slot] += content[start::4].count(0)
            size += len(content)
    quads = size // 4
    if quads >= 2:
        ratios = [count / quads for count in positions]
        if max(ratios[0], 0.0) <= 0.05 and min(ratios[1:]) >= 0.3:
            return "utf-32-le"
        if max(ratios[3], 0.0) <= 0.05 and min(ratios[:3]) >= 0.3:
            return "utf-32-be"
    pairs = size // 2
    if pairs < 2:
        return None
    even_nuls = positions[0] + positions[2]
    odd_nuls = positions[1] + positions[3]
    if odd_nuls / pairs >= 0.3 and even_nuls / pairs <= 0.05:
        return "utf-16-le"
    if even_nuls / pairs >= 0.3 and odd_nuls / pairs <= 0.05:
        return "utf-16-be"
    return None


def _decode_unicode(path: Path, encoding: str, output: BinaryIO | None) -> int:
    """Validate or stream one complete Unicode file as UTF-8."""
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    written = 0
    with path.open("rb") as stream:
        while content := stream.read(_COPY_CHUNK_BYTES):
            text = decoder.decode(content, final=False)
            if output is not None and text:
                encoded = text.encode("utf-8")
                output.write(encoded)
                written += len(encoded)
        text = decoder.decode(b"", final=True)
        if output is not None and text:
            encoded = text.encode("utf-8")
            output.write(encoded)
            written += len(encoded)
    return written


def _append_unicode_runs(path: Path, output: BinaryIO) -> int:
    """Expose localized ASCII-bearing UTF-16/32 runs inside mixed binary data."""
    if path.stat().st_size == 0:
        return 0
    written = 0
    with (
        path.open("rb") as stream,
        mmap.mmap(
            stream.fileno(),
            0,
            access=mmap.ACCESS_READ,
        ) as content,
    ):
        if content.find(b"\0") < 0:
            return 0
        for pattern, encoding in _UNICODE_RUNS:
            for match in pattern.finditer(content):
                encoded = (
                    match.group().decode(encoding, errors="strict").encode("utf-8")
                )
                output.write(b"\n\n")
                output.write(encoded)
                written += len(encoded) + 2
    return written


def _append_unicode(path: Path, output: BinaryIO) -> int:
    """Append complete or localized UTF-16/32 text without silent decode loss."""
    encoding = _whole_unicode_encoding(path)
    if encoding is None:
        return _append_unicode_runs(path, output)
    try:
        _decode_unicode(path, encoding, None)
    except (UnicodeDecodeError, UnicodeEncodeError) as exc:
        raise SnapshotError("Recognized Unicode text is malformed") from exc
    boundary = b"\n\n"
    output.write(boundary)
    return len(boundary) + _decode_unicode(path, encoding, output)


def _raw_scan_path(
    object_id: bytes,
    raw_path: bytes,
    preserve_paths: bool,
) -> bytes:
    """Keep policy bytes scan-visible without placing policy at its active path."""
    if not preserve_paths:
        return _historical_target_path(object_id, raw_path)
    if raw_path in _POLICY_PATHS:
        suffix = b".toml" if raw_path.endswith(b".toml") else b".ignore"
        return b".npa-gitleaks-policy/" + object_id + suffix
    return raw_path


def _wrapper_scan_path(object_id: bytes, raw_path: bytes) -> bytes:
    """Isolate one blob and path in the matching test/non-test policy context."""
    root = SCAN_TEST_ROOT if raw_path.startswith(b"npa/tests/") else SCAN_NON_TEST_ROOT
    path_digest = hashlib.sha256(raw_path).hexdigest().encode("ascii")
    return root + b"/" + object_id + b"-" + path_digest + b".txt"


def _context_wrapper_scan_path(object_id: bytes, raw_path: bytes) -> bytes:
    """Retain a logical path suffix while keeping one blob in an isolated root."""
    root = SCAN_TEST_ROOT if raw_path.startswith(b"npa/tests/") else SCAN_NON_TEST_ROOT
    return root + b"/path-context/" + object_id + b"/" + raw_path


def _path_scan_path(raw_path: bytes) -> bytes:
    """Isolate one historical path that was deduplicated from blob wrappers."""
    root = SCAN_TEST_ROOT if raw_path.startswith(b"npa/tests/") else SCAN_NON_TEST_ROOT
    path_digest = hashlib.sha256(raw_path).hexdigest().encode("ascii")
    return root + b"/path-" + path_digest + b".txt"


def _context_path_scan_path(raw_path: bytes) -> bytes:
    """Retain a historical path suffix for path-constrained scanner rules."""
    root = SCAN_TEST_ROOT if raw_path.startswith(b"npa/tests/") else SCAN_NON_TEST_ROOT
    path_digest = hashlib.sha256(raw_path).hexdigest().encode("ascii")
    return root + b"/path-context/path-" + path_digest + b"/" + raw_path


def _historical_target_path(object_id: bytes, raw_path: bytes) -> bytes:
    """Give one historical blob a collision-free path with its exact basename."""
    name = raw_path.rsplit(b"/", 1)[-1]
    path_digest = hashlib.sha256(raw_path).hexdigest().encode("ascii")
    root = (
        b"npa/tests/.npa-gitleaks-history/"
        if raw_path.startswith(b"npa/tests/")
        else b".npa-gitleaks-history/"
    )
    return root + object_id + b"-" + path_digest + b"/" + name


def _candidate_exact_path_targets(
    entries: list[_Entry],
    root: Path | None,
) -> dict[tuple[bytes, bytes], Path] | None:
    """Map each candidate wrapper to its exact logical path in a separate view."""
    if root is None:
        return None
    root.mkdir(mode=0o700)
    return {
        (object_id, raw_path): _target_path(root, raw_path)
        for _mode, object_id, raw_path in entries
    }


def _history_exact_path_targets(
    entries: list[_Entry],
    root: Path | None,
) -> tuple[dict[tuple[bytes, bytes], Path] | None, int]:
    """Shard historical wrappers so each source exposes exact logical paths."""
    if root is None:
        return None, 0
    root.mkdir(mode=0o700)
    targets: dict[tuple[bytes, bytes], Path] = {}
    full_paths: list[set[tuple[bytes, ...]]] = []
    strict_prefixes: list[set[tuple[bytes, ...]]] = []
    for _mode, object_id, raw_path in entries:
        parts = tuple(raw_path.split(b"/"))
        ancestors = {parts[:index] for index in range(1, len(parts))}
        shard = 0
        while shard < len(full_paths):
            if (
                parts not in full_paths[shard]
                and parts not in strict_prefixes[shard]
                and not ancestors & full_paths[shard]
            ):
                break
            shard += 1
        if shard >= _MAX_EXACT_PATH_SHARDS:
            raise SnapshotError("Historical exact-path views exceed the shard limit")
        if shard == len(full_paths):
            full_paths.append(set())
            strict_prefixes.append(set())
            shard_root = root / str(shard)
            shard_root.mkdir(mode=0o700)
        else:
            shard_root = root / str(shard)
        full_paths[shard].add(parts)
        strict_prefixes[shard].update(ancestors)
        targets[(object_id, raw_path)] = _target_path(shard_root, raw_path)
    return targets, len(full_paths)


def _start_blob_process(repo: Path) -> subprocess.Popen[bytes]:
    """Start one authenticated Git blob stream."""
    process = subprocess.Popen(
        _git_command(repo, "cat-file", "--batch"),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=_git_environment(),
    )
    if (
        process.stdin is not None
        and process.stdout is not None
        and process.stderr is not None
    ):
        return process
    process.kill()
    process.wait()
    raise SnapshotError("Could not open the Git object stream")


def _write_path_wrappers(
    destination: Path,
    scan_paths: list[bytes],
    receipt: _WriteReceipt,
) -> None:
    """Write isolated historical path-only wrappers."""
    for raw_path in scan_paths:
        payload = b"\n" + raw_path + b"\n"
        file_bytes = len(SCAN_PREFIX) + len(payload)
        if receipt.materialized_bytes + file_bytes > _MAX_SCAN_VIEW_BYTES:
            raise SnapshotError("Generated scan view exceeds the bounded byte limit")
        target = _target_path(destination, _path_scan_path(raw_path))
        with _open_target(target) as stream:
            stream.write(SCAN_PREFIX)
            stream.write(payload)
        context_target = _target_path(
            destination,
            _context_path_scan_path(raw_path),
        )
        context_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(target, context_target)
        receipt.wrapper_scan_files += 2
        receipt.path_bytes += len(payload)
        receipt.scan_bytes += 2 * (len(SCAN_PREFIX) + len(payload))
        receipt.materialized_bytes += file_bytes


def _request_blob_size(
    process: subprocess.Popen[bytes],
    expected_id: bytes,
) -> int:
    """Request one blob and validate its Git batch header."""
    if process.stdin is None or process.stdout is None:
        raise SnapshotError("Git object stream is unavailable")
    process.stdin.write(expected_id + b"\n")
    process.stdin.flush()
    header = process.stdout.readline().removesuffix(b"\n").split()
    if len(header) != 3:
        raise SnapshotError("Git returned an invalid object header")
    object_id, kind, raw_size = header
    if object_id != expected_id or kind != b"blob":
        raise SnapshotError("Git returned an unexpected source object")
    size = int(raw_size)
    if size < 0:
        raise SnapshotError("Git returned an invalid source object size")
    return size


def _write_scan_blob(
    source: BinaryIO,
    destination: Path,
    expected_id: bytes,
    raw_path: bytes,
    preserve_paths: bool,
    size: int,
    digest: _Digest,
    exact_path_target: Path | None,
) -> tuple[int, int, int]:
    """Write exact raw bytes plus neutral and logical-path text wrappers."""
    target_path = _raw_scan_path(expected_id, raw_path, preserve_paths)
    raw_target = _target_path(destination, target_path)
    wrapper_path = _wrapper_scan_path(expected_id, raw_path)
    wrapper_target = _target_path(destination, wrapper_path)
    path_payload = b"\n" + raw_path + b"\n"
    with _open_target(wrapper_target) as wrapper:
        wrapper.write(SCAN_PREFIX)
        wrapper.write(path_payload)
        wrapper.write(_SCAN_BOUNDARY)
        with _open_target(raw_target) as raw_stream:
            _copy_exact(source, (raw_stream, wrapper), size, digest)
        wrapper.write(_SCAN_BOUNDARY)
        transcoded = _append_unicode(raw_target, wrapper)
    context_target = _target_path(
        destination,
        _context_wrapper_scan_path(expected_id, raw_path),
    )
    context_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.link(wrapper_target, context_target)
    if exact_path_target is not None:
        exact_path_target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.link(wrapper_target, exact_path_target)
    wrapper_bytes = (
        len(SCAN_PREFIX)
        + len(path_payload)
        + (2 * len(_SCAN_BOUNDARY))
        + size
        + transcoded
    )
    wrapper_copies = 2 + int(exact_path_target is not None)
    scan_bytes = size + (wrapper_copies * wrapper_bytes)
    return scan_bytes, transcoded, size + wrapper_bytes


def _record_scan_blob(
    receipt: _WriteReceipt,
    raw_path: bytes,
    scan_bytes: int,
    transcoded: int,
    materialized_bytes: int,
    has_exact_path_wrapper: bool,
) -> None:
    """Record one raw blob and its neutral/logical text wrappers."""
    receipt.raw_scan_files += 1
    receipt.wrapper_scan_files += 2 + int(has_exact_path_wrapper)
    receipt.exact_path_wrapper_files += int(has_exact_path_wrapper)
    receipt.path_bytes += len(raw_path) + 2
    receipt.scan_bytes += scan_bytes
    receipt.materialized_bytes += materialized_bytes
    if transcoded:
        receipt.transcoded_files += 1
        receipt.transcoded_bytes += transcoded


def _write_blob_entry(
    process: subprocess.Popen[bytes],
    destination: Path,
    entry: _Entry,
    scan_view: bool,
    preserve_paths: bool,
    object_format: bytes,
    receipt: _WriteReceipt,
    exact_path_target: Path | None,
) -> None:
    """Write and cryptographically verify one requested blob."""
    _, expected_id, raw_path = entry
    size = _request_blob_size(process, expected_id)
    if scan_view and size > _MAX_SCAN_BLOB_BYTES:
        raise SnapshotError("Git blob exceeds the bounded scan-source byte limit")
    if scan_view and receipt.total_bytes + size > _MAX_SCAN_SOURCE_BYTES:
        raise SnapshotError("Git blobs exceed the aggregate scan-source byte limit")
    path_payload_bytes = len(raw_path) + 2
    wrapper_upper_bound = (
        len(SCAN_PREFIX)
        + path_payload_bytes
        + (2 * len(_SCAN_BOUNDARY))
        + (5 * size)
        + (2 * ((size // 8) + 1))
    )
    if (
        scan_view
        and receipt.materialized_bytes + size + wrapper_upper_bound
        > _MAX_SCAN_VIEW_BYTES
    ):
        raise SnapshotError("Generated scan view exceeds the bounded byte limit")
    digest = hashlib.new(_OBJECT_HASHES[object_format])
    digest.update(b"blob " + str(size).encode("ascii") + b"\0")
    if process.stdout is None:
        raise SnapshotError("Git object stream is unavailable")
    if scan_view:
        scan_bytes, transcoded, materialized_bytes = _write_scan_blob(
            process.stdout,
            destination,
            expected_id,
            raw_path,
            preserve_paths,
            size,
            digest,
            exact_path_target,
        )
        _record_scan_blob(
            receipt,
            raw_path,
            scan_bytes,
            transcoded,
            materialized_bytes,
            exact_path_target is not None,
        )
    else:
        target = _target_path(destination, raw_path)
        with _open_target(target) as stream:
            _copy_exact(process.stdout, (stream,), size, digest)
        receipt.scan_bytes += size
        receipt.materialized_bytes += size
    if digest.hexdigest().encode("ascii") != expected_id:
        raise SnapshotError("Git object bytes do not match their object ID")
    if process.stdout.read(1) != b"\n":
        raise SnapshotError("Git returned an incomplete source object")
    receipt.total_bytes += size


def _finish_blob_process(process: subprocess.Popen[bytes]) -> None:
    """Close one successful Git batch stream and validate its exit."""
    if process.stdin is None or process.stdout is None or process.stderr is None:
        raise SnapshotError("Git object stream is unavailable")
    process.stdin.close()
    if process.stdout.read(1):
        raise SnapshotError("Git returned unexpected trailing object bytes")
    stderr = process.stderr.read()
    status = process.wait()
    if status:
        detail = stderr.decode("utf-8", errors="replace").strip()
        suffix = f": {detail}" if detail else ""
        raise SnapshotError(f"Git object stream failed with status {status}{suffix}")


def _write_blobs(
    repo: Path,
    entries: list[_Entry],
    destination: Path,
    scan_view: bool,
    object_format: bytes,
    *,
    preserve_paths: bool = True,
    scan_paths: list[bytes] | None = None,
    exact_path_targets: dict[tuple[bytes, bytes], Path] | None = None,
) -> _WriteReceipt:
    """Stream, validate, and write every requested Git object."""
    process = _start_blob_process(repo)
    receipt = _WriteReceipt()
    try:
        if scan_view and scan_paths is not None:
            _write_path_wrappers(destination, scan_paths, receipt)
        for entry in entries:
            _write_blob_entry(
                process,
                destination,
                entry,
                scan_view,
                preserve_paths,
                object_format,
                receipt,
                (
                    None
                    if exact_path_targets is None
                    else exact_path_targets.get((entry[1], entry[2]))
                ),
            )
        _finish_blob_process(process)
    finally:
        active_error = sys.exc_info()[1]
        cleanup_errors = _cleanup_blob_process(process, ())
        if cleanup_errors and active_error is None:
            raise SnapshotError("Could not clean up the Git object stream") from (
                cleanup_errors[0]
            )
    return receipt


def _scan_receipt(
    receipt: _WriteReceipt,
    scan_view: bool,
    source_files: int,
) -> dict[str, object]:
    """Render common scan-view counts from one write receipt."""
    scan_files = source_files
    if scan_view:
        scan_files = receipt.raw_scan_files + receipt.wrapper_scan_files
    return {
        "scan_view": scan_view,
        "scan_prefix_bytes": len(SCAN_PREFIX) if scan_view else 0,
        "scan_files": scan_files,
        "raw_scan_files": receipt.raw_scan_files,
        "wrapper_scan_files": receipt.wrapper_scan_files,
        "exact_path_wrapper_files": receipt.exact_path_wrapper_files,
        "scan_bytes": receipt.scan_bytes,
        "materialized_bytes": receipt.materialized_bytes,
        "scan_path_bytes": receipt.path_bytes,
        "transcoded_files": receipt.transcoded_files,
        "transcoded_bytes": receipt.transcoded_bytes,
    }


def _candidate_receipt(
    identity: dict[str, object],
    tree: str,
    entries: list[_Entry],
    receipt: _WriteReceipt,
    scan_view: bool,
) -> dict[str, object]:
    """Render one candidate tree's materialization receipt."""
    return {
        "commit": identity["commit"],
        "tree": tree,
        "parents": identity["parents"],
        "object_format": identity["object_format"],
        "files": len(entries),
        "regular_files": sum(mode in REGULAR_MODES for mode, _, _ in entries),
        "symlink_blobs": sum(mode == _SYMLINK_MODE for mode, _, _ in entries),
        "bytes": receipt.total_bytes,
        **_scan_receipt(receipt, scan_view, len(entries)),
    }


def materialize(
    repo: Path,
    destination: Path,
    revision: str,
    *,
    scan_view: bool = False,
    exact_path_view: Path | None = None,
) -> dict[str, object]:
    """Materialize one exact commit tree.

    Args:
        repo: Repository containing the authenticated Git objects.
        destination: Empty private directory that receives the tree.
        revision: Revision resolving to the required commit.
        scan_view: Whether to add isolated scanner wrappers.
    Returns:
        A non-sensitive materialization receipt.
    Raises:
        SnapshotError: If exact materialization cannot be proven.
    """
    if exact_path_view is not None and not scan_view:
        raise SnapshotError("Exact-path wrappers require the scanner view")
    identity = resolve_commit(repo, revision)
    object_format = str(identity["object_format"]).encode("ascii")
    tree = str(identity["tree"])
    tree_budget = _TreeBudget()
    entries = _tree_entries(
        repo,
        tree.encode("ascii"),
        object_format,
        budget=tree_budget,
    )
    exact_path_targets = _candidate_exact_path_targets(entries, exact_path_view)
    receipt = _write_blobs(
        repo,
        entries,
        destination,
        scan_view,
        object_format,
        exact_path_targets=exact_path_targets,
    )
    rendered = _candidate_receipt(identity, tree, entries, receipt, scan_view)
    rendered["tree_metadata_bytes"] = tree_budget.metadata_bytes
    rendered["tree_paths"] = tree_budget.paths
    rendered["tree_path_bytes"] = tree_budget.path_bytes
    rendered["exact_path_shards"] = int(bool(exact_path_targets))
    rendered["root_ignore_exact_path_exclusions"] = int(
        exact_path_view is not None
        and any(raw_path == b".gitleaksignore" for _, _, raw_path in entries)
    )
    return rendered


def materialize_commit_metadata(
    repo: Path,
    destination: Path,
    revisions: list[str],
) -> dict[str, object]:
    """Materialize authenticated raw metadata for exact selected-range commits."""
    if not revisions:
        raise SnapshotError("Commit metadata scan requires at least one revision")
    object_format = _object_format(repo)
    seen: set[str] = set()
    total_bytes = 0
    scan_bytes = 0
    transcoded_bytes = 0
    transcoded_files = 0
    for revision in revisions:
        if revision in seen:
            raise SnapshotError("Commit metadata scan contains a duplicate revision")
        seen.add(revision)
        content = read_verified_commit(repo, revision)
        if len(content) > _MAX_COMMIT_METADATA_BYTES:
            raise SnapshotError("Commit metadata exceeds the bounded byte limit")
        if total_bytes + len(content) > _MAX_COMMIT_METADATA_TOTAL_BYTES:
            raise SnapshotError(
                "Commit metadata exceeds the aggregate bounded byte limit"
            )
        target = destination / f"{revision}.txt"
        wrapper_target = _target_path(
            destination,
            SCAN_NON_TEST_ROOT + b"/commit-" + revision.encode("ascii") + b".txt",
        )
        with _open_target(target) as stream:
            stream.write(content)
        with _open_target(wrapper_target) as wrapper:
            wrapper.write(SCAN_PREFIX)
            wrapper.write(_SCAN_BOUNDARY)
            wrapper.write(content)
            wrapper.write(_SCAN_BOUNDARY)
            transcoded = _append_unicode(target, wrapper)
        total_bytes += len(content)
        transcoded_bytes += transcoded
        transcoded_files += int(bool(transcoded))
        scan_bytes += (
            len(SCAN_PREFIX)
            + (2 * len(_SCAN_BOUNDARY))
            + (2 * len(content))
            + transcoded
        )
    return {
        "commits": list(revisions),
        "object_format": object_format.decode("ascii"),
        "files": len(revisions),
        "bytes": total_bytes,
        "wrapper_files": len(revisions),
        "scan_bytes": scan_bytes,
        "transcoded_files": transcoded_files,
        "transcoded_bytes": transcoded_bytes,
    }


def _index_history(
    repo: Path,
    revisions: list[str],
) -> tuple[bytes, _HistoryIndex, _TreeBudget]:
    """Verify and deduplicate source objects across historical trees."""
    object_format = _object_format(repo)
    member_cache: dict[bytes, list[_Entry]] = {}
    index = _HistoryIndex()
    tree_budget = _TreeBudget()
    for revision in revisions:
        identity = resolve_commit(repo, revision)
        if identity["commit"] != revision:
            raise SnapshotError("Historical scan revision must be one full commit ID")
        if identity["object_format"] != object_format.decode("ascii"):
            raise SnapshotError("Historical commits use inconsistent object formats")
        commit = str(identity["commit"])
        tree = str(identity["tree"])
        index.commits.append(commit)
        index.trees.append(tree)
        entries = _tree_entries(
            repo,
            tree.encode("ascii"),
            object_format,
            member_cache,
            tree_budget,
        )
        index.source_entries += len(entries)
        for mode, object_id, raw_path in entries:
            index.entries.setdefault(
                (object_id, raw_path),
                (mode, object_id, raw_path),
            )
            index.paths.add(raw_path)
    return object_format, index, tree_budget


def materialize_history_scan(
    repo: Path,
    destination: Path,
    revisions: list[str],
    *,
    exact_path_root: Path | None = None,
) -> dict[str, object]:
    """Materialize a deduplicated historical scan view.

    Args:
        repo: Repository containing the authenticated Git objects.
        destination: Empty private directory that receives the scan view.
        revisions: Exact historical commit IDs in scan order.
    Returns:
        A non-sensitive historical materialization receipt.
    Raises:
        SnapshotError: If history or exact materialization cannot be proven.
    """
    if not revisions:
        raise SnapshotError("Historical scan requires at least one revision")
    object_format, index, tree_budget = _index_history(repo, revisions)
    entries = list(index.entries.values())
    exact_path_targets, exact_path_shards = _history_exact_path_targets(
        entries,
        exact_path_root,
    )
    receipt = _write_blobs(
        repo,
        entries,
        destination,
        True,
        object_format,
        preserve_paths=False,
        scan_paths=sorted(index.paths),
        exact_path_targets=exact_path_targets,
    )
    return {
        "commits": index.commits,
        "trees": index.trees,
        "object_format": object_format.decode("ascii"),
        "source_entries": index.source_entries,
        "unique_blob_contexts": len(index.entries),
        "unique_paths": len(index.paths),
        "bytes": receipt.total_bytes,
        "tree_metadata_bytes": tree_budget.metadata_bytes,
        "tree_paths": tree_budget.paths,
        "tree_path_bytes": tree_budget.path_bytes,
        "exact_path_shards": exact_path_shards,
        "root_ignore_exact_path_exclusions": (
            sum(raw_path == b".gitleaksignore" for _, _, raw_path in entries)
            if exact_path_root is not None
            else 0
        ),
        **_scan_receipt(receipt, True, len(index.entries)),
    }


def _parser() -> argparse.ArgumentParser:
    """Build the command-line parser."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--revision", default="HEAD")
    parser.add_argument(
        "--scan-view",
        action="store_true",
        help=(
            "write an isolated text wrapper for each exact blob and path so "
            "Gitleaks cannot omit binary content or cross blob boundaries"
        ),
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Materialize committed bytes for a fail-closed Gitleaks filesystem scan.

    Args:
        argv: Optional command-line arguments.
    Returns:
        Zero on success and one when the tree cannot be materialized exactly.
    Raises:
        None.
    """
    args = _parser().parse_args(argv)
    try:
        repo = args.repo_root.resolve(strict=True)
        if not repo.is_dir():
            raise SnapshotError("Repository root must be an existing directory")
        if args.output_dir.is_symlink():
            raise SnapshotError("Output directory must not be a symbolic link")
        destination = args.output_dir.resolve(strict=True)
        if not destination.is_dir():
            raise SnapshotError("Output directory must be an existing real directory")
        if any(destination.iterdir()):
            raise SnapshotError("Output directory must be empty")
        receipt = materialize(
            repo,
            destination,
            args.revision,
            scan_view=args.scan_view,
        )
    except (OSError, ValueError, subprocess.CalledProcessError, SnapshotError) as exc:
        print(f"Gitleaks tree materialization failed: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(receipt, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
