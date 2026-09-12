"""Secret-sourced confidentiality denylist scanner.

The scanner intentionally reports only locations, never matched text.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tarfile

from npa.guardrails.ncore_attribution import (
    ATTRIBUTION_LINES as NCORE_ATTRIBUTION_LINES,
    REPOSITORY_PATH as NCORE_REPOSITORY_PATH,
    verify_public_notice,
)
from npa.guardrails.robomimic_attribution import (
    ATTRIBUTION_LINES as ROBOMIMIC_ATTRIBUTION_LINES,
    LOCK_GIT_BLOB_SHA1 as ROBOMIMIC_LOCK_GIT_BLOB_SHA1,
    REPOSITORY_PATH as ROBOMIMIC_REPOSITORY_PATH,
    verify_public_license_lock,
)


BUILTIN_NEBIUS_INFRA_PATTERN = r"""(?ix)
(?:
    # Opaque account/resource IDs. Ignore 40/64-character hexadecimal hashes.
    (?<![a-z0-9])(?:e|u)00
    (?!(?:[0-9a-f]{37}|[0-9a-f]{61})(?![0-9a-f]))
    [a-z0-9]{12,}(?![a-z0-9])
  |
    # Task-specific private object-store or registry locations.
    \bs3://[a-z0-9][a-z0-9.-]*
    (?:-[0-9a-f]{8,}|-(?:19|20)[0-9]{6})(?=[/\s`'\"]|$)
  |
    \bcr\.[a-z0-9-]+\.nebius\.cloud/[a-z0-9][a-z0-9._-]*
    (?:-[0-9a-f]{8,}|-(?:19|20)[0-9]{6})(?=[/:\s`'\"]|$)
  |
    # Dated operational names when attached to a resource/campaign/job field.
    \b(?:tenant|project|capacity[ _-]?block|cluster|node[ _-]?group|
       compute[ _-]?instance|bucket|registry|network|campaign|job)
    (?:[ _-]?(?:id|name))?\b[^\n]{0,48}[:=|/]\s*[`'\"]?
    [a-z0-9][a-z0-9-]{4,}-(?:19|20)[0-9]{6}\b
)
"""


@dataclass(frozen=True)
class ScanHit:
    """A redacted denylist match location."""

    source: str
    line_number: int
    repository_path: str | None = None


@dataclass(frozen=True)
class DenylistPattern:
    """Operator-provided denylist regex plus its redacted source name."""

    pattern: str
    source: str


def default_pattern_file(pattern_env: str) -> Path:
    """Return the local operator-private fallback path for a denylist env name."""

    file_name = pattern_env.lower().replace("_", "-")
    return Path("~/.config/npa").expanduser() / f"{file_name}.regex"


def load_denylist_pattern(
    pattern_env: str,
    *,
    pattern_file: Path | None = None,
    environ: Mapping[str, str] = os.environ,
) -> DenylistPattern:
    """Load a denylist regex from an env var or an operator-private file."""

    env_pattern = environ.get(pattern_env)
    if env_pattern and env_pattern.strip():
        return DenylistPattern(pattern=env_pattern, source=pattern_env)

    pattern_file_env = f"{pattern_env}_FILE"
    configured_file = environ.get(pattern_file_env)
    if configured_file:
        source_path = Path(configured_file).expanduser()
        source_name = f"{pattern_file_env}:{source_path}"
    elif pattern_file:
        source_path = pattern_file.expanduser()
        source_name = f"--pattern-file:{source_path}"
    else:
        source_path = default_pattern_file(pattern_env)
        source_name = f"default-file:{source_path}"

    if source_path.is_file():
        return DenylistPattern(
            pattern=source_path.read_text(encoding="utf-8"),
            source=source_name,
        )

    raise ValueError(
        f"{pattern_env} is empty and no denylist file was found at {source_path} "
        f"(or via {pattern_file_env})"
    )


def compile_denylist(
    pattern: str,
    *,
    source: str = "denylist",
    ignore_case: bool = False,
) -> re.Pattern[str]:
    """Compile the operator-provided denylist regex."""

    if not pattern.strip():
        raise ValueError(f"{source} is empty")
    flags = re.IGNORECASE if ignore_case else 0
    return re.compile(pattern, flags=flags)


def compile_builtin_nebius_infra() -> re.Pattern[str]:
    """Compile the public, deterministic Nebius infrastructure leak guard."""

    return compile_denylist(
        BUILTIN_NEBIUS_INFRA_PATTERN,
        source="built-in-nebius-infra",
    )


def scan_text(text: str, denylist: re.Pattern[str], *, source: str) -> list[ScanHit]:
    """Return redacted hit locations for text."""

    hits: list[ScanHit] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if denylist.search(line):
            hits.append(ScanHit(source=source, line_number=line_number))
    return hits


def scan_diff_text(
    text: str, denylist: re.Pattern[str], *, source: str
) -> list[ScanHit]:
    """Return redacted hits from added diff lines, ignoring removals and headers."""

    hits: list[ScanHit] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if not line.startswith("+") or line.startswith("+++"):
            continue
        if denylist.search(line[1:]):
            hits.append(ScanHit(source=source, line_number=line_number))
    return hits


def tracked_text_files(repo_root: Path) -> list[Path]:
    """Return Git-tracked files that Git classifies as text."""

    result = subprocess.run(
        ["git", "grep", "-Ilz", "-e", "", "--", "."],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if result.returncode not in (0, 1):
        result.check_returncode()
    names = [name for name in result.stdout.decode("utf-8").split("\0") if name]
    return [repo_root / name for name in names]


def scan_paths(
    paths: list[Path], denylist: re.Pattern[str], *, repo_root: Path
) -> list[ScanHit]:
    """Scan paths and report redacted locations."""

    hits: list[ScanHit] = []
    for path in paths:
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo_root))
        text = path.read_bytes().decode("utf-8", errors="ignore")
        for hit in scan_text(text, denylist, source=rel):
            hits.append(ScanHit(hit.source, hit.line_number, repository_path=rel))
    return hits


_DIFF_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@")


def _scan_git_diff_text(
    text: str, denylist: re.Pattern[str], *, diff_range: str
) -> list[ScanHit]:
    """Map added Git diff lines to post-image paths and line numbers."""
    hits: list[ScanHit] = []
    repository_path: str | None = None
    post_image_line: int | None = None
    for line in text.splitlines():
        if line.startswith("diff --git "):
            repository_path = None
            post_image_line = None
            continue
        if post_image_line is None and line.startswith("+++ b/"):
            repository_path = line[6:]
            continue
        hunk = _DIFF_HUNK.match(line)
        if hunk:
            post_image_line = int(hunk.group(1))
            continue
        if post_image_line is None or not line:
            continue
        if line.startswith("+"):
            if denylist.search(line[1:]):
                source = f"diff:{diff_range}:{repository_path or 'unknown'}"
                hits.append(ScanHit(source, post_image_line, repository_path))
            post_image_line += 1
        elif line.startswith(" "):
            post_image_line += 1
    return hits


def scan_git_diff(
    repo_root: Path, diff_range: str, denylist: re.Pattern[str]
) -> list[ScanHit]:
    """Scan a Git diff range and report redacted locations."""

    result = subprocess.run(
        [
            "git",
            "diff",
            "--unified=0",
            "--no-ext-diff",
            "--no-textconv",
            "--end-of-options",
            diff_range,
        ],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    hits = _scan_git_diff_text(result.stdout, denylist, diff_range=diff_range)
    canonical_checks = (
        (NCORE_REPOSITORY_PATH, _canonical_diff_matches),
        (ROBOMIMIC_REPOSITORY_PATH, _robomimic_diff_matches),
    )
    for repository_path, check in canonical_checks:
        has_candidate = any(hit.repository_path == repository_path for hit in hits)
        if has_candidate and not check(repo_root, diff_range):
            hits = [
                ScanHit(hit.source, hit.line_number)
                if hit.repository_path == repository_path
                else hit
                for hit in hits
            ]
    return hits


def _canonical_diff_matches(repo_root: Path, diff_range: str) -> bool:
    """Bind an eligible diff to the exact canonical post-image Git blob."""
    try:
        notice = _canonical_notice_bytes(repo_root)
        return _diff_has_post_image(
            repo_root, diff_range, NCORE_REPOSITORY_PATH, notice
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _robomimic_diff_matches(repo_root: Path, diff_range: str) -> bool:
    """Bind an eligible robomimic diff to its exact canonical Git blob."""
    try:
        lock = _canonical_robomimic_lock_bytes(repo_root)
        return _diff_has_post_image(
            repo_root, diff_range, ROBOMIMIC_REPOSITORY_PATH, lock
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return False


def _diff_has_post_image(
    repo_root: Path, diff_range: str, repository_path: str, payload: bytes
) -> bool:
    object_id = subprocess.check_output(
        ["git", "hash-object", "--stdin"], cwd=repo_root, input=payload
    ).strip()
    raw = subprocess.check_output(
        [
            "git",
            "diff",
            "--raw",
            "--no-abbrev",
            "--no-renames",
            "--no-ext-diff",
            "--no-textconv",
            "-z",
            "--end-of-options",
            diff_range,
            "--",
            repository_path,
        ],
        cwd=repo_root,
    )
    fields = raw.split(b"\0")
    if len(fields) != 3 or fields[1] != repository_path.encode() or fields[2]:
        return False
    header = fields[0].split()
    return len(header) == 5 and header[1] == b"100644" and header[3] == object_id


def _canonical_notice_bytes(repo_root: Path) -> bytes:
    """Return canonical bytes only for the exact regular Git index entry."""
    return _canonical_git_bytes(repo_root, NCORE_REPOSITORY_PATH)


def _canonical_robomimic_lock_bytes(repo_root: Path) -> bytes:
    """Return the exact regular robomimic lock Git index entry."""
    return _canonical_git_bytes(
        repo_root,
        ROBOMIMIC_REPOSITORY_PATH,
        expected_object_id=ROBOMIMIC_LOCK_GIT_BLOB_SHA1,
    )


def _canonical_git_bytes(
    repo_root: Path,
    repository_path: str,
    *,
    expected_object_id: str | None = None,
) -> bytes:
    """Return bytes only for one exact, unaliased, regular Git entry."""
    path = repo_root / repository_path
    details = path.lstat()
    if (
        not stat.S_ISREG(details.st_mode)
        or path.is_symlink()
        or details.st_nlink != 1
    ):
        raise ValueError("canonical attribution is not an unaliased regular file")
    payload = path.read_bytes()
    object_id = subprocess.run(
        ["git", "hash-object", "--stdin"],
        cwd=repo_root,
        check=True,
        input=payload,
        stdout=subprocess.PIPE,
    ).stdout.strip()
    if expected_object_id is not None and object_id.decode() != expected_object_id:
        raise ValueError("canonical attribution Git object mismatch")
    result = subprocess.run(
        ["git", "ls-files", "--stage", "-z"],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
    )
    records = [record for record in result.stdout.split(b"\0") if record]
    expected = b"100644 " + object_id + b" 0\t" + repository_path.encode()
    same_objects = [
        record for record in records
        if record.split(b"\t", 1)[0].split()[1] == object_id
    ]
    if expected not in records or same_objects != [expected]:
        raise ValueError("canonical attribution is not an exact regular Git entry")
    canonical_target = path.resolve()
    for record in records:
        if not record.startswith(b"120000 "):
            continue
        candidate = repo_root / record.split(b"\t", 1)[1].decode()
        if candidate.is_symlink() and candidate.resolve() == canonical_target:
            raise ValueError("canonical attribution has a tracked symlink alias")
    return payload


def _ncore_disposition_indexes(
    hits: list[ScanHit], repo_root: Path, proof_directory: Path
) -> set[int]:
    """Return indexes of exact NCore attribution findings after public proof."""
    candidates = {
        index
        for index, hit in enumerate(hits)
        if hit.repository_path == NCORE_REPOSITORY_PATH
        and hit.line_number in NCORE_ATTRIBUTION_LINES
    }
    if not candidates:
        return set()
    verify_public_notice(_canonical_notice_bytes(repo_root), proof_directory)
    return candidates


def _robomimic_disposition_indexes(
    hits: list[ScanHit], repo_root: Path, proof_directory: Path
) -> set[int]:
    """Return exact robomimic attribution indexes after official-byte proof."""
    candidates = {
        index
        for index, hit in enumerate(hits)
        if hit.repository_path == ROBOMIMIC_REPOSITORY_PATH
        and hit.line_number in ROBOMIMIC_ATTRIBUTION_LINES
    }
    if not candidates:
        return set()
    verify_public_license_lock(
        _canonical_robomimic_lock_bytes(repo_root), proof_directory
    )
    return candidates


def should_skip_unconfigured_fork_pull_request(
    pattern_env: str,
    *,
    environ: Mapping[str, str] = os.environ,
) -> bool:
    """Return True when a fork PR workflow cannot access repository secrets."""

    if environ.get(pattern_env, "").strip():
        return False
    if environ.get("GITHUB_EVENT_NAME") != "pull_request":
        return False

    event_path = environ.get("GITHUB_EVENT_PATH")
    if not event_path:
        return False

    try:
        event = json.loads(Path(event_path).read_text(encoding="utf-8"))
    except OSError:
        return False

    pull_request = event.get("pull_request") or {}
    head_repo = (pull_request.get("head") or {}).get("repo") or {}
    head_full_name = head_repo.get("full_name")
    base_full_name = (event.get("repository") or {}).get("full_name") or environ.get(
        "GITHUB_REPOSITORY"
    )
    if not head_full_name or not base_full_name:
        return False
    return head_full_name != base_full_name


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--diff-range", default="")
    parser.add_argument("--tree", action="store_true")
    parser.add_argument(
        "--built-in-nebius-infra",
        action="store_true",
        help="Use the deterministic public Nebius infrastructure leak patterns.",
    )
    parser.add_argument(
        "--stdin-source",
        default="",
        help="Scan stdin and report redacted hits under this source label.",
    )
    parser.add_argument(
        "--stdin-is-diff",
        action="store_true",
        help="Treat stdin as a Git diff and scan only added lines.",
    )
    parser.add_argument(
        "--pattern-env",
        default="CUSTOMER_DENYLIST",
        help="Environment variable containing the denylist regex.",
    )
    parser.add_argument(
        "--pattern-file",
        type=Path,
        help=(
            "Operator-private file containing the denylist regex. If omitted, "
            "the scanner checks ${PATTERN_ENV}_FILE, then "
            "~/.config/npa/<lowercase-pattern-env>.regex."
        ),
    )
    parser.add_argument(
        "--ignore-case",
        action="store_true",
        help="Match denylist regexes case-insensitively.",
    )
    parser.add_argument(
        "--ncore-attribution-proof-directory",
        type=Path,
        help=(
            "Private caller-owned directory used to verify exact public "
            "attributions against independently pinned official sources."
        ),
    )
    args = parser.parse_args(argv)

    if (
        not args.built_in_nebius_infra
        and (args.tree or args.diff_range or args.stdin_source)
        and should_skip_unconfigured_fork_pull_request(args.pattern_env)
    ):
        print(
            "confidentiality scan skipped on fork pull request "
            f"({args.pattern_env} secrets are unavailable to fork workflows)"
        )
        return 0

    try:
        if args.built_in_nebius_infra:
            denylist = compile_builtin_nebius_infra()
        else:
            source_pattern = load_denylist_pattern(
                args.pattern_env,
                pattern_file=args.pattern_file,
            )
            denylist = compile_denylist(
                source_pattern.pattern,
                source=source_pattern.source,
                ignore_case=args.ignore_case,
            )
    except ValueError as exc:
        print(f"confidentiality scan not configured: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"confidentiality scan source cannot be read: {exc}", file=sys.stderr)
        return 2
    except re.error as exc:
        print(f"confidentiality scan regex is invalid: {exc}", file=sys.stderr)
        return 2

    repo_root = args.repo_root.resolve()
    hits: list[ScanHit] = []
    if args.tree:
        hits.extend(
            scan_paths(tracked_text_files(repo_root), denylist, repo_root=repo_root)
        )
    if args.diff_range:
        hits.extend(scan_git_diff(repo_root, args.diff_range, denylist))
    if args.stdin_source:
        stdin_text = sys.stdin.read()
        if args.stdin_is_diff:
            hits.extend(scan_diff_text(stdin_text, denylist, source=args.stdin_source))
        else:
            hits.extend(scan_text(stdin_text, denylist, source=args.stdin_source))

    if hits:
        print("confidentiality scan raw redacted hit locations:", file=sys.stderr)
        for hit in hits:
            print(f"{hit.source}:{hit.line_number}", file=sys.stderr)

    disposition_indexes: set[int] = set()
    may_verify_ncore = (
        args.pattern_env == "CUSTOMER_DENYLIST"
        and not args.built_in_nebius_infra
        and args.ncore_attribution_proof_directory is not None
    )
    if hits and may_verify_ncore:
        verifiers = (
            ("NCore", _ncore_disposition_indexes),
            ("robomimic", _robomimic_disposition_indexes),
        )
        for label, verifier in verifiers:
            try:
                disposition_indexes |= verifier(
                    hits,
                    repo_root,
                    args.ncore_attribution_proof_directory,
                )
            except (
                OSError,
                ValueError,
                subprocess.SubprocessError,
                tarfile.TarError,
            ):
                print(
                    f"{label} public-attribution proof could not be verified",
                    file=sys.stderr,
                )

    unresolved = len(hits) - len(disposition_indexes)
    summary = (
        f"raw={len(hits)} dispositioned={len(disposition_indexes)} "
        f"unresolved={unresolved}"
    )
    if unresolved:
        print(f"confidentiality scan failed; {summary}", file=sys.stderr)
        return 1
    print(f"confidentiality scan passed; {summary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
