"""Secret-sourced confidentiality denylist scanner.

The scanner intentionally reports only locations, never matched text.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import sys


@dataclass(frozen=True)
class ScanHit:
    """A redacted denylist match location."""

    source: str
    line_number: int


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


def scan_text(text: str, denylist: re.Pattern[str], *, source: str) -> list[ScanHit]:
    """Return redacted hit locations for text."""

    hits: list[ScanHit] = []
    for line_number, line in enumerate(text.splitlines(), start=1):
        if denylist.search(line):
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


def scan_paths(paths: list[Path], denylist: re.Pattern[str], *, repo_root: Path) -> list[ScanHit]:
    """Scan paths and report redacted locations."""

    hits: list[ScanHit] = []
    for path in paths:
        if not path.is_file():
            continue
        rel = str(path.relative_to(repo_root))
        text = path.read_bytes().decode("utf-8", errors="ignore")
        hits.extend(scan_text(text, denylist, source=rel))
    return hits


def scan_git_diff(repo_root: Path, diff_range: str, denylist: re.Pattern[str]) -> list[ScanHit]:
    """Scan a Git diff range and report redacted locations."""

    result = subprocess.run(
        ["git", "diff", "--unified=0", "--no-ext-diff", diff_range],
        cwd=repo_root,
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    )
    return scan_text(result.stdout, denylist, source=f"diff:{diff_range}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--diff-range", default="")
    parser.add_argument("--tree", action="store_true")
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
    args = parser.parse_args(argv)

    try:
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
        hits.extend(scan_paths(tracked_text_files(repo_root), denylist, repo_root=repo_root))
    if args.diff_range:
        hits.extend(scan_git_diff(repo_root, args.diff_range, denylist))

    if hits:
        print("confidentiality scan failed; redacted hit locations:", file=sys.stderr)
        for hit in hits:
            print(f"{hit.source}:{hit.line_number}", file=sys.stderr)
        return 1
    print("confidentiality scan passed; hits=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
