"""Reject security regressions between a Git base and a proposed merge snapshot."""

from __future__ import annotations

import argparse
import collections
import io
import json
import os
import shutil
import subprocess
from pathlib import Path

from security_dependencies import scan_dependencies
from security_source import scan_source


def _git(root: Path, *arguments: str) -> bytes:
    return subprocess.run(["git", "-C", str(root), *arguments], check=True,
                          stdout=subprocess.PIPE).stdout


def _tree_entries(root: Path, commit: str) -> list[tuple[str, str]]:
    entries = []
    for record in _git(root, "ls-tree", "-r", "-z", "--full-tree", commit).split(b"\0"):
        if not record:
            continue
        metadata, raw_path = record.split(b"\t", 1)
        mode, kind, digest = metadata.decode().split()
        path = raw_path.decode()
        if mode == "120000" and path in {".agents/skills", ".claude/skills"}:
            continue
        if mode not in {"100644", "100755"} or kind != "blob":
            raise ValueError(f"Unsupported source entry: {path}")
        entries.append((path, digest))
    return entries


def _write_blobs(entries: list[tuple[str, str]], stream: io.BytesIO, destination: Path) -> None:
    for relative, expected_digest in entries:
        digest, kind, size = stream.readline().decode().split()
        if digest != expected_digest or kind != "blob":
            raise ValueError("Git returned an unexpected source object")
        content = stream.read(int(size))
        if len(content) != int(size) or stream.read(1) != b"\n":
            raise ValueError("Git returned an incomplete source object")
        target = destination / relative
        if destination.resolve() not in target.resolve().parents:
            raise ValueError("Git source path escapes the snapshot")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)


def _snapshot_revision(root: Path, revision: str, destination: Path) -> str:
    commit = _git(root, "rev-parse", "--verify", f"{revision}^{{commit}}").decode().strip()
    entries = _tree_entries(root, commit)
    requested = "".join(f"{digest}\n" for _, digest in entries).encode()
    # Reading raw blobs prevents export-ignore/export-subst from hiding candidate code.
    result = subprocess.run(["git", "-C", str(root), "cat-file", "--batch"],
                            input=requested, stdout=subprocess.PIPE, check=True)
    _write_blobs(entries, io.BytesIO(result.stdout), destination)
    return commit


def _snapshot_working(root: Path, destination: Path) -> None:
    paths = _git(root, "ls-files", "--cached", "--others", "--exclude-standard", "-z")
    for relative in paths.decode().split("\0"):
        if not relative or relative in {".agents/skills", ".claude/skills"}:
            continue
        source = root / relative
        if source.is_symlink() or root not in source.resolve().parents:
            raise ValueError(f"Unsupported source symlink: {relative}")
        if not source.exists():
            continue
        destination_file = destination / relative
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination_file)


def regressions(base: list[dict], candidate: list[dict]) -> list[dict]:
    """Find added occurrences without allowing duplicate findings to cancel out.

    Args:
        base: Findings from the base revision under the same scanner policy.
        candidate: Findings from the proposed merge.
    Returns:
        Candidate findings whose matching base occurrence has not been consumed.
    Raises:
        KeyError: A scanner omitted a required identity field.
    """
    def key(finding: dict) -> tuple:
        return tuple(finding[field] for field in ("scanner", "path", "rule", "identity"))

    remaining = collections.Counter(key(finding) for finding in base)
    added = []
    for finding in candidate:
        identity = key(finding)
        if remaining[identity]:
            remaining[identity] -= 1
        else:
            added.append(finding)
    return added


def _scan(root: Path, output: Path, cache: Path) -> list[dict]:
    if not (root / "npa/pyproject.toml").is_file():
        raise ValueError("Required application dependency manifest is missing")
    findings = scan_source(root, output / "source")
    findings.extend(scan_dependencies(root, output / "dependencies", cache))
    (output / "findings.json").write_text(json.dumps(findings, indent=2))
    return findings


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True, help="Actual target branch commit")
    parser.add_argument("--head", help="Candidate merge commit; defaults to working files")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path, required=True,
                        help="New private directory outside the checkout")
    return parser.parse_args()


def _report_regressions(added: list[dict], summary: dict, output: Path) -> int:
    summary["regressions"] = added
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    for finding in added:
        print(f"{finding['path']}:{finding['line']}: {finding['scanner']} "
              f"{finding['rule']}: {finding['message']}")
    print(f"Security regression gate: {len(added)} new findings")
    return int(bool(added))


def main() -> int:
    """Run the security gate and write an actionable, source-free summary.

    Args:
        None; arguments are read from the command line.
    Returns:
        Zero for no regressions, one for findings, two for operational failure.
    Raises:
        OSError: The private output directory cannot be created.
    """
    arguments = _arguments()
    root = arguments.repo_root.resolve()
    output = arguments.output_dir.resolve()
    if output == root or root in output.parents:
        raise ValueError("Keep security reports outside the checkout")
    os.umask(0o077)
    output.mkdir(parents=True, exist_ok=False)
    try:
        base_commit = _snapshot_revision(root, arguments.base, output / "base")
        if arguments.head:
            head_commit = _snapshot_revision(root, arguments.head, output / "candidate")
        else:
            _snapshot_working(root, output / "candidate")
            head_commit = "working-files"
        base = _scan(output / "base", output / "base-report", output / "cache")
        candidate = _scan(output / "candidate", output / "candidate-report", output / "cache")
        added = regressions(base, candidate)
        summary = {"base": base_commit, "head": head_commit, "base_findings": len(base),
                   "candidate_findings": len(candidate)}
        return _report_regressions(added, summary, output)
    except (OSError, ValueError, KeyError, TypeError, RuntimeError, subprocess.CalledProcessError) as error:
        (output / "failure.txt").write_text(str(error))
        print(f"Security gate could not complete ({type(error).__name__}); "
              "rerun the documented command locally and inspect private failure.txt and scanner logs")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
