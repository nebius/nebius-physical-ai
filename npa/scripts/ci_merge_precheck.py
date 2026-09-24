"""Check committed merge inputs without checking out or executing candidate code."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import ci_requirements


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def _supports_merge_tree_write_tree(root: Path) -> bool:
    """Return whether the installed Git supports virtual merge-tree writes."""
    result = subprocess.run(
        ["git", "merge-tree", "-h"],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    return "--write-tree" in result.stdout + result.stderr


def _legacy_merge_tree(root: Path, base: str, head: str) -> str:
    """Create a virtual merge tree with the plumbing available in Git 2.34."""
    merge_bases = _git(root, "merge-base", "--all", base, head).splitlines()
    with tempfile.TemporaryDirectory(prefix="npa-merge-tree-") as directory:
        snapshot = Path(directory)
        work_tree = snapshot / "worktree"
        work_tree.mkdir()
        environment = {
            **os.environ,
            "GIT_INDEX_FILE": str(snapshot / "index"),
            "GIT_WORK_TREE": str(work_tree),
        }
        checkout = subprocess.run(
            ["git", "read-tree", "--reset", "-u", base],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        checkout.check_returncode()
        result = subprocess.run(
            ["git", "merge-recursive", *merge_bases, "--", base, head],
            cwd=root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode == 1:
            unresolved = subprocess.check_output(
                ["git", "ls-files", "-u", "-z"],
                cwd=root,
                env=environment,
                text=True,
            )
            conflicts = sorted(
                {
                    entry.split("\t", 1)[1]
                    for entry in unresolved.split("\0")
                    if "\t" in entry
                }
            )
            raise ValueError("Merge conflicts: " + ", ".join(conflicts))
        result.check_returncode()
        return subprocess.check_output(
            ["git", "write-tree"], cwd=root, env=environment, text=True
        ).strip()


def _merge_tree(root: Path, base: str, head: str) -> str:
    if not _supports_merge_tree_write_tree(root):
        return _legacy_merge_tree(root, base, head)
    result = subprocess.run(
        ["git", "merge-tree", "--write-tree", "--name-only", base, head],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode == 1:
        conflicts = result.stdout.split("\n\n", 1)[0].splitlines()[1:]
        raise ValueError("Merge conflicts: " + ", ".join(conflicts))
    result.check_returncode()
    return result.stdout.splitlines()[0]


def check_merge(root: Path, base_ref: str, head_ref: str) -> dict:
    """Validate the combined dependency inputs of two committed revisions.

    Args:
        root: Repository containing both revisions.
        base_ref: Current target revision, fetched by the caller.
        head_ref: Committed candidate revision; staged/working files are excluded.
    Returns:
        Exact compared SHAs, merged tree, and verified dependency fingerprint.
    Raises:
        ValueError: The merge conflicts or its dependency fingerprint is stale.
        subprocess.CalledProcessError: Git cannot resolve or read the revisions.
    """
    base, head = (
        _git(root, "rev-parse", "--verify", "--end-of-options", f"{ref}^{{commit}}")
        for ref in (base_ref, head_ref)
    )
    tree = _merge_tree(root, base, head)
    with tempfile.TemporaryDirectory(prefix="npa-merge-precheck-") as directory:
        snapshot = Path(directory)
        for name in ("pyproject.toml", "ci/constraints.in", "ci/requirements.txt"):
            path = snapshot / "npa" / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(
                subprocess.check_output(["git", "show", f"{tree}:npa/{name}"], cwd=root)
            )
        ci_requirements._check(snapshot)
        fingerprint = ci_requirements._fingerprint(snapshot)
    return {"base": base, "head": head, "tree": tree, "fingerprint": fingerprint}


def _main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    args = parser.parse_args()
    try:
        report = check_merge(ci_requirements._ROOT, args.base, args.head)
    except (ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f"Merge precheck failed: {error}\n")
    print(json.dumps(report, indent=2))
    print("Committed merge inputs passed; this does not run the full CI suite.")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
