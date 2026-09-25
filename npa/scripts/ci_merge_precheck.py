"""Check committed merge inputs without checking out or executing candidate code."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import tempfile

import ci_requirements


def _git(root: Path, *arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=root, text=True).strip()


def _merge_tree(root: Path, base: str, head: str) -> str:
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
