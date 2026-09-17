"""Classify whether a candidate needs the expensive image-security checks."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

_EXACT_PATHS = frozenset(
    {
        ".gitleaks.toml",
        "npa/scripts/scan_base_images.py",
        "npa/scripts/scan_image_omniverse_payload.py",
        "npa/src/npa/deploy/images.py",
        "scripts/ci_image_security_scope.py",
        "scripts/security-requirements.txt",
        "scripts/security_install.sh",
        "trivy.yaml",
    }
)
_PATH_PREFIXES = (
    ".github/",
    "docs/security/",
    "npa/docker/",
    "npa/scripts/image_byte_scan/",
    "npa/tests/docker/",
)


def changed_paths(repo_root: Path, base: str, head: str) -> list[str]:
    """Return paths changed between two verified Git commits.

    Args:
        repo_root: Repository containing both commits.
        base: Trusted target commit.
        head: Candidate commit.
    Returns:
        Repository-relative changed paths.
    Raises:
        subprocess.CalledProcessError: A revision or diff cannot be resolved.
    """

    for revision in (base, head):
        subprocess.run(
            ["git", "rev-parse", "--verify", f"{revision}^{{commit}}"],
            cwd=repo_root,
            check=True,
            capture_output=True,
        )
    result = subprocess.run(
        ["git", "diff", "--name-only", "-z", base, head, "--"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    )
    return [path.decode() for path in result.stdout.split(b"\0") if path]


def needs_deep_image_security(paths: list[str]) -> bool:
    """Return whether changed paths can affect image or packaging security.

    Args:
        paths: Repository-relative changed paths.
    Returns:
        ``True`` when the complete image gate must run.
    Raises:
        None.
    """

    return any(
        path in _EXACT_PATHS or path.startswith(_PATH_PREFIXES) for path in paths
    )


def main() -> int:
    """Print the fail-closed image-security scope decision.

    Args:
        None.
    Returns:
        Process exit status.
    Raises:
        subprocess.CalledProcessError: Git cannot prove the comparison range.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--base", required=True)
    parser.add_argument("--head", required=True)
    arguments = parser.parse_args()
    paths = changed_paths(arguments.repo_root, arguments.base, arguments.head)
    print("true" if needs_deep_image_security(paths) else "false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
