"""Keep NCore publication subprocess output and evidence outside the checkout."""

import hashlib
import json
import os
from pathlib import Path
import subprocess

from image_byte_scan import core as W

ROOT = Path(__file__).resolve().parents[3]
PYTHON = ROOT / "npa/.venv/bin/python"
CONTEXT = ("npa/src/npa", "npa/docker/workbench/ncore")
SOURCE_PATHS = ("npa/docker/workbench/ncore", "npa/scripts/image_byte_scan", "npa/scripts/ncore_publication",
                "npa/src/npa/deploy/images.py", "npa/src/npa/deploy/publish_public.py",
                "npa/src/npa/deploy/ncore_selected_sbom.py", "npa/src/npa/_public_https.py",
                "npa/src/npa/workflows/ncore_runtime.py", "npa/tests/conftest.py", "npa/pyproject.toml",
                "npa/scripts/publish_ncore_oci.py", "npa/scripts/scan_image_bytes.py",
                "npa/scripts/scan_image_omniverse_payload.py",
                "npa/tests/docker/test_image_byte_go_build.py",
                "npa/tests/docker/test_packaging_contract.py",
                "npa/tests/docker/test_ncore_image_contract.py",
                "npa/docker/workbench/packaging-contract.yaml", ".gitleaks.toml",
                ".trivyignore", ".github/workflows/publish-public-images.yml")


def run(argv, output, *, env=None, input_bytes=None):
    """Execute a gate with private output and sanitized failure text.

    Args:
        argv: Argument vector without credentials.
        output: New private stdout path; stderr uses the .stderr suffix.
        env: Explicit subprocess environment, or the current environment.
        input_bytes: Optional stdin bytes.
    Returns:
        None.
    Raises:
        OSError: A process or private output cannot be opened.
        ValueError: The subprocess failed.
    """
    with output.open("xb") as stdout, output.with_suffix(output.suffix + ".stderr").open("xb") as stderr:
        result = subprocess.run(argv, cwd=ROOT, env=env, input=input_bytes,
                                stdout=stdout, stderr=stderr, check=False)
    W.require(result.returncode == 0, "publication_subprocess_failed_see_private_evidence")


def file_sha(path):
    """Hash a file without retaining its contents.

    Args:
        path: File to hash.
    Returns:
        SHA256 hex digest.
    Raises:
        OSError: The file cannot be read.
    """
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def write_json(path, value):
    """Create private evidence without replacing prior evidence.

    Args:
        path: New file path.
        value: JSON value.
    Returns:
        None.
    Raises:
        OSError: The file exists or cannot be written.
    """
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, sort_keys=True, indent=2)
        stream.write("\n")


def committed_source(sha):
    """Require the full checked-out SHA and unchanged gate/build source closure.

    Args:
        sha: Exact reviewed commit.
    Returns:
        SHA256 of the complete committed build-context archive.
    Raises:
        ValueError: Source identity or the relevant worktree differs.
        subprocess.CalledProcessError: Git cannot read the source.
    """
    import re

    W.require(re.fullmatch("[0-9a-f]{40}", sha), "full_source_sha_required")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
    W.require(head == sha, "source_sha_must_equal_head")
    changed = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=all", "--", *SOURCE_PATHS], cwd=ROOT)
    W.require(not changed, "committed_build_and_gate_source_required")
    # git archive exports tracked files and their committed modes, including the
    # entire NPA source tree. Generated catalog/cache/private inputs are absent.
    archive = subprocess.check_output(["git", "archive", sha, *CONTEXT], cwd=ROOT)
    return hashlib.sha256(archive).hexdigest()


def public_environment():
    """Return an environment without private denylist and runtime credentials.

    Args:
        None.
    Returns:
        Minimal environment for build/runtime/scanner subprocesses.
    Raises:
        None.
    """
    names = ("PATH", "HOME", "DOCKER_CONFIG", "DOCKER_HOST", "DOCKER_CONTEXT",
             "SSL_CERT_FILE", "SSL_CERT_DIR", "LANG")
    return {name: os.environ[name] for name in names if name in os.environ}
