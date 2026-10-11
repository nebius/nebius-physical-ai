"""Catch unsupported stdlib calls before the overnight interpreter audit."""

import ast
import hashlib
from pathlib import Path
import subprocess

import pytest

from npa.exception_notes import add_exception_note
from npa.workflows.navigation.artifacts import file_sha256


ROOT = Path(__file__).resolve().parents[2]
PRODUCTION_ROOTS = (
    "npa/src/",
    "npa/scripts/",
    "npa/docker/",
    "npa/workflows/",
    "workflows/implementations/",
)


def _unsupported_api(node):
    if isinstance(node, ast.ImportFrom):
        unsupported = {"datetime": "UTC", "hashlib": "file_digest"}
        return any(alias.name == unsupported.get(node.module) for alias in node.names)
    if isinstance(node, ast.Attribute):
        if node.attr == "add_note":
            return True
        return isinstance(node.value, ast.Name) and (node.value.id, node.attr) in {
            ("hashlib", "file_digest"),
            ("datetime", "UTC"),
        }
    return False


def test_production_code_keeps_python310_stdlib_compatibility():
    """Reject known unsupported stdlib APIs across shipped source and scripts.

    Args:
        None.
    Returns:
        None.
    Raises:
        AssertionError: Production code requires a Python 3.11-only API.
    """
    paths = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.py"],
        cwd=ROOT,
        capture_output=True,
        check=True,
        text=True,
    ).stdout.split("\0")
    violations = []
    for name in paths:
        if not name.startswith(PRODUCTION_ROOTS):
            continue
        source = (ROOT / name).read_text()
        if not any(token in source for token in ("file_digest", "UTC", "add_note")):
            continue
        for node in ast.walk(ast.parse(source)):
            if _unsupported_api(node):
                violations.append(f"{name}:{node.lineno}")
    assert not violations, "Python 3.11-only stdlib APIs: " + ", ".join(violations)


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("native_notes", [False, True])
def test_secondary_failure_notes_preserve_original_error(error_type, native_notes):
    """Retain all secondary notes and the exact original exception.

    Args:
        error_type: Failure or interruption class.
        native_notes: Whether the native note method remains available.
    Returns:
        None.
    Raises:
        AssertionError: Notes disappear or the authoritative failure changes.
    """
    error = error_type("original failure")
    if not native_notes:
        error.add_note = None
    add_exception_note(error, "cleanup failed")
    add_exception_note(error, "receipt failed")
    assert error.__notes__ == ["cleanup failed", "receipt failed"]
    assert str(error) == "original failure"
    with pytest.raises(error_type) as caught:
        raise error
    assert caught.value is error


@pytest.mark.parametrize("payload", [b"", b"binary\x00payload" * 100_000])
def test_navigation_checksum_keeps_exact_bytes_across_chunks(tmp_path, payload):
    """Verify empty and multi-chunk binary checksums without optional APIs.

    Args:
        tmp_path: Test-owned directory.
        payload: Exact file bytes spanning zero or multiple chunks.
    Returns:
        None.
    Raises:
        AssertionError: The checksum differs from the exact file bytes.
    """
    path = tmp_path / "checkpoint.bin"
    path.write_bytes(payload)
    assert file_sha256(path) == hashlib.sha256(payload).hexdigest()
