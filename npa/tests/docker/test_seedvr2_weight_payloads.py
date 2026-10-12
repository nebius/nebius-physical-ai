"""SeedVR2 final-layer weight scanning must fail closed."""

from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path

import pytest

SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "docker"
    / "workbench"
    / "seedvr2"
    / "verify_no_weight_payloads.py"
)
SPEC = importlib.util.spec_from_file_location("verify_no_weight_payloads", SCRIPT)
assert SPEC and SPEC.loader
scanner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(scanner)


def _tree(tmp_path: Path) -> tuple[tuple[Path, ...], Path]:
    source = tmp_path / "npa-src"
    service = tmp_path / "npa-venv"
    source.mkdir()
    allowed = service / "lib/python3.12/site-packages/rerun_sdk.pth"
    allowed.parent.mkdir(parents=True)
    allowed.write_bytes(scanner.RERUN_PATH_FILE_BYTES)
    return (source, service), allowed


def test_exact_rerun_path_file_is_reported(tmp_path: Path) -> None:
    roots, allowed = _tree(tmp_path)
    report = scanner.verify(
        roots,
        allowed_path_file=allowed,
        expected_path_file_bytes=scanner.RERUN_PATH_FILE_BYTES,
    )
    record = report["allowed_python_path_file"]
    assert record["bytes"] == 10
    assert record["sha256"] == hashlib.sha256(b"rerun_sdk\n").hexdigest()
    assert report["rejected_payloads"] == []


@pytest.mark.parametrize(
    "name",
    [
        "weights.pth",
        "weights.PT",
        "weights.safetensors",
        "weights.ckpt",
        "pytorch_model-00001-of-00002.bin",
    ],
)
def test_every_other_model_like_file_is_rejected(tmp_path: Path, name: str) -> None:
    roots, allowed = _tree(tmp_path)
    (roots[0] / name).write_bytes(b"payload")
    with pytest.raises(ValueError, match="model-like payloads are forbidden"):
        scanner.verify(
            roots,
            allowed_path_file=allowed,
            expected_path_file_bytes=scanner.RERUN_PATH_FILE_BYTES,
        )


def test_model_like_directory_is_rejected(tmp_path: Path) -> None:
    roots, allowed = _tree(tmp_path)
    (roots[0] / "hidden.pth").mkdir()
    with pytest.raises(ValueError, match="model-like payloads are forbidden"):
        scanner.verify(
            roots,
            allowed_path_file=allowed,
            expected_path_file_bytes=scanner.RERUN_PATH_FILE_BYTES,
        )


def test_allowed_path_is_verified_after_mutation(tmp_path: Path) -> None:
    roots, allowed = _tree(tmp_path)
    allowed.write_bytes(b"model payload")
    with pytest.raises(ValueError, match="bytes differ"):
        scanner.verify(
            roots,
            allowed_path_file=allowed,
            expected_path_file_bytes=scanner.RERUN_PATH_FILE_BYTES,
        )


def test_allowed_path_must_be_a_regular_file(tmp_path: Path) -> None:
    roots, allowed = _tree(tmp_path)
    allowed.unlink()
    target = tmp_path / "rerun-path-target"
    target.write_bytes(scanner.RERUN_PATH_FILE_BYTES)
    allowed.symlink_to(target)
    with pytest.raises(ValueError, match="not regular"):
        scanner.verify(
            roots,
            allowed_path_file=allowed,
            expected_path_file_bytes=scanner.RERUN_PATH_FILE_BYTES,
        )


def test_missing_root_fails_closed(tmp_path: Path) -> None:
    roots, allowed = _tree(tmp_path)
    with pytest.raises(ValueError, match="required scan root"):
        scanner.verify(
            (roots[0], tmp_path / "missing"),
            allowed_path_file=allowed,
            expected_path_file_bytes=scanner.RERUN_PATH_FILE_BYTES,
        )


def test_walk_error_is_not_masked(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    roots, allowed = _tree(tmp_path)

    def denied(_root, *, followlinks, onerror):
        assert followlinks is False
        onerror(PermissionError("denied"))
        yield

    monkeypatch.setattr(scanner.os, "walk", denied)
    with pytest.raises(PermissionError, match="denied"):
        scanner.verify(
            roots,
            allowed_path_file=allowed,
            expected_path_file_bytes=scanner.RERUN_PATH_FILE_BYTES,
        )
