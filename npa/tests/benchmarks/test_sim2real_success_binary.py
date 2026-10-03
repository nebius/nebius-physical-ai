"""Verify deterministic Rerun executable selection for Sim2Real evidence."""

from __future__ import annotations

from pathlib import Path
import subprocess
import sys

import pytest

from npa.benchmarks import sim2real_success


def _write_executable(path: Path, *, executable: bool = True) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(0o755 if executable else 0o644)


def test_rerun_binary_prefers_active_interpreter_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "venv/bin/python"
    sibling = python.with_name("rerun")
    ambient = tmp_path / "ambient/rerun"
    _write_executable(python)
    _write_executable(sibling)
    _write_executable(ambient)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(ambient.parent))

    assert sim2real_success._resolve_rerun_binary("rerun") == str(sibling)


def test_rerun_binary_ignores_non_executable_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "venv/bin/python"
    sibling = python.with_name("rerun")
    ambient = tmp_path / "ambient/rerun"
    _write_executable(python)
    _write_executable(sibling, executable=False)
    _write_executable(ambient)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(ambient.parent))

    assert sim2real_success._resolve_rerun_binary("rerun") == str(ambient)


def test_rerun_binary_preserves_explicit_path() -> None:
    explicit = "./tools/review-rerun"
    assert sim2real_success._resolve_rerun_binary(explicit) == explicit


def test_empty_rerun_binary_is_rejected() -> None:
    with pytest.raises(sim2real_success.VerificationError, match="non-empty"):
        sim2real_success._resolve_rerun_binary("")


def test_missing_bare_rerun_binary_fails_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "venv/bin/python"
    _write_executable(python)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", "")

    with pytest.raises(sim2real_success.VerificationError, match="rerun-sdk"):
        sim2real_success._resolve_rerun_binary("rerun")


def test_rrd_verifier_launches_resolved_sibling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    python = tmp_path / "venv/bin/python"
    sibling = python.with_name("rerun")
    recording = tmp_path / "sim2real.rrd"
    _write_executable(python)
    _write_executable(sibling)
    recording.write_bytes(b"rrd")
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", "")
    commands: list[list[str]] = []

    def run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess[str]:
        commands.append(command)
        return subprocess.CompletedProcess(command, 0, stdout="decoded", stderr="")

    monkeypatch.setattr(sim2real_success.subprocess, "run", run)
    result = sim2real_success._verify_rrd(recording, rerun_bin="rerun")

    assert commands == [[str(sibling), "rrd", "print", str(recording)]]
    assert result["size_bytes"] == 3


def test_rrd_verifier_reports_explicit_launch_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    recording = tmp_path / "sim2real.rrd"
    recording.write_bytes(b"rrd")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise FileNotFoundError("missing executable")

    monkeypatch.setattr(sim2real_success.subprocess, "run", fail)
    with pytest.raises(
        sim2real_success.VerificationError, match="could not be launched"
    ):
        sim2real_success._verify_rrd(recording, rerun_bin="./missing-explicit-rerun")
