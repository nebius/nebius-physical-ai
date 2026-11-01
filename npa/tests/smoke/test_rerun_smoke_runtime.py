from __future__ import annotations

import importlib
import json
import stat
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


def _write_executable(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    path.chmod(stat.S_IRUSR | stat.S_IWUSR | stat.S_IXUSR)


def _python_path(tmp_path: Path) -> Path:
    python = tmp_path / "venv/bin/python"
    _write_executable(python)
    return python


def test_rerun_cli_prefers_executable_interpreter_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolver = importlib.import_module("npa.rerun_cli")
    python = _python_path(tmp_path)
    sibling = python.with_name("rerun")
    ambient = tmp_path / "ambient/rerun"
    _write_executable(sibling)
    _write_executable(ambient)
    monkeypatch.setattr(resolver.sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(ambient.parent))

    assert resolver.resolve_rerun_cli() == str(sibling)


@pytest.mark.parametrize("invalid_kind", ["non-executable", "directory"])
def test_rerun_cli_falls_back_from_invalid_interpreter_sibling(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, invalid_kind: str
) -> None:
    resolver = importlib.import_module("npa.rerun_cli")
    python = _python_path(tmp_path)
    sibling = python.with_name("rerun")
    if invalid_kind == "directory":
        sibling.mkdir()
    else:
        sibling.write_text("not executable", encoding="utf-8")
    ambient = tmp_path / "ambient/rerun"
    _write_executable(ambient)
    monkeypatch.setattr(resolver.sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(ambient.parent))

    assert resolver.resolve_rerun_cli() == str(ambient)


def test_rerun_cli_fails_when_no_executable_is_available(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    resolver = importlib.import_module("npa.rerun_cli")
    python = _python_path(tmp_path)
    monkeypatch.setattr(resolver.sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(RuntimeError, match="Rerun CLI is unavailable"):
        resolver.resolve_rerun_cli()


class _Recording:
    def save(self, path: Path) -> None:
        Path(path).write_bytes(b"rrd")

    def log(self, *_args, **_kwargs) -> None:
        pass

    def set_time(self, *_args, **_kwargs) -> None:
        pass

    def flush(self) -> None:
        pass

    def disconnect(self) -> None:
        pass


class _Process:
    returncode = None

    def poll(self) -> None:
        return None

    def terminate(self) -> None:
        pass

    def wait(self, *, timeout: int) -> int:
        return 0

    def kill(self) -> None:
        pass


def test_viewer_smoke_uses_interpreter_sibling_for_every_subprocess(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.smoke import test_rerun_viewer_functional as viewer

    python = _python_path(tmp_path)
    sibling = python.with_name("rerun")
    ambient = tmp_path / "ambient/rerun"
    _write_executable(sibling)
    _write_executable(ambient)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(ambient.parent))
    output = tmp_path / "output"
    monkeypatch.setenv("NPA_IMAGE_SOURCE_SHA", "a" * 40)
    monkeypatch.setenv("NPA_SMOKE_OUTPUT_DIR", str(output))
    monkeypatch.setattr(viewer.rr, "RecordingStream", lambda *_args: _Recording())
    monkeypatch.setattr(viewer, "_free_port", lambda: 9000)
    monkeypatch.setattr(viewer, "_wait_for_http", lambda *_args: b"<html />")
    run_calls: list[list[str]] = []
    popen_calls: list[list[str]] = []

    def run(argv, **_kwargs):
        run_calls.append(argv)
        stdout = "robot/joints/positions" if argv[2] == "print" else ""
        return SimpleNamespace(stdout=stdout)

    def popen(argv, **_kwargs):
        popen_calls.append(argv)
        return _Process()

    monkeypatch.setattr(viewer.subprocess, "run", run)
    monkeypatch.setattr(viewer.subprocess, "Popen", popen)

    assert viewer.main() == 0
    assert [argv[0] for argv in [*run_calls, *popen_calls]] == [str(sibling)] * 3
    report = json.loads((output / "rerun-viewer-functional.json").read_text())
    assert report["status"] == "passed"


def test_curobo_decode_rejects_nonexecutable_sibling_for_path_copy(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.curobo import artifacts

    python = _python_path(tmp_path)
    sibling = python.with_name("rerun")
    sibling.write_text("not executable", encoding="utf-8")
    ambient = tmp_path / "ambient/rerun"
    _write_executable(ambient)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(ambient.parent))
    calls: list[list[str]] = []

    def reject_verify(argv, **_kwargs):
        calls.append(argv)
        return SimpleNamespace(returncode=1)

    monkeypatch.setattr(artifacts.subprocess, "run", reject_verify)
    with pytest.raises(artifacts.CuroboError, match="Rerun rejected"):
        artifacts._decode_rrd_to(
            tmp_path / "input.rrd",
            tmp_path / "decoded.txt",
            rows=[],
            run_id="resolver-control",
        )

    assert calls[0][0] == str(ambient)


def test_curobo_decode_fails_before_subprocess_without_rerun(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from npa.workbench.curobo import artifacts

    python = _python_path(tmp_path)
    monkeypatch.setattr(sys, "executable", str(python))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))
    monkeypatch.setattr(
        artifacts.subprocess,
        "run",
        lambda *_args, **_kwargs: pytest.fail("subprocess started"),
    )

    with pytest.raises(artifacts.CuroboError, match="Rerun CLI is unavailable"):
        artifacts._decode_rrd_to(
            tmp_path / "input.rrd",
            tmp_path / "decoded.txt",
            rows=[],
            run_id="missing-control",
        )
