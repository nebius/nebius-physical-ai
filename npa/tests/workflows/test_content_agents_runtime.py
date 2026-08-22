from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

import pytest

from npa.workflows import content_agents_runtime as runtime


def _reviewed_test_lock(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    lock = tmp_path / "pylock.toml"
    lock.write_text(
        'name = "ovrtx"\nversion = "0.3.0.312915"\n'
        'url = "https://pypi.nvidia.invalid/exact-x86.whl"\n'
        'sha256 = "reviewed-x86-sha"\n',
        encoding="utf-8",
    )
    monkeypatch.setattr(
        runtime,
        "OVRTX_RUNTIME_LOCK_SHA256",
        hashlib.sha256(lock.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(
        runtime,
        "OVRTX_WHEELS",
        {"x86_64": ("https://pypi.nvidia.invalid/exact-x86.whl", "reviewed-x86-sha")},
    )
    return lock


def test_runtime_identity_is_version_architecture_and_complete_lock_bound(
    tmp_path, monkeypatch
) -> None:
    lock = _reviewed_test_lock(tmp_path, monkeypatch)
    target, architecture, tier = runtime.runtime_path(
        environ={runtime.RUNTIME_CACHE_ENV: str(tmp_path / "cache")}, machine="amd64"
    )

    assert architecture == "x86_64"
    assert tier == "configured-filesystem"
    assert str(target).endswith(
        f"ovrtx/{runtime.OVRTX_VERSION}/x86_64/lock-{runtime.OVRTX_RUNTIME_LOCK_SHA256}"
    )
    runtime._verify_lock(lock, architecture)


def test_bootstrap_uses_unique_temp_and_atomically_publishes_once(
    tmp_path, monkeypatch
) -> None:
    lock = _reviewed_test_lock(tmp_path, monkeypatch)
    calls: list[dict[str, str]] = []

    def fake_run(argv, *, check, env):
        assert argv[-1] == "--provision-only"
        temporary = Path(env["WU_OVRTX_VENV_DIR"])
        assert ".tmp-" in temporary.name
        (temporary / "bin").mkdir(parents=True)
        (temporary / "bin/python").write_text("python", encoding="utf-8")
        (temporary / runtime.UPSTREAM_READY_MARKER).write_text(
            "ready", encoding="utf-8"
        )
        calls.append(env)
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(runtime.subprocess, "run", fake_run)
    monkeypatch.setattr(runtime, "_probe_runtime", lambda target: None)
    env = {
        runtime.RUNTIME_CACHE_ENV: str(tmp_path / "cache"),
        "NPA_CONTENT_AGENTS_OVRTX_LOCK": str(lock),
        "HOME": str(tmp_path / "home"),
    }

    first = runtime.bootstrap_runtime(environ=env, machine="x86_64")
    second = runtime.bootstrap_runtime(environ=env, machine="x86_64")

    assert first == second
    assert len(calls) == 1
    target = Path(first["runtime_path"])
    assert target.is_dir()
    assert not list(target.parent.glob(".*.tmp-*"))
    marker = json.loads((target / runtime.READY_MARKER).read_text(encoding="utf-8"))
    assert marker["cache_identity"] == target.name
    assert not any("TOKEN" in key or "KEY" in key for key in marker)


def test_existing_unverified_identity_fails_closed_without_overwrite(
    tmp_path, monkeypatch
) -> None:
    lock = _reviewed_test_lock(tmp_path, monkeypatch)
    env = {
        runtime.RUNTIME_CACHE_ENV: str(tmp_path / "cache"),
        "NPA_CONTENT_AGENTS_OVRTX_LOCK": str(lock),
    }
    target, _, _ = runtime.runtime_path(environ=env, machine="x86_64")
    target.mkdir(parents=True)
    sentinel = target / "do-not-overwrite"
    sentinel.write_text("partial", encoding="utf-8")

    with pytest.raises(runtime.ContentAgentsRuntimeError, match="not ready"):
        runtime.bootstrap_runtime(environ=env, machine="x86_64")

    assert sentinel.read_text(encoding="utf-8") == "partial"


def test_unsupported_architecture_fails_before_download(tmp_path) -> None:
    with pytest.raises(
        runtime.ContentAgentsRuntimeError, match="no reviewed lock entry"
    ):
        runtime.runtime_path(
            environ={runtime.RUNTIME_CACHE_ENV: str(tmp_path)}, machine="riscv64"
        )


def test_unconfigured_fallback_is_explicitly_node_ephemeral(tmp_path) -> None:
    target, _, tier = runtime.runtime_path(
        environ={"XDG_CACHE_HOME": str(tmp_path)}, machine="x86_64"
    )
    assert tier == "node-ephemeral"
    assert target.is_relative_to(tmp_path / "npa" / "runtime-cache" / "content-agents")


def test_bootstrap_source_contains_single_writer_and_atomic_publication() -> None:
    source = Path(runtime.__file__).read_text(encoding="utf-8")
    for required in (
        "fcntl.LOCK_EX",
        "tempfile.mkdtemp",
        "os.rename(temporary, target)",
        "READY_MARKER",
    ):
        assert required in source


def test_image_inspection_fails_closed_without_uv(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runtime.shutil, "which", lambda _name: None)

    with pytest.raises(runtime.ContentAgentsRuntimeError, match="lacks the pinned uv"):
        runtime.inspect_image()
