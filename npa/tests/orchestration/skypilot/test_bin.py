from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.skypilot._bin import SkyPilotNotInstalledError, resolve_sky_bin


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


def test_resolve_sky_bin_explicit_path_wins(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    explicit = _executable(tmp_path / "explicit-sky")
    env = _executable(tmp_path / "env-sky")
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(env))

    assert resolve_sky_bin(explicit) == explicit.resolve()


def test_resolve_sky_bin_env_var(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env = _executable(tmp_path / "env-sky")
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(env))
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    assert resolve_sky_bin() == env.resolve()


def test_resolve_sky_bin_path_discovery(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    discovered = _executable(bin_dir / "sky")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.setenv("PATH", str(bin_dir))

    assert resolve_sky_bin() == discovered.resolve()


def test_resolve_sky_bin_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    with pytest.raises(SkyPilotNotInstalledError, match="SkyPilot CLI executable was not found"):
        resolve_sky_bin()
