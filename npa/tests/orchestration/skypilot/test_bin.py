from __future__ import annotations

from pathlib import Path

import pytest

from npa.orchestration.skypilot import _bin as bin_module
from npa.orchestration.skypilot._bin import (
    REQUIRED_SKYPILOT_VERSION,
    SkyPilotConfigError,
    SkyPilotNotInstalledError,
    SkyPilotVersionError,
    clear_skypilot_version_cache,
    ensure_skypilot_version,
    resolve_config,
    resolve_sky_bin,
)


def _executable(path: Path) -> Path:
    path.write_text("#!/bin/sh\n", encoding="utf-8")
    path.chmod(0o755)
    return path


@pytest.fixture(autouse=True)
def _isolated_skypilot_resolution(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_skypilot_version_cache()
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing-config.yaml")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG", raising=False)
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", raising=False)


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


def test_resolve_sky_bin_config_file_default(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    discovered = _executable(tmp_path / "config-sky")
    config = tmp_path / "config.yaml"
    config.write_text(f"skypilot:\n  sky_bin: {discovered}\n", encoding="utf-8")
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)

    assert resolve_sky_bin() == discovered.resolve()


def test_resolve_sky_bin_missing_raises(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("NPA_SKYPILOT_BIN", raising=False)
    monkeypatch.setattr(bin_module, "CONFIG_PATH", tmp_path / "missing.yaml")

    with pytest.raises(SkyPilotNotInstalledError, match="SkyPilot CLI executable is not configured"):
        resolve_sky_bin()


def test_resolve_config_precedence_explicit_env_config(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    explicit = _executable(tmp_path / "explicit-sky")
    env = _executable(tmp_path / "env-sky")
    default = _executable(tmp_path / "default-sky")
    config_global = tmp_path / "config-global.yaml"
    env_global = tmp_path / "env-global.yaml"
    explicit_global = tmp_path / "explicit-global.yaml"
    config_isolated = tmp_path / "config-isolated"
    env_isolated = tmp_path / "env-isolated"
    explicit_isolated = tmp_path / "explicit-isolated"
    config = tmp_path / "config.yaml"
    config.write_text(
        "\n".join(
            [
                "skypilot:",
                f"  sky_bin: {default}",
                f"  global_config_path: {config_global}",
                f"  isolated_config_dir: {config_isolated}",
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)
    monkeypatch.setenv("NPA_SKYPILOT_BIN", str(env))
    monkeypatch.setenv("SKYPILOT_GLOBAL_CONFIG", str(env_global))
    monkeypatch.setenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", str(env_isolated))

    resolved = resolve_config(
        sky_bin=explicit,
        global_config_path=explicit_global,
        isolated_config_dir=explicit_isolated,
    )
    assert resolved.sky_bin == explicit.resolve()
    assert resolved.global_config_path == explicit_global
    assert resolved.isolated_config_dir == explicit_isolated

    resolved = resolve_config()
    assert resolved.sky_bin == env.resolve()
    assert resolved.global_config_path == env_global
    assert resolved.isolated_config_dir == env_isolated

    monkeypatch.delenv("NPA_SKYPILOT_BIN")
    monkeypatch.delenv("SKYPILOT_GLOBAL_CONFIG")
    monkeypatch.delenv("NPA_SKYPILOT_ISOLATED_CONFIG_DIR")
    resolved = resolve_config()
    assert resolved.sky_bin == default.resolve()
    assert resolved.global_config_path == config_global
    assert resolved.isolated_config_dir == config_isolated


def test_resolve_config_rejects_unknown_config_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sky = _executable(tmp_path / "sky")
    config = tmp_path / "config.yaml"
    config.write_text(f"skypilot:\n  sky_bin: {sky}\n  typo_key: true\n", encoding="utf-8")
    monkeypatch.setattr(bin_module, "CONFIG_PATH", config)

    with pytest.raises(SkyPilotConfigError, match="typo_key.*Valid keys"):
        resolve_config()


def test_ensure_skypilot_version_accepts_required_version(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_skypilot_version_cache()
    sky = _executable(tmp_path / "sky")
    calls: list[list[str]] = []

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        return bin_module.subprocess.CompletedProcess(cmd, 0, stdout=f"SkyPilot {REQUIRED_SKYPILOT_VERSION}\n", stderr="")

    monkeypatch.setattr(bin_module.subprocess, "run", fake_run)

    assert ensure_skypilot_version(sky) == sky.resolve()
    assert ensure_skypilot_version(sky) == sky.resolve()
    assert calls == [[str(sky.resolve()), "--version"]]


def test_ensure_skypilot_version_rejects_mismatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    clear_skypilot_version_cache()
    sky = _executable(tmp_path / "sky")

    def fake_run(cmd, **kwargs):
        return bin_module.subprocess.CompletedProcess(cmd, 0, stdout="SkyPilot 0.12.1\n", stderr="")

    monkeypatch.setattr(bin_module.subprocess, "run", fake_run)

    with pytest.raises(SkyPilotVersionError, match="expected 0.12.2, got 0.12.1"):
        ensure_skypilot_version(sky)
