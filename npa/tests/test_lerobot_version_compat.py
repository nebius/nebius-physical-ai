"""Unit tests for LeRobot multi-version compatibility helpers."""

from __future__ import annotations

import pytest

from npa.workbench.lerobot.version_compat import (
    LeRobotVersionError,
    eval_checkpoint_arg,
    lerobot_pip_spec,
    resolve_lerobot_version,
    supported_lerobot_versions,
    torch_install_pins,
    train_env_eval_arg,
    train_env_eval_flag,
)


def test_supported_versions_include_default_and_060() -> None:
    versions = supported_lerobot_versions()
    assert "0.5.1" in versions
    assert "0.6.0" in versions
    assert resolve_lerobot_version(None) == "0.5.1"


def test_pip_spec_and_train_flags_differ_by_version() -> None:
    assert lerobot_pip_spec("0.5.1") == "lerobot[pusht,libero]==0.5.1"
    assert lerobot_pip_spec("0.6.0") == "lerobot[training,evaluation,pusht,libero]==0.6.0"
    assert train_env_eval_flag("0.5.1") == "eval_freq"
    assert train_env_eval_flag("0.6.0") == "env_eval_freq"
    assert train_env_eval_arg(100, version="0.5.1") == "--eval_freq=100"
    assert train_env_eval_arg(100, version="0.6.0") == "--env_eval_freq=100"


def test_eval_checkpoint_and_torch_pins() -> None:
    assert eval_checkpoint_arg("/ckpt", version="0.5.1") == "--policy.path=/ckpt"
    assert (
        eval_checkpoint_arg("/ckpt", version="0.6.0", style="policy")
        == "--policy.pretrained_path=/ckpt"
    )
    assert "torch==2.12.1" in torch_install_pins("0.5.1")
    assert torch_install_pins("0.6.0") == []


def test_unsupported_version_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_LEROBOT_VERSION", raising=False)
    with pytest.raises(LeRobotVersionError, match="Unsupported"):
        resolve_lerobot_version("9.9.9")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    assert resolve_lerobot_version(None) == "0.6.0"
