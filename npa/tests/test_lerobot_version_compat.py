"""Unit tests for LeRobot multi-version compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path
import re

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
    assert (
        lerobot_pip_spec("0.6.0")
        == "lerobot[training,evaluation,pusht,libero,diffusion,smolvla]==0.6.0"
    )
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


def test_060_requests_the_extras_its_policies_gate_on() -> None:
    """0.6.0 moved diffusers/transformers behind extras and enforces them in
    each policy's ``__init__``, so a missing extra builds fine and only fails
    once ``make_policy`` runs. 0.5.1 ships both as core dependencies."""

    spec = lerobot_pip_spec("0.6.0")
    assert "diffusion" in spec, "--policy.type=diffusion needs lerobot[diffusion]"
    assert "smolvla" in spec, "--policy.type=smolvla needs lerobot[smolvla]"


def test_every_supported_version_resolves_to_immutable_image_evidence() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "npa/src/npa/deploy/lerobot_version_manifest.json").read_text(
            encoding="utf-8"
        )
    )

    for version in manifest["supported_versions"]:
        entry = manifest["versions"][version]
        assert entry["image_tag"] != version
        assert re.fullmatch(r"sha256:[0-9a-f]{64}", entry["image_digest"])


def test_060_resolves_to_the_validated_d6_image() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "npa/src/npa/deploy/lerobot_version_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    entry = manifest["versions"]["0.6.0"]

    assert entry["image_tag"] == "0.6.0-d6-extras-20260912"
    assert entry["image_digest"] == (
        "sha256:8d513f8558253fc484808a1e53a63a5"
        "da5a0c280ff973c4e590dd3e04b228643"
    )


def test_060_image_build_and_smoke_cover_real_diffusion_construction() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "npa/docker/workbench/lerobot/Dockerfile").read_text(
        encoding="utf-8"
    )
    smoke = (root / "npa/src/npa/smoke/test_lerobot_env.py").read_text(
        encoding="utf-8"
    )
    cloud_init = (root / "npa/src/npa/deploy/terraform/cloud_init.yaml.tpl").read_text(
        encoding="utf-8"
    )

    assert "lerobot[training,evaluation,pusht,libero,diffusion,smolvla]" in dockerfile
    assert "python -m npa.smoke.test_lerobot_env" in dockerfile
    assert "DiffusionPolicy(config)" in smoke
    assert "NPA_LEROBOT_SMOKE_REQUIRE_CUDA" in smoke
    assert 'policy.to("cuda")' in smoke
    assert (
        "lerobot[training,evaluation,pusht,libero,diffusion,smolvla]" in cloud_init
    )


def test_unsupported_version_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_LEROBOT_VERSION", raising=False)
    with pytest.raises(LeRobotVersionError, match="Unsupported"):
        resolve_lerobot_version("9.9.9")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    assert resolve_lerobot_version(None) == "0.6.0"
