"""Unit tests for LeRobot multi-version compatibility helpers."""

from __future__ import annotations

import json
from pathlib import Path
import re

from packaging.requirements import Requirement
import pytest

from npa.deploy.images import resolve_lerobot_image_tag
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
    assert torch_install_pins("0.5.1") == [
        "torch==2.13.0",
        "torchvision==0.28.0",
        "torchcodec==0.16.0",
        "diffusers==0.38.0",
        "datasets==5.0.1",
        "wandb==0.30.0",
    ]
    assert torch_install_pins("0.6.0") == []


def test_default_secure_integration_manifest_matches_its_constraints() -> None:
    root = Path(__file__).resolve().parents[2]
    manifest = json.loads(
        (root / "npa/src/npa/deploy/lerobot_version_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    constraints = {
        requirement.name: str(requirement)
        for line in (
            root / "npa/docker/workbench/lerobot/default-runtime-requirements.txt"
        )
        .read_text(encoding="utf-8")
        .splitlines()
        if line and not line.startswith("#")
        for requirement in [Requirement(line)]
    }
    entry = manifest["versions"]["0.5.1"]
    manifest_pins = {
        "torch": entry["torch_pin"],
        "torchvision": entry["torchvision_pin"],
        "torchcodec": entry["torchcodec_pin"],
        "diffusers": entry["diffusers_pin"],
        "datasets": entry["datasets_pin"],
        "wandb": entry["wandb_pin"],
    }
    assert manifest_pins == constraints


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
        "sha256:8d513f8558253fc484808a1e53a63a5da5a0c280ff973c4e590dd3e04b228643"
    )
    assert resolve_lerobot_image_tag("0.6.0") == entry["image_tag"]
    assert resolve_lerobot_image_tag("0.6.0") != "0.6.0"


def test_060_image_build_and_smoke_cover_real_diffusion_construction() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "npa/docker/workbench/lerobot/Dockerfile").read_text(
        encoding="utf-8"
    )
    lock = (root / "npa/docker/workbench/lerobot/requirements-060.lock").read_text(
        encoding="utf-8"
    )
    smoke = (root / "npa/src/npa/smoke/test_lerobot_env.py").read_text(encoding="utf-8")
    cloud_init = (root / "npa/src/npa/deploy/terraform/cloud_init.yaml.tpl").read_text(
        encoding="utf-8"
    )

    assert (
        "--require-hashes --no-deps -r /opt/lerobot/requirements-060.lock" in dockerfile
    )
    for dependency in ("diffusers==", "transformers==", "accelerate==", "num2words=="):
        assert dependency in lock
    assert "python -m npa.smoke.test_lerobot_env" in dockerfile
    assert "DiffusionPolicy(config)" in smoke
    assert "NPA_LEROBOT_SMOKE_REQUIRE_CUDA" in smoke
    assert 'policy.to("cuda")' in smoke
    assert "lerobot[training,evaluation,pusht,libero,diffusion,smolvla]" in cloud_init


def test_lerobot_image_build_smoke_imports_npa_dataset_runtime() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "npa/docker/workbench/lerobot/Dockerfile").read_text(
        encoding="utf-8"
    )
    assert dockerfile.count('"pyarrow>=23.0.1"') == 2
    assert dockerfile.count('"av==15.1.0"') == 2
    assert dockerfile.count('"kubernetes==36.0.3"') == 2
    assert "NPA_BAKED_PYTHON=/opt/lerobot/venv/bin/python" in dockerfile
    assert "pip install --no-deps -e /opt/npa" in dockerfile
    assert 'test "$(command -v npa)" = /opt/lerobot/venv/bin/npa' in dockerfile
    assert "metadata.version('npa')" in dockerfile
    assert "assert av.__version__ == '15.1.0'" in dockerfile
    assert '"$NPA_BAKED_PYTHON" -c' in dockerfile
    assert "hasattr(av, 'option')" in dockerfile
    assert "from lerobot.datasets.lerobot_dataset import LeRobotDataset" in dockerfile
    assert "import pyarrow.parquet" in dockerfile
    assert "import npa.workflows.lerobot_dataset" in dockerfile
    assert "import npa.workbench.lerobot.policy_container" in dockerfile
    assert "from kubernetes import client, config" in dockerfile
    assert "import npa.workflows.sim2real.smolvla_service_lifecycle" in dockerfile


def test_lerobot_image_bakes_exact_npa_workflow_source() -> None:
    root = Path(__file__).resolve().parents[2]
    dockerfile = (root / "npa/docker/workbench/lerobot/Dockerfile").read_text(
        encoding="utf-8"
    )
    publication = (root / ".github/workflows/publish-public-images.yml").read_text(
        encoding="utf-8"
    )

    assert "ARG NPA_SOURCE_SHA=unknown" in dockerfile
    assert "NPA_IMAGE_SOURCE_SHA=${NPA_SOURCE_SHA}" in dockerfile
    assert "COPY --chown=ubuntu:ubuntu src/npa /opt/npa/src/npa" in dockerfile
    assert "COPY --chown=ubuntu:ubuntu workflows /opt/npa/workflows" in dockerfile
    assert '{"isaac-lab", "lerobot", "openpi", "rerun-viewer"}' in publication


def test_unsupported_version_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_LEROBOT_VERSION", raising=False)
    with pytest.raises(LeRobotVersionError, match="Unsupported"):
        resolve_lerobot_version("9.9.9")


def test_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    assert resolve_lerobot_version(None) == "0.6.0"


def test_smoke_identifies_local_integration_without_changing_upstream_selection(
    monkeypatch,
):
    from npa.smoke import _versions

    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.5.1")
    monkeypatch.setattr(_versions, "_has_secure_lerobot_integration", lambda: True)
    assert _versions.expected_lerobot_version(__file__) == "0.5.1+npa.secure2"
    assert resolve_lerobot_version(None) == "0.5.1"
    monkeypatch.setattr(_versions, "_has_secure_lerobot_integration", lambda: False)
    assert _versions.expected_lerobot_version(__file__) == "0.5.1"
    monkeypatch.delenv("NPA_LEROBOT_VERSION")
    monkeypatch.setattr(_versions, "_has_secure_lerobot_integration", lambda: True)
    assert _versions.expected_lerobot_version(__file__) == "0.5.1+npa.secure2"
    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    assert _versions.expected_lerobot_version(__file__) == "0.6.0"


def test_smoke_identifies_matching_060_secure_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.smoke import _versions

    monkeypatch.setenv("NPA_LEROBOT_VERSION", "0.6.0")
    monkeypatch.setattr(
        _versions,
        "_secure_lerobot_integration_version",
        lambda: "0.6.0+npa.secure2",
    )
    assert _versions.expected_lerobot_version(__file__) == "0.6.0+npa.secure2"


def test_env_smoke_uses_distribution_metadata_for_secure_integration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.smoke import test_lerobot_env

    monkeypatch.setattr(
        test_lerobot_env.metadata,
        "version",
        lambda name: "0.5.1+npa.secure2" if name == "lerobot" else "unexpected",
    )

    assert test_lerobot_env._package_version("lerobot") == "0.5.1+npa.secure2"
