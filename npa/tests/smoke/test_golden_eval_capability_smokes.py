"""Unit tests for new capability golden-eval smoke modules (no GPU required)."""

from __future__ import annotations

from importlib.util import find_spec
from unittest.mock import patch

import pytest


def test_retargeting_functional_passes() -> None:
    from npa.smoke import test_retargeting_functional

    assert test_retargeting_functional.main() == 0


def test_sim2real_envgen_raw_generation_passes() -> None:
    from npa.smoke.test_sim2real_envgen_functional import check_raw_env_generation

    result = check_raw_env_generation()
    assert result.ok, result.detail


def test_cosmos3_reason_cache_wiring_passes() -> None:
    from npa.smoke.test_cosmos3_reason_functional import check_reason_cache_wiring

    result = check_reason_cache_wiring()
    assert result.ok, result.detail


def test_lerobot_vlm_rl_signal_step_passes() -> None:
    from unittest.mock import MagicMock, patch

    with patch(
        "npa.smoke.test_lerobot_vlm_rl_functional.run_vlm_signal_training_step",
        return_value=MagicMock(checkpoint_path="/tmp/checkpoint.pt", policy_delta_l2=0.1),
    ):
        with patch("pathlib.Path.exists", return_value=True):
            from npa.smoke.test_lerobot_vlm_rl_functional import check_vlm_signal_step

            result = check_vlm_signal_step()
    assert result.ok, result.detail


@pytest.mark.skipif(
    find_spec("torch") is None,
    reason="torch not installed",
)
@patch("npa.genesis.env_pick_place.FrankaPickPlaceEnv")
@patch("torch.cuda.is_available", return_value=True)
def test_sim2real_envgen_genesis_step_mocked(mock_env, _cuda) -> None:
    from npa.smoke.test_sim2real_envgen_functional import check_genesis_cuda_step

    instance = mock_env.return_value
    instance.act_dim = 4
    instance.device = "cuda:0"
    result = check_genesis_cuda_step()
    assert result.ok, result.detail


def test_manifest_covers_all_tools_with_container_smokes_or_server_smokes() -> None:
    from npa.deploy.images import CONTAINER_IMAGE_NAMES
    from npa.smoke.manifest import load_manifest

    specs = load_manifest()
    missing = set(CONTAINER_IMAGE_NAMES) - set(specs)
    assert not missing
    weak = []
    for name in CONTAINER_IMAGE_NAMES:
        ge = specs[name].golden_eval
        if ge.command.endswith("--help"):
            weak.append(name)
        if ge.command.startswith("python -c \"import npa.workbench"):
            weak.append(name)
    assert not weak, f"containers still using import/help-only smokes: {sorted(set(weak))}"


def test_run_all_dry_run_includes_all_tools() -> None:
    from npa.deploy.images import CONTAINER_IMAGE_NAMES
    from npa.smoke.batch import iter_containers, run_all
    from npa.smoke.manifest import load_manifest

    blocked = {
        name
        for name, spec in load_manifest().items()
        if spec.golden_eval.status == "blocked-on-upstream"
    }
    expected = set(CONTAINER_IMAGE_NAMES) - blocked
    names = iter_containers(tools_only=True, include_foundation=False)
    batch = run_all(names, serverless=False, execute=False)
    assert {r.name for r in batch.results} == expected
    assert batch.ok
