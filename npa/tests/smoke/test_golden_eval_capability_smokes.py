"""Unit tests for new capability golden-eval smoke modules (no GPU required)."""

from __future__ import annotations

from importlib.util import find_spec
import subprocess
from unittest.mock import patch

import pytest


def test_retargeting_functional_passes() -> None:
    from npa.smoke import test_retargeting_functional

    assert test_retargeting_functional.main() == 0


def test_sim2real_envgen_raw_generation_passes() -> None:
    from npa.smoke.test_sim2real_envgen_functional import check_raw_env_generation

    result = check_raw_env_generation()
    assert result.ok, result.detail


def test_sim2real_envgen_runtime_readability_probe_is_non_root_and_drops_pythonpath(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.smoke import test_sim2real_envgen_functional as envgen_smoke

    observed: dict[str, object] = {}

    def fake_run(args, **kwargs):
        observed["args"] = args
        observed["environment"] = kwargs["env"]
        return subprocess.CompletedProcess(
            args,
            0,
            stdout="/opt/npa/compat/tetgen.py | /opt/npa/src/npa/workflows/sim2real_envgen.py\n",
            stderr="",
        )

    monkeypatch.setattr(envgen_smoke.os, "geteuid", lambda: 1000)
    monkeypatch.setenv("PYTHONPATH", "/operator/overlay")
    monkeypatch.setattr(envgen_smoke.subprocess, "run", fake_run)

    result = envgen_smoke.check_non_root_runtime_readability()

    assert result.ok, result.detail
    assert observed["args"][:2] == [envgen_smoke.sys.executable, "-c"]
    assert "PYTHONPATH" not in observed["environment"]
    assert "uid=1000" in result.detail


def test_sim2real_envgen_runtime_readability_probe_rejects_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from npa.smoke import test_sim2real_envgen_functional as envgen_smoke

    monkeypatch.setattr(envgen_smoke.os, "geteuid", lambda: 0)

    result = envgen_smoke.check_non_root_runtime_readability()

    assert not result.ok
    assert "unexpectedly ran as root" in result.detail


def test_cosmos3_reason_cache_wiring_passes() -> None:
    from npa.smoke.test_cosmos3_reason_functional import check_reason_cache_wiring

    result = check_reason_cache_wiring()
    assert result.ok, result.detail


def test_lancedb_bdd100k_dhash_runtime_passes() -> None:
    from npa.smoke.test_lancedb_functional import check_bdd100k_dhash_runtime

    result = check_bdd100k_dhash_runtime()

    assert result.ok, result.detail
    assert "Pillow=12." in result.detail
    assert "dhash=0" in result.detail


def test_lerobot_vlm_rl_signal_step_passes() -> None:
    from unittest.mock import MagicMock, patch

    with patch(
        "npa.smoke.test_lerobot_vlm_rl_functional.run_vlm_signal_training_step",
        return_value=MagicMock(
            checkpoint_path="/tmp/checkpoint.pt", policy_delta_l2=0.1
        ),
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
def test_sim2real_envgen_genesis_step_mocked(_cuda, mock_env) -> None:
    from npa.smoke.test_sim2real_envgen_functional import check_genesis_cuda_step

    instance = mock_env.return_value
    instance.act_dim = 4
    # Simulation is replaced at its boundary; keep tensor construction on the
    # CPU so this unit check does not require a physical CUDA device.
    instance.device = "cpu"
    result = check_genesis_cuda_step()
    assert result.ok, result.detail
    instance.reset.assert_called_once_with()
    instance.step.assert_called_once()
    action = instance.step.call_args.args[0]
    assert tuple(action.shape) == (1, 4)
    assert action.count_nonzero().item() == 0


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
        if ge.command.startswith('python -c "import npa.workbench'):
            weak.append(name)
    assert not weak, (
        f"containers still using import/help-only smokes: {sorted(set(weak))}"
    )


def test_run_all_dry_run_includes_all_runnable_tools() -> None:
    from npa.deploy.images import CONTAINER_IMAGE_NAMES
    from npa.smoke.batch import iter_containers, run_all
    from npa.smoke.manifest import load_manifest

    unrunnable = {
        name
        for name, spec in load_manifest().items()
        if spec.golden_eval.status in {"blocked-on-upstream", "needs-image-update"}
    }
    expected = set(CONTAINER_IMAGE_NAMES) - unrunnable
    names = iter_containers(tools_only=True, include_foundation=False)
    batch = run_all(names, serverless=False, execute=False)
    assert {r.name for r in batch.results} == expected
    assert batch.ok
