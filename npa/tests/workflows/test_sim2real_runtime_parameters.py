"""Canonical Sim2Real parameter wiring and provenance contracts."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import yaml

from npa.workflows.sim2real.camera_views import camera_metadata, camera_view_names
from npa.workflows.sim2real.capture import capture_settings, ppo_settings
from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real.constants import (
    DEFAULT_ENV_COUNT,
    DEFAULT_HELDOUT_ENVS,
    DEFAULT_INNER_ITERATIONS,
    DEFAULT_OUTER_ITERATIONS,
    DEFAULT_ROLLOUT_COUNT,
    DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE,
    DEFAULT_STEPS_PER_ROLLOUT,
    DEFAULT_THRESHOLD,
)
from npa.workflows.sim2real.k8s_submit import _validate_real_runtime_env
from npa.workflows.sim2real.k8s_components import _component_job_manifest
from npa.workbench.cosmos.reason import (
    DEFAULT_REASON_EVENT_FRAMES,
    DEFAULT_REASON_MAX_NEW_TOKENS,
)
from npa.workflows.sim2real.materialize import default_runbook_path, materialize_k8s_job
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.workflows.sim2real import engine


RUNBOOK = default_runbook_path()


def _canonical_envs() -> dict[str, str]:
    payload = yaml.safe_load(Path(RUNBOOK).read_text(encoding="utf-8"))
    return {str(key): str(value) for key, value in payload["envs"].items()}


def test_canonical_yaml_defaults_match_engine_and_real_components() -> None:
    envs = _canonical_envs()
    assert float(envs["SUCCESS_THRESHOLD"]) == DEFAULT_THRESHOLD
    assert int(envs["INNER_ITERATIONS"]) == DEFAULT_INNER_ITERATIONS
    assert int(envs["OUTER_ITERATIONS"]) == DEFAULT_OUTER_ITERATIONS
    assert int(envs["NPA_ENV_COUNT"]) == DEFAULT_ENV_COUNT
    assert int(envs["ROLLOUT_COUNT"]) == DEFAULT_ROLLOUT_COUNT
    assert int(envs["STEPS_PER_ROLLOUT"]) == DEFAULT_STEPS_PER_ROLLOUT
    assert int(envs["HELDOUT_ENV_COUNT"]) == DEFAULT_HELDOUT_ENVS
    assert envs["NPA_SIM2REAL_EARLY_EXIT"] == "0"
    assert envs["NPA_SIM2REAL_HELDOUT_EVAL_LIMIT"] == "0"
    assert envs["NPA_SIM2REAL_K8S_JOB_TIMEOUT_S"] == "0"
    assert int(envs["NPA_COSMOS_REASON_MAX_FRAMES"]) == DEFAULT_REASON_EVENT_FRAMES
    assert (
        int(envs["NPA_COSMOS_REASON_MAX_NEW_TOKENS"]) == DEFAULT_REASON_MAX_NEW_TOKENS
    )
    assert float(envs["LEARNING_RATE"]) == DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE
    config = Sim2RealLoopConfig(run_id="parameter-defaults")
    assert config.k8s_job_timeout_s == 0
    assert config.learning_rate == DEFAULT_SIGNAL_ADAPTER_LEARNING_RATE
    assert envs["BYO_TRAINER_COMMAND"].endswith("byo_isaac_trainer")
    assert envs["BYO_POLICY_COMMAND"].endswith("byo_isaac_policy_rollout")
    assert envs["BYO_EVAL_COMMAND"].endswith("byo_isaac_eval")
    assert envs["NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS"] == "1"
    _validate_real_runtime_env(
        {
            **envs,
            "OMNI_KIT_ACCEPT_EULA": "YES",
            "ISAACSIM_ACCEPT_EULA": "YES",
        }
    )


def test_capture_and_ppo_defaults_are_inspection_and_training_grade() -> None:
    envs = _canonical_envs()
    capture = capture_settings(envs)
    ppo = ppo_settings(envs)
    assert capture == {
        "width": 640,
        "height": 480,
        "rollout_stride": 1,
        "heldout_stride": 20,
        "png_compress_level": 3,
        "fps": 10.0,
    }
    assert ppo == {
        "num_envs": 1024,
        "iterations": 500,
        "steps_per_env": 24,
        "total_environment_steps": 12_288_000,
    }


def test_camera_metadata_has_three_named_views_pose_and_intrinsics() -> None:
    metadata = cast(
        list[dict[str, Any]],
        camera_metadata("front,side,top", width=1280, height=720),
    )
    assert [item["name"] for item in metadata] == ["primary", "side", "overhead"]
    for item in metadata:
        assert item["pose_frame"] == "isaac_world"
        assert len(item["position"]) == 3
        assert len(item["rotation"]) == 4
        assert item["width"] == 1280 and item["height"] == 720
        assert item["intrinsics_px"]["fx"] > 0
        assert item["intrinsics_px"]["fy"] > 0


def test_materialized_job_carries_capture_ppo_and_visualization_knobs() -> None:
    envs = _canonical_envs()
    image = "cr.example.com/npa/npa-lerobot-vlm-rl:test"
    manifest = materialize_k8s_job(
        RUNBOOK,
        image=image,
        run_id="parameter-contract",
    ).manifest
    container_env = {
        item["name"]: item["value"]
        for item in manifest["spec"]["template"]["spec"]["containers"][0]["env"]
    }
    for name in (
        "NPA_SIM2REAL_CAPTURE_WIDTH",
        "NPA_SIM2REAL_CAPTURE_HEIGHT",
        "NPA_SIM2REAL_CAMERA_VIEWS",
        "NPA_BYO_ISAAC_NUM_ENVS",
        "NPA_BYO_ISAAC_ITERATIONS",
        "NPA_BYO_ISAAC_STEPS_PER_ENV",
        "NPA_BYO_ISAAC_VALIDATION_INTERVAL",
        "NPA_SIM2REAL_MCAP",
        "NPA_SIM2REAL_REQUIRE_VISUALIZATION",
        "NPA_COSMOS_REASON_MAX_FRAMES",
        "NPA_COSMOS_REASON_MAX_NEW_TOKENS",
        "NPA_SIM2REAL_K8S_JOB_TIMEOUT_S",
        "NPA_BYO_ISAAC_JOB_TIMEOUT_S",
    ):
        assert container_env[name] == envs[name]


@pytest.mark.parametrize(
    "candidates",
    [
        "RTX PRO 6000,L40S",
        "RTX PRO 6000;L40S",
        ("RTX PRO 6000", "L40S"),
        ["RTX PRO 6000", "L40S"],
    ],
)
def test_gpu_candidate_config_round_trip_never_stringifies_sequences(
    candidates: object,
) -> None:
    config = build_config_from_env(
        run_id="gpu-candidate-round-trip",
        k8s_gpu_candidates=candidates,
    )
    assert config.k8s_gpu_candidates == ("RTX PRO 6000", "L40S")
    assert all("'" not in candidate for candidate in config.k8s_gpu_candidates)

    from npa.workflows import sim2real_loop as compatibility_loop

    compatibility_config = compatibility_loop.build_config_from_env(
        run_id="gpu-candidate-compatibility-round-trip",
        k8s_gpu_candidates=candidates,
    )
    assert compatibility_config.k8s_gpu_candidates == ("RTX PRO 6000", "L40S")


@pytest.mark.parametrize(
    ("override", "message"),
    [
        ({"NPA_SIM2REAL_CAPTURE_WIDTH": "319"}, "CAPTURE_WIDTH"),
        ({"NPA_BYO_ISAAC_STEPS_PER_ENV": "0"}, "STEPS_PER_ENV"),
        ({"NPA_BYO_ISAAC_VALIDATION_INTERVAL": "0"}, "VALIDATION_INTERVAL"),
        (
            {"NPA_SIM2REAL_ROLLOUT_HORIZON_STEPS": "31"},
            "ROLLOUT_HORIZON_STEPS",
        ),
        ({"NPA_COSMOS_REASON_MAX_FRAMES": "31"}, "MAX_FRAMES"),
        ({"NPA_COSMOS_REASON_MAX_NEW_TOKENS": "2047"}, "MAX_NEW_TOKENS"),
        ({"NPA_SIM2REAL_CAMERA_VIEWS": "rear"}, "camera view"),
        ({"SUCCESS_THRESHOLD": "1.1"}, "SUCCESS_THRESHOLD"),
        ({"NPA_BYO_ISAAC_SUCCESS_DIST_M": "0"}, "SUCCESS_DIST_M"),
        ({"NPA_SIM2REAL_K8S_JOB_TIMEOUT_S": "-1"}, "JOB_TIMEOUT"),
        ({"BYO_TRAINER_COMMAND": ""}, "BYO_TRAINER_COMMAND"),
        ({"NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS": "0"}, "REQUIRE_REAL_COMPONENTS"),
        ({"OMNI_KIT_ACCEPT_EULA": ""}, "OMNI_KIT_ACCEPT_EULA"),
    ],
)
def test_invalid_or_inert_real_tier_parameters_fail_before_submit(
    override: dict[str, str], message: str
) -> None:
    envs = {
        **_canonical_envs(),
        "OMNI_KIT_ACCEPT_EULA": "YES",
        "ISAACSIM_ACCEPT_EULA": "YES",
        **override,
    }
    with pytest.raises(ValueError, match=message):
        _validate_real_runtime_env(envs)


def test_camera_selection_preserves_primary_compatibility() -> None:
    assert camera_view_names("side") == ("primary", "side")


def test_component_command_default_has_no_deadline(monkeypatch, tmp_path: Path) -> None:
    observed: dict[str, object] = {}

    def fake_run(*args, **kwargs):
        observed.update(kwargs)
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    monkeypatch.setattr(engine.subprocess, "run", fake_run)
    engine._run_component_command("true", cwd=tmp_path, env={}, component="trainer")
    assert observed["timeout"] is None


def test_sibling_component_default_has_no_deadline(monkeypatch, tmp_path: Path) -> None:
    config = Sim2RealLoopConfig(run_id="sim2real-unlimited")
    manifest = _component_job_manifest(
        "cr.example.com/npa/transfer:test",
        component="cosmos2_transfer",
        env={},
        config=config,
        namespace="default",
        job_name="s2r-transfer",
        timeout_s=config.k8s_job_timeout_s,
    )
    assert "activeDeadlineSeconds" not in manifest["spec"]

    observed: dict[str, object] = {}

    def fake_component(*args, **kwargs):
        observed.update(kwargs)
        return {"returncode": 0}

    monkeypatch.setattr(engine, "_run_kubernetes_image_component", fake_component)
    engine._run_image_component(
        "cr.example.com/npa/transfer:test",
        component="cosmos2_transfer",
        env={},
        output_json=tmp_path / "out.json",
        output_uri="s3://bucket/out.json",
        config=config,
    )
    assert observed["timeout_s"] == 0
