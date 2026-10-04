"""Contract tests for the Cosmos3 Edge FastWAM-K2 closed-loop workflow."""

from __future__ import annotations

import json
from pathlib import Path

from npa.workbench.cosmos.fastwam_k2 import (
    COMPARISON_SCHEMA,
    CONDITIONING_LATENT_FRAMES,
    KEEP_GENERATED_VISION_FRAMES,
    PREPARED_SCHEMA,
    ROBOLAB_REPOSITORY,
    TOKENS_PER_VISION_LATENT_FRAME,
    VARIANT_SCHEMA,
    EvaluationRequest,
    FastWamK2Error,
    _isaac_runtime_env,
    _prepared_payload,
    _rrd_recording_id,
    _run_robolab,
    _select_cuda_topology,
    apply_k2_runtime_overlay,
    compare_variants,
    server_argv,
    summarize_episode_metrics,
)
from npa.workbench.cosmos.policy_artifacts import publish_bundle, write_local_json


def _framework_source(root: Path) -> Path:
    framework = root / "framework"
    server = framework / "cosmos_framework/scripts/action_policy_server_robolab.py"
    model = framework / "cosmos_framework/model/generator/omni_mot_model.py"
    server.parent.mkdir(parents=True)
    model.parent.mkdir(parents=True)
    server.write_text(
        "    format_prompt_as_json: bool | None = None\n"
        '    """Serve prompts as structured JSON (matching training ``format_prompt_as_json``)."""\n'
        "        self.model = pipe.model\n        self.model.eval()\n"
    )
    model.write_text(
        "def encode(condition_indexes, condition_mask):\n"
        "            full_shape = list(prefix_latent.shape)\n"
        "            full_shape[temporal_dim] = num_latent_frames\n"
    )
    return framework


def _variant_bundle(path: Path, variant: str, prepared_sha: str) -> Path:
    artifacts = path / "artifacts"
    artifacts.mkdir(parents=True)
    result = {
        "schema": VARIANT_SCHEMA,
        "status": "succeeded",
        "variant": variant,
        "prepared_sha256": prepared_sha,
        "screening_only": True,
        "metrics": {
            "tasks": {
                "RubiksCubesInBinTask": {
                    "success_rate": 0.25 if variant == "full-wam" else 0.5,
                    "policy_inference_avg_ms": 12.0 if variant == "full-wam" else 9.0,
                }
            },
            "overall": {"success_rate": 0.25 if variant == "full-wam" else 0.5},
        },
    }
    write_local_json(artifacts / "variant-result.json", result)
    publish_bundle(artifacts, str(path / "published"), result, f"{variant}.json")
    return path / "published" / f"{variant}.json"


def test_overlay_retains_k2_generated_frames_and_native_condition_mask(
    tmp_path: Path,
) -> None:
    framework = _framework_source(tmp_path)
    record = apply_k2_runtime_overlay(framework, tmp_path / "artifacts")

    assert record["keep_generated_vision_frames"] == KEEP_GENERATED_VISION_FRAMES
    assert record["conditioning_latent_frames"] == CONDITIONING_LATENT_FRAMES
    assert record["expected_vision_tokens"] == TOKENS_PER_VISION_LATENT_FRAME * 3
    assert record["expected_mse_target_tokens"] == TOKENS_PER_VISION_LATENT_FRAME * 2
    model = (
        framework / "cosmos_framework/model/generator/omni_mot_model.py"
    ).read_text()
    assert (
        "kept_latent_frames = min(num_latent_frames, needed_latent_frames + keep_generated)"
        in model
    )
    assert "condition_indexes" in model
    assert "condition_mask" in model


def test_robolab_checkout_uses_the_upstream_published_repository() -> None:
    """Pin the repository namespace named by both upstream policy cards."""
    assert ROBOLAB_REPOSITORY == "https://github.com/NVlabs/RoboLab.git"


def test_server_argv_never_uses_k0_drop_generated_vision(tmp_path: Path) -> None:
    command = server_argv(tmp_path / "python", tmp_path / "checkpoint", "fastwam-k2")
    baseline = server_argv(tmp_path / "python", tmp_path / "checkpoint", "full-wam")

    assert command[-2:] == ["--keep-generated-vision-frames", "2"]
    assert "--drop-generated-vision" not in command
    assert "--drop-generated-vision" not in baseline
    assert "--keep-generated-vision-frames" not in baseline


def test_robolab_uses_only_the_shared_isaac_acceptance_surface() -> None:
    from npa.orchestration.npa_workflow.skypilot_render import isaac_eula_envs

    runtime_env = _isaac_runtime_env({"ACCEPT_EULA": "Y"})

    assert runtime_env["ACCEPT_EULA"] == "Y"
    assert runtime_env["OMNI_KIT_ACCEPT_EULA"] == "Y"
    assert isaac_eula_envs("workbench.cosmos3.fastwam_k2_full_wam") == {
        "ACCEPT_EULA": "Y"
    }
    assert isaac_eula_envs("workbench.cosmos3.fastwam_k2_eval", accepted=False) == {
        "ACCEPT_EULA": ""
    }
    try:
        _isaac_runtime_env({"ACCEPT_EULA": ""})
    except FastWamK2Error as exc:
        assert "explicitly opted out" in str(exc)
    else:  # pragma: no cover - an opt-out must never reach a runtime fetch
        raise AssertionError("explicit Isaac opt-out was accepted")


def test_robolab_client_keeps_the_shared_isaac_acceptance_surface(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, str] = {}
    output = tmp_path / "robolab/output/npa-fastwam-k2-full-wam"
    output.mkdir(parents=True)

    def fake_run(_argv, *, cwd, env, log) -> None:
        captured.update(env)

    monkeypatch.setattr("npa.workbench.cosmos.fastwam_k2._run", fake_run)
    result = _run_robolab(
        tmp_path / "python",
        tmp_path / "robolab",
        EvaluationRequest(),
        "full-wam",
        {"ACCEPT_EULA": "Y"},
        tmp_path / "robolab.log",
        cuda_visible_devices="0",
    )

    assert result == output
    assert captured["ACCEPT_EULA"] == "Y"
    assert captured["OMNI_KIT_ACCEPT_EULA"] == "Y"
    assert captured["CUDA_VISIBLE_DEVICES"] == "0"


def test_cuda_topology_preserves_two_gpu_split_and_allows_one_gpu_colocation() -> None:
    assert _select_cuda_topology(2) == {
        "visible_cuda_device_count": 2,
        "policy_server_cuda_visible_devices": "0",
        "robolab_cuda_visible_devices": "1",
        "shared_cuda_device": False,
    }
    assert _select_cuda_topology(1) == {
        "visible_cuda_device_count": 1,
        "policy_server_cuda_visible_devices": "0",
        "robolab_cuda_visible_devices": "0",
        "shared_cuda_device": True,
    }
    try:
        _select_cuda_topology(0)
    except FastWamK2Error as exc:
        assert "at least one CUDA device" in str(exc)
    else:  # pragma: no cover - zero CUDA devices must not start native evaluation
        raise AssertionError("zero CUDA devices were accepted")


def test_prepared_payload_hashes_matched_task_sources(tmp_path: Path) -> None:
    robolab = tmp_path / "robolab"
    task_dir = robolab / "robolab/tasks/benchmark"
    task_dir.mkdir(parents=True)
    (task_dir / "rubiks.py").write_text("class RubiksCubesInBinTask: pass\n")
    (task_dir / "yellow.py").write_text("class StackYellowOnRedTask: pass\n")

    payload = _prepared_payload(EvaluationRequest(), robolab)

    assert payload["schema"] == PREPARED_SCHEMA
    assert payload["benchmark_claim"] is False
    assert set(payload["task_sources"]) == {
        "RubiksCubesInBinTask",
        "StackYellowOnRedTask",
    }
    assert len(payload["prepared_sha256"]) == 64


def test_comparison_requires_matched_closed_loop_manifests(tmp_path: Path) -> None:
    full_wam = _variant_bundle(tmp_path / "full", "full-wam", "matched")
    fastwam_k2 = _variant_bundle(tmp_path / "k2", "fastwam-k2", "matched")

    result = compare_variants(
        full_wam_path=str(full_wam),
        k2_path=str(fastwam_k2),
        output_path=str(tmp_path / "comparison"),
    )
    persisted = json.loads((tmp_path / "comparison" / "comparison.json").read_text())

    assert result["schema"] == COMPARISON_SCHEMA
    assert (
        persisted["decision_basis"]
        == "closed-loop task success; open-loop errors are intentionally excluded"
    )
    assert persisted["benchmark_claim"] is False
    assert (
        persisted["paired_task_metrics"]["RubiksCubesInBinTask"]["success_rate_delta"]
        == 0.25
    )


def test_episode_metrics_require_native_latency_for_every_closed_loop_row() -> None:
    rows = [
        {
            "task_name": "RubiksCubesInBinTask",
            "success": True,
            "timing": {"policy_inference_avg_ms": 12.5},
        },
        {
            "task_name": "RubiksCubesInBinTask",
            "success": False,
            "timing": {"policy_inference_avg_ms": 7.5},
        },
    ]

    metrics = summarize_episode_metrics(rows, ["RubiksCubesInBinTask"])

    assert metrics["tasks"]["RubiksCubesInBinTask"]["success_rate"] == 0.5
    assert metrics["overall"]["policy_inference_avg_ms"] == 10.0
    rows[1]["timing"] = {}
    try:
        summarize_episode_metrics(rows, ["RubiksCubesInBinTask"])
    except FastWamK2Error as exc:
        assert "native policy latency" in str(exc)
    else:  # pragma: no cover - a non-measurement must not produce a report
        raise AssertionError("missing native latency was accepted")


def test_rrd_uses_the_renderer_run_identity_when_present(monkeypatch) -> None:
    comparison = {"prepared_sha256": "prepared-evidence"}

    assert _rrd_recording_id(comparison) == "prepared-evidence"
    monkeypatch.setenv("NPA_WORKFLOW_RUN_ID", "workflow-run-identity")
    assert _rrd_recording_id(comparison) == "workflow-run-identity"
