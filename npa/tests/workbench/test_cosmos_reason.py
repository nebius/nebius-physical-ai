"""Unit tests for workbench-hosted Cosmos Reason2/3 helpers."""

from __future__ import annotations

import os

from npa.workbench.cosmos.reason import (
    DEFAULT_REASON2_CACHE,
    DEFAULT_REASON2_MODEL,
    DEFAULT_REASON3_CACHE,
    DEFAULT_REASON3_MODEL,
    apply_cosmos_reason_kubernetes_env,
    cosmos_reason_k8s_shell_preamble,
    cosmos_reason_runtime_env,
    default_reason_cache_dir,
    merge_dual_reason_evaluations,
    prepare_cosmos_reason_cache,
    resolve_cosmos_reason_model_id,
    task_description_from_manifest,
    vlm_k8s_component,
)


def test_resolve_cosmos_reason_alias_defaults_to_reason2() -> None:
    assert (
        resolve_cosmos_reason_model_id("npa-cosmos3-reason")
        == DEFAULT_REASON2_MODEL
    )


def test_default_reason_cache_dir_uses_writable_tmp_hf_home(monkeypatch) -> None:
    monkeypatch.delenv("NPA_COSMOS_REASON2_CACHE", raising=False)
    monkeypatch.delenv("NPA_COSMOS_REASON3_CACHE", raising=False)
    assert default_reason_cache_dir(DEFAULT_REASON2_MODEL) == DEFAULT_REASON2_CACHE
    assert default_reason_cache_dir(DEFAULT_REASON3_MODEL) == DEFAULT_REASON3_CACHE
    assert DEFAULT_REASON2_CACHE.startswith("/tmp/hf_home/")


def test_cosmos_reason_runtime_env_defaults_to_writable_cache() -> None:
    runtime = cosmos_reason_runtime_env()
    assert runtime["HF_HOME"] == "/tmp/hf_home"
    assert runtime["NPA_COSMOS_REASON2_CACHE"] == DEFAULT_REASON2_CACHE


def test_apply_cosmos_reason_kubernetes_env_preserves_existing_values() -> None:
    safe = apply_cosmos_reason_kubernetes_env(
        {"HF_HOME": "/custom/hf", "NPA_SIM2REAL_RUN_ID": "r1"}
    )
    assert safe["HF_HOME"] == "/custom/hf"
    assert safe["NPA_COSMOS_REASON3_CACHE"] == DEFAULT_REASON3_CACHE


def test_prepare_cosmos_reason_cache_creates_directory(tmp_path, monkeypatch) -> None:
    cache_root = tmp_path / "hf"
    monkeypatch.setenv("NPA_COSMOS_REASON2_CACHE", str(cache_root / "reason2"))
    monkeypatch.delenv("HF_HOME", raising=False)
    cache_dir = prepare_cosmos_reason_cache(model_id=DEFAULT_REASON2_MODEL)
    try:
        assert cache_dir == str(cache_root / "reason2")
        assert (cache_root / "reason2").is_dir()
        assert os.environ["HF_HOME"] == str(cache_root)
    finally:
        os.environ.pop("HF_HOME", None)


def test_vlm_k8s_shell_preamble_creates_hf_home() -> None:
    preamble = cosmos_reason_k8s_shell_preamble()
    assert 'export HF_HOME="${HF_HOME:-/tmp/hf_home}"' in preamble
    assert "mkdir -p" in preamble
    assert vlm_k8s_component("vlm_eval_reason2")
    assert not vlm_k8s_component("policy_actions")


def test_engine_vlm_job_script_prepares_hf_cache(monkeypatch) -> None:
    monkeypatch.delenv("HF_HOME", raising=False)
    from npa.workflows.sim2real.engine import (
        _component_job_script,
        _kubernetes_component_env,
    )
    from npa.workflows.sim2real.models import Sim2RealLoopConfig

    script = _component_job_script("vlm_eval_reason2")
    assert 'export HF_HOME="${HF_HOME:-/tmp/hf_home}"' in script
    safe = _kubernetes_component_env({}, Sim2RealLoopConfig(run_id="r"))
    assert safe["HF_HOME"] == "/tmp/hf_home"
    assert safe["NPA_COSMOS_REASON3_CACHE"] == DEFAULT_REASON3_CACHE


def test_task_description_from_manifest_prefers_task_description() -> None:
    manifest = {"task_description": "Pick up the cube.", "task": "ignored"}
    assert task_description_from_manifest(manifest) == "Pick up the cube."


def test_merge_dual_reason_evaluations_averages_scores_and_requires_both_success() -> None:
    reason2 = {
        "rollout_id": "rollout-0000",
        "model": DEFAULT_REASON2_MODEL,
        "success": True,
        "score": 0.8,
        "per_step": [
            {
                "step": 0,
                "critique_text": "aligned",
                "error_tags": ["ok"],
                "action": [0.0, 0.0, 0.0],
                "camera_observation": "camera-000.ppm",
            }
        ],
        "summary": "reason2 ok",
    }
    reason3 = {
        "rollout_id": "rollout-0000",
        "model": DEFAULT_REASON3_MODEL,
        "success": False,
        "score": 0.4,
        "per_step": [
            {
                "step": 0,
                "critique_text": "late grasp",
                "error_tags": ["late_grasp"],
                "action": [0.0, 0.0, 0.0],
                "camera_observation": "camera-000.ppm",
            }
        ],
        "summary": "reason3 miss",
    }

    merged = merge_dual_reason_evaluations(reason2, reason3, threshold=0.75)

    assert merged["dual_reason"] is True
    assert merged["component_source"] == "cosmos_dual_reason_vlm"
    assert merged["score"] == 0.6
    assert merged["success"] is False
    assert merged["per_step"][0]["error_tags"] == ["ok", "late_grasp"]
    assert "reason2_critique" in merged["per_step"][0]
    assert "reason3_critique" in merged["per_step"][0]
