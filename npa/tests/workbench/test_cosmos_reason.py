"""Unit tests for workbench-hosted Cosmos Reason2/Cosmos3 helpers."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

from npa.workbench.cosmos import reason as reason_module

from npa.workbench.cosmos.reason import (
    DEFAULT_REASON2_CACHE,
    DEFAULT_REASON2_MODEL,
    DEFAULT_REASON3_CACHE,
    DEFAULT_COSMOS3_MODEL,
    DEFAULT_REASON_EVENT_FRAMES,
    DEFAULT_REASON_MAX_NEW_TOKENS,
    apply_cosmos_reason_kubernetes_env,
    cosmos_reason_k8s_shell_preamble,
    cosmos_reason_runtime_env,
    default_reason_cache_dir,
    cosmos_reason_family,
    merge_reason_evaluations,
    merge_dual_reason_evaluations,
    prepare_cosmos_reason_cache,
    resolve_cosmos_reason_model_id,
    task_description_from_manifest,
    run_token_factory_rollout_vlm,
    select_hosted_event_frames,
    vlm_k8s_component,
)


def test_reason_import_in_minimal_component_image_does_not_load_httpx() -> None:
    code = """
import importlib.abc
import sys

class BlockHttpx(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == "httpx" or fullname.startswith("httpx."):
            raise RuntimeError("httpx must not be imported")
        return None

sys.meta_path.insert(0, BlockHttpx())
import npa.workbench.cosmos.reason
"""
    env = {**os.environ, "NPA_SKIP_EAGER_IMPORTS": "1"}
    subprocess.run([sys.executable, "-c", code], env=env, check=True)


def test_resolve_cosmos_reason_alias_defaults_to_reason2() -> None:
    assert resolve_cosmos_reason_model_id("npa-cosmos3-reason") == DEFAULT_REASON2_MODEL


def test_default_reason_cache_dir_uses_writable_tmp_hf_home(monkeypatch) -> None:
    monkeypatch.delenv("NPA_COSMOS_REASON2_CACHE", raising=False)
    monkeypatch.delenv("NPA_COSMOS_REASON3_CACHE", raising=False)
    assert default_reason_cache_dir(DEFAULT_REASON2_MODEL) == DEFAULT_REASON2_CACHE
    assert default_reason_cache_dir("nvidia/Cosmos-Reason2-2B") == DEFAULT_REASON3_CACHE
    assert DEFAULT_REASON2_CACHE.startswith("/tmp/hf_home/")


def test_cosmos_reason_runtime_env_defaults_to_writable_cache() -> None:
    runtime = cosmos_reason_runtime_env()
    assert runtime["HF_HOME"] == "/tmp/hf_home"
    assert runtime["NPA_COSMOS_REASON2_CACHE"] == DEFAULT_REASON2_CACHE


def test_cosmos_reason_runtime_env_prefers_the_durable_cache(monkeypatch) -> None:
    # The Reason checkpoints are gated, so no image may bake them and every Job
    # downloads them; /tmp means paying for that download once per Job.
    for name in (
        "HF_HOME",
        "NPA_COSMOS_REASON_CACHE",
        "NPA_COSMOS_REASON2_CACHE",
        "NPA_COSMOS_REASON3_CACHE",
    ):
        monkeypatch.delenv(name, raising=False)
    # The renderer exports the resolved root into the container; a claim name is
    # meaningless to code that cannot mount anything itself.
    monkeypatch.setenv("NPA_MODEL_CACHE_DIR", "/opt/npa-model-cache")

    runtime = cosmos_reason_runtime_env()

    assert runtime["HF_HOME"] == "/opt/npa-model-cache/huggingface"
    assert (
        runtime["NPA_COSMOS_REASON2_CACHE"]
        == "/opt/npa-model-cache/huggingface/cosmos-reason2"
    )
    assert (
        runtime["NPA_COSMOS_REASON3_CACHE"]
        == "/opt/npa-model-cache/huggingface/cosmos-reason2-2b"
    )


def test_reason_defaults_cover_every_canonical_decision_event() -> None:
    assert DEFAULT_REASON_EVENT_FRAMES == 32
    assert DEFAULT_REASON_MAX_NEW_TOKENS >= 8192


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
    assert vlm_k8s_component("vlm_eval_cosmos3")
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


def test_model_family_distinguishes_real_cosmos3_edge_from_reason2() -> None:
    assert cosmos_reason_family(DEFAULT_REASON2_MODEL) == "reason2"
    assert cosmos_reason_family(DEFAULT_COSMOS3_MODEL) == "cosmos3"
    assert cosmos_reason_family("nvidia/Cosmos-Reason2-2B") == "reason2"


def test_hosted_frame_selection_is_bounded_and_rollout_wide() -> None:
    frames = [Path(f"camera-{index:03d}.png") for index in range(32)]
    selected = select_hosted_event_frames(frames)
    assert len(selected) == 8
    assert selected[0] == frames[0]
    assert selected[-1] == frames[-1]
    assert selected == select_hosted_event_frames(frames)


def test_token_factory_rollout_evaluator_returns_event_local_contract(tmp_path) -> None:
    frames = []
    for index in range(10):
        frame = tmp_path / f"camera-{index:03d}.png"
        frame.write_bytes(b"public-synthetic-frame")
        frames.append(frame)

    class Client:
        last_request_metrics = {"latency_seconds": 1.25, "retries": 1}

        def chat_completion(self, **kwargs):
            assert kwargs["model"] == DEFAULT_COSMOS3_MODEL
            images = kwargs["messages"][0]["content"][1:]
            assert len(images) == 8
            assert all(item["image_url"]["url"].startswith("data:image/png;base64,") for item in images)
            return {
                "id": "request-public-1",
                "choices": [{"message": {"content": json.dumps({
                    "success": True,
                    "score": 0.9,
                    "summary": "stable cube grasp",
                    "per_step": [
                        {"step": index, "critique_text": f"event {index} stable", "error_tags": ["ok"], "confidence": 0.8}
                        for index in range(10)
                    ],
                })}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            }

    result = run_token_factory_rollout_vlm(
        model_id=DEFAULT_COSMOS3_MODEL,
        image_paths=frames,
        actions=[{"step": index, "action": [0.0]} for index in range(10)],
        task_description="strict cube grasp",
        rollout_id="rollout-0000",
        threshold=0.5,
        client=Client(),
    )
    assert len(result["per_step"]) == 10
    assert result["schema"] == "npa.sim2real.vlm_eval.v3"
    assert result["backend"] == "token_factory"
    assert result["request"] == {
        "request_id": "request-public-1",
        "input_tokens": 100,
        "output_tokens": 50,
        "total_tokens": 150,
        "latency_seconds": 1.25,
        "retries": 1,
        "cost_usd": None,
        "cost_source": "unavailable",
    }


def test_cosmos3_success_cannot_bypass_the_fixed_threshold() -> None:
    payload = reason_module._parse_cosmos_reason_output(
        json.dumps(
            {
                "success": True,
                "score": 0.49,
                "summary": "model claimed success below the workflow threshold",
                "per_step": [
                    {
                        "step": 0,
                        "critique_text": "cube remains unstable",
                        "error_tags": ["unstable"],
                    }
                ],
            }
        ),
        actions=[{"step": 0, "action": [0.0]}],
        rollout_id="rollout-threshold",
        threshold=0.5,
        family="cosmos3",
    )
    assert payload["schema"] == "npa.sim2real.vlm_eval.v3"
    assert payload["success"] is False


def test_task_description_from_manifest_prefers_task_description() -> None:
    manifest = {"task_description": "Pick up the cube.", "task": "ignored"}
    assert task_description_from_manifest(manifest) == "Pick up the cube."


def test_merge_dual_reason_evaluations_averages_scores_and_requires_both_success() -> (
    None
):
    reason2 = {
        "rollout_id": "rollout-0000",
        "model": DEFAULT_REASON2_MODEL,
        "success": True,
        "score": 0.9,
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
    cosmos3 = {
        "rollout_id": "rollout-0000",
        "model": DEFAULT_COSMOS3_MODEL,
        "success": False,
        "score": 0.8,
        "per_step": [
            {
                "step": 0,
                "critique_text": "late grasp",
                "error_tags": ["late_grasp"],
                "action": [0.0, 0.0, 0.0],
                "camera_observation": "camera-000.ppm",
            }
        ],
        "summary": "cosmos3 miss",
    }

    merged = merge_reason_evaluations(reason2, cosmos3, threshold=0.75)

    assert merged["two_evaluator"] is True
    assert merged["schema"] == "npa.sim2real.vlm_eval.v2"
    assert merged["component_source"] == "cosmos_reason2_cosmos3_vlm"
    assert merged["score"] == 0.85
    assert merged["success"] is False
    assert merged["per_step"][0]["error_tags"] == ["ok", "late_grasp"]
    assert "reason2_critique" in merged["per_step"][0]
    assert "cosmos3_critique" in merged["per_step"][0]
    assert "cosmos3_tags" in merged["per_step"][0]
    assert merged["cosmos3"]["model"] == DEFAULT_COSMOS3_MODEL

    archived_alias = merge_dual_reason_evaluations(reason2, cosmos3, threshold=0.75)
    assert archived_alias["score"] == 0.85
    assert archived_alias["success"] is False


def test_summary_only_output_is_rejected_without_temporal_broadcast() -> None:
    actions = [
        {
            "step": step,
            "action": [float(step)],
            "simulator_ground_truth": {"secret": step},
        }
        for step in range(4)
    ]
    payload = reason_module._parse_cosmos_reason_output(
        json.dumps(
            {
                "success": False,
                "score": 0.2,
                "summary": "The rollout missed the target.",
            }
        ),
        actions=actions,
        rollout_id="rollout-0001",
        threshold=0.5,
        family="reason2",
    )

    assert payload["summary"] == "The rollout missed the target."
    assert len(payload["per_step"]) == 4
    assert {row["critique_source"] for row in payload["per_step"]} == {"model_missing"}
    assert {row["confidence"] for row in payload["per_step"]} == {0.0}
    assert len({row["critique_text"] for row in payload["per_step"]}) == 4
    assert all(
        "The rollout missed the target" not in row["critique_text"]
        for row in payload["per_step"]
    )


def test_single_array_wrapped_evaluation_preserves_event_local_critiques() -> None:
    payload = reason_module._parse_cosmos_reason_output(
        """```json
[{"success": false, "score": 0.27, "summary": "no stable grasp",
  "per_step": [
    {"step": 0, "critique_text": "hovering above cube", "error_tags": ["late_grasp"], "confidence": 0.91},
    {"step": 1, "critique_text": "fingers remain open", "error_tags": ["unstable"], "confidence": 0.87}
  ]}]
```""",
        actions=[
            {"step": 0, "action": [0.1]},
            {"step": 1, "action": [0.2]},
        ],
        rollout_id="rollout-array-wrapper",
        threshold=0.5,
        family="cosmos3",
    )

    assert payload["summary"] == "no stable grasp"
    assert [row["step"] for row in payload["per_step"]] == [0, 1]
    assert {row["critique_source"] for row in payload["per_step"]} == {"model_per_step"}
    assert [row["confidence"] for row in payload["per_step"]] == [0.91, 0.87]
    assert len({row["critique_text"] for row in payload["per_step"]}) == 2
    assert (
        reason_module._json_object_from_text(
            '[{"success": false, "score": 0.2}, {"step": 31}]'
        )
        is None
    )


def test_token_truncated_output_recovers_complete_rows_and_explicit_false() -> None:
    payload = reason_module._parse_cosmos_reason_output(
        """```json
[{"success": false, "score": 0.17, "summary": "missed cube",
  "per_step": [
    {"step": 0, "critique_text": "hovered above cube", "error_tags": ["late_grasp"], "confidence": 0.9},
    {"step": 1, "critique_text": "fingers remained open", "error_tags": ["unstable"], "confidence": 0.8},
    {"step": 2, "critique_text": "unterminated""",
        actions=[
            {"step": 0, "action": [0.1]},
            {"step": 1, "action": [0.2]},
            {"step": 2, "action": [0.3]},
        ],
        rollout_id="rollout-truncated",
        threshold=0.5,
        family="reason2",
    )

    assert payload["success"] is False
    assert payload["score"] == 0.17
    assert [row["step"] for row in payload["per_step"]] == [0, 1, 2]
    assert [row["critique_source"] for row in payload["per_step"]] == [
        "model_per_step",
        "model_per_step",
        "model_missing",
    ]
    assert payload["per_step"][2]["confidence"] == 0.0


def test_unstructured_explicit_false_is_not_misread_as_success() -> None:
    payload = reason_module._parse_unstructured_vlm_output(
        'result: {"success": false, "score": 0.1, unfinished', threshold=0.5
    )
    assert payload["success"] is False


def test_malformed_step_is_rejected_instead_of_copying_summary() -> None:
    payload = reason_module._parse_cosmos_reason_output(
        json.dumps(
            {
                "success": False,
                "score": 0.1,
                "summary": "rollout summary",
                "per_step": [{"step": 0, "error_tags": ["missed_target"]}],
            }
        ),
        actions=[{"step": 0, "action": [0.0]}],
        rollout_id="rollout-0002",
        threshold=0.5,
        family="cosmos3",
    )

    step = payload["per_step"][0]
    assert step["critique_source"] == "model_malformed"
    assert step["confidence"] == 0.0
    assert step["error_tags"] == ["ok"]
    assert "rollout summary" not in step["critique_text"]


def test_reason_prompt_requires_every_step_and_hides_simulator_truth() -> None:
    prompt = reason_module._cosmos_reason_prompt(
        family="reason2",
        task_description="Lift the cube.",
        actions=[
            {
                "step": step,
                "sim_step": step * 10,
                "action": [0.1, -0.2],
                "simulator_ground_truth": {"do_not_leak": "ANSWER"},
            }
            for step in range(32)
        ],
        frame_names=[f"camera-{step:03d}.png" for step in range(32)],
    )

    assert "Required per_step indices: [0, 1, 2" in prompt
    assert "exactly one" in prompt
    assert "never copy or broadcast" in prompt
    assert "never use a top-level array" in prompt
    assert "simulator_ground_truth" not in prompt
    assert "do_not_leak" not in prompt
    assert "ANSWER" not in prompt


def test_dual_reason_rejects_missing_local_model_label() -> None:
    base = {
        "rollout_id": "rollout-0003",
        "success": False,
        "score": 0.2,
        "summary": "summary only",
    }
    reason2 = {
        **base,
        "model": DEFAULT_REASON2_MODEL,
        "per_step": [
            {
                "step": 0,
                "critique_text": "reason2 returned no local label for step 0",
                "error_tags": ["ok"],
                "critique_source": "model_missing",
                "confidence": 0.0,
            }
        ],
    }
    cosmos3 = {
        **base,
        "model": DEFAULT_COSMOS3_MODEL,
        "per_step": [
            {
                "step": 0,
                "critique_text": "specific late grasp",
                "error_tags": ["late_grasp"],
                "critique_source": "model_per_step",
                "confidence": 0.8,
            }
        ],
    }

    merged = merge_reason_evaluations(reason2, cosmos3, threshold=0.5)

    assert merged["per_step"][0]["critique_source"] == "model_missing"
    assert merged["per_step"][0]["confidence"] == 0.0
