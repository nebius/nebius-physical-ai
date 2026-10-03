"""Contract tests for the native five-stage LeRobot FastWAM workflow."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path

import pytest

from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.workflows import fastwam_policy as fastwam
from npa.workflows.lerobot_dataset import LeRobotDatasetSummary


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows" / "testing" / "fastwam-policy-qualification.yaml"


def _runtime_args() -> argparse.Namespace:
    return argparse.Namespace(
        fastwam_base_revision="a" * 40,
        wan_revision="b" * 40,
        wan_diffusers_revision="c" * 40,
        umt5_revision="d" * 40,
        device="cuda",
        train_steps=300_000,
        batch_size=8,
        environment="libero",
        environment_task="libero_10",
        episode_length=200,
        observation_height=224,
        observation_width=224,
        eval_batch_size=1,
        policy_dtype="float32",
        n_action_steps=10,
        episodes=50,
        seed=42,
        compile_action_infer=True,
    )


def test_workflow_has_five_connected_substantive_native_stages() -> None:
    spec = load_spec(WORKFLOW)
    plan = build_plan(spec, run_id="fastwam-contract")

    assert [step.state for step in plan.steps] == [
        "prepare",
        "train",
        "rollout",
        "evaluate",
        "report",
    ]
    assert [step.tool_ref for step in plan.steps] == [
        "workbench.lerobot.fastwam_prepare",
        "workbench.lerobot.fastwam_train",
        "workbench.lerobot.fastwam_rollout",
        "workbench.lerobot.fastwam_evaluate",
        "workbench.lerobot.fastwam_report",
    ]
    assert all(step.argv or step.shell for step in plan.steps)
    assert spec.states["train"].inputs[0].uri == "{{config.prepared_uri}}recipe.json"
    assert spec.states["rollout"].inputs[1].uri == "{{config.training_uri}}checkpoint/"
    assert spec.states["evaluate"].inputs[-1].uri == "{{config.rollouts_uri}}rollout.json"
    assert spec.states["report"].outputs[1].uri == "{{config.report_uri}}fastwam.rrd"


def test_prepare_seals_disjoint_split_and_upstream_identity(tmp_path, monkeypatch: pytest.MonkeyPatch) -> None:
    source = tmp_path / "dataset"
    (source / "meta").mkdir(parents=True)
    (source / "meta" / "info.json").write_text('{"features": {}}\n')
    summary = LeRobotDatasetSummary(
        source_uri="unused",
        local_path=str(source),
        repo_id="operator/robot-data",
        revision="f" * 40,
        license="CC-BY-4.0",
        total_episodes=4,
        total_frames=40,
        fps=30,
        episode_indices=[2, 3, 7, 11],
        feature_keys=["action", "observation.images.top", "observation.state"],
        camera_keys=["observation.images.top"],
        state_keys=["observation.state"],
        action_keys=["action"],
        loaded_with_lerobot_dataset=True,
    )
    monkeypatch.setattr(fastwam, "_materialize_dataset", lambda *_args: source)
    monkeypatch.setattr(fastwam, "summarize_lerobot_dataset", lambda *_args, **_kwargs: summary)
    monkeypatch.setattr(
        fastwam,
        "_validate_native_fastwam_dataset_contract",
        lambda *_args: {"has_task_or_precomputed_context": True},
    )
    args = argparse.Namespace(
        input_path="s3://input/",
        dataset_repo_id="operator/robot-data",
        dataset_revision="f" * 40,
        dataset_license="CC-BY-4.0",
        train_fraction=0.75,
        seed=42,
    )

    fastwam._prepare(args, tmp_path / "work", tmp_path / "prepared")

    recipe = json.loads((tmp_path / "prepared" / "recipe.json").read_text())
    assert set(recipe["train_episode_indices"]).isdisjoint(recipe["heldout_episode_indices"])
    assert sorted(recipe["train_episode_indices"] + recipe["heldout_episode_indices"]) == [2, 3, 7, 11]
    assert recipe["policy"] == "fastwam"
    assert recipe["native_fastwam_feature_contract"]["has_task_or_precomputed_context"] is True
    assert recipe["physical_robot_tested"] is False
    assert "Cosmos3" in recipe["upstream"]["distinction"]


def test_native_commands_preserve_split_and_direct_action_contract(tmp_path) -> None:
    recipe = {
        "dataset": {"repo_id": "operator/robot-data", "revision": "f" * 40},
        "train_episode_indices": [1, 4, 8],
    }
    models = {key: tmp_path / key for key in ("fastwam_base", "wan", "wan_diffusers", "umt5")}
    train = fastwam._fastwam_train_command(_runtime_args(), recipe, tmp_path / "data", tmp_path / "out", models)
    rollout = fastwam._fastwam_eval_command(_runtime_args(), tmp_path / "checkpoint", tmp_path / "eval", models)

    assert train[0] == "lerobot-train"
    assert "--policy.type=fastwam" in train
    assert "--policy.device=cuda" in train
    assert "--dataset.episodes=[1, 4, 8]" in train
    assert "--env_eval_freq=0" in train
    assert rollout[0] == "lerobot-eval"
    assert "--policy.compile_action_infer=true" in rollout
    assert "--env.task=libero_10" in rollout
    assert "--env.observation_height=224" in rollout
    assert "--policy.dtype=float32" in rollout
    assert "--policy.n_action_steps=10" in rollout
    assert all("generate_video" not in item for item in rollout)


def test_runtime_fetch_requires_distinct_immutable_component_revisions(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[str, str]] = []

    def fake_download(*, repo_id: str, revision: str, cache_dir: str | None) -> str:
        calls.append((repo_id, revision))
        return f"/cache/{repo_id.replace('/', '__')}/{revision}"

    class Hub:
        snapshot_download = staticmethod(fake_download)

    monkeypatch.setitem(__import__("sys").modules, "huggingface_hub", Hub)
    resolved = fastwam._fetch_runtime_models(_runtime_args())

    assert set(resolved) == {"fastwam_base", "wan", "wan_diffusers", "umt5"}
    assert calls[1] == (fastwam.WAN_REPOSITORY, "b" * 40)
    assert calls[2] == (fastwam.WAN_DIFFUSERS_REPOSITORY, "c" * 40)


def test_latency_stage_loads_the_exact_checkpoint_not_a_fresh_base_model() -> None:
    source = inspect.getsource(fastwam._measure_direct_action_latency)
    assert "FastWAMPolicy.from_pretrained(" in source
    assert "checkpoint," in source


def test_recipe_refuses_missing_or_overlapping_episode_contract(tmp_path) -> None:
    (tmp_path / "recipe.json").write_text(
        json.dumps({"schema": "npa.fastwam.recipe.v1", "policy": "fastwam", "train_episode_indices": [1], "heldout_episode_indices": []})
    )
    with pytest.raises(fastwam.FastWAMPolicyError, match="episode-disjoint"):
        fastwam._read_recipe(tmp_path)
