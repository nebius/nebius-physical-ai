from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from npa.workflows.sim2real.byo_isaac_eval import (
    per_env_from_distances,
    select_stratified_eval_envs,
)
from npa.workflows.sim2real.byo_isaac_trainer import parse_ppo_training_log
from npa.workflows.sim2real.checkpoint_selection import (
    assert_no_split_leakage,
    select_best_checkpoint,
)
from npa.workflows.sim2real.isaac_scenario_task import (
    PLACEMENT_APPROACH_STD_M,
    PLACEMENT_COMPLETION_REWARD_WEIGHT,
    PLACEMENT_DWELL_SCALE,
    PLACEMENT_DROP_PENALTY_WEIGHT,
    PLACEMENT_GOAL_CURRICULUM_LIFT_M,
    PLACEMENT_HOLD_STD_M,
    PLACEMENT_NEAR_STD_M,
    PLACEMENT_MINIMAL_LIFT_M,
    PLACEMENT_PROGRESS_REWARD_WEIGHT,
    PLACEMENT_PROGRESS_SCALE_M,
    STABLE_PLACEMENT_DISTANCE_M,
    STABLE_PLACEMENT_REWARD_WEIGHT,
    STABLE_PLACEMENT_SPEED_MPS,
    STABLE_PLACEMENT_STEPS,
    ScenarioContractError,
    module_source,
    placement_curriculum_signal,
    goal_curriculum_fraction,
    read_scenarios,
)
from npa.workflows.sim2real.task_contract import (
    LIFT_DATASET_ID,
    LIFT_TASK_ID,
    PUSHT_DATASET_ID,
    SEED_DATASET_SCHEMA,
    TaskContractError,
    build_task_contract,
    validate_seed_dataset_manifest,
    validate_task_dataset,
)
from npa.workflows.sim2real.temporal_credit import convert_evaluation
from npa.workflows.sim2real_envgen import (
    EnvGenConfig,
    SceneSpec,
    Sim2RealEnvGenError,
    _bounded_stratified_quotas,
    curate_envs,
    generate_raw_envs,
    load_raw_shards,
    write_split_manifest,
)


def test_lift_contract_rejects_pusht_and_validates_seed_manifest() -> None:
    with pytest.raises(TaskContractError, match="incompatible"):
        validate_task_dataset(
            task_id=LIFT_TASK_ID,
            dataset_id=PUSHT_DATASET_ID,
            dataset_uri="s3://bucket/pusht/",
            real_required=True,
        )
    contract = build_task_contract(
        task_id=LIFT_TASK_ID,
        dataset_id=LIFT_DATASET_ID,
        dataset_uri="s3://bucket/lift/",
    )
    manifest = {
        "schema": SEED_DATASET_SCHEMA,
        "task_id": LIFT_TASK_ID,
        "dataset_id": LIFT_DATASET_ID,
        "task_contract_digest": contract["task_contract_digest"],
        "source_backend": "isaac",
        "source_run_id": "source-run",
        "relabeled_from_another_task": False,
        "trajectory_count": 3,
        "action_count": 96,
        "camera_observation_count": 288,
        "sample_rollout_manifest_uri": "s3://bucket/lift/rollout/manifest.json",
    }
    proof = validate_seed_dataset_manifest(manifest, contract=contract)
    assert proof["source_backend"] == "isaac"
    manifest["dataset_id"] = PUSHT_DATASET_ID
    with pytest.raises(TaskContractError, match="mismatch"):
        validate_seed_dataset_manifest(manifest, contract=contract)


def test_matching_unimplemented_task_cannot_reuse_lift_contract() -> None:
    with pytest.raises(TaskContractError, match="no normalized simulator-facing"):
        build_task_contract(
            task_id="PushT-v0",
            dataset_id=PUSHT_DATASET_ID,
            dataset_uri="s3://bucket/pusht/",
        )


def test_curated_splits_are_disjoint_and_consume_stage3_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Storage:
        def upload_file(self, _source: str, destination: str) -> str:
            return destination

    monkeypatch.setattr(
        "npa.workflows.sim2real_envgen.StorageClient.from_environment",
        lambda: Storage(),
    )
    contract = build_task_contract(
        task_id=LIFT_TASK_ID,
        dataset_id=LIFT_DATASET_ID,
        dataset_uri="s3://bucket/lift/",
    )
    scene = SceneSpec(
        augmented_frames_uri="s3://bucket/augment/frames/",
        augmented_frame_uris=tuple(
            f"s3://bucket/augment/frames/frame-{index:05d}.png" for index in range(12)
        ),
        task_contract=contract,
        task_id=LIFT_TASK_ID,
        dataset_id=LIFT_DATASET_ID,
    )
    config = EnvGenConfig(
        run_id="efficacy",
        output_uri="s3://bucket/run",
        env_count=60,
        scene_spec=scene,
    )
    result = write_split_manifest(config, tmp_path)
    assert result["train_count"] == 48
    assert result["validation_count"] == 6
    assert result["gold_heldout_count"] == 6
    assert result["config_digest_leakage"] == {
        "train_validation": [],
        "train_gold_heldout": [],
        "validation_gold_heldout": [],
    }
    curation = json.loads((tmp_path / "curation-manifest.json").read_text())
    assert curation["augmentation_records_consumed"] == curation["accepted_count"]
    assert (
        curation["augmentation_consumer_contract"]["direct_state_policy_pixels"]
        is False
    )

    rows = generate_raw_envs(config)
    duplicate = dict(rows[0])
    invalid = dict(rows[1])
    invalid["validity"] = {**invalid["validity"], "reachable": False}
    accepted, rejected, reasons = curate_envs([rows[0], duplicate, invalid])
    assert len(accepted) == 1
    assert len(rejected) == 2
    assert reasons == {"duplicate_config_digest": 1, "unreachable_placement": 1}


def test_canonical_stratified_quotas_are_exact() -> None:
    sizes = {"easy": 3334, "medium": 3333, "hard": 3333}
    train = _bounded_stratified_quotas(
        sizes,
        target=8000,
        lower={name: 1 for name in sizes},
        upper={name: size - 2 for name, size in sizes.items()},
    )
    assert sum(train.values()) == 8000
    remaining = {name: sizes[name] - train[name] for name in sizes}
    validation = _bounded_stratified_quotas(
        remaining,
        target=1000,
        lower={name: 1 for name in sizes},
        upper={name: size - 1 for name, size in remaining.items()},
    )
    assert sum(validation.values()) == 1000
    assert sum(remaining[name] - validation[name] for name in sizes) == 1000


def test_stage5_loads_and_hashes_exact_stage4_shards(tmp_path: Path) -> None:
    contract = build_task_contract(
        task_id=LIFT_TASK_ID,
        dataset_id=LIFT_DATASET_ID,
        dataset_uri="s3://bucket/lift/",
    )
    base = dict(
        run_id="raw-handoff",
        output_uri="s3://bucket/run",
        env_count=12,
        train_fraction=0.5,
        seed=9,
        shard_count=4,
        scene_spec=SceneSpec(
            augmented_frame_uris=("s3://bucket/augment/frame.png",),
            task_contract=contract,
            task_id=LIFT_TASK_ID,
            dataset_id=LIFT_DATASET_ID,
        ),
    )
    config = EnvGenConfig(shard_index=0, **base)
    for shard_index in range(config.shard_count):
        shard_config = EnvGenConfig(shard_index=shard_index, **base)
        path = tmp_path / (
            f"raw-shard-{shard_index:02d}-of-{config.shard_count:02d}.jsonl"
        )
        path.write_text(
            "".join(
                json.dumps(row, sort_keys=True) + "\n"
                for row in generate_raw_envs(shard_config)
            ),
            encoding="utf-8",
        )

    class Storage:
        def download_directory(self, _source: str, _destination: str) -> str:
            return str(tmp_path)

    rows, proof = load_raw_shards(config, tmp_path, storage=Storage())
    assert len(rows) == 12
    assert proof["mode"] == "downloaded_stage_04_raw_shards"
    assert proof["shard_count"] == 4
    assert all(len(shard["sha256"]) == 64 for shard in proof["shards"])

    (tmp_path / "raw-shard-03-of-04.jsonl").unlink()
    with pytest.raises(Sim2RealEnvGenError, match="raw shard set mismatch"):
        load_raw_shards(config, tmp_path, storage=Storage())


def test_split_leakage_gate_fails_closed() -> None:
    assert_no_split_leakage({"train"}, {"validation"}, {"gold"})
    with pytest.raises(ValueError, match="leakage"):
        assert_no_split_leakage({"same"}, {"same"}, {"gold"})


def test_isaac_scenario_split_matches_authoritative_task_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    contract = build_task_contract(
        task_id=LIFT_TASK_ID,
        dataset_id=LIFT_DATASET_ID,
        dataset_uri="s3://bucket/lift/",
    )
    config = EnvGenConfig(
        run_id="scenario-contract",
        output_uri="s3://bucket/run",
        env_count=3,
        scene_spec=SceneSpec(
            augmented_frame_uris=("s3://bucket/augment/frame.png",),
            task_contract=contract,
            task_id=LIFT_TASK_ID,
            dataset_id=LIFT_DATASET_ID,
        ),
    )
    scenario_path = tmp_path / "envs.jsonl"
    scenario_path.write_text(
        "\n".join(json.dumps(row, sort_keys=True) for row in generate_raw_envs(config))
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("NPA_SIM2REAL_TASK_CONTRACT_DIGEST", "wrong")
    with pytest.raises(ScenarioContractError, match="authoritative"):
        read_scenarios(str(scenario_path))
    monkeypatch.setenv(
        "NPA_SIM2REAL_TASK_CONTRACT_DIGEST", contract["task_contract_digest"]
    )
    assert len(read_scenarios(str(scenario_path))) == 3


def test_scenario_task_ships_strict_stable_placement_curriculum() -> None:
    assert STABLE_PLACEMENT_DISTANCE_M == 0.05
    assert STABLE_PLACEMENT_SPEED_MPS == 0.03
    assert PLACEMENT_MINIMAL_LIFT_M == 0.04
    assert PLACEMENT_APPROACH_STD_M == 0.35
    assert PLACEMENT_NEAR_STD_M == 0.08
    assert PLACEMENT_HOLD_STD_M == 0.15
    assert PLACEMENT_DWELL_SCALE == 2.0
    assert PLACEMENT_GOAL_CURRICULUM_LIFT_M == 0.08
    assert PLACEMENT_PROGRESS_SCALE_M == 0.02
    assert PLACEMENT_PROGRESS_REWARD_WEIGHT == 128.0
    assert PLACEMENT_DROP_PENALTY_WEIGHT == -5000.0
    assert PLACEMENT_COMPLETION_REWARD_WEIGHT == 5000.0
    assert STABLE_PLACEMENT_REWARD_WEIGHT == 32.0
    assert STABLE_PLACEMENT_STEPS == 3
    source = module_source()
    assert "def stable_placement_curriculum" in source
    assert "lifted * (dense + strict)" in source
    assert "env_cfg.rewards.stable_placement_curriculum" in source
    assert "env_cfg.rewards.monotonic_placement_progress" in source
    assert "env_cfg.rewards.object_drop_penalty" in source
    assert "env_cfg.rewards.stable_placement_completion" in source
    assert "npa_best_placement_distance" in source
    assert "combine_frame_transforms" in source
    assert "mdp.is_terminated_term" in source
    assert "env_cfg.terminations.stable_placement_success" in source
    assert "NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM" in source
    assert "npa_goal_curriculum_true_assignments" in source
    assert 'NPA_SIM2REAL_ENABLE_SUCCESS_TERMINATION", "0"' in source
    assert "distance < float(success_distance_m)" in source
    assert "speed < float(stable_speed_mps)" in source


def test_stable_placement_curriculum_is_dense_across_live_canary_basin() -> None:
    # The 20-35 cm region observed in the failed live canary must retain a
    # meaningful approach gradient after lift; it may not collapse near zero.
    canary_basin_approach = 1.0 - math.tanh(0.25 / PLACEMENT_APPROACH_STD_M)
    assert canary_basin_approach > 0.2
    assert STABLE_PLACEMENT_REWARD_WEIGHT * canary_basin_approach > 10.0
    slow = placement_curriculum_signal(0.057, 0.01, 0.02, tanh=math.tanh)
    fly_through = placement_curriculum_signal(0.057, 0.20, 0.02, tanh=math.tanh)
    assert slow > 1.0
    assert slow > fly_through + 0.4


def test_stable_placement_curriculum_does_not_suppress_transport() -> None:
    transporting_far = placement_curriculum_signal(0.35, 0.20, 0.02, tanh=math.tanh)
    stopped_far = placement_curriculum_signal(0.45, 0.0, 0.02, tanh=math.tanh)
    assert STABLE_PLACEMENT_REWARD_WEIGHT * transporting_far > 4.0
    assert STABLE_PLACEMENT_REWARD_WEIGHT * stopped_far < 5.0


def test_stable_placement_curriculum_rejects_thrown_object_reward() -> None:
    held = placement_curriculum_signal(0.20, 0.20, 0.02, tanh=math.tanh)
    thrown = placement_curriculum_signal(0.20, 0.20, 0.50, tanh=math.tanh)
    assert held > thrown * 100


def test_goal_curriculum_reaches_exact_target_and_fails_closed() -> None:
    assert goal_curriculum_fraction(0, 7200) == 0.0
    assert goal_curriculum_fraction(3600, 7200) == 0.5
    assert goal_curriculum_fraction(7200, 7200) == 1.0
    assert goal_curriculum_fraction(9000, 7200) == 1.0
    with pytest.raises(ScenarioContractError, match="positive"):
        goal_curriculum_fraction(1, 0)


def test_temporal_credit_is_grounded_bounded_and_non_degenerate() -> None:
    evaluation = {
        "rollout_id": "rollout-1",
        "per_step": [
            {
                "step": index,
                "action": [0.1, -0.1],
                "error_tags": ["minor_alignment"],
                "confidence": 0.9,
                "model_disagreement": index == 1,
                "simulator_ground_truth": {
                    "object_goal_distance_m": distance,
                    "end_effector_object_distance_m": 0.20 - index * 0.05,
                    "contact": index >= 1,
                    "stable_grasp": index >= 2,
                    "object_lift_m": 0.05 * index,
                    "placement_stable": index == 3,
                    "scenario_config_digest": "cfg",
                },
            }
            for index, distance in enumerate((0.30, 0.24, 0.16, 0.03))
        ],
    }
    signal = convert_evaluation(evaluation)
    rewards = [row["reward"] for row in signal["per_step"]]
    assert all(-1.0 <= value <= 1.0 for value in rewards)
    assert len(set(rewards)) > 1
    assert signal["calibration"]["nonzero_advantage_count"] > 0
    assert signal["calibration"]["model_disagreement_steps"] == 1
    assert signal["calibration"]["vlm_accepted_steps"] == 2
    assert signal["calibration"]["vlm_rejected_or_downweighted_steps"] == 2
    assert signal["calibration"]["vlm_disagreement_downweighted_steps"] == 1
    assert signal["calibration"]["vlm_contradictory_steps"] == 1
    assert signal["per_step"][1]["confidence"] < signal["per_step"][0]["confidence"]


def test_temporal_credit_calibration_rejects_untrustworthy_vlm_rows() -> None:
    sources = (
        ("model_missing", 0.95, False, ["minor_alignment"]),
        ("model_malformed", 0.95, False, ["minor_alignment"]),
        ("model_per_step", 0.2, False, ["minor_alignment"]),
        ("model_per_step", 0.9, True, ["minor_alignment"]),
        ("summary_broadcast", 0.9, False, ["minor_alignment"]),
        ("model_per_step", 0.9, False, ["ok"]),
        ("model_per_step", 0.9, False, ["minor_alignment"]),
    )
    evaluation = {
        "rollout_id": "calibration-reasons",
        "per_step": [
            {
                "step": index,
                "action": [0.1],
                "critique_source": source,
                "confidence": confidence,
                "model_disagreement": disagreement,
                "error_tags": tags,
                "simulator_ground_truth": {
                    "object_goal_distance_m": 0.30 - index * 0.01,
                    "end_effector_object_distance_m": 0.20,
                    "contact": False,
                    "stable_grasp": False,
                    "object_lift_m": 0.0,
                    "placement_stable": False,
                    "scenario_config_digest": "cfg",
                },
            }
            for index, (source, confidence, disagreement, tags) in enumerate(sources)
        ],
    }

    signal = convert_evaluation(evaluation)
    calibration = signal["calibration"]
    assert calibration["step_count"] == 7
    assert calibration["vlm_accepted_steps"] == 1
    assert calibration["vlm_calibrated_steps"] == 1
    assert calibration["vlm_rejected_or_downweighted_steps"] == 6
    assert calibration["vlm_missing_or_malformed_steps"] == 2
    assert calibration["vlm_low_confidence_steps"] == 3
    assert calibration["vlm_disagreement_downweighted_steps"] == 1
    assert calibration["vlm_summary_broadcast_steps"] == 1
    assert calibration["vlm_contradictory_steps"] == 1
    assert signal["per_step"][0]["confidence"] == 0.0
    assert signal["per_step"][1]["confidence"] == 0.0


def test_checkpoint_selection_uses_validation_and_prefers_earlier_exact_tie() -> None:
    report = {
        "success_rate": 0.25,
        "per_env": [{"env_id": "validation-0"}],
        "success_summary": {"mean_object_goal_distance_m": 0.12},
        "decomposed_metrics": {"place": {"rate": 0.25}},
    }
    candidates = [
        {
            "evaluation_split": "validation",
            "outer_iteration": 1,
            "inner_iteration": iteration,
            "checkpoint_uri": f"s3://bucket/model-{iteration}.pt",
            "validation_report": report,
        }
        for iteration in (1, 2)
    ]
    selected = select_best_checkpoint(candidates)
    assert selected["inner_iteration"] == 1
    same_pass = [
        {
            "evaluation_split": "validation",
            "outer_iteration": 1,
            "inner_iteration": 1,
            "training_iteration": training_iteration,
            "checkpoint_uri": f"s3://bucket/model-{training_iteration}.pt",
            "validation_report": report,
        }
        for training_iteration in (100, 200)
    ]
    assert select_best_checkpoint(same_pass)["training_iteration"] == 100
    candidates[0]["evaluation_split"] = "gold_heldout"
    with pytest.raises(ValueError, match="only validation"):
        select_best_checkpoint(candidates)


def test_checkpoint_selection_accepts_component_native_strict_rate() -> None:
    report = {
        "strict_success": {"rate": 1 / 3},
        "per_env": [{"env_id": "validation-0"}],
        "success_summary": {"mean_object_goal_distance_m": 0.04},
        "decomposed_metrics": {"place": {"rate": 1 / 3}},
    }
    selected = select_best_checkpoint(
        [
            {
                "evaluation_split": "validation",
                "training_iteration": 100,
                "checkpoint_uri": "s3://bucket/model-100.pt",
                "validation_report": report,
            }
        ]
    )
    assert selected["rank_key"][0] == pytest.approx(1 / 3)


def test_eval_is_stratified_and_strict_success_requires_stability() -> None:
    rows = [
        {
            "env_id": f"{difficulty}-{index}",
            "difficulty": difficulty,
            "scenario_config_digest": f"{difficulty}-{index}",
        }
        for difficulty in ("easy", "medium", "hard")
        for index in range(30)
    ]
    selected = select_stratified_eval_envs(rows, count=64, split="gold_heldout")
    counts = {
        difficulty: sum(row["difficulty"] == difficulty for row in selected)
        for difficulty in ("easy", "medium", "hard")
    }
    assert max(counts.values()) - min(counts.values()) <= 1

    unstable = per_env_from_distances(
        [0.01],
        success_dist_m=0.05,
        runtime_metrics=[{"placement_stable": False}],
    )
    stable = per_env_from_distances(
        [0.01],
        success_dist_m=0.05,
        runtime_metrics=[{"placement_stable": True}],
    )
    assert unstable[0]["success"] is False
    assert stable[0]["success"] is True


def test_parse_ppo_telemetry_rejects_empty_and_surfaces_losses() -> None:
    log = """
Learning iteration 0/500
Mean action noise std: 1.00
Mean value_function loss: 0.0200
Mean surrogate loss: -0.0040
Mean entropy loss: 11.2000
Mean reward: 0.75
Episode_Reward/reaching_object: 0.1000
Episode_Reward/lifting_object: 0.2000
Episode_Reward/object_goal_tracking: 0.0500
Metrics/object_pose/position_error: 0.1800
Episode_Termination/stable_placement_success: 0.1250
Total timesteps: 24576
"""
    telemetry = parse_ppo_training_log(log)
    assert telemetry["configured_iterations"] == 500
    assert telemetry["final_iteration"]["value_loss"] == 0.02
    assert telemetry["final_iteration"]["total_timesteps"] == 24576
    assert telemetry["final_iteration"]["stable_placement_termination_rate"] == 0.125
    with pytest.raises(ValueError, match="no Learning iteration"):
        parse_ppo_training_log("no telemetry")
