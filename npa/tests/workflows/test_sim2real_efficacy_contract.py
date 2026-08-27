from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

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
    PLACEMENT_ARM_SETTLING_SPEED_RADPS,
    PLACEMENT_ARM_STILLNESS_REWARD_WEIGHT,
    PLACEMENT_BASIN_SETTLING_REWARD_WEIGHT,
    PLACEMENT_BASIN_WIDTH_M,
    PLACEMENT_COMPLETION_REWARD_WEIGHT,
    PLACEMENT_DWELL_REWARD_EXPONENT,
    PLACEMENT_DWELL_SCALE,
    PLACEMENT_DROP_PENALTY_WEIGHT,
    PLACEMENT_GOAL_CURRICULUM_LIFT_M,
    PLACEMENT_HOLD_MAX_DISTANCE_M,
    PLACEMENT_HOLD_REWARD_FLOOR,
    PLACEMENT_HOLD_STD_M,
    PLACEMENT_NEAR_STD_M,
    PLACEMENT_MINIMAL_LIFT_M,
    PLACEMENT_PROGRESS_REWARD_WEIGHT,
    PLACEMENT_PROGRESS_SCALE_M,
    PLACEMENT_SETTLING_SPEED_MPS,
    PLACEMENT_STRICT_DWELL_REWARD_WEIGHT,
    STABLE_PLACEMENT_DISTANCE_M,
    STABLE_PLACEMENT_REWARD_WEIGHT,
    STABLE_PLACEMENT_SPEED_MPS,
    STABLE_PLACEMENT_STEPS,
    ScenarioContractError,
    _scheduled_drop_penalty_type,
    drop_penalty_schedule_fraction,
    module_source,
    near_goal_arm_stillness_signal,
    placement_curriculum_signal,
    placement_progress_signal,
    stable_placement_dwell_signal,
    stable_placement_retention_signal,
    strict_basin_settling_signal,
    goal_curriculum_fraction,
    read_scenarios,
    scenario_assignment_indices,
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


def test_scenario_assignment_cursor_covers_tail_before_wrapping() -> None:
    first = scenario_assignment_indices(count=16, row_count=18)
    second = scenario_assignment_indices(count=4, row_count=18, cursor=len(first))

    assert first == list(range(16))
    assert second == [16, 17, 0, 1]
    assert set(first + second) == set(range(18))


def test_scenario_assignment_cursor_applies_offset_and_validates_bounds() -> None:
    assert scenario_assignment_indices(
        count=5, row_count=3, cursor=2, offset=1
    ) == [0, 1, 2, 0, 1]
    with pytest.raises(ValueError, match="non-negative"):
        scenario_assignment_indices(count=-1, row_count=3)
    with pytest.raises(ValueError, match="at least one"):
        scenario_assignment_indices(count=1, row_count=0)


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


def test_reduced_split_clamps_train_count_to_keep_validation_and_gold_sealed(
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
    config = EnvGenConfig(
        run_id="reduced-live-proof",
        output_uri="s3://bucket/run",
        env_count=24,
        train_fraction=0.8,
        scene_spec=SceneSpec(
            augmented_frame_uris=("s3://bucket/augment/frame.png",),
            task_contract=contract,
            task_id=LIFT_TASK_ID,
            dataset_id=LIFT_DATASET_ID,
        ),
    )

    result = write_split_manifest(config, tmp_path)

    assert result["requested_train_count"] == 19
    assert result["train_count"] == 18
    assert result["validation_count"] == 3
    assert result["gold_heldout_count"] == 3
    assert result["effective_train_fraction"] == 0.75
    assert result["split_count_adjusted_for_stratification"] is True
    assert all(
        all(
            result["coverage"][split]["difficulty"][difficulty] >= 1
            for difficulty in ("easy", "medium", "hard")
        )
        for split in ("train", "validation", "gold_heldout")
    )


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
    base: dict[str, Any] = dict(
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
    assert PLACEMENT_BASIN_WIDTH_M == 0.05
    assert PLACEMENT_NEAR_STD_M == 0.08
    assert PLACEMENT_HOLD_STD_M == 0.15
    assert PLACEMENT_HOLD_REWARD_FLOOR == 0.20
    assert PLACEMENT_HOLD_MAX_DISTANCE_M == 0.30
    assert PLACEMENT_DWELL_SCALE == 2.0
    assert PLACEMENT_SETTLING_SPEED_MPS == 0.20
    assert PLACEMENT_GOAL_CURRICULUM_LIFT_M == 0.08
    assert PLACEMENT_PROGRESS_SCALE_M == 0.02
    assert PLACEMENT_PROGRESS_REWARD_WEIGHT == 512.0
    assert PLACEMENT_DROP_PENALTY_WEIGHT == -2000.0
    assert PLACEMENT_COMPLETION_REWARD_WEIGHT == 5000.0
    assert STABLE_PLACEMENT_REWARD_WEIGHT == 32.0
    assert PLACEMENT_BASIN_SETTLING_REWARD_WEIGHT == 256.0
    assert PLACEMENT_ARM_SETTLING_SPEED_RADPS == 1.0
    assert PLACEMENT_ARM_STILLNESS_REWARD_WEIGHT == 512.0
    assert PLACEMENT_STRICT_DWELL_REWARD_WEIGHT == 4096.0
    assert PLACEMENT_DWELL_REWARD_EXPONENT == 2.0
    assert STABLE_PLACEMENT_STEPS == 3
    source = module_source()
    assert "def stable_placement_curriculum" in source
    assert "lifted * (dense + strict)" in source
    assert "env_cfg.rewards.stable_placement_curriculum" in source
    assert "env_cfg.rewards.potential_placement_progress" in source
    assert "env_cfg.rewards.strict_basin_settling" in source
    assert "env_cfg.rewards.near_goal_arm_stillness" in source
    assert 'env.action_manager.get_term("arm_action")' in source
    assert "env_cfg.rewards.stable_placement_dwell" in source
    assert "env_cfg.rewards.stable_placement_dwell_break" not in source
    assert "env_cfg.rewards.stable_placement_departure" not in source
    assert "def stable_placement_departure" not in source
    assert "PLACEMENT_POST_SUCCESS_DEPARTURE_WEIGHT" not in source
    assert "env_cfg.rewards.object_drop_penalty" in source
    assert "env_cfg.rewards.stable_placement_completion" in source
    assert "func=stable_placement_completion" in source
    assert source.index("env_cfg.rewards.stable_placement_completion") < source.index(
        "if success_termination_enabled"
    )
    assert "npa_previous_placement_distance" in source
    assert "combine_frame_transforms" in source
    assert "scheduled_drop_penalty" in source
    assert "func=_scheduled_drop_penalty_type()" in source
    assert "return mdp.is_terminated_term(env" not in source
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
    assert slow > 0.9
    assert slow > fly_through + 0.4


def test_stable_placement_curriculum_rewards_braking_in_strict_basin() -> None:
    stopped = placement_curriculum_signal(0.027, 0.0, 0.02, tanh=math.tanh)
    fly_through = placement_curriculum_signal(0.027, 0.20, 0.02, tanh=math.tanh)
    assert stopped > 1.5
    assert fly_through > 0.0
    assert stopped > fly_through + 0.8


def test_near_goal_arm_stillness_rewards_directly_controllable_braking() -> None:
    stopped = near_goal_arm_stillness_signal(0.04, 0.0, 0.02, tanh=math.tanh)
    moving = near_goal_arm_stillness_signal(0.04, 0.50, 0.02, tanh=math.tanh)
    fast = near_goal_arm_stillness_signal(0.04, 3.0, 0.02, tanh=math.tanh)
    far_stopped = near_goal_arm_stillness_signal(0.30, 0.0, 0.02, tanh=math.tanh)
    # The live Train32 scale must not saturate the signal to zero at ordinary
    # arm speed; PPO needs a dense distinction between moving and braking.
    assert moving > 0.20
    assert moving > fast + 0.20
    assert stopped > moving + 0.20
    assert stopped > fast + 0.50
    assert stopped > far_stopped + 0.4
    assert moving >= 0.0
    with pytest.raises(ValueError, match="positive"):
        near_goal_arm_stillness_signal(
            0.04,
            0.0,
            0.02,
            tanh=math.tanh,
            arm_settling_speed_radps=0.0,
        )


def test_signed_placement_progress_penalizes_departure_and_is_reset_safe() -> None:
    assert placement_progress_signal(math.inf, 0.30) == 0.0
    assert placement_progress_signal(0.30, 0.27) == 1.0
    assert placement_progress_signal(0.027, 0.047) == pytest.approx(-1.0)
    # Positive drive tapers before the exact basin and reaches zero inside it,
    # while later departure remains fully negative.
    assert placement_progress_signal(0.047, 0.027) == 0.0
    assert placement_progress_signal(0.10, 0.08) == pytest.approx(0.6)
    assert placement_progress_signal(0.12, 0.10) == pytest.approx(1.0)
    assert placement_progress_signal(0.030, 0.032) == pytest.approx(-0.1)
    with pytest.raises(ValueError, match="positive"):
        placement_progress_signal(0.03, 0.02, progress_scale_m=0.0)
    with pytest.raises(ValueError, match="positive"):
        placement_progress_signal(0.03, 0.02, braking_width_m=0.0)


def test_strict_dwell_reward_requires_three_unchanged_stable_steps() -> None:
    steps, reward = stable_placement_dwell_signal(True, 0)
    assert (steps, reward) == (1, pytest.approx(1 / 9))
    steps, reward = stable_placement_dwell_signal(True, steps)
    assert (steps, reward) == (2, pytest.approx(4 / 9))
    steps, reward = stable_placement_dwell_signal(True, steps)
    assert (steps, reward) == (3, 1.0)
    assert stable_placement_dwell_signal(True, steps) == (3, 1.0)
    assert stable_placement_dwell_signal(False, steps) == (0, 0.0)
    with pytest.raises(ValueError, match="positive"):
        stable_placement_dwell_signal(True, 0, required_steps=0)
    with pytest.raises(ValueError, match="positive"):
        stable_placement_dwell_signal(True, 0, reward_exponent=0)


def test_stable_placement_retention_is_positive_only_after_completion() -> None:
    state = stable_placement_retention_signal(True, 0, False)
    assert state == (1, pytest.approx(1 / 9), False, 0.0)
    state = stable_placement_retention_signal(True, state[0], state[2])
    assert state == (2, pytest.approx(4 / 9), False, 0.0)
    state = stable_placement_retention_signal(True, state[0], state[2])
    assert state == (3, 1.0, True, 1.0)
    state = stable_placement_retention_signal(True, state[0], state[2])
    assert state == (3, 1.0, True, 0.0)
    state = stable_placement_retention_signal(False, state[0], state[2])
    assert state == (0, 0.0, True, 0.0)


def test_strict_basin_settling_rewards_braking_without_target_avoidance() -> None:
    stopped = strict_basin_settling_signal(0.03, 0.0, 0.02, tanh=math.tanh)
    threshold = strict_basin_settling_signal(0.03, 0.03, 0.02, tanh=math.tanh)
    fly_through = strict_basin_settling_signal(0.03, 0.20, 0.02, tanh=math.tanh)
    prearrival_stopped = strict_basin_settling_signal(0.10, 0.0, 0.02, tanh=math.tanh)
    prearrival_fast = strict_basin_settling_signal(0.10, 0.20, 0.02, tanh=math.tanh)
    transporting = strict_basin_settling_signal(0.20, 0.20, 0.02, tanh=math.tanh)
    assert stopped > 0.5
    assert stopped > threshold > fly_through > 0.0
    assert stopped > fly_through + 0.5
    assert prearrival_stopped > prearrival_fast + 0.05
    assert prearrival_fast > transporting > 0.0
    # Nominal value before Isaac's dt multiplier stays negligible at 20 cm.
    assert transporting * PLACEMENT_BASIN_SETTLING_REWARD_WEIGHT < 0.1
    with pytest.raises(ValueError, match="positive"):
        strict_basin_settling_signal(0.03, 0.0, 0.02, tanh=math.tanh, basin_width_m=0.0)


def test_stable_placement_curriculum_does_not_suppress_transport() -> None:
    transporting_far = placement_curriculum_signal(0.35, 0.20, 0.02, tanh=math.tanh)
    stopped_far = placement_curriculum_signal(0.45, 0.0, 0.02, tanh=math.tanh)
    assert STABLE_PLACEMENT_REWARD_WEIGHT * transporting_far > 4.0
    assert STABLE_PLACEMENT_REWARD_WEIGHT * stopped_far < 5.0


def test_stable_placement_curriculum_rejects_thrown_object_reward() -> None:
    held = placement_curriculum_signal(0.20, 0.20, 0.02, tanh=math.tanh)
    thrown = placement_curriculum_signal(0.20, 0.20, 0.50, tanh=math.tanh)
    assert held > thrown * 3


def test_drop_penalty_ramps_only_during_first_pass() -> None:
    assert drop_penalty_schedule_fraction(0, 7200, curriculum_enabled=True) == 0.0
    assert drop_penalty_schedule_fraction(3600, 7200, curriculum_enabled=True) == 0.5
    assert drop_penalty_schedule_fraction(7200, 7200, curriculum_enabled=True) == 1.0
    assert drop_penalty_schedule_fraction(0, 7200, curriculum_enabled=False) == 1.0


def test_drop_penalty_preserves_isaac_stateful_manager_term_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[object, object]] = []

    class FakeIsaacTerm:
        def __init__(self, cfg: object, env: object) -> None:
            calls.append((cfg, env))

        def __call__(self, env: object, term_keys: str | list[str] = ".*") -> float:
            calls.append((env, term_keys))
            return 2.0

    class Cfg:
        decimation = 4

    class Env:
        cfg = Cfg()
        _sim_step_counter = 0

    monkeypatch.delenv("NPA_SIM2REAL_ENABLE_GOAL_CURRICULUM", raising=False)
    term_type = _scheduled_drop_penalty_type(FakeIsaacTerm)
    env = Env()
    cfg = object()
    term = term_type(cfg, env)
    assert term(env, term_keys="object_dropping", full_goal_step=7200) == 2.0
    assert calls == [(cfg, env), (env, "object_dropping")]


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
Episode_Reward/near_goal_arm_stillness: 0.6250
Episode_Reward/stable_placement_dwell: 0.3750
Episode_Reward/stable_placement_dwell_break: -0.2500
Episode_Reward/stable_placement_completion: 0.1250
Episode_Reward/stable_placement_departure: -0.5000
Metrics/object_pose/position_error: 0.1800
Episode_Termination/stable_placement_success: 0.1250
Total timesteps: 24576
"""
    telemetry = parse_ppo_training_log(log)
    assert telemetry["configured_iterations"] == 500
    assert telemetry["final_iteration"]["value_loss"] == 0.02
    assert telemetry["final_iteration"]["total_timesteps"] == 24576
    assert telemetry["final_iteration"]["stable_placement_termination_rate"] == 0.125
    assert telemetry["final_iteration"]["near_goal_arm_stillness_reward"] == 0.625
    assert telemetry["final_iteration"]["stable_placement_dwell_reward"] == 0.375
    assert telemetry["final_iteration"]["stable_placement_dwell_break_reward"] == -0.25
    assert telemetry["final_iteration"]["stable_placement_completion_reward"] == 0.125
    assert telemetry["final_iteration"]["stable_placement_departure_reward"] == -0.5
    with pytest.raises(ValueError, match="no Learning iteration"):
        parse_ppo_training_log("no telemetry")
