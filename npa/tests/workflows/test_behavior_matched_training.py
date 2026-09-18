"""Exercise the paired stage-conditioning data and selection contracts."""

# ruff: noqa: E402

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

IMPLEMENTATION = (
    Path(__file__).parents[3] / "workflows/implementations/behavior-matched-training"
)
sys.path.insert(0, str(IMPLEMENTATION))

from action_partition import EXPECTED_TRAINABLE_PATHS, assert_partition
from build_replay_trace import build
from checkpoint_selection import LossRow, select_checkpoint
from export_selected import export_selected
from gpu_preflight import _logits_equal, _result
from generate_prefix_records import _transform_raw_sample
from panel_data import PanelDataset, ReplanDataset, TaskBalancedSampler
from stage_conditioning import (
    ApplyStageCondition,
    FilterState,
    FrameKey,
    ReplayRow,
    load_trace,
    replay_episode,
    update_filter,
    write_trace,
)


def _scores(stage: int) -> list[float]:
    scores = [-10.0] * 15
    scores[stage] = 10.0
    return scores


def _digests(count: int) -> list[tuple[str, str]]:
    return [(f"{index:064x}", f"{index + 100:064x}") for index in range(count)]


def test_public_package_has_no_campaign_locations() -> None:
    forbidden = ("/task/", "/Users/", "npa-bucket-", "behavior-2026-pr-550")
    for path in IMPLEMENTATION.iterdir():
        if path.is_file():
            assert not any(value in path.read_text() for value in forbidden), path


def test_matched_configs_differ_only_by_stage_source() -> None:
    teacher = json.loads((IMPLEMENTATION / "config-teacher.json").read_text())
    replay = json.loads((IMPLEMENTATION / "config-replay.json").read_text())
    differences = {key for key in teacher if teacher[key] != replay[key]}
    assert differences == {"arm", "candidate"}


def test_replay_uses_pre_update_native_hysteresis() -> None:
    rows = replay_episode(
        task_id=0,
        episode_index=7,
        episode_length=80,
        logits=[_scores(1), _scores(0), _scores(1), _scores(2)],
        sample_digests=_digests(4),
    )
    assert [row.replay_stage for row in rows] == [0, 0, 0, 1]
    assert [row.post_update_stage for row in rows] == [0, 0, 1, 1]
    assert [row.teacher_stage for row in rows] == [0, 1, 2, 3]


def test_task0_reset_precedes_vote() -> None:
    state = FilterState(stage=4)
    state.history.extend([4, 4])
    updated, reset = update_filter(state, raw_prediction=3, task_id=0)
    assert reset is True
    assert updated.stage == 2
    assert tuple(updated.history) == (3,)


def test_gpu_preflight_compares_storage_dtypes_as_float32(monkeypatch) -> None:
    direct = np.array([[1.0, 2.0, np.nan]], dtype=np.float16)
    native = np.array([[1.0, 2.0, np.nan]], dtype=np.float32)
    original = np.allclose

    def require_float32(left, right, **kwargs):
        assert left.dtype == right.dtype == np.dtype(np.float32)
        return original(left, right, **kwargs)

    monkeypatch.setattr(np, "allclose", require_float32)
    assert _logits_equal(direct, native) == (True, 2)


def test_gpu_receipt_uses_real_panel_index_contract() -> None:
    class EpisodeDataset:
        episodes = [311]
        episode_data_index = {"from": np.array([0]), "to": np.array([21])}

    class FakePanel(PanelDataset):
        def __getitem__(self, index):
            assert index == 0
            return {"task_index": 0, "episode_index": 311}

    panel = FakePanel.__new__(FakePanel)
    panel.task_ids = (0,)
    panel.offsets = [0]
    panel.datasets = [EpisodeDataset()]

    class Device:
        device_kind = "NVIDIA B200"

    class Jax:
        @staticmethod
        def devices():
            return [Device()]

    receipt = _result(Jax(), panel, finite_logits=15)
    assert receipt["real_training_sample"] == {
        "task_id": 0,
        "episode_index": 311,
        "episode_relative_frame": 0,
    }


def test_prefix_transform_reuses_decoded_sample_with_native_bytes() -> None:
    panel = object()

    class Transformed:
        _dataset = panel

        @staticmethod
        def _transform(sample):
            return {"value": np.asarray(sample["value"], dtype=np.float32) * 2}

    raw = {"value": np.array([1, 2, 3], dtype=np.int16)}
    actual = _transform_raw_sample(Transformed(), panel, raw)
    expected = Transformed()._transform(raw)
    assert actual["value"].tobytes() == expected["value"].tobytes()
    with pytest.raises(ValueError, match="wrap"):
        _transform_raw_sample(Transformed(), object(), raw)


def test_trace_preserves_raw_argmax_before_native_clamp() -> None:
    scores = _scores(14)
    scores[2] = 9.0
    rows = replay_episode(
        task_id=0,
        episode_index=8,
        episode_length=20,
        logits=[scores],
        sample_digests=_digests(1),
    )
    assert rows[0].raw_argmax == 14
    assert rows[0].history_after == (2,)


def test_native_filter_does_not_move_back_from_maximum_stage() -> None:
    state = FilterState(stage=5)
    state.history.extend([4, 4])
    updated, reset = update_filter(state, raw_prediction=4, task_id=1)
    assert reset is False
    assert updated.stage == 5
    assert tuple(updated.history) == (4, 4, 4)


def test_trace_rejects_identity_and_duplicate_drift(tmp_path: Path) -> None:
    rows = replay_episode(
        task_id=1,
        episode_index=200,
        episode_length=40,
        logits=[_scores(0), _scores(1)],
        sample_digests=_digests(2),
    )
    path = tmp_path / "trace.jsonl"
    identities = {"split_sha256": "a" * 64, "checkpoint_sha256": "b" * 64}
    write_trace(path, rows, split="training", identities=identities)
    assert list(load_trace(path, split="training", identities=identities)) == [
        row.key for row in rows
    ]
    with pytest.raises(ValueError, match="header"):
        load_trace(path, split="holdout", identities=identities)
    duplicate = dataclasses_replace(rows[1], key=rows[0].key)
    with pytest.raises(ValueError, match="duplicate"):
        write_trace(
            tmp_path / "bad.jsonl",
            [rows[0], duplicate],
            split="training",
            identities=identities,
        )


def dataclasses_replace(row: ReplayRow, **changes) -> ReplayRow:
    """Replace fields without hiding the dataclasses dependency in assertions."""

    import dataclasses

    return dataclasses.replace(row, **changes)


def test_arm_transform_changes_only_condition() -> None:
    source = {
        "tokenized_prompt": np.array([22, 4], dtype=np.int32),
        "teacher_stage": np.array(4, dtype=np.int32),
        "replay_stage": np.array(2, dtype=np.int32),
        "state": np.arange(32, dtype=np.float32),
    }
    teacher = ApplyStageCondition("teacher")(source)
    replay = ApplyStageCondition("replay")(source)
    assert teacher["tokenized_prompt"].tolist() == [22, 4]
    assert replay["tokenized_prompt"].tolist() == [22, 2]
    assert np.array_equal(teacher["state"], replay["state"])
    with pytest.raises(ValueError, match="equal-time"):
        ApplyStageCondition("replay")({**source, "teacher_stage": np.array(3)})


def _loss(step: int, task: int, value: float) -> LossRow:
    return LossRow(step, task, task * 100, 0, value, value + 1, (value,) * 23)


def test_selection_equal_weights_tasks_and_breaks_tie_early() -> None:
    rows = []
    for step in (600, 1200):
        rows.extend([_loss(step, 0, 1.0), _loss(step, 1, 2.0), _loss(step, 22, 3.0)])
    result = select_checkpoint(rows)
    assert result["selected_step"] == 600
    assert result["selected_score"] == 2.0


def test_partition_is_exact_and_excludes_stage_modules() -> None:
    assert len(EXPECTED_TRAINABLE_PATHS) == 23
    assert_partition(EXPECTED_TRAINABLE_PATHS | {("stage_pred_from_vlm", "kernel")})
    with pytest.raises(ValueError, match="lacks"):
        assert_partition(
            set(EXPECTED_TRAINABLE_PATHS) - {("action_out_proj", "kernel")}
        )


def test_frozen_configs_form_a_matched_pair() -> None:
    teacher = json.loads((IMPLEMENTATION / "config-teacher.json").read_text())
    replay = json.loads((IMPLEMENTATION / "config-replay.json").read_text())
    differing = {key for key in teacher if teacher[key] != replay[key]}
    assert differing == {"arm", "candidate"}


def test_training_prefix_has_equal_quotas_without_task0_reuse() -> None:
    dataset = ReplanDataset.__new__(ReplanDataset)
    dataset.indices_by_task = [
        list(range(19388)),
        list(range(20000, 20000 + 47820)),
        list(range(70000, 70000 + 69882)),
    ]
    prefix = []
    for position, index in enumerate(TaskBalancedSampler(dataset, seed=0)):
        if position == 57600:
            break
        prefix.append(index)
    assert len(prefix) == 57600
    assert len({index for index in prefix if index < 19388}) == 19200
    assert sum(index < 19388 for index in prefix) == 19200
    assert sum(20000 <= index < 67820 for index in prefix) == 19200
    assert sum(index >= 70000 for index in prefix) == 19200


def test_replan_join_rejects_unmapped_or_holdout_key() -> None:
    class EpisodeDataset:
        def __init__(self, episode: int):
            self.episodes = [episode]
            self.episode_data_index = {"from": np.array([0]), "to": np.array([21])}

    class Panel:
        def __init__(self):
            self.task_ids = (0, 1, 22)
            self.offsets = [0, 21, 42]
            self.datasets = [
                EpisodeDataset(0),
                EpisodeDataset(200),
                EpisodeDataset(4400),
            ]

        def __len__(self):
            return 63

        def __getitem__(self, index):
            slot = index // 21
            return {
                "task_index": np.array(self.task_ids[slot]),
                "episode_index": np.array(self.datasets[slot].episodes[0]),
            }

    trace = {}
    for task_id, episode in ((0, 0), (1, 200), (22, 4400)):
        for frame in (0, 20):
            row = ReplayRow(
                key=FrameKey(task_id, episode, frame),
                timestamp=frame / 30,
                teacher_stage=0,
                replay_stage=0,
                post_update_stage=0,
                raw_argmax=0,
                raw_logits=tuple(_scores(0)),
                history_before=(),
                history_after=(0,),
                task0_reset_applied=False,
                stage_count={0: 5, 1: 6, 22: 9}[task_id],
                observation_sha256="a" * 64,
                action_sha256="b" * 64,
            )
            trace[row.key] = row
    joined = ReplanDataset(Panel(), trace)
    assert joined.flat_indices == (0, 20, 21, 41, 42, 62)
    trace[FrameKey(0, 199, 0)] = next(iter(trace.values()))
    with pytest.raises(ValueError, match="held-out"):
        ReplanDataset(Panel(), trace)


def test_trace_builder_rejects_noncanonical_episode_order(tmp_path: Path) -> None:
    identities = tmp_path / "identities.json"
    identities.write_text(json.dumps({"split": "a" * 64}))
    source = tmp_path / "prefix.jsonl"
    record = {
        "task_id": 0,
        "episode_index": 2,
        "episode_length": 20,
        "frames": [0],
        "raw_logits": [_scores(0)],
        "sample_identities": [
            {"observation_sha256": "a" * 64, "action_sha256": "b" * 64}
        ],
    }
    source.write_text("\n".join(json.dumps(row) for row in (record, record)) + "\n")
    with pytest.raises(ValueError, match="strictly ordered"):
        build(source, tmp_path / "trace.jsonl", identities, "training")


def test_export_copies_only_selected_inference_tree(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoints" / "600"
    (checkpoint / "params").mkdir(parents=True)
    (checkpoint / "assets").mkdir()
    (checkpoint / "train_state").mkdir()
    (checkpoint / "params" / "weights").write_bytes(b"params")
    (checkpoint / "assets" / "norm").write_bytes(b"norm")
    (checkpoint / "train_state" / "optimizer").write_bytes(b"private")
    selection = tmp_path / "selection.json"
    selection.write_text(
        json.dumps(
            {
                "schema": "npa.behavior.rlc-holdout-selection.v1",
                "selected_step": 600,
            }
        )
    )
    output = tmp_path / "model"
    receipt = export_selected(checkpoint.parent, selection, output)
    assert set(receipt["files"]) == {"assets/norm", "params/weights"}
    assert not (output / "train_state").exists()
