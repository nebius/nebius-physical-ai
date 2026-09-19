"""Protect exact sample order, final partial batches, and prefix shard membership."""
# ruff: noqa: E402

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

IMPLEMENTATION = (
    Path(__file__).resolve().parents[3]
    / "workflows/implementations/behavior-matched-training"
)
sys.path.insert(0, str(IMPLEMENTATION))

import generate_prefix_records as prefix
import merge_prefix_records as merge_module
from ordered_data import OrderedBatches


class NumberSamples:
    """Vary worker completion order while keeping deterministic sample bytes."""

    def __len__(self):
        return 7

    def __getitem__(self, index):
        if index == 0:
            time.sleep(0.02)
        return np.array([index, index * 2], dtype=np.int32)


@pytest.mark.parametrize("workers", [0, 2])
def test_ordered_workers_preserve_indices_partial_batch_and_second_pass(workers):
    pytest.importorskip("torch")
    expected = ((6, 0, 3), (2, 5))
    with OrderedBatches(NumberSamples(), expected, workers=workers) as loader:
        for _pass in range(2):
            observed = list(loader)
            assert tuple(indices for indices, _ in observed) == expected
            assert [
                [sample.tolist() for sample in samples] for _, samples in observed
            ] == [[[6, 12], [0, 0], [3, 6]], [[2, 4], [5, 10]]]
    assert loader.loader._iterator is None


def test_auxiliary_padding_never_adds_replay_frames(monkeypatch):
    seen = []
    monkeypatch.setattr(prefix, "_stack", lambda rows: (np.asarray(rows), None))

    def predict(_model, observation):
        seen.append(len(observation))
        return observation[:, None]

    assert prefix._predict_batches(None, predict, [1, 2, 3, 4, 5], 4) == [
        [1.0],
        [2.0],
        [3.0],
        [4.0],
        [5.0],
    ]
    assert seen == [4, 4]


def _record(task: int, episode: int) -> dict:
    return {
        "schema": "npa.behavior.rlc-prefix-record.v2",
        "task_id": task,
        "episode_index": episode,
        "episode_length": 21,
        "frames": [0, 20],
        "auxiliary_raw_logits": [[0.0] * 15] * 2,
        "served_valid_logits": [[0.0] * 5] * 2,
        "sample_identities": [
            {"observation_sha256": "a" * 64, "action_sha256": "b" * 64}
        ]
        * 2,
    }


def _merge_args(tmp_path, monkeypatch):
    split = {
        "tasks": {
            str(task): {"training": [task * 100, task * 100 + 1]} for task in (0, 1, 22)
        }
    }
    monkeypatch.setattr(merge_module, "load_split", lambda _path: split)
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps(split))
    shards = []
    for index in range(2):
        path = tmp_path / f"shard-{index}.jsonl"
        rows = [_record(task, task * 100 + index) for task in (0, 1, 22)]
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        shards.append(path)
    return SimpleNamespace(
        episode_split=split_path,
        split="training",
        shard_input=shards,
        output=tmp_path / "merged.jsonl",
        receipt=tmp_path / "receipt.json",
    )


def test_prefix_merge_restores_full_canonical_order(tmp_path, monkeypatch):
    args = _merge_args(tmp_path, monkeypatch)
    receipt = merge_module.merge(args)
    records = [json.loads(line) for line in args.output.read_text().splitlines()]
    assert [(row["task_id"], row["episode_index"]) for row in records] == [
        (0, 0),
        (0, 1),
        (1, 100),
        (1, 101),
        (22, 2200),
        (22, 2201),
    ]
    assert receipt["episodes"] == 6 and receipt["frames"] == 12
    assert [row["episodes"] for row in receipt["shards"]] == [3, 3]


@pytest.mark.parametrize(
    "damage", ["missing", "duplicate", "reorder", "frames", "foreign"]
)
def test_prefix_merge_rejects_incomplete_or_cross_shard_data(
    tmp_path, monkeypatch, damage
):
    args = _merge_args(tmp_path, monkeypatch)
    path = args.shard_input[0]
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    if damage == "missing":
        rows.pop()
    elif damage == "duplicate":
        rows.append(rows[0])
    elif damage == "reorder":
        rows.reverse()
    elif damage == "frames":
        rows[0]["frames"] = [0, 0]
    else:
        rows[0]["episode_index"] = 1
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    with pytest.raises(ValueError):
        merge_module.merge(args)
    assert not args.output.exists() and not args.receipt.exists()
