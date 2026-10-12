"""Native LeRobot v3 task-index compatibility at training admission."""

import json
from types import SimpleNamespace

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.workbench.lerobot import policy_container as trainer


def _write_tasks(root, table):
    (root / "meta").mkdir(parents=True)
    pq.write_table(table, root / "meta/tasks.parquet")


@pytest.mark.parametrize("index_name", [None, "task", "instruction"])
def test_native_pandas_task_index_reaches_training(
    monkeypatch, tmp_path, capsys, index_name
):
    dataset = tmp_path / "dataset"
    frame = pd.DataFrame(
        {"task_index": [1, 0]}, index=pd.Index(["Close", "Open"], name=index_name)
    )
    (dataset / "meta").mkdir(parents=True)
    # Native LeRobot v3 persists this indexed DataFrame directly to Parquet.
    frame.to_parquet(dataset / "meta/tasks.parquet")
    assert pq.read_table(dataset / "meta/tasks.parquet").to_pandas().index.tolist() == [
        "Close",
        "Open",
    ]
    output = tmp_path / "output"
    calls = []

    def train(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(output_dir=output, to_dict=lambda: {"status": "trained"})

    monkeypatch.setattr(trainer, "run_lerobot_training", train)
    assert (
        trainer.main(
            [
                "train",
                "--dataset-path",
                str(dataset),
                "--output-dir",
                str(output),
                "--steps",
                "1",
            ]
        )
        == 0
    )
    assert calls[0]["dataset_path"] == dataset
    payload = json.loads(capsys.readouterr().out)["training_dataset_provenance"]
    assert payload["dataset_tasks"] == ["Open", "Close"]
    assert payload["declared_training_tasks_verified"] is False
    assert json.loads(
        (output / trainer.TRAINING_DATASET_PROVENANCE_FILENAME).read_text()
    )["dataset_tasks"] == ["Open", "Close"]


def test_explicit_task_column_wins_over_pandas_index(tmp_path):
    frame = pd.DataFrame({"task_index": [0], "task": ["Open"]}, index=["not-the-task"])
    _write_tasks(tmp_path, pa.Table.from_pandas(frame))
    assert trainer._read_dataset_tasks(tmp_path) == ["Open"]


@pytest.mark.parametrize(
    "case",
    [
        "no_metadata",
        "range",
        "multi",
        "numeric",
        "missing_index",
        "missing_task_index",
        "duplicate",
    ],
)
def test_ambiguous_or_invalid_native_tasks_remain_rejected(monkeypatch, tmp_path, case):
    frame = pd.DataFrame({"task_index": [0, 1]}, index=["Open", "Close"])
    if case == "range":
        frame = frame.reset_index(drop=True)
    elif case == "multi":
        frame.index = pd.MultiIndex.from_tuples([("Open", "a"), ("Close", "b")])
    elif case == "numeric":
        frame.index = [10, 20]
    elif case == "duplicate":
        frame.index = ["Open", "Open"]
    table = pa.Table.from_pandas(frame)
    if case == "no_metadata":
        table = table.replace_schema_metadata(None)
    elif case == "missing_index":
        table = table.drop(["__index_level_0__"])
    elif case == "missing_task_index":
        table = table.drop(["task_index"])
    _write_tasks(tmp_path, table)
    monkeypatch.setattr(
        trainer,
        "run_lerobot_training",
        lambda **kwargs: pytest.fail("invalid task metadata must not reach training"),
    )
    with pytest.raises(trainer.PolicyContainerError):
        trainer.main(
            [
                "train",
                "--dataset-path",
                str(tmp_path),
                "--output-dir",
                str(tmp_path / "output"),
                "--steps",
                "1",
            ]
        )
    assert not (tmp_path / "output").exists()


def test_repository_adapter_task_producer_is_accepted(tmp_path):
    from npa.adapter.sim_to_lerobot import _write_tasks_parquet

    _write_tasks_parquet(["Open", "Close"], tmp_path / "meta/tasks.parquet")
    assert trainer._read_dataset_tasks(tmp_path) == ["Open", "Close"]


def test_jsonl_tasks_keep_precedence_over_parquet(tmp_path):
    _write_tasks(tmp_path, pa.table({"task_index": [0], "task": ["Parquet"]}))
    (tmp_path / "meta/tasks.jsonl").write_text('{"task_index": 0, "task": "JSONL"}\n')
    assert trainer._read_dataset_tasks(tmp_path) == ["JSONL"]


def test_native_task_provenance_retains_declared_order_and_content_binding(tmp_path):
    frame = pd.DataFrame({"task_index": [1, 0]}, index=["Close", "Open"])
    _write_tasks(tmp_path, pa.Table.from_pandas(frame))
    args = {
        "dataset_source": "fixture",
        "declared_training_env_ids": "robocasa/Open,robocasa/Close",
    }
    first = trainer.build_training_dataset_provenance(tmp_path, **args)
    assert first["declared_training_tasks_verified"] is True
    (tmp_path / "data.bin").write_bytes(b"changed dataset")
    assert (
        trainer.build_training_dataset_provenance(tmp_path, **args)[
            "dataset_tree_sha256"
        ]
        != first["dataset_tree_sha256"]
    )
    with pytest.raises(trainer.PolicyContainerError, match="exactly match"):
        trainer.build_training_dataset_provenance(
            tmp_path,
            dataset_source="fixture",
            declared_training_env_ids="robocasa/Close,robocasa/Open",
        )
