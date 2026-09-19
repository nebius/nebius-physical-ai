"""Proof that reviewed subtask labels survive in real LeRobot Parquet rows."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from npa.fiftyone_lerobot_subtasks import SubtaskSegment, apply_subtask_segments
from npa.orchestration.npa_workflow import build_plan, load_spec, run_workflow
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.orchestration.npa_workflow.submit_matrix import SUBMIT_LIVE_MATRIX
from npa.workflows.lerobot_subtask_proof import (
    LeRobotSubtaskProofError,
    main,
    prove_lerobot_subtasks,
)

SPEC = Path(__file__).parents[3] / "workflows" / "testing" / "lerobot-subtask-proof.yaml"


def _write_lerobot_dataset(root: Path) -> Path:
    metadata = root / "meta"
    data = root / "data" / "chunk-000"
    metadata.mkdir(parents=True)
    data.mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "total_episodes": 2,
        "total_frames": 8,
        "features": {},
    }
    (metadata / "info.json").write_text(json.dumps(info), encoding="utf-8")
    table = pa.table(
        {
            "episode_index": [0, 0, 0, 0, 1, 1, 1, 1],
            "frame_index": [0, 1, 2, 3, 0, 1, 2, 3],
            "timestamp": [0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
            "action": [[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]],
        }
    )
    pq.write_table(table, data / "file-000.parquet")
    return root


def _reviewed_dataset(root: Path) -> Path:
    dataset = _write_lerobot_dataset(root)
    apply_subtask_segments(
        dataset,
        [
            SubtaskSegment(0, "approach", 0, 200_000_000),
            SubtaskSegment(0, "grasp", 200_000_000, 400_000_000),
            SubtaskSegment(1, "align", 0, 100_000_000),
            SubtaskSegment(1, "place", 100_000_000, 400_000_000),
        ],
    )
    return dataset


def test_proof_reads_a_concrete_subtask_from_lerobot_parquet(tmp_path: Path) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    data_path = dataset / "data" / "chunk-000" / "file-000.parquet"
    before = hashlib.sha256(data_path.read_bytes()).hexdigest()
    output = tmp_path / "evidence" / "subtask-proof.json"

    result = prove_lerobot_subtasks(str(dataset), str(output), expected_label="grasp")

    assert json.loads(output.read_text(encoding="utf-8")) == result
    assert result["status"] == "verified"
    assert result["summary"] == {
        "episode_count": 2,
        "frame_count": 8,
        "labeled_frame_count": 8,
        "unlabeled_frame_count": 0,
        "subtask_count": 4,
        "segment_count": 4,
    }
    assert result["proof"] == {
        "episode_index": 0,
        "frame_index": 2,
        "timestamp": 0.2,
        "subtask_index": 2,
        "source_data_file": "data/chunk-000/file-000.parquet",
        "source_parquet_sha256": before,
        "subtask": "grasp",
        "row_sha256": result["proof"]["row_sha256"],
    }
    unhashed = dict(result["proof"])
    recorded_row_hash = unhashed.pop("row_sha256")
    canonical = json.dumps(unhashed, sort_keys=True, separators=(",", ":")).encode()
    assert recorded_row_hash == hashlib.sha256(canonical).hexdigest()
    assert hashlib.sha256(data_path.read_bytes()).hexdigest() == before


@pytest.mark.parametrize(
    ("index", "message"),
    [(99, "unknown subtask_index 99"), (-1, "nonnegative subtask_index"), (None, "invalid subtask fields")],
)
def test_proof_rejects_invalid_subtask_indices(tmp_path: Path, index, message: str) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    path = dataset / "data" / "chunk-000" / "file-000.parquet"
    table = pq.read_table(path)
    indices = table.column("subtask_index").to_pylist()
    indices[0] = index
    table = table.set_column(
        table.column_names.index("subtask_index"),
        "subtask_index",
        pa.array(indices, type=pa.int64()),
    )
    pq.write_table(table, path)

    with pytest.raises(LeRobotSubtaskProofError, match=message):
        prove_lerobot_subtasks(str(dataset), str(tmp_path / "proof.json"))
    assert not (tmp_path / "proof.json").exists()


def test_missing_label_does_not_publish_a_proof(tmp_path: Path) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    with pytest.raises(LeRobotSubtaskProofError, match="expected subtask label was not found"):
        prove_lerobot_subtasks(str(dataset), str(tmp_path / "proof.json"), expected_label="lift")
    assert not (tmp_path / "proof.json").exists()


def test_proof_rejects_missing_dataset_frames(tmp_path: Path) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    path = dataset / "data" / "chunk-000" / "file-000.parquet"
    pq.write_table(pq.read_table(path).slice(0, 7), path)
    with pytest.raises(LeRobotSubtaskProofError, match="total_frames does not match"):
        prove_lerobot_subtasks(str(dataset), str(tmp_path / "proof.json"))


def test_proof_cannot_overwrite_the_source_dataset(tmp_path: Path) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    path = dataset / "meta" / "info.json"
    original = path.read_bytes()
    with pytest.raises(LeRobotSubtaskProofError, match="outside the source dataset"):
        prove_lerobot_subtasks(str(dataset), str(path))
    assert path.read_bytes() == original


@pytest.mark.parametrize("suffix", ["meta/info.json", "data/chunk-000/file-000.parquet"])
def test_s3_proof_cannot_overwrite_source_objects(suffix: str) -> None:
    with pytest.raises(LeRobotSubtaskProofError, match="outside the source dataset"):
        prove_lerobot_subtasks("s3://example-bucket/reviewed/", f"s3://example-bucket/reviewed/{suffix}")


def test_rendered_yaml_arguments_execute_the_proof(tmp_path: Path, capsys) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    output = tmp_path / "proof.json"
    spec = merge_config_overrides(load_spec(SPEC), {
        "reviewed_dataset_uri": str(dataset), "proof_uri": str(output),
    })
    step = build_plan(spec, run_id="local-proof").steps[0]
    assert main(step.argv[3:]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == json.loads(output.read_text())
    assert payload["proof"]["subtask"] == "grasp"


def test_workflow_interpreter_executes_the_declared_module(tmp_path: Path, monkeypatch) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    output = tmp_path / "proof.json"
    monkeypatch.setenv("PATH", str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", ""))
    monkeypatch.setenv("PYTHONPATH", str(SPEC.parents[2] / "npa" / "src"))
    spec = merge_config_overrides(load_spec(SPEC), {
        "reviewed_dataset_uri": str(dataset), "proof_uri": str(output),
    })

    report = run_workflow(spec, run_id="local-subtask-proof", execute=True)

    assert report["status"] == "completed"
    assert report["steps"][0]["returncode"] == 0
    proof = json.loads(output.read_text())
    assert proof["proof"]["subtask"] == "grasp"
    assert proof["summary"]["labeled_frame_count"] == 8


def test_s3_round_trip_only_downloads_label_metadata_and_frame_data(tmp_path: Path) -> None:
    dataset = _reviewed_dataset(tmp_path / "reviewed")
    downloaded: list[str] = []
    published: dict[str, dict] = {}

    class Storage:
        def download_file(self, uri: str, destination: str) -> None:
            downloaded.append(uri)
            Path(destination).parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(dataset / "meta" / uri.rsplit("/", 1)[1], destination)

        def download_directory(self, uri: str, destination: str) -> None:
            downloaded.append(uri)
            shutil.copytree(dataset / "data", destination)

        def upload_file(self, path: str, uri: str) -> str:
            published[uri] = json.loads(Path(path).read_text())
            return uri

    source = "s3://example-bucket/reviewed"
    output = "s3://example-bucket/evidence/proof.json"
    payload = prove_lerobot_subtasks(source, output, expected_label="grasp", storage_client=Storage())
    assert downloaded == [f"{source}/meta/info.json", f"{source}/meta/subtasks.parquet", f"{source}/data/"]
    assert published[output] == payload
    assert payload["source_catalog_sha256"] == hashlib.sha256(
        (dataset / "meta" / "subtasks.parquet").read_bytes()
    ).hexdigest()


def test_workflow_runs_argv_safe_proof_module_and_is_live_registered() -> None:
    plan = build_plan(load_spec(SPEC), run_id="subtask-proof-test")
    step = plan.steps[0]

    assert step.shell == ""
    assert step.argv[:3] == ["python3", "-m", "npa.workflows.lerobot_subtask_proof"]
    assert step.argv[step.argv.index("--expected-label") + 1] == "grasp"
    assert step.inputs[0]["schema"] == "lerobot.dataset.v3.subtasks"
    assert step.outputs[0]["schema"] == "npa.lerobot.subtask_proof.v1"
    case = next(case for case in SUBMIT_LIVE_MATRIX if case.spec == SPEC.name)
    assert case.tier == "cpu"
    assert not case.plan_only
