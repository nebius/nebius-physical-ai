"""Record a synthetic LeRobot before/after proof by executing the checked-in YAML."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import rerun as rr
import rerun.blueprint as rrb
from rerun.recording import load_recording

from npa.fiftyone_lerobot_subtasks import (
    apply_subtask_segments,
    segments_from_temporal_tags,
)

REPOSITORY = Path(__file__).resolve().parents[2]
WORKFLOW = REPOSITORY / "workflows/testing/lerobot-subtask-proof.yaml"
FRAME_FILE = "data/chunk-000/file-000.parquet"
APPLICATION_ID = "npa-lerobot-subtask-proof"
TIMELINE = "dataset_row"
LIMITATIONS = (
    "Synthetic numeric LeRobot-format fixture, not captured robot data. "
    "Temporal tags are supplied programmatically to the production label parser; "
    "this is not a recording of FiftyOne UI interaction or native dataset export. "
    "No camera footage, model inference, policy training, or cloud execution is claimed."
)


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): _digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _input_frames() -> pa.Table:
    return pa.table(
        {
            "episode_index": [0, 0, 0, 0, 1, 1, 1, 1],
            "frame_index": [0, 1, 2, 3, 0, 1, 2, 3],
            "timestamp": [0.0, 0.1, 0.2, 0.3, 0.0, 0.1, 0.2, 0.3],
            "index": list(range(8)),
            "task_index": [0] * 8,
            "action": [[0.0], [0.1], [0.2], [0.3], [1.0], [1.1], [1.2], [1.3]],
        }
    )


def _write_input(root: Path) -> None:
    table = _input_frames()
    features = {
        name: {
            "dtype": "float64" if name in {"timestamp", "action"} else "int64",
            "shape": [1],
            "names": None,
        }
        for name in table.column_names
    }
    _write_json(
        root / "meta/info.json",
        {
            "codebase_version": "v3.0",
            "robot_type": "synthetic-test-fixture",
            "fps": 10,
            "total_episodes": 2,
            "total_frames": 8,
            "total_tasks": 1,
            "features": features,
            "splits": {"train": "0:2"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        },
    )
    (root / FRAME_FILE).parent.mkdir(parents=True)
    pq.write_table(table, root / FRAME_FILE)
    task = "Pick and place the object (synthetic contract fixture)"
    pq.write_table(
        pa.table({"task_index": [0], "task": [task]}), root / "meta/tasks.parquet"
    )
    episodes = root / "meta/episodes/chunk-000/file-000.parquet"
    episodes.parent.mkdir(parents=True)
    pq.write_table(
        pa.table(
            {
                "episode_index": [0, 1],
                "length": [4, 4],
                "tasks": [[task], [task]],
                "data/chunk_index": [0, 0],
                "data/file_index": [0, 0],
                "dataset_from_index": [0, 4],
                "dataset_to_index": [4, 8],
            }
        ),
        episodes,
    )


def _temporal_tags() -> list[dict]:
    intervals = [
        (0, "approach", 0, 2),
        (0, "grasp", 2, 4),
        (1, "align", 0, 1),
        (1, "place", 1, 4),
    ]
    return [
        {
            "sample_id": f"episode-{episode}",
            "tag": f"subtask:{label}",
            "index_type": 2,
            "start": start * 100_000_000,
            "end": end * 100_000_000,
        }
        for episode, label, start, end in intervals
    ]


def _label_copy(root: Path) -> None:
    tags = _temporal_tags()
    _write_json(root / "temporal-tags.json", tags)
    segments = segments_from_temporal_tags({"episode-0": 0, "episode-1": 1}, tags)
    shutil.copytree(root / "input", root / "reviewed")
    _write_json(
        root / "export-report.json", apply_subtask_segments(root / "reviewed", segments)
    )


def _execute_yaml(root: Path, run_id: str) -> dict:
    environment = dict(os.environ)
    environment["PATH"] = (
        str(Path(sys.executable).parent) + os.pathsep + environment.get("PATH", "")
    )
    environment["PYTHONPATH"] = str(REPOSITORY / "npa/src")
    command = [
        sys.executable,
        "-m",
        "npa",
        "workbench",
        "workflow",
        "run-spec",
        str(WORKFLOW),
        "--run-id",
        run_id,
        "--execute",
        "--json",
        "--var",
        "reviewed_dataset_uri=reviewed",
        "--var",
        "proof_uri=subtask-proof.json",
    ]
    result = subprocess.run(
        command, cwd=root, env=environment, text=True, capture_output=True, check=True
    )
    report = json.loads(result.stdout)
    if report["status"] != "completed" or any(
        step["returncode"] != 0 for step in report["steps"]
    ):
        raise ValueError("The YAML proof did not complete successfully")
    # Retain the factual execution report without publishing the checkout path.
    report = json.loads(json.dumps(report).replace(str(REPOSITORY), "<checkout>"))
    _write_json(root / "workflow-report.json", report)
    shutil.copyfile(WORKFLOW, root / "workflow.yaml")
    return report


def _before_after(root: Path) -> list[dict]:
    original_table = pq.read_table(root / "input" / FRAME_FILE)
    reviewed_table = pq.read_table(root / "reviewed" / FRAME_FILE)
    if not original_table.equals(
        reviewed_table.select(original_table.column_names), check_metadata=True
    ):
        raise ValueError("Labeling changed an original frame column")
    original = original_table.to_pylist()
    reviewed = reviewed_table.to_pylist()
    catalog = pq.read_table(root / "reviewed/meta/subtasks.parquet").to_pylist()
    labels = {row["subtask_index"]: row["subtask"] for row in catalog}
    result = []
    for before, after in zip(original, reviewed, strict=True):
        result.append(
            {
                "input": before,
                "reviewed": after,
                "subtask": labels[after["subtask_index"]],
            }
        )
    _write_json(root / "before-after.json", result)
    return result


def _provenance(root: Path, run_id: str, source_hashes: dict) -> dict:
    files = [
        Path(__file__),
        WORKFLOW,
        REPOSITORY / "npa/src/npa/fiftyone_lerobot_subtasks.py",
        REPOSITORY / "npa/src/npa/workflows/lerobot_subtask_proof.py",
    ]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPOSITORY,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.strip()
    return {
        "run_id": run_id,
        "producer": "npa/scripts/record_lerobot_subtask_proof.py",
        "checkout_base_revision": revision,
        "recipe_sha256": {
            path.relative_to(REPOSITORY).as_posix(): _digest(path) for path in files
        },
        "input_sha256": source_hashes,
        "limitations": LIMITATIONS,
        "timeline": "dataset_row is a row ordinal, not elapsed capture or execution time",
        "versions": {"pyarrow": pa.__version__, "rerun": rr.__version__},
        "proof": json.loads((root / "subtask-proof.json").read_text()),
    }


def _blueprint() -> rrb.Blueprint:
    return rrb.Blueprint(
        rrb.Vertical(
            rrb.TextDocumentView(origin="summary", name="Evidence and limitations"),
            rrb.Horizontal(
                rrb.TextDocumentView(origin="input/frame", name="Original LeRobot row"),
                rrb.TextDocumentView(
                    origin="reviewed/frame", name="Same row with subtask label"
                ),
            ),
            rrb.Horizontal(
                rrb.TimeSeriesView(origin="input/action_0", name="Original action[0]"),
                rrb.TimeSeriesView(
                    origin="subtasks/index", name="Subtask index (categorical)"
                ),
            ),
            row_shares=[1.0, 1.6, 1.0],
        ),
        rrb.TimePanel(timeline=TIMELINE, state=rrb.PanelState.Expanded),
        auto_layout=False,
    )


def _summary_text(provenance: dict) -> str:
    proof = provenance["proof"]
    row = proof["proof"]
    return (
        "# LeRobot input → temporal tags → labeled copy → YAML proof\n\n"
        f"{LIMITATIONS}\n\n"
        f"**{proof['summary']['labeled_frame_count']}/{proof['summary']['frame_count']} frames labeled.** "
        "All original columns preserved; input file hashes unchanged.\n\n"
        f"Scrub dataset_row to **2**: episode {row['episode_index']}, frame {row['frame_index']}, "
        f"timestamp {row['timestamp']} s → **{row['subtask']}**, subtask_index **{row['subtask_index']}**.\n\n"
        "Catalog: 0 = align · 1 = approach · 2 = grasp · 3 = place.\n\n"
        "The row ordinal is not elapsed capture time; timestamps reset at each episode."
    )


def _write_recording(root: Path, rows: list[dict], provenance: dict) -> Path:
    path = root / "lerobot-subtasks.rrd"
    recording = rr.RecordingStream(APPLICATION_ID, recording_id=provenance["run_id"])
    try:
        recording.save(path, default_blueprint=_blueprint())
        recording.log(
            "provenance/run",
            rr.TextDocument(json.dumps(provenance, indent=2)),
            static=True,
        )
        recording.log(
            "summary",
            rr.TextDocument(_summary_text(provenance), media_type="text/markdown"),
            static=True,
        )
        for index, row in enumerate(rows):
            recording.set_time(TIMELINE, sequence=index)
            recording.log(
                "input/frame", rr.TextDocument(json.dumps(row["input"], indent=2))
            )
            after = {**row["reviewed"], "resolved_subtask": row["subtask"]}
            recording.log(
                "reviewed/frame", rr.TextDocument(json.dumps(after, indent=2))
            )
            recording.log("input/action_0", rr.Scalars(row["input"]["action"][0]))
            recording.log(
                "subtasks/index", rr.Scalars(row["reviewed"]["subtask_index"])
            )
        recording.flush()
    finally:
        recording.disconnect()
    return path


def _decoded_entities(path: Path) -> dict[str, list]:
    observed: dict[str, list] = {}
    for chunk in load_recording(path).chunks():
        batch = chunk.to_record_batch()
        if TIMELINE not in batch.schema.names:
            continue
        column = (
            "TextDocument:text"
            if chunk.entity_path.endswith("/frame")
            else "Scalars:scalars"
        )
        if column not in batch.schema.names:
            continue
        rows = zip(
            batch.column(TIMELINE).to_pylist(),
            batch.column(column).to_pylist(),
            strict=True,
        )
        observed.setdefault(chunk.entity_path, []).extend(
            (index, values[0]) for index, values in rows
        )
    return {entity: sorted(rows) for entity, rows in observed.items()}


def _inspect_recording(path: Path, rows: list[dict], run_id: str) -> dict:
    executable = str(Path(sys.executable).with_name("rerun"))
    checked = subprocess.run(
        [executable, "rrd", "verify", str(path)],
        text=True,
        capture_output=True,
        check=True,
    )
    printed = subprocess.run(
        [executable, "rrd", "print", "-vv", str(path)],
        text=True,
        capture_output=True,
        check=True,
    )
    inspection = printed.stdout + printed.stderr
    if any(
        value not in inspection
        for value in (APPLICATION_ID, run_id, TIMELINE, "provenance/run")
    ):
        raise ValueError("Decoded recording is missing identity or provenance")
    recording = load_recording(path)
    if (
        recording.application_id() != APPLICATION_ID
        or recording.recording_id() != run_id
    ):
        raise ValueError("Decoded recording identity does not match the run")
    observed = _decoded_entities(path)
    expected = {
        "/input/action_0": [
            (index, row["input"]["action"][0]) for index, row in enumerate(rows)
        ],
        "/subtasks/index": [
            (index, row["reviewed"]["subtask_index"]) for index, row in enumerate(rows)
        ],
        "/input/frame": [
            (index, json.dumps(row["input"], indent=2))
            for index, row in enumerate(rows)
        ],
        "/reviewed/frame": [
            (
                index,
                json.dumps(
                    {**row["reviewed"], "resolved_subtask": row["subtask"]}, indent=2
                ),
            )
            for index, row in enumerate(rows)
        ],
    }
    if observed != expected:
        raise ValueError(
            "Decoded recording does not match every source frame and subtask"
        )
    return {
        "status": "verified",
        "sha256": _digest(path),
        "bytes": path.stat().st_size,
        "application_id": recording.application_id(),
        "recording_id": recording.recording_id(),
        "timeline": TIMELINE,
        "row_ordinals": list(range(len(rows))),
        "samples_per_entity": {
            entity: len(samples) for entity, samples in observed.items()
        },
        "rrd_verify": (checked.stdout + checked.stderr).strip(),
        "rrd_print_sha256": hashlib.sha256(inspection.encode()).hexdigest(),
    }


def _finish_bundle(
    root: Path, provenance: dict, inspection: dict, source_hashes: dict
) -> dict:
    manifest = {
        "schema": "npa.lerobot.subtask_recording.v1",
        "status": "verified",
        "provenance": provenance,
        "recording": inspection,
        "input_unchanged": source_hashes == _inventory(root / "input"),
        "original_frame_columns_unchanged": True,
        "artifacts": _inventory(root),
    }
    if not manifest["input_unchanged"]:
        raise ValueError("The original input dataset was modified")
    _write_json(root / "manifest.json", manifest)
    with zipfile.ZipFile(
        root / "proof-bundle.zip", "w", compression=zipfile.ZIP_DEFLATED
    ) as bundle:
        for relative in [*manifest["artifacts"], "manifest.json"]:
            bundle.write(root / relative, relative)
    return manifest


def record_proof(output_dir: Path, run_id: str) -> dict:
    """Save a synthetic before/after bundle with executed YAML and decoded RRD evidence.

    Args:
        output_dir: New local directory; existing directories are never overwritten.
        run_id: Identifier shared by the YAML run and Rerun recording.

    Returns:
        Manifest containing source hashes, proof, and independent recording checks.

    Raises:
        FileExistsError: The destination already exists.
        ValueError: Source preservation or recording content checks fail.
        subprocess.CalledProcessError: YAML execution or Rerun verification fails.
    """
    root = output_dir.expanduser().resolve()
    root.mkdir(parents=True, exist_ok=False)
    _write_input(root / "input")
    source_hashes = _inventory(root / "input")
    _label_copy(root)
    _execute_yaml(root, run_id)
    rows = _before_after(root)
    provenance = _provenance(root, run_id, source_hashes)
    recording = _write_recording(root, rows, provenance)
    inspection = _inspect_recording(recording, rows, run_id)
    return _finish_bundle(root, provenance, inspection, source_hashes)


def main(argv: list[str] | None = None) -> int:
    """Create a local recorded proof without requiring cloud or FiftyOne services.

    Args:
        argv: Optional arguments excluding the program name.

    Returns:
        Zero after the complete bundle has passed its validation checks.

    Raises:
        FileExistsError: The output directory already exists.
        ValueError: Input preservation or decoded recording verification fails.
        subprocess.CalledProcessError: The YAML or Rerun CLI fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", default="lerobot-subtask-recorded-proof")
    args = parser.parse_args(argv)
    manifest = record_proof(args.output_dir, args.run_id)
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
