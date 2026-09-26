"""Bind prepared scene artifacts to submitted source and independent simulator/dataset checks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import subprocess
import time

import numpy as np

from matrix import file_digest, load_matrix
from media import verify_video
from replay import replay_episode


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _inventory(root, manifest):
    _require(not any(path.is_symlink() for path in root.rglob("*")), "linked artifact")
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    _require(actual == set(manifest) | {"report.json"}, "artifact inventory differs")
    for relative, expected in manifest.items():
        path = root / relative
        _require(
            not path.is_symlink() and path.resolve().is_relative_to(root.resolve()),
            "artifact escapes output",
        )
        _require(path.stat().st_size == expected["bytes"], "artifact size differs")
        _require(file_digest(path) == expected["sha256"], "artifact hash differs")


def _preview(episode):
    cameras = [
        np.load(episode / f"obs_{key}.npy", mmap_mode="r", allow_pickle=False)
        for key in ("workspace", "wrist")
    ]
    return verify_video(episode / "preview.mp4", cameras)


def _record_identity(row, case, index):
    _require(
        row["id"] == case["id"] and row["scene"] == case["scene"],
        "scene provenance differs",
    )
    _require(
        row["simulation_seed"] == case["seed"]
        and row["controller"] == case["controller"],
        "reset/controller differs",
    )
    relative = f"episodes/episode_{index:04d}"
    _require(row["episode_path"] == relative, "episode path differs")
    scene, simulation = case["scene"], row["simulation"]
    task = f"Pick up the {scene['object_color']} cube and place it on the {scene['target_color']} target, then release it."
    _require(simulation["task"] == task, "scene task label differs")
    _require(
        simulation["frames"] == 185
        and simulation["fps"] == 25
        and simulation["environment"] == "FetchPickAndPlace-v4",
        "simulation contract differs",
    )
    return relative


def _metrics(claimed, observed):
    for name in ("final_distance_m", "lift_height_m", "settled_max_speed_m_s"):
        value = claimed[name]
        _require(
            type(value) in (int, float)
            and math.isfinite(value)
            and abs(value - observed[name]) <= 1e-6,
            "simulation metric differs: " + name,
        )
    _require(
        type(claimed["bilateral_contact_frames"]) is int
        and claimed["bilateral_contact_frames"] == observed["bilateral_contact_frames"],
        "simulation contact count differs",
    )
    _require(claimed["judge"] == observed["judge"], "simulation judge differs")


def _acceptance(row, observed, accepted):
    _metrics(row["simulation"], observed)
    _require(
        row["status"] == ("accepted" if observed["accepted"] else "rejected"),
        "claimed acceptance differs",
    )
    _require(
        row["simulation"]["accepted"] == observed["accepted"],
        "simulation acceptance differs",
    )
    _require(
        row["simulation"]["checks"] == observed["checks"], "simulation checks differ"
    )
    if observed["accepted"]:
        _require(
            row.get("dataset_episode_index") == len(accepted),
            "accepted indices are not contiguous",
        )
        accepted.append(row)
    else:
        _require("dataset_episode_index" not in row, "rejected episode entered dataset")


def _records(root, matrix):
    rows = [
        json.loads(line)
        for line in (root / "provenance.jsonl").read_text().splitlines()
    ]
    _require(len(rows) == len(matrix["cases"]), "case count differs")
    accepted, outcomes = [], []
    for index, (row, case) in enumerate(zip(rows, matrix["cases"], strict=True)):
        relative = _record_identity(row, case, index)
        observed = replay_episode(root / relative, case)
        _acceptance(row, observed, accepted)
        outcomes.append(
            {
                "id": case["id"],
                **observed,
                "decoded_preview_frames": _preview(root / relative),
            }
        )
    return accepted, outcomes


def _native_reader(root, accepted, native_python, output):
    if not accepted:
        _require(not (root / "dataset").exists(), "rejected cases produced a dataset")
        return {"status": "passed", "native_episodes": 0, "native_frames": 0}
    expected = output / "expected-episodes.json"
    expected.write_text(json.dumps(accepted, indent=2) + "\n")
    with (output / "native-reader.log").open("x") as log:
        subprocess.run(
            [
                str(native_python),
                str(Path(__file__).with_name("native_dataset.py")),
                "--input",
                str(root),
                "--expected",
                str(expected),
                "--output",
                str(output / "native-reader.json"),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
        )
    return json.loads((output / "native-reader.json").read_text())


def _verify(root, matrix_path, expected_source, native_python, output):
    matrix, digest = load_matrix(matrix_path)
    report = json.loads((root / "report.json").read_text())
    _require(
        report["schema"] == "npa.robot-workflow.result.v1"
        and report["status"] == "completed",
        "incomplete workflow",
    )
    _require(
        report["matrix_sha256"] == digest
        and report["source_sha256"] == expected_source,
        "matrix/source binding differs",
    )
    _inventory(root, report["artifacts"])
    accepted, outcomes = _records(root, matrix)
    _require(report["case_count"] == len(outcomes), "case total differs")
    _require(report["accepted_count"] == len(accepted), "accepted total differs")
    _require(
        report["rejected_count"] == len(outcomes) - len(accepted),
        "rejected total differs",
    )
    _require(
        report["total_frames"] == 185 * len(accepted) and report["fps"] == 25,
        "training frame total differs",
    )
    return {
        "status": "passed",
        "matrix_sha256": digest,
        "source_sha256": expected_source,
        "outcomes": outcomes,
        "native_dataset": _native_reader(root, accepted, native_python, output),
    }


def verify_workflow(
    root: Path,
    matrix_path: Path,
    expected_source: dict,
    native_python: Path,
    output: Path,
) -> dict:
    """Verify current-source results using native replay and an isolated LeRobot reader.

    Args: root: Completed run. matrix_path: Frozen input. expected_source: Operator hashes.
        native_python: LeRobot interpreter. output: New verification evidence directory.
    Returns: Complete acceptance receipt, only after every independent check succeeds.
    Raises: ValueError, AssertionError, OSError, CalledProcessError: Verification fails.
    """
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    result = {"status": "interrupted"}
    try:
        result = _verify(root, matrix_path, expected_source, native_python, output)
    except (Exception, KeyboardInterrupt) as error:
        result = {
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
        }
        raise
    finally:
        result["seconds"] = time.monotonic() - started
        (output / "result.json").write_text(
            json.dumps(result, indent=2, allow_nan=False) + "\n"
        )
    return result


def main() -> None:
    """Verify existing local artifacts without model calls or cloud submissions.

    Args: None; command-line paths select immutable inputs and new evidence output.
    Returns: None; prints the final verification receipt.
    Raises: ValueError, AssertionError, OSError: Any verification boundary fails.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "matrix", "expected-source", "native-python", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(
        json.dumps(
            verify_workflow(
                args.input,
                args.matrix,
                json.loads(args.expected_source.read_text()),
                args.native_python,
                args.output,
            )
        )
    )


if __name__ == "__main__":
    main()
