"""Validate native teacher episodes and export train-only LeRobot data."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from typing import Any

from npa.adapter.sim_to_lerobot import convert
from npa.workflows.lerobot_transfer_data import materialize, publish, write_json
from npa.workflows.sim2real.pi05_data import install_policy_adapter
from npa.workflows.sim2real.workflow_io import (
    POLICY_INTERCHANGE_COMPONENT_SCHEMA,
    publish_component_record,
    read_json,
)


SCHEMA = "npa.sim2real.policy_interchange.lerobot_dataset.v1"
COLLECTION_SCHEMA = "npa.sim2real.policy_interchange.teacher_collection.v1"
EPISODE_FILES = (
    "obs_workspace.npy",
    "obs_wrist.npy",
    "state.npy",
    "actions.npy",
    "timestamps.npy",
    "metadata.json",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_episodes(root: Path, report: dict[str, Any]) -> dict[str, list[Path]]:
    if (
        report.get("schema") != COLLECTION_SCHEMA
        or report.get("status") != "passed"
        or report.get("all_teacher_episodes_successful") is not True
    ):
        raise RuntimeError("teacher collection is not a successful sealed input")
    splits = {name: [] for name in ("train", "validation", "gold")}
    for row in report.get("episodes") or ():
        if not isinstance(row, dict) or row.get("task_success") is not True:
            raise RuntimeError("teacher collection contains an unsuccessful episode")
        split = str(row.get("split") or "")
        index = row.get("episode_index")
        if split not in splits or not isinstance(index, int) or index < 0:
            raise RuntimeError("teacher collection episode identity is invalid")
        episode = root / "splits" / split / f"episode_{index:04d}"
        expected = row.get("artifacts")
        if not isinstance(expected, dict) or set(expected) != set(EPISODE_FILES):
            raise RuntimeError("teacher episode artifact manifest is incomplete")
        for name in EPISODE_FILES:
            path = episode / name
            if not path.is_file() or _sha256(path) != expected[name]:
                raise RuntimeError(
                    "teacher episode bytes differ from collection receipt"
                )
        splits[split].append(episode)
    counts = report.get("counts") or {}
    if any(len(splits[name]) != counts.get(name) for name in splits):
        raise RuntimeError("teacher collection split counts differ from its receipt")
    return splits


def run(args: argparse.Namespace) -> dict[str, Any]:
    """Export only training episodes while retaining immutable held-out lineage."""

    with tempfile.TemporaryDirectory(prefix="npa-policy-teacher-export-") as value:
        work = Path(value)
        collection = materialize(args.collection_uri, work / "collection")
        collection_report = json.loads(
            (collection / "report.json").read_text(encoding="utf-8")
        )
        splits = _validated_episodes(collection, collection_report)
        adapter = read_json(args.policy_adapter_uri, directory=work / "adapter")
        adapter_path = work / "policy-adapter.json"
        write_json(adapter_path, adapter)
        lerobot_input = work / "lerobot-input"
        for index, source in enumerate(splits["train"]):
            destination = lerobot_input / f"episode_{index:06d}"
            destination.mkdir(parents=True)
            for name in EPISODE_FILES[:-1]:
                shutil.copy2(source / name, destination / name)
        output = work / "output"
        dataset = convert(
            lerobot_input,
            output / "lerobot-v3",
            fps=args.fps,
            robot_type=str(collection_report["robot_identity"]),
            task="Pick up the rigid object and place it inside the target.",
        )
        install_policy_adapter(adapter_path, dataset)
        info = json.loads((dataset / "meta" / "info.json").read_text(encoding="utf-8"))
        if info.get("total_episodes") != len(splits["train"]):
            raise RuntimeError("LeRobot readback episode count mismatch")
        result = {
            "schema": SCHEMA,
            "status": "passed",
            "robot_identity": collection_report["robot_identity"],
            "object_identity": collection_report["object_identity"],
            "task_identity": collection_report["task_identity"],
            "training_episodes": len(splits["train"]),
            "validation_episodes_held_out": len(splits["validation"]),
            "gold_episodes_held_out": len(splits["gold"]),
            "lerobot_frames": info.get("total_frames"),
            "timestamp_source": info.get("timestamp_source"),
            "policy_adapter": "lerobot-v3/meta/npa_policy_adapter.json",
            "normalization_source": "training_split_only",
            "heldout_episode_artifacts": {
                split: [
                    {
                        "episode_index": row["episode_index"],
                        "scenario": row["scenario"],
                        "artifacts": row["artifacts"],
                    }
                    for row in collection_report["episodes"]
                    if row["split"] == split
                ]
                for split in ("validation", "gold")
            },
        }
        write_json(output / "dataset-report.json", result)
        publish(output, args.output_uri)
    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=3,
        name="policy_interchange_lerobot_dataset",
        tier="WORKS",
        evidence=(
            "Exported native dynamic-dimension teacher data with training-only "
            "normalization and sealed held-out lineage."
        ),
        artifacts={"dataset": args.output_uri},
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-uri", required=True)
    parser.add_argument("--policy-adapter-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--component-root-uri", required=True)
    parser.add_argument("--fps", type=int, default=15)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fps < 1:
        raise ValueError("fps must be positive")
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
