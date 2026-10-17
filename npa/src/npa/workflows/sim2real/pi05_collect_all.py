"""Collect size and randomized-scenario holdouts for one multicolor cube family."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.pi05_contract import validate_dense_episode
from npa.workflows.sim2real.interchange_object_asset import (
    MaterializedObjectAsset,
    assert_collection_object_binding,
    materialize_prepared_object,
)
from npa.workflows.sim2real.workflow_io import (
    PI05_COMPONENT_SCHEMA,
    POLICY_INTERCHANGE_COMPONENT_SCHEMA,
    publish_component_record,
    read_json,
)


SPLITS = {
    "train": {
        "seed_offset": 0,
        "object_identity": "isaac-multicolor-cube-048mm",
        "object_scale": (0.8, 0.8, 0.8),
        "scene_id": "pi05-single-body-support-layout-v1",
    },
    "validation": {
        "seed_offset": 10_000,
        "object_identity": "isaac-multicolor-cube-060mm",
        "object_scale": (1.0, 1.0, 1.0),
        "scene_id": "pi05-single-body-support-layout-v1",
    },
    "gold": {
        "seed_offset": 20_000,
        "object_identity": "isaac-multicolor-cube-072mm",
        "object_scale": (1.2, 1.2, 1.2),
        "scene_id": "pi05-single-body-support-layout-v1",
    },
}


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Args: None. Returns: Parser. Raises: None."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root-uri", required=True)
    parser.add_argument("--episodes-per-split", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--object-usd", default="")
    parser.add_argument("--preparation-uri", default="")
    parser.add_argument("--component-root-uri", required=True)
    parser.add_argument(
        "--component-schema",
        choices=(PI05_COMPONENT_SCHEMA, POLICY_INTERCHANGE_COMPONENT_SCHEMA),
        default=PI05_COMPONENT_SCHEMA,
    )
    return parser


def _collector_identity(args: argparse.Namespace, split: str) -> str:
    config = SPLITS[split]
    if args.object_usd or getattr(args, "preparation_uri", ""):
        percent = round(float(config["object_scale"][0]) * 100)
        return f"operator-custom-usd-scale-{percent:03d}"
    return str(config["object_identity"])


def _collector_argv(
    args: argparse.Namespace,
    split: str,
    prepared_object: MaterializedObjectAsset | None = None,
) -> list[str]:
    config = SPLITS[split]
    # The outer NPA interpreter excludes Isaac; the shim selects its verified cache.
    result = [
        "/isaac-sim/python.sh",
        "-m",
        "npa.workflows.sim2real.pi05_isaac",
        "--output-uri",
        f"{args.output_root_uri.rstrip('/')}/{split}/",
        "--episodes",
        str(args.episodes_per_split),
        "--seed",
        str(args.seed + int(config["seed_offset"])),
        "--object-identity",
        _collector_identity(args, split),
        "--object-usd",
        str(prepared_object.local_path) if prepared_object else args.object_usd,
        "--object-scale",
        ",".join(str(value) for value in config["object_scale"]),
        "--scene-id",
        str(config["scene_id"]),
    ]
    if prepared_object:
        for key, value in prepared_object.receipt().items():
            result.extend([f"--object-{key.replace('_', '-')}", value])
    return result


def _run_collector(
    args: argparse.Namespace,
    split: str,
    prepared_object: MaterializedObjectAsset | None = None,
) -> dict[str, Any]:
    process = subprocess.run(_collector_argv(args, split, prepared_object), check=False)
    if process.returncode:
        raise RuntimeError(f"{split} collector failed with exit {process.returncode}")
    # Kit can exit during app.close(); the uploaded report precedes that boundary.
    uri = f"{args.output_root_uri.rstrip('/')}/{split}/collection.json"
    with tempfile.TemporaryDirectory(prefix="npa-pi05-collection-readback-") as path:
        report = read_json(uri, directory=Path(path))
    _validate_collection_report(report, args, split, prepared_object)
    return report


def _validate_collection_report(
    report: dict[str, Any],
    args: argparse.Namespace,
    split: str,
    prepared_object: MaterializedObjectAsset | None = None,
) -> None:
    config = SPLITS[split]
    attempts = report.get("attempts")
    asset = report.get("object_asset") or {}
    if (
        report.get("schema") != "npa.sim2real.pi05.expert_collection.v1"
        or not isinstance(attempts, list)
        or not isinstance(asset, dict)
        or len(attempts) != args.episodes_per_split
        or report.get("attempt_count") != len(attempts)
        or report.get("scene_id") != config["scene_id"]
        or asset.get("identity") != _collector_identity(args, split)
        or asset.get("usd")
        != (
            prepared_object.profile.source_uri
            if prepared_object
            else (args.object_usd or "task_default")
        )
        or asset.get("scale") != list(config["object_scale"])
    ):
        raise RuntimeError(f"{split} collection report does not match its request")
    if prepared_object:
        assert_collection_object_binding(asset, prepared_object.profile)
    successes = _validate_collector_successes(attempts, args, split)
    if successes < 1 or report.get("successful_episode_count") != successes:
        raise RuntimeError(f"{split} collection success count is inconsistent")


def _validate_collector_successes(
    attempts: list[Any], args: argparse.Namespace, split: str
) -> int:
    config = SPLITS[split]
    expected_identity = _collector_identity(args, split)
    successes = 0
    for index, attempt in enumerate(attempts):
        if not isinstance(attempt, dict):
            raise RuntimeError(f"{split} collection attempt is not an object")
        if not attempt.get("success_evidence", {}).get(
            "released_supported_surface_placement"
        ):
            continue
        validate_dense_episode(attempt)
        expected_seed = args.seed + int(config["seed_offset"]) + index
        if (
            attempt.get("scene_seed") != expected_seed
            or attempt.get("object_identity") != expected_identity
        ):
            raise RuntimeError(f"{split} successful episode does not match its request")
        successes += 1
    return successes


def main(argv: list[str] | None = None) -> int:
    """Collect all splits. Args: argv. Returns: Exit status. Raises: RuntimeError."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.episodes_per_split < 1:
        parser.error("--episodes-per-split must be positive")
    if args.preparation_uri and args.object_usd:
        parser.error("--preparation-uri and --object-usd are mutually exclusive")
    with tempfile.TemporaryDirectory(prefix="npa-pi05-prepared-object-") as value:
        prepared_object = None
        if args.preparation_uri:
            root = Path(value)
            preparation = read_json(args.preparation_uri, directory=root)
            prepared_object = materialize_prepared_object(preparation, root / "asset")
        results = {
            split: _run_collector(args, split, prepared_object) for split in SPLITS
        }
    publish_component_record(
        component_schema=args.component_schema,
        root_uri=args.component_root_uri,
        stage=2,
        name=(
            "policy_interchange_franka_expert_collection"
            if args.component_schema == POLICY_INTERCHANGE_COMPONENT_SCHEMA
            else "pi05_expert_collection"
        ),
        tier="WORKS",
        require_gpu=True,
        evidence="Collected physics-only released placements with dense exterior and moving-wrist observations.",
        artifacts={
            split: f"{args.output_root_uri.rstrip('/')}/{split}/collection.json"
            for split in SPLITS
        },
    )
    print(json.dumps(results, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
