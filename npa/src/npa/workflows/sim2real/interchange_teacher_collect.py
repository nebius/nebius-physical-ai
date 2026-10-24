"""Materialize inputs and collect native model-neutral teacher episodes in Isaac."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from npa.workflows.lerobot_transfer_data import materialize, publish, write_json
from npa.workflows.sim2real import isaac_robot_asset
from npa.workflows.sim2real.workflow_io import read_json
from npa.workflows.sim2real.workflow_io import (
    POLICY_INTERCHANGE_COMPONENT_SCHEMA,
    publish_component_record,
)


SCHEMA = "npa.sim2real.policy_interchange.teacher_collection.v1"


def _inputs(args: argparse.Namespace, work: Path) -> tuple[Path, dict, Path]:
    """Materialize the prepared bundle and child preparation JSON."""

    prepared = materialize(args.prepared_robot_uri, work / "prepared-robot")
    preparation = read_json(args.preparation_uri, directory=work / "preparation")
    preparation_path = work / "preparation.json"
    write_json(preparation_path, preparation)
    return prepared, preparation, preparation_path


def _fetch_robot(preparation: dict, work: Path) -> Path:
    """Fetch the resolved robot asset for the native child."""

    asset = preparation.get("robot_asset")
    spec = asset.get("robot_spec") if isinstance(asset, dict) else None
    if not isinstance(spec, dict):
        raise RuntimeError("preparation lacks the exact resolved RobotSpec")
    robot_work = work / "robot-asset"
    os.environ["NPA_BYO_ROBOT_SPEC_JSON"] = json.dumps(
        spec, sort_keys=True, separators=(",", ":")
    )
    os.environ["NPA_ROBOT_WORK_DIR"] = str(robot_work)
    isaac_robot_asset.fetch()
    return robot_work


def _command(
    args: argparse.Namespace,
    prepared: Path,
    preparation: Path,
    robot: Path,
    output: Path,
) -> list[str]:
    """Build the native teacher command from explicit materialized paths."""

    return [
        os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh"),
        "-m",
        "npa.workflows.sim2real.interchange_teacher_runtime",
        "--input-path",
        str(prepared),
        "--preparation-path",
        str(preparation),
        "--robot-usd-path",
        str(robot / "resolved" / "robot.usd"),
        "--robot-asset-manifest-path",
        str(robot / "resolved" / "manifest.json"),
        "--output-path",
        str(output),
        "--train-episodes",
        str(args.train_episodes),
        "--validation-episodes",
        str(args.validation_episodes),
        "--gold-episodes",
        str(args.gold_episodes),
        "--seed",
        str(args.seed),
    ]


def _execute(command: list[str], output: Path, log: Path) -> dict:
    """Execute the native collector and load its report."""

    environment = dict(os.environ)
    environment["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command,
            env=environment,
            stdout=stream,
            stderr=subprocess.STDOUT,
            check=False,
        )
    report_path = output / "report.json"
    if result.returncode or not report_path.is_file():
        excerpt = log.read_text(encoding="utf-8", errors="replace")[-12_000:]
        raise RuntimeError(
            f"native teacher collection failed with {result.returncode}\n{excerpt}"
        )
    return json.loads(report_path.read_text(encoding="utf-8"))


def _validate(report: dict, args: argparse.Namespace) -> None:
    """Validate collection completeness before publication."""

    expected = args.train_episodes + args.validation_episodes + args.gold_episodes
    if (
        report.get("schema") != SCHEMA
        or report.get("status") != "passed"
        or report.get("all_teacher_episodes_successful") is not True
        or len(report.get("episodes") or ()) != expected
        or report.get("privileged_object_state_policy_input") is not False
    ):
        raise RuntimeError("native teacher collection report is incomplete")


def _publish_component(args: argparse.Namespace) -> None:
    """Publish the standard teacher collection ComponentRecord."""

    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=2,
        name="policy_interchange_non_franka_teacher_collection",
        tier="WORKS",
        evidence=(
            "Collected successful geometry-aware teacher episodes with causal "
            "exterior/wrist observations and measured post-control targets."
        ),
        artifacts={"collection": args.output_uri},
    )


def run(args: argparse.Namespace) -> dict:
    """Run the native collector and publish its complete immutable output tree."""

    with tempfile.TemporaryDirectory(prefix="npa-policy-teacher-") as value:
        work = Path(value)
        prepared, preparation, preparation_path = _inputs(args, work)
        robot = _fetch_robot(preparation, work)
        output, log = work / "output", work / "isaac-teacher.log"
        report = _execute(
            _command(args, prepared, preparation_path, robot, output), output, log
        )
        _validate(report, args)
        (output / "isaac-teacher.log").write_bytes(log.read_bytes())
        publish(output, args.output_uri)
    _publish_component(args)
    return report


def build_parser() -> argparse.ArgumentParser:
    """Build the native teacher collection argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-robot-uri", required=True)
    parser.add_argument("--preparation-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--component-root-uri", required=True)
    parser.add_argument("--train-episodes", type=int, required=True)
    parser.add_argument("--validation-episodes", type=int, required=True)
    parser.add_argument("--gold-episodes", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Collect and publish native embodiment-aware teacher episodes."""

    args = build_parser().parse_args(argv)
    if min(args.train_episodes, args.validation_episodes, args.gold_episodes) < 1:
        raise ValueError("every teacher split requires at least one episode")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
