"""Materialize and publish one native non-Franka policy-interchange trial."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile

from npa.workflows.lerobot_transfer_data import materialize, publish, write_json
from npa.workflows.sim2real.workflow_io import read_json
from npa.workflows.sim2real.workflow_io import (
    POLICY_INTERCHANGE_COMPONENT_SCHEMA,
    publish_component_record,
)
from npa.workflows.sim2real import isaac_robot_asset


SCHEMA = "npa.sim2real.policy_interchange.non_franka_execution.v1"
COHORT_SCHEMA = "npa.sim2real.policy_interchange.non_franka_execution_cohort.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _materialized_inputs(
    args: argparse.Namespace, work: Path
) -> tuple[Path, dict, Path, Path]:
    """Materialize immutable inputs and write child-process JSON bindings."""

    prepared = materialize(args.prepared_robot_uri, work / "prepared-robot")
    preparation = read_json(args.preparation_uri, directory=work / "preparation")
    service = read_json(args.service_receipt_uri, directory=work / "service")
    preparation_path = work / "preparation.json"
    service_path = work / "service.json"
    write_json(preparation_path, preparation)
    write_json(service_path, service)
    return prepared, preparation, preparation_path, service_path


def _fetch_robot(preparation: dict, work: Path) -> tuple[dict, Path, Path]:
    """Fetch the exact resolved robot USD declared by preparation."""

    asset = preparation.get("robot_asset")
    spec = asset.get("robot_spec") if isinstance(asset, dict) else None
    if not isinstance(spec, dict):
        raise RuntimeError("preparation lacks the exact resolved RobotSpec")
    robot_work = work / "robot-asset"
    os.environ["NPA_BYO_ROBOT_SPEC_JSON"] = json.dumps(
        spec, sort_keys=True, separators=(",", ":")
    )
    os.environ["NPA_ROBOT_WORK_DIR"] = str(robot_work)
    manifest = isaac_robot_asset.fetch()
    return (
        manifest,
        robot_work / "resolved" / "manifest.json",
        robot_work / "resolved" / "robot.usd",
    )


def _child_command(
    args: argparse.Namespace,
    prepared: Path,
    preparation: Path,
    service: Path,
    manifest: Path,
    usd: Path,
    output: Path,
    *,
    seed: int | None = None,
) -> list[str]:
    """Build the native Isaac child command from explicit file bindings."""

    return [
        os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh"),
        "-m",
        "npa.workflows.sim2real.interchange_isaac_runtime",
        "--input-path",
        str(prepared),
        "--preparation-path",
        str(preparation),
        "--service-receipt-path",
        str(service),
        "--robot-usd-path",
        str(usd),
        "--robot-asset-manifest-path",
        str(manifest),
        "--output-path",
        str(output),
        "--decisions",
        str(args.decisions),
        "--seed",
        str(args.seed if seed is None else seed),
    ]


def _cohort_report(reports: list[dict], *, seed: int) -> dict:
    """Summarize a split-disjoint native cohort without hiding failures."""

    identity_fields = (
        "robot_identity",
        "object_identity",
        "task_identity",
        "checkpoint_sha256",
        "normalization_identity",
    )
    if not reports or any(
        report.get("seed") != seed + index
        or report.get("execution_valid") is not True
        or any(
            report.get(field) != reports[0].get(field) or not report.get(field)
            for field in identity_fields
        )
        for index, report in enumerate(reports)
    ):
        raise RuntimeError("native object cohort identities or seeds are inconsistent")

    return {
        "schema": COHORT_SCHEMA,
        "execution_valid": all(
            report.get("execution_valid") is True for report in reports
        ),
        "task_success_measured": True,
        "episode_count": len(reports),
        "success_count": sum(report.get("task_success") is True for report in reports),
        "success_rate": (
            sum(report.get("task_success") is True for report in reports) / len(reports)
            if reports
            else 0.0
        ),
        "base_seed": seed,
        "episodes": [
            {
                "episode_index": index,
                "seed": report.get("seed"),
                "execution_valid": report.get("execution_valid") is True,
                "task_success": report.get("task_success") is True,
                "ordered_event_steps": report.get("ordered_event_steps") or {},
                "robot_identity": report.get("robot_identity"),
                "object_identity": report.get("object_identity"),
                "task_identity": report.get("task_identity"),
                "checkpoint_sha256": report.get("checkpoint_sha256"),
            }
            for index, report in enumerate(reports)
        ],
    }


def _run_child(command: list[str], output: Path, log: Path) -> dict:
    """Execute native Isaac and return its complete report."""

    env = dict(os.environ)
    env["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(
            command, env=env, stdout=stream, stderr=subprocess.STDOUT, check=False
        )
    if result.returncode or not (output / "report.json").is_file():
        excerpt = log.read_text(encoding="utf-8", errors="replace")[-12_000:]
        failure_path = output / "runtime_failure.json"
        failure = (
            failure_path.read_text(encoding="utf-8", errors="replace")
            if failure_path.is_file()
            else "native runtime emitted no structured failure receipt"
        )
        raise RuntimeError(
            "native non-Franka policy trial failed with "
            f"{result.returncode}; failure={failure}\n{excerpt}"
        )
    return json.loads((output / "report.json").read_text(encoding="utf-8"))


def _validate_report_shape(report: dict, decisions: int) -> dict:
    """Validate required native policy and teacher evidence collections."""

    teacher = report.get("teacher") or {}
    if (
        report.get("schema") != SCHEMA
        or report.get("execution_valid") is not True
        or report.get("decisions") != decisions
        or len(report.get("controls") or []) != decisions
        or len(report.get("camera_hashes") or []) != decisions
        or report.get("all_camera_decisions_hashed") is not True
        or not report.get("visual_evidence")
        or teacher.get("schema") != "npa.sim2real.policy_interchange.native_teacher.v1"
        or teacher.get("execution_valid") is not True
        or not teacher.get("controls")
        or not teacher.get("visual_evidence")
        or len(teacher.get("camera_hashes") or ()) != len(teacher.get("controls") or ())
        or len(teacher.get("camera_clocks") or ()) != len(teacher.get("controls") or ())
        or len(report.get("observation_clocks") or ()) != decisions
    ):
        raise RuntimeError("native non-Franka execution report is incomplete")
    return teacher


def _validate_camera_clocks(report: dict, teacher: dict) -> None:
    """Prove teacher and policy images causally precede applied controls."""

    for clock in teacher["camera_clocks"]:
        poses = clock.get("camera_world_poses")
        if not isinstance(poses, dict) or set(poses) != set(
            clock["capture_timestamps_s"]
        ):
            raise RuntimeError("native teacher camera poses are incomplete")
        limit = float(clock["command_application_time_s"]) + 1.0e-9
        if any(
            float(value) > limit for value in clock["capture_timestamps_s"].values()
        ):
            raise RuntimeError("native teacher camera/control clock is noncausal")
    for clock, control in zip(
        report["observation_clocks"], report["controls"], strict=True
    ):
        state_time = float(clock["state_timestamp_s"])
        command_time = float(control["command_application_time_s"])
        poses = clock.get("camera_world_poses")
        if (
            not isinstance(poses, dict)
            or set(poses) != set(clock["capture_timestamps_s"])
            or any(
                float(value) > state_time + 1.0e-9
                for value in clock["capture_timestamps_s"].values()
            )
            or state_time > command_time + 1.0e-9
            or control.get("terminated") is not False
            or control.get("truncated") is not False
        ):
            raise RuntimeError("native policy camera/control clock is noncausal")


def _validate_outputs(report: dict, teacher: dict, output: Path, robot: dict) -> None:
    """Validate visual bytes and exact fetched-USD consumption."""

    for visual in [*teacher["visual_evidence"], *report["visual_evidence"]]:
        path = output / str(visual.get("path") or "")
        if (
            not path.is_file()
            or path.stat().st_size != visual.get("bytes")
            or _sha256(path) != visual.get("sha256")
        ):
            raise RuntimeError("native non-Franka visual evidence is incomplete")
    if report.get("robot_asset", {}).get("usd_sha256") != robot.get("usd_sha256"):
        raise RuntimeError("native execution did not consume the fetched robot USD")


def _publish_component(args: argparse.Namespace) -> None:
    """Publish the standard component record after native evidence is durable."""

    publish_component_record(
        component_schema=POLICY_INTERCHANGE_COMPONENT_SCHEMA,
        root_uri=args.component_root_uri,
        stage=args.component_stage,
        name=f"policy_interchange_non_franka_{args.policy_role}_execution",
        tier="WORKS",
        evidence=(
            "Executed the geometry-aware native teacher and applied real typed "
            f"{args.policy_role} policy actions to the sealed non-Franka task."
        ),
        artifacts={"execution": args.output_uri},
    )


def run(args: argparse.Namespace) -> dict:
    """Run, validate, and publish a native non-Franka policy cohort."""

    with tempfile.TemporaryDirectory(prefix="npa-policy-interchange-isaac-") as value:
        work = Path(value)
        prepared, preparation, prep_path, service_path = _materialized_inputs(
            args, work
        )
        robot, manifest_path, usd_path = _fetch_robot(preparation, work)
        published = work / "published"
        if args.episodes > 1:
            published.mkdir()
        reports = []
        for episode_index in range(args.episodes):
            if args.episodes == 1:
                output = work / "output"
                log = work / "isaac-runtime.log"
            else:
                episode = published / "episodes" / f"episode-{episode_index:03d}"
                output = episode / "native"
                log = episode / "isaac-runtime.log"
                episode.mkdir(parents=True)
            command = _child_command(
                args,
                prepared,
                prep_path,
                service_path,
                manifest_path,
                usd_path,
                output,
                seed=args.seed + episode_index,
            )
            report = _run_child(command, output, log)
            teacher = _validate_report_shape(report, args.decisions)
            _validate_camera_clocks(report, teacher)
            _validate_outputs(report, teacher, output, robot)
            if args.episodes == 1:
                (output / "isaac-runtime.log").write_bytes(log.read_bytes())
            reports.append(report)
        result = (
            reports[0]
            if args.episodes == 1
            else _cohort_report(reports, seed=args.seed)
        )
        if args.episodes > 1:
            write_json(published / "report.json", result)
        publish(output if args.episodes == 1 else published, args.output_uri)
    _publish_component(args)
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the non-Franka Isaac entrypoint argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-robot-uri", required=True)
    parser.add_argument("--preparation-uri", required=True)
    parser.add_argument("--service-receipt-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--component-root-uri", required=True)
    parser.add_argument("--component-stage", type=int, choices=(6, 11), required=True)
    parser.add_argument("--policy-role", choices=("base", "adapted"), required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--episodes", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Materialize inputs, execute Isaac, and publish the report."""

    args = build_parser().parse_args(argv)
    if args.decisions < 1 or args.episodes < 1 or args.seed < 0:
        raise ValueError("decisions and episodes must be positive and seed nonnegative")
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
