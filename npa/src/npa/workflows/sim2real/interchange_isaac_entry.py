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
from npa.workflows.sim2real import isaac_robot_asset


SCHEMA = "npa.sim2real.policy_interchange.non_franka_execution.v1"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def run(args: argparse.Namespace) -> dict:
    with tempfile.TemporaryDirectory(prefix="npa-policy-interchange-isaac-") as value:
        work = Path(value)
        prepared = materialize(args.prepared_robot_uri, work / "prepared-robot")
        preparation = read_json(args.preparation_uri, directory=work / "preparation")
        service = read_json(args.service_receipt_uri, directory=work / "service")
        preparation_path = work / "preparation.json"
        service_path = work / "service.json"
        write_json(preparation_path, preparation)
        write_json(service_path, service)
        asset = preparation.get("robot_asset")
        spec = asset.get("robot_spec") if isinstance(asset, dict) else None
        if not isinstance(spec, dict):
            raise RuntimeError("preparation lacks the exact resolved RobotSpec")
        robot_work = work / "robot-asset"
        os.environ["NPA_BYO_ROBOT_SPEC_JSON"] = json.dumps(
            spec, sort_keys=True, separators=(",", ":")
        )
        os.environ["NPA_ROBOT_WORK_DIR"] = str(robot_work)
        robot_manifest = isaac_robot_asset.fetch()
        robot_manifest_path = robot_work / "resolved" / "manifest.json"
        robot_usd_path = robot_work / "resolved" / "robot.usd"
        output = work / "output"
        log = work / "isaac-runtime.log"
        child = [
            os.environ.get("ISAAC_LAB_PYTHON", "/isaac-sim/python.sh"),
            "-m",
            "npa.workflows.sim2real.interchange_isaac_runtime",
            "--input-path",
            str(prepared),
            "--preparation-path",
            str(preparation_path),
            "--service-receipt-path",
            str(service_path),
            "--robot-usd-path",
            str(robot_usd_path),
            "--robot-asset-manifest-path",
            str(robot_manifest_path),
            "--output-path",
            str(output),
            "--decisions",
            str(args.decisions),
            "--seed",
            str(args.seed),
        ]
        env = dict(os.environ)
        env["OMNI_TELEMETRY_DISABLE_ANONYMOUS_DATA"] = "1"
        with log.open("w", encoding="utf-8") as stream:
            result = subprocess.run(
                child,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                check=False,
            )
        if result.returncode or not (output / "report.json").is_file():
            excerpt = log.read_text(encoding="utf-8", errors="replace")[-12_000:]
            raise RuntimeError(
                f"native non-Franka policy trial failed with {result.returncode}\n{excerpt}"
            )
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        teacher = report.get("teacher") or {}
        if (
            report.get("schema") != SCHEMA
            or report.get("execution_valid") is not True
            or report.get("decisions") != args.decisions
            or len(report.get("controls") or []) != args.decisions
            or len(report.get("camera_hashes") or []) != args.decisions
            or report.get("all_camera_decisions_hashed") is not True
            or not report.get("visual_evidence")
            or teacher.get("schema")
            != "npa.sim2real.policy_interchange.native_teacher.v1"
            or teacher.get("execution_valid") is not True
            or not teacher.get("controls")
            or not teacher.get("visual_evidence")
            or len(teacher.get("camera_hashes") or ())
            != len(teacher.get("controls") or ())
            or len(teacher.get("camera_clocks") or ())
            != len(teacher.get("controls") or ())
            or len(report.get("observation_clocks") or ()) != args.decisions
        ):
            raise RuntimeError("native non-Franka execution report is incomplete")
        for clock in teacher["camera_clocks"]:
            if any(
                float(value) > float(clock["command_application_time_s"]) + 1.0e-9
                for value in clock["capture_timestamps_s"].values()
            ):
                raise RuntimeError("native teacher camera/control clock is noncausal")
        for clock, control in zip(
            report["observation_clocks"], report["controls"], strict=True
        ):
            state_time = float(clock["state_timestamp_s"])
            command_time = float(control["command_application_time_s"])
            if (
                any(
                    float(value) > state_time + 1.0e-9
                    for value in clock["capture_timestamps_s"].values()
                )
                or state_time > command_time + 1.0e-9
                or control.get("terminated") is not False
                or control.get("truncated") is not False
            ):
                raise RuntimeError("native policy camera/control clock is noncausal")
        for visual in [
            *teacher["visual_evidence"],
            *report["visual_evidence"],
        ]:
            path = output / str(visual.get("path") or "")
            if (
                not path.is_file()
                or path.stat().st_size != visual.get("bytes")
                or _sha256(path) != visual.get("sha256")
            ):
                raise RuntimeError("native non-Franka visual evidence is incomplete")
        if report.get("robot_asset", {}).get("usd_sha256") != robot_manifest.get(
            "usd_sha256"
        ):
            raise RuntimeError("native execution did not consume the fetched robot USD")
        (output / "isaac-runtime.log").write_bytes(log.read_bytes())
        publish(output, args.output_uri)
        return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-robot-uri", required=True)
    parser.add_argument("--preparation-uri", required=True)
    parser.add_argument("--service-receipt-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--decisions", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.decisions < 1 or args.seed < 0:
        raise ValueError("decisions must be positive and seed nonnegative")
    result = run(args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
