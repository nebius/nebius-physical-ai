"""Materialize and publish one native non-Franka policy-interchange trial."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

from npa.workflows.lerobot_transfer_data import materialize, publish, write_json
from npa.workflows.sim2real.workflow_io import read_json


SCHEMA = "npa.sim2real.policy_interchange.non_franka_execution.v1"


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
        if (
            report.get("schema") != SCHEMA
            or report.get("execution_valid") is not True
            or report.get("decisions") != args.decisions
            or len(report.get("controls") or []) != args.decisions
            or len(report.get("camera_hashes") or []) != args.decisions
        ):
            raise RuntimeError("native non-Franka execution report is incomplete")
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
