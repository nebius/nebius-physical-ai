"""Convert and publish the exact robot asset required by an interchange run."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile

from npa.workflows.lerobot_transfer_data import write_json
from npa.workflows.sim2real import isaac_robot_asset
from npa.workflows.sim2real.workflow_io import read_json


SCHEMA = "npa.sim2real.policy_interchange.robot_asset_prepare.v1"


def _launch_and_convert() -> tuple[dict, object]:
    """Keep Kit alive until the converted manifest and stage receipt are durable."""

    from isaaclab.app import AppLauncher

    app = AppLauncher(
        headless=True,
        kit_args=os.environ.get(
            "NPA_ISAAC_KIT_ARGS", "--portable-root /tmp/npa-isaac-kit"
        ),
    ).app
    return isaac_robot_asset.prepare_with_running_app(), app


def run(args: argparse.Namespace) -> dict:
    """Convert the preparation's source bundle and publish exact asset evidence."""

    with tempfile.TemporaryDirectory(prefix="npa-policy-robot-asset-") as value:
        work = Path(value)
        preparation = read_json(args.preparation_uri, directory=work / "preparation")
        asset = preparation.get("robot_asset")
        spec = asset.get("robot_spec") if isinstance(asset, dict) else None
        if not isinstance(spec, dict):
            raise RuntimeError("preparation lacks the exact resolved RobotSpec")
        os.environ["NPA_BYO_ROBOT_SPEC_JSON"] = json.dumps(
            spec, sort_keys=True, separators=(",", ":")
        )
        os.environ["NPA_ROBOT_WORK_DIR"] = str(work / "robot-asset")
        manifest, app = _launch_and_convert()
        receipt = {
            "schema": SCHEMA,
            "status": "passed",
            "run_id": args.run_id,
            "preparation_uri": args.preparation_uri,
            "resolved_usd_uri": spec["resolved_usd_uri"],
            "resolved_manifest_uri": spec["resolved_manifest_uri"],
            "embodiment_digest": spec["embodiment_digest"],
            "conversion_recipe": spec["conversion_recipe"],
            "conversion_recipe_sha256": spec["conversion_recipe_sha256"],
            "usd_sha256": manifest["usd_sha256"],
            "asset_tree_sha256": manifest["asset_tree_sha256"],
            "complete_asset_tree_published": True,
        }
        write_json(args.output_uri, receipt, directory=work / "receipt")
        # Isaac Kit may terminate the interpreter from close(). Both the complete
        # converted tree and this declared workflow output are durable first.
        app.close()
        return receipt


def build_parser() -> argparse.ArgumentParser:
    """Build the robot-asset preparation argument parser."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--preparation-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Convert and publish one content-addressed robot asset."""

    result = run(build_parser().parse_args(argv))
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
