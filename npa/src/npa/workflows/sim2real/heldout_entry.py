"""Minimal Isaac sibling entrypoint for held-out eval (no full sim2real CLI)."""

from __future__ import annotations

import argparse
import os

from npa.workflows.sim2real.constants import (
    DEFAULT_ISAAC_TASK,
    DEFAULT_SIM_BACKEND,
    DEFAULT_THRESHOLD,
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="npa.workflows.sim2real.heldout_entry")
    parser.add_argument("--heldout-envs-uri", required=True)
    parser.add_argument("--inner-evidence-uri", required=True)
    parser.add_argument("--output-uri", required=True)
    parser.add_argument("--threshold", type=float, default=DEFAULT_THRESHOLD)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--scene-spec-uri", default="")
    parser.add_argument("--assets-uri", default="")
    parser.add_argument("--byo-mesh-uri", default="")
    parser.add_argument("--robot-spec-uri", default="")
    parser.add_argument("--robot-source", default="")
    parser.add_argument("--robot-preset", default="")
    parser.add_argument(
        "--sim-backend",
        default=os.environ.get("NPA_SIM2REAL_SIM_BACKEND", DEFAULT_SIM_BACKEND),
    )
    parser.add_argument(
        "--isaac-task",
        default=os.environ.get("NPA_SIM2REAL_ISAAC_TASK", DEFAULT_ISAAC_TASK),
    )
    args = parser.parse_args()
    from npa.workflows.sim2real.engine import run_heldout_eval_component_from_s3

    run_heldout_eval_component_from_s3(
        heldout_envs_uri=args.heldout_envs_uri,
        inner_evidence_uri=args.inner_evidence_uri,
        output_uri=args.output_uri,
        threshold=args.threshold,
        limit=args.limit,
        scene_spec_uri=args.scene_spec_uri,
        assets_uri=args.assets_uri,
        byo_mesh_uri=args.byo_mesh_uri,
        robot_spec_uri=args.robot_spec_uri,
        robot_source=args.robot_source,
        robot_preset=args.robot_preset,
        sim_backend=args.sim_backend,
        isaac_task=args.isaac_task,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
