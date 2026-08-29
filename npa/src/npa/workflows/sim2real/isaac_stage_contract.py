"""Shared canonical Isaac stage environment and embodiment evidence loading."""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.workflows.sim2real.robot_contract import (
    assert_embodiment_evidence,
    isaac_environment,
)
from npa.workflows.sim2real.workflow_io import read_json, source_sha


def _root_parts(root_uri: str, run_id: str) -> tuple[str, str]:
    parsed = urlparse(root_uri)
    path = parsed.path.lstrip("/").rstrip("/")
    suffix = "/" + run_id
    if not path.endswith(suffix):
        raise ValueError("run root must end with the exact workflow run ID")
    return parsed.netloc, path[: -len(suffix)]


def common_environment(args: Any, *, split_uri: str) -> dict[str, Any]:
    """Build the stock environment, adding custom values only for a BYO contract."""

    root = str(args.root_uri).rstrip("/")
    bucket, base_prefix = _root_parts(root, args.run_id)
    with tempfile.TemporaryDirectory(prefix="npa-s2r-isaac-contract-") as raw:
        input_dir = Path(raw)
        task_contract = read_json(
            f"{root}/stage_02_assets/task-contract.json", directory=input_dir
        )
        env = {
            "NPA_SIM2REAL_INLINE_TASK": "1",
            "NPA_SIM2REAL_RUN_ID": args.run_id,
            "NPA_SIM2REAL_BUCKET": bucket,
            "NPA_SIM2REAL_S3_BUCKET": bucket,
            "NPA_SIM2REAL_PREFIX": base_prefix,
            "NPA_SIM2REAL_ISAAC_IMAGE": os.environ["NPA_TASK_IMAGE"],
            "ISAAC_IMAGE": os.environ["NPA_TASK_IMAGE"],
            "NPA_SIM2REAL_SOURCE_SHA": source_sha(),
            "NPA_SIM2REAL_ISAAC_TASK": args.task_id,
            "NPA_BYO_ISAAC_TASK": args.task_id,
            "NPA_SIM2REAL_TASK_CONTRACT_DIGEST": task_contract[
                "task_contract_digest"
            ],
            "NPA_SIM2REAL_TRAIN_ENVS_URI": split_uri,
            "NPA_SIM2REAL_CAMERA_VIEWS": "primary,side,overhead",
            "NPA_SIM2REAL_CAPTURE_FPS": args.capture_fps,
            "NPA_SIM2REAL_CAPTURE_WIDTH": args.capture_width,
            "NPA_SIM2REAL_CAPTURE_HEIGHT": args.capture_height,
            "NPA_SIM2REAL_PNG_COMPRESS_LEVEL": args.png_compress_level,
        }
        robot_uri = f"{root}/stage_02_assets/consumed_robot_spec.json"
        robot = read_json(robot_uri, directory=input_dir)
        env.update(isaac_environment(robot, contract_uri=robot_uri, stage=args.stage))
    return env


def verify_evidence(
    *, root: str, payload: dict[str, Any], stage: str
) -> dict[str, Any]:
    """Load Stage 2's contract and verify one downstream stage's evidence."""

    with tempfile.TemporaryDirectory(prefix="npa-s2r-isaac-contract-") as raw:
        contract = read_json(
            f"{root}/stage_02_assets/consumed_robot_spec.json",
            directory=Path(raw),
        )
        return assert_embodiment_evidence(contract, payload=payload, stage=stage)
