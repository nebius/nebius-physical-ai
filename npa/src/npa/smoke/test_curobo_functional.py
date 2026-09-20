"""Real single-pose cuRobo GPU qualification, separate from full benchmark runs."""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re

from pydantic import ValidationError

from npa.workbench.dataset.storage import read_bytes_uri, uri_join, write_bytes_uri
from npa.workbench.curobo.audit import audit_bytes
from npa.workbench.curobo.artifacts import (
    build_rrd,
    canonical,
    decode_rrd,
    read_journal,
)
from npa.workbench.curobo.runner import execute
from npa.workbench.curobo.replay import replay_rows
from npa.workbench.curobo.schemas import PlanManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _publish(uri: str, payload: bytes) -> None:
    write_bytes_uri(uri, payload)
    if hashlib.sha256(read_bytes_uri(uri)).digest() != hashlib.sha256(payload).digest():
        raise RuntimeError("golden evidence upload digest mismatch")


def _require_feasible_distance(audit: dict) -> None:
    distance = audit.get("terminal_goal_distance_m", {}).get("max")
    orientation = audit.get("terminal_goal_orientation_rad", {}).get("max")
    if (
        isinstance(distance, bool)
        or not isinstance(distance, (int, float))
        or not math.isfinite(distance)
        or distance > 0.005
        or isinstance(orientation, bool)
        or not isinstance(orientation, (int, float))
        or not math.isfinite(orientation)
        or orientation > 0.05
    ):
        raise RuntimeError("feasible control terminal pose exceeds tolerance")


def _run_workload(
    root: Path, *, run_id: str, source_commit: str, image_digest: str
) -> dict:
    manifest = {
        "problems": [
            {
                "id": "franka-pose",
                "start": [0, -1.3, 0, -2.5, 0, 1.0, 0],
                "goal_pose": {
                    "position_xyz": [0.5, 0.0, 0.3],
                    "quaternion_wxyz": [1, 0, 0, 0],
                },
                "cuboids": {
                    "table": {
                        "dims": [2, 2, 0.2],
                        "pose": [0, 0, -0.2, 1, 0, 0, 0],
                    }
                },
            },
            {
                "id": "blocked-goal-control",
                "start": [0, -1.3, 0, -2.5, 0, 1.0, 0],
                "goal_pose": {
                    "position_xyz": [0.5, 0.0, 0.3],
                    "quaternion_wxyz": [1, 0, 0, 0],
                },
                "cuboids": {
                    "goal-blocker": {
                        "dims": [0.4, 0.4, 0.4],
                        "pose": [0.5, 0.0, 0.3, 1, 0, 0, 0],
                    }
                },
            },
        ]
    }
    (root / "input.json").write_bytes(canonical(manifest))
    report = execute("plan", manifest, root / "output", run_id=run_id)
    rows = read_journal(root / "output/problems.jsonl")
    statuses = {row["problem_id"]: row["status"] for row in rows}
    if statuses != {
        "franka-pose": "success",
        "blocked-goal-control": "failed",
    }:
        raise RuntimeError("positive or valid-but-infeasible control disagrees")
    report["input_sha256"] = _sha256(root / "input.json")
    report["journal_sha256"] = _sha256(root / "output/problems.jsonl")
    (root / "output/result.json").write_bytes(canonical(report))
    result_bytes = (root / "output/result.json").read_bytes()
    journal_bytes = (root / "output/problems.jsonl").read_bytes()
    audit = audit_bytes(result_bytes, journal_bytes, run_id=run_id)
    replay = replay_rows(rows, report)
    _require_feasible_distance(replay)
    audit["independent_replay"] = replay
    (root / "independent-validation.json").write_bytes(canonical(audit))
    rrd_manifest = build_rrd(
        root / "output/problems.jsonl",
        root / "planning.rrd",
        run_id=run_id,
    )
    rrd_manifest["result_sha256"] = hashlib.sha256(result_bytes).hexdigest()
    rrd_manifest["decode"] = decode_rrd(
        root / "planning.rrd",
        rows=rows,
        run_id=run_id,
        decoded_output=root / "rrd-print.txt",
    )
    (root / "rrd-manifest.json").write_bytes(canonical(rrd_manifest))
    malformed_rejected = False
    try:
        PlanManifest.model_validate(
            {
                "robot": "/unreviewed-absolute-robot.yml",
                "problems": [manifest["problems"][0]],
            }
        )
    except ValidationError as exc:
        if not any(error["loc"] == ("robot",) for error in exc.errors()):
            raise RuntimeError(
                "malformed control failed outside the robot field"
            ) from exc
        malformed_rejected = True
    if not malformed_rejected:
        raise RuntimeError("malformed manifest control was not rejected")
    controls = {
        "schema_version": "npa.curobo.controls.v1",
        "run_id": run_id,
        "valid_but_infeasible": {
            "problem_id": "blocked-goal-control",
            "observed_status": statuses["blocked-goal-control"],
            "accepted_status": "failed",
        },
        "malformed_manifest": {
            "description": "absolute robot configuration path",
            "rejected": malformed_rejected,
        },
    }
    (root / "controls.json").write_bytes(canonical(controls))
    artifacts = {
        str(path.relative_to(root)): {
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }
    artifact_manifest = {
        "schema_version": "npa.curobo.smoke-artifacts.v1",
        "run_id": run_id,
        "source_commit": source_commit,
        "image_digest": image_digest,
        "gpu": report["gpu"],
        "artifacts": artifacts,
        "objective": {
            "success": 1,
            "failed_control": 1,
            "invalid": 0,
            "independent_validation": audit["valid"],
            "independent_replay": replay["valid"],
            "rrd_decode": rrd_manifest["decode"]["print"],
        },
        "limitations": report["limitations"],
    }
    (root / "artifact-manifest.json").write_bytes(canonical(artifact_manifest))
    return {
        "artifact_manifest_sha256": _sha256(root / "artifact-manifest.json"),
    }


def _upload_tree(
    root: Path,
    *,
    output_uri: str,
    run_id: str,
    image_digest: str,
    workload_status: str,
) -> dict:
    uploaded = []
    errors = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == "upload-receipt.json":
            continue
        relative = str(path.relative_to(root))
        payload = path.read_bytes()
        try:
            _publish(uri_join(output_uri, relative), payload)
        except Exception as exc:
            errors.append({"path": relative, "error": type(exc).__name__})
        else:
            uploaded.append(
                {
                    "path": relative,
                    "bytes": len(payload),
                    "sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    upload_receipt = {
        "schema_version": "npa.curobo.smoke-upload.v1",
        "run_id": run_id,
        "image_digest": image_digest,
        "workload_status": workload_status,
        "objects": uploaded,
        "errors": errors,
        "readback_verified": not errors,
    }
    receipt_bytes = canonical(upload_receipt)
    (root / "upload-receipt.json").write_bytes(receipt_bytes)
    try:
        _publish(uri_join(output_uri, "upload-receipt.json"), receipt_bytes)
    except Exception as exc:
        errors.append({"path": "upload-receipt.json", "error": type(exc).__name__})
        upload_receipt["errors"] = errors
        upload_receipt["readback_verified"] = False
        (root / "upload-receipt.json").write_bytes(canonical(upload_receipt))
    if errors:
        print(
            json.dumps(
                {
                    "status": "evidence-upload-failed",
                    "run_id": run_id,
                    "failed_paths": [item["path"] for item in errors],
                    "upload_receipt_sha256": _sha256(root / "upload-receipt.json"),
                },
                sort_keys=True,
            )
        )
        raise RuntimeError("golden evidence upload or read-back failed")
    return upload_receipt


def main():
    run_id = os.environ.get("NPA_SMOKE_RUN_ID", "curobo-functional")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise RuntimeError("NPA_SMOKE_RUN_ID must be a filesystem-safe identity")
    source_commit = os.environ.get("NPA_IMAGE_SOURCE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("NPA_IMAGE_SOURCE_SHA must be an exact lowercase commit")
    image_digest = os.environ.get("NPA_IMAGE_DIGEST", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image_digest):
        raise RuntimeError("NPA_IMAGE_DIGEST must be an exact sha256 digest")
    expected_image_digest = os.environ.get("NPA_EXPECTED_IMAGE_DIGEST", "")
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", expected_image_digest):
        raise RuntimeError("NPA_EXPECTED_IMAGE_DIGEST must be an exact sha256 digest")
    if image_digest != expected_image_digest:
        raise RuntimeError("executed image differs from the frozen candidate digest")
    output_uri = os.environ.get("NPA_OUTPUT_PATH", "")
    if not re.fullmatch(r"s3://[^/\s]+/.+", output_uri):
        raise RuntimeError("NPA_OUTPUT_PATH must be a non-root S3 prefix")
    output_dir = os.environ.get("NPA_SMOKE_OUTPUT_DIR", "")
    if not output_dir:
        raise RuntimeError("NPA_SMOKE_OUTPUT_DIR is required")
    output_root = Path(output_dir)
    if not output_root.is_absolute():
        raise RuntimeError("NPA_SMOKE_OUTPUT_DIR must be absolute")
    root = output_root / run_id
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    root.chmod(0o700)
    try:
        summary = _run_workload(
            root,
            run_id=run_id,
            source_commit=source_commit,
            image_digest=image_digest,
        )
    except Exception as exc:
        (root / "failure.json").write_bytes(
            canonical(
                {
                    "schema_version": "npa.curobo.smoke-failure.v1",
                    "run_id": run_id,
                    "source_commit": source_commit,
                    "image_digest": image_digest,
                    "failure_type": type(exc).__name__,
                    "message": "cuRobo golden workload failed before acceptance",
                }
            )
        )
        upload_failure_type = None
        try:
            _upload_tree(
                root,
                output_uri=output_uri,
                run_id=run_id,
                image_digest=image_digest,
                workload_status="failed",
            )
        except Exception as upload_exc:
            upload_failure_type = type(upload_exc).__name__
        print(
            json.dumps(
                {
                    "status": "failed",
                    "run_id": run_id,
                    "failure_type": type(exc).__name__,
                    "evidence_upload_failure_type": upload_failure_type,
                    "failure_sha256": _sha256(root / "failure.json"),
                    "upload_receipt_sha256": (
                        _sha256(root / "upload-receipt.json")
                        if (root / "upload-receipt.json").is_file()
                        else None
                    ),
                },
                sort_keys=True,
            )
        )
        raise
    _upload_tree(
        root,
        output_uri=output_uri,
        run_id=run_id,
        image_digest=image_digest,
        workload_status="passed",
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "run_id": run_id,
                "source_commit": source_commit,
                **summary,
                "upload_receipt_sha256": _sha256(root / "upload-receipt.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
