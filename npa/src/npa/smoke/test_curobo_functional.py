"""Real single-pose cuRobo GPU qualification, separate from full benchmark runs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re

from pydantic import ValidationError

from npa.workbench.curobo.audit import audit_bytes
from npa.workbench.curobo.artifacts import (
    build_rrd,
    canonical,
    decode_rrd,
    read_journal,
)
from npa.workbench.curobo.runner import execute
from npa.workbench.curobo.schemas import PlanManifest


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def main():
    run_id = os.environ.get("NPA_SMOKE_RUN_ID", "curobo-functional")
    if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
        raise RuntimeError("NPA_SMOKE_RUN_ID must be a filesystem-safe identity")
    source_commit = os.environ.get("NPA_IMAGE_SOURCE_SHA", "")
    if not re.fullmatch(r"[0-9a-f]{40}", source_commit):
        raise RuntimeError("NPA_IMAGE_SOURCE_SHA must be an exact lowercase commit")
    root = Path(os.environ.get("NPA_SMOKE_OUTPUT_DIR", "/tmp/npa-golden")) / run_id
    root.mkdir(parents=True, exist_ok=False)
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
        PlanManifest.model_validate({"robot": "/tmp/unreviewed.yml", "problems": []})
    except ValidationError:
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
        "image_digest": os.environ.get("NPA_IMAGE_DIGEST", "unavailable"),
        "gpu": report["gpu"],
        "artifacts": artifacts,
        "objective": {
            "success": 1,
            "failed_control": 1,
            "invalid": 0,
            "independent_validation": audit["valid"],
            "rrd_decode": rrd_manifest["decode"]["print"],
        },
        "limitations": report["limitations"],
    }
    (root / "artifact-manifest.json").write_bytes(canonical(artifact_manifest))
    print(
        json.dumps(
            {
                "status": "passed",
                "run_id": run_id,
                "source_commit": source_commit,
                "artifact_manifest_sha256": _sha256(root / "artifact-manifest.json"),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
