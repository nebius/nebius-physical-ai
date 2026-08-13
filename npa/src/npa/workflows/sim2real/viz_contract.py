"""Run-level policy and artifact provenance supplied to Stage 14."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.capture import runtime_parameter_metadata


def visualization_run_metadata(
    *,
    config: Any,
    artifact_root: str,
    policy_checkpoint: str = "",
    candidate: dict[str, Any] | None = None,
    heldout_report: dict[str, Any] | None = None,
    progress: bool = False,
) -> dict[str, Any]:
    """Build one stable Rerun/MCAP access and checkpoint provenance contract."""

    candidate = candidate or {}
    heldout = heldout_report or {}
    report_name = "sim2real-progress.rrd" if progress else "sim2real.rrd"
    viewer_command = (
        "npa workbench sim2real rerun serve "
        f"--run-id {config.run_id} --s3-bucket {config.s3_bucket} "
        f"--s3-prefix {config.s3_prefix}"
    )
    inference = heldout.get("policy_inference_provenance") or {}
    return {
        "run_id": config.run_id,
        "artifact_root": artifact_root + "/",
        "rrd_s3_uri": f"{artifact_root}/reports/{report_name}",
        "candidate_s3_uri": f"{artifact_root}/checkpoints/candidate/candidate.json",
        "policy_checkpoint": policy_checkpoint,
        "policy_checkpoint_identity": candidate.get("policy_checkpoint_identity", ""),
        "policy_checkpoint_sha256": candidate.get("policy_checkpoint_sha256", ""),
        "policy_checkpoint_size_bytes": candidate.get(
            "policy_checkpoint_size_bytes", ""
        ),
        "policy_download_command": candidate.get("policy_download_command", ""),
        "policy_ui_action": candidate.get("policy_ui_action", ""),
        "policy_deployable": candidate.get("deployable_policy", False),
        "heldout_policy_checkpoint": str(heldout.get("policy_checkpoint") or ""),
        "heldout_policy_checkpoint_sha256": str(
            heldout.get("policy_checkpoint_sha256") or ""
        ),
        "heldout_policy_checkpoint_size_bytes": int(
            heldout.get("policy_checkpoint_size_bytes") or 0
        ),
        "heldout_policy_loaded_for_inference": bool(
            inference.get("loaded_for_inference")
        ),
        "runtime_parameters": runtime_parameter_metadata(),
        "orchestrator_job_name": config.run_id,
        "orchestrator_node_product": config.k8s_gpu_product,
        "viewer_command": viewer_command,
    }
