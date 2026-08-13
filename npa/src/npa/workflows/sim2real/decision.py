"""Stage 11 threshold decision and candidate-checkpoint packaging."""

from __future__ import annotations

import os
import shlex
import tempfile
import time
from pathlib import Path
from typing import Any

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.constants import SCHEMA_THRESHOLD_DECISION
from npa.workflows.sim2real.hashing import sha256_file
from npa.workflows.sim2real.models import Sim2RealLoopConfig, Sim2RealLoopError
from npa.workflows.sim2real.utils import _utc_now, _write_json_artifact


def threshold_decision(
    config: Sim2RealLoopConfig,
    *,
    local_dir: Path,
    heldout_report: dict[str, Any],
    outer_iteration: int,
) -> dict[str, Any]:
    """Apply Stage 11 threshold gate and write promote/loop-back artifacts."""

    stage_started = time.monotonic()
    success_rate = float(heldout_report["success_rate"])
    checkpoint_dir = local_dir / "checkpoints" / "candidate"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    candidate_path = checkpoint_dir / "candidate.json"
    # When a BYO trainer produced a real policy checkpoint (surfaced by the heldout
    # eval as policy_checkpoint), promote should reference those real weights and be
    # deployable — not the reference-metadata stub.
    real_checkpoint = str(heldout_report.get("policy_checkpoint") or "").strip()
    is_real_policy = real_checkpoint.startswith("s3://") and real_checkpoint.endswith(
        ".pt"
    )
    inference = dict(heldout_report.get("policy_inference_provenance") or {})
    scenario_proof = dict(heldout_report.get("applied_scenario_proof") or {})
    split = str(heldout_report.get("evaluation_split") or "")
    promotion_gates = {
        "strict_success_threshold": success_rate >= config.threshold,
        "final_gold_split": split in {"", "gold_heldout"},
        "checkpoint_loaded_for_inference": (
            bool(inference.get("loaded_for_inference")) if is_real_policy else True
        ),
        "stock_or_scripted_policy_false": (
            inference.get("stock_or_scripted_policy") is False
            if is_real_policy
            else True
        ),
        "applied_scenario_digest_match": (
            bool(scenario_proof.get("exact_digest_match")) if is_real_policy else True
        ),
        "minimum_episode_coverage": len(heldout_report.get("per_env") or [])
        >= (config.heldout_env_count if split == "gold_heldout" else 1),
    }
    promoted = all(promotion_gates.values())
    checkpoint_uri = real_checkpoint if is_real_policy else str(checkpoint_dir)
    checkpoint_metadata: dict[str, Any] = {}
    if is_real_policy:
        with tempfile.TemporaryDirectory(prefix="npa-policy-proof-") as temporary:
            local_checkpoint = Path(temporary) / Path(real_checkpoint).name
            try:
                StorageClient.from_environment(
                    endpoint_url=config.s3_endpoint
                ).download_file(real_checkpoint, str(local_checkpoint))
                checkpoint_metadata = {
                    "policy_checkpoint_identity": Path(real_checkpoint).name,
                    "policy_checkpoint_sha256": sha256_file(local_checkpoint),
                    "policy_checkpoint_size_bytes": local_checkpoint.stat().st_size,
                    "policy_download_command": (
                        "aws s3 cp "
                        f"{shlex.quote(real_checkpoint)} ./model.pt "
                        ' --endpoint-url "$AWS_ENDPOINT_URL"'
                    ),
                    "policy_ui_action": (
                        "Open Artifacts for this run, select the .pt checkpoint, "
                        "and choose Download; the Rerun viewer links it but does not execute it."
                    ),
                }
            except Exception as exc:
                if (
                    os.environ.get("NPA_SIM2REAL_REQUIRE_REAL_COMPONENTS", "").strip()
                    == "1"
                ):
                    raise Sim2RealLoopError(
                        f"deployable policy checkpoint could not be hashed: {exc}"
                    ) from exc
                checkpoint_metadata = {"policy_metadata_error": str(exc)}
    decision = {
        "schema": SCHEMA_THRESHOLD_DECISION,
        "stage": 11,
        "outer_iteration": outer_iteration,
        "success_rate": round(success_rate, 6),
        "threshold": config.threshold,
        "threshold_met": promotion_gates["strict_success_threshold"],
        "decision": "promote_checkpoint" if promoted else "loop_back_to_inner_loop",
        "evaluation_split": split or "legacy_heldout",
        "promotion_gates": promotion_gates,
        "checkpoint_uri": checkpoint_uri,
        "max_outer_iterations": config.outer_iterations,
        "remaining_outer_iterations": max(0, config.outer_iterations - outer_iteration),
        "effective_learning_rate": config.learning_rate,
        "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
        "duration_s": round(time.monotonic() - stage_started, 3),
    }
    # Package real weights even below threshold; promotion remains a distinct
    # quality decision so fixed-count runs never lose candidate access.
    candidate_payload: dict[str, Any] = {}
    if promoted or is_real_policy:
        candidate_payload = {
            "schema": "npa.sim2real.candidate_checkpoint.v1",
            "run_id": config.run_id,
            "source": (
                "isaac-rsl-rl-ppo" if is_real_policy else "vlm-rl-reference-update"
            ),
            "deployable_policy": bool(is_real_policy and promoted),
            "policy_bytes_available": is_real_policy,
            "policy_artifact_kind": (
                "isaac_rsl_rl_checkpoint" if is_real_policy else "reference_metadata"
            ),
            "policy_checkpoint_uri": real_checkpoint if is_real_policy else "",
            **checkpoint_metadata,
            "handoff_doc": "docs/workbench/guides/sim2real-customer-assets.md#real-world-policy-deployment-stage-12-seam",
            "heldout_success_rate": round(success_rate, 6),
            "threshold": config.threshold,
            "threshold_met": promotion_gates["strict_success_threshold"],
            "promotion_gates": promotion_gates,
            "effective_learning_rate": config.learning_rate,
            "learning_rate_scope": "vlm_signal_adapter_and_no_signal_control",
            "promotion_decision": (
                "promote_checkpoint" if promoted else "loop_back_to_inner_loop"
            ),
            "candidate_status": (
                "promoted" if promoted else "below_threshold_policy_artifact"
            ),
            "evaluated_at": _utc_now(),
            "promoted_at": _utc_now() if promoted else "",
        }
        _write_json_artifact(candidate_path, candidate_payload)
        # The pod filesystem is disposable.  Keep the complete policy handoff
        # in the identity-bound Stage 11 durable unit so finalization can
        # reconstruct candidate.json after a controller restart.
        decision["candidate"] = candidate_payload
    if not promoted:
        remaining = max(0, config.outer_iterations - outer_iteration)
        _write_json_artifact(
            local_dir / "outer_loop" / "loopback.json",
            {
                "schema": "npa.sim2real.loopback.v1",
                "run_id": config.run_id,
                "from_stage": 11,
                "to_stage": 7,
                "reason": "heldout threshold not met",
                "outer_iteration": outer_iteration,
                "score": round(success_rate, 6),
                "threshold": config.threshold,
                "threshold_met": promotion_gates["strict_success_threshold"],
                "promotion_gates": promotion_gates,
                "real_policy": is_real_policy,
                "policy_checkpoint_uri": real_checkpoint if is_real_policy else "",
                "candidate_path": str(candidate_path)
                if candidate_path.is_file()
                else "",
                "remaining_outer_iterations": remaining,
                "remaining_work": (
                    "run_next_outer_iteration"
                    if remaining
                    else "configured_outer_iterations_exhausted"
                ),
                "decision": decision,
            },
        )
    path = local_dir / "outer_loop" / "decision.json"
    _write_json_artifact(path, decision)
    return {**decision, "decision_uri": str(path)}
