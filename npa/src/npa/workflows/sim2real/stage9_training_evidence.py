"""Materialize temporal credit and durable checkpoint-selection evidence for Stage 9."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.workflow_io import json_file_sha256


_SELECTION_FIELDS = (
    "checkpoint_uri",
    "checkpoint_sha256",
    "checkpoint_size_bytes",
    "generator_policy_sha256",
    "outer_iteration",
    "inner_iteration",
    "training_iteration",
    "checkpoint_completed_updates",
    "checkpoint_iteration_semantics",
    "validation_report_uri",
    "validation_report_sha256",
    "rank_key",
    "selection_policy",
    "candidate_count",
    "audited_candidate_count",
)


def _selection_summary(selection: dict[str, Any]) -> dict[str, Any]:
    return {
        field: selection[field] for field in _SELECTION_FIELDS if field in selection
    }


def validation_selection_record(
    *,
    run_id: str,
    source_sha: str,
    outer_iteration: int,
    selection: dict[str, Any],
) -> dict[str, Any]:
    """Build a compact, content-addressed validation-only selection record.

    Args:
        run_id: Exact workflow run identifier.
        source_sha: Source SHA attested by the workflow image.
        outer_iteration: Canonical outer-loop coordinate.
        selection: Full validation-only selection returned by checkpoint ranking.
    Returns:
        Compact selection record without embedded validation reports or media.
    Raises:
        ValueError: If the selection is not a complete validation-only ranking.
    """
    if selection.get("evaluation_split") != "validation":
        raise ValueError("Stage 9 compact selection must remain validation-only")
    ranked = selection.get("ranked_candidates")
    if not isinstance(ranked, list) or not ranked:
        raise ValueError("Stage 9 compact selection lacks ranked candidates")
    if ranked[0].get("checkpoint_uri") != selection.get("checkpoint_uri"):
        raise ValueError("Stage 9 compact selection does not lead with its selection")
    if not all(
        isinstance(row, dict) and "validation_report" not in row for row in ranked
    ):
        raise ValueError("Stage 9 compact selection leaked an embedded report")
    summary = _selection_summary(selection)
    required = {
        "checkpoint_uri",
        "checkpoint_sha256",
        "checkpoint_size_bytes",
        "generator_policy_sha256",
        "validation_report_uri",
        "validation_report_sha256",
        "rank_key",
        "selection_policy",
    }
    if required - summary.keys():
        raise ValueError("Stage 9 compact selection lacks selected checkpoint identity")
    payload: dict[str, Any] = {
        "schema": "npa.sim2real.validation_selection.v1",
        "run_id": run_id,
        "source_sha": source_sha,
        "outer_iteration": outer_iteration,
        "selection_input_split": "validation",
        "selected_checkpoint": summary,
        "ranked_candidates": ranked,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def publish_validation_selection_record(
    *,
    run_id: str,
    source_sha: str,
    outer_iteration: int,
    root: str,
    selection: dict[str, Any],
    work: Path,
    publish_json: Callable[..., str],
) -> dict[str, str]:
    """Publish one compact validation-selection record for a Stage 9 pass.

    Args:
        run_id: Exact workflow run identifier.
        source_sha: Source SHA attested by the workflow image.
        outer_iteration: Canonical outer-loop coordinate.
        root: Run-owned durable artifact root.
        selection: Full validation-only selection returned by checkpoint ranking.
        work: Owned stage work directory.
        publish_json: JSON artifact publisher with the workflow IO contract.
    Returns:
        Durable selection-record URI and content digest.
    Raises:
        ValueError: If the selection cannot become a compact validation record.
        OSError: If the local artifact directory cannot be prepared.
    """
    payload = validation_selection_record(
        run_id=run_id,
        source_sha=source_sha,
        outer_iteration=outer_iteration,
        selection=selection,
    )
    uri = f"{root}/checkpoints/validation-selection/outer-{outer_iteration:02d}.json"
    # The embedded canonical digest avoids a self-referential file hash. The
    # returned digest is instead the exact JSON bytes that ``publish_json``
    # writes, allowing the ComponentRecord to bind the durable artifact.
    artifact_sha256 = json_file_sha256(payload)
    publish_json(uri, payload, directory=work / "selection")
    return {
        "uri": uri,
        "sha256": artifact_sha256,
        "content_sha256": payload["content_sha256"],
    }


def temporal_signals(args, work, evaluations):
    """Write validated rollout evaluations and their temporal training signals.

    Args:
        args: Stage settings, including the configured evaluation threshold.
        work: Owned temporary stage directory.
        evaluations: Validated hosted evaluations keyed by rollout identity.
    Returns:
        Local directories and the ordered evaluation and training-signal rows.
    Raises:
        ValueError: Temporal credit violates the evaluator binding contract.
        OSError: Owned output files cannot be written.
    """
    from npa.workflows.sim2real.temporal_credit import convert_evaluation

    evaluation_dir, signal_dir = work / "evaluations", work / "signals"
    evaluation_dir.mkdir()
    signal_dir.mkdir()
    evaluation_rows, signals = [], []
    for rollout_id in sorted(evaluations):
        evaluation = dict(evaluations[rollout_id])
        evaluation["threshold"] = args.threshold
        signal = convert_evaluation(evaluation)
        (evaluation_dir / f"{rollout_id}.json").write_text(
            json.dumps(evaluation, indent=2)
        )
        (signal_dir / f"{rollout_id}.json").write_text(json.dumps(signal, indent=2))
        evaluation_rows.append(evaluation)
        signals.append(signal)
    return {
        "evaluation_dir": evaluation_dir,
        "signal_dir": signal_dir,
        "evaluations": evaluation_rows,
        "signals": signals,
    }


def _stage9_iteration(args, inputs, update, embodiment):
    return {
        "iteration": args.inner_iteration,
        "actions_uri": inputs["actions_uri"],
        "vlm_eval_uri": inputs["evaluation_uri"],
        "signal_uri": inputs["signal_uri"],
        "trainer_component_invocation": update.get("component_invocation"),
        "update": update,
        "embodiment": embodiment,
        "sample_vlm_eval": inputs["evaluations"][0],
        "sample_signal": inputs["signals"][0],
        "evaluator_usage": inputs["evaluator_usage"],
    }


def component_record_kwargs(args, root, evidence_uri, selection, update, embodiment):
    """Bind a completed native pass or its replay to the Stage 9 component record.

    Args:
        args: The configured full PPO pass.
        root: Run-owned output root.
        evidence_uri: Durable inner-loop evidence document.
        selection: Exact validation-selected checkpoint and report.
        update: Completed native trainer result.
        embodiment: Verified training identity, or None for durable replay.
    Returns:
        Keyword arguments for the stage adapter's component publication.
    Raises:
        KeyError: Required selection evidence is absent.
    """
    artifacts = {
        "evidence": evidence_uri,
        "checkpoint": selection["checkpoint_uri"],
        "checkpoint_sha256": selection.get("checkpoint_sha256", ""),
        "validation_report": selection["validation_report_uri"],
        "ppo_iterations": args.ppo_iterations,
        "component_invocation": update.get("component_invocation"),
    }
    selection_record = selection.get("validation_selection_record")
    if isinstance(selection_record, dict):
        artifacts["validation_selection_record"] = selection_record["uri"]
        artifacts["validation_selection_record_sha256"] = selection_record["sha256"]
    if embodiment is None:
        artifacts["idempotent_replay"] = True
        description = "Re-adopted exact durable same-iteration PPO and validation evidence after a standard-runtime Job retry."
    else:
        artifacts["embodiment"] = embodiment
        description = "Genuine BYO Isaac RSL-RL PPO consumed bounded temporal Reason signals and published an exact checkpoint; validation alone selected it."
    return {
        "root_uri": root,
        "stage": 9,
        "name": "stage_09_training_signal",
        "tier": "WORKS",
        "evidence": description,
        "artifacts": artifacts,
        "require_gpu": True,
    }


def loop_evidence(args, inputs, prior, update, embodiment, candidates, selection):
    """Bind a complete PPO pass and validation-only selection into durable evidence.

    Args:
        args: Requested outer and inner training coordinates.
        inputs: Bound source prefixes, sample evaluations, and temporal signals.
        prior: Previous complete training passes and checkpoint candidates.
        update: Completed native trainer result.
        embodiment: Verified training embodiment identity.
        candidates: Complete current and previous validation checkpoint pool.
        selection: Validation-only selected checkpoint and its report.
    Returns:
        The complete compositional inner-loop evidence document.
    Raises:
        KeyError: Required bound input or selection evidence is missing.
    """
    validation = dict(selection.get("validation_report") or {})
    evidence = {
        "schema": "npa.sim2real.inner_loop_evidence.v1",
        "run_id": args.run_id,
        "outer_iteration": args.outer_iteration,
        "iterations": list(prior.get("iterations") or [])
        + [_stage9_iteration(args, inputs, update, embodiment)],
        "checkpoint_candidates": candidates,
        "selected_checkpoint_uri": selection["checkpoint_uri"],
        "final_checkpoint_uri": selection["checkpoint_uri"],
        "checkpoint_selection": selection,
        "selected_validation_report": validation,
        "selected_validation_strict_success": float(
            validation.get("success_rate") or 0.0
        ),
    }
    selection_record = selection.get("validation_selection_record")
    if isinstance(selection_record, dict):
        evidence["validation_selection_record"] = selection_record
    return evidence


def prepared_loop_evidence(
    args,
    root: str,
    work: Path,
    inputs,
    prior,
    update,
    embodiment,
    candidates,
    selection,
    source_sha: str,
    publish_json: Callable[..., str],
):
    """Publish compact selection lineage, then build canonical Stage 9 evidence.

    Args:
        args: Requested outer and inner training coordinates.
        root: Run-owned durable artifact root.
        work: Owned Stage 9 work directory.
        inputs: Bound source prefixes, sample evaluations, and temporal signals.
        prior: Previous complete training passes and checkpoint candidates.
        update: Completed native trainer result.
        embodiment: Verified training embodiment identity.
        candidates: Complete current and previous validation checkpoint pool.
        selection: Validation-only selected checkpoint and its report.
        source_sha: Source SHA attested by the workflow image.
        publish_json: JSON artifact publisher with the workflow IO contract.
    Returns:
        Complete canonical Stage 9 evidence with compact selection lineage.
    Raises:
        ValueError: If the validation selection cannot be compactly published.
        OSError: If the local artifact directory cannot be prepared.
    """
    selection["validation_selection_record"] = publish_validation_selection_record(
        run_id=args.run_id,
        source_sha=source_sha,
        outer_iteration=args.outer_iteration,
        root=root,
        selection=selection,
        work=work,
        publish_json=publish_json,
    )
    return loop_evidence(args, inputs, prior, update, embodiment, candidates, selection)
