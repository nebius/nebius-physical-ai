"""Materialize temporal credit and durable checkpoint-selection evidence for Stage 9."""

from __future__ import annotations

import json


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
    return {
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
