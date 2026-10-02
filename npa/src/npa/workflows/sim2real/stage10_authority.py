"""Validate Stage 10 checkpoint and render-producer authority."""

from __future__ import annotations

import argparse
from typing import Any

from npa.workflows.sim2real.byo_isaac_trainer import artifact_tag, k8s_job_name
from npa.workflows.sim2real.checkpoint_selection import (
    resolve_run_scoped_checkpoint,
)


def _canonical_trainer_checkpoint_uri(
    *,
    root: str,
    run_id: str,
    outer_iteration: int,
    candidate: dict[str, Any],
) -> str:
    candidate_outer = candidate.get("outer_iteration")
    candidate_inner = candidate.get("inner_iteration")
    if (
        type(candidate_outer) is not int
        or candidate_outer != outer_iteration
        or type(candidate_inner) is not int
        or candidate_inner <= 0
    ):
        raise RuntimeError(
            "Stage 10 checkpoint lacks exact producer outer/inner iteration"
        )
    job = k8s_job_name("s2r-byo-isaac-train", run_id)
    tag = artifact_tag(f"outer-{candidate_outer:02d}-iter-{candidate_inner:02d}")
    return f"{root}/byo-trainer/{job}/{tag}/model_latest.pt"


def _resolve_stage10_checkpoint(
    args: argparse.Namespace,
    root: str,
    evidence: dict[str, Any],
) -> tuple[str, dict[str, Any], dict[str, Any]]:
    run_id = str(getattr(args, "run_id", "") or root.rstrip("/").rsplit("/", 1)[-1])
    try:
        selection, candidate = resolve_run_scoped_checkpoint(
            evidence,
            run_root=root,
            run_id=run_id,
        )
    except ValueError as exc:
        raise RuntimeError("Stage 10 checkpoint is outside the current run") from exc
    recorded_outer = evidence.get("outer_iteration")
    if (
        not isinstance(recorded_outer, int)
        or isinstance(recorded_outer, bool)
        or recorded_outer != args.outer_iteration
    ):
        raise RuntimeError("Stage 10 outer iteration disagrees with selection")
    return run_id, selection, candidate


def _assert_modern_trainer_scope(
    args: argparse.Namespace,
    root: str,
    evidence: dict[str, Any],
    run_id: str,
    selection: dict[str, Any],
    candidate: dict[str, Any],
) -> None:
    if evidence.get("schema") != "npa.sim2real.inner_loop_evidence.v1":
        raise RuntimeError("Stage 10 requires canonical Stage 9 inner-loop evidence")
    if evidence.get("run_id") != run_id:
        raise RuntimeError("Stage 10 evidence run_id disagrees with execution")
    expected_uri = _canonical_trainer_checkpoint_uri(
        root=root,
        run_id=run_id,
        outer_iteration=args.outer_iteration,
        candidate=candidate,
    )
    if (
        selection.get("outer_iteration") != candidate.get("outer_iteration")
        or selection.get("inner_iteration") != candidate.get("inner_iteration")
        or selection["checkpoint_uri"] != expected_uri
    ):
        raise RuntimeError(
            "Stage 10 checkpoint is not the exact canonical BYO trainer output"
        )


def _validate_stage10_input_scope(
    args: argparse.Namespace,
    root: str,
    evidence: dict[str, Any],
) -> str:
    run_id, selection, candidate = _resolve_stage10_checkpoint(args, root, evidence)
    _assert_modern_trainer_scope(
        args,
        root,
        evidence,
        run_id,
        selection,
        candidate,
    )
    return run_id


def validate_stage10_input_scope(
    args: argparse.Namespace,
    *,
    root: str,
    evidence: dict[str, Any],
) -> str:
    """Reject cross-run or cross-iteration evidence before Isaac sees it.

    Args:
        args: Stage adapter arguments containing run and outer-loop identity.
        root: Exact S3 artifact root selected for this workflow run.
        evidence: Persisted inner-loop checkpoint-selection evidence.
    Returns:
        The run identifier used by the Stage 10 producer.
    Raises:
        RuntimeError: If checkpoint or outer-iteration authority disagrees.
    """
    return _validate_stage10_input_scope(args, root, evidence)


def expected_byo_render_prefix(
    *,
    root: str,
    run_id: str,
    outer_iteration: int,
    evaluation_tag: str | None = None,
) -> str:
    """Return the exact BYO Isaac render prefix for one gold evaluation.

    Args:
        root: Exact S3 artifact root selected for this workflow run.
        run_id: Workflow run identifier used in the Kubernetes job name.
        outer_iteration: Current outer-loop iteration.
        evaluation_tag: Optional immutable current-attempt evaluation tag.

    Returns:
        The run-owned BYO evaluation render prefix.

    Raises:
        None.
    """

    job_name = k8s_job_name(
        "s2r-byo-isaac-eval",
        run_id,
        artifact_tag(evaluation_tag or f"gold-o{outer_iteration:02d}"),
    )
    return f"{root}/byo-eval/{job_name}/renders/"
