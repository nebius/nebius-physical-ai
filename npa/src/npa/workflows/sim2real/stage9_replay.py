"""Durable replay and fan-in helpers for compositional Sim2Real Stage 9."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from npa.workflows.sim2real.checkpoint_selection import select_best_checkpoint
from npa.workflows.sim2real.workflow_io import (
    aggregate_parallel_provenance,
    publish_component_record,
    read_json,
)


def existing_replay(
    *,
    prior: dict[str, Any],
    outer_iteration: int,
    inner_iteration: int,
    actions_uri: str,
    merged_uri: str,
    signal_uri: str,
    sample_vlm_eval: dict[str, Any],
    sample_signal: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]] | None:
    """Validate and adopt evidence from an exact Stage 9 retry.

    Stage 9 writes its durable evidence before the Kubernetes Job becomes terminal.
    A replacement Job therefore treats an exact record at the same loop coordinate
    as committed output. Any partial, divergent, or out-of-order record fails closed.
    """

    iterations = list(prior.get("iterations") or [])
    candidates = list(prior.get("checkpoint_candidates") or [])
    iteration_matches = [
        item
        for item in iterations
        if int(item.get("iteration") or 0) == inner_iteration
    ]
    candidate_matches = [
        item
        for item in candidates
        if int(item.get("outer_iteration") or 0) == outer_iteration
        and int(item.get("inner_iteration") or 0) == inner_iteration
    ]
    if not iteration_matches and not candidate_matches:
        if any(
            int(item.get("iteration") or 0) > inner_iteration for item in iterations
        ):
            raise RuntimeError(
                "Stage 9 cannot insert evidence before a later iteration"
            )
        return None
    if len(iteration_matches) != 1 or len(candidate_matches) != 1:
        raise RuntimeError("Stage 9 found partial or duplicate same-iteration evidence")
    if int(prior.get("outer_iteration") or 0) != outer_iteration:
        raise RuntimeError(
            "Stage 9 replay evidence belongs to a different outer iteration"
        )
    if max(int(item.get("iteration") or 0) for item in iterations) != inner_iteration:
        raise RuntimeError("Stage 9 may replay only the latest committed iteration")

    iteration = iteration_matches[0]
    candidate = candidate_matches[0]
    expected = {
        "iteration": inner_iteration,
        "actions_uri": actions_uri,
        "vlm_eval_uri": merged_uri,
        "signal_uri": signal_uri,
        "sample_vlm_eval": sample_vlm_eval,
        "sample_signal": sample_signal,
    }
    if any(iteration.get(key) != value for key, value in expected.items()):
        raise RuntimeError(
            "Stage 9 same-iteration replay conflicts with durable evidence"
        )
    update = dict(iteration.get("update") or {})
    if candidate.get("evaluation_split") != "validation" or candidate.get(
        "checkpoint_uri"
    ) != update.get("checkpoint_path"):
        raise RuntimeError("Stage 9 replay checkpoint lineage is inconsistent")
    selection = select_best_checkpoint(candidates)
    if (
        prior.get("selected_checkpoint_uri") != selection["checkpoint_uri"]
        or prior.get("final_checkpoint_uri") != selection["checkpoint_uri"]
        or prior.get("checkpoint_selection") != selection
        or prior.get("selected_validation_report") != selection.get("validation_report")
    ):
        raise RuntimeError("Stage 9 replay selection conflicts with its candidates")
    return candidate, selection, update


def publish_stage8_join(
    *,
    root: str,
    work: Path,
    lane_base: str,
    reason2: dict[str, Any],
    reason3: dict[str, Any],
    merged_uri: str,
    rollout_count: int,
    outer_iteration: int,
    inner_iteration: int,
) -> None:
    """Publish the canonical Stage 8 record after validating both lane records."""

    lane_records = [
        read_json(
            f"{root}/components/lanes/stage_08/"
            f"{lane}-o{outer_iteration}-i{inner_iteration}.json",
            directory=work / f"lane-{lane}",
        )
        for lane in ("reason2", "reason3")
    ]
    expected_lanes = [
        f"{lane}-o{outer_iteration}-i{inner_iteration}"
        for lane in ("reason2", "reason3")
    ]
    if [item.get("lane") for item in lane_records] != expected_lanes or any(
        item.get("artifacts", {}).get("result") != lane_base + f"{lane}.json"
        for item, lane in zip(lane_records, ("reason2", "reason3"), strict=True)
    ):
        raise RuntimeError(
            "Stage 8 lane records do not match the declared Reason fan-out"
        )
    provenance = aggregate_parallel_provenance(
        [reason2["provenance"], reason3["provenance"]], stage=8
    )
    publish_component_record(
        root_uri=root,
        stage=8,
        name="stage_08_vlm_eval_train",
        tier="WORKS",
        evidence="Two parallel real Cosmos Reason lanes evaluated event-local Isaac observations and were deterministically merged.",
        artifacts={
            "reason2": lane_base + "reason2.json",
            "reason3": lane_base + "reason3.json",
            "merged": merged_uri,
            "rollout_count": rollout_count,
            "reason_lane_provenance": [reason2["provenance"], reason3["provenance"]],
            "lane_records": lane_records,
        },
        require_gpu=True,
        execution_provenance=provenance,
    )
