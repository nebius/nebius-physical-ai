"""Stage 9 evaluator-coverage boundary for compositional Sim2Real."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any


class EvaluatorContractError(RuntimeError):
    """An evaluator boundary cannot be reused by the configured current run."""


def _migration_error(reason: str) -> EvaluatorContractError:
    return EvaluatorContractError(
        f"{reason}. Stage 9 has not started PPO or checkpoint selection. "
        "Preserve the existing run and immutable evaluator artifacts; start a new "
        "run ID and output root with the current evaluator contract and configured "
        "reason model, then rerun Stage 8 before resuming Stage 9. Do not relabel "
        "or rewrite old evaluator artifacts in place."
    )


def _integer(value: Any, *, minimum: int = 0) -> bool:
    return type(value) is int and value >= minimum


def _number(value: Any, *, positive: bool = False) -> bool:
    return (
        type(value) in (int, float)
        and math.isfinite(value)
        and (value > 0 if positive else value >= 0)
    )


def _provenance_matches(
    evaluator: dict[str, Any], record: dict[str, Any], expected_source_sha: str
) -> bool:
    provenance = evaluator.get("provenance")
    if not isinstance(provenance, dict):
        return False
    image = provenance.get("image")
    if (
        not isinstance(image, str)
        or not re.fullmatch(r"[^\s]+@sha256:[a-f0-9]{64}", image)
        or not re.fullmatch(r"[a-f0-9]{40}", expected_source_sha)
        or provenance.get("source_sha") != expected_source_sha
        or provenance.get("image_digest") != image.split("@", 1)[1]
        or provenance.get("execution_mode") != "standard_npa_workflow_skypilot"
        or not isinstance(provenance.get("workflow_job"), str)
        or not provenance["workflow_job"].strip()
    ):
        return False
    artifacts = record.get("artifacts") or {}
    if any(
        artifacts.get(key) != provenance[key]
        for key in (
            "image",
            "image_digest",
            "source_sha",
            "execution_mode",
            "workflow_job",
        )
    ):
        return False
    digest = record.get("content_sha256")
    material = {key: value for key, value in record.items() if key != "content_sha256"}
    return (
        record.get("schema") == "npa.sim2real.component_record.v1"
        and record.get("tier") == "WORKS"
        and digest
        == hashlib.sha256(
            json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest()
    )


def _request_valid(request: dict[str, Any]) -> bool:
    return (
        isinstance(request.get("request_id"), str)
        and bool(request["request_id"].strip())
        and all(
            _integer(request.get(key), minimum=1)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        and request["total_tokens"]
        == request["input_tokens"] + request["output_tokens"]
        and _number(request.get("latency_seconds"), positive=True)
        and _integer(request.get("retries"))
        and "cost_usd" in request
        and (request["cost_usd"] is None or _number(request["cost_usd"]))
    )


def _usage_matches(
    usage: dict[str, Any], requests: list[dict[str, Any]], model: str
) -> bool:
    if not requests or not all(_request_valid(request) for request in requests):
        return False
    ids = [request["request_id"] for request in requests]
    priced = all(request["cost_usd"] is not None for request in requests)
    return (
        usage.get("model") == model
        and usage.get("provider") == "nebius"
        and usage.get("backend") == "token_factory"
        and _integer(usage.get("request_count"), minimum=1)
        and usage["request_count"] == len(requests)
        and len(set(ids)) == len(ids)
        and usage.get("request_ids") == ids
        and all(
            _integer(usage.get(key), minimum=1)
            and usage[key] == sum(request[key] for request in requests)
            for key in ("input_tokens", "output_tokens", "total_tokens")
        )
        and usage.get("per_request_latency_seconds")
        == [request["latency_seconds"] for request in requests]
        and _number(usage.get("aggregate_latency_seconds"), positive=True)
        and usage["aggregate_latency_seconds"]
        == round(sum(request["latency_seconds"] for request in requests), 6)
        and _integer(usage.get("retries"))
        and usage["retries"] == sum(request["retries"] for request in requests)
        and "cost_usd" in usage
        and (
            usage["cost_usd"] is None
            if not priced
            else (
                _number(usage["cost_usd"])
                and usage["cost_usd"]
                == round(sum(request["cost_usd"] for request in requests), 8)
            )
        )
        and usage.get("cost_source") == ("response_usage" if priced else "unavailable")
    )


def validate_hosted_evaluator(
    *,
    stage7: dict[str, Any],
    evaluator: dict[str, Any],
    stage8_record: dict[str, Any],
    expected_model: str,
    expected_source_sha: str,
    evaluator_uri: str,
    outer_iteration: int,
    inner_iteration: int,
) -> dict[str, dict[str, Any]]:
    """Validate the complete Stage 8 boundary before any Stage 9 side effects.

    This is shared by the real training adapter and live admission checks. It
    reads no storage and never repairs or mutates archived evaluator artifacts.
    Canonical Stage 7 uses ``SAMPLE_INDEX = {step: index for index, step in
    enumerate(SAMPLE_STEPS)}``: decision indices are contiguous from zero even
    when the underlying simulator steps are sparse. Generic hosted evaluations
    with other action indices are outside this canonical Stage 9 contract.
    """

    from npa.workbench.cosmos.reason import hosted_rollout_model_family

    try:
        expected_family = hosted_rollout_model_family(expected_model)
        artifacts = stage8_record.get("artifacts") or {}
        if (
            evaluator.get("schema")
            not in {
                "npa.sim2real.cosmos3_evaluator.v1",
                "npa.sim2real.cosmos_reason_lane.v2",
            }
            or (evaluator.get("evaluator") or evaluator.get("lane")) != "cosmos3"
            or evaluator.get("model") != expected_model
            or evaluator.get("reason_family") != expected_family
            or evaluator.get("backend") != "token_factory"
            or evaluator.get("provider") != "nebius"
            or not _provenance_matches(evaluator, stage8_record, expected_source_sha)
            or stage8_record.get("stage") != 8
            or stage8_record.get("name") != "stage_08_vlm_eval_train"
            or artifacts.get("result") != evaluator_uri
            or artifacts.get("backend") != "token_factory"
            or artifacts.get("provider") != "nebius"
            or artifacts.get("model") != expected_model
            or artifacts.get("reason_family") != expected_family
            or int(artifacts.get("outer_iteration") or 0) != outer_iteration
            or int(artifacts.get("inner_iteration") or 0) != inner_iteration
        ):
            raise _migration_error(
                "Stage 8 requires the configured hosted model identity, family, and provenance"
            )
        rows = validate_stage7_cosmos3_coverage(stage7, evaluator)
        usage = dict(evaluator.get("evaluator_usage") or {})
        requests = [dict(item.get("request") or {}) for item in rows.values()]
        if (
            artifacts.get("evaluator_usage") != usage
            or not _integer(artifacts.get("rollout_count"), minimum=1)
            or artifacts["rollout_count"] != len(rows)
            or not _usage_matches(usage, requests, expected_model)
        ) or any(
            item.get("schema") != "npa.sim2real.vlm_eval.v3"
            or item.get("backend") != "token_factory"
            or item.get("provider") != "nebius"
            or item.get("model") != evaluator.get("model")
            or item.get("reason_family") != expected_family
            or not isinstance(item.get("request"), dict)
            or not _integer(item.get("action_count"), minimum=1)
            or len(item.get("per_step") or []) != item["action_count"]
            or any(
                not isinstance(step, dict) or not _integer(step.get("step"))
                for step in item.get("per_step") or []
            )
            or {step["step"] for step in item.get("per_step") or []}
            != set(range(item["action_count"]))
            for item in rows.values()
        ):
            raise _migration_error(
                "Stage 8 hosted evaluations have incompatible accounting or per-action coverage"
            )
    except EvaluatorContractError:
        raise
    except (AttributeError, TypeError, ValueError) as exc:
        raise _migration_error(
            "Stage 8 evaluator contract contains malformed metadata"
        ) from exc
    return rows


def validate_stage7_cosmos3_coverage(
    stage7: dict[str, Any], cosmos3: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Return evaluations only when one Cosmos3 result covers Stage 7 exactly."""

    rollout_dirs = stage7.get("rollout_dirs")
    stage7_ids = (
        [Path(str(item).rstrip("/")).name for item in rollout_dirs]
        if isinstance(rollout_dirs, list)
        else []
    )
    evaluations = cosmos3.get("evaluations")
    evaluation_rows = evaluations if isinstance(evaluations, list) else []
    evaluation_ids = [
        str(item.get("rollout_id") or "")
        for item in evaluation_rows
        if isinstance(item, dict)
    ]
    source_ids_value = cosmos3.get("source_rollout_ids")
    source_ids = (
        [str(item) for item in source_ids_value]
        if isinstance(source_ids_value, list)
        else []
    )
    if (
        stage7.get("schema") != "npa.sim2real.policy_rollouts.v1"
        or not stage7_ids
        or any(not item for item in stage7_ids)
        or len(set(stage7_ids)) != len(stage7_ids)
        or not evaluation_rows
        or len(evaluation_ids) != len(evaluation_rows)
        or any(not item for item in evaluation_ids)
        or len(set(evaluation_ids)) != len(evaluation_ids)
        or len(source_ids) != len(evaluation_ids)
        or len(set(source_ids)) != len(source_ids)
        or set(source_ids) != set(evaluation_ids)
        or len(stage7_ids) != len(evaluation_ids)
        or set(stage7_ids) != set(evaluation_ids)
    ):
        raise _migration_error(
            "Stage 8 Cosmos3 evaluations do not exactly cover Stage 7 rollouts"
        )
    return {
        rollout_id: item for rollout_id, item in zip(evaluation_ids, evaluation_rows)
    }
