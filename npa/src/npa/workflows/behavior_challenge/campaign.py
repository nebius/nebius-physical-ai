"""Declare and compare immutable BEHAVIOR evaluation panels without cherry-picking."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import Any

from .protocol import SPLITS, UPSTREAM_COMMIT, WRAPPER, require_supported_upstream

_SHA256 = re.compile(r"[0-9a-f]{64}")
_IDENTIFIER = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}")
_ARTIFACT_NAME = re.compile(r"[a-z][a-z0-9_-]{0,63}")
_POLICY_ROLES = ("baseline", "candidate")
_PANELS = ("development", "reporting")
_SPLIT_BY_PANEL = {"development": "development", "reporting": "report"}


def canonical_digest(value: Any) -> str:
    """Hash one JSON value with stable key and whitespace normalization.

    Args:
        value: JSON-compatible value to hash.
    Returns:
        Lowercase SHA-256 digest of the canonical JSON bytes.
    Raises:
        TypeError: The value is not JSON serializable.
        ValueError: The value contains a non-finite number.
    """
    payload = json.dumps(
        value,
        allow_nan=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _exact_keys(value: object, expected: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) != expected:
        raise ValueError(f"{label} requires exactly {sorted(expected)}")
    return value


def _valid_sha256(value: object) -> bool:
    return isinstance(value, str) and _SHA256.fullmatch(value) is not None


def _artifact(name: str, value: object) -> dict[str, Any]:
    artifact = _exact_keys(value, {"sha256", "bytes"}, f"artifact {name}")
    if _ARTIFACT_NAME.fullmatch(name) is None:
        raise ValueError("Artifact names must be stable lowercase identifiers")
    if not _valid_sha256(artifact["sha256"]):
        raise ValueError(f"Artifact {name} requires a lowercase SHA-256")
    if type(artifact["bytes"]) is not int or artifact["bytes"] <= 0:
        raise ValueError(f"Artifact {name} requires a positive byte count")
    return {"sha256": artifact["sha256"], "bytes": artifact["bytes"]}


def freeze_policy_identity(
    policy_id: str, artifacts: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    """Freeze a policy as immutable checkpoint and serving artifact identities.

    Args:
        policy_id: Human-readable stable policy identifier.
        artifacts: Named artifact SHA-256 and byte identities. The checkpoint and
            serving entries are required.
    Returns:
        Canonical policy identity with its own digest.
    Raises:
        ValueError: The identifier or artifact identity is invalid.
    """
    if not isinstance(policy_id, str) or _IDENTIFIER.fullmatch(policy_id) is None:
        raise ValueError("policy_id must be a stable lowercase identifier")
    if not isinstance(artifacts, dict) or not {"checkpoint", "serving"}.issubset(
        artifacts
    ):
        raise ValueError("Policy identity requires checkpoint and serving artifacts")
    frozen = {name: _artifact(name, artifacts[name]) for name in sorted(artifacts)}
    payload = {
        "schema": "npa.behavior.policy-identity.v1",
        "policy_id": policy_id,
        "artifacts": frozen,
    }
    return {**payload, "identity_sha256": canonical_digest(payload)}


def validate_policy_identity(identity: object) -> dict[str, Any]:
    """Validate and normalize an immutable policy identity.

    Args:
        identity: Policy identity produced by :func:`freeze_policy_identity`.
    Returns:
        Newly constructed canonical policy identity.
    Raises:
        ValueError: The schema, fields, artifacts, or digest differ.
    """
    value = _exact_keys(
        identity,
        {"schema", "policy_id", "artifacts", "identity_sha256"},
        "policy identity",
    )
    if value["schema"] != "npa.behavior.policy-identity.v1":
        raise ValueError("Unsupported policy identity schema")
    expected = freeze_policy_identity(value["policy_id"], value["artifacts"])
    if value != expected:
        raise ValueError("Policy identity digest or canonical form differs")
    return expected


def _registry_tasks(tasks: object) -> tuple[str, ...]:
    if (
        not isinstance(tasks, (list, tuple))
        or len(tasks) != 100
        or not all(isinstance(task, str) and task for task in tasks)
        or len(set(tasks)) != 100
    ):
        raise ValueError("Expected the ordered official 100-task registry")
    return tuple(tasks)


def _selected_tasks(registry: tuple[str, ...], selected: object) -> tuple[str, ...]:
    if (
        not isinstance(selected, (list, tuple))
        or not selected
        or not all(isinstance(task, str) for task in selected)
        or len(set(selected)) != len(selected)
        or not set(selected).issubset(registry)
    ):
        raise ValueError("Select one or more distinct official tasks")
    selected_set = set(selected)
    return tuple(task for task in registry if task in selected_set)


def _case(split: str, task: str, index: int) -> dict[str, Any]:
    core = {
        "split": split,
        "task": task,
        "index": index,
        "instance_id": 301 + index,
        "rollout_id": 0,
    }
    return {**core, "case_id": canonical_digest(core)}


def _panel_cases(split: str, tasks: tuple[str, ...]) -> list[dict[str, Any]]:
    return [_case(split, task, index) for task in tasks for index in SPLITS[split]]


def _panel_payload(policy, registry, tasks, split, revision) -> dict[str, Any]:
    return {
        "schema": "npa.behavior.campaign-panel.v1",
        "upstream_commit": revision,
        "wrapper": WRAPPER,
        "registry_sha256": canonical_digest(list(registry)),
        "split": split,
        "selected_tasks": list(tasks),
        "task_count": len(tasks),
        "case_count": len(tasks) * len(SPLITS[split]),
        "policy": policy,
        "cases": _panel_cases(split, tasks),
    }


def declare_panel(
    policy: dict[str, Any],
    registry_tasks: list[str] | tuple[str, ...],
    selected_tasks: list[str] | tuple[str, ...],
    split: str,
    *,
    upstream_commit: str = UPSTREAM_COMMIT,
) -> dict[str, Any]:
    """Declare one reusable policy panel over fixed official cases.

    Args:
        policy: Immutable policy identity.
        registry_tasks: Ordered official 100-task registry.
        selected_tasks: Nonempty subset to include, canonicalized to registry order.
        split: Existing protocol split, development or report.
        upstream_commit: Supported official evaluator revision to freeze.
    Returns:
        Panel whose ID is independent of campaigns and worker partitions.
    Raises:
        ValueError: The policy, registry, task subset, or split is invalid.
    """
    frozen_policy = validate_policy_identity(policy)
    registry = _registry_tasks(registry_tasks)
    tasks = _selected_tasks(registry, selected_tasks)
    if split not in SPLITS:
        raise ValueError("Panel split must be development or report")
    revision = require_supported_upstream(upstream_commit)
    payload = _panel_payload(frozen_policy, registry, tasks, split, revision)
    return {**payload, "panel_id": canonical_digest(payload)}


def validate_panel(panel: object) -> dict[str, Any]:
    """Validate a reusable panel without requiring its parent campaign.

    Args:
        panel: Panel declaration to validate.
    Returns:
        The panel after structural and digest validation.
    Raises:
        ValueError: The panel schema, cases, policy, or digest differs.
    """
    keys = {
        "schema",
        "upstream_commit",
        "wrapper",
        "registry_sha256",
        "split",
        "selected_tasks",
        "task_count",
        "case_count",
        "policy",
        "cases",
        "panel_id",
    }
    value = _exact_keys(panel, keys, "panel")
    _validate_panel_fields(value)
    payload = {key: value[key] for key in keys - {"panel_id"}}
    if value["panel_id"] != canonical_digest(payload):
        raise ValueError("Panel ID differs from immutable panel content")
    return value


def _validate_panel_fields(panel: dict[str, Any]) -> None:
    if panel["schema"] != "npa.behavior.campaign-panel.v1":
        raise ValueError("Unsupported campaign panel schema")
    if panel["wrapper"] != WRAPPER:
        raise ValueError("Panel protocol identity differs")
    require_supported_upstream(panel["upstream_commit"])
    if not _valid_sha256(panel["registry_sha256"]):
        raise ValueError("Panel registry identity is invalid")
    policy = validate_policy_identity(panel["policy"])
    tasks = tuple(panel["selected_tasks"])
    if not tasks or len(set(tasks)) != len(tasks):
        raise ValueError("Panel tasks must be nonempty and distinct")
    if panel["split"] not in SPLITS:
        raise ValueError("Panel split is unsupported")
    cases = _panel_cases(panel["split"], tasks)
    if panel["task_count"] != len(tasks) or panel["case_count"] != len(cases):
        raise ValueError("Panel counts differ from selected tasks")
    if panel["cases"] != cases or panel["policy"] != policy:
        raise ValueError("Panel cases or policy are not canonical")


def _partition_payload(panel: dict[str, Any], worker_count: int) -> dict[str, Any]:
    workers = [
        {
            "worker_index": worker_index,
            "case_ids": [
                case["case_id"]
                for ordinal, case in enumerate(panel["cases"])
                if ordinal % worker_count == worker_index
            ],
        }
        for worker_index in range(worker_count)
    ]
    return {
        "schema": "npa.behavior.campaign-partition.v1",
        "panel_id": panel["panel_id"],
        "worker_count": worker_count,
        "workers": workers,
    }


def partition_panel(panel: dict[str, Any], worker_count: int) -> dict[str, Any]:
    """Assign every panel case once using deterministic round-robin ownership.

    Args:
        panel: Valid reusable panel declaration.
        worker_count: Positive worker count no larger than the case count.
    Returns:
        Immutable worker partition separate from the reusable panel ID.
    Raises:
        ValueError: The panel or worker count is invalid.
    """
    valid_panel = validate_panel(panel)
    if (
        type(worker_count) is not int
        or worker_count <= 0
        or worker_count > valid_panel["case_count"]
    ):
        raise ValueError("worker_count must be between one and the panel case count")
    payload = _partition_payload(valid_panel, worker_count)
    return {**payload, "partition_sha256": canonical_digest(payload)}


def validate_partition(partition: object, panel: dict[str, Any]) -> dict[str, Any]:
    """Validate exact deterministic worker ownership for a panel.

    Args:
        partition: Partition declaration to validate.
        panel: Referenced panel declaration.
    Returns:
        Canonical partition declaration.
    Raises:
        ValueError: Worker ownership, panel binding, or digest differs.
    """
    value = _exact_keys(
        partition,
        {"schema", "panel_id", "worker_count", "workers", "partition_sha256"},
        "partition",
    )
    if value["schema"] != "npa.behavior.campaign-partition.v1":
        raise ValueError("Unsupported campaign partition schema")
    expected = partition_panel(panel, value["worker_count"])
    if value != expected:
        raise ValueError("Partition differs from deterministic case ownership")
    return expected


def _campaign_panels(
    policies: dict[str, dict[str, Any]],
    registry: tuple[str, ...],
    tasks: tuple[str, ...],
    upstream_commit: str,
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        name: {
            role: declare_panel(
                policy,
                registry,
                tasks,
                _SPLIT_BY_PANEL[name],
                upstream_commit=upstream_commit,
            )
            for role, policy in policies.items()
        }
        for name in _PANELS
    }


def _campaign_partitions(
    panels: dict[str, dict[str, dict[str, Any]]], worker_count: int
) -> dict[str, dict[str, dict[str, Any]]]:
    return {
        name: {
            role: partition_panel(panel, worker_count)
            for role, panel in role_panels.items()
        }
        for name, role_panels in panels.items()
    }


def declare_campaign(
    campaign_id: str,
    registry_tasks: list[str] | tuple[str, ...],
    selected_tasks: list[str] | tuple[str, ...],
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    worker_count: int,
    minimum_paired_cases: int = 10,
    upstream_commit: str = UPSTREAM_COMMIT,
) -> dict[str, Any]:
    """Freeze a paired development/reporting campaign before execution.

    Args:
        campaign_id: Human-readable campaign identifier.
        registry_tasks: Ordered official 100-task registry.
        selected_tasks: Any nonempty subset of official tasks.
        baseline: Immutable baseline policy identity.
        candidate: Immutable candidate policy identity.
        worker_count: Worker ownership count for every policy panel.
        minimum_paired_cases: Reviewed evidence threshold for comparisons.
        upstream_commit: Supported official evaluator revision for every panel.
    Returns:
        Campaign declaration containing reusable panels and worker partitions.
    Raises:
        ValueError: Any identity, task, worker, or comparison threshold is invalid.
    """
    if not isinstance(campaign_id, str) or _IDENTIFIER.fullmatch(campaign_id) is None:
        raise ValueError("campaign_id must be a stable lowercase identifier")
    registry = _registry_tasks(registry_tasks)
    tasks = _selected_tasks(registry, selected_tasks)
    policies = _campaign_policies(baseline, candidate)
    panels = _campaign_panels(policies, registry, tasks, upstream_commit)
    _validate_paired_threshold(minimum_paired_cases, len(tasks) * 10)
    partitions = _campaign_partitions(panels, worker_count)
    payload = _campaign_payload(
        campaign_id, registry, tasks, policies, panels, partitions, minimum_paired_cases
    )
    return {**payload, "campaign_sha256": canonical_digest(payload)}


def _campaign_policies(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    policies = {
        "baseline": validate_policy_identity(baseline),
        "candidate": validate_policy_identity(candidate),
    }
    baseline_artifacts = canonical_digest(policies["baseline"]["artifacts"])
    candidate_artifacts = canonical_digest(policies["candidate"]["artifacts"])
    if baseline_artifacts == candidate_artifacts:
        raise ValueError("Baseline and candidate policy identities must differ")
    return policies


def _validate_paired_threshold(minimum: int, panel_cases: int) -> None:
    if type(minimum) is not int or not 1 <= minimum <= panel_cases:
        raise ValueError("minimum_paired_cases must fit inside a complete panel")


def _campaign_payload(
    campaign_id: str,
    registry: tuple[str, ...],
    tasks: tuple[str, ...],
    policies: dict[str, dict[str, Any]],
    panels: dict[str, dict[str, dict[str, Any]]],
    partitions: dict[str, dict[str, dict[str, Any]]],
    minimum_paired_cases: int,
) -> dict[str, Any]:
    return {
        "schema": "npa.behavior.campaign.v1",
        "campaign_id": campaign_id,
        "registry_sha256": canonical_digest(list(registry)),
        "selected_tasks": list(tasks),
        "task_count": len(tasks),
        "policy_roles": list(_POLICY_ROLES),
        "policies": policies,
        "panels": panels,
        "partitions": partitions,
        "minimum_paired_cases": minimum_paired_cases,
        "task_panel_comparison_supported": True,
        "full_challenge_competitive_claim_allowed": False,
    }


def validate_campaign(
    campaign: object, registry_tasks: list[str] | tuple[str, ...]
) -> dict[str, Any]:
    """Validate a campaign against the exact official task registry.

    Args:
        campaign: Campaign declaration to validate.
        registry_tasks: Ordered official 100-task registry.
    Returns:
        Newly constructed canonical campaign declaration.
    Raises:
        ValueError: The campaign differs from a fresh canonical declaration.
    """
    if not isinstance(campaign, dict):
        raise ValueError("Campaign must be an object")
    try:
        worker_count = campaign["partitions"]["development"]["baseline"]["worker_count"]
        revisions = {
            panel["upstream_commit"]
            for split in campaign["panels"].values()
            for panel in split.values()
        }
        if len(revisions) != 1:
            raise ValueError("Campaign panels must use one evaluator revision")
        expected = declare_campaign(
            campaign["campaign_id"],
            registry_tasks,
            campaign["selected_tasks"],
            campaign["policies"]["baseline"],
            campaign["policies"]["candidate"],
            worker_count=worker_count,
            minimum_paired_cases=campaign["minimum_paired_cases"],
            upstream_commit=revisions.pop(),
        )
    except (KeyError, TypeError) as exc:
        raise ValueError("Campaign is missing required fields") from exc
    if campaign != expected:
        raise ValueError("Campaign differs from its canonical declaration")
    return expected


def _case_from_record(record: dict[str, Any], split: str) -> dict[str, Any]:
    core = {
        "split": split,
        **{
            key: record.get(key)
            for key in ("task", "index", "instance_id", "rollout_id")
        },
    }
    if "split" in record and record["split"] != split:
        raise ValueError("Rollout split differs from its panel")
    return {**core, "case_id": canonical_digest(core)}


def _rollout_files(record: dict[str, Any], case: dict[str, Any]) -> dict[str, str]:
    stem = f"{case['task']}_{case['instance_id']}_{case['rollout_id']}"
    expected = {f"json/{stem}.json", f"videos/{stem}.mp4"}
    files = record.get("files")
    if (
        not isinstance(files, dict)
        or set(files) != expected
        or not all(_valid_sha256(value) for value in files.values())
    ):
        raise ValueError("Rollout requires exact JSON and video SHA-256 identities")
    return {name: files[name] for name in sorted(files)}


def _rollout_measurements(record: dict[str, Any]) -> dict[str, Any]:
    score = record.get("q_score")
    if (
        type(score) not in (int, float)
        or not math.isfinite(score)
        or not 0 <= score <= 1
    ):
        raise ValueError("Rollout q_score must be finite and between zero and one")
    frames = record.get("video_frames")
    if type(frames) is not int or frames <= 0:
        raise ValueError("Rollout requires a positive decoded video frame count")
    optional = (record.get("success"), record.get("steps"))
    if optional == (None, None):
        return {"q_score": float(score), "video_frames": frames}
    if (
        type(optional[0]) is not bool
        or type(optional[1]) is not int
        or optional[1] <= 0
    ):
        raise ValueError("Rollout success and positive steps must appear together")
    return {
        "q_score": float(score),
        "video_frames": frames,
        "success": optional[0],
        "steps": optional[1],
    }


def bind_inspected_rollout(
    panel: dict[str, Any], rollout: dict[str, Any]
) -> dict[str, Any]:
    """Bind one direct ``inspect_rollout`` result to an immutable panel.

    Args:
        panel: Reusable policy panel that prescribed the case.
        rollout: Direct result of ``artifacts.inspect_rollout``. Optional success
            and steps fields must be provided together.
    Returns:
        Case receipt bound to the panel and policy identity.
    Raises:
        ValueError: The result is not one of the prescribed cases or is malformed.
    """
    valid_panel = validate_panel(panel)
    case = _case_from_record(rollout, valid_panel["split"])
    expected = {item["case_id"]: item for item in valid_panel["cases"]}
    if case["case_id"] not in expected or case != expected[case["case_id"]]:
        raise ValueError("Rollout is not a prescribed panel case")
    payload = {
        "schema": "npa.behavior.campaign-case-receipt.v1",
        "panel_id": valid_panel["panel_id"],
        "policy_identity_sha256": valid_panel["policy"]["identity_sha256"],
        "case": case,
        **_rollout_measurements(rollout),
        "files": _rollout_files(rollout, case),
        "artifact_validation_contract": "npa.behavior.inspect-rollout.v1",
        "artifact_bytes_verified_by_aggregator": False,
    }
    return {**payload, "case_receipt_sha256": canonical_digest(payload)}


def _validate_case_receipt(panel: dict[str, Any], receipt: object) -> dict[str, Any]:
    if not isinstance(receipt, dict):
        raise ValueError("Case receipt must be an object")
    case = receipt.get("case")
    if not isinstance(case, dict):
        raise ValueError("Case receipt is missing its case")
    reconstructed = bind_inspected_rollout(
        panel,
        {
            **case,
            **{key: receipt[key] for key in ("q_score", "video_frames")},
            **{key: receipt[key] for key in ("success", "steps") if key in receipt},
            "files": receipt.get("files"),
        },
    )
    if receipt != reconstructed:
        raise ValueError("Case receipt differs from its canonical panel binding")
    return reconstructed


def aggregate_panel(
    panel: dict[str, Any], case_receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    """Aggregate a complete panel without selecting or imputing any case.

    Args:
        panel: Reusable panel declaration.
        case_receipts: Receipts bound from direct ``inspect_rollout`` results.
    Returns:
        Complete ordered panel aggregate with Q and optional success metrics.
    Raises:
        ValueError: Any prescribed case is missing, duplicated, extra, or invalid.
    """
    valid_panel = validate_panel(panel)
    receipts = [
        _validate_case_receipt(valid_panel, receipt) for receipt in case_receipts
    ]
    by_case = {receipt["case"]["case_id"]: receipt for receipt in receipts}
    expected_ids = [case["case_id"] for case in valid_panel["cases"]]
    if len(by_case) != len(receipts) or set(by_case) != set(expected_ids):
        raise ValueError(
            "Panel aggregation requires every prescribed case exactly once"
        )
    ordered = [by_case[case_id] for case_id in expected_ids]
    payload = _aggregate_payload(valid_panel, ordered)
    return {**payload, "aggregate_sha256": canonical_digest(payload)}


def _aggregate_payload(
    panel: dict[str, Any], receipts: list[dict[str, Any]]
) -> dict[str, Any]:
    scores = [receipt["q_score"] for receipt in receipts]
    has_success = ["success" in receipt for receipt in receipts]
    if any(has_success) and not all(has_success):
        raise ValueError(
            "Success and steps must be present for every panel case or none"
        )
    successes = (
        sum(receipt["success"] for receipt in receipts) if all(has_success) else None
    )
    return {
        "schema": "npa.behavior.campaign-panel-aggregate.v1",
        "panel_id": panel["panel_id"],
        "policy_identity_sha256": panel["policy"]["identity_sha256"],
        "split": panel["split"],
        "task_count": panel["task_count"],
        "case_count": len(receipts),
        "complete": True,
        "sum_q": math.fsum(scores),
        "mean_q": math.fsum(scores) / len(scores),
        "success_count": successes,
        "success_rate": successes / len(receipts) if successes is not None else None,
        "mean_steps": (
            math.fsum(receipt["steps"] for receipt in receipts) / len(receipts)
            if successes is not None
            else None
        ),
        "case_receipts": receipts,
        "artifact_validation_contract": "npa.behavior.inspect-rollout.v1",
        "artifact_bytes_verified_by_aggregator": False,
        "full_challenge_competitive_claim_allowed": False,
    }


def validate_panel_aggregate(
    aggregate: object, panel: dict[str, Any]
) -> dict[str, Any]:
    """Validate a complete aggregate against its reusable panel.

    Args:
        aggregate: Aggregate produced by :func:`aggregate_panel`.
        panel: Referenced immutable panel.
    Returns:
        Newly reconstructed canonical aggregate.
    Raises:
        ValueError: The aggregate is incomplete, altered, or for another panel.
    """
    if not isinstance(aggregate, dict):
        raise ValueError("Panel aggregate must be an object")
    receipts = aggregate.get("case_receipts")
    if not isinstance(receipts, list):
        raise ValueError("Panel aggregate requires case receipts")
    expected = aggregate_panel(panel, receipts)
    if aggregate != expected:
        raise ValueError("Panel aggregate differs from complete canonical evidence")
    return expected


def _paired_rows(
    baseline: dict[str, Any], candidate: dict[str, Any]
) -> list[dict[str, Any]]:
    rows = []
    for base, changed in zip(
        baseline["case_receipts"], candidate["case_receipts"], strict=True
    ):
        if base["case"] != changed["case"]:
            raise ValueError("Baseline and candidate case ordering differs")
        row = {
            "case": base["case"],
            "baseline_q": base["q_score"],
            "candidate_q": changed["q_score"],
            "q_delta": changed["q_score"] - base["q_score"],
        }
        if "success" in base and "success" in changed:
            row["baseline_success"] = base["success"]
            row["candidate_success"] = changed["success"]
        rows.append(row)
    return rows


def _paired_protocol(panels: dict) -> tuple[dict, dict]:
    baseline = validate_panel(panels["baseline"])
    candidate = validate_panel(panels["candidate"])
    fields = (
        "upstream_commit",
        "wrapper",
        "registry_sha256",
        "split",
        "selected_tasks",
        "cases",
    )
    if any(baseline[field] != candidate[field] for field in fields):
        raise ValueError("Paired panels must use one exact evaluator protocol")
    return baseline, candidate


def compare_panels(
    campaign: dict[str, Any],
    panel_name: str,
    baseline_aggregate: dict[str, Any],
    candidate_aggregate: dict[str, Any],
) -> dict[str, Any]:
    """Compare the campaign's complete paired baseline and candidate panels.

    Args:
        campaign: Valid campaign declaration.
        panel_name: Development or reporting panel name.
        baseline_aggregate: Complete reusable baseline aggregate.
        candidate_aggregate: Complete reusable candidate aggregate.
    Returns:
        Paired comparison with Q, optional success gains, and claim limits.
    Raises:
        ValueError: The panel name, aggregate binding, completeness, or pairing differs.
    """
    if panel_name not in _PANELS:
        raise ValueError("panel_name must be development or reporting")
    _validate_campaign_digest(campaign)
    panels = campaign.get("panels", {}).get(panel_name, {})
    if set(panels) != set(_POLICY_ROLES):
        raise ValueError("Campaign does not contain the requested paired panels")
    baseline_panel, candidate_panel = _paired_protocol(panels)
    if baseline_aggregate.get("panel_id") != baseline_panel["panel_id"]:
        raise ValueError("Baseline aggregate is not bound to the referenced panel")
    if candidate_aggregate.get("panel_id") != candidate_panel["panel_id"]:
        raise ValueError("Candidate aggregate is not bound to the referenced panel")
    baseline = validate_panel_aggregate(baseline_aggregate, baseline_panel)
    candidate = validate_panel_aggregate(candidate_aggregate, candidate_panel)
    rows = _paired_rows(baseline, candidate)
    threshold = campaign.get("minimum_paired_cases")
    if type(threshold) is not int or threshold <= 0:
        raise ValueError("Campaign paired-case threshold is invalid")
    payload = _comparison_payload(campaign, panel_name, baseline, candidate, rows)
    return {**payload, "comparison_sha256": canonical_digest(payload)}


def _validate_campaign_digest(campaign: dict[str, Any]) -> None:
    if campaign.get("schema") != "npa.behavior.campaign.v1":
        raise ValueError("Unsupported campaign schema")
    digest = campaign.get("campaign_sha256")
    payload = {
        key: value for key, value in campaign.items() if key != "campaign_sha256"
    }
    if not _valid_sha256(digest) or digest != canonical_digest(payload):
        raise ValueError("Campaign digest differs from its declaration")


def _comparison_payload(
    campaign: dict[str, Any],
    panel_name: str,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    rows: list[dict[str, Any]],
) -> dict[str, Any]:
    success_available = baseline["success_count"] is not None
    if success_available != (candidate["success_count"] is not None):
        raise ValueError("Success metrics must be present for both policies or neither")
    q_gain = candidate["mean_q"] - baseline["mean_q"]
    return {
        "schema": "npa.behavior.campaign-comparison.v1",
        "campaign_sha256": campaign.get("campaign_sha256"),
        "panel_name": panel_name,
        "split": baseline["split"],
        "baseline_panel_id": baseline["panel_id"],
        "candidate_panel_id": candidate["panel_id"],
        "paired_case_count": len(rows),
        "minimum_paired_cases": campaign["minimum_paired_cases"],
        "paired_case_threshold_met": len(rows) >= campaign["minimum_paired_cases"],
        "baseline_mean_q": baseline["mean_q"],
        "candidate_mean_q": candidate["mean_q"],
        "mean_q_gain": q_gain,
        "q_outcome": _q_outcome(q_gain),
        "baseline_success_rate": baseline["success_rate"],
        "candidate_success_rate": candidate["success_rate"],
        "success_rate_gain": (
            candidate["success_rate"] - baseline["success_rate"]
            if success_available
            else None
        ),
        "paired_cases": rows,
        "complete_panels_only": True,
        "no_case_selection_after_observation": True,
        "task_panel_comparison_supported": True,
        "claim_scope": _claim_scope(campaign),
        "full_challenge_competitive_claim_allowed": False,
    }


def _q_outcome(q_gain: float) -> str:
    if q_gain > 0:
        return "candidate"
    if q_gain < 0:
        return "baseline"
    return "tie"


def _claim_scope(campaign: dict[str, Any]) -> str:
    if campaign.get("task_count") == 100:
        return "complete_public_100_task_panel"
    return "focused_task_panel"
