"""Canonical artifact authority for compositional Sim2Real ComponentRecords."""

from __future__ import annotations

from typing import Any

from npa.workflows.sim2real.checkpoint_selection import resolve_selected_checkpoint
from npa.workflows.sim2real.decision_authority import gold_report_sha256
from npa.workflows.sim2real.workflow_io import validate_component_record


COMPONENT_CONTRACTS: dict[int, tuple[str, str, tuple[str, ...]]] = {
    1: ("stage_01_trigger", "WORKS", ("trigger", "task_contract_digest")),
    2: (
        "stage_02_assets",
        "WORKS",
        ("task_contract", "scene", "robot_contract"),
    ),
    3: ("stage_03_augment", "WORKS", ("manifest", "result", "frames")),
    4: (
        "stage_04_envs_raw",
        "WORKS",
        ("raw_envs", "shard_provenance", "shard_count", "lane_records"),
    ),
    5: (
        "stage_05_envs_train",
        "WORKS",
        ("manifest", "train_envs", "validation_envs", "gold_envs"),
    ),
    6: ("stage_06_tokens", "WORKS", ("tokens", "split_digest")),
    7: ("stage_07_actions_train", "WORKS", ("prefix", "component_invocation")),
    8: ("stage_08_vlm_eval_train", "WORKS", ("result", "evaluator_usage")),
    9: (
        "stage_09_training_signal",
        "WORKS",
        ("evidence", "checkpoint", "validation_report"),
    ),
    10: (
        "stage_10_eval_heldout",
        "WORKS",
        (
            "report",
            "gold_report_sha256",
            "checkpoint",
            "renders",
            "render_lineage",
        ),
    ),
    11: ("stage_11_outer_loop", "WORKS", ("decision", "gold_report")),
    12: ("stage_12_external_validation", "SEAM", ("seam",)),
    13: ("stage_13_retrigger", "WORKS", ("record", "decision")),
}

COMPONENT_URI_KEYS: dict[int, tuple[str, ...]] = {
    1: ("trigger",),
    2: ("task_contract", "scene", "robot_contract"),
    3: ("manifest", "result", "frames"),
    4: ("raw_envs",),
    5: ("manifest", "train_envs", "validation_envs", "gold_envs"),
    6: ("tokens",),
    7: ("prefix",),
    8: ("result",),
    9: ("evidence", "checkpoint", "validation_report"),
    10: ("report", "checkpoint", "renders"),
    11: ("decision", "gold_report"),
    12: ("seam",),
    13: ("record", "decision"),
}


def _require_source_sha(value: object) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 40
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(
            "canonical component authority requires a lowercase 40-character source SHA"
        )
    return value


def _authority_uris(
    root: str,
    evidence: dict[str, Any],
    gold: dict[str, Any],
) -> dict[int, dict[str, str]]:
    selection, _candidate = resolve_selected_checkpoint(evidence)
    outer = evidence.get("outer_iteration")
    iterations = evidence.get("iterations")
    if type(outer) is not int or not isinstance(iterations, list) or not iterations:
        raise ValueError("ComponentRecord authority lacks terminal loop evidence")
    inner_values = [
        item.get("iteration") for item in iterations if isinstance(item, dict)
    ]
    if not inner_values or any(type(value) is not int for value in inner_values):
        raise ValueError("ComponentRecord authority has invalid inner iterations")
    inner = max(inner_values)
    lane = f"{root}/vlm_eval/train/outer-{outer:02d}/iter-{inner:02d}/"
    gold_report = f"{root}/eval/gold-heldout/outer-{outer:02d}/report.json"
    renders = str((gold.get("render_lineage") or {}).get("canonical_s3_uri") or "")
    return {
        1: {"trigger": f"{root}/stage_01_trigger/trigger.json"},
        2: {
            "task_contract": f"{root}/stage_02_assets/task-contract.json",
            "scene": f"{root}/stage_02_assets/consumed_scene_spec.json",
            "robot_contract": f"{root}/stage_02_assets/consumed_robot_spec.json",
        },
        3: {
            "manifest": f"{root}/augment/manifest.json",
            "result": f"{root}/augment/cosmos2-transfer-result.json",
            "frames": f"{root}/augment/frames/",
        },
        4: {"raw_envs": f"{root}/envs/raw/"},
        5: {
            "manifest": f"{root}/envs/manifest/split-manifest.json",
            "train_envs": f"{root}/envs/train/envs.jsonl",
            "validation_envs": f"{root}/envs/validation/envs.jsonl",
            "gold_envs": f"{root}/envs/gold-heldout/envs.jsonl",
        },
        6: {"tokens": f"{root}/tokens/manifest.json"},
        7: {"prefix": (f"{root}/actions/train/outer-{outer:02d}/iter-{inner:02d}/")},
        8: {"result": lane + "cosmos3.json"},
        9: {
            "evidence": f"{root}/inner_loop/outer-{outer:02d}/evidence.json",
            "checkpoint": selection["checkpoint_uri"],
            "validation_report": selection["validation_report_uri"],
        },
        10: {
            "report": gold_report,
            "checkpoint": selection["checkpoint_uri"],
            "renders": renders,
        },
        11: {
            "decision": f"{root}/outer_loop/decision.json",
            "gold_report": gold_report,
        },
        12: {"seam": f"{root}/external_validation/seam.json"},
        13: {
            "record": f"{root}/retrigger/record.json",
            "decision": f"{root}/outer_loop/decision.json",
        },
    }


def _assert_artifact_scope(
    root: str,
    stage: int,
    component: dict[str, Any],
    expected_uris: dict[int, dict[str, str]],
    gold: dict[str, Any],
) -> None:
    root_prefix = root.rstrip("/") + "/"
    for key in COMPONENT_URI_KEYS[stage]:
        uri = component["artifacts"][key]
        suffix = uri[len(root_prefix) :] if isinstance(uri, str) else ""
        parts = suffix.rstrip("/").split("/")
        if (
            not isinstance(uri, str)
            or not uri.startswith(root_prefix)
            or not suffix
            or any(token in suffix for token in ("\\", "%", "?", "#"))
            or any(part in {"", ".", ".."} for part in parts)
            or uri != expected_uris[stage][key]
        ):
            raise ValueError(
                f"Stage {stage} ComponentRecord {key} is not its canonical artifact"
            )
    if stage == 10 and component["artifacts"].get("render_lineage") != gold.get(
        "render_lineage"
    ):
        raise ValueError("Stage 10 ComponentRecord render lineage is not canonical")
    if stage == 10 and component["artifacts"].get(
        "gold_report_sha256"
    ) != gold_report_sha256(gold):
        raise ValueError("Stage 10 ComponentRecord gold report digest is stale")
    if stage == 4:
        artifacts = component["artifacts"]
        shard_count = artifacts.get("shard_count")
        if (
            artifacts.get("execution_mode") != "standard_npa_workflow_parallel_join"
            or type(shard_count) is not int
            or shard_count <= 0
            or artifacts.get("lane_count") != shard_count
            or not isinstance(artifacts.get("shard_provenance"), list)
            or len(artifacts["shard_provenance"]) != shard_count
            or not isinstance(artifacts.get("lane_records"), list)
            or len(artifacts["lane_records"]) != shard_count
        ):
            raise ValueError("Stage 4 ComponentRecord parallel join is incomplete")


def validate_component_records(
    components: list[dict[str, Any]],
    *,
    root: str,
    evidence: dict[str, Any],
    gold: dict[str, Any],
    expected_source_sha: str,
) -> None:
    """Validate exact artifact and provenance authority for Stages 1 through 13."""

    expected_source_sha = _require_source_sha(expected_source_sha)
    if len(components) != 13:
        raise ValueError("canonical ComponentRecord chain must contain 13 stages")
    expected_uris = _authority_uris(root, evidence, gold)
    for stage, component in enumerate(components, start=1):
        name, tier, required = COMPONENT_CONTRACTS[stage]
        validate_component_record(
            component,
            expected_stage=stage,
            expected_name=name,
            expected_tier=tier,
            required_artifacts=required,
            expected_source_sha=expected_source_sha,
        )
        _assert_artifact_scope(root, stage, component, expected_uris, gold)


def validate_stage14_component_record(
    record: dict[str, Any],
    report: dict[str, Any],
    *,
    expected_source_sha: str,
) -> None:
    """Validate one regenerated Stage 14 record against its sealed report."""

    expected_source_sha = _require_source_sha(expected_source_sha)
    required = ("rrd", "report")
    if report.get("mcap_uri"):
        required += ("mcap",)
    validate_component_record(
        record,
        expected_stage=14,
        expected_name="stage_14_rerun_viz",
        expected_tier="WORKS",
        required_artifacts=required,
        expected_source_sha=expected_source_sha,
    )
    artifacts = record["artifacts"]
    if (
        artifacts.get("rrd") != report.get("rrd_uri")
        or artifacts.get("report") != report.get("report_uri")
        or artifacts.get("mcap", "") != report.get("mcap_uri", "")
    ):
        raise ValueError("Stage 14 ComponentRecord artifact authority is stale")
