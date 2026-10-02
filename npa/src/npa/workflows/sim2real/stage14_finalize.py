"""Bounded artifact materialization and final evidence for Sim2Real Stage 14."""

from __future__ import annotations

import argparse
import json
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.checkpoint_selection import (
    CHECKPOINT_DIGEST_ALIASES,
    CHECKPOINT_SIZE_ALIASES,
    CHECKPOINT_URI_ALIASES,
    GENERATOR_DIGEST_ALIASES,
    resolve_selected_checkpoint,
)
from npa.workflows.sim2real.workflow_io import (
    parse_json_object,
    publish_component_record,
    read_json,
    source_sha,
    storage,
    write_json,
)


def _field_evidence(
    payload: dict[str, Any],
    key: str,
    source: str,
) -> tuple[tuple[str, object], ...]:
    return ((source, payload[key]),) if key in payload else ()


def _aliased_evidence(
    *groups: tuple[dict[str, Any], str, tuple[str, ...]],
) -> tuple[tuple[str, object], ...]:
    evidence: tuple[tuple[str, object], ...] = ()
    for payload, source, keys in groups:
        for key in keys:
            evidence += _field_evidence(payload, key, f"{source}.{key}")
    return evidence


def _checkpoint_uri_evidence(
    selection: dict[str, Any],
    candidate: dict[str, Any],
    decision: dict[str, Any],
    decision_candidate: dict[str, Any],
) -> tuple[tuple[str, object], ...]:
    return _aliased_evidence(
        (selection, "checkpoint_selection", CHECKPOINT_URI_ALIASES),
        (candidate, "selected candidate", CHECKPOINT_URI_ALIASES),
        (decision, "outer_loop.decision", ("checkpoint_uri",)),
        (
            decision_candidate,
            "outer_loop.decision.candidate",
            CHECKPOINT_URI_ALIASES,
        ),
    )


def _candidate_digest_evidence(
    candidate: dict[str, Any],
    decision_candidate: dict[str, Any],
) -> tuple[tuple[str, object], ...]:
    return _aliased_evidence(
        (candidate, "selected candidate", CHECKPOINT_DIGEST_ALIASES),
        (
            decision_candidate,
            "outer_loop.decision.candidate",
            CHECKPOINT_DIGEST_ALIASES,
        ),
    )


def _candidate_size_evidence(
    candidate: dict[str, Any],
    decision_candidate: dict[str, Any],
) -> tuple[tuple[str, object], ...]:
    return _aliased_evidence(
        (candidate, "selected candidate", CHECKPOINT_SIZE_ALIASES),
        (
            decision_candidate,
            "outer_loop.decision.candidate",
            CHECKPOINT_SIZE_ALIASES,
        ),
    )


def _generator_digest_evidence(
    selection: dict[str, Any],
    candidate: dict[str, Any],
    decision_candidate: dict[str, Any],
) -> tuple[tuple[str, object], ...]:
    return _aliased_evidence(
        (selection, "checkpoint_selection", GENERATOR_DIGEST_ALIASES),
        (candidate, "selected candidate", GENERATOR_DIGEST_ALIASES),
        (
            decision_candidate,
            "outer_loop.decision.candidate",
            GENERATOR_DIGEST_ALIASES,
        ),
    )


def _candidate_leaf_evidence(
    candidate: dict[str, Any],
    decision_candidate: dict[str, Any],
) -> tuple[tuple[str, object], ...]:
    return _aliased_evidence(
        (
            candidate,
            "selected candidate",
            ("policy_checkpoint_identity", "identity"),
        ),
        (
            decision_candidate,
            "outer_loop.decision.candidate",
            ("policy_checkpoint_identity", "identity"),
        ),
    )


def _assert_promotion_checkpoint_uri(decision: dict[str, Any]) -> None:
    if decision.get("decision") != "promote_checkpoint":
        return
    from npa.workflows.sim2real.viz_contract import _checkpoint_uri

    if _checkpoint_uri(decision.get("checkpoint_uri")) is None:
        raise RuntimeError("Stage 14 promotion decision lacks a valid checkpoint URI")


def _assert_candidate_leaf_identity(
    candidate: dict[str, Any],
    decision_candidate: dict[str, Any],
    *,
    checkpoint_uri: str,
) -> None:
    expected = Path(checkpoint_uri).name
    for source, value in _candidate_leaf_evidence(candidate, decision_candidate):
        if not isinstance(value, str) or value != expected:
            raise RuntimeError(
                f"Stage 14 selected checkpoint identity: {source} "
                "disagrees with checkpoint URI"
            )


def _stage14_policy_metadata(
    evidence: dict[str, Any],
    decision: dict[str, Any],
    gold: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    from npa.workflows.sim2real_viz import _heldout_policy_metadata

    _assert_promotion_checkpoint_uri(decision)
    try:
        selection, candidate = resolve_selected_checkpoint(evidence)
    except ValueError as exc:
        raise RuntimeError(f"Stage 14 selected checkpoint identity: {exc}") from exc
    decision_candidate = decision.get("candidate", {})
    if not isinstance(decision_candidate, dict):
        raise RuntimeError("Stage 14 decision candidate must be an object")
    _assert_candidate_leaf_identity(
        candidate,
        decision_candidate,
        checkpoint_uri=selection["checkpoint_uri"],
    )
    metadata = _heldout_policy_metadata(
        gold,
        checkpoint_fallback=evidence["selected_checkpoint_uri"],
        checkpoint_sha256_fallback=selection["checkpoint_sha256"],
        checkpoint_size_fallback=selection["checkpoint_size_bytes"],
        checkpoint_uri_evidence=_checkpoint_uri_evidence(
            selection, candidate, decision, decision_candidate
        ),
        checkpoint_sha256_evidence=_candidate_digest_evidence(
            candidate, decision_candidate
        ),
        checkpoint_size_evidence=_candidate_size_evidence(
            candidate, decision_candidate
        ),
        generator_sha256_evidence=_generator_digest_evidence(
            selection, candidate, decision_candidate
        ),
        checkpoint_identity_evidence=_candidate_leaf_evidence(
            candidate, decision_candidate
        ),
    )
    return metadata, selection, candidate


def download_plan(
    *,
    root: str,
    outer_iteration: int,
    evidence: dict[str, Any],
    gold: dict[str, Any],
) -> list[tuple[str, str, bool]]:
    """Select only artifacts consumed by the final Rerun/MCAP encoders.

    Entries are ``(source URI, run-relative destination, is_prefix)``. Keeping the
    plan explicit prevents a visualization retry from mirroring an unbounded
    production run prefix onto node-local ephemeral storage.
    """

    outer = f"outer-{outer_iteration:02d}"
    entries: list[tuple[str, str, bool]] = [
        (
            f"{root}/stage_02_assets/consumed_robot_spec.json",
            "stage_02_assets/consumed_robot_spec.json",
            False,
        ),
        (f"{root}/augment/manifest.json", "augment/manifest.json", False),
        (f"{root}/augment/frames/", "augment/frames", True),
        (f"{root}/tokens/manifest.json", "tokens/manifest.json", False),
        (
            f"{root}/envs/manifest/split-manifest.json",
            "envs/manifest/split-manifest.json",
            False,
        ),
        (f"{root}/envs/train/envs.jsonl", "envs/train/envs.jsonl", False),
        (
            f"{root}/envs/validation/envs.jsonl",
            "envs/validation/envs.jsonl",
            False,
        ),
        (
            f"{root}/envs/gold-heldout/envs.jsonl",
            "envs/gold-heldout/envs.jsonl",
            False,
        ),
        (f"{root}/outer_loop/decision.json", "outer_loop/decision.json", False),
        (
            f"{root}/inner_loop/{outer}/evidence.json",
            f"inner_loop/{outer}/evidence.json",
            False,
        ),
        (
            f"{root}/eval/gold-heldout/{outer}/report.json",
            f"eval/gold-heldout/{outer}/report.json",
            False,
        ),
    ]
    for item in evidence.get("iterations") or []:
        inner_iteration = int(item.get("iteration") or 0)
        if inner_iteration < 1:
            raise RuntimeError("Stage 14 evidence contains an invalid inner iteration")
        inner = f"iter-{inner_iteration:02d}"
        entries.extend(
            [
                (
                    str(item.get("actions_uri") or ""),
                    f"actions/train/{outer}/{inner}",
                    True,
                ),
                (
                    str(item.get("vlm_eval_uri") or ""),
                    f"vlm_eval/train/{outer}/{inner}/evaluations",
                    True,
                ),
                (
                    str(item.get("signal_uri") or ""),
                    f"vlm_eval/train/{outer}/{inner}/signals",
                    True,
                ),
            ]
        )
    lineage = dict(gold.get("render_lineage") or {})
    render_uri = str(lineage.get("canonical_s3_uri") or "")
    raw_render_relative = lineage.get("local_relative_dir")
    render_relative = (
        raw_render_relative if isinstance(raw_render_relative, str) else ""
    )
    render_path = Path(render_relative)
    expected_relative = f"eval/gold-heldout/{outer}/renders"
    expected_uri = f"{root.rstrip('/')}/{expected_relative}/"
    recorded_outer = gold.get("outer_iteration")
    if (
        gold.get("evaluation_split") != "gold_heldout"
        or not isinstance(recorded_outer, int)
        or isinstance(recorded_outer, bool)
        or recorded_outer != outer_iteration
        or lineage.get("evaluation_split") != "gold_heldout"
        or render_uri != expected_uri
        or render_relative != expected_relative
        or render_relative != render_relative.strip()
        or render_path.is_absolute()
        or render_path == Path(".")
        or ".." in render_path.parts
    ):
        raise RuntimeError("Stage 14 gold report lacks safe canonical render lineage")
    entries.append((render_uri, render_relative, True))

    seen: set[tuple[str, str, bool]] = set()
    plan: list[tuple[str, str, bool]] = []
    root_prefix = root.rstrip("/") + "/"
    for source, destination, is_prefix in entries:
        normalized = source.rstrip("/") + "/" if is_prefix else source
        destination_path = Path(destination)
        if (
            not source.startswith(root_prefix)
            or normalized == root_prefix
            or not destination
            or destination_path.is_absolute()
            or destination_path == Path(".")
            or ".." in destination_path.parts
        ):
            raise RuntimeError(
                f"Stage 14 rejected unscoped artifact selection: {source}"
            )
        entry = (normalized, destination, is_prefix)
        if entry not in seen:
            seen.add(entry)
            plan.append(entry)
    return plan


def materialize_plan(plan: list[tuple[str, str, bool]], *, local: Path) -> None:
    client = storage()
    root = local.resolve()
    for source, destination, is_prefix in plan:
        target = local / destination
        resolved = target.resolve()
        if resolved == root or not resolved.is_relative_to(root):
            raise RuntimeError(f"Stage 14 rejected unsafe local path: {destination}")
        if is_prefix:
            client.download_directory(source, str(target))
        else:
            client.download_file(source, str(target))


@dataclass(frozen=True)
class _Stage14State:
    args: argparse.Namespace
    root: str
    work: Path
    local: Path
    evidence: dict[str, Any]
    gold: dict[str, Any]
    rrd_uri: str
    mcap_uri: str
    report_uri: str


def _localize_iteration_paths(
    evidence: dict[str, Any],
    local: Path,
    *,
    outer_iteration: int,
) -> None:
    outer = f"outer-{outer_iteration:02d}"
    for item in evidence.get("iterations") or []:
        inner = f"iter-{int(item.get('iteration') or 1):02d}"
        item["actions_dir"] = str(local / "actions" / "train" / outer / inner)
        item["vlm_eval_dir"] = str(
            local / "vlm_eval" / "train" / outer / inner / "evaluations"
        )
        item["signal_dir"] = str(
            local / "vlm_eval" / "train" / outer / inner / "signals"
        )


def _stage14_state(
    args: argparse.Namespace,
    root: str,
    work: Path,
    local: Path,
    evidence: dict[str, Any],
    gold: dict[str, Any],
) -> _Stage14State:
    return _Stage14State(
        args=args,
        root=root,
        work=work,
        local=local,
        evidence=evidence,
        gold=gold,
        rrd_uri=f"{root}/reports/sim2real.rrd",
        mcap_uri=f"{root}/reports/sim2real.mcap",
        report_uri=f"{root}/reports/sim2real-report.json",
    )


def _materialize_stage14(
    args: argparse.Namespace,
    *,
    root: str,
    work: Path,
) -> _Stage14State:
    outer = args.outer_iteration
    evidence = read_json(
        f"{root}/inner_loop/outer-{outer:02d}/evidence.json",
        directory=work / "evidence-input",
    )
    gold = read_json(
        f"{root}/eval/gold-heldout/outer-{outer:02d}/report.json",
        directory=work / "gold-input",
    )
    local = work / "run"
    materialize_plan(
        download_plan(
            root=root,
            outer_iteration=outer,
            evidence=evidence,
            gold=gold,
        ),
        local=local,
    )
    _localize_iteration_paths(evidence, local, outer_iteration=outer)
    gold["local_renders_dir"] = str(
        local / str((gold.get("render_lineage") or {}).get("local_relative_dir") or "")
    )
    return _stage14_state(args, root, work, local, evidence, gold)


def _load_component_records(state: _Stage14State) -> list[dict[str, Any]]:
    publish_component_record(
        root_uri=state.root,
        stage=14,
        name="stage_14_rerun_viz",
        tier="WORKS",
        evidence=(
            "Finalization is executing in the standard workflow runtime and will "
            "publish full Rerun and MCAP evidence."
        ),
        artifacts={
            "rrd": state.rrd_uri,
            "mcap": state.mcap_uri,
            "report": state.report_uri,
        },
    )
    components = [
        read_json(
            f"{state.root}/components/stage_{stage:02d}.json",
            directory=state.work / f"component-{stage:02d}",
        )
        for stage in range(1, 15)
    ]
    if [item["stage"] for item in components] != list(range(1, 15)):
        raise RuntimeError("Stage 14 requires exactly 14 ordered ComponentRecords")
    if components[11]["tier"] != "SEAM" or any(
        item["tier"] != "WORKS" for index, item in enumerate(components) if index != 11
    ):
        raise RuntimeError(
            "ComponentRecord tiers violate the 13 WORKS + Stage 12 SEAM contract"
        )
    return components


def _stage14_report_payload(
    state: _Stage14State,
    components: list[dict[str, Any]],
    decision: dict[str, Any],
    selection: dict[str, Any],
    selected_candidate: dict[str, Any],
    robot_contract: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "npa.sim2real.e2e_report.v1",
        "run_id": state.args.run_id,
        "source_sha": source_sha(),
        "status": "completed",
        "architecture": "npa.workflow/v0.0.1_compositional_standard_runtime",
        "component_records": components,
        "outer_loop": {"decision": decision, "latest_heldout_report": state.gold},
        "checkpoint_selection": selection,
        "selected_checkpoint_candidate": selected_candidate,
        "stage8_evaluator_usage": [
            item.get("evaluator_usage")
            for item in state.evidence.get("iterations") or []
        ],
        "strict_gold_success_rate": float(state.gold.get("success_rate") or 0.0),
        "policy_quality_is_pipeline_gate": False,
        "robot_contract": robot_contract,
        "embodiment": state.gold.get("embodiment", {}),
        "rrd_uri": state.rrd_uri,
        "mcap_uri": state.mcap_uri,
    }


def _build_stage14_report(
    state: _Stage14State,
    components: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    decision = parse_json_object(
        (state.local / "outer_loop" / "decision.json").read_text(),
        source="Stage 14 outer-loop decision",
    )
    heldout_policy_metadata, selection, selected_candidate = _stage14_policy_metadata(
        state.evidence, decision, state.gold
    )
    robot_contract = parse_json_object(
        (state.local / "stage_02_assets" / "consumed_robot_spec.json").read_text(),
        source="Stage 14 consumed robot contract",
    )
    report = _stage14_report_payload(
        state,
        components,
        decision,
        selection,
        selected_candidate,
        robot_contract,
    )
    reports = state.local / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "sim2real-report.json").write_text(json.dumps(report, indent=2))
    return report, heldout_policy_metadata, reports


def _stage14_run_metadata(
    state: _Stage14State,
    report: dict[str, Any],
    heldout_policy_metadata: dict[str, Any],
) -> dict[str, Any]:
    selection = report["checkpoint_selection"]
    return {
        "run_id": state.args.run_id,
        "artifact_root": state.root,
        "policy_checkpoint": state.evidence["selected_checkpoint_uri"],
        "policy_checkpoint_sha256": selection.get("checkpoint_sha256", ""),
        "policy_checkpoint_size_bytes": selection.get(
            "checkpoint_size_bytes",
            state.gold.get("policy_checkpoint_size_bytes", 0),
        ),
        **heldout_policy_metadata,
        "rrd_s3_uri": state.rrd_uri,
        "embodiment": state.gold.get("embodiment", {}),
    }


def _emit_stage14_outputs(
    state: _Stage14State,
    components: list[dict[str, Any]],
    report: dict[str, Any],
    heldout_policy_metadata: dict[str, Any],
    reports: Path,
) -> tuple[Any, Any]:
    from npa.workflows.sim2real_viz import (
        emit_sim2real_mcap,
        emit_sim2real_rerun,
    )

    decision = report["outer_loop"]["decision"]
    rrd = emit_sim2real_rerun(
        local_dir=state.local,
        inner_evidence=state.evidence,
        heldout_report=state.gold,
        stage_components=components,
        outer_history=[
            {
                "decision": decision,
                "checkpoint_uri": state.evidence.get("selected_checkpoint_uri", ""),
            }
        ],
        run_metadata=_stage14_run_metadata(state, report, heldout_policy_metadata),
        output_rrd=reports / "sim2real.rrd",
        allow_progress_only=False,
    )
    mcap = emit_sim2real_mcap(
        local_dir=state.local,
        inner_evidence=state.evidence,
        heldout_report=state.gold,
        output_mcap=reports / "sim2real.mcap",
    )
    return rrd, mcap


def _publish_stage14_outputs(
    state: _Stage14State,
    report: dict[str, Any],
    reports: Path,
    rrd: Any,
    mcap: Any,
) -> None:
    for filename, uri in (
        ("sim2real.rrd", state.rrd_uri),
        ("sim2real.mcap", state.mcap_uri),
    ):
        path = reports / filename
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Stage 14 produced empty {filename}")
        storage().upload_file(str(path), uri)
    final_record = publish_component_record(
        root_uri=state.root,
        stage=14,
        name="stage_14_rerun_viz",
        tier="WORKS",
        evidence="Published independently decodable Rerun and MCAP recordings with multi-camera gold footage, policy, progress, and evaluation evidence.",
        artifacts={
            "rrd": state.rrd_uri,
            "rrd_bytes": (reports / "sim2real.rrd").stat().st_size,
            "rrd_summary": rrd.to_dict(),
            "mcap": state.mcap_uri,
            "mcap_bytes": (reports / "sim2real.mcap").stat().st_size,
            "mcap_summary": mcap.to_dict(),
            "report": state.report_uri,
        },
    )
    report["component_records"][-1] = final_record
    write_json(state.report_uri, report, directory=state.work / "final-report")


def finalize_in_work(args: argparse.Namespace, *, root: str, work: Path) -> None:
    state = _materialize_stage14(args, root=root, work=work)
    components = _load_component_records(state)
    report, policy_metadata, reports = _build_stage14_report(state, components)
    rrd, mcap = _emit_stage14_outputs(
        state, components, report, policy_metadata, reports
    )
    _publish_stage14_outputs(state, report, reports, rrd, mcap)


def finalize(args: argparse.Namespace) -> None:
    root = str(args.root_uri).rstrip("/")
    with tempfile.TemporaryDirectory(prefix="npa-s2r-stage-14-") as directory:
        finalize_in_work(args, root=root, work=Path(directory))
