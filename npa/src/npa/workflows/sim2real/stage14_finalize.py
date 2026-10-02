"""Bounded artifact materialization and final evidence for Sim2Real Stage 14."""

from __future__ import annotations

import argparse
import hashlib
import json
import secrets
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.checkpoint_selection import (
    CHECKPOINT_DIGEST_ALIASES,
    CHECKPOINT_SIZE_ALIASES,
    CHECKPOINT_URI_ALIASES,
    GENERATOR_DIGEST_ALIASES,
    resolve_selected_checkpoint,
    resolve_run_scoped_checkpoint,
)
from npa.workflows.sim2real.component_authority import (
    COMPONENT_CONTRACTS as _COMPONENT_CONTRACTS,  # noqa: F401 - compatibility
    COMPONENT_URI_KEYS as _COMPONENT_URI_KEYS,  # noqa: F401 - compatibility
    stage14_report_authority_sha256,
    validate_component_records,
    validate_remote_stage4_authority,
)
from npa.workflows.sim2real.decision_authority import (
    gold_report_sha256,
    validate_stage11_decision,
)
from npa.workflows.sim2real.publication import (
    MutablePublicationTransaction,
    RemoteObjectSnapshot,
    recover_interrupted_publication,
    remote_object_snapshot,
    remote_object_version,
    upload_immutable_file,
)
from npa.workflows.sim2real.stage10_execution import (
    materialize_verified_render_snapshot,
)
from npa.workflows.sim2real.stage10_authority import validate_stage10_input_scope
from npa.workflows.sim2real.workflow_io import (
    build_component_record,
    component_record_history_uri,
    parse_json_object,
    publish_built_component_history,
    publish_built_component_pointer,
    read_json,
    source_sha,
    storage,
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


def _resolve_stage14_selection(
    evidence: dict[str, Any],
    run_root: str | None,
    run_id: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        if run_root is None:
            selection, candidate = resolve_selected_checkpoint(evidence)
        else:
            outer_iteration = evidence.get("outer_iteration")
            if type(outer_iteration) is not int or run_id is None:
                raise RuntimeError("Stage 14 selected outer iteration is invalid")
            validate_stage10_input_scope(
                argparse.Namespace(
                    run_id=run_id,
                    outer_iteration=outer_iteration,
                ),
                root=run_root,
                evidence=evidence,
            )
            selection, candidate = resolve_run_scoped_checkpoint(
                evidence,
                run_root=run_root,
                run_id=run_id,
            )
    except (RuntimeError, ValueError) as exc:
        raise RuntimeError(f"Stage 14 selected checkpoint identity: {exc}") from exc
    return selection, candidate


def _validated_decision_candidate(
    decision: dict[str, Any],
    candidate: dict[str, Any],
    selection: dict[str, Any],
) -> dict[str, Any]:
    decision_candidate = decision.get("candidate", {})
    if not isinstance(decision_candidate, dict):
        raise RuntimeError("Stage 14 decision candidate must be an object")
    _assert_candidate_leaf_identity(
        candidate,
        decision_candidate,
        checkpoint_uri=selection["checkpoint_uri"],
    )
    return decision_candidate


def _assert_stage14_decision_authority(
    evidence: dict[str, Any],
    decision: dict[str, Any],
    gold: dict[str, Any],
    selection: dict[str, Any],
    run_root: str | None,
    run_id: str | None,
    expected_threshold: float,
    expected_early_exit: bool,
    expected_gold_report_sha256: str | None,
) -> None:
    if run_root is None or run_id is None:
        return
    outer_iteration = evidence.get("outer_iteration")
    if type(outer_iteration) is not int:
        raise RuntimeError("Stage 14 selected outer iteration is invalid")
    try:
        validate_stage11_decision(
            decision,
            run_id=run_id,
            root=run_root,
            outer_iteration=outer_iteration,
            gold_report=gold,
            checkpoint_uri=selection["checkpoint_uri"],
            expected_threshold=expected_threshold,
            expected_early_exit=expected_early_exit,
            gold_report_bytes_sha256=expected_gold_report_sha256,
        )
    except ValueError as exc:
        raise RuntimeError(f"Stage 14 decision authority: {exc}") from exc


def _heldout_metadata(
    evidence: dict[str, Any],
    decision: dict[str, Any],
    gold: dict[str, Any],
    selection: dict[str, Any],
    candidate: dict[str, Any],
    decision_candidate: dict[str, Any],
) -> dict[str, Any]:
    from npa.workflows.sim2real_viz import _heldout_policy_metadata

    return _heldout_policy_metadata(
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


def _stage14_policy_metadata(
    evidence: dict[str, Any],
    decision: dict[str, Any],
    gold: dict[str, Any],
    *,
    run_root: str | None = None,
    run_id: str | None = None,
    expected_threshold: float = 0.5,
    expected_early_exit: bool = False,
    expected_gold_report_sha256: str | None = None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _assert_promotion_checkpoint_uri(decision)
    selection, candidate = _resolve_stage14_selection(evidence, run_root, run_id)
    decision_candidate = _validated_decision_candidate(decision, candidate, selection)
    _assert_stage14_decision_authority(
        evidence,
        decision,
        gold,
        selection,
        run_root,
        run_id,
        expected_threshold,
        expected_early_exit,
        expected_gold_report_sha256,
    )
    metadata = _heldout_metadata(
        evidence, decision, gold, selection, candidate, decision_candidate
    )
    return metadata, selection, candidate


def _assert_publishable_policy_metadata(
    metadata: dict[str, Any],
    gold: dict[str, Any],
) -> None:
    missing = [
        field
        for field in (
            "heldout_policy_loaded_for_inference",
            "heldout_policy_identity_verified",
            "heldout_policy_learned_actor_only",
        )
        if metadata.get(field) is not True
    ]
    if gold.get("deployable_policy_eval") is not True:
        missing.append("deployable_policy_eval")
    if missing:
        raise RuntimeError(
            "Stage 14 refuses to publish nondeployable or unverified learned-policy "
            f"evidence; required true fields: {', '.join(missing)}"
        )


def _iteration_artifact_uris(
    root: str,
    *,
    outer_iteration: int,
    inner_iteration: int,
) -> tuple[str, str, str]:
    outer = f"outer-{outer_iteration:02d}"
    inner = f"iter-{inner_iteration:02d}"
    lane = f"{root}/vlm_eval/train/{outer}/{inner}/"
    return (
        f"{root}/actions/train/{outer}/{inner}/",
        lane + "evaluations/",
        lane + "signals/",
    )


def _base_download_entries(root: str, outer: str) -> list[tuple[str, str, bool]]:
    relative_files = (
        "stage_02_assets/consumed_robot_spec.json",
        "augment/manifest.json",
        "tokens/manifest.json",
        "envs/manifest/split-manifest.json",
        "envs/train/envs.jsonl",
        "envs/validation/envs.jsonl",
        "envs/gold-heldout/envs.jsonl",
        "outer_loop/decision.json",
        f"inner_loop/{outer}/evidence.json",
        f"eval/gold-heldout/{outer}/report.json",
    )
    entries = [(f"{root}/{path}", path, False) for path in relative_files]
    entries.insert(2, (f"{root}/augment/frames/", "augment/frames", True))
    return entries


def _inner_iteration(item: object, seen: set[int]) -> tuple[int, dict[str, Any]]:
    if not isinstance(item, dict):
        raise RuntimeError("Stage 14 iteration evidence must be an object")
    iteration = item.get("iteration")
    if type(iteration) is not int or iteration < 1:
        raise RuntimeError("Stage 14 evidence contains an invalid inner iteration")
    if iteration in seen:
        raise RuntimeError("Stage 14 evidence contains a duplicate inner iteration")
    seen.add(iteration)
    return iteration, item


def _iteration_download_entries(
    root: str,
    outer_iteration: int,
    iterations: object,
) -> list[tuple[str, str, bool]]:
    if not isinstance(iterations, list):
        raise RuntimeError("Stage 14 evidence iterations must be a list")
    outer = f"outer-{outer_iteration:02d}"
    entries: list[tuple[str, str, bool]] = []
    seen: set[int] = set()
    for raw_item in iterations:
        inner_iteration, item = _inner_iteration(raw_item, seen)
        expected = _iteration_artifact_uris(
            root,
            outer_iteration=outer_iteration,
            inner_iteration=inner_iteration,
        )
        actual = (
            item.get("actions_uri"),
            item.get("vlm_eval_uri"),
            item.get("signal_uri"),
        )
        if actual != expected:
            raise RuntimeError(
                "Stage 14 iteration artifact URIs do not match the requested run "
                "and outer/inner iteration"
            )
        inner = f"iter-{inner_iteration:02d}"
        destinations = (
            f"actions/train/{outer}/{inner}",
            f"vlm_eval/train/{outer}/{inner}/evaluations",
            f"vlm_eval/train/{outer}/{inner}/signals",
        )
        entries.extend(zip(expected, destinations, (True, True, True), strict=True))
    return entries


def _render_attempt_uri(
    root: str,
    outer_iteration: int,
    attempt_tag: object,
) -> str:
    outer = f"outer-{outer_iteration:02d}"
    if attempt_tag in (None, ""):
        return f"{root.rstrip('/')}/eval/gold-heldout/{outer}/renders/"
    prefix = f"gold-o{outer_iteration:02d}-attempt-"
    if (
        not isinstance(attempt_tag, str)
        or not attempt_tag.startswith(prefix)
        or len(attempt_tag) != len(prefix) + 32
        or any(char not in "0123456789abcdef" for char in attempt_tag[len(prefix) :])
    ):
        raise RuntimeError("Stage 14 gold report has invalid render attempt lineage")
    return (
        f"{root.rstrip('/')}/eval/gold-heldout/{outer}/attempts/{attempt_tag}/renders/"
    )


def _render_download_entry(
    root: str,
    outer_iteration: int,
    gold: dict[str, Any],
) -> tuple[str, str, bool]:
    outer = f"outer-{outer_iteration:02d}"
    lineage = dict(gold.get("render_lineage") or {})
    render_uri = str(lineage.get("canonical_s3_uri") or "")
    raw_relative = lineage.get("local_relative_dir")
    relative = raw_relative if isinstance(raw_relative, str) else ""
    expected_relative = f"eval/gold-heldout/{outer}/renders"
    expected_uri = _render_attempt_uri(
        root, outer_iteration, lineage.get("evaluation_attempt_tag")
    )
    recorded_outer = gold.get("outer_iteration")
    path = Path(relative)
    if (
        gold.get("evaluation_split") != "gold_heldout"
        or type(recorded_outer) is not int
        or recorded_outer != outer_iteration
        or lineage.get("evaluation_split") != "gold_heldout"
        or render_uri != expected_uri
        or relative != expected_relative
        or relative != relative.strip()
        or path.is_absolute()
        or path == Path(".")
        or ".." in path.parts
    ):
        raise RuntimeError("Stage 14 gold report lacks safe canonical render lineage")
    return render_uri, relative, True


def _deduplicate_download_entries(
    root: str,
    entries: list[tuple[str, str, bool]],
) -> list[tuple[str, str, bool]]:
    plan: list[tuple[str, str, bool]] = []
    root_prefix = root.rstrip("/") + "/"
    for source, destination, is_prefix in entries:
        normalized = source.rstrip("/") + "/" if is_prefix else source
        path = Path(destination)
        if (
            not source.startswith(root_prefix)
            or normalized == root_prefix
            or not destination
            or path.is_absolute()
            or path == Path(".")
            or ".." in path.parts
        ):
            raise RuntimeError(
                f"Stage 14 rejected unscoped artifact selection: {source}"
            )
        entry = (normalized, destination, is_prefix)
        if entry not in plan:
            plan.append(entry)
    return plan


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
    recorded_outer = evidence.get("outer_iteration")
    if type(recorded_outer) is not int or recorded_outer != outer_iteration:
        raise RuntimeError(
            "Stage 14 evidence outer iteration disagrees with the requested path"
        )
    entries = _base_download_entries(root, outer)
    entries += _iteration_download_entries(
        root, outer_iteration, evidence.get("iterations")
    )
    entries.append(_render_download_entry(root, outer_iteration, gold))
    return _deduplicate_download_entries(root, entries)


def _assert_no_symlinked_ancestors(
    path: Path,
    *,
    containment_root: Path,
) -> None:
    root = containment_root.absolute()
    candidate = path.absolute()
    if candidate == root or not candidate.is_relative_to(root):
        raise RuntimeError(f"Stage 14 rejected unsafe local path: {path}")
    cursor = candidate
    while True:
        if cursor.is_symlink():
            raise RuntimeError(
                f"Stage 14 rejected a symlinked local ancestor: {cursor}"
            )
        if cursor == root:
            break
        cursor = cursor.parent


def materialize_plan(plan: list[tuple[str, str, bool]], *, local: Path) -> None:
    client = storage()
    root = local.resolve()
    for source, destination, is_prefix in plan:
        target = local / destination
        _assert_no_symlinked_ancestors(target, containment_root=local)
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
    gold_report_bytes_sha256: str = ""
    publication_id: str = ""
    canonical_rrd_uri: str = ""
    canonical_mcap_uri: str = ""
    canonical_report_uri: str = ""
    publication_snapshots: dict[str, RemoteObjectSnapshot | None] | None = None


def _localize_iteration_paths(
    evidence: dict[str, Any],
    local: Path,
    *,
    root: str,
    outer_iteration: int,
) -> None:
    outer = f"outer-{outer_iteration:02d}"
    for item in evidence.get("iterations") or []:
        inner_iteration = item.get("iteration")
        if (
            not isinstance(inner_iteration, int)
            or isinstance(inner_iteration, bool)
            or inner_iteration < 1
        ):
            raise RuntimeError("Stage 14 evidence contains an invalid inner iteration")
        expected = _iteration_artifact_uris(
            root,
            outer_iteration=outer_iteration,
            inner_iteration=inner_iteration,
        )
        actual = (
            item.get("actions_uri"),
            item.get("vlm_eval_uri"),
            item.get("signal_uri"),
        )
        if actual != expected:
            raise RuntimeError(
                "Stage 14 refuses to localize iteration artifacts without exact "
                "run and outer/inner authority"
            )
        inner = f"iter-{inner_iteration:02d}"
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
    gold_report_bytes_sha256: str = "",
) -> _Stage14State:
    publication_id = secrets.token_hex(16)
    publication_root = f"{root}/reports/generations/{publication_id}"
    return _Stage14State(
        args=args,
        root=root,
        work=work,
        local=local,
        evidence=evidence,
        gold=gold,
        rrd_uri=f"{publication_root}/sim2real.rrd",
        mcap_uri=f"{publication_root}/sim2real.mcap",
        report_uri=f"{publication_root}/sim2real-report.json",
        gold_report_bytes_sha256=(gold_report_bytes_sha256 or gold_report_sha256(gold)),
        publication_id=publication_id,
        canonical_rrd_uri=f"{root}/reports/sim2real.rrd",
        canonical_mcap_uri=f"{root}/reports/sim2real.mcap",
        canonical_report_uri=f"{root}/reports/sim2real-report.json",
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
    gold_path = work / "gold-input" / "report.json"
    gold_bytes_sha256 = (
        hashlib.sha256(gold_path.read_bytes()).hexdigest()
        if gold_path.is_file()
        else gold_report_sha256(gold)
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
    _localize_iteration_paths(
        evidence,
        local,
        root=root,
        outer_iteration=outer,
    )
    return _stage14_state(
        args,
        root,
        work,
        local,
        evidence,
        gold,
        gold_report_bytes_sha256=gold_bytes_sha256,
    )


def _load_component_records(state: _Stage14State) -> list[dict[str, Any]]:
    components = [
        read_json(
            f"{state.root}/components/stage_{stage:02d}.json",
            directory=state.work / f"component-{stage:02d}",
        )
        for stage in range(1, 14)
    ]
    try:
        expected_source = source_sha()
        validate_component_records(
            components,
            root=state.root,
            evidence=state.evidence,
            gold=state.gold,
            expected_source_sha=expected_source,
        )
        for stage, component in enumerate(components, start=1):
            history = read_json(
                component_record_history_uri(
                    state.root, stage, component["content_sha256"]
                ),
                directory=state.work / f"component-history-{stage:02d}",
            )
            if history != component:
                raise ValueError(
                    f"Stage {stage} ComponentRecord pointer/history mismatch"
                )
        validate_remote_stage4_authority(
            state.root,
            components[3],
            lambda uri: read_json(
                uri,
                directory=state.work / "stage-04-nested-authority",
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise RuntimeError("Stage 14 rejected an invalid ComponentRecord") from exc
    return components


def _assert_stage14_publication_preconditions(state: _Stage14State) -> None:
    decision = parse_json_object(
        (state.local / "outer_loop" / "decision.json").read_text(),
        source="Stage 14 outer-loop decision",
    )
    metadata, _selection, _candidate = _stage14_policy_metadata(
        state.evidence,
        decision,
        state.gold,
        run_root=state.root,
        run_id=state.args.run_id,
        expected_threshold=float(getattr(state.args, "threshold", 0.5)),
        expected_early_exit=bool(getattr(state.args, "allow_early_exit", False)),
        expected_gold_report_sha256=state.gold_report_bytes_sha256,
    )
    _assert_publishable_policy_metadata(metadata, state.gold)


def _validate_stage14_materialized_frames(state: _Stage14State) -> Path:
    """Revalidate downloaded frames against the sealed Stage 10 byte manifest."""

    lineage = state.gold.get("render_lineage")
    manifest = state.gold.get("render_manifest")
    if not isinstance(lineage, dict) or not isinstance(manifest, dict):
        raise RuntimeError("Stage 14 gold report lacks render byte authority")
    relative = lineage.get("local_relative_dir")
    if not isinstance(relative, str) or not relative:
        raise RuntimeError("Stage 14 gold report lacks a local render path")
    render_dir = state.local / relative
    _selection, candidate = _resolve_stage14_selection(
        state.evidence,
        state.root,
        state.args.run_id,
    )
    return materialize_verified_render_snapshot(
        render_dir,
        state.local / ".verified-stage14-renders",
        manifest,
        candidate,
        require_frame_identity=True,
    )


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
        "policy_gate_config": {
            "threshold": decision["threshold"],
            "early_exit": decision["early_exit_enabled"],
        },
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
        "report_uri": state.report_uri,
        "publication_id": state.publication_id,
        "publication_journal_uri": (f"{state.root}/reports/.sim2real-publication.json"),
        "canonical_rrd_uri": state.canonical_rrd_uri or state.rrd_uri,
        "canonical_mcap_uri": state.canonical_mcap_uri or state.mcap_uri,
        "canonical_report_uri": state.canonical_report_uri or state.report_uri,
    }


def _build_stage14_record(state: _Stage14State) -> dict[str, Any]:
    return build_component_record(
        stage=14,
        name="stage_14_rerun_viz",
        tier="WORKS",
        evidence=(
            "Published independently decodable Rerun and MCAP recordings with "
            "multi-camera gold footage, policy, progress, and evaluation evidence."
        ),
        artifacts={
            "rrd": state.rrd_uri,
            "mcap": state.mcap_uri,
            "report": state.report_uri,
        },
    )


def _bind_stage14_report_authority(
    report: dict[str, Any],
    component: dict[str, Any],
) -> None:
    component["artifacts"]["report_authority_sha256"] = stage14_report_authority_sha256(
        report
    )
    material = {
        key: value for key, value in component.items() if key != "content_sha256"
    }
    component["content_sha256"] = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def _build_stage14_report(
    state: _Stage14State,
    components: list[dict[str, Any]],
) -> tuple[dict[str, Any], dict[str, Any], Path]:
    decision = parse_json_object(
        (state.local / "outer_loop" / "decision.json").read_text(),
        source="Stage 14 outer-loop decision",
    )
    heldout_policy_metadata, selection, selected_candidate = _stage14_policy_metadata(
        state.evidence,
        decision,
        state.gold,
        run_root=state.root,
        run_id=state.args.run_id,
        expected_threshold=float(getattr(state.args, "threshold", 0.5)),
        expected_early_exit=bool(getattr(state.args, "allow_early_exit", False)),
        expected_gold_report_sha256=state.gold_report_bytes_sha256,
    )
    _assert_publishable_policy_metadata(heldout_policy_metadata, state.gold)
    components.append(_build_stage14_record(state))
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
    _bind_stage14_report_authority(report, components[-1])
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
    verified_renders: Path,
) -> tuple[Any, Any]:
    from npa.workflows.sim2real_viz import (
        emit_sim2real_mcap,
        emit_sim2real_rerun,
    )

    localized_gold = json.loads(json.dumps(state.gold))
    localized_gold["local_renders_dir"] = str(verified_renders)
    decision = report["outer_loop"]["decision"]
    rrd = emit_sim2real_rerun(
        local_dir=state.local,
        inner_evidence=state.evidence,
        heldout_report=localized_gold,
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
        heldout_report=localized_gold,
        output_mcap=reports / "sim2real.mcap",
    )
    return rrd, mcap


def _assert_recording_result(label: str, result: Any) -> None:
    heldout_frames = getattr(result, "heldout_frame_count", None)
    if type(heldout_frames) is not int or heldout_frames <= 0:
        raise RuntimeError(
            f"Stage 14 refuses to publish {label} evidence with zero held-out frames"
        )


def _stage14_recording_path(
    state: _Stage14State,
    reports: Path,
    filename: str,
) -> Path:
    path = reports / filename
    _assert_no_symlinked_ancestors(path, containment_root=state.local)
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Stage 14 produced empty {filename}")
    return path


def _seal_stage14_report(
    report: dict[str, Any],
    rrd: Any,
    mcap: Any,
) -> None:
    report["recording_summaries"] = {
        "rrd": rrd.to_dict(),
        "mcap": mcap.to_dict(),
    }


def _publish_stage14_outputs(
    state: _Stage14State,
    report: dict[str, Any],
    reports: Path,
    rrd: Any,
    mcap: Any,
) -> None:
    _assert_recording_result("Rerun", rrd)
    _assert_recording_result("MCAP", mcap)
    recordings = [
        (
            _stage14_recording_path(state, reports, "sim2real.rrd"),
            state.rrd_uri,
            state.canonical_rrd_uri or state.rrd_uri,
        ),
        (
            _stage14_recording_path(state, reports, "sim2real.mcap"),
            state.mcap_uri,
            state.canonical_mcap_uri or state.mcap_uri,
        ),
    ]
    client = storage()
    canonical_report = state.canonical_report_uri or state.report_uri
    component_pointer = f"{state.root}/components/stage_14.json"
    lock_uri = f"{state.root}/reports/.sim2real-publication.json"
    mutable_uris = (
        canonical_report,
        *(alias_uri for _path, _immutable_uri, alias_uri in recordings),
        component_pointer,
        lock_uri,
    )
    snapshots = state.publication_snapshots
    if snapshots is None:
        raise RuntimeError(
            "Stage 14 publication snapshots were not captured before validation"
        )
    if any(uri not in snapshots for uri in mutable_uris):
        raise RuntimeError("Stage 14 publication snapshot set is incomplete")
    for path, immutable_uri, _alias_uri in recordings:
        upload_immutable_file(client, path, immutable_uri)
    _seal_stage14_report(report, rrd, mcap)
    report_path = reports / "sim2real-report.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    upload_immutable_file(client, report_path, state.report_uri)
    component_record = report["component_records"][-1]
    component_kwargs = {
        "root_uri": state.root,
        "record": component_record,
        "expected_stage": 14,
        "expected_name": "stage_14_rerun_viz",
        "expected_tier": "WORKS",
        "required_artifacts": ("rrd", "mcap", "report"),
    }
    publish_built_component_history(**component_kwargs, client=client)
    with MutablePublicationTransaction(
        client,
        lock_uri=lock_uri,
        lock_snapshot=snapshots[lock_uri],
        transaction_id=state.publication_id,
    ) as transaction:
        for path, immutable_uri, alias_uri in recordings:
            if alias_uri != immutable_uri:
                transaction.replace_file(
                    path,
                    alias_uri,
                    snapshots[alias_uri],
                    immutable_uri=immutable_uri,
                )
        if canonical_report != state.report_uri:
            transaction.replace_file(
                report_path,
                canonical_report,
                snapshots[canonical_report],
                immutable_uri=state.report_uri,
            )
        publish_built_component_pointer(
            **component_kwargs,
            client=client,
            snapshot=snapshots[component_pointer],
            transaction=transaction,
            immutable_uri=component_record_history_uri(
                state.root,
                14,
                component_record["content_sha256"],
            ),
        )


def _capture_stage14_publication_snapshots(
    state: _Stage14State,
) -> _Stage14State:
    """Capture mutable authority versions before validating source authority."""

    canonical_report = state.canonical_report_uri or state.report_uri
    lock_uri = f"{state.root}/reports/.sim2real-publication.json"
    uris = (
        canonical_report,
        state.canonical_rrd_uri or state.rrd_uri,
        state.canonical_mcap_uri or state.mcap_uri,
        f"{state.root}/components/stage_14.json",
        lock_uri,
    )
    client = storage()
    lock_snapshot = remote_object_snapshot(client, lock_uri)
    recover_interrupted_publication(
        client,
        lock_uri=lock_uri,
        lock_snapshot=lock_snapshot,
    )
    return replace(
        state,
        publication_snapshots={
            uri: (
                remote_object_snapshot(client, uri)
                if uri == lock_uri
                else remote_object_version(client, uri)
            )
            for uri in uris
        },
    )


def finalize_in_work(args: argparse.Namespace, *, root: str, work: Path) -> None:
    state = _materialize_stage14(args, root=root, work=work)
    state = _capture_stage14_publication_snapshots(state)
    _assert_stage14_publication_preconditions(state)
    verified_renders = _validate_stage14_materialized_frames(state)
    components = _load_component_records(state)
    report, policy_metadata, reports = _build_stage14_report(state, components)
    rrd, mcap = _emit_stage14_outputs(
        state,
        components,
        report,
        policy_metadata,
        reports,
        verified_renders,
    )
    _publish_stage14_outputs(state, report, reports, rrd, mcap)


def finalize(args: argparse.Namespace) -> None:
    root = str(args.root_uri).rstrip("/")
    with tempfile.TemporaryDirectory(prefix="npa-s2r-stage-14-") as directory:
        finalize_in_work(args, root=root, work=Path(directory))
