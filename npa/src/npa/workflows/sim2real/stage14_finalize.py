"""Bounded artifact materialization and final evidence for Sim2Real Stage 14."""

from __future__ import annotations

import argparse
import json
import secrets
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
    resolve_run_scoped_checkpoint,
)
from npa.workflows.sim2real.decision_authority import validate_stage11_decision
from npa.workflows.sim2real.workflow_io import (
    build_component_record,
    parse_json_object,
    publish_built_component_record,
    read_json,
    source_sha,
    storage,
    validate_component_record,
    write_json,
)


_COMPONENT_CONTRACTS: dict[int, tuple[str, str, tuple[str, ...]]] = {
    1: ("stage_01_trigger", "WORKS", ("trigger", "task_contract_digest")),
    2: (
        "stage_02_assets",
        "WORKS",
        ("task_contract", "scene", "robot_contract"),
    ),
    3: ("stage_03_augment", "WORKS", ("manifest", "result", "frames")),
    4: ("stage_04_envs_raw", "WORKS", ("raw_envs", "shard_provenance")),
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
        ("report", "checkpoint", "renders", "render_lineage"),
    ),
    11: ("stage_11_outer_loop", "WORKS", ("decision", "gold_report")),
    12: ("stage_12_external_validation", "SEAM", ("seam",)),
    13: ("stage_13_retrigger", "WORKS", ("record", "decision")),
}
_COMPONENT_URI_KEYS: dict[int, tuple[str, ...]] = {
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
            selection, candidate = resolve_run_scoped_checkpoint(
                evidence,
                run_root=run_root,
                run_id=run_id,
            )
    except ValueError as exc:
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
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    _assert_promotion_checkpoint_uri(decision)
    selection, candidate = _resolve_stage14_selection(evidence, run_root, run_id)
    decision_candidate = _validated_decision_candidate(decision, candidate, selection)
    _assert_stage14_decision_authority(
        evidence, decision, gold, selection, run_root, run_id
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
    publication_id: str = ""
    canonical_rrd_uri: str = ""
    canonical_mcap_uri: str = ""
    canonical_report_uri: str = ""


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
    gold["local_renders_dir"] = str(
        local / str((gold.get("render_lineage") or {}).get("local_relative_dir") or "")
    )
    return _stage14_state(args, root, work, local, evidence, gold)


def _assert_component_artifact_scope(
    state: _Stage14State,
    stage: int,
    component: dict[str, Any],
) -> None:
    root_prefix = state.root.rstrip("/") + "/"
    for key in _COMPONENT_URI_KEYS[stage]:
        uri = component["artifacts"][key]
        suffix = uri[len(root_prefix) :] if isinstance(uri, str) else ""
        parts = suffix.rstrip("/").split("/")
        if (
            not isinstance(uri, str)
            or not uri.startswith(root_prefix)
            or not suffix
            or any(token in suffix for token in ("\\", "%", "?", "#"))
            or any(part in {"", ".", ".."} for part in parts)
        ):
            raise ValueError(f"Stage {stage} ComponentRecord {key} is outside the run")


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
        for stage, component in enumerate(components, start=1):
            name, tier, required = _COMPONENT_CONTRACTS[stage]
            validate_component_record(
                component,
                expected_stage=stage,
                expected_name=name,
                expected_tier=tier,
                required_artifacts=required,
                expected_source_sha=expected_source,
            )
            _assert_component_artifact_scope(state, stage, component)
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
    )
    _assert_publishable_policy_metadata(metadata, state.gold)


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
        "publication_id": state.publication_id,
        "canonical_rrd_uri": state.canonical_rrd_uri or state.rrd_uri,
        "canonical_mcap_uri": state.canonical_mcap_uri or state.mcap_uri,
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


def _assert_recording_result(label: str, result: Any) -> None:
    heldout_frames = getattr(result, "heldout_frame_count", None)
    if type(heldout_frames) is not int or heldout_frames <= 0:
        raise RuntimeError(
            f"Stage 14 refuses to publish {label} evidence with zero held-out frames"
        )


def _upload_stage14_recording(
    state: _Stage14State,
    reports: Path,
    filename: str,
    uri: str,
    alias_uri: str,
) -> None:
    path = reports / filename
    _assert_no_symlinked_ancestors(path, containment_root=state.local)
    if path.stat().st_size <= 0:
        raise RuntimeError(f"Stage 14 produced empty {filename}")
    storage().upload_file(str(path), uri)
    if alias_uri != uri:
        storage().upload_file(str(path), alias_uri)


def _publish_stage14_reports(
    state: _Stage14State,
    report: dict[str, Any],
    rrd: Any,
    mcap: Any,
) -> None:
    report["recording_summaries"] = {
        "rrd": rrd.to_dict(),
        "mcap": mcap.to_dict(),
    }
    write_json(state.report_uri, report, directory=state.work / "final-report")
    canonical = state.canonical_report_uri or state.report_uri
    if canonical != state.report_uri:
        write_json(canonical, report, directory=state.work / "canonical-final-report")


def _publish_stage14_outputs(
    state: _Stage14State,
    report: dict[str, Any],
    reports: Path,
    rrd: Any,
    mcap: Any,
) -> None:
    _assert_recording_result("Rerun", rrd)
    _assert_recording_result("MCAP", mcap)
    recordings = (
        (
            "sim2real.rrd",
            state.rrd_uri,
            state.canonical_rrd_uri or state.rrd_uri,
        ),
        (
            "sim2real.mcap",
            state.mcap_uri,
            state.canonical_mcap_uri or state.mcap_uri,
        ),
    )
    for filename, uri, alias_uri in recordings:
        _upload_stage14_recording(state, reports, filename, uri, alias_uri)
    _publish_stage14_reports(state, report, rrd, mcap)
    publish_built_component_record(
        root_uri=state.root,
        record=report["component_records"][-1],
        expected_stage=14,
        expected_name="stage_14_rerun_viz",
        expected_tier="WORKS",
        required_artifacts=("rrd", "mcap", "report"),
    )


def finalize_in_work(args: argparse.Namespace, *, root: str, work: Path) -> None:
    state = _materialize_stage14(args, root=root, work=work)
    _assert_stage14_publication_preconditions(state)
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
