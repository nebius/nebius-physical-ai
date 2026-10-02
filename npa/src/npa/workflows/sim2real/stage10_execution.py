"""Execute the canonical Stage 10 gold held-out evaluation."""

from __future__ import annotations

import argparse
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from npa.workflows.sim2real.decision_authority import gold_report_sha256
from npa.workflows.sim2real.publication import upload_immutable_tree
from npa.workflows.sim2real.stage10_authority import (
    expected_byo_render_prefix,
    validate_stage10_input_scope,
)


@dataclass(frozen=True)
class Stage10Operations:
    """I/O and validation operations supplied by the stage adapter."""

    read_json: Callable[..., dict[str, Any]]
    run_eval: Callable[..., dict[str, Any]]
    checkpoint_identity: Callable[[dict[str, Any], dict[str, Any]], dict[str, Any]]
    embodiment_evidence: Callable[..., dict[str, Any]]
    storage: Callable[[], Any]
    write_loop_output: Callable[..., Any]
    publish_component_record: Callable[..., Any]


def assert_gold_checkpoint_identity(
    evidence: dict[str, Any],
    report: dict[str, Any],
) -> dict[str, Any]:
    """Bind gold evaluation bytes to the validation-selected learned actor."""

    from npa.workflows.sim2real.checkpoint_selection import (
        resolve_selected_checkpoint,
    )
    from npa.workflows.sim2real.viz_contract import (
        selected_checkpoint_policy_metadata,
    )

    try:
        selection, candidate = resolve_selected_checkpoint(evidence)
    except ValueError as exc:
        raise RuntimeError(
            "Stage 10 selected checkpoint identity is incomplete or inconsistent"
        ) from exc
    metadata = selected_checkpoint_policy_metadata(report, selection, candidate)
    if not all(
        metadata[field] is True
        for field in (
            "heldout_policy_identity_verified",
            "heldout_policy_loaded_for_inference",
            "heldout_policy_learned_actor_only",
        )
    ):
        issues = ", ".join(metadata["heldout_policy_identity_errors"]) or (
            "learned-actor-only provenance is incomplete or contradictory"
        )
        raise RuntimeError(f"Stage 10 selected checkpoint identity mismatch: {issues}")
    return candidate


def _assert_render_checkpoint_identity(
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    require_checkpoint_identity: bool,
) -> None:
    checkpoint = manifest.get("policy_checkpoint")
    if not require_checkpoint_identity and checkpoint is None:
        return
    if not isinstance(checkpoint, dict):
        raise RuntimeError("Stage 10 render manifest lacks checkpoint identity")
    expected = (
        candidate.get("checkpoint_uri"),
        candidate.get("checkpoint_sha256"),
        candidate.get("checkpoint_size_bytes"),
    )
    actual = (
        checkpoint.get("uri"),
        checkpoint.get("sha256"),
        checkpoint.get("size_bytes"),
    )
    if actual != expected:
        raise RuntimeError(
            "Stage 10 render manifest checkpoint disagrees with selection"
        )


def _episode_render_frames(episode: object) -> set[str]:
    if not isinstance(episode, dict):
        raise RuntimeError("Stage 10 render episode must be an object")
    env_id = episode.get("env_id")
    if not isinstance(env_id, str) or not env_id or Path(env_id).name != env_id:
        raise RuntimeError("Stage 10 render episode has an unsafe env_id")
    names = list(episode.get("frames") or [])
    views = episode.get("camera_views") or {}
    if not isinstance(views, dict):
        raise RuntimeError("Stage 10 render camera views must be an object")
    for values in views.values():
        if not isinstance(values, list):
            raise RuntimeError("Stage 10 render camera frame list is invalid")
        names.extend(values)
    frames: set[str] = set()
    for name in names:
        path = Path(str(name))
        if path.name != str(name) or not str(name).endswith(".png"):
            raise RuntimeError("Stage 10 render manifest frame path is unsafe")
        frames.add((Path(env_id) / path).as_posix())
    return frames


def _declared_render_frames(
    manifest: dict[str, Any],
    *,
    candidate: dict[str, Any],
    require_checkpoint_identity: bool,
) -> set[str]:
    _assert_render_checkpoint_identity(manifest, candidate, require_checkpoint_identity)
    episodes = manifest.get("episodes")
    if not isinstance(episodes, list) or not episodes:
        raise RuntimeError("Stage 10 render manifest has no episodes")
    declared = set().union(*(_episode_render_frames(item) for item in episodes))
    if not declared:
        raise RuntimeError("Stage 10 render manifest declares no camera frames")
    return declared


def _assert_exact_render_frames(render_local: Path, declared: set[str]) -> None:
    actual = {
        path.relative_to(render_local).as_posix()
        for path in render_local.rglob("*.png")
        if path.is_file()
    }
    if actual != declared:
        raise RuntimeError(
            "Stage 10 render bytes do not exactly match the declared frame set"
        )


def _assert_render_frame_bytes(
    render_local: Path,
    manifest: dict[str, Any],
    declared: set[str],
    *,
    required: bool,
) -> None:
    frame_artifacts = manifest.get("frame_artifacts")
    if not required and frame_artifacts is None:
        return
    if not isinstance(frame_artifacts, dict) or set(frame_artifacts) != declared:
        raise RuntimeError("Stage 10 render manifest lacks exact frame-byte authority")
    for relative in sorted(declared):
        identity = frame_artifacts.get(relative)
        path = render_local / relative
        if not isinstance(identity, dict) or path.is_symlink() or not path.is_file():
            raise RuntimeError(
                f"Stage 10 render frame authority is invalid: {relative}"
            )
        payload = path.read_bytes()
        digest = hashlib.sha256(payload).hexdigest()
        size = identity.get("size_bytes")
        if (
            len(payload) <= 8
            or not payload.startswith(b"\x89PNG\r\n\x1a\n")
            or type(size) is not int
            or size != len(payload)
            or identity.get("sha256") != digest
        ):
            raise RuntimeError(
                f"Stage 10 render frame bytes disagree with manifest: {relative}"
            )


def validate_materialized_render_tree(
    render_local: Path,
    manifest: dict[str, Any],
    candidate: dict[str, Any],
    *,
    require_checkpoint_identity: bool = True,
    require_frame_identity: bool | None = None,
) -> None:
    """Bind one local render tree to the manifest and checkpoint that produced it."""

    declared = _declared_render_frames(
        manifest,
        candidate=candidate,
        require_checkpoint_identity=require_checkpoint_identity,
    )
    _assert_exact_render_frames(render_local, declared)
    _assert_render_frame_bytes(
        render_local,
        manifest,
        declared,
        required=(
            manifest.get("schema") == "npa.sim2real.heldout_renders.v2"
            if require_frame_identity is None
            else require_frame_identity
        ),
    )


def _canonical_render_attempt_tag(
    manifest: dict[str, Any],
    *,
    outer_iteration: int,
) -> str:
    attempt_tag = manifest.get("evaluation_attempt_tag")
    prefix = f"gold-o{outer_iteration:02d}-attempt-"
    if (
        not isinstance(attempt_tag, str)
        or not attempt_tag.startswith(prefix)
        or len(attempt_tag) != len(prefix) + 32
        or any(char not in "0123456789abcdef" for char in attempt_tag[len(prefix) :])
    ):
        raise RuntimeError("Stage 10 render producer lacks immutable attempt identity")
    return attempt_tag


def _normalize_gold_report_scope(
    report: dict[str, Any],
    *,
    outer_iteration: int,
) -> None:
    if report.get("evaluation_split") not in (None, "", "gold_heldout"):
        raise RuntimeError("Stage 10 report evaluation split disagrees with gold")
    recorded_outer = report.get("outer_iteration")
    if recorded_outer is not None and (
        type(recorded_outer) is not int or recorded_outer != outer_iteration
    ):
        raise RuntimeError("Stage 10 report outer iteration disagrees with execution")
    report["evaluation_split"] = "gold_heldout"
    report["outer_iteration"] = outer_iteration


def _stage10_embodiment(
    operations: Stage10Operations,
    root: str,
    report: dict[str, Any],
    selected_candidate: dict[str, Any],
) -> dict[str, Any]:
    evaluated = operations.embodiment_evidence(
        root=root, payload=report, stage="Stage 10 gold evaluation"
    )
    if not evaluated:
        return {}
    trained = dict(selected_candidate.get("embodiment") or {})
    if not trained:
        raise RuntimeError("Stage 10 selected checkpoint has no training embodiment")
    if evaluated != trained:
        raise RuntimeError("Stage 10 checkpoint train/eval embodiment parity mismatch")
    return evaluated


def _stage10_render_source(
    args: argparse.Namespace,
    root: str,
    run_id: str,
    evidence: dict[str, Any],
    report: dict[str, Any],
) -> tuple[dict[str, Any], str, str | None]:
    manifest = dict(report.get("render_manifest") or {})
    source = str(manifest.get("renders_s3_uri") or "")
    if not source or not manifest.get("episodes"):
        raise RuntimeError("Stage 10 gold evaluation lacks explicit render lineage")
    modern = evidence.get("schema") == "npa.sim2real.inner_loop_evidence.v1"
    attempt = (
        _canonical_render_attempt_tag(manifest, outer_iteration=args.outer_iteration)
        if modern
        else None
    )
    expected = expected_byo_render_prefix(
        root=root,
        run_id=run_id,
        outer_iteration=args.outer_iteration,
        evaluation_tag=attempt,
    )
    if source.rstrip("/") + "/" != expected:
        raise RuntimeError(
            "Stage 10 render source is not the exact current heldout-eval producer output"
        )
    return manifest, expected, attempt


def _render_destination(
    root: str,
    outer_iteration: int,
    relative: str,
    attempt: str | None,
) -> str:
    if attempt is None:
        return f"{root}/{relative}/"
    return (
        f"{root}/eval/gold-heldout/outer-{outer_iteration:02d}/"
        f"attempts/{attempt}/renders/"
    )


def _materialize_stage10_renders(
    operations: Stage10Operations,
    args: argparse.Namespace,
    root: str,
    work: Path,
    evidence: dict[str, Any],
    report: dict[str, Any],
    candidate: dict[str, Any],
    run_id: str,
) -> str:
    manifest, source, attempt = _stage10_render_source(
        args, root, run_id, evidence, report
    )
    relative = f"eval/gold-heldout/outer-{args.outer_iteration:02d}/renders"
    local = work / relative
    client = operations.storage()
    client.download_directory(source, str(local))
    validate_materialized_render_tree(
        local,
        manifest,
        candidate,
        require_checkpoint_identity=attempt is not None,
        require_frame_identity=attempt is not None,
    )
    destination = _render_destination(root, args.outer_iteration, relative, attempt)
    upload_immutable_tree(client, local, destination)
    report["render_lineage"] = {
        "evaluation_split": "gold_heldout",
        "evaluation_attempt_tag": attempt or "",
        "source_s3_uri": source,
        "canonical_s3_uri": destination,
        "local_relative_dir": relative,
    }
    report["local_renders_dir"] = relative
    return destination


def _publish_stage10_report(
    operations: Stage10Operations,
    args: argparse.Namespace,
    root: str,
    work: Path,
    evidence: dict[str, Any],
    report: dict[str, Any],
    render_uri: str,
    embodiment: dict[str, Any],
) -> None:
    report_uri = (
        f"{root}/eval/gold-heldout/outer-{args.outer_iteration:02d}/report.json"
    )
    operations.write_loop_output(report_uri, report, work / "out", args.outer_iteration)
    operations.publish_component_record(
        root_uri=root,
        stage=10,
        name="stage_10_eval_heldout",
        tier="WORKS",
        evidence="Isaac loaded the validation-selected checkpoint and evaluated only the untouched gold split with strict 5 cm stable placement.",
        artifacts={
            "report": report_uri,
            "gold_report_sha256": gold_report_sha256(report),
            "evaluation_split": "gold_heldout",
            "checkpoint": evidence["selected_checkpoint_uri"],
            "checkpoint_sha256": report.get("policy_checkpoint_sha256", ""),
            "renders": render_uri,
            "render_lineage": report["render_lineage"],
            "component_invocation": report.get("component_invocation"),
            "embodiment": embodiment,
        },
        require_gpu=True,
    )


def run_stage10(
    args: argparse.Namespace,
    *,
    root: str,
    work: Path,
    operations: Stage10Operations,
) -> None:
    """Run gold evaluation, seal exact renders, and publish its component record."""

    evidence_uri = f"{root}/inner_loop/outer-{args.outer_iteration:02d}/evidence.json"
    evidence = operations.read_json(evidence_uri, directory=work / "input")
    run_id = validate_stage10_input_scope(args, root=root, evidence=evidence)
    report = operations.run_eval(
        args,
        split="gold_heldout",
        envs_uri=f"{root}/envs/gold-heldout/envs.jsonl",
        env_count=args.gold_count,
        evidence=evidence,
        output_path=work / "report.json",
        tag=f"gold-o{args.outer_iteration:02d}",
    )
    _normalize_gold_report_scope(report, outer_iteration=args.outer_iteration)
    selected = operations.checkpoint_identity(evidence, report)
    embodiment = _stage10_embodiment(operations, root, report, selected)
    render_uri = _materialize_stage10_renders(
        operations, args, root, work, evidence, report, selected, run_id
    )
    _publish_stage10_report(
        operations, args, root, work, evidence, report, render_uri, embodiment
    )
