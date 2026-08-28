"""Bounded artifact materialization and final evidence for Sim2Real Stage 14."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

from npa.workflows.sim2real.workflow_io import (
    publish_component_record,
    read_json,
    source_sha,
    storage,
    write_json,
)


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
    render_relative = str(lineage.get("local_relative_dir") or "").strip("/")
    if not render_uri or not render_relative or ".." in Path(render_relative).parts:
        raise RuntimeError("Stage 14 gold report lacks safe canonical render lineage")
    entries.append((render_uri, render_relative, True))

    seen: set[tuple[str, str, bool]] = set()
    plan: list[tuple[str, str, bool]] = []
    root_prefix = root.rstrip("/") + "/"
    for source, destination, is_prefix in entries:
        normalized = source.rstrip("/") + "/" if is_prefix else source
        if (
            not source.startswith(root_prefix)
            or normalized == root_prefix
            or not destination
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
    for source, destination, is_prefix in plan:
        target = local / destination
        if not target.resolve().is_relative_to(local.resolve()):
            raise RuntimeError(f"Stage 14 rejected unsafe local path: {destination}")
        if is_prefix:
            client.download_directory(source, str(target))
        else:
            client.download_file(source, str(target))


def finalize_in_work(args: argparse.Namespace, *, root: str, work: Path) -> None:
    from npa.workflows.sim2real_viz import emit_sim2real_mcap, emit_sim2real_rerun

    evidence = read_json(
        f"{root}/inner_loop/outer-{args.outer_iteration:02d}/evidence.json",
        directory=work / "evidence-input",
    )
    gold = read_json(
        f"{root}/eval/gold-heldout/outer-{args.outer_iteration:02d}/report.json",
        directory=work / "gold-input",
    )
    local = work / "run"
    plan = download_plan(
        root=root,
        outer_iteration=args.outer_iteration,
        evidence=evidence,
        gold=gold,
    )
    materialize_plan(plan, local=local)
    for item in evidence.get("iterations") or []:
        inner = int(item.get("iteration") or 1)
        item["actions_dir"] = str(
            local
            / "actions"
            / "train"
            / f"outer-{args.outer_iteration:02d}"
            / f"iter-{inner:02d}"
        )
        item["vlm_eval_dir"] = str(
            local
            / "vlm_eval"
            / "train"
            / f"outer-{args.outer_iteration:02d}"
            / f"iter-{inner:02d}"
            / "evaluations"
        )
        item["signal_dir"] = str(
            local
            / "vlm_eval"
            / "train"
            / f"outer-{args.outer_iteration:02d}"
            / f"iter-{inner:02d}"
            / "signals"
        )
        item["validation_report"] = evidence.get("selected_validation_report") or {}
    gold["local_renders_dir"] = str(
        local / str((gold.get("render_lineage") or {}).get("local_relative_dir") or "")
    )
    rrd_uri = f"{root}/reports/sim2real.rrd"
    mcap_uri = f"{root}/reports/sim2real.mcap"
    report_uri = f"{root}/reports/sim2real-report.json"
    publish_component_record(
        root_uri=root,
        stage=14,
        name="stage_14_rerun_viz",
        tier="WORKS",
        evidence="Finalization is executing in the standard workflow runtime and will publish full Rerun and MCAP evidence.",
        artifacts={"rrd": rrd_uri, "mcap": mcap_uri, "report": report_uri},
    )
    components = [
        read_json(
            f"{root}/components/stage_{stage:02d}.json",
            directory=work / f"component-{stage:02d}",
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
    decision = json.loads((local / "outer_loop" / "decision.json").read_text())
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "run_id": args.run_id,
        "source_sha": source_sha(),
        "status": "completed",
        "architecture": "npa.workflow/v0.0.1_compositional_standard_runtime",
        "component_records": components,
        "outer_loop": {"decision": decision, "latest_heldout_report": gold},
        "checkpoint_selection": evidence.get("checkpoint_selection"),
        "stage8_evaluator_usage": [
            item.get("evaluator_usage") for item in evidence.get("iterations") or []
        ],
        "strict_gold_success_rate": float(gold.get("success_rate") or 0.0),
        "policy_quality_is_pipeline_gate": False,
        "rrd_uri": rrd_uri,
        "mcap_uri": mcap_uri,
    }
    reports = local / "reports"
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "sim2real-report.json").write_text(json.dumps(report, indent=2))
    run_metadata = {
        "run_id": args.run_id,
        "artifact_root": root,
        "policy_checkpoint": evidence.get("selected_checkpoint_uri", ""),
        "policy_checkpoint_sha256": gold.get("policy_checkpoint_sha256", ""),
        "policy_checkpoint_size_bytes": gold.get("policy_checkpoint_size_bytes", 0),
        "heldout_policy_loaded_for_inference": True,
        "heldout_policy_checkpoint": evidence.get("selected_checkpoint_uri", ""),
        "heldout_policy_checkpoint_sha256": gold.get("policy_checkpoint_sha256", ""),
        "heldout_policy_checkpoint_size_bytes": gold.get(
            "policy_checkpoint_size_bytes", 0
        ),
        "rrd_s3_uri": rrd_uri,
    }
    rrd = emit_sim2real_rerun(
        local_dir=local,
        inner_evidence=evidence,
        heldout_report=gold,
        stage_components=components,
        outer_history=[
            {
                "decision": decision,
                "checkpoint_uri": evidence.get("selected_checkpoint_uri", ""),
            }
        ],
        run_metadata=run_metadata,
        output_rrd=reports / "sim2real.rrd",
        allow_progress_only=False,
    )
    mcap = emit_sim2real_mcap(
        local_dir=local,
        inner_evidence=evidence,
        heldout_report=gold,
        output_mcap=reports / "sim2real.mcap",
    )
    for filename, uri in (("sim2real.rrd", rrd_uri), ("sim2real.mcap", mcap_uri)):
        path = reports / filename
        if path.stat().st_size <= 0:
            raise RuntimeError(f"Stage 14 produced empty {filename}")
        storage().upload_file(str(path), uri)
    final_record = publish_component_record(
        root_uri=root,
        stage=14,
        name="stage_14_rerun_viz",
        tier="WORKS",
        evidence="Published independently decodable Rerun and MCAP recordings with multi-camera gold footage, policy, progress, and evaluation evidence.",
        artifacts={
            "rrd": rrd_uri,
            "rrd_bytes": (reports / "sim2real.rrd").stat().st_size,
            "rrd_summary": rrd.to_dict(),
            "mcap": mcap_uri,
            "mcap_bytes": (reports / "sim2real.mcap").stat().st_size,
            "mcap_summary": mcap.to_dict(),
            "report": report_uri,
        },
    )
    report["component_records"][-1] = final_record
    write_json(report_uri, report, directory=work / "final-report")


def finalize(args: argparse.Namespace) -> None:
    root = str(args.root_uri).rstrip("/")
    with tempfile.TemporaryDirectory(prefix="npa-s2r-stage-14-") as directory:
        finalize_in_work(args, root=root, work=Path(directory))
