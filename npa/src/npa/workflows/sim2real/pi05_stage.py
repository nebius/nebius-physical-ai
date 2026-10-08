"""Small CPU stage adapters for the declarative pi0.5 Sim2Real variant."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

from npa.workflows.sim2real.pi05_contract import assert_contract, build_contract
from npa.workflows.sim2real.workflow_io import (
    publish_component_record,
    read_json,
    storage,
    write_json,
)


def _prepare(args: argparse.Namespace) -> dict:
    contract = build_contract()
    assert_contract(contract)
    result = {
        "schema": "npa.sim2real.pi05.preparation.v1",
        "run_id": args.run_id,
        "task_contract": contract,
        "planning_prerequisites": {
            "immutable_openpi_source": contract["upstream"]["openpi_commit"],
            "digest_pinned_images_required": True,
            "disjoint_object_and_scene_splits_required": True,
        },
        "execution_prerequisites": {
            "gemma_terms": "operator-owned task-scoped NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES",
            "isaac_eula": "standard submit-time product routing",
            "private_clusterip_and_scoped_rbac": True,
        },
    }
    write_json(args.output_uri, result, directory=Path(tempfile.mkdtemp()))
    publish_component_record(
        root_uri=args.component_root_uri,
        stage=1,
        name="pi05_prepare_contract",
        tier="WORKS",
        evidence="Published the digest-bound released-surface pi0.5 contract.",
        artifacts={"preparation": args.output_uri},
    )
    return result


def _external_seam(args: argparse.Namespace) -> dict:
    result = {
        "schema": "npa.sim2real.pi05.external_validation_seam.v1",
        "run_id": args.run_id,
        "status": "operator_seam_not_executed",
        "physical_robot_deployed": False,
        "required_before_robot_claim": [
            "operator robot selection and calibration",
            "joint/gripper/camera contract verification on that robot",
            "workspace and force safety review",
            "supervised physical trials",
        ],
    }
    write_json(args.output_uri, result, directory=Path(tempfile.mkdtemp()))
    publish_component_record(
        root_uri=args.component_root_uri,
        stage=12,
        name="pi05_physical_robot_seam",
        tier="SEAM",
        evidence="Physical robot deployment remains an explicit operator seam.",
        artifacts={"seam": args.output_uri},
    )
    return result


def _copy_artifact(source: str, destination: str, work: Path) -> dict[str, object]:
    work.mkdir(parents=True, exist_ok=True)
    target = work / Path(source).name
    storage().download_file(source, str(target))
    payload = target.read_bytes()
    if not payload:
        raise RuntimeError(f"review artifact is empty: {source}")
    storage().upload_file(str(target), destination)
    return {
        "uri": destination,
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _comparison(baseline: dict, adapted: dict) -> dict[str, object]:
    baseline_protocol = baseline.get("protocol") or {}
    adapted_protocol = adapted.get("protocol") or {}
    comparable = (
        baseline.get("split") == adapted.get("split") == "gold"
        and baseline_protocol.get("sha256") == adapted_protocol.get("sha256")
        and baseline_protocol.get("scenario_sha256")
        == adapted_protocol.get("scenario_sha256")
        and baseline.get("episode_count") == adapted.get("episode_count")
    )
    before, after = (
        float(baseline.get("success_rate", 0.0)),
        float(adapted.get("success_rate", 0.0)),
    )
    improvement = after - before if comparable else None
    return {
        "comparison_protocol_identical": comparable,
        "baseline_success_rate": before,
        "adapted_success_rate": after,
        "measured_improvement": improvement,
        "improvement_claimed": bool(
            comparable and improvement is not None and improvement > 0
        ),
    }


def _checkpoint_evidence(
    args: argparse.Namespace, artifacts: dict, work: Path
) -> dict[str, object]:
    manifest = read_json(args.checkpoint_manifest_uri, directory=work / "manifest")
    files = manifest.get("files")
    if manifest.get(
        "schema"
    ) != "npa.workbench.openpi.checkpoint-manifest.v1" or not isinstance(files, list):
        raise RuntimeError("selected checkpoint manifest is invalid")
    digest = hashlib.sha256(
        json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if manifest.get("content_manifest_sha256") != digest:
        raise RuntimeError("selected checkpoint content manifest digest is invalid")
    training_digest = (artifacts["training"].get("checkpoint") or {}).get(
        "content_manifest_sha256"
    )
    reload_digest = (artifacts["reload"].get("lineage") or {}).get(
        "trained_checkpoint_manifest_sha256"
    )
    if training_digest != digest or reload_digest != digest:
        raise RuntimeError("training, selected checkpoint, and reload hashes differ")
    return {
        "training_artifact": args.training_uri,
        "reload_evaluation": args.reload_uri,
        "selected_manifest": args.checkpoint_manifest_uri,
        "content_manifest_sha256": digest,
        "reload_consumed_exact_selected_checkpoint": True,
    }


def _load_final_inputs(args: argparse.Namespace, work: Path) -> dict[str, dict]:
    return {
        "baseline": read_json(args.baseline_uri, directory=work / "baseline"),
        "adapted": read_json(args.adapted_uri, directory=work / "adapted"),
        "training": read_json(args.training_uri, directory=work / "training"),
        "reload": read_json(args.reload_uri, directory=work / "reload"),
        "external": read_json(args.external_seam_uri, directory=work / "external"),
    }


def _finalize(args: argparse.Namespace) -> dict:
    work = Path(tempfile.mkdtemp(prefix="npa-pi05-finalize-"))
    artifacts = _load_final_inputs(args, work)
    copied = {
        "mp4": _copy_artifact(
            args.adapted_artifact_root_uri.rstrip("/") + "/rollouts.mp4",
            args.final_mp4_uri,
            work / "media",
        ),
        "rrd": _copy_artifact(
            args.adapted_artifact_root_uri.rstrip("/") + "/rollouts.rrd",
            args.final_rrd_uri,
            work / "media",
        ),
    }
    result = {
        "schema": "npa.sim2real.pi05.final_report.v1",
        "run_id": args.run_id,
        "task": "released_supported_surface_placement_v1",
        **_comparison(artifacts["baseline"], artifacts["adapted"]),
        "canonical_ppo_comparison": "not_comparable_airborne_goal_historical_diagnostic_only",
        "checkpoint": _checkpoint_evidence(args, artifacts, work),
        "review_artifacts": copied,
        "physical_robot": artifacts["external"],
        "orchestration_succeeded_independent_of_task_quality": True,
    }
    write_json(args.output_uri, result, directory=work / "report")
    publish_component_record(
        root_uri=args.component_root_uri,
        stage=14,
        name="pi05_finalize",
        tier="WORKS",
        evidence="Bound identical-protocol metrics, checkpoint lineage, MP4 and Rerun outputs.",
        artifacts={
            "report": args.output_uri,
            "mp4": args.final_mp4_uri,
            "rrd": args.final_rrd_uri,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI. Args: None. Returns: Parser. Raises: None."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare")
    prepare.add_argument("--run-id", required=True)
    prepare.add_argument("--output-uri", required=True)
    prepare.add_argument("--component-root-uri", required=True)
    external = commands.add_parser("external-seam")
    external.add_argument("--run-id", required=True)
    external.add_argument("--output-uri", required=True)
    external.add_argument("--component-root-uri", required=True)
    final = commands.add_parser("finalize")
    final.add_argument("--run-id", required=True)
    for name in (
        "baseline-uri",
        "adapted-uri",
        "training-uri",
        "reload-uri",
        "checkpoint-manifest-uri",
        "external-seam-uri",
        "adapted-artifact-root-uri",
        "final-mp4-uri",
        "final-rrd-uri",
        "output-uri",
        "component-root-uri",
    ):
        final.add_argument(f"--{name}", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a stage. Args: argv. Returns: Exit status. Raises: RuntimeError."""
    args = build_parser().parse_args(argv)
    handlers = {
        "prepare": _prepare,
        "external-seam": _external_seam,
        "finalize": _finalize,
    }
    result = handlers[args.command](args)
    print(json.dumps(result, sort_keys=True), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
