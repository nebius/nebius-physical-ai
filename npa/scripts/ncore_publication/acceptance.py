"""Assemble and verify receipt-derived NCore publication acceptance."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import re
from typing import Any

from image_byte_scan import core as W, prepare as P
from npa.deploy import images
from npa.workbench.nurec.evidence import validate_runtime_attestation

from .process import ROOT, committed_source, file_sha, write_json
from .vlm_evidence import verify_complete_evidence


STATEMENT_FORMAT = "npa_ncore_acceptance_statement_v1"
REVIEW_FORMAT = "npa_ncore_acceptance_review_v1"
ACCEPTANCE_FORMAT = "npa_ncore_receipt_derived_acceptance_v1"
STATEMENT_PATH = Path("acceptance/statement.json")
REVIEW_PATH = Path("acceptance/review.json")
FINAL_PATH = Path("acceptance/accepted-manifest.json")
_HASH = re.compile(r"[0-9a-f]{64}")


def _canonical(value: Any) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _sha_value(value: Any) -> str:
    return hashlib.sha256(_canonical(value)).hexdigest()


def _private_root(root: Path) -> Path:
    root = root.absolute()
    if (
        root.is_symlink()
        or not root.is_dir()
        or root.stat().st_uid != os.getuid()
        or root.stat().st_mode & 0o077
    ):
        raise ValueError("acceptance analysis root is not private")
    return root


def _relative_file(root: Path, path: Path, label: str) -> Path:
    path = path.absolute()
    if (
        not path.is_relative_to(root)
        or path.is_symlink()
        or not path.is_file()
        or path.stat().st_uid != os.getuid()
        or path.stat().st_nlink != 1
        or path.stat().st_mode & 0o077
    ):
        raise ValueError(f"{label} is outside the private acceptance root")
    return path


def _json(path: Path) -> dict[str, Any]:
    payload = W.bound_json(P.binding(path))
    W.require(isinstance(payload, dict), "acceptance_evidence_not_object")
    return payload


def _inventory(root: Path, evidence_root: Path, gate_dir: Path) -> list[dict[str, Any]]:
    records = []
    for directory in (root / "build", evidence_root, gate_dir):
        for path in sorted(directory.rglob("*")):
            W.require(not path.is_symlink(), "acceptance_evidence_symlink")
            if path.is_dir():
                continue
            W.require(path.is_file(), "acceptance_evidence_special_file")
            records.append(
                {
                    "path": path.relative_to(root).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": file_sha(path),
                }
            )
    paths = [record["path"] for record in records]
    W.require(paths and len(paths) == len(set(paths)), "acceptance_evidence_population")
    return records


def _hash_field(record: dict[str, Any], field: str, path: Path) -> None:
    W.require(record.get(field) == file_sha(path), f"acceptance_{field}_binding")


def _trivy_counts(payload: dict[str, Any]) -> dict[str, int]:
    results = payload.get("Results")
    W.require(isinstance(results, list), "acceptance_trivy_results")
    vulnerabilities = []
    secrets = []
    for result in results:
        W.require(isinstance(result, dict), "acceptance_trivy_result")
        vulnerabilities.extend(
            finding
            for finding in (result.get("Vulnerabilities") or [])
            if isinstance(finding, dict)
            and str(finding.get("Severity") or "").upper() == "CRITICAL"
        )
        secrets.extend(
            finding
            for finding in (result.get("Secrets") or [])
            if isinstance(finding, dict)
        )
    fixed = [
        finding
        for finding in vulnerabilities
        if str(finding.get("FixedVersion") or "").strip()
    ]
    return {
        "critical_total": len(vulnerabilities),
        "critical_with_fix": len(fixed),
        "critical_unfixed": len(vulnerabilities) - len(fixed),
        "secrets": len(secrets),
    }


def _prepublication(
    manifest: dict[str, Any], analysis_root: Path, gate_dir: Path
) -> None:
    build = _json(analysis_root / "build/build.json")
    prepublication = _json(gate_dir / "prepublication.json")
    evidence_manifest_path = gate_dir / "evidence-manifest.json"
    evidence_manifest = _json(evidence_manifest_path)
    W.require(
        manifest.get("development_sha") == build.get("source_sha")
        and committed_source(manifest["development_sha"]) == build.get("context_sha256")
        and manifest.get("oci_digest") == build.get("image_digest")
        and manifest.get("amd64_manifest") == prepublication.get("platform_digest")
        and manifest.get("config_digest") == prepublication.get("config_digest")
        and prepublication.get("source_sha") == build.get("source_sha")
        and prepublication.get("image_digest") == build.get("image_digest")
        and prepublication.get("archive_sha256") == build.get("archive_sha256"),
        "acceptance_candidate_identity",
    )
    for item in evidence_manifest.get("files", []):
        W.require(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and file_sha(gate_dir / item["path"]) == item.get("sha256")
            and (gate_dir / item["path"]).stat().st_size == item.get("bytes"),
            "acceptance_gate_evidence_manifest",
        )
    pre = manifest["prepublication"]
    W.require(
        pre.get("evidence_manifest_sha256") == file_sha(evidence_manifest_path),
        "acceptance_prepublication_manifest",
    )
    paths = {
        "graph_receipt_sha256": "graph.json",
        "raw_byte_report_sha256": "bytes/report.json",
        "raw_byte_ledger_sha256": "bytes/records.jsonl",
        "attribution_receipt_sha256": "attribution.json",
        "attribution_replay_report_sha256": "attribution-replay/report.json",
        "attribution_replay_ledger_sha256": "attribution-replay/records.jsonl",
        "provenance_sbom_sha256": "buildx.spdx.json",
        "source_delivery_receipt_sha256": "source-delivery.log",
        "component_receipt_sha256": "components/receipt.json",
        "selected_base_receipt_sha256": "selected-base/selected.receipt.json",
        "bootstrap_receipt_sha256": "local-image-binding.json",
        "payload_receipt_sha256": "payload.json",
        "vulnerability_receipt_sha256": "trivy-policy.json",
        "license_receipt_sha256": "trivy-all.json",
    }
    for field, relative in paths.items():
        _hash_field(pre, field, gate_dir / relative)
    byte_report = _json(gate_dir / "bytes/report.json")
    attribution = _json(gate_dir / "attribution.json")
    byte_scan = manifest["byte_scan"]
    W.require(
        byte_report.get("complete") is True
        and byte_report.get("valid") is True
        and byte_scan.get("status") == "pass"
        and byte_scan.get("report_sha256") == file_sha(gate_dir / "bytes/report.json")
        and byte_scan.get("image_digest") == manifest["oci_digest"]
        and byte_scan.get("archive_sha256") == prepublication["archive_sha256"]
        and byte_scan.get("config_digest") == manifest["config_digest"]
        and byte_scan.get("policy_sha256") == attribution.get("policy_sha256")
        and byte_scan.get("complete") is byte_report["complete"]
        and byte_scan.get("bytes_scanned") == byte_report.get("scanned_bytes")
        and byte_scan.get("files_scanned") == byte_report.get("regular_files")
        and byte_scan.get("unresolved_findings") == byte_report.get("findings"),
        "acceptance_byte_scan_results",
    )
    payload = _json(gate_dir / "payload.json")
    payload_scan = manifest["payload_scan"]
    W.require(
        payload.get("scan_complete") is True
        and payload.get("verdict") == "clean"
        and payload_scan.get("status") == "pass"
        and payload_scan.get("report_sha256") == file_sha(gate_dir / "payload.json")
        and payload_scan.get("image_digest") == manifest["oci_digest"]
        and payload_scan.get("entries_scanned") == payload.get("entries_scanned")
        and payload_scan.get("payload_hits") == len(payload.get("payload_hits") or [])
        and payload_scan.get("history_hits") == len(payload.get("history_hits") or [])
        and payload_scan.get("weight_shaped_paths")
        == len(payload.get("weight_shaped_paths") or [])
        and payload_scan.get("weight_review_sha256")
        == file_sha(gate_dir / "payload-history.json"),
        "acceptance_payload_scan_results",
    )
    vulnerability = manifest["vulnerability_scan"]
    W.require(
        vulnerability.get("status") == "pass"
        and vulnerability.get("report_sha256") == file_sha(gate_dir / "trivy-all.json")
        and vulnerability.get("image_digest") == manifest["oci_digest"]
        and {
            field: vulnerability.get(field)
            for field in (
                "critical_total",
                "critical_with_fix",
                "critical_unfixed",
                "secrets",
            )
        }
        == _trivy_counts(_json(gate_dir / "trivy-all.json")),
        "acceptance_vulnerability_scan_results",
    )
    selected = _json(gate_dir / "selected-base/selected.receipt.json")
    W.require(
        manifest.get("source", {}).get("lock_sha256")
        == file_sha(ROOT / "npa/docker/workbench/ncore/base-source-lock.json")
        and manifest.get("source", {}).get("post_patch_inventory_sha256")
        == file_sha(gate_dir / "inventory.json")
        and manifest.get("selected_base_scan") == selected,
        "acceptance_selected_base_scan_results",
    )


def _qualification(
    manifest: dict[str, Any],
    evidence_root: Path,
) -> dict[str, Any]:
    controls = manifest["qualification_controls"]
    control_paths = {
        "source_acquisition_receipt_sha256": "source-acquisition.json",
        "s3_probe_receipt_sha256": "s3-handoff-probe.json",
        "source_staging_receipt_sha256": "source-staging.json",
        "candidate_image_receipt_sha256": "candidate-image.json",
        "qualification_execution_receipt_sha256": "qualification-execution.json",
        "wrong_source_execution_receipt_sha256": "wrong-source-execution.json",
        "conversion_execution_receipt_sha256": "convert-execution.json",
        "audit_execution_receipt_sha256": "audit-execution.json",
        "wrong_source_receipt_sha256": "wrong-source.json",
    }
    for field, relative in control_paths.items():
        _hash_field(controls, field, evidence_root / relative)
    acquisition = _json(evidence_root / "source-acquisition.json")
    probe = _json(evidence_root / "s3-handoff-probe.json")
    staging = _json(evidence_root / "source-staging.json")
    candidate = _json(evidence_root / "candidate-image.json")
    execution = _json(evidence_root / "qualification-execution.json")
    execution_receipts = {
        role: _json(evidence_root / f"{role}-execution.json")
        for role in ("wrong-source", "convert", "audit")
    }
    wrong = _json(evidence_root / "wrong-source.json")
    W.require(
        acquisition.get("format") == "npa_ncore_public_source_acquisition_v1"
        and acquisition.get("status") == "pass"
        and acquisition.get("archive_sha256")
        == manifest["conversion"]["source_archive_sha256"]
        and probe.get("format") == "npa_ncore_s3_handoff_probe_v1"
        and probe.get("status") == "pass"
        and staging.get("format") == "npa_ncore_source_staging_v1"
        and staging.get("status") == "pass"
        and candidate.get("format") == "npa.ncore.qualification-candidate-image.v1"
        and candidate.get("status") == "pass"
        and candidate.get("source_sha") == manifest["development_sha"]
        and candidate.get("image_digest")
        == manifest["conversion"]["observed_image_digest"]
        and execution.get("format") == "npa.ncore.candidate-qualification.v1"
        and execution.get("status") == "pass"
        and execution.get("chronology") == ["wrong-source", "convert", "audit"]
        and execution.get("candidate_image_receipt_sha256")
        == file_sha(evidence_root / "candidate-image.json")
        and execution.get("wrong_source_receipt_sha256")
        == file_sha(evidence_root / "wrong-source.json")
        and execution.get("executions")
        == {
            role: file_sha(evidence_root / f"{role}-execution.json")
            for role in ("wrong-source", "convert", "audit")
        }
        and all(
            receipt.get("format") == "npa.ncore.host-container-execution.v1"
            and receipt.get("status") == "pass"
            and receipt.get("role") == role
            and receipt.get("exit_code") == 0
            and receipt.get("candidate_image_receipt_sha256")
            == file_sha(evidence_root / "candidate-image.json")
            and receipt.get("local_image_id") == candidate.get("local_image_id")
            for role, receipt in execution_receipts.items()
        )
        and wrong.get("format") == controls.get("wrong_source_format")
        and wrong.get("status") == "pass"
        and wrong.get("failure_phase") == controls.get("wrong_source_failure_phase")
        and wrong.get("native_started") is controls.get("wrong_source_native_started")
        and execution_receipts["wrong-source"].get("exit_code")
        == controls.get("control_command_exit_code")
        and wrong.get("after_output_objects")
        == controls.get("wrong_source_output_objects"),
        "acceptance_qualification_controls",
    )
    readback_root = evidence_root / "readback"
    readback_receipt_path = evidence_root / "qualification-readback.json"
    objective_path = evidence_root / "qualification-audit.json"
    cleanup_path = evidence_root / "cleanup.json"
    readback = _json(readback_receipt_path)
    objective = _json(objective_path)
    cleanup = _json(cleanup_path)
    runtime_path = readback_root / "evidence/nre-runtime.json"
    workflow_status_path = readback_root / "evidence/workflow-status.json"
    final_report_path = readback_root / "reports/final.json"
    conversion_path = readback_root / "ncore/sequence/conversion.json"
    conversion_audit_path = readback_root / "evidence/ncore-conversion-audit.json"
    reconstruction_path = readback_root / "reconstruction/reconstruction.json"
    render_path = readback_root / "novel_views/nre-render.json"
    rrd_path = readback_root / "reports/sim2real.rrd"
    runtime = _json(runtime_path)
    conversion_report = _json(conversion_path)
    conversion_audit = _json(conversion_audit_path)
    reconstruction = _json(reconstruction_path)
    render = _json(render_path)
    validate_runtime_attestation(
        runtime,
        expected_image=manifest["rtx_proof"]["nre_image"],
        required_stages=("reconstruct", "render"),
    )
    proof = manifest["rtx_proof"]
    conversion = manifest["conversion"]
    audit_source = conversion_audit.get("source")
    audit_conversion = conversion_audit.get("conversion")
    W.require(
        readback.get("format") == "npa_ncore_qualification_readback_v1"
        and readback.get("status") == "pass"
        and objective.get("format") == "npa_ncore_qualification_audit_v1"
        and objective.get("status") == "pass"
        and proof.get("report_sha256") == file_sha(objective_path)
        and proof.get("complete_readback_receipt_sha256")
        == file_sha(readback_receipt_path)
        and proof.get("runtime_image_attestation_sha256") == file_sha(runtime_path)
        and proof.get("final_workflow_status_sha256") == file_sha(workflow_status_path)
        and proof.get("final_report_sha256") == file_sha(final_report_path)
        and conversion.get("report_sha256") == file_sha(conversion_path)
        and conversion.get("audit_sha256") == file_sha(conversion_audit_path)
        and objective.get("conversion_report_sha256") == conversion["report_sha256"]
        and objective.get("conversion_audit_sha256") == conversion["audit_sha256"]
        and objective.get("runtime_attestation_sha256")
        == proof["runtime_image_attestation_sha256"]
        and objective.get("complete_readback", {}).get("receipt_sha256")
        == proof["complete_readback_receipt_sha256"]
        and objective.get("workflow_status", {}).get("sha256")
        == proof["final_workflow_status_sha256"]
        and objective.get("final_report_sha256") == proof["final_report_sha256"]
        and objective.get("metrics") == proof.get("observed_metrics")
        and objective.get("recipe", {}).get("max_epochs_argument")
        == proof.get("max_epochs")
        and objective.get("usdz", {}).get("sha256") == proof.get("usdz_sha256")
        and objective.get("usdz", {}).get("bytes") == proof.get("usdz_bytes")
        and objective.get("render", {}).get("novel_view") is proof.get("novel_view"),
        "acceptance_objective_evidence",
    )
    W.require(
        isinstance(audit_source, dict)
        and isinstance(audit_conversion, dict)
        and conversion.get("source_archive_sha256")
        == audit_source.get("archive_sha256")
        and conversion.get("source_inventory_sha256")
        == audit_source.get("inventory_sha256")
        and conversion.get("converted_inventory_sha256")
        == audit_conversion.get("inventory_sha256")
        and conversion.get("source_counts") == audit_source.get("counts")
        and conversion.get("converted_counts") == audit_conversion.get("counts")
        and conversion.get("origin_points_filtered")
        == audit_conversion.get("origin_points_filtered")
        and conversion.get("all_members_reopened")
        is audit_conversion.get("all_members_reopened")
        and conversion.get("member_hashes_verified")
        is audit_conversion.get("member_hashes_verified")
        and conversion.get("calibration_verified")
        is audit_conversion.get("calibration_verified")
        and conversion.get("poses_verified") is audit_conversion.get("poses_verified")
        and conversion.get("finite_geometry") is audit_conversion.get("finite_geometry")
        and conversion.get("rig_mode")
        == conversion_report.get("options", {}).get("rig_mode")
        and conversion.get("poses_component_group")
        == audit_conversion.get("poses_component_group")
        and conversion.get("camera_frame_counts")
        == audit_source.get("camera_frame_counts")
        and conversion.get("camera_frame_inventory_sha256")
        == audit_source.get("camera_frame_inventory_sha256"),
        "acceptance_conversion_evidence",
    )
    training = proof.get("training_input")
    rrd = proof.get("rrd")
    W.require(
        isinstance(training, dict)
        and isinstance(rrd, dict)
        and reconstruction.get("status") == "pass"
        and render.get("status") == "pass"
        and proof.get("train_exit_code")
        == reconstruction.get("invocation", {}).get("train_exit_code")
        and proof.get("render_exit_code")
        == render.get("invocation", {}).get("render_exit_code")
        and proof.get("usdz_sha256")
        == reconstruction.get("outputs", {}).get("usdz", {}).get("sha256")
        and proof.get("usdz_bytes")
        == reconstruction.get("outputs", {}).get("usdz", {}).get("bytes")
        and proof.get("usd_runtime_version")
        == objective.get("usdz", {}).get("usd_runtime_version")
        and proof.get("rendered_usdz_sha256")
        == render.get("input_usdz", {}).get("sha256")
        and proof.get("render_sha256") == file_sha(render_path)
        and proof.get("render_bytes") == render.get("output", {}).get("bytes")
        and proof.get("decoded_frames") == render.get("output", {}).get("frame_count")
        and proof.get("video_count") == render.get("output", {}).get("video_count")
        and proof.get("decoded_video_frames")
        == render.get("output", {}).get("decoded_video_frames")
        and proof.get("finite_pixels") is render.get("output", {}).get("finite_pixels")
        and proof.get("novel_view") is render.get("invocation", {}).get("novel_view")
        and training.get("reconstruction_receipt_sha256")
        == file_sha(reconstruction_path)
        and training.get("render_receipt_sha256") == file_sha(render_path)
        and training.get("parsed_config_sha256")
        == reconstruction.get("outputs", {}).get("parsed_config", {}).get("sha256")
        and training.get("native_recipe")
        == reconstruction.get("recipe", {}).get("name")
        and training.get("max_epochs_argument")
        == reconstruction.get("recipe", {}).get("max_epochs_argument")
        and training.get("resolved_epochs")
        == reconstruction.get("recipe", {}).get("resolved_epochs")
        and training.get("resolved_samples_per_epoch")
        == reconstruction.get("recipe", {}).get("resolved_samples_per_epoch")
        and training.get("source_camera_frame_counts")
        == conversion.get("camera_frame_counts")
        and training.get("source_frame_inventory_sha256")
        == conversion.get("camera_frame_inventory_sha256")
        and training.get("conversion_report_sha256") == conversion.get("report_sha256")
        and training.get("sequence_inventory_sha256")
        == reconstruction.get("input", {}).get("sequence_inventory_sha256")
        and rrd.get("report_sha256") == file_sha(objective_path)
        and rrd.get("sha256") == objective.get("rrd", {}).get("sha256")
        and rrd.get("bytes") == objective.get("rrd", {}).get("bytes")
        and rrd.get("decoded_rows") == objective.get("rrd", {}).get("decoded_rows")
        and rrd.get("decoded_frames") == objective.get("rrd", {}).get("decoded_frames")
        and rrd.get("conversion_report_sha256") == conversion.get("report_sha256")
        and rrd.get("source_inventory_sha256")
        == conversion.get("source_inventory_sha256")
        and rrd.get("lineage_verified")
        is objective.get("rrd", {}).get("lineage_verified")
        and rrd.get("frame_artifacts_verified")
        is objective.get("rrd", {}).get("frame_artifacts_verified")
        and file_sha(rrd_path) == objective.get("rrd", {}).get("sha256"),
        "acceptance_native_receipt_evidence",
    )
    _hash_field(manifest["cleanup"], "receipt_sha256", cleanup_path)
    for field in (
        "format",
        "status",
        "build_receipt_sha256",
        "workflow_status_sha256",
        "managed_job_identities_sha256",
        "managed_jobs",
        "storage_inventory_sha256",
        "storage_objects",
        "active_job_pods",
        "jobs_terminal_or_absent",
        "cancel_before_destroy",
        "controller_disposition",
        "storage_disposition",
        "registry_disposition",
        "local_runtime_disposition",
        "orphan_count",
    ):
        W.require(
            manifest["cleanup"].get(field) == cleanup.get(field),
            f"acceptance_cleanup_{field}",
        )
    return {
        "objective_sha256": file_sha(objective_path),
        "readback_sha256": file_sha(readback_receipt_path),
        "cleanup_sha256": file_sha(cleanup_path),
    }


def _visual(manifest: dict[str, Any], evidence_root: Path) -> dict[str, Any]:
    visual = manifest["rtx_proof"]["visual_review"]
    root = evidence_root / "vlm"
    prefix_path = root / "external-attempt-prefix.txt"
    prefix = (
        _relative_file(evidence_root, prefix_path, "attempt prefix")
        .read_text(encoding="utf-8")
        .strip()
    )
    paths = {
        "control_manifest_sha256": "freeze.json",
        "label_commitment_sha256": "labels.json",
        "freeze_acceptance_sha256": "freeze-acceptance.json",
        "calibration_result_sha256": "calibration.json",
        "final_frame_manifest_sha256": "final-frame-manifest.json",
        "final_result_sha256": "final.json",
        "raw_transport_manifest_sha256": "transport-manifest.json",
    }
    for field, relative in paths.items():
        _hash_field(visual, field, root / relative)
    acceptance = _json(root / "freeze-acceptance.json")
    W.require(
        visual.get("freeze_review_receipt_sha256")
        == acceptance.get("review_receipt_sha256")
        == file_sha(root / "freeze-review.json")
        and visual.get("external_attempt_prefix_sha256")
        == hashlib.sha256(prefix.encode()).hexdigest(),
        "acceptance_visual_review_binding",
    )
    verified = verify_complete_evidence(
        root,
        freeze_sha256=visual["control_manifest_sha256"],
        freeze_acceptance_sha256=visual["freeze_acceptance_sha256"],
        calibration_sha256=visual["calibration_result_sha256"],
        final_sha256=visual["final_result_sha256"],
        external_attempt_prefix=prefix,
        rubric_path=ROOT / "npa/scripts/ncore_publication/vlm-rubric-v2.txt",
        calibration_task_path=ROOT
        / "npa/scripts/ncore_publication/vlm-calibration-task-v2.txt",
        final_task_path=ROOT / "npa/scripts/ncore_publication/vlm-final-task-v2.txt",
    )
    calibration = verified["calibration"]
    final = verified["final"]
    W.require(
        visual.get("calibration_total") == calibration.get("total")
        and visual.get("true_positives") == calibration.get("true_positives")
        and visual.get("true_negatives") == calibration.get("true_negatives")
        and visual.get("false_positives") == calibration.get("false_positives")
        and visual.get("false_negatives") == calibration.get("false_negatives")
        and visual.get("final_score") == final.get("score")
        and visual.get("final_passed") is (final.get("status") == "pass")
        and visual.get("attempt_count") == final.get("attempt_count")
        and visual.get("external_attempt_markers") == final.get("attempt_count"),
        "acceptance_visual_results",
    )
    return {
        "freeze_sha256": visual["control_manifest_sha256"],
        "final_sha256": visual["final_result_sha256"],
        "transport_manifest_sha256": verified["transport_manifest_sha256"],
    }


def build_statement(
    *,
    analysis_root: Path,
    gate_dir: Path,
    evidence_root: Path,
    proposed_manifest_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Build one immutable statement from actual gate and workload receipts."""
    analysis_root = _private_root(analysis_root)
    gate_dir = _relative_file(
        analysis_root, gate_dir / "prepublication.json", "gate directory"
    ).parent
    evidence_root = _relative_file(
        analysis_root,
        evidence_root / "qualification-audit.json",
        "qualification evidence",
    ).parent
    proposed_manifest_path = _relative_file(
        analysis_root, proposed_manifest_path, "proposed manifest"
    )
    W.require(
        output_path.absolute() == analysis_root / STATEMENT_PATH,
        "acceptance_statement_path",
    )
    manifest = _json(proposed_manifest_path)
    manifest.pop("acceptance_verification", None)
    validated = copy.deepcopy(manifest)
    validated["acceptance_verification"] = {
        "format": ACCEPTANCE_FORMAT,
        "statement_sha256": "0" * 64,
        "evidence_inventory_sha256": "0" * 64,
        "review_receipt_sha256": "0" * 64,
        "reviewer_id_sha256": "0" * 64,
    }
    images.validate_ncore_accepted_image_manifest(validated)
    _prepublication(manifest, analysis_root, gate_dir)
    objective = _qualification(manifest, evidence_root)
    visual = _visual(manifest, evidence_root)
    inventory = _inventory(analysis_root, evidence_root, gate_dir)
    statement = {
        "format": STATEMENT_FORMAT,
        "status": "pending_independent_review",
        "candidate_commit": manifest["development_sha"],
        "manifest": manifest,
        "manifest_sha256": _sha_value(manifest),
        "evidence_inventory": inventory,
        "evidence_inventory_sha256": _sha_value(inventory),
        "objective": objective,
        "visual": visual,
    }
    output_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    write_json(output_path, statement)
    return statement


def finalize_acceptance(
    *,
    analysis_root: Path,
    statement_path: Path,
    review_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Require an exact independent review before emitting accepted manifest."""
    analysis_root = _private_root(analysis_root)
    W.require(
        statement_path.absolute() == analysis_root / STATEMENT_PATH
        and review_path.absolute() == analysis_root / REVIEW_PATH
        and output_path.absolute() == analysis_root / FINAL_PATH,
        "acceptance_finalization_paths",
    )
    statement = _json(_relative_file(analysis_root, statement_path, "statement"))
    review = _json(_relative_file(analysis_root, review_path, "review"))
    statement_sha = file_sha(statement_path)
    W.require(
        statement.get("format") == STATEMENT_FORMAT
        and statement.get("status") == "pending_independent_review"
        and review.get("format") == REVIEW_FORMAT
        and review.get("verdict") == "ACCEPTED"
        and review.get("candidate_commit") == statement.get("candidate_commit")
        and review.get("statement_sha256") == statement_sha
        and review.get("evidence_inventory_sha256")
        == statement.get("evidence_inventory_sha256")
        and review.get("manifest_sha256") == statement.get("manifest_sha256")
        and review.get("objective_evidence_reviewed") is True
        and review.get("visual_evidence_reviewed") is True
        and review.get("cleanup_reviewed") is True
        and isinstance(review.get("reviewer_id"), str)
        and bool(review["reviewer_id"].strip()),
        "acceptance_independent_review",
    )
    manifest = copy.deepcopy(statement["manifest"])
    manifest["acceptance_verification"] = {
        "format": ACCEPTANCE_FORMAT,
        "statement_sha256": statement_sha,
        "evidence_inventory_sha256": statement["evidence_inventory_sha256"],
        "review_receipt_sha256": file_sha(review_path),
        "reviewer_id_sha256": hashlib.sha256(
            review["reviewer_id"].encode()
        ).hexdigest(),
    }
    images.validate_ncore_accepted_image_manifest(manifest)
    write_json(output_path, manifest)
    return manifest


def verify_final_acceptance(
    analysis_root: Path, acceptance_path: Path
) -> dict[str, Any]:
    """Recheck the protected statement, review, and every bound evidence file."""
    analysis_root = _private_root(analysis_root)
    W.require(
        acceptance_path.absolute() == analysis_root / FINAL_PATH,
        "accepted_manifest_path",
    )
    manifest = _json(_relative_file(analysis_root, acceptance_path, "acceptance"))
    statement_path = analysis_root / STATEMENT_PATH
    review_path = analysis_root / REVIEW_PATH
    statement = _json(_relative_file(analysis_root, statement_path, "statement"))
    review = _json(_relative_file(analysis_root, review_path, "review"))
    verification = manifest.get("acceptance_verification")
    W.require(
        isinstance(verification, dict)
        and verification.get("statement_sha256") == file_sha(statement_path)
        and verification.get("review_receipt_sha256") == file_sha(review_path)
        and verification.get("evidence_inventory_sha256")
        == statement.get("evidence_inventory_sha256")
        and review.get("statement_sha256") == file_sha(statement_path)
        and review.get("verdict") == "ACCEPTED",
        "accepted_manifest_review_binding",
    )
    for item in statement.get("evidence_inventory", []):
        W.require(
            isinstance(item, dict)
            and isinstance(item.get("path"), str)
            and _HASH.fullmatch(str(item.get("sha256") or ""))
            and _relative_file(
                analysis_root,
                analysis_root / item["path"],
                "accepted evidence",
            )
            .stat()
            .st_size
            == item.get("bytes")
            and file_sha(analysis_root / item["path"]) == item["sha256"],
            "accepted_evidence_inventory_changed",
        )
    without_verification = copy.deepcopy(manifest)
    without_verification.pop("acceptance_verification", None)
    W.require(
        without_verification == statement.get("manifest")
        and _sha_value(without_verification) == statement.get("manifest_sha256"),
        "accepted_manifest_statement_changed",
    )
    return images.validate_ncore_accepted_image_manifest(manifest)
