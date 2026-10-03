"""Build synthetic receipt-binding fixtures; these are never live qualification proof."""

import copy
import importlib.util
import json
from pathlib import Path

from npa.workbench.nurec.qualification_audit import _usdz


ROOT = Path(__file__).resolve().parents[3]


def _module(relative):
    spec = importlib.util.spec_from_file_location(Path(relative).stem, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write(root, relative, value):
    path = root / relative
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True))
    path.chmod(0o600)
    return path


def _acquisition(root, manifest):
    candidate = {
        "format": "npa.ncore.qualification-candidate-image.v1",
        "status": "pass",
        "source_sha": manifest["development_sha"],
        "image_digest": manifest["conversion"]["observed_image_digest"],
        "local_image_id": "sha256:" + "e" * 64,
    }
    _write(root, "candidate-image.json", candidate)
    _write(
        root,
        "source-acquisition.json",
        {
            "format": "npa_ncore_public_source_acquisition_v1",
            "status": "pass",
            "archive_sha256": manifest["conversion"]["source_archive_sha256"],
        },
    )
    for role, format_name in (
        ("s3-handoff-probe", "s3_handoff_probe"),
        ("source-staging", "source_staging"),
    ):
        _write(
            root,
            role + ".json",
            {"format": f"npa_ncore_{format_name}_v1", "status": "pass"},
        )
    return candidate


def _execution_controls(root, candidate, sha):
    _write(
        root,
        "wrong-source.json",
        {
            "format": "npa_ncore_wrong_source_control_v1",
            "status": "pass",
            "failure_phase": "source_digest_pre_extract",
            "native_started": False,
            "after_output_objects": 0,
        },
    )
    executions = {}
    for role in ("wrong-source", "convert", "audit"):
        path = _write(
            root,
            f"{role}-execution.json",
            {
                "format": "npa.ncore.host-container-execution.v1",
                "status": "pass",
                "role": role,
                "exit_code": 0,
                "local_image_id": candidate["local_image_id"],
                "candidate_image_receipt_sha256": sha(root / "candidate-image.json"),
            },
        )
        executions[role] = sha(path)
    return executions


def _controls(root, manifest, sha):
    candidate = _acquisition(root, manifest)
    executions = _execution_controls(root, candidate, sha)
    _write(
        root,
        "qualification-execution.json",
        {
            "format": "npa.ncore.candidate-qualification.v1",
            "status": "pass",
            "chronology": ["wrong-source", "convert", "audit"],
            "executions": executions,
            "candidate_image_receipt_sha256": sha(root / "candidate-image.json"),
            "wrong_source_receipt_sha256": sha(root / "wrong-source.json"),
        },
    )
    names = (
        "source_acquisition:source-acquisition",
        "s3_probe:s3-handoff-probe",
        "source_staging:source-staging",
        "candidate_image:candidate-image",
        "qualification_execution:qualification-execution",
        "wrong_source_execution:wrong-source-execution",
        "conversion_execution:convert-execution",
        "audit_execution:audit-execution",
        "wrong_source:wrong-source",
    )
    for item in names:
        field, filename = item.split(":")
        manifest["qualification_controls"][field + "_receipt_sha256"] = sha(
            root / (filename + ".json")
        )


def _conversion(root, manifest, sha):
    conversion = manifest["conversion"]
    report = {"options": {"rig_mode": conversion["rig_mode"]}}
    report_path = _write(root, "ncore/sequence/conversion.json", report)
    source, converted = _conversion_metadata(conversion)
    audit = _write(
        root,
        "evidence/ncore-conversion-audit.json",
        {"source": source, "conversion": converted},
    )
    conversion.update(report_sha256=sha(report_path), audit_sha256=sha(audit))


def _conversion_metadata(conversion):
    source = {
        name: conversion[key]
        for name, key in (
            ("archive_sha256", "source_archive_sha256"),
            ("inventory_sha256", "source_inventory_sha256"),
            ("counts", "source_counts"),
            ("camera_frame_counts", "camera_frame_counts"),
            ("camera_frame_inventory_sha256", "camera_frame_inventory_sha256"),
        )
    }
    converted = {
        name: conversion[name]
        for name in (
            "origin_points_filtered",
            "all_members_reopened",
            "member_hashes_verified",
            "calibration_verified",
            "poses_verified",
            "finite_geometry",
            "poses_component_group",
        )
    }
    converted.update(
        inventory_sha256=conversion["converted_inventory_sha256"],
        counts=conversion["converted_counts"],
    )
    return source, converted


def _native_documents(proof, usdz):
    training = proof["training_input"]
    reconstruction = {
        "status": "pass",
        "invocation": {"train_exit_code": 0},
        "outputs": {
            "usdz": usdz,
            "parsed_config": {"sha256": training["parsed_config_sha256"]},
        },
        "recipe": {
            "name": training["native_recipe"],
            "max_epochs_argument": 0,
            "resolved_epochs": 1,
            "resolved_samples_per_epoch": 30000,
        },
        "input": {"sequence_inventory_sha256": training["sequence_inventory_sha256"]},
    }
    render = {
        "status": "pass",
        "invocation": {"render_exit_code": 0, "novel_view": True},
        "input_usdz": usdz,
        "output": {
            "bytes": proof["render_bytes"],
            "frame_count": proof["decoded_frames"],
            "video_count": proof["video_count"],
            "decoded_video_frames": proof["decoded_video_frames"],
            "finite_pixels": True,
        },
    }
    return reconstruction, render


def _native(root, manifest, usdz, sha):
    proof = manifest["rtx_proof"]
    training = proof["training_input"]
    proof.update(
        usdz_sha256=usdz["sha256"],
        usdz_bytes=usdz["bytes"],
        rendered_usdz_sha256=usdz["sha256"],
        usd_runtime_version=usdz["usd_runtime_version"],
    )
    reconstruction, render = _native_documents(proof, usdz)
    reconstruction_path = _write(
        root, "reconstruction/reconstruction.json", reconstruction
    )
    render_path = _write(root, "novel_views/nre-render.json", render)
    training.update(
        reconstruction_receipt_sha256=sha(reconstruction_path),
        render_receipt_sha256=sha(render_path),
        conversion_report_sha256=manifest["conversion"]["report_sha256"],
    )
    proof["render_sha256"] = sha(render_path)


def _runtime_observation(readback):
    runtime_fixture = _module("npa/tests/workbench/test_nurec_runtime_attestation.py")
    _, reconstruct = runtime_fixture._observe(readback, "reconstruct")
    _, render = runtime_fixture._observe(readback, "render")
    runtime = readback / "evidence/nre-runtime.json"
    runtime_fixture.runtime_attestation.bundle_runtime_attestations(
        reconstruct_path=reconstruct, render_path=render, output_path=runtime
    )
    return runtime


def _readback_summary(root, manifest, sha):
    proof = manifest["rtx_proof"]
    readback = root / "readback"
    status = _write(readback, "evidence/workflow-status.json", {"status": "SUCCEEDED"})
    final = _write(readback, "reports/final.json", {"has_usdz": True})
    rrd = readback / "reports/sim2real.rrd"
    rrd.write_bytes(b"synthetic binding fixture, not a decoded RRD")
    rrd.chmod(0o600)
    proof["rrd"].update(
        sha256=sha(rrd),
        bytes=rrd.stat().st_size,
        conversion_report_sha256=manifest["conversion"]["report_sha256"],
    )
    readback_receipt = _write(
        root,
        "qualification-readback.json",
        {"format": "npa_ncore_qualification_readback_v1", "status": "pass"},
    )
    return status, final, readback_receipt


def _objective(root, manifest, usdz, sha):
    proof = manifest["rtx_proof"]
    runtime = _runtime_observation(root / "readback")
    status, final, readback_receipt = _readback_summary(root, manifest, sha)
    objective = {
        "format": "npa_ncore_qualification_audit_v1",
        "status": "pass",
        "conversion_report_sha256": manifest["conversion"]["report_sha256"],
        "conversion_audit_sha256": manifest["conversion"]["audit_sha256"],
        "runtime_attestation_sha256": sha(runtime),
        "complete_readback": {"receipt_sha256": sha(readback_receipt)},
        "workflow_status": {"sha256": sha(status)},
        "final_report_sha256": sha(final),
        "metrics": proof["observed_metrics"],
        "recipe": {"max_epochs_argument": 0},
        "usdz": usdz,
        "render": {"novel_view": True},
        "rrd": copy.deepcopy(proof["rrd"]),
    }
    objective_path = _write(root, "qualification-audit.json", objective)
    proof.update(
        report_sha256=sha(objective_path),
        complete_readback_receipt_sha256=sha(readback_receipt),
        runtime_image_attestation_sha256=sha(runtime),
        final_workflow_status_sha256=sha(status),
        final_report_sha256=sha(final),
        conversion_report_sha256=manifest["conversion"]["report_sha256"],
        conversion_audit_sha256=manifest["conversion"]["audit_sha256"],
    )
    proof["rrd"]["report_sha256"] = sha(objective_path)


def synthetic_statement_inputs(root, sha):
    """Create a real-USD, synthetic-GPU fixture for receipt contract integration.

    Args:
        root: Private temporary fixture directory.
        sha: Production file hashing function.
    Returns:
        Manifest and evidence root; these cannot be used as live acceptance proof.
    Raises:
        AssertionError: The synthetic USD package cannot be created.
    """
    root.chmod(0o700)
    fixture = _module("npa/tests/deploy/test_ncore_publication.py")
    manifest = fixture.accepted.__wrapped__()
    evidence = root / "evidence"
    readback = evidence / "readback"
    (readback / "reconstruction").mkdir(mode=0o700, parents=True)
    package = readback / "reconstruction/last.usdz"
    _module(
        "npa/tests/orchestration/npa_workflow/test_nurec_colmap_workflow.py"
    )._write_synthetic_usdz(package)
    usdz = _usdz(package)
    _controls(evidence, manifest, sha)
    _conversion(readback, manifest, sha)
    _native(readback, manifest, usdz, sha)
    _objective(evidence, manifest, usdz, sha)
    cleanup = _write(evidence, "cleanup.json", manifest["cleanup"])
    manifest["cleanup"]["receipt_sha256"] = sha(cleanup)
    for path in root.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    return manifest, evidence
