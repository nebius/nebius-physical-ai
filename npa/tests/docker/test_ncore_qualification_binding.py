"""Transplant valid foreign-run receipts through real acceptance binders."""

import copy
import hashlib
import json
import shutil

import pytest

from ncore_acceptance_fixture import synthetic_statement_inputs, _module
from test_ncore_acceptance import ROOT, W, acceptance, _write
from test_ncore_vlm_evidence import _calibrated_schedule
from ncore_publication import vlm_evidence as V


def _foreign_cleanup(manifest, evidence, dimension):
    """Produce another valid cleanup through the real producer, not edited claims."""
    fixture = _module("npa/tests/workbench/test_nurec_qualification_cleanup.py")
    status_path = evidence / "readback/evidence/workflow-status.json"
    status = json.loads(status_path.read_text())
    build_path = evidence.parent / "build/build.json"
    build = json.loads(build_path.read_text())
    run_id = "foreign-run" if dimension == "run" else status["run_id"]
    status["run_id"] = run_id
    for stage in status["stages"].values():
        if dimension == "jobs":
            stage["managed_job_id"] = str(int(stage["managed_job_id"]) + 100)
    if dimension == "build":
        build["source_sha"] = "b" * 40
    _write(status_path, status)
    _write(build_path, build)
    prefix = (
        "s3://private/foreign/evidence/"
        if dimension == "prefix"
        else "s3://private/run/evidence/"
    )
    readback_path = evidence / "qualification-readback.json"
    readback = json.loads(readback_path.read_text())
    readback["prefix_sha256"] = hashlib.sha256(prefix.encode()).hexdigest()
    _write(readback_path, readback)
    # Storage and native provider boundaries only. Actual producer binds identity.
    storage = fixture._Storage()
    storage.s3.get_paginator = lambda _: type(
        "Paginator",
        (),
        {
            "paginate": lambda self, **kw: [
                {
                    "Contents": [
                        {
                            "Key": kw["Prefix"] + "report.json",
                            "Size": 10,
                            "ETag": '"synthetic"',
                        }
                    ]
                }
            ]
        },
    )()
    receipt_path = evidence / "foreign-cleanup.json"
    receipt = fixture.cleanup_qualification(
        run_id=run_id,
        workflow_status_path=status_path,
        context="synthetic-context",
        namespace="synthetic-namespace",
        storage_prefix=prefix,
        local_image="local/ncore:candidate",
        builder="synthetic-builder",
        build_receipt_path=build_path,
        source_sha=build["source_sha"],
        output_path=receipt_path,
        storage_client=storage,
        process_runner=fixture._runner(),
        workflow_cleaner=fixture._cleaner,
    )
    _write(evidence / "cleanup.json", receipt)
    manifest["cleanup"] = {
        **receipt,
        "receipt_sha256": acceptance.file_sha(evidence / "cleanup.json"),
    }
    _refresh_objective(manifest, evidence)


def _refresh_objective(manifest, evidence):
    proof = manifest["rtx_proof"]
    readback_sha = acceptance.file_sha(evidence / "qualification-readback.json")
    status_sha = acceptance.file_sha(
        evidence / "readback/evidence/workflow-status.json"
    )
    objective_path = evidence / "qualification-audit.json"
    objective = json.loads(objective_path.read_text())
    objective["complete_readback"]["receipt_sha256"] = readback_sha
    objective["workflow_status"]["sha256"] = status_sha
    _write(objective_path, objective)
    proof.update(
        complete_readback_receipt_sha256=readback_sha,
        final_workflow_status_sha256=status_sha,
        report_sha256=acceptance.file_sha(objective_path),
    )
    proof["rrd"]["report_sha256"] = proof["report_sha256"]


@pytest.mark.parametrize("dimension", ["run", "jobs", "build", "prefix"])
def test_valid_foreign_cleanup_cannot_certify_another_qualification(
    tmp_path, dimension
):
    a, b = tmp_path / "a", tmp_path / "b"
    a.mkdir(mode=0o700)
    b.mkdir(mode=0o700)
    manifest_a, evidence_a = synthetic_statement_inputs(a, acceptance.file_sha)
    manifest_b, evidence_b = synthetic_statement_inputs(b, acceptance.file_sha)
    _foreign_cleanup(manifest_b, evidence_b, dimension)
    with W.authorized_roots(tmp_path, ROOT):
        for manifest, evidence in ((manifest_a, evidence_a), (manifest_b, evidence_b)):
            acceptance.images.validate_ncore_accepted_image_manifest(manifest)
            assert acceptance._qualification(manifest, evidence, evidence.parent)
        # Both entire qualification bundles pass alone. Transplant receipt plus
        # its matching manifest claims; internal hash checks cannot detect this.
        manifest_a["cleanup"] = copy.deepcopy(manifest_b["cleanup"])
        (evidence_a / "cleanup.json").write_bytes(
            (evidence_b / "cleanup.json").read_bytes()
        )
        with pytest.raises(
            ValueError, match="acceptance_cleanup_qualification_binding"
        ):
            acceptance._qualification(manifest_a, evidence_a, a)


def _visual_fixture(monkeypatch, root, *, camera, color_offset, source_sha):
    root.mkdir(mode=0o700)
    manifest, evidence = synthetic_statement_inputs(
        root, acceptance.file_sha, source_archive_sha256=source_sha
    )
    media = root / "media"
    media.mkdir(mode=0o700)
    args, calls, storage = _calibrated_schedule(
        monkeypatch,
        media,
        qualification={
            "evidence": evidence,
            "camera": camera,
            "color_offset": color_offset,
            "source_archive_sha256": source_sha,
        },
    )
    V.final(args)
    (args.evidence_root / "freeze-review.json").write_bytes(
        args.review_path.read_bytes()
    )
    args.evidence_root.rename(evidence / "vlm")
    visual_root = evidence / "vlm"
    calibration = json.loads((visual_root / "calibration.json").read_text())
    final = json.loads((visual_root / "final.json").read_text())
    fields = {
        "control_manifest_sha256": "freeze.json",
        "label_commitment_sha256": "labels.json",
        "freeze_acceptance_sha256": "freeze-acceptance.json",
        "calibration_result_sha256": "calibration.json",
        "final_frame_manifest_sha256": "final-frame-manifest.json",
        "final_result_sha256": "final.json",
        "raw_transport_manifest_sha256": "transport-manifest.json",
        "freeze_review_receipt_sha256": "freeze-review.json",
    }
    visual = {key: V._sha_file(visual_root / name) for key, name in fields.items()}
    visual.update(
        external_attempt_prefix_sha256=V._sha_bytes(
            args.external_attempt_prefix.encode()
        ),
        calibration_total=calibration["total"],
        **{
            k: calibration[k]
            for k in (
                "true_positives",
                "true_negatives",
                "false_positives",
                "false_negatives",
            )
        },
        final_score=final["score"],
        final_passed=True,
        attempt_count=5,
        external_attempt_markers=5,
    )
    manifest["rtx_proof"]["visual_review"] = visual
    _refresh_objective(manifest, evidence)
    for path in evidence.rglob("*"):
        path.chmod(0o700 if path.is_dir() else 0o600)
    assert len(calls) == len(storage.objects) == 5
    return manifest, evidence


@pytest.mark.parametrize(
    "dimension,match",
    [("source", "source"), ("camera", "camera"), ("pixels", "inventory")],
)
def test_valid_foreign_visual_schedule_cannot_certify_another_run(
    tmp_path, monkeypatch, dimension, match
):
    manifest_a, evidence_a = _visual_fixture(
        monkeypatch,
        tmp_path / "a",
        camera="camera-1",
        color_offset=0,
        source_sha="a" * 64,
    )
    manifest_b, evidence_b = _visual_fixture(
        monkeypatch,
        tmp_path / "b",
        camera="camera-2" if dimension == "camera" else "camera-1",
        color_offset=10 if dimension == "pixels" else 0,
        source_sha="b" * 64 if dimension == "source" else "a" * 64,
    )
    with W.authorized_roots(tmp_path, ROOT):
        for manifest, evidence in ((manifest_a, evidence_a), (manifest_b, evidence_b)):
            acceptance.images.validate_ncore_accepted_image_manifest(manifest)
            assert acceptance._qualification(manifest, evidence, evidence.parent)
        assert acceptance._visual(manifest_a, evidence_a)
        assert acceptance._visual(manifest_b, evidence_b)
        (evidence_b / "vlm").rename(evidence_b / "own-vlm")
        shutil.copytree(evidence_a / "vlm", evidence_b / "vlm")
        manifest_b["rtx_proof"]["visual_review"] = copy.deepcopy(
            manifest_a["rtx_proof"]["visual_review"]
        )
        with pytest.raises(ValueError, match="acceptance_visual_" + match + "_binding"):
            acceptance._visual(manifest_b, evidence_b)
