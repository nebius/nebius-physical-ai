import hashlib
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import acceptance  # noqa: E402


SHA = "a" * 40


def _change_usd_contract(manifest, evidence, changed):
    if changed == "manifest":
        manifest["rtx_proof"]["usd_runtime_version"] = "25.11"
    elif changed == "objective":
        path = evidence / "qualification-audit.json"
        objective = json.loads(path.read_text())
        objective["usdz"]["usd_runtime_version"] = "25.11"
        _write(path, objective)
        manifest["rtx_proof"]["report_sha256"] = acceptance.file_sha(path)
        manifest["rtx_proof"]["rrd"]["report_sha256"] = acceptance.file_sha(path)


@pytest.mark.parametrize("changed", [None, "manifest", "objective"])
def test_statement_binds_real_usd_runtime_across_receipt_and_manifest(
    tmp_path, monkeypatch, changed
):
    from ncore_acceptance_fixture import synthetic_statement_inputs

    manifest, evidence = synthetic_statement_inputs(tmp_path, acceptance.file_sha)
    assert manifest["rtx_proof"]["usd_runtime_version"] == "26.8"
    _change_usd_contract(manifest, evidence, changed)
    proposed = _write(tmp_path / "proposed.json", manifest)
    gates = tmp_path / "gates"
    _write(gates / "prepublication.json", {"synthetic_fixture": True})
    # This cross-layer CPU test exercises the real USD decoder, complete manifest
    # validator, qualification receipt binder, inventory and statement builder.
    # Scanner and hosted-VLM boundaries have separate tests; no live proof is claimed.
    monkeypatch.setattr(acceptance, "_prepublication", lambda *_: None)
    monkeypatch.setattr(acceptance, "_visual", lambda *_: {"synthetic_fixture": True})
    with W.authorized_roots(tmp_path, ROOT):
        if changed:
            match = "usd_runtime_version" if changed == "manifest" else "native_receipt"
            with pytest.raises((RuntimeError, ValueError), match=match):
                acceptance.build_statement(
                    analysis_root=tmp_path,
                    gate_dir=gates,
                    evidence_root=evidence,
                    proposed_manifest_path=proposed,
                    output_path=tmp_path / acceptance.STATEMENT_PATH,
                )
        else:
            statement = acceptance.build_statement(
                analysis_root=tmp_path,
                gate_dir=gates,
                evidence_root=evidence,
                proposed_manifest_path=proposed,
                output_path=tmp_path / acceptance.STATEMENT_PATH,
            )
            assert statement["status"] == "pending_independent_review"
            assert statement["manifest"]["rtx_proof"]["usd_runtime_version"] == "26.8"


def _write(path: Path, value) -> Path:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.write_text(json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")
    path.chmod(0o600)
    return path


def _fixture(root: Path):
    root.chmod(0o700)
    evidence = root / "qualification-audit.json"
    evidence.write_text('{"status":"pass"}\n')
    evidence.chmod(0o600)
    inventory = [
        {
            "path": evidence.relative_to(root).as_posix(),
            "bytes": evidence.stat().st_size,
            "sha256": hashlib.sha256(evidence.read_bytes()).hexdigest(),
        }
    ]
    manifest = {"development_sha": SHA, "status": "accepted"}
    statement = {
        "format": acceptance.STATEMENT_FORMAT,
        "status": "pending_independent_review",
        "candidate_commit": SHA,
        "manifest": manifest,
        "manifest_sha256": acceptance._sha_value(manifest),
        "evidence_inventory": inventory,
        "evidence_inventory_sha256": acceptance._sha_value(inventory),
        "objective": {},
        "visual": {},
    }
    statement_path = _write(root / acceptance.STATEMENT_PATH, statement)
    review = {
        "format": acceptance.REVIEW_FORMAT,
        "verdict": "ACCEPTED",
        "candidate_commit": SHA,
        "statement_sha256": hashlib.sha256(statement_path.read_bytes()).hexdigest(),
        "evidence_inventory_sha256": statement["evidence_inventory_sha256"],
        "manifest_sha256": statement["manifest_sha256"],
        "objective_evidence_reviewed": True,
        "visual_evidence_reviewed": True,
        "cleanup_reviewed": True,
        "reviewer_id": "independent-review-lane",
    }
    review_path = _write(root / acceptance.REVIEW_PATH, review)
    return manifest, statement_path, review_path, evidence


def test_finalize_and_verify_bind_independent_review_and_evidence(
    tmp_path, monkeypatch
):
    manifest, statement_path, review_path, _ = _fixture(tmp_path)
    monkeypatch.setattr(
        acceptance.images, "validate_ncore_accepted_image_manifest", lambda value: value
    )
    output = tmp_path / acceptance.FINAL_PATH
    with W.authorized_roots(tmp_path, ROOT):
        finalized = acceptance.finalize_acceptance(
            analysis_root=tmp_path,
            statement_path=statement_path,
            review_path=review_path,
            output_path=output,
        )
    assert finalized["development_sha"] == SHA
    with W.authorized_roots(tmp_path, ROOT):
        assert (
            acceptance.verify_final_acceptance(tmp_path, output)["development_sha"]
            == SHA
        )
    assert (
        finalized["acceptance_verification"]["review_receipt_sha256"]
        == hashlib.sha256(review_path.read_bytes()).hexdigest()
    )
    assert manifest == {
        key: value
        for key, value in finalized.items()
        if key != "acceptance_verification"
    }


def test_final_acceptance_rejects_changed_evidence(tmp_path, monkeypatch):
    _, statement_path, review_path, evidence = _fixture(tmp_path)
    monkeypatch.setattr(
        acceptance.images, "validate_ncore_accepted_image_manifest", lambda value: value
    )
    output = tmp_path / acceptance.FINAL_PATH
    with W.authorized_roots(tmp_path, ROOT):
        acceptance.finalize_acceptance(
            analysis_root=tmp_path,
            statement_path=statement_path,
            review_path=review_path,
            output_path=output,
        )
    evidence.write_text('{"status":"fabricated"}\n')
    with W.authorized_roots(tmp_path, ROOT):
        with pytest.raises(ValueError, match="accepted_evidence_inventory_changed"):
            acceptance.verify_final_acceptance(tmp_path, output)


def test_finalize_rejects_review_for_another_statement(tmp_path, monkeypatch):
    _, statement_path, review_path, _ = _fixture(tmp_path)
    review = json.loads(review_path.read_text())
    review["statement_sha256"] = "0" * 64
    _write(review_path, review)
    monkeypatch.setattr(
        acceptance.images, "validate_ncore_accepted_image_manifest", lambda value: value
    )
    with W.authorized_roots(tmp_path, ROOT):
        with pytest.raises(ValueError, match="acceptance_independent_review"):
            acceptance.finalize_acceptance(
                analysis_root=tmp_path,
                statement_path=statement_path,
                review_path=review_path,
                output_path=tmp_path / acceptance.FINAL_PATH,
            )
