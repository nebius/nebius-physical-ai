"""Bind a fresh publication scan to the exact independently reviewed population."""

import copy
from pathlib import Path

from image_byte_scan import core as W
from . import acceptance, attribution, byte_acceptance, publication_security
from .process import file_sha, write_json


def _reviewed_gate(root, accepted):
    statement = acceptance._json(root / acceptance.STATEMENT_PATH)
    digest = accepted["prepublication"]["evidence_manifest_sha256"]
    matches = [
        root / item["path"]
        for item in statement["evidence_inventory"]
        if Path(item["path"]).name == "evidence-manifest.json"
        and item["sha256"] == digest
    ]
    W.require(len(matches) == 1, "publication_reviewed_gate_required")
    return matches[0].parent


def _manifest(path, accepted):
    manifest = acceptance._json(path)
    expected = {
        "schema": "npa.ncore.prepublication-evidence.v1",
        "source_sha": accepted["development_sha"],
        "image_digest": accepted["oci_digest"],
        "platform_digest": accepted["amd64_manifest"],
        "config_digest": accepted["config_digest"],
        "archive_sha256": accepted["prepublication"]["archive_sha256"],
    }
    W.require(
        all(manifest.get(key) == value for key, value in expected.items()),
        "publication_gate_identity_changed",
    )
    rows = manifest.get("files")
    W.require(isinstance(rows, list) and rows, "publication_gate_inventory_empty")
    seen = set()
    for row in rows:
        target = acceptance._relative_file(
            path.parent, path.parent / row["path"], "publication gate evidence"
        )
        W.require(
            row["path"] not in seen
            and file_sha(target) == row["sha256"]
            and target.stat().st_size == row["bytes"],
            "publication_gate_inventory_changed",
        )
        seen.add(row["path"])
    return manifest


def _report_population(report):
    population = copy.deepcopy(report)
    # Only invocation bindings differ: both complete authorizations are verified
    # separately, while every input role/hash, graph byte and finding stays bound.
    population.pop("authorization_sha256")
    population["input_snapshot_receipts"] = [
        {key: value for key, value in row.items() if key != "stat"}
        for row in population["input_snapshot_receipts"]
    ]
    return population


def _fresh_manifest(accepted, directory, report, bindings):
    manifest = copy.deepcopy(accepted)
    prepublication = manifest["prepublication"]
    for field, relative in byte_acceptance.ATTRIBUTION_FILES.items():
        if accepted["byte_scan"]["resolution"] == "public-attribution":
            prepublication[field] = file_sha(directory / relative)
        else:
            prepublication.pop(field, None)
    resolution, _exit = byte_acceptance._resolution(report, directory, prepublication)
    policy = byte_acceptance._policy_identity(report)
    W.require(policy[0] == "regex-v1", "publication_requires_ci_regex_policy")
    manifest["byte_scan"] = byte_acceptance._result(
        report, bindings, resolution, policy
    )
    return manifest


def _disposition(accepted, original, fresh):
    if accepted["byte_scan"]["resolution"] == "raw-clean":
        return
    receipts = []
    for directory in (original, fresh):
        receipt = acceptance._json(directory / "attribution.json")
        # These hashes necessarily bind different authorizations; the raw ledger,
        # exact occurrences, source notice/provenance and disposition remain equal.
        receipt.pop("authorization_sha256")
        receipt.pop("report_sha256")
        receipts.append(receipt)
    W.require(receipts[0] == receipts[1], "publication_reviewed_disposition_changed")


def _scan_population(accepted, root, original, fresh):
    old_auth, old_report, _rows, old_bindings = attribution._scan_inputs(original)
    auth, report, _rows, bindings = attribution._scan_inputs(fresh)
    W.require(
        old_bindings["report_sha256"]
        == accepted["prepublication"]["raw_byte_report_sha256"]
        and old_bindings["records_sha256"]
        == accepted["prepublication"]["raw_byte_ledger_sha256"]
        == bindings["records_sha256"]
        and _report_population(old_report) == _report_population(report)
        and attribution._policy(old_auth, old_report)
        == attribution._policy(auth, report),
        "publication_reviewed_byte_population_changed",
    )
    manifest = _fresh_manifest(accepted, fresh, report, bindings)
    byte_acceptance.verify(manifest, root, fresh)
    W.require(
        manifest["byte_scan"]["resolution"] == accepted["byte_scan"]["resolution"]
        and manifest["byte_scan"]["policy_sha256"]
        == accepted["byte_scan"]["policy_sha256"],
        "publication_reviewed_policy_or_resolution_changed",
    )
    _disposition(accepted, original, fresh)
    return bindings


def _receipt(accepted, acceptance_path, before, bindings):
    return {
        "schema": "npa.ncore.reviewed-publication-binding.v1",
        "acceptance_sha256": file_sha(acceptance_path),
        "review": accepted["acceptance_verification"],
        "reviewed_evidence_manifest_sha256": accepted["prepublication"][
            "evidence_manifest_sha256"
        ],
        "fresh_evidence_manifest_sha256": before,
        "reviewed_raw_report_sha256": accepted["prepublication"][
            "raw_byte_report_sha256"
        ],
        "fresh_raw_report_sha256": bindings["report_sha256"],
        "identical_raw_ledger_sha256": bindings["records_sha256"],
        "policy_sha256": accepted["byte_scan"]["policy_sha256"],
        "resolution": accepted["byte_scan"]["resolution"],
    }


def verify(accepted, acceptance_path, evidence_manifest_path):
    """Retain distinct fresh evidence without extending the reviewed disposition.

    Args:
        accepted: Fully verified original acceptance and protected evidence inventory.
        acceptance_path: Original private accepted-manifest path.
        evidence_manifest_path: Fresh successful gate manifest, before registry writes.
    Returns:
        A receipt binding both manifests and the exact reviewed byte population.
    Raises:
        ValueError, OSError: Candidate, population, policy, disposition or files changed.
    """
    root = acceptance_path.parents[1]
    original = _reviewed_gate(root, accepted)
    fresh = evidence_manifest_path.parent
    W.require(original != fresh, "publication_requires_fresh_gate_directory")
    before = file_sha(evidence_manifest_path)
    _manifest(evidence_manifest_path, accepted)
    bindings = _scan_population(accepted, root, original, fresh)
    receipt = _receipt(accepted, acceptance_path, before, bindings)
    receipt["security_population_sha256"] = publication_security.verify(original, fresh)
    _manifest(evidence_manifest_path, accepted)
    W.require(
        file_sha(evidence_manifest_path) == before, "publication_manifest_changed"
    )
    acceptance.verify_final_acceptance(root, acceptance_path)
    write_json(fresh / "accepted-publication-binding.json", receipt)
    return receipt
