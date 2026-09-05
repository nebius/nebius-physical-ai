import copy
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/tests/docker"))
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import public_native_policy as P  # noqa: E402
from test_image_byte_adjudication import W, record  # noqa: E402


def fixture():
    native = {"rule_id": "generic-api-key", "start_line": 0, "end_line": 0}
    rows = [
        record(1, b"synthetic noncredential fixture", [native, copy.deepcopy(native)]),
        record(2, b"synthetic noncredential fixture", [native, copy.deepcopy(native)]),
    ]
    size = sum(r["bytes"] for r in rows)
    report = {
        "schema_version": "npa.image-byte-scan.v1",
        "complete": True,
        "helper_joined": True,
        "valid": False,
        "records": 2,
        "scanned_bytes": size,
        "regular_files": 2,
        "regular_bytes": size,
        "findings": 4,
        "verified_zero_bytes": 0,
        "helper_summary": {"type": "summary", "files": 2, "bytes": size, "findings": 4},
    }
    binding = {"path": "synthetic-public-policy-evidence", "sha256": "a" * 64}
    catalog = {
        "schema_version": P.SCHEMA,
        "detector_identity": {"config_sha256": "c" * 64},
        "entries": [
            {
                "record_kind": "layer_regular_content",
                "record_sha256": rows[0]["sha256"],
                "record_bytes": rows[0]["bytes"],
                "native_findings": [copy.deepcopy(native), copy.deepcopy(native)],
                "semantic_role": "cryptographic-self-test",
                "operational_credential": False,
                "public_provenance": binding,
                "semantic_proof": copy.deepcopy(binding),
            }
        ],
    }
    return catalog, report, rows


def compile_catalog(catalog):
    return P.compile_catalog(catalog, {"config_sha256": "c" * 64}, lambda binding: None)


def test_duplicate_coordinates_and_identical_ancestor_content_conserved():
    catalog, report, rows = fixture()
    original = copy.deepcopy(report)
    result = P.match_fresh_population(compile_catalog(catalog), report, rows)
    assert (
        result["accepted_native_occurrences"] == 4 and result["raw_scan_valid"] is False
    )
    assert report == original


@pytest.mark.parametrize(
    "mutation",
    [
        "changed_record",
        "changed_size",
        "missing_native",
        "new_native",
        "changed_native",
        "unsafe_role",
        "active_credential",
        "duplicate_entry",
        "changed_detector",
        "bool_size",
        "missing_proof",
        "bad_proof_hash",
    ],
)
def test_unreviewed_catalog_changes_refuse(mutation):
    catalog, report, rows = fixture()
    entry = catalog["entries"][0]
    if mutation == "changed_record":
        entry["record_sha256"] = "0" * 64
    elif mutation == "changed_size":
        entry["record_bytes"] += 1
    elif mutation == "missing_native":
        entry["native_findings"].pop()
    elif mutation == "new_native":
        entry["native_findings"].append(copy.deepcopy(entry["native_findings"][0]))
    elif mutation == "changed_native":
        entry["native_findings"][0]["rule_id"] = "different-rule"
    elif mutation == "unsafe_role":
        entry["semantic_role"] = "runtime-authentication"
    elif mutation == "active_credential":
        entry["operational_credential"] = True
    elif mutation == "duplicate_entry":
        catalog["entries"].append(copy.deepcopy(entry))
    elif mutation == "changed_detector":
        catalog["detector_identity"]["config_sha256"] = "d" * 64
    elif mutation == "bool_size":
        entry["record_bytes"] = True
    elif mutation == "missing_proof":
        del entry["semantic_proof"]
    else:
        entry["semantic_proof"]["sha256"] = "untrusted"
    with pytest.raises(W.ScanError):
        P.match_fresh_population(compile_catalog(catalog), report, rows)


@pytest.mark.parametrize("family", ["regex", "literal", "structural"])
def test_confidentiality_and_structural_findings_cannot_be_accepted(family):
    catalog, report, rows = fixture()
    if family == "regex":
        rows[0]["findings"].append(
            {
                "rule_id": "infra-denylist",
                "start_byte": 0,
                "end_byte": 4,
                "start_line": 0,
                "end_line": 0,
                "views": ["record"],
            }
        )
    elif family == "literal":
        rows.append(
            {
                "type": "finding",
                "rule_id": "private_literal",
                "record_ordinal": 1,
                "literal_index": 0,
                "literal_sha256": "a" * 64,
                "byte_start": 0,
                "byte_end": 4,
                "scope": "layer",
                "entry_ordinal": 1,
            }
        )
    else:
        rows.append({"type": "finding", "rule_id": "pkcs12_payload", "scope": "layer"})
    report["findings"] += 1
    with pytest.raises(W.ScanError):
        P.match_fresh_population(compile_catalog(catalog), report, rows)


@pytest.mark.parametrize(
    "mutation",
    ["incomplete", "helper_pending", "failure_code", "count_bool", "new_record_hash"],
)
def test_incomplete_or_changed_raw_population_refuses(mutation):
    catalog, report, rows = fixture()
    if mutation == "incomplete":
        report["complete"] = False
    elif mutation == "helper_pending":
        report["helper_joined"] = False
    elif mutation == "failure_code":
        report["failure_code"] = "input_changed"
    elif mutation == "count_bool":
        report["findings"] = True
    else:
        rows[1]["sha256"] = "f" * 64
    with pytest.raises(W.ScanError):
        P.match_fresh_population(compile_catalog(catalog), report, rows)


def test_complete_zero_finding_scan_keeps_zero_verdict():
    catalog, report, rows = fixture()
    for row in rows:
        row["findings"] = []
    report["valid"] = True
    report["findings"] = 0
    report["helper_summary"]["findings"] = 0
    result = P.match_fresh_population(compile_catalog(catalog), report, rows)
    assert (
        result["raw_scan_valid"] is True and result["accepted_native_occurrences"] == 0
    )


def test_same_bytes_in_archive_metadata_are_not_library_proof():
    catalog, report, rows = fixture()
    for row in rows:
        row["kind"] = "raw_tar_header"
    report["regular_files"] = report["regular_bytes"] = 0
    with pytest.raises(W.ScanError, match="unsupported_record_kind"):
        P.match_fresh_population(compile_catalog(catalog), report, rows)


def test_exact_typed_metadata_bytes_can_be_reviewed_without_path_suppression():
    catalog, report, rows = fixture()
    for row in rows:
        row["kind"] = "logical_tar_path"
    report["regular_files"] = report["regular_bytes"] = 0
    catalog["entries"][0]["record_kind"] = "logical_tar_path"
    catalog["entries"][0]["semantic_role"] = "public-package-path-metadata"
    result = P.match_fresh_population(compile_catalog(catalog), report, rows)
    assert result["accepted_native_occurrences"] == 4


def test_exact_bytes_cannot_cross_regular_file_and_metadata_contexts():
    catalog, report, rows = fixture()
    for row in rows:
        row["kind"] = "logical_tar_path"
    report["regular_files"] = report["regular_bytes"] = 0
    with pytest.raises(W.ScanError, match="unreviewed_content_or_population"):
        P.match_fresh_population(compile_catalog(catalog), report, rows)


def test_metadata_requires_its_exact_review_role():
    catalog, _report, _rows = fixture()
    catalog["entries"][0]["record_kind"] = "logical_tar_path"
    with pytest.raises(W.ScanError, match="metadata_role"):
        compile_catalog(catalog)
