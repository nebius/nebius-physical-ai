"""Prove NCore raw-finding diagnostics are bounded and privacy-safe."""

import hashlib
import json
import os
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import byte_findings  # noqa: E402


ATTACKER_HEX = "a" * 64
DIGEST = "b" * 64


@pytest.fixture(autouse=True)
def _authorize_private_evidence(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, ROOT):
        yield


def _finding(rule="customer-denylist"):
    return {"rule_id": rule, "start_byte": 1, "end_byte": 2,
            "start_line": 1, "end_line": 1, "views": ["record"]}


def _record(findings=None, kind="layer_regular_content", size=7):
    return {"type": "record", "record_ordinal": 1, "kind": kind, "bytes": size,
            "sha256": DIGEST, "findings": [_finding()] if findings is None else findings,
            "scope": "layer", "layer_ordinal": 0, "entry_ordinal": 1, "tar_offset": 512}


def _confidentiality():
    return {"type": "confidentiality_record", "record_ordinal": 1,
            "policy_sha256": "c" * 64, "sha256": DIGEST, "bytes": 7,
            "line_count": 1, "composed_findings": 1}


def _evidence(tmp_path, *, report_change=None, rows=None):
    directory = tmp_path / "bytes"
    directory.mkdir(mode=0o700)
    rows = [_record()] if rows is None else rows
    ledger = b"".join(json.dumps(row, sort_keys=True).encode() + b"\n" for row in rows)
    records = [row for row in rows if row.get("type") == "record"]
    findings = sum(len(row.get("findings", [])) + (row.get("type") == "finding") for row in rows)
    scanned = sum(row.get("bytes", 0) for row in records if type(row.get("bytes")) is int)
    regular = [row for row in records if row.get("kind") == "layer_regular_content"]
    native = sum(set(finding) == {"rule_id", "start_line", "end_line"}
                 for row in records for finding in row.get("findings", []))
    report = {"schema_version": "npa.image-byte-scan.v1", "complete": True,
              "authorization_sha256": DIGEST, "archive_sha256": DIGEST,
              "image_config_digest": "sha256:" + DIGEST,
              "image_manifest_digest": "sha256:" + DIGEST,
              "expected_image_id": "sha256:" + DIGEST,
              "private_literals_configured": False, "private_literal_count": 0,
              "literal_matching_policy": "exact-substring-v1", "layers": [],
              "oci_graph": {}, "outer": {}, "confidentiality_policy": {},
              "literal_engine": {}, "input_snapshot_receipts": [],
              "valid": findings == 0, "helper_joined": True, "records": len(records),
              "scanned_bytes": scanned, "verified_zero_bytes": 0,
              "regular_files": len(regular),
              "regular_bytes": sum(row.get("bytes", 0) for row in regular
                                     if type(row.get("bytes")) is int),
              "findings": findings,
              "helper_summary": {"type": "summary", "files": len(records),
                                  "bytes": scanned, "findings": native}}
    if report_change:
        report_change(report)
    report_path = directory / "report.json"
    ledger_path = directory / "records.jsonl"
    report_path.write_bytes(json.dumps(report).encode())
    ledger_path.write_bytes(ledger)
    report_path.chmod(0o600)
    ledger_path.chmod(0o600)
    return report, ledger


def test_fixed_counts_and_second_order_content_fingerprint(tmp_path, capsys):
    report, ledger = _evidence(tmp_path, rows=[_confidentiality(), _record()])
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    fingerprint = hashlib.sha256(DIGEST.encode("ascii")).hexdigest()
    assert "byte-scan-detail available=true" in output
    assert "rule=customer-denylist findings=1" in output
    assert "class=native-credential findings=0" in output
    assert "kind=layer_regular_content records=1 findings=1" in output
    assert f"content-fingerprint={fingerprint} bytes=7 findings=1" in output
    assert hashlib.sha256(json.dumps(report).encode()).hexdigest() in output
    assert hashlib.sha256(ledger).hexdigest() in output
    assert DIGEST not in output


def test_all_fixed_finding_classes_are_counted(tmp_path, capsys):
    native = {"rule_id": "generic-api-key", "start_line": 0, "end_line": 1}
    literal = {"type": "finding", "rule_id": "private_literal", "record_ordinal": 1,
               "literal_index": 0, "literal_sha256": "d" * 64,
               "byte_start": 0, "byte_end": 1, "scope": "layer", "layer_ordinal": 0,
               "entry_ordinal": 1, "tar_offset": 512}
    structural = {"type": "finding", "rule_id": "pkcs12-file", "scope": "layer",
                  "layer_ordinal": 0, "entry_ordinal": 1, "tar_offset": 512}
    _evidence(tmp_path, rows=[literal, structural, _record(findings=[_finding(), native])])
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    assert "rule=private_literal findings=1" in output
    assert "class=native-credential findings=1" in output
    assert "class=structural findings=1" in output
    assert "kind=layer_regular_content records=1 findings=3" in output
    assert "bytes=7 findings=3" in output


@pytest.mark.parametrize("change", [
    lambda report: report.update(records=True),
    lambda report: report.update(findings=1 << 80),
    lambda report: report.update(helper_summary={"type": "summary", "files": 1,
                                                  "bytes": 7, "findings": True}),
    lambda report: report.update(schema_version=ATTACKER_HEX),
    lambda report: report.update(unexpected_metadata=ATTACKER_HEX),
])
def test_report_types_and_huge_counts_make_detail_unavailable(tmp_path, capsys, change):
    _evidence(tmp_path, report_change=change)
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    assert "byte-scan-detail available=false" in output
    assert ATTACKER_HEX not in output


@pytest.mark.parametrize("mutation", [
    lambda row: row.update(kind=ATTACKER_HEX),
    lambda row: row["findings"][0].update(rule_id=ATTACKER_HEX),
])
def test_unknown_classes_use_only_unclassified_indicator(tmp_path, capsys, mutation):
    row = _record()
    mutation(row)
    _evidence(tmp_path, rows=[row])
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    assert "byte-scan-detail available=true" in output
    assert "unclassified" in output
    assert ATTACKER_HEX not in output


@pytest.mark.parametrize("rows", [
    [{"type": ATTACKER_HEX}],
    [{**_record(), "metadata": ATTACKER_HEX}],
    [{**_record(), "bytes": True}],
    [{**_record(), "bytes": 1 << 80}],
    [{"type": "confidentiality_record", "record_ordinal": 1,
      "policy_sha256": DIGEST, "sha256": DIGEST, "bytes": 7,
      "line_count": 1, "composed_findings": 1}],
])
def test_malformed_partial_and_fake_metadata_are_not_interpreted(tmp_path, capsys, rows):
    _evidence(tmp_path, rows=rows)
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    assert "byte-scan-detail available=false" in output
    assert ATTACKER_HEX not in output


def test_malformed_and_unreadable_artifacts_keep_only_safe_hashes(tmp_path, monkeypatch, capsys):
    _evidence(tmp_path)
    (tmp_path / "bytes/records.jsonl").write_bytes(b"{partial")
    original = byte_findings._read
    monkeypatch.setattr(byte_findings, "_read", lambda path: None if path.name == "report.json" else original(path))
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    assert "artifact=raw-report available=false" in output
    assert "artifact=raw-ledger available=true sha256=" in output
    assert "byte-scan-detail available=false" in output


@pytest.mark.parametrize("raw", [b"{partial", b'{"records":1,"records":2}'])
def test_malformed_report_is_hashed_but_not_interpreted(tmp_path, capsys, raw):
    _evidence(tmp_path)
    (tmp_path / "bytes/report.json").write_bytes(raw)
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    assert hashlib.sha256(raw).hexdigest() in output
    assert "byte-scan-detail available=false" in output


def test_attribution_receipt_hashes_bytes_without_echoing_fields(tmp_path, capsys):
    raw = json.dumps({"metadata": ATTACKER_HEX}).encode()
    receipt = tmp_path / "attribution.json"
    receipt.write_bytes(raw)
    receipt.chmod(0o600)
    byte_findings.emit_attribution_receipt(tmp_path, json.loads(raw))
    output = capsys.readouterr().err
    assert hashlib.sha256(raw).hexdigest() in output
    assert ATTACKER_HEX not in output


def test_replaced_attribution_receipt_is_not_reported_as_accepted(tmp_path, capsys):
    raw = json.dumps({"replacement": ATTACKER_HEX}).encode()
    path = tmp_path / "attribution.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    byte_findings.emit_attribution_receipt(tmp_path, {"accepted": True})
    output = capsys.readouterr().err
    assert "artifact=attribution-acceptance available=false" in output
    assert hashlib.sha256(raw).hexdigest() not in output
    assert ATTACKER_HEX not in output


@pytest.mark.parametrize("accepted,replacement", [(True, 1), (False, 0), (1, 1.0)])
def test_receipt_binding_rejects_equal_values_with_different_json_types(
    tmp_path, capsys, accepted, replacement
):
    raw = json.dumps({"nested": [{"value": replacement}]}).encode()
    path = tmp_path / "attribution.json"
    path.write_bytes(raw)
    path.chmod(0o600)
    byte_findings.emit_attribution_receipt(tmp_path, {"nested": [{"value": accepted}]})
    output = capsys.readouterr().err
    assert "artifact=attribution-acceptance available=false" in output
    assert hashlib.sha256(raw).hexdigest() not in output


@pytest.mark.parametrize("operation", ["raw", "attribution"])
def test_diagnostic_allocation_failure_does_not_escape(tmp_path, monkeypatch, capsys, operation):
    _evidence(tmp_path)
    path = tmp_path / "attribution.json"
    path.write_text("{}")
    path.chmod(0o600)

    def allocation_failure(_fd):
        raise MemoryError(ATTACKER_HEX)

    monkeypatch.setattr(W, "descriptor_bytes", allocation_failure)
    if operation == "raw":
        byte_findings.emit_raw(tmp_path)
    else:
        byte_findings.emit_attribution_receipt(tmp_path, {})
    output = capsys.readouterr().err
    assert "available=false" in output
    assert ATTACKER_HEX not in output


@pytest.mark.parametrize("damage", [None, "ordinal", "context", "range", "orphan"])
def test_literal_findings_bind_to_the_exact_following_record(tmp_path, capsys, damage):
    literal = {"type": "finding", "rule_id": "private_literal", "record_ordinal": 1,
               "literal_index": 0, "literal_sha256": "d" * 64,
               "byte_start": 0, "byte_end": 1, "scope": "layer", "layer_ordinal": 0,
               "entry_ordinal": 1, "tar_offset": 512}
    if damage == "ordinal":
        literal["record_ordinal"] = 2
    elif damage == "context":
        literal["entry_ordinal"] = 2
    elif damage == "range":
        literal["byte_end"] = 8
    rows = [literal] if damage == "orphan" else [literal, _record(findings=[])]
    _evidence(tmp_path, rows=rows)
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    if damage is None:
        assert "kind=layer_regular_content records=1 findings=1" in output
        assert "category=affected-regular-content" in output
    else:
        assert "byte-scan-detail available=false" in output
        assert "category=affected-regular-content" not in output


def test_path_records_never_receive_content_fingerprints(tmp_path, capsys):
    _evidence(tmp_path, rows=[_record(kind="logical_tar_path")])
    byte_findings.emit_raw(tmp_path)
    output = capsys.readouterr().err
    assert "byte-scan-detail available=true" in output
    assert "category=affected-regular-content" not in output
    assert DIGEST not in output


def test_symlinked_private_evidence_is_unavailable_without_target_disclosure(
    tmp_path, capsys
):
    _report, ledger = _evidence(tmp_path)
    report = tmp_path / "bytes/report.json"
    target = tmp_path / "private-report-target.json"
    target.write_bytes(b"private target material")
    target.chmod(0o600)
    report.unlink()
    report.symlink_to(target)

    byte_findings.emit_raw(tmp_path)

    output = capsys.readouterr().err
    assert "artifact=raw-report available=false" in output
    assert hashlib.sha256(ledger).hexdigest() in output
    assert "byte-scan-detail available=false" in output
    assert str(target) not in output
    assert "private target material" not in output


def test_descriptor_mutation_makes_evidence_unavailable(
    tmp_path, monkeypatch, capsys
):
    _evidence(tmp_path)
    report = tmp_path / "bytes/report.json"
    report_inode = report.stat().st_ino
    descriptor_bytes = W.descriptor_bytes

    def mutate_after_read(fd):
        raw = descriptor_bytes(fd)
        if os.fstat(fd).st_ino == report_inode:
            os.fchmod(fd, 0o400)
        return raw

    monkeypatch.setattr(W, "descriptor_bytes", mutate_after_read)
    byte_findings.emit_raw(tmp_path)

    output = capsys.readouterr().err
    assert "artifact=raw-report available=false" in output
    assert "artifact=raw-ledger available=true sha256=" in output
    assert "byte-scan-detail available=false" in output
    assert "ncore_byte_diagnostic_evidence_changed" not in output
