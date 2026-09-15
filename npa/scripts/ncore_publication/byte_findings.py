"""Emit privacy-safe diagnostics derived from NCore raw byte-scan evidence."""

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import re
import sys

from image_byte_scan import core as W


__all__ = ["emit_attribution_receipt", "emit_raw"]

_MAXIMUM = (1 << 63) - 1
_SHA256 = re.compile(r"[0-9a-f]{64}")
_RECORD_KINDS = (
    "layer_regular_content", "logical_tar_link", "logical_tar_path",
    "nonzero_tar_padding", "outer_regular_content", "raw_gzip_header",
    "raw_tar_extension", "raw_tar_header", "unexplained_tar_trailer",
    "verified_zero_content",
)
_ROW_KINDS = ("encoded_layer_blob", "verified_zero_range")
_RULE_IDENTIFIERS = ("customer-denylist", "infra-denylist", "private_literal")
_FINDING_CLASSES = ("native-credential", "structural", "unclassified")
_FINDING_GROUPS = (*_RULE_IDENTIFIERS, *_FINDING_CLASSES)
_STRUCTURAL_RULES = frozenset({"nonzero_tar_padding", "nonzero_tar_trailer", "pkcs12-file"})
_CONTEXT = frozenset({"scope", "layer_ordinal", "entry_ordinal", "tar_offset", "compressed_offset"})
_REPORT_FIELDS = frozenset({
    "schema_version", "valid", "complete", "authorization_sha256", "archive_sha256",
    "image_config_digest", "image_manifest_digest", "expected_image_id",
    "private_literals_configured", "private_literal_count", "literal_matching_policy",
    "layers", "oci_graph", "outer", "confidentiality_policy", "helper_summary",
    "literal_engine", "input_snapshot_receipts", "helper_joined", "records",
    "scanned_bytes", "verified_zero_bytes", "regular_files", "regular_bytes", "findings",
})


class _DiagnosticError(ValueError):
    """Identify evidence that cannot safely support detailed diagnostics."""


def _integer(value, *, positive=False):
    valid = type(value) is int and 0 <= value <= _MAXIMUM
    if positive:
        valid = valid and value > 0
    if not valid:
        raise _DiagnosticError("invalid_bounded_integer")
    return value


def _digest(value):
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise _DiagnosticError("invalid_digest")
    return value


def _json(raw):
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise _DiagnosticError("duplicate_json_key")
            result[key] = value
        return result

    try:
        return json.loads(raw, object_pairs_hook=unique)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise _DiagnosticError("invalid_json") from error


def _context(row):
    names = set(row) & _CONTEXT
    values = {name: row[name] for name in names}
    if values.get("scope") not in {"outer", "layer"}:
        raise _DiagnosticError("invalid_context")
    for name, value in values.items():
        if name != "scope":
            _integer(value)
    return names


def _finding_class(finding, size):
    if type(finding) is not dict:
        raise _DiagnosticError("invalid_finding")
    native = {"rule_id", "start_line", "end_line"}
    regex = {"rule_id", "start_byte", "end_byte", "start_line", "end_line", "views"}
    if set(finding) == native:
        start, end = _integer(finding["start_line"]), _integer(finding["end_line"])
        if type(finding["rule_id"]) is not str or re.fullmatch(r"[a-z0-9_-]+", finding["rule_id"]) is None:
            raise _DiagnosticError("invalid_native_rule")
        if size == 0 or start > end or end > size:
            raise _DiagnosticError("invalid_native_range")
        return "native-credential"
    if set(finding) != regex:
        raise _DiagnosticError("invalid_finding_schema")
    for name in ("start_byte", "end_byte", "start_line", "end_line"):
        _integer(finding[name])
    if not (finding["start_byte"] <= finding["end_byte"] <= size):
        raise _DiagnosticError("invalid_finding_range")
    if not (1 <= finding["start_line"] <= finding["end_line"] <= size + 1):
        raise _DiagnosticError("invalid_finding_lines")
    if finding["views"] not in (["line"], ["record"], ["line", "record"], ["record", "line"]):
        raise _DiagnosticError("invalid_finding_views")
    if type(finding["rule_id"]) is not str or re.fullmatch(r"[a-z0-9_-]+", finding["rule_id"]) is None:
        raise _DiagnosticError("invalid_regex_rule")
    return finding["rule_id"] if finding["rule_id"] in _RULE_IDENTIFIERS[:2] else "unclassified"


def _record(row, expected_ordinal, counts, fingerprints, confidentiality, literals):
    base = {"type", "record_ordinal", "kind", "bytes", "sha256", "findings"}
    context = _context(row)
    if set(row) != base | context or row.get("type") != "record":
        raise _DiagnosticError("invalid_record_schema")
    if _integer(row["record_ordinal"], positive=True) != expected_ordinal:
        raise _DiagnosticError("invalid_record_order")
    size, digest = _integer(row["bytes"]), _digest(row["sha256"])
    if type(row["findings"]) is not list:
        raise _DiagnosticError("invalid_record_findings")
    if type(row["kind"]) is not str:
        raise _DiagnosticError("invalid_record_kind")
    kind = row["kind"] if row["kind"] in _RECORD_KINDS else "unclassified"
    counts["kind", kind] += 1
    counts["bytes", kind] += size
    if confidentiality is not None:
        bound = (row["record_ordinal"], digest, size)
        if confidentiality != bound:
            raise _DiagnosticError("confidentiality_record_binding")
    for finding in row["findings"]:
        rule = _finding_class(finding, size)
        counts["finding", rule] += 1
        counts["kind-findings", kind] += 1
    for literal in literals:
        if (literal["record_ordinal"] != expected_ordinal
                or {key: literal[key] for key in _context(literal)} != {key: row[key] for key in context}
                or not 0 <= literal["byte_start"] < literal["byte_end"] <= size):
            raise _DiagnosticError("literal_record_binding")
        counts["kind-findings", kind] += 1
    finding_count = len(row["findings"]) + len(literals)
    if kind == "layer_regular_content" and finding_count:
        opaque = hashlib.sha256(digest.encode("ascii")).hexdigest()
        fingerprints.append((opaque, size, _integer(finding_count)))


def _standalone_finding(row, counts):
    context = _context(row)
    rule = row.get("rule_id")
    literal = {"type", "rule_id", "record_ordinal", "literal_index",
               "literal_sha256", "byte_start", "byte_end"} | context
    structural = {"type", "rule_id"} | context
    if rule == "private_literal" and set(row) == literal:
        _integer(row["record_ordinal"], positive=True)
        for name in ("literal_index", "byte_start", "byte_end"):
            _integer(row[name])
        _digest(row["literal_sha256"])
        counts["finding", "private_literal"] += 1
        return row
    if set(row) != structural:
        raise _DiagnosticError("invalid_standalone_finding")
    if type(rule) is not str or re.fullmatch(r"[a-z0-9_-]+", rule) is None:
        raise _DiagnosticError("invalid_standalone_rule")
    category = "structural" if rule in _STRUCTURAL_RULES else "unclassified"
    counts["finding", category] += 1


def _auxiliary(row, counts):
    context = _context(row)
    expected = {"type", "bytes", "sha256"} | context
    if set(row) != expected:
        raise _DiagnosticError("invalid_auxiliary_row")
    size = _integer(row["bytes"], positive=row["type"] == "verified_zero_range")
    _digest(row["sha256"])
    counts["kind", row["type"]] += 1
    counts["bytes", row["type"]] += size


def _confidentiality(row):
    expected = {"type", "record_ordinal", "policy_sha256", "sha256", "bytes",
                "line_count", "composed_findings"}
    if set(row) != expected:
        raise _DiagnosticError("invalid_confidentiality_row")
    _integer(row["record_ordinal"], positive=True)
    for name in ("bytes", "line_count", "composed_findings"):
        _integer(row[name])
    _digest(row["policy_sha256"])
    digest = _digest(row["sha256"])
    return row["record_ordinal"], digest, row["bytes"]


def _rows(raw):
    rows = []
    for line in raw.splitlines():
        if not line:
            raise _DiagnosticError("empty_ledger_line")
        row = _json(line)
        if type(row) is not dict:
            raise _DiagnosticError("invalid_ledger_row")
        rows.append(row)
    if not rows:
        raise _DiagnosticError("empty_ledger")
    return rows


def _ledger(raw):
    counts, fingerprints, ordinal, pending = Counter(), [], 0, None
    literals = []
    for row in _rows(raw):
        row_type = row.get("type")
        if row_type == "record":
            ordinal += 1
            _record(row, ordinal, counts, fingerprints, pending, literals)
            pending = None
            literals = []
        elif row_type == "finding":
            if pending is not None:
                raise _DiagnosticError("invalid_ledger_order")
            literal = _standalone_finding(row, counts)
            if literal is not None:
                literals.append(literal)
        elif row_type == "confidentiality_record":
            if pending is not None:
                raise _DiagnosticError("invalid_ledger_order")
            pending = _confidentiality(row)
        elif row_type in _ROW_KINDS:
            if pending is not None or literals:
                raise _DiagnosticError("invalid_ledger_order")
            _auxiliary(row, counts)
        else:
            raise _DiagnosticError("invalid_row_type")
    if pending is not None or literals:
        raise _DiagnosticError("partial_confidentiality_record")
    return counts, fingerprints, ordinal


def _report(raw, counts, records):
    report = _json(raw)
    if (type(report) is not dict or set(report) != _REPORT_FIELDS
            or report.get("schema_version") != "npa.image-byte-scan.v1"):
        raise _DiagnosticError("invalid_report_schema")
    for name in ("complete", "valid", "helper_joined"):
        if type(report.get(name)) is not bool:
            raise _DiagnosticError("invalid_report_boolean")
    for name in ("records", "scanned_bytes", "verified_zero_bytes", "regular_files",
                 "regular_bytes", "findings"):
        _integer(report.get(name))
    if report["complete"] is not True or report["helper_joined"] is not True or "failure_code" in report:
        raise _DiagnosticError("partial_report")
    total = sum(counts["finding", group] for group in _FINDING_GROUPS)
    if report["records"] != records or report["findings"] != total:
        raise _DiagnosticError("report_ledger_population")
    if report["valid"] is not (total == 0):
        raise _DiagnosticError("report_verdict_population")
    scanned = sum(counts["bytes", kind] for kind in (*_RECORD_KINDS, "unclassified"))
    if report["scanned_bytes"] != scanned:
        raise _DiagnosticError("report_ledger_bytes")
    if report["regular_files"] != counts["kind", "layer_regular_content"]:
        raise _DiagnosticError("report_regular_files")
    if report["regular_bytes"] != counts["bytes", "layer_regular_content"]:
        raise _DiagnosticError("report_regular_bytes")
    if report["verified_zero_bytes"] != counts["bytes", "verified_zero_range"]:
        raise _DiagnosticError("report_zero_bytes")
    helper = report.get("helper_summary")
    if type(helper) is not dict or set(helper) != {"type", "files", "bytes", "findings"}:
        raise _DiagnosticError("invalid_helper_summary")
    for name in ("files", "bytes", "findings"):
        _integer(helper[name])
    if (helper["type"] != "summary" or helper["files"] != records
            or helper["bytes"] != scanned
            or helper["findings"] != counts["finding", "native-credential"]):
        raise _DiagnosticError("helper_ledger_population")


def _read(path):
    fd = None
    try:
        _path, fd, before = W.open_private_fd(path)
        raw = W.descriptor_bytes(fd)
        after = os.fstat(fd)
        W.require(
            W.stat_fingerprint(after) == W.stat_fingerprint(before),
            "ncore_byte_diagnostic_evidence_changed",
        )
        return raw
    except (OSError, ValueError):
        return None
    finally:
        if fd is not None:
            os.close(fd)


def _artifact(name, raw):
    if raw is None:
        print(f"NCore OCI byte-scan-evidence artifact={name} available=false", file=sys.stderr, flush=True)
        return
    digest = hashlib.sha256(raw).hexdigest()
    print(f"NCore OCI byte-scan-evidence artifact={name} available=true sha256={digest}",
          file=sys.stderr, flush=True)


def _counts(counts):
    for rule in _RULE_IDENTIFIERS:
        print(f"NCore OCI byte-scan-detail category=rule rule={rule} "
              f"findings={counts['finding', rule]}", file=sys.stderr, flush=True)
    for finding_class in _FINDING_CLASSES:
        print("NCore OCI byte-scan-detail category=finding-class "
              f"class={finding_class} findings={counts['finding', finding_class]}",
              file=sys.stderr, flush=True)
    for kind in (*_RECORD_KINDS, *_ROW_KINDS, "unclassified"):
        findings = counts["kind-findings", kind]
        print(f"NCore OCI byte-scan-detail category=record-kind kind={kind} "
              f"records={counts['kind', kind]} findings={findings}", file=sys.stderr, flush=True)


def _fingerprints(fingerprints):
    for digest, size, findings in sorted(fingerprints):
        print("NCore OCI byte-scan-detail category=affected-regular-content "
              f"content-fingerprint={digest} bytes={size} findings={findings}",
              file=sys.stderr, flush=True)


def _emit_raw(directory):
    """Emit hashes and bounded classifications for raw NCore scan evidence.

    Args:
        directory: NCore gate directory containing the private bytes directory.
    Returns:
        None.
    Raises:
        None. Missing or invalid detail evidence is reported as unavailable.
    """
    report = _read(Path(directory) / "bytes/report.json")
    ledger = _read(Path(directory) / "bytes/records.jsonl")
    _artifact("raw-report", report)
    _artifact("raw-ledger", ledger)
    try:
        counts, fingerprints, records = _ledger(ledger) if ledger is not None else ({}, [], 0)
        if report is None or ledger is None:
            raise _DiagnosticError("missing_diagnostic_evidence")
        _report(report, counts, records)
    except (_DiagnosticError, MemoryError, OverflowError, TypeError, ValueError):
        print("NCore OCI byte-scan-detail available=false", file=sys.stderr, flush=True)
        return
    print("NCore OCI byte-scan-detail available=true", file=sys.stderr, flush=True)
    _counts(counts)
    _fingerprints(fingerprints)


def emit_raw(directory):
    """Emit optional diagnostics without replacing the publication gate outcome."""
    try:
        _emit_raw(directory)
    except (OSError, ValueError, TypeError, MemoryError, OverflowError, RecursionError):
        print("NCore OCI byte-scan-detail available=false", file=sys.stderr, flush=True)


def emit_attribution_receipt(directory, accepted):
    """Emit a hash of an attribution receipt only after successful acceptance.

    Args:
        directory: NCore gate directory containing the private receipt.
        accepted: Receipt returned by the successful attribution verifier.
    Returns:
        None.
    Raises:
        None. A missing receipt is reported as unavailable.
    """
    try:
        raw = _read(Path(directory) / "attribution.json")
        if raw is None or W.canonical(_json(raw)) != W.canonical(accepted):
            raw = None
        _artifact("attribution-acceptance", raw)
    except (OSError, ValueError, TypeError, MemoryError, OverflowError, RecursionError):
        _artifact("attribution-acceptance", None)
