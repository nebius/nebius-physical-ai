#!/usr/bin/env python3
"""Verify an explicitly authorized review of complete image-byte findings.

This preserves the raw scanner's verdict and produces a separate acceptance
receipt. Expected manifest/review hashes are authorization inputs obtained from
an actual independent review; they must not be calculated automatically from
unreviewed files. A hash authenticates bytes relative to that authorization, not
the identity or correctness of a purported reviewer. This module never invents
semantic acceptance from a filename, origin URL, package name or role label.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys
import tarfile
from pathlib import Path

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from image_byte_scan import core as W
else:
    from . import core as W


SCHEMA = "npa.image-byte-adjudication.v1"
MANIFEST_SCHEMA = "npa.image-byte-disposition-manifest.v1"
REVIEW_SCHEMA = "npa.image-byte-independent-review.v1"
PROOF_SCHEMA = "npa.image-byte-occurrence-provenance.v1"
ROLES = frozenset({"cryptographic-self-test", "parser-format-delimiter",
                   "package-integrity-metadata", "non-operational-source-example",
                   "public-source-symbol", "public-license-reference"})
HEX = re.compile(r"[0-9a-f]{64}")


def digest(value):
    W.require(isinstance(value, str) and HEX.fullmatch(value) is not None,
              "adjudication_digest")
    return value


def integer(value):
    W.require(type(value) is int and value >= 0, "adjudication_integer")
    return value


def fields(value, expected, code):
    W.require(isinstance(value, dict) and set(value) == set(expected), code)


def decode(data):
    # Duplicate keys can otherwise erase an earlier denial, occurrence or hash.
    def unique(pairs):
        result = {}
        for key, value in pairs:
            W.require(key not in result, "adjudication_duplicate_json_key")
            result[key] = value
        return result
    return json.loads(data, object_pairs_hook=unique)


def bound_bytes(spec):
    fields(spec, {"path", "sha256"}, "adjudication_file_binding")
    digest(spec["sha256"])
    with W.bound_open(spec, secret=True) as (_path, fd, initial):
        data = bytearray()
        os.lseek(fd, 0, os.SEEK_SET)
        while chunk := os.read(fd, W.CHUNK):
            data.extend(chunk)
        W.require(hashlib.sha256(data).hexdigest() == spec["sha256"],
                  "adjudication_file_changed")
        W.require(W.stat_fingerprint(os.fstat(fd)) == W.stat_fingerprint(initial),
                  "adjudication_file_changed")
    return bytes(data)


def pinned_json(path, expected):
    return decode(bound_bytes({"path": str(path), "sha256": digest(expected)}))


def population(report, rows):
    """Conserve each native/regex/literal occurrence, including duplicates.

    Identical detections receive distinct identities by their ledger position.
    Structural archive findings are never waivable by this first protocol.
    """
    W.require(report.get("schema_version") == "npa.image-byte-scan.v1"
              and report.get("complete") is True
              and report.get("helper_joined") is True
              and "failure_code" not in report, "adjudication_incomplete_scan")
    records, issues = {}, []
    native_count = zero_bytes = 0
    for line, row in enumerate(rows, 1):
        W.require(isinstance(row, dict), "adjudication_ledger_row")
        kind = row.get("type")
        if kind == "record":
            ordinal = integer(row.get("record_ordinal"))
            W.require(ordinal == len(records) + 1, "adjudication_record_order")
            digest(row.get("sha256"))
            integer(row.get("bytes"))
            W.require(isinstance(row.get("findings"), list),
                      "adjudication_record_findings")
            records[ordinal] = row
            for index, finding in enumerate(row["findings"]):
                W.require(isinstance(finding, dict) and isinstance(finding.get("rule_id"), str)
                          and re.fullmatch(r"[a-z0-9_-]+", finding["rule_id"]),
                          "adjudication_finding_schema")
                if set(finding) == {"rule_id", "start_line", "end_line"}:
                    integer(finding["start_line"])
                    integer(finding["end_line"])
                    # The pinned detector uses zero-based lines for a fragment
                    # with StartLine=0. Byte length bounds its LF line count.
                    W.require(row["bytes"] > 0 and finding["start_line"] <= finding["end_line"] <= row["bytes"],
                              "adjudication_native_range")
                    native_count += 1
                else:
                    fields(finding, {"rule_id", "start_byte", "end_byte", "start_line", "end_line", "views"},
                           "adjudication_regex_finding_schema")
                    W.require(finding["rule_id"] in {"customer-denylist", "infra-denylist"},
                              "adjudication_regex_rule")
                    for name in ("start_byte", "end_byte", "start_line", "end_line"):
                        integer(finding[name])
                    W.require(0 <= finding["start_byte"] <= finding["end_byte"] <= row["bytes"],
                              "adjudication_regex_range")
                    W.require(isinstance(finding["views"], list) and bool(finding["views"])
                              and all(view in {"line", "record"} for view in finding["views"])
                              and len(finding["views"]) == len(set(finding["views"])),
                              "adjudication_regex_views")
                issues.append((line, index, ordinal, finding))
        elif kind == "finding":
            # Path/structural failures without a content record need separate
            # controls. Refuse them instead of converting a format error to safe.
            W.require(type(row.get("record_ordinal")) is int,
                      "adjudication_structural_finding")
            literal_keys = {"type", "rule_id", "record_ordinal", "literal_index",
                            "literal_sha256", "byte_start", "byte_end"}
            W.require(literal_keys <= set(row) <= literal_keys | {
                "scope", "layer_ordinal", "entry_ordinal", "tar_offset", "compressed_offset"}
                and row["rule_id"] == "private_literal", "adjudication_literal_finding_schema")
            for name in ("literal_index", "byte_start", "byte_end"):
                integer(row[name])
            for name in ("layer_ordinal", "entry_ordinal", "tar_offset", "compressed_offset"):
                if name in row:
                    integer(row[name])
            digest(row["literal_sha256"])
            issues.append((line, None, row["record_ordinal"], row))
        elif kind == "verified_zero_range":
            zero_bytes += integer(row.get("bytes"))
            digest(row.get("sha256"))
        else:
            W.require(kind in {"confidentiality_record", "encoded_layer_blob"},
                      "adjudication_unknown_ledger_row")
    W.require(len(records) == integer(report.get("records")),
              "adjudication_record_population")
    W.require(sum(row["bytes"] for row in records.values()) == integer(report.get("scanned_bytes")),
              "adjudication_byte_population")
    regular = [r for r in records.values() if r["kind"] == "layer_regular_content"]
    W.require(len(regular) == integer(report.get("regular_files"))
              and sum(r["bytes"] for r in regular) == integer(report.get("regular_bytes")),
              "adjudication_regular_population")
    W.require(len(issues) == integer(report.get("findings")),
              "adjudication_finding_population")
    W.require(zero_bytes == integer(report.get("verified_zero_bytes")),
              "adjudication_zero_population")
    helper = report.get("helper_summary")
    fields(helper, {"type", "files", "bytes", "findings"}, "adjudication_helper_schema")
    for name in ("files", "bytes", "findings"):
        integer(helper[name])
    W.require(report.get("helper_summary") == {"type": "summary", "files": len(records),
              "bytes": report["scanned_bytes"], "findings": native_count},
              "adjudication_native_population")
    W.require(report.get("valid") is (len(issues) == 0), "adjudication_raw_verdict")
    result = {}
    for line, index, ordinal, finding in issues:
        W.require(ordinal in records, "adjudication_orphan_finding")
        subject = records[ordinal]
        if index is None:
            W.require(finding["byte_start"] < finding["byte_end"] <= subject["bytes"],
                      "adjudication_literal_range")
        identity = {"ledger_line": line, "finding_index": index,
                    "record_ordinal": ordinal, "record_sha256": subject["sha256"],
                    "record_bytes": subject["bytes"], "finding": finding}
        occurrence = W.sha(W.canonical(identity))
        W.require(occurrence not in result, "adjudication_duplicate_occurrence")
        result[occurrence] = identity
    return result


def context(authorization, report, report_hash, records_hash, authorization_file_hash,
            image_source_sha, scanner_source_sha):
    return {"authorization_sha256": W.sha(W.canonical(authorization)),
            "authorization_file_sha256": authorization_file_hash,
            "archive_sha256": authorization["archive"]["sha256"],
            "expected_image_id": authorization["expected_image_id"],
            "image_config_digest": report["image_config_digest"],
            "image_manifest_digest": report.get("image_manifest_digest"),
            "scanner_sources_sha256": W.sha(W.canonical(authorization["sources"])),
            "confidentiality_policy_sha256": W.sha(W.canonical(report["confidentiality_policy"])),
            "image_source_sha": image_source_sha, "scanner_source_sha": scanner_source_sha,
            "report_sha256": report_hash, "records_sha256": records_hash}


def image_revision(authorization, verification):
    """Read the config of a Docker save graph accepted by core.graph.

    Like core.graph, this requires manifest.json. OCI-only layout directories
    are not an input format for this command.
    """
    with W.bound_open(authorization["archive"], secret=True) as (_path, fd, info):
        W.graph(fd, info.st_size, verification, authorization["expected_image_id"])
        os.lseek(fd, 0, os.SEEK_SET)
        with os.fdopen(os.dup(fd), "rb") as stream, tarfile.open(fileobj=stream, mode="r:") as archive:
            manifest = decode(archive.extractfile("manifest.json").read())
            payload = archive.extractfile(manifest[0]["Config"]).read()
            W.require("sha256:" + W.sha(payload) == verification["image_config_digest"],
                      "adjudication_config_changed")
            config = decode(payload)
            revision = config["config"]["Labels"]["org.opencontainers.image.revision"]
            W.require(isinstance(revision, str) and re.fullmatch(r"[0-9a-f]{40}", revision),
                      "adjudication_image_revision")
            return revision


def committed_sources(authorization):
    root = W._ROOTS.get()[1]
    sha = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"],
                                  text=True).strip()
    W.require(re.fullmatch(r"[0-9a-f]{40}", sha), "adjudication_scanner_revision")
    W.require(authorization["sources"] == W.source_bindings(), "adjudication_source_population")
    for name, binding in authorization["sources"].items():
        try:
            content = subprocess.check_output(["git", "-C", str(root), "cat-file", "blob", f"{sha}:{name}"],
                                              stderr=subprocess.DEVNULL)
        except subprocess.CalledProcessError as error:
            raise W.ScanError("adjudication_uncommitted_source") from error
        W.require(W.sha(content) == binding["sha256"], "adjudication_uncommitted_source")
    return sha


def policy_receipt(authorization):
    literal = authorization.get("literal_inventory")
    literal_receipt = typed = None
    if literal is not None:
        inventory = W.bound_json(literal)
        values = inventory.get("literals")
        W.require(isinstance(values, list) and all(isinstance(value, str) and value for value in values),
                  "adjudication_literal_inventory")
        matching = literal["matching_policy"]
        W.require(matching in {"exact-substring-v1", W.POLICY}, "adjudication_literal_policy")
        matcher_sha = (W.AHO_PINS["source"] if authorization.get("literal_engine") is not None
                       else W.source_bindings()["npa/scripts/image_byte_scan/core.py"]["sha256"])
        literal_receipt = {"kind": matching, "inventory_sha256": literal["sha256"],
                           "pattern_count": len(values), "matcher_sha256": matcher_sha}
        typed = W.C.LiteralPolicyBinding(W.sha(W.canonical(literal_receipt)), matcher_sha, len(values))
    if authorization.get("confidentiality") is not None:
        policy = W.bound_json(authorization["confidentiality"])
        W.require(isinstance(policy, dict) and set(policy) <= {"customer_pattern", "infra_pattern"},
                  "adjudication_policy_schema")
        return W.C.compile_policy(policy.get("customer_pattern"), policy.get("infra_pattern"),
                                  literal_policy=typed).receipt()
    W.require(literal_receipt is not None and literal_receipt["pattern_count"] > 0,
              "adjudication_missing_confidentiality")
    return {"mode": "exact-literals-v1", "binding": literal_receipt}


def dispositions(manifest, review, occurrence_map, expected_context, manifest_hash, loader):
    """Verify the authorized review, with no automatic semantic classification."""
    fields(manifest, {"schema_version", "context", "dispositions"},
           "adjudication_manifest_schema")
    fields(review, {"schema_version", "decision", "manifest_sha256", "context",
                    "reviewed_occurrences"}, "adjudication_review_schema")
    W.require(manifest["schema_version"] == MANIFEST_SCHEMA
              and review["schema_version"] == REVIEW_SCHEMA,
              "adjudication_schema_version")
    W.require(manifest["context"] == expected_context == review["context"],
              "adjudication_stale_context")
    W.require(review["decision"] == "accept" and review["manifest_sha256"] == manifest_hash,
              "adjudication_review_not_accepted")
    W.require(isinstance(manifest["dispositions"], list)
              and isinstance(review["reviewed_occurrences"], list),
              "adjudication_dispositions_type")
    reviewed = {}
    for row in review["reviewed_occurrences"]:
        fields(row, {"occurrence_id", "proof_sha256", "decision"},
               "adjudication_review_occurrence_schema")
        key = digest(row["occurrence_id"])
        W.require(key not in reviewed and row["decision"] == "accept",
                  "adjudication_duplicate_or_unreviewed_occurrence")
        reviewed[key] = digest(row["proof_sha256"])
    accepted = set()
    for row in manifest["dispositions"]:
        fields(row, {"occurrence_id", "proof"}, "adjudication_disposition_schema")
        key = digest(row["occurrence_id"])
        W.require(key in occurrence_map and key not in accepted,
                  "adjudication_unknown_or_duplicate_disposition")
        proof_binding = row["proof"]
        fields(proof_binding, {"path", "sha256"}, "adjudication_proof_binding")
        W.require(reviewed.get(key) == proof_binding["sha256"],
                  "adjudication_occurrence_not_independently_reviewed")
        proof = loader(proof_binding)
        fields(proof, {"schema_version", "context", "occurrence_id", "record_sha256",
                       "record_bytes", "semantic_role", "operational_credential",
                       "provenance_evidence", "semantic_evidence"},
               "adjudication_proof_schema")
        expected = occurrence_map[key]
        W.require(proof["schema_version"] == PROOF_SCHEMA
                  and proof["context"] == expected_context
                  and proof["occurrence_id"] == key
                  and proof["record_sha256"] == expected["record_sha256"]
                  and integer(proof["record_bytes"]) == expected["record_bytes"],
                  "adjudication_proof_subject_changed")
        W.require(proof["semantic_role"] in ROLES and proof["operational_credential"] is False,
                  "adjudication_unsafe_semantics")
        # Both independently reviewed evidence bodies remain retrievable and
        # unchanged. Mere public origin is not semantic proof of non-use.
        for name in ("provenance_evidence", "semantic_evidence"):
            bindings = proof[name]
            W.require(isinstance(bindings, list) and bool(bindings),
                      "adjudication_missing_evidence")
            for binding in bindings:
                fields(binding, {"path", "sha256"}, "adjudication_evidence_binding")
                loader(binding, json_body=False)
        accepted.add(key)
    W.require(accepted == set(occurrence_map) == set(reviewed),
              "adjudication_unaccounted_occurrence")
    return len(accepted)


def verify(args):
    manifest = pinned_json(args.manifest, args.manifest_sha256)
    review = pinned_json(args.review, args.review_sha256)
    expected = manifest.get("context", {})
    authorization = pinned_json(args.authorization, expected.get("authorization_file_sha256"))
    required = {"schema_version", "accepted_verification", "archive", "verification_report",
                "expected_image_id", "helper", "config", "sources", "tools_receipt"}
    W.require(isinstance(authorization, dict) and required <= set(authorization) <= required | {
        "literal_inventory", "literal_engine", "confidentiality"}, "adjudication_authorization_fields")
    W.require(authorization.get("schema_version") == "npa.image-byte-scan-authorization.v1"
              and authorization.get("accepted_verification") is True, "adjudication_authorization_schema")
    helper, config = W.verified_tools(authorization["tools_receipt"])
    W.require(helper == authorization["helper"] and config == authorization["config"],
              "adjudication_tools_authorization")
    snapshots = W.input_snapshots(authorization)
    verification = W.bound_json(authorization["verification_report"])
    W.require(verification.get("valid") is True
              and verification.get("schema_version") == "npa.curobo.image-verification.v1",
              "adjudication_verification_failed")
    W.bound_file(authorization["archive"], secret=True)
    report = pinned_json(args.report, expected.get("report_sha256"))
    ledger_bytes = bound_bytes({"path": str(args.records), "sha256": expected.get("records_sha256")})
    rows = [decode(line) for line in ledger_bytes.splitlines()]
    occurrence_map = population(report, rows)
    image_source_sha = image_revision(authorization, verification)
    scanner_source_sha = committed_sources(authorization)
    actual = context(authorization, report, expected["report_sha256"], expected["records_sha256"],
                     expected["authorization_file_sha256"], image_source_sha, scanner_source_sha)
    W.require(report.get("authorization_sha256") == actual["authorization_sha256"]
              and report.get("archive_sha256") == actual["archive_sha256"]
              and verification.get("docker_save_sha256") == actual["archive_sha256"]
              and report.get("expected_image_id") == authorization["expected_image_id"]
              and verification.get("expected_image_id") == authorization["expected_image_id"]
              and verification.get("image_config_digest") == report["image_config_digest"]
              and verification.get("image_manifest_digest") == report.get("image_manifest_digest")
              and policy_receipt(authorization) == report.get("confidentiality_policy"),
              "adjudication_artifact_binding")
    expected_snapshots = [{"role": role, "sha256": spec["sha256"], "stat": list(before)}
                          for role, spec, _secret, _path, before in snapshots]
    W.require(report.get("input_snapshot_receipts") == expected_snapshots,
              "adjudication_scan_inputs_changed")
    consumed = {}
    json_cache = {}

    def load(binding, json_body=True):
        fields(binding, {"path", "sha256"}, "adjudication_evidence_binding")
        key = binding["path"]
        W.require(key not in consumed or consumed[key] == binding,
                  "adjudication_conflicting_evidence_binding")
        if key not in consumed:
            if json_body:
                json_cache[key] = decode(bound_bytes(binding))
            else:
                with W.bound_open(binding, secret=True) as (_path, _fd, info):
                    W.require(info.st_size > 0, "adjudication_empty_evidence")
            consumed[key] = binding
        if json_body:
            if key not in json_cache:
                json_cache[key] = decode(bound_bytes(binding))
            return json_cache[key]
        return None

    count = dispositions(manifest, review, occurrence_map, actual,
                         args.manifest_sha256, load)
    for spec in consumed.values():
        W.bound_file(spec, secret=True)
    W.recheck_snapshots(snapshots)
    for path, expected_hash in ((args.manifest, args.manifest_sha256),
                                (args.review, args.review_sha256),
                                (args.report, expected["report_sha256"]),
                                (args.records, expected["records_sha256"]),
                                (args.authorization, expected["authorization_file_sha256"])):
        W.bound_file({"path": str(path), "sha256": expected_hash}, secret=True)
    W.bound_file(authorization["archive"], secret=True)
    W.require(committed_sources(authorization) == scanner_source_sha, "adjudication_scanner_head_changed")
    return {"schema_version": SCHEMA, "accepted": True, "context": actual,
            "raw_scan_valid": report["valid"], "raw_scan_findings": report["findings"],
            "accepted_occurrences": count, "unresolved_occurrences": 0,
            "manifest_sha256": args.manifest_sha256, "review_sha256": args.review_sha256}


def main(argv=None):
    os.umask(0o077)
    parser = W.SanitizedArgumentParser(description=__doc__)
    for option in ("analysis-root", "trusted-root", "manifest", "review", "authorization",
                   "report", "records", "output-dir"):
        parser.add_argument("--" + option, type=Path, required=True)
    for option in ("manifest-sha256", "review-sha256"):
        parser.add_argument("--" + option, required=True)
    args = parser.parse_args(argv)
    try:
        with W.authorized_roots(args.analysis_root, args.trusted_root):
            result = verify(args)
            output, fd = W.create_output(args.output_dir)
            try:
                W.write_private_json(output, "adjudication.json", result)
            finally:
                os.close(fd)
        print("image byte findings independently accepted")
        return 0
    except (*W.INPUT_ERRORS, subprocess.SubprocessError):
        print("image byte adjudication failed")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
