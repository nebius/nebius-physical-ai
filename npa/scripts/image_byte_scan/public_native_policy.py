"""Conserve fresh native findings against reviewed exact public content.

The scanner constructs this state before scanning and binds its emitted ledger.
This module does not authenticate caller-supplied reports or review private
confidentiality findings. A successful receipt is separate from the raw verdict.
"""

from __future__ import annotations

from collections import Counter

from image_byte_scan import adjudicate as A

W = A.W
SCHEMA = "npa.image-native-content-policy.v1"
ROLES = {
    "cryptographic-self-test",
    "parser-format-delimiter",
    "package-integrity-metadata",
    "non-operational-source-example",
    "public-package-path-metadata",
    "linker-type-framing",
    "encoded-device-payload",
    "bytecode-framing",
    "public-source-symbol",
    "numeric-source-constant",
}
KINDS = {"layer_regular_content", "logical_tar_path"}


def native_multiset(findings):
    W.require(isinstance(findings, list), "public_policy_findings_type")
    result = Counter()
    for finding in findings:
        A.fields(
            finding,
            {"rule_id", "start_line", "end_line"},
            "public_policy_confidentiality_fatal",
        )
        W.require(
            isinstance(finding["rule_id"], str) and bool(finding["rule_id"]),
            "public_policy_rule",
        )
        A.integer(finding["start_line"])
        A.integer(finding["end_line"])
        W.require(
            finding["start_line"] <= finding["end_line"], "public_policy_native_range"
        )
        result[W.canonical(finding)] += 1
    return result


def compile_catalog(catalog, detector_identity, proof_loader):
    A.fields(
        catalog,
        {"schema_version", "detector_identity", "entries"},
        "public_policy_schema",
    )
    W.require(catalog["schema_version"] == SCHEMA, "public_policy_version")
    W.require(
        catalog["detector_identity"] == detector_identity,
        "public_policy_detector_changed",
    )
    W.require(isinstance(catalog["entries"], list), "public_policy_entries")
    entries = {}
    for entry in catalog["entries"]:
        A.fields(
            entry,
            {
                "record_kind",
                "record_sha256",
                "record_bytes",
                "native_findings",
                "semantic_role",
                "operational_credential",
                "public_provenance",
                "semantic_proof",
            },
            "public_policy_entry",
        )
        record_hash = A.digest(entry["record_sha256"])
        A.integer(entry["record_bytes"])
        W.require(
            entry["record_bytes"] > 0
            and entry["semantic_role"] in ROLES
            and entry["operational_credential"] is False,
            "public_policy_unsafe_semantics",
        )
        W.require(entry["record_kind"] in KINDS, "public_policy_record_kind")
        W.require(
            entry["record_kind"] != "logical_tar_path"
            or entry["semantic_role"] == "public-package-path-metadata",
            "public_policy_metadata_role",
        )
        key = (entry["record_kind"], record_hash)
        W.require(key not in entries, "public_policy_duplicate_content")
        findings = native_multiset(entry["native_findings"])
        W.require(bool(findings), "public_policy_empty_native_population")
        for binding in (entry["public_provenance"], entry["semantic_proof"]):
            A.fields(binding, {"path", "sha256"}, "public_policy_proof_binding")
            A.digest(binding["sha256"])
            # The exact evidence bodies are committed public product policy.
            # Their actual semantics must have received independent review; this
            # validates bindings, never infers safety from labels or origin URLs.
            proof_loader(binding)
        entries[key] = (entry["record_bytes"], findings)
    return entries


def match_fresh_population(entries, report, rows):
    """Conserve every fresh occurrence; confidentiality findings are always fatal."""
    occurrences = A.population(report, rows)
    accepted = set()
    records = {r["record_ordinal"]: r for r in rows if r.get("type") == "record"}
    for row in rows:
        if row.get("type") == "finding":
            raise W.ScanError("public_policy_confidentiality_fatal")
        if row.get("type") != "record" or not row["findings"]:
            continue
        W.require(row["kind"] in KINDS, "public_policy_unsupported_record_kind")
        native = native_multiset(row["findings"])
        expected = entries.get((row["kind"], row["sha256"]))
        W.require(
            expected is not None and expected == (row["bytes"], native),
            "public_policy_unreviewed_content_or_population",
        )
    for identity, occurrence in occurrences.items():
        row = records[occurrence["record_ordinal"]]
        W.require(
            row["kind"] in KINDS and occurrence["finding_index"] is not None,
            "public_policy_unreviewed_occurrence",
        )
        accepted.add(identity)
    W.require(
        len(accepted) == report["findings"], "public_policy_occurrence_conservation"
    )
    return {
        "raw_scan_valid": report["valid"],
        "raw_findings": report["findings"],
        "accepted_native_occurrences": len(accepted),
        "unresolved_occurrences": 0,
    }


class FreshPolicyReview:
    """Internal state created by core.main before its fresh scan; no report CLI."""

    def __init__(
        self, catalog_path, catalog_sha256, authorization, authorization_binding
    ):
        import hashlib
        from pathlib import Path

        self.accepted = False
        self.authorization = authorization
        self.authorization_binding = authorization_binding
        W.bound_file(authorization_binding, secret=True)
        self.snapshots = W.input_snapshots(authorization)
        self.scanner_revision = self._committed_revision()
        self.sources = W.source_bindings()
        self.catalog_path = Path(catalog_path).absolute()
        self.catalog_sha256 = A.digest(catalog_sha256)
        self._ledger_digest = hashlib.sha256()
        helper, config = W.verified_tools(authorization["tools_receipt"])
        self.detector_identity = {
            "helper_sha256": helper["sha256"],
            "config_sha256": config["sha256"],
        }
        self.proof_bindings = {}
        catalog = A.decode(self._public_source(self.catalog_path, self.catalog_sha256))
        self.entries = compile_catalog(catalog, self.detector_identity, self._proof)

    def _committed_revision(self):
        import subprocess

        try:
            return A.committed_sources(self.authorization)
        except subprocess.SubprocessError as error:
            raise W.ScanError("public_policy_scanner_revision_unavailable") from error

    def _public_source(self, path, expected):
        from pathlib import Path

        path = Path(path).absolute()
        root = W._ROOTS.get()[1]
        W.require(path.is_relative_to(root), "public_policy_source_scope")
        name = str(path.relative_to(root))
        W.require(
            name in self.sources and self.sources[name]["sha256"] == expected,
            "public_policy_uncommitted_evidence",
        )
        with W.open_source_fd(path) as (fd, _info):
            data = W.descriptor_bytes(fd)
        W.require(
            bool(data) and W.sha(data) == expected, "public_policy_evidence_changed"
        )
        self.proof_bindings[name] = self.sources[name]
        return data

    def _proof(self, binding):
        from pathlib import PurePosixPath

        name = binding["path"]
        W.require(
            isinstance(name, str)
            and not PurePosixPath(name).is_absolute()
            and ".." not in PurePosixPath(name).parts,
            "public_policy_proof_scope",
        )
        return self._public_source(W._ROOTS.get()[1] / name, binding["sha256"])

    def observe(self, emitted_bytes):
        # core.Ledger calls after the exact bytes are flushed to the ledger. Detection
        # input and every finding remain untouched, including denied findings.
        W.require(type(emitted_bytes) is bytes, "public_policy_observer_bytes")
        self._ledger_digest.update(emitted_bytes)

    def accept_fresh_scan(self, report, directory):
        import json

        # Only core.main calls this after _scan returns and the raw report is
        # written. There is no input-report CLI and no hash synthesized to imply
        # that an untrusted caller's purported scan had actually executed.
        rows_spec = {
            "path": str(directory / "records.jsonl"),
            "sha256": self._ledger_digest.hexdigest(),
        }
        row_bytes = A.bound_bytes(rows_spec)
        rows = [A.decode(line) for line in row_bytes.splitlines()]
        verdict = match_fresh_population(self.entries, report, rows)
        report_bytes = (json.dumps(report, sort_keys=True, indent=2) + "\n").encode()
        report_spec = {
            "path": str(directory / "report.json"),
            "sha256": W.sha(report_bytes),
        }
        W.bound_file(report_spec, secret=True)
        W.require(
            report.get("authorization_sha256")
            == W.sha(W.canonical(self.authorization)),
            "public_policy_authorization_changed",
        )
        verification = W.bound_json(self.authorization["verification_report"])
        image_revision = A.image_revision(self.authorization, verification)
        W.require(
            self._committed_revision() == self.scanner_revision,
            "public_policy_scanner_changed",
        )
        for binding in self.proof_bindings.values():
            W.bound_file(binding, secret=False)
        W.bound_file(self.authorization["archive"], secret=True)
        W.bound_file(rows_spec, secret=True)
        W.bound_file(report_spec, secret=True)
        receipt = {
            "schema_version": "npa.image-byte-policy-acceptance.v1",
            "accepted": True,
            "mode": "reviewed-exact-native-content",
            **verdict,
            "catalog_sha256": self.catalog_sha256,
            "scanner_source_sha": self.scanner_revision,
            "image_source_sha": image_revision,
            "scanner_sources_sha256": W.sha(W.canonical(self.sources)),
            "public_proof_sources_sha256": W.sha(W.canonical(self.proof_bindings)),
            "detector_identity": self.detector_identity,
            "authorization_sha256": report["authorization_sha256"],
            "archive_sha256": report["archive_sha256"],
            "image_config_digest": report["image_config_digest"],
            "image_manifest_digest": report["image_manifest_digest"],
            "confidentiality_policy_sha256": W.sha(
                W.canonical(report["confidentiality_policy"])
            ),
            "report_sha256": report_spec["sha256"],
            "records_sha256": rows_spec["sha256"],
        }
        W.bound_file(self.authorization_binding, secret=True)
        W.recheck_snapshots(self.snapshots)
        W.require(
            self._committed_revision() == self.scanner_revision,
            "public_policy_scanner_changed",
        )
        W.write_private_json(directory, "public-policy-acceptance.json", receipt)
        self.accepted = True
        return receipt
