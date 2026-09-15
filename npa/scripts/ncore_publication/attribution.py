"""Verify the one provenance-proven CPython notice in an NCore image scan."""

from __future__ import annotations

import gzip
import hashlib
import os
import posixpath
from pathlib import Path
import subprocess
import tarfile

from image_byte_scan import core as W
from . import artifact
from .process import ROOT, PYTHON, committed_source, file_sha, run_byte_scanner, write_json

SCHEMA = "npa.ncore.public-attribution-acceptance.v1"
_REPOSITORY_PATH = "npa/docker/workbench/ncore/notices/cpython/LICENSE.third-party"
_IMAGE_PATH = "usr/share/doc/npa-ncore/cpython/LICENSE.third-party"
_LINES = frozenset({633, 640})
_PHYSICAL_RECORD_KINDS = frozenset({
    "raw_tar_header", "raw_tar_extension", "outer_regular_content",
    "layer_regular_content", "nonzero_tar_padding", "unexplained_tar_trailer",
    "verified_zero_content",
})
_LOGICAL_RECORD_KINDS = frozenset({"logical_tar_path", "logical_tar_link"})
_COMPRESSED_RECORD_KINDS = frozenset({"raw_gzip_header"})
_KNOWN_RECORD_KINDS = (_PHYSICAL_RECORD_KINDS | _LOGICAL_RECORD_KINDS
                       | _COMPRESSED_RECORD_KINDS)


def _notice_api():
    from npa.guardrails import ncore_attribution

    return ncore_attribution


def _contract(api):
    W.require(api.REPOSITORY_PATH == _REPOSITORY_PATH
              and api.IMAGE_PATH == _IMAGE_PATH
              and api.ATTRIBUTION_LINES == _LINES
              and isinstance(api.NOTICE_SHA256, str)
              and len(api.NOTICE_SHA256) == 64
              and type(api.NOTICE_SIZE) is int and api.NOTICE_SIZE > 0,
              "ncore_attribution_api_contract")


def _git_tree(source_sha):
    raw = subprocess.check_output(["git", "ls-tree", "-r", "-z", source_sha], cwd=ROOT)
    rows = []
    for entry in raw.split(b"\0"):
        if not entry:
            continue
        header, path = entry.split(b"\t", 1)
        mode, kind, object_id = header.decode("ascii").split(" ")
        rows.append((mode, kind, object_id, path.decode("utf-8")))
    return rows


def _repository_notice(source_sha, api):
    rows = _git_tree(source_sha)
    selected = [row for row in rows if row[3] == api.REPOSITORY_PATH]
    W.require(len(selected) == 1 and selected[0][:2] == ("100644", "blob"),
              "ncore_attribution_repository_record")
    object_id = selected[0][2]
    W.require(sum(row[2] == object_id for row in rows) == 1,
              "ncore_attribution_repository_copy")
    _reject_repository_aliases(rows, api.REPOSITORY_PATH)
    notice = subprocess.check_output(["git", "cat-file", "blob", object_id], cwd=ROOT)
    W.require(len(notice) == api.NOTICE_SIZE and W.sha(notice) == api.NOTICE_SHA256,
              "ncore_attribution_repository_bytes")
    return notice, object_id


def _reject_repository_aliases(rows, target):
    links = {}
    for mode, kind, object_id, path in rows:
        if mode != "120000" or kind != "blob":
            continue
        raw = subprocess.check_output(["git", "cat-file", "blob", object_id], cwd=ROOT)
        try:
            links[path] = raw.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("ncore_attribution_repository_alias") from error
    W.require(not any(_alias_reaches(path, links, target) for path in links),
              "ncore_attribution_repository_alias")


def _alias_reaches(path, links, target):
    visited = set()
    while path in links and path not in visited:
        visited.add(path)
        path = _resolved_link(path, links[path])
        if path == target:
            return True
    return False


def _read_private(path):
    path, fd, before = W.open_private_fd(path)
    try:
        raw = W.descriptor_bytes(fd)
        W.require(W.stat_fingerprint(os.fstat(fd)) == W.stat_fingerprint(before),
                  "ncore_attribution_evidence_changed")
    finally:
        os.close(fd)
    return raw, W.sha(raw), W.stat_fingerprint(before)


def _scan_inputs(directory):
    authorization_raw, authorization_sha, _ = _read_private(
        directory / "authorization/authorization.json")
    report_raw, report_sha, report_stat = _read_private(directory / "bytes/report.json")
    records_raw, records_sha, records_stat = _read_private(directory / "bytes/records.jsonl")
    authorization = W.json_object(authorization_raw)
    report = W.json_object(report_raw)
    rows = [W.json_object(line) for line in records_raw.splitlines()]
    W.require(bool(rows), "ncore_attribution_empty_ledger")
    return authorization, report, rows, {
        "authorization_sha256": authorization_sha, "report_sha256": report_sha,
        "records_sha256": records_sha, "report_stat": report_stat, "records_stat": records_stat,
    }


def _context(row):
    allowed = {"scope", "layer_ordinal", "entry_ordinal", "tar_offset", "compressed_offset"}
    result = {name: row[name] for name in allowed if name in row}
    W.require(result.get("scope") in {"outer", "layer"}, "ncore_attribution_record_scope")
    for name, value in result.items():
        if name != "scope":
            W.require(type(value) is int and value >= 0, "ncore_attribution_record_context")
    return result


def _record(row, expected_ordinal):
    base = {"type", "record_ordinal", "kind", "bytes", "sha256", "findings"}
    context = _context(row)
    W.require(set(row) == base | set(context)
              and row["type"] == "record" and row["record_ordinal"] == expected_ordinal
              and isinstance(row["kind"], str) and type(row["bytes"]) is int and row["bytes"] >= 0
              and isinstance(row["sha256"], str) and len(row["sha256"]) == 64
              and isinstance(row["findings"], list), "ncore_attribution_record_schema")
    return context


def _confidentiality_row(row, record, policy_sha):
    expected = {"type", "record_ordinal", "policy_sha256", "sha256", "bytes",
                "line_count", "composed_findings"}
    W.require(set(row) == expected and row["type"] == "confidentiality_record"
              and row["record_ordinal"] == record["record_ordinal"]
              and row["policy_sha256"] == policy_sha and row["sha256"] == record["sha256"]
              and row["bytes"] == record["bytes"]
              and type(row["line_count"]) is int and row["line_count"] >= 0
              and row["composed_findings"] == len(record["findings"]),
              "ncore_attribution_confidentiality_record")


def _other_row(row):
    kind = row.get("type")
    if kind == "verified_zero_range":
        context = _context(row)
        expected = {"type", "bytes", "sha256", *context}
        W.require(set(row) == expected and type(row["bytes"]) is int and row["bytes"] > 0,
                  "ncore_attribution_zero_row")
        return row["bytes"], context
    W.require(kind == "encoded_layer_blob", "ncore_attribution_structural_finding")
    context = _context(row)
    expected = {"type", "bytes", "sha256", *context}
    W.require(set(row) == expected and type(row["bytes"]) is int and row["bytes"] >= 0,
              "ncore_attribution_encoded_layer")
    return 0, context


def _ledger_population(report, rows, policy_sha):
    records, issues, zero_bytes, scope_bytes = [], [], 0, {}
    pending = None
    for line_number, row in enumerate(rows, 1):
        if row.get("type") == "confidentiality_record":
            W.require(pending is None, "ncore_attribution_ledger_order")
            pending = row
        elif row.get("type") == "record":
            context = _record(row, len(records) + 1)
            W.require(pending is not None, "ncore_attribution_missing_confidentiality_record")
            _confidentiality_row(pending, row, policy_sha)
            pending = None
            records.append((line_number, row, context))
            W.require(row["kind"] in _KNOWN_RECORD_KINDS, "ncore_attribution_record_kind")
            if row["kind"] in _PHYSICAL_RECORD_KINDS:
                scope_bytes[_scope_key(context)] = scope_bytes.get(_scope_key(context), 0) + row["bytes"]
            issues.extend((line_number, index, row, finding)
                          for index, finding in enumerate(row["findings"]))
        else:
            W.require(pending is None, "ncore_attribution_ledger_order")
            amount, context = _other_row(row)
            zero_bytes += amount
    W.require(pending is None, "ncore_attribution_ledger_order")
    _report_population(report, records, issues, zero_bytes, scope_bytes)
    return records, issues


def _scope_key(context):
    return (context["scope"], context.get("layer_ordinal"))


def _report_population(report, records, issues, zero_bytes, scope_bytes):
    regular = [row for _, row, _ in records if row["kind"] == "layer_regular_content"]
    scanned = sum(row["bytes"] for _, row, _ in records)
    helper = report.get("helper_summary")
    W.require(report.get("schema_version") == "npa.image-byte-scan.v1"
              and report.get("complete") is True and report.get("helper_joined") is True
              and report.get("valid") is False and "failure_code" not in report
              and report.get("records") == len(records) and report.get("scanned_bytes") == scanned
              and report.get("regular_files") == len(regular)
              and report.get("regular_bytes") == sum(row["bytes"] for row in regular)
              and report.get("verified_zero_bytes") == zero_bytes
              and report.get("findings") == len(issues) == 2,
              "ncore_attribution_raw_population")
    W.require(helper == {"type": "summary", "files": len(records), "bytes": scanned,
                         "findings": 0}, "ncore_attribution_native_population")
    W.require(report["outer"]["decoded_bytes"] == scope_bytes.get(("outer", None), 0),
              "ncore_attribution_outer_ledger_population")
    for layer in report["layers"]:
        W.require(layer["decoded_bytes"] == scope_bytes.get(("layer", layer["ordinal"]), 0),
                  "ncore_attribution_layer_ledger_population")


def _policy(authorization, report):
    W.require("literal_inventory" not in authorization
              and authorization.get("confidentiality") is not None,
              "ncore_attribution_regex_policy_required")
    engine_receipt = _native_engine_receipt(authorization)
    W.require(report.get("literal_engine") == engine_receipt,
              "ncore_attribution_native_engine_changed")
    configured = W.bound_json(authorization["confidentiality"])
    W.require(isinstance(configured, dict)
              and set(configured) <= {"customer_pattern", "infra_pattern"},
              "ncore_attribution_policy_schema")
    policy = W.C.compile_policy(configured.get("customer_pattern"), configured.get("infra_pattern"))
    receipt = policy.receipt()
    W.require(receipt == report.get("confidentiality_policy"),
              "ncore_attribution_policy_changed")
    return receipt


def _native_engine_receipt(authorization):
    engine = authorization.get("literal_engine")
    W.require(isinstance(engine, dict) and set(engine) == {"kind", *W.AHO_PINS}
              and engine.get("kind") == "aho-corasick-v1",
              "ncore_attribution_native_engine_binding")
    for role, digest in W.AHO_PINS.items():
        binding = engine.get(role)
        W.require(isinstance(binding, dict) and set(binding) == {"path", "sha256"}
                  and binding.get("sha256") == digest,
                  "ncore_attribution_native_engine_binding")
    sources = authorization.get("sources")
    source_name = "npa/scripts/image_byte_scan/aho_matcher.py"
    W.require(isinstance(sources, dict) and engine["source"] == sources.get(source_name),
              "ncore_attribution_native_engine_source")
    return _expected_native_engine_receipt()


def _expected_native_engine_receipt():
    return {"kind": "aho-corasick-v1", "pinned_sha256": W.AHO_PINS,
            "sealed_native_copy": True}


def _report_graph(report, graph, verification, archive_size):
    expected_fields = {
        "schema_version", "valid", "complete", "authorization_sha256", "archive_sha256",
        "image_config_digest", "image_manifest_digest", "expected_image_id",
        "private_literals_configured", "private_literal_count", "literal_matching_policy",
        "layers", "oci_graph", "outer", "confidentiality_policy", "helper_summary",
        "literal_engine", "input_snapshot_receipts", "helper_joined", "records",
        "scanned_bytes", "verified_zero_bytes", "regular_files", "regular_bytes", "findings",
    }
    W.require(report.get("archive_sha256") == verification["archive_sha256"]
              and report.get("expected_image_id") == verification["expected_image_id"]
              and report.get("image_config_digest") == verification["image_config_digest"]
              and report.get("image_manifest_digest") == verification["image_manifest_digest"]
              and report.get("regular_files") == verification["regular_files_read"]
              and report.get("regular_bytes") == verification["content_bytes_read"]
              and set(report) == expected_fields and report.get("oci_graph") == graph["receipt"]
              and report.get("private_literals_configured") is False
              and report.get("private_literal_count") == 0
              and report.get("literal_engine") == _expected_native_engine_receipt(),
              "ncore_attribution_report_artifact_binding")
    layers = report.get("layers")
    W.require(isinstance(layers, list) and len(layers) == len(graph["layers"]),
              "ncore_attribution_report_layers")
    _report_layers(layers, graph["layers"])
    outer = report.get("outer")
    W.require(isinstance(outer, dict) and set(outer) == {"headers", "decoded_bytes", "zero_end_blocks"}
              and all(type(outer.get(name)) is int and outer[name] >= 0 for name in outer)
              and outer["decoded_bytes"] == archive_size,
              "ncore_attribution_outer_population")


def _report_layers(layers, expected_layers):
    for actual, expected in zip(layers, expected_layers, strict=True):
        expected_layer_fields = {"ordinal", "diff_id", "compressed_sha256", "compressed_bytes",
                                 "codec", "headers", "decoded_bytes", "zero_end_blocks"}
        descriptor = expected["descriptor"]
        codec = "gzip" if descriptor["mediaType"].endswith("+gzip") else "raw"
        W.require(set(actual) == expected_layer_fields
                  and actual.get("ordinal") == expected["ordinal"]
                  and actual.get("diff_id") == expected["diff_id"]
                  and actual.get("compressed_bytes") == expected["size"]
                  and "sha256:" + actual.get("compressed_sha256", "") == descriptor["digest"]
                  and actual.get("codec") == codec
                  and all(type(actual.get(name)) is int and actual[name] >= 0
                          for name in ("headers", "decoded_bytes", "zero_end_blocks")),
                  "ncore_attribution_report_layer_binding")


def _authorization_binding(authorization, report, archive, digest, verification):
    W.require(authorization.get("schema_version") == "npa.image-byte-scan-authorization.v1"
              and authorization.get("accepted_verification") is True
              and Path(authorization["archive"]["path"]).absolute() == archive.absolute()
              and authorization["archive"]["sha256"] == file_sha(archive)
              and authorization.get("expected_image_id") == digest
              and W.bound_json(authorization["verification_report"]) == verification
              and report.get("authorization_sha256") == W.sha(W.canonical(authorization)),
              "ncore_attribution_authorization_binding")
    snapshots = W.input_snapshots(authorization)
    expected = [{"role": role, "sha256": spec["sha256"], "stat": list(before)}
                for role, spec, _secret, _path, before in snapshots]
    W.require(report.get("input_snapshot_receipts") == expected,
              "ncore_attribution_input_snapshot_binding")
    return snapshots


def _layer_stream(archive, layer):
    member = archive.getmember(layer["name"])
    stream = archive.extractfile(member)
    W.require(stream is not None, "ncore_attribution_layer_missing")
    if layer["descriptor"]["mediaType"].endswith("+gzip"):
        return gzip.GzipFile(fileobj=stream)
    return stream


def _decoded_layer_sizes(archive_path, graph):
    sizes = []
    with tarfile.open(archive_path) as outer:
        for layer in graph["layers"]:
            digest, size = hashlib.sha256(), 0
            with _layer_stream(outer, layer) as stream:
                while chunk := stream.read(W.CHUNK):
                    digest.update(chunk)
                    size += len(chunk)
            W.require("sha256:" + digest.hexdigest() == layer["diff_id"],
                      "ncore_attribution_decoded_layer_changed")
            sizes.append(size)
    return sizes


def _resolved_link(name, link):
    if link.startswith("/"):
        return posixpath.normpath(link).removeprefix("/")
    return posixpath.normpath(posixpath.join(posixpath.dirname(name), link))


def _image_notice(archive_path, graph, notice, api):
    artifact._scratch_layers(archive_path, graph["layers"])
    decoded_sizes = _decoded_layer_sizes(archive_path, graph)
    matches, digest_copies, entries, links = [], [], {}, {}
    with tarfile.open(archive_path) as outer:
        with _layer_stream(outer, graph["layers"][0]) as stream, tarfile.open(fileobj=stream, mode="r|") as layer:
            for member in layer:
                name = W.safe_name(member.name)
                W.require(name not in entries, "ncore_attribution_image_duplicate_path")
                entries[name] = member
                if member.islnk():
                    links[name] = member.linkname
                elif member.issym():
                    links[name] = member.linkname
                if member.isfile():
                    raw = layer.extractfile(member).read()
                    if W.sha(raw) == api.NOTICE_SHA256:
                        digest_copies.append(name)
                    if name == api.IMAGE_PATH:
                        matches.append((member.offset_data, raw, member.mode))
    for parent in Path(api.IMAGE_PATH).parents:
        member = entries.get(parent.as_posix())
        W.require(member is None or member.isdir(), "ncore_attribution_image_ancestor_alias")
    W.require(not any(_alias_reaches(name, links, api.IMAGE_PATH) for name in links)
              and not any(member.islnk() for member in entries.values())
              and digest_copies == [api.IMAGE_PATH] and len(matches) == 1,
              "ncore_attribution_image_occurrence")
    data_offset, raw, mode = matches[0]
    W.require(raw == notice and len(raw) == api.NOTICE_SIZE and mode & 0o777 == 0o644,
              "ncore_attribution_image_notice")
    return {"layer_ordinal": 0, "data_offset": data_offset,
            "decoded_sizes": decoded_sizes}


def _accepted_occurrences(records, issues, image, api):
    path_sha = W.sha(api.IMAGE_PATH.encode("utf-8"))
    paths = [(row, context) for _, row, context in records
             if row["kind"] == "logical_tar_path" and row["sha256"] == path_sha
             and row["bytes"] == len(api.IMAGE_PATH.encode("utf-8"))]
    contents = [(line, row, context) for line, row, context in records
                if row["kind"] == "layer_regular_content"
                and row["sha256"] == api.NOTICE_SHA256 and row["bytes"] == api.NOTICE_SIZE]
    W.require(len(paths) == len(contents) == 1, "ncore_attribution_ledger_occurrence")
    path, path_context = paths[0]
    line, content, content_context = contents[0]
    expected = {"scope": "layer", "layer_ordinal": image["layer_ordinal"]}
    W.require({key: path_context.get(key) for key in expected} == expected
              and {key: content_context.get(key) for key in expected} == expected
              and path_context.get("entry_ordinal") == content_context.get("entry_ordinal")
              and content_context.get("tar_offset") == image["data_offset"]
              and not path["findings"], "ncore_attribution_ledger_image_binding")
    return _findings(issues, line, content, api)


def _findings(issues, ledger_line, content, api):
    accepted, lines = [], set()
    for line, index, record, finding in issues:
        expected = {"rule_id", "start_byte", "end_byte", "start_line", "end_line", "views"}
        W.require(line == ledger_line and record is content and set(finding) == expected
                  and finding["rule_id"] == "customer-denylist"
                  and finding["start_line"] == finding["end_line"] in api.ATTRIBUTION_LINES
                  and type(finding["start_byte"]) is int and type(finding["end_byte"]) is int
                  and 0 <= finding["start_byte"] <= finding["end_byte"] <= api.NOTICE_SIZE
                  and isinstance(finding["views"], list) and bool(finding["views"])
                  and set(finding["views"]) <= {"line", "record"}
                  and len(finding["views"]) == len(set(finding["views"])),
                  "ncore_attribution_finding_scope")
        identity = {"ledger_line": line, "finding_index": index,
                    "record_ordinal": content["record_ordinal"],
                    "record_sha256": content["sha256"], "record_bytes": content["bytes"],
                    "finding": finding}
        accepted.append(W.sha(W.canonical(identity)))
        lines.add(finding["start_line"])
    W.require(len(accepted) == 2 and len(set(accepted)) == 2 and lines == api.ATTRIBUTION_LINES,
              "ncore_attribution_finding_population")
    return sorted(accepted)


def _recheck(directory, bindings, snapshots, archive, verification):
    for relative, digest, fingerprint in (
        ("bytes/report.json", bindings["report_sha256"], bindings["report_stat"]),
        ("bytes/records.jsonl", bindings["records_sha256"], bindings["records_stat"]),
    ):
        raw, current, current_stat = _read_private(directory / relative)
        W.require(current == digest and current_stat == fingerprint and W.sha(raw) == digest,
                  "ncore_attribution_evidence_changed")
    W.recheck_snapshots(snapshots)
    artifact.assert_unchanged(archive, verification)


def _replay_population(args, directory, bindings):
    replay = directory / "attribution-replay"
    authorization = directory / "authorization/authorization.json"
    argv = [str(PYTHON), "npa/scripts/scan_image_bytes.py",
            "--analysis-root", str(args.analysis_root), "--trusted-root", str(ROOT),
            "--authorization", str(authorization), "--output-dir", str(replay)]
    status = run_byte_scanner(argv, directory / "attribution-replay.log")
    W.require(status == 1, "ncore_attribution_replay_exit")
    for name, expected in (("report.json", bindings["report_sha256"]),
                           ("records.jsonl", bindings["records_sha256"])):
        _raw, observed, _stat = _read_private(replay / name)
        W.require(observed == expected, "ncore_attribution_replay_population")


def _upstream_proof(api, notice, directory):
    proof_directory = directory / "attribution-upstream"
    proof_directory.mkdir(mode=0o700)
    proof = api.verify_public_notice(notice, proof_directory)
    W.require(isinstance(proof, dict), "ncore_attribution_upstream_proof")
    return proof


def recheck(directory, expected, archive, verification):
    """Recheck the in-memory acceptance and its raw evidence before transfer.

    Args:
        directory: Private gate evidence directory.
        expected: Acceptance receipt returned by :func:`verify` in this process.
        archive: Original OCI archive.
        verification: Native verification of that archive.
    Returns:
        None.
    Raises:
        ValueError, OSError: The receipt, raw evidence, or image changed.
    """
    raw, _digest, _stat = _read_private(directory / "attribution.json")
    W.require(W.json_object(raw) == expected, "ncore_attribution_receipt_changed")
    for name, field in (("report.json", "report_sha256"),
                        ("records.jsonl", "records_sha256")):
        _raw, observed, _fingerprint = _read_private(directory / "bytes" / name)
        W.require(observed == expected[field], "ncore_attribution_evidence_changed")
    artifact.assert_unchanged(archive, verification)


def verify(args, directory, archive, digest, verification, scanner_exit):
    """Accept only the exact provenance-proven NCore CPython attribution hits.

    Args:
        args: Validated NCore CLI arguments and reviewed source identity.
        directory: Private gate evidence directory containing raw scan outputs.
        archive: Original OCI archive scanned by the raw scanner.
        digest: Exact expected OCI publication index identity.
        verification: Native verification of the original OCI graph.
        scanner_exit: Raw scanner process exit status.
    Returns:
        A separate receipt; the raw report remains invalid and unchanged.
    Raises:
        ValueError, OSError: Any provenance, population, policy, source, or image binding fails.
    """
    W.require(scanner_exit == 1, "ncore_attribution_requires_finding_exit")
    api = _notice_api()
    _contract(api)
    context_sha = committed_source(args.source_sha)
    notice, object_id = _repository_notice(args.source_sha, api)
    authorization, report, rows, bindings = _scan_inputs(directory)
    policy = _policy(authorization, report)
    snapshots = _authorization_binding(authorization, report, archive, digest, verification)
    observed_graph, observed_verification = artifact.inspect(archive, digest)
    W.require(observed_verification == verification, "ncore_attribution_graph_changed")
    _report_graph(report, observed_graph, verification, archive.stat().st_size)
    records, issues = _ledger_population(report, rows, policy["policy_sha256"])
    image = _image_notice(archive, observed_graph, notice, api)
    W.require([layer["decoded_bytes"] for layer in report["layers"]] == image["decoded_sizes"],
              "ncore_attribution_decoded_population")
    occurrences = _accepted_occurrences(records, issues, image, api)
    _replay_population(args, directory, bindings)
    proof = _upstream_proof(api, notice, directory)
    receipt = _receipt(args, digest, verification, bindings, policy, proof,
                       context_sha, object_id, occurrences, api)
    _recheck(directory, bindings, snapshots, archive, verification)
    write_json(directory / "attribution.json", receipt)
    _recheck(directory, bindings, snapshots, archive, verification)
    return receipt


def _receipt(args, digest, verification, bindings, policy, proof,
             context_sha, object_id, occurrences, api):
    return {
        "schema": SCHEMA, "accepted": True, "source_sha": args.source_sha,
        "source_context_sha256": context_sha, "repository_path": api.REPOSITORY_PATH,
        "repository_blob": object_id, "image_path": api.IMAGE_PATH,
        "notice_sha256": api.NOTICE_SHA256, "notice_size": api.NOTICE_SIZE,
        "attribution_lines": sorted(api.ATTRIBUTION_LINES),
        "image_digest": digest, "archive_sha256": verification["archive_sha256"],
        "platform_digest": verification["image_manifest_digest"],
        "config_digest": verification["image_config_digest"],
        "authorization_sha256": bindings["authorization_sha256"],
        "report_sha256": bindings["report_sha256"], "records_sha256": bindings["records_sha256"],
        "policy_sha256": policy["policy_sha256"],
        "upstream_proof_sha256": W.sha(W.canonical(proof)), "upstream_archive_count": 2,
        "population_replay": "byte-identical",
        "occurrences": occurrences, "raw_valid": False, "raw_findings": 2,
        "dispositioned_findings": 2, "unresolved_findings": 0,
    }
