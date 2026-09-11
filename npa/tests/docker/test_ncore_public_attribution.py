"""Exercise the NCore-only public-attribution receipt and hostile boundaries."""

import io
import json
from copy import deepcopy
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))

from image_byte_scan import core as W  # noqa: E402
from ncore_publication import attribution, gates, process  # noqa: E402

REPOSITORY_PATH = "npa/docker/workbench/ncore/notices/cpython/LICENSE.third-party"
IMAGE_PATH = "usr/share/doc/npa-ncore/cpython/LICENSE.third-party"
NOTICE = (ROOT / REPOSITORY_PATH).read_bytes()
NOTICE_SHA = W.sha(NOTICE)
POLICY_SHA = "a" * 64
PHYSICAL_KINDS = {
    "raw_tar_header", "raw_tar_extension", "outer_regular_content",
    "layer_regular_content", "nonzero_tar_padding", "unexplained_tar_trailer",
    "verified_zero_content",
}


def _api():
    return SimpleNamespace(REPOSITORY_PATH=REPOSITORY_PATH, IMAGE_PATH=IMAGE_PATH,
                           ATTRIBUTION_LINES=frozenset({633, 640}),
                           NOTICE_SHA256=NOTICE_SHA, NOTICE_SIZE=len(NOTICE))


def _finding(line, rule="customer-denylist"):
    return {"rule_id": rule, "start_byte": line, "end_byte": line + 1,
            "start_line": line, "end_line": line, "views": ["line", "record"]}


def _confidentiality(ordinal, digest, size, findings):
    return {"type": "confidentiality_record", "record_ordinal": ordinal,
            "policy_sha256": POLICY_SHA, "sha256": digest, "bytes": size,
            "line_count": 1, "composed_findings": len(findings)}


def _record(ordinal, kind, raw, findings, entry, offset):
    return {"type": "record", "record_ordinal": ordinal, "kind": kind,
            "bytes": len(raw), "sha256": W.sha(raw), "findings": findings,
            "scope": "layer", "layer_ordinal": 0, "entry_ordinal": entry,
            "tar_offset": offset}


def _population():
    findings = [_finding(633), _finding(640)]
    path = _record(1, "logical_tar_path", IMAGE_PATH.encode(), [], 7, 1024)
    notice = _record(2, "layer_regular_content", NOTICE, findings, 7, 1536)
    rows = [_confidentiality(1, path["sha256"], path["bytes"], []), path,
            _confidentiality(2, notice["sha256"], notice["bytes"], findings), notice]
    report = {"schema_version": "npa.image-byte-scan.v1", "complete": True,
              "helper_joined": True, "valid": False, "records": 2,
              "scanned_bytes": path["bytes"] + notice["bytes"],
              "regular_files": 1, "regular_bytes": notice["bytes"],
              "verified_zero_bytes": 0, "findings": 2,
              "helper_summary": {"type": "summary", "files": 2,
                                 "bytes": path["bytes"] + notice["bytes"], "findings": 0},
              "outer": {"decoded_bytes": 0},
              "layers": [{"ordinal": 0, "decoded_bytes": notice["bytes"]}]}
    return report, rows


def _refresh_report(report, rows):
    records = [row for row in rows if row["type"] == "record"]
    regular = [row for row in records if row["kind"] == "layer_regular_content"]
    report["records"] = len(records)
    report["scanned_bytes"] = sum(row["bytes"] for row in records)
    report["regular_files"] = len(regular)
    report["regular_bytes"] = sum(row["bytes"] for row in regular)
    report["findings"] = sum(len(row["findings"]) for row in records)
    report["helper_summary"].update(files=len(records), bytes=report["scanned_bytes"])
    report["layers"][0]["decoded_bytes"] = sum(
        row["bytes"] for row in records
        if row["kind"] in PHYSICAL_KINDS and row.get("scope") == "layer"
    )


def test_exact_raw_population_remains_failed_and_two_occurrences_are_dispositioned():
    report, rows = _population()
    records, issues = attribution._ledger_population(report, rows, POLICY_SHA)
    image = {"layer_ordinal": 0, "data_offset": 1536}
    accepted = attribution._accepted_occurrences(records, issues, image, _api())
    assert report["valid"] is False and report["findings"] == 2
    assert len(accepted) == 2 and len(set(accepted)) == 2


def test_compressed_header_is_scanned_but_not_counted_as_decoded_layer_bytes():
    report, rows = _population()
    header = _record(3, "raw_gzip_header", b"gzip-header", [], 8, 2048)
    rows.extend([_confidentiality(3, header["sha256"], header["bytes"], []), header])
    decoded = report["layers"][0]["decoded_bytes"]
    _refresh_report(report, rows)
    report["layers"][0]["decoded_bytes"] = decoded
    attribution._ledger_population(report, rows, POLICY_SHA)


def test_population_counts_only_physical_records_and_not_zero_range_twice():
    report, rows = _population()
    logical = _record(3, "logical_tar_link", b"synthetic-target", [], 8, 2048)
    header = _record(4, "raw_tar_header", bytes(512), [], 8, 2560)
    zeros = _record(5, "verified_zero_content", bytes(1024), [], 9, 3072)
    outer = _record(6, "outer_regular_content", b"encoded-layer", [], 1, 512)
    outer["scope"] = "outer"
    outer.pop("layer_ordinal")
    for record in (logical, header, zeros, outer):
        rows.extend([
            _confidentiality(record["record_ordinal"], record["sha256"], record["bytes"], []),
            record,
        ])
    rows.insert(-2, {"type": "verified_zero_range", "bytes": zeros["bytes"],
                     "sha256": zeros["sha256"], "scope": "layer", "layer_ordinal": 0,
                     "entry_ordinal": 9, "tar_offset": 3072})
    rows.append({"type": "encoded_layer_blob", "bytes": outer["bytes"],
                 "sha256": outer["sha256"], "scope": "outer", "entry_ordinal": 1,
                 "tar_offset": 512})
    _refresh_report(report, rows)
    report["verified_zero_bytes"] = zeros["bytes"]
    report["outer"]["decoded_bytes"] = outer["bytes"]

    records, issues = attribution._ledger_population(report, rows, POLICY_SHA)

    assert len(records) == 6
    assert len(issues) == 2
    assert report["layers"][0]["decoded_bytes"] == len(NOTICE) + 512 + 1024


def test_population_rejects_unknown_record_kind():
    report, rows = _population()
    rows[1]["kind"] = "future_semantic_alias"
    with pytest.raises(ValueError, match="ncore_attribution_record_kind"):
        attribution._ledger_population(report, rows, POLICY_SHA)


def _native_authorization():
    bindings = {
        role: {"path": f"/synthetic/{role}", "sha256": digest}
        for role, digest in W.AHO_PINS.items()
    }
    return {
        "confidentiality": {"synthetic": "binding"},
        "literal_engine": {"kind": "aho-corasick-v1", **bindings},
        "sources": {"npa/scripts/image_byte_scan/aho_matcher.py": bindings["source"]},
    }


def _native_receipt():
    return {"kind": "aho-corasick-v1", "pinned_sha256": W.AHO_PINS,
            "sealed_native_copy": True}


def test_regex_policy_requires_and_authenticates_native_engine(monkeypatch):
    configured = {"customer_pattern": "synthetic-customer", "infra_pattern": None}
    policy = W.C.compile_policy(**configured).receipt()
    monkeypatch.setattr(W, "bound_json", lambda _: configured)
    report = {"confidentiality_policy": policy, "literal_engine": _native_receipt()}

    assert attribution._policy(_native_authorization(), report) == policy


@pytest.mark.parametrize("damage", [
    "missing", "inventory", "pin", "source", "receipt", "regex-receipt",
])
def test_regex_policy_rejects_unbound_or_misreported_native_engine(monkeypatch, damage):
    configured = {"customer_pattern": "synthetic-customer", "infra_pattern": None}
    monkeypatch.setattr(W, "bound_json", lambda _: configured)
    authorization = _native_authorization()
    report = {"confidentiality_policy": W.C.compile_policy(**configured).receipt(),
              "literal_engine": _native_receipt()}
    if damage == "missing":
        authorization.pop("literal_engine")
    elif damage == "inventory":
        authorization["literal_inventory"] = {"synthetic": "binding"}
    elif damage == "pin":
        authorization["literal_engine"]["extension"]["sha256"] = "0" * 64
    elif damage == "source":
        authorization["sources"]["npa/scripts/image_byte_scan/aho_matcher.py"] = {
            "path": "/synthetic/other", "sha256": W.AHO_PINS["source"],
        }
    elif damage == "receipt":
        report["literal_engine"]["sealed_native_copy"] = False
    else:
        report["literal_engine"] = {"kind": "regex-reference-v1"}
    with pytest.raises(ValueError):
        attribution._policy(authorization, report)


def _oci_report_graph():
    config_digest = "sha256:" + "b" * 64
    manifest_digest = "sha256:" + "c" * 64
    layer_digest = "sha256:" + "d" * 64
    descriptor = {"mediaType": "application/vnd.oci.image.layer.v1.tar",
                  "digest": layer_digest, "size": 2048}
    graph = {
        "receipt": {"schema_version": "npa.ncore.oci-graph.v1", "blobs": []},
        "layers": [{"ordinal": 0, "name": "blobs/sha256/" + "d" * 64,
                    "size": 2048, "diff_id": "sha256:" + "e" * 64,
                    "descriptor": descriptor}],
    }
    verification = {"archive_sha256": "f" * 64, "expected_image_id": "sha256:" + "1" * 64,
                    "image_config_digest": config_digest,
                    "image_manifest_digest": manifest_digest,
                    "regular_files_read": 1, "content_bytes_read": len(NOTICE)}
    report = {
        "schema_version": "npa.image-byte-scan.v1", "valid": False, "complete": True,
        "authorization_sha256": "2" * 64, "archive_sha256": verification["archive_sha256"],
        "image_config_digest": config_digest, "image_manifest_digest": manifest_digest,
        "expected_image_id": verification["expected_image_id"],
        "private_literals_configured": False, "private_literal_count": 0,
        "literal_matching_policy": "exact-substring-v1",
        "layers": [{"ordinal": 0, "diff_id": graph["layers"][0]["diff_id"],
                    "compressed_sha256": layer_digest.removeprefix("sha256:"),
                    "compressed_bytes": 2048, "codec": "raw", "headers": 1,
                    "decoded_bytes": 2048, "zero_end_blocks": 2}],
        "oci_graph": graph["receipt"],
        "outer": {"headers": 4, "decoded_bytes": 4096, "zero_end_blocks": 2},
        "confidentiality_policy": {"mode": "regex-v1"},
        "helper_summary": {"type": "summary", "files": 1, "bytes": len(NOTICE),
                           "findings": 0},
        "literal_engine": _native_receipt(), "input_snapshot_receipts": [],
        "helper_joined": True, "records": 1, "scanned_bytes": len(NOTICE),
        "verified_zero_bytes": 0, "regular_files": 1, "regular_bytes": len(NOTICE),
        "findings": 2,
    }
    return report, graph, verification


def test_native_oci_report_is_accepted_before_population_replay():
    report, graph, verification = _oci_report_graph()

    attribution._report_graph(report, graph, verification, 4096)


@pytest.mark.parametrize("damage", ["regex-engine", "changed-pin", "private", "private-count"])
def test_native_oci_report_rejects_engine_or_private_literal_drift(damage):
    report, graph, verification = _oci_report_graph()
    report = deepcopy(report)
    if damage == "regex-engine":
        report["literal_engine"] = {"kind": "regex-reference-v1"}
    elif damage == "changed-pin":
        report["literal_engine"]["pinned_sha256"] = dict(W.AHO_PINS, source="0" * 64)
    elif damage == "private":
        report["private_literals_configured"] = True
    else:
        report["private_literal_count"] = 1

    with pytest.raises(ValueError, match="ncore_attribution_report_artifact_binding"):
        attribution._report_graph(report, graph, verification, 4096)


@pytest.mark.parametrize("damage", [
    "other-line", "infra", "native", "structural", "literal", "copy", "path-copy",
    "incomplete", "joined", "raw-valid", "helper-finding", "extra-finding",
])
def test_population_rejects_every_non_attribution_or_incomplete_finding(damage):
    report, rows = _population()
    notice = rows[3]
    if damage == "other-line":
        notice["findings"][0].update(start_line=632, end_line=632)
    elif damage == "infra":
        notice["findings"][0]["rule_id"] = "infra-denylist"
    elif damage == "native":
        notice["findings"][0] = {"rule_id": "credential", "start_line": 633, "end_line": 633}
    elif damage in {"structural", "literal"}:
        rule = "private_literal" if damage == "literal" else "nonzero_tar_padding"
        rows.append({"type": "finding", "rule_id": rule, "record_ordinal": 2,
                     "scope": "layer", "layer_ordinal": 0, "entry_ordinal": 7,
                     "tar_offset": 2048})
        report["findings"] += 1
    elif damage in {"copy", "path-copy"}:
        raw = NOTICE if damage == "copy" else IMAGE_PATH.encode()
        kind = "layer_regular_content" if damage == "copy" else "logical_tar_path"
        duplicate = _record(3, kind, raw, [], 8, 2048)
        rows.extend([_confidentiality(3, duplicate["sha256"], duplicate["bytes"], []), duplicate])
        _refresh_report(report, rows)
    elif damage == "incomplete":
        report["complete"] = False
    elif damage == "joined":
        report["helper_joined"] = False
    elif damage == "raw-valid":
        report["valid"] = True
    elif damage == "helper-finding":
        report["helper_summary"]["findings"] = 1
    else:
        notice["findings"].append(_finding(633))
        rows[2]["composed_findings"] += 1
        _refresh_report(report, rows)
    with pytest.raises(ValueError):
        records, issues = attribution._ledger_population(report, rows, POLICY_SHA)
        image = {"layer_ordinal": 0, "data_offset": 1536}
        attribution._accepted_occurrences(records, issues, image, _api())


def test_two_findings_must_cover_both_exact_attribution_lines():
    report, rows = _population()
    rows[3]["findings"][1].update(start_line=633, end_line=633, start_byte=634, end_byte=635)
    records, issues = attribution._ledger_population(report, rows, POLICY_SHA)
    image = {"layer_ordinal": 0, "data_offset": 1536}
    with pytest.raises(ValueError, match="ncore_attribution_finding_population"):
        attribution._accepted_occurrences(records, issues, image, _api())


def _tar(entries):
    output = io.BytesIO()
    with tarfile.open(fileobj=output, mode="w") as archive:
        for name, kind, raw in entries:
            member = tarfile.TarInfo(name)
            member.mode = 0o644
            if kind == "file":
                member.size = len(raw)
                archive.addfile(member, io.BytesIO(raw))
            elif kind == "symlink":
                member.type = tarfile.SYMTYPE
                member.linkname = raw
                archive.addfile(member)
            else:
                member.type = tarfile.LNKTYPE
                member.linkname = raw
                archive.addfile(member)
    return output.getvalue()


def _image(tmp_path, entries):
    layer = _tar(entries)
    outer = _tar([("layer.tar", "file", layer)])
    path = tmp_path / "image.tar"
    path.write_bytes(outer)
    graph = {"layers": [{"ordinal": 0, "name": "layer.tar", "size": len(layer),
                         "diff_id": "sha256:" + W.sha(layer),
                         "descriptor": {"mediaType": "application/vnd.oci.image.layer.v1.tar",
                                        "digest": "sha256:" + W.sha(layer), "size": len(layer)}}]}
    return path, graph


@pytest.mark.parametrize("damage", [None, "changed", "copy", "symlink", "hardlink", "ancestor"])
def test_original_scratch_layer_requires_one_regular_canonical_notice(tmp_path, damage):
    raw = NOTICE + b"changed" if damage == "changed" else NOTICE
    entries = [(IMAGE_PATH, "file", raw)]
    if damage == "copy":
        entries.append(("copy", "file", NOTICE))
    elif damage == "symlink":
        entries.append(("alias", "symlink", IMAGE_PATH))
    elif damage == "hardlink":
        entries.append(("alias", "hardlink", IMAGE_PATH))
    elif damage == "ancestor":
        entries.insert(0, ("usr/share/doc", "symlink", "elsewhere"))
    image, graph = _image(tmp_path, entries)
    if damage is None:
        observed = attribution._image_notice(image, graph, NOTICE, _api())
        assert observed["layer_ordinal"] == 0
    else:
        with pytest.raises(ValueError):
            attribution._image_notice(image, graph, NOTICE, _api())


def test_repository_notice_is_one_regular_git_blob_at_the_canonical_path():
    source_sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    notice, object_id = attribution._repository_notice(source_sha, _api())
    assert notice == NOTICE and len(object_id) == 40


def test_byte_gate_keeps_clean_pass_and_routes_only_exit_one_to_attribution(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(gates, "run", lambda *_: None)
    monkeypatch.setattr(gates, "run_byte_scanner", lambda *_: 0)
    monkeypatch.setattr(attribution, "verify", lambda *_: pytest.fail("clean scan adjudicated"))
    args = SimpleNamespace(analysis_root=tmp_path, policy_mode="ci-regex")
    clean = tmp_path / "clean"
    (clean / "bytes").mkdir(parents=True)
    (clean / "bytes/report.json").write_text(
        json.dumps({"complete": True, "valid": True, "helper_joined": True})
    )
    gates.byte_scan(args, clean, tmp_path / "image", "synthetic", {})
    monkeypatch.setattr(gates, "run_byte_scanner", lambda *_: 1)
    calls = []
    monkeypatch.setattr(attribution, "verify", lambda *values: calls.append(values[-1]))
    findings = tmp_path / "findings"
    (findings / "bytes").mkdir(parents=True)
    (findings / "bytes/report.json").write_text(
        json.dumps({"complete": True, "valid": False, "helper_joined": True})
    )
    gates.byte_scan(args, findings, tmp_path / "image", "synthetic", {})
    assert calls == [1]


@pytest.mark.parametrize("status", [0, 1, 2, -1])
def test_raw_scanner_runner_accepts_only_documented_exit_semantics(tmp_path, monkeypatch, status):
    monkeypatch.setattr(process.subprocess, "run",
                        lambda argv, **_: subprocess.CompletedProcess(argv, status))
    output = tmp_path / "scanner.log"
    if status in {0, 1}:
        assert process.run_byte_scanner([str(process.PYTHON)], output) == status
    else:
        with pytest.raises(ValueError, match="image_byte_scanner_unexpected_exit"):
            process.run_byte_scanner([str(process.PYTHON)], output)


@pytest.mark.parametrize("damage", [None, "status", "report", "ledger"])
def test_population_replay_must_be_byte_identical(tmp_path, monkeypatch, damage):
    original = tmp_path / "original"
    (original / "bytes").mkdir(parents=True)
    report = b'{"valid":false}\n'
    records = b'{"type":"record"}\n'
    (original / "bytes/report.json").write_bytes(report)
    (original / "bytes/records.jsonl").write_bytes(records)
    (original / "bytes/report.json").chmod(0o600)
    (original / "bytes/records.jsonl").chmod(0o600)
    bindings = {"report_sha256": W.sha(report), "records_sha256": W.sha(records)}

    def replay(_argv, _output):
        target = original / "attribution-replay"
        target.mkdir()
        target.joinpath("report.json").write_bytes(report + (b"changed" if damage == "report" else b""))
        target.joinpath("records.jsonl").write_bytes(records + (b"changed" if damage == "ledger" else b""))
        target.joinpath("report.json").chmod(0o600)
        target.joinpath("records.jsonl").chmod(0o600)
        return 0 if damage == "status" else 1

    monkeypatch.setattr(attribution, "run_byte_scanner", replay)
    args = SimpleNamespace(analysis_root=tmp_path)
    with W.authorized_roots(tmp_path, ROOT):
        if damage is None:
            attribution._replay_population(args, original, bindings)
        else:
            with pytest.raises(ValueError):
                attribution._replay_population(args, original, bindings)


def test_receipt_exposes_bindings_and_counts_without_proof_or_finding_material():
    bindings = {"authorization_sha256": "b" * 64, "report_sha256": "c" * 64,
                "records_sha256": "d" * 64}
    verification = {"archive_sha256": "e" * 64, "image_manifest_digest": "sha256:" + "f" * 64,
                    "image_config_digest": "sha256:" + "1" * 64}
    hostile = {"private_path": "synthetic-private-material"}
    receipt = attribution._receipt(SimpleNamespace(source_sha="2" * 40), "sha256:" + "3" * 64,
                                   verification, bindings, {"policy_sha256": "4" * 64}, hostile,
                                   "5" * 64, "6" * 40, ["7" * 64, "8" * 64], _api())
    serialized = json.dumps(receipt)
    assert "synthetic-private-material" not in serialized
    assert receipt["raw_valid"] is False and receipt["unresolved_findings"] == 0


@pytest.mark.parametrize("damage", [None, "receipt", "report", "ledger", "archive"])
def test_late_evidence_mutation_is_rejected_before_transfer(tmp_path, damage):
    directory = tmp_path / "gate"
    (directory / "bytes").mkdir(parents=True)
    report, records, archive = b"report", b"records", b"archive"
    for path, raw in ((directory / "bytes/report.json", report),
                      (directory / "bytes/records.jsonl", records),
                      (tmp_path / "image.tar", archive)):
        path.write_bytes(raw)
        path.chmod(0o600)
    expected = {"report_sha256": W.sha(report), "records_sha256": W.sha(records)}
    (directory / "attribution.json").write_text(json.dumps(expected))
    (directory / "attribution.json").chmod(0o600)
    target = {"receipt": directory / "attribution.json", "report": directory / "bytes/report.json",
              "ledger": directory / "bytes/records.jsonl", "archive": tmp_path / "image.tar"}.get(damage)
    if target is not None:
        target.write_bytes(target.read_bytes() + b"changed")
    verification = {"archive_sha256": W.sha(archive)}
    with W.authorized_roots(tmp_path, ROOT):
        if damage is None:
            attribution.recheck(directory, expected, tmp_path / "image.tar", verification)
        else:
            with pytest.raises((ValueError, json.JSONDecodeError)):
                attribution.recheck(directory, expected, tmp_path / "image.tar", verification)
