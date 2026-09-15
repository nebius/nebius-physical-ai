"""Occurrence conservation and explicit review authorization, without network."""
from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
import sys
import tarfile
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHECKOUT / "npa/scripts"))
from image_byte_scan import adjudicate as A  # noqa: E402

W = A.W


@pytest.fixture
def committed_source_oracle(tmp_path, monkeypatch):
    """Hermetic Git-object oracle for the exact captured scanner source bytes.

    The production command rejects an uncommitted scanner. Tests must also run
    before a contributor commits, so emulate the immutable Git object store
    separately from the real live-file reads being tested. No branch is created.
    """
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        contents = {name: Path(binding["path"]).read_bytes()
                    for name, binding in W.source_bindings().items()}
    original = A.subprocess.check_output
    head = original(["git", "-C", str(CHECKOUT), "rev-parse", "HEAD"], text=True).strip()

    def objects(argv, **kwargs):
        if argv[:5] == ["git", "-C", str(CHECKOUT), "cat-file", "blob"]:
            commit, name = argv[5].split(":", 1)
            if commit != head or name not in contents:
                raise subprocess.CalledProcessError(128, argv)
            return contents[name]
        return original(argv, **kwargs)

    monkeypatch.setattr(A.subprocess, "check_output", objects)


def sha(data):
    return hashlib.sha256(data).hexdigest()


def record(ordinal, body, findings, kind="layer_regular_content"):
    return {"type": "record", "record_ordinal": ordinal, "bytes": len(body),
            "sha256": sha(body), "findings": findings, "kind": kind,
            "scope": "layer", "layer_ordinal": 0, "tar_offset": ordinal * 512}


def sample():
    same = {"rule_id": "generic-api-key", "start_line": 1, "end_line": 1}
    rows = [record(1, b"header", [same], "raw_tar_header"),
            {"type": "finding", "record_ordinal": 2, "rule_id": "private_literal",
             "literal_index": 0, "literal_sha256": sha(b"abc"), "byte_start": 0,
             "byte_end": 3, "scope": "layer", "layer_ordinal": 0, "tar_offset": 1024},
            record(2, b"abc body", [same, copy.deepcopy(same)]),
            record(3, b"", [])]
    report = {"schema_version": "npa.image-byte-scan.v1", "complete": True,
              "helper_joined": True, "valid": False, "records": 3,
              "scanned_bytes": 14, "verified_zero_bytes": 0,
              "regular_files": 2, "regular_bytes": 8, "findings": 4,
              "helper_summary": {"type": "summary", "files": 3, "bytes": 14, "findings": 3}}
    return report, rows


def reviewed(tmp_path):
    report, rows = sample()
    population = A.population(report, rows)
    ctx = {"archive_sha256": "a" * 64, "source_sha256": "b" * 64,
           "policy_sha256": "c" * 64}
    files = {}

    def write(name, data):
        path = tmp_path / name
        path.write_bytes(data)
        path.chmod(0o600)
        binding = {"path": str(path), "sha256": sha(data)}
        files[name] = binding
        return binding

    provenance = write("provenance.bin", b"synthetic immutable upstream package evidence")
    semantics = write("semantic-review.txt", b"synthetic reviewed non-operational example")
    manifest = {"schema_version": A.MANIFEST_SCHEMA, "context": ctx, "dispositions": []}
    decisions = []
    proofs = {}
    for index, (key, occurrence) in enumerate(population.items()):
        proof = {"schema_version": A.PROOF_SCHEMA, "context": ctx, "occurrence_id": key,
                 "record_sha256": occurrence["record_sha256"], "record_bytes": occurrence["record_bytes"],
                 "semantic_role": "non-operational-source-example", "operational_credential": False,
                 "provenance_evidence": [provenance], "semantic_evidence": [semantics]}
        binding = write(f"proof-{index}.json", W.canonical(proof))
        proofs[key] = proof
        manifest["dispositions"].append({"occurrence_id": key, "proof": binding})
        decisions.append({"occurrence_id": key, "proof_sha256": binding["sha256"], "decision": "accept"})
    manifest_hash = sha(W.canonical(manifest))
    review = {"schema_version": A.REVIEW_SCHEMA, "decision": "accept",
              "manifest_sha256": manifest_hash, "context": ctx, "reviewed_occurrences": decisions}

    def loader(binding, json_body=True):
        data = Path(binding["path"]).read_bytes()
        W.require(sha(data) == binding["sha256"], "test_evidence_changed")
        return A.decode(data) if json_body else data

    return manifest, review, population, ctx, manifest_hash, loader, proofs, files


def check(case):
    return A.dispositions(*case[:6])


def test_conserves_identical_native_findings_and_literal_occurrences(tmp_path):
    report, rows = sample()
    raw_before = copy.deepcopy(report)
    found = A.population(report, rows)
    assert len(found) == 4
    assert len({key for key, value in found.items() if value["record_ordinal"] == 2}) == 3
    assert check(reviewed(tmp_path)) == 4
    assert report == raw_before and report["valid"] is False


@pytest.mark.parametrize("key,value", [("complete", False), ("complete", 1),
    ("helper_joined", False), ("helper_joined", 1), ("failure_code", "input_binding_changed"),
    ("failure_code", None), ("valid", True), ("valid", 0)])
def test_refuses_incomplete_or_changed_raw_verdict(key, value):
    report, rows = sample()
    report[key] = value
    with pytest.raises(W.ScanError):
        A.population(report, rows)


@pytest.mark.parametrize("field", ["records", "scanned_bytes", "verified_zero_bytes",
    "regular_files", "regular_bytes", "findings"])
@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None])
def test_report_counts_are_strict_integers(field, value):
    report, rows = sample()
    report[field] = value
    with pytest.raises(W.ScanError):
        A.population(report, rows)


@pytest.mark.parametrize("field", ["files", "bytes", "findings"])
@pytest.mark.parametrize("value", [True, False, -1, 1.0, "1", None])
def test_native_counts_are_strict_integers(field, value):
    report, rows = sample()
    report["helper_summary"][field] = value
    with pytest.raises(W.ScanError):
        A.population(report, rows)


@pytest.mark.parametrize("mutation", ["missing_record", "duplicate_record", "new_finding",
    "missing_finding", "new_record", "reordered", "orphan", "byte_count", "record_bool",
    "literal_range", "literal_unknown_field", "structural", "unknown_row", "zero_unaccounted"])
def test_population_refuses_corruption(mutation):
    report, rows = sample()
    if mutation == "missing_record":
        rows.pop()
    elif mutation == "duplicate_record":
        rows.append(copy.deepcopy(rows[-1]))
    elif mutation == "new_finding":
        rows[2]["findings"].append(copy.deepcopy(rows[2]["findings"][0]))
    elif mutation == "missing_finding":
        rows[2]["findings"].pop()
    elif mutation == "new_record":
        rows.append(record(4, b"new", []))
    elif mutation == "reordered":
        rows[0], rows[-1] = rows[-1], rows[0]
    elif mutation == "orphan":
        rows[1]["record_ordinal"] = 99
    elif mutation == "byte_count":
        rows[2]["bytes"] += 1
    elif mutation == "record_bool":
        rows[0]["record_ordinal"] = True
    elif mutation == "literal_range":
        rows[1]["byte_end"] = 999
    elif mutation == "literal_unknown_field":
        rows[1]["secret"] = "not-permitted"
    elif mutation == "structural":
        rows[1] = {"type": "finding", "rule_id": "nonzero_tar_padding"}
    elif mutation == "unknown_row":
        rows.append({"type": "ignored_override"})
    else:
        rows.append({"type": "verified_zero_range", "bytes": 1, "sha256": sha(b"\0")})
    with pytest.raises(W.ScanError):
        A.population(report, rows)


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unknown", "review_missing",
    "review_duplicate", "review_unknown", "review_denied", "overall_denied",
    "manifest_changed", "archive_changed", "policy_changed", "source_changed"])
def test_dispositions_need_every_exact_independent_review(tmp_path, mutation):
    case = reviewed(tmp_path)
    manifest, review = case[:2]
    if mutation == "missing":
        manifest["dispositions"].pop()
    elif mutation == "duplicate":
        manifest["dispositions"].append(copy.deepcopy(manifest["dispositions"][0]))
    elif mutation == "unknown":
        manifest["dispositions"][0]["occurrence_id"] = "d" * 64
    elif mutation == "review_missing":
        review["reviewed_occurrences"].pop()
    elif mutation == "review_duplicate":
        review["reviewed_occurrences"].append(copy.deepcopy(review["reviewed_occurrences"][0]))
    elif mutation == "review_unknown":
        review["reviewed_occurrences"][0]["occurrence_id"] = "d" * 64
    elif mutation == "review_denied":
        review["reviewed_occurrences"][0]["decision"] = "unreviewed"
    elif mutation == "overall_denied":
        review["decision"] = "deny"
    elif mutation == "manifest_changed":
        review["manifest_sha256"] = "d" * 64
    else:
        manifest["context"] = copy.deepcopy(manifest["context"])
        manifest["context"][mutation.removesuffix("_changed") + "_sha256"] = "d" * 64
    with pytest.raises(W.ScanError):
        check(case)


@pytest.mark.parametrize("mutation", ["live_credential", "missing_provenance", "missing_semantics",
    "unrecognized_role", "wrong_record", "wrong_occurrence", "wrong_context", "bool_bytes",
    "stale_proof", "stale_provenance", "stale_semantics"])
def test_independent_acceptance_never_waives_changed_or_unsafe_evidence(tmp_path, mutation):
    case = reviewed(tmp_path)
    manifest, review, _population, _ctx, _digest, _loader, proofs, files = case
    first = manifest["dispositions"][0]
    proof = proofs[first["occurrence_id"]]
    if mutation == "live_credential":
        proof["operational_credential"] = True
    elif mutation == "missing_provenance":
        proof["provenance_evidence"] = []
    elif mutation == "missing_semantics":
        proof["semantic_evidence"] = []
    elif mutation == "unrecognized_role":
        proof["semantic_role"] = "anything-from-public-package"
    elif mutation == "wrong_record":
        proof["record_sha256"] = "d" * 64
    elif mutation == "wrong_occurrence":
        proof["occurrence_id"] = "d" * 64
    elif mutation == "wrong_context":
        proof["context"] = {"archive_sha256": "d" * 64}
    elif mutation == "bool_bytes":
        proof["record_bytes"] = True
    elif mutation == "stale_provenance":
        Path(files["provenance.bin"]["path"]).write_bytes(b"changed archive")
    elif mutation == "stale_semantics":
        Path(files["semantic-review.txt"]["path"]).write_bytes(b"changed review")
    else:
        Path(first["proof"]["path"]).write_bytes(b"changed proof")
    if not mutation.startswith("stale_"):
        payload = W.canonical(proof)
        Path(first["proof"]["path"]).write_bytes(payload)
        first["proof"]["sha256"] = sha(payload)
        review["reviewed_occurrences"][0]["proof_sha256"] = sha(payload)
    with pytest.raises(W.ScanError):
        check(case)


def test_duplicate_json_keys_refuse_even_if_later_claim_is_accept():
    with pytest.raises(W.ScanError, match="duplicate_json_key"):
        A.decode(b'{"decision":"deny","decision":"accept"}')


def test_exact_authorization_is_required_before_reading_untrusted_manifest(tmp_path):
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps({"decision": "accept"}))
    path.chmod(0o600)
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT), pytest.raises(W.ScanError):
        A.pinned_json(path, "0" * 64)


def complete_case(tmp_path):
    """Generate real archive/ledger bindings with the existing protocol oracle."""
    # This is the existing hermetic framing detector, never a security claim.
    sys.path.insert(0, str(CHECKOUT / "npa/tests/docker"))
    from test_image_byte_scan import FakeDetector, file, fixture, js, tar_data, write

    auth = fixture(tmp_path, entries=[file("opt/sample", b"neutral private-operator-marker body")], repeat=2)
    archive = Path(auth["archive"]["path"])
    with tarfile.open(archive) as saved:
        payloads = {row.name: saved.extractfile(row).read() for row in saved if row.isfile()}
    saved = json.loads(payloads["manifest.json"])
    old_config_name = saved[0]["Config"]
    config = json.loads(payloads.pop(old_config_name))
    config["config"] = {"Labels": {"org.opencontainers.image.revision": "a" * 40}}
    config_bytes = js(config)
    config_name = "blobs/sha256/" + sha(config_bytes)
    payloads[config_name] = config_bytes
    saved[0]["Config"] = config_name
    payloads["manifest.json"] = js(saved)
    index = json.loads(payloads["index.json"])
    old_manifest_name = "blobs/sha256/" + index["manifests"][0]["digest"][7:]
    manifest = json.loads(payloads.pop(old_manifest_name))
    manifest["config"].update(digest="sha256:" + sha(config_bytes), size=len(config_bytes))
    manifest_bytes = js(manifest)
    image_id = "sha256:" + sha(manifest_bytes)
    payloads["blobs/sha256/" + image_id[7:]] = manifest_bytes
    index["manifests"][0].update(digest=image_id, size=len(manifest_bytes))
    payloads["index.json"] = js(index)
    auth["archive"] = write(archive, tar_data([file(name, body) for name, body in payloads.items()]))
    verification = json.loads(Path(auth["verification_report"]["path"]).read_bytes())
    verification.update(expected_image_id=image_id, image_config_digest="sha256:" + sha(config_bytes),
                        image_manifest_digest=image_id, docker_save_sha256=auth["archive"]["sha256"])
    auth["verification_report"] = write(Path(auth["verification_report"]["path"]), js(verification))
    auth["expected_image_id"] = image_id
    output = tmp_path / "raw"
    output.mkdir(mode=0o700)
    report = W._scan(auth, output, detector_type=FakeDetector)
    assert report["complete"] and not report["valid"] and report["findings"] > 0
    report_spec = write(output / "report.json", js(report))
    auth_spec = write(tmp_path / "authorization.json", js(auth))
    records = output / "records.jsonl"
    ctx = A.context(auth, report, report_spec["sha256"], sha(records.read_bytes()),
                    auth_spec["sha256"], "a" * 40,
                    subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
    occurrences = A.population(report, [json.loads(row) for row in records.read_text().splitlines()])
    evidence = write(tmp_path / "source.evidence", b"inert synthetic source evidence")
    semantic = write(tmp_path / "semantic.evidence", b"inert synthetic semantic review")
    dispositions, decisions = [], []
    for ordinal, (key, occurrence) in enumerate(occurrences.items()):
        proof = {"schema_version": A.PROOF_SCHEMA, "context": ctx, "occurrence_id": key,
                 "record_sha256": occurrence["record_sha256"], "record_bytes": occurrence["record_bytes"],
                 "semantic_role": "non-operational-source-example", "operational_credential": False,
                 "provenance_evidence": [evidence], "semantic_evidence": [semantic]}
        proof_spec = write(tmp_path / f"proof-{ordinal}.json", js(proof))
        dispositions.append({"occurrence_id": key, "proof": proof_spec})
        decisions.append({"occurrence_id": key, "proof_sha256": proof_spec["sha256"], "decision": "accept"})
    manifest_spec = write(tmp_path / "manifest.json", js({"schema_version": A.MANIFEST_SCHEMA,
                          "context": ctx, "dispositions": dispositions}))
    review_spec = write(tmp_path / "review.json", js({"schema_version": A.REVIEW_SCHEMA,
                        "decision": "accept", "manifest_sha256": manifest_spec["sha256"],
                        "context": ctx, "reviewed_occurrences": decisions}))
    return SimpleNamespace(manifest=Path(manifest_spec["path"]), manifest_sha256=manifest_spec["sha256"],
                           review=Path(review_spec["path"]), review_sha256=review_spec["sha256"],
                           authorization=Path(auth_spec["path"]), report=Path(report_spec["path"]),
                           records=records, output_dir=tmp_path / "accepted"), auth, report


def test_real_archive_binding_preserves_raw_failure_and_separates_source_revisions(tmp_path, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _auth, report = complete_case(tmp_path)
        result = A.verify(args)
        assert result["accepted"] is True and result["raw_scan_valid"] is False
        assert result["accepted_occurrences"] == report["findings"]
        assert json.loads(args.report.read_bytes()) == report
        assert result["context"]["image_source_sha"] == "a" * 40
        assert result["context"]["scanner_source_sha"] != "a" * 40


@pytest.mark.parametrize("target", ["archive", "authorization", "report", "records", "manifest", "review",
                                     "verification", "source_evidence", "semantic_evidence", "policy"])
def test_real_bound_input_changes_refuse_before_acceptance_output(tmp_path, target, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, auth, _report = complete_case(tmp_path)
        paths = {"archive": Path(auth["archive"]["path"]), "verification": Path(auth["verification_report"]["path"]),
                 "source_evidence": tmp_path / "source.evidence", "semantic_evidence": tmp_path / "semantic.evidence",
                 "policy": Path(auth["literal_inventory"]["path"])}
        path = paths[target] if target in paths else getattr(args, target)
        with path.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises(W.ScanError):
            A.verify(args)
        assert not args.output_dir.exists()


def test_shared_evidence_is_hashed_once_then_rechecked_once(tmp_path, monkeypatch, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _auth, _report = complete_case(tmp_path)
        original = W.bound_open
        calls = []

        @contextmanager
        def observed(binding, **kwargs):
            calls.append(binding["path"])
            with original(binding, **kwargs) as opened:
                yield opened

        monkeypatch.setattr(W, "bound_open", observed)
        A.verify(args)
        assert calls.count(str(tmp_path / "source.evidence")) == 2
        assert calls.count(str(tmp_path / "semantic.evidence")) == 2


def test_private_evidence_symlink_is_rejected(tmp_path, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _auth, _report = complete_case(tmp_path)
        path = tmp_path / "source.evidence"
        retained = tmp_path / "retained.evidence"
        os.rename(path, retained)
        path.symlink_to(retained)
        with pytest.raises(W.ScanError):
            A.verify(args)


@pytest.mark.parametrize("target", ["report", "records", "authorization", "manifest", "review"])
def test_binding_changed_after_initial_read_still_refuses(tmp_path, monkeypatch, target, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _auth, _report = complete_case(tmp_path)
        original = A.bound_bytes
        changed = False

        def racing(binding):
            nonlocal changed
            data = original(binding)
            if Path(binding["path"]).name == "proof-0.json" and not changed:
                changed = True
                with getattr(args, target).open("ab") as stream:
                    stream.write(b"changed after initial bound read")
            return data

        monkeypatch.setattr(A, "bound_bytes", racing)
        with pytest.raises(W.ScanError):
            A.verify(args)
        assert changed


def test_evidence_changed_after_first_hash_is_rechecked(tmp_path, monkeypatch, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _auth, _report = complete_case(tmp_path)
        original = W.bound_open
        changed = False

        @contextmanager
        def racing(binding, **kwargs):
            nonlocal changed
            with original(binding, **kwargs) as opened:
                yield opened
            if Path(binding["path"]).name == "source.evidence" and not changed:
                changed = True
                Path(binding["path"]).write_bytes(b"changed after verified hash")

        monkeypatch.setattr(W, "bound_open", racing)
        with pytest.raises(W.ScanError):
            A.verify(args)
        assert changed


def test_scanner_head_movement_during_adjudication_refuses(tmp_path, monkeypatch, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _auth, _report = complete_case(tmp_path)
        original = A.committed_sources
        checks = 0

        def moving(authorization):
            nonlocal checks
            checks += 1
            actual = original(authorization)
            return actual if checks == 1 else "0" * 40

        monkeypatch.setattr(A, "committed_sources", moving)
        with pytest.raises(W.ScanError, match="scanner_head_changed"):
            A.verify(args)
        assert checks == 2


def test_uncommitted_scanner_source_is_not_claimed_as_commit(tmp_path, monkeypatch, committed_source_oracle):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _auth, _report = complete_case(tmp_path)
        original = A.subprocess.check_output

        def changed_git_blob(argv, **kwargs):
            data = original(argv, **kwargs)
            return data + b"different committed source" if "cat-file" in argv else data

        monkeypatch.setattr(A.subprocess, "check_output", changed_git_blob)
        with pytest.raises(W.ScanError, match="uncommitted_source"):
            A.verify(args)
