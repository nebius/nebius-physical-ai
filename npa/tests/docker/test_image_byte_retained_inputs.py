"""Exercise original OCI identity and explicit transported-authorization boundaries."""

from __future__ import annotations

import copy
import json
from pathlib import Path
import shutil
from types import SimpleNamespace

import pytest

from test_image_byte_adjudication import (
    A,
    W,
    CHECKOUT,
    committed_source_oracle,  # noqa: F401
)
from test_image_byte_scan import FakeDetector, js, write
from test_image_byte_scan_oci import authorize, oci_fixture
from image_byte_scan import oci_verification as O, retained_inputs as R

pytestmark = pytest.mark.usefixtures("committed_source_oracle")


def original_case(directory):
    directory.mkdir(mode=0o700)
    files, verification, _ = oci_fixture(marker="ancestor", revision="a" * 40)
    auth = authorize(directory, files, verification)
    report = O.verify(auth["archive"], auth["expected_image_id"])
    auth["verification_report"] = write(directory / "verification.json", js(report))
    del auth["literal_inventory"]
    auth["confidentiality"] = write(
        directory / "policy.json",
        W.canonical(
            {
                "customer_pattern": "synthetic-private-operator-marker",
                "infra_pattern": None,
            }
        ),
    )
    output = directory / "raw"
    output.mkdir(mode=0o700)
    report = W._scan(auth, output, detector_type=FakeDetector)
    assert report["complete"] and report["findings"] > 0, report
    write(output / "report.json", js(report))
    write(directory / "authorization.json", js(auth))
    write(
        directory / "native-checks.json",
        js(
            {
                "schema_version": "npa.image-byte-native-checks.v1",
                "passed": True,
                "synthetic_only": True,
                "source_bindings": auth["sources"],
                "helper_sha256": auth["helper"]["sha256"],
            }
        ),
    )
    return auth, report


def capture_args(directory):
    return SimpleNamespace(
        authorization=directory / "authorization.json",
        report=directory / "raw/report.json",
        records=directory / "raw/records.jsonl",
        output_dir=directory / "retained",
        run_id="12345-1",
        interface_revision="b" * 40,
        native_checks=directory / "native-checks.json",
    )


def move_retained(root, original, receipt, authorization):
    transported = root / "transported"
    transported.mkdir(mode=0o700)
    shutil.copytree(original / "retained", transported / "retained")
    archive = transported / "image.tar"
    policy = transported / "policy.json"
    shutil.copyfile(authorization["archive"]["path"], archive)
    shutil.copyfile(authorization["confidentiality"]["path"], policy)
    archive.chmod(0o600)
    policy.chmod(0o600)
    shutil.rmtree(original)
    return SimpleNamespace(
        retention=transported / "retained/retention.json",
        retention_sha256=W.sha((transported / "retained/retention.json").read_bytes()),
        archive=archive,
        policy=policy,
        **{
            role: transported / "retained" / receipt["files"][role]
            for role in ("authorization", "report", "records")
        },
    )


def make_review(args, auth, report):
    context = A.context(
        auth,
        report,
        W.sha(args.report.read_bytes()),
        W.sha(args.records.read_bytes()),
        W.sha(args.authorization.read_bytes()),
        "a" * 40,
        A.committed_sources(auth),
    )
    rows = [json.loads(line) for line in args.records.read_text().splitlines()]
    occurrences = A.population(report, rows)
    evidence = write(
        args.retention.parent / "evidence",
        b"synthetic typed producer and consumer proof",
    )
    dispositions, decisions = [], []
    for ordinal, (identity, occurrence) in enumerate(occurrences.items()):
        proof = {
            "schema_version": A.PROOF_SCHEMA,
            "context": context,
            "occurrence_id": identity,
            "record_sha256": occurrence["record_sha256"],
            "record_bytes": occurrence["record_bytes"],
            "semantic_role": "non-operational-source-example",
            "operational_credential": False,
            "provenance_evidence": [evidence],
            "semantic_evidence": [evidence],
        }
        binding = write(args.retention.parent / f"proof-{ordinal}.json", js(proof))
        dispositions.append({"occurrence_id": identity, "proof": binding})
        decisions.append(
            {
                "occurrence_id": identity,
                "proof_sha256": binding["sha256"],
                "decision": "accept",
            }
        )
    manifest = {
        "schema_version": A.MANIFEST_SCHEMA,
        "context": context,
        "dispositions": dispositions,
    }
    bound_manifest = write(args.retention.parent / "manifest.json", js(manifest))
    review = {
        "schema_version": A.REVIEW_SCHEMA,
        "context": context,
        "manifest_sha256": bound_manifest["sha256"],
        "decision": "accept",
        "reviewed_occurrences": decisions,
    }
    bound_review = write(args.retention.parent / "review.json", js(review))
    args.manifest, args.manifest_sha256 = (
        Path(bound_manifest["path"]),
        bound_manifest["sha256"],
    )
    args.review, args.review_sha256 = Path(bound_review["path"]), bound_review["sha256"]
    return manifest, review


@pytest.fixture
def retained_case(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        original = tmp_path / "original"
        auth, report = original_case(original)
        auth_bytes = (original / "authorization.json").read_bytes()
        receipt = R.capture(capture_args(original))
        args = move_retained(tmp_path, original, receipt, auth)
        yield args, auth, report, receipt, auth_bytes


def test_roundtrip_preserves_authorization_and_conserves_all_occurrences(retained_case):
    args, auth, report, receipt, original_bytes = retained_case
    assert args.authorization.read_bytes() == original_bytes
    assert receipt["policy_values_retained"] is False
    assert auth["confidentiality"]["sha256"] not in receipt["files"].values()
    assert not (args.retention.parent / auth["confidentiality"]["sha256"]).exists()
    with R.relocated(args):
        make_review(args, auth, report)
        result = A.verify(args)
    assert result["accepted"] is True
    assert result["raw_scan_valid"] is False
    assert result["accepted_occurrences"] == report["findings"]
    assert result["context"]["retention_sha256"] == args.retention_sha256
    assert result["context"]["expected_image_id"] == auth["expected_image_id"]
    assert (
        result["transport"]["current_input_snapshot_receipts"]
        != receipt["original_snapshot_receipts"]
    )
    assert json.loads(args.report.read_bytes()) == report
    assert W._RETAINED_INPUTS.get() is None


@pytest.mark.parametrize(
    "target", ["retention", "authorization", "report", "records", "archive", "policy"]
)
def test_changed_transported_bytes_refuse_before_acceptance(retained_case, target):
    args, *_ = retained_case
    getattr(args, target).write_bytes(b"changed synthetic input")
    with pytest.raises(W.ScanError), R.relocated(args):
        pytest.fail("changed bytes admitted")
    assert W._RETAINED_INPUTS.get() is None


@pytest.mark.parametrize(
    "field",
    ["scanner_revision", "original_snapshot_receipts", "policy_receipt", "archive"],
)
def test_even_reauthorized_transport_seal_cannot_rewrite_original_claims(
    retained_case, field
):
    args, auth, report, receipt, _ = retained_case
    changed = copy.deepcopy(receipt)
    if field == "scanner_revision":
        changed[field] = "f" * 40
    elif field == "original_snapshot_receipts":
        changed[field][0]["stat"][1] += 1
    elif field == "archive":
        changed[field]["sha256"] = "f" * 64
    else:
        changed[field] = {}
    args.retention.write_bytes(js(changed))
    args.retention_sha256 = W.sha(args.retention.read_bytes())
    with pytest.raises(W.ScanError), R.relocated(args):
        make_review(args, auth, report)
        A.verify(args)


def test_transport_never_restores_previous_inode_identity(retained_case):
    args, auth, report, receipt, _ = retained_case
    with R.relocated(args):
        current = W.input_snapshots(auth)
        originals = R.snapshot_receipts(current)
        assert originals == receipt["original_snapshot_receipts"]
        assert any(
            list(now[-1]) != before["stat"]
            for now, before in zip(current, originals, strict=True)
        )
        make_review(args, auth, report)
        result = A.verify(args)
    assert result["transport"]["original_snapshot_receipts_preserved"]


def test_current_input_mutation_after_verification_refuses(retained_case):
    args, auth, report, *_ = retained_case
    with pytest.raises(W.ScanError), R.relocated(args):
        make_review(args, auth, report)
        A.verify(args)
        args.policy.write_bytes(b"changed current policy")


def test_missing_occurrence_still_refuses_transported_oci(retained_case):
    args, auth, report, *_ = retained_case
    with R.relocated(args):
        manifest, review = make_review(args, auth, report)
        manifest["dispositions"].pop()
        args.manifest.write_bytes(js(manifest))
        args.manifest_sha256 = W.sha(args.manifest.read_bytes())
        review["manifest_sha256"] = args.manifest_sha256
        args.review.write_bytes(js(review))
        args.review_sha256 = W.sha(args.review.read_bytes())
        with pytest.raises(W.ScanError, match="unaccounted_occurrence"):
            A.verify(args)


def test_generic_oci_config_revision_requires_same_original_root(retained_case):
    args, auth, report, *_ = retained_case
    with R.relocated(args):
        verifier = W.bound_json(auth["verification_report"])
        assert A.image_revision(auth, verifier, report) == "a" * 40
        verifier["expected_image_id"] = "sha256:" + "f" * 64
        with pytest.raises(W.ScanError):
            A.image_revision(auth, verifier, report)


def test_hosted_relative_evidence_roundtrip_uses_actual_cli(retained_case):
    args, auth, report, _receipt, original_authorization = retained_case
    with R.relocated(args):
        manifest, review = make_review(args, auth, report)
    evidence_dir = args.retention.parent.parent / "review"
    evidence_dir.mkdir(mode=0o700)
    decisions = []
    for row in manifest["dispositions"]:
        proof = json.loads(Path(row["proof"]["path"]).read_bytes())
        for role in ("provenance_evidence", "semantic_evidence"):
            for binding in proof[role]:
                data = Path(binding["path"]).read_bytes()
                path = evidence_dir / binding["sha256"]
                path.write_bytes(data)
                path.chmod(0o600)
                binding["path"] = "review/" + binding["sha256"]
        body = js(proof)
        digest = W.sha(body)
        write(evidence_dir / digest, body)
        row["proof"] = {"path": "review/" + digest, "sha256": digest}
        decisions.append(
            {
                "occurrence_id": row["occurrence_id"],
                "proof_sha256": digest,
                "decision": "accept",
            }
        )
    manifest_bytes = js(manifest)
    args.manifest_sha256 = W.sha(manifest_bytes)
    review.update(manifest_sha256=args.manifest_sha256, reviewed_occurrences=decisions)
    args.manifest = evidence_dir / "manifest.json"
    args.review = evidence_dir / "review.json"
    write(args.manifest, manifest_bytes)
    args.review_sha256 = write(args.review, js(review))["sha256"]
    args.evidence_root = evidence_dir
    args.analysis_root, args.trusted_root = W._ROOTS.get()
    args.output_dir = evidence_dir.parent / "accepted"
    argv = []
    for name, value in vars(args).items():
        argv.extend(["--" + name.replace("_", "-"), str(value)])
    assert A.main(argv) == 0
    result = json.loads((args.output_dir / "adjudication.json").read_bytes())
    assert result["accepted_occurrences"] == report["findings"]
    assert result["raw_scan_valid"] is False
    assert args.authorization.read_bytes() == original_authorization
