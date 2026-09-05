"""Exercise authorization pins and sanitized failures through the real CLI."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

CHECKOUT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(CHECKOUT / "npa/tests/docker"))
from test_image_byte_adjudication import (  # noqa: E402
    A,
    W,
    committed_source_oracle,  # noqa: F401
    complete_case,
    reviewed,
)


def cli_args(tmp_path, args):
    values = ["--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT)]
    for name in (
        "manifest",
        "review",
        "authorization",
        "report",
        "records",
        "output_dir",
        "manifest_sha256",
        "review_sha256",
    ):
        values.extend(["--" + name.replace("_", "-"), str(getattr(args, name))])
    return values


@pytest.mark.usefixtures("committed_source_oracle")
def test_main_preserves_raw_failure_and_emits_separate_private_receipt(
    tmp_path, capsys
):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _, report = complete_case(tmp_path)
    assert A.main(cli_args(tmp_path, args)) == 0
    assert capsys.readouterr().out == "image byte findings independently accepted\n"
    output = args.output_dir / "adjudication.json"
    assert output.stat().st_mode & 0o077 == 0
    result = json.loads(output.read_bytes())
    assert result["accepted"] is True and result["raw_scan_valid"] is False
    assert json.loads(args.report.read_bytes()) == report


@pytest.mark.parametrize(
    "target",
    ["manifest", "review", "authorization", "report", "records", "source.evidence"],
)
def test_main_invalid_evidence_emits_only_fixed_failure(tmp_path, capsys, target):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, _, _ = complete_case(tmp_path)
    path = tmp_path / target if target == "source.evidence" else getattr(args, target)
    path.write_bytes(b"unreviewed synthetic private marker must not appear in output")
    assert A.main(cli_args(tmp_path, args)) == 1
    captured = capsys.readouterr()
    assert captured.out == "image byte adjudication failed\n" and captured.err == ""
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    "mutation", ["accepted_false", "accepted_boolish", "missing", "extra", "schema"]
)
def test_authorization_schema_failure_is_exercised_after_valid_outer_pins(
    tmp_path, mutation
):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        args, auth, _ = complete_case(tmp_path)
        code = "adjudication_authorization_schema"
        if mutation == "accepted_false":
            auth["accepted_verification"] = False
        elif mutation == "accepted_boolish":
            auth["accepted_verification"] = 1
        elif mutation == "missing":
            del auth["archive"]
            code = "adjudication_authorization_fields"
        elif mutation == "extra":
            auth["unexpected"] = True
            code = "adjudication_authorization_fields"
        else:
            auth["schema_version"] = 1
        body = json.dumps(auth).encode()
        args.authorization.write_bytes(body)
        manifest = json.loads(args.manifest.read_bytes())
        manifest["context"]["authorization_file_sha256"] = hashlib.sha256(
            body
        ).hexdigest()
        body = json.dumps(manifest).encode()
        args.manifest.write_bytes(body)
        args.manifest_sha256 = hashlib.sha256(body).hexdigest()
        with pytest.raises(W.ScanError, match=code):
            A.verify(args)


@pytest.mark.parametrize(
    "role",
    ["encoded-geometry-data", "public-package-reference", "encoded-gzip-payload"],
)
@pytest.mark.parametrize(
    "mutation", ["none", "changed_evidence", "unreviewed", "active_credential"]
)
def test_local_literal_roles_require_exact_independent_proof(tmp_path, role, mutation):
    manifest, review, population, context, _, loader, proofs, _ = reviewed(tmp_path)
    literal_key = next(
        key
        for key, occurrence in population.items()
        if occurrence["finding_index"] is None
    )
    proof = proofs[literal_key]
    proof["semantic_role"] = role
    proof["operational_credential"] = mutation == "active_credential"
    disposition = next(
        row for row in manifest["dispositions"] if row["occurrence_id"] == literal_key
    )
    body = W.canonical(proof)
    Path(disposition["proof"]["path"]).write_bytes(body)
    disposition["proof"]["sha256"] = hashlib.sha256(body).hexdigest()
    if mutation != "unreviewed":
        decision = next(
            row
            for row in review["reviewed_occurrences"]
            if row["occurrence_id"] == literal_key
        )
        decision["proof_sha256"] = disposition["proof"]["sha256"]
    manifest_hash = hashlib.sha256(W.canonical(manifest)).hexdigest()
    review["manifest_sha256"] = manifest_hash
    if mutation == "changed_evidence":
        Path(proof["semantic_evidence"][0]["path"]).write_bytes(
            b"changed synthetic evidence"
        )
    if mutation == "none":
        assert (
            A.dispositions(manifest, review, population, context, manifest_hash, loader)
            == 4
        )
    else:
        with pytest.raises(W.ScanError):
            A.dispositions(manifest, review, population, context, manifest_hash, loader)
