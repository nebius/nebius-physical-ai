import copy
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/tests/docker"))
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import public_native_policy as P  # noqa: E402
from test_image_byte_adjudication import W, record  # noqa: E402


def test_complete_shipped_catalog_compiles_with_all_bound_proofs(tmp_path, monkeypatch):
    """Check the real evidence graph with the production proof validator."""
    catalog = json.loads(
        (
            ROOT / "npa/scripts/image_byte_scan/public_policies/curobo-v2.json"
        ).read_text()
    )
    loaded = set()
    reviewer = object.__new__(P.FreshPolicyReview)

    def bound_source(path, expected):
        data = path.read_bytes()
        assert W.sha(data) == expected
        loaded.add(path)
        return data

    monkeypatch.setattr(reviewer, "_public_source", bound_source)
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, ROOT):
        entries = P.compile_catalog(
            catalog, catalog["detector_identity"], reviewer._proof
        )
    assert len(entries) == len(catalog["entries"]) > 0
    assert loaded == {
        ROOT / entry[kind]["path"]
        for entry in catalog["entries"]
        for kind in ("public_provenance", "semantic_proof")
    }


def test_curobo_same_byte_remediation_binds_all_ten_exact_record_classes():
    root = ROOT / "npa/scripts/image_byte_scan/public_policies"
    policy = json.loads((root / "curobo-v2.json").read_text())
    sources = {
        row["record_sha256"]: row
        for row in json.loads((root / "curobo-v2-sources.json").read_text())["content"]
    }
    semantics = {
        row["record_sha256"]: row
        for row in json.loads((root / "curobo-v2-semantics.json").read_text())[
            "content"
        ]
    }
    expected = {
        "09339f848f60bb1111a61ee0fe91ed0c25132b7ff63298244d7ac14e61b58466": (
            "layer_regular_content",
            162565256,
            2,
            "encoded-device-payload",
        ),
        "404ccd5811e79ed47c95dc7d5ab6f6d66c15cc2ee576a5150f224db28c07a01f": (
            "layer_regular_content",
            170430,
            1,
            "public-source-symbol",
        ),
        "707a84f218ac88c75c7a82164cd74ec85684dace82e015d2dc49b4b356f3a54c": (
            "layer_regular_content",
            245899,
            2,
            "public-source-symbol",
        ),
        "80d4c51e0d3ace8f28f108c7c7f9ac984e9a636da7833e3aa41cf0fa93bec8b1": (
            "layer_regular_content",
            170466,
            1,
            "public-source-symbol",
        ),
        "8e49cebd373d90b55f3626da71dd97fe8e95c68dc4c064d8962b27907690736e": (
            "layer_regular_content",
            696512,
            1,
            "package-integrity-metadata",
        ),
        "c571d524fc5571e6c8a5784d51f75fbe8cf17590463c397a06e48428cb926c48": (
            "layer_regular_content",
            141152872,
            1,
            "encoded-device-payload",
        ),
        "ddbae2995750875c07bc218d12a062d73f8250678f54dedda7e9be5068781c98": (
            "layer_regular_content",
            2065824,
            10,
            "cryptographic-self-test",
        ),
        "e03287709e8e16898a6834541beaf96414631cae385a54a63fcfb55826381725": (
            "logical_tar_path",
            45,
            1,
            "public-package-path-metadata",
        ),
        "e12607320178caa224182665d31c28501194573ece6d104d2cd7ccdea6439dc7": (
            "layer_regular_content",
            114691,
            1,
            "public-source-symbol",
        ),
        "fb46c104e3ef44f1353c323c847b2ed7e224eb3d23b1e7df90ec0253f10f9270": (
            "logical_tar_path",
            49,
            1,
            "public-package-path-metadata",
        ),
    }
    entries = {
        row["record_sha256"]: row
        for row in policy["entries"]
        if row["record_sha256"] in expected
    }
    assert entries.keys() == expected.keys() <= sources.keys() & semantics.keys()
    for digest, (kind, size, findings, role) in expected.items():
        row = entries[digest]
        assert (
            row["record_kind"],
            row["record_bytes"],
            len(row["native_findings"]),
            row["semantic_role"],
            row["operational_credential"],
        ) == (kind, size, findings, role, False)
        assert sources[digest]["public_origin"]
        assert semantics[digest]["lexical_source_context"]
        assert "Deferred" not in semantics[digest]["reason"]

    gnutls = sources["ddbae2995750875c07bc218d12a062d73f8250678f54dedda7e9be5068781c98"]
    assert gnutls["public_origin"]["version"] == "3.8.3-1.1ubuntu3.6"
    assert gnutls["semantic_source"]["patch_application"]["applied_patch_count"] == 46
    assert len(gnutls["semantic_source"]["constant_declarations"]) == 10

    for digest, count in (
        (
            "c571d524fc5571e6c8a5784d51f75fbe8cf17590463c397a06e48428cb926c48",
            1,
        ),
        (
            "09339f848f60bb1111a61ee0fe91ed0c25132b7ff63298244d7ac14e61b58466",
            2,
        ),
    ):
        containers = semantics[digest]["lexical_source_context"][0]["containers"]
        assert len(containers) == count
        assert all(
            row["original_native_pattern_survives_decoding"] is False
            for row in containers
        )

    bytecode = sources[
        "80d4c51e0d3ace8f28f108c7c7f9ac984e9a636da7833e3aa41cf0fa93bec8b1"
    ]["public_origin"]
    assert bytecode["kind"] == "exact-checked-hash-bytecode"
    assert bytecode["header"]["source_hash_matches_exact_source"] is True
    assert bytecode["marshal"]["recursive_code_objects"] == 117
    assert bytecode["whole_file_reproduction"]["whole_file_equal"] is True

    openssl = sources[
        "8e49cebd373d90b55f3626da71dd97fe8e95c68dc4c064d8962b27907690736e"
    ]
    assert openssl["semantic_source"]["section"]["name"] == ".gnu_debuglink"


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


@pytest.mark.parametrize(
    "role",
    [
        "public-documentation-text",
        "public-protocol-field",
        "numeric-debug-metadata",
        "numeric-unwind-metadata",
        "machine-instruction-bytes",
        "encoded-archive-member-payload",
        "encoded-bzip2-payload",
        "encoded-video-payload",
        "public-service-endpoint-reference",
        "public-provider-hardware-reference",
        "public-provider-location-reference",
        "product-installation-path",
        "public-application-protocol-reference",
        "bytecode-instruction-bytes",
    ],
)
def test_local_adjudication_roles_do_not_extend_hosted_policy(role):
    catalog, _report, _rows = fixture()
    catalog["entries"][0]["semantic_role"] = role
    with pytest.raises(W.ScanError, match="public_policy_unsafe_semantics"):
        compile_catalog(catalog)


@pytest.mark.parametrize(
    "url",
    [
        None,
        False,
        "",
        "http://example.com/package.whl",
        "https://user@example.com/package.whl",
        "https://example.com/package.whl?credential=fixture",
        "https://example.com/package.whl#fragment",
        "https://example.com/package.tar.gz",
        "https://example.com:bad/package.whl",
        "https:///package.whl",
        "https://example.com/../package.whl",
        "https://example.com/unsafe path.whl",
    ],
)
def test_typed_public_wheel_requires_credential_free_reproducible_url(
    tmp_path, monkeypatch, url
):
    tmp_path.chmod(0o700)
    payload = {
        "schema_version": "npa.public-native-source-proofs.v1",
        "content": [
            {
                "public_origin": {
                    "kind": "exact-checked-hash-bytecode",
                    "source": {
                        "kind": "locked-public-wheel-member",
                        "artifact_sha256": "a" * 64,
                        "artifact_url": url,
                    },
                }
            }
        ],
    }
    reviewer = object.__new__(P.FreshPolicyReview)
    monkeypatch.setattr(
        reviewer, "_public_source", lambda path, expected: W.canonical(payload)
    )
    with W.authorized_roots(tmp_path, ROOT):
        with pytest.raises(W.ScanError, match="public_policy_wheel_artifact_url"):
            reviewer._proof(
                {"path": "synthetic-public-source.json", "sha256": "b" * 64}
            )


def test_missing_typed_wheel_url_refuses_but_other_proof_kinds_keep_their_schema():
    P.validate_public_wheel_references(
        {"kind": "signed-ubuntu-package-member", "artifact_url": None}
    )
    P.validate_public_wheel_references(
        {
            "kind": "locked-public-wheel-member",
            "artifact_url": "https://example.com/package%2Bvariant.whl",
            "artifact_sha256": "a" * 64,
        }
    )
    with pytest.raises(W.ScanError, match="public_policy_wheel_artifact_url"):
        P.validate_public_wheel_references(
            {"kind": "locked-public-wheel-member", "artifact_sha256": "a" * 64}
        )
