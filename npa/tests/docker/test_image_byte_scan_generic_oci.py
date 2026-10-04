"""Exercise generic original-OCI verification without native tools or networking."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_image_byte_scan import (
    CHECKOUT,
    FakeDetector,
    digest,
    file,
    js,
    run,
    tar_data,
    write,
)
from test_image_byte_scan_oci import MARKER, authorize, oci_fixture
from image_byte_scan import (
    core as W,
    ncore_verification as N,
    oci_verification as O,
    prepare as P,
)


@pytest.fixture(autouse=True)
def roots(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        yield


def _authorization(tmp_path, **options):
    files, expected, raws = oci_fixture(**options)
    auth = authorize(tmp_path, files, expected)
    verified = O.verify(auth["archive"], auth["expected_image_id"])
    auth["verification_report"] = write(tmp_path / "verification.json", js(verified))
    return auth, verified, files, raws


@pytest.mark.parametrize("artifact", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_generic_original_oci_binds_every_physical_byte(tmp_path, artifact, nested):
    auth, verified, files, raws = _authorization(
        tmp_path, artifact=artifact, nested=nested
    )
    assert verified["schema_version"] == O.SCHEMA
    assert verified["regular_files_read"] == 3
    assert verified["content_bytes_read"] == len(b"originalcurrent")
    report, records = run(tmp_path, auth)
    assert report["valid"] and report["complete"] and report["helper_joined"], report
    assert report["oci_graph"]["image_index_digest"] == auth["expected_image_id"]
    encoded = [row for row in records if row.get("kind") == "outer_regular_content"]
    assert sorted((row["bytes"], row["sha256"]) for row in encoded) == sorted(
        (len(data), digest(data)) for data in files.values()
    )
    for scope, size in (
        ("outer", Path(auth["archive"]["path"]).stat().st_size),
        ("layer", sum(map(len, raws))),
    ):
        physical = [
            row
            for row in records
            if row.get("type") == "record"
            and row.get("scope") == scope
            and not row["kind"].startswith("logical_")
            and row["kind"] != "raw_gzip_header"
        ]
        assert sum(row["bytes"] for row in physical) == size
    assert [row["diff_id"] for row in report["layers"]] == verified[
        "verified_layer_diff_ids"
    ]


def test_ncore_schema_and_fields_remain_identical(tmp_path):
    auth, verified, _, _ = _authorization(tmp_path)
    ncore = N.verify(auth["archive"], auth["expected_image_id"])
    assert ncore == {**verified, "schema_version": N.SCHEMA}
    assert N.SCHEMA != O.SCHEMA


def test_generic_preparation_uses_existing_policy_and_dependency_contract(
    tmp_path, monkeypatch
):
    auth, verified, _, _ = _authorization(tmp_path)
    monkeypatch.setattr(P, "tools_bindings", lambda _: (auth["helper"], auth["config"]))
    engine = {"kind": "synthetic-unexecuted-binding"}
    engine.update(
        {role: write(tmp_path / role, b"synthetic binding") for role in W.AHO_PINS}
    )
    monkeypatch.setattr(P, "native_engine", lambda _: engine)
    monkeypatch.setattr(W, "Detector", FakeDetector)
    monkeypatch.setenv("CUSTOMER_DENYLIST", "synthetic-required-customer-pattern")
    monkeypatch.delenv("INFRA_DENYLIST", raising=False)
    args = SimpleNamespace(
        tools_receipt=auth["tools_receipt"]["path"],
        native_receipt=None,
        archive=auth["archive"]["path"],
        verification_report=auth["verification_report"]["path"],
        expected_image_id=auth["expected_image_id"],
        policy_mode="ci-regex",
        literal_inventory=None,
        literal_matching_policy="exact-substring-v1",
    )
    directory = tmp_path / "prepared"
    directory.mkdir(mode=0o700)
    prepared = P.authorize(args, directory)
    assert W.bound_json(prepared["verification_report"]) == verified
    assert (
        W.bound_json(prepared["confidentiality"])["customer_pattern"]
        == "synthetic-required-customer-pattern"
    )
    assert FakeDetector.instances[-1].joined


@pytest.mark.parametrize(
    "marker",
    [
        "ancestor",
        "runtime",
        "layer_metadata",
        "config",
        "history",
        "manifest",
        "index",
        "attestation",
    ],
)
def test_generic_scan_detects_each_original_graph_record(tmp_path, marker):
    auth, _, _, _ = _authorization(tmp_path, marker=marker)
    report, records = run(tmp_path, auth)
    assert report["complete"] and not report["valid"]
    assert any(row.get("rule_id") == "private_literal" for row in records)
    assert MARKER not in json.dumps(report) + json.dumps(records)


def test_generic_scan_keeps_nonzero_padding_failure(tmp_path):
    auth, _, _, _ = _authorization(tmp_path, padding=True)
    report, records = run(tmp_path, auth)
    assert report["complete"] and not report["valid"]
    assert any(row.get("rule_id") == "nonzero_tar_padding" for row in records)


@pytest.mark.parametrize(
    "field",
    [
        "expected_image_id",
        "image_index_digest",
        "image_manifest_digest",
        "image_config_digest",
        "verified_layer_diff_ids",
        "layer_count",
        "regular_files_read",
        "content_bytes_read",
    ],
)
def test_generic_retained_report_tampering_fails_closed(tmp_path, field):
    auth, verified, _, _ = _authorization(tmp_path)
    if field == "verified_layer_diff_ids":
        verified[field] = list(reversed(verified[field]))
    elif type(verified[field]) is int:
        verified[field] += 1
    else:
        verified[field] = "sha256:" + "0" * 64
    auth["verification_report"] = write(tmp_path / "verification.json", js(verified))
    report, _ = run(tmp_path, auth)
    assert not report["valid"] and not report["complete"], report


@pytest.mark.parametrize(
    "change",
    ["missing", "extra", "digest", "size", "duplicate", "platform", "unknown_codec"],
)
def test_generic_verifier_rejects_malformed_original_graph(tmp_path, change):
    files, _, _ = oci_fixture(nested=False)
    index = json.loads(files["index.json"])
    runtime = index["manifests"][0]
    name = "blobs/sha256/" + runtime["digest"][7:]
    if change == "missing":
        del files[name]
    elif change == "extra":
        files["blobs/sha256/" + digest(b"unreferenced")] = b"unreferenced"
    elif change == "digest":
        files[name] += b" "
        runtime["size"] += 1
    elif change == "size":
        runtime["size"] += 1
    elif change == "duplicate":
        index["manifests"].append(copy.deepcopy(runtime))
    elif change == "platform":
        runtime["platform"]["architecture"] = "arm64"
    else:
        runtime["mediaType"] = "application/unsupported"
    files["index.json"] = js(index)
    archive = write(
        tmp_path / "image.tar",
        tar_data([file(name, data) for name, data in files.items()]),
    )
    with pytest.raises(W.ScanError, match="oci_"):
        O.verify(archive, "sha256:" + digest(files["index.json"]))


@pytest.mark.parametrize("mode", ["missing", "empty"])
def test_generic_scan_requires_real_nonempty_policy(tmp_path, mode):
    auth, _, _, _ = _authorization(tmp_path)
    if mode == "missing":
        del auth["literal_inventory"]
    else:
        auth["literal_inventory"].update(
            write(tmp_path / "literals.json", js({"literals": []}))
        )
    with pytest.raises(W.ScanError, match="confidentiality_policy_required"):
        run(tmp_path, auth)


@pytest.mark.parametrize("target", ["archive", "policy", "source"])
def test_generic_scan_rejects_input_mutation_during_detection(
    tmp_path, monkeypatch, target
):
    auth, _, _, _ = _authorization(tmp_path)
    original = W.source_bindings

    class MutatingDetector(FakeDetector):
        def finish(self):
            summary = super().finish()
            if target == "source":
                monkeypatch.setattr(
                    W,
                    "source_bindings",
                    lambda: {**original(), "synthetic-extra-source": {}},
                )
            else:
                binding = auth[
                    "archive" if target == "archive" else "literal_inventory"
                ]
                with Path(binding["path"]).open("ab") as stream:
                    stream.write(b" ")
            return summary

    report, _ = run(tmp_path, auth, detector_type=MutatingDetector)
    assert not report["valid"] and not report["complete"], report


@pytest.mark.parametrize("failure", ["scan_cancelled", "helper_not_reaped"])
def test_generic_scan_keeps_interrupted_and_unjoined_helpers_failed(tmp_path, failure):
    auth, _, _, _ = _authorization(tmp_path)

    class FailedDetector(FakeDetector):
        def finish(self):
            raise W.ScanError(failure)

        def abort(self):
            self.joined = failure != "helper_not_reaped"

    report, _ = run(tmp_path, auth, detector_type=FailedDetector)
    assert not report["valid"] and not report["complete"]
    assert report["failure_code"] == failure
    assert report["helper_joined"] == (failure != "helper_not_reaped")


def test_generic_verifier_cli_writes_only_private_structural_report(tmp_path, capsys):
    auth, verified, _, _ = _authorization(tmp_path)
    output = tmp_path / "verified"
    result = O.main(
        [
            "--analysis-root",
            str(tmp_path),
            "--trusted-root",
            str(CHECKOUT),
            "--archive",
            auth["archive"]["path"],
            "--expected-image-id",
            auth["expected_image_id"],
            "--output-dir",
            str(output),
        ]
    )
    assert result == 0
    assert json.loads((output / "verification.json").read_text()) == verified
    assert (output / "verification.json").stat().st_mode & 0o077 == 0
    assert capsys.readouterr().out == "OCI graph verification completed\n"
