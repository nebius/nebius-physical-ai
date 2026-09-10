"""Small real OCI graphs; the detector double proves framing, not secret safety."""
from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_image_byte_scan import (
    CHECKOUT, FakeDetector, digest, file, fixture, fixture_tools_receipt, js,
    run, tar_data, write,
)
from image_byte_scan import core as W, prepare as P

INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
LAYER = "application/vnd.oci.image.layer.v1.tar"
INTOTO = "application/vnd.in-toto+json"
MARKER = "synthetic-private-operator-marker"


@pytest.fixture(autouse=True)
def roots(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        yield


def oci_fixture(*, artifact=False, nested=True, marker=None, padding=False):
    """Two ancestor layers, one whiteout, and a BuildKit attestation manifest."""
    blobs = {}
    def blob(data, media):
        desc = {"mediaType": media, "digest": "sha256:" + digest(data), "size": len(data)}
        blobs["blobs/sha256/" + digest(data)] = data
        return desc

    def tagged(role, neutral):
        return MARKER if marker == role else neutral

    ancestor_entries = [file("opt/removed", tagged("ancestor", "original").encode(),
                             pax={"SCHILY.xattr.user.review": tagged("layer_metadata", "neutral")})]
    current_entries = [file("opt/.wh.removed"), file("opt/current", tagged("runtime", "current").encode())]
    raws = [tar_data(ancestor_entries), tar_data(current_entries)]
    if padding:
        raw = bytearray(raws[1])
        # Second regular member body occupies offset 1024; its padding starts at 1031.
        raw[1031:1031 + len(MARKER)] = MARKER.encode()
        raws[1] = bytes(raw)
    layers = [blob(raws[0], LAYER), blob(gzip.compress(raws[1], mtime=0), LAYER + "+gzip")]
    config = blob(js({"architecture": "amd64", "os": "linux", "rootfs": {"type": "layers", "diff_ids": ["sha256:" + digest(raw) for raw in raws]},
                      "config": {"User": "1000", "Labels": {"review": tagged("config", "neutral")}},
                      "history": [{"created_by": tagged("history", "synthetic build")}]}), CONFIG)
    runtime = blob(js({"schemaVersion": 2, "mediaType": MANIFEST, "config": config, "layers": layers,
                       "annotations": {"review": tagged("manifest", "neutral")}}), MANIFEST)
    statement = blob(js({"_type": "https://in-toto.io/Statement/v0.1", "subject": [{"name": "synthetic", "digest": {"sha256": runtime["digest"][7:]}}],
                         "predicateType": "https://spdx.dev/Document", "predicate": {"review": tagged("attestation", "neutral")}}), INTOTO)
    att_config = blob(js({} if artifact else {"architecture": "unknown", "os": "unknown", "rootfs": {"type": "layers", "diff_ids": [statement["digest"]]}}),
                      "application/vnd.oci.empty.v1+json" if artifact else CONFIG)
    if artifact:
        att_config["data"] = "e30="
    att = {"schemaVersion": 2, "mediaType": MANIFEST, "config": att_config, "layers": [statement]}
    if artifact:
        att.update(artifactType="application/vnd.docker.attestation.manifest.v1+json", subject=runtime)
    attestation = blob(js(att), MANIFEST)
    runtime["platform"] = {"os": "linux", "architecture": "amd64"}
    attestation.update(platform={"os": "unknown", "architecture": "unknown"}, annotations={
        "vnd.docker.reference.type": "attestation-manifest", "vnd.docker.reference.digest": runtime["digest"]})
    root = js({"schemaVersion": 2, "mediaType": INDEX, "manifests": [runtime, attestation],
               "annotations": {"review": tagged("index", "neutral")}})
    publication_digest = "sha256:" + digest(root)
    if nested:
        root = js({"schemaVersion": 2, "mediaType": INDEX, "manifests": [blob(root, INDEX)]})
    files = {"oci-layout": b'{"imageLayoutVersion":"1.0.0"}', "index.json": root, **blobs}
    counts = {"regular_files_read": 3, "content_bytes_read": sum(len(row[1]) for row in ancestor_entries + current_entries)}
    verification = {"schema_version": "npa.ncore.oci-verification.v1", "valid": True,
                    "expected_image_id": publication_digest, "image_index_digest": publication_digest,
                    "image_manifest_digest": runtime["digest"], "image_config_digest": config["digest"],
                    "verified_layer_diff_ids": ["sha256:" + digest(raw) for raw in raws], "layer_count": 2, **counts}
    return files, verification, raws


def authorize(tmp_path, files, verification):
    # Reuse only synthetic dependency receipts, never a cuRobo verification report.
    auth = {"schema_version": "npa.image-byte-scan-authorization.v1", "accepted_verification": True,
            "archive": write(tmp_path / "image.tar", tar_data([file(name, data) for name, data in files.items()])),
            "expected_image_id": verification["expected_image_id"],
            "helper": {**write(tmp_path / "helper", b"synthetic framing oracle"), "ready_sha256": "0" * 64},
            "config": write(tmp_path / "config", (CHECKOUT / ".gitleaks.toml").read_bytes()),
            "literal_inventory": {**write(tmp_path / "literals.json", js({"literals": [MARKER]})), "matching_policy": "exact-substring-v1"},
            "sources": W.source_bindings()}
    verification = {**verification, "archive_sha256": auth["archive"]["sha256"]}
    auth["verification_report"] = write(tmp_path / "verification.json", js(verification))
    fixture_tools_receipt(auth, tmp_path)
    return auth


@pytest.mark.parametrize("artifact", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_attested_oci_complete_graph_and_physical_byte_receipts(tmp_path, artifact, nested):
    files, verification, raws = oci_fixture(artifact=artifact, nested=nested)
    auth = authorize(tmp_path, files, verification)
    report, records = run(tmp_path, auth)
    assert report["valid"] and report["complete"] and report["helper_joined"], report
    assert report["regular_files"] == 3 and len(report["layers"]) == 2
    assert report["oci_graph"]["image_index_digest"] == verification["image_index_digest"]
    assert report["oci_graph"]["blob_count"] == len(files) - 2
    assert report["oci_graph"]["blob_bytes"] == sum(len(data) for name, data in files.items() if name.startswith("blobs/"))
    bodies = [row for row in records if row.get("kind") == "outer_regular_content"]
    assert sorted((row["bytes"], row["sha256"]) for row in bodies) == sorted((len(data), digest(data)) for data in files.values())
    # Physical accounting excludes additive logical paths and zero-range receipts.
    for scope, size in [("outer", Path(auth["archive"]["path"]).stat().st_size), ("layer", sum(map(len, raws)))]:
        physical = [row for row in records if row.get("type") == "record" and row.get("scope") == scope
                    and not row["kind"].startswith("logical_") and row["kind"] != "raw_gzip_header"]
        assert sum(row["bytes"] for row in physical) == size


@pytest.mark.parametrize("marker", ["ancestor", "runtime", "layer_metadata", "config", "history", "manifest", "index", "attestation"])
def test_every_graph_role_is_scanned_and_private_findings_stay_redacted(tmp_path, marker):
    files, verification, _ = oci_fixture(marker=marker)
    report, records = run(tmp_path, authorize(tmp_path, files, verification))
    assert report["complete"] and not report["valid"]
    assert any(row.get("rule_id") == "private_literal" for row in records)
    assert MARKER not in json.dumps(report) + json.dumps(records)


def test_oci_nonzero_layer_padding_is_scanned_and_rejected(tmp_path):
    files, verification, _ = oci_fixture(padding=True)
    report, records = run(tmp_path, authorize(tmp_path, files, verification))
    assert report["complete"] and not report["valid"]
    assert any(row.get("rule_id") == "nonzero_tar_padding" for row in records)


@pytest.mark.parametrize("change", ["missing", "extra", "digest", "size", "digest_format", "duplicate", "root", "platform", "descriptor_urls", "unknown_codec"])
def test_oci_graph_mutations_fail_closed(tmp_path, change):
    files, verification, _ = oci_fixture(nested=False)
    index = json.loads(files["index.json"])
    runtime = index["manifests"][0]
    if change == "missing":
        del files["blobs/sha256/" + runtime["digest"][7:]]
    elif change == "extra":
        files["blobs/sha256/" + digest(b"unreferenced")] = b"unreferenced"
    elif change == "digest":
        files["blobs/sha256/" + runtime["digest"][7:]] += b" "
        runtime["size"] += 1
    elif change == "size":
        runtime["size"] += 1
    elif change == "digest_format":
        runtime["digest"] = "sha256:../invalid"
    elif change == "duplicate":
        index["manifests"].append(copy.deepcopy(runtime))
    elif change == "root":
        verification["expected_image_id"] = "sha256:" + "0" * 64
    elif change == "platform":
        runtime["platform"]["architecture"] = "arm64"
    elif change == "descriptor_urls":
        runtime["urls"] = ["https://example.invalid/external"]
    else:
        runtime["mediaType"] = "application/unsupported"
    files["index.json"] = js(index)
    if change != "root":
        verification["image_index_digest"] = verification["expected_image_id"] = "sha256:" + digest(files["index.json"])
    report, _ = run(tmp_path, authorize(tmp_path, files, verification))
    assert not report["complete"] and not report["valid"]
    assert report["failure_code"].startswith("oci_"), report


def test_ncore_verifier_generates_real_archive_binding_and_counts(tmp_path):
    from image_byte_scan import ncore_verification as N
    files, expected, _ = oci_fixture()
    archive = write(tmp_path / "image.tar", tar_data([file(name, data) for name, data in files.items()]))
    report = N.verify(archive, expected["expected_image_id"])
    assert report == {**expected, "archive_sha256": archive["sha256"]}


@pytest.mark.parametrize("schema", ["npa.unrecognized.image-verification.v1", None, [], {}])
def test_curobo_authorization_still_requires_its_own_exact_verifier_schema(tmp_path, schema):
    auth = fixture(tmp_path)
    verification = W.bound_json(auth["verification_report"])
    verification["schema_version"] = schema
    auth["verification_report"] = write(tmp_path / "verification.json", js(verification))
    with pytest.raises(W.ScanError, match="verification_schema"):
        run(tmp_path, auth)


def test_preparation_accepts_legitimate_ncore_verification(tmp_path, monkeypatch):
    files, verification, _ = oci_fixture()
    auth = authorize(tmp_path, files, verification)
    monkeypatch.setattr(P, "tools_bindings", lambda _: (auth["helper"], auth["config"]))
    monkeypatch.setattr(P, "native_engine", lambda _: {"kind": "synthetic-unexecuted-binding"})
    monkeypatch.setattr(W, "Detector", FakeDetector)
    monkeypatch.setattr(W, "input_snapshots", lambda _: [])
    args = SimpleNamespace(tools_receipt=auth["tools_receipt"]["path"], native_receipt=None,
                           archive=auth["archive"]["path"], verification_report=auth["verification_report"]["path"],
                           expected_image_id=auth["expected_image_id"], policy_mode="exact-literals",
                           literal_inventory=auth["literal_inventory"]["path"], literal_matching_policy="exact-substring-v1")
    directory = tmp_path / "prepared"
    directory.mkdir(mode=0o700)
    result = P.authorize(args, directory)
    assert W.bound_json(result["verification_report"])["schema_version"] == "npa.ncore.oci-verification.v1"


def rebind(files, verification, old_digest, data):
    """Replace one blob and transitively update actual parent bytes/descriptors."""
    new_digest = "sha256:" + digest(data)
    del files["blobs/sha256/" + old_digest[7:]]
    files["blobs/sha256/" + new_digest[7:]] = data
    for name, body in list(files.items()):
        if name != "index.json" and not name.startswith("blobs/"):
            continue
        try:
            value = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            continue
        changed = False
        def replace(item):
            nonlocal changed
            if isinstance(item, dict):
                if item.get("digest") == old_digest:
                    item.update(digest=new_digest, size=len(data))
                    changed = True
                for key, child in item.items():
                    if key == "vnd.docker.reference.digest" and child == old_digest:
                        item[key], changed = new_digest, True
                    else:
                        replace(child)
            elif isinstance(item, list):
                for child in item:
                    replace(child)
        replace(value)
        if changed:
            if name == "index.json":
                files[name] = js(value)
                verification["image_index_digest"] = verification["expected_image_id"] = "sha256:" + digest(files[name])
            else:
                rebind(files, verification, "sha256:" + name.rsplit("/", 1)[1], js(value))
    return new_digest


@pytest.mark.parametrize("role", ["runtime_config", "ancestor", "runtime_layer", "attestation_manifest", "attestation_config", "attestation_blob", "child_index"])
@pytest.mark.parametrize("change", ["missing", "corrupt"])
def test_all_descriptor_roles_require_exact_present_bytes(tmp_path, role, change):
    from image_byte_scan import ncore_verification as N
    files, verification, _ = oci_fixture()
    index_desc = json.loads(files["index.json"])["manifests"][0]
    index = json.loads(files["blobs/sha256/" + index_desc["digest"][7:]])
    runtime_desc, att_desc = index["manifests"]
    runtime = json.loads(files["blobs/sha256/" + runtime_desc["digest"][7:]])
    att = json.loads(files["blobs/sha256/" + att_desc["digest"][7:]])
    desc = {"runtime_config": runtime["config"], "ancestor": runtime["layers"][0], "runtime_layer": runtime["layers"][1],
            "attestation_manifest": att_desc, "attestation_config": att["config"], "attestation_blob": att["layers"][0], "child_index": index_desc}[role]
    name = "blobs/sha256/" + desc["digest"][7:]
    if change == "missing":
        del files[name]
    else:
        files[name] = files[name][:-1] + bytes([files[name][-1] ^ 1])
    archive = write(tmp_path / "image.tar", tar_data([file(name, body) for name, body in files.items()]))
    with W.bound_open(archive) as (_, fd, info), pytest.raises(W.ScanError, match="oci_missing_blob" if change == "missing" else "oci_blob_digest"):
        N.inspect(fd, info.st_size, verification["expected_image_id"])


@pytest.mark.parametrize("change,code", [
    ("subject", "oci_attestation_subject_disagreement"), ("inline", "oci_inline_descriptor"),
    ("size_bool", "oci_descriptor_size"), ("size_negative", "oci_descriptor_size"),
    ("attestation_media", "oci_descriptor_media_type"), ("malformed_json", "oci_json_object"),
])
def test_attestation_graph_schema_mutations_reach_specific_refusal(tmp_path, change, code):
    from image_byte_scan import ncore_verification as N
    files, verification, _ = oci_fixture(artifact=True, nested=False)
    index = json.loads(files["index.json"])
    runtime_desc, att_desc = index["manifests"]
    att = json.loads(files["blobs/sha256/" + att_desc["digest"][7:]])
    if change == "subject":
        # A conflicting existing manifest is still bound, then refused as target.
        att["subject"] = copy.deepcopy(att_desc)
    elif change == "inline":
        att["config"]["data"] = "e30K"
    elif change.startswith("size_"):
        att["layers"][0]["size"] = True if change == "size_bool" else -1
    elif change == "attestation_media":
        att["layers"][0]["mediaType"] = LAYER
    else:
        rebind(files, verification, att["layers"][0]["digest"], b"[]")
    if change != "malformed_json":
        if change == "subject":
            # Avoid recursive self-reference: subject is the runtime, annotation conflicts.
            att["subject"] = runtime_desc
            index["manifests"][1]["annotations"]["vnd.docker.reference.digest"] = "sha256:" + "0" * 64
            files["index.json"] = js(index)
        rebind(files, verification, att_desc["digest"], js(att))
    archive = write(tmp_path / "image.tar", tar_data([file(name, body) for name, body in files.items()]))
    with W.bound_open(archive) as (_, fd, info), pytest.raises(W.ScanError, match=code):
        N.inspect(fd, info.st_size, verification["expected_image_id"])


@pytest.mark.parametrize("where", ["metadata", "padding", "trailer", "duplicate_path", "truncated"])
def test_outer_archive_metadata_and_padding_are_not_omitted(tmp_path, where):
    files, verification, _ = oci_fixture()
    auth = authorize(tmp_path, files, verification)
    entries = [file(name, body, pax={"SCHILY.xattr.user.review": MARKER} if where == "metadata" else {}) for name, body in files.items()]
    if where == "duplicate_path":
        entries.append(entries[0])
    outer = bytearray(tar_data(entries))
    if where == "padding":
        start = 512 + len(files["oci-layout"])
        outer[start:start + len(MARKER)] = MARKER.encode()
    elif where == "trailer":
        outer += MARKER.encode().ljust(512, b"\0")
    elif where == "truncated":
        outer = outer[:-1]
    auth["archive"] = write(tmp_path / "image.tar", bytes(outer))
    verification["archive_sha256"] = auth["archive"]["sha256"]
    auth["verification_report"] = write(tmp_path / "verification.json", js(verification))
    report, records = run(tmp_path, auth)
    assert not report["valid"]
    if where in {"duplicate_path", "truncated"}:
        assert not report["complete"]
    else:
        assert report["complete"] and any(row.get("rule_id") == "private_literal" for row in records)
    assert MARKER not in json.dumps(report) + json.dumps(records)


@pytest.mark.parametrize("field", ["image_config_digest", "image_manifest_digest", "image_index_digest", "verified_layer_diff_ids", "layer_count", "regular_files_read", "content_bytes_read"])
def test_ncore_verifier_binding_or_count_disagreement_never_passes(tmp_path, field):
    files, verification, _ = oci_fixture(artifact=True)
    if field.endswith("digest"):
        verification[field] = "sha256:" + "0" * 64
    elif field == "verified_layer_diff_ids":
        verification[field] = list(reversed(verification[field]))
    else:
        verification[field] += 1
    report, _ = run(tmp_path, authorize(tmp_path, files, verification))
    assert not report["valid"] and not report["complete"]
    assert report["failure_code"] in {"oci_verifier_image_binding", "oci_expected_identity", "oci_verifier_layer_binding", "verifier_regular_population_disagreement"}


def test_ncore_cli_emits_only_private_receipt_and_sanitized_status(tmp_path, capsys):
    from image_byte_scan import ncore_verification as N
    files, expected, _ = oci_fixture(artifact=True, marker="config")
    archive = write(tmp_path / "image.tar", tar_data([file(name, data) for name, data in files.items()]))
    output = tmp_path / "verification-output"
    assert N.main(["--analysis-root", str(tmp_path), "--trusted-root", str(CHECKOUT),
                   "--archive", archive["path"], "--expected-image-id", expected["expected_image_id"], "--output-dir", str(output)]) == 0
    assert capsys.readouterr().out == "NCore OCI graph verification completed\n"
    report = output / "verification.json"
    assert report.stat().st_mode & 0o077 == 0 and MARKER not in report.read_text()
    assert N.main(["--unexpected", MARKER]) == 1
    captured = capsys.readouterr()
    assert captured.out == "NCore OCI graph verification failed\n" and captured.err == ""


@pytest.mark.parametrize("change,code", [("diff_id", "oci_attestation_config_layers"), ("duplicate_json", "duplicate_json_key"),
                                       ("codec", "declared_layer_codec_mismatch"), ("decoded_digest", "layer_uncompressed_diff_id")])
def test_rebound_graph_still_rejects_invalid_config_or_layer_semantics(tmp_path, change, code):
    files, verification, _ = oci_fixture(nested=False)
    index = json.loads(files["index.json"])
    desc = index["manifests"][1 if change == "diff_id" else 0]
    manifest = json.loads(files["blobs/sha256/" + desc["digest"][7:]])
    if change == "diff_id":
        config = json.loads(files["blobs/sha256/" + manifest["config"]["digest"][7:]])
        config["rootfs"]["diff_ids"] = []
        rebind(files, verification, manifest["config"]["digest"], js(config))
    elif change == "duplicate_json":
        files["index.json"] = files["index.json"][:-1] + b',"schemaVersion":2}'
        verification["image_index_digest"] = verification["expected_image_id"] = "sha256:" + digest(files["index.json"])
    elif change == "codec":
        manifest["layers"][1]["mediaType"] = LAYER
        verification["image_manifest_digest"] = rebind(files, verification, desc["digest"], js(manifest))
    else:
        raw = files["blobs/sha256/" + manifest["layers"][0]["digest"][7:]]
        # Replace a neutral regular body, keeping a valid tar and declared sizes.
        altered = raw.replace(b"original", b"modified")
        assert altered != raw
        rebind(files, verification, manifest["layers"][0]["digest"], altered)
        verification["image_manifest_digest"] = json.loads(files["index.json"])["manifests"][0]["digest"]
    report, _ = run(tmp_path, authorize(tmp_path, files, verification))
    assert not report["complete"] and not report["valid"] and report["failure_code"] == code


@pytest.mark.parametrize("missing_blob", [False, True])
def test_fixed_inline_empty_artifact_config_has_no_unscanned_payload(tmp_path, missing_blob):
    files, verification, _ = oci_fixture(artifact=True)
    if missing_blob:
        del files["blobs/sha256/" + digest(b"{}")]
    report, records = run(tmp_path, authorize(tmp_path, files, verification))
    assert report["valid"] and report["complete"], report
    assert report["oci_graph"]["blob_count"] == len(files) - 2
    assert any(row["digest"] == "sha256:" + digest(b"{}") for row in report["oci_graph"]["blobs"])
    assert all(row.get("findings", []) == [] for row in records if row.get("type") == "record")


@pytest.mark.parametrize("change", ["missing_policy", "not_accepted", "false_verifier", "boolean_count"])
def test_ncore_inputs_keep_existing_authorization_and_confidentiality_refusals(tmp_path, change):
    files, verification, _ = oci_fixture(artifact=True)
    if change == "false_verifier":
        verification["valid"] = False
    elif change == "boolean_count":
        verification["regular_files_read"] = True
    auth = authorize(tmp_path, files, verification)
    if change == "missing_policy":
        del auth["literal_inventory"]
    elif change == "not_accepted":
        auth["accepted_verification"] = False
    if change == "boolean_count":
        report, _ = run(tmp_path, auth)
        assert not report["valid"] and not report["complete"] and report["failure_code"] == "oci_verifier_population"
    else:
        code = {"missing_policy": "confidentiality_policy_required", "not_accepted": "verification_not_accepted", "false_verifier": "verification_did_not_pass"}[change]
        with pytest.raises(W.ScanError, match=code):
            run(tmp_path, auth)


def test_ncore_graph_does_not_pass_under_curobo_verifier_identity(tmp_path):
    files, verification, _ = oci_fixture()
    auth = authorize(tmp_path, files, verification)
    verification.update(schema_version="npa.curobo.image-verification.v1", docker_save_sha256=auth["archive"]["sha256"])
    auth["verification_report"] = write(tmp_path / "verification.json", js(verification))
    report, _ = run(tmp_path, auth)
    assert not report["valid"] and not report["complete"]
    assert "oci_graph" not in report
