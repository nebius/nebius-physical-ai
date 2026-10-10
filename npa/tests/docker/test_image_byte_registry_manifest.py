"""Exercise original manifest identity and complete ancestor byte accounting."""

from __future__ import annotations

import copy
import gzip
import json
from pathlib import Path
import tarfile

import pytest

from test_image_byte_scan import CHECKOUT, digest, file, js, run, tar_data, write
from test_image_byte_scan_oci import MARKER, authorize, oci_fixture
from image_byte_scan import core as W, oci_verification as O
from image_byte_scan import registry_manifest_verification as R


@pytest.fixture(autouse=True)
def roots(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        yield


def manifest_fixture(*, docker=False, marker=None, repeated_layer=False):
    files, old, raws = oci_fixture(marker=marker)
    manifest = json.loads(files["blobs/sha256/" + old["image_manifest_digest"][7:]])
    config = files["blobs/sha256/" + manifest["config"]["digest"][7:]]
    kept = {}
    if docker:
        manifest["mediaType"] = R.DOCKER + "distribution.manifest.v2+json"
        manifest["config"]["mediaType"] = R.DOCKER + "container.image.v1+json"
        for layer, raw in zip(manifest["layers"], raws, strict=True):
            encoded = gzip.compress(raw, mtime=0)
            layer.update(
                mediaType=R.DOCKER + "image.rootfs.diff.tar.gzip",
                digest="sha256:" + digest(encoded),
                size=len(encoded),
            )
            kept["blobs/sha256/" + digest(encoded)] = encoded
    else:
        kept.update(
            {
                "blobs/sha256/" + d["digest"][7:]: files[
                    "blobs/sha256/" + d["digest"][7:]
                ]
                for d in manifest["layers"]
            }
        )
    if repeated_layer:
        manifest["layers"].append(copy.deepcopy(manifest["layers"][0]))
        value = json.loads(config)
        value["rootfs"]["diff_ids"].append(value["rootfs"]["diff_ids"][0])
        config = js(value)
        manifest["config"].update(digest="sha256:" + digest(config), size=len(config))
    kept["blobs/sha256/" + digest(config)] = config
    kept["oci-layout"] = files["oci-layout"]
    expected = replace_manifest(kept, manifest, marker=marker)
    return kept, expected


def replace_manifest(files, manifest, *, marker=None):
    data = js(manifest)
    expected = "sha256:" + digest(data)
    if "index.json" in files:
        previous = json.loads(files["index.json"])["manifests"][0]["digest"]
        files.pop("blobs/sha256/" + previous[7:], None)
    files["blobs/sha256/" + expected[7:]] = data
    files["index.json"] = js(
        {
            "schemaVersion": 2,
            "mediaType": R.OCI + "index.v1+json",
            "manifests": [
                {
                    "mediaType": manifest["mediaType"],
                    "digest": expected,
                    "size": len(data),
                }
            ],
            "annotations": {
                "review": MARKER if marker == "index" else "transport-wrapper"
            },
        }
    )
    return expected


def verified(tmp_path, files, expected):
    binding = write(
        tmp_path / "verify-image.tar", tar_data([file(n, b) for n, b in files.items()])
    )
    return R.verify(binding, expected)


@pytest.mark.parametrize("docker", [False, True])
@pytest.mark.parametrize("repeated_layer", [False, True])
def test_original_manifest_retains_identity_and_complete_accounting(
    tmp_path, docker, repeated_layer
):
    files, expected = manifest_fixture(docker=docker, repeated_layer=repeated_layer)
    verification = verified(tmp_path, files, expected)
    report, records = run(tmp_path, authorize(tmp_path, files, verification))
    assert report["valid"] and report["complete"] and report["helper_joined"]
    assert report["image_manifest_digest"] == expected
    assert verification["attestations"] == R.ATTESTATIONS
    assert "image_index_digest" not in verification
    assert len(report["layers"]) == (3 if repeated_layer else 2)
    assert report["regular_files"] == (4 if repeated_layer else 3)
    assert report["oci_graph"]["attestations"]["external_referrers"] == "not-inspected"
    physical = [
        r
        for r in records
        if r.get("type") == "record"
        and r.get("scope") == "outer"
        and not r["kind"].startswith("logical_")
    ]
    assert (
        sum(r["bytes"] for r in physical) == Path(tmp_path / "image.tar").stat().st_size
    )
    bodies = [r for r in records if r.get("kind") == "outer_regular_content"]
    assert sorted((r["bytes"], r["sha256"]) for r in bodies) == sorted(
        (len(b), digest(b)) for b in files.values()
    )


@pytest.mark.parametrize(
    "marker",
    ["ancestor", "runtime", "layer_metadata", "config", "history", "manifest", "index"],
)
@pytest.mark.parametrize("docker", [False, True])
def test_deleted_ancestor_history_metadata_and_live_bytes_are_scanned(
    tmp_path, marker, docker
):
    files, expected = manifest_fixture(docker=docker, marker=marker)
    report, records = run(
        tmp_path, authorize(tmp_path, files, verified(tmp_path, files, expected))
    )
    assert report["complete"] and not report["valid"]
    assert any(r.get("rule_id") == "private_literal" for r in records)
    assert MARKER not in json.dumps(report) + json.dumps(records)


@pytest.mark.parametrize("kind", ["config", "transport-index", "unrelated"])
def test_expected_identity_cannot_be_replaced_by_config_or_local_wrapper(
    tmp_path, kind
):
    files, expected = manifest_fixture(docker=True)
    manifest = json.loads(files["blobs/sha256/" + expected[7:]])
    wrong = {
        "config": manifest["config"]["digest"],
        "transport-index": "sha256:" + digest(files["index.json"]),
        "unrelated": "sha256:" + "1" * 64,
    }[kind]
    with pytest.raises(W.ScanError, match="registry_original_manifest_binding"):
        verified(tmp_path, files, wrong)


@pytest.mark.parametrize(
    "damage",
    ["config-byte", "compressed-byte", "missing-layer", "extra-file", "extra-blob"],
)
def test_original_content_and_exact_graph_closure_are_required(tmp_path, damage):
    files, expected = manifest_fixture(docker=True)
    manifest = json.loads(files["blobs/sha256/" + expected[7:]])
    config = "blobs/sha256/" + manifest["config"]["digest"][7:]
    layer = "blobs/sha256/" + manifest["layers"][0]["digest"][7:]
    if damage == "config-byte":
        files[config] = files[config].replace(b"amd64", b"arm64")
    elif damage == "compressed-byte":
        value = bytearray(files[layer])
        value[-1] ^= 1
        files[layer] = bytes(value)
    elif damage == "missing-layer":
        del files[layer]
    else:
        files[
            "unexpected"
            if damage == "extra-file"
            else "blobs/sha256/" + digest(b"extra")
        ] = b"extra"
    with pytest.raises(W.ScanError):
        verified(tmp_path, files, expected)


@pytest.mark.parametrize(
    "damage",
    [
        "foreign-url",
        "config-as-layer",
        "size-bool",
        "wrong-platform",
        "subject",
        "index-root",
    ],
)
def test_unsupported_original_graph_shapes_fail_closed(tmp_path, damage):
    files, expected = manifest_fixture(docker=True)
    manifest = json.loads(files["blobs/sha256/" + expected[7:]])
    if damage == "foreign-url":
        manifest["layers"][0]["urls"] = ["https://example.invalid/blob"]
    elif damage == "config-as-layer":
        manifest["layers"][0] = dict(manifest["config"])
    elif damage == "size-bool":
        manifest["layers"][0]["size"] = True
    elif damage == "wrong-platform":
        manifest["config"]["platform"] = {"os": "linux", "architecture": "arm64"}
    elif damage == "subject":
        manifest["subject"] = dict(manifest["config"])
    else:
        manifest["mediaType"] = R.OCI + "index.v1+json"
    expected = replace_manifest(files, manifest)
    with pytest.raises(W.ScanError):
        verified(tmp_path, files, expected)


@pytest.mark.parametrize("damage", ["duplicate", "symlink", "padding"])
def test_archive_ambiguity_or_hidden_padding_cannot_qualify(tmp_path, damage):
    files, expected = manifest_fixture()
    entries = [file(n, b) for n, b in files.items()]
    if damage == "duplicate":
        entries.append(entries[0])
    elif damage == "symlink":
        entries.append(file("unexpected-link", kind=tarfile.SYMTYPE, link="index.json"))
    raw = tar_data(entries)
    if damage != "padding":
        with pytest.raises(W.ScanError):
            R.verify(write(tmp_path / "bad.tar", raw), expected)
        return
    verification = verified(tmp_path, files, expected)
    auth = authorize(tmp_path, files, verification)
    raw = bytearray(Path(auth["archive"]["path"]).read_bytes())
    raw[-1] = 1
    auth["archive"] = write(Path(auth["archive"]["path"]), bytes(raw))
    verification["archive_sha256"] = auth["archive"]["sha256"]
    auth["verification_report"] = write(
        Path(auth["verification_report"]["path"]), js(verification)
    )
    report, _ = run(tmp_path, auth)
    assert not report["valid"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("attestations", {"status": "verified"}),
        ("identity_kind", "oci-index"),
        ("layer_count", True),
        ("regular_files_read", -1),
        ("verified_layer_diff_ids", []),
        ("image_config_digest", "sha256:" + "2" * 64),
    ],
)
def test_retained_receipt_cannot_overstate_or_rebind_graph(tmp_path, field, value):
    files, expected = manifest_fixture()
    verification = verified(tmp_path, files, expected)
    auth = authorize(tmp_path, files, {**verification, field: value})
    report, _ = run(tmp_path, auth)
    assert not report["valid"]


def test_existing_index_profile_still_rejects_unattested_manifest_root(tmp_path):
    files, expected = manifest_fixture()
    binding = write(
        tmp_path / "original.tar", tar_data([file(n, b) for n, b in files.items()])
    )
    with pytest.raises(W.ScanError, match="oci_descriptor_media_type"):
        O.verify(binding, expected)


def test_config_diff_ids_are_verified_against_actual_decoded_bytes(tmp_path):
    files, expected = manifest_fixture(docker=True)
    manifest = json.loads(files["blobs/sha256/" + expected[7:]])
    old = "blobs/sha256/" + manifest["config"]["digest"][7:]
    config = json.loads(files.pop(old))
    config["rootfs"]["diff_ids"][0] = "sha256:" + "3" * 64
    data = js(config)
    files["blobs/sha256/" + digest(data)] = data
    manifest["config"].update(digest="sha256:" + digest(data), size=len(data))
    expected = replace_manifest(files, manifest)
    with pytest.raises(W.ScanError, match="oci_layer_diff_id"):
        verified(tmp_path, files, expected)
