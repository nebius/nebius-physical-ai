"""Exercise direct-manifest graph integrity without weakening attested defaults."""

from __future__ import annotations

import copy
import gzip
import json
import os
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
from test_image_byte_scan_oci import authorize, oci_fixture
from image_byte_scan import (
    core as W,
    oci_graph as G,
    prepare as P,
    seedvr2_verification as S,
)


@pytest.fixture(autouse=True)
def roots(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        yield


def _blob(files, payload, media):
    descriptor = {
        "mediaType": media,
        "digest": "sha256:" + digest(payload),
        "size": len(payload),
    }
    files["blobs/sha256/" + descriptor["digest"][7:]] = payload
    return descriptor


def _direct(
    *,
    docker=True,
    marker=None,
    platform=None,
    config=None,
    raw=None,
    encoded=None,
    layer_media=None,
):
    files, original, _ = oci_fixture(nested=False, marker=marker)
    root = json.loads(files["index.json"])
    descriptor = root["manifests"][0]
    manifest = json.loads(files["blobs/sha256/" + descriptor["digest"][7:]])
    selected = {"oci-layout": files["oci-layout"]}
    _direct_layers(files, selected, manifest, docker, raw, encoded, layer_media)
    payload = config or json.loads(
        files["blobs/sha256/" + manifest["config"]["digest"][7:]]
    )
    if raw is not None:
        payload["rootfs"]["diff_ids"] = ["sha256:" + digest(raw)]
    manifest["config"] = _blob(
        selected, js(payload), G.DOCKER_CONFIG if docker else G.CONFIG
    )
    manifest["mediaType"] = G.DOCKER_MANIFEST if docker else G.MANIFEST
    descriptor = _blob(selected, js(manifest), manifest["mediaType"])
    if platform is not None:
        descriptor["platform"] = platform
    root["manifests"] = [descriptor]
    selected["index.json"] = js(root)
    selected["manifest.json"] = _compatibility(manifest)
    return selected, descriptor["digest"], original


def _compatibility(manifest):
    return js(
        [
            {
                "Config": "blobs/sha256/" + manifest["config"]["digest"][7:],
                "Layers": [
                    "blobs/sha256/" + row["digest"][7:] for row in manifest["layers"]
                ],
            }
        ]
    )


def _direct_layers(files, selected, manifest, docker, raw, encoded, layer_media):
    layers = []
    for index, row in enumerate(manifest["layers"]):
        if raw is not None and index:
            continue
        content = (
            (encoded if encoded is not None else gzip.compress(raw, mtime=0))
            if raw is not None
            else files["blobs/sha256/" + row["digest"][7:]]
        )
        media = row["mediaType"]
        if docker:
            media = "application/vnd.docker.image.rootfs.diff.tar" + (
                ".gzip" if media.endswith("+gzip") else ""
            )
        if raw is not None:
            media = layer_media or "application/vnd.docker.image.rootfs.diff.tar.gzip"
        layers.append(_blob(selected, content, media))
    manifest["layers"] = layers


def _archive(tmp_path, files):
    return write(
        tmp_path / "direct.tar",
        tar_data([file(name, data) for name, data in files.items()]),
    )


@pytest.mark.parametrize("docker", [False, True])
@pytest.mark.parametrize("platform", [None, S.PLATFORM])
def test_direct_graph_complete_scan_and_truthful_identity(tmp_path, docker, platform):
    files, expected, original = _direct(docker=docker, platform=platform)
    verification = S.verify(_archive(tmp_path, files), expected)
    assert verification["schema_version"] == S.SCHEMA
    assert verification["image_manifest_digest"] == expected
    assert "image_index_digest" not in verification
    assert verification["attestation_manifest_count"] == 0
    for key in ("regular_files_read", "content_bytes_read", "verified_layer_diff_ids"):
        assert verification[key] == original[key]
    report, _ = run(tmp_path, authorize(tmp_path, files, verification))
    assert report["complete"] and report["valid"], report
    assert report["direct_manifest_graph"]["image_manifest_digest"] == expected
    assert "image_index_digest" not in report["direct_manifest_graph"]
    assert "oci_graph" not in report
    assert report["direct_manifest_graph"]["blob_count"] == len(files) - 3


@pytest.mark.parametrize(
    "marker",
    ["ancestor", "runtime", "config", "history", "manifest", "index", "layer_metadata"],
)
def test_direct_scan_detects_complete_ancestor_and_metadata_population(
    tmp_path, marker
):
    files, expected, _ = _direct(marker=marker)
    verification = S.verify(_archive(tmp_path, files), expected)
    report, _ = run(tmp_path, authorize(tmp_path, files, verification))
    assert report["complete"] and not report["valid"], report


@pytest.mark.parametrize(
    "change",
    [
        "extra",
        "missing",
        "digest",
        "size",
        "multiple",
        "wrong_id",
        "platform",
        "config_platform",
        "urls",
        "mime",
        "media_disagreement",
        "compatibility",
        "attestation",
    ],
)
def test_direct_graph_rejects_mutations(tmp_path, change):
    wrong_platform = {
        "os": "linux",
        "architecture": "arm64",
        "rootfs": {"type": "layers", "diff_ids": []},
    }
    files, expected, _ = _direct(
        config=wrong_platform if change == "config_platform" else None
    )
    index = json.loads(files["index.json"])
    desc = index["manifests"][0]
    path = "blobs/sha256/" + expected[7:]
    if change == "extra":
        files["blobs/sha256/" + digest(b"unreferenced")] = b"unreferenced"
    elif change == "missing":
        del files[path]
    elif change == "digest":
        files[path] += b" "
        desc["size"] += 1
    elif change == "size":
        desc["size"] += 1
    elif change == "multiple":
        index["manifests"].append(copy.deepcopy(desc))
    elif change == "wrong_id":
        expected = "sha256:" + "0" * 64
    elif change == "platform":
        desc["platform"] = {"os": "linux", "architecture": "arm64"}
    elif change == "urls":
        desc["urls"] = ["https://example.invalid/blob"]
    elif change == "mime":
        desc["mediaType"] = "application/unsupported"
    elif change == "media_disagreement":
        desc["mediaType"] = G.MANIFEST
    elif change == "compatibility":
        files["manifest.json"] = js([{"Config": "wrong", "Layers": []}])
    elif change == "attestation":
        desc["annotations"] = {"vnd.docker.reference.type": "attestation-manifest"}
    files["index.json"] = js(index)
    with pytest.raises(W.ScanError):
        S.verify(_archive(tmp_path, files), expected)


@pytest.mark.parametrize(
    "damage", ["trailing", "concatenated", "truncated", "wrong_diff", "wrong_codec"]
)
def test_direct_decoded_layer_integrity(tmp_path, damage):
    raw = tar_data([file("opt/control", b"real synthetic body")])
    encoded = gzip.compress(raw, mtime=0)
    media = None
    if damage == "trailing":
        encoded += b"unaccounted bytes"
    elif damage == "concatenated":
        encoded += gzip.compress(b"extra member", mtime=0)
    elif damage == "truncated":
        encoded = encoded[:-1]
    elif damage == "wrong_diff":
        encoded = gzip.compress(raw + b"different", mtime=0)
    else:
        media = "application/vnd.docker.image.rootfs.diff.tar"
    files, expected, _ = _direct(raw=raw, encoded=encoded, layer_media=media)
    with pytest.raises((W.ScanError, EOFError)):
        S.verify(_archive(tmp_path, files), expected)


@pytest.mark.parametrize("docker", [False, True])
def test_attested_default_still_refuses_unattested_manifest(tmp_path, docker):
    files, expected, _ = _direct(docker=docker)
    archive = _archive(tmp_path, files)
    with W.bound_open(archive) as (_, descriptor, info):
        with pytest.raises(W.ScanError):
            G.inspect(
                descriptor,
                info.st_size,
                expected,
                platform=S.PLATFORM,
                allow_docker_manifest=True,
            )


def test_direct_mode_does_not_select_runtime_from_attested_index(tmp_path):
    files, original, _ = oci_fixture(nested=False)
    with pytest.raises(W.ScanError):
        S.verify(_archive(tmp_path, files), original["image_manifest_digest"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("image_index_digest", "sha256:" + "a" * 64),
        ("attestation_manifest_count", 1),
        ("layer_count", True),
        ("regular_files_read", -1),
        ("image_config_digest", "sha256:" + "0" * 64),
    ],
)
def test_direct_receipt_tampering_fails(tmp_path, key, value):
    files, expected, _ = _direct()
    verification = S.verify(_archive(tmp_path, files), expected)
    verification[key] = value
    report, _ = run(tmp_path, authorize(tmp_path, files, verification))
    assert not report["complete"] and not report["valid"]


def test_direct_cli_private_receipt(tmp_path, capsys):
    files, expected, _ = _direct()
    archive = _archive(tmp_path, files)
    assert (
        S.main(
            [
                "--analysis-root",
                str(tmp_path),
                "--trusted-root",
                str(CHECKOUT),
                "--archive",
                archive["path"],
                "--expected-image-id",
                expected,
                "--output-dir",
                str(tmp_path / "receipt"),
            ]
        )
        == 0
    )
    assert capsys.readouterr().out == "SeedVR2 direct-manifest verification completed\n"
    assert os.stat(tmp_path / "receipt/verification.json").st_mode & 0o777 == 0o600


def test_preparation_accepts_only_exact_direct_manifest_schema(tmp_path, monkeypatch):
    files, expected, _ = _direct()
    verification = S.verify(_archive(tmp_path, files), expected)
    auth = authorize(tmp_path, files, verification)
    monkeypatch.setattr(P, "tools_bindings", lambda _: (auth["helper"], auth["config"]))
    monkeypatch.setattr(
        P, "native_engine", lambda _: {"kind": "synthetic-unexecuted-binding"}
    )
    monkeypatch.setattr(W, "Detector", FakeDetector)
    monkeypatch.setattr(W, "input_snapshots", lambda _: [])
    args = SimpleNamespace(
        tools_receipt=auth["tools_receipt"]["path"],
        native_receipt=None,
        archive=auth["archive"]["path"],
        verification_report=auth["verification_report"]["path"],
        expected_image_id=expected,
        policy_mode="exact-literals",
        literal_inventory=auth["literal_inventory"]["path"],
        literal_matching_policy="exact-substring-v1",
    )
    directory = tmp_path / "prepared"
    directory.mkdir(mode=0o700)
    result = P.authorize(args, directory)
    assert W.bound_json(result["verification_report"])["schema_version"] == S.SCHEMA


def test_decoded_records_and_transport_are_accounted_separately(tmp_path):
    files, expected, _ = _direct()
    verification = S.verify(_archive(tmp_path, files), expected)
    report, _ = run(tmp_path, authorize(tmp_path, files, verification))
    assert report["valid"] and report["complete"]
    records = [
        json.loads(line)
        for line in (tmp_path / "output/records.jsonl").read_text().splitlines()
    ]
    encoded = [row for row in records if row.get("type") == "encoded_layer_blob"]
    assert len(encoded) == 2
    assert all(row["bytes"] > 0 and len(row["sha256"]) == 64 for row in encoded)
    assert report["regular_files"] == verification["regular_files_read"] == 3
    assert report["regular_bytes"] == verification["content_bytes_read"]
