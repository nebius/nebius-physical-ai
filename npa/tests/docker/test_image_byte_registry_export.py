"""Verify original registry export preserves bytes and refuses ambiguous inputs."""

from __future__ import annotations

import json
import os
import tarfile

import pytest

from test_image_byte_scan import CHECKOUT, digest
from test_image_byte_registry_manifest import manifest_fixture
from image_byte_scan import core as W, registry_manifest_export as E


@pytest.fixture(autouse=True)
def roots(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        yield


def source_fixture(tmp_path, docker=True):
    files, expected = manifest_fixture(docker=docker, repeated_layer=True)
    source = tmp_path / "source"
    source.mkdir(mode=0o700)
    manifest_name = "blobs/sha256/" + expected[7:]
    for name, data in files.items():
        if not name.startswith("blobs/"):
            continue
        target = source / (
            "manifest.json" if name == manifest_name else name.rsplit("/", 1)[1]
        )
        target.write_bytes(data)
        target.chmod(0o600)
    (source / "version").write_bytes(E.DIRECTORY_VERSION)
    (source / "version").chmod(0o600)
    return source, files, expected


def export(tmp_path, source, expected):
    directory, descriptor = W.create_output(tmp_path / "output")
    try:
        return E.export(source, expected, directory, descriptor)
    finally:
        os.close(descriptor)


@pytest.mark.parametrize("docker", [False, True])
def test_original_config_manifest_and_compressed_blobs_are_unchanged(tmp_path, docker):
    source, original, expected = source_fixture(tmp_path, docker)
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    report = export(tmp_path, source, expected)
    assert report["structural_verification"]["image_manifest_digest"] == expected
    assert report["structural_verification"]["layer_count"] == 3
    assert not report["complete_byte_qualified"] and not report["native_qualified"]
    with tarfile.open(tmp_path / "output/image.tar", mode="r:") as archive:
        assert len(archive.getnames()) == len(set(archive.getnames()))
        for name, data in original.items():
            if name.startswith("blobs/"):
                assert archive.extractfile(name).read() == data
    assert before == {p.name: p.read_bytes() for p in source.iterdir()}


@pytest.mark.parametrize(
    "damage",
    [
        "changed-manifest",
        "changed-blob",
        "missing-blob",
        "extra",
        "signature",
        "version",
        "symlink",
        "public",
    ],
)
def test_bad_source_export_does_not_produce_acceptance(tmp_path, damage):
    source, _, expected = source_fixture(tmp_path)
    manifest = json.loads((source / "manifest.json").read_bytes())
    blob = source / manifest["layers"][0]["digest"][7:]
    if damage == "changed-manifest":
        (source / "manifest.json").write_bytes(b"{}")
    elif damage == "changed-blob":
        blob.write_bytes(b"changed")
    elif damage == "missing-blob":
        blob.unlink()
    elif damage in {"extra", "signature"}:
        (source / ("signature-1" if damage == "signature" else "extra")).write_bytes(
            b"retained"
        )
    elif damage == "version":
        (source / "version").write_bytes(b"unsupported transport")
    elif damage == "symlink":
        target = tmp_path / "retained-blob"
        blob.rename(target)
        blob.symlink_to(target)
    else:
        blob.chmod(0o644)
    with pytest.raises(W.INPUT_ERRORS):
        export(tmp_path, source, expected)
    assert not (tmp_path / "output/export.json").exists()


def test_export_refuses_config_identity_and_existing_output(tmp_path):
    source, _, expected = source_fixture(tmp_path)
    manifest = json.loads((source / "manifest.json").read_bytes())
    with pytest.raises(W.ScanError, match="input_binding_changed"):
        export(tmp_path, source, manifest["config"]["digest"])
    retained = tmp_path / "output/retained"
    retained.write_bytes(b"existing evidence")
    with pytest.raises(FileExistsError):
        export(tmp_path, source, expected)
    assert retained.read_bytes() == b"existing evidence"


def test_real_export_cli_writes_only_structural_receipt(tmp_path, capsys):
    source, _, expected = source_fixture(tmp_path)
    output = tmp_path / "cli-output"
    code = E.main(
        [
            "--analysis-root",
            str(tmp_path),
            "--trusted-root",
            str(CHECKOUT),
            "--source-dir",
            str(source),
            "--expected-image-id",
            expected,
            "--output-dir",
            str(output),
        ]
    )
    assert code == 0
    assert capsys.readouterr().out == "Original registry manifest export completed\n"
    report = json.loads((output / "export.json").read_bytes())
    assert report["archive_sha256"] == digest((output / "image.tar").read_bytes())
    assert report["structural_verification"]["expected_image_id"] == expected
    assert report["complete_byte_qualified"] is False
