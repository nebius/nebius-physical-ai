"""Exercise generic Docker-save identity before the complete-byte scanner."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import core  # noqa: E402
from image_byte_scan import docker_save_verification as VERIFIER  # noqa: E402
from image_byte_scan import prepare  # noqa: E402


def _tar(entries: list[tuple[str, bytes]]) -> bytes:
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, payload in entries:
            member = tarfile.TarInfo(name)
            member.size = len(payload)
            archive.addfile(member, io.BytesIO(payload))
    return stream.getvalue()


def _archive(
    tmp_path: Path,
    *,
    compressed: bool = False,
    oci_layout: bool = False,
    layer_entries: list[tuple[str, bytes]] | None = None,
    layer_payload: bytes | None = None,
) -> tuple[Path, str]:
    raw_layer = _tar(layer_entries or [("opt/result.txt", b"physical-ai")])
    stored_layer = (
        layer_payload
        if layer_payload is not None
        else gzip.compress(raw_layer, mtime=0)
        if compressed
        else raw_layer
    )
    config = {
        "rootfs": {
            "type": "layers",
            "diff_ids": ["sha256:" + hashlib.sha256(raw_layer).hexdigest()],
        }
    }
    config_bytes = json.dumps(config, sort_keys=True).encode()
    config_digest = hashlib.sha256(config_bytes).hexdigest()
    stored_layer_digest = hashlib.sha256(stored_layer).hexdigest()
    config_name = (
        "blobs/sha256/" + config_digest if oci_layout else config_digest + ".json"
    )
    layer_name = (
        "blobs/sha256/" + stored_layer_digest if oci_layout else "layer/layer.tar"
    )
    manifest = [
        {
            "Config": config_name,
            "Layers": [layer_name],
            "RepoTags": ["npa-robocasa:test"],
        }
    ]
    members = [
        ("manifest.json", json.dumps(manifest).encode()),
        (config_name, config_bytes),
        (layer_name, stored_layer),
    ]
    image_id = "sha256:" + config_digest
    if oci_layout:
        manifest_document = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "config": {
                "mediaType": "application/vnd.docker.container.image.v1+json",
                "digest": "sha256:" + config_digest,
                "size": len(config_bytes),
            },
            "layers": [
                {
                    "mediaType": (
                        "application/vnd.oci.image.layer.v1.tar+gzip"
                        if compressed
                        else "application/vnd.oci.image.layer.v1.tar"
                    ),
                    "digest": "sha256:" + stored_layer_digest,
                    "size": len(stored_layer),
                }
            ],
        }
        manifest_bytes = json.dumps(manifest_document, sort_keys=True).encode()
        manifest_digest = hashlib.sha256(manifest_bytes).hexdigest()
        index = {
            "schemaVersion": 2,
            "mediaType": "application/vnd.oci.image.index.v1+json",
            "manifests": [
                {
                    "mediaType": manifest_document["mediaType"],
                    "digest": "sha256:" + manifest_digest,
                    "size": len(manifest_bytes),
                    "platform": {"os": "linux", "architecture": "amd64"},
                }
            ],
        }
        members.extend(
            [
                ("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
                ("index.json", json.dumps(index).encode()),
                ("blobs/sha256/" + manifest_digest, manifest_bytes),
            ]
        )
        image_id = "sha256:" + manifest_digest
    archive = tmp_path / "image.tar"
    archive.write_bytes(_tar(members))
    return archive, image_id


@pytest.mark.parametrize(
    ("compressed", "oci_layout"),
    [(False, False), (True, False), (False, True), (True, True)],
)
def test_verifier_binds_complete_layer_and_config(
    tmp_path: Path, compressed: bool, oci_layout: bool
) -> None:
    archive, image_id = _archive(tmp_path, compressed=compressed, oci_layout=oci_layout)

    report = VERIFIER.verify(archive, image_id)

    assert report["schema_version"] == VERIFIER.SCHEMA
    assert report["valid"] is True
    assert report["expected_image_id"] == image_id
    if oci_layout:
        assert report["image_manifest_digest"] == image_id
    else:
        assert report["image_config_digest"] == image_id
    assert report["layer_count"] == 1
    assert report["regular_files_read"] == 1
    assert report["content_bytes_read"] == len(b"physical-ai")
    assert (
        report["docker_save_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert core.verification_archive_digest(report) == report["docker_save_sha256"]
    fd = os.open(archive, os.O_RDONLY)
    try:
        layers = core.graph(fd, os.fstat(fd).st_size, report, image_id)
    finally:
        os.close(fd)
    assert len(layers) == 1


def test_verifier_rejects_wrong_inspected_image_id(tmp_path: Path) -> None:
    archive, _image_id = _archive(tmp_path)

    with pytest.raises(core.ScanError, match="docker_save_expected_image"):
        VERIFIER.verify(archive, "sha256:" + "f" * 64)


def test_verifier_rejects_layer_that_disagrees_with_diff_id(tmp_path: Path) -> None:
    archive, image_id = _archive(tmp_path, layer_payload=_tar([("changed", b"bytes")]))

    with pytest.raises(core.ScanError, match="docker_save_layer_diff_id"):
        VERIFIER.verify(archive, image_id)


def test_verifier_rejects_unsafe_inner_path(tmp_path: Path) -> None:
    archive, image_id = _archive(
        tmp_path, layer_entries=[("../escape", b"not extracted")]
    )

    with pytest.raises(core.ScanError, match="docker_save_safe_path"):
        VERIFIER.verify(archive, image_id)


def test_preparation_accepts_bound_generic_verification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True)
    verification = tmp_path / "verification.json"
    verification.write_text(json.dumps(VERIFIER.verify(archive, image_id)))
    inventory = tmp_path / "literals.json"
    inventory.write_text(json.dumps({"literals": ["absent-private-marker"]}))
    tools = tmp_path / "tools.json"
    native = tmp_path / "native.json"
    tools.write_text("{}")
    native.write_text("{}")
    for path in (archive, verification, inventory, tools, native):
        path.chmod(0o600)

    class FakeDetector:
        def __init__(self, *_args: object) -> None:
            self.joined = False

        def finish(self) -> dict[str, object]:
            self.joined = True
            return {}

        def abort(self) -> None:
            self.joined = True

    monkeypatch.setattr(
        prepare,
        "tools_bindings",
        lambda _path: (
            {"path": str(tools), "sha256": "synthetic"},
            {"path": str(ROOT / ".gitleaks.toml"), "sha256": "synthetic"},
        ),
    )
    monkeypatch.setattr(
        prepare,
        "native_engine",
        lambda _path: {"kind": "synthetic-unexecuted-binding"},
    )
    monkeypatch.setattr(core, "Detector", FakeDetector)
    monkeypatch.setattr(core, "input_snapshots", lambda _authorization: [])
    args = SimpleNamespace(
        tools_receipt=tools,
        native_receipt=native,
        archive=archive,
        verification_report=verification,
        expected_image_id=image_id,
        policy_mode="exact-literals",
        literal_inventory=inventory,
        literal_matching_policy="exact-substring-v1",
    )
    output = tmp_path / "authorization"
    output.mkdir(mode=0o700)
    with core.authorized_roots(tmp_path, ROOT):
        result = prepare.authorize(args, output)
        assert (
            core.bound_json(result["verification_report"])["schema_version"]
            == VERIFIER.SCHEMA
        )


def test_cli_writes_owner_only_bound_report(tmp_path: Path) -> None:
    archive, image_id = _archive(tmp_path, oci_layout=True)
    output = tmp_path / "cli-output"

    result = subprocess.run(
        [
            sys.executable,
            str(ROOT / "npa/scripts/image_byte_scan/docker_save_verification.py"),
            "--analysis-root",
            str(tmp_path),
            "--trusted-root",
            str(ROOT),
            "--archive",
            str(archive),
            "--expected-image-id",
            image_id,
            "--output-dir",
            str(output),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result
    report = json.loads((output / "verification.json").read_text())
    assert report["expected_image_id"] == image_id
    assert (output / "verification.json").stat().st_mode & 0o777 == 0o600
