"""Exercise generic Docker-save identity before the complete-byte scanner."""

from __future__ import annotations

import gzip
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
from image_byte_scan import core  # noqa: E402
from image_byte_scan import docker_save_verification as VERIFIER  # noqa: E402


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
    config_name = config_digest + ".json"
    layer_name = "layer/layer.tar"
    manifest = [
        {
            "Config": config_name,
            "Layers": [layer_name],
            "RepoTags": ["npa-robocasa:test"],
        }
    ]
    archive = tmp_path / "image.tar"
    archive.write_bytes(
        _tar(
            [
                ("manifest.json", json.dumps(manifest).encode()),
                (config_name, config_bytes),
                (layer_name, stored_layer),
            ]
        )
    )
    return archive, "sha256:" + config_digest


@pytest.mark.parametrize("compressed", [False, True])
def test_verifier_binds_complete_layer_and_config(
    tmp_path: Path, compressed: bool
) -> None:
    archive, image_id = _archive(tmp_path, compressed=compressed)

    report = VERIFIER.verify(archive, image_id)

    assert report["schema_version"] == VERIFIER.SCHEMA
    assert report["valid"] is True
    assert report["expected_image_id"] == image_id
    assert report["image_config_digest"] == image_id
    assert report["layer_count"] == 1
    assert report["regular_files_read"] == 1
    assert report["content_bytes_read"] == len(b"physical-ai")
    assert (
        report["docker_save_sha256"] == hashlib.sha256(archive.read_bytes()).hexdigest()
    )
    assert core.verification_archive_digest(report) == report["docker_save_sha256"]


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
