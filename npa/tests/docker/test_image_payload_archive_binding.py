"""Bind full streamed archive identity to the graph verified before scanning."""

import hashlib
import io
import json
import tarfile

import pytest

from test_image_payload_scan import scanner


def _tar(entries):
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        for name, content in entries:
            entry = tarfile.TarInfo(name)
            entry.size = len(content)
            archive.addfile(entry, io.BytesIO(content))
    return stream.getvalue()


@pytest.fixture
def bound_archive(tmp_path):
    layer = _tar([("opt/neutral.txt", b"neutral body")])
    diff_id = "sha256:" + hashlib.sha256(layer).hexdigest()
    config = json.dumps({"rootfs": {"type": "layers", "diff_ids": [diff_id]}}).encode()
    config_sha = hashlib.sha256(config).hexdigest()
    manifest = [{"Config": config_sha + ".json", "Layers": ["layer.tar"]}]
    archive = tmp_path / "image.tar"
    archive.write_bytes(
        _tar(
            [
                ("manifest.json", json.dumps(manifest).encode()),
                (config_sha + ".json", config),
                ("layer.tar", layer),
            ]
        )
    )
    verification = tmp_path / "verification.json"
    verification.write_text(
        json.dumps(
            {
                "schema_version": "npa.curobo.image-verification.v1",
                "valid": True,
                "findings": [],
                "docker_save_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "image_manifest_digest": None,
                "image_config_digest": "sha256:" + config_sha,
                "expected_image_id": "sha256:" + config_sha,
                "verified_layer_diff_ids": [diff_id],
                "layer_count": 1,
            }
        )
    )
    return archive, verification


def test_verified_graph_and_full_stream_identity_are_both_preserved(bound_archive):
    archive, verification = bound_archive
    report = scanner.scan(None, archive, verification_report=verification).to_dict()
    expected = hashlib.sha256(archive.read_bytes()).hexdigest()
    assert report["archive_sha256"] == expected
    assert report["archive_bytes"] == archive.stat().st_size
    assert report["archive_binding"]["docker_save_sha256"] == expected


def test_archive_mutation_after_real_graph_binding_cannot_emit_report(
    bound_archive, monkeypatch
):
    archive, verification = bound_archive
    verify = scanner._verify_archive_binding
    verified = []

    def mutate_after_verification(*args, **kwargs):
        binding = verify(*args, **kwargs)
        verified.append(binding)
        with archive.open("ab") as output:
            output.write(b"changed trailer after graph verification")
        return binding

    monkeypatch.setattr(scanner, "_verify_archive_binding", mutate_after_verification)
    with pytest.raises(RuntimeError, match="differs from complete verification report"):
        scanner.scan(None, archive, verification_report=verification)
    assert len(verified) == 1
