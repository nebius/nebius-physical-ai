"""Exercise Habitat's OCI quarantine verifier with synthetic image bytes."""

from __future__ import annotations

import copy
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile

import pytest

from test_image_byte_scan import file, js, tar_data


ROOT = Path(__file__).resolve().parents[3]
SCRIPTS = ROOT / "npa/scripts"
sys.path.insert(0, str(SCRIPTS))
from image_byte_scan import habitat_sim_verification as H  # noqa: E402


INDEX = "application/vnd.oci.image.index.v1+json"
MANIFEST = "application/vnd.oci.image.manifest.v1+json"
CONFIG = "application/vnd.oci.image.config.v1+json"
LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
INTOTO = "application/vnd.in-toto+json"
PACKAGE = ROOT / "npa/docker/workbench/habitat-sim"
CONTRACT = json.loads((PACKAGE / "runtime-payload.json").read_text())


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _oci(
    tmp_path: Path,
    layers: list[list[tuple]],
    *,
    user: str = "ubuntu",
    diff_ids: list[str] | None = None,
) -> tuple[Path, str]:
    blobs: dict[str, bytes] = {}

    def blob(data: bytes, media: str) -> dict[str, object]:
        digest = "sha256:" + _digest(data)
        blobs["blobs/sha256/" + digest.removeprefix("sha256:")] = data
        return {"mediaType": media, "digest": digest, "size": len(data)}

    raw_layers = [tar_data(rows) for rows in layers]
    packed_layers = [blob(gzip.compress(raw, mtime=0), LAYER) for raw in raw_layers]
    labels = copy.deepcopy(CONTRACT["required_labels"])
    config = blob(
        js(
            {
                "architecture": "amd64",
                "os": "linux",
                "rootfs": {
                    "type": "layers",
                    "diff_ids": diff_ids
                    or ["sha256:" + _digest(raw) for raw in raw_layers],
                },
                "config": {
                    "User": user,
                    "Entrypoint": ["/usr/local/bin/npa-habitat-entrypoint"],
                    "Labels": labels,
                },
            }
        ),
        CONFIG,
    )
    runtime = blob(
        js(
            {
                "schemaVersion": 2,
                "mediaType": MANIFEST,
                "config": config,
                "layers": packed_layers,
            }
        ),
        MANIFEST,
    )
    statement = blob(js({"predicateType": "https://spdx.dev/Document"}), INTOTO)
    attestation_config = blob(
        js(
            {
                "architecture": "unknown",
                "os": "unknown",
                "rootfs": {"type": "layers", "diff_ids": [statement["digest"]]},
            }
        ),
        CONFIG,
    )
    attestation = blob(
        js(
            {
                "schemaVersion": 2,
                "mediaType": MANIFEST,
                "config": attestation_config,
                "layers": [statement],
            }
        ),
        MANIFEST,
    )
    runtime["platform"] = {"os": "linux", "architecture": "amd64"}
    attestation.update(
        platform={"os": "unknown", "architecture": "unknown"},
        annotations={
            "vnd.docker.reference.type": "attestation-manifest",
            "vnd.docker.reference.digest": runtime["digest"],
        },
    )
    index = js(
        {
            "schemaVersion": 2,
            "mediaType": INDEX,
            "manifests": [runtime, attestation],
        }
    )
    expected = "sha256:" + _digest(index)
    files = {
        "oci-layout": b'{"imageLayoutVersion":"1.0.0"}',
        "index.json": index,
        **blobs,
    }
    archive = tmp_path / "habitat.oci.tar"
    archive.write_bytes(tar_data([file(name, data) for name, data in files.items()]))
    return archive, expected


def _required_entries() -> list[tuple]:
    paths = [
        "opt/venv/bin/python3",
        "opt/npa-runtime/npa/workflows/habitat_sim_smoke.py",
        "var/lib/dpkg/status",
        "usr/src/habitat-sim/LICENSE",
        "usr/src/habitat-sim/data/default.physics_config.json",
        "usr/share/doc/npa-habitat-sim/source-manifest.json",
        "usr/share/doc/npa-habitat-sim/licenses.json",
        "usr/share/doc/npa-habitat-sim/REDISTRIBUTION.md",
    ]
    return [
        file(path, b"Package: python3\n" if path.endswith("dpkg/status") else b"exact")
        for path in paths
    ]


def _verify(tmp_path: Path, layers: list[list[tuple]], **kwargs) -> dict[str, object]:
    archive, expected = _oci(tmp_path, layers, **kwargs)
    fd = os.open(archive, os.O_RDONLY)
    try:
        return H.verify(
            fd,
            archive.stat().st_size,
            expected,
            CONTRACT,
            _digest(archive.read_bytes()),
        )
    finally:
        os.close(fd)


def _codes(report: dict[str, object]) -> set[str]:
    return {row["code"] for row in report["findings"]}


def test_valid_attested_oci_has_complete_graph_and_payload_receipt(tmp_path) -> None:
    report = _verify(tmp_path, [_required_entries()])
    assert report["valid"] is True
    assert report["layer_count"] == 1
    assert report["regular_files_read"] == 8
    assert report["installed_package_count"] == 1
    assert report["oci_graph"]["attestation_manifest_count"] == 1


def test_every_layer_rejects_scene_paths_and_known_payload_hashes(tmp_path) -> None:
    contract = copy.deepcopy(CONTRACT)
    contract["forbidden_content_sha256"].append(_digest(b"scene payload"))
    archive, expected = _oci(
        tmp_path,
        [
            [
                file("opt/scene_datasets/skokloster-castle.glb", b"scene payload"),
                *_required_entries(),
            ]
        ],
    )
    fd = os.open(archive, os.O_RDONLY)
    try:
        report = H.verify(
            fd,
            archive.stat().st_size,
            expected,
            contract,
            _digest(archive.read_bytes()),
        )
    finally:
        os.close(fd)
    assert {"forbidden_path", "forbidden_payload_hash"} <= _codes(report)


def test_empty_base_cache_directories_are_allowed_but_cache_bytes_are_not(
    tmp_path,
) -> None:
    directories = [
        file("var/cache/apt", kind=tarfile.DIRTYPE),
        file("var/lib/apt/lists", kind=tarfile.DIRTYPE),
    ]
    assert _verify(tmp_path, [[*directories, *_required_entries()]])["valid"] is True

    cached = file("var/lib/apt/lists/archive.example_Packages", b"package metadata")
    assert "forbidden_path" in _codes(
        _verify(tmp_path, [[*directories, cached, *_required_entries()]])
    )


def test_final_runtime_refuses_builder_package_and_root_user(tmp_path) -> None:
    entries = _required_entries()
    status = next(
        index for index, row in enumerate(entries) if row[0].endswith("dpkg/status")
    )
    entries[status] = file("var/lib/dpkg/status", b"Package: cmake\nPackage: python3\n")
    report = _verify(tmp_path, [entries], user="root")
    assert {"forbidden_runtime_package", "final_user_not_ubuntu"} <= _codes(report)


def test_whiteout_and_required_symlink_cannot_fake_final_payload(tmp_path) -> None:
    required = _required_entries()
    target = "usr/src/habitat-sim/LICENSE"
    lower = required
    upper = [file("usr/src/habitat-sim/.wh.LICENSE")]
    assert "required_path_missing" in _codes(_verify(tmp_path, [lower, upper]))

    replaced = [row for row in required if row[0] != target]
    replaced.append(file(target, kind=tarfile.SYMTYPE, link="/elsewhere"))
    assert "required_path_not_file_or_directory" in _codes(
        _verify(tmp_path, [replaced])
    )


def test_wrong_uncompressed_diff_id_and_malformed_whiteout_fail(tmp_path) -> None:
    with pytest.raises(Exception, match="habitat_oci_layer_diff_id"):
        _verify(tmp_path, [_required_entries()], diff_ids=["sha256:" + "0" * 64])
    with pytest.raises(Exception, match="habitat_oci_malformed_whiteout"):
        _verify(
            tmp_path,
            [[file("opt/.wh.payload", b"nonempty"), *_required_entries()]],
        )


def test_duplicate_canonical_layer_path_fails_closed(tmp_path) -> None:
    entries = _required_entries()
    entries.append(file("./" + entries[0][0], b"duplicate"))
    with pytest.raises(Exception, match="habitat_oci_duplicate_layer_path"):
        _verify(tmp_path, [entries])
