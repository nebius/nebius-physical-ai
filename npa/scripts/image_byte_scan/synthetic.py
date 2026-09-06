"""Synthetic archive construction for the explicit native regression command.

These fixture graph receipts are never evidence for an actual image.
"""
from __future__ import annotations
import gzip
import hashlib
import io
import json
import tarfile
from . import core as W

def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path, data):
    path.write_bytes(data)
    path.chmod(0o600)
    return {"path": str(path), "sha256": digest(data)}


def js(value):
    return json.dumps(value).encode()


def tar_data(entries, *, format=tarfile.PAX_FORMAT):
    result = io.BytesIO()
    with tarfile.open(fileobj=result, mode="w", format=format) as archive:
        for name, data, kind, link, pax in entries:
            item = tarfile.TarInfo(name)
            item.type, item.linkname, item.pax_headers = kind, link, pax
            item.size = len(data) if kind in {tarfile.REGTYPE, tarfile.AREGTYPE} else 0
            archive.addfile(item, io.BytesIO(data) if item.isfile() else None)
    return result.getvalue()


def file(name, data=b"", *, kind=tarfile.REGTYPE, link="", pax=None):
    return name, data, kind, link, pax or {}


def fixture_tools_receipt(authorization, directory, ready=None):
    ready = {} if ready is None else ready
    authorization["helper"]["ready_sha256"] = W.sha(W.canonical(ready))
    receipt = {"schema_version": "npa.image-byte-scan-tools.v1", "source": W.current_go_sources(),
               "helper": {key: value for key, value in authorization["helper"].items() if key != "ready_sha256"},
               "config": authorization["config"], "ready": write(directory / "fixture-ready.json", js(ready))}
    authorization["tools_receipt"] = write(directory / "fixture-tools.json", js(receipt))


def fixture(tmp_path, *, entries=None, raw=None, compressed=None, literals=None, policy="exact-substring-v1", repeat=1, codec="gzip"):
    entries = entries if entries is not None else [file("opt/sample", b"actual synthetic neutral body")]
    raw = raw if raw is not None else tar_data(entries)
    data = (gzip.compress(raw, mtime=0) if codec == "gzip" else raw) if compressed is None else compressed
    config = js({"rootfs": {"type": "layers", "diff_ids": ["sha256:" + digest(raw)] * repeat}})
    config_id = "sha256:" + digest(config)
    manifest = {"schemaVersion": 2, "mediaType": "application/vnd.oci.image.manifest.v1+json",
                "config": {"mediaType": "application/vnd.oci.image.config.v1+json", "digest": config_id, "size": len(config)},
                "layers": [{"mediaType": "application/vnd.oci.image.layer.v1.tar" + ("+gzip" if codec == "gzip" else ""), "digest": "sha256:" + digest(data), "size": len(data)}] * repeat}
    manifest_bytes = js(manifest)
    image_id = "sha256:" + digest(manifest_bytes)
    index = js({"schemaVersion": 2, "manifests": [{"mediaType": manifest["mediaType"], "digest": image_id, "size": len(manifest_bytes)}]})
    saved = js([{"Config": "blobs/sha256/" + digest(config), "Layers": ["blobs/sha256/" + digest(data)] * repeat}])
    outer = tar_data([file("manifest.json", saved), file("index.json", index), file("oci-layout", b'{"imageLayoutVersion":"1.0.0"}'),
                      file("blobs/sha256/" + digest(config), config), file("blobs/sha256/" + digest(manifest_bytes), manifest_bytes),
                      file("blobs/sha256/" + digest(data), data)])
    archive_binding = write(tmp_path / "image.tar", outer)
    regulars = [row for row in entries if row[2] in {tarfile.REGTYPE, tarfile.AREGTYPE}]
    verification = {"schema_version": "npa.curobo.image-verification.v1", "valid": True, "expected_image_id": image_id,
                    "image_config_digest": config_id, "image_manifest_digest": image_id, "docker_save_sha256": digest(outer),
                    "verified_layer_diff_ids": ["sha256:" + digest(raw)] * repeat, "layer_count": repeat,
                    "regular_files_read": len(regulars) * repeat, "content_bytes_read": sum(len(row[1]) for row in regulars) * repeat}
    inventory = write(tmp_path / "literals.json", js({"literals": literals or ["private-operator-marker"]}))
    inventory["matching_policy"] = policy
    authorization = {"schema_version": "npa.image-byte-scan-authorization.v1", "accepted_verification": True,
                     "archive": archive_binding, "verification_report": write(tmp_path / "verification.json", js(verification)),
                     "expected_image_id": image_id, "helper": {**write(tmp_path / "helper-fixture", b"Synthetic framing oracle input"), "ready_sha256": "0" * 64},
                     "config": write(tmp_path / "config-fixture", W.bound_bytes({"path": str(W._ROOTS.get()[1] / ".gitleaks.toml"),
                                      "sha256": W.sha((W._ROOTS.get()[1] / ".gitleaks.toml").read_bytes())})),
                     "literal_inventory": inventory, "sources": W.source_bindings()}
    fixture_tools_receipt(authorization, tmp_path)
    return authorization


def optional_gzip(raw, *, flags=0, extra=b"AB\x03\x00oneXY\x03\x00two", name=b"advisory.tar", comment=b"advisory comment"):
    import struct
    import zlib

    header = bytearray(b"\x1f\x8b\x08" + bytes([flags]) + b"\0" * 4 + b"\0\xff")
    if flags & 4:
        header.extend(struct.pack("<H", len(extra)) + extra)
    if flags & 8:
        header.extend(name + b"\0")
    if flags & 16:
        header.extend(comment + b"\0")
    if flags & 2:
        header.extend(struct.pack("<H", zlib.crc32(header) & 0xFFFF))
    compressor = zlib.compressobj(wbits=-15)
    body = compressor.compress(raw) + compressor.flush()
    return bytes(header) + body + struct.pack("<II", zlib.crc32(raw), len(raw) & 0xFFFFFFFF), bytes(header)
