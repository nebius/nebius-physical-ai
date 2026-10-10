"""Encoded layers are graph-bound containers, not whole logical file records."""

import io
import json
import tarfile
from pathlib import Path

import pytest

from test_image_byte_scan import (
    CHECKOUT,
    W,
    file,
    fixture,
    js,
    raw_tar_entry,
    run,
    tar_data,
    write,
)


@pytest.fixture(autouse=True)
def explicit_roots(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, CHECKOUT):
        yield


class MetadataSink:
    """No detection claims: only observe tar admission before body dispatch."""

    def data(self, *_args):
        pass

    def zeros(self, *_args):
        pass

    def flush_zeros(self):
        pass


def _large_header(name):
    member = tarfile.TarInfo(name)
    member.size = W.COMPLETE_RECORD_LIMIT + 1
    return member.tobuf(format=tarfile.GNU_FORMAT)


def test_declared_encoded_layer_above_one_gib_reaches_bound_handler():
    class ReachedHandler(Exception):
        pass

    name = "blobs/sha256/" + "a" * 64

    def handler(_reader, size, observed_name, _context):
        assert size == (1 << 30) + 1
        assert observed_name == name
        raise ReachedHandler

    # This tests admission only, not the absent body or a complete image scan.
    with pytest.raises(ReachedHandler):
        W.walk_tar(
            io.BytesIO(_large_header(name)),
            MetadataSink(),
            {"scope": "outer"},
            handler,
            encoded_layer_sizes={name: (1 << 30) + 1},
        )


@pytest.mark.parametrize("scope", ["outer", "layer"])
def test_unknown_outer_and_decoded_large_records_rejected_before_handler(scope):
    with pytest.raises(W.ScanError, match="^complete_record_limit$"):
        W.walk_tar(
            io.BytesIO(_large_header("blobs/sha256/" + "a" * 64)),
            MetadataSink(),
            {"scope": scope},
            lambda *_args: pytest.fail("unadmitted record body reached handler"),
        )


def test_encoded_size_binding_and_outer_only_scope_are_required():
    for scope, size, expected in (
        ("outer", 1 << 30, "encoded_layer_size_binding"),
        ("layer", (1 << 30) + 1, "encoded_layer_scope"),
    ):
        with pytest.raises(W.ScanError, match=f"^{expected}$"):
            W.walk_tar(
                io.BytesIO(_large_header("layer.tar")),
                MetadataSink(),
                {"scope": scope},
                lambda *_args: pytest.fail("invalid binding reached handler"),
                encoded_layer_sizes={"layer.tar": size},
            )


def test_encoded_admission_uses_effective_name_not_misleading_header():
    extension = raw_tar_entry(
        "long-name", b"unbound.tar\0", kind=tarfile.GNUTYPE_LONGNAME
    )
    with pytest.raises(W.ScanError, match="^complete_record_limit$"):
        W.walk_tar(
            io.BytesIO(extension + _large_header("bound.tar")),
            MetadataSink(),
            {"scope": "outer"},
            lambda *_args: pytest.fail("misleading name reached handler"),
            encoded_layer_sizes={"bound.tar": (1 << 30) + 1},
        )


def _rewrite_outer(authorization, transform):
    archive_path = Path(authorization["archive"]["path"])
    with tarfile.open(archive_path) as archive:
        entries = [
            file(member.name, archive.extractfile(member).read()) for member in archive
        ]
    authorization["archive"] = write(archive_path, tar_data(transform(entries)))
    report_path = Path(authorization["verification_report"]["path"])
    report = json.loads(report_path.read_text())
    report["docker_save_sha256"] = authorization["archive"]["sha256"]
    authorization["verification_report"] = write(report_path, js(report))


def test_graph_bound_large_encoded_layer_gets_full_decoded_accounting(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(W, "COMPLETE_RECORD_LIMIT", 16 * 1024)
    entries = [file(f"opt/file-{index}", b"neutral body") for index in range(32)]
    authorization = fixture(tmp_path, codec="raw", entries=entries)
    report, rows = run(tmp_path, authorization)
    assert report["valid"] and report["complete"]
    assert report["regular_files"] == 32
    assert report["layers"][0]["decoded_bytes"] > W.COMPLETE_RECORD_LIMIT
    assert any(row.get("kind") == "layer_regular_content" for row in rows)


def test_large_encoded_layer_does_not_admit_oversized_decoded_file(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(W, "COMPLETE_RECORD_LIMIT", 16 * 1024)
    authorization = fixture(
        tmp_path, entries=[file("large", b"x" * (16 * 1024 + 1))], codec="raw"
    )
    report, _ = run(tmp_path, authorization)
    assert not report["complete"] and not report["valid"]
    assert report["failure_code"] == "complete_record_limit"


@pytest.mark.parametrize("oversized_file", [False, True])
def test_real_generic_verification_preserves_encoded_and_decoded_record_boundary(
    tmp_path, monkeypatch, oversized_file
):
    from image_byte_scan import docker_save_verification

    limit = 16 * 1024
    entries = [file(f"opt/file-{index}", b"neutral body") for index in range(32)]
    if oversized_file:
        entries.append(file("opt/oversized", b"x" * (limit + 1)))
    authorization = fixture(tmp_path, codec="raw", entries=entries)
    # Obtain the real verifier's report before reducing only the scanner record
    # cap. A graph proof must not exempt any decoded file from that cap.
    verification = docker_save_verification.verify(
        Path(authorization["archive"]["path"]), authorization["expected_image_id"]
    )
    assert verification["schema_version"] == "npa.docker-save.image-verification.v1"
    assert verification["valid"] is True
    authorization["verification_report"] = write(
        Path(authorization["verification_report"]["path"]), js(verification)
    )
    monkeypatch.setattr(W, "COMPLETE_RECORD_LIMIT", limit)
    report, rows = run(tmp_path, authorization)
    if oversized_file:
        assert not report["valid"] and not report["complete"]
        assert report["failure_code"] == "complete_record_limit"
    else:
        assert report["valid"] and report["complete"]
        assert report["regular_files"] == 32
        assert report["layers"][0]["decoded_bytes"] > limit
        assert any(row.get("kind") == "layer_regular_content" for row in rows)


@pytest.mark.parametrize("mutation", ["unknown", "size", "digest"])
def test_encoded_layer_exception_preserves_graph_and_unknown_file_gates(
    tmp_path, monkeypatch, mutation
):
    monkeypatch.setattr(W, "COMPLETE_RECORD_LIMIT", 16 * 1024)
    entries = [file(f"opt/file-{index}", b"neutral body") for index in range(32)]
    authorization = fixture(tmp_path, codec="raw", entries=entries)

    def transform(entries):
        if mutation == "unknown":
            return [
                *entries,
                file("blobs/sha256/" + "b" * 64, b"x" * (16 * 1024 + 1)),
            ]
        name, payload, kind, link, pax = entries[-1]
        changed = payload + b"x" if mutation == "size" else b"x" + payload[1:]
        return [*entries[:-1], (name, changed, kind, link, pax)]

    _rewrite_outer(authorization, transform)
    report, _ = run(tmp_path, authorization)
    assert not report["complete"] and not report["valid"]
    assert (
        report["failure_code"]
        == {
            "unknown": "complete_record_limit",
            "size": "graph_layer_descriptor",
            "digest": "layer_compressed_digest",
        }[mutation]
    )
