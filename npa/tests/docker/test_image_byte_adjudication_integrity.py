"""Boundary coverage for the narrow private evidence-conservation delta."""
from __future__ import annotations

import errno
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "npa/scripts"))
sys.path.insert(0, str(ROOT / "npa/tests/docker"))
import test_image_byte_adjudication as T  # noqa: E402
from image_byte_scan import adjudicate as A  # noqa: E402

W = A.W


@pytest.mark.parametrize("field", ["scope", "layer_ordinal", "entry_ordinal", "tar_offset", "compressed_offset"])
@pytest.mark.parametrize("side", ["record", "finding"])
def test_context_fields_require_exact_type_and_identity(field, side):
    report, rows = T.sample()
    record, finding = rows[2], rows[1]
    value = "layer" if field == "scope" else 0
    record[field] = finding[field] = value
    assert len(A.population(report, rows)) == 4
    changed = record if side == "record" else finding
    changed[field] = 1 if field == "scope" else True
    with pytest.raises(W.ScanError):
        A.population(report, rows)


@pytest.mark.parametrize("side", ["record", "finding"])
def test_missing_context_cannot_rebind_occurrence(side):
    report, rows = T.sample()
    (rows[2] if side == "record" else rows[1]).pop("layer_ordinal")
    with pytest.raises(W.ScanError, match="finding_context_changed"):
        A.population(report, rows)


@pytest.mark.parametrize("body,start,end,line", [(b"", 0, 0, 1), (b"\n", 1, 1, 2), (b"a", 0, 1, 1)])
def test_real_confidentiality_empty_and_eof_coordinates_remain_supported(body, start, end, line):
    receipt = W.C.compile_policy("$" if start == end else "a").scan_record(body)
    found = next(row for row in receipt.findings if row.start_byte == start and row.end_byte == end)
    assert found.start_line == found.end_line == line
    regex = json.loads(json.dumps(asdict(found)))
    record = T.record(1, body, [regex])
    report = {"schema_version": "npa.image-byte-scan.v1", "complete": True,
              "helper_joined": True, "valid": False, "records": 1,
              "scanned_bytes": len(body), "verified_zero_bytes": 0,
              "regular_files": 1, "regular_bytes": len(body), "findings": 1,
              "helper_summary": {"type": "summary", "files": 1,
                                 "bytes": len(body), "findings": 0}}
    assert len(A.population(report, [record])) == 1


def bound_population(codec="gzip", repeat=2):
    descriptor = {"digest": "sha256:" + "b" * 64,
                  "mediaType": "application/vnd.oci.image.layer.v1.tar" + ("+gzip" if codec == "gzip" else "")}
    layers = [{"ordinal": i, "name": "blobs/sha256/" + "b" * 64,
               "size": 10, "diff_id": "sha256:" + "a" * 64,
               "descriptor": descriptor} for i in range(repeat)]
    report = {"regular_files": 2, "regular_bytes": 8,
              "outer": {"headers": 6, "decoded_bytes": 10240, "zero_end_blocks": 2},
              "layers": [{"ordinal": i, "diff_id": "sha256:" + "a" * 64,
                          "compressed_bytes": 10, "compressed_sha256": "b" * 64,
                          "codec": codec, "headers": 1, "decoded_bytes": 10240,
                          "zero_end_blocks": 2} for i in range(repeat)]}
    verifier = {"regular_files_read": 2, "content_bytes_read": 8, "layer_count": repeat}
    return report, verifier, layers, 10240


@pytest.mark.parametrize("codec", ["raw", "gzip"])
def test_ordered_duplicate_layer_occurrences_remain_distinct(codec):
    args = bound_population(codec)
    A.verified_population(*args)
    args[0]["layers"][1]["ordinal"] = 0
    with pytest.raises(W.ScanError, match="layer_binding"):
        A.verified_population(*args)


@pytest.mark.parametrize("target,key,value", [
    ("verification", "regular_files_read", True),
    ("verification", "content_bytes_read", 8.0),
    ("verification", "layer_count", 1),
    ("report", "regular_files", 0), ("report", "regular_bytes", 0),
    ("outer", "decoded_bytes", 1), ("outer", "headers", False),
    ("layer", "ordinal", True), ("layer", "diff_id", "sha256:" + "c" * 64),
    ("layer", "compressed_bytes", 11), ("layer", "compressed_sha256", "c" * 64),
    ("layer", "codec", "raw"), ("layer", "decoded_bytes", -1),
    ("layer", "headers", 1.0), ("layer", "zero_end_blocks", False),
])
def test_native_verifier_and_graph_fields_refuse_contradictions(target, key, value):
    args = bound_population()
    objects = {"report": args[0], "verification": args[1],
               "outer": args[0]["outer"], "layer": args[0]["layers"][1]}
    objects[target][key] = value
    with pytest.raises(W.ScanError):
        A.verified_population(*args)


@pytest.mark.parametrize("when", ["before_link", "after_link"])
def test_replacement_during_publication_never_receives_success(tmp_path, monkeypatch, when):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, ROOT):
        directory, fd = W.create_output(tmp_path / "accepted")
        held = tmp_path / "held-original"
        original = A.os.link

        def swapping(*args, **kwargs):
            if when == "before_link":
                directory.rename(held)
                directory.mkdir(mode=0o700)
            result = original(*args, **kwargs)
            if when == "after_link":
                directory.rename(held)
                directory.mkdir(mode=0o700)
            return result

        monkeypatch.setattr(A.os, "link", swapping)
        try:
            with pytest.raises(W.ScanError, match="output_directory_changed"):
                A.write_result(directory, fd, {"accepted": True})
            assert not list(directory.iterdir())
            assert not list(held.iterdir())
        finally:
            os.close(fd)


def test_held_output_success_is_private_and_exact(tmp_path):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, ROOT):
        directory, fd = W.create_output(tmp_path / "accepted")
        try:
            result = {"accepted": True, "raw_scan_valid": False, "raw_scan_findings": 7}
            A.write_result(directory, fd, result)
            files = list(directory.iterdir())
            assert [p.name for p in files] == ["adjudication.json"]
            assert files[0].stat().st_mode & 0o777 == 0o600
            assert json.loads(files[0].read_bytes()) == result
        finally:
            os.close(fd)


def test_failed_write_cleans_pending_without_success(tmp_path, monkeypatch):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, ROOT):
        directory, fd = W.create_output(tmp_path / "accepted")
        monkeypatch.setattr(A.os, "fsync", lambda _fd: (_ for _ in ()).throw(OSError(errno.ENOSPC, "synthetic")))
        try:
            with pytest.raises(OSError):
                A.write_result(directory, fd, {"accepted": True})
            assert not list(directory.iterdir())
        finally:
            os.close(fd)


@pytest.mark.parametrize("failure", [KeyboardInterrupt, OSError])
def test_interrupted_write_closes_exact_owned_file(tmp_path, monkeypatch, failure):
    tmp_path.chmod(0o700)
    with W.authorized_roots(tmp_path, ROOT):
        directory, fd = W.create_output(tmp_path / "accepted")
        descriptors = []

        def interrupted(target, _data):
            descriptors.append(target)
            raise failure()

        monkeypatch.setattr(A.os, "write", interrupted)
        try:
            with pytest.raises(failure):
                A.write_result(directory, fd, {"accepted": True})
            for target in descriptors:
                with pytest.raises(OSError):
                    os.fstat(target)
            assert not list(directory.iterdir())
        finally:
            os.close(fd)
