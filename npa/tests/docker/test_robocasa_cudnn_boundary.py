"""Exercise exact-content cuDNN packaging without removing runtime or notices."""

import base64
import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/robocasa"
SPEC = importlib.util.spec_from_file_location(
    "robocasa_cudnn_boundary", IMAGE / "filter_cudnn_runtime.py"
)
assert SPEC is not None and SPEC.loader is not None
boundary = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(boundary)
DIST = "nvidia_cudnn_cu12-9.20.0.48.dist-info"


def _record_row(name, body):
    digest = (
        base64.urlsafe_b64encode(hashlib.sha256(body).digest()).decode().rstrip("=")
    )
    return [name, f"sha256={digest}", str(len(body))]


def _installed_fixture(tmp_path):
    root = tmp_path / "site"
    manifest = json.loads((IMAGE / "cudnn-runtime-boundary.json").read_text())
    rows = []
    for item in manifest["files"]:
        body = (b"\x7fELF" if item["role"] == "retain_runtime" else b"") + item[
            "path"
        ].encode()
        if item["path"].endswith("/METADATA"):
            body = b"Name: nvidia-cudnn-cu12\nVersion: 9.20.0.48\n"
        path = root / item["path"]
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
        item.update(bytes=len(body), sha256=hashlib.sha256(body).hexdigest())
        rows.append(_record_row(item["path"], body))
    for name, body in (("INSTALLER", b"pip\n"), ("REQUESTED", b"")):
        (root / DIST / name).write_bytes(body)
        rows.append(_record_row(f"{DIST}/{name}", body))
    rows.append([f"{DIST}/RECORD", "", ""])
    with (root / DIST / "RECORD").open("w", newline="") as stream:
        csv.writer(stream).writerows(rows)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(json.dumps(manifest))
    return root, manifest_path, manifest


def test_exact_runtime_and_notices_remain_while_headers_are_omitted(tmp_path):
    root, manifest_path, manifest = _installed_fixture(tmp_path)
    before = {
        item["path"]: (root / item["path"]).read_bytes() for item in manifest["files"]
    }
    receipt = boundary.filter_cudnn_runtime(root, manifest_path)
    omitted = {
        item["path"] for item in manifest["files"] if item["role"] == "omit_header"
    }
    assert len(omitted) == 14 and set(receipt["omitted_headers"]) == omitted
    for name, body in before.items():
        if name in omitted:
            assert not (root / name).exists()
        else:
            assert (root / name).read_bytes() == body
            assert receipt["retained_sha256"][name] == hashlib.sha256(body).hexdigest()
    with (root / DIST / "RECORD").open(newline="") as stream:
        recorded = {row[0] for row in csv.reader(stream)}
    assert recorded == (set(before) - omitted) | boundary._GENERATED


@pytest.mark.parametrize(
    "fault",
    ["bytes", "missing", "unknown", "symlink", "fifo", "record", "version", "notice"],
)
def test_bad_installations_reject_before_any_header_is_removed(tmp_path, fault):
    root, manifest_path, manifest = _installed_fixture(tmp_path)
    runtime = root / "nvidia/cudnn/lib/libcudnn.so.9"
    if fault == "bytes":
        runtime.write_bytes(b"altered runtime")
    elif fault in {"missing", "symlink", "fifo"}:
        runtime.unlink()
        if fault == "symlink":
            outside = tmp_path / "outside"
            outside.write_bytes(b"preserve outside")
            runtime.symlink_to(outside)
        elif fault == "fifo":
            os.mkfifo(runtime)
    elif fault == "unknown":
        (runtime.parent / "unreviewed.so").write_bytes(b"unreviewed")
    elif fault == "record":
        with (root / DIST / "RECORD").open("a") as stream:
            stream.write("duplicate,,\n")
    elif fault == "version":
        (root / DIST / "METADATA").write_text(
            "Name: nvidia-cudnn-cu12\nVersion: 9.21\n"
        )
    elif fault == "notice":
        (root / DIST / "licenses/License.txt").unlink()
    with pytest.raises(ValueError):
        boundary.filter_cudnn_runtime(root, manifest_path)
    assert all(
        (root / item["path"]).is_file()
        for item in manifest["files"]
        if item["role"] == "omit_header"
    )
    if fault == "symlink":
        assert outside.read_bytes() == b"preserve outside"


@pytest.mark.parametrize("fault", ["duplicate", "role", "escape", "digest"])
def test_bad_manifest_or_record_digests_reject_before_deletion(tmp_path, fault):
    root, manifest_path, manifest = _installed_fixture(tmp_path)
    if fault == "duplicate":
        manifest["files"].append(manifest["files"][0])
    elif fault == "role":
        manifest["files"][0]["role"] = "accept_unknown"
    elif fault == "escape":
        manifest["files"][0]["path"] = "../outside"
    else:
        record = root / DIST / "RECORD"
        record.write_text(record.read_text().replace("sha256=", "sha512=", 1))
    manifest_path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        boundary.filter_cudnn_runtime(root, manifest_path)
    assert (root / "nvidia/cudnn/include/cudnn.h").is_file()


def test_parent_symlink_never_removes_outside_headers(tmp_path):
    root, manifest_path, _ = _installed_fixture(tmp_path)
    vendor = root / "nvidia"
    outside = tmp_path / "outside-vendor"
    vendor.rename(outside)
    vendor.symlink_to(outside, target_is_directory=True)
    marker = outside / "cudnn/include/cudnn.h"
    original = marker.read_bytes()
    with pytest.raises(ValueError, match="symlink"):
        boundary.filter_cudnn_runtime(root, manifest_path)
    assert marker.read_bytes() == original


def test_runtime_filter_executes_in_the_original_install_layer():
    text = (IMAGE / "Dockerfile").read_text()
    install = text.index("-r /opt/robocasa/locks/requirements.lock")
    filtered = text.index(
        "&& python /opt/robocasa/derivative/filter_cudnn_runtime.py", install
    )
    assert "\nRUN " not in text[install:filtered]
    assert "\nFROM nvidia/cuda:12.9.1-base-ubuntu22.04@sha256:" in text
    assert "FROM nvidia/cuda:12.4.1-cudnn-devel" not in text
