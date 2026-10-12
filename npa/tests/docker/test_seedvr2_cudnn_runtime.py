"""SeedVR2 cuDNN runtime-only filtering controls."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/seedvr2"
DIST_INFO = "nvidia_cudnn_cu13-9.20.0.48.dist-info"
HEADER = "nvidia/cudnn/include/cudnn.h"
STATIC = "nvidia/cudnn/lib/libcudnn_static.a"
LIBRARY = "nvidia/cudnn/lib/libcudnn.so.9"
LICENSE = f"{DIST_INFO}/licenses/License.txt"


@pytest.fixture
def package(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "seedvr2_cudnn_filter", IMAGE / "filter_cudnn_runtime.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {
        HEADER: b"synthetic SDK header\n",
        STATIC: b"synthetic SDK archive",
        LIBRARY: b"\x7fELFsynthetic runtime library",
        LICENSE: b"synthetic notice\n",
        f"{DIST_INFO}/METADATA": (b"Name: nvidia-cudnn-cu13\nVersion: 9.20.0.48\n"),
    }
    for name, payload in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def write_record(extra=()):
        with (tmp_path / DIST_INFO / "RECORD").open("w", newline="") as stream:
            csv.writer(stream).writerows(
                (name, "", "") for name in [*files, *extra, f"{DIST_INFO}/RECORD"]
            )

    write_record()
    return tmp_path, module.filter_cudnn_runtime, files, write_record


def test_only_shared_runtime_and_notice_remain(package) -> None:
    root, apply_filter, files, _ = package

    report = apply_filter(root)

    assert report["schema_version"] == "npa.seedvr2.cudnn-runtime.v1"
    assert report["omitted_sdk_files"] == [HEADER, STATIC]
    assert not (root / HEADER).exists()
    assert not (root / STATIC).exists()
    for name in (LIBRARY, LICENSE):
        assert (root / name).read_bytes() == files[name]
        assert (
            report["retained_sha256"][name] == hashlib.sha256(files[name]).hexdigest()
        )
    record = (root / DIST_INFO / "RECORD").read_text()
    assert HEADER not in record and STATIC not in record


@pytest.mark.parametrize(
    "extra",
    [
        "nvidia/cudnn/include/unreviewed.hpp",
        "nvidia/cudnn/hidden/payload.bin",
    ],
)
def test_unreviewed_registered_payload_fails_before_deletion(
    package, extra: str
) -> None:
    root, apply_filter, _, write_record = package
    path = root / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"unreviewed")
    write_record([extra])

    with pytest.raises(ValueError, match="unreviewed file"):
        apply_filter(root)

    assert (root / HEADER).exists()


def test_unrecorded_namespace_payload_fails_before_deletion(package) -> None:
    root, apply_filter, _, _ = package
    (root / "nvidia/cudnn/lib/hidden.bin").write_bytes(b"unrecorded")

    with pytest.raises(ValueError, match="unrecorded file"):
        apply_filter(root)

    assert (root / HEADER).exists()


def test_unrecorded_distribution_metadata_fails_before_deletion(package) -> None:
    root, apply_filter, _, _ = package
    (root / DIST_INFO / "hidden.bin").write_bytes(b"unrecorded")

    with pytest.raises(ValueError, match="distribution metadata"):
        apply_filter(root)

    assert (root / HEADER).exists()
