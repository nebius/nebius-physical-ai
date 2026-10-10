"""SeedVR2 NVSHMEM runtime-only filtering controls."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/seedvr2"
DIST_INFO = "nvidia_nvshmem_cu13-3.4.5.dist-info"
HEADER = "nvidia/nvshmem/include/device/nvshmem.cuh"
STATIC = "nvidia/nvshmem/lib/libnvshmem_device.a"
BITCODE = "nvidia/nvshmem/lib/libnvshmem_device.bc"
LIBRARY = "nvidia/nvshmem/lib/libnvshmem_host.so.3"
PLUGIN = "nvidia/nvshmem/lib/nvshmem_bootstrap_uid.so.3"
LICENSE = f"{DIST_INFO}/licenses/License.txt"
OFFICIAL_NOTICE = (
    b"Portions derived from DF-NVSHMEM-prototype\n"
    b"Portions derived from Sandia OpenSHMEM\n"
)


@pytest.fixture
def package(tmp_path):
    spec = importlib.util.spec_from_file_location(
        "seedvr2_nvshmem_filter", IMAGE / "filter_nvshmem_runtime.py"
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    official_license = tmp_path / "NVSHMEM-License-v3.4.5-0.txt"
    official_license.write_bytes(OFFICIAL_NOTICE)
    module.OFFICIAL_LICENSE_SHA256 = hashlib.sha256(
        official_license.read_bytes()
    ).hexdigest()
    files = {
        HEADER: b"synthetic SDK header\n",
        STATIC: b"synthetic SDK archive",
        BITCODE: b"synthetic device bitcode",
        LIBRARY: b"\x7fELFsynthetic runtime library",
        PLUGIN: b"\x7fELFsynthetic runtime plugin",
        LICENSE: b"synthetic NVIDIA notice\n",
        f"{DIST_INFO}/METADATA": (b"Name: nvidia-nvshmem-cu13\nVersion: 3.4.5\n"),
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

    def apply_filter(root):
        return module.filter_nvshmem_runtime(root, official_license)

    return tmp_path, apply_filter, files, write_record


def test_only_shared_runtime_and_notice_remain(package) -> None:
    root, apply_filter, files, _ = package

    report = apply_filter(root)

    assert report["schema_version"] == "npa.seedvr2.nvshmem-runtime.v1"
    assert report["official_product_license"] == {
        "revision": "v3.4.5-0",
        "source": (
            "https://raw.githubusercontent.com/NVIDIA/nvshmem/v3.4.5-0/License.txt"
        ),
        "sha256": hashlib.sha256(OFFICIAL_NOTICE).hexdigest(),
    }
    assert report["omitted_sdk_files"] == [HEADER, STATIC, BITCODE]
    for name in (HEADER, STATIC, BITCODE):
        assert not (root / name).exists()
    for name in (LIBRARY, PLUGIN, LICENSE):
        assert (root / name).read_bytes() == files[name]
        assert (
            report["retained_sha256"][name] == hashlib.sha256(files[name]).hexdigest()
        )
    record = (root / DIST_INFO / "RECORD").read_text()
    assert HEADER not in record and STATIC not in record and BITCODE not in record


@pytest.mark.parametrize(
    "extra",
    [
        "nvidia/nvshmem/share/unreviewed.bin",
        "nvidia/nvshmem/lib/unreviewed.so",
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


def test_unrecorded_payload_fails_before_deletion(package) -> None:
    root, apply_filter, _, _ = package
    (root / "nvidia/nvshmem/lib/hidden.bin").write_bytes(b"unrecorded")

    with pytest.raises(ValueError, match="unrecorded file"):
        apply_filter(root)

    assert (root / HEADER).exists()
