"""Retained base binaries require their actual source/notice obligations."""

import json
from pathlib import Path


PACKAGING = Path(__file__).resolve().parents[2] / "docker/workbench/ncore"


def test_every_retained_debian_binary_has_required_source_or_notice_delivery():
    lock = json.loads((PACKAGING / "base-source-lock.json").read_text())
    sources = {c["id"]: c for c in lock["components"]}
    artifacts = {
        a.get("transformation", {}).get("input_sha256", a["sha256"]): a
        for a in lock["artifacts"]
    }
    for package in lock["debian_binaries"]:
        source = sources[package["source"]]
        assert source["kind"] == "debian-source"
        assert source["license_reason"]
        if source["delivery"] == "notice":
            # Only the reviewed permissive grants permit notice-only delivery.
            # Covered libraries cannot silently lose their corresponding source.
            grants = set(source["license"].split(" AND "))
            assert grants <= {
                "Apache-2.0", "BSD-2-Clause", "BSD-3-Clause", "ISC", "MIT",
                "Zlib", "bzip2-1.0.6", "Beerware", "LicenseRef-Public-Domain",
                "LicenseRef-MIT-SIPB", "LicenseRef-TCP-Wrappers",
            }
            assert grants
            assert not source["artifacts"]
            continue
        assert source["delivery"] == "source"
        filenames = [artifacts[h]["filename"] for h in source["artifacts"]]
        assert any(f.endswith(".dsc") for f in filenames)
        assert any(".tar." in f or f.endswith(".tar.gz") for f in filenames)
    # The final scratch image inherits no superseded binary versions. Actual
    # image coverage still rejects unexpected/unmapped ancestor ELF bytes.
    assert lock["base_diff_ids"] == []
    assert lock["notices"]
    assert lock["cpython_elf_files"]
    assert lock["debian_keyring_sha256"]


def test_native_application_libraries_are_runtime_only():
    dockerfile = (PACKAGING / "Dockerfile").read_text()
    for obsolete in (
        "build_native.py",
        "build_ffmpeg.py",
        "native-sources",
        "libblas3",
        "liblapack3",
        "libgfortran5",
        "libquadmath0",
        "libgomp1",
    ):
        assert obsolete not in dockerfile
    lock = json.loads((PACKAGING / "runtime-lock.json").read_text())
    artifacts = {a["name"]: a for a in lock["artifacts"]}
    assert artifacts["numpy"]["version"] == "1.26.4"
    assert artifacts["scipy"]["version"] == "1.15.2"
    assert "av" not in artifacts
    assert "ffmpeg" not in artifacts
    assert artifacts["numpy"]["filename"].endswith(".whl")
    assert artifacts["scipy"]["filename"].endswith(".whl")
