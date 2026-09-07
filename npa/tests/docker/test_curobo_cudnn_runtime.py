"""cuDNN runtime-only packaging, preserving notices and rejecting unseen bytes."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path
import re

import pytest


ROOT = Path(__file__).resolve().parents[3]
IMAGE = ROOT / "npa/docker/workbench/curobo"
DIST_INFO = "nvidia_cudnn_cu13-9.13.0.50.dist-info"
HEADER = "nvidia/cudnn/include/cudnn.h"
LIBRARY = "nvidia/cudnn/lib/libcudnn.so.9"
LICENSE = f"{DIST_INFO}/licenses/License.txt"


@pytest.fixture
def package(tmp_path):
    spec = importlib.util.spec_from_file_location("curobo_cudnn_filter", IMAGE / "filter_cudnn_runtime.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {
        HEADER: b"synthetic SDK header\n",
        LIBRARY: b"\x7fELFsynthetic test library",
        "nvidia/cudnn/lib/libcudnn_static.a": b"synthetic SDK archive",
        LICENSE: b"synthetic notice must remain byte-identical\n",
        f"{DIST_INFO}/METADATA": b"Name: nvidia-cudnn-cu13\nVersion: 9.13.0.50\n",
    }
    for name, payload in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)

    def record(extra=()):
        with (tmp_path / DIST_INFO / "RECORD").open("w", newline="") as stream:
            csv.writer(stream).writerows((name, "", "") for name in [*files, *extra, f"{DIST_INFO}/RECORD"])

    record()
    return tmp_path, module.filter_cudnn_runtime, files, record


def test_only_runtime_and_license_bytes_remain_identical(package):
    root, apply_filter, files, _ = package
    unrelated = root / "other_package.py"
    unrelated.write_text("unrelated dependency")
    report = apply_filter(root)
    assert report["omitted_sdk_files"] == [HEADER, "nvidia/cudnn/lib/libcudnn_static.a"]
    assert not (root / HEADER).exists()
    assert not (root / "nvidia/cudnn/lib/libcudnn_static.a").exists()
    for name in (LIBRARY, LICENSE):
        assert (root / name).read_bytes() == files[name]
        assert report["retained_sha256"][name] == hashlib.sha256(files[name]).hexdigest()
    assert unrelated.read_text() == "unrelated dependency"
    recorded = (root / DIST_INFO / "RECORD").read_text()
    assert HEADER not in recorded and ".a," not in recorded
    assert LIBRARY in recorded and LICENSE in recorded


@pytest.mark.parametrize(
    "extra",
    ["nvidia/cudnn/include/unreviewed.hpp", "nvidia/cudnn/hidden/cudnn.h", "nvidia/other/cudnn.h"],
)
def test_new_registered_payload_fails_before_any_deletion(package, extra):
    root, apply_filter, _, record = package
    path = root / extra
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("unreviewed SDK payload")
    record([extra])
    with pytest.raises(ValueError, match="Unreviewed file"):
        apply_filter(root)
    assert (root / HEADER).exists()


def test_unrecorded_namespace_payload_fails_before_deletion(package):
    root, apply_filter, _, _ = package
    (root / "nvidia/cudnn/lib/hidden.h").write_text("unrecorded SDK payload")
    with pytest.raises(ValueError, match="Unrecorded file"):
        apply_filter(root)
    assert (root / HEADER).exists()


def test_runtime_symlink_cannot_escape_payload_boundary(package, tmp_path):
    root, apply_filter, _, _ = package
    target = tmp_path / "outside.so"
    target.write_bytes(b"\x7fELFoutside")
    (root / LIBRARY).unlink()
    (root / LIBRARY).symlink_to(target)
    with pytest.raises(ValueError, match="regular contained files"):
        apply_filter(root)
    assert (root / HEADER).exists()


@pytest.mark.parametrize("change", ["license", "runtime", "version"])
def test_missing_notice_or_invalid_runtime_fails_closed(package, change):
    root, apply_filter, _, _ = package
    if change == "license":
        (root / LICENSE).unlink()
    elif change == "runtime":
        (root / LIBRARY).write_bytes(b"not an ELF object")
    else:
        (root / DIST_INFO / "METADATA").write_text("Name: nvidia-cudnn-cu13\nVersion: 10.0\n")
    with pytest.raises(ValueError):
        apply_filter(root)
    assert (root / HEADER).exists()


def test_docker_never_commits_unfiltered_cudnn_in_any_layer():
    text = (IMAGE / "Dockerfile").read_text()
    assert "-cudnn-" not in next(line for line in text.splitlines() if line.startswith("FROM "))
    instructions = re.sub(r"\\\n\s*", " ", text).splitlines()
    install = next(line for line in instructions if line.startswith("RUN pip install "))
    assert "--require-hashes" in install
    assert "&& python /opt/filter_cudnn_runtime.py" in install
    assert "&& pip check" in install
