"""Verify the OpenPI cuDNN filtering boundary against adversarial installed files."""

from __future__ import annotations

import csv
import hashlib
import importlib.util
from pathlib import Path

import pytest

SOURCE = Path(__file__).resolve().parents[2] / "docker/workbench/openpi/filter_cudnn_runtime.py"


@pytest.fixture
def installed_cudnn(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location("openpi_cudnn_filter", SOURCE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    files = {
        "nvidia/cudnn/lib/libcudnn.so.9": b"\x7fELFsynthetic shared library",
        "nvidia/cudnn/include/cudnn.h": b"synthetic SDK header",
        "nvidia/cudnn/lib/libcudnn_static.a": b"synthetic static SDK",
        module.LICENSE_PATH: b"synthetic unit fixture license",
        f"{module.DIST_INFO}/METADATA": b"Name: nvidia-cudnn-cu12\nVersion: 9.10.2.21\n",
    }
    monkeypatch.setattr(module, "LICENSE_SHA256", hashlib.sha256(files[module.LICENSE_PATH]).hexdigest())
    for name, body in files.items():
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(body)
    record = tmp_path / module.DIST_INFO / "RECORD"
    with record.open("w", newline="") as stream:
        csv.writer(stream).writerows([name, "", ""] for name in [*files, f"{module.DIST_INFO}/RECORD"])
    return module, tmp_path, files


def test_filter_removes_development_bytes_and_preserves_runtime_notice(installed_cudnn):
    module, root, original = installed_cudnn
    result = module.retain_cudnn_runtime(root)
    assert set(result["omitted_development_files"]) == {
        "nvidia/cudnn/include/cudnn.h", "nvidia/cudnn/lib/libcudnn_static.a"
    }
    for name in result["omitted_development_files"]:
        assert not (root / name).exists()
    for name, digest in result["retained_sha256"].items():
        assert (root / name).read_bytes() == original[name]
        assert digest == hashlib.sha256(original[name]).hexdigest()
    with (root / module.DIST_INFO / "RECORD").open(newline="") as stream:
        assert {row[0] for row in csv.reader(stream)}.isdisjoint(result["omitted_development_files"])
    assert module.retain_cudnn_runtime(root)["retained_sha256"] == result["retained_sha256"]


@pytest.mark.parametrize("mutation", ["unknown_recorded", "unknown_unrecorded", "bad_library", "missing_library", "changed_license", "symlink", "duplicate_record"])
def test_filter_refuses_before_any_deletion(installed_cudnn, mutation):
    module, root, _ = installed_cudnn
    record = root / module.DIST_INFO / "RECORD"
    library = root / "nvidia/cudnn/lib/libcudnn.so.9"
    if mutation.startswith("unknown_"):
        unexpected = "nvidia/cudnn/unreviewed.bin"
        (root / unexpected).write_bytes(b"unreviewed SDK payload")
        if mutation == "unknown_recorded":
            with record.open("a", newline="") as stream:
                csv.writer(stream).writerow([unexpected, "", ""])
    elif mutation == "bad_library":
        library.write_bytes(b"not an ELF library")
    elif mutation == "missing_library":
        library.unlink()
    elif mutation == "changed_license":
        (root / module.LICENSE_PATH).write_bytes(b"unreviewed replacement license")
    elif mutation == "symlink":
        target = root / "other-library"
        library.rename(target)
        library.symlink_to(target)
    else:
        with record.open("a", newline="") as stream:
            csv.writer(stream).writerow([module.LICENSE_PATH, "", ""])
    with pytest.raises(ValueError):
        module.retain_cudnn_runtime(root)
    assert (root / "nvidia/cudnn/include/cudnn.h").is_file()
    assert (root / "nvidia/cudnn/lib/libcudnn_static.a").is_file()


def test_filter_rejects_a_second_distribution(installed_cudnn):
    module, root, _ = installed_cudnn
    (root / "nvidia_cudnn_cu12-0.0.0.dist-info").mkdir()
    with pytest.raises(ValueError, match="exact cuDNN distribution"):
        module.retain_cudnn_runtime(root)
