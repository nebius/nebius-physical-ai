"""Mutation tests for the RoboTwin runtime-only image-byte boundary."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import tarfile
import zipfile
from pathlib import Path


_SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, _SCRIPTS / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_load("scan_image_wan_payload")
scanner = _load("scan_image_robotwin_payload")


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _zip(members: dict[str, bytes]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w") as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    return stream.getvalue()


def _kinds(findings) -> set[str]:
    return {finding.kind for finding in findings}


def test_private_runtime_and_source_are_allowed(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/robotwin/script/collect_data.py": b"official source",
            "usr/local/lib/python3.10/site-packages/curobo/__init__.py": b"",
            "usr/local/cuda/lib64/libcudart.so.12": b"authorized private runtime",
        },
    )
    config = {
        "history": [
            {"created_by": "FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04"},
            {"created_by": "RUN pip install -r /tmp/robotwin-requirements.lock"},
        ]
    }
    assert scanner.scan(rootfs, config) == []


def test_asset_archives_and_extracted_assets_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/robotwin/assets/objects.zip": b"asset bytes",
            "opt/robotwin/assets/embodiments/aloha-agilex/config.yml": b"x",
        },
    )
    kinds = _kinds(scanner.scan(rootfs, {}))
    assert {"robotwin_asset_archive", "robotwin_extracted_asset"} <= kinds


def test_renamed_nested_archive_is_traversed(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "opt/opaque/payload.bin": _zip(
                {"objects/020_hammer/base0/model_data0.json": b"asset bytes"}
            )
        },
    )
    assert "robotwin_extracted_asset" in _kinds(scanner.scan(rootfs, {}))


def test_cache_outputs_and_build_time_fetch_fail(tmp_path: Path) -> None:
    rootfs = _tar(
        tmp_path / "rootfs.tar",
        {
            "root/.cache/huggingface/hub/datasets--TianxingChen--RoboTwin2.0/ref": b"x",
            "opt/custom-cache/datasets--TianxingChen--RoboTwin2.0/snapshots/ref": b"x",
            "workspace/byof-runs/run/robotwin-native/episode_0/episode_0000000.hdf5": b"x",
        },
    )
    config = {
        "history": [
            {
                "created_by": (
                    "RUN huggingface-cli download TianxingChen/RoboTwin2.0 "
                    "objects.zip"
                )
            }
        ]
    }
    kinds = _kinds(scanner.scan(rootfs, config))
    assert {
        "robotwin_huggingface_cache",
        "robotwin_generated_output",
        "robotwin_asset_fetch_at_build",
    } <= kinds


def test_cli_report_records_a_clean_offline_scan(tmp_path: Path) -> None:
    rootfs = _tar(tmp_path / "rootfs.tar", {"opt/robotwin/LICENSE": b"MIT"})
    output = tmp_path / "report.json"
    assert scanner.main(["--rootfs-tar", str(rootfs), "--output", str(output)]) == 0
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report == {
        "format": "npa_robotwin_image_byte_scan_v1",
        "image": "offline-rootfs",
        "status": "pass",
        "archives_scanned": 1,
        "findings": [],
    }
