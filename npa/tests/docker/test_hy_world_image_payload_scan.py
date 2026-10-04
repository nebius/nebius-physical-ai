"""Mutation tests for the HY-World neutral-image payload scanner."""

from __future__ import annotations

import importlib.util
import io
import subprocess
import sys
import tarfile
from pathlib import Path


def _load(name: str):
    scripts = Path(__file__).resolve().parents[2] / "scripts"
    spec = importlib.util.spec_from_file_location(name, scripts / f"{name}.py")
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


scanner = _load("scan_image_hy_world_payload")
IMAGE_ROOT = Path(__file__).resolve().parents[2] / "docker" / "workbench" / "hy-world"


def _tar(path: Path, members: dict[str, bytes]) -> Path:
    with tarfile.open(path, "w") as archive:
        for name, payload in members.items():
            info = tarfile.TarInfo(name)
            info.size = len(payload)
            archive.addfile(info, io.BytesIO(payload))
    return path


def _kinds(findings) -> set[str]:
    return {finding.kind for finding in findings}


def test_runtime_fetch_plumbing_is_not_payload(tmp_path: Path) -> None:
    rootfs = {
        "usr/local/bin/hy-world-runtime": b"hf download tencent/HY-World-2.0\n",
        "opt/npa/hy-world/asset_contract.py": b"SOURCE_REF = 'df9988'\n",
        "usr/local/lib/python3.12/site-packages/uv/__init__.py": b"",
    }
    findings = scanner.scan(_tar(tmp_path / "clean.tar", rootfs), {"history": []})
    assert findings == [], _kinds(findings)


def test_source_and_weights_are_rejected(tmp_path: Path) -> None:
    rootfs = {
        "opt/runtime/HY-World-2.0/hyworld2/worldgen/video_gen.py": b"source",
        "workspace/model-cache/worldstereo-memory-dmd/model.safetensors": b"weights",
    }
    kinds = _kinds(
        scanner.scan(_tar(tmp_path / "payload.tar", rootfs), {"history": []})
    )
    assert "hy_world_source_tree" in kinds
    assert {"hy_world_or_worldstereo_weight", "checkpoint_or_weight"} & kinds


def test_build_time_fetch_is_rejected(tmp_path: Path) -> None:
    findings = scanner.scan(
        _tar(tmp_path / "history.tar", {"usr/local/bin/hy-world-runtime": b""}),
        {"history": [{"created_by": "RUN hy-world-runtime ensure"}]},
    )
    assert "runtime_bootstrap_at_build" in _kinds(findings)


def test_candidate_build_helper_refuses_direct_push() -> None:
    result = subprocess.run(
        [str(IMAGE_ROOT / "build.sh"), "--push"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "direct image pushes are disabled" in result.stderr


def test_runtime_bootstrap_compiles_upstream_native_extensions_and_pins_hub_refs() -> (
    None
):
    runtime = (IMAGE_ROOT / "hy_world_runtime.sh").read_text(encoding="utf-8")
    assert "gsplat_maskgaussian" in runtime
    assert "third_party/navmesh" in runtime
    assert "PYTORCH3D_REF=" in runtime
    assert "FLASH_ATTN_VERSION=" in runtime
    assert "register_hub_main_ref" in runtime
    assert '"$refs/main"' in runtime
