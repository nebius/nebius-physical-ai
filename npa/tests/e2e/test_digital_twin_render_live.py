"""Verify a materialized real NuRec GPU run and its private offline HTML handoff."""

from __future__ import annotations

import json
import os
from pathlib import Path
import struct
import zipfile

from PIL import Image, ImageStat
import pytest

from npa.workbench.nurec.render_evidence import verified_render_summary

pytestmark = [pytest.mark.e2e, pytest.mark.gpu]


def test_real_gpu_twin_has_bound_scene_media_and_portable_html():
    path = os.environ.get("NPA_DIGITAL_TWIN_LIVE_DIR", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        pytest.skip("requires a materialized operator-owned real GPU run")
    root = Path(path)
    summary = verified_render_summary(root)
    assert summary["render GPU"] == "RTX PRO 6000"
    assert summary["GPU telemetry samples"] > 0
    assert summary["peak GPU utilization (%)"] > 0
    with zipfile.ZipFile(root / "reconstruction" / "last.usdz") as scene:
        assert scene.namelist()
    images = sorted((root / "novel_views").rglob("*.png"))
    assert images
    with Image.open(images[0]) as image:
        assert image.width > 100 and image.height > 100
        assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 1
    html = (root / "reports" / "index.html").read_text()
    assert "connect-src 'none'" in html
    assert "Media hashes verified; GPU activity observed" in html
    encoded = html.split('<script id="preview-data" type="application/json">')[1].split(
        "</script>"
    )[0]
    groups = json.loads(encoded)
    assert {group["title"] for group in groups} >= {"Input capture", "Novel views"}
    assert all(frame["images"] for group in groups for frame in group["frames"])


def test_real_cuda_reference_scene_has_native_gpu_and_portable_outputs():
    from npa.workflows.digital_twin_render import verify_render

    path = os.environ.get("NPA_DIGITAL_TWIN_CUDA_LIVE_DIR", "")
    if os.environ.get("NPA_INTEGRATION_E2E") != "1" or not path:
        pytest.skip("requires a materialized native Blender CUDA run")
    root = Path(path)
    record = verify_render(root)
    assert record["gpu_models"] in [["NVIDIA B200"], ["NVIDIA RTX PRO 6000"]]
    assert record["telemetry"]["sample_count"] > 0
    assert record["telemetry"]["peak_utilization_percent"] > 0
    assert record["telemetry"]["peak_memory_mib"] > 0
    assert (
        len(
            {
                digest
                for name, digest in record["files"].items()
                if name.endswith(".png")
            }
        )
        == record["frame_count"]
    )
    for path in root.glob("frame-*.png"):
        with Image.open(path) as image:
            assert image.size == (1280, 720)
            assert max(ImageStat.Stat(image.convert("RGB")).stddev) > 10
    glb = (root / "scene.glb").read_bytes()
    assert struct.unpack_from("<I", glb, 4)[0] == 2
    assert struct.unpack_from("<I", glb, 8)[0] == len(glb)
    chunk_length = struct.unpack_from("<I", glb, 12)[0]
    scene = json.loads(glb[20 : 20 + chunk_length])
    assert len(scene["meshes"]) > 10 and len(scene["materials"]) >= 5
    html = (root / "index.html").read_text()
    assert "connect-src 'none'" in html
    assert "data:image/jpeg;base64," in html
    assert "Blender Cycles" in html and "authored demonstrator" in html
