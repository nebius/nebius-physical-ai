"""Unit coverage for the report writer, not a HY-World inference substitute."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path


def _load_report_module():
    script = (
        Path(__file__).resolve().parents[2]
        / "docker"
        / "workbench"
        / "hy-world"
        / "hy_world_report.py"
    )
    spec = importlib.util.spec_from_file_location("npa_hy_world_report_test", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


REPORT = _load_report_module()


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_report_requires_hashed_decodable_video_and_verifies_rrd(
    tmp_path: Path,
) -> None:
    """The synthetic clip here exercises only Rerun serialization mechanics."""

    video = tmp_path / "generated-camera.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "testsrc=size=64x64:rate=3",
            "-frames:v",
            "3",
            "-pix_fmt",
            "yuv420p",
            str(video),
        ],
        check=True,
    )
    evidence = {
        "pipeline_mode": "image_to_world",
        "capability": "hy_world_2_image_conditioned_world_generation",
        "rendered_camera_dataset": {
            "sha256": _sha256(video),
            "decode": {"decoded_frames": 3},
        },
    }
    evidence_path = tmp_path / "hy_world_image_to_world.json"
    evidence_path.write_text(json.dumps(evidence), encoding="utf-8")
    rrd, manifest = tmp_path / "scene.rrd", tmp_path / "scene_rrd_manifest.json"

    assert (
        REPORT.main(
            [
                "--evidence",
                str(evidence_path),
                "--video",
                str(video),
                "--rrd",
                str(rrd),
                "--manifest",
                str(manifest),
                "--run-id",
                "hy-world-report-unit-test",
            ]
        )
        == 0
    )
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    assert payload["status"] == "verified"
    assert payload["run_id"] == "hy-world-report-unit-test"
    assert payload["rrd"]["sha256"] == _sha256(rrd)
    assert payload["rerun_stats"]["num_rows"] > 0
