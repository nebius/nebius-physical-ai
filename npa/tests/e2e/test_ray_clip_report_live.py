"""Convert real downloaded native Ray CLIP CUDA results; never provision implicitly."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

pytestmark = pytest.mark.e2e


@pytest.mark.skipif(not os.environ.get("NPA_RAY_CLIP_RESULTS"), reason="Select real downloaded CLIP results explicitly")
def test_real_cuda_results_convert_and_decode(tmp_path):
    """Require CUDA actor evidence and independently verify actual RRD bytes."""
    from rerun.recording import load_recording

    root = Path(os.environ["NPA_RAY_CLIP_RESULTS"])
    source = Path(__file__).parents[2] / "workflows/workbench/ray-clip-development/report.py"
    report = json.loads((root / "report.json").read_text())
    actors = report.get("actors", report.get("model_initializations"))
    assert actors
    for actor in actors:
        assert actor["gpu_ids"]
        assert actor["cuda"]
        assert actor["gpu_name"]
        assert actor["model_load_seconds"] > 0
    final = report.get("final_actors", actors)
    assert sum(actor["inference_calls"] for actor in final) > 0
    output = tmp_path / "clip.rrd"
    converted = subprocess.run([sys.executable, str(source), "--input-path", str(root),
                                "--output-path", str(output), "--run-id", "clip-live-validation"],
                               capture_output=True, text=True)
    assert converted.returncode == 0, "Converter failed; inspect the private result artifacts"
    receipt = json.loads(converted.stdout)
    assert receipt["records"] == report["records"]
    cli = Path(sys.executable).parent / "rerun"
    verify = subprocess.run([str(cli), "rrd", "verify", str(output)], capture_output=True)
    assert verify.returncode == 0, "Rerun CLI rejected the recording"
    printed = subprocess.run([str(cli), "rrd", "print", "-vv", str(output)], capture_output=True, text=True)
    assert printed.returncode == 0
    assert "npa.ray-clip-development" in printed.stdout
    assert "clip-live-validation" in printed.stdout
    indices, entities = [], set()
    for chunk in load_recording(output).chunks():
        entity = str(chunk.entity_path)
        entities.add(entity)
        if entity == "/vectors/norm":
            indices.extend(chunk.to_record_batch().column("record_id").to_pylist())
    assert indices == list(range(report["records"]))
    assert {"/images/original", "/images/crop", "/vectors/embedding", "/provenance/run"} <= entities
    if "model_initializations" in report:
        assert "/checkpoints/materialized" in entities
        if report["recovery"] is not None:
            assert "/recovery/checkpoint_replay" in entities
