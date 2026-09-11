"""Exercise the shipped golden command with CPU fakes at the GPU boundary."""

from __future__ import annotations

import ast
import copy
import json
import os
from pathlib import Path
import shlex
from types import SimpleNamespace

import pytest

from npa.smoke.manifest import load_manifest
from npa.workbench.cosmos import nano_video, nano_video_golden, nano_video_server

REAL_RUNTIME = nano_video_server.NanoVideoRuntime


def command_source() -> str:
    argv = shlex.split(load_manifest()["cosmos3-nano-video"].golden_eval.command)
    assert argv == ["python", "-m", "npa.workbench.cosmos.nano_video_golden"]
    return Path(nano_video_golden.__file__).read_text()


@pytest.fixture
def golden(monkeypatch, tmp_path):
    root = tmp_path / "artifacts"
    monkeypatch.setenv("NPA_COSMOS3_GOLDEN_OUTPUT_ROOT", str(root))
    monkeypatch.setenv("NPA_COSMOS3_MODEL_PATH", str(tmp_path / "model"))
    monkeypatch.delenv("NPA_COSMOS3_VIDEO_TOKEN", raising=False)
    events = []
    state = SimpleNamespace(
        root=root,
        events=events,
        failure=None,
        report={
            "status": "succeeded",
            "pipeline": "Cosmos3OmniDiffusersPipeline",
            "dtype": "bfloat16",
            "tensor_parallel_size": 1,
            "guardrails": False,
            "device_peak_used_mib": 100000.0,
            "total_wall_seconds": 50.0,
            "chunks": [
                {"status": "succeeded", "requested_frames": frames,
                 "wall_seconds": 15.0, "inference_seconds": 14.0, "peak_memory_mb": 90000.0}
                for frames in (297, 297, 137)
            ],
        },
    )

    class Runtime:
        def __init__(self):
            events.append(("construct",))
            if state.failure == "cache":
                raise ValueError("private cache exception detail")
            assert os.environ["NPA_COSMOS3_VIDEO_TOKEN"]
            self.endpoint = "http://127.0.0.1:8100"
            self.replica_id = "unit-replica"
            self.process = SimpleNamespace(wait=lambda: events.append(("wait",)))

        async def start(self):
            events.append(("start",))
            if state.failure == "start":
                raise RuntimeError("private startup exception detail")

        async def check_health(self):
            events.append(("health",))

        def close(self):
            events.append(("close",))

    def rollout(**kwargs):
        events.append(("rollout", kwargs))
        if state.failure == "rollout":
            raise RuntimeError("private workload exception detail")
        output = kwargs["output_dir"]
        output.mkdir(parents=True)
        report = copy.deepcopy(state.report)
        nano_video.write_json(output / "report.json", report)
        for name in ["chunk-1.mp4", "chunk-2.mp4", "chunk-3.mp4", "video-30s.mp4"]:
            (output / name).write_bytes(b"unit fixture; decoder boundary is mocked")
        return report

    def validate(path, frames):
        assert path.is_file()
        events.append(("decode", path.name, frames))
        if state.failure == "decode" and frames == 720:
            raise nano_video.NanoVideoError("private decode exception detail")
        return {"valid": True, "full_decode_passed": True, "decoded_frames": frames,
                "fps": 24.0, "width": 832, "height": 480}

    monkeypatch.setattr(nano_video_golden, "NanoVideoRuntime", Runtime)
    monkeypatch.setattr(nano_video_golden, "run_rollout", rollout)
    monkeypatch.setattr(nano_video_golden, "validate_video", validate)
    state.run = nano_video_golden.main
    mask = os.umask(0o077)
    os.umask(mask)
    try:
        yield state
    finally:
        os.umask(mask)


def test_golden_command_runs_generation_decodes_every_artifact_and_cleans_up(golden, capsys):
    golden.run()
    result = json.loads((golden.root / "result.json").read_text())
    assert result["status"] == "succeeded"
    assert len(result["report"]["sha256"]) == len(result["video"]["sha256"]) == 64
    assert result["validation"]["decoded_frames"] == 720
    assert [event[0] for event in golden.events] == [
        "construct", "start", "health", "rollout",
        "decode", "decode", "decode", "decode", "close", "wait",
    ]
    assert [event[1:] for event in golden.events if event[0] == "decode"] == [
        ("chunk-1.mp4", 297), ("chunk-2.mp4", 297),
        ("chunk-3.mp4", 137), ("video-30s.mp4", 720),
    ]
    rollout = next(event[1] for event in golden.events if event[0] == "rollout")
    assert rollout["seed"] == 17 and rollout["prompt"] == nano_video.DEFAULT_PROMPT
    assert rollout["replica_id"] == "unit-replica"
    assert json.loads(capsys.readouterr().out)["status"] == "succeeded"


def test_rerun_preserves_prior_artifacts_and_uses_a_new_output_directory(golden):
    golden.run()
    old = json.loads((golden.root / "result.json").read_text())["run_directory"]
    golden.run()
    new = json.loads((golden.root / "result.json").read_text())["run_directory"]
    assert new != old
    assert (Path(old) / "generation/report.json").is_file()
    assert (Path(new) / "generation/report.json").is_file()


@pytest.mark.parametrize("failure", ["cache", "start", "rollout", "decode"])
def test_failure_replaces_stale_success_and_never_exposes_exception_detail(golden, failure):
    golden.run()
    golden.events.clear()
    golden.failure = failure
    with pytest.raises((RuntimeError, ValueError, nano_video.NanoVideoError)):
        golden.run()
    text = (golden.root / "result.json").read_text()
    assert json.loads(text)["status"] == "failed"
    assert "private" not in text
    if failure == "cache":
        assert [event[0] for event in golden.events] == ["construct"]
    else:
        assert [event[0] for event in golden.events][-2:] == ["close", "wait"]


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        ("report", "status", "running"),
        ("report", "pipeline", "reasoner"),
        ("report", "dtype", "float16"),
        ("report", "tensor_parallel_size", 2),
        ("report", "guardrails", True),
        ("report", "device_peak_used_mib", 0),
        ("report", "total_wall_seconds", True),
        ("chunk", "status", "failed"),
        ("chunk", "requested_frames", 301),
        ("chunk", "wall_seconds", -1),
        ("chunk", "inference_seconds", 0),
        ("chunk", "peak_memory_mb", "unmeasured"),
    ],
)
def test_missing_or_wrong_generation_evidence_cannot_pass(golden, section, field, value):
    target = golden.report if section == "report" else golden.report["chunks"][0]
    target[field] = value
    with pytest.raises(nano_video.NanoVideoError):
        golden.run()
    assert json.loads((golden.root / "result.json").read_text())["status"] == "failed"
    assert [event[0] for event in golden.events][-2:] == ["close", "wait"]


def test_all_imported_npa_modules_are_shipped_in_the_image():
    source = command_source()
    repo = Path(__file__).resolve().parents[3]
    spec = load_manifest()["cosmos3-nano-video"]
    dockerfile = (repo / spec.dockerfile).read_text()
    imported = [node.module for node in ast.walk(ast.parse(source))
                if isinstance(node, ast.ImportFrom) and node.module.startswith("npa.")]
    assert len(imported) == 2
    assert "stage_weights" not in source
    for module in imported:
        assert "src/" + module.replace(".", "/") + ".py" in dockerfile
    assert spec.golden_eval.execution_timeout is None
    assert spec.golden_eval.gpu == "required"
    assert spec.golden_eval.kind == "container-smoke"
    assert "src/npa/workbench/cosmos/nano_video_golden.py" in dockerfile


def test_missing_prestaged_weights_refuses_before_gpu_start(golden, monkeypatch):
    monkeypatch.setattr(nano_video_golden, "NanoVideoRuntime", REAL_RUNTIME)
    started = []

    async def start(self):
        started.append(True)
        raise AssertionError("missing cache must fail before GPU initialization")

    monkeypatch.setattr(REAL_RUNTIME, "start", start)
    with pytest.raises(FileNotFoundError):
        golden.run()
    assert not started
    result = json.loads((golden.root / "result.json").read_text())
    assert result["status"] == "failed" and result["error_type"] == "FileNotFoundError"


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1", 0, -1])
def test_golden_measurements_reject_nonfinite_and_nonpositive_values(value):
    assert not nano_video_golden._positive(value)
