"""Exercise the warehouse acceptance oracle and artifact failure boundaries."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from npa.workflows.antioch_warehouse import runner
from npa.workflows.antioch_warehouse.evidence import PHASES, VIEWS, verify_evidence


def _write(path: Path, value) -> None:
    path.write_text(json.dumps(value))


@pytest.fixture
def evidence(tmp_path):
    """Synthetic oracle inputs; these do not prove simulator execution."""
    measured = {
        "physics_backend": "physx",
        "physics_dt": 1 / 120,
        "stage_prims": 4500,
        "rigid_bodies": 8,
        "completed_batches": 1,
        "total_placements": 6,
        "max_attachment_error_m": 0.015,
        "accumulation_speed_m_s": 0.01,
        "max_conveyor_displacement_m": 4.1,
        "settled": [
            {"carton": i, "error_m": 0.005, "speed_m_s": 0.002} for i in range(6)
        ],
    }
    _write(tmp_path / "validation.json", measured)
    events = [
        {"carton": i, "event": event, "sim_s": i * 20 + j}
        for i in range(6)
        for j, event in enumerate(PHASES)
    ]
    _write(tmp_path / "events.json", events)
    _write(
        tmp_path / "trajectory.json",
        [
            {"sim_s": 0.0, "placed": 0, "cartons": [[0.0, 0.0, 1.0]] * 6},
            {"sim_s": 140.0, "placed": 6, "cartons": [[2.0, 1.0, 1.0]] * 6},
        ],
    )
    pixels = np.tile(np.linspace(20, 200, 1280, dtype=np.uint8), (720, 1))
    for view in VIEWS:
        Image.fromarray(pixels).convert("RGB").save(tmp_path / f"{view}.png")
    return tmp_path


def test_checks_real_pixels_and_hashes_every_required_artifact(evidence):
    report = verify_evidence(evidence)
    assert report["all_passed"]
    assert len(report["checks"]) == 21
    assert len(report["artifacts_sha256"]) == 6
    for name, digest in report["artifacts_sha256"].items():
        assert digest == hashlib.sha256((evidence / name).read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "field,value,check",
    [
        ("physics_backend", "newton", "physx_120hz"),
        ("completed_batches", 0, "complete_batch"),
        ("accumulation_speed_m_s", float("nan"), "stopped_before_pickup"),
        ("max_attachment_error_m", 0.1, "attachment_error_under_4cm"),
        ("max_conveyor_displacement_m", 0.0, "contact_transport_over_3_5m"),
    ],
)
def test_rejects_failed_or_nonfinite_measurements(evidence, field, value, check):
    measured = json.loads((evidence / "validation.json").read_text())
    measured[field] = value
    _write(evidence / "validation.json", measured)
    report = verify_evidence(evidence)
    assert not report["all_passed"]
    assert not report["checks"][check]


@pytest.mark.parametrize("mutation", ["duplicate", "missing", "late_failure"])
def test_requires_every_carton_to_pass(evidence, mutation):
    measured = json.loads((evidence / "validation.json").read_text())
    if mutation == "duplicate":
        measured["settled"][-1]["carton"] = 0
    elif mutation == "missing":
        measured["settled"].pop()
    else:
        measured["settled"][-1]["error_m"] = 0.05
    _write(evidence / "validation.json", measured)
    assert not verify_evidence(evidence)["all_passed"]


def test_requires_ordered_events_for_last_carton_too(evidence):
    events = json.loads((evidence / "events.json").read_text())
    events[-1]["event"] = "vacuum_released"
    _write(evidence / "events.json", events)
    assert not verify_evidence(evidence)["checks"]["ordered_workflow_each_carton"]


@pytest.mark.parametrize("pixels", [0, 120, 255])
def test_rejects_blank_images_without_trusting_metadata(evidence, pixels):
    Image.new("RGB", (1280, 720), (pixels,) * 3).save(evidence / "packing.png")
    assert not verify_evidence(evidence)["all_passed"]


def test_missing_image_is_a_failure(evidence):
    (evidence / "aisle.png").unlink()
    with pytest.raises(FileNotFoundError):
        verify_evidence(evidence)


def test_verify_command_publishes_failed_report_and_raises(evidence, tmp_path):
    Image.new("RGB", (1280, 720)).save(evidence / "packing.png")
    output = tmp_path / "report"
    with pytest.raises(RuntimeError, match="failed acceptance"):
        runner.main(
            ["verify", "--input-path", str(evidence), "--output-path", str(output)]
        )
    assert not json.loads((output / "verification.json").read_text())["all_passed"]


def test_remote_verify_reads_storage_instead_of_producer_verdict(
    evidence, tmp_path, monkeypatch
):
    import shutil

    class Storage:
        def download_directory(self, uri, destination):
            assert uri == "s3://fixture/warehouse/"
            shutil.copytree(evidence, destination, dirs_exist_ok=True)

    monkeypatch.setattr(runner.StorageClient, "from_environment", lambda: Storage())
    output = tmp_path / "report"
    runner.main(
        [
            "verify",
            "--input-path",
            "s3://fixture/warehouse/",
            "--output-path",
            str(output),
        ]
    )
    assert json.loads((output / "verification.json").read_text())["all_passed"]


def test_publication_failure_cannot_be_hidden_by_kit_close(tmp_path, monkeypatch):
    from npa.workflows.antioch_warehouse import runtime

    closed = []
    application = SimpleNamespace(close=lambda: closed.append(True))
    monkeypatch.setattr(runtime, "simulate", lambda *args: application)
    monkeypatch.setattr(runner, "verify_evidence", lambda _: {"all_passed": True})

    def deny(*args):
        raise OSError("storage denied")

    monkeypatch.setattr(runner, "_publish", deny)
    with pytest.raises(OSError, match="storage denied"):
        runner.main(
            ["simulate", "--output-path", str(tmp_path / "output"), "--run-id", "test"]
        )
    assert not closed


def test_simulation_failure_retains_failure_artifact(tmp_path, monkeypatch):
    from npa.workflows.antioch_warehouse import runtime

    def fail(*args):
        raise RuntimeError("native runtime unavailable")

    monkeypatch.setattr(runtime, "simulate", fail)
    output = tmp_path / "output"
    with pytest.raises(RuntimeError, match="runtime unavailable"):
        runner.main(["simulate", "--output-path", str(output), "--run-id", "test"])
    assert json.loads((output / "failure.json").read_text()) == {
        "error_type": "RuntimeError"
    }


def test_module_import_does_not_need_isaac():
    from npa.workflows.antioch_warehouse import runtime

    assert callable(runtime.simulate)


def test_module_failure_survives_native_atexit_success():
    import subprocess
    import sys

    script = """
import atexit, os, runpy
from npa.workflows.antioch_warehouse import runner
atexit.register(lambda: os._exit(0))
def fail():
    raise RuntimeError("native failure must stay failed")
runner.main = fail
runpy.run_module("npa.workflows.antioch_warehouse", run_name="__main__")
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True)
    assert result.returncode == 1
    assert b"native failure must stay failed" in result.stderr


def test_workflow_routes_simulation_to_isaac_and_readback_to_cpu(monkeypatch):
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://fixture/source/npa")
    import yaml
    from npa.orchestration.npa_workflow.spec import load_spec
    from npa.orchestration.npa_workflow.interpreter import build_plan
    from npa.orchestration.npa_workflow.skypilot_render import (
        SkypilotRenderOptions,
        render_skypilot_yaml,
    )

    root = Path(__file__).resolve().parents[3]
    spec = load_spec(root / "workflows/testing/antioch-warehouse.yaml")
    rendered = render_skypilot_yaml(
        spec,
        build_plan(spec, run_id="test"),
        run_id="test",
        options=SkypilotRenderOptions(
            registry="registry.example", materialize_registry_secrets=False
        ),
    )
    simulate, verify = [
        row for row in yaml.safe_load_all(rendered) if row and row.get("run")
    ]
    assert "npa-isaac-lab" in simulate["resources"]["image_id"]
    assert simulate["envs"]["ACCEPT_EULA"] == "Y"
    assert "antioch_warehouse simulate" in simulate["run"]
    assert "antioch_warehouse verify" in verify["run"]
    assert "accelerators" not in verify["resources"]
    assert "ACCEPT_EULA" not in verify["envs"]


def test_final_telemetry_replaces_same_tick_without_rounding_away_time(tmp_path):
    from npa.workflows.antioch_warehouse._scene import _WarehouseCell

    cell = object.__new__(_WarehouseCell)
    cell.output_directory = tmp_path
    cell.t, cell.phase, cell.episode = 1.0004, "SETTLING", 0
    cell.placed, cell.grip_position, cell.samples = [], [0.0, 0.0, 1.0], []
    cell.packages = [
        SimpleNamespace(get_world_pose=lambda: ([0.0, 0.0, 1.0], None))
    ] * 6
    cell.record_sample()
    cell.placed = list(range(6))
    cell.phase = "BATCH_COMPLETE"
    cell.record_sample()
    assert len(cell.samples) == 1
    assert cell.samples[0]["sim_s"] == 1.0004
    assert cell.samples[0]["placed"] == 6
