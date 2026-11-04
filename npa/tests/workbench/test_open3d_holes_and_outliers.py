"""Fail-closed contracts for the withdrawn holes/outliers measurement."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

EVIDENCE = Path(__file__).resolve().parents[3] / "docs" / "evidence" / "open3d"
HARNESS_PATH = EVIDENCE / "harness" / "holes_and_outliers.py"
REPORT_PATH = EVIDENCE / "holes-and-outliers.json"
RECORD_PATH = EVIDENCE / "capability-record-holes-and-outliers.json"


def _harness():
    spec = importlib.util.spec_from_file_location("_holes_harness", HARNESS_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    stub = types.ModuleType("open3d")
    saved = sys.modules.get("open3d")
    sys.modules["open3d"] = stub
    try:
        spec.loader.exec_module(module)
    finally:
        if saved is None:
            del sys.modules["open3d"]
        else:
            sys.modules["open3d"] = saved
    return module


def test_unrepeatable_quantitative_claims_are_withdrawn() -> None:
    report = json.loads(REPORT_PATH.read_text())
    record = json.loads(RECORD_PATH.read_text())
    assert report["status"] == "withdrawn"
    assert record["status"] == "withdrawn"
    assert "unhashbound" in report["reason"]
    assert "no new robotics capability" in report["retained_truth"]
    assert record["disposition"]["quantitative_claims"] == "withdrawn"
    serialized = json.dumps(report) + json.dumps(record)
    for invalid_claim in ('"interior_holes": 13', '"answer": "fitted"', "0.9628"):
        assert invalid_claim not in serialized


def test_boundary_cycles_use_every_edge_once_and_split_a_bowtie() -> None:
    harness = _harness()
    triangles = np.array([(0, 1, 2), (0, 3, 4)], dtype=int)
    loops, closure = harness.closed_boundary_loops(triangles)
    assert sorted(len(loop) for loop in loops) == [3, 3]
    assert closure == {
        "boundary_edges": 6,
        "consumed_boundary_edges": 6,
        "boundary_vertices_degree_not_two": 1,
    }


def test_closed_boundary_cycles_reject_an_open_fragment() -> None:
    harness = _harness()
    assert harness._find_simple_cycle({(0, 1), (1, 2)}, (0, 1)) is None


def test_detector_controls_cover_closed_single_bowtie_and_open_topology() -> None:
    validation = _harness().detector_validation()
    assert validation["closed_tetrahedron_loops"] == 0
    assert validation["single_triangle_loops"] == 1
    assert validation["bowtie_loops"] == 2
    assert validation["open_fragment_rejected"] is True
    assert validation["detector_is_valid"] is True


def test_nonplanar_loop_area_is_explicitly_projected() -> None:
    harness = _harness()
    vertices = np.array(
        [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.2), (0.0, 1.0, 0.0)]
    )
    measured = harness.projected_loop_area(vertices, [0, 1, 2, 3])
    assert measured["projected_boundary_area_m2"] > 0
    assert measured["max_distance_from_best_fit_plane_m"] > 0
    assert "surface_area" not in measured


def test_empty_input_geometry_fails_closed() -> None:
    harness = _harness()
    mesh = types.SimpleNamespace(
        vertices=np.empty((0, 3)), triangles=np.empty((0, 3), dtype=int)
    )
    with pytest.raises(ValueError, match="mesh is empty"):
        harness.topology(mesh)


@pytest.mark.parametrize(
    "report",
    [
        {"detector_validation": {"detector_is_valid": False}, "inputs": {}},
        {"detector_validation": {"detector_is_valid": True}, "inputs": {}},
        {
            "detector_validation": {"detector_is_valid": True},
            "inputs": {"candidate": {"sha256": "short"}},
        },
    ],
)
def test_uncontrolled_or_unbound_report_is_quarantined(report: dict) -> None:
    canonical = Path("/evidence/holes-and-outliers.json")
    destination, published = _harness().publication(report, canonical)
    assert published is False
    assert destination != canonical
    assert destination.name.endswith(".invalid.json")


def test_hash_bound_controlled_report_can_publish() -> None:
    canonical = Path("/evidence/holes-and-outliers.json")
    report = {
        "detector_validation": {"detector_is_valid": True},
        "inputs": {"candidate": {"sha256": "a" * 64}},
    }
    assert _harness().publication(report, canonical) == (canonical, True)


def test_withdrawal_record_binds_current_method_and_report() -> None:
    record = json.loads(RECORD_PATH.read_text())
    provenance = record["provenance"]
    assert (
        provenance["harness_sha256"]
        == hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest()
    )
    assert (
        provenance["withdrawal_report_sha256"]
        == hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()
    )
