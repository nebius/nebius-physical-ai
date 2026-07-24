"""Tier-0 tests for Rerun recording identity (no stock-demo-as-run-data rule)."""

from __future__ import annotations

from npa.cli import agent_recordings as R

# Minimal byte fixtures mimicking the entity-path strings embedded in .rrd files.
_RUN_RRD = b"RRF2\x00...world/table.../heldout/per_env/env-0000.../rollout/0.../scores..."
_DEMO_RRD = b"RRF2\x00...world/franka/base.../franka/gripper...world/table...world/cube...demo/active_camera..."
_EMPTY = b""


def test_run_recording_detected_as_run_specific():
    assert R.recording_has_run_entities(_RUN_RRD) is True
    assert R.is_stock_demo_recording(_RUN_RRD) is False


def test_stock_demo_detected_and_not_run_specific():
    assert R.recording_has_run_entities(_DEMO_RRD) is False
    assert R.is_stock_demo_recording(_DEMO_RRD) is True


def test_empty_or_none_is_not_ready():
    assert R.recording_has_run_entities(_EMPTY) is False
    assert R.recording_has_run_entities(None) is False
    assert R.is_stock_demo_recording(_EMPTY) is False


def test_unknown_geometry_only_is_not_run_and_not_demo():
    other = b"RRF2\x00...world/floor...world/lamp..."
    assert R.recording_has_run_entities(other) is False
    assert R.is_stock_demo_recording(other) is False


def test_run_recording_basename_is_filesystem_safe():
    assert R.run_recording_basename("agent-run-1843d1e8b64d") == "agent-run-1843d1e8b64d.rrd"
    assert R.run_recording_basename("sim2real-2026:07") == "sim2real-2026:07.rrd"
    # Traversal / odd input is sanitized.
    assert "/" not in R.run_recording_basename("../../evil")
    assert ".." not in R.run_recording_basename("../../evil")
    assert R.run_recording_basename("") == "run.rrd"
