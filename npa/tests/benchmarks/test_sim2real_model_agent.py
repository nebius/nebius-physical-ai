from __future__ import annotations

import json
from pathlib import Path

import pytest

from npa.benchmarks.sim2real_model_agent import _inside, _run_tool
from npa.benchmarks.sim2real_model_server import render_server_resources
from npa.benchmarks.sim2real_success import VerificationError, _lift_evidence


def _manifest(*, duration: float = 2.0, timestamped: bool = True) -> dict:
    rows = []
    for index, when in enumerate((0.0, 1.0, duration)):
        row = {
            "step": index,
            "sim_step": index,
            "simulator_ground_truth": {
                "stable_grasp": True,
                "object_lift_m": 0.051,
            },
        }
        if timestamped:
            row["sim_time_seconds"] = when
        rows.append(row)
    return {
        "schema": "npa.sim2real.action_rollout.v1",
        "source": "byo_isaac_policy_rollout",
        "sim_backend": "isaac",
        "policy_trained": True,
        "policy_checkpoint_sha256": "a" * 64,
        "policy_checkpoint_size_bytes": 10,
        "rollout_id": "rollout-1",
        "simulation_step_seconds": 1.0,
        "actions": rows,
    }


def test_lift_evidence_requires_physical_two_second_dwell(tmp_path: Path) -> None:
    found = _lift_evidence(
        tmp_path / "manifest.json",
        _manifest(),
        minimum_lift_m=0.05,
        minimum_hold_seconds=2.0,
    )
    assert found is not None
    assert found.duration_seconds == 2.0
    assert found.minimum_lift_m == pytest.approx(0.051)


def test_lift_evidence_rejects_short_or_broken_hold(tmp_path: Path) -> None:
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            _manifest(duration=1.99),
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )
    broken = _manifest()
    broken["actions"][1]["simulator_ground_truth"]["stable_grasp"] = False
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            broken,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )


def test_lift_evidence_refuses_sample_count_as_time(tmp_path: Path) -> None:
    manifest = _manifest(timestamped=False)
    manifest.pop("simulation_step_seconds")
    with pytest.raises(VerificationError, match="simulation_step_seconds"):
        _lift_evidence(
            tmp_path / "manifest.json",
            manifest,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )


def test_lift_evidence_rejects_gaps_in_temporal_coverage(tmp_path: Path) -> None:
    manifest = _manifest()
    manifest["actions"][-1]["sim_step"] = 3
    assert (
        _lift_evidence(
            tmp_path / "manifest.json",
            manifest,
            minimum_lift_m=0.05,
            minimum_hold_seconds=2.0,
        )
        is None
    )


def test_file_tools_cannot_escape_workspace(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="escapes"):
        _inside(tmp_path, "../outside")
    result = _run_tool(
        "write_file", {"path": "notes/result.txt", "content": "ok"}, tmp_path, {}
    )
    assert result["bytes"] == 2
    assert (tmp_path / "notes/result.txt").read_text() == "ok"


def test_tool_schema_round_trips_json_arguments(tmp_path: Path) -> None:
    path = tmp_path / "data.json"
    path.write_text(json.dumps({"ok": True}))
    result = _run_tool("read_file", {"path": "data.json"}, tmp_path, {})
    assert json.loads(result["content"]) == {"ok": True}


def test_model_server_renderer_pins_image_and_isolates_multinode_endpoint() -> None:
    model = {
        "repository": "org/model",
        "revision": "a" * 40,
        "server": "sglang",
        "server_image": "registry/server@sha256:" + "b" * 64,
        "tool_call_parser": "parser",
        "reasoning_parser": "reasoner",
        "context_limit": 1000,
        "tensor_parallel_size": 16,
        "server_arguments": ["--random-seed=7"],
    }
    resources = render_server_resources(model, namespace="trial-a", service_name="m")
    endpoint, statefulset = resources[2], resources[3]
    assert endpoint["spec"]["selector"] == {"statefulset.kubernetes.io/pod-name": "m-0"}
    assert statefulset["spec"]["replicas"] == 2
    container = statefulset["spec"]["template"]["spec"]["containers"][0]
    assert container["image"].startswith("registry/server@sha256:")
    assert container["resources"]["limits"]["nvidia.com/gpu"] == 8
    assert "ORDINAL=${POD_NAME##*-}" in container["command"][-1]
    assert "--node-rank ${ORDINAL}" in container["command"][-1]
    pod_spec = statefulset["spec"]["template"]["spec"]
    assert pod_spec["hostNetwork"] is True
    assert any(volume["name"] == "infiniband" for volume in pod_spec["volumes"])
