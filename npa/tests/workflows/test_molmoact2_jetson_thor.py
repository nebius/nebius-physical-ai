"""Contract tests for the MolmoAct2 Jetson-Thor edge evidence workflow."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import npa.workflows.molmoact2_jetson_thor as thor
from npa.orchestration.npa_workflow import build_plan, load_spec
from npa.orchestration.npa_workflow.submit import merge_config_overrides
from npa.workflows.molmoact2_jetson_thor import (
    BASE_MODEL,
    BASE_MODEL_REVISION,
    BUNDLE_REPOSITORY,
    BUNDLE_REVISION,
    QUALIFICATION_SCHEMA,
    TARGET_REQUIREMENTS,
    TRACE_SCHEMA,
    VLA_EDGE_REPOSITORY,
    VLA_EDGE_REVISION,
    MolmoAct2JetsonThorError,
    evaluate_action_trace,
    prepare_observations,
    target_action_trace,
    verify_target_action_trace,
    verify_target_qualification,
    visualize_action_trace,
)

SPEC = Path(__file__).parents[3] / "workflows/testing/molmoact2-jetson-thor-edge.yaml"
READINESS = SPEC.with_suffix(".readiness.json")
DOCKERFILE = (
    Path(__file__).parents[2] / "docker/workbench/molmoact2-jetson-thor/Dockerfile"
)


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_observations(path: Path) -> None:
    rows = 4
    np.savez_compressed(
        path,
        image=np.arange(rows * 2 * 3 * 3, dtype=np.uint8).reshape(rows, 2, 3, 3),
        wrist_image=np.full((rows, 2, 3, 3), 37, dtype=np.uint8),
        state=np.arange(rows * 8, dtype=np.float32).reshape(rows, 8),
        instruction=np.asarray(
            ["open drawer", "open drawer", "close drawer", "close drawer"]
        ),
        episode_id=np.asarray(["episode-0", "episode-1", "episode-2", "episode-3"]),
        split=np.asarray(["calibration", "calibration", "evaluation", "evaluation"]),
        seed=np.asarray([11, 12, 13, 14], dtype=np.int64),
    )


def _write_qualification(path: Path, prepared_digest: str) -> None:
    payload = {
        "schema": QUALIFICATION_SCHEMA,
        "status": "passed",
        "run_id": "thor-test",
        "prepared_observations_sha256": prepared_digest,
        "bundle_repository": BUNDLE_REPOSITORY,
        "bundle_revision": BUNDLE_REVISION,
        "vla_edge_repository": VLA_EDGE_REPOSITORY,
        "vla_edge_revision": VLA_EDGE_REVISION,
        "base_model": BASE_MODEL,
        "base_model_revision": BASE_MODEL_REVISION,
        "engine_manifest_sha256": "a" * 64,
        "target": dict(TARGET_REQUIREMENTS),
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def _write_trace(path: Path) -> None:
    target = np.full((2, 10, 7), 0.25, dtype=np.float32)
    reference = np.full((2, 10, 7), 0.5, dtype=np.float32)
    np.savez_compressed(
        path,
        episode_id=np.asarray(["episode-2", "episode-3"]),
        seed=np.asarray([13, 14], dtype=np.int64),
        target_actions=target,
        reference_actions=reference,
        target_dt_ms=np.asarray([110.0, 120.0]),
        reference_dt_ms=np.asarray([600.0, 620.0]),
    )


def test_connected_artifact_pipeline_emits_verified_rrd(tmp_path: Path) -> None:
    observations = tmp_path / "observations.npz"
    prepared = tmp_path / "prepared.npz"
    prepared_manifest = tmp_path / "prepared.json"
    qualification = tmp_path / "qualification.json"
    verified_qualification = tmp_path / "verified-qualification.json"
    trace = tmp_path / "trace.npz"
    trace_manifest = tmp_path / "trace.json"
    verified_trace = tmp_path / "verified-trace.json"
    evaluation = tmp_path / "evaluation.json"
    rrd = tmp_path / "reports/trace.rrd"
    visualization = tmp_path / "reports/visualization.json"
    _write_observations(observations)

    prepared_result = prepare_observations(
        observations_uri=str(observations),
        prepared_uri=str(prepared),
        manifest_uri=str(prepared_manifest),
        run_id="thor-test",
    )
    assert prepared_result["calibration_rows"] == 2
    assert prepared_result["evaluation_rows"] == 2
    _write_qualification(qualification, _digest(prepared))
    qualification_result = verify_target_qualification(
        prepared_uri=str(prepared),
        qualification_uri=str(qualification),
        verified_uri=str(verified_qualification),
        run_id="thor-test",
    )
    _write_trace(trace)
    trace_payload = {
        "schema": TRACE_SCHEMA,
        "run_id": "thor-test",
        "execution_mode": "open_loop_observation_trace",
        "closed_loop_rollout_status": "not_evaluated_by_vla_edge",
        "qualification_sha256": qualification_result["qualification_sha256"],
        "trace_uri": str(trace),
        "trace_sha256": _digest(trace),
        "rows": 2,
        "action_horizon": 10,
        "action_dim": 7,
        "target_backend": "tensorrt",
        "reference_backend": "torch",
    }
    trace_manifest.write_text(json.dumps(trace_payload), encoding="utf-8")
    verified_result = verify_target_action_trace(
        trace_manifest_uri=str(trace_manifest),
        verified_qualification_uri=str(verified_qualification),
        verified_trace_uri=str(verified_trace),
        run_id="thor-test",
    )
    evaluation_result = evaluate_action_trace(
        verified_trace_uri=str(verified_trace),
        evaluation_uri=str(evaluation),
        run_id="thor-test",
    )
    visualization_result = visualize_action_trace(
        verified_trace_uri=str(verified_trace),
        evaluation_uri=str(evaluation),
        rrd_uri=str(rrd),
        manifest_uri=str(visualization),
        run_id="thor-test",
    )

    assert verified_result["rows"] == 2
    assert evaluation_result["action_mae"] == pytest.approx(0.25)
    assert evaluation_result["action_rmse"] == pytest.approx(0.25)
    assert (
        evaluation_result["closed_loop_rollout_status"] == "not_evaluated_by_vla_edge"
    )
    assert visualization_result["rrd_bytes"] > 1024
    rerun = Path(sys.executable).with_name("rerun")
    verified = subprocess.run(
        [str(rerun), "rrd", "verify", str(rrd)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr
    printed = subprocess.run(
        [str(rerun), "rrd", "print", "-vv", str(rrd)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert printed.returncode == 0, printed.stderr
    assert "evaluation_row" in printed.stdout
    assert "latency/target_ms" in printed.stdout


def test_qualification_rejects_non_thor_evidence(tmp_path: Path) -> None:
    observations = tmp_path / "observations.npz"
    prepared = tmp_path / "prepared.npz"
    manifest = tmp_path / "prepared.json"
    qualification = tmp_path / "qualification.json"
    _write_observations(observations)
    prepare_observations(
        observations_uri=str(observations),
        prepared_uri=str(prepared),
        manifest_uri=str(manifest),
        run_id="thor-test",
    )
    _write_qualification(qualification, _digest(prepared))
    payload = json.loads(qualification.read_text())
    payload["target"]["arch"] = "x86_64"
    qualification.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(MolmoAct2JetsonThorError, match="target arch"):
        verify_target_qualification(
            prepared_uri=str(prepared),
            qualification_uri=str(qualification),
            verified_uri=str(tmp_path / "verified.json"),
            run_id="thor-test",
        )


def test_target_compute_capability_allows_only_upstream_thor_spelling() -> None:
    assert thor._canonical_compute_capability("sm_110") == "sm_110a"
    assert thor._canonical_compute_capability("sm_110a") == "sm_110a"
    assert thor._canonical_compute_capability("sm_120") == "sm_120"


def test_target_trace_binds_the_staged_raw_qualification(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The device can emit both target artifacts before cloud verification starts."""

    class Contract:
        action_horizon = 3
        action_dim = 7

    class Client:
        def act(self, *args, **kwargs):  # noqa: ANN002,ANN003 - mirrors upstream client
            del args, kwargs
            return np.full((3, 7), 0.5, dtype=np.float32), 12.5

        def close(self) -> None:
            return None

    observations = tmp_path / "observations.npz"
    prepared = tmp_path / "prepared.npz"
    qualification = tmp_path / "qualification.json"
    _write_observations(observations)
    prepare_observations(
        observations_uri=str(observations),
        prepared_uri=str(prepared),
        manifest_uri=str(tmp_path / "prepared.json"),
        run_id="thor-test",
    )
    _write_qualification(qualification, _digest(prepared))
    monkeypatch.setattr(thor, "_bind_edge_endpoint", lambda *_: (Client(), Contract()))

    result = target_action_trace(
        prepared_uri=str(prepared),
        qualification_uri=str(qualification),
        target_endpoint="http://target.invalid:8202",
        reference_endpoint="http://reference.invalid:8203",
        trace_uri=str(tmp_path / "trace.npz"),
        manifest_uri=str(tmp_path / "trace.json"),
        run_id="thor-test",
    )

    expected = hashlib.sha256(
        json.dumps(json.loads(qualification.read_text()), sort_keys=True).encode()
    ).hexdigest()
    assert result["qualification_sha256"] == expected
    assert result["rows"] == 2


def test_workflow_has_five_connected_real_stages() -> None:
    spec = load_spec(SPEC)
    plan = build_plan(spec, run_id="thor-test")
    assert spec.config["source_sha"] == ""
    assert spec.config["require_baked_npa"] == "1"
    assert spec.config["baked_npa_import"] == "npa.workflows.molmoact2_jetson_thor"
    assert spec.resources["cpu"]["image"] == "{{config.runtime_image}}"
    assert spec.config["runtime_image"].endswith("@sha256:" + "0" * 64)
    assert [step.state for step in plan.steps] == [
        "prepare-observations",
        "verify-target-engine",
        "verify-target-action-trace",
        "evaluate-target-action-trace",
        "visualize-target-evidence",
    ]
    assert all(
        step.argv[:3] == ["python3", "-m", "npa.workflows.molmoact2_jetson_thor"]
        for step in plan.steps
    )
    commands = [step.argv[3] for step in plan.steps]
    assert commands == [
        "prepare",
        "verify-qualification",
        "verify-action-trace",
        "evaluate",
        "visualize",
    ]
    assert "target-qualify" not in commands
    assert "target-action-trace" not in commands
    assert (
        spec.states["visualize-target-evidence"].outputs[0].schema
        == "application/vnd.rerun.rrd"
    )


def test_readiness_record_binds_workflow_and_target_blocker() -> None:
    readiness = json.loads(READINESS.read_text(encoding="utf-8"))
    assert readiness["workflow_sha256"] == _digest(SPEC)
    assert readiness["planning"]["validation"]["status"] == "verified"
    assert (
        "does not run target-qualify or target-action-trace"
        in readiness["planning"]["task_fidelity"]["reason"]
    )
    assert readiness["prerequisites"]["target_runtime"]["status"] == "blocked"
    assert (
        "Before native acceptance"
        in readiness["prerequisites"]["target_runtime"]["reason"]
    )


def test_workflow_argv_preserves_hostile_config_as_one_argument() -> None:
    hostile = "s3://bucket/a'; touch /tmp/edge-injection; #"
    spec = merge_config_overrides(load_spec(SPEC), {"observations_uri": hostile})
    first = build_plan(spec, run_id="thor-test").steps[0]
    assert first.argv[first.argv.index("--observations-uri") + 1] == hostile
    assert first.shell == ""


def test_evidence_worker_applies_debian_security_updates() -> None:
    dockerfile = DOCKERFILE.read_text(encoding="utf-8")
    assert "apt-get update" in dockerfile
    assert "apt-get upgrade -y --no-install-recommends" in dockerfile
    assert dockerfile.index("apt-get update") < dockerfile.index("apt-get upgrade")
    assert dockerfile.index("apt-get upgrade") < dockerfile.index("apt-get install")
