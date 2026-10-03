"""Contracts for the LIBERO-Plus matched-robustness workflow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml

from npa.workflows import libero_plus


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/libero-plus-robustness.yaml"


def test_workflow_has_five_connected_substantive_stages() -> None:
    payload = yaml.safe_load(SPEC.read_text())
    states = payload["states"]
    assert list(states) == [
        "prepare-tasks",
        "baseline-rollouts",
        "candidate-rollouts",
        "calculate-deltas",
        "emit-results",
    ]
    assert all("argv" in state["run"] for state in states.values())
    assert states["prepare-tasks"]["next"] == "baseline-rollouts"
    assert states["baseline-rollouts"]["next"] == "candidate-rollouts"
    assert states["candidate-rollouts"]["next"] == "calculate-deltas"
    assert states["calculate-deltas"]["next"] == "emit-results"
    assert states["emit-results"]["terminal"] is True
    candidate_argv = states["candidate-rollouts"]["run"]["argv"]
    assert "--matched-baseline-uri" in candidate_argv
    assert "{{state.baseline-rollouts.uri}}" in candidate_argv
    assert any(
        output["schema"] == "application/vnd.rerun.rrd"
        for output in states["emit-results"]["outputs"]
    )


def test_compare_requires_same_prepared_protocol_and_reports_all_dimensions(
    tmp_path: Path,
) -> None:
    protocol = tmp_path / "protocol.json"
    baseline = tmp_path / "baseline.json"
    candidate = tmp_path / "candidate.json"
    delta = tmp_path / "delta.json"
    protocol.write_text(json.dumps({"tasks": [{"id": index} for index in range(1, 8)]}))
    digest = libero_plus.hashlib.sha256(protocol.read_bytes()).hexdigest()
    rows = [
        {"task_id": index, "category": dimension, "success": index % 2 == 0}
        for index, dimension in enumerate(libero_plus.DIMENSIONS, start=1)
    ]
    baseline.write_text(
        json.dumps({"protocol_sha256": digest, "results": rows, "smoke_only": True})
    )
    candidate_rows = [dict(row, success=not row["success"]) for row in rows]
    candidate.write_text(
        json.dumps(
            {"protocol_sha256": digest, "results": candidate_rows, "smoke_only": True}
        )
    )

    libero_plus.compare(str(protocol), str(baseline), str(candidate), str(delta))

    observed = json.loads(delta.read_text())
    assert observed["deltas"][libero_plus.DIMENSIONS[0]] == 1.0
    assert observed["deltas"][libero_plus.DIMENSIONS[1]] == -1.0
    assert set(observed["deltas"]) == set(libero_plus.DIMENSIONS)
    assert observed["smoke_only"] is True

    rows_with_media = [
        dict(
            row,
            episode_media={
                "uri": f"s3://unit/{row['task_id']}.mp4",
                "sha256": "a",
                "bytes": 1,
            },
        )
        for row in rows
    ]
    baseline.write_text(
        json.dumps(
            {"protocol_sha256": digest, "results": rows_with_media, "smoke_only": True}
        )
    )
    candidate.write_text(
        json.dumps(
            {"protocol_sha256": digest, "results": rows_with_media, "smoke_only": True}
        )
    )
    report = tmp_path / "report.json"
    rrd = tmp_path / "report.rrd"
    libero_plus.report(str(delta), str(baseline), str(candidate), str(report), str(rrd))
    assert json.loads(report.read_text())["rrd"]["sha256"]
    verified = subprocess.run(
        [str(Path(sys.executable).with_name("rerun")), "rrd", "verify", str(rrd)],
        capture_output=True,
        check=False,
    )
    assert verified.returncode == 0, verified.stderr.decode()


def test_full_protocol_refuses_smoke_policy(tmp_path: Path) -> None:
    protocol = tmp_path / "protocol.json"
    protocol.write_text(json.dumps({"mode": "benchmark"}))
    try:
        libero_plus.rollout(
            str(protocol),
            str(tmp_path / "out.json"),
            str(tmp_path / "media"),
            "baseline",
            "smoke-zero",
            1,
        )
    except libero_plus.LiberoPlusError as error:
        assert "cannot produce a complete benchmark" in str(error)
    else:
        raise AssertionError("full benchmark accepted smoke-only adapter")
