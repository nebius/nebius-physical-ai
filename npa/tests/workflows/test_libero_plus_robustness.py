"""Contracts for the LIBERO-Plus matched-robustness workflow."""

from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import yaml
import pytest

from npa.workflows import libero_plus


ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/libero-plus-robustness.yaml"
ADMISSION_DOCKERFILE = ROOT / "npa/docker/workbench/libero-plus/Dockerfile.admission"
ADMISSION_BUILD = ROOT / "npa/docker/workbench/libero-plus/build-private.sh"
ADMISSION_NOTICE = ROOT / "npa/docker/workbench/libero-plus/THIRD_PARTY_NOTICES.md"


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
            {
                "protocol_sha256": digest,
                "results": candidate_rows,
                "smoke_only": True,
                "matched_baseline_sha256": libero_plus.hashlib.sha256(
                    baseline.read_bytes()
                ).hexdigest(),
            }
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
            {
                "protocol_sha256": digest,
                "results": rows_with_media,
                "smoke_only": True,
                "matched_baseline_sha256": libero_plus.hashlib.sha256(
                    baseline.read_bytes()
                ).hexdigest(),
            }
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


def test_unlicensed_pinned_source_fails_before_any_upstream_fetch() -> None:
    try:
        libero_plus._ensure_upstream()
    except libero_plus.LiberoPlusError as error:
        assert "source execution is blocked" in str(error)
    else:
        raise AssertionError("unlicensed source fetch was not blocked")


def test_approved_source_cache_requires_operator_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A future authorization must not silently use an ambient temporary cache."""
    monkeypatch.delenv("NPA_MODEL_CACHE_DIR", raising=False)
    with pytest.raises(libero_plus.LiberoPlusError, match="NPA_MODEL_CACHE_DIR"):
        libero_plus._cache_root()


def test_private_admission_image_contains_no_upstream_payload() -> None:
    dockerfile = ADMISSION_DOCKERFILE.read_text()
    notice = ADMISSION_NOTICE.read_text()

    assert "COPY src/npa/workflows/libero_plus.py" in dockerfile
    assert "USER ubuntu" in dockerfile
    assert 'org.nebius.npa.upstream-payload="absent"' in dockerfile
    assert "git clone" not in dockerfile
    assert "hf_hub_download" not in dockerfile
    assert "LIBERO-Plus source" in notice
    assert "is not copied into this image" in notice
    assert "asset archive is not copied into this image" in notice


def test_private_admission_build_refuses_public_targets_and_scans_output() -> None:
    build = ADMISSION_BUILD.read_text()

    assert "ghcr.io/nebius/nebius-physical-ai/*|docker.io/*|index.docker.io/*" in build
    assert "refusing public image target" in build
    assert "--provenance=mode=max" in build
    assert "--sbom=true" in build
    assert "scan_image_omniverse_payload.py" in build
    assert "built-local-not-pushed" in build
