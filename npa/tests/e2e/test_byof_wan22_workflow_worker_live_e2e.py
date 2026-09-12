"""Read-only live verification of an existing standard-workflow Wan worker run."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

import pytest

from npa.deploy.images import wan_accepted_image_manifest
from npa.orchestration.npa_workflow.run_resolution import validate_run_id
from npa.solutions.wan2_2.rerun import (
    MULTI_GPU_LAYOUT,
    WanRunLayout,
    _source_filenames,
    layout_for_solution,
    validate_wan_run,
)

from .test_byof_wan22_live_e2e import _read_s3_json, _s3_client, _verify_published_rrd


@dataclass(frozen=True)
class _CompletedRun:
    """Bind verification to explicit operator inputs, never discovered defaults."""

    run_id: str
    project: str
    prefix_uri: str
    bucket: str
    key_prefix: str
    controls: dict[str, int]


def _required_environment(name: str) -> str:
    value = os.environ.get(name, "").strip()
    assert value, f"Completed-run verification requires {name}"
    return value


def _completed_run() -> _CompletedRun:
    run_id = validate_run_id(_required_environment("NPA_BYOF_WAN22_WORKER_RUN_ID"))
    project = _required_environment("NPA_BYOF_WAN22_WORKER_PROJECT")
    prefix_uri = _required_environment("NPA_BYOF_WAN22_WORKER_PREFIX")
    parsed = urlparse(prefix_uri)
    assert parsed.scheme == "s3" and parsed.netloc and parsed.path.strip("/")
    assert not (parsed.query or parsed.fragment or parsed.username or parsed.password)
    controls = {
        key: int(_required_environment(f"NPA_BYOF_WAN22_LIVE_{key.upper()}"))
        for key in ("frames", "steps", "seed")
    }
    assert controls["frames"] >= 5 and (controls["frames"] - 1) % 4 == 0
    assert controls["steps"] > 0 and controls["seed"] >= 0
    return _CompletedRun(
        run_id, project, prefix_uri.rstrip("/") + "/", parsed.netloc,
        parsed.path.strip("/") + "/", controls,
    )


def _verify_worker_summary(summary: dict, target: _CompletedRun) -> WanRunLayout:
    assert summary["schema"] == "npa.byof.worker-smoke.v1"
    assert summary["status"] == "success" and summary["smoke_exit_code"] == 0
    assert summary["execution"] == "workflow-worker"
    assert summary["tool"] == "byof" and summary["workload"] == "solution-smoke"
    assert summary["smoke_artifact_present"] is True
    assert summary["run_id"] == target.run_id
    # Both identities must be recorded; a worker can retain the outer run ID.
    validate_run_id(summary["workflow_worker_run_id"])
    assert str(summary["workflow_state"]).strip()
    accepted = wan_accepted_image_manifest()
    assert summary["image"].endswith("@" + accepted["oci_digest"])
    assert summary["metadata"]["repo"] == accepted["source"]["repository"]
    assert summary["metadata"]["ref"] == accepted["source"]["revision"]
    layout = layout_for_solution(summary["solution_name"])
    assert layout is not None, "Summary must identify a supported Wan GPU layout"
    assert summary["smoke_artifact_name"] in layout.primary_filenames
    return layout


def _download_worker_evidence(s3, target, summary, layout, directory: Path) -> None:
    names = _source_filenames(layout, summary["smoke_artifact_name"])
    names.append("wan2_2_generation_request.json")
    inventory = {item["path"]: item for item in summary["artifacts"]}
    for name in names:
        path = directory / name
        s3.download_file(target.bucket, target.key_prefix + name, str(path))
        if name == "npa_byof_summary.json":
            assert json.loads(path.read_text()) == summary
            continue
        recorded = inventory[name]
        assert path.stat().st_size == recorded["size_bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == recorded["sha256"], name


def _verify_generation(evidence: dict, target: _CompletedRun, directory: Path) -> None:
    request = json.loads((directory / "wan2_2_generation_request.json").read_text())
    assert request == target.controls
    assert all(type(value) is int for value in request.values())
    primary = evidence["primary"]
    if evidence["layout"] is MULTI_GPU_LAYOUT:
        actual = {key: primary["generation"][key] for key in target.controls}
    else:
        actual = {
            "frames": primary["requested"]["frame_count"],
            "steps": primary["requested"]["inference_steps"],
            "seed": primary["seed"],
        }
    assert actual == target.controls
    assert evidence["video"]["observed"]["frame_count"] == target.controls["frames"]


def _verify_manifest_sources(
    manifest: dict, target: _CompletedRun, directory: Path, source_names: list[str]
) -> None:
    assert manifest["source_prefix_uri"] == target.prefix_uri
    assert {item["uri"] for item in manifest["source_objects"]} == {
        target.prefix_uri + name for name in source_names
    }
    for source in manifest["source_objects"]:
        assert source["uri"].startswith(target.prefix_uri)
        name = source["uri"][len(target.prefix_uri):]
        assert Path(name).name == name, "Manifest source must be a direct run artifact"
        path = directory / name
        assert path.stat().st_size == source["size_bytes"], name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == source["sha256"], name


@pytest.mark.e2e
@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_BYOF_WAN22_WORKER_VERIFY") != "1",
    reason="Requires explicit completed-run verification gates; never submits GPU jobs",
)
def test_completed_standard_workflow_wan_worker(tmp_path: Path) -> None:
    target = _completed_run()
    s3 = _s3_client(target.project)
    summary = _read_s3_json(s3, target.bucket, target.key_prefix + "npa_byof_summary.json")
    layout = _verify_worker_summary(summary, target)
    _download_worker_evidence(s3, target, summary, layout, tmp_path)
    evidence = validate_wan_run(tmp_path, layout)
    assert evidence["run_id"] == target.run_id
    _verify_generation(evidence, target, tmp_path)
    manifest = _verify_published_rrd(
        s3, bucket=target.bucket, key_prefix=target.key_prefix, layout=layout,
        run_id=target.run_id, video_path=tmp_path / layout.video_filename,
        expected_frame_count=target.controls["frames"], expected_fps=24.0,
        expected_rank_count=len(layout.rank_filenames), tmp_path=tmp_path,
    )
    _verify_manifest_sources(
        manifest, target, tmp_path, _source_filenames(layout, summary["smoke_artifact_name"])
    )
