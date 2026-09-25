"""Opt-in native Isaac PhysX qualification of an operator-supplied scene handoff."""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import uuid
from pathlib import Path

import pytest

pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot, pytest.mark.gpu]
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "workflows/testing/scan-to-isaac-navigation.yaml"


def _require(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        pytest.skip(f"{name} is required for the live scene handoff")
    return value


def _submit_scene(run_id: str, bucket: str, project: str) -> None:
    command = [
        sys.executable,
        "-m",
        "npa",
        "workbench",
        "workflow",
        "submit",
        str(SPEC),
        "--runtime",
        "--max-wait-seconds",
        "0",
        "--run-id",
        run_id,
        "--project",
        project,
        "--infra",
        _require("NPA_SCAN_TO_ISAAC_INFRA"),
        "--var",
        f"bucket={bucket}",
        "--var",
        f"input_path={_require('NPA_SCAN_TO_ISAAC_INPUT_URI')}",
        "--var",
        f"isaac_image={_require('NPA_SCAN_TO_ISAAC_ISAAC_IMAGE')}",
        "--secret-env",
        "AWS_ACCESS_KEY_ID",
        "--secret-env",
        "AWS_SECRET_ACCESS_KEY",
    ]
    for key in ("assembly_image",):
        value = os.environ.get(f"NPA_SCAN_TO_ISAAC_{key.upper()}", "").strip()
        if value:
            command.extend(["--var", f"{key}={value}"])
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, "scene handoff failed; inspect private workflow logs"


def _read_artifacts(client, bucket: str, prefix: str, output: Path) -> tuple:
    from npa.workbench.nurec.navigation_publication import verify_publication

    artifacts = {}
    for relative in (
        "assembled/scene.usdz",
        "assembled/provenance.json",
        "assembled/.npa-navigation-claim.json",
        "assembled/.npa-navigation-complete.json",
        "reports/physics_validation.json",
        "reports/.npa-navigation-claim.json",
        "reports/.npa-navigation-complete.json",
    ):
        destination = output / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        client.download_file(bucket, f"{prefix}/{relative}", str(destination))
        assert destination.stat().st_size > 0
        artifacts[relative] = destination
    for directory in ("assembled", "reports"):
        verify_publication(output / directory)
    scene = artifacts["assembled/scene.usdz"]
    provenance = json.loads(artifacts["assembled/provenance.json"].read_text())
    physics = json.loads(artifacts["reports/physics_validation.json"].read_text())
    assert (
        physics["assembly_provenance_sha256"]
        == hashlib.sha256(
            artifacts["assembled/provenance.json"].read_bytes()
        ).hexdigest()
    )
    return scene, provenance, physics


def _assert_measured_probes(provenance: dict, physics: dict) -> None:
    assert physics["schema"] == "npa.nurec.navigation_physics.v1"
    assert physics["physics_validated"] is True
    assert physics["visual_render_validated"] is False
    assert physics["runtime_versions"]["isaac_sim"]["version"]
    assert physics["runtime_image"] == {
        "reference": _require("NPA_SCAN_TO_ISAAC_ISAAC_IMAGE"),
        "attestation_scope": "operator-declared workload image reference",
        "running_image_identity_verified": False,
    }
    requested = provenance["ray_probes"]
    measured = physics["probes"]
    assert requested and len(measured) == len(requested)
    for expected, actual in zip(requested, measured, strict=True):
        distance = actual["distance_m"]
        assert math.isfinite(distance)
        assert expected["min_distance"] <= distance <= expected["max_distance"]
        assert actual["expected"] == expected
        assert actual["collision"] in {item["path"] for item in provenance["colliders"]}


def test_scene_handoff_runs_real_isaac_physics(tmp_path: Path) -> None:
    """Submit the scene workflow and verify its real package and measured ray hits.

    Args:
        tmp_path: Private temporary directory for retrieved artifacts.
    Returns:
        None.
    Raises:
        AssertionError: Submission or artifact qualification fails.
    """
    if any(
        os.environ.get(name) != "1"
        for name in ("NPA_INTEGRATION_E2E", "NPA_SCAN_TO_ISAAC_LIVE")
    ):
        pytest.skip("set both NPA_INTEGRATION_E2E=1 and NPA_SCAN_TO_ISAAC_LIVE=1")
    from npa.clients.project_credentials import s3_client_for_project
    from npa.workbench.nurec.navigation_scene import verify_scene

    bucket = _require("NPA_SCAN_TO_ISAAC_BUCKET")
    project = _require("NPA_SCAN_TO_ISAAC_PROJECT")
    run_id = f"scene-handoff-{uuid.uuid4().hex}"
    _submit_scene(run_id, bucket, project)
    client = s3_client_for_project(project, allow_host_creds=True)
    scene, provenance, physics = _read_artifacts(
        client, bucket, f"scan-to-isaac-navigation/{run_id}", tmp_path
    )
    digest = hashlib.sha256(scene.read_bytes()).hexdigest()
    assert provenance["schema"] == "npa.nurec.navigation_scene.v1"
    assert provenance["scene_sha256"] == physics["scene_sha256"] == digest
    assert provenance["collider_count"] > 0 and provenance["triangle_count"] > 0
    assert provenance["physics_validated"] is False
    verify_scene(scene, provenance["colliders"])
    _assert_measured_probes(provenance, physics)
