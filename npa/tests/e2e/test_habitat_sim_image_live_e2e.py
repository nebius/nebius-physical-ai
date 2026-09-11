"""Validate one completed exact-image Habitat-Sim RTX workflow run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import subprocess

import boto3
import pytest


ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/habitat-sim-smoke.yaml"
SOURCE_REVISION = "57ee4941dc4765240f0f91f70b2c97a919bf9038"
SCENE_SHA256 = "b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56"
NAVMESH_SHA256 = "1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d"
DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}$")


def _private_receipt() -> dict[str, object]:
    path = Path(os.environ["NPA_HABITAT_SIM_IMAGE_LIVE_RECEIPT"]).resolve()
    assert path.is_file() and path.stat().st_mode & 0o077 == 0
    assert path.parent.stat().st_mode & 0o077 == 0
    receipt = json.loads(path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == "npa.habitat-sim.image-live.v1"
    assert (
        receipt["head"]
        == subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip()
    )
    workflow_hash = hashlib.sha256(WORKFLOW.read_bytes()).hexdigest()
    assert receipt["workflow_sha256"] == workflow_hash
    assert DIGEST.fullmatch(str(receipt["image"]))
    target = receipt["target"]
    assert target == {
        **target,
        "policy": "STRICT",
        "accelerator": "RTX PRO 6000 Blackwell",
        "gpu_count": 1,
    }
    return receipt


def _pod(receipt: dict[str, object]) -> dict[str, object]:
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(receipt["kubeconfig"]),
            "--context",
            str(receipt["kubernetes_context"]),
            "-n",
            str(receipt["namespace"]),
            "get",
            "pod",
            str(receipt["pod_name"]),
            "-o",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    pod = json.loads(result.stdout)
    assert pod["metadata"]["uid"] == receipt["pod_uid"]
    return pod


def _proof(receipt: dict[str, object]) -> tuple[dict[str, object], bytes]:
    storage = receipt["storage"]
    client = boto3.client("s3", endpoint_url=storage["endpoint"])
    key = storage["prefix"].rstrip("/") + "/habitat-sim-smoke.json"
    payload = client.get_object(Bucket=storage["bucket"], Key=key)["Body"].read()
    assert hashlib.sha256(payload).hexdigest() == receipt["proof_sha256"]
    return json.loads(payload), payload


def _assert_observation_readback(
    proof: dict[str, object], receipt: dict[str, object]
) -> None:
    storage = receipt["storage"]
    client = boto3.client("s3", endpoint_url=storage["endpoint"])
    prefix = storage["prefix"].rstrip("/")
    frames = proof["rendered_observations"]["frames"]
    assert len(frames) == proof["rendered_rgb_frame_count"]
    for frame in frames:
        for kind in ("rgb_png", "depth_npy", "depth_preview_png"):
            relative = frame[f"{kind}_path"]
            payload = client.get_object(
                Bucket=storage["bucket"], Key=f"{prefix}/{relative}"
            )["Body"].read()
            assert len(payload) == frame[f"{kind}_bytes"]
            assert hashlib.sha256(payload).hexdigest() == frame[f"{kind}_sha256"]


def _assert_proof(proof: dict[str, object], receipt: dict[str, object]) -> None:
    assert proof["solution"] == "habitat-sim"
    assert proof["source_revision"] == SOURCE_REVISION
    assert proof["scene_sha256"] == SCENE_SHA256
    assert proof["scene"]["navmesh_sha256"] == NAVMESH_SHA256
    assert proof["scene_license"] == "CC BY 4.0"
    assert proof["rendered_rgb_frame_count"] > 1
    assert proof["rendered_depth_frame_count"] == proof["rendered_rgb_frame_count"]
    assert proof["agent_displacement"] > 0.1
    assert proof["bullet_step_count"] > 0
    assert proof["measured_fps"] > 0
    assert proof["renderer_egl_evidence"]["backend"] == "EGL"
    assert proof["observed_rtx_gpu_architecture"] == "Blackwell"
    assert proof["observed_rtx_gpu_count"] == 1
    assert proof["exit_status"] == 0
    assert proof["pod_observed_image_digest"] == str(receipt["image"]).rsplit("@", 1)[1]


@pytest.mark.skipif(
    os.environ.get("NPA_INTEGRATION_E2E") != "1"
    or os.environ.get("NPA_HABITAT_SIM_IMAGE_LIVE") != "1"
    or not os.environ.get("NPA_HABITAT_SIM_IMAGE_LIVE_RECEIPT", "").strip(),
    reason="requires the owner-only exact-image Habitat-Sim live receipt",
)
def test_exact_habitat_sim_image_rgb_depth_bullet_egl_traversal() -> None:
    """Require immutable pod, GPU, storage, and genuine capability evidence."""

    receipt = _private_receipt()
    pod = _pod(receipt)
    digest = str(receipt["image"]).rsplit("@", 1)[1]
    statuses = pod["status"].get("containerStatuses", [])
    assert len(statuses) == 1 and digest in statuses[0]["imageID"]
    proof, payload = _proof(receipt)
    assert payload and proof["pod_observed_image_digest"] in statuses[0]["imageID"]
    _assert_proof(proof, receipt)
    _assert_observation_readback(proof, receipt)
