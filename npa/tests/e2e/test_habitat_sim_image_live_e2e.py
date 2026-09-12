"""Validate one completed exact-image Habitat-Sim RTX workflow run."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import stat
import subprocess
import urllib.parse

import boto3
import pytest

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/habitat-sim-smoke.yaml"
SOURCE_REVISION = "57ee4941dc4765240f0f91f70b2c97a919bf9038"
SCENE_SHA256 = "b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56"
NAVMESH_SHA256 = "1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d"
DIGEST = re.compile(r".+@sha256:[0-9a-f]{64}$")
SHA256 = re.compile(r"[0-9a-f]{64}")
PUBLIC_REGISTRY_HOSTS = {
    "docker.io",
    "ghcr.io",
    "index.docker.io",
    "public.ecr.aws",
    "quay.io",
    "registry-1.docker.io",
    "registry.k8s.io",
}
MAX_PRIVATE_RECEIPT_BYTES = 1024 * 1024


def _file_identity(value: os.stat_result) -> tuple[int, int, int, int, int]:
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


def _read_bounded(file_fd: int) -> bytes:
    payload = bytearray()
    while chunk := os.read(
        file_fd, min(65536, MAX_PRIVATE_RECEIPT_BYTES + 1 - len(payload))
    ):
        payload.extend(chunk)
        assert len(payload) <= MAX_PRIVATE_RECEIPT_BYTES
    return bytes(payload)


def _private_bytes(path_value: object) -> tuple[bytes, Path]:
    supplied = Path(str(path_value))
    assert (
        supplied.is_absolute()
        and supplied.parent.resolve(strict=True) == supplied.parent
    )
    directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC | os.O_NOFOLLOW
    directory_fd = os.open(supplied.parent, directory_flags)
    try:
        directory = os.fstat(directory_fd)
        assert stat.S_ISDIR(directory.st_mode)
        assert directory.st_uid == os.geteuid() and directory.st_mode & 0o077 == 0
        file_flags = os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW
        file_fd = os.open(supplied.name, file_flags, dir_fd=directory_fd)
        try:
            before = os.fstat(file_fd)
            assert stat.S_ISREG(before.st_mode)
            assert before.st_uid == os.geteuid() and before.st_mode & 0o077 == 0
            assert before.st_size <= MAX_PRIVATE_RECEIPT_BYTES
            payload = _read_bounded(file_fd)
            after = os.fstat(file_fd)
            named = os.stat(supplied.name, dir_fd=directory_fd, follow_symlinks=False)
            assert (
                _file_identity(before) == _file_identity(after) == _file_identity(named)
            )
            return payload, supplied
        finally:
            os.close(file_fd)
    finally:
        os.close(directory_fd)


def _private_json(path_value: object) -> tuple[dict[str, object], Path, bytes]:
    payload, path = _private_bytes(path_value)
    value = json.loads(payload)
    assert isinstance(value, dict)
    return value, path, payload


def _assert_private_image(receipt: dict[str, object]) -> None:
    registry = str(receipt["registry"]).rstrip("/")
    image = str(receipt["image"])
    parsed = urllib.parse.urlsplit("//" + registry)
    assert (
        registry
        and parsed.hostname
        and parsed.username is None
        and parsed.password is None
        and not parsed.query
        and not parsed.fragment
    )
    host = parsed.hostname.lower().rstrip(".")
    assert host not in PUBLIC_REGISTRY_HOSTS
    assert image.startswith(registry + "/")
    evidence, _path, payload = _private_json(receipt["registry_evidence"]["path"])
    assert SHA256.fullmatch(str(receipt["registry_evidence"]["sha256"]))
    assert hashlib.sha256(payload).hexdigest() == receipt["registry_evidence"]["sha256"]
    assert evidence == {
        **evidence,
        "schema_version": "npa.registry.private-pull-refusal.v1",
        "registry": registry,
        "image": image,
        "resolved_digest": image.rsplit("@", 1)[1],
        "anonymous_pull_denied": True,
        "anonymous_status": evidence["anonymous_status"],
        "authenticated_pull_succeeded": True,
    }
    assert evidence["anonymous_status"] in {401, 403}


def _assert_provider_binding(receipt: dict[str, object]) -> dict[str, object]:
    reservation = receipt["reservation"]
    provider, _path, payload = _private_json(reservation["provider_receipt_path"])
    provider_hash = hashlib.sha256(payload).hexdigest()
    assert SHA256.fullmatch(str(reservation["provider_receipt_sha256"]))
    assert provider_hash == reservation["provider_receipt_sha256"]
    assert provider["schema_version"] == "npa.nebius.strict-capacity-binding.v1"
    target = receipt["target"]
    for key in ("accelerator", "gpu_count"):
        assert provider[key] == reservation[key] == target[key]
    for key in (
        "capacity_block_group_id",
        "node_group_id",
        "kubernetes_node",
        "state",
        "verified_at",
    ):
        assert provider[key] == reservation[key]
    for key in ("kubernetes_context", "project", "project_id"):
        assert provider[key] == receipt[key]
    expected_policy = {
        "policy": "STRICT",
        "reservation_ids": [reservation["capacity_block_group_id"]],
    }
    assert provider["node_group_reservation_policy"] == expected_policy
    readback = provider["provider_readback_sha256"]
    assert readback == reservation["provider_readback_sha256"]
    assert set(readback) == {"capacity", "cluster", "node-group"}
    assert all(SHA256.fullmatch(str(value)) for value in readback.values())
    node = provider["kubernetes_node"]
    assert set(node) == {"name", "provider_id"}
    assert all(isinstance(node[key], str) and node[key] for key in node)
    assert isinstance(provider["node_group_id"], str) and provider["node_group_id"]
    return provider


def _assert_pod_completion(pod: dict[str, object], image: str) -> str:
    digest = image.rsplit("@", 1)[1]
    containers = pod["spec"].get("containers", [])
    statuses = pod["status"].get("containerStatuses", [])
    assert pod["status"].get("phase") == "Succeeded"
    assert len(containers) == len(statuses) == 1
    assert containers[0].get("image") == image
    assert containers[0].get("name") == statuses[0].get("name")
    image_id = str(statuses[0].get("imageID", ""))
    assert re.fullmatch(r"(?:docker-pullable://)?.+@" + re.escape(digest), image_id)
    terminated = statuses[0].get("state", {}).get("terminated")
    assert terminated and terminated.get("exitCode") == 0
    return image_id


def _private_receipt() -> dict[str, object]:
    receipt, _, _ = _private_json(os.environ["NPA_HABITAT_SIM_IMAGE_LIVE_RECEIPT"])
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
    _assert_private_image(receipt)
    _assert_provider_binding(receipt)
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


def _node(receipt: dict[str, object], name: str) -> dict[str, object]:
    result = subprocess.run(
        [
            "kubectl",
            "--kubeconfig",
            str(receipt["kubeconfig"]),
            "--context",
            str(receipt["kubernetes_context"]),
            "get",
            "node",
            name,
            "-o",
            "json",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(result.stdout)


def _assert_pod_provider_node(
    pod: dict[str, object], node: dict[str, object], provider: dict[str, object]
) -> None:
    expected = provider["kubernetes_node"]
    assert pod["spec"]["nodeName"] == expected["name"]
    assert node["metadata"]["name"] == expected["name"]
    assert node["spec"]["providerID"] == expected["provider_id"]


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
    provider = _assert_provider_binding(receipt)
    pod = _pod(receipt)
    node = _node(receipt, str(pod["spec"]["nodeName"]))
    _assert_pod_provider_node(pod, node, provider)
    image_id = _assert_pod_completion(pod, str(receipt["image"]))
    proof, payload = _proof(receipt)
    assert payload and proof["pod_observed_image_digest"] in image_id
    _assert_proof(proof, receipt)
    _assert_observation_readback(proof, receipt)
