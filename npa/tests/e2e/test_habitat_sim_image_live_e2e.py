"""Validate one completed exact-image Habitat-Sim RTX workflow run."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import math
import os
from pathlib import Path
import re
import stat
import subprocess
import urllib.parse

import boto3
import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
WORKFLOW = ROOT / "workflows/testing/habitat-sim-smoke.yaml"
SOURCE_REVISION = "57ee4941dc4765240f0f91f70b2c97a919bf9038"
SCENE_SHA256 = "b14e29e17f5e31d86a1002eefd77b7d345b265006481739ae480a847e6623f56"
NAVMESH_SHA256 = "1a9a5bd123af8001f0ea2c5c8d326cb3fd39808ca771fc766856af8f0772391d"
ARCHIVE_SHA256 = "1231420c6482e79e25beea7ab25121e0421a5fd67b68dd9502145442c288db06"
ARCHIVE_BYTES = 94_590_970
ARCHIVE_URL = "https://dl.fbaipublicfiles.com/habitat/habitat-test-scenes.zip"
ASSET_LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/legalcode.en"
ATTRIBUTION = "The King's Hall, Skokloster Castle; scan by Erik Lernestål"
CAPABILITY = "skokloster_castle_rgb_depth_bullet_traversal"
READY_MARKER_NAME = "habitat-sim-publication-ready.json"
PLAN_PLACEHOLDER = "__NPA_RENDERED_PLAN_SHA256__"
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
PLATFORM_LITERAL_ENV = {
    "KUBERNETES_SERVICE_HOST",
    "KUBERNETES_SERVICE_PORT",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
}
PLATFORM_SECRET_ENV = {
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
}
MAX_PRIVATE_RECEIPT_BYTES = 1024 * 1024
RFC3339_UTC = re.compile(
    r"[0-9]{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12][0-9]|3[01])"
    r"T(?:[01][0-9]|2[0-3]):[0-5][0-9]:[0-5][0-9](?:\.[0-9]+)?Z"
)


def _utc_instant(value: object) -> datetime:
    assert isinstance(value, str) and RFC3339_UTC.fullmatch(value)
    instant = datetime.fromisoformat(value.removesuffix("Z") + "+00:00")
    assert instant.utcoffset() is not None and instant.utcoffset().total_seconds() == 0
    return instant


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
    assert provider["state"] == reservation["state"] == "READY"
    assert _utc_instant(provider["verified_at"]) >= _utc_instant(
        receipt["transaction_started_at"]
    )
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


def _exact_keys(value: object, keys: set[str]) -> dict[str, object]:
    assert isinstance(value, dict) and set(value) == keys
    return value


def _expected_plan_argv(
    plan: dict[str, object], receipt: dict[str, object], output_uri: str
) -> list[object]:
    return [
        "python3",
        "-m",
        "npa.workflows.habitat_sim_smoke",
        "--output-dir",
        plan["output_dir"],
        "--output-uri",
        output_uri,
        "--run-id",
        receipt["run_id"],
        "--plan-sha256",
        PLAN_PLACEHOLDER,
    ]


def _rendered_plan(
    receipt: dict[str, object], provider: dict[str, object]
) -> tuple[dict[str, object], str, dict[str, object]]:
    reference = _exact_keys(receipt["rendered_plan"], {"path", "sha256"})
    plan, _path, payload = _private_json(reference["path"])
    plan_sha256 = hashlib.sha256(payload).hexdigest()
    assert SHA256.fullmatch(str(reference["sha256"]))
    assert plan_sha256 == reference["sha256"]
    _exact_keys(
        plan,
        {
            "schema_version",
            "workflow_name",
            "workflow_sha256",
            "run_id",
            "image",
            "namespace",
            "pod_name",
            "node_name",
            "output_dir",
            "output_uri",
            "pod_command",
            "pod_args",
            "environment",
            "runtime_uid",
            "gpu",
            "skypilot_sha256",
            "setup_sha256",
            "run_sha256",
            "submission_id",
        },
    )
    assert plan["schema_version"] == "npa.habitat-sim.rendered-plan.v1"
    assert plan["workflow_name"] == "habitat-sim-smoke"
    assert plan["workflow_sha256"] == receipt["workflow_sha256"]
    for key in ("run_id", "image", "namespace", "pod_name"):
        assert plan[key] == receipt[key]
    assert plan["node_name"] == provider["kubernetes_node"]["name"]
    assert plan["runtime_uid"] == 1000
    core_environment = {
        "NPA_TASK_IMAGE": receipt["image"],
        "NPA_WORKFLOW_NAME": "habitat-sim-smoke",
        "NPA_WORKFLOW_RUN_ID": receipt["run_id"],
        "NPA_WORKFLOW_STATE": "render-traversal",
    }
    assert isinstance(plan["environment"], dict)
    assert all(
        plan["environment"].get(key) == value for key, value in core_environment.items()
    )
    assert plan["gpu"] == {
        "accelerator": "RTX PRO 6000 Blackwell",
        "count": 1,
        "compute_capability": "12.0",
    }
    output_uri = f"s3://{receipt['storage']['bucket']}/{receipt['storage']['prefix'].rstrip('/')}/"
    assert plan["output_uri"] == output_uri
    expected = _expected_plan_argv(plan, receipt, output_uri)
    assert plan["pod_command"] + plan["pod_args"] == expected
    assert sum(item == PLAN_PLACEHOLDER for item in expected) == 1
    task = _canonical_skypilot_task(receipt, plan)
    submission = _submitted_task(receipt, plan, plan_sha256, task)
    return plan, plan_sha256, submission


def _canonical_skypilot_task(
    receipt: dict[str, object], plan: dict[str, object]
) -> dict[str, object]:
    reference = _exact_keys(receipt["rendered_skypilot"], {"path", "sha256"})
    payload, _path = _private_bytes(reference["path"])
    digest = hashlib.sha256(payload).hexdigest()
    assert SHA256.fullmatch(str(reference["sha256"]))
    assert digest == reference["sha256"] == plan["skypilot_sha256"]
    documents = list(yaml.safe_load_all(payload))
    assert len(documents) == 2 and documents[0] == {
        "name": "habitat-sim-smoke",
        "execution": "serial",
    }
    task = _exact_keys(documents[1], {"name", "resources", "envs", "setup", "run"})
    assert task["name"] == "render-traversal"
    assert task["envs"] == plan["environment"]
    assert task["resources"]["image_id"] == "docker:" + str(receipt["image"])
    assert hashlib.sha256(task["setup"].encode()).hexdigest() == plan["setup_sha256"]
    assert hashlib.sha256(task["run"].encode()).hexdigest() == plan["run_sha256"]
    assert "Habitat-Sim refuses a Python source overlay" in task["setup"]
    assert "/opt/venv/bin/python" in task["setup"]
    assert (
        " ".join(
            str(item) for item in _expected_plan_argv(plan, receipt, plan["output_uri"])
        )
        in task["run"]
    )
    return task


def _submitted_task(
    receipt: dict[str, object],
    plan: dict[str, object],
    plan_sha256: str,
    task: dict[str, object],
) -> dict[str, object]:
    reference = _exact_keys(receipt["submitted_task"], {"path", "sha256"})
    submission, _path, payload = _private_json(reference["path"])
    assert hashlib.sha256(payload).hexdigest() == reference["sha256"]
    _exact_keys(
        submission,
        {
            "schema_version",
            "task_id",
            "task_name",
            "skypilot_sha256",
            "setup_sha256",
            "run_sha256",
            "image",
            "pod_command",
            "pod_args",
            "platform_environment",
        },
    )
    assert submission["schema_version"] == "npa.habitat-sim.submitted-task.v1"
    assert submission["task_id"] == plan["submission_id"]
    assert submission["task_name"] == task["name"]
    assert submission["image"] == receipt["image"]
    for key in ("skypilot_sha256", "setup_sha256", "run_sha256"):
        assert submission[key] == plan[key]
    expected = _expected_plan_argv(plan, receipt, plan["output_uri"])
    expected = [plan_sha256 if item == PLAN_PLACEHOLDER else item for item in expected]
    assert submission["pod_command"] + submission["pod_args"] == expected
    assert isinstance(submission["platform_environment"], list)
    return submission


def _termination_receipt(
    terminated: dict[str, object], receipt: dict[str, object], plan_sha256: str
) -> dict[str, object]:
    termination = json.loads(terminated.get("message", ""))
    _exact_keys(
        termination,
        {
            "schema_version",
            "workflow_name",
            "run_id",
            "rendered_plan_sha256",
            "image_digest",
            "proof_sha256",
            "manifest_key",
            "manifest_sha256",
            "ready_key",
            "ready_sha256",
            "exit_status",
        },
    )
    expected = {
        "schema_version": "npa.habitat-sim.pod-termination.v1",
        "workflow_name": "habitat-sim-smoke",
        "run_id": receipt["run_id"],
        "rendered_plan_sha256": plan_sha256,
        "image_digest": str(receipt["image"]).rsplit("@", 1)[1],
        "exit_status": 0,
    }
    assert all(termination[key] == value for key, value in expected.items())
    for key in ("proof_sha256", "manifest_sha256", "ready_sha256"):
        assert SHA256.fullmatch(str(termination[key]))
    return termination


def _assert_pod_completion(
    pod: dict[str, object],
    receipt: dict[str, object],
    plan: dict[str, object],
    plan_sha256: str,
    submission: dict[str, object],
) -> tuple[str, dict[str, object]]:
    image = str(receipt["image"])
    digest = image.rsplit("@", 1)[1]
    containers = pod["spec"].get("containers", [])
    statuses = pod["status"].get("containerStatuses", [])
    assert pod["status"].get("phase") == "Succeeded"
    assert len(containers) == len(statuses) == 1
    assert (
        pod["metadata"].get("labels", {}).get("npa.nebius.com/task-name")
        == submission["task_name"]
    )
    assert (
        pod["metadata"].get("annotations", {}).get("npa.nebius.com/task-id")
        == submission["task_id"]
    )
    assert containers[0].get("image") == image
    assert containers[0].get("name") == statuses[0].get("name")
    assert containers[0].get("command", []) == submission["pod_command"]
    assert containers[0].get("args", []) == submission["pod_args"]
    environment = containers[0].get("env", [])
    assert len({row.get("name") for row in environment}) == len(environment)
    assert not containers[0].get("envFrom")
    declared = [
        {"name": name, "value": value} for name, value in plan["environment"].items()
    ]
    allowed = declared + submission["platform_environment"]
    assert environment == allowed
    assert len({row["name"] for row in allowed}) == len(allowed)
    for row in submission["platform_environment"]:
        assert set(row) in ({"name", "value"}, {"name", "valueFrom"})
        if "valueFrom" in row:
            assert row["name"] in PLATFORM_SECRET_ENV
            assert set(row["valueFrom"]) == {"secretKeyRef"}
            assert set(row["valueFrom"]["secretKeyRef"]) <= {"name", "key", "optional"}
            assert {"name", "key"} <= set(row["valueFrom"]["secretKeyRef"])
        else:
            assert row["name"] in PLATFORM_LITERAL_ENV
    security = containers[0].get("securityContext", {})
    assert security.get("runAsNonRoot") is True
    assert security.get("runAsUser") == plan["runtime_uid"]
    image_id = str(statuses[0].get("imageID", ""))
    assert re.fullmatch(r"(?:docker-pullable://)?.+@" + re.escape(digest), image_id)
    terminated = statuses[0].get("state", {}).get("terminated")
    assert terminated and terminated.get("exitCode") == 0
    termination = _termination_receipt(terminated, receipt, plan_sha256)
    return image_id, termination


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
    assert "proof_sha256" not in receipt
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


def _storage_client(receipt: dict[str, object]):
    storage = receipt["storage"]
    return boto3.client("s3", endpoint_url=storage["endpoint"])


def _listed_keys(client: object, bucket: str, prefix: str) -> set[str]:
    keys: set[str] = set()
    token: str | None = None
    while True:
        kwargs: dict[str, object] = {"Bucket": bucket, "Prefix": prefix}
        if token is not None:
            kwargs["ContinuationToken"] = token
        response = client.list_objects_v2(**kwargs)
        keys.update(str(row["Key"]) for row in response.get("Contents", []))
        if not response.get("IsTruncated"):
            return keys
        token = str(response["NextContinuationToken"])


def _ready_marker(
    client: object,
    bucket: str,
    prefix: str,
    termination: dict[str, object],
) -> dict[str, object]:
    ready_key = f"{prefix}/{READY_MARKER_NAME}"
    assert termination["ready_key"] == ready_key
    ready_payload = client.get_object(Bucket=bucket, Key=ready_key)["Body"].read()
    assert hashlib.sha256(ready_payload).hexdigest() == termination["ready_sha256"]
    ready = json.loads(ready_payload)
    _exact_keys(
        ready,
        {
            "schema_version",
            "stage_prefix",
            "manifest_key",
            "manifest_sha256",
            "inventory_sha256",
            "object_count",
            "proof_key",
            "proof_sha256",
        },
    )
    assert ready["schema_version"] == "npa.habitat-sim.publication-ready.v1"
    assert re.fullmatch(
        re.escape(prefix) + r"/\.staging/[0-9a-f]{32}", ready["stage_prefix"]
    )
    assert (
        ready["manifest_key"]
        == ready["stage_prefix"] + "/habitat-sim-publication-manifest.json"
    )
    for key in ("manifest_sha256", "inventory_sha256", "proof_sha256"):
        assert SHA256.fullmatch(str(ready[key]))
    for key in ("manifest_key", "manifest_sha256", "proof_sha256"):
        assert ready[key] == termination[key]
    return ready


def _publication_manifest(
    client: object, bucket: str, ready: dict[str, object]
) -> dict[str, object]:
    manifest_payload = client.get_object(Bucket=bucket, Key=ready["manifest_key"])[
        "Body"
    ].read()
    assert hashlib.sha256(manifest_payload).hexdigest() == ready["manifest_sha256"]
    manifest = json.loads(manifest_payload)
    _exact_keys(
        manifest,
        {
            "schema_version",
            "solution",
            "capability",
            "execution_binding",
            "provenance",
            "object_count",
            "objects",
        },
    )
    assert manifest["schema_version"] == "npa.habitat-sim.publication-manifest.v1"
    assert manifest["solution"] == "habitat-sim"
    assert manifest["capability"] == CAPABILITY
    objects = manifest["objects"]
    assert isinstance(objects, list) and objects
    assert manifest["object_count"] == ready["object_count"] == len(objects)
    for row in objects:
        _exact_keys(row, {"path", "key", "bytes", "sha256", "media_type"})
        assert row["key"] == ready["stage_prefix"] + "/" + row["path"]
        assert isinstance(row["bytes"], int) and row["bytes"] > 0
        assert SHA256.fullmatch(str(row["sha256"]))
    assert len({row["path"] for row in objects}) == len(objects)
    assert len({row["key"] for row in objects}) == len(objects)
    canonical = (
        json.dumps(objects, separators=(",", ":"), sort_keys=True) + "\n"
    ).encode()
    assert hashlib.sha256(canonical).hexdigest() == ready["inventory_sha256"]
    expected_keys = {row["key"] for row in objects} | {ready["manifest_key"]}
    assert _listed_keys(client, bucket, ready["stage_prefix"] + "/") == expected_keys
    proof_rows = [row for row in objects if row["path"] == "habitat-sim-smoke.json"]
    assert len(proof_rows) == 1
    assert proof_rows[0]["key"] == ready["proof_key"]
    assert proof_rows[0]["sha256"] == ready["proof_sha256"]
    return manifest


def _publication(
    receipt: dict[str, object], termination: dict[str, object]
) -> tuple[dict[str, object], dict[str, object], object]:
    storage = receipt["storage"]
    client = _storage_client(receipt)
    bucket = str(storage["bucket"])
    prefix = str(storage["prefix"]).rstrip("/")
    ready = _ready_marker(client, bucket, prefix, termination)
    manifest = _publication_manifest(client, bucket, ready)
    return ready, manifest, client


def _proof(
    receipt: dict[str, object],
    ready: dict[str, object],
    client: object,
) -> tuple[dict[str, object], bytes]:
    storage = receipt["storage"]
    payload = client.get_object(Bucket=storage["bucket"], Key=ready["proof_key"])[
        "Body"
    ].read()
    assert hashlib.sha256(payload).hexdigest() == ready["proof_sha256"]
    return json.loads(payload), payload


def _assert_observation_readback(
    proof: dict[str, object],
    receipt: dict[str, object],
    manifest: dict[str, object],
    client: object,
) -> None:
    storage = receipt["storage"]
    rows = {row["path"]: row for row in manifest["objects"]}
    frames = proof["rendered_observations"]["frames"]
    assert len(frames) == proof["rendered_rgb_frame_count"]
    expected_paths = {"habitat-sim-smoke.json"}
    for frame in frames:
        for kind in ("rgb_png", "depth_npy", "depth_preview_png"):
            relative = frame[f"{kind}_path"]
            expected_paths.add(relative)
            row = rows[relative]
            assert row["bytes"] == frame[f"{kind}_bytes"]
            assert row["sha256"] == frame[f"{kind}_sha256"]
            assert row["media_type"] == frame[f"{kind}_media_type"]
            payload = client.get_object(Bucket=storage["bucket"], Key=row["key"])[
                "Body"
            ].read()
            assert len(payload) == frame[f"{kind}_bytes"]
            assert hashlib.sha256(payload).hexdigest() == frame[f"{kind}_sha256"]
    assert set(rows) == expected_paths


def _assert_proof_root(proof: dict[str, object]) -> None:
    keys = {
        "schema_version",
        "solution",
        "capability",
        "capabilities_exercised",
        "execution_binding",
        "source_revision",
        "source",
        "scene_id",
        "scene_sha256",
        "scene_license",
        "scene",
        "rendered_rgb_frame_count",
        "rendered_depth_frame_count",
        "rendered_observations",
        "finite_depth_statistics",
        "agent",
        "agent_start",
        "agent_end",
        "agent_displacement",
        "bullet",
        "bullet_step_count",
        "measured_fps",
        "renderer_egl_evidence",
        "observed_gpu",
        "observed_rtx_gpu_model",
        "observed_rtx_gpu_architecture",
        "observed_rtx_gpu_count",
        "pod_observed_immutable_image",
        "pod_observed_image_digest",
        "image_observation_source",
        "exit_status",
    }
    _exact_keys(proof, keys)
    assert proof["schema_version"] == "npa.habitat-sim.smoke.v1"
    assert proof["solution"] == "habitat-sim"
    assert proof["capability"] == CAPABILITY
    assert set(proof["capabilities_exercised"]) == {
        CAPABILITY,
        "headless_nvidia_egl_rgb_depth_render",
        "bullet_physics_world_step",
        "greedy_geodesic_agent_traversal",
    }


def _assert_execution_and_source(
    proof: dict[str, object], receipt: dict[str, object], plan_sha256: str
) -> tuple[dict[str, object], dict[str, object]]:
    binding = _exact_keys(
        proof["execution_binding"],
        {
            "workflow_name",
            "run_id",
            "rendered_plan_sha256",
            "runtime_uid",
            "runtime_gid",
        },
    )
    assert binding["workflow_name"] == "habitat-sim-smoke"
    assert binding["run_id"] == receipt["run_id"]
    assert binding["rendered_plan_sha256"] == plan_sha256
    assert binding["runtime_uid"] == 1000 and binding["runtime_gid"] != 0
    assert proof["source_revision"] == SOURCE_REVISION
    source = _exact_keys(
        proof["source"],
        {
            "repository",
            "requested_revision",
            "observed_revision",
            "license",
            "manifest_sha256",
        },
    )
    assert source["repository"] == "https://github.com/facebookresearch/habitat-sim"
    assert (
        source["requested_revision"] == source["observed_revision"] == SOURCE_REVISION
    )
    assert source["license"] == "MIT"
    expected_source_manifest = (
        ROOT / "npa/docker/workbench/habitat-sim/source-manifest.json"
    )
    assert (
        source["manifest_sha256"]
        == hashlib.sha256(expected_source_manifest.read_bytes()).hexdigest()
    )
    return binding, source


def _assert_archive_members(scene: dict[str, object]) -> None:
    expected = (
        (
            scene["archive_member"],
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.glb",
            SCENE_SHA256,
        ),
        (
            scene["navmesh_archive_member"],
            "data/scene_datasets/habitat-test-scenes/skokloster-castle.navmesh",
            NAVMESH_SHA256,
        ),
    )
    for member, expected_name, expected_hash in expected:
        _exact_keys(member, {"name", "bytes", "compressed_bytes", "crc32", "sha256"})
        assert member["name"] == expected_name and member["sha256"] == expected_hash
        assert member["bytes"] > 0 and member["compressed_bytes"] > 0
        assert re.fullmatch(r"[0-9a-f]{8}", member["crc32"])


def _assert_scene(proof: dict[str, object]) -> dict[str, object]:
    assert proof["scene_id"] == "habitat_test_scenes/skokloster-castle.glb"
    assert proof["scene_sha256"] == SCENE_SHA256
    scene = _exact_keys(
        proof["scene"],
        {
            "source",
            "archive",
            "id",
            "sha256",
            "bytes",
            "archive_member",
            "license",
            "license_url",
            "attribution",
            "original_asset",
            "modification_notice",
            "immutability_boundary",
            "navmesh_sha256",
            "navmesh_bytes",
            "navmesh_archive_member",
        },
    )
    _exact_keys(
        scene["archive"],
        {
            "url",
            "url_role",
            "checked_at_utc",
            "response_metadata",
            "bytes",
            "sha256",
            "url_is_mutable",
            "zip_integrity",
            "ephemeral_copy_removed",
            "unrelated_members_extracted",
        },
    )
    _exact_keys(
        scene["archive"]["response_metadata"],
        {"status", "content_length", "content_type", "etag", "last_modified"},
    )
    assert scene["archive"]["url"] == ARCHIVE_URL
    assert scene["archive"]["bytes"] == ARCHIVE_BYTES
    assert scene["archive"]["sha256"] == ARCHIVE_SHA256
    assert scene["archive"]["url_is_mutable"] is True
    assert scene["archive"]["zip_integrity"] == "pass"
    assert scene["archive"]["ephemeral_copy_removed"] is True
    assert scene["archive"]["unrelated_members_extracted"] is False
    assert scene["id"] == proof["scene_id"]
    assert scene["sha256"] == SCENE_SHA256
    _assert_archive_members(scene)
    assert scene["navmesh_sha256"] == NAVMESH_SHA256
    assert scene["license"] == "CC BY 4.0"
    assert scene["license_url"] == ASSET_LICENSE_URL
    assert scene["attribution"] == ATTRIBUTION
    _exact_keys(scene["original_asset"], {"name", "creator", "scan_credit", "url"})
    assert scene["original_asset"] == {
        "name": "The King's Hall",
        "creator": "Skokloster Castle",
        "scan_credit": "Erik Lernestål",
        "url": "https://sketchfab.com/3d-models/the-kings-hall-d18155613363445b9b68c0c67196d98d",
    }
    assert scene["modification_notice"]
    assert proof["scene_license"] == "CC BY 4.0"
    return scene


def _assert_frame(frame: dict[str, object], index: int) -> None:
    keys = {
        "index",
        "action",
        "rgb_shape",
        "rgb_raw_sha256",
        "rgb_png_path",
        "rgb_png_media_type",
        "rgb_png_bytes",
        "rgb_png_sha256",
        "depth_shape",
        "depth_raw_sha256",
        "depth_npy_path",
        "depth_npy_media_type",
        "depth_npy_bytes",
        "depth_npy_sha256",
        "depth_preview_png_path",
        "depth_preview_png_media_type",
        "depth_preview_png_bytes",
        "depth_preview_png_sha256",
    }
    _exact_keys(frame, keys)
    assert frame["index"] == index
    assert frame["rgb_shape"] == [240, 320, 4]
    assert frame["depth_shape"] == [240, 320]
    hashes = (
        "rgb_raw_sha256",
        "rgb_png_sha256",
        "depth_raw_sha256",
        "depth_npy_sha256",
        "depth_preview_png_sha256",
    )
    assert all(SHA256.fullmatch(frame[key]) for key in hashes)
    assert frame["rgb_png_media_type"] == "image/png"
    assert frame["depth_npy_media_type"] == "application/x-npy"
    assert frame["depth_preview_png_media_type"] == "image/png"


def _assert_observations(proof: dict[str, object]) -> list[dict[str, object]]:
    assert proof["rendered_rgb_frame_count"] > 1
    assert proof["rendered_depth_frame_count"] == proof["rendered_rgb_frame_count"]
    observations = _exact_keys(
        proof["rendered_observations"], {"rgb", "depth", "frames", "saved_directory"}
    )
    _exact_keys(
        observations["rgb"], {"frame_count", "shape", "dtype", "aggregate_raw_sha256"}
    )
    _exact_keys(
        observations["depth"], {"frame_count", "shape", "dtype", "aggregate_raw_sha256"}
    )
    frames = observations["frames"]
    assert len(frames) == proof["rendered_rgb_frame_count"]
    assert observations["rgb"]["frame_count"] == len(frames)
    assert observations["depth"]["frame_count"] == len(frames)
    assert observations["rgb"]["shape"] == [240, 320, 4]
    assert observations["depth"]["shape"] == [240, 320]
    assert observations["rgb"]["dtype"] == "uint8"
    assert observations["depth"]["dtype"] == "float32"
    assert SHA256.fullmatch(observations["rgb"]["aggregate_raw_sha256"])
    assert SHA256.fullmatch(observations["depth"]["aggregate_raw_sha256"])
    for index, frame in enumerate(frames):
        _assert_frame(frame, index)
    assert len({row["rgb_raw_sha256"] for row in frames}) > 1
    assert len({row["depth_raw_sha256"] for row in frames}) > 1
    return frames


def _assert_depth_and_dynamics(
    proof: dict[str, object], frames: list[dict[str, object]]
) -> None:
    finite = proof["finite_depth_statistics"]
    assert set(finite) == {"count", "minimum", "maximum", "mean", "standard_deviation"}
    assert finite["count"] > 0
    assert all(math.isfinite(float(finite[key])) for key in finite if key != "count")
    assert 0 <= finite["minimum"] <= finite["mean"] <= finite["maximum"]
    assert finite["maximum"] > 0 and finite["standard_deviation"] >= 0
    agent = _exact_keys(
        proof["agent"],
        {
            "start",
            "end",
            "goal",
            "displacement",
            "planned_geodesic_distance",
            "actions",
            "collision_count",
        },
    )
    assert proof["agent_start"] == agent["start"]
    assert proof["agent_end"] == agent["end"]
    assert proof["agent_displacement"] == agent["displacement"] > 0.1
    assert len(agent["actions"]) == len(frames)
    assert "move_forward" in agent["actions"]
    assert proof["bullet_step_count"] > 0
    bullet = _exact_keys(
        proof["bullet"],
        {
            "built_with_bullet",
            "enabled",
            "step_count",
            "world_time_start",
            "world_time_end",
        },
    )
    assert bullet["built_with_bullet"] is True
    assert bullet["enabled"] is True
    assert bullet["step_count"] == proof["bullet_step_count"] == len(frames)
    assert bullet["world_time_end"] > bullet["world_time_start"]
    assert math.isfinite(proof["measured_fps"]) and proof["measured_fps"] > 0


def _assert_renderer_and_gpu(
    proof: dict[str, object], receipt: dict[str, object], provider: dict[str, object]
) -> None:
    egl = _exact_keys(
        proof["renderer_egl_evidence"], {"backend", "display_unset", "gl", "libraries"}
    )
    _exact_keys(egl["gl"], {"vendor", "renderer", "version"})
    assert egl["backend"] == "EGL" and egl["display_unset"] is True
    assert "nvidia" in egl["gl"]["vendor"].lower()
    assert any("libEGL_nvidia.so" in value for value in egl["libraries"])
    gpu = _exact_keys(
        proof["observed_gpu"],
        {"count", "model", "architecture", "compute_capability", "observation_source"},
    )
    assert gpu["count"] == 1 and gpu["architecture"] == "Blackwell"
    assert gpu["compute_capability"] == "12.0"
    normalized_model = re.sub(r"[^A-Z0-9]+", "", gpu["model"].upper())
    assert "RTXPRO6000BLACKWELL" in normalized_model and "B200" not in normalized_model
    assert proof["observed_rtx_gpu_architecture"] == "Blackwell"
    assert proof["observed_rtx_gpu_count"] == 1
    assert proof["observed_rtx_gpu_model"] == gpu["model"]
    assert gpu["observation_source"] == "nvidia-smi inside the workload pod"
    assert provider["accelerator"] == "RTX PRO 6000 Blackwell"
    assert proof["exit_status"] == 0
    assert proof["pod_observed_immutable_image"] == receipt["image"]
    assert proof["pod_observed_image_digest"] == str(receipt["image"]).rsplit("@", 1)[1]
    assert (
        proof["image_observation_source"]
        == "NPA_TASK_IMAGE set from the submitted exact digest"
    )


def _assert_manifest_provenance(
    manifest: dict[str, object],
    source: dict[str, object],
    scene: dict[str, object],
    binding: dict[str, object],
) -> None:
    provenance = _exact_keys(
        manifest["provenance"],
        {
            "source_revision",
            "source_license",
            "source_manifest_sha256",
            "archive_url",
            "archive_sha256",
            "scene_id",
            "scene_sha256",
            "navmesh_sha256",
            "asset_license",
            "asset_license_url",
            "attribution",
            "original_asset",
            "modification_notice",
        },
    )
    assert provenance["source_revision"] == SOURCE_REVISION
    assert provenance["source_manifest_sha256"] == source["manifest_sha256"]
    assert provenance["archive_url"] == ARCHIVE_URL
    assert provenance["archive_sha256"] == ARCHIVE_SHA256
    assert provenance["scene_sha256"] == SCENE_SHA256
    assert provenance["navmesh_sha256"] == NAVMESH_SHA256
    assert provenance["asset_license"] == "CC BY 4.0"
    assert provenance["asset_license_url"] == ASSET_LICENSE_URL
    assert provenance["attribution"] == ATTRIBUTION
    assert provenance["original_asset"] == scene["original_asset"]
    assert provenance["modification_notice"] == scene["modification_notice"]
    assert manifest["execution_binding"] == binding


def _assert_proof(
    proof: dict[str, object],
    receipt: dict[str, object],
    provider: dict[str, object],
    plan_sha256: str,
    manifest: dict[str, object],
) -> None:
    _assert_proof_root(proof)
    binding, source = _assert_execution_and_source(proof, receipt, plan_sha256)
    scene = _assert_scene(proof)
    frames = _assert_observations(proof)
    _assert_depth_and_dynamics(proof, frames)
    _assert_renderer_and_gpu(proof, receipt, provider)
    _assert_manifest_provenance(manifest, source, scene, binding)


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
    plan, plan_sha256, submission = _rendered_plan(receipt, provider)
    pod = _pod(receipt)
    node = _node(receipt, str(pod["spec"]["nodeName"]))
    _assert_pod_provider_node(pod, node, provider)
    image_id, termination = _assert_pod_completion(
        pod, receipt, plan, plan_sha256, submission
    )
    ready, manifest, client = _publication(receipt, termination)
    proof, payload = _proof(receipt, ready, client)
    assert payload and proof["pod_observed_image_digest"] in image_id
    assert hashlib.sha256(payload).hexdigest() == termination["proof_sha256"]
    _assert_proof(proof, receipt, provider, plan_sha256, manifest)
    _assert_observation_readback(proof, receipt, manifest, client)
