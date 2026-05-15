from __future__ import annotations

import base64
import io
import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path
from typing import Any

import boto3
import pytest


NAMESPACE = "argo"
TEMPLATE_NAME = "curate-augment-train"
FIXTURE_URI = (
    "s3://YOUR_S3_BUCKET/"
    "argo-artifacts/fixtures/curate-augment-train-v1/fixture-dataset.txt"
)
BUCKET = "YOUR_S3_BUCKET"
S3_ENDPOINT = "https://storage.eu-north1.nebius.cloud"
S3_PREFIX = "argo-artifacts"
POLL_INTERVAL_SECONDS = 10
TERMINAL_PHASES = {"Succeeded", "Failed", "Error"}
REQUIRED_CHAIN = ["step-curate", "step-augment", "step-train"]
SENTINEL = "W8_WORKFLOW_YAML_BOOTSTRAP_FIXTURE_SENTINEL"


@pytest.mark.e2e_pipeline
def test_curate_augment_train_workflow_template() -> None:
    _require_workflow_e2e()
    namespace = os.environ.get("NPA_ARGO_NAMESPACE", NAMESPACE)
    fixture_uri = os.environ.get("NPA_WORKFLOW_FIXTURE_URI", FIXTURE_URI)

    _run(["argo", "version"])
    _run(["kubectl", "cluster-info"])

    workflow_name = _submit_workflow(namespace, fixture_uri)
    try:
        workflow = _poll_workflow(namespace, workflow_name)
        assert workflow["status"]["phase"] == "Succeeded", json.dumps(
            workflow.get("status", {}),
            indent=2,
            sort_keys=True,
        )

        markers = _load_step_markers(namespace, workflow_name)
        assert sorted(markers) == sorted(REQUIRED_CHAIN)

        final_marker = markers["step-train"]
        assert markers["step-curate"]["chain"] == ["step-curate"]
        assert markers["step-augment"]["chain"] == ["step-curate", "step-augment"]
        assert final_marker["chain"] == REQUIRED_CHAIN
        for marker in markers.values():
            assert marker["source_sentinel_detected"] is True
            assert marker["source_sentinel"] == SENTINEL
        assert markers["step-curate"]["input_size_bytes"] > 0
        assert markers["step-augment"]["input_size_bytes"] > 0
        assert markers["step-train"]["input_size_bytes"] > 0
        _write_result_artifact(workflow_name, workflow, markers)
    finally:
        _run(["argo", "delete", "-n", namespace, workflow_name], check=False)


def _require_workflow_e2e() -> None:
    if os.environ.get("NPA_INTEGRATION_E2E") != "1":
        pytest.skip("set NPA_INTEGRATION_E2E=1 to run WorkflowTemplate e2e tests")
    if not os.environ.get("KUBECONFIG"):
        pytest.skip("KUBECONFIG must point at the isolated Argo cluster kubeconfig")
    for tool in ("argo", "kubectl"):
        if not shutil.which(tool):
            pytest.fail(f"{tool} CLI is required")


def _submit_workflow(namespace: str, fixture_uri: str) -> str:
    result = _run(
        [
            "argo",
            "submit",
            "-n",
            namespace,
            "--from",
            f"workflowtemplate/{TEMPLATE_NAME}",
            "-p",
            f"dataset-uri={fixture_uri}",
            "-o",
            "name",
        ]
    )
    return result.stdout.strip().split("/")[-1]


def _poll_workflow(namespace: str, workflow_name: str) -> dict[str, Any]:
    while True:
        result = _run(["argo", "get", "-n", namespace, workflow_name, "-o", "json"])
        workflow = json.loads(result.stdout)
        phase = workflow.get("status", {}).get("phase")
        if phase in TERMINAL_PHASES:
            return workflow
        time.sleep(POLL_INTERVAL_SECONDS)


def _load_step_markers(namespace: str, workflow_name: str) -> dict[str, dict[str, Any]]:
    s3 = _s3_client_from_secret(namespace)
    prefix = f"{S3_PREFIX}/{workflow_name}/"
    markers: dict[str, dict[str, Any]] = {}

    for key in _list_s3_keys(s3, prefix):
        if not key.endswith("/output.tgz"):
            continue
        marker = _read_marker_from_tgz(s3.get_object(Bucket=BUCKET, Key=key)["Body"].read())
        markers[marker["step_name"]] = marker

    return markers


def _s3_client_from_secret(namespace: str):
    secret = json.loads(
        _run(
            [
                "kubectl",
                "-n",
                namespace,
                "get",
                "secret",
                "argo-s3-credentials",
                "-o",
                "json",
            ]
        ).stdout
    )
    access_key = base64.b64decode(secret["data"]["accessKey"]).decode("utf-8")
    secret_key = base64.b64decode(secret["data"]["secretKey"]).decode("utf-8")
    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("NPA_WORKFLOW_S3_ENDPOINT", S3_ENDPOINT),
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def _list_s3_keys(s3, prefix: str) -> list[str]:
    keys: list[str] = []
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=BUCKET, Prefix=prefix):
        keys.extend(obj["Key"] for obj in page.get("Contents", []))
    return keys


def _read_marker_from_tgz(payload: bytes) -> dict[str, Any]:
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isfile() and member.name.endswith("marker.json"):
                extracted = archive.extractfile(member)
                if extracted is None:
                    break
                return json.load(extracted)
    raise AssertionError("marker.json not found in output artifact")


def _write_result_artifact(
    workflow_name: str,
    workflow: dict[str, Any],
    markers: dict[str, dict[str, Any]],
) -> None:
    path = os.environ.get("NPA_WORKFLOW_TEST_RESULT_PATH")
    if not path:
        return
    payload = {
        "workflow_name": workflow_name,
        "phase": workflow["status"]["phase"],
        "marker_steps": sorted(markers),
        "final_chain": markers["step-train"]["chain"],
    }
    Path(path).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _run(
    args: list[str],
    *,
    check: bool = True,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=Path(__file__).resolve().parents[2],
    )
