"""Qualify real admission, quota, and S3 isolation on explicitly selected team allocations."""

import json
import os
import uuid
from pathlib import Path

import pytest
from botocore.exceptions import ClientError

from npa.workbench.team.deployment import allocations
from npa.workbench.team.enrollment import kubectl, verify_enrollment
from npa.workbench.team.models import load_config
from npa.workbench.team.storage import PersonalStorage

pytestmark = [pytest.mark.e2e, pytest.mark.timeout(0)]


@pytest.fixture(scope="module")
def live_allocations():
    if (
        os.environ.get("NPA_TEAM_LIVE_E2E") != "1"
        or os.environ.get("NPA_INTEGRATION_E2E") != "1"
    ):
        pytest.skip("Explicit team live qualification is disabled")
    path = os.environ.get("NPA_TEAM_LIVE_CONFIG")
    assert path, "Select a private team configuration"
    bindings = list(allocations(load_config(Path(path))))
    selected = []
    for binding in bindings:
        if not selected or binding.allocation.subject != selected[0].allocation.subject:
            selected.append(binding)
        if len(selected) == 2:
            break
    assert len(selected) == 2, "Qualification requires two distinct people"
    for binding in selected:
        verify_enrollment(binding)
    return selected


def _pod(binding):
    return {
        "apiVersion": "v1",
        "kind": "Pod",
        "metadata": {
            "name": "team-qualification-" + uuid.uuid4().hex[:12],
            "namespace": binding.namespace,
        },
        "spec": {
            "serviceAccountName": "npa-team-worker",
            "automountServiceAccountToken": False,
            "restartPolicy": "Never",
            "containers": [
                {"name": "probe", "image": "ubuntu:24.04", "command": ["true"]}
            ],
        },
    }


def _admit(binding, pod):
    return kubectl(
        binding, "create", "--dry-run=server", "-f", "-", input=json.dumps(pod)
    )


def test_real_admission_rejects_identity_and_token_overrides(live_allocations):
    binding = live_allocations[0]
    assert _admit(binding, _pod(binding)).returncode == 0, (
        "Valid worker pod must pass admission"
    )
    for field, value in (
        ("serviceAccountName", "default"),
        ("automountServiceAccountToken", True),
    ):
        pod = _pod(binding)
        pod["spec"][field] = value
        assert _admit(binding, pod).returncode != 0, "Identity override must be denied"
    pod = _pod(binding)
    pod["spec"]["volumes"] = [
        {
            "name": "token",
            "projected": {"sources": [{"serviceAccountToken": {"path": "token"}}]},
        }
    ]
    assert _admit(binding, pod).returncode != 0, (
        "Explicit token projection must be denied"
    )


def test_real_quota_rejects_personal_overallocation(live_allocations):
    binding = live_allocations[0]
    pod = _pod(binding)
    count = binding.allocation.clusters[binding.cluster] + 1
    pod["spec"]["containers"][0]["resources"] = {"limits": {"nvidia.com/gpu": count}}
    result = _admit(binding, pod)
    assert result.returncode != 0 and "quota" in result.stderr.lower(), (
        "GPU quota must reject over-allocation"
    )


@pytest.mark.parametrize("field", ["containers", "initContainers"])
def test_real_admission_rejects_unaccounted_gpu_resources(live_allocations, field):
    binding = live_allocations[0]
    pod = _pod(binding)
    if field == "initContainers":
        pod["spec"][field] = [dict(pod["spec"]["containers"][0], name="initialize")]
    pod["spec"][field][0]["resources"] = {"limits": {"nvidia.com/gpu.shared": 1}}
    result = _admit(binding, pod)
    assert result.returncode != 0
    assert "team quotas currently support whole NVIDIA GPUs only" in result.stderr


def test_real_storage_principals_cannot_read_or_write_each_others_objects(
    live_allocations,
):
    stores = [
        PersonalStorage(binding.allocation.storage) for binding in live_allocations
    ]
    keys = [
        f"{store.grant.prefix}/qualification/{uuid.uuid4().hex}" for store in stores
    ]
    try:
        for store, key in zip(stores, keys, strict=True):
            store.client.put_object(
                Bucket=store.grant.bucket, Key=key, Body=b"team-qualification"
            )
        for index, store in enumerate(stores):
            other = stores[1 - index]
            key = keys[1 - index]
            for operation in ("get_object", "put_object"):
                arguments = {"Bucket": other.grant.bucket, "Key": key}
                if operation == "put_object":
                    arguments["Body"] = b"must-be-denied"
                with pytest.raises(ClientError) as denied:
                    getattr(store.client, operation)(**arguments)
                assert (
                    denied.value.response["ResponseMetadata"]["HTTPStatusCode"] == 403
                )
    finally:
        for store, key in zip(stores, keys, strict=True):
            store.client.delete_object(Bucket=store.grant.bucket, Key=key)
