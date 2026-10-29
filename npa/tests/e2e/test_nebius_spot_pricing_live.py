"""Validate real serverless pricing requests without allocating any resources."""

from __future__ import annotations

import os
import subprocess
import uuid

import pytest

from npa.clients import nebius
from npa.clients.serverless import EndpointSpec, ServerlessClient

pytestmark = pytest.mark.e2e


class _DryRunValidated(Exception):
    """Stop before a successful validation can be mistaken for a creation."""


def _validate_only(args, **kwargs):
    if args[1:4] not in (["ai", "job", "create"], ["ai", "endpoint", "create"]):
        pytest.fail(
            "Only serverless create-request validation is allowed", pytrace=False
        )
    result = subprocess.run(
        [*args, "--dry-run", "--no-browser", "--no-check-update"],
        **kwargs,
        stdin=subprocess.DEVNULL,
        check=False,
    )
    if result.returncode:
        pytest.fail("The live provider rejected the pricing dry-run", pytrace=False)
    raise _DryRunValidated


@pytest.mark.parametrize("resource", ["job", "endpoint"])
@pytest.mark.parametrize("preemptible", [False, True])
def test_serverless_pricing_request_is_accepted_without_allocation(
    resource, preemptible
):
    if os.environ.get("NPA_SPOT_PRICING_LIVE") != "1":
        pytest.skip("Set NPA_SPOT_PRICING_LIVE=1 for live request validation")
    project = os.environ.get("NPA_E2E_PROJECT_ID", "")
    if not project:
        pytest.fail("An explicit NPA_E2E_PROJECT_ID is required", pytrace=False)
    client = ServerlessClient(
        nebius_bin=nebius._require_nebius(), subprocess_runner=_validate_only
    )
    inputs = {
        "project_id": project,
        "name": "npa-spot-validation-" + uuid.uuid4().hex[:12],
        "image": "docker.io/library/ubuntu:24.04",
        "preset": "1gpu-24vcpu-218gb",
        "preemptible": preemptible,
        "subnet_id": os.environ.get("NPA_E2E_SUBNET_ID", ""),
    }
    with pytest.raises(_DryRunValidated):
        if resource == "endpoint":
            client.create_endpoint(
                EndpointSpec(
                    **inputs,
                    platform="gpu-rtx6000",
                    public=False,
                    container_ports=[8080],
                )
            )
        else:
            client.create_job(
                **inputs,
                gpu_type="gpu-rtx6000",
                gpu_count=1,
                command="true",
                output_path="/tmp/pricing-validation",
            )
