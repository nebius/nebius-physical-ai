from __future__ import annotations

import os

import pytest

from npa.orchestration.skypilot.image_bootstrap_contract import (
    probe_image_capabilities,
)


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot]


def test_image_bootstrap_terminal_probe_live() -> None:
    """Exercise terminal observation and exact cleanup on an operator-selected cluster."""

    image = os.environ.get("NPA_E2E_IMAGE_PROBE_IMAGE", "").strip()
    digest = os.environ.get("NPA_E2E_IMAGE_PROBE_DIGEST", "").strip()
    context = os.environ.get("NPA_E2E_KUBECONTEXT", "").strip()
    if not (image and digest and context):
        pytest.skip(
            "set NPA_E2E_IMAGE_PROBE_IMAGE, NPA_E2E_IMAGE_PROBE_DIGEST, "
            "and NPA_E2E_KUBECONTEXT for the disposable Kubernetes probe"
        )

    evidence = probe_image_capabilities(
        image=image,
        digest=digest,
        context=context,
        kubeconfig=os.environ.get("KUBECONFIG", "").strip(),
    )

    assert evidence.state in {"compatible", "incompatible"}
    assert evidence.cleanup == "verified_deleted"
    assert evidence.detail or evidence.checks


def test_vendor_image_runtime_bootstrap_live() -> None:
    """Require a real vendor image to pass prepared worker checks and cleanup."""

    image = os.environ.get("NPA_E2E_IMAGE_PROBE_IMAGE", "").strip()
    digest = os.environ.get("NPA_E2E_IMAGE_PROBE_DIGEST", "").strip()
    context = os.environ.get("NPA_E2E_KUBECONTEXT", "").strip()
    if not (image and digest and context):
        pytest.skip("Requires an operator-selected vendor image and exact cluster")

    evidence = probe_image_capabilities(
        image=image,
        digest=digest,
        context=context,
        kubeconfig=os.environ.get("KUBECONFIG", "").strip(),
        image_pull_secrets=tuple(
            name.strip()
            for name in os.environ.get("NPA_E2E_IMAGE_PULL_SECRETS", "").split(",")
            if name.strip()
        ),
        runtime_bootstrap=True,
        observation_timeout_seconds=0,
    )

    assert evidence.ok, evidence.detail
    assert evidence.source == "ephemeral_runtime_bootstrap_probe"
    assert "kubernetes_command_override" in evidence.checks
    assert evidence.cleanup == "verified_deleted"
