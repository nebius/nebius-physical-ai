from __future__ import annotations

import os
import secrets

import pytest

from npa.orchestration.skypilot.image_bootstrap_contract import (
    probe_image_capabilities,
)
from npa.orchestration.skypilot.registry_preflight import verify_kubernetes_image_pull


pytestmark = [pytest.mark.e2e, pytest.mark.e2e_skypilot]


def _target_pull_inputs() -> dict:
    image = os.environ.get("NPA_E2E_IMAGE_PROBE_IMAGE", "").strip()
    context = os.environ.get("NPA_E2E_KUBECONTEXT", "").strip()
    namespace = os.environ.get("NPA_E2E_KUBE_NAMESPACE", "").strip()
    if not (image and context and namespace):
        pytest.skip("Requires an operator-selected image, exact context and namespace")
    return {
        "image": image,
        "context": context,
        "namespace": namespace,
        "kubeconfig": os.environ.get("KUBECONFIG", "").strip(),
        "service_account_name": os.environ.get(
            "NPA_E2E_IMAGE_PULL_SERVICE_ACCOUNT", "skypilot-service-account"
        ),
        "secret_names": tuple(
            name.strip()
            for name in os.environ.get("NPA_E2E_IMAGE_PULL_SECRETS", "").split(",")
            if name.strip()
        ),
    }


def test_target_image_pull_without_deadline_live() -> None:
    """Verify the selected image through real kubelet pull and owned cleanup."""
    evidence = verify_kubernetes_image_pull(**_target_pull_inputs(), timeout_seconds=0)
    assert evidence.ok, evidence.status
    assert evidence.digest.startswith("sha256:")
    assert evidence.cleanup_status == "verified"


def test_target_image_pull_deadline_live() -> None:
    """Expire observation of a deliberately unschedulable owned probe."""
    evidence = verify_kubernetes_image_pull(
        **_target_pull_inputs(),
        timeout_seconds=1,
        pod_placement={
            "nodeSelector": {"npa.nebius.com/probe-only": secrets.token_hex(16)}
        },
    )
    assert evidence.status == "timed_out"
    assert evidence.cleanup_status == "verified"


def test_image_bootstrap_terminal_probe_live() -> None:
    """Exercise terminal observation and exact cleanup on an operator-selected cluster."""

    image = os.environ.get("NPA_E2E_IMAGE_PROBE_IMAGE", "").strip()
    digest = os.environ.get("NPA_E2E_IMAGE_PROBE_DIGEST", "").strip()
    context = os.environ.get("NPA_E2E_KUBECONTEXT", "").strip()
    namespace = os.environ.get("NPA_E2E_KUBE_NAMESPACE", "").strip()
    if not (image and digest and context and namespace):
        pytest.skip(
            "set NPA_E2E_IMAGE_PROBE_IMAGE, NPA_E2E_IMAGE_PROBE_DIGEST, "
            "NPA_E2E_KUBECONTEXT, and NPA_E2E_KUBE_NAMESPACE for the "
            "disposable Kubernetes probe"
        )

    evidence = probe_image_capabilities(
        image=image,
        digest=digest,
        context=context,
        namespace=namespace,
        kubeconfig=os.environ.get("KUBECONFIG", "").strip(),
    )

    assert evidence.state in {"compatible", "incompatible"}
    assert evidence.cleanup == "verified"
    assert evidence.detail or evidence.checks


def test_vendor_image_runtime_bootstrap_live() -> None:
    """Require a real vendor image to pass prepared worker checks and cleanup."""

    image = os.environ.get("NPA_E2E_IMAGE_PROBE_IMAGE", "").strip()
    digest = os.environ.get("NPA_E2E_IMAGE_PROBE_DIGEST", "").strip()
    context = os.environ.get("NPA_E2E_KUBECONTEXT", "").strip()
    namespace = os.environ.get("NPA_E2E_KUBE_NAMESPACE", "").strip()
    if not (image and digest and context and namespace):
        pytest.skip(
            "Requires an operator-selected vendor image, exact cluster, and namespace"
        )

    evidence = probe_image_capabilities(
        image=image,
        digest=digest,
        context=context,
        namespace=namespace,
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
    assert evidence.cleanup == "verified"
