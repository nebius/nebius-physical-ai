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
