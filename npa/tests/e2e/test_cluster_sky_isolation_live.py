"""Validate cluster readiness against an explicitly selected owned SkyPilot API."""

from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.request import urlopen

import pytest
import yaml

from npa.cli.cluster import terraform_lifecycle
from npa.orchestration.skypilot.k8s_gpu_catalog import discover_kubernetes_gpu_catalog
from npa.orchestration.skypilot.local_api import IsolatedApiError


pytestmark = [
    pytest.mark.e2e,
    pytest.mark.skipif(
        os.environ.get("NPA_INTEGRATION_E2E") != "1",
        reason="requires an explicitly selected live cluster",
    ),
]


def test_cluster_sky_readiness_and_ambient_api_refusal(monkeypatch) -> None:
    selected = os.environ.get("NPA_TEST_CLUSTER_ISOLATION_KUBECONFIG", "")
    context = os.environ.get("NPA_TEST_CLUSTER_ISOLATION_CONTEXT", "")
    isolated = os.environ.get("NPA_SKYPILOT_ISOLATED_CONFIG_DIR", "")
    sky = os.environ.get("NPA_SKYPILOT_BIN", "")
    completed_validation = os.environ.get("NPA_TEST_CLUSTER_ISOLATION_SKY_CLUSTER", "")
    if not all((selected, context, isolated, sky, completed_validation)):
        pytest.fail("Live cluster isolation requires explicit kubeconfig, context, state directory, SkyPilot binary, and completed validation cluster name")
    kubeconfig = Path(selected).resolve(strict=True)
    _executable, environment, _override = terraform_lifecycle._check_skypilot_kubernetes(
        kubeconfig, context, sky_bin=sky,
    )
    scope = Path(environment["NPA_SKYPILOT_ISOLATED_API_DIR"])
    assert Path(isolated).resolve() in scope.resolve().parents
    assert Path(environment["KUBECONFIG"]).resolve() == kubeconfig
    assert (Path(environment["HOME"]) / ".kube/config").resolve() == kubeconfig
    configuration = yaml.safe_load(Path(environment["SKYPILOT_GLOBAL_CONFIG"]).read_text())
    assert configuration["kubernetes"]["allowed_contexts"] == [context]
    with urlopen(environment["SKYPILOT_API_SERVER_ENDPOINT"] + "/api/health") as response:
        health = json.load(response)
    assert health["version"] == "0.12.2"
    assert health["status"].lower() == "healthy"
    catalog = discover_kubernetes_gpu_catalog(context=context, kubeconfig=kubeconfig, sky_bin=sky)
    assert not catalog.is_empty
    accelerator = terraform_lifecycle._detect_skypilot_gpu(
        sky, f"k8s/{context}", environment, config_override=_override,
        cwd=kubeconfig.parent,
    )
    name, quantity = accelerator.rsplit(":", 1)
    assert quantity == "1" and 1 in catalog.quantities_by_accelerator[name]
    terraform_lifecycle._wait_for_sky_down(
        sky, completed_validation, environment, config_override=_override,
        cwd=kubeconfig.parent,
    )

    monkeypatch.setenv("SKYPILOT_API_SERVER_ENDPOINT", "http://127.0.0.1:9")
    with pytest.raises(IsolatedApiError, match="different configured API endpoint"):
        terraform_lifecycle._check_skypilot_kubernetes(kubeconfig, context, sky_bin=sky)
