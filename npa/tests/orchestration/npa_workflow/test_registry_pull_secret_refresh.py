"""A pinned-image submit refreshes the cluster's Nebius registry pull secret.

Kubernetes pulls private images with an ``imagePullSecret``, and the Nebius registry
only accepts short-lived IAM tokens, so a cluster whose secret was written days ago
fails every pull with ``401 Unauthorized``. SkyPilot surfaces that as ``ErrImagePull``
and then as resources-unavailable, which reads like a capacity problem and costs a
whole run — so the submit path mints a fresh token first.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pytest

from npa.workflows.sim2real import registry_auth
from npa.workflows.sim2real.k8s_client import KubernetesJobClient

from npa.cli.workbench.workflow import (
    _refresh_kubernetes_pull_secrets,
    nebius_registry_hosts,
)

# Shaped like a rendered plan for the Physical AI Data Factory blueprint, whose
# stages pull three different private images.
RENDERED = """
name: physical-ai-data-factory
resources:
  image_id: docker:cr.us-central1.nebius.cloud/u00j7q/npa-cosmos2-transfer:2.5.1
---
resources:
  image_id: docker:cr.us-central1.nebius.cloud/u00j7q/npa-cosmos-evaluator:0.1.0
---
resources:
  image_id: docker:cr.eu-north1.nebius.cloud/e00cm0/npa-cosmos-curate:0.1.0
"""


@pytest.fixture(autouse=True)
def _registry_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SKYPILOT_DOCKER_SERVER", "cr.eu-north1.nebius.cloud")
    monkeypatch.setenv("SKYPILOT_DOCKER_USERNAME", "iam")
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "test-token")
    # Refresh deliberately ignores the ambient render-time password. Unit tests
    # must never reach a live Docker helper or profile exchange.
    monkeypatch.setattr(registry_auth, "_docker_helper_credential", lambda *_: None)
    monkeypatch.setattr(
        registry_auth, "mint_nebius_registry_token", lambda **_: "test-token"
    )


def test_hosts_are_deduplicated_per_registry() -> None:
    assert nebius_registry_hosts(RENDERED) == [
        "cr.eu-north1.nebius.cloud",
        "cr.us-central1.nebius.cloud",
    ]


@pytest.mark.parametrize(
    "rendered",
    [
        "resources:\n  cloud: kubernetes\n",  # no image pin at all
        "resources:\n  image_id: docker:docker.io/library/python:3.12\n",  # public
        "resources:\n  image_id: docker:ghcr.io/org/image:tag\n",  # other registry
    ],
)
def test_no_nebius_image_means_no_hosts(rendered: str) -> None:
    assert nebius_registry_hosts(rendered) == []


def test_every_host_lands_in_one_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A plan spanning two registries must produce a single, merged refresh.

    The secret holds one dockerconfigjson and each apply replaces it, so refreshing
    host by host would leave only the last host authenticated.
    """

    calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_nebius_registry_pull_secret",
        lambda **kwargs: calls.append(kwargs),
    )
    rendered = tmp_path / "workflow.yaml"
    rendered.write_text(RENDERED, encoding="utf-8")

    _refresh_kubernetes_pull_secrets(
        rendered, k8s_context="target", kubeconfig="/tmp/target-kubeconfig"
    )
    assert calls == [
        {
            "registry_servers": [
                "cr.eu-north1.nebius.cloud",
                "cr.us-central1.nebius.cloud",
            ],
            "kubeconfig": "/tmp/target-kubeconfig",
            "k8s_context": "target",
        }
    ]


def test_each_refresh_delegates_credential_resolution_despite_stale_ambient_password(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Long runtime loops must not reinstall the token minted for their first wave."""

    applied: list[dict[str, object]] = []
    monkeypatch.setenv("SKYPILOT_DOCKER_PASSWORD", "expired-initial-token")
    monkeypatch.setenv("NPA_REGISTRY_PASSWORD", "also-expired")
    monkeypatch.setattr(
        registry_auth,
        "ensure_nebius_registry_pull_secret",
        lambda **kwargs: applied.append(kwargs),
    )
    rendered = tmp_path / "workflow.yaml"
    rendered.write_text(RENDERED, encoding="utf-8")

    _refresh_kubernetes_pull_secrets(rendered)
    _refresh_kubernetes_pull_secrets(rendered)

    assert len(applied) == 2
    assert all("token" not in call for call in applied)
    assert all("username" not in call for call in applied)


def test_the_applied_secret_authenticates_every_host(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """End to end through the real writer: both hosts survive in the applied secret."""

    applied: dict[str, object] = {}

    class _Client:
        def apply_secret(self, manifest):
            applied.update(manifest)

    monkeypatch.setattr(registry_auth, "mint_nebius_registry_token", lambda **_: "tok")
    monkeypatch.setattr(
        KubernetesJobClient,
        "from_environment",
        classmethod(lambda _cls, **_kwargs: _Client()),
    )

    registry_auth.ensure_nebius_registry_pull_secret(
        registry_servers=["cr.eu-north1.nebius.cloud", "cr.us-central1.nebius.cloud"]
    )

    payload = base64.b64decode(applied["data"][".dockerconfigjson"]).decode("utf-8")
    auths = json.loads(payload)["auths"]
    assert sorted(auths) == ["cr.eu-north1.nebius.cloud", "cr.us-central1.nebius.cloud"]
    assert all(entry["password"] == "tok" for entry in auths.values())


def test_non_nebius_hosts_are_dropped_before_applying(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A mixed plan must not produce a secret claiming to cover a foreign registry."""

    applied: dict[str, object] = {}

    class _Client:
        def apply_secret(self, manifest):
            applied.update(manifest)

    monkeypatch.setattr(registry_auth, "mint_nebius_registry_token", lambda **_: "tok")
    monkeypatch.setattr(
        KubernetesJobClient,
        "from_environment",
        classmethod(lambda _cls, **_kwargs: _Client()),
    )

    registry_auth.ensure_nebius_registry_pull_secret(
        registry_servers=["ghcr.io", "cr.us-central1.nebius.cloud", "docker.io"]
    )

    payload = base64.b64decode(applied["data"][".dockerconfigjson"]).decode("utf-8")
    assert sorted(json.loads(payload)["auths"]) == ["cr.us-central1.nebius.cloud"]


def test_no_apply_when_no_host_is_a_nebius_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        registry_auth,
        "mint_nebius_registry_token",
        lambda **_: pytest.fail("must not mint a token with nothing to authenticate"),
    )
    registry_auth.ensure_nebius_registry_pull_secret(registry_servers=["ghcr.io"])


def test_no_refresh_without_a_private_image(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[object] = []
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_nebius_registry_pull_secret",
        lambda **kwargs: calls.append(kwargs),
    )
    rendered = tmp_path / "workflow.yaml"
    rendered.write_text("resources:\n  cloud: kubernetes\n", encoding="utf-8")

    _refresh_kubernetes_pull_secrets(rendered)
    assert calls == []


def test_a_refresh_failure_blocks_the_submit_before_launch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A private-image job must not launch without its Kubernetes pull secret."""

    def boom(**kwargs: object) -> None:
        raise RuntimeError("kubectl not found")

    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_nebius_registry_pull_secret", boom
    )
    rendered = tmp_path / "workflow.yaml"
    rendered.write_text(RENDERED, encoding="utf-8")

    with pytest.raises(RuntimeError, match="imagePullSecret"):
        _refresh_kubernetes_pull_secrets(rendered)


def test_a_missing_rendered_file_is_ignored(tmp_path: Path) -> None:
    _refresh_kubernetes_pull_secrets(tmp_path / "absent.yaml")  # must not raise


def test_a_branch_overlay_is_put_ahead_of_a_baked_npa() -> None:
    """Installing the overlay editable is not enough to displace an image's own npa.

    A workbench image whose npa was installed by a different pip/backend keeps a path
    hook pointing at its baked tree. The overlay install then reports success while
    the stage silently runs the image's older code — observed live as
    `No such command 'cosmos2'` from an image built for cosmos2.
    """

    from npa.orchestration.npa_workflow.skypilot_render import render_task_run_script

    script = render_task_run_script(["npa", "workbench", "cosmos2", "transfer"])
    assert "PYTHONPATH=/tmp/npa-src-overlay/src" in script
    assert 'PYTHONPATH="/tmp/npa-src-overlay/src:$PYTHONPATH"' in script
    # Ahead of the command, or it has no effect on it.
    assert script.index("PYTHONPATH") < script.index("npa workbench")
    # SkyPilot reads ${...} as one of its own placeholders.
    assert "${" not in script
