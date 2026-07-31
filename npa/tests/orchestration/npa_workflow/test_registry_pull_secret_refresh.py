"""A pinned-image submit refreshes the cluster's Nebius registry pull secret.

Kubernetes pulls private images with an ``imagePullSecret``, and the Nebius registry
only accepts short-lived IAM tokens, so a cluster whose secret was written days ago
fails every pull with ``401 Unauthorized``. SkyPilot surfaces that as ``ErrImagePull``
and then as resources-unavailable, which reads like a capacity problem and costs a
whole run — so the submit path mints a fresh token first.
"""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_refresh_is_called_once_per_registry_host(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_nebius_registry_pull_secret",
        lambda *, registry_server, **kwargs: calls.append(registry_server),
    )
    rendered = tmp_path / "workflow.yaml"
    rendered.write_text(RENDERED, encoding="utf-8")

    _refresh_kubernetes_pull_secrets(rendered)
    assert calls == ["cr.eu-north1.nebius.cloud", "cr.us-central1.nebius.cloud"]


def test_no_refresh_without_a_private_image(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_nebius_registry_pull_secret",
        lambda *, registry_server, **kwargs: calls.append(registry_server),
    )
    rendered = tmp_path / "workflow.yaml"
    rendered.write_text("resources:\n  cloud: kubernetes\n", encoding="utf-8")

    _refresh_kubernetes_pull_secrets(rendered)
    assert calls == []


def test_a_refresh_failure_does_not_block_the_submit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An operator without kubectl reach, or a public-image cluster, still submits."""

    def boom(*, registry_server: str, **kwargs: object) -> None:
        raise RuntimeError("kubectl not found")

    monkeypatch.setattr(
        "npa.workflows.sim2real.registry_auth.ensure_nebius_registry_pull_secret", boom
    )
    rendered = tmp_path / "workflow.yaml"
    rendered.write_text(RENDERED, encoding="utf-8")

    _refresh_kubernetes_pull_secrets(rendered)  # must not raise


def test_a_missing_rendered_file_is_ignored(tmp_path: Path) -> None:
    _refresh_kubernetes_pull_secrets(tmp_path / "absent.yaml")  # must not raise
