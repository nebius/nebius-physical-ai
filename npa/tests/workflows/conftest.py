"""Shared fixtures for workflow unit tests."""

from __future__ import annotations

from typing import Any

import pytest


@pytest.fixture
def operator_sim2real_image_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """Route quarantined Sim2Real defaults to a synthetic operator registry.

    Algorithm, artifact, and resume tests need concrete image strings but do
    not exercise publication acceptance. Dedicated resolver and CLI tests cover
    the fail-closed public path.
    """

    from npa.deploy.images import PUBLICATION_QUARANTINE_TOOLS
    from npa.workflows.sim2real import models

    resolve = models.container_image_for_tool

    def resolve_for_test(tool: str, **kwargs: Any) -> str:
        if (
            tool in PUBLICATION_QUARANTINE_TOOLS
            and not kwargs.get("registry")
            and not kwargs.get("tag")
        ):
            kwargs["registry"] = "registry.example.invalid/operator/workbench"
        return resolve(tool, **kwargs)

    monkeypatch.setattr(models, "container_image_for_tool", resolve_for_test)
