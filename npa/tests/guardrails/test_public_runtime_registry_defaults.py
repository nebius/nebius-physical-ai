"""Guard supported workload defaults against private-registry configuration drift."""

from __future__ import annotations

from dataclasses import fields

from npa.deploy.images import (
    DEFAULT_PUBLIC_CONTAINER_REGISTRY,
    container_image_for_tool,
    publicly_publishable_tools,
)
from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real.models import Sim2RealLoopConfig


def test_every_published_tool_ignores_ambient_private_registry(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.invalid/operator/private")
    prefix = f"{DEFAULT_PUBLIC_CONTAINER_REGISTRY}/"

    images = [container_image_for_tool(tool) for tool in publicly_publishable_tools()]

    assert images
    assert all(image.startswith(prefix) for image in images)


def test_sim2real_owned_images_ignore_generic_registry_configuration(
    monkeypatch,
) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.invalid/operator/private")
    monkeypatch.delenv("NPA_SIM2REAL_REGISTRY", raising=False)
    prefix = f"{DEFAULT_PUBLIC_CONTAINER_REGISTRY}/"

    config = build_config_from_env(run_id="registry-guard")
    image_fields = [
        item.name for item in fields(Sim2RealLoopConfig) if item.name.endswith("_image")
    ]

    assert image_fields
    assert all(str(getattr(config, name)).startswith(prefix) for name in image_fields)


def test_sim2real_custom_registry_is_scoped_and_explicit(monkeypatch) -> None:
    custom = "registry.invalid/operator/validated"
    monkeypatch.setenv("NPA_SIM2REAL_REGISTRY", custom)

    config = build_config_from_env(run_id="custom-registry-guard")

    assert config.augment_image.startswith(f"{custom}/")
