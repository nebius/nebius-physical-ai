"""Guard supported workload defaults against private-registry configuration drift."""

from __future__ import annotations

from dataclasses import fields
from pathlib import Path

import yaml

from npa.deploy.images import (
    DEFAULT_PUBLIC_CONTAINER_REGISTRY,
    container_image_for_tool,
    publicly_publishable_tools,
)
from npa.workflows.sim2real.config import build_config_from_env
from npa.workflows.sim2real.models import Sim2RealLoopConfig
from npa.orchestration.npa_workflow.skypilot_render import SkypilotRenderOptions
from npa.orchestration.npa_workflow.spec import load_spec
from npa.orchestration.npa_workflow.submit import prepare_npa_workflow_for_submit


WORKFLOW_DIR = (
    Path(__file__).resolve().parents[3]
    / "npa"
    / "workflows"
    / "workbench"
    / "npa-workflows"
)


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


def test_every_shipped_workflow_keeps_owned_images_on_public_ghcr(
    monkeypatch,
) -> None:
    hostile_registry = "registry.invalid/operator/private"
    monkeypatch.setenv("NPA_REGISTRY", hostile_registry)
    monkeypatch.setenv("NPA_SRC_S3_URI", "s3://test-fixtures/npa-source")
    public_prefix = f"{DEFAULT_PUBLIC_CONTAINER_REGISTRY}/npa-"

    rendered_images: set[str] = set()
    for spec_path in sorted(WORKFLOW_DIR.glob("*.yaml")):
        spec = load_spec(spec_path)
        requires_baked_image = str(
            spec.config.get("require_baked_npa") or ""
        ).lower() in {"1", "true", "yes", "on"}
        prepared = prepare_npa_workflow_for_submit(
            spec_path,
            run_id=f"registry-guard-{spec_path.stem}",
            assume_decision="promote_checkpoint",
            config_overrides=(
                {"source_sha": "0" * 40} if requires_baked_image else None
            ),
            render_options=SkypilotRenderOptions(
                image_overrides=(
                    {"*": f"{public_prefix}runtime@sha256:{'0' * 64}"}
                    if requires_baked_image
                    else {}
                ),
                materialize_registry_secrets=False,
            ),
        )
        try:
            for document in yaml.safe_load_all(
                prepared.skypilot_yaml_path.read_text(encoding="utf-8")
            ):
                if not isinstance(document, dict):
                    continue
                image = str((document.get("resources") or {}).get("image_id") or "")
                if image:
                    rendered_images.add(image.removeprefix("docker:"))
        finally:
            prepared.temp_dir.cleanup()

    assert rendered_images
    assert not any(hostile_registry in image for image in rendered_images)
    assert all(
        image.startswith(public_prefix)
        for image in rendered_images
        if image.rsplit("/", 1)[-1].startswith("npa-")
    )
