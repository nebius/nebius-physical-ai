from __future__ import annotations

from npa.deploy.images import (
    DEFAULT_CONTAINER_REGISTRY,
    container_image_for_tool,
    default_vlm_image,
    default_workbench_image,
)


def test_default_registry_is_real_first_party_registry() -> None:
    assert DEFAULT_CONTAINER_REGISTRY == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw"


def test_non_sonic_workbench_images_resolve_from_supported_tools() -> None:
    assert (
        container_image_for_tool("lancedb")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.2"
    )
    assert container_image_for_tool("detection-training") == (
        "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/"
        "npa-detection-training:bdd100k-real-labelmap-eval-w9-registry-fix-20260519T214847Z"
    )
    assert (
        container_image_for_tool("groot")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-groot:0.1.0"
    )


def test_byo_workflow_images_have_pushed_defaults(monkeypatch) -> None:
    monkeypatch.delenv("NPA_VLM_IMAGE", raising=False)
    monkeypatch.delenv("NPA_WORKBENCH_IMAGE", raising=False)
    monkeypatch.delenv("NPA_REGISTRY", raising=False)

    assert (
        default_vlm_image()
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-cosmos:1.0.9"
    )
    assert (
        default_workbench_image()
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-genesis:0.4.6"
    )


def test_byo_workflow_images_honor_env(monkeypatch) -> None:
    monkeypatch.setenv("NPA_VLM_IMAGE", "registry.example/npa-vlm:custom")
    monkeypatch.setenv("NPA_WORKBENCH_IMAGE", "registry.example/npa-workbench:custom")

    assert default_vlm_image() == "registry.example/npa-vlm:custom"
    assert default_workbench_image() == "registry.example/npa-workbench:custom"
