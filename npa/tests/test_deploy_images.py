from __future__ import annotations

from pathlib import Path

from npa.deploy.images import (
    DEFAULT_CONTAINER_REGISTRY,
    SUPPORTED_TOOL_VERSIONS,
    container_image_for_tool,
    default_vlm_image,
    default_workbench_image,
)


def test_default_registry_id_is_placeholder_not_a_real_nebius_id() -> None:
    import re

    # The committed default must be an inert placeholder, never a real Nebius
    # registry identifier (which matches the e0<...> pattern).
    assert DEFAULT_CONTAINER_REGISTRY == "cr.eu-north1.nebius.cloud/<your-registry-id>"
    assert not re.search(r"\be0[0-9][a-z0-9]{12,}\b", DEFAULT_CONTAINER_REGISTRY)


def test_default_registry_id_honors_env(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.example/team")
    assert container_image_for_tool("lancedb") == "registry.example/team/npa-lancedb:0.30.2"


def test_non_sonic_workbench_images_resolve_from_supported_tools() -> None:
    assert (
        container_image_for_tool("lancedb")
        == "cr.eu-north1.nebius.cloud/<your-registry-id>/npa-lancedb:0.30.2"
    )
    assert container_image_for_tool("detection-training") == (
        "cr.eu-north1.nebius.cloud/<your-registry-id>/"
        "npa-detection-training:bdd100k-real-labelmap-eval-w9-registry-fix-20260519T214847Z"
    )
    assert (
        container_image_for_tool("groot")
        == "cr.eu-north1.nebius.cloud/<your-registry-id>/npa-groot:0.1.0"
    )
    assert (
        container_image_for_tool("cosmos3-reason")
        == "cr.eu-north1.nebius.cloud/<your-registry-id>/npa-cosmos3-reason:3.0.1-genuine-sm120"
    )


def test_packaged_supported_tool_versions_match_pyproject() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert SUPPORTED_TOOL_VERSIONS == data["tool"]["npa"]["supported-tools"]


def test_byo_workflow_images_have_pushed_defaults(monkeypatch) -> None:
    monkeypatch.delenv("NPA_VLM_IMAGE", raising=False)
    monkeypatch.delenv("NPA_WORKBENCH_IMAGE", raising=False)
    monkeypatch.delenv("NPA_REGISTRY", raising=False)

    assert (
        default_vlm_image()
        == "cr.eu-north1.nebius.cloud/<your-registry-id>/npa-cosmos:1.0.9"
    )
    assert (
        default_workbench_image()
        == "cr.eu-north1.nebius.cloud/<your-registry-id>/npa-genesis:0.4.6"
    )


def test_byo_workflow_images_honor_env(monkeypatch) -> None:
    monkeypatch.setenv("NPA_VLM_IMAGE", "registry.example/npa-vlm:custom")
    monkeypatch.setenv("NPA_WORKBENCH_IMAGE", "registry.example/npa-workbench:custom")

    assert default_vlm_image() == "registry.example/npa-vlm:custom"
    assert default_workbench_image() == "registry.example/npa-workbench:custom"
