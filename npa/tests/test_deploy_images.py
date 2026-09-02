from __future__ import annotations

from pathlib import Path

from npa.deploy.images import (
    DEFAULT_CONTAINER_REGISTRY,
    SUPPORTED_TOOL_VERSIONS,
    development_image_for_tool,
    container_image_for_tool,
    default_vlm_image,
    default_workbench_image,
    development_tag,
    execution_container_registry,
    registry_from_env,
)


def test_registry_from_env_prefers_npa_registry(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.example/team")
    assert registry_from_env() == "registry.example/team"


def test_registry_from_env_empty_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    assert registry_from_env() == ""


def test_execution_container_registry_defaults_to_public_releases(monkeypatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    assert execution_container_registry() == DEFAULT_CONTAINER_REGISTRY


def test_official_ghcr_namespace_is_public_only() -> None:
    assert DEFAULT_CONTAINER_REGISTRY == "ghcr.io/nebius/nebius-physical-ai"


def test_non_sonic_workbench_images_resolve_from_supported_tools() -> None:
    assert (
        container_image_for_tool("lancedb")
        == "ghcr.io/nebius/nebius-physical-ai/npa-lancedb:"
        "cuda13-b300-0.30.3-sm80-sm90-sm100-sm103-sm120-20260803T031514Z"
    )
    assert container_image_for_tool("detection-training") == (
        "ghcr.io/nebius/nebius-physical-ai/"
        "npa-detection-training:bdd100k-golden-eval-smoke-20260614T210000Z"
    )
    assert (
        container_image_for_tool("groot")
        == "ghcr.io/nebius/nebius-physical-ai/npa-groot:0.1.0"
    )
    assert (
        container_image_for_tool("cosmos2-transfer")
        == "ghcr.io/nebius/nebius-physical-ai/"
        "npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z"
    )
    assert (
        container_image_for_tool("cosmos3") == "ghcr.io/nebius/nebius-physical-ai/"
        "npa-cosmos3:1.2.2-cu130-r6"
    )
    assert (
        container_image_for_tool("cosmos3-reason")
        == "ghcr.io/nebius/nebius-physical-ai/npa-cosmos3-reason:"
        "cuda13-b300-3.0.1-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )
    assert (
        container_image_for_tool("envgen")
        == "ghcr.io/nebius/nebius-physical-ai/npa-envgen:"
        "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )
    assert (
        container_image_for_tool("reference-policy")
        == "ghcr.io/nebius/nebius-physical-ai/npa-reference-policy:"
        "cuda13-b300-0.1.2-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )
    assert (
        container_image_for_tool("loop-eval")
        == "ghcr.io/nebius/nebius-physical-ai/npa-loop-eval:"
        "cuda13-b300-0.1.3-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )


def test_repository_image_defaults_ignore_ambient_private_registry(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.example/private-builds")

    assert container_image_for_tool("retargeting") == (
        "ghcr.io/nebius/nebius-physical-ai/npa-retargeting:0.1.1"
    )
    assert default_workbench_image().startswith(
        "ghcr.io/nebius/nebius-physical-ai/npa-genesis:"
    )


def test_explicit_custom_registry_remains_available() -> None:
    assert container_image_for_tool(
        "retargeting", registry="registry.example/custom"
    ) == "registry.example/custom/npa-retargeting:0.1.1"


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
        default_vlm_image() == "ghcr.io/nebius/nebius-physical-ai/"
        "npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z"
    )
    assert (
        default_workbench_image() == "ghcr.io/nebius/nebius-physical-ai/npa-genesis:"
        "cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )


def test_byo_workflow_images_honor_env(monkeypatch) -> None:
    monkeypatch.setenv("NPA_VLM_IMAGE", "registry.example/npa-vlm:custom")
    monkeypatch.setenv("NPA_WORKBENCH_IMAGE", "registry.example/npa-workbench:custom")

    assert default_vlm_image() == "registry.example/npa-vlm:custom"
    assert default_workbench_image() == "registry.example/npa-workbench:custom"


def test_development_images_use_public_package_and_full_sha() -> None:
    sha = "a" * 40
    assert development_tag(sha) == f"dev-{sha}"
    assert development_image_for_tool("genesis", git_sha=sha) == (
        f"ghcr.io/nebius/nebius-physical-ai/npa-genesis:dev-{sha}"
    )


def test_restricted_image_cannot_enter_official_development_channel() -> None:
    import pytest
    from npa.deploy import images

    original = images.RESTRICTED_PUBLICATION_TOOLS
    images.RESTRICTED_PUBLICATION_TOOLS = frozenset({"genesis"})
    try:
        with pytest.raises(ValueError, match="restricted/build-your-own"):
            development_image_for_tool("genesis", git_sha="b" * 40)
    finally:
        images.RESTRICTED_PUBLICATION_TOOLS = original
