from __future__ import annotations

from pathlib import Path

from npa.deploy.images import (
    DEFAULT_CONTAINER_REGISTRY,
    SUPPORTED_TOOL_VERSIONS,
    container_image_for_tool,
    default_vlm_image,
    default_workbench_image,
    primary_container_registry,
    registry_from_env,
    registry_from_id,
)


def test_registry_from_id_expands_against_primary_region() -> None:
    assert registry_from_id("myregid123") == "cr.eu-north1.nebius.cloud/myregid123"
    # Surrounding whitespace is stripped.
    assert registry_from_id("  myregid123 ") == "cr.eu-north1.nebius.cloud/myregid123"


def test_registry_from_env_prefers_npa_registry(monkeypatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.example/team")
    monkeypatch.setenv("NPA_REGISTRY_ID", "myregid123")
    assert registry_from_env() == "registry.example/team"


def test_registry_from_env_falls_back_to_registry_id(monkeypatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.setenv("NPA_REGISTRY_ID", "myregid123")
    assert registry_from_env() == "cr.eu-north1.nebius.cloud/myregid123"


def test_registry_from_env_empty_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_ID", raising=False)
    assert registry_from_env() == ""


def test_primary_container_registry_honors_registry_id(monkeypatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.setenv("NPA_REGISTRY_ID", "myregid123")
    assert primary_container_registry() == "cr.eu-north1.nebius.cloud/myregid123"


def test_primary_container_registry_defaults_when_unset(monkeypatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_ID", raising=False)
    assert primary_container_registry() == DEFAULT_CONTAINER_REGISTRY


def test_default_registry_is_real_first_party_registry() -> None:
    assert DEFAULT_CONTAINER_REGISTRY == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw"


def test_non_sonic_workbench_images_resolve_from_supported_tools() -> None:
    assert (
        container_image_for_tool("lancedb")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.3"
    )
    assert container_image_for_tool("detection-training") == (
        "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/"
        "npa-detection-training:bdd100k-golden-eval-smoke-20260614T210000Z"
    )
    assert (
        container_image_for_tool("groot")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-groot:0.1.0"
    )
    assert (
        container_image_for_tool("cosmos2-transfer")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/"
        "npa-cosmos2-transfer:2.5.1-golden-eval-smoke-20260616T033000Z"
    )
    assert (
        container_image_for_tool("cosmos3-reason")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-cosmos3-reason:3.0.1-genuine-sm120"
    )
    assert (
        container_image_for_tool("envgen")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-envgen:0.1.2"
    )
    assert (
        container_image_for_tool("reference-policy")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-reference-policy:0.1.2"
    )
    assert (
        container_image_for_tool("loop-eval")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-loop-eval:0.1.3-genuine-sm120"
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
