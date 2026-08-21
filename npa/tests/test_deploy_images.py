from __future__ import annotations

from pathlib import Path

from npa.deploy.images import (
    DEFAULT_CONTAINER_REGISTRY,
    DEFAULT_SOURCE_CONTAINER_REGISTRY,
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


def test_default_registry_is_the_anonymous_public_mirror() -> None:
    assert DEFAULT_CONTAINER_REGISTRY == "ghcr.io/nebius/nebius-physical-ai"


def test_maintainer_source_registry_remains_separate() -> None:
    assert DEFAULT_SOURCE_CONTAINER_REGISTRY.startswith(
        "cr.eu-north1.nebius.cloud/"
    )


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
        container_image_for_tool("cosmos3")
        == "ghcr.io/nebius/nebius-physical-ai/"
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


def test_packaged_supported_tool_versions_match_pyproject() -> None:
    try:
        import tomllib
    except ModuleNotFoundError:
        import tomli as tomllib

    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    data = tomllib.loads(pyproject.read_text(encoding="utf-8"))

    assert SUPPORTED_TOOL_VERSIONS == data["tool"]["npa"]["supported-tools"]


def test_paidf_worker_defaults_select_bootstrap_capable_releases() -> None:
    for tool in (
        "cosmos-curate",
        "cosmos-evaluator",
        "fiftyone",
        "rerun-viewer",
    ):
        assert "skypilot-v1" in SUPPORTED_TOOL_VERSIONS[tool]


def test_byo_workflow_images_have_pushed_defaults(monkeypatch) -> None:
    monkeypatch.delenv("NPA_VLM_IMAGE", raising=False)
    monkeypatch.delenv("NPA_WORKBENCH_IMAGE", raising=False)
    monkeypatch.delenv("NPA_REGISTRY", raising=False)

    assert (
        default_vlm_image()
        == "ghcr.io/nebius/nebius-physical-ai/"
        "npa-cosmos:cu128-torch27-sm100-1.0.9-20260803T002017Z"
    )
    assert (
        default_workbench_image()
        == "ghcr.io/nebius/nebius-physical-ai/npa-genesis:"
        "cuda13-b300-0.4.6-sm80-sm90-sm100-sm103-sm120-20260803T034152Z"
    )


def test_byo_workflow_images_honor_env(monkeypatch) -> None:
    monkeypatch.setenv("NPA_VLM_IMAGE", "registry.example/npa-vlm:custom")
    monkeypatch.setenv("NPA_WORKBENCH_IMAGE", "registry.example/npa-workbench:custom")

    assert default_vlm_image() == "registry.example/npa-vlm:custom"
    assert default_workbench_image() == "registry.example/npa-workbench:custom"
