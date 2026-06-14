from __future__ import annotations

from pathlib import Path

import pytest

from npa.deploy.images import (
    BACKUP_CONTAINER_REGISTRY,
    DEFAULT_CONTAINER_REGISTRY,
    SUPPORTED_TOOL_VERSIONS,
    backup_container_registry,
    container_image_candidates,
    container_image_for_tool,
    default_vlm_image,
    default_workbench_image,
)


def test_default_registry_is_real_first_party_registry() -> None:
    assert DEFAULT_CONTAINER_REGISTRY == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw"


def test_backup_registry_is_distinct_region() -> None:
    assert BACKUP_CONTAINER_REGISTRY.startswith("cr.us-central1.nebius.cloud/")
    assert BACKUP_CONTAINER_REGISTRY != DEFAULT_CONTAINER_REGISTRY


def test_backup_registry_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_BACKUP_REGISTRY", "registry.example/backup")
    assert backup_container_registry() == "registry.example/backup"


def test_container_image_candidates_include_primary_then_backup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("NPA_REGISTRY", raising=False)
    monkeypatch.delenv("NPA_REGISTRY_ID", raising=False)
    monkeypatch.delenv("NPA_BACKUP_REGISTRY", raising=False)
    candidates = container_image_candidates("lancedb")
    assert candidates[0] == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-lancedb:0.30.2"
    assert candidates[1] == f"{BACKUP_CONTAINER_REGISTRY}/npa-lancedb:0.30.2"


def test_container_image_candidates_dedup_when_backup_matches_primary(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("NPA_REGISTRY", "registry.example/same")
    monkeypatch.setenv("NPA_BACKUP_REGISTRY", "registry.example/same")
    candidates = container_image_candidates("lancedb")
    assert candidates == ["registry.example/same/npa-lancedb:0.30.2"]


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
    assert (
        container_image_for_tool("cosmos2-transfer")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-cosmos2-transfer:2.5.0"
    )
    assert (
        container_image_for_tool("cosmos3-reason")
        == "cr.eu-north1.nebius.cloud/e00cm0vc6t09m0z5gw/npa-cosmos3-reason:3.0.1-genuine-sm120"
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
