from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from npa.workflows.paidf_upstream import (
    PAIDF_ORCHESTRATION_REVISION,
    PHYSICAL_AI_DATA_FACTORY_REVISION,
    SCHEMA,
    upstream_contract,
    write_upstream_contract,
)


def test_upstream_contract_distinguishes_sources_and_orchestrators() -> None:
    payload = upstream_contract("cosmos-transfer2.5")

    assert payload["schema"] == SCHEMA
    sources = {item["repository"]: item for item in payload["sources"]}
    ecosystem = sources["https://github.com/NVIDIA/physical-ai-data-factory"]
    scaler = sources["https://github.com/NVIDIA/paidf-orchestration"]
    assert ecosystem["revision"] == PHYSICAL_AI_DATA_FACTORY_REVISION
    assert ecosystem["licenses"] == ["CC-BY-4.0", "Apache-2.0"]
    assert ecosystem["upstream_orchestrator"] == "NVIDIA OSMO"
    assert scaler["revision"] == PAIDF_ORCHESTRATION_REVISION
    assert scaler["licenses"] == ["Apache-2.0"]
    assert scaler["role"] == "airflow-kubernetes-scaled-iaa-and-evg-reference"
    assert ecosystem["executed_by_npa"] is False
    assert scaler["executed_by_npa"] is False
    assert payload["npa_integration"]["orchestrator"] == "SkyPilot"
    assert payload["npa_integration"]["components"]["execution"] == (
        "workbench.cosmos2.transfer_execute"
    )


def test_cosmos3_contract_names_real_framework_command() -> None:
    payload = upstream_contract("cosmos3-video2video")
    assert payload["npa_integration"]["components"]["execution"] == (
        "workbench.cosmos3.generate_variants"
    )
    assert (
        payload["npa_integration"]["components"]["translation"]
        == "npa-specific-variant"
    )


@pytest.mark.parametrize(
    ("variant", "workflow"),
    [
        (
            "defect-image-generation-day1-manual-roi",
            "Defect Image Generation Day 1 manual-ROI",
        ),
        ("image-attribute-augmentation", "image_attribute_augmentation_dag"),
        ("event-video-generation", "event_video_generation_dag"),
    ],
)
def test_new_direct_translation_contracts(variant: str, workflow: str) -> None:
    payload = upstream_contract(variant)
    components = payload["npa_integration"]["components"]
    assert components["translation"] == "direct"
    assert components["upstream_workflow"] == workflow
    pinned_images = components.get("runtime_images") or components.get(
        "source_reference_images"
    )
    assert pinned_images
    assert all("@sha256:" in image for image in pinned_images)


def test_service_model_revisions_are_recorded_in_direct_translation_provenance() -> (
    None
):
    iaa = upstream_contract("image-attribute-augmentation")
    evg = upstream_contract("event-video-generation")
    assert iaa["npa_integration"]["components"]["models"] == {
        "Qwen/Qwen-Image-Edit-2511": "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
    }
    assert evg["npa_integration"]["components"]["models"] == {
        "nvidia/Cosmos3-Super-Image2Video": "4f847566f3d3388fbf0ac07b99dd1a6432db9ecd"
    }


@pytest.mark.parametrize(
    ("variant", "executed_repositories"),
    [
        (
            "defect-image-generation-day1-manual-roi",
            {"https://github.com/NVIDIA/paidf-anomalygen"},
        ),
        (
            "image-attribute-augmentation",
            {
                "https://github.com/NVIDIA/paidf-augmentation",
                "https://github.com/NVIDIA/paidf-auto-labeling",
            },
        ),
        (
            "event-video-generation",
            {
                "https://github.com/NVIDIA/paidf-augmentation",
                "https://github.com/NVIDIA/paidf-auto-labeling",
            },
        ),
    ],
)
def test_component_execution_flags_are_variant_specific(
    variant: str, executed_repositories: set[str]
) -> None:
    payload = upstream_contract(variant)
    actual = {
        str(source["repository"])
        for source in payload["sources"]
        if source.get("executed_by_npa") is True
    }
    assert actual == executed_repositories


def test_write_upstream_contract_local(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    target = tmp_path / "reports" / "upstream.json"
    result = write_upstream_contract("cosmos-transfer2.5", str(target))

    written = json.loads(target.read_text(encoding="utf-8"))
    assert written == {
        key: value for key, value in result.items() if key != "written_uri"
    }
    assert json.loads(capsys.readouterr().out)["status"] == "completed"


def test_write_upstream_contract_records_resolved_private_runtime_image(
    tmp_path: Path,
) -> None:
    image = "registry.invalid/npa-paidf-anomalygen-sky@sha256:" + "a" * 64
    result = write_upstream_contract(
        "defect-image-generation-day1-manual-roi",
        str(tmp_path / "upstream.json"),
        image,
    )
    assert result["npa_integration"]["resolved_runtime_image"] == image


def test_unknown_variant_fails_closed() -> None:
    with pytest.raises(ValueError, match="unsupported PAIDF workflow variant"):
        upstream_contract("airflow")


@pytest.mark.parametrize(
    ("filename", "variant", "successor"),
    [
        ("physical-ai-data-factory.yaml", "cosmos-transfer2.5", "generate-configs"),
        ("paidf-cosmos3.yaml", "cosmos3-video2video", "prepare-input"),
        (
            "paidf-defect-image-generation.yaml",
            "defect-image-generation-day1-manual-roi",
            "prepare-base-checkpoints",
        ),
        (
            "paidf-image-attribute-augmentation.yaml",
            "image-attribute-augmentation",
            "prepare-input",
        ),
        (
            "paidf-event-video-generation.yaml",
            "event-video-generation",
            "prepare-input",
        ),
    ],
)
def test_shipped_workflows_record_upstream_before_processing(
    filename: str, variant: str, successor: str
) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (
            repo_root / "npa" / "workflows" / "workbench" / "npa-workflows" / filename
        ).read_text(encoding="utf-8")
    )

    assert workflow["initial"] == "record-upstream"
    state = workflow["states"]["record-upstream"]
    expected_tail = [
        variant,
        "{{config.upstream_contract_uri}}",
    ]
    if filename == "paidf-defect-image-generation.yaml":
        expected_tail.append("{{config.anomalygen_image}}")
    assert state["run"]["argv"][-len(expected_tail) :] == expected_tail
    assert state["outputs"] == [
        {
            "uri": "{{config.upstream_contract_uri}}",
            "schema": SCHEMA,
        }
    ]
    assert state["next"] == successor
    assert workflow["states"][successor]["needs"] == ["record-upstream"]


def test_native_iaa_preserves_postprocess_and_attribute_search_boundaries() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (
            repo_root
            / "npa/workflows/workbench/npa-workflows/paidf-image-attribute-augmentation.yaml"
        ).read_text(encoding="utf-8")
    )

    states = workflow["states"]
    assert states["validate-outputs"]["next"] == "cosmos-post-processing"
    assert (
        states["cosmos-post-processing"]["toolRef"] == "workflow.paidf.postprocess_iaa"
    )
    assert (
        states["cosmos-post-processing"]["next"] == "event-and-person-attribute-search"
    )
    assert states["event-and-person-attribute-search"]["toolRef"] == (
        "workflow.paidf.run_auto_label"
    )


def test_native_evg_preserves_published_sequential_labeling_chain() -> None:
    repo_root = Path(__file__).resolve().parents[3]
    workflow = yaml.safe_load(
        (
            repo_root
            / "npa/workflows/workbench/npa-workflows/paidf-event-video-generation.yaml"
        ).read_text(encoding="utf-8")
    )

    states = workflow["states"]
    chain = [
        "detection-and-tracking",
        "captioning",
        "anomaly-visual-qa",
        "person-attribute-visual-qa",
        "person-attribute-search",
        "generate-anomaly-dataset",
    ]
    for current, successor in zip(chain, chain[1:]):
        assert states[current]["next"] == successor
