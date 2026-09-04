"""Pinned upstream provenance for NPA's Physical AI Data Factory workflows.

NVIDIA publishes two repositories with different responsibilities: the
``physical-ai-data-factory`` repository is the ecosystem and agent-skill entry
point, while ``paidf-orchestration`` is an Airflow-on-Kubernetes implementation
for scaled IAA and EVG DAGs.  NPA executes neither OSMO nor Airflow.  This module
records that boundary as a durable run artifact so an NPA result cannot be
mistaken for an execution of either upstream orchestrator.
"""

from __future__ import annotations

import json
import re
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "npa.paidf.upstream.v1"
PHYSICAL_AI_DATA_FACTORY_REVISION = "e4c663cbbdcf159ad952751274c883c81d3ab4be"
PAIDF_ORCHESTRATION_REVISION = "f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9"
PAIDF_AUGMENTATION_REVISION = "bc5719362492a1e3b40bd7d33b43c46dd89efad5"
PAIDF_AUTO_LABELING_REVISION = "36dc1114dea00d9986df97325a664520993964de"
PAIDF_ANOMALYGEN_REVISION = "dbaf7d7d9003f048230f9026da5969e9e5931785"
PAIDF_SIMULATION_REVISION = "498751aceeea3dc3bac0d5fb043bf3553aec46a6"
PAIDF_CURATION_REVISION = "02079bbd272900c837ebdbd0bf44384dfdf1f25e"
QWEN_IMAGE_EDIT_REVISION = "6f3ccc0b56e431dc6a0c2b2039706d7d26f22cb9"
COSMOS3_SUPER_IMAGE2VIDEO_REVISION = "4f847566f3d3388fbf0ac07b99dd1a6432db9ecd"

_VARIANTS: dict[str, dict[str, Any]] = {
    "cosmos-transfer2.5": {
        "translation": "direct",
        "upstream_workflow": "Video Data Augmentation",
        "augmentation": "NVIDIA Cosmos Transfer 2.5",
        "execution": "workbench.cosmos2.transfer_execute",
        "evaluation": "NVIDIA Cosmos Evaluator plus NPA quality policy",
        "curation": "NVIDIA Cosmos Curator plus FiftyOne Brain",
    },
    "cosmos3-video2video": {
        "translation": "npa-specific-variant",
        "upstream_workflow": "Video Data Augmentation alternative",
        "augmentation": "NVIDIA Cosmos Framework 3 video2video",
        "execution": "workbench.cosmos3.generate_variants",
        "evaluation": "NVIDIA Cosmos Evaluator plus NPA quality policy",
        "curation": "NVIDIA Cosmos Curator plus FiftyOne Brain",
    },
    "defect-image-generation-day1-manual-roi": {
        "translation": "direct",
        "upstream_workflow": "Defect Image Generation Day 1 manual-ROI",
        "preparation": "canonical clean-image, ROI-mask, and defect-spec validation",
        "training": "NVIDIA PAIDF AnomalyGen 1.1.0 fresh fine-tuning with pinned use-case recipe",
        "generation": "NVIDIA PAIDF AnomalyGen 1.1.0 inference and native labeling",
        "execution": "workflow.paidf.dig_train -> workflow.paidf.dig_infer",
        "source_reference_images": [
            "nvcr.io/nvidia/paidf-anomalygen@sha256:e62a87d1dc58b6de8b8a352dc8ec2a2e3e400288d66b2b8b19b92d97e7a0bc09"
        ],
        "runtime_image_boundary": (
            "the vendor image fails SkyPilot bootstrap; paidf-anomalygen-sky is "
            "built from the pinned Apache-2.0 source on public CUDA bases, and "
            "the run must supply it by exact private-registry digest"
        ),
        "outputs": "generated images, masks, COCO labels, and provenance",
    },
    "image-attribute-augmentation": {
        "translation": "direct",
        "upstream_workflow": "image_attribute_augmentation_dag",
        "preparation": "input validation and deterministic attribute sampling",
        "generation": "NVIDIA PAIDF Augmentation 1.1.0 image-edit protocol",
        "models": {"Qwen/Qwen-Image-Edit-2511": QWEN_IMAGE_EDIT_REVISION},
        "labeling": "NVIDIA PAIDF event/person attribute search protocol",
        "execution": "workflow.paidf.run_iaa_augmentation",
        "reference_runtime_images": [
            "docker.io/vllm/vllm-omni@sha256:5d8c7e742c98858f257d82307e378391f0e7d77065e141c733cc4778042128ab",
            "nvcr.io/nvidia/paidf-event-and-person-attribute-search-service@sha256:0f581ff6d92efd391281e5787a8b1fda76556443ade47c1f5d59d4c345a01f6a",
        ],
        "outputs": "augmented image dataset, attributes, skips, and provenance",
    },
    "event-video-generation": {
        "translation": "direct",
        "upstream_workflow": "event_video_generation_dag",
        "preparation": "input validation and deterministic anomaly/environment sampling",
        "generation": "NVIDIA PAIDF Augmentation 1.1.0 Cosmos3 image2video protocol",
        "models": {
            "nvidia/Cosmos3-Super-Image2Video": COSMOS3_SUPER_IMAGE2VIDEO_REVISION
        },
        "labeling": (
            "NVIDIA PAIDF detection/tracking, captioning, visual-QA, and "
            "event/person attribute-search protocols"
        ),
        "execution": "workflow.paidf.run_evg_augmentation",
        "reference_runtime_images": [
            "docker.io/vllm/vllm-omni@sha256:970dee6658ea223f615b2438ce41e47f1d5322225482546e6e6bc5d8134f757c",
            "nvcr.io/nvidia/paidf-detection-and-tracking-rfdetr-service@sha256:6b35e63b95cab7cd772906bcb08be978de7526427f0d1925ab84439dd4a9561e",
            "nvcr.io/nvidia/paidf-captioning-service@sha256:17e1e3f53cc66342183f7d0b6eed76907993bb325a13db90c46d9a8cf664d804",
            "nvcr.io/nvidia/paidf-visual-qa-service@sha256:e681c8dee849c7ac9fc5b182f51e9efd0da460972b08850d40f00aa9d5e3c97c",
            "nvcr.io/nvidia/paidf-event-and-person-attribute-search-service@sha256:0f581ff6d92efd391281e5787a8b1fda76556443ade47c1f5d59d4c345a01f6a",
        ],
        "outputs": "anomaly video dataset, annotations, sidecars, and provenance",
    },
}

_COMPONENT_SOURCES = [
    {
        "repository": "https://github.com/NVIDIA/paidf-augmentation",
        "revision": PAIDF_AUGMENTATION_REVISION,
        "role": "published-augmentation-protocol",
        "licenses": ["Apache-2.0"],
        "workflow_variants": ["image-attribute-augmentation", "event-video-generation"],
    },
    {
        "repository": "https://github.com/NVIDIA/paidf-auto-labeling",
        "revision": PAIDF_AUTO_LABELING_REVISION,
        "role": "published-auto-labeling-protocols",
        "licenses": ["Apache-2.0"],
        "workflow_variants": ["image-attribute-augmentation", "event-video-generation"],
    },
    {
        "repository": "https://github.com/NVIDIA/paidf-anomalygen",
        "revision": PAIDF_ANOMALYGEN_REVISION,
        "role": "defect-generation-implementation",
        "licenses": ["Apache-2.0"],
        "workflow_variants": ["defect-image-generation-day1-manual-roi"],
    },
    {
        "repository": "https://github.com/NVIDIA/paidf-simulation",
        "revision": PAIDF_SIMULATION_REVISION,
        "role": "dig-usd-simulation-ecosystem-module-not-executed-by-manual-roi-translation",
        "licenses": ["Apache-2.0"],
        "workflow_variants": [],
    },
    {
        "repository": "https://github.com/NVIDIA/paidf-curation-and-retrieval",
        "revision": PAIDF_CURATION_REVISION,
        "role": "published-curation-and-retrieval-protocol",
        "licenses": ["Apache-2.0"],
        "workflow_variants": [],
    },
]


def upstream_contract(workflow_variant: str) -> dict[str, Any]:
    """Return the immutable public-source and execution-boundary contract."""

    variant = str(workflow_variant or "").strip()
    if variant not in _VARIANTS:
        raise ValueError(
            "unsupported PAIDF workflow variant; expected one of: "
            + ", ".join(sorted(_VARIANTS))
        )
    component_sources = []
    for source in _COMPONENT_SOURCES:
        record = dict(source)
        applicable = list(record.pop("workflow_variants"))
        record["executed_by_npa"] = variant in applicable
        record["applicable_workflow_variants"] = applicable
        component_sources.append(record)
    return {
        "schema": SCHEMA,
        "workflow_variant": variant,
        "sources": [
            {
                "repository": "https://github.com/NVIDIA/physical-ai-data-factory",
                "revision": PHYSICAL_AI_DATA_FACTORY_REVISION,
                "role": "ecosystem-entrypoint-and-video-data-augmentation-skill",
                "licenses": ["CC-BY-4.0", "Apache-2.0"],
                "upstream_orchestrator": "NVIDIA OSMO",
                "executed_by_npa": False,
            },
            {
                "repository": "https://github.com/NVIDIA/paidf-orchestration",
                "revision": PAIDF_ORCHESTRATION_REVISION,
                "role": "airflow-kubernetes-scaled-iaa-and-evg-reference",
                "licenses": ["Apache-2.0"],
                "upstream_orchestrator": "Apache Airflow on Kubernetes",
                "executed_by_npa": False,
            },
            *component_sources,
        ],
        "npa_integration": {
            "api_version": "npa.workflow/v0.0.1",
            "orchestrator": "SkyPilot",
            "data_handoff": "S3-compatible object storage",
            "component_policy": "real registered workbench commands; no manifest stubs",
            "runtime_fetch_boundary": (
                "gated model weights are fetched at runtime under the operator identity"
            ),
            "redistribution_boundary": (
                "no PAIDF orchestration runtime, gated model weights, or input data "
                "is embedded in NPA workflow artifacts"
            ),
            "components": dict(_VARIANTS[variant]),
        },
    }


def write_upstream_contract(
    workflow_variant: str,
    output_uri: str,
    runtime_image: str = "",
    *,
    generation_model: str = "",
    generation_revision: str = "",
) -> dict[str, Any]:
    """Write one truthful PAIDF upstream contract to a local path or S3 URI."""

    payload = upstream_contract(workflow_variant)
    if generation_model or generation_revision:
        if not generation_model.strip() or not re.fullmatch(
            r"[0-9a-f]{40}", generation_revision
        ):
            raise ValueError("generation model requires an exact immutable revision")
        components = payload["npa_integration"]["components"]
        components["reference_models"] = components.get("models", {})
        components["models"] = {generation_model: generation_revision}
    if "reference_runtime_images" in payload["npa_integration"]["components"]:
        payload["npa_integration"]["runtime_image_evidence"] = (
            "reference_runtime_images are reviewed upstream image defaults; "
            "executed stage reports record the actual renderer-selected runtime_image, "
            "including operator overrides"
        )
    if runtime_image:
        if not re.fullmatch(r".+@sha256:[0-9a-f]{64}", runtime_image):
            raise ValueError("runtime_image must be immutable by sha256 digest")
        payload["npa_integration"]["resolved_runtime_image"] = runtime_image
    encoded = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if output_uri.startswith("s3://"):
        from npa.clients.storage import StorageClient

        with tempfile.TemporaryDirectory(prefix="npa-paidf-upstream-") as tmp:
            local = Path(tmp) / "upstream.json"
            local.write_text(encoded, encoding="utf-8")
            written_uri = StorageClient.from_environment().upload_file(
                str(local), output_uri
            )
    else:
        destination = Path(output_uri)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(encoded, encoding="utf-8")
        written_uri = str(destination)
    print(
        json.dumps(
            {
                "stage": "record-paidf-upstream",
                "status": "completed",
                "workflow_variant": workflow_variant,
            }
        )
    )
    return {**payload, "written_uri": written_uri}


__all__ = [
    "PAIDF_ORCHESTRATION_REVISION",
    "PAIDF_ANOMALYGEN_REVISION",
    "PAIDF_AUGMENTATION_REVISION",
    "PAIDF_AUTO_LABELING_REVISION",
    "PAIDF_CURATION_REVISION",
    "PAIDF_SIMULATION_REVISION",
    "PHYSICAL_AI_DATA_FACTORY_REVISION",
    "COSMOS3_SUPER_IMAGE2VIDEO_REVISION",
    "QWEN_IMAGE_EDIT_REVISION",
    "SCHEMA",
    "upstream_contract",
    "write_upstream_contract",
]
