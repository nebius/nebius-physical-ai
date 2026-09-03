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
import tempfile
from pathlib import Path
from typing import Any


SCHEMA = "npa.paidf.upstream.v1"
PHYSICAL_AI_DATA_FACTORY_REVISION = "0c37222a7cd1b2a1fc36086a7a03d7978d891506"
PAIDF_ORCHESTRATION_REVISION = "f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9"

_VARIANTS: dict[str, dict[str, Any]] = {
    "cosmos-transfer2.5": {
        "augmentation": "NVIDIA Cosmos Transfer 2.5",
        "execution": "workbench.cosmos2.transfer_execute",
        "evaluation": "NVIDIA Cosmos Evaluator plus NPA quality policy",
        "curation": "NVIDIA Cosmos Curator plus FiftyOne Brain",
    },
    "cosmos3-video2video": {
        "augmentation": "NVIDIA Cosmos Framework 3 video2video",
        "execution": "workbench.cosmos3.generate_video_variants",
        "evaluation": "NVIDIA Cosmos Evaluator plus NPA quality policy",
        "curation": "NVIDIA Cosmos Curator plus FiftyOne Brain",
    },
}


def upstream_contract(workflow_variant: str) -> dict[str, Any]:
    """Return the immutable public-source and execution-boundary contract."""

    variant = str(workflow_variant or "").strip()
    if variant not in _VARIANTS:
        raise ValueError(
            "unsupported PAIDF workflow variant; expected one of: "
            + ", ".join(sorted(_VARIANTS))
        )
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


def write_upstream_contract(workflow_variant: str, output_uri: str) -> dict[str, Any]:
    """Write one truthful PAIDF upstream contract to a local path or S3 URI."""

    payload = upstream_contract(workflow_variant)
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
    "PHYSICAL_AI_DATA_FACTORY_REVISION",
    "SCHEMA",
    "upstream_contract",
    "write_upstream_contract",
]
