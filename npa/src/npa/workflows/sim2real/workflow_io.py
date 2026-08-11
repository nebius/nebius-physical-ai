"""S3 and evidence primitives for compositional Sim2Real workflow stages."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.clients.storage import StorageClient


def storage() -> StorageClient:
    return StorageClient.from_environment()


def read_json(uri: str, *, directory: Path) -> dict[str, Any]:
    target = directory / Path(urlparse(uri).path).name
    storage().download_file(uri, str(target))
    payload = json.loads(target.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {uri}")
    return payload


def read_jsonl(uri: str, *, directory: Path) -> list[dict[str, Any]]:
    target = directory / Path(urlparse(uri).path).name
    storage().download_file(uri, str(target))
    rows = [
        json.loads(line) for line in target.read_text().splitlines() if line.strip()
    ]
    if not rows or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"expected non-empty JSONL objects at {uri}")
    return rows


def write_json(uri: str, payload: dict[str, Any], *, directory: Path) -> str:
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / Path(urlparse(uri).path).name
    target.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return storage().upload_file(str(target), uri)


def source_sha() -> str:
    actual = os.environ.get("NPA_IMAGE_SOURCE_SHA", "").strip().lower()
    expected = os.environ.get("NPA_SIM2REAL_SOURCE_SHA", "").strip().lower()
    for label, value in (("image", actual), ("workflow", expected)):
        if len(value) != 40 or any(char not in "0123456789abcdef" for char in value):
            raise RuntimeError(
                f"workflow stage lacks an exact 40-character {label} source SHA"
            )
    if actual != expected:
        raise RuntimeError(
            "workflow source SHA does not match the immutable task image attestation"
        )
    return actual


def image_provenance(*, require_gpu: bool) -> dict[str, Any]:
    image = os.environ.get("NPA_TASK_IMAGE", "").removeprefix("docker:").strip()
    if "@sha256:" not in image:
        raise RuntimeError("workflow stage lacks an immutable NPA_TASK_IMAGE digest")
    proof: dict[str, Any] = {
        "image": image,
        "image_digest": image.split("@", 1)[1],
        "source_sha": source_sha(),
        "workflow_job": os.environ.get("SKYPILOT_TASK_ID", "")
        or os.environ.get("SKYPILOT_CLUSTER_NAME", ""),
        "execution_mode": "standard_npa_workflow_skypilot",
    }
    if require_gpu:
        query = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,uuid",
                "--format=csv,noheader",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
        rows = [line.strip() for line in query.stdout.splitlines() if line.strip()]
        if not rows:
            raise RuntimeError("GPU stage has no nvidia-smi device evidence")
        products = [row.split(",", 1)[0].strip() for row in rows]
        proof.update({"gpu_products": products, "gpu_rows": rows})
    return proof


def publish_component_record(
    *,
    root_uri: str,
    stage: int,
    name: str,
    tier: str,
    evidence: str,
    artifacts: dict[str, Any],
    require_gpu: bool = False,
    next_action: str = "CONTINUE",
    execution_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if stage == 12:
        if tier != "SEAM":
            raise ValueError("Stage 12 must remain an explicit SEAM")
        provenance: dict[str, Any] = {"source_sha": source_sha()}
    else:
        if tier != "WORKS":
            raise ValueError(
                f"Stage {stage} must fail closed instead of publishing {tier}"
            )
        provenance = dict(
            execution_provenance or image_provenance(require_gpu=require_gpu)
        )
        if "@sha256:" not in str(provenance.get("image") or ""):
            raise ValueError(
                f"Stage {stage} execution provenance lacks an image digest"
            )
        if require_gpu and not provenance.get("gpu_products"):
            raise ValueError(f"Stage {stage} execution provenance lacks GPU products")
    payload = {
        "schema": "npa.sim2real.component_record.v1",
        "stage": stage,
        "name": name,
        "tier": tier,
        "evidence": evidence,
        "artifacts": {**artifacts, **provenance},
        "next_action": next_action,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    payload["content_sha256"] = digest
    work = Path("/tmp/npa-sim2real-component") / f"stage-{stage:02d}"
    history = (
        f"{root_uri.rstrip('/')}/components/history/stage_{stage:02d}/{digest}.json"
    )
    pointer = f"{root_uri.rstrip('/')}/components/stage_{stage:02d}.json"
    write_json(history, payload, directory=work)
    write_json(pointer, payload, directory=work)
    return payload


def list_prefix(uri: str) -> list[dict[str, Any]]:
    parsed = urlparse(uri)
    paginator = storage().s3.get_paginator("list_objects_v2")
    return [
        item
        for page in paginator.paginate(
            Bucket=parsed.netloc, Prefix=parsed.path.lstrip("/")
        )
        for item in page.get("Contents", []) or []
    ]
