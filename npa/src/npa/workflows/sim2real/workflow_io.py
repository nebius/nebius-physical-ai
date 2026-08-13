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


def declared_loop_uri(
    canonical_uri: str,
    outer_iteration: int,
    inner_iteration: int | None = None,
) -> str:
    """Map stable padded lineage to the integer URI rendered by loops."""

    outer_segment = f"/outer-{outer_iteration:02d}/"
    if outer_segment not in canonical_uri:
        raise ValueError(f"canonical loop URI lacks expected segment {outer_segment!r}")
    declared = canonical_uri.replace(
        outer_segment,
        f"/outer-{outer_iteration}/",
        1,
    )
    if inner_iteration is not None:
        inner_segment = f"/iter-{inner_iteration:02d}/"
        if inner_segment not in declared:
            raise ValueError(
                f"canonical loop URI lacks expected segment {inner_segment!r}"
            )
        declared = declared.replace(
            inner_segment,
            f"/iter-{inner_iteration}/",
            1,
        )
    return declared


def write_loop_output(
    canonical_uri: str,
    payload: dict[str, Any],
    directory: Path,
    outer_iteration: int,
    inner_iteration: int | None = None,
) -> str:
    """Publish canonical lineage plus the standard runtime checkpoint alias.

    Historical Sim2Real artifacts use ``outer-01/iter-01`` while standard
    ``npa.workflow`` loop substitutions use ``outer-1/iter-1``. Keep the
    established lineage and publish the same small JSON payload at the output
    URI that durable runtime reconciliation checks.
    """

    result = write_json(canonical_uri, payload, directory=directory)
    declared_uri = declared_loop_uri(
        canonical_uri,
        outer_iteration,
        inner_iteration,
    )
    if declared_uri != canonical_uri:
        write_json(declared_uri, payload, directory=directory / "declared")
    return result


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


def publish_component_lane_record(
    *,
    root_uri: str,
    stage: int,
    lane: str,
    evidence: str,
    artifacts: dict[str, Any],
    require_gpu: bool = True,
    execution_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish immutable per-lane evidence without impersonating the stage join.

    Parallel leaves own their execution proof. The downstream barrier consumer owns
    the single canonical ``components/stage_XX.json`` aggregation record, preserving
    the 14-record audit contract while making every lane independently attributable.
    """

    safe_lane = lane.strip()
    if not safe_lane or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789-_" for char in safe_lane
    ):
        raise ValueError(f"invalid component lane name: {lane!r}")
    provenance = dict(execution_provenance or image_provenance(require_gpu=require_gpu))
    if "@sha256:" not in str(provenance.get("image") or ""):
        raise ValueError(
            f"Stage {stage} lane {safe_lane} execution provenance lacks an image digest"
        )
    if require_gpu and not provenance.get("gpu_products"):
        raise ValueError(
            f"Stage {stage} lane {safe_lane} execution provenance lacks GPU products"
        )
    payload = {
        "schema": "npa.sim2real.component_lane_record.v1",
        "stage": stage,
        "lane": safe_lane,
        "tier": "WORKS",
        "evidence": evidence,
        "artifacts": {**artifacts, **provenance},
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    digest = hashlib.sha256(encoded).hexdigest()
    payload["content_sha256"] = digest
    work = Path("/tmp/npa-sim2real-component-lane") / f"stage-{stage:02d}-{safe_lane}"
    prefix = f"{root_uri.rstrip('/')}/components/lanes/stage_{stage:02d}/{safe_lane}"
    write_json(f"{prefix}/history/{digest}.json", payload, directory=work)
    write_json(f"{prefix}.json", payload, directory=work)
    return payload


def aggregate_parallel_provenance(
    provenances: list[dict[str, Any]], *, stage: int
) -> dict[str, Any]:
    """Build an honest join provenance record from every parallel leaf."""

    if not provenances:
        raise ValueError(f"Stage {stage} parallel aggregation has no lane provenance")
    images = {str(item.get("image") or "") for item in provenances}
    source_shas = {str(item.get("source_sha") or "") for item in provenances}
    image = next(iter(images)) if len(images) == 1 else ""
    repository = image.split("@", 1)[0]
    registry = repository.split("/", 1)[0]
    if (
        len(images) != 1
        or "/" not in repository
        or not ("." in registry or ":" in registry or registry == "localhost")
    ):
        raise ValueError(
            f"Stage {stage} parallel lanes do not share one qualified image"
        )
    source_sha_value = next(iter(source_shas)) if len(source_shas) == 1 else ""
    if (
        "@sha256:" not in image
        or len(source_sha_value) != 40
        or any(char not in "0123456789abcdef" for char in source_sha_value)
    ):
        raise ValueError(f"Stage {stage} parallel lane provenance is not immutable")
    gpu_products = sorted(
        {
            str(product)
            for item in provenances
            for product in (item.get("gpu_products") or [])
            if str(product)
        }
    )
    workflow_jobs = [str(item.get("workflow_job") or "") for item in provenances]
    if (
        not gpu_products
        or any(not job for job in workflow_jobs)
        or len(set(workflow_jobs)) != len(provenances)
    ):
        raise ValueError(f"Stage {stage} parallel lane provenance is incomplete")
    return {
        "image": image,
        "image_digest": image.split("@", 1)[1],
        "source_sha": source_sha_value,
        "workflow_jobs": workflow_jobs,
        "gpu_products": gpu_products,
        "lane_count": len(provenances),
        "execution_mode": "standard_npa_workflow_parallel_join",
    }


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
