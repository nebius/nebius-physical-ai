"""S3 and evidence primitives for compositional Sim2Real workflow stages."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from npa.clients.storage import StorageClient
from npa.workflows.sim2real.publication import (
    MutablePublicationTransaction,
    RemoteObjectSnapshot,
    replace_mutable_file,
    upload_immutable_file,
)


def storage() -> StorageClient:
    return StorageClient.from_environment()


def parse_json_object(payload: str, *, source: str = "JSON payload") -> dict[str, Any]:
    """Parse one JSON object while rejecting ambiguous duplicate fields.

    Args:
        payload: UTF-8-decoded JSON text.
        source: Human-readable artifact name for validation errors.

    Returns:
        The decoded JSON object.

    Raises:
        ValueError: The text is invalid, non-object, or contains duplicate fields.
    """

    def unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{source} contains duplicate JSON field {key!r}")
            result[key] = value
        return result

    try:
        decoded = json.loads(payload, object_pairs_hook=unique_object)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{source} is not valid JSON: {exc.msg}") from exc
    if not isinstance(decoded, dict):
        raise ValueError(f"{source} must contain a JSON object")
    return decoded


def read_json(uri: str, *, directory: Path) -> dict[str, Any]:
    target = directory / Path(urlparse(uri).path).name
    storage().download_file(uri, str(target))
    return parse_json_object(target.read_text(encoding="utf-8"), source=uri)


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


def _gpu_device_evidence_is_valid(products: object, rows: object) -> bool:
    if (
        not isinstance(products, list)
        or not products
        or not isinstance(rows, list)
        or len(rows) != len(products)
    ):
        return False
    seen_uuids: set[str] = set()
    for product, row in zip(products, rows, strict=True):
        if (
            not isinstance(product, str)
            or not product
            or product != product.strip()
            or not isinstance(row, str)
            or row != row.strip()
        ):
            return False
        fields = row.split(",")
        if len(fields) != 2:
            return False
        row_product, uuid = (field.strip() for field in fields)
        if (
            row_product != product
            or not uuid.startswith("GPU-")
            or len(uuid) <= 4
            or any(character.isspace() or character == "," for character in uuid)
            or uuid in seen_uuids
        ):
            return False
        seen_uuids.add(uuid)
    return True


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
        if not _gpu_device_evidence_is_valid(products, rows):
            raise RuntimeError(
                "GPU stage returned malformed nvidia-smi device evidence"
            )
        proof.update({"gpu_products": products, "gpu_rows": rows})
    return proof


def _component_provenance(
    stage: int,
    tier: str,
    require_gpu: bool,
    execution_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    if stage == 12:
        if tier != "SEAM":
            raise ValueError("Stage 12 must remain an explicit SEAM")
        return {"source_sha": source_sha()}
    if tier != "WORKS":
        raise ValueError(f"Stage {stage} must fail closed instead of publishing {tier}")
    provenance = dict(execution_provenance or image_provenance(require_gpu=require_gpu))
    if require_gpu and not provenance.get("gpu_products"):
        raise ValueError(f"Stage {stage} execution provenance lacks GPU products")
    if not _record_provenance_is_valid(provenance, stage=stage):
        raise ValueError(f"Stage {stage} execution provenance is invalid")
    return provenance


def _component_record_payload(
    stage: int,
    name: str,
    tier: str,
    evidence: str,
    artifacts: dict[str, Any],
    require_gpu: bool = False,
    next_action: str = "CONTINUE",
    execution_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    provenance = _component_provenance(stage, tier, require_gpu, execution_provenance)
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
    payload["content_sha256"] = hashlib.sha256(encoded).hexdigest()
    return payload


def build_component_record(
    *,
    stage: int,
    name: str,
    tier: str,
    evidence: str,
    artifacts: dict[str, Any],
    require_gpu: bool = False,
    next_action: str = "CONTINUE",
    execution_provenance: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one content-addressed ComponentRecord without publishing it.
    Args:
        stage: Canonical one-based workflow stage number.
        name: Canonical stage component name.
        tier: ``WORKS`` except for the explicit Stage 12 ``SEAM``.
        evidence: Human-readable factual completion evidence.
        artifacts: Stage outputs and lineage fields.
        require_gpu: Whether execution provenance must name GPU products.
        next_action: Durable orchestration action after this stage.
        execution_provenance: Optional already-verified task provenance.
    Returns:
        A complete ComponentRecord including its content SHA-256.
    Raises:
        ValueError: If tier or execution provenance is invalid.
        RuntimeError: If ambient image/source provenance is unavailable.
    """

    return _component_record_payload(
        stage,
        name,
        tier,
        evidence,
        artifacts,
        require_gpu,
        next_action,
        execution_provenance,
    )


def _record_shape_is_valid(
    record: dict[str, Any],
    *,
    expected_stage: int,
    expected_name: str,
    expected_tier: str,
    required_artifacts: tuple[str, ...],
) -> bool:
    artifacts = record.get("artifacts")
    material = {key: value for key, value in record.items() if key != "content_sha256"}
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return bool(
        record.get("schema") == "npa.sim2real.component_record.v1"
        and type(record.get("stage")) is int
        and record["stage"] == expected_stage
        and record.get("name") == expected_name
        and record.get("tier") == expected_tier
        and isinstance(record.get("evidence"), str)
        and record["evidence"].strip()
        and isinstance(record.get("next_action"), str)
        and record["next_action"]
        in {
            "CONTINUE",
            "LOOP_OR_COMPLETE_BUDGET",
            "EXTERNAL_OPERATOR_VALIDATION",
        }
        and isinstance(artifacts, dict)
        and all(
            key in artifacts and artifacts[key] not in (None, "", {}, [])
            for key in required_artifacts
        )
        and record.get("content_sha256") == digest
    )


def _record_provenance_is_valid(
    artifacts: dict[str, Any],
    *,
    stage: int,
) -> bool:
    source = artifacts.get("source_sha")
    source_valid = (
        isinstance(source, str)
        and len(source) == 40
        and all(char in "0123456789abcdef" for char in source)
    )
    if not source_valid:
        return False
    if stage == 12:
        return not any(
            key in artifacts
            for key in ("image", "image_digest", "execution_mode", "workflow_job")
        )
    image = artifacts.get("image")
    image_digest = artifacts.get("image_digest")
    image_valid = bool(
        isinstance(image, str)
        and "@sha256:" in image
        and isinstance(image_digest, str)
        and len(image_digest) == 71
        and image_digest.startswith("sha256:")
        and all(char in "0123456789abcdef" for char in image_digest[7:])
        and image_digest == image.split("@", 1)[1]
    )
    if not image_valid:
        return False
    if artifacts.get("execution_mode") == "standard_npa_workflow_parallel_join":
        jobs = artifacts.get("workflow_jobs")
        return bool(
            stage == 4
            and isinstance(jobs, list)
            and jobs
            and all(isinstance(job, str) and job.strip() for job in jobs)
            and len(set(jobs)) == len(jobs)
            and artifacts.get("lane_count") == len(jobs)
            and isinstance(artifacts.get("gpu_products"), list)
            and artifacts["gpu_products"]
        )
    gpu_valid = _gpu_device_evidence_is_valid(
        artifacts.get("gpu_products"),
        artifacts.get("gpu_rows"),
    )
    return bool(
        artifacts.get("execution_mode") == "standard_npa_workflow_skypilot"
        and isinstance(artifacts.get("workflow_job"), str)
        and artifacts["workflow_job"].strip()
        and (stage != 4 or gpu_valid)
    )


def _validate_component_record(
    record: dict[str, Any],
    expected_stage: int,
    expected_name: str,
    expected_tier: str,
    required_artifacts: tuple[str, ...],
    expected_source_sha: str | None,
) -> None:
    if not isinstance(record, dict):
        raise ValueError(f"Stage {expected_stage} ComponentRecord must be an object")
    if not _record_shape_is_valid(
        record,
        expected_stage=expected_stage,
        expected_name=expected_name,
        expected_tier=expected_tier,
        required_artifacts=required_artifacts,
    ):
        raise ValueError(f"Stage {expected_stage} ComponentRecord is invalid")
    if not _record_provenance_is_valid(record["artifacts"], stage=expected_stage):
        raise ValueError(f"Stage {expected_stage} execution provenance is invalid")
    if (
        expected_source_sha is not None
        and record["artifacts"]["source_sha"] != expected_source_sha
    ):
        raise ValueError(f"Stage {expected_stage} source revision is stale")


def validate_component_record(
    record: dict[str, Any],
    *,
    expected_stage: int,
    expected_name: str,
    expected_tier: str,
    required_artifacts: tuple[str, ...],
    expected_source_sha: str | None = None,
) -> None:
    """Reject incomplete or tampered canonical ComponentRecords.

    Args:
        record: Decoded ComponentRecord read from durable storage.
        expected_stage: Stage number selected by the pointer path.
        expected_name: Canonical component name for that stage.
        expected_tier: Required ``WORKS`` or ``SEAM`` tier.
        required_artifacts: Artifact keys required from the stage producer.
        expected_source_sha: Optional exact workflow source revision.
    Returns:
        None.
    Raises:
        ValueError: If schema, provenance, artifacts, or digest are invalid.
    """
    _validate_component_record(
        record,
        expected_stage,
        expected_name,
        expected_tier,
        required_artifacts,
        expected_source_sha,
    )


def component_record_history_uri(root_uri: str, stage: int, digest: str) -> str:
    """Return the immutable content-addressed URI for one ComponentRecord."""

    if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
        raise ValueError("ComponentRecord history digest must be lowercase SHA-256")
    return f"{root_uri.rstrip('/')}/components/history/stage_{stage:02d}/{digest}.json"


def _publish_record(root_uri: str, record: dict[str, Any], stage: int) -> None:
    digest = record["content_sha256"]
    work = Path("/tmp/npa-sim2real-component") / f"stage-{stage:02d}"
    history = component_record_history_uri(root_uri, stage, digest)
    pointer = f"{root_uri.rstrip('/')}/components/stage_{stage:02d}.json"
    write_json(history, record, directory=work)
    write_json(pointer, record, directory=work)


def _build_and_publish_record(
    root_uri: str,
    stage: int,
    name: str,
    tier: str,
    evidence: str,
    artifacts: dict[str, Any],
    require_gpu: bool,
    next_action: str,
    execution_provenance: dict[str, Any] | None,
) -> dict[str, Any]:
    payload = build_component_record(
        stage=stage,
        name=name,
        tier=tier,
        evidence=evidence,
        artifacts=artifacts,
        require_gpu=require_gpu,
        next_action=next_action,
        execution_provenance=execution_provenance,
    )
    _publish_record(root_uri, payload, stage)
    return payload


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
    """Publish one immutable ComponentRecord history object and live pointer.
    Args:
        root_uri: Run-owned artifact root.
        stage: Canonical one-based workflow stage number.
        name: Canonical stage component name.
        tier: ``WORKS`` except for Stage 12 ``SEAM``.
        evidence: Human-readable factual completion evidence.
        artifacts: Stage outputs and lineage fields.
        require_gpu: Whether provenance must name GPU products.
        next_action: Durable orchestration action after this stage.
        execution_provenance: Optional already-verified task provenance.
    Returns:
        The published content-addressed ComponentRecord.
    Raises:
        ValueError or RuntimeError: If validation or provenance fails.
    """
    return _build_and_publish_record(
        root_uri,
        stage,
        name,
        tier,
        evidence,
        artifacts,
        require_gpu,
        next_action,
        execution_provenance,
    )


def publish_built_component_record(
    *,
    root_uri: str,
    record: dict[str, Any],
    expected_stage: int,
    expected_name: str,
    expected_tier: str,
    required_artifacts: tuple[str, ...],
) -> dict[str, Any]:
    """Publish an already-built ComponentRecord without changing its identity.

    Args:
        root_uri: Run-owned artifact root.
        record: Stable record previously embedded in final evidence.
        expected_stage: Stage number owned by the caller.
        expected_name: Canonical component name owned by the caller.
        expected_tier: Canonical publication tier owned by the caller.
        required_artifacts: Artifact keys required from this producer.

    Returns:
        The exact published record.

    Raises:
        ValueError: If the record was changed or is incomplete.
    """

    publish_built_component_history(
        root_uri=root_uri,
        record=record,
        expected_stage=expected_stage,
        expected_name=expected_name,
        expected_tier=expected_tier,
        required_artifacts=required_artifacts,
    )
    publish_built_component_pointer(
        root_uri=root_uri,
        record=record,
        expected_stage=expected_stage,
        expected_name=expected_name,
        expected_tier=expected_tier,
        required_artifacts=required_artifacts,
    )
    return record


def _validate_built_component_record(
    record: dict[str, Any],
    *,
    expected_stage: int,
    expected_name: str,
    expected_tier: str,
    required_artifacts: tuple[str, ...],
) -> None:
    validate_component_record(
        record,
        expected_stage=expected_stage,
        expected_name=expected_name,
        expected_tier=expected_tier,
        required_artifacts=required_artifacts,
    )


def publish_built_component_history(
    *,
    root_uri: str,
    record: dict[str, Any],
    expected_stage: int,
    expected_name: str,
    expected_tier: str,
    required_artifacts: tuple[str, ...],
    client: Any | None = None,
) -> dict[str, Any]:
    """Publish only the immutable history object for an already-built record."""

    _validate_built_component_record(
        record,
        expected_stage=expected_stage,
        expected_name=expected_name,
        expected_tier=expected_tier,
        required_artifacts=required_artifacts,
    )
    uri = component_record_history_uri(
        root_uri, expected_stage, record["content_sha256"]
    )
    with tempfile.TemporaryDirectory(
        prefix=f"npa-sim2real-component-{expected_stage:02d}-"
    ) as directory:
        if client is None:
            write_json(uri, record, directory=Path(directory) / "history")
        else:
            path = Path(directory) / "history" / Path(urlparse(uri).path).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            upload_immutable_file(client, path, uri)
    return record


def publish_built_component_pointer(
    *,
    root_uri: str,
    record: dict[str, Any],
    expected_stage: int,
    expected_name: str,
    expected_tier: str,
    required_artifacts: tuple[str, ...],
    client: Any | None = None,
    snapshot: RemoteObjectSnapshot | None = None,
    transaction: MutablePublicationTransaction | None = None,
    immutable_uri: str = "",
) -> dict[str, Any]:
    """Publish only the mutable canonical pointer for an already-built record."""

    _validate_built_component_record(
        record,
        expected_stage=expected_stage,
        expected_name=expected_name,
        expected_tier=expected_tier,
        required_artifacts=required_artifacts,
    )
    uri = f"{root_uri.rstrip('/')}/components/stage_{expected_stage:02d}.json"
    with tempfile.TemporaryDirectory(
        prefix=f"npa-sim2real-component-{expected_stage:02d}-"
    ) as directory:
        if transaction is not None:
            transaction.replace_bytes(
                (json.dumps(record, indent=2, sort_keys=True) + "\n").encode(),
                uri,
                snapshot,
                immutable_uri=immutable_uri,
            )
        elif client is None:
            write_json(uri, record, directory=Path(directory) / "pointer")
        else:
            path = Path(directory) / "pointer" / Path(urlparse(uri).path).name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps(record, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            replace_mutable_file(client, path, uri, snapshot)
    return record


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
    if not _record_provenance_is_valid(provenance, stage=stage):
        raise ValueError(
            f"Stage {stage} lane {safe_lane} execution provenance is invalid"
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
    prefix = f"{root_uri.rstrip('/')}/components/lanes/stage_{stage:02d}/{safe_lane}"
    with tempfile.TemporaryDirectory(
        prefix=f"npa-sim2real-component-lane-{stage:02d}-{safe_lane}-"
    ) as directory:
        work = Path(directory)
        write_json(f"{prefix}/history/{digest}.json", payload, directory=work)
        write_json(f"{prefix}.json", payload, directory=work)
    return payload


def validate_component_lane_record(
    record: dict[str, Any],
    *,
    expected_stage: int,
    expected_lane: str,
    required_artifacts: tuple[str, ...],
    expected_source_sha: str,
) -> None:
    """Validate one embedded parallel-lane record and its execution proof."""

    if not isinstance(record, dict):
        raise ValueError(f"Stage {expected_stage} lane record must be an object")
    artifacts = record.get("artifacts")
    material = {key: value for key, value in record.items() if key != "content_sha256"}
    digest = hashlib.sha256(
        json.dumps(material, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        record.get("schema") != "npa.sim2real.component_lane_record.v1"
        or record.get("stage") != expected_stage
        or record.get("lane") != expected_lane
        or record.get("tier") != "WORKS"
        or not isinstance(record.get("evidence"), str)
        or not record["evidence"].strip()
        or not isinstance(artifacts, dict)
        or any(
            key not in artifacts or artifacts[key] in (None, "", {}, [])
            for key in required_artifacts
        )
        or record.get("content_sha256") != digest
    ):
        raise ValueError(
            f"Stage {expected_stage} lane {expected_lane} record is invalid"
        )
    if artifacts.get("source_sha") != expected_source_sha:
        raise ValueError(
            f"Stage {expected_stage} lane {expected_lane} source authority is stale"
        )
    if not _record_provenance_is_valid(artifacts, stage=expected_stage):
        raise ValueError(
            f"Stage {expected_stage} lane {expected_lane} provenance is invalid"
        )


def aggregate_parallel_provenance(
    provenances: list[dict[str, Any]], *, stage: int
) -> dict[str, Any]:
    """Build an honest join provenance record from every parallel leaf."""

    if not provenances:
        raise ValueError(f"Stage {stage} parallel aggregation has no lane provenance")
    if any(
        not _gpu_device_evidence_is_valid(
            provenance.get("gpu_products"),
            provenance.get("gpu_rows"),
        )
        for provenance in provenances
    ):
        raise ValueError(
            f"Stage {stage} parallel lane GPU provenance is incomplete or inconsistent"
        )
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
