"""Native SkyPilot adapters for NVIDIA Physical AI Data Factory workflows.

The adapters in this module deliberately do not reimplement model behavior. They
prepare the published PAIDF protocol payloads, invoke the real upstream
executables in their vendor images (or from an exact runtime-fetched revision),
and validate the media/metadata handoff before publishing a stage result.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.request
from pathlib import Path, PurePosixPath
from typing import Any
from urllib.parse import urlparse

import yaml

from npa.clients.storage import StorageClient
from npa.workflows.paidf_upstream import (
    PAIDF_AUGMENTATION_REVISION,
    PAIDF_AUTO_LABELING_REVISION,
    PAIDF_ORCHESTRATION_REVISION,
    PHYSICAL_AI_DATA_FACTORY_REVISION,
    RFDETR_BASE_SHA256,
    RFDETR_BASE_URL,
    validate_direct_generation_model,
    validate_token_factory_endpoint,
)


SCHEMA_PREFIX = "npa.paidf.native"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
IAA_LABEL_ARTIFACTS = (
    "sidecars/person_attribute_search/bundle_attributes.json",
    "sidecars/person_attribute_search/bundle_queries.json",
)
DIG_RUNTIME_CACHE_PROBES = {
    "nvidia/Cosmos-Guardrail1": "blocklist",
    "Qwen/Qwen3Guard-Gen-0.6B": "model.safetensors",
    "Qwen/Qwen3-VL-8B-Instruct": "tokenizer.json",
    "nvidia/Cosmos3-Edge": "processor_config.json",
    "Wan-AI/Wan2.2-TI2V-5B": "Wan2.2_VAE.pth",
}
DIG_PRETRAINED_CONTENT_MANIFEST = "npa-pretrained-content.json"
_SERVICE_WORKFLOWS = {"image-edit": "iaa", "image2video": "evg"}
_PAIDF_EXECUTOR_PATH = "modules/generation/executors/base.py"
_PAIDF_EXECUTOR_SHA256 = (
    "650283999eb6ac6f0b3bb943dccf73a1fa507b5790be9d282038893a91f55b46"
)
_PAIDF_EXECUTOR_PATCHED_SHA256 = (
    "0a28db07ba1fc9703659e5e94d8a867be9ae05d8276c691778289a3506c7fa59"
)
_EVG_LABEL_STAGES = (
    "detection", "captioning", "visual-qa-anomaly", "visual-qa-person",
    "person-attribute-search",
)
_EVG_VQA_MEDIA_SOURCE_PATH = "packages/tasks/visual_qa/src/visual_qa/media.py"
_EVG_VQA_MEDIA_SOURCE_SHA256 = (
    "85f09956d93a39a8e0d92be6774efaf2baaadc02072fea2e682939b4f0ea3c75"
)
_EVG_VQA_HOSTED_MAX_IMAGES = 10


class PaidfNativeError(RuntimeError):
    """A PAIDF protocol or artifact contract failed closed."""


def _evg_vqa_request_media_contract(stage: str) -> dict[str, Any]:
    """Use the vendor's existing sampler within this hosted endpoint's capacity."""

    controls = {
        "visual-qa-anomaly": ("--max-frames", 16, "_sample_frame_ids", "candidate-frame-ids"),
        "visual-qa-person": ("--max-crops-per-track", 12, "_sample_even", "track-crop-list"),
    }
    if stage not in controls:
        raise PaidfNativeError("request media contract requires an EVG VQA stage")
    control, upstream_value, function, scope = controls[stage]
    return {
        "schema": f"{SCHEMA_PREFIX}.evg-vqa-request-media.v1",
        "upstream_repository": "https://github.com/NVIDIA/paidf-auto-labeling",
        "upstream_revision": PAIDF_AUTO_LABELING_REVISION,
        "source_path": _EVG_VQA_MEDIA_SOURCE_PATH,
        "source_sha256": _EVG_VQA_MEDIA_SOURCE_SHA256,
        "hosted_provider": "nebius-token-factory",
        "hosted_max_images": _EVG_VQA_HOSTED_MAX_IMAGES,
        "stage": stage,
        "control": control,
        "value": _EVG_VQA_HOSTED_MAX_IMAGES,
        "upstream_value": upstream_value,
        "sampling_strategy": "endpoint-inclusive-even-subsampling",
        "sampling_function": function,
        "sampling_scope": scope,
    }


def _require_evg_vqa_request_media_contract(payload: dict[str, Any]) -> None:
    expected = _evg_vqa_request_media_contract(payload.get("stage", ""))
    actual = payload.get("request_media_contract")
    if (
        actual != expected
        or any(type(actual.get(key)) is not int for key in ("value", "upstream_value", "hosted_max_images"))
    ):
        raise PaidfNativeError("EVG VQA request media contract is missing or changed")


def _bind_detection_checkpoint() -> None:
    """Keep custom cache paths subject to the published checkpoint digest."""

    expected = os.environ.get("RFDETR_MODEL_SHA256", "").strip().lower()
    if expected and expected != RFDETR_BASE_SHA256:
        raise PaidfNativeError(
            "direct EVG detection requires the published RF-DETR checkpoint SHA-256"
        )
    # Upstream otherwise permits a custom RFDETR_MODEL_PATH basename to skip
    # hashing. Bind its existing streaming verifier even for those cache paths.
    os.environ["RFDETR_MODEL_SHA256"] = RFDETR_BASE_SHA256


def _require_token_factory_endpoint(endpoint: str, role: str) -> None:
    """Bind the injected Token Factory credential to its approved HTTPS origin."""

    try:
        validate_token_factory_endpoint(endpoint, role)
    except ValueError as exc:
        raise PaidfNativeError(str(exc)) from exc


def _require_direct_generation_model(
    workflow: str, generation_model: str, generation_revision: str | None = None
) -> None:
    """Keep direct IAA/EVG translations on their reviewed model artifacts."""

    try:
        validate_direct_generation_model(
            workflow, generation_model, generation_revision
        )
    except ValueError as exc:
        raise PaidfNativeError(str(exc)) from exc


def _validate_configured_token_factory_endpoints(config: Any) -> None:
    """Reject a rendered PAIDF config that could route injected keys elsewhere."""

    if not isinstance(config, dict):
        raise PaidfNativeError("rendered PAIDF config is not an object")
    endpoints = config.get("endpoints")
    if endpoints is None:
        return
    if not isinstance(endpoints, list):
        raise PaidfNativeError("rendered PAIDF endpoints must be a list")
    for endpoint in endpoints:
        if not isinstance(endpoint, dict):
            raise PaidfNativeError("rendered PAIDF endpoint is not an object")
        key_environment = endpoint.get("api_key_env")
        if key_environment in {"VLM_API_KEY", "LLM_API_KEY", "NVIDIA_API_KEY"}:
            _require_token_factory_endpoint(
                str(endpoint.get("url") or ""), str(endpoint.get("role") or "model")
            )


def _validate_local_generation_endpoint(
    config: Any, workflow: str, port: int | None = None
) -> None:
    """Bind the direct translation to the exact local model service it starts."""

    _validate_configured_token_factory_endpoints(config)
    endpoints = config.get("endpoints")
    role = "image_edit" if workflow == "iaa" else "image2video"
    matches = [item for item in endpoints or [] if item.get("role") == role]
    if len(matches) != 1:
        raise PaidfNativeError("direct generation requires exactly one local model endpoint")
    endpoint = matches[0]
    parsed = urlparse(str(endpoint.get("url") or ""))
    try:
        actual_port = parsed.port
    except ValueError as exc:
        raise PaidfNativeError("direct generation endpoint has an invalid port") from exc
    if (
        parsed.scheme != "http" or parsed.hostname != "127.0.0.1"
        or actual_port is None or (port is not None and actual_port != port)
        or parsed.path.rstrip("/") != "/v1" or parsed.query or parsed.fragment
        or parsed.username or parsed.password
        or endpoint.get("api_key_env") != "GENERATION_API_KEY"
    ):
        raise PaidfNativeError("direct generation endpoint must name its local model service")
    _require_direct_generation_model(workflow, str(endpoint.get("model") or ""))
    if any(item.get("api_key_env") == "GENERATION_API_KEY" and item is not endpoint for item in endpoints):
        raise PaidfNativeError("generation credential alias may only name the local model service")
    if workflow == "evg":
        _require_evg_request_guardrails(config)


def _require_evg_request_guardrails(config: Any) -> None:
    augmentation = config.get("augmentation") if isinstance(config, dict) else None
    parameters = augmentation.get("parameters") if isinstance(augmentation, dict) else None
    extra = parameters.get("extra_params") if isinstance(parameters, dict) else None
    if not isinstance(extra, dict) or extra.get("guardrails") is not True:
        raise PaidfNativeError("EVG requires explicit enabled guardrails in every executed request")


def _run_component(
    argv: list[str],
    *,
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    retry_delay_seconds: float = 30.0,
) -> None:
    """Match the upstream DAG's three retries and 30-second retry delay."""

    for attempt in range(4):
        try:
            subprocess.run(argv, check=True, cwd=cwd, env=env)
            return
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(retry_delay_seconds)


def _configure_multistorage(*uris: str) -> None:
    """Expose NPA's run-scoped S3 route in the upstream MSC protocol.

    PAIDF's Kubernetes manifests construct this JSON from a Secret. NPA already
    injects the equivalent run-scoped S3 environment into each SkyPilot task, so
    the native translation constructs the payload in memory immediately before
    invoking the vendor CLI. It is never written to an artifact or printed.
    """

    if os.environ.get("MULTISTORAGECLIENT_CONFIGURATION", "").strip():
        return
    buckets = sorted(
        {
            parsed.netloc
            for value in uris
            if (parsed := urlparse(str(value))).scheme == "s3" and parsed.netloc
        }
    )
    if not buckets:
        return
    endpoint = (
        os.environ.get("AWS_ENDPOINT_URL", "").strip()
        or os.environ.get("NEBIUS_S3_ENDPOINT", "").strip()
    )
    access_key = os.environ.get("AWS_ACCESS_KEY_ID", "").strip()
    secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY", "").strip()
    if not (endpoint and access_key and secret_key):
        raise PaidfNativeError(
            "PAIDF remote storage requires the run-scoped S3 endpoint and credentials"
        )
    region = os.environ.get("AWS_DEFAULT_REGION", "").strip() or "us-east-1"
    profiles = {}
    path_mapping = {}
    for bucket in buckets:
        profiles[bucket] = {
            "storage_provider": {
                "type": "s3",
                "options": {
                    "base_path": bucket,
                    "region_name": region,
                    "endpoint_url": endpoint,
                    "infer_content_type": True,
                },
            },
            "credentials_provider": {
                "type": "S3Credentials",
                "options": {"access_key": access_key, "secret_key": secret_key},
            },
        }
        path_mapping[f"s3://{bucket}/"] = f"msc://{bucket}/"
    os.environ["MULTISTORAGECLIENT_CONFIGURATION"] = json.dumps(
        {"profiles": profiles, "path_mapping": path_mapping}, separators=(",", ":")
    )


def _is_s3(value: str) -> bool:
    return value.startswith("s3://")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_video(path: Path) -> None:
    """Decode a real video stream without relying on host ffmpeg executables."""

    try:
        import av
    except ImportError as exc:
        raise PaidfNativeError("PyAV is required to validate generated video") from exc

    frame_count = 0
    try:
        with av.open(str(path)) as container:
            streams = list(container.streams.video)
            if not streams:
                raise PaidfNativeError("generated media has no video stream")
            for frame in container.decode(streams[0]):
                if frame.width <= 0 or frame.height <= 0:
                    raise PaidfNativeError(
                        "generated video decoded a frame with invalid dimensions"
                    )
                frame_count += 1
    except PaidfNativeError:
        raise
    except (av.FFmpegError, OSError, ValueError) as exc:
        raise PaidfNativeError("generated video could not be decoded") from exc
    if frame_count < 2:
        raise PaidfNativeError("generated video decoded fewer than two frames")


def _materialize(uri: str, destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if _is_s3(uri):
        StorageClient.from_environment().download_path(uri, str(destination))
        return destination
    source = Path(uri)
    if not source.exists():
        raise PaidfNativeError(f"input does not exist: {uri}")
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(source, destination)
    return destination


def _publish(path: Path, uri: str) -> str:
    if _is_s3(uri):
        return StorageClient.from_environment().upload_path(str(path), uri)
    destination = Path(uri)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if path.is_dir():
        shutil.copytree(path, destination, dirs_exist_ok=True)
    else:
        shutil.copy2(path, destination)
    return str(destination)


def _write_json(payload: dict[str, Any], uri: str) -> dict[str, Any]:
    if str(payload.get("schema", "")).startswith(f"{SCHEMA_PREFIX}."):
        if not isinstance(payload.get("run_id"), str) or not payload["run_id"].strip():
            raise PaidfNativeError("native artifacts require the workflow run identity")
        runtime_image = os.environ.get("NPA_TASK_IMAGE", "").strip()
        if runtime_image:
            payload["runtime_image"] = runtime_image
    with tempfile.TemporaryDirectory(prefix="npa-paidf-json-") as tmp:
        local = Path(tmp) / "payload.json"
        local.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        _publish(local, uri)
    return payload


def _uri_is_file(uri: str) -> bool:
    """Return whether an exact local/S3 artifact exists without downloading it."""

    if not _is_s3(uri):
        return Path(uri).is_file()
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if not key or key.endswith("/"):
        return False
    try:
        StorageClient.from_environment().s3.head_object(Bucket=parsed.netloc, Key=key)
    except Exception:  # noqa: BLE001 - provider-specific not-found responses
        return False
    return True


def _uri_prefix_has_objects(uri: str) -> bool:
    """Return whether a local/S3 directory prefix contains at least one object."""

    if not _is_s3(uri):
        path = Path(uri)
        return path.is_file() or (
            path.is_dir() and any(item.is_file() for item in path.rglob("*"))
        )
    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/").rstrip("/") + "/"
    response = StorageClient.from_environment().s3.list_objects_v2(
        Bucket=parsed.netloc, Prefix=prefix, MaxKeys=1
    )
    return bool(response.get("Contents"))


def _list_s3_images(uri: str) -> list[str]:
    parsed = urlparse(uri)
    prefix = parsed.path.lstrip("/")
    if prefix and not prefix.endswith("/"):
        prefix += "/"
    client = StorageClient.from_environment().s3
    paginator = client.get_paginator("list_objects_v2")
    values: list[str] = []
    for page in paginator.paginate(Bucket=parsed.netloc, Prefix=prefix):
        for item in page.get("Contents", []):
            key = str(item.get("Key") or "")
            if Path(key).suffix.lower() in IMAGE_SUFFIXES:
                values.append(f"s3://{parsed.netloc}/{key}")
    return sorted(values)


def _list_images(uri: str) -> list[str]:
    if _is_s3(uri):
        return _list_s3_images(uri)
    path = Path(uri)
    if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES:
        return [str(path)]
    if path.is_dir():
        return sorted(
            str(item)
            for item in path.rglob("*")
            if item.suffix.lower() in IMAGE_SUFFIXES
        )
    raise PaidfNativeError(f"input URI contains no readable image path: {uri}")


def prepare_images(
    input_uri: str, output_uri: str, manifest_uri: str, run_id: str
) -> dict[str, Any]:
    """Validate real image bytes and stage the canonical IAA/EVG input set."""

    images = _list_images(input_uri)
    if not images:
        raise PaidfNativeError("PAIDF input preparation found no supported images")
    from PIL import Image

    records: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="npa-paidf-input-") as tmp:
        root = Path(tmp)
        for index, source_uri in enumerate(images):
            suffix = Path(urlparse(source_uri).path).suffix.lower()
            source = root / f"source-{index:04d}{suffix}"
            local = root / f"input-{index:04d}.jpg"
            _materialize(source_uri, source)
            source_sha256 = _sha256(source)
            try:
                with Image.open(source) as image:
                    image.verify()
                with Image.open(source) as image:
                    width, height = image.size
                    image.convert("RGB").save(
                        local,
                        format="JPEG",
                        quality=95,
                        optimize=False,
                        progressive=False,
                        subsampling=0,
                    )
            except Exception as exc:  # noqa: BLE001 - media decoder boundary
                raise PaidfNativeError(
                    f"invalid input image at index {index}: {exc}"
                ) from exc
            if width < 64 or height < 64:
                raise PaidfNativeError(
                    f"input image {index} is too small: {width}x{height}"
                )
            target_uri = f"{output_uri.rstrip('/')}/{local.name}"
            _publish(local, target_uri)
            pane_metadata = {
                "image_order": [local.name],
                "widths": [width],
                "height": height,
                "original_resolutions": [{"width": width, "height": height}],
            }
            pane_metadata_uri = f"{output_uri.rstrip('/')}/{local.stem}.json"
            _write_json(pane_metadata, pane_metadata_uri)
            records.append(
                {
                    "input_key": f"input-{index:04d}",
                    "source_uri": source_uri,
                    "source_sha256": source_sha256,
                    "prepared_uri": target_uri,
                    "pane_metadata_uri": pane_metadata_uri,
                    "sha256": _sha256(local),
                    "width": width,
                    "height": height,
                }
            )
    payload = {
        "schema": f"{SCHEMA_PREFIX}.prepared-input.v1",
        "run_id": run_id,
        "count": len(records),
        "images": records,
    }
    return _write_json(payload, manifest_uri)


_IAA_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "top_outer_color": (
        "beige",
        "black",
        "blue",
        "brown",
        "green",
        "grey",
        "red",
        "white",
    ),
    "top_outer_type": ("hoodie", "jacket", "sweater", "vest"),
    "bottom_type": ("jeans", "leggings", "shorts", "skirt"),
    "bottom_color": ("beige", "black", "blue", "brown", "grey", "white"),
    "shoe_type": ("boots", "sandals", "sneakers"),
    "shoe_color": ("black", "brown", "grey", "white"),
}


_EVG_DISTRIBUTIONS: dict[str, tuple[str, ...]] = {
    "anomaly_type": (
        "person_falling",
        "person_climbing",
        "person_running",
        "smoking_or_vaping",
        "fire_or_smoke",
    ),
    "env_type": ("warehouse", "retail", "office", "education_facility"),
}


def _read_json(uri: str) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="npa-paidf-read-") as tmp:
        local = Path(tmp) / "value.json"
        _materialize(uri, local)
        value = json.loads(local.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise PaidfNativeError(f"expected JSON object: {uri}")
    return value


def _producer_descriptor(uri: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Bind a producer's semantic JSON document, including its runtime identity."""

    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return {
        "uri": uri,
        "document_sha256": hashlib.sha256(encoded).hexdigest(),
        **{key: payload[key] for key in (
            "schema", "run_id", "workflow", "stage", "runtime_image",
            "upstream_revision", "component", "source_adaptation", "generation_runtime",
            "request_media_contract",
        ) if key in payload},
    }


def _require_evg_generation_runtime(payload: dict[str, Any]) -> None:
    from npa.workflows.paidf_guardrails import PaidfGuardrailError, require_evg_generation_runtime

    try:
        require_evg_generation_runtime(payload.get("generation_runtime"))
    except PaidfGuardrailError as exc:
        raise PaidfNativeError(str(exc)) from exc


def _lineage_kinds(workflow: str) -> list[str]:
    kinds = [f"{workflow}-augmentation", f"{workflow}-validation"]
    if workflow == "iaa":
        kinds.append("iaa-postprocess")
        stages = ("person-attribute-search",)
    else:
        stages = _EVG_LABEL_STAGES
    return [*kinds, *(f"{workflow}-auto-label-{stage}" for stage in stages)]


def _verified_producers(
    descriptors: Any, kinds: list[str], run_id: str, workflow: str
) -> list[dict[str, Any]]:
    if not isinstance(descriptors, list) or len(descriptors) != len(kinds):
        raise PaidfNativeError("producer lineage is missing a required stage")
    documents = []
    for descriptor, kind in zip(descriptors, kinds, strict=True):
        if not isinstance(descriptor, dict) or not descriptor.get("uri"):
            raise PaidfNativeError("producer lineage has no artifact reference")
        document = _read_run_artifact(descriptor["uri"], kind, run_id, workflow)
        if descriptor != _producer_descriptor(descriptor["uri"], document):
            raise PaidfNativeError("producer document changed after its handoff")
        if "-auto-label-" in kind and document.get("stage") != kind.split("-auto-label-", 1)[1]:
            raise PaidfNativeError("producer stage identity does not match its schema")
        if kind in {"evg-auto-label-visual-qa-anomaly", "evg-auto-label-visual-qa-person"}:
            _require_evg_vqa_request_media_contract(document)
        if document.get("producers", []) != descriptors[:len(documents)]:
            raise PaidfNativeError("producer lineage does not preserve its predecessors")
        if kind.endswith("-augmentation"):
            _require_paidf_image_output_adaptation(document.get("source_adaptation"))
            if workflow == "evg":
                _require_evg_generation_runtime(document)
            outputs = document.get("outputs")
            if not isinstance(outputs, list) or not outputs:
                raise PaidfNativeError("augmentation producer has no completed artifacts")
            for output in outputs:
                _verify_fingerprints(output.get("artifacts"))
                if workflow == "evg":
                    with tempfile.TemporaryDirectory(prefix="npa-paidf-guardrail-lineage-") as tmp:
                        config = _materialize(output["config_uri"], Path(tmp) / "config.yaml")
                        _require_evg_request_guardrails(yaml.safe_load(config.read_text(encoding="utf-8")))
        documents.append(document)
    return documents


def _artifact_fingerprints(uris: dict[str, str]) -> dict[str, Any]:
    records = {}
    with tempfile.TemporaryDirectory(prefix="npa-paidf-handoff-") as tmp:
        for index, (name, uri) in enumerate(uris.items()):
            local = _materialize(uri, Path(tmp) / str(index))
            if not local.is_file() or not local.stat().st_size:
                raise PaidfNativeError("producer omitted a required nonempty artifact")
            records[name] = {"uri": uri, "sha256": _sha256(local), "size_bytes": local.stat().st_size}
    return records


def _verify_fingerprints(records: Any) -> None:
    if not isinstance(records, dict) or not records:
        raise PaidfNativeError("producer has no artifact content manifest")
    try:
        actual = _artifact_fingerprints({name: record["uri"] for name, record in records.items()})
    except (KeyError, TypeError, OSError) as exc:
        raise PaidfNativeError("producer artifact content manifest is malformed or its bytes are missing") from exc
    if actual != records:
        raise PaidfNativeError("producer artifact bytes changed after their handoff")


def _token_factory_child_env() -> dict[str, str]:
    token = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if not token:
        raise PaidfNativeError("NEBIUS_TOKEN_FACTORY_KEY is required by the upstream protocol")
    return {
        **os.environ,
        **dict.fromkeys(("VLM_API_KEY", "LLM_API_KEY", "NVIDIA_API_KEY"), token),
        "GENERATION_API_KEY": "local",
    }


def _verify_label_handoffs(
    documents: list[dict[str, Any]], accepted: list[dict[str, Any]],
    auto_label_root_uri: str | None = None,
) -> None:
    expected = {f"{item['input_key']}_aug{item['augmentation_index']}": item for item in accepted}
    if len(expected) != len(accepted):
        raise PaidfNativeError("validated media contains duplicate scene identities")
    scene_paths: dict[str, str] = {}
    for document in documents:
        if "-auto-label-" not in document.get("schema", ""):
            continue
        outputs = document.get("outputs")
        if not isinstance(outputs, list) or document.get("count") != len(outputs):
            raise PaidfNativeError("labeling producer has an invalid output count")
        actual = {item.get("key"): item for item in outputs}
        if len(actual) != len(outputs) or actual.keys() != expected.keys():
            raise PaidfNativeError("labeling producer scene set differs from validated media")
        for key, output in actual.items():
            item = expected[key]
            path = output.get("data_path")
            if not path or output.get("media_uri") != item["media_uri"]:
                raise PaidfNativeError("labeling producer names foreign media or no scene")
            if auto_label_root_uri is not None and path != f"{auto_label_root_uri.rstrip('/')}/{item['input_key']}/{item['augmentation_index']}":
                raise PaidfNativeError("labeling producer names another scene root")
            if key in scene_paths and scene_paths[key] != path:
                raise PaidfNativeError("labeling producers disagree on the scene path")
            scene_paths[key] = path
            records = output.get("artifacts")
            if records:
                if any(record.get("uri") != f"{path}/{name}" for name, record in records.items()):
                    raise PaidfNativeError("labeling content manifest names a foreign scene")
                _verify_fingerprints(records)
            elif not (document.get("stage") == "person-attribute-search" and output.get("trackless") is True):
                raise PaidfNativeError("labeling producer has no output content manifest")


def _require_artifact_identity(
    payload: dict[str, Any], schema: str, run_id: str, workflow: str | None = None
) -> None:
    if (
        not run_id.strip()
        or payload.get("schema") != schema
        or payload.get("run_id") != run_id
    ):
        raise PaidfNativeError(
            "artifact schema or run identity does not match the consuming stage"
        )
    if workflow is not None and payload.get("workflow") != workflow:
        raise PaidfNativeError("artifact workflow does not match the consuming stage")


def _read_run_artifact(
    uri: str, kind: str, run_id: str, workflow: str | None = None
) -> dict[str, Any]:
    payload = _read_json(uri)
    _require_artifact_identity(payload, f"{SCHEMA_PREFIX}.{kind}.v1", run_id, workflow)
    return payload


def _read_config_manifest(uri: str, run_id: str) -> dict[str, Any]:
    manifest = _read_json(uri)
    workflow = manifest.get("workflow")
    if workflow not in {"iaa", "evg"}:
        raise PaidfNativeError("augmentation config manifest has no supported workflow")
    _require_artifact_identity(
        manifest, f"{SCHEMA_PREFIX}.{workflow}-configs.v1", run_id, workflow
    )
    return manifest


def _require_upstream_identity(
    payload: dict[str, Any], workflow: str, run_id: str
) -> None:
    _require_artifact_identity(payload, "npa.paidf.upstream.v1", run_id)
    expected_variant = {
        "iaa": "image-attribute-augmentation",
        "evg": "event-video-generation",
    }[workflow]
    if payload.get("workflow_variant") != expected_variant:
        raise PaidfNativeError("upstream provenance belongs to another workflow")


def build_augmentation_configs(
    workflow: str,
    prepared_manifest_uri: str,
    output_uri: str,
    config_manifest_uri: str,
    num_augmentations: int,
    seed: int,
    vlm_url: str,
    vlm_model: str,
    llm_url: str,
    llm_model: str,
    generation_url: str,
    generation_model: str,
    run_id: str,
) -> dict[str, Any]:
    """Render the published PAIDF Augmentation request protocol deterministically."""

    if workflow not in {"iaa", "evg"}:
        raise PaidfNativeError("workflow must be iaa or evg")
    _require_token_factory_endpoint(vlm_url, "VLM")
    _require_token_factory_endpoint(llm_url, "LLM")
    _require_direct_generation_model(workflow, generation_model)
    if not all(
        value.strip()
        for value in (vlm_url, vlm_model, llm_url, llm_model, generation_url)
    ):
        raise PaidfNativeError(
            "VLM, LLM, and generation endpoints/models must be explicit"
        )
    prepared = _read_run_artifact(prepared_manifest_uri, "prepared-input", run_id)
    images = prepared.get("images")
    if not isinstance(images, list) or not images:
        raise PaidfNativeError("prepared input manifest contains no images")
    if num_augmentations < 1:
        raise PaidfNativeError("num_augmentations must be positive")

    rng = random.Random(seed)
    distributions = _IAA_DISTRIBUTIONS if workflow == "iaa" else _EVG_DISTRIBUTIONS
    configs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="npa-paidf-config-") as tmp:
        root = Path(tmp)
        orchestration = _runtime_fetch(
            "https://github.com/NVIDIA/paidf-orchestration.git",
            PAIDF_ORCHESTRATION_REVISION,
            root / "orchestration",
        )
        workflow_dir = (
            "image_attribute_augmentation_dag"
            if workflow == "iaa"
            else "event_video_generation_dag"
        )
        template_path = (
            orchestration
            / "airflow/dags/workflows"
            / workflow_dir
            / "configs/cosmos_config.yaml"
        )
        template = yaml.safe_load(template_path.read_text(encoding="utf-8"))
        if not isinstance(template, dict):
            raise PaidfNativeError("upstream PAIDF cosmos_config.yaml is not an object")
        for image in images:
            input_key = str(image["input_key"])
            for augmentation_index in range(num_augmentations):
                variables = {
                    key: rng.choice(values) for key, values in distributions.items()
                }
                base_uri = (
                    f"{output_uri.rstrip('/')}/cosmos/{input_key}/{augmentation_index}"
                )
                suffix = "jpg" if workflow == "iaa" else "mp4"
                config = copy.deepcopy(template)
                config["data"][0]["inputs"]["rgb"] = image["prepared_uri"]
                config["data"][0]["output"] = {
                    "video": f"{base_uri}/output.{suffix}",
                    "caption": f"{base_uri}/caption.txt",
                    "metadata": f"{base_uri}/output_metadata.json",
                }
                role_values = {
                    "vlm": (vlm_url, vlm_model, "VLM_API_KEY"),
                    "llm": (llm_url, llm_model, "LLM_API_KEY"),
                    "image_edit": (
                        generation_url,
                        generation_model,
                        "GENERATION_API_KEY",
                    ),
                    "image2video": (
                        generation_url,
                        generation_model,
                        "GENERATION_API_KEY",
                    ),
                }
                for endpoint in config["endpoints"]:
                    role = endpoint.get("role")
                    if role in role_values:
                        endpoint["url"], endpoint["model"], endpoint["api_key_env"] = (
                            role_values[role]
                        )
                config["captioning"]["llm"]["variables"] = {
                    name: [value] for name, value in variables.items()
                }
                parameters = config["augmentation"]["parameters"]
                if workflow == "iaa":
                    parameters["extra_body"]["seed"] = seed + len(configs)
                else:
                    parameters["seed"] = seed + len(configs)
                    if not isinstance(parameters.get("extra_params"), dict):
                        raise PaidfNativeError("upstream EVG config has no extra_params object")
                    parameters["extra_params"]["guardrails"] = True
                    _require_evg_request_guardrails(config)
                local = root / f"{input_key}-{augmentation_index}.yaml"
                local.write_text(
                    yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
                )
                config_uri = f"{output_uri.rstrip('/')}/configs/{local.name}"
                _publish(local, config_uri)
                configs.append(
                    {
                        "input_key": input_key,
                        "augmentation_index": augmentation_index,
                        "variables": variables,
                        "config_uri": config_uri,
                        "media_uri": config["data"][0]["output"]["video"],
                        "caption_uri": config["data"][0]["output"]["caption"],
                        "metadata_uri": config["data"][0]["output"]["metadata"],
                    }
                )
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{workflow}-configs.v1",
        "run_id": run_id,
        "workflow": workflow,
        "seed": seed,
        "count": len(configs),
        "upstream_protocol": "NVIDIA/paidf-augmentation@" + PAIDF_AUGMENTATION_REVISION,
        "configs": configs,
    }
    return _write_json(payload, config_manifest_uri)


def _runtime_fetch(repository: str, revision: str, destination: Path) -> Path:
    subprocess.run(["git", "init", "-q", str(destination)], check=True)
    subprocess.run(
        ["git", "-C", str(destination), "remote", "add", "origin", repository],
        check=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(destination),
            "fetch",
            "-q",
            "--depth",
            "1",
            "origin",
            revision,
        ],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(destination), "checkout", "-q", "--detach", "FETCH_HEAD"],
        check=True,
    )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != revision:
        raise PaidfNativeError("runtime-fetched upstream revision mismatch")
    return destination


def _paidf_image_output_patch_bytes(original: bytes) -> bytes:
    """Apply the reviewed executor edit after its caller verifies source identity."""

    text = original.decode("utf-8")
    import_anchor = "import multistorageclient as msc\n"
    media_anchor = '_MEDIA_KINDS = frozenset({"image", "video", "control"})\n\n\n'
    write_anchor = (
        '        if result.media_bytes is not None:\n'
        '            with msc.open(output_path, "wb") as f:\n'
        '                f.write(result.media_bytes)\n'
    )
    if any(text.count(anchor) != 1 for anchor in (import_anchor, media_anchor, write_anchor)):
        raise PaidfNativeError(
            "pinned PAIDF executor no longer has the reviewed MIME patch anchors"
        )
    text = text.replace(
        import_anchor,
        "import cv2\nimport multistorageclient as msc\nimport numpy as np\n",
        1,
    )
    text = text.replace(
        media_anchor,
        '''_MEDIA_KINDS = frozenset({"image", "video", "control"})


def _media_bytes_for_output(media_bytes: bytes, output_path: str) -> bytes:
    """Make image bytes agree with a JPEG output path before VLM reuse."""
    extension = os.path.splitext(output_path)[1].lower()
    if extension not in {".jpg", ".jpeg"}:
        return media_bytes
    decoded = cv2.imdecode(
        np.frombuffer(media_bytes, dtype=np.uint8), cv2.IMREAD_COLOR
    )
    if decoded is None:
        raise ValueError("adapter returned undecodable image bytes for JPEG output")
    encoded_ok, encoded = cv2.imencode(
        ".jpg", decoded, [cv2.IMWRITE_JPEG_QUALITY, 95]
    )
    if not encoded_ok or not encoded.size:
        raise ValueError("could not encode adapter output as JPEG")
    payload = encoded.tobytes()
    if not payload.startswith(b"\\xff\\xd8\\xff"):
        raise ValueError("encoded JPEG output has invalid magic bytes")
    return payload


''',
        1,
    )
    text = text.replace(
        write_anchor,
        '        if result.media_bytes is not None:\n'
        '            payload = _media_bytes_for_output(result.media_bytes, output_path)\n'
        '            with msc.open(output_path, "wb") as f:\n'
        '                f.write(payload)\n',
        1,
    )
    return text.encode("utf-8")


def _paidf_image_output_adaptation() -> dict[str, Any]:
    manifest = {
        "schema": "npa.paidf.upstream-source-adaptation.v1",
        "upstream_revision": PAIDF_AUGMENTATION_REVISION,
        "purpose": "jpeg-output-byte-contract",
        "path": _PAIDF_EXECUTOR_PATH,
        "original_sha256": _PAIDF_EXECUTOR_SHA256,
        "patched_sha256": _PAIDF_EXECUTOR_PATCHED_SHA256,
    }
    manifest["patch_sha256"] = hashlib.sha256(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return manifest


def _require_paidf_image_output_adaptation(value: Any) -> None:
    if value != _paidf_image_output_adaptation():
        raise PaidfNativeError(
            "augmentation producer lacks the reviewed image MIME source adaptation"
        )


def _patch_paidf_image_output_contract(source: Path) -> dict[str, Any]:
    """Bind a narrow JPEG writer fix to the reviewed upstream source bytes.

    The image-edit API returns PNG bytes while the published IAA config names a
    ``.jpg`` output. Upstream writes those bytes unchanged and later declares
    them ``image/jpeg`` to its VLM, which Token Factory correctly rejects. The
    patch keeps the published request and output paths intact and makes the
    writer honor the existing JPEG extension before any evaluator reads it.
    """

    target = source / _PAIDF_EXECUTOR_PATH
    try:
        original = target.read_bytes()
    except OSError as exc:
        raise PaidfNativeError(
            "pinned PAIDF executor required for the image MIME adaptation is missing"
        ) from exc
    original_sha256 = hashlib.sha256(original).hexdigest()
    if original_sha256 != _PAIDF_EXECUTOR_SHA256:
        raise PaidfNativeError(
            "pinned PAIDF executor bytes differ from the reviewed MIME adaptation"
        )
    patched = _paidf_image_output_patch_bytes(original)
    patched_sha256 = hashlib.sha256(patched).hexdigest()
    if patched_sha256 != _PAIDF_EXECUTOR_PATCHED_SHA256:
        raise PaidfNativeError("PAIDF image MIME adaptation produced unexpected bytes")
    target.write_bytes(patched)
    return _paidf_image_output_adaptation()


def run_augmentation(
    config_manifest_uri: str, result_uri: str, run_id: str,
    *, generation_port: int | None = None, generation_runtime: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Invoke the real paidf-augmentation CLI once for every rendered config."""

    manifest = _read_config_manifest(config_manifest_uri, run_id)
    configs = manifest.get("configs")
    if not isinstance(configs, list) or not configs:
        raise PaidfNativeError("augmentation config manifest is empty")
    workflow = manifest.get("workflow")
    if workflow not in {"iaa", "evg"}:
        raise PaidfNativeError("augmentation config manifest has no supported workflow")
    _configure_multistorage(
        config_manifest_uri,
        result_uri,
        *(str(value) for item in configs for value in item.values()),
    )
    component_env = _token_factory_child_env()

    with tempfile.TemporaryDirectory(prefix="npa-paidf-augmentation-") as tmp:
        root = Path(tmp)
        local_configs: list[Path] = []
        for index, item in enumerate(configs):
            local = root / f"config-{index:04d}.yaml"
            _materialize(str(item["config_uri"]), local)
            _validate_local_generation_endpoint(
                yaml.safe_load(local.read_text(encoding="utf-8")),
                workflow,
                generation_port,
            )
            local_configs.append(local)
        source = _runtime_fetch(
            "https://github.com/NVIDIA/paidf-augmentation.git",
            PAIDF_AUGMENTATION_REVISION,
            root / "source",
        )
        command_prefix = [
            "uv",
            "run",
            "--project",
            str(source),
            "--no-sync",
            "--python",
            sys.executable,
            "python",
            str(source / "modules/cli.py"),
        ]
        source_adaptation = _patch_paidf_image_output_contract(source)
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(source),
                "--frozen",
                "--python",
                sys.executable,
            ],
            check=True,
        )
        completed: list[dict[str, Any]] = []
        failed: list[dict[str, Any]] = []
        for item, local in zip(configs, local_configs, strict=True):
            try:
                _run_component([*command_prefix, "--config", str(local)], env=component_env)
                content = _artifact_fingerprints({
                    field: item[field] for field in ("config_uri", "media_uri", "caption_uri", "metadata_uri")
                })
            except (subprocess.CalledProcessError, PaidfNativeError, OSError) as exc:
                # IAA's mapped join retains successful siblings after exhausted
                # component retries. EVG requires every expected augmentation.
                if workflow == "evg":
                    raise
                failed.append(
                    {
                        "config_uri": item["config_uri"],
                        "input_key": item["input_key"],
                        "augmentation_index": item["augmentation_index"],
                        "exit_code": getattr(exc, "returncode", None),
                        "reason": "component_retries_exhausted" if isinstance(exc, subprocess.CalledProcessError) else "component_output_missing_or_empty",
                    }
                )
                continue
            completed.append(
                {
                    **item,
                    "artifacts": content,
                }
            )
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{workflow}-augmentation.v1",
        "run_id": run_id,
        "workflow": workflow,
        "count": len(completed),
        "attempted_count": len(configs),
        "failed_count": len(failed),
        "failed": failed,
        "component": "NVIDIA paidf-augmentation 1.1.0",
        "upstream_revision": PAIDF_AUGMENTATION_REVISION,
        "source_adaptation": source_adaptation,
        "outputs": completed,
        "config_manifest_uri": config_manifest_uri,
    }
    if generation_runtime is not None:
        payload["generation_runtime"] = generation_runtime
    _write_json(payload, result_uri)
    if not completed:
        raise PaidfNativeError(
            "every mapped PAIDF augmentation exhausted component retries"
        )
    return payload


def run_local_augmentation(
    config_manifest_uri: str,
    result_uri: str,
    generation_model: str,
    generation_revision: str,
    service_kind: str,
    port: int,
    parallel_size: int,
    run_id: str,
) -> dict[str, Any]:
    """Keep a genuine vLLM-Omni service alive while PAIDF runs its batch.

    SkyPilot task setup cannot own a long-lived child process, so the service and
    its consuming PAIDF batch intentionally share one state/run boundary.
    """

    workflow = _SERVICE_WORKFLOWS.get(service_kind)
    if workflow is None:
        raise PaidfNativeError("service_kind must be image-edit or image2video")
    _require_direct_generation_model(workflow, generation_model, generation_revision)
    manifest = _read_config_manifest(config_manifest_uri, run_id)
    if manifest["workflow"] != workflow:
        raise PaidfNativeError(
            "augmentation manifest workflow does not match the generation service"
        )

    configs = manifest.get("configs")
    if not isinstance(configs, list) or not configs:
        raise PaidfNativeError("local generation requires a nonempty configured batch")
    with tempfile.TemporaryDirectory(prefix="npa-paidf-service-preflight-") as tmp:
        for index, item in enumerate(configs):
            local = _materialize(str(item["config_uri"]), Path(tmp) / f"{index}.yaml")
            _validate_local_generation_endpoint(
                yaml.safe_load(local.read_text(encoding="utf-8")), workflow, port
            )

    if service_kind == "image-edit":
        command = [
            "vllm",
            "serve",
            generation_model,
            "--revision",
            generation_revision,
            "--omni",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
        ]
    elif service_kind == "image2video":
        if parallel_size not in {1, 2}:
            raise PaidfNativeError("the installed Cosmos3 CFG protocol supports parallel_size 1 or 2")
        command = [
            "vllm",
            "serve",
            generation_model,
            "--revision",
            generation_revision,
            "--omni",
            "--host",
            "127.0.0.1",
            "--port",
            str(port),
            "--cfg-parallel-size",
            str(parallel_size),
            "--use-hsdp",
            "--hsdp-shard-size",
            str(parallel_size),
            "--init-timeout",
            "1800",
        ]
    else:
        raise PaidfNativeError("service_kind must be image-edit or image2video")
    generation_runtime = None
    service_options: dict[str, Any] = {"start_new_session": True}
    if workflow == "evg":
        from npa.workflows.paidf_guardrails import (
            PaidfGuardrailError,
            prepare_evg_generation_environment,
        )

        try:
            environment, generation_runtime = prepare_evg_generation_environment()
        except PaidfGuardrailError as exc:
            raise PaidfNativeError(str(exc)) from exc
        service_options["env"] = environment
    service = subprocess.Popen(command, **service_options)  # noqa: S603 - fixed executable contract
    try:
        health = f"http://127.0.0.1:{port}/health"
        while True:
            if service.poll() is not None:
                raise PaidfNativeError(
                    f"vLLM-Omni exited before readiness ({service.returncode})"
                )
            try:
                with urllib.request.urlopen(health, timeout=5) as response:  # noqa: S310 - loopback only
                    if 200 <= response.status < 300:
                        break
            except OSError:
                time.sleep(5)
        batch_options: dict[str, Any] = {"generation_port": port}
        if generation_runtime is not None:
            batch_options["generation_runtime"] = generation_runtime
        return run_augmentation(config_manifest_uri, result_uri, run_id, **batch_options)
    finally:
        service.terminate()
        try:
            service.wait(timeout=30)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()


def validate_augmentation(
    config_manifest_uri: str, validation_uri: str, run_id: str,
    augmentation_result_uri: str,
) -> dict[str, Any]:
    """Require decodable media plus non-empty caption and metadata for each output."""

    manifest = _read_config_manifest(config_manifest_uri, run_id)
    workflow = manifest.get("workflow")
    if workflow not in {"iaa", "evg"}:
        raise PaidfNativeError("augmentation config manifest has no supported workflow")
    producer = _read_run_artifact(augmentation_result_uri, f"{workflow}-augmentation", run_id, workflow)
    _require_paidf_image_output_adaptation(producer.get("source_adaptation"))
    if workflow == "evg":
        _require_evg_generation_runtime(producer)
    configs = manifest.get("configs")
    outputs = producer.get("outputs")
    failed = producer.get("failed")
    if not isinstance(configs, list) or not isinstance(outputs, list) or not isinstance(failed, list):
        raise PaidfNativeError("augmentation producer has no completed/failed set")
    configured = {item["config_uri"]: item for item in configs}
    completed = {item["config_uri"]: item for item in outputs}
    failed_by_config = {item["config_uri"]: item for item in failed}
    if (
        producer.get("config_manifest_uri") != config_manifest_uri
        or len(configured) != len(configs) or len(completed) != len(outputs)
        or len(failed_by_config) != len(failed)
        or completed.keys() & failed_by_config.keys()
        or completed.keys() | failed_by_config.keys() != configured.keys()
        or producer.get("count") != len(outputs)
        or producer.get("failed_count") != len(failed)
        or producer.get("attempted_count") != len(configs)
        or (workflow == "evg" and failed)
    ):
        raise PaidfNativeError("augmentation producer completed set does not match the configured batch")
    for uri, output in completed.items():
        if any(output.get(field) != value for field, value in configured[uri].items()):
            raise PaidfNativeError("augmentation producer changed a configured output")
        records = output.get("artifacts")
        if not isinstance(records, dict) or set(records) != {"config_uri", "media_uri", "caption_uri", "metadata_uri"}:
            raise PaidfNativeError("augmentation producer lacks its full content manifest")
        if any(record.get("uri") != output[field] for field, record in records.items()):
            raise PaidfNativeError("augmentation producer content manifest names another output")
        _verify_fingerprints(records)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = [dict(item) for item in failed]
    with tempfile.TemporaryDirectory(prefix="npa-paidf-validate-") as tmp:
        root = Path(tmp)
        for index, item in enumerate(outputs):
            try:
                media = (
                    root
                    / f"media-{index:04d}{Path(urlparse(item['media_uri']).path).suffix}"
                )
                caption = root / f"caption-{index:04d}.txt"
                metadata = root / f"metadata-{index:04d}.json"
                _materialize(item["media_uri"], media)
                _materialize(item["caption_uri"], caption)
                _materialize(item["metadata_uri"], metadata)
                if (
                    media.stat().st_size < 1024
                    or not caption.read_text(encoding="utf-8").strip()
                ):
                    raise PaidfNativeError("media or caption is empty")
                parsed_metadata = json.loads(metadata.read_text(encoding="utf-8"))
                if not isinstance(parsed_metadata, dict):
                    raise PaidfNativeError("metadata is not an object")
                config_path = root / f"config-{index:04d}.yaml"
                _materialize(item["config_uri"], config_path)
                config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
                if workflow == "evg":
                    _require_evg_request_guardrails(config)
                evaluators = (
                    config.get("evaluators") if isinstance(config, dict) else None
                )
                if not isinstance(evaluators, list):
                    raise PaidfNativeError("executed config has no evaluator contract")
                for evaluator in evaluators:
                    if not isinstance(evaluator, dict):
                        raise PaidfNativeError("executed evaluator config is malformed")
                    if "attribute_verification" not in evaluator:
                        continue
                    settings = evaluator["attribute_verification"]
                    if not isinstance(settings, dict):
                        raise PaidfNativeError(
                            "attribute-verification config is malformed"
                        )
                    if settings.get("enabled", True) is not False:
                        verdict = parsed_metadata.get("attribute_verification")
                        if (
                            not isinstance(verdict, dict)
                            or verdict.get("passed") is not True
                        ):
                            raise PaidfNativeError(
                                "attribute_verification did not affirmatively pass"
                            )
                for evaluator in ("attribute_verification", "hallucination_check"):
                    verdict = parsed_metadata.get(evaluator)
                    if isinstance(verdict, dict) and verdict.get("passed") is False:
                        raise PaidfNativeError(f"{evaluator} failed")
                if media.suffix.lower() in IMAGE_SUFFIXES:
                    from PIL import Image

                    with Image.open(media) as image:
                        image.verify()
                else:
                    _validate_video(media)
                accepted.append(
                    {
                        **item,
                        "sha256": _sha256(media),
                        "size_bytes": media.stat().st_size,
                    }
                )
            except Exception as exc:  # noqa: BLE001 - one mapped augmentation may exhaust retries
                skipped.append(
                    {
                        "input_key": item.get("input_key"),
                        "augmentation_index": item.get("augmentation_index"),
                        "reason": str(exc),
                    }
                )
    if not accepted:
        raise PaidfNativeError("every mapped PAIDF augmentation failed validation")
    if workflow == "evg" and skipped:
        raise PaidfNativeError(
            f"EVG requires every expected augmentation; {len(skipped)} failed validation"
        )
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{manifest['workflow']}-validation.v1",
        "run_id": run_id,
        "workflow": workflow,
        "accepted_count": len(accepted),
        "skipped_count": len(skipped),
        "accepted": accepted,
        "skipped": skipped,
        "producers": [_producer_descriptor(augmentation_result_uri, producer)],
    }
    return _write_json(payload, validation_uri)


def postprocess_iaa(
    validation_uri: str,
    prepared_manifest_uri: str,
    output_root_uri: str,
    result_uri: str,
    vlm_url: str,
    vlm_model: str,
    run_id: str,
) -> dict[str, Any]:
    """Run the upstream IAA pane splitter and dataset-label generator."""

    validation = _read_run_artifact(validation_uri, "iaa-validation", run_id, "iaa")
    _require_token_factory_endpoint(vlm_url, "VLM")
    prepared = _read_run_artifact(prepared_manifest_uri, "prepared-input", run_id)
    prepared_by_key = {
        str(item["input_key"]): item for item in prepared.get("images", [])
    }
    accepted = validation.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise PaidfNativeError("IAA post-processing requires validated augmentations")
    _configure_multistorage(
        validation_uri,
        prepared_manifest_uri,
        output_root_uri,
        result_uri,
        *(str(value) for item in accepted for value in item.values()),
    )
    component_env = _token_factory_child_env()
    _verified_producers(validation.get("producers"), ["iaa-augmentation"], run_id, "iaa")

    outputs: list[dict[str, Any]] = []
    skipped = [dict(item) for item in validation.get("skipped", [])]
    with tempfile.TemporaryDirectory(prefix="npa-paidf-iaa-postprocess-") as tmp:
        root = Path(tmp)
        source = _runtime_fetch(
            "https://github.com/NVIDIA/paidf-augmentation.git",
            PAIDF_AUGMENTATION_REVISION,
            root / "source",
        )
        subprocess.run(
            [
                "uv",
                "sync",
                "--project",
                str(source),
                "--frozen",
                "--python",
                sys.executable,
            ],
            check=True,
        )
        script = (
            source / "modules/data_processing/create_attribute_augmented_dataset.py"
        )
        for item in accepted:
            input_key = str(item["input_key"])
            prepared_item = prepared_by_key.get(input_key)
            if prepared_item is None:
                raise PaidfNativeError(
                    f"IAA output has no prepared input for {input_key}"
                )
            augmentation_dir = item["media_uri"].rsplit("/", 1)[0]
            dataset_dir = (
                f"{output_root_uri.rstrip('/')}/{input_key}/"
                f"aug_{item['augmentation_index']}"
            )
            try:
                _run_component(
                    [
                        "uv",
                        "run",
                        "--project",
                        str(source),
                        "--no-sync",
                        "--python",
                        sys.executable,
                        "python",
                        str(script),
                        "--base-dir",
                        str(prepared_item["pane_metadata_uri"]),
                        "--augmented-folders",
                        str(augmentation_dir),
                        "--output-dir",
                        dataset_dir,
                        "--output-json",
                        "augmented_data.json",
                        "--vlm-endpoint",
                        vlm_url,
                        "--vlm-model",
                        vlm_model,
                        "--vlm-api-key-env",
                        "VLM_API_KEY",
                    ],
                    env=component_env,
                )
                dataset_json_uri = f"{dataset_dir}/augmented_data.json"
                dataset_json = _read_json(dataset_json_uri)
                entries = dataset_json.get("entries")
                if not isinstance(entries, list) or not entries:
                    raise PaidfNativeError(
                        f"upstream IAA post-processing produced no entries for {input_key}"
                    )
                item_outputs: list[dict[str, Any]] = []
                for entry in entries:
                    images = entry.get("images")
                    if not isinstance(images, list) or not images:
                        raise PaidfNativeError(
                            "upstream IAA post-processing entry has no images"
                        )
                    if not all(
                        isinstance(entry.get(field), dict)
                        for field in (
                            "attributes",
                            "selected_attributes",
                            "queries",
                            "attribute_verification",
                        )
                    ):
                        raise PaidfNativeError(
                            "upstream IAA post-processing entry lacks attribute labels"
                        )
                    media_uri = (
                        f"{dataset_dir.rstrip('/')}/{images[0].split('/', 1)[-1]}"
                    )
                    # The pane splitter re-encodes the generated image. Its bytes,
                    # rather than the pre-split image's digest, identify this handoff.
                    split_image = root / f"split-{len(outputs) + len(item_outputs)}.jpg"
                    _materialize(media_uri, split_image)
                    from PIL import Image

                    with Image.open(split_image) as image:
                        image.verify()
                    item_outputs.append(
                        {
                            **item,
                            "key": str(entry["person_key"]),
                            "media_uri": media_uri,
                            "sha256": _sha256(split_image),
                            "size_bytes": split_image.stat().st_size,
                            "generation_media_uri": item["media_uri"],
                            "generation_sha256": item["sha256"],
                            "attributes": entry["attributes"],
                            "selected_attributes": entry["selected_attributes"],
                            "queries": entry["queries"],
                            "attribute_verification": entry["attribute_verification"],
                            "postprocess_dataset_uri": dataset_json_uri,
                        }
                    )
                outputs.extend(item_outputs)
            except Exception as exc:  # noqa: BLE001 - mirrors the mapped Airflow join
                failure = {
                    "input_key": item.get("input_key"),
                    "augmentation_index": item.get("augmentation_index"),
                    "stage": "iaa-postprocess",
                    "reason": "component_retries_exhausted"
                    if isinstance(exc, subprocess.CalledProcessError)
                    else "postprocess_output_invalid",
                    "error_type": type(exc).__name__,
                }
                if isinstance(exc, subprocess.CalledProcessError):
                    failure["exit_code"] = exc.returncode
                skipped.append(failure)
    if not outputs:
        raise PaidfNativeError("every mapped IAA post-processing task failed")
    payload = {
        "schema": f"{SCHEMA_PREFIX}.iaa-postprocess.v1",
        "run_id": run_id,
        "workflow": "iaa",
        "accepted_count": len(outputs),
        "accepted": outputs,
        "skipped": skipped,
        "component": "NVIDIA paidf-augmentation create_attribute_augmented_dataset.py",
        "upstream_revision": PAIDF_AUGMENTATION_REVISION,
        "producers": [*validation["producers"], _producer_descriptor(validation_uri, validation)],
    }
    return _write_json(payload, result_uri)


def _asset_to_uri(source: Path, destination_uri: str) -> str:
    if not source.is_file() or not source.stat().st_size:
        raise PaidfNativeError(f"required upstream protocol asset is missing: {source}")
    return _publish(source, destination_uri)


def _validate_iaa_labels(data_path: str) -> None:
    """Validate the pinned PAS image-bundle protocol, excluding input assets.

    Upstream's export/bundle.py publishes matching nonempty ``people`` maps in
    the attribute and query documents. A staged config or prompt cannot satisfy
    this output boundary.
    """

    documents = []
    for relative in IAA_LABEL_ARTIFACTS:
        uri = f"{data_path.rstrip('/')}/{relative}"
        if not _uri_is_file(uri):
            raise PaidfNativeError("IAA attribute search omitted a required bundle")
        document = _read_json(uri)
        people = document.get("people")
        if (
            not isinstance(people, dict)
            or not people
            or document.get("n_people") != len(people)
            or not all(isinstance(person, dict) for person in people.values())
        ):
            raise PaidfNativeError(
                "IAA attribute search produced an invalid people bundle"
            )
        documents.append(document)
    attributes, queries = documents
    if attributes.get("chunk_id") != queries.get("chunk_id") or set(
        attributes["people"]
    ) != set(queries["people"]):
        raise PaidfNativeError("IAA attribute and query bundle identities disagree")
    for key, person in attributes["people"].items():
        if not isinstance(person.get("attributes"), dict):
            raise PaidfNativeError("IAA bundle is missing person attributes")
        query_values = queries["people"][key].get("queries")
        if not isinstance(query_values, dict):
            raise PaidfNativeError("IAA bundle is missing tiered queries")
        for tier in ("easy", "medium", "hard"):
            values = query_values.get(tier)
            if (
                not isinstance(values, list)
                or not values
                or not all(isinstance(value, str) and value.strip() for value in values)
            ):
                raise PaidfNativeError("IAA bundle has empty or invalid tiered queries")


def run_auto_label(
    workflow: str,
    stage: str,
    validation_uri: str,
    auto_label_root_uri: str,
    result_uri: str,
    vlm_url: str,
    vlm_model: str,
    llm_url: str,
    llm_model: str,
    run_id: str,
    previous_result_uri: str = "",
) -> dict[str, Any]:
    """Invoke one genuine paidf-auto-labeling service CLI over validated media."""

    _require_token_factory_endpoint(vlm_url, "VLM")
    _require_token_factory_endpoint(llm_url, "LLM")
    allowed = {
        "iaa": {"person-attribute-search"},
        "evg": {
            "detection",
            "captioning",
            "visual-qa-anomaly",
            "visual-qa-person",
            "person-attribute-search",
        },
    }
    if workflow not in allowed or stage not in allowed[workflow]:
        raise PaidfNativeError(f"unsupported {workflow!r} auto-label stage {stage!r}")
    if stage == "detection":
        _bind_detection_checkpoint()
    request_media_contract = (
        _evg_vqa_request_media_contract(stage)
        if stage in {"visual-qa-anomaly", "visual-qa-person"}
        else None
    )
    validation_kind = "iaa-postprocess" if workflow == "iaa" else "evg-validation"
    validation = _read_run_artifact(validation_uri, validation_kind, run_id, workflow)
    accepted = validation.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise PaidfNativeError("auto-labeling requires validated media")
    _configure_multistorage(
        validation_uri,
        auto_label_root_uri,
        result_uri,
        *(str(value) for item in accepted for value in item.values()),
    )
    component_env = _token_factory_child_env()
    kinds = _lineage_kinds(workflow)
    current_kind = f"{workflow}-auto-label-{stage}"
    expected_kinds = kinds[:kinds.index(current_kind)]
    previous = validation
    previous_uri = validation_uri
    if workflow == "evg" and stage != "detection":
        if not previous_result_uri:
            raise PaidfNativeError("EVG labeling requires its preceding producer result")
        previous_uri = previous_result_uri
        previous = _read_run_artifact(previous_uri, expected_kinds[-1], run_id, workflow)
    elif previous_result_uri and previous_result_uri != validation_uri:
        raise PaidfNativeError("first labeling stage must consume its validation producer")
    producers = [*previous.get("producers", []), _producer_descriptor(previous_uri, previous)]
    documents = _verified_producers(producers, expected_kinds, run_id, workflow)
    validation_index = expected_kinds.index(validation_kind)
    if producers[validation_index] != _producer_descriptor(validation_uri, validation):
        raise PaidfNativeError("labeling producer consumed a different validation artifact")
    _verify_label_handoffs(documents, accepted, auto_label_root_uri)

    with tempfile.TemporaryDirectory(prefix="npa-paidf-label-") as tmp:
        root = Path(tmp)
        upstream = _runtime_fetch(
            "https://github.com/NVIDIA/paidf-orchestration.git",
            PAIDF_ORCHESTRATION_REVISION,
            root / "orchestration",
        )
        evg_assets = (
            upstream / "airflow/dags/workflows/event_video_generation_dag/configs"
        )
        iaa_assets = (
            upstream / "airflow/dags/workflows/image_attribute_augmentation_dag/configs"
        )
        completed: list[dict[str, Any]] = []
        for item in accepted:
            key = f"{item['input_key']}_aug{item['augmentation_index']}"
            data_path = (
                f"{auto_label_root_uri.rstrip('/')}/{item['input_key']}/"
                f"{item['augmentation_index']}"
            )
            input_json = json.dumps(
                [{"media_path": item["media_uri"], "data_path": data_path}],
                separators=(",", ":"),
            )
            # Keep the pinned vendor service environment separate from the NPA
            # orchestration interpreter bootstrapped by SkyPilot. All reviewed
            # paidf-auto-labeling images install this console entrypoint here.
            args = ["/app/.venv/bin/main", "--input", input_json]
            if stage == "detection":
                args.extend(
                    [
                        "--tracker",
                        "rfdetr-boosttrack",
                        "--classes",
                        "person",
                        "--threshold",
                        "0.5",
                        "--extract-crops",
                        "--crop-classes",
                        "person",
                        "--crops-per-track",
                        "16",
                        "--crop-padding",
                        "0.1",
                        "--min-crop-size",
                        "48",
                        "--allow-model-download",
                    ]
                )
            elif stage == "captioning":
                args.extend(
                    [
                        "--input-source",
                        "original",
                        "--window-seconds",
                        "4.0",
                        "--window-frames",
                        "0",
                        "--remainder-threshold",
                        "0",
                        "--sampling-fps",
                        "2.0",
                        "--max-frames",
                        "24",
                        "--resolution",
                        "768",
                        "--vlm-provider",
                        "openai-compatible",
                        "--vlm-endpoint-url",
                        vlm_url,
                        "--vlm-model",
                        vlm_model,
                    ]
                )
            elif stage in {"visual-qa-anomaly", "visual-qa-person"}:
                person = stage == "visual-qa-person"
                asset = evg_assets / (
                    "question_bank.person_attributes.json"
                    if person
                    else "question_bank.anomaly_tags.json"
                )
                question_uri = _asset_to_uri(
                    asset, f"{data_path}/sidecars/assets/{asset.name}"
                )
                args.extend(
                    [
                        "--generation-mode",
                        "window-direct-vlm",
                        "--question-bank-file",
                        question_uri,
                    ]
                )
                if person:
                    args.extend(
                        [
                            "--track-crops-sidecar",
                            "detection_and_tracking/tracks.json",
                            "--max-crops-per-track",
                            str(request_media_contract["value"]),
                            "--resolution",
                            "896",
                            "--raw-windows-sidecar",
                            "visual_qa_per_track/windows.json",
                            "--output-items-sidecar",
                            "visual_qa_per_track/items.json",
                            "--output-windows-sidecar",
                            "visual_qa_per_track/windows.normalized.json",
                            "--state-artifacts-key",
                            "visual_qa_per_track",
                        ]
                    )
                else:
                    args.extend(
                        [
                            "--input-source",
                            "original",
                            "--single-window",
                            "--max-frames",
                            str(request_media_contract["value"]),
                            "--sampling-fps",
                            "3.0",
                            "--resolution",
                            "768",
                            "--raw-windows-sidecar",
                            "visual_qa_anomaly/windows.json",
                            "--output-items-sidecar",
                            "visual_qa_anomaly/items.json",
                            "--output-windows-sidecar",
                            "visual_qa_anomaly/windows.normalized.json",
                            "--state-artifacts-key",
                            "visual_qa_anomaly",
                        ]
                    )
                args.extend(
                    [
                        "--temperature",
                        "0",
                        "--max-tokens",
                        "4096",
                        "--no-flat-qa-tasks",
                        "--vlm-provider",
                        "openai-compatible",
                        "--vlm-endpoint-url",
                        vlm_url,
                        "--vlm-model",
                        vlm_model,
                    ]
                )
            else:
                config_asset = (
                    iaa_assets / "event_and_person_attribute_search_config.yaml"
                    if workflow == "iaa"
                    else evg_assets / "person_attribute_search_config.yaml"
                )
                if workflow == "iaa":
                    attribute_uri = str(item.get("postprocess_dataset_uri") or "")
                    if not attribute_uri or not _uri_is_file(attribute_uri):
                        raise PaidfNativeError(
                            "IAA attribute search requires the postprocessed attribute dataset"
                        )
                    assets_uri = f"{data_path}/sidecars/person_attribute_search/assets"
                    prompt_asset = (
                        iaa_assets
                        / "image_attribute_augmentation_synonymous_query_prompt.json"
                    )
                    prompt_uri = _asset_to_uri(
                        prompt_asset, f"{assets_uri}/{prompt_asset.name}"
                    )
                    remote_config = yaml.safe_load(
                        config_asset.read_text(encoding="utf-8")
                    )
                    if not isinstance(remote_config, dict):
                        raise PaidfNativeError(
                            "IAA attribute-search config is not an object"
                        )
                    remote_config.update(
                        attribute_json=attribute_uri,
                        query_prompt_file=prompt_uri,
                        llm_endpoint_url=llm_url,
                        llm_model=llm_model,
                    )
                    local_config = root / f"{key}-attribute-search.yaml"
                    local_config.write_text(
                        yaml.safe_dump(remote_config, sort_keys=False), encoding="utf-8"
                    )
                    config_uri = _asset_to_uri(
                        local_config, f"{assets_uri}/{config_asset.name}"
                    )
                else:
                    config_uri = _asset_to_uri(
                        config_asset, f"{data_path}/sidecars/assets/{config_asset.name}"
                    )
                args.extend(
                    [
                        "--config-file",
                        config_uri,
                        "--llm-provider",
                        "openai-compatible",
                        "--llm-endpoint-url",
                        llm_url,
                        "--llm-model",
                        llm_model,
                    ]
                )
                if workflow == "iaa":
                    args.extend(["--attribute-json", attribute_uri])
            _run_component(args, env=component_env)
            required: tuple[str, ...]
            if stage == "detection":
                required = ("contextual/objects.json", "contextual/instances.json")
            elif stage == "captioning":
                required = ("sidecars/captioning/video_captions.json",)
            elif stage == "visual-qa-anomaly":
                required = ("sidecars/visual_qa_anomaly/items.json",)
            elif stage == "visual-qa-person":
                required = (
                    "sidecars/visual_qa_per_track/items.json",
                    "sidecars/visual_qa_per_track/windows.normalized.json",
                )
            elif workflow == "iaa":
                required = IAA_LABEL_ARTIFACTS
            elif _uri_is_file(
                f"{data_path}/sidecars/detection_and_tracking/tracks.json"
            ):
                required = (
                    "sidecars/person_attribute_search/pas.json",
                    "sidecars/person_attribute_search/chunk_queries.json",
                    "sidecars/person_attribute_search/pas_anomaly.json",
                    "contextual/person_attributes.json",
                    "contextual/pas_queries.json",
                )
            else:
                required = ()
            missing = [
                relative
                for relative in required
                if not _uri_is_file(f"{data_path}/{relative}")
            ]
            if missing:
                raise PaidfNativeError(
                    f"{stage} omitted {len(missing)} required published sidecar(s)"
                )
            if workflow == "iaa":
                _validate_iaa_labels(data_path)
            content_paths = list(required)
            if stage == "detection" and _uri_is_file(f"{data_path}/sidecars/detection_and_tracking/tracks.json"):
                content_paths.append("sidecars/detection_and_tracking/tracks.json")
            completed.append(
                {
                    "artifacts": _artifact_fingerprints({relative: f"{data_path}/{relative}" for relative in content_paths}) if content_paths else {},
                    "key": key,
                    "data_path": data_path,
                    "media_uri": item["media_uri"],
                    "required_artifacts": list(required),
                    "trackless": stage == "person-attribute-search" and not required,
                }
            )
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{workflow}-auto-label-{stage}.v1",
        "run_id": run_id,
        "workflow": workflow,
        "stage": stage,
        "component": "NVIDIA paidf-auto-labeling 1.1.0",
        "component_executable": "/app/.venv/bin/main",
        "upstream_revision": PAIDF_AUTO_LABELING_REVISION,
        "count": len(completed),
        "outputs": completed,
        "producers": producers,
        "validation_uri": validation_uri,
    }
    if stage == "detection":
        payload["detection_checkpoint"] = {
            "url": RFDETR_BASE_URL,
            "sha256": RFDETR_BASE_SHA256,
            "verification": "upstream streaming SHA-256 verifier",
        }
    if request_media_contract is not None:
        payload["request_media_contract"] = request_media_contract
    return _write_json(payload, result_uri)


def finalize_dataset(
    workflow: str,
    validation_uri: str,
    upstream_uri: str,
    labels_uri: str,
    output_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Assemble the upstream DAFT scene layout and publish its NPA lineage index."""

    if workflow not in {"iaa", "evg"}:
        raise PaidfNativeError("dataset assembly requires a supported workflow")
    validation_kind = "iaa-postprocess" if workflow == "iaa" else "evg-validation"
    validation = _read_run_artifact(validation_uri, validation_kind, run_id, workflow)
    upstream = _read_json(upstream_uri)
    _require_upstream_identity(upstream, workflow, run_id)
    labels = _read_run_artifact(
        labels_uri, f"{workflow}-auto-label-person-attribute-search", run_id, workflow
    )
    accepted = validation.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise PaidfNativeError("cannot finalize a dataset with no accepted media")
    producers = [*labels.get("producers", []), _producer_descriptor(labels_uri, labels)]
    kinds = _lineage_kinds(workflow)
    documents = _verified_producers(producers, kinds, run_id, workflow)
    if producers[kinds.index(validation_kind)] != _producer_descriptor(validation_uri, validation):
        raise PaidfNativeError("final labeling producer consumed a different validation artifact")
    _verify_label_handoffs(documents, accepted)
    label_outputs = labels.get("outputs")
    if not isinstance(label_outputs, list):
        raise PaidfNativeError("labeling result has no output records")
    label_by_key = {str(item.get("key")): item for item in label_outputs}
    entries = []
    assembled_file_count = 0
    validated_artifact_count = 0
    trackless_count = 0
    for item in accepted:
        key = f"{item['input_key']}_aug{item['augmentation_index']}"
        label = label_by_key.get(key)
        if not isinstance(label, dict) or not str(label.get("data_path") or ""):
            raise PaidfNativeError(f"final dataset has no labeling output for {key}")
        data_path = str(label["data_path"])
        if workflow == "evg":
            required = (
                "contextual/objects.json",
                "contextual/instances.json",
                "sidecars/captioning/video_captions.json",
                "sidecars/visual_qa_anomaly/items.json",
                "sidecars/visual_qa_per_track/items.json",
                "sidecars/visual_qa_per_track/windows.normalized.json",
            )
            missing = [
                relative
                for relative in required
                if not _uri_is_file(f"{data_path}/{relative}")
            ]
            tracks = _uri_is_file(
                f"{data_path}/sidecars/detection_and_tracking/tracks.json"
            )
            if tracks:
                track_required = (
                    "sidecars/person_attribute_search/pas.json",
                    "sidecars/person_attribute_search/chunk_queries.json",
                    "sidecars/person_attribute_search/pas_anomaly.json",
                    "contextual/person_attributes.json",
                    "contextual/pas_queries.json",
                )
                missing.extend(
                    relative
                    for relative in track_required
                    if not _uri_is_file(f"{data_path}/{relative}")
                )
                validated_artifact_count += 1 + len(track_required)
            else:
                trackless_count += 1
            if missing:
                raise PaidfNativeError(
                    f"EVG terminal validation found {len(missing)} missing "
                    f"published sidecar(s) for {key}"
                )
            validated_artifact_count += len(required)
        else:
            _validate_iaa_labels(data_path)
            validated_artifact_count += len(IAA_LABEL_ARTIFACTS)
        scene_path = f"{output_uri.rsplit('/', 1)[0]}/{key}"
        with tempfile.TemporaryDirectory(prefix="npa-paidf-scene-") as tmp:
            scene = _materialize(data_path, Path(tmp) / "scene")
            if not scene.is_dir() or not any(
                path.is_file() for path in scene.rglob("*")
            ):
                raise PaidfNativeError(
                    "the auto-labeling scene contains no files to assemble"
                )
            cosmos = scene / "sidecars/cosmos"
            raw = scene / "raw"
            raw.mkdir(parents=True, exist_ok=True)
            cosmos.mkdir(parents=True, exist_ok=True)
            if workflow == "evg":
                local_media = _materialize(item["media_uri"], raw / "video.mp4")
            else:
                # The producer may emit multiple views; the scene owns all of
                # them, matching upstream copy_cosmos_augmented_images.
                _materialize(item["media_uri"].rsplit("/", 1)[0], raw)
                local_media = raw / Path(urlparse(item["media_uri"]).path).name
                _materialize(
                    item["postprocess_dataset_uri"],
                    scene / "sidecars/augmented_data.json",
                )
            if (
                not local_media.is_file()
                or local_media.stat().st_size != item["size_bytes"]
                or _sha256(local_media) != item["sha256"]
            ):
                raise PaidfNativeError(
                    "assembled media does not match its validated digest"
                )
            config_uri = str(item.get("config_uri") or "")
            if not config_uri:
                raise PaidfNativeError(
                    "dataset assembly requires the executed generation config"
                )
            _materialize(config_uri, cosmos / "config.yaml")
            _materialize(item["caption_uri"], cosmos / "caption.txt")
            _materialize(item["metadata_uri"], cosmos / "metadata.json")
            scene_artifacts = sorted(
                str(path.relative_to(scene))
                for path in scene.rglob("*")
                if path.is_file()
            )
            scene_content = {
                relative: {
                    "uri": f"{scene_path}/{relative}",
                    "sha256": _sha256(scene / relative),
                    "size_bytes": (scene / relative).stat().st_size,
                }
                for relative in scene_artifacts
            }
            assembled_file_count += len(scene_artifacts)
            _publish(scene, scene_path)
            media_relative = str(local_media.relative_to(scene))
        paths = {
            "config": f"{scene_path}/sidecars/cosmos/config.yaml",
            "video": f"{scene_path}/{media_relative}",
            "caption": f"{scene_path}/sidecars/cosmos/caption.txt",
            "metadata": f"{scene_path}/sidecars/cosmos/metadata.json",
            "raw": f"{scene_path}/raw",
            "contextual": f"{scene_path}/contextual",
            "sidecars": f"{scene_path}/sidecars",
        }
        entry = {
            "key": key,
            "scene_id": key,
            "input_key": item["input_key"],
            "augmentation_index": item["augmentation_index"],
            "scene_path": scene_path,
            "auto_labeling_source_path": data_path,
            "paths": paths,
            "media": paths["video"],
            "caption": paths["caption"],
            "metadata": paths["metadata"],
            "variables": item["variables"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
            "labels": scene_path,
            "scene_artifacts": scene_artifacts,
            "scene_content": scene_content,
        }
        if workflow == "iaa":
            config_relative = "sidecars/person_attribute_search/assets/event_and_person_attribute_search_config.yaml"
            if config_relative not in scene_artifacts:
                raise PaidfNativeError(
                    "IAA assembled scene lacks the executed attribute-search config"
                )
            entry.update(
                person_key=key,
                person_id=item["input_key"],
                source_person_key=item["input_key"],
                augmentation_id=f"aug_{item['augmentation_index']}",
                selected_attributes=item["selected_attributes"],
                queries=item["queries"],
                attribute_verification=item["attribute_verification"],
            )
            paths["config"] = f"{scene_path}/{config_relative}"
            paths["task"] = f"{scene_path}/task"
            entry["postprocess_dataset"] = f"{scene_path}/sidecars/augmented_data.json"
            entry["postprocess_source"] = item["postprocess_dataset_uri"]
            entry["generation_media"] = item["generation_media_uri"]
            entry["generation_sha256"] = item["generation_sha256"]
        entries.append(entry)
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{workflow}-dataset.v1",
        "run_id": run_id,
        "workflow": workflow,
        "status": "completed",
        "entry_count": len(entries),
        "assembled_file_count": assembled_file_count,
        "validated_artifact_count": validated_artifact_count,
        "trackless_scene_count": trackless_count,
        "entries": entries,
        "metadata": {
            "description": "Image Attribute Augmentation dataset"
            if workflow == "iaa"
            else "Event Video Generation anomaly dataset",
            "total_scenes": len(entries),
            "total_ids": len(entries),
            "original_ids": len({entry["input_key"] for entry in entries}),
            "original_inputs": len({entry["input_key"] for entry in entries}),
            "skipped_augmentations": validation.get("skipped", []),
        },
        "upstream": upstream,
        "lineage": {
            "validation_uri": validation_uri,
            "labels_uri": labels_uri,
            "upstream_uri": upstream_uri,
            "producers": producers,
        },
    }
    if workflow == "iaa":
        upstream_dataset_uri = f"{output_uri.rsplit('/', 1)[0]}/augmented_data.json"
        payload["upstream_dataset_uri"] = upstream_dataset_uri
        _write_json(payload, upstream_dataset_uri)
    return _write_json(payload, output_uri)


def validate_dataset(dataset_uri: str, report_uri: str, run_id: str) -> dict[str, Any]:
    """Fail closed on the published native dataset and emit a terminal decision."""

    dataset = _read_json(dataset_uri)
    workflow = dataset.get("workflow")
    if workflow not in {"iaa", "evg"}:
        raise PaidfNativeError("terminal dataset has no supported workflow")
    _require_artifact_identity(
        dataset, f"{SCHEMA_PREFIX}.{workflow}-dataset.v1", run_id, workflow
    )
    upstream = dataset.get("upstream")
    if not isinstance(upstream, dict):
        raise PaidfNativeError("terminal dataset has no upstream provenance")
    _require_upstream_identity(upstream, workflow, run_id)
    entries = dataset.get("entries")
    if (
        dataset.get("status") != "completed"
        or not isinstance(entries, list)
        or not entries
        or dataset.get("entry_count") != len(entries)
    ):
        raise PaidfNativeError("terminal dataset manifest is incomplete")
    lineage = dataset.get("lineage")
    if not isinstance(lineage, dict):
        raise PaidfNativeError("terminal dataset has no producer lineage")
    documents = _verified_producers(lineage.get("producers"), _lineage_kinds(workflow), run_id, workflow)
    validation_kind = "iaa-postprocess" if workflow == "iaa" else "evg-validation"
    validated = documents[_lineage_kinds(workflow).index(validation_kind)]
    _verify_label_handoffs(documents, validated["accepted"])
    missing = []
    for entry in entries:
        records = entry.get("scene_content")
        if not isinstance(records, dict) or set(records) != set(entry.get("scene_artifacts", [])):
            raise PaidfNativeError("assembled scene has no complete content manifest")
        if any(record.get("uri") != f"{entry['scene_path']}/{name}" for name, record in records.items()):
            raise PaidfNativeError("assembled scene content manifest names another scene")
        _verify_fingerprints(records)
        for field in ("media", "caption", "metadata"):
            value = str(entry.get(field) or "")
            if not value or not _uri_is_file(value):
                missing.append(f"{entry.get('key', '<unknown>')}:{field}")
        labels = str(entry.get("labels") or "")
        if not labels or not _uri_prefix_has_objects(labels):
            missing.append(f"{entry.get('key', '<unknown>')}:labels")
        elif dataset.get("workflow") == "iaa":
            _validate_iaa_labels(labels)
            postprocess = str(entry.get("postprocess_dataset") or "")
            if not postprocess or not _uri_is_file(postprocess):
                missing.append(f"{entry.get('key', '<unknown>')}:postprocess_dataset")
        for relative in entry.get("scene_artifacts", []):
            if not _uri_is_file(f"{entry['scene_path']}/{relative}"):
                missing.append(f"{entry.get('key', '<unknown>')}:scene_artifact")
    if missing:
        raise PaidfNativeError(
            f"terminal dataset validation found {len(missing)} missing artifact(s)"
        )
    encoded = json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{dataset['workflow']}-terminal-validation.v1",
        "run_id": run_id,
        "workflow": workflow,
        "status": "passed",
        "dataset_uri": dataset_uri,
        "dataset_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "entry_count": len(entries),
        "assembled_file_count": dataset.get("assembled_file_count", 0),
        "validated_artifact_count": dataset.get("validated_artifact_count", 0),
        "trackless_scene_count": dataset.get("trackless_scene_count", 0),
    }
    return _write_json(payload, report_uri)


def _dig_model_revisions() -> dict[str, str]:
    """Use the same exact revisions that the required access preflight checked."""

    from npa.workbench.model_access import HF, assets_for

    revisions = {
        asset.repo: asset.revision
        for asset in assets_for(["paidf-dig"])
        if asset.provider == HF
    }
    if not revisions or any(
        not re.fullmatch(r"[0-9a-f]{40}", revision) for revision in revisions.values()
    ):
        raise PaidfNativeError(
            "DIG dependencies require exact access-preflight revisions"
        )
    return revisions


def _dig_cache_manifest(
    pretrained: Path, run_id: str, *, initialize: bool = False
) -> dict[str, Any]:
    """Bind upstream's unpinned runtime loads to the approved cached snapshots."""

    if not run_id.strip():
        raise PaidfNativeError("DIG runtime cache requires the workflow run identity")
    revisions = _dig_model_revisions()
    snapshots = []
    for repository, probe in DIG_RUNTIME_CACHE_PROBES.items():
        revision = revisions[repository]
        cache = pretrained / "hf" / f"models--{repository.replace('/', '--')}"
        snapshot = cache / "snapshots" / revision
        if not (snapshot / probe).exists():
            raise PaidfNativeError(
                f"DIG pinned runtime cache is incomplete for {repository}"
            )
        reference = cache / "refs/main"
        if initialize:
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_text(revision, encoding="utf-8")
        if (
            not reference.is_file()
            or reference.read_text(encoding="utf-8").strip() != revision
        ):
            raise PaidfNativeError(
                f"DIG runtime cache revision disagrees with preflight for {repository}"
            )
        snapshots.append({"repository": repository, "revision": revision})
    manifest = {
        "schema": f"{SCHEMA_PREFIX}.dig-runtime-cache.v1",
        "run_id": run_id,
        "workflow": "dig",
        "models": snapshots,
    }
    recorded = pretrained / "runtime-hf-snapshots.json"
    if initialize:
        recorded.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    elif (
        not recorded.is_file()
        or json.loads(recorded.read_text(encoding="utf-8")) != manifest
    ):
        raise PaidfNativeError(
            "DIG runtime cache provenance does not match approved revisions"
        )
    return manifest


def _dig_pretrained_content_manifest(
    pretrained: Path, run_id: str, *, initialize: bool = False
) -> tuple[dict[str, Any], str]:
    """Create or verify the complete published pretrained byte closure."""

    if not run_id.strip():
        raise PaidfNativeError("DIG pretrained content requires the workflow run identity")
    if not pretrained.is_dir():
        raise PaidfNativeError("DIG pretrained content is not a directory")
    root = pretrained.resolve()
    records: list[dict[str, Any]] = []
    for item in sorted(pretrained.rglob("*"), key=lambda path: path.as_posix()):
        relative = item.relative_to(pretrained).as_posix()
        if relative == DIG_PRETRAINED_CONTENT_MANIFEST or not item.is_file():
            continue
        resolved = item.resolve()
        if not resolved.is_relative_to(root):
            raise PaidfNativeError(
                f"DIG pretrained content escapes its published tree: {relative}"
            )
        records.append(
            {
                "path": relative,
                "size_bytes": item.stat().st_size,
                "sha256": _sha256(item),
            }
        )
    if not records:
        raise PaidfNativeError("DIG pretrained content manifest would be empty")
    payload = {
        "schema": f"{SCHEMA_PREFIX}.dig-pretrained-content.v1",
        "run_id": run_id,
        "workflow": "dig",
        "file_count": len(records),
        "total_bytes": sum(record["size_bytes"] for record in records),
        "files": records,
    }
    recorded = pretrained / DIG_PRETRAINED_CONTENT_MANIFEST
    if initialize:
        recorded.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
    else:
        if not recorded.is_file():
            raise PaidfNativeError("DIG pretrained content manifest is missing")
        try:
            expected = json.loads(recorded.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise PaidfNativeError("DIG pretrained content manifest is unreadable") from exc
        _require_artifact_identity(
            expected,
            f"{SCHEMA_PREFIX}.dig-pretrained-content.v1",
            run_id,
            "dig",
        )
        if expected != payload:
            raise PaidfNativeError(
                "DIG pretrained content does not match its complete hash manifest"
            )
    return payload, _sha256(recorded)


def _dig_offline_environment(pretrained: Path, run_id: str) -> dict[str, str]:
    if not run_id.strip():
        raise PaidfNativeError("DIG runtime cache requires the workflow run identity")
    _dig_cache_manifest(pretrained, run_id)
    environment = {
        **_dig_vendor_environment(),
        "CKPT_DIR": str(pretrained),
        "HF_HUB_CACHE": str(pretrained / "hf"),
        "HUGGINGFACE_HUB_CACHE": str(pretrained / "hf"),
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
    }
    for name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HF_TOKEN_PATH"):
        environment.pop(name, None)
    return environment


def _dig_vendor_environment() -> dict[str, str]:
    """Run AnomalyGen children in its pinned environment, outside NPA's venv."""

    env = dict(os.environ)
    excluded = {"/tmp/npa-shim", "/opt/npa-venv/bin"}
    inherited = [
        entry
        for entry in env.get("PATH", "").split(os.pathsep)
        if entry and entry not in excluded and entry != "/opt/venv/bin"
    ]
    env["PATH"] = os.pathsep.join(["/opt/venv/bin", *inherited])
    env["VIRTUAL_ENV"] = "/opt/venv"
    env["UV_PYTHON"] = "/opt/venv/bin/python"
    # The pinned preflight script reads PYTHON; torchrun reads PYTHON_EXEC for
    # its actual workers independently of the interpreter that starts torchrun.
    env["PYTHON"] = "/opt/venv/bin/python"
    env["PYTHON_EXEC"] = "/opt/venv/bin/python"
    return env


def run_dig_train(
    dataset_uri: str,
    pretrained_uri: str,
    output_uri: str,
    result_uri: str,
    usecase: str,
    run_id: str,
) -> dict[str, Any]:
    """Run the upstream default Day-1 AnomalyGen fine-tuning task."""

    if usecase not in {"pcb", "metal_surface", "glass"}:
        raise PaidfNativeError("DIG usecase must be pcb, metal_surface, or glass")
    workspace = Path("/workspace/paidf-anomalygen")
    if not workspace.is_dir():
        raise PaidfNativeError(
            "the selected image is not NVIDIA paidf-anomalygen 1.1.0"
        )
    with tempfile.TemporaryDirectory(prefix="npa-paidf-dig-train-") as tmp:
        root = Path(tmp)
        dataset = _materialize(dataset_uri, root / "dataset")
        pretrained = _materialize(pretrained_uri, root / "pretrained")
        _, pretrained_manifest_sha256 = _dig_pretrained_content_manifest(
            pretrained, run_id
        )
        source = _runtime_fetch(
            "https://github.com/NVIDIA/physical-ai-data-factory.git",
            PHYSICAL_AI_DATA_FACTORY_REVISION,
            root / "physical-ai-data-factory",
        )
        skill = source / "skills/physical-ai-defect-image-generation-v1-1"
        script = skill / "scripts/anomalygen_train.sh"
        recipe = skill / f"assets/cookbooks/{usecase}/ag_config.yaml"
        train_output = root / "finetune"
        env = {
            **_dig_offline_environment(pretrained, run_id),
            "PRETRAINED_SRC": str(pretrained),
            "DATASET_DIR": str(dataset),
            "RECIPE_TEMPLATE": str(recipe),
            "TRAIN_OUTPUT": str(train_output),
            "NUM_GPUS": "1",
        }
        _run_component(["bash", str(script)], env=env)
        best = list(train_output.rglob("best_checkpoint.txt"))
        if len(best) != 1:
            raise PaidfNativeError(
                "AnomalyGen fine-tuning did not produce one best checkpoint"
            )
        selected = (
            best[0].parent / "model" / best[0].read_text(encoding="utf-8").strip()
        )
        selected_resolved = selected.resolve()
        if (
            not selected_resolved.is_relative_to(train_output.resolve())
            or not selected_resolved.is_file()
        ):
            raise PaidfNativeError("AnomalyGen best-checkpoint pointer is invalid")
        payload = {
            "schema": f"{SCHEMA_PREFIX}.dig-finetune.v1",
            "run_id": run_id,
            "workflow": "dig",
            "status": "completed",
            "component": "NVIDIA paidf-anomalygen 1.1.0",
            "upstream_workflow_revision": PHYSICAL_AI_DATA_FACTORY_REVISION,
            "selected_checkpoint": selected_resolved.relative_to(
                train_output.resolve()
            ).as_posix(),
            "selected_checkpoint_sha256": _sha256(selected_resolved),
            "pretrained_content_manifest_sha256": pretrained_manifest_sha256,
            "output_uri": output_uri,
        }
        _write_json(payload, str(train_output / "npa-finetune.json"))
        _publish(train_output, output_uri)
    return _write_json(payload, result_uri)


def prepare_dig_pretrained(
    output_uri: str,
    result_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Fetch AnomalyGen's gated base checkpoints at runtime under operator access."""

    workspace = Path("/workspace/paidf-anomalygen")
    script = workspace / "scripts/download_checkpoints.sh"
    if not script.is_file():
        raise PaidfNativeError("the selected image lacks AnomalyGen checkpoint setup")
    with tempfile.TemporaryDirectory(prefix="npa-paidf-dig-pretrained-") as tmp:
        output = Path(tmp) / "pretrained"
        converted_manifest = workspace / "assets/checkpoint_manifest_converted.sha256"
        revisions = _dig_model_revisions()
        env = {
            **_dig_vendor_environment(),
            "CKPT_DIR": str(output),
            "HF_HUB_OFFLINE": "0",
            "TRANSFORMERS_OFFLINE": "0",
            "HF_HUB_CACHE": str(output / "hf"),
            "HF_CLI_VERSION": "1.26.0",
        }
        # The pinned framework requests complete snapshots during model
        # construction, including the Qwen tokenizer and Edge processor paths.
        # Stage all files before credential-free, offline conversion and use.
        for repository, probe in DIG_RUNTIME_CACHE_PROBES.items():
            command = [
                "uvx",
                "hf==1.26.0",
                "download",
                repository,
                "--revision",
                revisions[repository],
                "--cache-dir",
                str(output / "hf"),
            ]
            # Both converters construct the vision tokenizer before the
            # original downloader runs. Its pinned HF lookup needs this file
            # in the shared cache, not only the later wan2pt2 handoff directory.
            if repository == "Wan-AI/Wan2.2-TI2V-5B":
                command.insert(4, probe)
            _run_component(command, env=env)
        _dig_cache_manifest(output, run_id, initialize=True)
        # The pinned upstream converter's named models resolve revision=main.
        # Fetch approved revisions first and give that real converter local
        # snapshots; the original downloader then sees completed DCP outputs.
        for model in ("Cosmos3-Nano", "Cosmos3-Edge"):
            repository = f"nvidia/{model}"
            source = (
                output / repository
                if model == "Cosmos3-Nano"
                else Path(tmp) / "edge-source"
            )
            _run_component(
                [
                    "uvx",
                    "hf==1.26.0",
                    "download",
                    repository,
                    "--revision",
                    revisions[repository],
                    "--local-dir",
                    str(source),
                ],
                env=env,
            )
            _run_component(
                [
                    "python",
                    "-m",
                    "cosmos_framework.scripts.convert_model_to_dcp",
                    "-o",
                    str(output / model),
                    "--checkpoint-path",
                    str(source),
                ],
                cwd=workspace,
                env=_dig_offline_environment(output, run_id),
            )
        for prefix, repository in {
            "DINOV2": "facebook/dinov2-large",
            "CRADIO": "nvidia/C-RADIOv3-B",
            "WAN_VAE": "Wan-AI/Wan2.2-TI2V-5B",
            "SAM2": "facebook/sam2.1-hiera-large",
            "COSMOS_VLM": "nvidia/Cosmos3-Nano",
            "GUARDRAIL": "nvidia/Cosmos-Guardrail1",
            "QWEN_GUARD": "Qwen/Qwen3Guard-Gen-0.6B",
            "QWEN_VLM": "Qwen/Qwen3-VL-8B-Instruct",
            "EDGE_VLM": "nvidia/Cosmos3-Edge",
        }.items():
            env[f"{prefix}_REPO"] = repository
            env[f"{prefix}_REV"] = revisions[repository]
        env.update(
            CRADIO_FILE="model.safetensors",
            WAN_VAE_FILE="Wan2.2_VAE.pth",
            SAM2_FILE="sam2.1_hiera_large.pt",
        )
        _run_component(["bash", str(script)], cwd=workspace, env=env)
        required = [
            output / "Cosmos3-Nano",
            output / "Cosmos3-Edge",
            output / "wan2pt2/Wan2.2_VAE.pth",
        ]
        if (
            any(not path.exists() for path in required)
            or not converted_manifest.is_file()
        ):
            raise PaidfNativeError("AnomalyGen base-checkpoint setup is incomplete")
        shutil.copy2(converted_manifest, output / converted_manifest.name)
        _dig_cache_manifest(output, run_id, initialize=True)
        content, content_manifest_sha256 = _dig_pretrained_content_manifest(
            output, run_id, initialize=True
        )
        _publish(output, output_uri)
        payload = {
            "schema": f"{SCHEMA_PREFIX}.dig-pretrained.v1",
            "run_id": run_id,
            "workflow": "dig",
            "status": "completed",
            "component": "NVIDIA paidf-anomalygen 1.1.0 checkpoint setup",
            "file_count": content["file_count"],
            "total_bytes": content["total_bytes"],
            "manifest_sha256": _sha256(output / converted_manifest.name),
            "content_manifest_sha256": content_manifest_sha256,
            "content_manifest": DIG_PRETRAINED_CONTENT_MANIFEST,
            "output_uri": output_uri,
        }
    return _write_json(payload, result_uri)


def _verify_dig_finetune_handoff(
    checkpoint: Path,
    checkpoint_uri: str,
    finetune_result_uri: str,
    run_id: str,
) -> tuple[dict[str, Any], Path]:
    """Bind inference to the exact best checkpoint selected by this run."""

    finetune_result = _read_run_artifact(
        finetune_result_uri, "dig-finetune", run_id, "dig"
    )
    checkpoint_record = _read_run_artifact(
        str(checkpoint / "npa-finetune.json"), "dig-finetune", run_id, "dig"
    )
    if checkpoint_record != finetune_result:
        raise PaidfNativeError(
            "DIG embedded checkpoint record disagrees with the finetune result"
        )
    if (
        checkpoint_record.get("status") != "completed"
        or str(checkpoint_record.get("output_uri") or "").rstrip("/")
        != checkpoint_uri.rstrip("/")
    ):
        raise PaidfNativeError("DIG checkpoint handoff is not completed")
    selected_value = str(checkpoint_record.get("selected_checkpoint") or "")
    selected_relative = PurePosixPath(selected_value)
    if (
        not selected_value
        or selected_relative.is_absolute()
        or ".." in selected_relative.parts
    ):
        raise PaidfNativeError("DIG selected-checkpoint identity is invalid")
    selected = (checkpoint / Path(*selected_relative.parts)).resolve()
    if not selected.is_relative_to(checkpoint.resolve()) or not selected.is_file():
        raise PaidfNativeError("DIG selected checkpoint is missing from the handoff")
    selected_sha256 = str(
        checkpoint_record.get("selected_checkpoint_sha256") or ""
    ).lower()
    if (
        not re.fullmatch(r"[0-9a-f]{64}", selected_sha256)
        or _sha256(selected) != selected_sha256
    ):
        raise PaidfNativeError("DIG selected checkpoint content hash does not match")
    best = list(checkpoint.rglob("best_checkpoint.txt"))
    if len(best) != 1:
        raise PaidfNativeError("DIG checkpoint handoff has no unique best pointer")
    best_target = (
        best[0].parent / "model" / best[0].read_text(encoding="utf-8").strip()
    ).resolve()
    if best_target != selected:
        raise PaidfNativeError(
            "DIG best-checkpoint pointer disagrees with the selected checkpoint"
        )
    return checkpoint_record, selected


def run_dig_inference(
    dataset_uri: str,
    pretrained_uri: str,
    checkpoint_uri: str,
    finetune_result_uri: str,
    output_uri: str,
    result_uri: str,
    num_sdg: int,
    run_id: str,
) -> dict[str, Any]:
    """Run AnomalyGen's published Day-1 manual-ROI inference and native labels."""

    from npa.workflows.paidf_dig_guardrails import (
        DIG_IMPORT_PROBE,
        DIG_QWEN_PATCHED_SHA256,
        DIG_QWEN_PATH,
        DIG_VENDOR_PYTHON,
        dig_guardrail_runtime,
        prepare_dig_guardrail_overlay,
        verify_dig_guardrail_overlay,
    )
    from npa.workflows.paidf_guardrails import PaidfGuardrailError

    workspace = Path("/workspace/paidf-anomalygen")
    if not workspace.is_dir():
        raise PaidfNativeError(
            "the selected image is not NVIDIA paidf-anomalygen 1.1.0"
        )
    if num_sdg < 1:
        raise PaidfNativeError("invalid AnomalyGen inference settings")

    with tempfile.TemporaryDirectory(prefix="npa-paidf-dig-") as tmp:
        root = Path(tmp)
        dataset = _materialize(dataset_uri, root / "dataset")
        pretrained = _materialize(pretrained_uri, root / "pretrained")
        _, pretrained_manifest_sha256 = _dig_pretrained_content_manifest(
            pretrained, run_id
        )
        checkpoint = _materialize(checkpoint_uri, root / "checkpoint")
        checkpoint_record, selected = _verify_dig_finetune_handoff(
            checkpoint, checkpoint_uri, finetune_result_uri, run_id
        )
        if (
            checkpoint_record.get("pretrained_content_manifest_sha256")
            != pretrained_manifest_sha256
        ):
            raise PaidfNativeError(
                "DIG inference pretrained content differs from its finetune handoff"
            )
        selected_value = str(checkpoint_record["selected_checkpoint"])
        selected_sha256 = str(checkpoint_record["selected_checkpoint_sha256"])
        defect_specs = list(dataset.rglob("defect_spec.jsonl"))
        if len(defect_specs) != 1:
            raise PaidfNativeError("DIG requires exactly one defect_spec.jsonl")
        source = _runtime_fetch(
            "https://github.com/NVIDIA/physical-ai-data-factory.git",
            PHYSICAL_AI_DATA_FACTORY_REVISION,
            root / "physical-ai-data-factory",
        )
        script = (
            source
            / "skills/physical-ai-defect-image-generation-v1-1/scripts/anomalygen_generate.sh"
        )
        generated = root / "generated"
        env = {
            **_dig_offline_environment(pretrained, run_id),
            "PRETRAINED_SRC": str(pretrained),
            "DATASET_DIR": str(defect_specs[0].parent),
            "DEFECT_SPEC": str(defect_specs[0]),
            "FINETUNE_DIR": str(checkpoint),
            "OUTPUT_DIR": str(generated),
            "NUM_SDG": str(num_sdg),
            "NUM_GPUS": "1",
            "CHECKPOINT_STEP": "",
        }
        overlay = root / "guardrail-code"
        try:
            env, guardrail_source = prepare_dig_guardrail_overlay(overlay, env)
        except PaidfGuardrailError as exc:
            raise PaidfNativeError(str(exc)) from exc
        _run_component(
            [DIG_VENDOR_PYTHON, "-c", DIG_IMPORT_PROBE,
             str(overlay / DIG_QWEN_PATH), DIG_QWEN_PATCHED_SHA256],
            env=env,
        )
        _run_component(["bash", str(script)], env=env)
        images = sorted((generated / "reconstructed_image").glob("*"))
        labels = generated / "pseudo_labels/coco_annotations.json"
        if not images or not labels.is_file() or not labels.stat().st_size:
            raise PaidfNativeError(
                "AnomalyGen returned no generated images or label metadata"
            )
        try:
            verify_dig_guardrail_overlay(overlay, guardrail_source)
            guardrail_runtime = dig_guardrail_runtime(
                generated / "timing_summary.json", guardrail_source, len(images)
            )
        except PaidfGuardrailError as exc:
            raise PaidfNativeError(str(exc)) from exc
        _publish(generated, output_uri)
        payload = {
            "schema": f"{SCHEMA_PREFIX}.dig-result.v1",
            "run_id": run_id,
            "workflow": "dig",
            "status": "completed",
            "component": "NVIDIA paidf-anomalygen 1.1.0",
            "upstream_workflow_revision": PHYSICAL_AI_DATA_FACTORY_REVISION,
            "guardrail_runtime": guardrail_runtime,
            "pretrained_content_manifest_sha256": pretrained_manifest_sha256,
            "finetune_result_uri": finetune_result_uri,
            "selected_checkpoint": selected_value,
            "selected_checkpoint_sha256": selected_sha256,
            "image_count": len(images),
            "label_file_count": 1,
            "images": [
                {
                    "name": item.name,
                    "sha256": _sha256(item),
                    "size_bytes": item.stat().st_size,
                }
                for item in images
            ],
            "output_uri": output_uri,
        }
    return _write_json(payload, result_uri)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser("prepare-images")
    for name in ("input-uri", "output-uri", "manifest-uri", "run-id"):
        prepare.add_argument(f"--{name}", required=True)

    configs = subparsers.add_parser("build-configs")
    configs.add_argument("--workflow", choices=("iaa", "evg"), required=True)
    for name in (
        "prepared-manifest-uri",
        "output-uri",
        "config-manifest-uri",
        "vlm-url",
        "vlm-model",
        "llm-url",
        "llm-model",
        "generation-url",
        "generation-model",
        "run-id",
    ):
        configs.add_argument(f"--{name}", required=True)
    configs.add_argument("--num-augmentations", type=int, required=True)
    configs.add_argument("--seed", type=int, required=True)

    augment = subparsers.add_parser("run-augmentation")
    for name in ("config-manifest-uri", "result-uri", "run-id"):
        augment.add_argument(f"--{name}", required=True)

    local_augment = subparsers.add_parser("run-local-augmentation")
    for name in (
        "config-manifest-uri",
        "result-uri",
        "generation-model",
        "generation-revision",
        "run-id",
    ):
        local_augment.add_argument(f"--{name}", required=True)
    local_augment.add_argument(
        "--service-kind", choices=("image-edit", "image2video"), required=True
    )
    local_augment.add_argument("--port", type=int, default=8000)
    local_augment.add_argument("--parallel-size", type=int, choices=(1, 2), default=1)

    validate = subparsers.add_parser("validate-augmentation")
    for name in ("config-manifest-uri", "augmentation-result-uri", "validation-uri", "run-id"):
        validate.add_argument(f"--{name}", required=True)

    label = subparsers.add_parser("run-auto-label")
    label.add_argument("--workflow", choices=("iaa", "evg"), required=True)
    label.add_argument(
        "--stage",
        choices=(
            "detection",
            "captioning",
            "visual-qa-anomaly",
            "visual-qa-person",
            "person-attribute-search",
        ),
        required=True,
    )
    for name in (
        "validation-uri",
        "auto-label-root-uri",
        "result-uri",
        "vlm-url",
        "vlm-model",
        "llm-url",
        "llm-model",
        "run-id",
    ):
        label.add_argument(f"--{name}", required=True)

    label.add_argument("--previous-result-uri", default="")

    final = subparsers.add_parser("finalize-dataset")
    final.add_argument("--workflow", choices=("iaa", "evg"), required=True)
    for name in (
        "validation-uri",
        "upstream-uri",
        "labels-uri",
        "output-uri",
        "run-id",
    ):
        final.add_argument(f"--{name}", required=True)

    terminal = subparsers.add_parser("validate-dataset")
    for name in ("dataset-uri", "report-uri", "run-id"):
        terminal.add_argument(f"--{name}", required=True)

    dig = subparsers.add_parser("dig-infer")
    for name in (
        "dataset-uri",
        "pretrained-uri",
        "checkpoint-uri",
        "finetune-result-uri",
        "output-uri",
        "result-uri",
        "run-id",
    ):
        dig.add_argument(f"--{name}", required=True)
    dig.add_argument("--num-sdg", type=int, required=True)

    train = subparsers.add_parser("dig-train")
    for name in (
        "dataset-uri",
        "pretrained-uri",
        "output-uri",
        "result-uri",
        "usecase",
        "run-id",
    ):
        train.add_argument(f"--{name}", required=True)

    pretrained = subparsers.add_parser("dig-prepare-pretrained")
    for name in ("output-uri", "result-uri", "run-id"):
        pretrained.add_argument(f"--{name}", required=True)

    postprocess = subparsers.add_parser("postprocess-iaa")
    for name in (
        "validation-uri",
        "prepared-manifest-uri",
        "output-root-uri",
        "result-uri",
        "vlm-url",
        "vlm-model",
        "run-id",
    ):
        postprocess.add_argument(f"--{name}", required=True)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    values = vars(args)
    command = values.pop("command")
    functions = {
        "prepare-images": prepare_images,
        "build-configs": build_augmentation_configs,
        "run-augmentation": run_augmentation,
        "run-local-augmentation": run_local_augmentation,
        "validate-augmentation": validate_augmentation,
        "postprocess-iaa": postprocess_iaa,
        "run-auto-label": run_auto_label,
        "finalize-dataset": finalize_dataset,
        "validate-dataset": validate_dataset,
        "dig-train": run_dig_train,
        "dig-prepare-pretrained": prepare_dig_pretrained,
        "dig-infer": run_dig_inference,
    }
    try:
        result = functions[command](
            **{key.replace("-", "_"): value for key, value in values.items()}
        )
    except (
        PaidfNativeError,
        subprocess.CalledProcessError,
        OSError,
        ValueError,
    ) as exc:
        raise SystemExit(f"PAIDF {command} failed: {exc}") from exc
    print(
        json.dumps(
            {"status": "completed", "command": command, "schema": result["schema"]},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
