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
import shutil
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import yaml

from npa.clients.storage import StorageClient
from npa.workflows.paidf_upstream import (
    PAIDF_AUGMENTATION_REVISION,
    PAIDF_AUTO_LABELING_REVISION,
    PAIDF_ORCHESTRATION_REVISION,
    PHYSICAL_AI_DATA_FACTORY_REVISION,
)


SCHEMA_PREFIX = "npa.paidf.native"
IMAGE_SUFFIXES = {".bmp", ".gif", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}


class PaidfNativeError(RuntimeError):
    """A PAIDF protocol or artifact contract failed closed."""


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
        return path.is_file() or (path.is_dir() and any(item.is_file() for item in path.rglob("*")))
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
            local = root / f"input-{index:04d}{suffix}"
            _materialize(source_uri, local)
            try:
                with Image.open(local) as image:
                    image.verify()
                with Image.open(local) as image:
                    width, height = image.size
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
    if not all(
        value.strip()
        for value in (vlm_url, vlm_model, llm_url, llm_model, generation_url)
    ):
        raise PaidfNativeError(
            "VLM, LLM, and generation endpoints/models must be explicit"
        )
    prepared = _read_json(prepared_manifest_uri)
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


def run_augmentation(
    config_manifest_uri: str, result_uri: str, run_id: str
) -> dict[str, Any]:
    """Invoke the real paidf-augmentation CLI once for every rendered config."""

    manifest = _read_json(config_manifest_uri)
    configs = manifest.get("configs")
    if not isinstance(configs, list) or not configs:
        raise PaidfNativeError("augmentation config manifest is empty")
    _configure_multistorage(
        config_manifest_uri,
        result_uri,
        *(str(value) for item in configs for value in item.values()),
    )
    token = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if token:
        os.environ.setdefault("VLM_API_KEY", token)
        os.environ.setdefault("LLM_API_KEY", token)
    os.environ.setdefault(
        "GENERATION_API_KEY",
        os.environ.get("IMAGE_EDIT_API_KEY", "")
        or os.environ.get("COSMOS_API_KEY", "")
        or "local",
    )

    with tempfile.TemporaryDirectory(prefix="npa-paidf-augmentation-") as tmp:
        root = Path(tmp)
        baked = Path("/workspace/modules/cli.py")
        if baked.is_file():
            command_prefix = ["/workspace/.venv/bin/python", str(baked)]
        else:
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
                "python",
                str(source / "modules/cli.py"),
            ]
            subprocess.run(
                ["uv", "sync", "--project", str(source), "--frozen"], check=True
            )
        completed: list[dict[str, Any]] = []
        for index, item in enumerate(configs):
            local = root / f"config-{index:04d}.yaml"
            _materialize(str(item["config_uri"]), local)
            _run_component([*command_prefix, "--config", str(local)])
            completed.append(
                {"config_uri": item["config_uri"], "media_uri": item["media_uri"]}
            )
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{manifest['workflow']}-augmentation.v1",
        "run_id": run_id,
        "count": len(completed),
        "component": "NVIDIA paidf-augmentation 1.1.0",
        "upstream_revision": PAIDF_AUGMENTATION_REVISION,
        "outputs": completed,
    }
    return _write_json(payload, result_uri)


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

    if service_kind == "image-edit":
        command = [
            "vllm-omni",
            "serve",
            generation_model,
            "--revision",
            generation_revision,
            "--omni",
            "--port",
            str(port),
        ]
    elif service_kind == "image2video":
        if parallel_size < 1:
            raise PaidfNativeError("parallel_size must be positive")
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
    service = subprocess.Popen(command, start_new_session=True)  # noqa: S603 - fixed executable contract
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
        return run_augmentation(config_manifest_uri, result_uri, run_id)
    finally:
        service.terminate()
        try:
            service.wait(timeout=30)
        except subprocess.TimeoutExpired:
            service.kill()
            service.wait()


def validate_augmentation(
    config_manifest_uri: str, validation_uri: str, run_id: str
) -> dict[str, Any]:
    """Require decodable media plus non-empty caption and metadata for each output."""

    manifest = _read_json(config_manifest_uri)
    accepted: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="npa-paidf-validate-") as tmp:
        root = Path(tmp)
        for index, item in enumerate(manifest.get("configs") or []):
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
                if media.suffix.lower() in IMAGE_SUFFIXES:
                    from PIL import Image

                    with Image.open(media) as image:
                        image.verify()
                else:
                    probe = subprocess.run(
                        [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "stream=codec_type",
                            "-of",
                            "json",
                            str(media),
                        ],
                        text=True,
                        capture_output=True,
                    )
                    if probe.returncode or "video" not in probe.stdout:
                        raise PaidfNativeError("ffprobe found no video stream")
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
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{manifest['workflow']}-validation.v1",
        "run_id": run_id,
        "accepted_count": len(accepted),
        "skipped_count": len(skipped),
        "accepted": accepted,
        "skipped": skipped,
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

    validation = _read_json(validation_uri)
    prepared = _read_json(prepared_manifest_uri)
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
    token = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if not token:
        raise PaidfNativeError(
            "NEBIUS_TOKEN_FACTORY_KEY is required by IAA visual attribute extraction"
        )
    os.environ.setdefault("VLM_API_KEY", token)

    outputs: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="npa-paidf-iaa-postprocess-") as tmp:
        root = Path(tmp)
        source = _runtime_fetch(
            "https://github.com/NVIDIA/paidf-augmentation.git",
            PAIDF_AUGMENTATION_REVISION,
            root / "source",
        )
        subprocess.run(["uv", "sync", "--project", str(source), "--frozen"], check=True)
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
            _run_component(
                [
                    "uv",
                    "run",
                    "--project",
                    str(source),
                    "--no-sync",
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
                ]
            )
            dataset_json_uri = f"{dataset_dir}/augmented_data.json"
            dataset_json = _read_json(dataset_json_uri)
            entries = dataset_json.get("entries")
            if not isinstance(entries, list) or not entries:
                raise PaidfNativeError(
                    f"upstream IAA post-processing produced no entries for {input_key}"
                )
            for entry in entries:
                images = entry.get("images")
                if not isinstance(images, list) or not images:
                    raise PaidfNativeError(
                        "upstream IAA post-processing entry has no images"
                    )
                media_uri = f"{dataset_dir.rstrip('/')}/{images[0].split('/', 1)[-1]}"
                outputs.append(
                    {
                        **item,
                        "key": str(entry["person_key"]),
                        "media_uri": media_uri,
                        "postprocess_dataset_uri": dataset_json_uri,
                    }
                )
    payload = {
        "schema": f"{SCHEMA_PREFIX}.iaa-postprocess.v1",
        "run_id": run_id,
        "accepted_count": len(outputs),
        "accepted": outputs,
        "component": "NVIDIA paidf-augmentation create_attribute_augmented_dataset.py",
        "upstream_revision": PAIDF_AUGMENTATION_REVISION,
    }
    return _write_json(payload, result_uri)


def _asset_to_uri(source: Path, destination_uri: str) -> str:
    if not source.is_file() or not source.stat().st_size:
        raise PaidfNativeError(f"required upstream protocol asset is missing: {source}")
    return _publish(source, destination_uri)


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
) -> dict[str, Any]:
    """Invoke one genuine paidf-auto-labeling service CLI over validated media."""

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
    validation = _read_json(validation_uri)
    accepted = validation.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise PaidfNativeError("auto-labeling requires validated media")
    _configure_multistorage(
        validation_uri,
        auto_label_root_uri,
        result_uri,
        *(str(value) for item in accepted for value in item.values()),
    )
    token = os.environ.get("NEBIUS_TOKEN_FACTORY_KEY", "").strip()
    if not token:
        raise PaidfNativeError(
            "NEBIUS_TOKEN_FACTORY_KEY is required by the OpenAI-compatible labeling protocol"
        )
    os.environ.setdefault("NVIDIA_API_KEY", token)
    os.environ.setdefault("VLM_API_KEY", token)
    os.environ.setdefault("LLM_API_KEY", token)

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
            args = ["main", "--input", input_json]
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
                            "12",
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
                            "16",
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
                    args.extend(["--attribute-json", item["metadata_uri"]])
            _run_component(args)
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
            elif workflow == "evg" and _uri_is_file(
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
            if workflow == "iaa" and not _uri_prefix_has_objects(data_path):
                raise PaidfNativeError(
                    "IAA attribute search produced no dataset objects"
                )
            completed.append(
                {
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
        "stage": stage,
        "component": "NVIDIA paidf-auto-labeling 1.1.0",
        "upstream_revision": PAIDF_AUTO_LABELING_REVISION,
        "count": len(completed),
        "outputs": completed,
    }
    return _write_json(payload, result_uri)


def finalize_dataset(
    workflow: str,
    validation_uri: str,
    upstream_uri: str,
    labels_uri: str,
    output_uri: str,
    run_id: str,
) -> dict[str, Any]:
    """Publish the final IAA/EVG dataset index only after media validation."""

    validation = _read_json(validation_uri)
    upstream = _read_json(upstream_uri)
    labels = _read_json(labels_uri)
    accepted = validation.get("accepted")
    if not isinstance(accepted, list) or not accepted:
        raise PaidfNativeError("cannot finalize a dataset with no accepted media")
    label_outputs = labels.get("outputs")
    if not isinstance(label_outputs, list):
        raise PaidfNativeError("labeling result has no output records")
    label_by_key = {str(item.get("key")): item for item in label_outputs}
    entries = []
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
        elif not _uri_prefix_has_objects(data_path):
            raise PaidfNativeError(f"IAA terminal dataset is empty for {key}")
        else:
            validated_artifact_count += 1
        entry = {
            "key": key,
            "media": item["media_uri"],
            "caption": item["caption_uri"],
            "metadata": item["metadata_uri"],
            "variables": item["variables"],
            "sha256": item["sha256"],
            "size_bytes": item["size_bytes"],
            "labels": data_path,
        }
        entries.append(entry)
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{workflow}-dataset.v1",
        "run_id": run_id,
        "workflow": workflow,
        "status": "completed",
        "entry_count": len(entries),
        "validated_artifact_count": validated_artifact_count,
        "trackless_scene_count": trackless_count,
        "entries": entries,
        "upstream": upstream,
        "lineage": {
            "validation_uri": validation_uri,
            "labels_uri": labels_uri,
            "upstream_uri": upstream_uri,
        },
    }
    return _write_json(payload, output_uri)


def validate_dataset(dataset_uri: str, report_uri: str, run_id: str) -> dict[str, Any]:
    """Fail closed on the published native dataset and emit a terminal decision."""

    dataset = _read_json(dataset_uri)
    entries = dataset.get("entries")
    if (
        dataset.get("status") != "completed"
        or not isinstance(entries, list)
        or not entries
        or dataset.get("entry_count") != len(entries)
    ):
        raise PaidfNativeError("terminal dataset manifest is incomplete")
    missing = []
    for entry in entries:
        for field in ("media", "caption", "metadata"):
            value = str(entry.get(field) or "")
            if not value or not _uri_is_file(value):
                missing.append(f"{entry.get('key', '<unknown>')}:{field}")
        labels = str(entry.get("labels") or "")
        if not labels or not _uri_prefix_has_objects(labels):
            missing.append(f"{entry.get('key', '<unknown>')}:labels")
    if missing:
        raise PaidfNativeError(
            f"terminal dataset validation found {len(missing)} missing artifact(s)"
        )
    encoded = json.dumps(dataset, sort_keys=True, separators=(",", ":")).encode()
    payload = {
        "schema": f"{SCHEMA_PREFIX}.{dataset['workflow']}-terminal-validation.v1",
        "run_id": run_id,
        "status": "passed",
        "dataset_uri": dataset_uri,
        "dataset_manifest_sha256": hashlib.sha256(encoded).hexdigest(),
        "entry_count": len(entries),
        "validated_artifact_count": dataset.get("validated_artifact_count", 0),
        "trackless_scene_count": dataset.get("trackless_scene_count", 0),
    }
    return _write_json(payload, report_uri)


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
            **os.environ,
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
        if not selected.is_file():
            raise PaidfNativeError("AnomalyGen best-checkpoint pointer is invalid")
        _publish(train_output, output_uri)
        payload = {
            "schema": f"{SCHEMA_PREFIX}.dig-finetune.v1",
            "run_id": run_id,
            "status": "completed",
            "component": "NVIDIA paidf-anomalygen 1.1.0",
            "upstream_workflow_revision": PHYSICAL_AI_DATA_FACTORY_REVISION,
            "selected_checkpoint": selected.name,
            "selected_checkpoint_sha256": _sha256(selected),
            "output_uri": output_uri,
        }
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
        env = {**os.environ, "CKPT_DIR": str(output)}
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
        files = [item for item in output.rglob("*") if item.is_file()]
        _publish(output, output_uri)
        payload = {
            "schema": f"{SCHEMA_PREFIX}.dig-pretrained.v1",
            "run_id": run_id,
            "status": "completed",
            "component": "NVIDIA paidf-anomalygen 1.1.0 checkpoint setup",
            "file_count": len(files),
            "total_bytes": sum(item.stat().st_size for item in files),
            "manifest_sha256": _sha256(output / converted_manifest.name),
            "output_uri": output_uri,
        }
    return _write_json(payload, result_uri)


def run_dig_inference(
    dataset_uri: str,
    pretrained_uri: str,
    checkpoint_uri: str,
    output_uri: str,
    result_uri: str,
    num_sdg: int,
    checkpoint_step: str,
    run_id: str,
) -> dict[str, Any]:
    """Run AnomalyGen's published Day-1 manual-ROI inference and native labels."""

    workspace = Path("/workspace/paidf-anomalygen")
    if not workspace.is_dir():
        raise PaidfNativeError(
            "the selected image is not NVIDIA paidf-anomalygen 1.1.0"
        )
    if num_sdg < 1 or (checkpoint_step and not checkpoint_step.isdigit()):
        raise PaidfNativeError("invalid AnomalyGen inference settings")

    with tempfile.TemporaryDirectory(prefix="npa-paidf-dig-") as tmp:
        root = Path(tmp)
        dataset = _materialize(dataset_uri, root / "dataset")
        pretrained = _materialize(pretrained_uri, root / "pretrained")
        checkpoint = _materialize(checkpoint_uri, root / "checkpoint")
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
            **os.environ,
            "PRETRAINED_SRC": str(pretrained),
            "DATASET_DIR": str(defect_specs[0].parent),
            "DEFECT_SPEC": str(defect_specs[0]),
            "FINETUNE_DIR": str(checkpoint),
            "OUTPUT_DIR": str(generated),
            "NUM_SDG": str(num_sdg),
            "NUM_GPUS": "1",
            "CHECKPOINT_STEP": checkpoint_step,
        }
        _run_component(["bash", str(script)], env=env)
        images = sorted((generated / "reconstructed_image").glob("*"))
        labels = generated / "pseudo_labels/coco_annotations.json"
        if not images or not labels.is_file() or not labels.stat().st_size:
            raise PaidfNativeError(
                "AnomalyGen returned no generated images or label metadata"
            )
        _publish(generated, output_uri)
        payload = {
            "schema": f"{SCHEMA_PREFIX}.dig-result.v1",
            "run_id": run_id,
            "status": "completed",
            "component": "NVIDIA paidf-anomalygen 1.1.0",
            "upstream_workflow_revision": PHYSICAL_AI_DATA_FACTORY_REVISION,
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
    local_augment.add_argument("--parallel-size", type=int, default=1)

    validate = subparsers.add_parser("validate-augmentation")
    for name in ("config-manifest-uri", "validation-uri", "run-id"):
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
        "output-uri",
        "result-uri",
        "run-id",
    ):
        dig.add_argument(f"--{name}", required=True)
    dig.add_argument("--num-sdg", type=int, required=True)
    dig.add_argument("--checkpoint-step", required=True)

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
