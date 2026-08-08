"""Validated GR00T training telemetry recordings for Rerun and MCAP viewers.

The workflow functions in this module are deliberately S3-to-S3.  A GPU stage
produces a real GR00T checkpoint and evidence manifest, then CPU stages:

1. validate the checkpoint, distributed evidence, finite loss, and source data;
2. decode a small representative sequence from the real LeRobot video dataset;
3. write and inspect a real Foxglove-schema MCAP;
4. decode that MCAP into native Rerun archetypes; and
5. independently inspect both recordings before publishing the final index.

Dataset frames have no robot capture clock.  Their recording timeline is always
labelled ``dataset/synthetic-fps`` and must never be presented as sensor time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Iterable, Mapping
from urllib.parse import urlparse

from npa.workbench.foxglove.inspect import McapInfo, summarize_mcap
from npa.workbench.foxglove.mcap_writer import (
    FrameInput,
    LogInput,
    MetricsInput,
    write_run_mcap,
)
from npa.workbench.lichtblick import build_rerun_rrd_from_mcap
from npa.workflows.artifacts import redact_artifact_text


SOURCE_SCHEMA = "npa.groot.visualization_source.v1"
VISUALIZATION_SCHEMA = "npa.groot.visualization.v1"
TIMESTAMP_SEMANTICS = "dataset/synthetic-fps"
RERUN_APPLICATION_ID = "npa_groot_training"
RERUN_TIMELINE = "mcap_time"
REQUIRED_MCAP_SCHEMAS = {
    "/camera": "foxglove.CompressedImage",
    "/log": "foxglove.Log",
}
REQUIRED_METRIC_TOPICS = {
    "/metrics/loss",
    "/metrics/gpu_count",
    "/metrics/world_size",
    "/metrics/checkpoint_bytes",
}


class GrootVisualizationError(RuntimeError):
    """Raised when factual training visualization cannot be produced safely."""


@dataclass(frozen=True)
class S3Ref:
    bucket: str
    key: str

    @property
    def uri(self) -> str:
        return f"s3://{self.bucket}/{self.key}"


def _split_s3(uri: str, *, require_key: bool = True) -> S3Ref:
    parsed = urlparse(str(uri or "").strip())
    if parsed.scheme != "s3" or not parsed.netloc:
        raise GrootVisualizationError(f"expected an s3:// URI, got: {uri!r}")
    key = parsed.path.lstrip("/")
    if require_key and not key:
        raise GrootVisualizationError(f"S3 object URI has no key: {uri!r}")
    return S3Ref(parsed.netloc, key)


def _s3_client(client: Any | None = None) -> Any:
    if client is not None:
        return client
    import boto3

    kwargs: dict[str, Any] = {}
    endpoint = os.environ.get("NEBIUS_S3_ENDPOINT") or os.environ.get(
        "AWS_ENDPOINT_URL"
    )
    if endpoint:
        kwargs["endpoint_url"] = endpoint
    return boto3.client("s3", **kwargs)


def _read_s3_bytes(client: Any, uri: str) -> bytes:
    ref = _split_s3(uri)
    try:
        return client.get_object(Bucket=ref.bucket, Key=ref.key)["Body"].read()
    except Exception as exc:  # noqa: BLE001 - preserve URI context, never credentials
        raise GrootVisualizationError(
            f"unable to read required object {uri}: {exc}"
        ) from exc


def _read_s3_json(client: Any, uri: str) -> dict[str, Any]:
    try:
        payload = json.loads(_read_s3_bytes(client, uri))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GrootVisualizationError(
            f"required JSON object is invalid: {uri}: {exc}"
        ) from exc
    if not isinstance(payload, dict):
        raise GrootVisualizationError(f"required JSON object is not a mapping: {uri}")
    return payload


def _put_bytes(
    client: Any,
    uri: str,
    body: bytes,
    *,
    content_type: str = "application/octet-stream",
) -> dict[str, Any]:
    if not body:
        raise GrootVisualizationError(f"refusing to upload an empty artifact: {uri}")
    ref = _split_s3(uri)
    client.put_object(
        Bucket=ref.bucket, Key=ref.key, Body=body, ContentType=content_type
    )
    head = client.head_object(Bucket=ref.bucket, Key=ref.key)
    size = int(head.get("ContentLength") or 0)
    if size <= 0:
        raise GrootVisualizationError(f"uploaded artifact is empty: {uri}")
    return {
        "uri": uri,
        "bytes": size,
        "sha256": hashlib.sha256(body).hexdigest(),
        "etag": str(head.get("ETag") or "").strip('"'),
    }


def _put_json(client: Any, uri: str, payload: Mapping[str, Any]) -> dict[str, Any]:
    body = (json.dumps(dict(payload), indent=2, sort_keys=True) + "\n").encode("utf-8")
    return _put_bytes(client, uri, body, content_type="application/json")


def _list_objects(client: Any, uri: str) -> list[dict[str, Any]]:
    ref = _split_s3(uri, require_key=False)
    prefix = ref.key.rstrip("/") + "/" if ref.key else ""
    rows: list[dict[str, Any]] = []
    token = ""
    while True:
        kwargs: dict[str, Any] = {"Bucket": ref.bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.list_objects_v2(**kwargs)
        for item in page.get("Contents") or ():
            rows.append(
                {
                    "bucket": ref.bucket,
                    "key": str(item.get("Key") or ""),
                    "size": int(item.get("Size") or 0),
                    "etag": str(item.get("ETag") or "").strip('"'),
                    "last_modified": str(item.get("LastModified") or ""),
                }
            )
        if not page.get("IsTruncated"):
            break
        token = str(page.get("NextContinuationToken") or "")
        if not token:
            break
    return sorted(rows, key=lambda item: item["key"])


def _download(client: Any, uri: str, path: Path) -> None:
    ref = _split_s3(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    client.download_file(ref.bucket, ref.key, str(path))
    if not path.is_file() or path.stat().st_size <= 0:
        raise GrootVisualizationError(f"downloaded artifact is empty: {uri}")


def _require_bool(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not True:
        raise GrootVisualizationError(f"training manifest requires {key}=true")


def _finite_number(value: Any, *, field: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise GrootVisualizationError(
            f"training manifest {field} is not numeric"
        ) from exc
    if not math.isfinite(parsed):
        raise GrootVisualizationError(f"training manifest {field} is not finite")
    return parsed


def _validate_training_manifest(
    payload: Mapping[str, Any], *, run_id: str, expected_gpu_count: int
) -> dict[str, Any]:
    if payload.get("schema") != "npa.groot.finetune.v1":
        raise GrootVisualizationError("unexpected GR00T finetune manifest schema")
    if payload.get("status") != "completed":
        raise GrootVisualizationError("GR00T finetune manifest is not completed")
    if payload.get("run_id") != run_id:
        raise GrootVisualizationError(
            "GR00T manifest run_id does not match this workflow run"
        )
    if expected_gpu_count < 1:
        raise GrootVisualizationError("expected_gpu_count must be positive")
    gpu_count = int(payload.get("num_gpus") or 0)
    world_size = int(payload.get("world_size") or 0)
    distinct = int(payload.get("distinct_gpu_count") or 0)
    if (gpu_count, world_size, distinct) != (
        expected_gpu_count,
        expected_gpu_count,
        expected_gpu_count,
    ):
        raise GrootVisualizationError(
            "GPU/world-size evidence mismatch: "
            f"expected {expected_gpu_count}, got gpu_count={gpu_count}, "
            f"world_size={world_size}, distinct_gpu_count={distinct}"
        )
    uuids = [str(value) for value in payload.get("gpu_uuids") or () if str(value)]
    if len(uuids) != expected_gpu_count or len(set(uuids)) != expected_gpu_count:
        raise GrootVisualizationError("GPU UUID evidence is missing or not distinct")
    _require_bool(payload, "collective_ok")
    _require_bool(payload, "optimizer_step_ok")
    _require_bool(payload, "loss_finite")
    step = int(payload.get("training_step") or 0)
    if step < 1:
        raise GrootVisualizationError("no completed training step was recorded")
    loss = _finite_number(payload.get("loss"), field="loss")
    checkpoint_bytes = int(payload.get("checkpoint_bytes") or 0)
    checkpoint_objects = int(payload.get("checkpoint_object_count") or 0)
    if checkpoint_bytes <= 0 or checkpoint_objects <= 0:
        raise GrootVisualizationError("checkpoint summary is empty")
    return {
        "gpu_count": gpu_count,
        "world_size": world_size,
        "distinct_gpu_count": distinct,
        "gpu_uuids": uuids,
        "collective_ok": True,
        "collective_sum": _finite_number(
            payload.get("collective_sum"), field="collective_sum"
        ),
        "collective_expected": _finite_number(
            payload.get("collective_expected"), field="collective_expected"
        ),
        "training_step": step,
        "optimizer_step_ok": True,
        "loss": loss,
        "loss_finite": True,
        "checkpoint_bytes": checkpoint_bytes,
        "checkpoint_object_count": checkpoint_objects,
    }


def _decode_dataset_video(
    video_path: Path, frames_dir: Path, *, max_frames: int
) -> tuple[list[Path], float]:
    try:
        import av
    except ModuleNotFoundError as exc:  # pragma: no cover - renderer installs PyAV
        raise GrootVisualizationError(
            "PyAV is required to decode representative LeRobot dataset video frames"
        ) from exc
    frames_dir.mkdir(parents=True, exist_ok=True)
    decoded: list[Path] = []
    with av.open(str(video_path)) as container:
        if not container.streams.video:
            raise GrootVisualizationError(
                f"dataset object has no video stream: {video_path.name}"
            )
        stream = container.streams.video[0]
        rate = stream.average_rate or stream.guessed_rate
        fps = float(rate) if rate else 0.0
        for index, frame in enumerate(container.decode(stream)):
            target = frames_dir / f"{index:06d}.png"
            frame.to_image().convert("RGB").save(target, format="PNG")
            decoded.append(target)
            if len(decoded) >= max(1, max_frames):
                break
    if not decoded:
        raise GrootVisualizationError(
            f"dataset video decoded zero frames: {video_path.name}"
        )
    if fps <= 0:
        fps = 10.0
    return decoded, fps


def _metric_payloads(
    training: Mapping[str, Any], *, run_id: str
) -> dict[str, dict[str, Any]]:
    return {
        "loss": {
            "value": training["loss"],
            "step": training["training_step"],
            "run_id": run_id,
        },
        "gpu_count": {"value": training["gpu_count"], "run_id": run_id},
        "world_size": {"value": training["world_size"], "run_id": run_id},
        "checkpoint_bytes": {
            "value": training.get(
                "uploaded_checkpoint_bytes", training["checkpoint_bytes"]
            ),
            "run_id": run_id,
        },
        "training_step": {"value": training["training_step"], "run_id": run_id},
    }


def validate_visualization_source(
    manifest_uri: str,
    data_uri: str,
    source_manifest_uri: str,
    run_id: str,
    expected_gpu_count: int,
    max_frames: int = 8,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Validate a completed GR00T run and publish one factual source bundle."""

    client = _s3_client(s3_client)
    manifest = _read_s3_json(client, manifest_uri)
    training = _validate_training_manifest(
        manifest, run_id=run_id, expected_gpu_count=int(expected_gpu_count)
    )

    checkpoint_uri = str(manifest.get("checkpoint_uri") or "")
    checkpoint_objects = [
        item for item in _list_objects(client, checkpoint_uri) if item["size"] > 0
    ]
    actual_checkpoint_bytes = sum(item["size"] for item in checkpoint_objects)
    if len(checkpoint_objects) < training["checkpoint_object_count"]:
        raise GrootVisualizationError(
            "uploaded checkpoint object count is smaller than the manifest"
        )
    if actual_checkpoint_bytes < training["checkpoint_bytes"]:
        raise GrootVisualizationError(
            "uploaded checkpoint bytes are smaller than the manifest"
        )
    training["uploaded_checkpoint_object_count"] = len(checkpoint_objects)
    training["uploaded_checkpoint_bytes"] = actual_checkpoint_bytes

    log_uri = str(manifest.get("training_log_uri") or "")
    if not log_uri:
        raise GrootVisualizationError("training manifest has no training_log_uri")
    raw_log = _read_s3_bytes(client, log_uri).decode("utf-8", errors="replace")
    safe_log, redacted = redact_artifact_text(raw_log)
    if not safe_log.strip():
        raise GrootVisualizationError("training log is empty after validation")

    dataset_objects = [
        item
        for item in _list_objects(client, data_uri)
        if item["size"] > 0 and item["key"].lower().endswith(".mp4")
    ]
    if not dataset_objects:
        raise GrootVisualizationError("real dataset has no representative MP4 sequence")
    source_object = dataset_objects[0]
    source_uri = f"s3://{source_object['bucket']}/{source_object['key']}"

    source_ref = _split_s3(source_manifest_uri)
    source_prefix = str(PurePosixPath(source_ref.key).parent).rstrip("/")
    with tempfile.TemporaryDirectory(prefix="npa-groot-viz-source-") as tmp:
        root = Path(tmp)
        video_path = root / PurePosixPath(source_object["key"]).name
        _download(client, source_uri, video_path)
        video_sha256 = hashlib.sha256(video_path.read_bytes()).hexdigest()
        frames, fps = _decode_dataset_video(
            video_path, root / "frames", max_frames=max(1, int(max_frames))
        )

        frame_records: list[dict[str, Any]] = []
        for index, frame_path in enumerate(frames):
            uri = f"s3://{source_ref.bucket}/{source_prefix}/frames/{frame_path.name}"
            record = _put_bytes(
                client, uri, frame_path.read_bytes(), content_type="image/png"
            )
            record.update(
                {
                    "frame_index": index,
                    "timeline_seconds": index / fps,
                    "timestamp_semantics": TIMESTAMP_SEMANTICS,
                }
            )
            frame_records.append(record)

        metrics: dict[str, dict[str, Any]] = {}
        for name, payload in _metric_payloads(training, run_id=run_id).items():
            uri = f"s3://{source_ref.bucket}/{source_prefix}/metrics/{name}.json"
            metrics[name] = _put_json(client, uri, payload)

        bundled_log_uri = f"s3://{source_ref.bucket}/{source_prefix}/logs/training.log"
        bundled_log = _put_bytes(
            client, bundled_log_uri, safe_log.encode("utf-8"), content_type="text/plain"
        )

    source = {
        "schema": SOURCE_SCHEMA,
        "status": "validated",
        "run_id": run_id,
        "training_manifest_uri": manifest_uri,
        "training_log_uri": log_uri,
        "training_log_redacted": redacted,
        "data_uri": data_uri,
        "checkpoint_uri": checkpoint_uri,
        "training": training,
        "dataset": {
            "source_object_uri": source_uri,
            "source_object_bytes": source_object["size"],
            "source_object_etag": source_object["etag"],
            "source_object_sha256": video_sha256,
            "fps": fps,
            "frame_count": len(frame_records),
            "timestamp_semantics": TIMESTAMP_SEMANTICS,
            "is_robot_capture_time": False,
            "frames": frame_records,
        },
        "metrics": metrics,
        "log": bundled_log,
        "provenance": {
            "method": "validated GR00T outputs plus decoded frames from the run's real LeRobot dataset",
            "source_uris": [manifest_uri, log_uri, source_uri, checkpoint_uri],
        },
    }
    _put_json(client, source_manifest_uri, source)
    print(json.dumps(source, indent=2, sort_keys=True))
    return source


def _download_source_bundle(
    client: Any, source: Mapping[str, Any], root: Path
) -> tuple[list[FrameInput], list[MetricsInput], list[LogInput]]:
    frames: list[FrameInput] = []
    for item in source.get("dataset", {}).get("frames", []):
        uri = str(item.get("uri") or "")
        target = root / "frames" / PurePosixPath(_split_s3(uri).key).name
        _download(client, uri, target)
        frames.append(FrameInput(path=target, camera="camera"))
    metrics: list[MetricsInput] = []
    for name, item in sorted((source.get("metrics") or {}).items()):
        uri = str(item.get("uri") or "")
        target = root / "metrics" / f"{name}.json"
        _download(client, uri, target)
        metrics.append(MetricsInput(path=target, name=name))
    log_uri = str(source.get("log", {}).get("uri") or "")
    log_path = root / "logs" / "training.log"
    _download(client, log_uri, log_path)
    return frames, metrics, [LogInput(path=log_path, name="groot_training")]


def _validate_mcap(info: McapInfo, *, run_id: str) -> None:
    if not info.valid_magic or info.size_bytes <= 0 or info.message_count <= 0:
        raise GrootVisualizationError(
            "MCAP inspection found an empty or invalid recording"
        )
    for topic, schema in REQUIRED_MCAP_SCHEMAS.items():
        if info.schemas.get(topic) != schema or info.channels.get(topic, 0) <= 0:
            raise GrootVisualizationError(
                f"MCAP missing required {schema} topic {topic}"
            )
    missing = sorted(
        topic for topic in REQUIRED_METRIC_TOPICS if info.channels.get(topic, 0) <= 0
    )
    if missing:
        raise GrootVisualizationError(
            f"MCAP missing required metric topics: {', '.join(missing)}"
        )
    metadata = info.metadata.get("npa") or {}
    if metadata.get("run_id") != run_id:
        raise GrootVisualizationError("MCAP metadata run_id mismatch")
    if metadata.get("timestamps") != TIMESTAMP_SEMANTICS:
        raise GrootVisualizationError(
            "MCAP timestamp semantics are not dataset/synthetic-fps"
        )


def emit_mcap(
    source_manifest_uri: str,
    output_uri: str,
    run_id: str,
    fps: float,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Write and independently inspect the workflow's factual MCAP recording."""

    client = _s3_client(s3_client)
    source = _read_s3_json(client, source_manifest_uri)
    if source.get("schema") != SOURCE_SCHEMA or source.get("run_id") != run_id:
        raise GrootVisualizationError(
            "visualization source manifest is invalid for this run"
        )
    source_fps = float(source.get("dataset", {}).get("fps") or fps)
    if not math.isfinite(source_fps) or source_fps <= 0:
        raise GrootVisualizationError("visualization source has no valid dataset FPS")
    with tempfile.TemporaryDirectory(prefix="npa-groot-mcap-") as tmp:
        root = Path(tmp)
        frames, metrics, logs = _download_source_bundle(client, source, root)
        output = root / "groot-training.mcap"
        summary = write_run_mcap(
            output=output,
            frames=frames,
            metrics=metrics,
            logs=logs,
            fps=source_fps,
            run_id=run_id,
            camera_topic_prefix="",
            metadata={
                "source_manifest_uri": source_manifest_uri,
                "training_manifest_uri": str(source.get("training_manifest_uri") or ""),
                "dataset_source_uri": str(
                    source.get("dataset", {}).get("source_object_uri") or ""
                ),
                "timestamps": TIMESTAMP_SEMANTICS,
                "is_robot_capture_time": "false",
            },
        )
        info = summarize_mcap(output)
        _validate_mcap(info, run_id=run_id)
        artifact = _put_bytes(client, output_uri, output.read_bytes())
    result = {
        "status": "written",
        "artifact": artifact,
        "writer": summary.to_dict(),
        "inspect": info.to_dict(),
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _rerun_executable() -> str:
    candidate = shutil.which("rerun")
    if candidate:
        return candidate
    sibling = Path(sys.executable).with_name("rerun")
    if sibling.is_file():
        return str(sibling)
    raise GrootVisualizationError("rerun CLI is unavailable for .rrd verification")


def inspect_rrd(
    path: str | Path,
    *,
    application_id: str,
    recording_id: str,
    expected_entities: Iterable[str],
    timeline: str = RERUN_TIMELINE,
    runner: Any = subprocess.run,
) -> dict[str, Any]:
    """Verify an RRD with Rerun's supported CLI and inspect its printed contents."""

    target = Path(path)
    if not target.is_file() or target.stat().st_size <= 0:
        raise GrootVisualizationError(f"RRD is missing or empty: {target}")
    executable = _rerun_executable()
    verify = runner(
        [executable, "rrd", "verify", str(target)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if verify.returncode != 0:
        raise GrootVisualizationError(
            f"Rerun could not verify RRD: {verify.stderr[-1000:]}"
        )
    printed = runner(
        [executable, "rrd", "print", "-vv", str(target)],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if printed.returncode != 0:
        raise GrootVisualizationError(
            f"Rerun could not inspect RRD: {printed.stderr[-1000:]}"
        )
    text = f"{printed.stdout}\n{printed.stderr}"
    required = [application_id, recording_id, timeline, *expected_entities]
    missing = [value for value in required if value and value not in text]
    if missing:
        raise GrootVisualizationError(
            "RRD inspection is missing expected identity/timeline/entities: "
            + ", ".join(missing)
        )
    return {
        "parseable": True,
        "bytes": target.stat().st_size,
        "application_id": application_id,
        "recording_id": recording_id,
        "timelines": [timeline],
        "entities": list(expected_entities),
        "verify_stdout": verify.stdout.strip()[-1000:],
    }


def emit_rrd(
    mcap_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Decode the validated MCAP into native Rerun archetypes and upload it."""

    client = _s3_client(s3_client)
    expected_entities = [
        "camera",
        "log",
        "metrics/loss",
        "metrics/gpu_count",
        "metrics/world_size",
        "metrics/checkpoint_bytes",
        "run/provenance",
    ]
    with tempfile.TemporaryDirectory(prefix="npa-groot-rrd-") as tmp:
        root = Path(tmp)
        mcap_path = root / "groot-training.mcap"
        rrd_path = root / "groot-training.rrd"
        _download(client, mcap_uri, mcap_path)
        mcap_info = summarize_mcap(mcap_path)
        _validate_mcap(mcap_info, run_id=run_id)
        converted = build_rerun_rrd_from_mcap(
            str(mcap_path),
            str(rrd_path),
            application_id=RERUN_APPLICATION_ID,
            recording_id=run_id,
        )
        inspection = inspect_rrd(
            rrd_path,
            application_id=RERUN_APPLICATION_ID,
            recording_id=run_id,
            expected_entities=expected_entities,
        )
        artifact = _put_bytes(client, output_uri, rrd_path.read_bytes())
    result = {
        "status": "written",
        "artifact": artifact,
        "converter": converted,
        "inspect": inspection,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def _head_artifact(client: Any, uri: str) -> dict[str, Any]:
    ref = _split_s3(uri)
    head = client.head_object(Bucket=ref.bucket, Key=ref.key)
    size = int(head.get("ContentLength") or 0)
    if size <= 0:
        raise GrootVisualizationError(f"required published artifact is empty: {uri}")
    return {
        "uri": uri,
        "bytes": size,
        "etag": str(head.get("ETag") or "").strip('"'),
        "content_type": str(head.get("ContentType") or "application/octet-stream"),
    }


def publish_visualizations(
    source_manifest_uri: str,
    mcap_uri: str,
    rrd_uri: str,
    workflow_uri: str,
    output_uri: str,
    run_id: str,
    *,
    s3_client: Any | None = None,
) -> dict[str, Any]:
    """Terminal gate: independently inspect and index every required output."""

    client = _s3_client(s3_client)
    source = _read_s3_json(client, source_manifest_uri)
    if source.get("schema") != SOURCE_SCHEMA or source.get("run_id") != run_id:
        raise GrootVisualizationError(
            "source provenance does not match the publish run"
        )
    workflow_bytes = _read_s3_bytes(client, workflow_uri)
    try:
        import yaml

        workflow = yaml.safe_load(workflow_bytes)
    except Exception as exc:  # noqa: BLE001
        raise GrootVisualizationError(
            f"submitted workflow YAML is invalid: {exc}"
        ) from exc
    if (
        not isinstance(workflow, dict)
        or workflow.get("apiVersion") != "npa.workflow/v0.0.1"
    ):
        raise GrootVisualizationError(
            "submitted workflow artifact is not npa.workflow/v0.0.1"
        )

    expected_entities = [
        "camera",
        "log",
        "metrics/loss",
        "metrics/gpu_count",
        "metrics/world_size",
        "metrics/checkpoint_bytes",
        "run/provenance",
    ]
    with tempfile.TemporaryDirectory(prefix="npa-groot-publish-") as tmp:
        root = Path(tmp)
        mcap_path = root / "groot-training.mcap"
        rrd_path = root / "groot-training.rrd"
        _download(client, mcap_uri, mcap_path)
        _download(client, rrd_uri, rrd_path)
        mcap_info = summarize_mcap(mcap_path)
        _validate_mcap(mcap_info, run_id=run_id)
        rrd_info = inspect_rrd(
            rrd_path,
            application_id=RERUN_APPLICATION_ID,
            recording_id=run_id,
            expected_entities=expected_entities,
        )
        mcap_sha256 = hashlib.sha256(mcap_path.read_bytes()).hexdigest()
        rrd_sha256 = hashlib.sha256(rrd_path.read_bytes()).hexdigest()

    result = {
        "schema": VISUALIZATION_SCHEMA,
        "status": "published",
        "run_id": run_id,
        "workflow": str(workflow.get("metadata", {}).get("name") or ""),
        "timestamp_semantics": TIMESTAMP_SEMANTICS,
        "is_robot_capture_time": False,
        "source": {
            "manifest_uri": source_manifest_uri,
            "training_manifest_uri": source.get("training_manifest_uri"),
            "dataset_source_uri": source.get("dataset", {}).get("source_object_uri"),
            "dataset_source_sha256": source.get("dataset", {}).get(
                "source_object_sha256"
            ),
            "fps": source.get("dataset", {}).get("fps"),
            "frame_count": source.get("dataset", {}).get("frame_count"),
        },
        "training": source.get("training"),
        "artifacts": {
            "mcap": {
                **_head_artifact(client, mcap_uri),
                "sha256": mcap_sha256,
                "schema": "mcap.v1",
            },
            "rrd": {
                **_head_artifact(client, rrd_uri),
                "sha256": rrd_sha256,
                "schema": "rerun.rrd.v1",
            },
            "workflow": {
                **_head_artifact(client, workflow_uri),
                "sha256": hashlib.sha256(workflow_bytes).hexdigest(),
                "schema": "npa.workflow/v0.0.1",
            },
            "visualization_manifest": {
                "uri": output_uri,
                "schema": VISUALIZATION_SCHEMA,
            },
        },
        "mcap_inspection": mcap_info.to_dict(),
        "rrd_inspection": rrd_info,
        "provenance_valid": True,
    }
    _put_json(client, output_uri, result)
    result["artifacts"]["visualization_manifest"].update(
        _head_artifact(client, output_uri)
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return result


def build_parser() -> argparse.ArgumentParser:
    """Build the checked module CLI used by GR00T workflow toolRefs."""

    parser = argparse.ArgumentParser(description="Validate and publish GR00T telemetry")
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate")
    validate.add_argument("--training-manifest-uri", required=True)
    validate.add_argument("--data-uri", required=True)
    validate.add_argument("--output-uri", required=True)
    validate.add_argument("--run-id", required=True)
    validate.add_argument("--gpu-count", required=True, type=int)
    validate.add_argument("--max-frames", required=True, type=int)

    mcap = commands.add_parser("emit-mcap")
    mcap.add_argument("--source-manifest-uri", required=True)
    mcap.add_argument("--output-uri", required=True)
    mcap.add_argument("--run-id", required=True)
    mcap.add_argument("--fps", required=True, type=float)

    rrd = commands.add_parser("emit-rrd")
    rrd.add_argument("--mcap-uri", required=True)
    rrd.add_argument("--output-uri", required=True)
    rrd.add_argument("--run-id", required=True)

    publish = commands.add_parser("publish")
    publish.add_argument("--source-manifest-uri", required=True)
    publish.add_argument("--mcap-uri", required=True)
    publish.add_argument("--rrd-uri", required=True)
    publish.add_argument("--workflow-uri", required=True)
    publish.add_argument("--output-uri", required=True)
    publish.add_argument("--run-id", required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run a GR00T visualization phase from the module CLI."""

    args = build_parser().parse_args(argv)
    if args.command == "validate":
        validate_visualization_source(
            args.training_manifest_uri,
            args.data_uri,
            args.output_uri,
            args.run_id,
            args.gpu_count,
            args.max_frames,
        )
    elif args.command == "emit-mcap":
        emit_mcap(args.source_manifest_uri, args.output_uri, args.run_id, args.fps)
    elif args.command == "emit-rrd":
        emit_rrd(args.mcap_uri, args.output_uri, args.run_id)
    else:
        publish_visualizations(
            args.source_manifest_uri,
            args.mcap_uri,
            args.rrd_uri,
            args.workflow_uri,
            args.output_uri,
            args.run_id,
        )
    return 0


__all__ = [
    "GrootVisualizationError",
    "RERUN_APPLICATION_ID",
    "RERUN_TIMELINE",
    "SOURCE_SCHEMA",
    "TIMESTAMP_SEMANTICS",
    "VISUALIZATION_SCHEMA",
    "emit_mcap",
    "emit_rrd",
    "build_parser",
    "inspect_rrd",
    "main",
    "publish_visualizations",
    "validate_visualization_source",
]


if __name__ == "__main__":
    raise SystemExit(main())
