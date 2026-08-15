"""Canonical S3 MCAP contract shared by the deployed agent and unit tests."""

from __future__ import annotations

import hashlib
import json
import logging
import tempfile
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

RICH_VISUALIZATION_CONTRACT = "npa.foxglove.robot-motion.v3"
_RICH_VISUALIZATION_TOPICS = {
    "/camera": "foxglove.CompressedImage",
    "/robot/diagnostic_scene": "foxglove.SceneUpdate",
    "/robot/diagnostic_pose": "foxglove.PoseInFrame",
    "/robot/diagnostic_trajectory": "foxglove.PosesInFrame",
    "/robot/diagnostic_joint_states": "foxglove.JointStates",
    "/actuators/commands": "npa.ActuatorCommands",
    "/run/state": "npa.RunState",
}

CANONICAL_MCAP_DEFAULT_STATE: dict[str, Any] = {
    "canonical_mcap_s3_uri": "",
    "canonical_mcap_key": "",
    "canonical_mcap_sha256": "",
    "canonical_mcap_size_bytes": 0,
    "canonical_mcap_source": "",
    "canonical_mcap_provenance": {},
    "transport_state": "",
    "foxglove_cloud": {},
    "foxglove_cloud_layout": {},
    "foxglove_selected_artifact": {},
}


def clear_cross_run_mcap_state(state: dict, run_id: str) -> None:
    """Clear viewer/publication identity when the selected run changes."""
    previous_run = str(state.get("run_id") or "")
    if not previous_run or previous_run == run_id:
        return
    state.update(CANONICAL_MCAP_DEFAULT_STATE)
    state.update(
        {
            "mcap_uri": "",
            "mcap_updated_at": "",
            "lichtblick_ready": False,
            "lichtblick_iframe_url": "/lichtblick/",
            "foxglove_ready": False,
            "foxglove_url": "",
        }
    )


def run_relative_artifact_key(
    key: str, run_id: str, *, safe_key: Callable[[str], str]
) -> str:
    """Return a traversal-checked key relative to its run directory."""
    value = safe_key(key)
    marker = f"/{run_id}/"
    if marker in value:
        return value.split(marker, 1)[1]
    prefix = f"{run_id}/"
    if value.startswith(prefix):
        return value[len(prefix) :]
    raise RuntimeError(f"artifact key is not scoped to run {run_id}: {value}")


def canonical_key_for_run(
    key: str, run_id: str, *, safe_key: Callable[[str], str]
) -> str:
    """Return the reserved reports key beside a run's existing artifacts."""
    relative = run_relative_artifact_key(key, run_id, safe_key=safe_key)
    return key[: len(key) - len(relative)] + "reports/sim2real.mcap"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def has_rich_visualization_contract(info: dict[str, Any]) -> bool:
    """Prove that a rich canonical MCAP has the current robot-motion contract."""
    schemas = dict(info.get("schemas") or {})
    metadata = dict(info.get("metadata") or {})
    npa_metadata = dict(metadata.get("npa") or {})
    contract = str(
        npa_metadata.get("visualization_contract")
        or info.get("visualization_contract")
        or ""
    )
    return (
        contract == RICH_VISUALIZATION_CONTRACT
        and all(
            schemas.get(topic) == schema
            for topic, schema in _RICH_VISUALIZATION_TOPICS.items()
        )
        and any(schema == "foxglove.Log" for schema in schemas.values())
        and any(
            schema.startswith("npa.RunMetrics")
            for schema in schemas.values()
            if isinstance(schema, str)
        )
    )


def rich_run_provenance_from_manifest(
    manifest: dict[str, Any], *, run_id: str, manifest_key: str, manifest_sha256: str
) -> dict[str, Any]:
    """Validate and compact the optional honest rich-visualization contract."""
    if manifest.get("schema") != "npa.foxglove.rich-run.v1":
        return {}
    if str(manifest.get("run_id") or "") != run_id:
        raise RuntimeError("rich-run manifest run_id does not match the selected run")
    engine = manifest.get("engine_provenance")
    cameras = manifest.get("camera_counts")
    limitations = manifest.get("limitations")
    if not isinstance(engine, dict) or not str(engine.get("engine") or ""):
        raise RuntimeError("rich-run manifest lacks engine provenance")
    if not isinstance(cameras, dict) or not cameras:
        raise RuntimeError("rich-run manifest lacks camera counts")
    if not isinstance(limitations, list):
        raise RuntimeError("rich-run manifest limitations must be a list")
    duration = float(manifest.get("duration_seconds") or 0)
    samples = int(manifest.get("sample_count") or 0)
    if duration <= 0 or samples <= 0:
        raise RuntimeError(
            "rich-run manifest duration and sample count must be positive"
        )
    return {
        "schema": "npa.foxglove.rich-run.v1",
        "manifest_key": manifest_key,
        "manifest_sha256": manifest_sha256,
        "engine_provenance": {str(k): v for k, v in engine.items()},
        "duration_seconds": duration,
        "sample_count": samples,
        "fps": float(manifest.get("fps") or 0),
        "camera_counts": {str(k): int(v) for k, v in cameras.items()},
        "timestamp_semantics": str(manifest.get("timestamp_semantics") or ""),
        "trajectory_semantics": str(manifest.get("trajectory_semantics") or ""),
        "limitations": [str(value) for value in limitations],
    }


def prepare_canonical_mcap(
    *,
    run_id: str,
    source_bucket: str = "",
    source_prefix: str = "",
    fps: float,
    max_frames: int,
    validate_run_id: Callable[[str], str],
    s3_client: Callable[[], tuple[Any, dict]],
    list_buckets: Callable[[Any, dict], list[str]],
    find_artifacts: Callable[..., tuple[str, list]],
    safe_key: Callable[[str], str],
    download: Callable[..., Path],
    convert: Callable[..., Any],
    summarize: Callable[[Path], Any],
    invalidate_cache: Callable[[], None],
    now_iso: Callable[[], str],
    recordings_dir: Path,
) -> dict:
    """Validate/reuse or generate/persist one canonical MCAP and provenance."""
    normalized = validate_run_id(run_id)
    s3, settings = s3_client()
    available_buckets = list_buckets(s3, settings)
    if source_bucket and source_bucket not in available_buckets:
        raise RuntimeError("run_ref bucket is not configured for this agent")
    bucket, artifacts = find_artifacts(
        [source_bucket] if source_bucket else available_buckets,
        base_prefix=settings.get("prefix", ""),
        run_id=normalized,
        exact_bucket=source_bucket,
        exact_prefix=source_prefix if source_bucket else None,
        s3=s3,
    )
    if not bucket or not artifacts:
        raise RuntimeError(
            f"no S3 artifacts found for run {normalized}; upload completed run inputs first"
        )
    canonical_key = canonical_key_for_run(
        str(artifacts[0].key), normalized, safe_key=safe_key
    )
    provenance_key = canonical_key + ".provenance.json"
    native = next((item for item in artifacts if str(item.key) == canonical_key), None)
    cache_name = hashlib.sha256(canonical_key.encode("utf-8")).hexdigest()[:12]
    local_path = recordings_dir / f"{cache_name}-sim2real.mcap"
    source = "native-reused" if native is not None else "generated-from-s3-artifacts"
    source_keys = [
        str(item.key)
        for item in artifacts
        if str(item.key) not in {canonical_key, provenance_key}
    ]
    rich_run: dict[str, Any] = {}
    rich_manifest = next(
        (
            item
            for item in artifacts
            if str(item.key).endswith("/reports/rich-run-manifest.json")
        ),
        None,
    )
    if rich_manifest is not None:
        response = s3.get_object(Bucket=bucket, Key=str(rich_manifest.key))
        raw_manifest = response["Body"].read()
        try:
            manifest = json.loads(raw_manifest)
        except (TypeError, ValueError) as exc:
            raise RuntimeError("rich-run manifest is not valid JSON") from exc
        if not isinstance(manifest, dict):
            raise RuntimeError("rich-run manifest must be a JSON object")
        rich_run = rich_run_provenance_from_manifest(
            manifest,
            run_id=normalized,
            manifest_key=str(rich_manifest.key),
            manifest_sha256=hashlib.sha256(raw_manifest).hexdigest(),
        )

    converted: dict = {}
    saved_provenance: dict = {}

    def generate_rich_canonical() -> dict:
        with tempfile.TemporaryDirectory(prefix=f"npa-mcap-{normalized}-") as tmp:
            source_dir = Path(tmp) / normalized
            for item in artifacts:
                if str(item.key) in {canonical_key, provenance_key}:
                    continue
                relative = run_relative_artifact_key(
                    str(item.key), normalized, safe_key=safe_key
                )
                download(str(item.s3_uri), source_dir / relative, s3=s3)
            result = convert(
                input_path=source_dir,
                output_path=local_path,
                fps=fps,
                max_frames=max_frames,
                run_id=normalized,
            ).to_dict()
        with local_path.open("rb") as body:
            s3.put_object(
                Bucket=bucket,
                Key=canonical_key,
                Body=body,
                ContentType="application/octet-stream",
                Metadata={
                    "npa-sha256": sha256_file(local_path),
                    "npa-canonical": "true",
                },
            )
        return result

    if native is not None:
        download(str(native.s3_uri), local_path, s3=s3)
        native_info = summarize(local_path).to_dict()
        if rich_run and not has_rich_visualization_contract(native_info):
            converted = generate_rich_canonical()
            source = "regenerated-rich-visualization-v3"
            saved_provenance = {}
        prior = next(
            (item for item in artifacts if str(item.key) == provenance_key), None
        )
        if prior is not None and not converted:
            try:
                response = s3.get_object(Bucket=bucket, Key=provenance_key)
                saved = json.loads(response["Body"].read())
                if isinstance(saved, dict):
                    saved_provenance = saved
            except Exception as exc:
                # A stale/unreadable sidecar is repaired from the validated MCAP.
                logger.debug("repairing unreadable canonical MCAP provenance: %s", exc)
    else:
        converted = generate_rich_canonical()

    info = summarize(local_path).to_dict()
    digest = sha256_file(local_path)
    if not info.get("valid_magic") or not int(info.get("message_count") or 0):
        raise RuntimeError("reports/sim2real.mcap is malformed or contains no messages")
    head = s3.head_object(Bucket=bucket, Key=canonical_key)
    if int(head.get("ContentLength") or -1) != int(info["size_bytes"]):
        raise RuntimeError(
            "canonical MCAP S3 size does not match the validated local bytes"
        )
    if saved_provenance.get("sha256") == digest:
        source = str(saved_provenance.get("source") or source)
        saved_sources = saved_provenance.get("source_artifacts")
        if isinstance(saved_sources, list):
            source_keys = [str(value) for value in saved_sources]
    metadata = dict(info.get("metadata") or {})
    timestamps = str((metadata.get("npa") or {}).get("timestamps") or "source")
    provenance = {
        "schema": "npa.canonical-mcap.v2",
        "run_id": normalized,
        "canonical_key": canonical_key,
        "canonical_s3_uri": f"s3://{bucket}/{canonical_key}",
        "sha256": digest,
        "size_bytes": int(info["size_bytes"]),
        "source": source,
        "source_artifacts": source_keys,
        "message_count": int(info["message_count"]),
        "channels": dict(info.get("channels") or {}),
        "schemas": dict(info.get("schemas") or {}),
        "numeric_paths": dict(info.get("numeric_paths") or {}),
        "channel_time_ranges": dict(info.get("channel_time_ranges") or {}),
        "start_time_ns": int(info.get("start_time_ns") or 0),
        "end_time_ns": int(info.get("end_time_ns") or 0),
        "duration_s": float(info.get("duration_s") or 0),
        "timestamps": timestamps,
        "fps": str((metadata.get("npa") or {}).get("fps") or ""),
        "visualization_contract": str(
            (metadata.get("npa") or {}).get("visualization_contract") or ""
        ),
        "scene_update_schema_source": str(
            (metadata.get("npa") or {}).get("scene_update_schema_source") or ""
        ),
        "visualization_fixed_frame": str(
            (metadata.get("npa") or {}).get("visualization_fixed_frame") or ""
        ),
        "visualization_fidelity": str(
            (metadata.get("npa") or {}).get("visualization_fidelity") or ""
        ),
        "updated_at": now_iso(),
    }
    if rich_run:
        provenance["rich_run"] = rich_run
    s3.put_object(
        Bucket=bucket,
        Key=provenance_key,
        Body=(json.dumps(provenance, indent=2, sort_keys=True) + "\n").encode(),
        ContentType="application/json",
    )
    invalidate_cache()
    summary = converted or {
        "output": str(local_path),
        "size_bytes": int(info["size_bytes"]),
        "message_count": int(info["message_count"]),
        "channels": dict(info.get("channels") or {}),
        "channel_time_ranges": dict(info.get("channel_time_ranges") or {}),
        "start_time_ns": int(info.get("start_time_ns") or 0),
        "end_time_ns": int(info.get("end_time_ns") or 0),
        "timestamps": timestamps,
        "reused_native": True,
    }
    return {
        "artifact_key": canonical_key,
        "s3_uri": provenance["canonical_s3_uri"],
        "local_path": str(local_path),
        "sha256": digest,
        "size_bytes": int(info["size_bytes"]),
        "source": source,
        "created": native is None or bool(converted),
        "provenance": provenance,
        "summary": summary,
    }
