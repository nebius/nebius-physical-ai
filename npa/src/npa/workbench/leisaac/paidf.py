"""S3-native bridge between LeIsaac demonstrations and the PAIDF workflow."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import tempfile
import uuid
from pathlib import Path
from typing import Any

from npa.workbench.leisaac.dataset import (
    DERIVED_SCHEMA,
    LEROBOT_TARGET_VERSION,
    VIDEO_KEY,
    DatasetError,
    _ffmpeg_executable,
    resolve_s3_endpoint,
    sha256_file,
    split_s3_uri,
    utc_now,
)


def _client(
    endpoint_url: str | None = None, *, config_endpoint: str | None = None
) -> Any:
    import boto3

    return boto3.client(
        "s3",
        endpoint_url=resolve_s3_endpoint(endpoint_url, config_endpoint=config_endpoint),
        region_name=os.environ.get("AWS_REGION") or "eu-north1",
    )


def _json_object(client: Any, bucket: str, key: str) -> dict[str, Any]:
    body = client.get_object(Bucket=bucket, Key=key)["Body"].read()
    try:
        payload = json.loads(body)
    except (UnicodeDecodeError, ValueError) as exc:
        raise DatasetError(f"S3 object is not valid JSON: s3://{bucket}/{key}") from exc
    if not isinstance(payload, dict):
        raise DatasetError(f"S3 JSON object must be a mapping: s3://{bucket}/{key}")
    return payload


def _metadata_sha256(response: dict[str, Any]) -> str:
    """Read S3 user metadata without assuming a provider's key casing."""

    metadata = response.get("Metadata")
    if not isinstance(metadata, dict):
        return ""
    return str(
        next(
            (value for key, value in metadata.items() if str(key).lower() == "sha256"),
            "",
        )
        or ""
    )


def _dataset_manifest(client: Any, dataset_uri: str) -> tuple[str, str, dict[str, Any]]:
    bucket, prefix = split_s3_uri(dataset_uri, label="dataset version URI")
    manifest = _json_object(client, bucket, f"{prefix}/npa-dataset.json")
    if manifest.get("schema") != "npa.leisaac.dataset.v1":
        raise DatasetError("source is not an immutable NPA LeIsaac dataset version")
    if manifest.get("lerobot_version") != LEROBOT_TARGET_VERSION:
        raise DatasetError("source dataset does not target LeRobot 0.5.1")
    if str(manifest.get("dataset_uri") or "").rstrip("/") != dataset_uri.rstrip("/"):
        raise DatasetError(
            "source dataset manifest URI does not match the selected version"
        )
    return bucket, prefix, manifest


def _episode_commit(
    client: Any, manifest: dict[str, Any], episode_index: int
) -> dict[str, Any]:
    commits = manifest.get("episode_commits")
    if (
        not isinstance(commits, list)
        or episode_index < 0
        or episode_index >= len(commits)
    ):
        raise DatasetError("episode index is outside the finalized dataset")
    bucket, key = split_s3_uri(str(commits[episode_index]), label="episode commit URI")
    commit = _json_object(client, bucket, key)
    if int(commit.get("episode_index", -1)) != episode_index:
        raise DatasetError("episode commit index does not match the requested episode")
    return commit


def export_episode_to_paidf(
    *,
    dataset_uri: str,
    episode_index: int,
    paidf_run_id: str,
    paidf_output_path: str,
    client: Any | None = None,
) -> dict[str, Any]:
    """Copy a finalized episode's real viewport clip into a clean PAIDF input prefix."""

    client = client or _client()
    source_bucket, source_prefix, manifest = _dataset_manifest(client, dataset_uri)
    commit = _episode_commit(client, manifest, episode_index)
    manifest_head = client.head_object(
        Bucket=source_bucket, Key=f"{source_prefix}/npa-dataset.json"
    )
    manifest_sha256 = _metadata_sha256(manifest_head)
    if len(manifest_sha256) != 64:
        raise DatasetError("source dataset manifest has no SHA-256 object metadata")
    commit_uri = str(manifest["episode_commits"][episode_index])
    commit_bucket, commit_key = split_s3_uri(commit_uri, label="episode commit URI")
    commit_sha256 = _metadata_sha256(
        client.head_object(Bucket=commit_bucket, Key=commit_key)
    )
    frames = commit.get("objects", {}).get("frames")
    if (
        len(commit_sha256) != 64
        or not isinstance(frames, list)
        or len(frames) != int(commit["metadata"].get("frame_count", -1))
    ):
        raise DatasetError("source episode commit has incomplete checksum provenance")
    frame_checksums = [str(item.get("sha256") or "") for item in frames]
    if not all(len(value) == 64 for value in frame_checksums):
        raise DatasetError("source episode has an invalid raw-frame checksum")
    frame_checksums_sha256 = hashlib.sha256(
        json.dumps(frame_checksums, separators=(",", ":")).encode()
    ).hexdigest()
    output_bucket, output_prefix = split_s3_uri(
        paidf_output_path, label="PAIDF output path"
    )
    expected_suffix = f"physical-ai-data-factory/{paidf_run_id}"
    if not (
        output_prefix.rstrip("/") == expected_suffix
        or output_prefix.rstrip("/").endswith("/" + expected_suffix)
    ):
        raise DatasetError(
            "PAIDF output path must be the clean run prefix "
            f"s3://BUCKET/{expected_suffix}"
        )
    video = commit["objects"]["video"]
    input_key = f"{output_prefix}/input/leisaac-episode-{episode_index:06d}.mp4"
    client.copy_object(
        Bucket=output_bucket,
        Key=input_key,
        CopySource={"Bucket": source_bucket, "Key": video["key"]},
        MetadataDirective="COPY",
        IfNoneMatch="*",
    )
    annotation_count = min(8, len(frames))
    annotation_indexes = (
        [0]
        if annotation_count == 1
        else [
            round(index * (len(frames) - 1) / (annotation_count - 1))
            for index in range(annotation_count)
        ]
    )
    annotation_frames: list[dict[str, Any]] = []
    for frame_index in annotation_indexes:
        frame = frames[frame_index]
        annotation_key = f"{output_prefix}/input/leisaac-frame-{frame_index:06d}.jpg"
        client.copy_object(
            Bucket=output_bucket,
            Key=annotation_key,
            CopySource={"Bucket": source_bucket, "Key": frame["key"]},
            MetadataDirective="COPY",
            IfNoneMatch="*",
        )
        annotation_frames.append(
            {
                "frame_index": frame_index,
                "uri": f"s3://{output_bucket}/{annotation_key}",
                "sha256": frame["sha256"],
            }
        )
    lineage = {
        "schema": "npa.leisaac.paidf-input.v1",
        "created_at": utc_now(),
        "source": {
            "dataset_uri": dataset_uri.rstrip("/"),
            "dataset_manifest_sha256": manifest_sha256,
            "episode_commit_uri": commit_uri,
            "episode_commit_sha256": commit_sha256,
            "episode_index": episode_index,
            "episode_uuid": commit["episode_uuid"],
            "task": commit["metadata"]["task"],
            "environment_id": commit["metadata"]["environment_id"],
            "environment_index": commit["metadata"]["environment_index"],
            "seed": commit["metadata"]["seed"],
            "video_sha256": video["sha256"],
            "records_sha256": commit["objects"]["records"]["sha256"],
            "frame_count": commit["metadata"]["frame_count"],
            "frame_checksums_sha256": frame_checksums_sha256,
        },
        "paidf": {
            "run_id": paidf_run_id,
            "run_uri": f"s3://{output_bucket}/{output_prefix}",
            "input_uri": f"s3://{output_bucket}/{input_key}",
            "annotation_frames": annotation_frames,
            "required_tool_ref": "workbench.cosmos2.transfer_execute",
            "required_engine": "cosmos_transfer2.5_gpu",
            "condition_on_input": True,
        },
        "semantics": (
            "Cosmos Transfer appearance augmentation preserves demonstrated motion "
            "through input conditioning; it is not action augmentation or new success evidence."
        ),
    }
    lineage_bytes = (json.dumps(lineage, indent=2, sort_keys=True) + "\n").encode()
    lineage_key = f"{output_prefix}/input/leisaac-lineage.json"
    client.put_object(
        Bucket=output_bucket,
        Key=lineage_key,
        Body=lineage_bytes,
        ContentType="application/json",
        Metadata={"sha256": hashlib.sha256(lineage_bytes).hexdigest()},
        IfNoneMatch="*",
    )
    command = (
        "NPA_COSMOS_CONDITION_ON_INPUT=1 npa workbench workflow submit "
        "npa/workflows/physical-ai-data-factory.yaml "
        f"--run-id {paidf_run_id} --assume-decision promote_checkpoint "
        f"--var bucket={output_bucket} --var prefix={output_prefix} "
        "--secret-env NEBIUS_TOKEN_FACTORY_KEY --secret-env AWS_ACCESS_KEY_ID "
        "--secret-env AWS_SECRET_ACCESS_KEY --secret-env HF_TOKEN"
    )
    return {
        "status": "exported",
        "input_uri": lineage["paidf"]["input_uri"],
        "lineage_uri": f"s3://{output_bucket}/{lineage_key}",
        "paidf_run_uri": lineage["paidf"]["run_uri"],
        "workflow": "npa/workflows/physical-ai-data-factory.yaml",
        "tool_ref": "workbench.cosmos2.transfer_execute",
        "condition_on_input": True,
        "annotation_frame_count": len(annotation_frames),
        "command": command,
    }


def _video_timestamps(path: Path) -> list[float]:
    ffprobe = shutil.which("ffprobe")
    if ffprobe:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise DatasetError(
                f"ffprobe failed for {path.name}: {result.stderr.strip()}"
            )
        frames = json.loads(result.stdout).get("frames", [])
        values = [float(item["best_effort_timestamp_time"]) for item in frames]
    else:
        result = subprocess.run(
            [
                _ffmpeg_executable(),
                "-hide_banner",
                "-loglevel",
                "info",
                "-i",
                str(path),
                "-vf",
                "showinfo",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise DatasetError(
                f"ffmpeg failed to inspect {path.name}: {result.stderr.strip()}"
            )
        values = [
            float(value)
            for value in re.findall(r"\bpts_time:([-+0-9.eE]+)", result.stderr)
        ]
    if len(values) < 2:
        raise DatasetError("video has fewer than two decoded frames")
    return values


def _assert_nonblank(path: Path) -> None:
    result = subprocess.run(
        [
            _ffmpeg_executable(),
            "-hide_banner",
            "-loglevel",
            "info",
            "-i",
            str(path),
            "-vf",
            "signalstats,metadata=print:key=lavfi.signalstats.YAVG",
            "-f",
            "null",
            "-",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    values: list[float] = []
    for line in (result.stdout + result.stderr).splitlines():
        if "lavfi.signalstats.YAVG=" in line:
            try:
                values.append(float(line.rsplit("=", 1)[-1]))
            except ValueError:
                pass
    if (
        result.returncode
        or not values
        or max(values) - min(values) < 0.1
        or max(values) < 2.0
    ):
        raise DatasetError("augmented video is blank or cannot be decoded")


def assert_temporal_alignment(
    source: Path, augmented: Path, *, tolerance_s: float = 0.001
) -> dict[str, Any]:
    source_ts = _video_timestamps(source)
    augmented_ts = _video_timestamps(augmented)
    if len(source_ts) != len(augmented_ts):
        raise DatasetError(
            f"source/augmented frame-count mismatch ({len(source_ts)} != {len(augmented_ts)})"
        )
    maximum_error = max(
        abs(left - right) for left, right in zip(source_ts, augmented_ts, strict=True)
    )
    if maximum_error > tolerance_s:
        raise DatasetError(
            f"source/augmented timestamp mismatch ({maximum_error:.6f}s > {tolerance_s:.6f}s)"
        )
    _assert_nonblank(augmented)
    return {"frame_count": len(source_ts), "maximum_timestamp_error_s": maximum_error}


def _find_variant(manifest: dict[str, Any], variant_index: int) -> dict[str, Any]:
    variants: list[dict[str, Any]] = []
    for clip in manifest.get("clips", []):
        if isinstance(clip, dict):
            variants.extend(
                item for item in clip.get("variants", []) if isinstance(item, dict)
            )
    if not variants:
        variants = [
            item for item in manifest.get("variants", []) if isinstance(item, dict)
        ]
    if variant_index < 0 or variant_index >= len(variants):
        raise DatasetError("PAIDF variant index is outside the real transfer manifest")
    return variants[variant_index]


def materialize_paidf_dataset(
    *,
    dataset_uri: str,
    episode_index: int,
    paidf_run_uri: str,
    output_path: str,
    variant_index: int = 0,
    client: Any | None = None,
) -> dict[str, Any]:
    """Publish a derived version only after exact video sequence alignment."""

    client = client or _client()
    source_bucket, source_prefix, source_manifest = _dataset_manifest(
        client, dataset_uri
    )
    commit = _episode_commit(client, source_manifest, episode_index)
    paidf_bucket, paidf_prefix = split_s3_uri(paidf_run_uri, label="PAIDF run URI")
    lineage = _json_object(
        client, paidf_bucket, f"{paidf_prefix}/input/leisaac-lineage.json"
    )
    source_manifest_head = client.head_object(
        Bucket=source_bucket, Key=f"{source_prefix}/npa-dataset.json"
    )
    source_manifest_sha256 = _metadata_sha256(source_manifest_head)
    source = lineage.get("source", {})
    source_commit_uri = str(source.get("episode_commit_uri") or "")
    try:
        source_commit_bucket, source_commit_key = split_s3_uri(
            source_commit_uri, label="source episode commit URI"
        )
        source_commit_sha256 = _metadata_sha256(
            client.head_object(Bucket=source_commit_bucket, Key=source_commit_key)
        )
    except Exception as exc:
        raise DatasetError("source episode commit checksum is unavailable") from exc
    commit_frames = commit.get("objects", {}).get("frames")
    frame_checksums = (
        [str(item.get("sha256") or "") for item in commit_frames]
        if isinstance(commit_frames, list)
        else []
    )
    frame_checksums_sha256 = hashlib.sha256(
        json.dumps(frame_checksums, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        lineage.get("schema") != "npa.leisaac.paidf-input.v1"
        or source.get("dataset_uri") != dataset_uri.rstrip("/")
        or source.get("dataset_manifest_sha256") != source_manifest_sha256
        or source_commit_uri != str(source_manifest["episode_commits"][episode_index])
        or source.get("episode_commit_sha256") != source_commit_sha256
        or int(source.get("episode_index", -1)) != episode_index
        or source.get("episode_uuid") != commit.get("episode_uuid")
        or source.get("task") != commit["metadata"].get("task")
        or source.get("environment_id") != commit["metadata"].get("environment_id")
        or int(source.get("environment_index", -1))
        != int(commit["metadata"].get("environment_index", -2))
        or source.get("video_sha256") != commit["objects"]["video"]["sha256"]
        or source.get("records_sha256") != commit["objects"]["records"]["sha256"]
        or int(source.get("frame_count", -1))
        != int(commit["metadata"].get("frame_count", -2))
        or len(frame_checksums) != int(source.get("frame_count", -1))
        or source.get("frame_checksums_sha256") != frame_checksums_sha256
    ):
        raise DatasetError(
            "PAIDF input lineage does not match the selected source episode"
        )
    transfer_key = f"{paidf_prefix}/cosmos_augmented/manifest.json"
    transfer = _json_object(client, paidf_bucket, transfer_key)
    if (
        transfer.get("mode") != "cosmos_transfer2.5_gpu"
        or transfer.get("status") != "executed"
        or transfer.get("input_conditioned") is not True
        or str(transfer.get("conditioned_input") or "")
        != f"leisaac-episode-{episode_index:06d}.mp4"
        or str(transfer.get("control") or "") not in {"edge", "vis"}
    ):
        raise DatasetError(
            "PAIDF manifest does not prove real input-conditioned Cosmos Transfer execution"
        )
    variant = _find_variant(transfer, variant_index)
    augmented_uri = str(variant.get("augmented_video_uri") or "")
    augmented_bucket, augmented_key = split_s3_uri(
        augmented_uri, label="augmented video URI"
    )
    if augmented_bucket != paidf_bucket or not augmented_key.startswith(
        paidf_prefix + "/cosmos_augmented/"
    ):
        raise DatasetError("PAIDF variant is outside the selected immutable run")
    source_video = commit["objects"]["video"]
    with tempfile.TemporaryDirectory(prefix="npa-leisaac-paidf-") as temporary:
        root = Path(temporary)
        source_local = root / "source.mp4"
        augmented_local = root / "augmented.mp4"
        client.download_file(source_bucket, source_video["key"], str(source_local))
        client.download_file(augmented_bucket, augmented_key, str(augmented_local))
        if sha256_file(source_local) != source_video["sha256"]:
            raise DatasetError("downloaded source episode checksum mismatch")
        alignment = assert_temporal_alignment(source_local, augmented_local)
        augmented_sha = sha256_file(augmented_local)
        output_bucket, output_prefix = split_s3_uri(
            output_path, label="derived dataset output path"
        )
        if output_bucket == source_bucket and (
            output_prefix == source_prefix
            or output_prefix.startswith(source_prefix + "/")
        ):
            raise DatasetError(
                "derived output must not overwrite or nest inside the source version"
            )
        version = f"derived-{uuid.uuid4().hex}"
        derived_prefix = f"{output_prefix}/versions/{version}"
        source_items: list[dict[str, Any]] = []
        token = None
        while True:
            list_args: dict[str, Any] = {
                "Bucket": source_bucket,
                "Prefix": source_prefix + "/",
            }
            if token:
                list_args["ContinuationToken"] = token
            page = client.list_objects_v2(**list_args)
            source_items.extend(page.get("Contents", []))
            if not page.get("IsTruncated"):
                break
            token = page.get("NextContinuationToken")
        derived_files: list[dict[str, Any]] = []
        source_files = {
            str(item.get("key") or ""): item
            for item in source_manifest.get("files", [])
            if isinstance(item, dict)
        }
        for item in source_items:
            key = str(item["Key"])
            relative = key[len(source_prefix) + 1 :]
            if relative in {
                "npa-dataset.json",
                f"videos/{VIDEO_KEY}/chunk-000/file-{episode_index:03d}.mp4",
            }:
                continue
            client.copy_object(
                Bucket=output_bucket,
                Key=f"{derived_prefix}/{relative}",
                CopySource={"Bucket": source_bucket, "Key": key},
                MetadataDirective="COPY",
            )
            source_file = source_files.get(key, {})
            derived_files.append(
                {
                    "key": f"{derived_prefix}/{relative}",
                    "sha256": str(source_file.get("sha256") or ""),
                    "bytes": int(source_file.get("bytes") or item.get("Size") or 0),
                }
            )
        video_key = f"{derived_prefix}/videos/{VIDEO_KEY}/chunk-000/file-{episode_index:03d}.mp4"
        with augmented_local.open("rb") as handle:
            client.put_object(
                Bucket=output_bucket,
                Key=video_key,
                Body=handle,
                Metadata={"sha256": augmented_sha},
                IfNoneMatch="*",
            )
        derived_files.append(
            {
                "key": video_key,
                "sha256": augmented_sha,
                "bytes": augmented_local.stat().st_size,
            }
        )
        transfer_bytes = client.get_object(Bucket=paidf_bucket, Key=transfer_key)[
            "Body"
        ].read()
        derived_uri = f"s3://{output_bucket}/{derived_prefix}"
        derived_lineage = {
            "schema": DERIVED_SCHEMA,
            "created_at": utc_now(),
            "dataset_uri": derived_uri,
            "parent_dataset_uri": dataset_uri.rstrip("/"),
            "source_episode_index": episode_index,
            "source_video_sha256": source_video["sha256"],
            "paidf_run_uri": paidf_run_uri.rstrip("/"),
            "paidf_manifest_sha256": hashlib.sha256(transfer_bytes).hexdigest(),
            "variant_index": variant_index,
            "augmented_video_uri": augmented_uri,
            "augmented_video_sha256": augmented_sha,
            "augmentation_engine": "cosmos_transfer2.5_gpu",
            "input_conditioned": True,
            "alignment": alignment,
            "nonvisual_labels": "byte-identical parent Parquet records",
            "success_evidence": "inherited source operator outcome; augmentation adds none",
        }
        lineage_bytes = (
            json.dumps(derived_lineage, indent=2, sort_keys=True) + "\n"
        ).encode()
        client.put_object(
            Bucket=output_bucket,
            Key=f"{derived_prefix}/meta/npa-lineage.json",
            Body=lineage_bytes,
            Metadata={"sha256": hashlib.sha256(lineage_bytes).hexdigest()},
            IfNoneMatch="*",
        )
        derived_files.append(
            {
                "key": f"{derived_prefix}/meta/npa-lineage.json",
                "sha256": hashlib.sha256(lineage_bytes).hexdigest(),
                "bytes": len(lineage_bytes),
            }
        )
        manifest = dict(source_manifest)
        manifest.update(
            {
                "schema": "npa.leisaac.dataset.v1",
                "dataset_uri": derived_uri,
                "version": version,
                "created_at": utc_now(),
                "derived": derived_lineage,
                "files": sorted(derived_files, key=lambda item: item["key"]),
            }
        )
        manifest_bytes = (
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        ).encode()
        client.put_object(
            Bucket=output_bucket,
            Key=f"{derived_prefix}/npa-dataset.json",
            Body=manifest_bytes,
            Metadata={"sha256": hashlib.sha256(manifest_bytes).hexdigest()},
            IfNoneMatch="*",
        )
    return {
        "status": "materialized",
        "dataset_uri": derived_uri,
        "episode_index": episode_index,
        "variant_index": variant_index,
        "alignment": alignment,
        "augmentation_engine": "cosmos_transfer2.5_gpu",
        "input_conditioned": True,
        "lerobot_version": LEROBOT_TARGET_VERSION,
    }
