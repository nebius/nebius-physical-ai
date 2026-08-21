"""Real Cosmos 3 video-conditioned stages for the PAIDF composition.

The workflow remains the orchestration surface.  These functions provide the
non-trivial, unit-testable glue needed to select one LeRobot episode/camera,
run multiple real ``video2video`` generations, publish the established PAIDF
layout, and verify the final aggregate without embedding Python in YAML.
"""

from __future__ import annotations

import concurrent.futures
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Mapping
from urllib.parse import urlparse


INPUT_SCHEMA = "npa.paidf.cosmos3.input.v1"
MANIFEST_SCHEMA = "npa.paidf.cosmos3.augment.v1"
ATTEMPT_SCHEMA = "npa.paidf.cosmos3.refinement.v1"
FINAL_SCHEMA = "npa.paidf.cosmos3.final.v1"
ENGINE = "nvidia-cosmos/cosmos-framework"
VIDEO_MODE = "video2video"


class PaidfCosmos3Error(RuntimeError):
    """A PAIDF Cosmos 3 contract could not be satisfied."""


def validate_committed_augment_manifest(
    document: Any, output_uri: str = ""
) -> list[dict[str, Any]]:
    """Validate the canonical, source-conditioned Cosmos 3 publication.

    Consumers must follow this manifest rather than infer outputs by listing the
    augment prefix.  Keep the checks fail closed: the document must prove real
    video2video execution with guardrails, source lineage, non-empty artifacts,
    distinct seeds, and videos confined to their declared variant directories.
    """

    if not isinstance(document, dict):
        raise PaidfCosmos3Error("canonical Cosmos 3 augment manifest is not an object")
    required = {
        "schema": MANIFEST_SCHEMA,
        "engine": ENGINE,
        "status": "executed",
        "mode": VIDEO_MODE,
        "input_conditioned": True,
        "input_conditioning": "source-video",
        "conditioned_input": "source.mp4",
        "guardrails": True,
        "weights_baked": False,
    }
    invalid = [key for key, expected in required.items() if document.get(key) != expected]
    lineage = document.get("lineage")
    if not isinstance(lineage, dict) or not str(
        lineage.get("input_provenance_uri") or ""
    ).strip():
        invalid.append("lineage.input_provenance_uri")
    if not str(document.get("model") or "").strip():
        invalid.append("model")
    if invalid:
        raise PaidfCosmos3Error(
            "canonical Cosmos 3 augment manifest has invalid fields: "
            + ", ".join(invalid)
        )

    variants = document.get("variants")
    try:
        variant_count = int(document.get("variant_count", -1))
        total_bytes = int(document.get("video_bytes", 0))
        total_frames = int(document.get("frame_count", 0))
    except (TypeError, ValueError) as exc:
        raise PaidfCosmos3Error(
            "canonical Cosmos 3 augment manifest has invalid aggregate counts"
        ) from exc
    if (
        not isinstance(variants, list)
        or not variants
        or variant_count != len(variants)
        or total_bytes <= 0
        or total_frames <= 0
    ):
        raise PaidfCosmos3Error(
            "canonical Cosmos 3 augment manifest has inconsistent variants or counts"
        )

    root = output_uri.rstrip("/") + "/" if output_uri else ""
    if not root:
        first_uri = str(variants[0].get("augmented_video_uri") or "")
        marker = "/cosmos_augmented/"
        if marker in first_uri:
            root = first_uri.split(marker, 1)[0] + marker
    if not root:
        raise PaidfCosmos3Error(
            "canonical Cosmos 3 augment manifest output root is indeterminate"
        )

    seen_clips: set[str] = set()
    seen_seeds: set[int] = set()
    variant_bytes = 0
    variant_frames = 0
    for item in variants:
        if not isinstance(item, dict):
            raise PaidfCosmos3Error(
                "canonical Cosmos 3 augment manifest has an invalid variant"
            )
        clip = str(item.get("clip") or "").strip()
        video_uri = str(item.get("augmented_video_uri") or "").strip()
        try:
            seed = int(item["seed"])
            video_bytes = int(item.get("video_bytes", 0))
            frame_count = int(item.get("frame_count", 0))
        except (KeyError, TypeError, ValueError) as exc:
            raise PaidfCosmos3Error(
                "canonical Cosmos 3 augment manifest has invalid variant metadata"
            ) from exc
        expected_prefix = f"{root}{clip}/"
        if (
            not clip
            or clip in seen_clips
            or seed in seen_seeds
            or not video_uri.startswith(expected_prefix)
            or video_uri.rsplit("/", 1)[-1] != "augmented_video.mp4"
            or video_bytes <= 0
            or frame_count <= 0
        ):
            raise PaidfCosmos3Error(
                "canonical Cosmos 3 augment manifest has duplicated, empty, or "
                "out-of-prefix variant evidence"
            )
        seen_clips.add(clip)
        seen_seeds.add(seed)
        variant_bytes += video_bytes
        variant_frames += frame_count
    if variant_bytes != total_bytes or variant_frames != total_frames:
        raise PaidfCosmos3Error(
            "canonical Cosmos 3 augment manifest aggregate counts do not match variants"
        )
    return variants


def _storage() -> Any:
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _is_s3(value: str) -> bool:
    return str(value or "").startswith("s3://")


def _split_s3(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise PaidfCosmos3Error("expected an s3:// URI")
    return parsed.netloc, parsed.path.lstrip("/")


def _read_json(uri: str, *, storage: Any | None = None) -> Any:
    if _is_s3(uri):
        client = storage or _storage()
        with tempfile.TemporaryDirectory(prefix="npa-paidf-c3-json-") as tmp:
            local = Path(client.download_path(uri, str(Path(tmp) / "value.json")))
            if local.is_dir():
                matches = sorted(local.rglob(Path(uri).name))
                if not matches:
                    raise PaidfCosmos3Error("required JSON artifact is missing")
                local = matches[0]
            return json.loads(local.read_text(encoding="utf-8"))
    return json.loads(Path(uri).read_text(encoding="utf-8"))


def _write_json(
    payload: Mapping[str, Any], uri: str, *, storage: Any | None = None
) -> str:
    body = json.dumps(dict(payload), indent=2, sort_keys=True) + "\n"
    if _is_s3(uri):
        client = storage or _storage()
        with tempfile.TemporaryDirectory(prefix="npa-paidf-c3-json-") as tmp:
            local = Path(tmp) / (Path(uri).name or "value.json")
            local.write_text(body, encoding="utf-8")
            return str(client.upload_file(str(local), uri))
    path = Path(uri)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return str(path)


def _list_keys(uri: str, *, storage: Any | None = None) -> list[str]:
    if not _is_s3(uri):
        root = Path(uri)
        return [
            str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()
        ]
    client = storage or _storage()
    bucket, prefix = _split_s3(uri if uri.endswith("/") else uri + "/")
    token: str | None = None
    keys: list[str] = []
    while True:
        kwargs: dict[str, Any] = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kwargs["ContinuationToken"] = token
        page = client.s3.list_objects_v2(**kwargs)
        keys.extend(
            str(item["Key"]) for item in page.get("Contents", []) if item.get("Key")
        )
        if not page.get("IsTruncated"):
            return keys
        token = str(page.get("NextContinuationToken") or "") or None


def _missing_object_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    if not isinstance(response, dict):
        return isinstance(exc, FileNotFoundError)
    error = response.get("Error")
    metadata = response.get("ResponseMetadata")
    code = str(error.get("Code") or "") if isinstance(error, dict) else ""
    status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
    return code in {"404", "NoSuchKey", "NotFound"} or status == 404


def _artifact_size(uri: str, *, storage: Any | None = None) -> int:
    if not _is_s3(uri):
        path = Path(uri)
        return path.stat().st_size if path.is_file() else 0
    client = storage or _storage()
    bucket, key = _split_s3(uri)
    try:
        response = client.s3.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        if _missing_object_error(exc):
            return 0
        raise
    return int(response.get("ContentLength") or 0)


def _download_source(
    uri: str, destination: Path, *, storage: Any | None = None
) -> Path:
    if _is_s3(uri):
        result = Path((storage or _storage()).download_path(uri, str(destination)))
        return result
    source = Path(uri.removeprefix("file://"))
    if not source.exists():
        raise PaidfCosmos3Error("configured input does not exist")
    if source.is_dir():
        shutil.copytree(source, destination, dirs_exist_ok=True)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _video_features(info: Mapping[str, Any]) -> list[str]:
    features = info.get("features")
    if not isinstance(features, dict):
        return []
    return sorted(
        str(name)
        for name, contract in features.items()
        if isinstance(contract, dict) and str(contract.get("dtype") or "") == "video"
    )


def _camera_feature(info: Mapping[str, Any], camera: str) -> str:
    choices = _video_features(info)
    requested = str(camera or "").strip()
    if not choices:
        raise PaidfCosmos3Error("LeRobot metadata declares no video camera features")
    if requested in choices:
        return requested
    suffix_matches = [name for name in choices if name.rsplit(".", 1)[-1] == requested]
    if len(suffix_matches) == 1:
        return suffix_matches[0]
    if not requested and len(choices) == 1:
        return choices[0]
    raise PaidfCosmos3Error(
        "camera selector must name exactly one LeRobot video feature; available: "
        + ", ".join(choices)
    )


def _episode_rows(root: Path) -> list[dict[str, Any]]:
    import pyarrow.parquet as pq

    rows: list[dict[str, Any]] = []
    for path in sorted((root / "meta" / "episodes").glob("chunk-*/file-*.parquet")):
        rows.extend(pq.read_table(path).to_pylist())
    if rows:
        return rows
    legacy = root / "meta" / "episodes.jsonl"
    if legacy.is_file():
        return [
            json.loads(line)
            for line in legacy.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    return []


def _select_lerobot_video(
    root: Path, episode: int, camera: str
) -> tuple[Path, tuple[float, float] | None, str]:
    info_path = root / "meta" / "info.json"
    if not info_path.is_file():
        raise PaidfCosmos3Error("LeRobot dataset is missing meta/info.json")
    info = json.loads(info_path.read_text(encoding="utf-8"))
    version = str(info.get("codebase_version") or "")
    if not (version.startswith("v2") or version.startswith("v3")):
        raise PaidfCosmos3Error("LeRobot codebase_version must be v2.x or v3.x")
    feature = _camera_feature(info, camera)
    rows = _episode_rows(root)
    row = next(
        (item for item in rows if int(item.get("episode_index", -1)) == episode), {}
    )
    if rows and not row:
        raise PaidfCosmos3Error(
            "selected LeRobot episode is absent from episode metadata"
        )
    chunk_index = int(row.get(f"videos/{feature}/chunk_index", 0) or 0)
    file_index = int(row.get(f"videos/{feature}/file_index", episode) or episode)
    pattern = str(
        info.get("video_path")
        or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    )
    values = {
        "video_key": feature,
        "chunk_index": chunk_index,
        "file_index": file_index,
        "episode_chunk": episode // int(info.get("chunks_size", 1000) or 1000),
        "episode_index": episode,
    }
    candidates = [
        root / pattern.format(**values),
        root
        / "videos"
        / feature
        / f"chunk-{chunk_index:03d}"
        / f"file-{file_index:03d}.mp4",
        root
        / "videos"
        / f"chunk-{values['episode_chunk']:03d}"
        / feature
        / f"episode_{episode:06d}.mp4",
    ]
    source = next((path for path in candidates if path.is_file()), None)
    if source is None:
        raise PaidfCosmos3Error("selected LeRobot episode/camera video is missing")
    start = row.get(f"videos/{feature}/from_timestamp")
    end = row.get(f"videos/{feature}/to_timestamp")
    timestamps: tuple[float, float] | None = None
    if start is not None or end is not None:
        if (
            start is None
            or end is None
            or float(start) < 0
            or float(end) <= float(start)
        ):
            raise PaidfCosmos3Error(
                "LeRobot episode video timestamps are incomplete or invalid"
            )
        timestamps = (float(start), float(end))
    return source, timestamps, feature


def _normalize_video(
    source: Path, destination: Path, timestamps: tuple[float, float] | None = None
) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PaidfCosmos3Error("ffmpeg is required to prepare the selected video")
    argv = [ffmpeg, "-y", "-v", "error"]
    if timestamps:
        argv.extend(["-ss", str(timestamps[0]), "-to", str(timestamps[1])])
    argv.extend(
        [
            "-i",
            str(source),
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ]
    )
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    if (
        completed.returncode
        or not destination.is_file()
        or destination.stat().st_size <= 0
    ):
        detail = str(
            completed.stderr or completed.stdout or "ffmpeg produced no output"
        )[:300]
        raise PaidfCosmos3Error(f"selected video could not be normalized: {detail}")


def _extract_frames(video: Path, destination: Path, count: int = 8) -> list[Path]:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PaidfCosmos3Error(
            "ffmpeg is required to extract caption/evaluator frames"
        )
    destination.mkdir(parents=True, exist_ok=True)
    argv = [
        ffmpeg,
        "-y",
        "-v",
        "error",
        "-i",
        str(video),
        "-vf",
        "fps=1,scale='min(1280,iw)':-2",
        "-frames:v",
        str(max(1, count)),
        str(destination / "frame-%05d.png"),
    ]
    completed = subprocess.run(argv, capture_output=True, text=True, check=False)
    frames = sorted(destination.glob("frame-*.png"))
    if completed.returncode or not frames:
        raise PaidfCosmos3Error(
            "selected video did not produce caption/evaluator frames"
        )
    return frames


def prepare_input(
    input_kind: str,
    input_video_uri: str,
    lerobot_dataset_uri: str,
    episode: str | int,
    camera: str,
    input_uri: str,
    provenance_uri: str,
    run_id: str = "",
    *,
    storage: Any | None = None,
) -> dict[str, Any]:
    """Select and stage one direct video or LeRobot v2/v3 episode/camera."""

    kind = str(input_kind or "").strip().lower()
    if kind not in {"video", "lerobot"}:
        raise PaidfCosmos3Error("input_kind must be 'video' or 'lerobot'")
    try:
        episode_index = int(episode)
    except (TypeError, ValueError) as exc:
        raise PaidfCosmos3Error("input episode must be a non-negative integer") from exc
    if episode_index < 0:
        raise PaidfCosmos3Error("input episode must be a non-negative integer")
    selected_uri = input_video_uri if kind == "video" else lerobot_dataset_uri
    if not str(selected_uri or "").strip():
        raise PaidfCosmos3Error(f"input_kind={kind} requires its corresponding URI")
    client = storage or (
        _storage() if _is_s3(selected_uri) or _is_s3(input_uri) else None
    )
    with tempfile.TemporaryDirectory(prefix="npa-paidf-c3-input-") as tmp:
        root = Path(tmp)
        timestamps: tuple[float, float] | None = None
        feature = ""
        if kind == "video":
            source = _download_source(
                selected_uri, root / "selected.mp4", storage=client
            )
            if source.is_dir():
                videos = sorted(source.rglob("*.mp4"))
                if len(videos) != 1:
                    raise PaidfCosmos3Error(
                        "video input URI must resolve to exactly one MP4"
                    )
                source = videos[0]
        else:
            dataset = _download_source(selected_uri, root / "dataset", storage=client)
            if dataset.is_file():
                raise PaidfCosmos3Error(
                    "LeRobot dataset URI must resolve to a directory"
                )
            source, timestamps, feature = _select_lerobot_video(
                dataset, episode_index, camera
            )
        canonical = root / "source.mp4"
        _normalize_video(source, canonical, timestamps)
        frames = _extract_frames(canonical, root / "frames")
        base = input_uri if input_uri.endswith("/") else input_uri + "/"
        if _is_s3(base):
            assert client is not None
            client.upload_file(str(canonical), base + "source.mp4")
            for frame in frames:
                client.upload_file(str(frame), base + frame.name)
        else:
            output = Path(base)
            output.mkdir(parents=True, exist_ok=True)
            shutil.copy2(canonical, output / "source.mp4")
            for frame in frames:
                shutil.copy2(frame, output / frame.name)
        payload = {
            "schema": INPUT_SCHEMA,
            "status": "prepared",
            "source_kind": "lerobot_dataset" if kind == "lerobot" else "video_uri",
            "lerobot_version_family": "v2/v3" if kind == "lerobot" else "",
            "episode": episode_index if kind == "lerobot" else None,
            "camera": feature if kind == "lerobot" else "",
            "staged_video_uri": base + "source.mp4",
            "conditioned_input": "source.mp4",
            "sha256": _sha256(canonical),
            "video_bytes": canonical.stat().st_size,
            "frame_count": len(frames),
            "run_id": str(run_id or ""),
        }
        payload["written_uri"] = _write_json(payload, provenance_uri, storage=client)
    print(
        json.dumps(
            {
                "stage": "prepare-input",
                "status": "prepared",
                "frame_count": payload["frame_count"],
            }
        )
    )
    return payload


def _complete_evaluator_report(report: Any) -> dict[str, Any]:
    if not isinstance(report, dict):
        raise PaidfCosmos3Error("prior evaluator report is not a JSON object")
    if report.get("status") != "completed" or not isinstance(
        report.get("passed"), bool
    ):
        raise PaidfCosmos3Error(
            "prior evaluator report is incomplete or has no boolean decision"
        )
    try:
        float(report["score"])
    except (KeyError, TypeError, ValueError) as exc:
        raise PaidfCosmos3Error("prior evaluator report has no numeric score") from exc
    return report


def _load_attempt(
    attempt_uri: str, scores_uri: str, *, storage: Any | None = None
) -> tuple[int, dict[str, Any] | None]:
    if not _is_s3(attempt_uri) and not Path(attempt_uri).is_file():
        return 0, None
    if _is_s3(attempt_uri):
        client = storage or _storage()
        bucket, key = _split_s3(attempt_uri)
        try:
            client.s3.head_object(Bucket=bucket, Key=key)
        except Exception as exc:
            if _missing_object_error(exc):
                return 0, None
            raise
    try:
        current = _read_json(attempt_uri, storage=storage)
    except Exception as exc:
        raise PaidfCosmos3Error(
            "refinement attempt artifact exists but cannot be read"
        ) from exc
    if not isinstance(current, dict) or current.get("schema") != ATTEMPT_SCHEMA:
        raise PaidfCosmos3Error("refinement attempt artifact is malformed")
    try:
        attempt = int(current["attempt"]) + 1
    except (KeyError, TypeError, ValueError) as exc:
        raise PaidfCosmos3Error("refinement attempt number is malformed") from exc
    report_uri = (
        scores_uri
        if scores_uri.endswith(".json")
        else scores_uri.rstrip("/") + "/cosmos_evaluator.json"
    )
    return attempt, _complete_evaluator_report(_read_json(report_uri, storage=storage))


def _visible_gpu_ids(environ: Mapping[str, str] | None = None) -> list[str]:
    env = environ if environ is not None else os.environ
    visible = str(env.get("CUDA_VISIBLE_DEVICES", "") or "").strip()
    if visible and visible not in {"-1", "none", "void"}:
        return [item.strip() for item in visible.split(",") if item.strip()]
    try:
        completed = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError:
        return []
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def _caption_text(payload: Any) -> str:
    if not isinstance(payload, dict) or not isinstance(payload.get("captions"), list):
        raise PaidfCosmos3Error("original caption report is missing or malformed")
    captions = [
        str(item.get("caption") or "").strip()
        for item in payload["captions"]
        if isinstance(item, dict)
    ]
    captions = [item for item in captions if item]
    if not captions:
        raise PaidfCosmos3Error("original caption report contains no captions")
    return " ".join(captions[:4])


def _preserve_source_motion(
    source: Path,
    generated: Path,
    destination: Path,
    *,
    source_weight: float,
) -> None:
    """Composite Cosmos appearance onto the source motion and camera geometry.

    Cosmos 3 Nano can produce a useful appearance treatment while drifting the
    camera or foreground trajectory.  This post-process retains the generated
    pixels, but anchors them to the source stream's resolution, frame cadence,
    and motion.  The unmodified framework artifact is published beside the
    composite so the transformation remains independently auditable.
    """

    if not 0.0 < source_weight < 1.0:
        raise PaidfCosmos3Error(
            "source motion weight must be strictly between 0 and 1"
        )
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise PaidfCosmos3Error(
            "ffmpeg is required for source-motion-preserving publication"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    cosmos_weight = 1.0 - source_weight
    filter_graph = (
        "[1:v][0:v]scale2ref[model][source];"
        f"[source][model]blend=all_expr='A*{source_weight:.6f}+"
        f"B*{cosmos_weight:.6f}':shortest=1[out]"
    )
    completed = subprocess.run(
        [
            ffmpeg,
            "-y",
            "-v",
            "error",
            "-i",
            str(source),
            "-i",
            str(generated),
            "-filter_complex",
            filter_graph,
            "-map",
            "[out]",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(destination),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if (
        completed.returncode
        or not destination.is_file()
        or destination.stat().st_size <= 0
    ):
        detail = str(
            completed.stderr or completed.stdout or "ffmpeg produced no output"
        )[:300]
        raise PaidfCosmos3Error(
            f"source-motion-preserving publication failed: {detail}"
        )


def _publish_variant(
    *,
    result: Mapping[str, Any],
    output_uri: str,
    clip: str,
    variables: Mapping[str, Any],
    metadata: Mapping[str, Any],
    storage: Any,
    source_video: Path,
    source_motion_weight: float,
) -> dict[str, Any]:
    if not _is_s3(output_uri):
        raise PaidfCosmos3Error(
            "PAIDF Cosmos 3 variant publication requires an s3:// output URI"
        )
    base = output_uri.rstrip("/") + f"/{clip}/"
    raw_artifact = Path(str(result["output_path"]))
    if not raw_artifact.is_file() or raw_artifact.stat().st_size <= 0:
        raise PaidfCosmos3Error("Cosmos 3 returned an empty video artifact")
    artifact = raw_artifact
    postprocess: dict[str, Any] | None = None
    if source_motion_weight:
        artifact = raw_artifact.with_name("motion-preserved.mp4")
        _preserve_source_motion(
            source_video,
            raw_artifact,
            artifact,
            source_weight=source_motion_weight,
        )
        raw_uri = base + "raw_cosmos_video.mp4"
        storage.upload_file(str(raw_artifact), raw_uri)
        postprocess = {
            "engine": "ffmpeg-source-motion-composite",
            "source_weight": source_motion_weight,
            "cosmos_weight": 1.0 - source_motion_weight,
            "raw_cosmos_video_uri": raw_uri,
            "raw_cosmos_video_bytes": raw_artifact.stat().st_size,
            "raw_cosmos_video_sha256": _sha256(raw_artifact),
            "published_video_sha256": _sha256(artifact),
        }
    storage.upload_file(str(artifact), base + "augmented_video.mp4")
    with tempfile.TemporaryDirectory(prefix="npa-paidf-c3-publish-") as tmp:
        frames = _extract_frames(artifact, Path(tmp) / "frames")
        for frame in frames:
            storage.upload_file(str(frame), base + frame.name)
        clip_meta = {
            "schema": MANIFEST_SCHEMA,
            "status": "executed",
            "engine": ENGINE,
            "mode": VIDEO_MODE,
            "clip": clip,
            "variables": dict(variables),
            "prompt": str(metadata["prompt"]),
            "input_conditioned": True,
            "input_conditioning": "source-video",
            "conditioned_input": "source.mp4",
            "model": str(metadata["model"]),
            "seed": int(metadata["seed"]),
            "guidance": float(metadata["guidance"]),
            "steps": int(metadata["steps"]),
            "guardrails": bool(metadata["guardrails"]),
            "weights_baked": False,
            "attempt": int(metadata["attempt"]),
            "lineage": {"input_provenance_uri": str(metadata["input_provenance_uri"])},
            "video_bytes": artifact.stat().st_size,
            "frame_count": len(frames),
            "motion_preservation": postprocess,
        }
        _write_json(clip_meta, base + "metadata.json", storage=storage)
    return {
        "clip": clip,
        "augmented_video_uri": base + "augmented_video.mp4",
        "video_bytes": artifact.stat().st_size,
        "frame_count": clip_meta["frame_count"],
        "seed": clip_meta["seed"],
        "guidance": clip_meta["guidance"],
        "steps": clip_meta["steps"],
        "variables": dict(variables),
        "motion_preservation": postprocess,
    }


def generate_variants(
    input_video_uri: str,
    input_provenance_uri: str,
    captions_uri: str,
    configs_uri: str,
    output_uri: str,
    scores_uri: str,
    attempt_uri: str,
    mode: str,
    checkpoint: str,
    prompt: str,
    negative_prompt: str,
    seed: str | int,
    guidance: str | float,
    steps: str | int,
    variant_count: str | int,
    variant_parallelism: str | int,
    retry_seed_stride: str | int,
    retry_guidance_delta: str | float,
    retry_steps_delta: str | int,
    parallelism_preset: str,
    guardrails: str | bool,
    run_id: str,
    source_motion_weight: str | float = 0.0,
    *,
    storage: Any | None = None,
    environ: Mapping[str, str] | None = None,
    generator: Any | None = None,
) -> dict[str, Any]:
    """Run one real Cosmos 3 video2video inference per configured variant."""

    if not _is_s3(output_uri):
        raise PaidfCosmos3Error(
            "PAIDF Cosmos 3 generate-variants publication requires an s3:// output URI"
        )
    if str(mode) != VIDEO_MODE:
        raise PaidfCosmos3Error("PAIDF Cosmos 3 generation must use video2video")
    enabled = str(guardrails).strip().lower() in {"1", "true", "yes", "on"}
    if not enabled:
        raise PaidfCosmos3Error("PAIDF Cosmos 3 guardrails must remain enabled")
    try:
        base_seed = int(seed)
        base_guidance = float(guidance)
        base_steps = int(steps)
        count = int(variant_count)
        requested_parallelism = int(variant_parallelism)
        seed_stride = int(retry_seed_stride)
        guidance_delta = float(retry_guidance_delta)
        steps_delta = int(retry_steps_delta)
        motion_weight = float(source_motion_weight)
    except (TypeError, ValueError) as exc:
        raise PaidfCosmos3Error(
            "Cosmos 3 generation and retry settings must be numeric"
        ) from exc
    if count < 1 or requested_parallelism < 1 or base_steps < 1:
        raise PaidfCosmos3Error(
            "variant count, parallelism, and steps must be positive"
        )
    if motion_weight and not 0.0 < motion_weight < 1.0:
        raise PaidfCosmos3Error(
            "source motion weight must be zero or strictly between 0 and 1"
        )
    client = storage or _storage()
    attempt, prior = _load_attempt(attempt_uri, scores_uri, storage=client)
    if attempt and seed_stride == 0 and guidance_delta == 0 and steps_delta == 0:
        raise PaidfCosmos3Error(
            "a refinement retry must change seed, guidance, or steps"
        )
    effective_guidance = base_guidance + attempt * guidance_delta
    effective_steps = base_steps + attempt * steps_delta
    if effective_guidance <= 0 or effective_steps < 1:
        raise PaidfCosmos3Error(
            "effective retry guidance/steps are outside supported bounds"
        )
    gpu_ids = _visible_gpu_ids(environ)
    if not gpu_ids:
        raise PaidfCosmos3Error("Cosmos 3 generation requires a visible GPU")
    concurrency = min(count, requested_parallelism, len(gpu_ids))
    config_manifest = _read_json(
        configs_uri.rstrip("/") + "/manifest.json", storage=client
    )
    combos = (
        config_manifest.get("augmentations")
        if isinstance(config_manifest, dict)
        else None
    )
    if not isinstance(combos, list) or len(combos) < count:
        raise PaidfCosmos3Error(
            "config manifest has fewer augmentation prompts than variant_count"
        )
    caption = _caption_text(
        _read_json(captions_uri.rstrip("/") + "/captions.json", storage=client)
    )
    provenance = _read_json(input_provenance_uri, storage=client)
    if not isinstance(provenance, dict) or provenance.get("status") != "prepared":
        raise PaidfCosmos3Error("input provenance is missing or incomplete")
    from npa.workbench.cosmos.generate import (
        generate_and_publish,
        materialize_vision_input,
    )

    run_generate = generator or generate_and_publish
    local_input = materialize_vision_input(input_video_uri)
    work_root = Path(tempfile.mkdtemp(prefix="npa-paidf-c3-generate-"))

    def run_one(index: int) -> tuple[int, dict[str, Any], dict[str, Any], str]:
        combo = dict(combos[index])
        variant_prompt = f"{str(prompt).strip()} Source understanding: {caption}. {str(combo.get('prompt') or '').strip()}"
        variant_seed = base_seed + attempt * seed_stride + index
        env = dict(environ if environ is not None else os.environ)
        env["CUDA_VISIBLE_DEVICES"] = gpu_ids[index % concurrency]
        result = run_generate(
            mode=VIDEO_MODE,
            prompt=variant_prompt,
            name=f"variant-{index:04d}",
            checkpoint=checkpoint,
            input_path=local_input,
            output_path=str(work_root / f"variant-{index:04d}"),
            negative_prompt=negative_prompt,
            seed=variant_seed,
            num_steps=effective_steps,
            guidance=effective_guidance,
            no_guardrails=False,
            parallelism_preset=parallelism_preset,
            run_id=run_id,
            environ=env,
        )
        return index, result, combo, variant_prompt

    generated: list[tuple[int, dict[str, Any], dict[str, Any], str]] = []
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(run_one, index) for index in range(count)]
            for future in concurrent.futures.as_completed(futures):
                generated.append(future.result())
        variants: list[dict[str, Any]] = []
        for index, result, combo, variant_prompt in sorted(generated):
            variants.append(
                _publish_variant(
                    result=result,
                    output_uri=output_uri,
                    clip=f"variant-{index:04d}",
                    variables=combo,
                    metadata={
                        "prompt": variant_prompt,
                        "model": checkpoint,
                        "seed": base_seed + attempt * seed_stride + index,
                        "guidance": effective_guidance,
                        "steps": effective_steps,
                        "guardrails": True,
                        "attempt": attempt,
                        "input_provenance_uri": input_provenance_uri,
                    },
                    storage=client,
                    source_video=Path(local_input),
                    source_motion_weight=motion_weight,
                )
            )
    finally:
        shutil.rmtree(work_root, ignore_errors=True)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "executed",
        "engine": ENGINE,
        "mode": VIDEO_MODE,
        "input_conditioned": True,
        "input_conditioning": "source-video",
        "conditioned_input": "source.mp4",
        "model": checkpoint,
        "guardrails": True,
        "weights_baked": False,
        "attempt": attempt,
        "variant_count": len(variants),
        "variant_parallelism": concurrency,
        "variants": variants,
        "video_bytes": sum(int(item["video_bytes"]) for item in variants),
        "frame_count": sum(int(item["frame_count"]) for item in variants),
        "lineage": {
            "input_provenance_uri": input_provenance_uri,
            "captions_uri": captions_uri,
        },
        "run_id": run_id,
        "motion_preservation": {
            "enabled": bool(motion_weight),
            "source_weight": motion_weight,
            "cosmos_weight": 1.0 - motion_weight if motion_weight else 1.0,
            "raw_cosmos_outputs_preserved": bool(motion_weight),
        },
    }
    _write_json(manifest, output_uri.rstrip("/") + "/manifest.json", storage=client)
    attempt_payload = {
        "schema": ATTEMPT_SCHEMA,
        "attempt": attempt,
        "base_seed": base_seed,
        "effective_seed_start": base_seed + attempt * seed_stride,
        "effective_guidance": effective_guidance,
        "effective_steps": effective_steps,
        "changed_from_prior": bool(attempt),
        "prior_score": float(prior["score"]) if prior else None,
        "prior_passed": bool(prior["passed"]) if prior else None,
    }
    _write_json(attempt_payload, attempt_uri, storage=client)
    _write_json(
        attempt_payload,
        attempt_uri.rsplit(".", 1)[0] + f"-{attempt:02d}.json",
        storage=client,
    )
    print(
        json.dumps(
            {
                "stage": "cosmos3-generate",
                "status": "executed",
                "attempt": attempt,
                "variant_count": len(variants),
            }
        )
    )
    return manifest


def reject_quality(disposition_uri: str) -> None:
    disposition = _read_json(disposition_uri)
    if (
        not isinstance(disposition, dict)
        or disposition.get("quality_status") != "rejected"
    ):
        raise PaidfCosmos3Error(
            "reject terminal requires a durable rejected disposition"
        )
    raise PaidfCosmos3Error(
        "quality rejected after bounded refinement; evidence was preserved"
    )


def route_quality_disposition(disposition_uri: str, decision_uri: str) -> str:
    """Route only from the durable evaluator-derived disposition."""

    from npa.orchestration.npa_workflow.decisions import write_decision

    try:
        disposition = _read_json(disposition_uri)
    except Exception as exc:
        raise PaidfCosmos3Error(
            "quality disposition is missing or unreadable"
        ) from exc
    if not isinstance(disposition, dict):
        raise PaidfCosmos3Error("quality disposition is not a JSON object")
    quality_status = disposition.get("quality_status")
    recorded_decision = disposition.get("decision")
    expected = (
        "promote_checkpoint" if quality_status == "accepted" else "loop_back"
    )
    if quality_status not in {"accepted", "rejected"}:
        raise PaidfCosmos3Error("quality disposition has an inconsistent decision")
    if recorded_decision is None:
        # Runs started by the pre-decision disposition writer can reach this state
        # after an operator repairs and resumes the driver.  Persist the uniquely
        # derivable route before continuing so downstream acceptance checks still
        # consume one durable, self-consistent disposition.  A present conflicting
        # value remains a hard failure below.
        disposition["decision"] = expected
        _write_json(disposition, disposition_uri)
    elif recorded_decision != expected:
        raise PaidfCosmos3Error("quality disposition has an inconsistent decision")
    write_decision(decision_uri, expected)
    return expected


def require_accepted_quality(disposition_uri: str) -> None:
    """Stop a statically promoted one-shot plan before downstream work."""

    try:
        disposition = _read_json(disposition_uri)
    except Exception as exc:
        raise PaidfCosmos3Error(
            "accepted quality disposition is missing or unreadable"
        ) from exc
    if not isinstance(disposition, dict):
        raise PaidfCosmos3Error("accepted quality disposition is not a JSON object")
    if (
        disposition.get("quality_status") != "accepted"
        or disposition.get("decision") != "promote_checkpoint"
        or disposition.get("evaluator_status") != "completed"
        or disposition.get("hard_checks_passed") is not True
    ):
        raise PaidfCosmos3Error(
            "annotation requires a complete accepted evaluator disposition"
        )


def finalize(
    run_root_uri: str, report_uri: str, *, storage: Any | None = None
) -> dict[str, Any]:
    """Fail closed unless every advertised downstream component produced evidence."""

    client = storage or (_storage() if _is_s3(run_root_uri) else None)
    root = run_root_uri.rstrip("/") + "/"
    manifest = _read_json(root + "cosmos_augmented/manifest.json", storage=client)
    if not isinstance(manifest, dict):
        raise PaidfCosmos3Error("Cosmos 3 manifest is not a JSON object")
    evaluator = _complete_evaluator_report(
        _read_json(root + "grade/cosmos_evaluator.json", storage=client)
    )
    disposition = _read_json(root + "grade/quality_disposition.json", storage=client)
    curator = _read_json(root + "curation/cosmos_curator.json", storage=client)
    fiftyone = _read_json(root + "curation/report.json", storage=client)
    if not isinstance(disposition, dict):
        raise PaidfCosmos3Error("quality disposition is not a JSON object")
    if not isinstance(curator, dict):
        raise PaidfCosmos3Error("Cosmos Curator report is not a JSON object")
    if not isinstance(fiftyone, dict):
        raise PaidfCosmos3Error("FiftyOne report is not a JSON object")
    required_manifest = {
        "schema": MANIFEST_SCHEMA,
        "status": "executed",
        "engine": ENGINE,
        "mode": VIDEO_MODE,
        "input_conditioned": True,
        "input_conditioning": "source-video",
        "conditioned_input": "source.mp4",
        "guardrails": True,
        "weights_baked": False,
    }
    missing_or_invalid = [
        field
        for field, expected in required_manifest.items()
        if manifest.get(field) != expected
    ]
    model = manifest.get("model")
    lineage = manifest.get("lineage")
    variants = manifest.get("variants")
    try:
        variant_count = int(manifest.get("variant_count", 0))
        video_bytes = int(manifest.get("video_bytes", 0))
    except (TypeError, ValueError) as exc:
        raise PaidfCosmos3Error(
            "Cosmos 3 manifest has invalid variant_count or video_bytes"
        ) from exc
    if not isinstance(model, str) or not model.strip():
        missing_or_invalid.append("model")
    if (
        not isinstance(lineage, dict)
        or not str(lineage.get("input_provenance_uri") or "").strip()
    ):
        missing_or_invalid.append("lineage.input_provenance_uri")
    variants_valid = isinstance(variants, list) and len(variants) == variant_count
    if variants_valid:
        for item in variants:
            try:
                item_bytes = int(item.get("video_bytes", 0)) if isinstance(item, dict) else 0
            except (TypeError, ValueError):
                item_bytes = 0
            if (
                not isinstance(item, dict)
                or not str(item.get("clip") or "").strip()
                or not str(item.get("augmented_video_uri") or "").strip()
                or item_bytes <= 0
            ):
                variants_valid = False
                break
    if not variants_valid:
        missing_or_invalid.append("variants")
    if missing_or_invalid or variant_count < 1 or video_bytes <= 0:
        raise PaidfCosmos3Error(
            "Cosmos 3 manifest does not prove real non-empty framework output; "
            f"missing or invalid fields: {', '.join(missing_or_invalid) or 'counts'}"
        )
    if (
        disposition.get("quality_status") != "accepted"
        or evaluator.get("passed") is not True
    ):
        raise PaidfCosmos3Error(
            "finalization requires an accepted complete evaluator result"
        )
    if (
        str(curator.get("engine") or "") in {"", "unavailable"}
        or int(curator.get("clip_count", 0)) < 1
    ):
        raise PaidfCosmos3Error(
            "Cosmos Curator report does not prove real curated clips"
        )
    if fiftyone.get("curation_engine") != "fiftyone-brain":
        raise PaidfCosmos3Error("FiftyOne report does not prove real Brain curation")
    keys = _list_keys(root, storage=client)
    rrd_uri = root + "reports/sim2real.rrd"
    if not any(key.endswith("reports/sim2real.rrd") for key in keys) or _artifact_size(
        rrd_uri, storage=client
    ) <= 0:
        raise PaidfCosmos3Error("Rerun recording is missing or empty")
    payload = {
        "schema": FINAL_SCHEMA,
        "status": "completed",
        "engine": ENGINE,
        "mode": VIDEO_MODE,
        "input_conditioned": True,
        "model": model,
        "guardrails": True,
        "variant_count": variant_count,
        "video_bytes": video_bytes,
        "evaluator_score": float(evaluator["score"]),
        "curated_clip_count": int(curator["clip_count"]),
        "fiftyone_engine": fiftyone["curation_engine"],
        "has_rrd": True,
        "artifact_count": len(keys),
    }
    payload["written_uri"] = _write_json(payload, report_uri, storage=client)
    print(
        json.dumps(
            {
                "stage": "finalize",
                "status": "completed",
                "variant_count": payload["variant_count"],
                "evaluator_score": payload["evaluator_score"],
            }
        )
    )
    return payload


__all__ = [
    "ATTEMPT_SCHEMA",
    "ENGINE",
    "FINAL_SCHEMA",
    "INPUT_SCHEMA",
    "MANIFEST_SCHEMA",
    "PaidfCosmos3Error",
    "finalize",
    "generate_variants",
    "prepare_input",
    "reject_quality",
    "require_accepted_quality",
    "route_quality_disposition",
]
