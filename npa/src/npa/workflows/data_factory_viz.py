"""Build a Rerun ``.rrd`` recording for a Physical AI Data Factory run.

The data-factory blueprint emits ``.mp4`` / ``.json`` artifacts, which the NPA
agent renders as video/json but NOT in the embedded Rerun viewer (that needs an
``.rrd``). This module logs a run's input frames, augmented frames, and captions
as Rerun streams and writes ``reports/sim2real.rrd`` so the run is viewable in
the agent's embedded Rerun panel (the agent prefers ``reports/sim2real.rrd``).

Kept dependency-light and importable so the blueprint's ``visualize`` stage can
call it inline (``python -c "from npa.workflows.data_factory_viz import
build_run_rrd; build_run_rrd(input_uri, output_uri)"``) in a task where ``npa``
is pip-installed.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient

_log = logging.getLogger(__name__)

APPLICATION_ID = "physical-ai-data-factory"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

#: Run sub-directories materialized from S3 before building a recording. Covers
#: both producers: the data-factory blueprint (input/cosmos_augmented/
#: cosmos_control/labeled_*/configs/grade/curation) and the NuRec
#: neural-reconstruction workflow (ncore/reconstruction/novel_views). Missing
#: subtrees are skipped.
RUN_SUBDIRS = (
    "input",
    "cosmos_augmented",
    "cosmos_control",
    "labeled_original",
    "labeled_augmented",
    "configs",
    "grade",
    "curation",
    "ncore",
    "reconstruction",
    "novel_views",
    "reports",
)


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


# Keep the .rrd small so the browser Rerun viewer loads fast. Full-res raw-RGB
# frames made a ~23 MB recording; downscaling + JPEG-encoding + subsampling frames
# cuts that ~10-20x with no meaningful loss for a review viewer. All overridable.
RRD_MAX_FRAME_DIM = _int_env("NPA_RRD_MAX_DIM", 512)
RRD_JPEG_QUALITY = _int_env("NPA_RRD_JPEG_QUALITY", 75)
RRD_MAX_FRAMES_PER_ENTITY = _int_env("NPA_RRD_MAX_FRAMES", 24)


class DataFactoryVizError(RuntimeError):
    """Raised when the Rerun recording cannot be built."""


def _run_id_from_uri(uri: str) -> str:
    return uri.rstrip("/").split("/")[-1] or APPLICATION_ID


def _frame_index(stem: str) -> int:
    # Parse the trailing frame number irrespective of the delimiter used by the
    # producer: both "video_0_frame_01" and "frame-00000" must yield a distinct
    # per-frame index so Rerun logs an animated sequence instead of collapsing
    # every frame onto time-sequence 0.
    m = re.search(r"(\d+)\D*$", stem)
    return int(m.group(1)) if m else 0


def _load_rgb(path: Path):
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        # Downscale to a review-friendly max dimension so the .rrd stays small.
        if RRD_MAX_FRAME_DIM > 0 and max(rgb.size) > RRD_MAX_FRAME_DIM:
            rgb.thumbnail((RRD_MAX_FRAME_DIM, RRD_MAX_FRAME_DIM))
        return np.asarray(rgb)


def _subsample(items: list, cap: int) -> list:
    """Evenly subsample ``items`` down to at most ``cap`` (keeps first + last)."""
    n = len(items)
    if cap <= 0 or n <= cap:
        return items
    step = n / float(cap)
    picked = [items[min(n - 1, int(i * step))] for i in range(cap)]
    # De-dupe while preserving order (integer stepping can repeat near the end).
    seen: set[int] = set()
    out = []
    for it in picked:
        key = id(it)
        if key not in seen:
            seen.add(key)
            out.append(it)
    return out


def _latest_iteration_dir(root: Path) -> Path:
    """Select the newest append-only PAIDF loop directory, with legacy fallback."""

    if not root.is_dir():
        return root
    candidates: list[tuple[int, Path]] = []
    for path in root.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"iteration-(\d+)", path.name)
        if match:
            candidates.append((int(match.group(1)), path))
    return max(candidates, default=(-1, root), key=lambda item: item[0])[1]


def _log_frame(rr: Any, rec: Any, entity: str, arr: Any) -> None:
    """Log a frame as a JPEG-encoded image (small) with a raw-RGB fallback."""
    try:
        import io

        from PIL import Image as _PILImage

        buf = io.BytesIO()
        _PILImage.fromarray(arr).save(buf, format="JPEG", quality=RRD_JPEG_QUALITY)
        rr.log(entity, rr.EncodedImage(contents=buf.getvalue(), media_type="image/jpeg"), recording=rec)
    except Exception:  # noqa: BLE001 - fall back to raw image if EncodedImage/PIL unavailable
        rr.log(entity, _image(rr, arr), recording=rec)


def _set_frame(rr: Any, rec: Any, idx: int) -> None:
    try:
        rr.set_time("frame", sequence=idx, recording=rec)
    except (TypeError, AttributeError):
        rr.set_time_sequence("frame", idx, recording=rec)


def _image(rr: Any, arr: Any):
    try:
        return rr.Image(arr, color_model="RGB")
    except TypeError:
        return rr.Image(arr)


def build_run_rrd(
    input_uri: str,
    output_uri: str,
    *,
    storage_client: "StorageClient | None" = None,
    app_id: str = APPLICATION_ID,
) -> dict[str, Any]:
    """Log a data-factory run's frames + captions to ``output_uri`` as an ``.rrd``.

    ``input_uri`` is the run root (``s3://.../<run_id>/`` or a local dir) that
    holds ``input/`` and ``cosmos_augmented/`` (and optionally ``labeled_*/``).
    ``output_uri`` is the destination ``.rrd`` (S3 or local).
    """

    if not input_uri:
        raise DataFactoryVizError("input_uri is required")
    if not output_uri or not output_uri.endswith(".rrd"):
        raise DataFactoryVizError(f"output_uri must end in .rrd, got: {output_uri!r}")

    try:
        import rerun as rr
    except ImportError as exc:  # pragma: no cover - rerun is a repo dependency
        raise DataFactoryVizError(f"rerun-sdk is required to build the recording: {exc}") from exc

    run_id = _run_id_from_uri(input_uri)
    active_storage = storage_client
    source_inventory: list[dict[str, Any]] = []
    output_object_key = ""
    output_exists = False
    if input_uri.startswith("s3://"):
        if active_storage is None:
            from npa.clients.storage import StorageClient

            active_storage = StorageClient.from_environment()
        source_bucket, source_prefix = _split_s3_prefix(input_uri)
        output_bucket, output_object_key = _split_s3_object(output_uri)
        if (
            output_bucket != source_bucket
            or not output_object_key.startswith(source_prefix)
        ):
            raise DataFactoryVizError(
                "remote RRD publication must remain inside the canonical run prefix"
            )
        source_inventory = _s3_inventory(active_storage, input_uri)
        output_exists = any(
            row["key"] == output_object_key for row in source_inventory
        )

    with tempfile.TemporaryDirectory(prefix="npa-df-viz-") as tmp:
        local = _materialize_run(
            input_uri, Path(tmp) / "run", storage_client=active_storage
        )
        captions = _load_captions(local)

        out_path = Path(tmp) / "sim2real.rrd"
        rec = rr.RecordingStream(app_id, recording_id=run_id)
        # A file sink must be attached before the first log call. Attaching it
        # afterwards happens to replay buffered rows, but leaves a streaming RRD
        # without its footer/manifest when the temporary directory is published.
        # Rerun can often read that stream, while `rerun rrd verify` correctly
        # rejects it as incomplete.
        rec.save(str(out_path))
        logged = 0

        input_root = local / "input"
        input_provenance = _read_json(input_root / "provenance.json")
        source_kind = (
            str(input_provenance.get("source_kind") or "")
            if isinstance(input_provenance, dict)
            else ""
        )
        for frame in _subsample(_image_files(input_root), RRD_MAX_FRAMES_PER_ENTITY):
            _set_frame(rr, rec, _frame_index(frame.stem))
            if frame.name.startswith("conditioning-frame-"):
                entity = "conditioning/derived"
            elif source_kind == "synthetic_fixture":
                entity = "fixture/synthetic_seeded"
            else:
                entity = f"source/{_input_entity(frame, input_root)}"
            _log_frame(rr, rec, entity, _load_rgb(frame))
            logged += 1

        if isinstance(input_provenance, dict):
            rr.log(
                "provenance/input",
                rr.TextDocument(
                    _json_block(
                        str(input_provenance.get("input_origin_label") or "Run input"),
                        input_provenance,
                    ),
                    media_type="text/markdown",
                ),
                static=True,
                recording=rec,
            )

        augmented_entities: set[str] = set()
        augmented_frame_count = 0
        augmented_video_count = 0
        variant_records = _committed_variant_records(local)
        if variant_records:
            disposition = _read_json(local / "grade" / "quality_disposition.json")
            quality_status = str(
                disposition.get("quality_status") or "UNKNOWN"
            ).upper() if isinstance(disposition, dict) else "UNKNOWN"
            for record in variant_records:
                d = record["directory"]
                label = _augmentation_label(d)
                candidate = str(record["candidate_id"])
                entity = f"augmented/{candidate}"
                augmented_entities.add(entity)
                for png in _subsample(sorted(d.glob("*.png")), RRD_MAX_FRAMES_PER_ENTITY):
                    _set_frame(rr, rec, _frame_index(png.stem))
                    _log_frame(rr, rec, entity, _load_rgb(png))
                    logged += 1
                    augmented_frame_count += 1
                video = record.get("video")
                if isinstance(video, Path) and video.is_file():
                    asset = rr.AssetVideo(path=video)
                    rr.log(f"{entity}/video", asset, static=True, recording=rec)
                    try:
                        timestamps = asset.read_frame_timestamps_nanos()
                        if len(timestamps):
                            rr.send_columns(
                                f"{entity}/video",
                                indexes=[
                                    rr.TimeColumn(
                                        "video_time", duration=1e-9 * timestamps
                                    )
                                ],
                                columns=rr.VideoFrameReference.columns_nanos(
                                    timestamps
                                ),
                                recording=rec,
                            )
                    except Exception as exc:  # noqa: BLE001 - asset remains reviewable
                        _log.debug(
                            "could not attach video frame references for %s: %s",
                            video,
                            exc,
                        )
                    augmented_video_count += 1
                if label:
                    rr.log(entity, rr.TextDocument(f"{d.name}: {label}"), static=True, recording=rec)
                rr.log(
                    f"{entity}/disposition",
                    rr.TextDocument(
                        _candidate_disposition_document(
                            local,
                            iteration=int(record["iteration"]),
                            clip=str(record["clip"]),
                            candidate_id=candidate,
                            quality_status=quality_status,
                            disposition=disposition,
                        ),
                        media_type="text/markdown",
                    ),
                    static=True,
                    recording=rec,
                )

            # A terminal PAIDF recording may never imply that a candidate was
            # reviewable when it contains only conditioning frames and text. A
            # committed candidate must contribute actual augmented image or video
            # components, including on the rejected branch.
            if augmented_frame_count == 0 and augmented_video_count == 0:
                rec.disconnect()
                raise DataFactoryVizError(
                    "committed augmented candidates produced no augmented media entities"
                )

        # The conditioning signal each variant was rendered from, as its own
        # entity tree. A segmentation-conditioned run is only reviewable if the
        # reviewer can see the segmentation next to the render, and these frames
        # live outside cosmos_augmented/ precisely so no consumer mistakes a
        # control map for an augmented frame.
        logged += _log_control_entities(rr, rec, local)

        # Neural-reconstruction runs contribute their own entities: the novel views
        # rendered from the trained Gaussians and NRE's validation renders. Both are
        # no-ops for a data-factory run, which has neither directory.
        #
        # NOTE: the input loop above is NOT a no-op for existing runs -- it was
        # widened from `rglob("*.png")` to every IMAGE_SUFFIXES entry, and entity
        # naming now groups by sub-directory when frames are nested. A flat,
        # all-PNG data-factory run is byte-identical, but a run with .jpg inputs or
        # nested directories now yields more/differently-named entities than before.
        logged += _log_nurec_entities(rr, rec, local)

        for name, body in captions.items():
            if body:
                rr.log(
                    f"captions/{name}",
                    rr.TextDocument(body, media_type="text/markdown"),
                    static=True,
                    recording=rec,
                )

        # Log every pipeline stage's report as a static text document so the whole
        # run — sampled scenarios, the hallucination / attribute-verify grade, the
        # curation report, the finalize aggregate, and a stage log/timeline — is
        # inspectable inside the embedded Rerun viewer alongside the input/output
        # images, not just the frames.
        for entity, body in _load_stage_docs(local).items():
            rr.log(
                entity,
                rr.TextDocument(body, media_type="text/markdown"),
                static=True,
                recording=rec,
            )

        if logged == 0 and augmented_video_count == 0:
            rec.disconnect()
            raise DataFactoryVizError(
                f"no input/augmented media found under {input_uri}; nothing to visualize"
            )

        # Flush batched rows and close the file sink before upload so the object
        # always contains Rerun's terminal manifest/footer.
        rec.flush()
        rec.disconnect()
        if output_exists:
            existing_path = Path(tmp) / "existing-sim2real.rrd"
            assert active_storage is not None
            active_storage.download_file(output_uri, str(existing_path))
            _verify_terminal_rrd_media(
                existing_path,
                variant_records=variant_records,
                quality_status=quality_status if variant_records else "UNKNOWN",
            )
            written_uri = output_uri
        else:
            written_uri = _publish(
                str(out_path), output_uri, storage_client=active_storage
            )

    inventory_proof: dict[str, Any] = {}
    if source_inventory:
        after = _s3_inventory(active_storage, input_uri)
        source_rows = _verify_additive_publication(
            source_inventory, after, output_object_key
        )
        inventory_proof = {
            "source_inventory_object_count": len(source_rows),
            "source_inventory_sha256": _inventory_sha256(source_rows),
            "source_inventory_unchanged_after_publication": True,
        }

    return {
        "status": "completed",
        "run_id": run_id,
        "input_uri": input_uri,
        "output_uri": written_uri,
        "frames_logged": logged,
        "augmented_media_entities": len(augmented_entities),
        "augmented_frame_components": augmented_frame_count,
        "augmented_video_components": augmented_video_count,
        **inventory_proof,
    }


def _split_s3_prefix(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    if parsed.scheme != "s3" or not parsed.netloc:
        raise DataFactoryVizError("expected an s3:// prefix")
    return parsed.netloc, parsed.path.lstrip("/").rstrip("/") + "/"


def _split_s3_object(uri: str) -> tuple[str, str]:
    parsed = urlparse(uri)
    key = parsed.path.lstrip("/")
    if parsed.scheme != "s3" or not parsed.netloc or not key or key.endswith("/"):
        raise DataFactoryVizError("expected an s3:// object URI")
    return parsed.netloc, key


def _s3_inventory(storage_client: Any, uri: str) -> list[dict[str, Any]]:
    bucket, prefix = _split_s3_prefix(uri)
    rows: list[dict[str, Any]] = []
    paginator = storage_client.s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        rows.extend(
            {
                "key": str(item.get("Key") or ""),
                "size": int(item.get("Size") or 0),
                "etag": str(item.get("ETag") or ""),
            }
            for item in page.get("Contents", [])
            if item.get("Key")
        )
    return sorted(rows, key=lambda row: row["key"])


def _inventory_sha256(rows: list[dict[str, Any]]) -> str:
    wire = json.dumps(rows, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(wire).hexdigest()


def _verify_additive_publication(
    before: list[dict[str, Any]], after: list[dict[str, Any]], output_key: str
) -> list[dict[str, Any]]:
    before_by_key = {str(row["key"]): row for row in before}
    after_by_key = {str(row["key"]): row for row in after}
    marker = "/reports/"
    run_prefix = output_key.split(marker, 1)[0] + "/" if marker in output_key else ""
    workflow_prefix = run_prefix + "npa-workflow/"
    source_before = [
        row
        for row in before
        if str(row["key"]) != output_key
        and not str(row["key"]).startswith(workflow_prefix)
    ]
    source_after = [
        row
        for row in after
        if str(row["key"]) != output_key
        and not str(row["key"]).startswith(workflow_prefix)
    ]
    if source_before != source_after:
        raise DataFactoryVizError(
            "RRD publication changed the canonical source object inventory"
        )
    workflow_before = {
        key for key in before_by_key if key.startswith(workflow_prefix)
    }
    if not workflow_before.issubset(after_by_key):
        raise DataFactoryVizError("RRD publication removed workflow evidence")
    unexpected = {
        key
        for key in after_by_key.keys() - before_by_key.keys()
        if key != output_key and not key.startswith(workflow_prefix)
    }
    if unexpected:
        raise DataFactoryVizError("RRD publication added undeclared run artifacts")
    output = after_by_key.get(output_key)
    if output is None or int(output.get("size") or 0) <= 0:
        raise DataFactoryVizError("RRD publication did not produce a non-empty artifact")
    if output_key in before_by_key and output != before_by_key[output_key]:
        raise DataFactoryVizError("RRD publication changed an existing recording")
    return source_before


def _verify_terminal_rrd_media(
    rrd_path: Path,
    *,
    variant_records: list[dict[str, Any]],
    quality_status: str,
) -> dict[str, int]:
    """Prove a preserved RRD contains each candidate's exact video and disposition."""

    try:
        from rerun.recording import load_recording
    except ImportError as exc:  # pragma: no cover - rerun is a runtime dependency
        raise DataFactoryVizError(
            "rerun recording loader is required to verify an existing RRD"
        ) from exc
    if not rrd_path.is_file() or rrd_path.stat().st_size <= 0:
        raise DataFactoryVizError("existing RRD is empty")
    chunks = list(load_recording(rrd_path).chunks())
    by_entity: dict[str, list[Any]] = {}
    for chunk in chunks:
        by_entity.setdefault(str(chunk.entity_path), []).append(chunk)

    verified_videos = 0
    verified_dispositions = 0
    expected_status = str(quality_status or "UNKNOWN").upper()
    for record in variant_records:
        candidate = str(record.get("candidate_id") or "")
        video_path = record.get("video")
        if not candidate or not isinstance(video_path, Path) or not video_path.is_file():
            raise DataFactoryVizError(
                "existing RRD verification requires every committed candidate video"
            )
        video_entity = f"/augmented/{candidate}/video"
        disposition_entity = f"/augmented/{candidate}/disposition"
        embedded: list[bytes] = []
        for chunk in by_entity.get(video_entity, []):
            batch = chunk.to_record_batch()
            if "AssetVideo:blob" not in batch.schema.names:
                continue
            for row in batch.column("AssetVideo:blob").to_pylist():
                if row:
                    embedded.append(bytes(row[0]))
        if len(embedded) != 1 or hashlib.sha256(embedded[0]).hexdigest() != _sha256_path(
            video_path
        ):
            raise DataFactoryVizError(
                "existing RRD augmented video differs from its canonical candidate"
            )
        verified_videos += 1

        text_values: list[str] = []
        for chunk in by_entity.get(disposition_entity, []):
            batch = chunk.to_record_batch()
            for name in batch.schema.names:
                if "text" not in name.lower() and "body" not in name.lower():
                    continue
                text_values.extend(str(value) for value in batch.column(name).to_pylist())
        if not text_values or not any(expected_status in value.upper() for value in text_values):
            raise DataFactoryVizError(
                "existing RRD candidate disposition is missing or inconsistent"
            )
        verified_dispositions += 1
    if not variant_records:
        raise DataFactoryVizError("existing RRD has no committed candidates to verify")
    return {
        "augmented_video_entities": verified_videos,
        "augmented_disposition_entities": verified_dispositions,
    }


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _image_files(root: Path) -> list[Path]:
    """Every image under ``root``, any supported suffix, deterministically ordered."""
    if not root.is_dir():
        return []
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _input_entity(frame: Path, root: Path) -> str:
    """Entity suffix for an ``input/`` frame.

    The data factory writes flat files (``input/video_0_frame_01.png``) and groups
    them by clip prefix. A neural-reconstruction run writes per-sensor
    sub-directories (``input/camera_images/camera2/000123.jpg``); those group by the
    owning directory, otherwise every frame would become its own entity and the
    Rerun timeline would collapse.
    """
    relative = frame.relative_to(root)
    if len(relative.parts) > 1:
        return relative.parts[-2]
    return "_".join(frame.stem.split("_")[:2]) or "clip"


def _grouped_images(root: Path) -> dict[str, list[Path]]:
    """Group images under ``root`` by their immediate parent directory name."""
    groups: dict[str, list[Path]] = {}
    for frame in _image_files(root):
        parent = frame.parent
        name = "frames" if parent == root else parent.name
        groups.setdefault(name, []).append(frame)
    return groups


def _log_nurec_entities(rr: Any, rec: Any, local: Path) -> int:
    """Log a neural-reconstruction run's renders as Rerun entities.

    ``novel_view/<camera>`` are the rig-offset views rendered from the trained
    Gaussians (the point of the capability), and ``reconstruction/<group>`` are
    NRE's own validation renders. Both entity prefixes are registered in
    ``npa.cli.agent_recordings.RUN_ENTITY_MARKERS`` so the agent recognises the
    recording as real run data.
    """
    logged = 0
    for directory, prefix in (("novel_views", "novel_view"), ("reconstruction", "reconstruction")):
        root = local / directory
        if not root.is_dir():
            continue
        for group, frames in sorted(_grouped_images(root).items()):
            for frame in _subsample(frames, RRD_MAX_FRAMES_PER_ENTITY):
                _set_frame(rr, rec, _frame_index(frame.stem))
                _log_frame(rr, rec, f"{prefix}/{group}", _load_rgb(frame))
                logged += 1
    return logged


def _log_control_entities(rr: Any, rec: Any, local: Path) -> int:
    """Log the control maps and region masks under ``cosmos_control/``.

    The augment stage publishes ``<clip>/control_<modality>/frame-*.png`` and, when
    a region mask was used, ``<clip>/mask_<modality>/frame-*.png``. Each becomes
    ``control/<clip>/<signal>`` so the segmentation that conditioned a variant sits
    beside ``augmented/<clip>`` on the same frame timeline.
    """

    logged = 0
    selected: list[tuple[str, Path]] = []
    augment_roots = _augment_roots(local)
    for iteration, augment_root in augment_roots:
        manifest_path = augment_root / "manifest.json"
        committed = _read_json(manifest_path)
        if isinstance(committed, dict):
            for variant in _validated_viz_manifest(committed):
                clip = str(variant.get("clip") or "")
                candidate = f"iteration-{iteration}/{clip}" if iteration else clip
                for uri in (variant.get("control_uris") or {}).values():
                    value = str(uri or "")
                    marker = "/cosmos_control/"
                    if marker not in value:
                        continue
                    relative = value.split(marker, 1)[1]
                    signal_dir = (
                        local
                        / "cosmos_control"
                        / Path(relative).parent
                        / Path(value).stem
                    )
                    selected.append((candidate, signal_dir))
            continue
        if manifest_path.exists():
            raise DataFactoryVizError("canonical augment manifest is unreadable")
        if (augment_root / "_attempts").exists():
            raise DataFactoryVizError(
                "augment attempts exist without a valid canonical manifest"
            )
    if not selected:
        root = local / "cosmos_control"
        if root.is_dir():
            selected = [
                (clip_dir.name, signal_dir)
                for clip_dir in sorted(p for p in root.iterdir() if p.is_dir())
                if clip_dir.name != "_attempts"
                for signal_dir in sorted(p for p in clip_dir.iterdir() if p.is_dir())
            ]
    for clip, signal_dir in selected:
        if signal_dir.is_dir():
            frames = _subsample(
                sorted(_image_files(signal_dir)), RRD_MAX_FRAMES_PER_ENTITY
            )
            entity = f"control/{clip}/{signal_dir.name}"
            for frame in frames:
                _set_frame(rr, rec, _frame_index(frame.stem))
                _log_frame(rr, rec, entity, _load_rgb(frame))
                logged += 1
    return logged


def _augment_roots(local: Path) -> list[tuple[int, Path]]:
    """Return every append-only augmentation iteration, or the legacy root."""

    root = local / "cosmos_augmented"
    if not root.is_dir():
        return []
    iterations = sorted(
        (
            (int(match.group(1)), path)
            for path in root.iterdir()
            if path.is_dir()
            if (match := re.fullmatch(r"iteration-(\d+)", path.name))
        ),
        key=lambda item: item[0],
    )
    return iterations or [(0, root)]


def _committed_variant_records(local: Path) -> list[dict[str, Any]]:
    """Map every canonical manifest candidate onto its preserved media tree."""

    root = local / "cosmos_augmented"
    records: list[dict[str, Any]] = []
    for iteration, iteration_root in _augment_roots(local):
        manifest_path = iteration_root / "manifest.json"
        manifest = _read_json(manifest_path)
        if isinstance(manifest, dict):
            for variant in _validated_viz_manifest(manifest):
                uri = str(variant.get("augmented_video_uri") or "")
                clip = str(variant.get("clip") or "").strip()
                marker = "/cosmos_augmented/"
                if marker not in uri or not clip:
                    raise DataFactoryVizError(
                        "canonical augment manifest variant has an invalid generated video URI"
                    )
                relative = Path(uri.split(marker, 1)[1])
                directory = root / relative.parent
                video = root / relative
                if not directory.is_dir() or not video.is_file():
                    raise DataFactoryVizError(
                        "canonical augment manifest references absent candidate media"
                    )
                records.append(
                    {
                        "iteration": iteration,
                        "clip": clip,
                        "candidate_id": (
                            f"iteration-{iteration}/{clip}" if iteration else clip
                        ),
                        "directory": directory,
                        "video": video,
                    }
                )
            continue
        if manifest_path.exists():
            raise DataFactoryVizError("canonical augment manifest is unreadable")
        if (iteration_root / "_attempts").exists():
            raise DataFactoryVizError(
                "augment attempts exist without a valid canonical manifest"
            )
        for directory in sorted(
            path
            for path in iteration_root.iterdir()
            if path.is_dir() and path.name != "_attempts"
        ):
            videos = sorted(directory.glob("*.mp4"))
            records.append(
                {
                    "iteration": iteration,
                    "clip": directory.name,
                    "candidate_id": (
                        f"iteration-{iteration}/{directory.name}"
                        if iteration
                        else directory.name
                    ),
                    "directory": directory,
                    "video": videos[0] if videos else None,
                }
            )
    return records


def _committed_variant_dirs(local: Path) -> list[Path]:
    """Compatibility projection of every committed candidate directory."""

    return [record["directory"] for record in _committed_variant_records(local)]


def _candidate_evaluation(local: Path, iteration: int, clip: str) -> dict[str, Any]:
    grade_root = local / "grade"
    grade_dir = (
        grade_root / f"iteration-{iteration}" / "ranking"
        if iteration
        else grade_root
    )
    try:
        from npa.workbench.cosmos_evaluator import RESULT_FILENAME as result_name
    except Exception:  # noqa: BLE001
        result_name = "cosmos_evaluator.json"
    report = _read_json(grade_dir / result_name)
    if not isinstance(report, dict):
        return {}
    return next(
        (
            item
            for item in report.get("clips", [])
            if isinstance(item, dict) and str(item.get("clip_id") or "") == clip
        ),
        {},
    )


def _candidate_disposition_document(
    local: Path,
    *,
    iteration: int,
    clip: str,
    candidate_id: str,
    quality_status: str,
    disposition: Any,
) -> str:
    """Truthful per-candidate disposition shown beside its actual media."""

    evaluation = _candidate_evaluation(local, iteration, clip)
    attributes = (
        evaluation.get("attribute_verification", {})
        if isinstance(evaluation.get("attribute_verification"), dict)
        else {}
    )
    failed_attributes = [
        str(check.get("variable") or "unknown")
        for check in attributes.get("checks", [])
        if isinstance(check, dict) and check.get("passed") is not True
    ]
    hallucination = (
        evaluation.get("hallucination", {})
        if isinstance(evaluation.get("hallucination"), dict)
        else {}
    )
    summary = {
        "candidate_id": candidate_id,
        "iteration": iteration,
        "clip_id": clip,
        "run_disposition": quality_status,
        "candidate_passed": evaluation.get("passed") is True,
        "promotion_eligible": quality_status == "ACCEPTED"
        and evaluation.get("passed") is True,
        "score": evaluation.get("score"),
        "failed_attributes": failed_attributes,
        "attribute_results": attributes.get("checks", []),
        "hallucination_status": (
            "passed" if hallucination.get("passed") is True else "failed"
        ),
        "hallucination": hallucination,
        "temporal_consistency": evaluation.get("temporal_consistency"),
        "appearance_fidelity": evaluation.get("appearance_fidelity"),
        "source_comparison_entity": "source/* or conditioning/derived",
        "output_media_entity": f"augmented/{candidate_id}",
        "final_disposition": disposition if isinstance(disposition, dict) else {},
    }
    return (
        f"# {quality_status} — candidate `{candidate_id}`\n\n"
        "This panel is review evidence only. Rejected media is never relabeled, "
        "curated, finalized, or promoted. Compare it directly with the source or "
        "conditioning entities on the shared timeline.\n\n"
        + _json_block("Candidate quality evidence", summary)
    )


def _validated_viz_manifest(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    from npa.workbench.cosmos.transfer import validate_committed_run_manifest

    try:
        return validate_committed_run_manifest(manifest)
    except (TypeError, ValueError) as exc:
        raise DataFactoryVizError(str(exc)) from exc


def _augmentation_label(clip_dir: Path) -> str:
    meta_path = clip_dir / "metadata.json"
    if not meta_path.is_file():
        return ""
    try:
        meta = json.loads(meta_path.read_text())
    except (ValueError, OSError):
        return ""
    variables = meta.get("variables", {}) if isinstance(meta, dict) else {}
    return ", ".join(f"{k}={v}" for k, v in variables.items())


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _json_block(title: str, payload: Any) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True)
    return f"## {title}\n\n```json\n{body}\n```\n"


def _load_stage_docs(local: Path) -> dict[str, str]:
    """Build per-stage markdown docs (scenarios, hallucination/grade, curation,
    finalize, and a stage log) so the full pipeline is viewable in the Rerun panel.

    Every entry is optional — a stage that did not persist its artifact is simply
    skipped, so this stays robust for partial runs.
    """
    docs: dict[str, str] = {}
    stage_log: list[str] = []

    # --- Neural-reconstruction stages (no-ops for a data-factory run) -------------
    docs.update(_load_nurec_docs(local, stage_log))

    # Stage 1 — sampled scenarios (Config Generation). This is the "various
    # scenarios" the augment stage multiplies over.
    cfg = _read_json(local / "configs" / "manifest.json")
    if isinstance(cfg, dict):
        combos = cfg.get("augmentations") or []
        lines = [f"**Scene:** {cfg.get('scene', 'n/a')}", f"**Scenarios sampled:** {len(combos)}", ""]
        for i, combo in enumerate(combos):
            if isinstance(combo, dict):
                prompt = str(combo.get("prompt") or "")
                attrs = ", ".join(f"{k}={v}" for k, v in combo.items() if k != "prompt")
                lines.append(f"- **scenario {i}** — {attrs}")
                if prompt:
                    lines.append(f"    - prompt: _{prompt}_")
        docs["pipeline/1_scenarios"] = "## Config generation — sampled scenarios\n\n" + "\n".join(lines) + "\n"
        stage_log.append(f"configs: {len(combos)} scenario(s) sampled")

    input_provenance = _read_json(local / "input" / "provenance.json")
    if isinstance(input_provenance, dict):
        label = str(input_provenance.get("input_origin_label") or "Run input")
        docs["pipeline/0_input_provenance"] = _json_block(label, input_provenance)
        stage_log.append(
            "input: "
            f"{label}, source_kind={input_provenance.get('source_kind', 'n/a')}, "
            f"sha256={input_provenance.get('sha256') or 'fixture-generated'}"
        )

    # Augment fan-out — how many Cosmos Transfer 2.5 variants were produced.
    aug_dir = _latest_iteration_dir(local / "cosmos_augmented")
    aug = _read_json(aug_dir / "manifest.json")
    if isinstance(aug, dict):
        variants = aug.get("variants") or aug.get("clips") or []
        docs["pipeline/2_augment"] = _json_block("Augment — Cosmos Transfer 2.5 (multiply)", aug)
        conditioning = f"control={aug.get('control') or 'n/a'}"
        if aug.get("control_prompt"):
            conditioning += f" on '{aug['control_prompt']}'"
        if aug.get("mask_prompt"):
            conditioning += f", masked to '{aug['mask_prompt']}'"
        stage_log.append(
            f"augment: {aug.get('variant_count', len(variants))} variant(s), "
            f"mode={aug.get('mode', 'n/a')}, input_conditioned={aug.get('input_conditioned')}, "
            f"{conditioning}"
        )

    # Evaluate & Validate — the hallucination / attribute-verification grade.
    grade_root = local / "grade"
    grade_dir = _latest_iteration_dir(grade_root)
    grade_docs: list[str] = []
    try:
        from npa.workbench.cosmos_evaluator import (
            RESULT_FILENAME as _cosmos_evaluator_result_filename,
        )
    except Exception:  # noqa: BLE001
        _cosmos_evaluator_result_filename = "cosmos_evaluator.json"
    try:
        from npa.workbench.vlm_eval import RESULT_FILENAME as _vlm_result_filename
    except Exception:  # noqa: BLE001
        _vlm_result_filename = "vlm_eval_stub.json"
    for name in (
        _cosmos_evaluator_result_filename,
        _vlm_result_filename,
        "vlm_eval.json",
    ):
        ev = _read_json(grade_dir / name)
        if isinstance(ev, dict):
            grade_docs.append(
                _json_block("Evaluator — integrity and appearance checks", ev)
            )
            stage_log.append(
                f"grade: score={ev.get('score')}, status={ev.get('status', 'n/a')}"
            )
            break
    dec = _read_json(grade_dir / "decision.json")
    if isinstance(dec, dict):
        grade_docs.append(_json_block("Quality gate decision", dec))
        stage_log.append(f"grade: decision={dec.get('decision', 'n/a')}")
    disposition = _read_json(grade_root / "quality_disposition.json")
    if isinstance(disposition, dict):
        grade_docs.append(_json_block("Final quality disposition", disposition))
        stage_log.append(
            "grade: disposition="
            f"{disposition.get('quality_status', 'n/a')}, "
            f"score={disposition.get('score', 'n/a')}"
        )
    if grade_docs:
        docs["pipeline/3_grade"] = "\n".join(grade_docs)

    # Curation report.
    cur = _read_json(local / "curation" / "report.json")
    if isinstance(cur, dict):
        docs["pipeline/4_curation"] = _json_block("Curation report", cur)
        stage_log.append(
            f"curation: {cur.get('augmented_clips', 0)} clip(s), "
            f"multiply={(cur.get('multiply') or {}).get('mode', 'n/a')}"
        )

    # Finalize aggregate report.
    fin = _read_json(local / "reports" / "final.json")
    if isinstance(fin, dict):
        docs["pipeline/5_finalize"] = _json_block("Finalize — aggregate report", fin)
        stage_log.append(
            f"finalize: {fin.get('artifact_count', 0)} artifacts, "
            f"multiply_mode={fin.get('multiply_mode', 'n/a')}"
        )

    if stage_log:
        docs["pipeline/0_log"] = (
            "## Pipeline stage log\n\n" + "\n".join(f"- {line}" for line in stage_log) + "\n"
        )
    return docs


def _read_yaml(path: Path) -> Any:
    try:
        import yaml

        return yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - optional artifact, optional dependency
        return None


def _load_nurec_docs(local: Path, stage_log: list[str]) -> dict[str, str]:
    """Build the neural-reconstruction stage docs for the Rerun panel.

    Every entry is optional, so a data-factory run (which has none of these
    artifacts) gets an empty dict and is completely unaffected.
    """
    docs: dict[str, str] = {}

    # Stage 1 — the real capture that was reconstructed, plus how the rig frame
    # NRE requires was obtained.
    manifest = _read_json(local / "ncore" / "manifest.json")
    if isinstance(manifest, dict):
        rig = manifest.get("rig_derivation") or {}
        lines = [
            f"**Dataset:** `{manifest.get('dataset_id', 'n/a')}`",
            f"**Scene:** `{manifest.get('scene', 'n/a')}` "
            f"(variant `{manifest.get('variant', 'n/a')}`)",
            f"**NCore shards:** {manifest.get('shard_count', 0)}",
            f"**Cameras:** {', '.join(manifest.get('camera_ids') or []) or 'n/a'}",
            f"**LiDARs:** {', '.join(manifest.get('lidar_ids') or []) or 'n/a'}",
        ]
        if rig:
            lines += [
                "",
                "_NRE requires a `rig -> world` pose edge that object-centric "
                "captures do not ship; it was derived from the reference camera._",
                f"**Reference camera:** `{rig.get('reference_camera', 'n/a')}` "
                f"({rig.get('pose_count', 0)} poses)",
                f"**Poses component group:** `{rig.get('poses_component_group', 'n/a')}`",
            ]
        docs["pipeline/1_ncore"] = "## NCore input capture\n\n" + "\n".join(lines) + "\n"
        stage_log.append(
            f"ncore: {manifest.get('scene', 'n/a')} "
            f"({manifest.get('shard_count', 0)} shard(s), "
            f"{len(manifest.get('camera_ids') or [])} camera(s))"
        )

    # Stage 2 — the trained Gaussian reconstruction and its real quality metrics.
    metrics = _read_yaml(local / "reconstruction" / "metrics.yaml")
    if isinstance(metrics, dict):
        docs["pipeline/2_reconstruct"] = _json_block(
            "Reconstruction — 3DGUT Gaussian training metrics", metrics
        )
        # `gaussians/*` is one of the run-entity markers the agent scans for, so the
        # metrics also land under that entity path.
        docs["gaussians/summary"] = _json_block(
            "Gaussian reconstruction quality (NRE validation)", metrics
        )
        flat = {
            key: value
            for key, value in _flatten_scalars(metrics)
            if any(token in key.lower() for token in ("psnr", "ssim", "lpips"))
        }
        if flat:
            stage_log.append(
                "reconstruct: "
                + ", ".join(f"{key}={value}" for key, value in sorted(flat.items())[:6])
            )
        else:
            stage_log.append("reconstruct: metrics recorded")

    # Stage 3 — novel views rendered from the trained scene.
    novel_root = local / "novel_views"
    if novel_root.is_dir():
        groups = _grouped_images(novel_root)
        videos = sorted(novel_root.rglob("*.mp4"))
        lines = [f"**Cameras rendered:** {len(groups)}", ""]
        for name, frames in sorted(groups.items()):
            lines.append(f"- `novel_view/{name}` — {len(frames)} frame(s)")
        for video in videos:
            lines.append(f"- video: `{video.name}`")
        docs["pipeline/3_novel_views"] = (
            "## Novel-view rendering (rig-offset, not training views)\n\n"
            + "\n".join(lines)
            + "\n"
        )
        stage_log.append(
            f"novel_views: {sum(len(v) for v in groups.values())} frame(s) "
            f"across {len(groups)} camera(s), {len(videos)} video(s)"
        )
    return docs


def _flatten_scalars(payload: Any, prefix: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(payload, dict):
        for key, value in payload.items():
            out.extend(_flatten_scalars(value, prefix=f"{prefix}{key}/"))
    elif isinstance(payload, (int, float)) and not isinstance(payload, bool):
        out.append((prefix.rstrip("/"), payload))
    return out


_CAPTION_HEADERS = {
    "labeled_original": (
        "## Derived conditioning-frame captions — Token Factory VLM\n\n"
        "_Descriptive labels of frames derived from the verified source (or the "
        "explicit synthetic fixture). This is captioning, "
        "not the quality gate — see `pipeline/3_grade` for the attribute-verify / "
        "hallucination check (score + promote/loop_back decision)._\n\n"
    ),
    "labeled_augmented": (
        "## Augmented-clip captions — Token Factory VLM\n\n"
        "_Descriptive per-frame labels of the Cosmos Transfer 2.5 OUTPUT. This is "
        "captioning, not the quality gate — see `pipeline/3_grade` for the "
        "attribute-verify / hallucination check (score + promote/loop_back decision)._\n\n"
    ),
}


def _load_captions(local: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("labeled_original", "labeled_augmented"):
        cj = local / name / "captions.json"
        if not cj.is_file():
            continue
        try:
            payload = json.loads(cj.read_text())
        except (ValueError, OSError):
            continue
        items = payload.get("captions", []) if isinstance(payload, dict) else []
        body = "\n\n".join(
            f"- {c.get('image')}: {c.get('caption')}" for c in items[:12] if isinstance(c, dict)
        )
        if body:
            # Prefix a self-identifying header so a caption panel is never confused
            # with the VLM eval / hallucination grade panel in the Rerun grid.
            out[name] = _CAPTION_HEADERS.get(name, "") + body
    return out


def _materialize_run(input_uri: str, dest: Path, *, storage_client: "StorageClient | None") -> Path:
    if not input_uri.startswith("s3://"):
        return Path(input_uri)
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    dest.mkdir(parents=True, exist_ok=True)
    root = input_uri.rstrip("/")
    for sub in RUN_SUBDIRS:
        try:
            client.download_path(f"{root}/{sub}/", str(dest / sub))
        except Exception:
            # Optional subtrees (labeled_*) may not exist; input/augmented drive the recording.
            continue
    return dest


def _publish(local_path: str, output_uri: str, *, storage_client: "StorageClient | None") -> str:
    if not output_uri.startswith("s3://"):
        out = Path(output_uri)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(Path(local_path).read_bytes())
        return str(out)
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    client.put_bytes_conditional(
        Path(local_path).read_bytes(),
        output_uri,
        if_none_match=True,
        content_type="application/octet-stream",
    )
    return output_uri
