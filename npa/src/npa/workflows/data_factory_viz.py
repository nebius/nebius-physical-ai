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
import math
import os
import re
import sqlite3
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any
from urllib.parse import urlparse

from npa.workflows.data_factory_review import _PaidfReview

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient

_log = logging.getLogger(__name__)

APPLICATION_ID = "physical-ai-data-factory"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

#: Run sub-directories materialized from S3 before building a recording. Covers
#: both producers: the data-factory blueprint (input/cosmos_augmented/
#: cosmos_control/labeled_*/configs/grade/curation) and the NuRec
#: neural-reconstruction workflow (source/ncore/reconstruction/novel_views).
#: Only source attribution is fetched; the original capture archive is not
#: needed to visualize the converted run. Missing subtrees are skipped unless
#: a COLMAP marker makes its three lineage documents mandatory.
RUN_SUBDIRS = (
    "input",
    "cosmos_augmented",
    "cosmos_control",
    "labeled_original",
    "labeled_augmented",
    "configs",
    "grade",
    "curation",
    "source",
    "ncore",
    "reconstruction",
    "novel_views",
    "reports",
)

_COLMAP_LINEAGE_ARTIFACTS = (
    (
        "source",
        "source/attribution.json",
        "COLMAP source attribution",
        "dataset revision sha256 creator license selected_capture source_counts",
    ),
    (
        "conversion",
        "ncore/sequence/conversion.json",
        "COLMAP to NCore conversion",
        "",
    ),
    (
        "rig",
        "ncore/sequence/npa-rig.json",
        "NCore rig derivation",
        "status reference_camera pose_count cameras already_present poses_component_group "
        "copied_dynamic_edges copied_static_edges",
    ),
)
_COLMAP_LINEAGE_PATHS = tuple(row[1] for row in _COLMAP_LINEAGE_ARTIFACTS)
# Rig sidecars also belong to preconverted NCore; they alone do not identify COLMAP.
_COLMAP_LINEAGE_MARKERS = _COLMAP_LINEAGE_PATHS[:2]


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
RRD_VERIFY_MAX_FILE_BYTES = 2**31
RRD_VERIFY_MAX_CHUNKS = 50_000
RRD_VERIFY_MAX_ROWS = 1_000_000
RRD_VERIFY_MAX_CHUNK_BYTES = 128 * 1024 * 1024
RRD_VERIFY_MAX_DECODED_BYTES = 2**31
RRD_VERIFY_MAX_MEDIA_BYTES = 2**30
RRD_VERIFY_MAX_ENTITIES = 100_000
_BLUEPRINT_UUID = re.compile(
    r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-"
    r"[89ab][0-9a-f]{3}-[0-9a-f]{12}\b"
)


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
    """Evenly subsample items, retaining both endpoints when cap permits."""
    n = len(items)
    if cap <= 0 or n <= cap:
        return items
    if cap == 1:
        return items[:1]
    return [items[i * (n - 1) // (cap - 1)] for i in range(cap)]


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
        rr.log(
            entity,
            rr.EncodedImage(contents=buf.getvalue(), media_type="image/jpeg"),
            recording=rec,
        )
    except Exception:  # noqa: BLE001 - fall back to raw image if EncodedImage/PIL unavailable
        rr.log(entity, _image(rr, arr), recording=rec)


def _set_frame(rr: Any, rec: Any, idx: int) -> None:
    try:
        rr.set_time("frame", sequence=idx, recording=rec)
    except (TypeError, AttributeError):
        rr.set_time_sequence("frame", idx, recording=rec)


def _log_video_asset(
    rr: Any,
    rec: Any,
    entity: str,
    path: Path,
    *,
    temporal_alignment: dict[str, Any] | None = None,
) -> int:
    """Log one video and its real decoded timestamps on the duration timeline."""

    asset = rr.AssetVideo(path=path)
    rr.log(entity, asset, static=True, recording=rec)
    timestamps = asset.read_frame_timestamps_nanos()
    if not len(timestamps):
        raise DataFactoryVizError(f"video has no decoded frame timestamps: {path.name}")
    timeline_timestamps = timestamps
    reference_timestamps = timestamps
    if temporal_alignment is not None:
        frame_map = temporal_alignment.get("frame_map")
        if not isinstance(frame_map, list) or not frame_map:
            raise DataFactoryVizError(
                "source video temporal alignment has no frame-reference map"
            )
        try:
            source_indices = [int(item["source_index"]) for item in frame_map]
            timeline_timestamps = [
                round(float(item["output_timestamp_seconds"]) * 1_000_000_000)
                for item in frame_map
            ]
        except (KeyError, TypeError, ValueError) as exc:
            raise DataFactoryVizError(
                "source video temporal alignment is malformed"
            ) from exc
        if any(index < 0 or index >= len(timestamps) for index in source_indices):
            raise DataFactoryVizError(
                "source video temporal alignment references an absent frame"
            )
        if any(
            right <= left
            for left, right in zip(timeline_timestamps, timeline_timestamps[1:])
        ):
            raise DataFactoryVizError(
                "source video temporal alignment is not strictly increasing"
            )
        reference_timestamps = [timestamps[index] for index in source_indices]
    references = rr.VideoFrameReference.columns_nanos(reference_timestamps)
    rr.send_columns(
        entity,
        indexes=[
            rr.TimeColumn(
                "video_time",
                duration=[1e-9 * value for value in timeline_timestamps],
            )
        ],
        columns=references,
        recording=rec,
    )
    # Keep ordinal navigation as an explicitly secondary timeline. The default
    # viewer uses ``video_time`` and therefore never invents playback speed.
    rr.send_columns(
        entity,
        indexes=[rr.TimeColumn("frame", sequence=range(len(reference_timestamps)))],
        columns=references,
        recording=rec,
    )
    return len(reference_timestamps)


def _source_video_temporal_alignment(
    media_evidence: dict[str, Any], entity: str
) -> dict[str, Any] | None:
    """Use only the decoded-PTS v3 map for exact source-frame remapping."""

    if (
        media_evidence.get("conditioning_policy") == "source-fidelity-v3"
        and entity == "source/original"
    ):
        value = media_evidence.get("temporal_alignment")
        return value if isinstance(value, dict) else None
    return None


def _image(rr: Any, arr: Any):
    try:
        return rr.Image(arr, color_model="RGB")
    except TypeError:
        return rr.Image(arr)


def _build_data_factory_blueprint(
    rrb: Any,
    *,
    source_entities: list[str],
    candidate_entities: list[str],
    has_controls: bool,
    has_captions: bool,
    has_pipeline_evidence: bool,
    default_timeline: str,
    foreground_controls: bool = False,
) -> Any:
    """Foreground media comparison while keeping factual evidence in another tab."""

    def media_group(entities: list[str], name: str) -> Any:
        views = [
            rrb.Spatial2DView(
                origin=entity,
                contents=f"{entity}/**",
                name=entity.rsplit("/", 1)[-1].replace("_", " "),
            )
            for entity in entities
        ]
        if len(views) == 1:
            return views[0]
        return rrb.Tabs(*views, active_tab=0, name=name)

    media_columns: list[Any] = []
    if source_entities:
        media_columns.append(media_group(source_entities, "Original source"))
    if has_controls and foreground_controls:
        media_columns.append(
            rrb.Spatial2DView(
                origin="control",
                contents="control/**",
                name="Conditioning controls",
            )
        )
    if candidate_entities:
        media_columns.append(media_group(candidate_entities, "Generated candidates"))
    media_tab: Any = (
        media_columns[0]
        if len(media_columns) == 1
        else rrb.Horizontal(
            *media_columns,
            column_shares=[1.0] * len(media_columns),
            name="Original versus generated",
        )
    )

    context_views: list[Any] = []
    if has_controls and not foreground_controls:
        context_views.append(
            rrb.Spatial2DView(
                origin="control",
                contents="control/**",
                name="Conditioning controls and masks",
            )
        )
    if has_captions:
        context_views.append(
            rrb.TextDocumentView(
                origin="captions", contents="captions/**", name="Prompt and captions"
            )
        )
    if has_pipeline_evidence:
        context_views.append(
            rrb.TextDocumentView(
                origin="pipeline",
                contents="pipeline/**",
                name="Evaluator scores and disposition",
            )
        )
    if candidate_entities:
        context_views.append(
            rrb.TextDocumentView(
                origin="augmented",
                contents="augmented/**/disposition",
                name="Per-candidate accept/reject disposition",
            )
        )
    foreground_rows: list[Any] = [media_tab]
    if context_views:
        foreground_rows.append(
            context_views[0]
            if len(context_views) == 1
            else rrb.Horizontal(
                *context_views,
                column_shares=[1.0] * len(context_views),
                name="Conditioning, evaluator scores, and disposition",
            )
        )
    layout: Any = (
        foreground_rows[0]
        if len(foreground_rows) == 1
        else rrb.Vertical(
            *foreground_rows,
            row_shares=[3.0, 1.0],
            name="Media-first quality review",
        )
    )
    return rrb.Blueprint(
        layout,
        rrb.BlueprintPanel(state=rrb.PanelState.Hidden),
        rrb.SelectionPanel(state=rrb.PanelState.Hidden),
        rrb.TimePanel(state=rrb.PanelState.Expanded, timeline=default_timeline),
        auto_layout=False,
        collapse_panels=True,
    )


def _resolve_existing_output_publication(
    storage_client: Any,
    output_uri: str,
    *,
    inventory_keys: set[str],
) -> tuple[str, Any | None]:
    """Resolve an existing reserved RRD through its committed journal."""

    from npa.agent_backend.publication_reader import publication_root_from_canonical_uri
    from npa.workflows.sim2real.publication import (
        resolve_committed_publication_snapshot,
    )

    try:
        publication_root_from_canonical_uri(output_uri)
    except ValueError:
        _, output_key = _split_s3_object(output_uri)
        return (output_uri, None) if output_key in inventory_keys else ("", None)
    _, output_key = _split_s3_object(output_uri)
    inventory_contains_output = output_key in inventory_keys
    publication = resolve_committed_publication_snapshot(storage_client, output_uri)
    if not publication.journaled:
        return (
            (output_uri, publication)
            if inventory_contains_output
            else ("", publication)
        )
    resolved = publication.resolve(output_uri)
    if not resolved:
        raise DataFactoryVizError(
            "committed publication does not contain the requested RRD"
        )
    return resolved, publication


def build_run_rrd(
    input_uri: str,
    output_uri: str,
    *,
    storage_client: "StorageClient | None" = None,
    app_id: str = APPLICATION_ID,
    html_preview: bool = False,
) -> dict[str, Any]:
    """Log a data-factory run's frames + captions to ``output_uri`` as an ``.rrd``.

    ``input_uri`` is the run root (``s3://.../<run_id>/`` or a local dir) that
    holds ``input/`` and ``cosmos_augmented/`` (and optionally ``labeled_*/``).
    ``output_uri`` is the destination ``.rrd`` (S3 or local).

    Args:
        input_uri: Canonical run prefix or local directory.
        output_uri: Destination recording inside the same remote run prefix.
        storage_client: Optional storage client used for download and publication.
        app_id: Recording application identity.
        html_preview: Also publish a compact index.html for NuRec runs. PAIDF
            retains its existing Rerun presentation and does not emit this page.

    Returns:
        Publication metadata, counts and optional NuRec HTML URI and SHA256.

    Raises:
        DataFactoryVizError: Media, lineage or recording validation fails.
        OSError: Local artifact reading or writing fails.
    """

    if not input_uri:
        raise DataFactoryVizError("input_uri is required")
    if not output_uri or not output_uri.endswith(".rrd"):
        raise DataFactoryVizError(f"output_uri must end in .rrd, got: {output_uri!r}")

    try:
        import rerun as rr
        import rerun.blueprint as rrb
    except ImportError as exc:  # pragma: no cover - rerun is a repo dependency
        raise DataFactoryVizError(
            f"rerun-sdk is required to build the recording: {exc}"
        ) from exc

    run_id = _run_id_from_uri(input_uri)
    active_storage = storage_client
    source_inventory: list[dict[str, Any]] = []
    require_colmap_lineage = False
    output_object_key = ""
    existing_output_uri = ""
    existing_publication = None
    inventory_output_key = ""
    preview_bytes = None
    if input_uri.startswith("s3://"):
        if active_storage is None:
            from npa.clients.storage import StorageClient

            active_storage = StorageClient.from_environment()
        source_bucket, source_prefix = _split_s3_prefix(input_uri)
        output_bucket, output_object_key = _split_s3_object(output_uri)
        if output_bucket != source_bucket or not output_object_key.startswith(
            source_prefix
        ):
            raise DataFactoryVizError(
                "remote RRD publication must remain inside the canonical run prefix"
            )
        source_inventory = _s3_inventory(active_storage, input_uri)
        require_colmap_lineage = _inventory_has_colmap_lineage(
            source_inventory, source_prefix
        )
        inventory_keys = {str(row["key"]) for row in source_inventory}
        existing_output_uri, existing_publication = (
            _resolve_existing_output_publication(
                active_storage,
                output_uri,
                inventory_keys=inventory_keys,
            )
        )
        inventory_output_key = output_object_key
        if existing_output_uri:
            existing_bucket, inventory_output_key = _split_s3_object(
                existing_output_uri
            )
            if existing_bucket != source_bucket:
                raise DataFactoryVizError(
                    "committed RRD resolved outside the canonical run bucket"
                )

    with tempfile.TemporaryDirectory(prefix="npa-df-viz-") as tmp:
        local = _materialize_run(
            input_uri,
            Path(tmp) / "run",
            storage_client=active_storage,
            require_colmap_lineage=require_colmap_lineage,
        )
        review = _PaidfReview(local, input_uri)
        captions = _load_captions(local, review=review)
        input_root = local / "input"
        raw_input_provenance = _read_json(input_root / "provenance.json")
        input_provenance = review.read(input_root / "provenance.json", "input")
        source_kind = (
            str(input_provenance.get("source_kind") or "")
            if isinstance(input_provenance, dict)
            else ""
        )
        source_entities: set[str] = set()
        for frame in _image_files(input_root):
            if frame.name.startswith("conditioning-frame-"):
                source_entities.add("conditioning/derived")
            elif source_kind == "synthetic_fixture":
                source_entities.add("fixture/synthetic_seeded")
            else:
                source_entities.add(f"source/{_input_entity(frame, input_root)}")
        source_video_records: list[dict[str, Any]] = []
        for name, entity in (
            ("source.mp4", "source/original"),
            ("conditioning.mp4", "conditioning/derived"),
        ):
            video = input_root / name
            if video.is_file():
                source_entities.add(entity)
                source_video_records.append({"entity": entity, "video": video})
        variant_records = _committed_variant_records(local)
        # ``workbench.nurec.visualize`` is intentionally shared by NuRec and
        # PAIDF workflow stages. Its CLI default is the NuRec application id,
        # but committed PAIDF candidates are authoritative run-shape evidence:
        # they must always receive the media-first PAIDF blueprint and
        # presentation metadata, regardless of that shared entrypoint default.
        effective_app_id = APPLICATION_ID if variant_records else app_id
        candidate_entities = [
            f"augmented/{record['candidate_id']}" for record in variant_records
        ]
        stage_docs = _load_stage_docs(local, review=review)
        media_evidence = _source_fidelity_media_evidence(
            local,
            # Internal synchronization checks need the source-derived policy and
            # temporal map. The portable review projection intentionally omits
            # those nested runtime details, so using it here silently disabled
            # the evidence panel for real source-fidelity runs.
            input_provenance=raw_input_provenance,
            variant_records=variant_records,
        )
        if media_evidence:
            stage_docs["pipeline/2_media_metadata"] = _json_block(
                "Synchronized source, conditioning, control, and generated media",
                media_evidence,
            )
        has_controls = bool(_image_files(local / "cosmos_control")) or bool(
            media_evidence.get("controls") if media_evidence else False
        )
        if (
            effective_app_id == APPLICATION_ID
            and variant_records
            and not source_entities
        ):
            raise DataFactoryVizError(
                "committed augmented candidates require source media for comparison"
            )
        has_video_timing = bool(source_video_records) and any(
            isinstance(record.get("video"), Path) and record["video"].is_file()
            for record in variant_records
        )
        default_timeline = "video_time" if has_video_timing else "frame"
        blueprint_source_entities = sorted(source_entities)
        if media_evidence:
            source_priority = {
                "source/original": 0,
                "conditioning/derived": 1,
            }
            blueprint_source_entities.sort(
                key=lambda entity: (source_priority.get(entity, 2), entity)
            )

        out_path = Path(tmp) / "sim2real.rrd"
        rec = rr.RecordingStream(effective_app_id, recording_id=run_id)
        # A file sink must be attached before the first log call. Attaching it
        # afterwards happens to replay buffered rows, but leaves a streaming RRD
        # without its footer/manifest when the temporary directory is published.
        # Rerun can often read that stream, while `rerun rrd verify` correctly
        # rejects it as incomplete.
        if effective_app_id == APPLICATION_ID and (
            source_entities or candidate_entities
        ):
            rec.save(
                str(out_path),
                default_blueprint=_build_data_factory_blueprint(
                    rrb,
                    source_entities=blueprint_source_entities,
                    candidate_entities=candidate_entities,
                    has_controls=has_controls,
                    has_captions=bool(captions),
                    has_pipeline_evidence=bool(stage_docs),
                    default_timeline=default_timeline,
                    foreground_controls=bool(media_evidence),
                ),
            )
        else:
            rec.save(str(out_path))
        logged = 0

        source_video_count = 0
        for source_video in source_video_records:
            temporal_alignment = (
                _source_video_temporal_alignment(media_evidence, source_video["entity"])
                if media_evidence
                else None
            )
            _log_video_asset(
                rr,
                rec,
                f"{source_video['entity']}/video",
                source_video["video"],
                temporal_alignment=temporal_alignment,
            )
            source_video_count += 1

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
        if variant_records:
            disposition = review.read(
                local / "grade" / "quality_disposition.json", "quality"
            )
            quality_status = (
                str(disposition.get("quality_status") or "UNKNOWN").upper()
                if isinstance(disposition, dict)
                else "UNKNOWN"
            )
            for record in variant_records:
                d = record["directory"]
                label = _augmentation_label(d, review)
                candidate = str(record["candidate_id"])
                entity = f"augmented/{candidate}"
                augmented_entities.add(entity)
                for png in _subsample(
                    sorted(d.glob("*.png")), RRD_MAX_FRAMES_PER_ENTITY
                ):
                    _set_frame(rr, rec, _frame_index(png.stem))
                    _log_frame(rr, rec, entity, _load_rgb(png))
                    logged += 1
                    augmented_frame_count += 1
                video = record.get("video")
                if isinstance(video, Path) and video.is_file():
                    try:
                        _log_video_asset(rr, rec, f"{entity}/video", video)
                    except Exception as exc:  # noqa: BLE001 - asset remains reviewable
                        rec.disconnect()
                        raise DataFactoryVizError(
                            "generated video timing could not be decoded"
                        ) from exc
                    augmented_video_count += 1
                if label:
                    rr.log(
                        entity,
                        rr.TextDocument(f"{review.identity(d.name)}: {label}"),
                        static=True,
                        recording=rec,
                    )
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
                            review=review,
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
        for entity, body in stage_docs.items():
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
        if existing_output_uri:
            existing_path = Path(tmp) / "existing-sim2real.rrd"
            assert active_storage is not None
            active_storage.download_file(existing_output_uri, str(existing_path))
            if existing_publication is not None:
                from npa.workflows.sim2real.publication import (
                    verify_committed_publication_file,
                )

                verify_committed_publication_file(
                    existing_publication,
                    output_uri,
                    existing_path,
                    client=active_storage,
                )
            _verify_existing_rrd_semantics(existing_path, out_path)
            _verify_terminal_rrd_media(
                existing_path,
                variant_records=variant_records,
                quality_status=quality_status if variant_records else "UNKNOWN",
                source_video_records=source_video_records,
            )
            written_uri = existing_output_uri
        else:
            written_uri = _publish(
                str(out_path), output_uri, storage_client=active_storage
            )
        if (
            html_preview
            and effective_app_id == "neural-reconstruction"
            and _has_nurec_outputs(local)
        ):
            from npa.workbench.nurec.preview import write_nurec_preview

            preview_path = Path(tmp) / "index.html"
            try:
                write_nurec_preview(local, preview_path)
            except (ValueError, OSError) as exc:
                raise DataFactoryVizError(
                    "NuRec HTML preview could not be built"
                ) from exc
            preview_bytes = preview_path.read_bytes()

    inventory_proof: dict[str, Any] = {}
    if source_inventory:
        after = _s3_inventory(active_storage, input_uri)
        source_rows = _verify_additive_publication(
            source_inventory, after, inventory_output_key
        )
        inventory_proof = {
            "source_inventory_object_count": len(source_rows),
            "source_inventory_sha256": _inventory_sha256(source_rows),
            "source_inventory_unchanged_after_publication": True,
        }

    preview_result = _publish_nurec_preview(preview_bytes, output_uri, active_storage)
    return {
        "status": "completed",
        "run_id": run_id,
        "input_uri": input_uri,
        "output_uri": written_uri,
        "frames_logged": logged,
        "augmented_media_entities": len(augmented_entities),
        "augmented_frame_components": augmented_frame_count,
        "augmented_video_components": augmented_video_count,
        "source_video_components": source_video_count,
        "presentation": {
            "default_timeline": default_timeline,
            "default_view": (
                "source-control-generated"
                if media_evidence
                else "original-versus-generated"
            ),
            "source_entities": sorted(source_entities),
            "source_video_entities": [
                str(record["entity"]) for record in source_video_records
            ],
            "candidate_entities": candidate_entities,
            "conditioning_context": has_controls,
            "evaluator_disposition_context": bool(stage_docs),
            "timing_basis": (
                "decoded-video-timestamps" if has_video_timing else "frame-sequence"
            ),
        }
        if effective_app_id == APPLICATION_ID
        else {},
        **inventory_proof,
        **preview_result,
    }


def _has_nurec_outputs(local):
    return (
        (local / "ncore" / "manifest.json").is_file()
        or (local / "reconstruction" / "metrics.yaml").is_file()
        or bool(_image_files(local / "novel_views"))
    )


def _publish_nurec_preview(payload, rrd_uri, storage):
    if payload is None:
        return {}
    target = (
        rrd_uri.rsplit("/", 1)[0] + "/index.html"
        if rrd_uri.startswith("s3://")
        else str(Path(rrd_uri).with_name("index.html"))
    )
    if target.startswith("s3://"):
        previous = storage.read_bytes_with_etag(target)
        if previous is not None and previous[0] != payload:
            raise DataFactoryVizError(
                "existing NuRec HTML preview differs; use a fresh run"
            )
        if previous is None:
            storage.put_bytes_conditional(
                payload, target, if_none_match=True, content_type="text/html"
            )
    else:
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_bytes(payload)
    return {
        "html_preview_uri": target,
        "html_preview_sha256": hashlib.sha256(payload).hexdigest(),
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


def _inventory_has_colmap_lineage(
    inventory: list[dict[str, Any]], run_prefix: str
) -> bool:
    lineage_keys = {run_prefix + relative for relative in _COLMAP_LINEAGE_MARKERS}
    return any(str(row.get("key") or "") in lineage_keys for row in inventory)


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
    workflow_before = {key for key in before_by_key if key.startswith(workflow_prefix)}
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
        raise DataFactoryVizError(
            "RRD publication did not produce a non-empty artifact"
        )
    if output_key in before_by_key and output != before_by_key[output_key]:
        raise DataFactoryVizError("RRD publication changed an existing recording")
    return source_before


def _verify_terminal_rrd_media(
    rrd_path: Path,
    *,
    variant_records: list[dict[str, Any]],
    quality_status: str,
    source_video_records: list[dict[str, Any]] | None = None,
) -> dict[str, int]:
    """Prove a preserved RRD contains each candidate's exact video and disposition."""

    expected_videos, dispositions, counts = _expected_terminal_rrd_media(
        variant_records,
        source_video_records or [],
    )
    observed_videos, observed_dispositions = _scan_terminal_rrd(
        rrd_path,
        expected_videos=set(expected_videos),
        expected_dispositions=set(dispositions),
        quality_status=str(quality_status or "UNKNOWN").upper(),
    )
    for entity, expected_digest in expected_videos.items():
        count, observed_digest = observed_videos.get(entity, (0, ""))
        if count != 1 or observed_digest != expected_digest:
            kind = "source" if entity.startswith("/source/") else "augmented"
            raise DataFactoryVizError(
                f"existing RRD {kind} video differs from its canonical input"
            )
    if not all(observed_dispositions.get(entity, False) for entity in dispositions):
        raise DataFactoryVizError(
            "existing RRD candidate disposition is missing or inconsistent"
        )
    return counts


def _expected_terminal_rrd_media(
    variant_records: list[dict[str, Any]],
    source_video_records: list[dict[str, Any]],
) -> tuple[dict[str, str], set[str], dict[str, int]]:
    if not variant_records:
        raise DataFactoryVizError("existing RRD has no committed candidates to verify")
    videos: dict[str, str] = {}
    dispositions: set[str] = set()
    for record in source_video_records:
        entity = str(record.get("entity") or "")
        video = record.get("video")
        if not entity or not isinstance(video, Path) or not video.is_file():
            raise DataFactoryVizError(
                "existing RRD verification requires every source video"
            )
        video_entity = f"/{entity}/video"
        if video_entity in videos:
            raise DataFactoryVizError("existing RRD verification has duplicate media")
        videos[video_entity] = _sha256_path(video)
    for record in variant_records:
        candidate = str(record.get("candidate_id") or "")
        video = record.get("video")
        if not candidate or not isinstance(video, Path) or not video.is_file():
            raise DataFactoryVizError(
                "existing RRD verification requires every committed candidate video"
            )
        video_entity = f"/augmented/{candidate}/video"
        if video_entity in videos:
            raise DataFactoryVizError("existing RRD verification has duplicate media")
        videos[video_entity] = _sha256_path(video)
        dispositions.add(f"/augmented/{candidate}/disposition")
    return (
        videos,
        dispositions,
        {
            "source_video_entities": len(source_video_records),
            "augmented_video_entities": len(variant_records),
            "augmented_disposition_entities": len(variant_records),
        },
    )


def _bounded_recording_batches(rrd_path: Path):
    try:
        from npa.viz.recordings import load_recording
    except ImportError as exc:  # pragma: no cover - rerun is a runtime dependency
        raise DataFactoryVizError(
            "rerun recording loader is required to verify an existing RRD"
        ) from exc
    _validate_rrd_file(rrd_path)
    yield from _bounded_rrd_batches(load_recording(rrd_path).chunks())


def _validate_rrd_file(path: Path) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise DataFactoryVizError("existing RRD is empty")
    if path.stat().st_size > RRD_VERIFY_MAX_FILE_BYTES:
        raise DataFactoryVizError("RRD exceeds the bounded file-size limit")


def _bounded_rrd_batches(chunks):
    contextual = ((None, chunk) for chunk in chunks)
    for _context, entity, batch in _bounded_context_rrd_batches(contextual):
        yield entity, batch


def _bounded_context_rrd_batches(items):
    chunk_count = row_count = decoded_bytes = 0
    entities: set[str] = set()
    for context, chunk in items:
        chunk_count += 1
        if chunk_count > RRD_VERIFY_MAX_CHUNKS:
            raise DataFactoryVizError("RRD exceeds the decoded chunk limit")
        chunk_rows = int(chunk.num_rows)
        if row_count + chunk_rows > RRD_VERIFY_MAX_ROWS:
            raise DataFactoryVizError("RRD exceeds bounded decoded data limits")
        batch = chunk.to_record_batch()
        if int(batch.num_rows) != chunk_rows:
            raise DataFactoryVizError("RRD chunk row count changed during decoding")
        row_count += chunk_rows
        decoded_bytes += int(batch.nbytes)
        if (
            row_count > RRD_VERIFY_MAX_ROWS
            or int(batch.nbytes) > RRD_VERIFY_MAX_CHUNK_BYTES
            or decoded_bytes > RRD_VERIFY_MAX_DECODED_BYTES
        ):
            raise DataFactoryVizError("RRD exceeds bounded decoded data limits")
        entity = str(chunk.entity_path)
        entities.add(entity)
        if len(entities) > RRD_VERIFY_MAX_ENTITIES:
            raise DataFactoryVizError("RRD exceeds the decoded entity limit")
        yield context, entity, batch


def _first_rrd_blob(value: Any) -> bytes | None:
    if isinstance(value, bytes):
        return value
    if isinstance(value, memoryview):
        return value.tobytes()
    if isinstance(value, (list, tuple)):
        if value and all(type(item) is int and 0 <= item <= 255 for item in value):
            return bytes(value)
        for item in value:
            blob = _first_rrd_blob(item)
            if blob is not None:
                return blob
    return None


def _rrd_blob_identities(scalar: Any) -> list[tuple[int, str]] | None:
    values = getattr(scalar, "values", None)
    if values is not None:
        identities: list[tuple[int, str]] = []
        for index in range(len(values)):
            inner = values[index]
            raw = getattr(inner, "values", None)
            if raw is None or str(getattr(raw, "type", "")) != "uint8":
                identities = []
                break
            view = memoryview(raw.to_numpy(zero_copy_only=False))
            identities.append((len(raw), hashlib.sha256(view).hexdigest()))
        if identities:
            return identities
    blob = _first_rrd_blob(scalar.as_py())
    if blob is None:
        return None
    return [(len(blob), hashlib.sha256(blob).hexdigest())]


def _scan_terminal_rrd(
    rrd_path: Path,
    *,
    expected_videos: set[str],
    expected_dispositions: set[str],
    quality_status: str,
) -> tuple[dict[str, tuple[int, str]], dict[str, bool]]:
    videos = {entity: (0, "") for entity in expected_videos}
    dispositions = {entity: False for entity in expected_dispositions}
    media_bytes = 0
    for entity, batch in _bounded_recording_batches(rrd_path):
        if entity in expected_videos and "AssetVideo:blob" in batch.schema.names:
            for index in range(batch.num_rows):
                blobs = _rrd_blob_identities(batch.column("AssetVideo:blob")[index])
                for size_bytes, blob_digest in blobs or []:
                    media_bytes += size_bytes
                    if media_bytes > RRD_VERIFY_MAX_MEDIA_BYTES:
                        raise DataFactoryVizError("RRD exceeds the decoded media limit")
                    count, digest = videos[entity]
                    videos[entity] = (
                        count + 1,
                        blob_digest if count == 0 else digest,
                    )
        if entity not in expected_dispositions:
            continue
        for name in batch.schema.names:
            if "text" not in name.lower() and "body" not in name.lower():
                continue
            for index in range(batch.num_rows):
                if quality_status in str(batch.column(name)[index].as_py()).upper():
                    dispositions[entity] = True
    return videos, dispositions


def _stable_rrd_value(value: Any) -> Any:
    if isinstance(value, bytes):
        return {
            "sha256": hashlib.sha256(value).hexdigest(),
            "size_bytes": len(value),
        }
    if isinstance(value, memoryview):
        return _stable_rrd_value(value.tobytes())
    if isinstance(value, dict):
        return {
            str(key): _stable_rrd_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_stable_rrd_value(item) for item in value]
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    return str(value)


def _stable_blueprint_value(value: Any) -> Any:
    if isinstance(value, bytes) and len(value) == 16:
        return {"blueprint_uuid": "<normalized>"}
    if isinstance(value, memoryview):
        return _stable_blueprint_value(value.tobytes())
    if isinstance(value, str):
        return _BLUEPRINT_UUID.sub("<blueprint-uuid>", value)
    if isinstance(value, dict):
        return {
            str(key): _stable_blueprint_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        if len(value) == 16 and all(
            type(item) is int and 0 <= item <= 255 for item in value
        ):
            return {"blueprint_uuid": "<normalized>"}
        return [_stable_blueprint_value(item) for item in value]
    return _stable_rrd_value(value)


def _rrd_store_identity(entry: Any) -> list[str]:
    kind = str(entry.kind)
    identity = [kind, str(entry.application_id)]
    if kind != "blueprint":
        identity.append(str(entry.recording_id))
    return identity


def _rrd_semantic_row(entry: Any, entity: str, batch: Any, index: int) -> dict:
    ephemeral = {
        "rerun.controls.RowId",
        "log_time",
        "RecordingInfo:start_time",
    }
    stable = (
        _stable_blueprint_value if str(entry.kind) == "blueprint" else _stable_rrd_value
    )
    names = [name for name in batch.schema.names if name not in ephemeral]
    normalized_entity = stable(entity)
    row = {}
    for name in names:
        scalar = batch.column(name)[index]
        blobs = None if str(entry.kind) == "blueprint" else _rrd_blob_identities(scalar)
        row[name] = (
            [{"size_bytes": size, "sha256": digest} for size, digest in blobs]
            if blobs is not None
            else stable(scalar.as_py())
        )
    return {
        "type": "row",
        "store": _rrd_store_identity(entry),
        "entity": normalized_entity,
        "row": row,
    }


def _insert_semantic_digest(connection: sqlite3.Connection, payload: dict) -> None:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    connection.execute(
        "INSERT INTO semantic_rows(digest) VALUES (?)",
        (hashlib.sha256(encoded).hexdigest(),),
    )


def _semantic_rrd_stores(path: Path):
    try:
        from rerun.chunk import RrdReader
    except ImportError as exc:  # pragma: no cover - rerun is a runtime dependency
        raise DataFactoryVizError(
            "rerun recording loader is required to compare an existing RRD"
        ) from exc
    _validate_rrd_file(path)
    reader = RrdReader(path)
    recordings = reader.recordings()
    if not recordings:
        raise DataFactoryVizError("RRD contains no recording store")
    stores = recordings + reader.blueprints()
    if len(stores) > 1024:
        raise DataFactoryVizError("RRD exceeds the decoded store limit")
    return reader, stores


def _rrd_semantic_sha256(path: Path) -> str:
    """Hash all RRD stores with bounded memory and normalized blueprint UUIDs."""

    reader, stores = _semantic_rrd_stores(path)
    with tempfile.TemporaryDirectory(prefix="npa-rrd-semantic-") as tmp:
        connection = sqlite3.connect(str(Path(tmp) / "rows.sqlite3"))
        try:
            connection.execute("CREATE TABLE semantic_rows (digest TEXT NOT NULL)")
            chunks = (
                (entry, chunk)
                for entry in stores
                for chunk in reader.stream(store=entry)
            )
            for entry in stores:
                _insert_semantic_digest(
                    connection,
                    {"type": "store", "identity": _rrd_store_identity(entry)},
                )
            for entry, entity, batch in _bounded_context_rrd_batches(chunks):
                for index in range(batch.num_rows):
                    payload = _rrd_semantic_row(entry, entity, batch, index)
                    if payload["row"]:
                        _insert_semantic_digest(connection, payload)
            connection.commit()
            digest = hashlib.sha256()
            query = (
                "SELECT digest, COUNT(*) FROM semantic_rows "
                "GROUP BY digest ORDER BY digest"
            )
            for row_digest, count in connection.execute(query):
                digest.update(str(row_digest).encode())
                digest.update(b":")
                digest.update(str(count).encode())
                digest.update(b"\n")
            return digest.hexdigest()
        finally:
            connection.close()


def _verify_existing_rrd_semantics(existing: Path, current: Path) -> None:
    if _rrd_semantic_sha256(existing) != _rrd_semantic_sha256(current):
        raise DataFactoryVizError("existing RRD does not match all current run inputs")


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _source_fidelity_media_evidence(
    local: Path,
    *,
    input_provenance: Any,
    variant_records: list[dict[str, Any]],
) -> dict[str, Any]:
    """Probe the exact VDA media bytes that the synchronized RRD embeds.

    This is intentionally opt-in to the NVIDIA VDA source-fidelity policy so
    existing DIG/IAA/EVG recordings and their defaults remain byte-for-byte
    behaviorally unchanged.
    """

    if not isinstance(input_provenance, dict):
        return {}
    derivation = input_provenance.get("derivation")
    if not isinstance(derivation, dict) or derivation.get("policy") not in {
        "source-fidelity-v2",
        "source-fidelity-v3",
    }:
        return {}
    conditioning_policy = str(derivation["policy"])

    from npa.workflows.data_factory_input import PaidfInputError, probe_video

    def probe(path: Path) -> dict[str, Any]:
        try:
            return {
                **probe_video(path),
                "sha256": _sha256_path(path),
                "byte_size": path.stat().st_size,
            }
        except PaidfInputError as exc:
            raise DataFactoryVizError(
                f"source-fidelity RRD media probe failed for {path.name}"
            ) from exc

    source = local / "input" / "source.mp4"
    conditioning = local / "input" / "conditioning.mp4"
    if not source.is_file() or not conditioning.is_file():
        raise DataFactoryVizError(
            "source-fidelity RRD requires source.mp4 and conditioning.mp4"
        )
    source_media = probe(source)
    conditioning_media = probe(conditioning)
    candidates = [
        {
            "candidate_id": str(record["candidate_id"]),
            "media": probe(record["video"]),
        }
        for record in variant_records
        if isinstance(record.get("video"), Path)
    ]
    if len(candidates) != len(variant_records):
        raise DataFactoryVizError(
            "source-fidelity RRD could not probe every committed candidate"
        )
    candidate_ids = {str(record["candidate_id"]) for record in variant_records}
    control_root = local / "cosmos_control"
    selected_control_videos: dict[Path, str] = {}
    controls_by_candidate = {candidate_id: 0 for candidate_id in candidate_ids}
    for iteration, augment_root in _augment_roots(local):
        manifest = _read_json(augment_root / "manifest.json")
        if not isinstance(manifest, dict):
            continue
        for variant in _validated_viz_manifest(manifest):
            clip = str(variant.get("clip") or "").strip()
            candidate_id = f"iteration-{iteration}/{clip}" if iteration else clip
            if candidate_id not in candidate_ids:
                continue
            control_uris = variant.get("control_uris") or {}
            if not isinstance(control_uris, dict) or not control_uris:
                raise DataFactoryVizError(
                    f"source-fidelity RRD candidate {candidate_id} has no committed control video"
                )
            for uri in control_uris.values():
                value = str(uri or "")
                marker = "/cosmos_control/"
                if marker not in value or not value.lower().endswith(".mp4"):
                    raise DataFactoryVizError(
                        f"source-fidelity RRD candidate {candidate_id} has an invalid control URI"
                    )
                path = control_root / value.split(marker, 1)[1]
                selected_control_videos[path] = candidate_id
                controls_by_candidate[candidate_id] += 1
    missing_candidate_controls = sorted(
        candidate_id
        for candidate_id, count in controls_by_candidate.items()
        if count == 0
    )
    if missing_candidate_controls:
        raise DataFactoryVizError(
            "source-fidelity RRD requires a committed control video for every candidate"
        )
    missing_controls = [
        path for path in sorted(selected_control_videos) if not path.is_file()
    ]
    if missing_controls:
        raise DataFactoryVizError(
            "source-fidelity RRD is missing a manifest-referenced control video"
        )
    controls = [
        {
            "candidate_id": selected_control_videos[path],
            "signal": path.relative_to(control_root).as_posix(),
            "media": probe(path),
        }
        for path in sorted(selected_control_videos)
    ]
    return {
        "schema": "npa.paidf.vda.media-evidence.v1",
        "conditioning_policy": conditioning_policy,
        "synchronization_timeline": "video_time",
        "source": source_media,
        "conditioning": conditioning_media,
        "controls": controls,
        "generated_candidates": candidates,
        "temporal_alignment": _validated_source_fidelity_alignment(
            derivation.get("temporal_alignment"),
            source_frame_count=int(source_media["frame_count"]),
            conditioning_frame_count=int(conditioning_media["frame_count"]),
        ),
    }


def _validated_source_fidelity_alignment(
    value: Any,
    *,
    source_frame_count: int,
    conditioning_frame_count: int,
) -> dict[str, Any]:
    """Validate the exact source-reference map before using it as RRD timing."""

    if not isinstance(value, dict) or not isinstance(value.get("frame_map"), list):
        raise DataFactoryVizError(
            "source-fidelity RRD requires a complete temporal-alignment map"
        )
    frame_map = value["frame_map"]
    if len(frame_map) != conditioning_frame_count:
        raise DataFactoryVizError(
            "source-fidelity temporal alignment does not match conditioning frames"
        )
    for expected_index, item in enumerate(frame_map):
        if not isinstance(item, dict):
            raise DataFactoryVizError("source-fidelity temporal alignment is malformed")
        source_index = item.get("source_index")
        output_index = item.get("output_index")
        output_timestamp = item.get("output_timestamp_seconds")
        if (
            output_index != expected_index
            or not isinstance(source_index, int)
            or not 0 <= source_index < source_frame_count
            or not isinstance(output_timestamp, (int, float))
            or not math.isclose(
                float(output_timestamp), expected_index / 16, rel_tol=0, abs_tol=1e-9
            )
        ):
            raise DataFactoryVizError("source-fidelity temporal alignment is malformed")
    return value


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
    """Group images by their full relative parent and original frame identity."""
    indexed: dict[str, dict[int, Path]] = {}
    for frame in _image_files(root):
        parent = frame.parent
        name = "frames" if parent == root else parent.relative_to(root).as_posix()
        frames = indexed.setdefault(name, {})
        index = _frame_index(frame.stem)
        if index in frames:
            raise DataFactoryVizError(
                f"duplicate frame index {index} in image group {name!r}: "
                f"{frames[index].name!r} and {frame.name!r}"
            )
        frames[index] = frame
    return {
        name: [frames[index] for index in sorted(frames)]
        for name, frames in indexed.items()
    }


def _log_nurec_entities(rr: Any, rec: Any, local: Path) -> int:
    """Log a neural-reconstruction run's renders as Rerun entities.

    ``novel_view/<camera>`` are the rig-offset views rendered from the trained
    Gaussians (the point of the capability), and ``reconstruction/<group>`` are
    NRE's own validation renders. Both entity prefixes are registered in
    ``npa.cli.agent_recordings.RUN_ENTITY_MARKERS`` so the agent recognises the
    recording as real run data.
    """
    logged = 0
    for directory, prefix in (
        ("novel_views", "novel_view"),
        ("reconstruction", "reconstruction"),
    ):
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
        control_video = signal_dir.with_suffix(".mp4")
        if signal_dir.is_dir() or control_video.is_file():
            frames = _subsample(
                sorted(_image_files(signal_dir)), RRD_MAX_FRAMES_PER_ENTITY
            )
            entity = f"control/{clip}/{signal_dir.name}"
            if control_video.is_file():
                _log_video_asset(rr, rec, f"{entity}/video", control_video)
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


def _candidate_evaluation(
    local: Path, iteration: int, clip: str, review: _PaidfReview
) -> dict[str, Any]:
    grade_root = local / "grade"
    grade_dir = (
        grade_root / f"iteration-{iteration}" / "ranking" if iteration else grade_root
    )
    try:
        from npa.workbench.cosmos_evaluator import RESULT_FILENAME as result_name
    except Exception:  # noqa: BLE001
        result_name = "cosmos_evaluator.json"
    return review.candidate_evaluation(grade_dir / result_name, clip)


def _candidate_quality_fields(evaluation: dict[str, Any]) -> dict[str, Any]:
    attributes = evaluation.get("attribute_verification") or {}
    hallucination = evaluation.get("hallucination") or {}
    return {
        "candidate_passed": evaluation.get("passed") is True,
        "score": evaluation.get("score"),
        "failed_attributes": [
            str(check.get("variable") or "unknown")
            for check in attributes.get("checks", [])
            if check.get("passed") is not True
        ],
        "attribute_results": attributes.get("checks", []),
        "hallucination_status": "passed"
        if hallucination.get("passed") is True
        else "failed",
        "hallucination": hallucination,
        "temporal_consistency": evaluation.get("temporal_consistency"),
        "appearance_fidelity": evaluation.get("appearance_fidelity"),
    }


def _candidate_disposition_document(
    local: Path,
    *,
    iteration: int,
    clip: str,
    candidate_id: str,
    quality_status: str,
    disposition: Any,
    review: _PaidfReview,
) -> str:
    """Truthful per-candidate disposition shown beside its actual media."""
    evaluation = _candidate_evaluation(local, iteration, clip, review)
    candidate_status = (
        "ACCEPTED"
        if quality_status == "ACCEPTED" and evaluation.get("passed") is True
        else "REJECTED"
    )
    summary = review.payload(
        {
            "candidate_id": candidate_id,
            "iteration": iteration,
            "clip_id": clip,
            "run_disposition": quality_status,
            "promotion_eligible": quality_status == "ACCEPTED"
            and evaluation.get("passed") is True,
            "candidate_disposition": candidate_status.lower(),
            **_candidate_quality_fields(evaluation),
            "source_comparison_entity": "source/* or conditioning/derived",
            "output_media_entity": f"augmented/{candidate_id}",
            "final_disposition": disposition if isinstance(disposition, dict) else {},
        },
        "candidate",
    )
    summary["source_reports"] = [
        source["source_report"]
        for source in (evaluation, disposition)
        if isinstance(source, dict) and "source_report" in source
    ]
    return (
        f"# {candidate_status} — candidate `{review.identity(candidate_id)}`\n\n"
        + "This panel is review evidence only. Rejected media is never relabeled, "
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


def _augmentation_label(clip_dir: Path, review: _PaidfReview) -> str:
    meta_path = clip_dir / "metadata.json"
    if not meta_path.is_file():
        return ""
    meta = review.read(meta_path, "metadata")
    variables = meta.get("variables", {}) if isinstance(meta, dict) else {}
    label = ", ".join(f"{k}={v}" for k, v in variables.items())
    if label:
        label += "\n\n" + _json_block("Metadata source", meta["source_report"])
    return label


def _read_json(path: Path) -> Any:
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return None


def _json_block(title: str, payload: Any) -> str:
    body = json.dumps(payload, indent=2, sort_keys=True)
    return f"## {title}\n\n```json\n{body}\n```\n"


def _load_stage_docs(
    local: Path, *, review: _PaidfReview | None = None
) -> dict[str, str]:
    """Build per-stage markdown docs (scenarios, hallucination/grade, curation,
    finalize, and a stage log) so the full pipeline is viewable in the Rerun panel.

    Every entry is optional — a stage that did not persist its artifact is simply
    skipped, so this stays robust for partial runs.
    """
    docs: dict[str, str] = {}
    stage_log: list[str] = []
    review = review or _PaidfReview(local)

    # --- Neural-reconstruction stages (no-ops for a data-factory run) -------------
    docs.update(_load_nurec_docs(local, stage_log))

    # Stage 1 — sampled scenarios (Config Generation). This is the "various
    # scenarios" the augment stage multiplies over.
    cfg = review.read(local / "configs" / "manifest.json", "config")
    if isinstance(cfg, dict):
        combos = cfg.get("augmentations") or []
        lines = [
            f"**Scene:** {cfg.get('scene', 'n/a')}",
            f"**Scenarios sampled:** {len(combos)}",
            "",
        ]
        for i, combo in enumerate(combos):
            if isinstance(combo, dict):
                prompt = str(combo.get("prompt") or "")
                attrs = ", ".join(f"{k}={v}" for k, v in combo.items() if k != "prompt")
                lines.append(f"- **scenario {i}** — {attrs}")
                if prompt:
                    lines.append(f"    - prompt: _{prompt}_")
        docs["pipeline/1_scenarios"] = (
            "## Config generation — sampled scenarios\n\n" + "\n".join(lines) + "\n"
        )
        docs["pipeline/1_scenarios"] += _json_block(
            "Scenario source", cfg["source_report"]
        )
        stage_log.append(f"configs: {len(combos)} scenario(s) sampled")

    input_provenance = review.read(local / "input" / "provenance.json", "input")
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
    aug = review.read(aug_dir / "manifest.json", "augment")
    if isinstance(aug, dict):
        variants = aug.get("variants") or aug.get("clips") or []
        docs["pipeline/2_augment"] = _json_block("Augment — generated variants", aug)
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
        ev = review.read(grade_dir / name, "evaluator")
        if isinstance(ev, dict):
            grade_docs.append(
                _json_block("Evaluator — integrity and appearance checks", ev)
            )
            stage_log.append(
                f"grade: score={ev.get('score')}, status={ev.get('status', 'n/a')}"
            )
            break
    dec = review.read(grade_dir / "decision.json", "quality")
    if isinstance(dec, dict):
        grade_docs.append(_json_block("Quality gate decision", dec))
        stage_log.append(f"grade: decision={dec.get('decision', 'n/a')}")
    disposition = review.read(grade_root / "quality_disposition.json", "quality")
    if isinstance(disposition, dict):
        grade_docs.append(_json_block("Final quality disposition", disposition))
        stage_log.append(
            "grade: disposition="
            f"{disposition.get('quality_status', 'n/a')}, "
            f"score={disposition.get('score', 'n/a')}"
        )
    if grade_docs:
        docs["pipeline/3_grade"] = "\n".join(grade_docs)

    # Curation reports from both real components, when available.
    curator = review.read(local / "curation" / "cosmos_curator.json", "curation")
    if isinstance(curator, dict):
        docs["pipeline/4_cosmos_curator"] = _json_block(
            "Cosmos Curator report", curator
        )
        stage_log.append(f"cosmos-curator: {curator.get('clip_count', 0)} clip(s)")
    cur = review.read(local / "curation" / "report.json", "curation")
    if isinstance(cur, dict):
        docs["pipeline/4_curation"] = _json_block("Curation report", cur)
        stage_log.append(
            f"curation: {cur.get('augmented_clips', 0)} clip(s), "
            f"multiply={(cur.get('multiply') or {}).get('mode', 'n/a')}"
        )

    # Finalize aggregate report.
    fin = review.read(local / "reports" / "final.json", "final")
    if isinstance(fin, dict):
        docs["pipeline/5_finalize"] = _json_block("Finalize — aggregate report", fin)
        stage_log.append(
            f"finalize: {fin.get('artifact_count', 0)} artifacts, "
            f"multiply_mode={fin.get('multiply_mode', 'n/a')}"
        )

    if stage_log:
        docs["pipeline/0_log"] = (
            "## Pipeline stage log\n\n"
            + "\n".join(f"- {line}" for line in stage_log)
            + "\n"
        )
    return docs


def _read_yaml(path: Path) -> Any:
    try:
        import yaml

        return yaml.safe_load(path.read_text())
    except Exception:  # noqa: BLE001 - optional artifact, optional dependency
        return None


def _lineage_fields(payload: dict, names: str) -> dict:
    return {name: payload[name] for name in names.split() if name in payload}


def _read_lineage(path: Path) -> dict | None:
    if not path.exists():
        return None
    payload = _read_json(path)
    if not isinstance(payload, dict) or not payload:
        raise DataFactoryVizError(f"NuRec lineage is unreadable: {path.name}")
    return payload


def _colmap_conversion_lineage(report: dict) -> dict:
    """Keep conversion facts without embedding source filenames or private paths."""
    payload = _lineage_fields(
        report,
        "schema_version status engine counts poses_component_group time_mapping point_filter",
    )
    payload["source"] = _lineage_fields(
        report.get("source", {}), "archive_sha256 counts origin_points_filtered"
    )
    payload["converter"] = _lineage_fields(
        report.get("converter", {}), "revision target runtime_sha256 license"
    )
    return payload


def _load_colmap_docs(local: Path, stage_log: list[str]) -> dict[str, str]:
    """Bind COLMAP capture, conversion and derived rig facts to their source bytes."""
    if not any((local / relative).exists() for relative in _COLMAP_LINEAGE_MARKERS):
        return {}
    docs: dict[str, str] = {}
    for entity, relative, title, fields in _COLMAP_LINEAGE_ARTIFACTS:
        path = local / relative
        report = _read_lineage(path)
        if report is None:
            raise DataFactoryVizError(f"NuRec lineage artifact is missing: {relative}")
        payload = (
            _colmap_conversion_lineage(report)
            if entity == "conversion"
            else _lineage_fields(report, fields)
        )
        payload["artifact_sha256"] = _sha256_path(path)
        docs[f"provenance/{entity}"] = _json_block(title, payload)
    if docs:
        docs["pipeline/1_ncore"] = (
            "## NCore input capture\n\n"
            "_Capture counts describe conversion input/output, not NRE training coverage. "
            "Photographic ordering is not synchronized capture time; sparse SfM points "
            "are not physical LiDAR._\n\n" + "\n".join(docs.values())
        )
        stage_log.append("ncore: COLMAP capture lineage recorded from run artifacts")
    return docs


def _load_nurec_docs(local: Path, stage_log: list[str]) -> dict[str, str]:
    """Describe actual NuRec artifacts, including either input capture format."""
    docs = _load_colmap_docs(local, stage_log)
    if not docs:
        docs = _load_ncore_manifest_docs(local, stage_log)
    docs.update(_load_nurec_metrics_docs(local, stage_log))
    docs.update(_load_novel_view_docs(local, stage_log))
    if (local / "novel_views").is_dir():
        docs["provenance/rrd_review"] = _json_block(
            "Novel-view review settings",
            {
                "schema": "npa.nurec.rrd-review.v1",
                "max_frames_per_entity": RRD_MAX_FRAMES_PER_ENTITY,
                "max_frame_dim": RRD_MAX_FRAME_DIM,
                "jpeg_quality": RRD_JPEG_QUALITY,
            },
        )
    return docs


def _load_ncore_manifest_docs(local: Path, stage_log: list[str]) -> dict[str, str]:
    """Describe the preconverted-NCore fetch manifest when present."""
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
        docs["pipeline/1_ncore"] = (
            "## NCore input capture\n\n" + "\n".join(lines) + "\n"
        )
        stage_log.append(
            f"ncore: {manifest.get('scene', 'n/a')} "
            f"({manifest.get('shard_count', 0)} shard(s), "
            f"{len(manifest.get('camera_ids') or [])} camera(s))"
        )

    return docs


def _load_nurec_metrics_docs(local: Path, stage_log: list[str]) -> dict[str, str]:
    """Describe the metrics emitted by NRE validation."""
    docs: dict[str, str] = {}
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

    return docs


def _load_novel_view_docs(local: Path, stage_log: list[str]) -> dict[str, str]:
    """Describe the rendered novel-view frames and videos."""
    docs: dict[str, str] = {}
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
        "_Descriptive per-frame labels of the generated video output. This is "
        "captioning, not the quality gate — see `pipeline/3_grade` for the "
        "attribute-verify / hallucination check (score + promote/loop_back decision)._\n\n"
    ),
}


def _load_captions(
    local: Path, *, review: _PaidfReview | None = None
) -> dict[str, str]:
    out: dict[str, str] = {}
    review = review or _PaidfReview(local)
    for name in ("labeled_original", "labeled_augmented"):
        cj = local / name / "captions.json"
        if not cj.is_file():
            continue
        payload = review.read(cj, "captions")
        items = payload.get("captions", []) if isinstance(payload, dict) else []
        body = "\n\n".join(
            f"- {c.get('image')}: {c.get('caption')}"
            for c in items[:12]
            if isinstance(c, dict)
        )
        if body:
            # Prefix a self-identifying header so a caption panel is never confused
            # with the VLM eval / hallucination grade panel in the Rerun grid.
            out[name] = _CAPTION_HEADERS.get(name, "") + body
            out[name] += "\n\n" + _json_block(
                "Caption source",
                {
                    "model": payload.get("model"),
                    **payload["source_report"],
                },
            )
    return out


def _materialize_run(
    input_uri: str,
    dest: Path,
    *,
    storage_client: "StorageClient | None",
    require_colmap_lineage: bool = False,
) -> Path:
    if not input_uri.startswith("s3://"):
        return Path(input_uri)
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    dest.mkdir(parents=True, exist_ok=True)
    root = input_uri.rstrip("/")
    for sub in RUN_SUBDIRS:
        if sub == "source":
            continue
        try:
            client.download_path(f"{root}/{sub}/", str(dest / sub))
        except Exception:
            # Optional subtrees (labeled_*) may not exist; input/augmented drive the recording.
            continue
    _download_colmap_lineage(client, root, dest, required=require_colmap_lineage)
    return dest


def _download_colmap_lineage(client, root: str, dest: Path, *, required: bool) -> None:
    lineage_paths = _COLMAP_LINEAGE_PATHS if required else ("source/attribution.json",)
    for relative in lineage_paths:
        local_path = dest / relative
        try:
            client.download_file(f"{root}/{relative}", str(local_path))
        except Exception:
            if required:
                raise


def _publish(
    local_path: str, output_uri: str, *, storage_client: "StorageClient | None"
) -> str:
    if not output_uri.startswith("s3://"):
        out = Path(output_uri)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_bytes(Path(local_path).read_bytes())
        return str(out)
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    from npa.agent_backend.publication_reader import (
        publication_root_from_canonical_uri,
    )
    from npa.workflows.sim2real.publication import (
        remote_object_snapshot,
        replace_unjournaled_legacy_file,
    )

    try:
        publication_root = publication_root_from_canonical_uri(output_uri)
    except ValueError:
        publication_root = ""
    if publication_root:
        lock_uri = f"{publication_root}/reports/.sim2real-publication.json"
        return replace_unjournaled_legacy_file(
            client,
            Path(local_path),
            output_uri,
            snapshot=None,
            lock_snapshot=remote_object_snapshot(client, lock_uri),
        )
    client.put_bytes_conditional(
        Path(local_path).read_bytes(),
        output_uri,
        if_none_match=True,
        content_type="application/octet-stream",
    )
    return output_uri
