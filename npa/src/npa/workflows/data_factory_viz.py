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

import json
import os
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient

APPLICATION_ID = "physical-ai-data-factory"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}

#: Run sub-directories materialized from S3 before building a recording. Covers
#: both producers: the data-factory blueprint (input/cosmos_augmented/labeled_*/
#: configs/grade/curation) and the NuRec neural-reconstruction workflow
#: (ncore/reconstruction/novel_views). Missing subtrees are skipped.
RUN_SUBDIRS = (
    "input",
    "cosmos_augmented",
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

    with tempfile.TemporaryDirectory(prefix="npa-df-viz-") as tmp:
        local = _materialize_run(input_uri, Path(tmp) / "run", storage_client=storage_client)
        captions = _load_captions(local)

        rec = rr.RecordingStream(app_id, recording_id=run_id)
        logged = 0

        input_root = local / "input"
        for frame in _subsample(_image_files(input_root), RRD_MAX_FRAMES_PER_ENTITY):
            _set_frame(rr, rec, _frame_index(frame.stem))
            _log_frame(rr, rec, f"input/{_input_entity(frame, input_root)}", _load_rgb(frame))
            logged += 1

        aug_root = local / "cosmos_augmented"
        if aug_root.is_dir():
            for d in sorted(p for p in aug_root.iterdir() if p.is_dir()):
                label = _augmentation_label(d)
                entity = f"augmented/{d.name}"
                for png in _subsample(sorted(d.glob("*.png")), RRD_MAX_FRAMES_PER_ENTITY):
                    _set_frame(rr, rec, _frame_index(png.stem))
                    _log_frame(rr, rec, entity, _load_rgb(png))
                    logged += 1
                if label:
                    rr.log(entity, rr.TextDocument(f"{d.name}: {label}"), static=True, recording=rec)

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

        if logged == 0:
            raise DataFactoryVizError(
                f"no input/augmented frames found under {input_uri}; nothing to visualize"
            )

        out_path = Path(tmp) / "sim2real.rrd"
        rr.save(str(out_path), recording=rec)
        written_uri = _publish(str(out_path), output_uri, storage_client=storage_client)

    return {
        "status": "completed",
        "run_id": run_id,
        "input_uri": input_uri,
        "output_uri": written_uri,
        "frames_logged": logged,
    }


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

    # Augment fan-out — how many Cosmos Transfer 2.5 variants were produced.
    aug = _read_json(local / "cosmos_augmented" / "manifest.json")
    if isinstance(aug, dict):
        variants = aug.get("variants") or aug.get("clips") or []
        docs["pipeline/2_augment"] = _json_block("Augment — Cosmos Transfer 2.5 (multiply)", aug)
        stage_log.append(
            f"augment: {aug.get('variant_count', len(variants))} variant(s), "
            f"mode={aug.get('mode', 'n/a')}, input_conditioned={aug.get('input_conditioned')}"
        )

    # Evaluate & Validate — the hallucination / attribute-verification grade.
    grade_dir = local / "grade"
    grade_docs: list[str] = []
    try:
        from npa.workbench.vlm_eval import RESULT_FILENAME as _vlm_result_filename
    except Exception:  # noqa: BLE001
        _vlm_result_filename = "vlm_eval_stub.json"
    for name in (_vlm_result_filename, "vlm_eval.json"):
        ev = _read_json(grade_dir / name)
        if isinstance(ev, dict):
            grade_docs.append(_json_block("Attribute verification / hallucination check (VLM)", ev))
            stage_log.append(f"grade: vlm score={ev.get('score')}, model={ev.get('model', 'n/a')}")
            break
    dec = _read_json(grade_dir / "decision.json")
    if isinstance(dec, dict):
        grade_docs.append(_json_block("Quality gate decision", dec))
        stage_log.append(f"grade: decision={dec.get('decision', 'n/a')}")
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
        "## Original-frame captions — Token Factory VLM\n\n"
        "_Descriptive per-frame labels of the SOURCE clip. This is captioning, "
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
    return client.upload_file(local_path, output_uri)
