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
import re
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from npa.clients.storage import StorageClient

APPLICATION_ID = "physical-ai-data-factory"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}


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
        return np.asarray(im.convert("RGB"))


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

        for png in sorted((local / "input").rglob("*.png")):
            clip = "_".join(png.stem.split("_")[:2]) or "clip"
            _set_frame(rr, rec, _frame_index(png.stem))
            rr.log(f"input/{clip}", _image(rr, _load_rgb(png)), recording=rec)
            logged += 1

        aug_root = local / "cosmos_augmented"
        if aug_root.is_dir():
            for d in sorted(p for p in aug_root.iterdir() if p.is_dir()):
                label = _augmentation_label(d)
                entity = f"augmented/{d.name}"
                for png in sorted(d.glob("*.png")):
                    _set_frame(rr, rec, _frame_index(png.stem))
                    rr.log(entity, _image(rr, _load_rgb(png)), recording=rec)
                    logged += 1
                if label:
                    rr.log(entity, rr.TextDocument(f"{d.name}: {label}"), static=True, recording=rec)

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
    for name in ("vlm_eval_stub.json", "vlm_eval.json"):
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
            out[name] = body
    return out


def _materialize_run(input_uri: str, dest: Path, *, storage_client: "StorageClient | None") -> Path:
    if not input_uri.startswith("s3://"):
        return Path(input_uri)
    from npa.clients.storage import StorageClient

    client = storage_client or StorageClient.from_environment()
    dest.mkdir(parents=True, exist_ok=True)
    root = input_uri.rstrip("/")
    for sub in (
        "input",
        "cosmos_augmented",
        "labeled_original",
        "labeled_augmented",
        "configs",
        "grade",
        "curation",
        "reports",
    ):
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
