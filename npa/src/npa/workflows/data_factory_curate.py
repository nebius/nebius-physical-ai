"""Real FiftyOne (Voxel51) Brain curation for the Physical AI Data Factory.

The paidf ``curate`` stage historically wrote a lightweight report (clip/frame
counts + multiply mode) -- a stand-in for FiftyOne review. This module runs
*real* FiftyOne Brain curation over the augmented scenario variants when FiftyOne
is importable (i.e. inside the ``npa-fiftyone`` workbench image):

1. builds a real ``fiftyone.Dataset`` from the augmented variant frames (one
   representative frame per Cosmos Transfer variant, tagged with its sampled
   appearance variables) plus the source input frames;
2. computes a per-sample embedding from the real pixels (no GPU / no torch --
   Pillow + numpy, which the image already ships);
3. runs FiftyOne Brain end to end -- ``compute_uniqueness`` (per-sample novelty),
   ``compute_similarity`` + ``find_duplicates`` (near-duplicate detection), and
   ``compute_visualization`` (a 2D embedding for the review grid); and
4. turns those signals into an actual curation decision: keep the most-unique
   representative of every near-duplicate cluster and flag redundant / low-signal
   variants for review.

The module is split into (a) pure, dependency-light logic (union-find,
``select_curated``, report assembly) that is unit-tested WITHOUT FiftyOne and (b)
a thin :func:`run_curation` glue that imports FiftyOne + Pillow/numpy lazily and
drives the Brain methods. Environments without FiftyOne (unit tests, the dev-VM
worktree python) raise :class:`FiftyoneUnavailable`, and callers fall back to the
report-only path -- so the pipeline never regresses when the image is absent.
"""

from __future__ import annotations

import logging
from typing import Any, Callable

_log = logging.getLogger(__name__)

CURATION_ENGINE_FIFTYONE = "fiftyone-brain"
CURATION_ENGINE_REPORT_ONLY = "report-only"

# Cosine-distance threshold below which two variant embeddings are treated as
# near-duplicates by FiftyOne's similarity index. Appearance-only augmentation
# variants are meant to be visibly distinct, so a small threshold catches combos
# that collapsed to near-identical renders (a real curation signal).
DEFAULT_DEDUP_THRESHOLD = 0.10
# Kept samples whose uniqueness falls at/below this quantile are flagged
# "redundant" (kept, but surfaced for human review), not dropped.
DEFAULT_REDUNDANT_QUANTILE = 0.15


class FiftyoneUnavailable(RuntimeError):
    """Raised when FiftyOne (or its curation deps) cannot be imported."""


# ---------------------------------------------------------------------------
# Pure logic (no FiftyOne / numpy / PIL) -- unit-tested directly.
# ---------------------------------------------------------------------------


def _quantile(values: list[float], q: float) -> float:
    """Linear-interpolated quantile of ``values`` (empty -> 0.0)."""
    if not values:
        return 0.0
    if q <= 0:
        return min(values)
    if q >= 1:
        return max(values)
    ordered = sorted(values)
    pos = q * (len(ordered) - 1)
    lo = int(pos)
    hi = min(lo + 1, len(ordered) - 1)
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _cluster_ids(sample_ids: list[str], duplicate_pairs: list[tuple[str, str]]) -> list[list[str]]:
    """Union-find the ids into near-duplicate clusters (singletons included)."""
    valid = set(sample_ids)
    parent = {sid: sid for sid in sample_ids}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    for a, b in duplicate_pairs:
        if a in valid and b in valid:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb

    groups: dict[str, list[str]] = {}
    for sid in sample_ids:
        groups.setdefault(find(sid), []).append(sid)
    return [sorted(members) for members in groups.values()]


def uniqueness_summary(uniqueness: dict[str, float]) -> dict[str, Any]:
    """Min / max / mean / count summary of per-sample uniqueness scores."""
    vals = [float(v) for v in uniqueness.values() if _is_finite(v)]
    if not vals:
        return {"count": 0, "min": 0.0, "max": 0.0, "mean": 0.0}
    return {
        "count": len(vals),
        "min": round(min(vals), 6),
        "max": round(max(vals), 6),
        "mean": round(sum(vals) / len(vals), 6),
    }


def select_curated(
    sample_ids: list[str],
    uniqueness: dict[str, float],
    duplicate_pairs: list[tuple[str, str]],
    *,
    redundant_quantile: float = DEFAULT_REDUNDANT_QUANTILE,
) -> dict[str, Any]:
    """Curate a set of variants from uniqueness + near-duplicate signals.

    For every near-duplicate cluster the most-unique member is kept and the rest
    are dropped (``reason: near_duplicate``, pointing at the kept representative).
    Kept members whose uniqueness sits at/below ``redundant_quantile`` are flagged
    ``redundant`` (kept, but surfaced for review). Pure/deterministic.
    """
    ids = sorted(set(sample_ids))

    def _u(sid: str) -> float:
        return float(uniqueness.get(sid, 0.0))

    clusters = _cluster_ids(ids, duplicate_pairs)
    kept: list[str] = []
    dropped: list[dict[str, Any]] = []
    near_dupe_clusters: list[dict[str, Any]] = []
    for members in clusters:
        if len(members) == 1:
            kept.append(members[0])
            continue
        # Highest uniqueness wins; stable tiebreak on id for determinism.
        rep = sorted(members, key=lambda m: (-_u(m), m))[0]
        kept.append(rep)
        losers = [m for m in members if m != rep]
        for m in losers:
            dropped.append(
                {
                    "id": m,
                    "reason": "near_duplicate",
                    "representative": rep,
                    "uniqueness": round(_u(m), 6),
                }
            )
        near_dupe_clusters.append(
            {"representative": rep, "members": sorted(members), "dropped": sorted(losers)}
        )

    kept = sorted(kept)
    threshold = _quantile([_u(s) for s in ids], redundant_quantile) if ids else 0.0
    redundant = sorted(s for s in kept if _u(s) <= threshold)

    return {
        "kept": kept,
        "dropped": sorted(dropped, key=lambda d: d["id"]),
        "kept_count": len(kept),
        "dropped_count": len(dropped),
        "near_duplicate_clusters": sorted(near_dupe_clusters, key=lambda c: c["representative"]),
        "near_duplicate_count": len(dropped),
        "redundant": redundant,
        "redundant_count": len(redundant),
        "redundant_uniqueness_threshold": round(threshold, 6),
    }


def merge_curation_into_report(
    base_report: dict[str, Any],
    *,
    engine: str,
    fiftyone_version: str,
    embedding_kind: str,
    dedup_threshold: float,
    uniqueness: dict[str, float],
    selection: dict[str, Any],
    visualization: list[dict[str, Any]] | None,
    fields: list[str],
    warn: str = "",
    uniqueness_method: str = "fiftyone-brain",
) -> dict[str, Any]:
    """Merge FiftyOne Brain results into the base (v1) curation report.

    Keeps every existing v1 field (so the finalize report + agent Voxel51 summary
    stay backward compatible) and adds a ``curation_engine`` tag plus a
    ``fiftyone`` block with the real Brain outputs and per-sample scores.
    """
    report = dict(base_report)
    report["curation_engine"] = engine
    per_sample = {
        sid: {
            "uniqueness": round(float(score), 6),
            "kept": sid in set(selection.get("kept", [])),
            "redundant": sid in set(selection.get("redundant", [])),
        }
        for sid, score in uniqueness.items()
    }
    fo_block: dict[str, Any] = {
        "engine": engine,
        "fiftyone_version": fiftyone_version,
        "embedding_kind": embedding_kind,
        "dedup_threshold": dedup_threshold,
        "fields": sorted(fields),
        "brain": {
            "uniqueness": uniqueness_summary(uniqueness),
            "uniqueness_method": uniqueness_method,
            "near_duplicate_clusters": selection.get("near_duplicate_clusters", []),
            "near_duplicate_count": selection.get("near_duplicate_count", 0),
            "visualization_method": "pca" if visualization else "",
        },
        "selection": {
            "kept": selection.get("kept", []),
            "dropped": selection.get("dropped", []),
            "kept_count": selection.get("kept_count", 0),
            "dropped_count": selection.get("dropped_count", 0),
            "redundant": selection.get("redundant", []),
            "redundant_count": selection.get("redundant_count", 0),
        },
        "samples": per_sample,
    }
    if visualization:
        fo_block["visualization"] = visualization
    if warn:
        fo_block["warn"] = warn[:300]
    report["fiftyone"] = fo_block
    # Curated counts at the top level so downstream readers (finalize / agent)
    # don't have to reach into the nested block.
    report["curated_kept"] = selection.get("kept_count", 0)
    report["curated_dropped"] = selection.get("dropped_count", 0)
    return report


# ---------------------------------------------------------------------------
# FiftyOne-backed glue (imports FiftyOne + Pillow/numpy lazily).
# ---------------------------------------------------------------------------


def _image_embedding(path: str) -> list[float]:
    """A small, GPU-free image embedding: downsampled RGB + per-channel histogram.

    Appearance-only augmentation varies lighting / background / cloth-color /
    surface, which a coarse spatial-color descriptor separates well enough for
    uniqueness + near-duplicate detection. Requires Pillow + numpy (present in the
    npa-fiftyone image).
    """
    import numpy as np
    from PIL import Image

    with Image.open(path) as im:
        rgb = im.convert("RGB")
        small = rgb.resize((16, 16))
        arr = np.asarray(small, dtype=np.float64) / 255.0
        spatial = arr.reshape(-1)
        hist = []
        full = np.asarray(rgb, dtype=np.float64)
        for ch in range(3):
            counts, _ = np.histogram(full[:, :, ch], bins=8, range=(0, 255))
            total = counts.sum() or 1.0
            hist.append(counts / total)
        hist_vec = np.concatenate(hist)
    return np.concatenate([spatial, hist_vec]).astype(np.float64).tolist()


def _is_finite(value: Any) -> bool:
    import math

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _embedding_novelty(order: list[str], emb: Any) -> dict[str, float]:
    """Deterministic per-sample novelty = normalized mean cosine distance to peers.

    Used as a robust fallback when FiftyOne's ``compute_uniqueness`` returns NaN or
    a degenerate (all-equal) result -- common for very small sample sets. Keeps the
    curation report meaningful (and JSON-valid, since NaN is not legal JSON).
    """
    import numpy as np

    n = int(getattr(emb, "shape", [0])[0]) if emb is not None else 0
    if n == 0:
        return {}
    if n == 1:
        return {order[0]: 1.0}
    norm = emb / (np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12)
    sim = norm @ norm.T
    np.fill_diagonal(sim, np.nan)
    dist = 1.0 - np.nanmean(sim, axis=1)
    lo, hi = float(np.min(dist)), float(np.max(dist))
    if hi - lo < 1e-12:
        vals = [0.5] * n
    else:
        vals = ((dist - lo) / (hi - lo)).tolist()
    return {order[i]: round(float(vals[i]), 6) for i in range(n)}


def _augmented_representatives(keys: list[str], augment_prefix: str) -> dict[str, dict[str, Any]]:
    """Map clip-id -> {frame_key, meta_key} for the augmented variants."""
    by_clip: dict[str, dict[str, Any]] = {}
    for key in keys:
        if not key.startswith(augment_prefix):
            continue
        rel = key[len(augment_prefix):]
        parts = rel.split("/")
        if len(parts) < 2 or not parts[0]:
            continue
        clip = parts[0]
        entry = by_clip.setdefault(clip, {"frames": [], "meta": ""})
        low = key.lower()
        if low.endswith(".png"):
            entry["frames"].append(key)
        elif parts[-1] == "metadata.json":
            entry["meta"] = key
    reps: dict[str, dict[str, Any]] = {}
    for clip, entry in by_clip.items():
        if entry["frames"]:
            reps[clip] = {"frame_key": sorted(entry["frames"])[0], "meta_key": entry["meta"]}
    return reps


def run_curation(
    *,
    keys: list[str],
    augment_prefix: str,
    base_report: dict[str, Any],
    download_key: Callable[[str, str], str],
    read_json: Callable[[str], dict[str, Any] | None],
    workdir: str,
    dedup_threshold: float = DEFAULT_DEDUP_THRESHOLD,
) -> dict[str, Any]:
    """Run real FiftyOne Brain curation and return the enriched report.

    ``keys`` are the augment prefix's object keys; ``download_key(key, dest)``
    downloads one object and returns the local path; ``read_json(key)`` returns
    parsed JSON (or ``None``). Raises :class:`FiftyoneUnavailable` if FiftyOne /
    Pillow / numpy are not importable.
    """
    try:
        import fiftyone as fo  # type: ignore
        import fiftyone.brain as fob  # type: ignore
        import numpy as np  # noqa: F401
        from PIL import Image  # noqa: F401
    except Exception as exc:  # noqa: BLE001 - any import failure => fall back
        raise FiftyoneUnavailable(str(exc)) from exc

    from importlib import metadata as _md

    try:
        fo_version = _md.version("fiftyone")
    except Exception:  # noqa: BLE001
        fo_version = getattr(fo, "__version__", "")

    reps = _augmented_representatives(keys, augment_prefix)
    if not reps:
        raise FiftyoneUnavailable("no augmented variant frames to curate")

    import os
    import uuid

    os.makedirs(workdir, exist_ok=True)
    dataset_name = f"paidf-curate-{uuid.uuid4().hex[:8]}"
    dataset = fo.Dataset(name=dataset_name, persistent=False)
    fields: set[str] = set()
    order: list[str] = []
    embeddings: list[list[float]] = []
    warn = ""
    try:
        for clip in sorted(reps):
            info = reps[clip]
            dest = os.path.join(workdir, clip)
            os.makedirs(dest, exist_ok=True)
            local = download_key(info["frame_key"], dest)
            variables: dict[str, Any] = {}
            if info["meta_key"]:
                meta = read_json(info["meta_key"]) or {}
                if isinstance(meta, dict) and isinstance(meta.get("variables"), dict):
                    variables = meta["variables"]
            sample = fo.Sample(filepath=local, tags=["augmented"])
            sample["clip_id"] = clip
            for fk, fv in variables.items():
                if fk == "prompt":
                    continue
                sample[str(fk)] = fv
                fields.add(str(fk))
            dataset.add_sample(sample)
            order.append(clip)
            embeddings.append(_image_embedding(local))

        emb = np.asarray(embeddings, dtype=np.float64)

        uniqueness_method = "fiftyone-brain"
        fob.compute_uniqueness(dataset, embeddings=emb, uniqueness_field="uniqueness")
        uniqueness: dict[str, float] = {}
        for sample in dataset:
            raw = sample["uniqueness"]
            uniqueness[str(sample["clip_id"])] = float(raw) if _is_finite(raw) else float("nan")
        # FiftyOne's uniqueness can be NaN / degenerate for tiny sample sets. Fall
        # back to a deterministic embedding-based novelty so the report stays
        # meaningful and JSON-valid (NaN is not legal JSON).
        finite = [v for v in uniqueness.values() if _is_finite(v)]
        distinct = {round(v, 9) for v in finite}
        if len(finite) != len(uniqueness) or (len(uniqueness) > 1 and len(distinct) <= 1):
            uniqueness = _embedding_novelty(order, emb)
            uniqueness_method = "embedding-fallback"
        else:
            uniqueness = {k: round(float(v), 6) for k, v in uniqueness.items()}

        duplicate_pairs: list[tuple[str, str]] = []
        try:
            sim = fob.compute_similarity(
                dataset, embeddings=emb, backend="sklearn", brain_key="paidf_sim"
            )
            sim.find_duplicates(thresh=dedup_threshold)
            id_to_clip = {sample.id: str(sample["clip_id"]) for sample in dataset}
            neighbors = getattr(sim, "neighbors_map", None) or {}
            for rep_id, dupes in neighbors.items():
                rep_clip = id_to_clip.get(rep_id, "")
                for entry in dupes:
                    dup_id = entry[0] if isinstance(entry, (list, tuple)) else entry
                    dup_clip = id_to_clip.get(dup_id, "")
                    if rep_clip and dup_clip and rep_clip != dup_clip:
                        duplicate_pairs.append((rep_clip, dup_clip))
        except Exception as exc:  # noqa: BLE001 - similarity is best-effort
            warn = f"similarity/find_duplicates failed: {exc}"

        visualization: list[dict[str, Any]] = []
        try:
            viz = fob.compute_visualization(
                dataset, embeddings=emb, method="pca", brain_key="paidf_viz"
            )
            points = getattr(viz, "points", None)
            if points is None:
                points = getattr(viz, "current_points", None)
            if points is not None:
                pts = np.asarray(points, dtype=np.float64)
                for clip, row in zip(order, pts):
                    visualization.append({"id": clip, "point": [round(float(x), 4) for x in row[:2]]})
        except Exception as exc:  # noqa: BLE001 - visualization is best-effort
            warn = (warn + "; " if warn else "") + f"visualization failed: {exc}"

        selection = select_curated(list(reps), uniqueness, duplicate_pairs)
        return merge_curation_into_report(
            base_report,
            engine=CURATION_ENGINE_FIFTYONE,
            fiftyone_version=fo_version,
            embedding_kind="rgb16-hist8",
            dedup_threshold=dedup_threshold,
            uniqueness=uniqueness,
            selection=selection,
            visualization=visualization,
            fields=sorted(fields),
            warn=warn,
            uniqueness_method=uniqueness_method,
        )
    finally:
        try:
            dataset.delete()
        except Exception as exc:  # noqa: BLE001 - cleanup is best-effort
            _log.debug("failed to delete curation dataset: %s", exc)
