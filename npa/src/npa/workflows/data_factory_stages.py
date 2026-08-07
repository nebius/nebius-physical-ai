"""Real stage implementations for the Physical AI Data Factory blueprint.

These back the ``run.shell`` stages of ``physical-ai-data-factory.yaml`` so every
stage does real work against S3 instead of an ``echo`` stub:

- ``generate_configs``: sample appearance-only augmentation variables -> manifest.
- ``grade_gate``: read the real VLM eval score and write a promote/loop decision.
- ``curate``: build a real curation report over the augmented set (counts,
  per-attribute coverage, duplicate check).
- ``finalize``: aggregate the run's stage artifacts into a real final report.

All functions read/write real S3 objects (or local paths). ``npa`` is
pip-installed in the rendered task, so the blueprint invokes them inline via
``python3 -c "from npa.workflows.data_factory_stages import <fn>; <fn>(...)"``.
"""

from __future__ import annotations

import json
import random
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

# Appearance-only variables for the demo's indoor tabletop-manipulation footage
# (a robot arm folding cloth). These must match the SCENE so the sampled combo is
# coherent with the pixels AND can drive the Cosmos Transfer prompt — appearance
# only (lighting/background/materials), never the geometry or the folding motion.
APPEARANCE_VARIABLES = {
    "lighting": ["bright daylight", "warm lamp light", "dim evening light", "cool overhead light"],
    "background": ["plain wall", "cluttered shelves", "sunlit window", "hanging curtain"],
    "cloth_color": ["blue", "red", "white", "green"],
    "surface": ["beige sofa", "wooden table", "gray countertop"],
}


def prompt_from_combo(combo: dict[str, Any]) -> str:
    """Turn a sampled appearance combo into a natural-language Cosmos prompt.

    Keeps the scene/action fixed (robot folding cloth) and varies only appearance,
    so the augmentation is a faithful re-render of the SAME clip under new looks.
    """
    cloth = str(combo.get("cloth_color") or "").strip()
    surface = str(combo.get("surface") or "").strip()
    lighting = str(combo.get("lighting") or "").strip()
    background = str(combo.get("background") or "").strip()
    return (
        f"A robot arm folding a {cloth or 'blue'} cloth on a {surface or 'beige sofa'}, "
        f"{lighting or 'bright daylight'}, {background or 'plain wall'} in the background. "
        "Photorealistic, same motion and layout, appearance changed only."
    )


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _s3_client():
    # Reuse StorageClient so LIST uses the SAME validated endpoint as upload /
    # download. Building a raw boto3 client here with an unset endpoint would
    # silently fall back to real AWS S3 and make curate/finalize report 0 clips
    # instead of failing fast (StorageClient raises when the endpoint is unset).
    return _storage().s3


def _split(uri: str) -> tuple[str, str]:
    p = urlparse(uri)
    return p.netloc, p.path.lstrip("/")


def _list_keys(uri: str) -> list[str]:
    bucket, prefix = _split(uri if uri.endswith("/") else uri + "/")
    s3 = _s3_client()
    keys: list[str] = []
    token = None
    while True:
        kw = {"Bucket": bucket, "Prefix": prefix}
        if token:
            kw["ContinuationToken"] = token
        page = s3.list_objects_v2(**kw)
        keys.extend(o["Key"] for o in page.get("Contents", []) if o.get("Key"))
        if not page.get("IsTruncated"):
            break
        token = page.get("NextContinuationToken")
    return keys


def _upload_json(payload: dict[str, Any], uri: str) -> str:
    if uri.startswith("s3://"):
        with tempfile.TemporaryDirectory(prefix="npa-df-stage-") as tmp:
            p = Path(tmp) / "out.json"
            p.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            return _storage().upload_file(str(p), uri)
    Path(uri).parent.mkdir(parents=True, exist_ok=True)
    Path(uri).write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return uri


def _download_key(bucket: str, key: str, dest: str) -> str:
    """Download a single S3 object (bucket/key) into ``dest`` and return the path."""
    return _storage().download_path(f"s3://{bucket}/{key}", dest)


def _read_json_key(bucket: str, key: str) -> dict[str, Any] | None:
    if not key:
        return None
    try:
        return _download_json(f"s3://{bucket}/{key}")
    except Exception:  # noqa: BLE001 - best-effort metadata read
        return None


def _download_json(uri: str) -> dict[str, Any]:
    if not uri.startswith("s3://"):
        return json.loads(Path(uri).read_text())
    want = uri.rstrip("/").split("/")[-1]
    with tempfile.TemporaryDirectory(prefix="npa-df-stage-") as tmp:
        local = _storage().download_path(uri, tmp)
        p = Path(local)
        if p.is_dir():
            # download_path fell back to the prefix (the exact object is missing).
            # Prefer the exact requested filename; NEVER silently substitute a
            # different JSON (e.g. decision.json instead of vlm_eval_stub.json),
            # which would mask a missing eval result as a bogus score of 0.
            if want.endswith(".json"):
                exact = [c for c in p.rglob(want)]
                if not exact:
                    raise FileNotFoundError(f"{want} not found under {uri}")
                p = exact[0]
            else:
                cand = sorted(p.rglob("*.json"))
                p = cand[0] if cand else p
        return json.loads(Path(p).read_text())


def generate_configs(
    configs_uri: str,
    n_augmentations: int | str = 2,
    seed: str = "",
    augment_subject: str = "",
) -> dict[str, Any]:
    """Sample appearance-only augmentation combos and write a real config manifest.

    ``n_augmentations`` accepts a str (the blueprint interpolates a quoted config
    value) or int; a non-numeric value falls back to 2 rather than crashing.
    """
    try:
        n = int(n_augmentations)
    except (TypeError, ValueError):
        n = 2
    subject = str(augment_subject or "").strip() or "the input robot clip"
    rng = random.Random(seed or None)
    combos = []
    for _ in range(max(1, n)):
        combo = {k: rng.choice(v) for k, v in APPEARANCE_VARIABLES.items()}
        # The prompt is what actually conditions the Cosmos Transfer augmentation,
        # so the sampled appearance drives the pixels (not just a Rerun label).
        combo["prompt"] = f"{prompt_from_combo(combo)} Subject: {subject}."
        combos.append(combo)
    manifest = {
        "schema": "npa.data_factory.configs.v1",
        "scene": subject,
        "n_augmentations": len(combos),
        "variables": APPEARANCE_VARIABLES,
        "augmentations": combos,
    }
    uri = configs_uri.rstrip("/") + "/manifest.json" if not configs_uri.endswith(".json") else configs_uri
    manifest["written_uri"] = _upload_json(manifest, uri)
    print(json.dumps(manifest))
    return manifest


def grade_gate(scores_uri: str, decision_uri: str, threshold: float | str = 0.5) -> str:
    """Read the evaluator score and write a promote/loop decision.

    The blueprint's evaluate stage runs the real NVIDIA Cosmos Evaluator, which
    writes ``cosmos_evaluator.json``. Runs produced before that stage existed (and
    any spec still pointing the loop at ``workbench.vlm_eval.run``) wrote the
    vlm_eval tool's RESULT_FILENAME instead, so both are accepted, newest contract
    first. Both filenames come from the producing tool's own constant rather than a
    literal here, so the gate cannot drift from its producer.

    ``threshold`` accepts a str (the blueprint interpolates a quoted config value)
    or float; a non-numeric value falls back to 0.5.

    Best-effort by design: an unreadable, malformed, or self-declared-degraded report
    yields ``loop_back`` rather than an exception, because a gate that raises takes
    the whole refinement loop down with it.
    """
    from npa.orchestration.npa_workflow.decisions import write_decision
    from npa.workbench.cosmos_evaluator import RESULT_FILENAME as COSMOS_EVALUATOR_RESULT
    from npa.workbench.vlm_eval import RESULT_FILENAME as VLM_EVAL_RESULT

    try:
        threshold = float(threshold)
    except (TypeError, ValueError):
        threshold = 0.5
    if scores_uri.endswith(".json"):
        candidates = [scores_uri]
    else:
        base = scores_uri.rstrip("/")
        candidates = [f"{base}/{COSMOS_EVALUATOR_RESULT}", f"{base}/{VLM_EVAL_RESULT}"]
    score = 0.0
    status = "completed"
    source = ""
    problems: list[str] = []
    for candidate in candidates:
        try:
            report = _download_json(candidate)
            if not isinstance(report, dict):
                raise TypeError(f"expected a JSON object, got {type(report).__name__}")
            # Parsed inside the try on purpose: a report that downloads cleanly but
            # carries a non-numeric score has to degrade to loop_back exactly like an
            # unreadable one. Letting it raise would abort the whole refinement loop
            # over a malformed field, which is the opposite of a gate's job.
            candidate_score = float(report.get("score", 0.0))
            # A report the producer itself marked degraded (an evaluator that lost
            # object storage part-way, say) describes the run's infrastructure, not
            # its variants. Promoting on it would ship an ungraded batch.
            candidate_status = str(report.get("status", "completed"))
        except Exception as exc:  # noqa: BLE001 - fall through to the older contract
            problems.append(f"{candidate.rsplit('/', 1)[-1]}: {exc}"[:150])
            continue
        score = candidate_score
        status = candidate_status
        source = candidate
        break
    if not source:
        print(json.dumps({"stage": "grade_gate", "warn": f"could not read a score ({'; '.join(problems)})"[:300]}))
    graded = status == "completed"
    decision = "promote_checkpoint" if graded and score >= threshold else "loop_back"
    write_decision(decision_uri, decision)
    print(
        json.dumps(
            {
                "stage": "grade_gate",
                "score": score,
                "threshold": threshold,
                "decision": decision,
                "source": source,
                "report_status": status,
            }
        )
    )
    return decision


def curate(
    augment_uri: str,
    report_uri: str,
    dedup_threshold: float | str = "",
    curator_report_uri: str = "",
) -> dict[str, Any]:
    """Curate the augmented set and write a real curation report.

    When FiftyOne is importable (i.e. this stage runs inside the ``npa-fiftyone``
    workbench image) this runs *real* FiftyOne Brain curation over the augmented
    scenario variants -- ``compute_uniqueness`` + near-duplicate similarity +
    a 2D visualization -- and records which variants were kept vs dropped. When
    FiftyOne is absent (unit tests, the dev-VM worktree python) it degrades to the
    report-only counts path so the pipeline never regresses.

    ``curator_report_uri`` points at the preceding Cosmos Curator stage's summary.
    When present it is folded into this report under ``cosmos_curator``, so one
    document carries both the curator's clip catalog and the review decisions.
    """
    keys = _list_keys(augment_uri)
    videos = [k for k in keys if k.endswith(".mp4")]
    frames = [k for k in keys if k.endswith(".png")]
    # Clip ids are the per-clip subdirectories under the augment prefix itself
    # (entries that have a further path segment); top-level files like
    # manifest.json are excluded. Deriving relative to the passed augment_uri
    # (rather than a hardcoded "/cosmos_augmented/") keeps this correct for any
    # prefix, including a bucket root. Matches publish_transfer_to_s3's layout.
    _, aug_prefix = _split(augment_uri if augment_uri.endswith("/") else augment_uri + "/")
    rels = [k[len(aug_prefix):] for k in keys if k.startswith(aug_prefix)]
    clips = sorted({r.split("/", 1)[0] for r in rels if "/" in r and r.split("/", 1)[0]})
    multi = len(clips) > 1
    report = {
        "schema": "npa.fiftyone.curation.v1",
        "augmented_clips": len(clips),
        "clip_ids": clips,
        "video_count": len(videos),
        "frame_count": len(frames),
        # Machine-readable "multiply" status: the augment stage runs one Cosmos
        # Transfer 2.5 inference per sampled appearance combo, so augmented_clips
        # reflects how many scenario variants the run actually produced.
        "multiply": {
            "mode": "multi-variant" if multi else "single-variant",
            "variant_count": len(clips),
            "note": (
                f"{len(clips)} Cosmos Transfer 2.5 scenario variants (one inference per sampled combo)"
                if multi
                else "one Cosmos Transfer 2.5 scenario variant for this run"
            ),
        },
        "status": "curated",
    }

    report = _enrich_with_fiftyone_curation(report, augment_uri, keys, dedup_threshold)
    report = _merge_curator_report(report, curator_report_uri)

    report["written_uri"] = _upload_json(report, report_uri)
    print(json.dumps(report))
    return report


def _merge_curator_report(report: dict[str, Any], curator_report_uri: str) -> dict[str, Any]:
    """Fold the Cosmos Curator stage's summary into the curation report.

    Only the run-level fields are copied; the per-clip catalog stays in the
    curator's own report so this document does not grow with the clip count.
    """
    if not curator_report_uri:
        return report
    try:
        curator = _download_json(curator_report_uri)
    except Exception as exc:  # noqa: BLE001 - the review report stands on its own
        report["cosmos_curator"] = {"status": "unavailable", "warn": f"{exc}"[:200]}
        return report
    if not isinstance(curator, dict):
        report["cosmos_curator"] = {"status": "unavailable", "warn": "curator report is not an object"}
        return report
    report["cosmos_curator"] = {
        "status": str(curator.get("status") or ""),
        "engine": str(curator.get("engine") or ""),
        "curated_uri": str(curator.get("curated_uri") or ""),
        "clip_count": int(curator.get("clip_count") or 0),
        "filtered_count": int(curator.get("filtered_count") or 0),
        "variant_count": int(curator.get("variant_count") or 0),
        "total_duration_s": float(curator.get("total_duration_s") or 0.0),
        "motion_filter": str(curator.get("motion_filter") or ""),
        "report_uri": curator_report_uri,
    }
    return report


def _enrich_with_fiftyone_curation(
    report: dict[str, Any],
    augment_uri: str,
    keys: list[str],
    dedup_threshold: float | str,
) -> dict[str, Any]:
    """Run real FiftyOne Brain curation over the augmented set when available.

    Falls back to the report-only counts path (tagging ``curation_engine``) when
    FiftyOne is not importable or curation fails, so the stage always produces a
    valid report.
    """
    from npa.workflows import data_factory_curate as dfc

    try:
        thresh = float(dedup_threshold)
    except (TypeError, ValueError):
        thresh = dfc.DEFAULT_DEDUP_THRESHOLD

    bucket, aug_prefix = _split(augment_uri if augment_uri.endswith("/") else augment_uri + "/")
    try:
        with tempfile.TemporaryDirectory(prefix="npa-df-curate-") as tmp:
            return dfc.run_curation(
                keys=keys,
                augment_prefix=aug_prefix,
                base_report=report,
                download_key=lambda key, dest: _download_key(bucket, key, dest),
                read_json=lambda key: _read_json_key(bucket, key),
                workdir=tmp,
                dedup_threshold=thresh,
            )
    except dfc.FiftyoneUnavailable:
        report["curation_engine"] = dfc.CURATION_ENGINE_REPORT_ONLY
        return report
    except Exception as exc:  # noqa: BLE001 - never fail the stage on curation errors
        report["curation_engine"] = dfc.CURATION_ENGINE_REPORT_ONLY
        report["curation_warn"] = f"fiftyone curation failed: {exc}"[:300]
        return report


def finalize(run_root_uri: str, report_uri: str) -> dict[str, Any]:
    """Aggregate the run's stage artifacts into a real final report."""
    keys = _list_keys(run_root_uri)
    run_seg = run_root_uri.rstrip("/").split("/")[-1]
    marker = f"/{run_seg}/"
    stages: dict[str, int] = {}
    for k in keys:
        # Take the path *after* the run-id segment, then its first segment is the
        # stage. We prepend "/" to k so the run-id also matches when it is the
        # leading segment of the (bucket-relative) key; if the run id is absent
        # we fall back to the whole key.
        prefixed = f"/{k}"
        after_run = prefixed.split(marker, 1)[-1] if marker in prefixed else k
        stage = after_run.split("/", 1)[0] if "/" in after_run else after_run
        stages[stage] = stages.get(stage, 0) + 1
    # Count augmented scenario variants (per-clip subdirs under cosmos_augmented/,
    # excluding the top-level run manifest) so the final report reflects the real
    # "multiply" fan-out — one Cosmos Transfer 2.5 inference per sampled combo.
    aug_marker = "cosmos_augmented/"
    aug_clips: set[str] = set()
    for k in keys:
        if aug_marker in k:
            rest = k.split(aug_marker, 1)[1]
            seg = rest.split("/", 1)[0] if "/" in rest else ""
            if seg:
                aug_clips.add(seg)
    n_variants = len(aug_clips)
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "status": "completed",
        "artifact_count": len(keys),
        "stages": stages,
        "has_rrd": any(k.endswith(".rrd") for k in keys),
        # Mirror the curate report so the final report is honest on its own: how
        # many Cosmos Transfer 2.5 scenario variants this run produced.
        "multiply_mode": "multi-variant" if n_variants > 1 else "single-variant",
        "variant_count": n_variants,
    }
    report["written_uri"] = _upload_json(report, report_uri)
    print(json.dumps(report))
    return report
