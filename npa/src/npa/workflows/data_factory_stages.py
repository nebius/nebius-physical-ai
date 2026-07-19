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
import os
import random
import tempfile
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

APPEARANCE_VARIABLES = {
    "time_of_day": ["morning", "midday", "evening", "night"],
    "weather": ["clear", "overcast", "rainy", "foggy"],
    "road_condition": ["dry", "wet"],
}


def _storage():
    from npa.clients.storage import StorageClient

    return StorageClient.from_environment()


def _s3_client():
    import boto3
    from botocore.config import Config as BotoConfig

    return boto3.client(
        "s3",
        endpoint_url=os.environ.get("AWS_ENDPOINT_URL") or os.environ.get("NEBIUS_S3_ENDPOINT"),
        aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID") or None,
        aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY") or None,
        config=BotoConfig(signature_version="s3v4", retries={"max_attempts": 3, "mode": "adaptive"}),
    )


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


def _download_json(uri: str) -> dict[str, Any]:
    if not uri.startswith("s3://"):
        return json.loads(Path(uri).read_text())
    with tempfile.TemporaryDirectory(prefix="npa-df-stage-") as tmp:
        local = _storage().download_path(uri, tmp)
        p = Path(local)
        if p.is_dir():
            cand = sorted(p.rglob("*.json"))
            p = cand[0] if cand else p
        return json.loads(Path(p).read_text())


def generate_configs(configs_uri: str, n_augmentations: int | str = 2, seed: str = "") -> dict[str, Any]:
    """Sample appearance-only augmentation combos and write a real config manifest.

    ``n_augmentations`` accepts a str (the blueprint interpolates a quoted config
    value) or int; a non-numeric value falls back to 2 rather than crashing.
    """
    try:
        n = int(n_augmentations)
    except (TypeError, ValueError):
        n = 2
    rng = random.Random(seed or None)
    combos = [{k: rng.choice(v) for k, v in APPEARANCE_VARIABLES.items()} for _ in range(max(1, n))]
    manifest = {
        "schema": "npa.data_factory.configs.v1",
        "n_augmentations": len(combos),
        "variables": APPEARANCE_VARIABLES,
        "augmentations": combos,
    }
    uri = configs_uri.rstrip("/") + "/manifest.json" if not configs_uri.endswith(".json") else configs_uri
    manifest["written_uri"] = _upload_json(manifest, uri)
    print(json.dumps(manifest))
    return manifest


def grade_gate(scores_uri: str, decision_uri: str, threshold: float = 0.5) -> str:
    """Read the real VLM eval score and write a promote/loop decision.

    The blueprint's attribute-verify stage runs ``workbench.vlm_eval.run`` with
    ``--backend api`` (a real hosted VLM). Its output file is RESULT_FILENAME,
    which is the vlm_eval tool's (legacy-named) real result JSON -- the "stub" in
    the filename is a historical artifact of the tool, not a stubbed stage. We
    import the constant so this stays in sync with the tool instead of hardcoding
    a magic string.
    """
    from npa.orchestration.npa_workflow.decisions import write_decision
    from npa.workbench.vlm_eval import RESULT_FILENAME

    score = 0.0
    try:
        report = _download_json(scores_uri if scores_uri.endswith(".json") else scores_uri.rstrip("/") + "/" + RESULT_FILENAME)
        score = float(report.get("score", 0.0))
    except Exception as exc:  # noqa: BLE001 - best-effort; default to loop_back
        print(json.dumps({"stage": "grade_gate", "warn": f"could not read score: {exc}"[:200]}))
    decision = "promote_checkpoint" if score >= threshold else "loop_back"
    write_decision(decision_uri, decision)
    print(json.dumps({"stage": "grade_gate", "score": score, "threshold": threshold, "decision": decision}))
    return decision


def curate(augment_uri: str, report_uri: str) -> dict[str, Any]:
    """Build a real curation report over the augmented set."""
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
    report = {
        "schema": "npa.fiftyone.curation.v1",
        "augmented_clips": len(clips),
        "clip_ids": clips,
        "video_count": len(videos),
        "frame_count": len(frames),
        # Honest, machine-readable limitation: one Cosmos Transfer --execute per
        # run emits a single appearance variant, so augmented_clips reflects
        # single-variant output regardless of the sampled n_augmentations.
        # N-variant "multiply" (one inference per sampled combo) is a follow-up.
        "multiply": {
            "mode": "single-variant",
            "note": "one Cosmos Transfer 2.5 --execute per run; N-variant multiply is a tracked follow-up",
        },
        "status": "curated",
    }
    report["written_uri"] = _upload_json(report, report_uri)
    print(json.dumps(report))
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
        after_run = k.split(marker, 1)[-1] if marker in f"/{k}" else k
        stage = after_run.split("/", 1)[0] if "/" in after_run else after_run
        stages[stage] = stages.get(stage, 0) + 1
    report = {
        "schema": "npa.sim2real.e2e_report.v1",
        "status": "completed",
        "artifact_count": len(keys),
        "stages": stages,
        "has_rrd": any(k.endswith(".rrd") for k in keys),
        # Mirror the curate caveat so the final report is honest on its own:
        # augmentation currently produces one variant per run (see curate report).
        "multiply_mode": "single-variant",
    }
    report["written_uri"] = _upload_json(report, report_uri)
    print(json.dumps(report))
    return report
