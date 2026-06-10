#!/usr/bin/env python3
"""Generate the BDD100K SkyPilot pipeline from a compact spec.

Prototype for the "thin YAML" direction:
- Vanilla base image (no custom workflow image); deps installed in `setup`.
- Train/eval are single parameterized templates looped over the failure-mode
  views (no copy-pasted x3 blocks); GPU lives in the in-cluster services, so the
  SkyPilot task pods are CPU-only.
- Two runners:
    --runner cli   tasks call the `npa` CLI (the target; the CLI is the tested
                   step library). Train/eval use --label-map / --wait /
                   --from-view-latest / --write-canonical-metrics (the small
                   additive CLI flags noted in the migration plan).
    --runner curl  behaviour-equivalent to the committed YAML (curl+jq); used to
                   validate the generated DAG against the mock-endpoint harness.

Env-var names match npa/scripts/run_bdd100k_pipeline.py so the same runner can
render + submit (or mock-validate) the generated file.

Usage:
    npa/.venv/bin/python npa/scripts/generate_bdd100k_pipeline.py \
        --runner curl --out /tmp/bdd100k-pipeline.generated.yaml
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VANILLA_IMAGE = "docker:python:3.11-slim"
LANCEDB_ENDPOINT = "http://npa-lancedb.workbench.svc.cluster.local:8686"
DETECTION_ENDPOINT = "http://npa-detection-training.workbench.svc.cluster.local:8790"
CPU_UDFS = ["has_person", "has_rider", "person_bbox_area_pct", "dhash", "is_duplicate"]
SYNTHETIC_LABEL_MAP = {
    "person": 0, "rider": 1, "car": 2, "truck": 3, "bus": 4,
    "train": 5, "motor": 6, "bike": 7, "traffic light": 8, "traffic sign": 9,
}
# (view slug, SQL filter) — one source of truth for the three failure modes.
VIEWS = [
    ("bdd100k_rider_train", "rider", "has_rider = true AND split = 'train'"),
    ("bdd100k_nighttime_person_train", "nighttime", "timeofday = 'night' AND has_person = true AND split = 'train'"),
    ("bdd100k_distant_person_train", "distant", "has_person = true AND person_bbox_area_pct < 0.01 AND split = 'train'"),
]

# Shared bash helpers for the curl runner — defined ONCE, prepended per task.
CURL_HELPERS = """set -euo pipefail
post() {  # post BASE PATH JSON TOKEN
  code=$(curl -sS -o /tmp/r.json -w '%{http_code}' -X POST "$1$2" -H 'Content-Type: application/json' ${4:+-H "Authorization: Bearer $4"} --data "$3")
  { [ "$code" -ge 200 ] && [ "$code" -lt 300 ]; } || { echo "POST $2 -> $code" >&2; cat /tmp/r.json >&2; exit 1; }
  cat /tmp/r.json
}
get() {  # get BASE PATH TOKEN
  code=$(curl -sS -o /tmp/r.json -w '%{http_code}' "$1$2" ${3:+-H "Authorization: Bearer $3"})
  { [ "$code" -ge 200 ] && [ "$code" -lt 300 ]; } || { echo "GET $2 -> $code" >&2; cat /tmp/r.json >&2; exit 1; }
  cat /tmp/r.json
}
"""

SETUP_CURL = ("set -e\n"
              "command -v curl >/dev/null && command -v jq >/dev/null || "
              "{ apt-get update && apt-get install -y --no-install-recommends ca-certificates curl jq; }")
SETUP_CLI = "set -e\npip install -q \"npa==${NPA_VERSION:-0.1.0}\""


@dataclass
class Task:
    name: str
    run: str
    cpus: int = 4
    memory: int = 16
    envs: dict = field(default_factory=dict)


def _doc(task: Task, *, runner: str) -> dict:
    return {
        "name": task.name,
        "resources": {"cloud": "kubernetes", "cpus": task.cpus, "memory": task.memory, "image_id": VANILLA_IMAGE},
        "envs": task.envs,
        "setup": SETUP_CLI if runner == "cli" else SETUP_CURL,
        "run": task.run,
    }


def _ingest_run(runner: str) -> str:
    if runner == "cli":
        return ('npa workbench lancedb import-bdd100k --service --endpoint "$LANCEDB_ENDPOINT" '
                '--table "$LANCE_TABLE" --lance-uri "$LANCE_URI" '
                '--synthetic "$BDD100K_SYNTHETIC_ROWS" --split train --split val')
    return CURL_HELPERS + (
        'payload=$(jq -n --arg t "$LANCE_TABLE" --arg u "$LANCE_URI" --argjson s "$BDD100K_SYNTHETIC_ROWS" '
        "'{table:$t,lance_uri:$u,synthetic:$s,splits:[\"train\",\"val\"]}')\n"
        'post "$LANCEDB_ENDPOINT" /import-bdd100k "$payload" "$LANCEDB_TOKEN" | jq -e ".total_rows >= 0"')


def _backfill_run(udfs: list[str], runner: str) -> str:
    if runner == "cli":
        return "\n".join('npa workbench lancedb backfill --service --endpoint "$LANCEDB_ENDPOINT" '
                         f'--table "$LANCE_TABLE" --lance-uri "$LANCE_URI" --udf {udf}' for udf in udfs)
    return CURL_HELPERS + (
        f'for udf in {" ".join(udfs)}; do\n'
        '  payload=$(jq -n --arg t "$LANCE_TABLE" --arg u "$LANCE_URI" --arg f "$udf" '
        "'{table:$t,lance_uri:$u,udf:$f,batch_size:512}')\n"
        '  post "$LANCEDB_ENDPOINT" /backfill "$payload" "$LANCEDB_TOKEN" | jq -e ".udf == \\"$udf\\""\n'
        'done')


def _create_mvs_run(runner: str) -> str:
    if runner == "cli":
        return "\n".join('npa workbench lancedb create-mv --service --endpoint "$LANCEDB_ENDPOINT" '
                         f'--name {name} --source-table "$LANCE_TABLE" --lance-uri "$LANCE_URI" --filter-sql {json.dumps(sql)}'
                         for name, _short, sql in VIEWS)
    lines = [CURL_HELPERS.rstrip("\n")]
    for name, _short, sql in VIEWS:
        lines.append(
            f'payload=$(jq -n --arg n {json.dumps(name)} --arg s "$LANCE_TABLE" --arg f {json.dumps(sql)} '
            "--arg u \"$LANCE_URI\" '{name:$n,source_table:$s,filter_sql:$f,lance_uri:$u}')")
        lines.append(f'post "$LANCEDB_ENDPOINT" /create-mv "$payload" "$LANCEDB_TOKEN" | jq -e \'.view_name == "{name}"\'')
    return "\n".join(lines)


def _train_run(runner: str) -> str:
    if runner == "cli":
        return ('npa workbench detection-training train --service --endpoint "$DETECTION_TRAINING_ENDPOINT" '
                '--view "$VIEW_NAME" --lance-uri "$LANCE_URI" --output-uri "$TRAIN_OUTPUT_URI" '
                '--label-map "$BDD100K_LABEL_MAP" --epochs "$TRAIN_EPOCHS" '
                '--wait --poll-seconds "$TRAIN_POLL_SECONDS" --timeout "$TRAIN_TIMEOUT_SECONDS"')
    return CURL_HELPERS + (
        'payload=$(jq -n --arg v "$VIEW_NAME" --arg u "$LANCE_URI" --arg o "$TRAIN_OUTPUT_URI" '
        '--argjson m "$BDD100K_LABEL_MAP" --argjson e "$TRAIN_EPOCHS" '
        "'{view:$v,lance_uri:$u,output_uri:$o,label_map:$m,epochs:$e}')\n"
        'run_id=$(post "$DETECTION_TRAINING_ENDPOINT" /train "$payload" "$DETECTION_TRAINING_TOKEN" | jq -r .run_id)\n'
        'while :; do\n'
        '  st=$(get "$DETECTION_TRAINING_ENDPOINT" "/status?run_id=$run_id" "$DETECTION_TRAINING_TOKEN" | jq -r .status)\n'
        '  [ "$st" = completed ] && break\n'
        '  [ "$st" = failed ] && { echo "training failed" >&2; exit 1; }\n'
        '  sleep "$TRAIN_POLL_SECONDS"\n'
        'done')


def _eval_run(runner: str) -> str:
    if runner == "cli":
        return ('CANON=$([ "${WRITE_CANONICAL_EVAL_METRICS:-1}" = 1 ] && echo --write-canonical-metrics)\n'
                'npa workbench detection-training eval --service --endpoint "$DETECTION_TRAINING_ENDPOINT" '
                '--from-view-latest "$VIEW_NAME" --eval-view "$VIEW_NAME" --lance-uri "$LANCE_URI" '
                '--output-uri "$EVAL_OUTPUT_URI" $CANON')
    return CURL_HELPERS + (
        'runs=$(get "$DETECTION_TRAINING_ENDPOINT" /runs "$DETECTION_TRAINING_TOKEN")\n'
        'pat=$(echo "$runs" | jq -r --arg s "/training/$VIEW_NAME/" '
        "'[.runs[]|select(.status==\"completed\" and (.checkpoint_uri_pattern|contains($s)))]|last|.checkpoint_uri_pattern // \"\"')\n"
        'ep=$(echo "$runs" | jq -r --arg s "/training/$VIEW_NAME/" '
        "'[.runs[]|select(.status==\"completed\" and (.checkpoint_uri_pattern|contains($s)))]|last|.total_epochs // \"\"')\n"
        '{ [ -n "$pat" ] && [ -n "$ep" ]; } || { echo "no completed run for $VIEW_NAME" >&2; exit 1; }\n'
        'ckpt=$(printf "%s" "$pat" | sed "s/{epoch}/$ep/g")\n'
        'payload=$(jq -n --arg c "$ckpt" --arg v "$VIEW_NAME" --arg u "$LANCE_URI" --arg o "$EVAL_OUTPUT_URI" '
        "'{checkpoint_uri:$c,eval_view:$v,lance_uri:$u,output_uri:$o}')\n"
        'post "$DETECTION_TRAINING_ENDPOINT" /eval "$payload" "$DETECTION_TRAINING_TOKEN" | jq -e "(.mAP|type==\\"number\\")"')


def build(runner: str) -> list[dict]:
    base = {
        "NPA_PIPELINE_RUN_ID": "<your-run-id>",
        "LANCE_URI": "s3://${NPA_S3_BUCKET}/bdd100k-pipeline/${NPA_PIPELINE_RUN_ID}/lancedb/",
        "LANCE_TABLE": "bdd100k",
        "LANCEDB_ENDPOINT": LANCEDB_ENDPOINT,
        "LANCEDB_TOKEN": "",
    }
    det = {"DETECTION_TRAINING_ENDPOINT": DETECTION_ENDPOINT, "DETECTION_TRAINING_TOKEN": ""}
    label_map = json.dumps(SYNTHETIC_LABEL_MAP, separators=(",", ":"))

    tasks = [
        Task("bdd100k-ingest", _ingest_run(runner),
             envs={**base, "BDD100K_SOURCE_URI": "s3://${NPA_S3_BUCKET}/raw-bdd100k/subset-demo/",
                   "BDD100K_LIMIT": "10000", "BDD100K_SYNTHETIC_ROWS": "0"}),
        Task("bdd100k-backfill-cpu", _backfill_run(CPU_UDFS, runner), envs=dict(base)),
        Task("bdd100k-backfill-clip", _backfill_run(["clip_embedding"], runner), envs=dict(base)),
        Task("bdd100k-create-mvs", _create_mvs_run(runner), envs=dict(base)),
    ]
    for slug, short, _sql in VIEWS:
        tasks.append(Task(f"bdd100k-train-{short}", _train_run(runner),
                          envs={**base, **det, "VIEW_NAME": slug, "VIEW_SLUG": slug,
                                "TRAIN_OUTPUT_URI": f"s3://${{NPA_S3_BUCKET}}/bdd100k-pipeline/${{NPA_PIPELINE_RUN_ID}}/training/{slug}",
                                "BDD100K_LABEL_MAP": label_map, "TRAIN_EPOCHS": "10",
                                "TRAIN_POLL_SECONDS": "30", "TRAIN_TIMEOUT_SECONDS": "21600"}))
    for slug, short, _sql in VIEWS:
        tasks.append(Task(f"bdd100k-eval-{short}", _eval_run(runner),
                          envs={**base, **det, "VIEW_NAME": slug, "VIEW_SLUG": slug,
                                "EVAL_OUTPUT_URI": f"s3://${{NPA_S3_BUCKET}}/bdd100k-pipeline/${{NPA_PIPELINE_RUN_ID}}/eval/{slug}"}))

    return [{"name": "bdd100k-pipeline", "execution": "serial"}] + [_doc(t, runner=runner) for t in tasks]


HEADER = (
    "# GENERATED FILE - do not edit by hand.\n"
    "# Source: npa/scripts/generate_bdd100k_pipeline.py  (runner={runner})\n"
    "# Regenerate: npa/.venv/bin/python npa/scripts/generate_bdd100k_pipeline.py "
    "--runner {runner} --out npa/workflows/workbench/skypilot/bdd100k-pipeline.generated.yaml\n"
)


def render(runner: str) -> str:
    """Return the full generated YAML text (header + documents)."""
    return HEADER.format(runner=runner) + yaml.safe_dump_all(build(runner), sort_keys=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--runner", choices=["cli", "curl"], default="cli")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.out.write_text(render(args.runner), encoding="utf-8")
    print(f"wrote {args.out} (runner={args.runner})")


if __name__ == "__main__":
    main()
