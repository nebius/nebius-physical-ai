---
name: detection-training
description: Use to train and evaluate Faster R-CNN detectors from LanceDB materialized views (BDD100K failure-mode slices) — direct versus deployed-service execution, the mandatory label map for string categories, and checkpoint discovery at eval time.
---

# Detection training (LanceDB view → Faster R-CNN)

This tool trains an object detector on a **slice** of a dataset rather than the
whole thing. The input is a LanceDB materialized view, which is what makes the
failure-mode workflow possible: curate a view of nighttime frames, or distant
objects, or riders, then train and evaluate only on that slice.

Upstream is `skills/tools/lancedb/SKILL.md` — create the view with
`npa workbench lancedb create-mv` / `refresh-mv` before anything here. The end-to-end
pipeline is `npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml`, with a
walkthrough in `docs/workbench/cookbooks/bdd100k-pipeline.md`.

## Two execution modes

Direct (the default) runs the work from your CLI invocation. Service mode calls a
deployed Kubernetes endpoint:

```bash
npa workbench detection-training train --view <mv-name> --output-uri s3://<bucket>/runs/<id>/
npa workbench detection-training train --view <mv-name> --service --endpoint <url>
```

Deploy the service only when you want a persistent endpoint several runs share:

```bash
npa workbench detection-training deploy \
  --project <alias> --cluster-name <name> \
  --input-path s3://<bucket>/lancedb/<db>/ \
  --output-path s3://<bucket>/detection/ \
  --gpu-type h100 --namespace default \
  --dry-run                       # prints the manifest without applying
npa workbench detection-training deploy --project <alias> --destroy
```

`--gpu-type` accepts `h100`, `b200`, `l40s`, and `rtxpro6000`; verify actual cluster labels and reservation placement before deploying. Auth defaults to `token` (the token comes from
the variable named by `--token-env`, default `DETECTION_TRAINING_TOKEN`);
`--insecure-no-auth` exists but should not be used. Official NPA GHCR images
pull anonymously, so `--image-pull-secret` defaults to empty. For an optional
operator-controlled private registry, pass the name of an existing standard
Kubernetes pull secret explicitly. Always `--dry-run` first and read the
manifest.

## Train

```bash
npa workbench detection-training train \
  --view <materialized-view> \
  --output-uri s3://<bucket>/detection/runs/<id>/ \
  --label-map '{"person":0,"rider":1,"car":2}' \
  --epochs 10 --batch-size 8 --learning-rate 0.005 \
  --wait --poll-seconds 30 --timeout-seconds 21600 \
  --output json
```

`--view` is the only required flag. `--data-path` (alias `--lance-uri` /
`--input-path`) overrides which LanceDB the view is read from. `--override
KEY=VALUE` sets request fields the flags do not expose. Weights and Biases
logging is opt-in and defaults to `offline` mode.

`--wait` polls `/status` until the run completes **and fails if it does not** —
without it, the command returns as soon as the run is accepted, which is easy to
misread as success. Default output format for `train` and `eval` is `json`.

## The label map is not optional for BDD100K

**`--label-map` is required whenever the dataset stores string categories, which
BDD100K does.** Without it, category names reach `int()` and the run dies on:

```
invalid literal for int() with base 10: 'train'
```

`train` is a real BDD100K category — the vehicle. Reading that traceback as a
bug in the training code, rather than a missing label map, costs more time than
it should. Accepted formats are JSON (`{"person":0,...}`) or comma-separated
pairs (`person=0,rider=1`). Class count is inferred from the map by default.
Source maps containing zero shift all IDs by one inside the detector because
Faster R-CNN reserves zero for background. Checkpoints retain both maps; string
and numeric annotations from materialized views keep the same category identity.
Explicit `--num-classes` includes background and must cover every mapped ID.

## Evaluate

```bash
npa workbench detection-training eval \
  --checkpoint-uri s3://<bucket>/detection/runs/<id>/ \
  --eval-view <validation-mv> \
  --output-uri s3://<bucket>/detection/eval/<id>/ \
  --label-map '{"person":0,"rider":1,"car":2}' \
  --discover-checkpoint \
  --write-canonical-metrics
```

`--eval-view` and `--output-uri` are required. `--discover-checkpoint` changes
what `--checkpoint-uri` means: it becomes the training **output prefix** to
search, and the checkpoint is selected from the last completed `/runs` entry
using its verified final checkpoint artifact. Legacy records without artifact
metadata retain the epoch-pattern fallback. Use it instead of hand-assembling a
checkpoint filename from an epoch count you assumed.

`--write-canonical-metrics` publishes the response under the output prefix so
downstream stages and `npa workbench insights` can pick it up. Evaluation reads
the checkpoint map automatically. An explicit map must match its category
identity; mismatches fail before model execution. Legacy checkpoints preserve
the original IDs. Missing metric dependencies fail instead of returning fake
zero scores; an undefined COCO metric remains `-1`.

## In workflows

toolRefs are per-slice pairs rather than one generic tool:
`workbench.detection_training.train_nighttime` / `.eval_nighttime`,
`.train_distant` / `.eval_distant`, `.train_rider` / `.eval_rider`. They appear
together in `bdd100k-pipeline.yaml`, which builds the views, trains each slice,
and evaluates each against its validation view.

## Gotchas

- **Train and eval must agree on category identity.** New checkpoints record it
  and evaluation rejects a conflicting explicit map.
- **Without `--wait`, "started" is not "succeeded".** Check `status` before
  reporting a result.
- **A view is a slice, so the metric is about that slice.** Nighttime mAP is not
  overall mAP; label it accordingly when reporting.
- **`--service` needs both a reachable `--endpoint` and the token variable set.**
  A missing token presents as an auth failure from the endpoint, not as a CLI
  validation error.
- **Refresh the view after the underlying table changes.** `refresh-mv` is not
  automatic, and training a stale view silently trains on old rows.

## Persistent service records and readiness

The default authenticated deployment probes `/readyz`, which returns only a
minimal readiness result. `/health`, training/evaluation, status and artifacts
remain protected. Tokens and storage credentials are provisioned separately
through private files/stdin; workload manifests contain Secret references.
Before any Kubernetes mutation, deploy binds the selected project to its exact
saved context, verifies the selected GPU product and node selector, and probes
the exact output directory with the same atomic credential pair injected into
the service. Ambiguous/foreign contexts, mixed credential pairs, denied output
writes, and unknown or mismatched GPU evidence fail before apply. GPU capacity
checks count free devices; busy allocations count toward a `Recreate` replacement
only with the exact pod → ReplicaSet → Deployment controller UID chain. Labels
do not prove ownership, and unbound GPU demand makes availability unknown. Dry-run only
renders; it does not establish execution readiness.

A retained PVC holds SQLite run records; `--state-pvc` reuses an existing claim.
One worker owns the volume under a process lock, while transactions serialize
concurrent updates. Restart preserves terminal results and marks unfinished work
`interrupted` with its original identity and progress. It never resumes training
automatically. `deploy --destroy` preserves the state claim.

`/status` and `/artifacts` expose exact artifact roles, locations, media/schema
metadata, sizes and SHA-256 hashes of the successfully written bytes. Authenticated
`/artifacts/content?run_id=...&sha256=...` checks current bytes before returning
them. Service writes and retrieval stay under its configured output prefix.
Direct synchronous CLI/SDK runs return artifact manifests; service runs own
persistent status records. Evaluation results and metric artifacts are also
durable under `eval_run_id`; identical completed requests reuse that result.
Active, failed, or interrupted evaluation identities are not silently rerun.

## Verify

```bash
# The checkpoint and Lance dataloader regressions require real CPU PyTorch.
# CI installs this runtime explicitly; it is not part of the lightweight dev extra.
uv pip install --python npa/.venv/bin/python --index-url https://download.pytorch.org/whl/cpu torch==2.12.1
# PyTorch also activates the full suite's SONIC ONNX export tests.
uv pip install --python npa/.venv/bin/python -e "npa[sonic]"
npa/.venv/bin/python -m pytest npa/tests/workbench/test_detection_runtime_contract.py npa/tests/workbench/test_detection_training.py -q
# Authorized live deployment and private config with prepared LanceDB views:
NPA_DETECTION_RUNTIME_LIVE=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_detection_runtime_live.py -q
```
