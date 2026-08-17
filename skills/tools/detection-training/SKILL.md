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

`--gpu-type` is `h100` or `l40s`. Auth defaults to `token` (the token comes from
the variable named by `--token-env`, default `DETECTION_TRAINING_TOKEN`);
`--insecure-no-auth` exists but should not be used. Private registries need
`--image-pull-secret` (default `npa-nebius-registry`). Always `--dry-run` first
and read the manifest.

## Train

```bash
npa workbench detection-training train \
  --view <materialized-view> \
  --output-uri s3://<bucket>/detection/runs/<id>/ \
  --num-classes 10 \
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
pairs (`person=0,rider=1`). Keep `--num-classes` consistent with the map.

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
search, and the checkpoint is resolved from the last completed `/runs` entry with
the trained epoch count substituted. Use it instead of hand-assembling a
checkpoint filename from an epoch count you assumed.

`--write-canonical-metrics` publishes the response under the output prefix so
downstream stages and `npa workbench insights` can pick it up. Pass the **same**
`--label-map` you trained with; a different map silently reindexes classes and
produces plausible-looking but wrong per-class numbers.

## In workflows

toolRefs are per-slice pairs rather than one generic tool:
`workbench.detection_training.train_nighttime` / `.eval_nighttime`,
`.train_distant` / `.eval_distant`, `.train_rider` / `.eval_rider`. They appear
together in `bdd100k-pipeline.yaml`, which builds the views, trains each slice,
and evaluates each against its validation view.

## Gotchas

- **Train and eval must agree on the label map and class count.** Mismatches do
  not error; they produce wrong metrics.
- **Without `--wait`, "started" is not "succeeded".** Check `status` before
  reporting a result.
- **A view is a slice, so the metric is about that slice.** Nighttime mAP is not
  overall mAP; label it accordingly when reporting.
- **`--service` needs both a reachable `--endpoint` and the token variable set.**
  A missing token presents as an auth failure from the endpoint, not as a CLI
  validation error.
- **Refresh the view after the underlying table changes.** `refresh-mv` is not
  automatic, and training a stale view silently trains on old rows.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```
