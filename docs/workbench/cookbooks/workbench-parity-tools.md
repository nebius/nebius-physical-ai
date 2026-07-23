# Cookbook: scenario-gen and dataset-of-record live smokes

Two CPU-only smoke workflows that exercise the `scenario_gen` and `dataset`
workbench tools end-to-end against real S3, with no GPU and no LanceDB/FiftyOne
dependency. Use them to validate the tools on a configured environment before
wiring the GPU-backed production pipelines
(`adversarial-scenario-hardening.yaml`, `dataset-ingest-curate.yaml`).

Both are ordinary `npa.workflow/v0.0.1` specs, so every stage is a catalog
`toolRef` and the whole thing runs through `npa workbench workflow`.

- `npa/workflows/workbench/npa-workflows/scenario-gen-smoke.yaml`
- `npa/workflows/workbench/npa-workflows/dataset-of-record-smoke.yaml`

## What each toolRef executes

Each `toolRef` is an `npa` CLI invocation run as a subprocess by the workflow
interpreter. In `--execute` mode the CLI runs the tool's Python in-process
(scenario-gen uses its deterministic default adversary backend — no GPU, no
container); the FastAPI service/container is only involved in `--service` mode.
Artifacts pass over S3.

## Prerequisites

- `npa` installed from this branch (so `npa workbench scenario-gen` and
  `npa workbench dataset` exist): `npa workbench scenario-gen --help`.
- S3-compatible credentials available to the CLI (loaded from
  `~/.npa/credentials.yaml`; the workbench group exports them to the
  environment). Confirm with `aws s3 ls s3://<your-bucket>/`.
- A writable bucket. The committed specs use `example-bucket`; substitute your
  bucket at run time (the specs stay hygienic).

## 1) scenario-gen smoke

`generate` (default adversary backend) -> `rank`. No fixture required — the
default backend synthesizes deterministic adversarial scenarios.

```bash
BUCKET=<your-bucket>
RUN_ID="scenario-gen-smoke-$(date +%Y%m%d%H%M%S)"
SPEC=/tmp/${RUN_ID}.yaml
sed "s/example-bucket/${BUCKET}/g" \
  npa/workflows/workbench/npa-workflows/scenario-gen-smoke.yaml > "${SPEC}"

npa workbench workflow validate-spec "${SPEC}" --json
npa workbench workflow run-spec "${SPEC}" --run-id "${RUN_ID}" --execute --json
```

Expected: `status: completed`, two steps (`generate`, `rank`). Artifacts:

```bash
aws s3 ls s3://${BUCKET}/scenario-gen-smoke/${RUN_ID}/adversarial/
aws s3 ls s3://${BUCKET}/scenario-gen-smoke/${RUN_ID}/ranked/
# adversarial/manifest.json      (schema npa.scenario_gen.adversarial_set.v1)
# adversarial/scenarios/*.json   (per-scenario configs)
# ranked/ranked.json             (schema npa.scenario_gen.ranked_set.v1)
```

## 2) dataset-of-record smoke

`ingest -> validate -> quality gate -> curate -> register(query)`. The gate
writes a decision to S3 that the interpreter branches on (`config.quality_gate`
>= 0.5 promotes). Requires a small raw fixture on S3.

```bash
BUCKET=<your-bucket>
RUN_ID="dataset-smoke-$(date +%Y%m%d%H%M%S)"

cat > /tmp/records.json <<'JSON'
{"records": [
  {"record_id": "r1", "modality": "camera", "uri": "s3://example/r1.png", "event": "cut_in", "location": "sf", "timestamp": "2026-01-01T00:00:00Z", "quality": {"corruption": 0.0}, "embedding": [0.1, 0.2]},
  {"record_id": "r2", "modality": "lidar",  "uri": "s3://example/r2.bin", "event": "cut_in", "location": "la", "timestamp": "2026-01-01T00:01:00Z", "quality": {"corruption": 0.0}},
  {"record_id": "r3", "modality": "camera", "uri": "s3://example/r3.png", "event": "jaywalk", "location": "sf", "timestamp": "2026-01-01T00:02:00Z", "quality": {"corruption": 0.0}}
]}
JSON
aws s3 cp /tmp/records.json s3://${BUCKET}/dataset-of-record-fixtures/records.json

SPEC=/tmp/${RUN_ID}.yaml
sed "s/example-bucket/${BUCKET}/g" \
  npa/workflows/workbench/npa-workflows/dataset-of-record-smoke.yaml > "${SPEC}"

npa workbench workflow validate-spec "${SPEC}" --json
npa workbench workflow run-spec "${SPEC}" --run-id "${RUN_ID}" --execute --json
```

Expected: `status: completed`, steps `ingest, validate, quality-gate, curate,
register` (the `reject` branch is taken only when `quality_gate < 0.5`).
Artifacts:

```bash
aws s3 ls s3://${BUCKET}/dataset-smoke/${RUN_ID}/dataset/smoke-fleet/v1/
aws s3 ls s3://${BUCKET}/dataset-smoke/${RUN_ID}/validation/
aws s3 ls s3://${BUCKET}/dataset-smoke/${RUN_ID}/gate/
aws s3 ls s3://${BUCKET}/dataset-smoke/${RUN_ID}/curated/
# dataset/.../manifest.json          (schema npa.dataset.manifest.v1, lineage)
# validation/validation_report.json  (schema npa.dataset.validation_report.v1)
# gate/decision.json                 (promote_checkpoint)
# curated/.../manifest.json          (parent lineage + filter predicate)
```

## Notes

- To force the reject branch, set `config.quality_gate` below `0.5` (or
  `--var`-style edit the temp spec) and confirm the plan ends at `reject`.
- `run-spec --execute` runs stages locally as subprocesses; for cluster
  execution submit the GPU-backed production pipelines with
  `npa workbench workflow submit`.
- These specs keep `example-bucket` in-repo for hygiene; always run against a
  substituted temp copy.
