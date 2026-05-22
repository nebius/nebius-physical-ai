# Serverless E2E Tests

Serverless e2e tests create real Nebius Serverless AI Endpoints and must run
only against sandbox projects. They are not part of the default unit-test suite.

## Self-discovery

The serverless build run sets these environment variables in its own shell:

- `NPA_INTEGRATION_E2E=1`
- `NPA_E2E_SERVERLESS_PROJECT=<auto-selected-project-id>`

`NPA_E2E_SERVERLESS_PROJECT` is a Nebius project ID, not a region name. The
selection step reads `~/.npa/config.yaml`, resolves each project entry to a
project ID from `project_id` when present or from the entry key otherwise, then
maps the entry to a region using `region`, `location`, or `zone`.

Primary selection:

1. Prefer an accessible non-production project whose region matches `eu-north1`.
2. Otherwise use the first accessible non-production project in config order.
3. Exclude project IDs or aliases containing `prod` or `production`.

To run manually:

```bash
export NPA_INTEGRATION_E2E=1
export NPA_E2E_SERVERLESS_PROJECT=project-eXXXXXXXXXXXX
pytest npa/tests/e2e/test_cosmos_serverless_e2e.py -v -m e2e_serverless
```

## Credentials

Cosmos serverless inference uses the gated
`nvidia/Cosmos-1.0-Diffusion-7B-Text2World` model. The e2e harness requires a
Hugging Face token from `~/.npa/credentials.yaml` or from `HF_TOKEN`,
`HUGGINGFACE_TOKEN`, or `HUGGINGFACE_HUB_TOKEN` in the operator environment.
Token values are never printed in pytest output.

When a token is available, Cosmos serverless deploy propagates it to the
Endpoint runtime as both `HF_TOKEN` and `HUGGINGFACE_HUB_TOKEN`.

## Inference Timing

The prompt inference e2e path deploys the real Cosmos 7B text-to-world model.
Cold endpoint startup, model load, and diffusion sampling can exceed a short
synchronous CLI wait.

`test_e2e_cli_infer_prompt` therefore dispatches inference with
`cosmos infer --submit-only --output-format json`, then polls the Endpoint job
status every 30 seconds for up to 2400 seconds. The full real-Nebius suite was
validated on 2026-05-12 with `8 passed, 1 skipped`; the skip is the documented
forced-NER scenario that only runs when `NPA_E2E_FORCE_NER` is set.

## NER Fallback Chain

NER means a capacity/resource condition, not a generic failure. The client maps
these Nebius CLI messages to `NotEnoughResourcesError`: quota exceeded, quota
limit, limit reached, insufficient capacity, no capacity available, scheduling
failure, no GPU available, no resources available, out of capacity, or resource
not available.

Auth errors, image pull errors, HTTP inference failures, and timeout waiting for
`RUNNING` are not NER conditions. They fail the test instead of rotating.

The fallback chain starts with the selected primary project, then appends the
remaining non-production project IDs from `~/.npa/config.yaml` in declaration
order. Rotation is run-wide: once a project returns NER, subsequent tests skip
that project for the rest of the run.

## Cleanup

Every test tracks the project that actually created each endpoint. Fixture
teardown deletes the endpoint from that same project and removes transient
`npa-e2e-*` aliases from local config. If cleanup fails, the test prints a loud
`!!! ORPHANED ENDPOINT` line with the project and endpoint name so it can be
removed manually.

Gate checks compare preflight and postflight endpoint counts for every project
in the fallback chain and assert there are no leftover `npa-e2e-*` aliases.

## Subnets

Some projects have multiple VPC subnets. Cosmos serverless deploy accepts
`--subnet-id`; the e2e harness discovers a READY subnet with
`nebius vpc subnet list --parent-id <project-id> --format json` and passes that
subnet explicitly.

## Cosmos × Jobs (training)

The Cosmos training workload runs as a Nebius Serverless Job under
`--runtime serverless`. W1/W1.5 validated 5 of 6 hardening dimensions against
real Nebius:

- happy-path submission and completion: PASS
- NotEnoughResources fallback (project rotation): NOT REPRODUCED; the valid
  `gpu-h200-sxm` 8-GPU request was accepted in the sandbox
- cancel mid-execution: PASS
- status lifecycle transitions: PASS
- HF token propagation to the Job container env: PASS
- idempotent re-submission: PASS

Run: `pytest npa/tests/e2e/test_cosmos_jobs_serverless_e2e.py -v -m e2e_serverless`

Requires `NPA_INTEGRATION_E2E=1` and
`NPA_E2E_SERVERLESS_PROJECT=<sandbox-project-id>`. The NER test platform can be
overridden via `NPA_E2E_NER_PLATFORM`; related resource knobs are
`NPA_E2E_NER_PRESET` and `NPA_E2E_NER_GPU_COUNT`.

## NER E2E Deferral

The Cosmos Jobs NER e2e test (`ner_handling`) is currently marked as not
reproducible in the standard sandbox. The largest valid H200 request was
accepted by Nebius rather than rejected with `NotEnoughResources`, so the
project-rotation code path was not exercised end-to-end.

NER detection and classification logic is covered by unit tests in
`npa/tests/clients/test_serverless.py`, including the `_NER_PATTERNS`
classifier and structured exception metadata.

The customer-facing NER UX is covered by `docs/cli-errors.md` and
`docs/sdk/errors.md`, which describe how capacity failures surface in CLI
output, status commands, and SDK exceptions.

To verify NER UX manually, temporarily inject a `NotEnoughResourcesError` via a
mock test or use a quota-bound sandbox project coordinated with the Nebius
platform team.

## LeRobot × Jobs (training)

`npa workbench lerobot train --runtime serverless` runs LeRobot policy training
as a Nebius Serverless Job. Mirrors the Cosmos × Jobs pattern.

E2E validation: 7 of 10 hardening dimensions confirmed against real Nebius:

- happy-path: PASS
- NER handling: PLATFORM (high-GPU LeRobot preset mapping is not deterministic)
- cancel: TEST_FAILURE (Nebius internal cancel error; cleanup succeeded)
- status lifecycle: PASS
- HF propagation: PASS
- idempotent submit: PASS
- dataset from HF: PASS
- dataset from S3: SKIP (`NPA_E2E_LEROBOT_S3_DATASET` not set)
- Diffusion on H200: PASS
- `--submit-only`: PASS

Run: `pytest npa/tests/e2e/test_lerobot_jobs_serverless_e2e.py -v -m e2e_serverless`

Requires `NPA_INTEGRATION_E2E=1` and
`NPA_E2E_SERVERLESS_PROJECT=<sandbox-project-id>`. NER test platform can be
overridden via `NPA_E2E_NER_PLATFORM`.

Default GPU type per policy (encoded from May 2026 LeRobot GPU benchmark
research):

- Diffusion Policy: H200 preferred (~2.5x faster than B300 with stock PyTorch)
- Transformer-heavy (ACT, SmolVLA, VQ-BeT): H200 default, B300 acceptable

`--gpu-type b300 --policy-type diffusion` emits a CLI warning per the benchmark
findings.

## Cosmos Serverless Jobs E2E

Smoke command used by W7-parallel-tools:

```bash
npa workbench cosmos -p uk-south1 -n w7p-cosmos train \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/cosmos-smoke/ \
  --job-name cosmos-smoke2-<run-id> \
  --smoke \
  --smoke-seconds 5
```

Expected artifact: `checkpoint.json`.

## Isaac Lab Serverless Jobs E2E

```bash
npa workbench isaac-lab -p uk-south1 -n w7p-isaac train \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --task Isaac-Reach-Franka-v0 \
  --num-envs 1 \
  --steps 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/isaac-lab-smoke/ \
  --job-name isaac-lab-smoke3-<run-id>
```

Expected artifacts: `npa_isaac_lab_train_summary.json`, `npa_isaac_lab_checkpoint.pt`, `npa_isaac_lab_checkpoint_manifest.json`.

## FiftyOne Serverless Jobs E2E

```bash
npa workbench fiftyone -p uk-south1 -n w7p-fiftyone load-dataset \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --name w7p-curated \
  --input-path Voxel51/VisDrone2019-DET \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/fiftyone-smoke/ \
  --job-name fiftyone-smoke-<run-id>
```

Expected artifact: `npa_fiftyone_dataset_summary.json`.

## Genesis Serverless Jobs E2E

```bash
npa workbench genesis -p uk-south1 -n w7p-genesis train-teacher \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --n-envs 1 \
  --max-iterations 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/genesis-smoke/ \
  --job-name genesis-smoke-<run-id>
```

Expected artifacts: `train_teacher_summary.json`, `model.pt`.

## GR00T Serverless Jobs E2E

```bash
npa workbench groot -p uk-south1 -n w7p-groot infer \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --input-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-input/checkpoint/ \
  --dataset-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-input/dataset/ \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-smoke/ \
  --gpu-type h200 \
  --gpu-count 1 \
  --model-variant nvidia/GR00T-N1.7-3B \
  --steps 1 \
  --action-horizon 1 \
  --job-name groot-smoke-<run-id>
```

W7-parallel-tools result: code path and unit coverage are present, but three
smoke attempts failed before logs. W7p-groot-debug classified those failures as
a missing image tag: the jobs used `npa-groot:n1.7`, while the pushed GR00T
runtime image is `npa-groot:0.1.0`.

W7p-groot-debug fixed the default serverless image tag and retried once on
2026-05-14:

```bash
npa workbench groot -p uk-south1 -n w7pgd-groot infer \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --input-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-input/checkpoint/ \
  --dataset-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-input/dataset/ \
  --output-path s3://${NPA_S3_BUCKET}/<run-prefix>/groot-smoke/ \
  --gpu-type h200 \
  --gpu-count 1 \
  --model-variant nvidia/GR00T-N1.7-3B \
  --steps 1 \
  --action-horizon 1 \
  --job-name groot-smoke-retry-<run-id> \
  --timeout 3600 \
  --poll-interval 15
```

Retry result: `FAIL_DIFFERENT`. The job submitted image `npa-groot:0.1.0`,
advanced to `STARTING`, and allocated a running compute instance, but produced
no logs before cleanup. Treat GR00T as `SMOKE_FAILED` until Nebius investigates
`aijob-test-00000000000` or a later retry reaches terminal success and uploads
artifacts.

## LanceDB

LanceDB does not use Serverless Jobs for `deploy`; it is a persistent,
CPU-only service. Validate it with the local container smoke before any VM
smoke:

```bash
docker build -f npa/docker/lancedb/Dockerfile -t npa-lancedb:0.30.2 npa/

npa workbench lancedb deploy \
  --runtime container \
  --storage-path /tmp/npa-lancedb-smoke \
  --port 8686 \
  --auth-mode none \
  --replace \
  --image npa-lancedb:0.30.2

npa workbench lancedb create-table \
  --endpoint http://localhost:8686 \
  --table smoke_test \
  --input-path /tmp/tiny_dataset.json \
  --mode overwrite

npa workbench lancedb query \
  --endpoint http://localhost:8686 \
  --table smoke_test \
  --vector '[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]' \
  --top-k 5
```

W7-lancedb result: local container smoke passed with 10 rows and an
8-dimensional query vector. The public parent command still needs the
Workbench parent Typer registration follow-up; the smoke was run through the
new LanceDB subapp directly to stay inside the W7-lancedb write allowlist.

## SONIC

SONIC uses Nebius Serverless Jobs for Isaac Lab training smoke. The smoke target
is a short Unitree G1 run that validates the self-contained SONIC image,
`gear_sonic`, Isaac Lab imports, and S3 artifact upload:

```bash
SMOKE_TS=$(date -u +%Y%m%dT%H%M%SZ)
npa workbench sonic -p uk-south1 -n w7sonic train \
  --runtime serverless \
  --project-id <YOUR_PROJECT_ID> \
  --gpu-type l40s \
  --gpu-count 1 \
  --embodiment unitree-g1 \
  --steps 10 \
  --output-path s3://${NPA_S3_BUCKET}/w7sonic-smoke/$SMOKE_TS/ \
  --job-name sonic-smoke-$SMOKE_TS \
  --timeout 3600 \
  --poll-interval 15
```

Expected artifacts:

- `sonic_smoke_result.json`
- `sonic_train_summary.json`
- `checkpoint_smoke.json`

W7-sonic result: `FAIL_PLATFORM`. The CLI, tests, docs, and Dockerfile are in
place, but the local SONIC image did not build within the three-attempt Phase B
budget. No L40S job was submitted and no GPU spend was incurred. Re-run the
serverless smoke after the linux/amd64 image builds and is pushed to the Nebius
container registry.

## LanceDB S3-backed E2E (VM Downgrade)

The LanceDB e2e test is marked `e2e_serverless` so it stays outside default
pytest runs with the other real-infrastructure tests. Phase 0 of W7-lancedb-e2e
found that `npa workbench lancedb deploy --runtime vm --dry-run` is accepted,
but the non-dry-run VM/BYOVM path currently exits before provisioning. Until
managed VM provisioning is wired, the test exercises the real public CLI with
the container runtime and Nebius S3-backed storage.

Run:

```bash
export NPA_INTEGRATION_E2E=1
export NPA_E2E_SERVERLESS_PROJECT=<YOUR_PROJECT_ID>
export NPA_E2E_PROJECT=eu-north1

npa/.venv/bin/pytest npa/tests/e2e/test_lancedb_e2e.py -m e2e_serverless -k lancedb -v
```

Resources used:

- Project: `<YOUR_PROJECT_ID>` (`eu-north1`)
- Runtime: local Docker container, image `npa-lancedb:0.30.2`
- S3 bucket: `<your-bucket>`
- Storage prefix: `s3://${NPA_S3_BUCKET}/w7lancedb-e2e-<timestamp>-<id>/db/`
- Test duration: about 20 seconds when the image is already built
- Nebius compute cost: $0 for this downgraded path; S3 writes are a few small
  Lance files

What the test validates:

- Public `npa workbench lancedb` CLI deploys a LanceDB wrapper container
- Endpoint health check reaches `status: ok`
- A 100-row, 8-dimensional vector table can be created
- `list` returns the created table
- Basic vector query returns nearest-neighbor rows with distance fields
- Scalar filtered query returns only matching `label = 'robot'` rows
- LanceDB persists table files to Nebius S3 under a unique prefix
- Teardown removes the Docker container even if assertions fail after deploy

What remains pending:

- Full `--runtime vm` provisioning on Nebius CPU VM
- VM endpoint readiness and teardown through the LanceDB deploy command
- VM orphan-resource verification as part of the test itself
