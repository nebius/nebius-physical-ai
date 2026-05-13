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
  --project-id YOUR_PROJECT_ID \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/cosmos-smoke/ \
  --job-name cosmos-smoke2-20260513T225839Z \
  --smoke \
  --smoke-seconds 5
```

Expected artifact: `checkpoint.json`.

## Isaac Lab Serverless Jobs E2E

```bash
npa workbench isaac-lab -p uk-south1 -n w7p-isaac train \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --task Isaac-Reach-Franka-v0 \
  --num-envs 1 \
  --steps 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/isaac-lab-smoke/ \
  --job-name isaac-lab-smoke3-20260513T225839Z
```

Expected artifacts: `npa_isaac_lab_train_summary.json`, `npa_isaac_lab_random_policy_checkpoint.json`.

## FiftyOne Serverless Jobs E2E

```bash
npa workbench fiftyone -p uk-south1 -n w7p-fiftyone load-dataset \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --name w7p-curated \
  --input-path Voxel51/VisDrone2019-DET \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/fiftyone-smoke/ \
  --job-name fiftyone-smoke-20260513T225839Z
```

Expected artifact: `npa_fiftyone_dataset_summary.json`.

## Genesis Serverless Jobs E2E

```bash
npa workbench genesis -p uk-south1 -n w7p-genesis train-teacher \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --n-envs 1 \
  --max-iterations 1 \
  --gpu-type l40s \
  --gpu-count 1 \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/genesis-smoke/ \
  --job-name genesis-smoke-20260513T225839Z
```

Expected artifacts: `train_teacher_summary.json`, `model.pt`.

## GR00T Serverless Jobs E2E

```bash
npa workbench groot -p uk-south1 -n w7p-groot infer \
  --runtime serverless \
  --project-id YOUR_PROJECT_ID \
  --input-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/groot-input/checkpoint/ \
  --dataset-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/groot-input/dataset/ \
  --output-path s3://YOUR_S3_BUCKET_2/w7p-fresh/20260513T225839Z/groot-smoke/ \
  --gpu-type h200 \
  --gpu-count 1 \
  --model-variant nvidia/GR00T-N1.7-3B \
  --steps 1 \
  --action-horizon 1 \
  --job-name groot-smoke-20260513T225839Z
```

W7-parallel-tools result: code path and unit coverage are present, but three smoke attempts failed before logs with Nebius internal Job errors. Treat this as `SMOKE_FAILED` until a follow-up run gets a terminal successful Job and uploaded artifacts.
