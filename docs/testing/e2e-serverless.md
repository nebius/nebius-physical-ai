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
