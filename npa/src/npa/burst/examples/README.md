# Burst task examples

[Workflow catalog](../../../../../workflows/README.md) · [Burst operating guide](../../../../../skills/tools/burst/SKILL.md)

These are single-task inputs to `npa burst submit-yaml`, **not workflow templates**.
The current example runs a small Isaac Lab Cartpole training exercise on one
L40S VM, then writes a **local Cosmos SDG contract manifest**. It does not run
Cosmos generation or upload that manifest to S3.

## Before submitting

Install NPA and [bootstrap SkyPilot](../../../../../docs/orchestration/skypilot-setup.md).
Configure an authorized Nebius project, verify credentials, and check L40S
capacity and the selected Isaac image. This YAML uses `cloud: nebius`; it does
not consume an existing Kubernetes GPU pool. Isaac runtime use requires the
operator's accepted NVIDIA terms and an RT-core GPU.

Public GHCR images need no registry credentials. For a private image, configure
exact-host SkyPilot Docker credentials as described in the Burst guide.

## Submit and inspect

Run from the repository root with your own run label and destination contract:

```bash
npa burst submit-yaml npa/src/npa/burst/examples/isaac-lab-cosmos-sdg-burst-smoke.yaml \
  --name '<run-id>' \
  --var 'NPA_RUN_ID=<run-id>' \
  --var 'ISAAC_LAB_IMAGE=ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab:3.0.0b2.post1' \
  --var 'NPA_OUTPUT_URI=s3://<your-bucket>/<prefix>/<run-id>/' \
  --json
```

Save the returned job ID and handle. Use the same runtime configuration to
inspect it:

```bash
npa burst status '<job-id-or-handle-json>'
npa burst logs '<job-id-or-handle-json>' --follow
```

Expect completed Isaac training and `cosmos_sdg_manifest.json` in the task's
local output directory, with the manifest printed in logs. `NPA_OUTPUT_URI`
populates URI fields; it is not an uploader. Preserve needed logs and files
before deleting compute. Burst has no durable workflow artifact ledger or
resume command. Follow [teardown](../../../../../docs/teardown.md) for the exact
managed job and its owned resources.

## Rules for changes

- **One task per file.** Multi-stage pipelines belong in `workflows/` as
  `npa.workflow/v0.0.1` specs.
- Keep `${VAR}` placeholders; submission refuses unresolved values. Never commit
  live resource IDs, bucket names, or credentials.
- The example set is pinned by
  [`test_burst_examples.py`](../../../../tests/guardrails/test_burst_examples.py).
