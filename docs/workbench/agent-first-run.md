# Run your first workflow with a coding agent

Use your existing coding agent with terminal access to this checkout. Install
`npa` and the Nebius CLI using the [quickstart](../quickstart.md), then follow
the setup and workflow prompts below. The [self-hosted browser agent](../agent.md)
is optional and requires a separate deployment.

## Configure the project and model access

If an agent is operating the checkout, attach these values to its **private
environment** instead of pasting token values into chat:

```text
NEBIUS_TENANT_ID=<your-tenant-id>
NEBIUS_PROJECT_ID=<your-project-id>
NEBIUS_REGION=<your-project-region>
HF_TOKEN=<your-hugging-face-read-token>
NGC_API_KEY=<your-ngc-api-key>
NEBIUS_TOKEN_FACTORY_KEY=<your-token-factory-key>
```

The Hugging Face account behind `HF_TOKEN` must already have access to the
models used by the workflow. A fine-grained token must include those repos.
Gated terms can only be accepted by you on Hugging Face; the access check below
prints every exact page still requiring action. NGC is part of the general
Workbench setup even though the PAIDF + Cosmos 3 path below currently obtains
its model weights from Hugging Face. Token Factory is required by that path for
captioning and evaluation.

First-time storage setup needs **admin permission on the target project**.
`npa configure` creates a project service account, access key, and project-scoped
IAM group with a bucket-scoped `storage.object-editor` permit. Tenant-wide admin
permission and tenant-wide project listing are not required.

Then give your agent this prompt:

```text
Set up Nebius Physical AI Workbench in this checkout. Read AGENTS.md and
skills/index.yaml first and follow the relevant first-run, credential-preflight,
and GPU guidance.

Use NEBIUS_TENANT_ID, NEBIUS_PROJECT_ID, NEBIUS_REGION, HF_TOKEN, NGC_API_KEY,
and NEBIUS_TOKEN_FACTORY_KEY from the private process environment. Never print
secret values, put them in command arguments, or write them into the repository.
Never run `env`, `printenv`, `set`, `export -p`, or another command that dumps
the process environment; inspect only allowlisted names and report present or
missing. Do not read credential files except through npa's credential APIs.
Use NPA_PROJECT_ALIAS if it is set; otherwise use "workbench" as the local alias.

Install or verify npa and its reported host prerequisites, including Terraform
and, for SkyPilot Kubernetes on Debian/Ubuntu, socat. Verify the active Nebius
CLI identity and configure the known tenant, project, and region
non-interactively. Persist supported environment credentials with npa configure
--save-env-credentials and use explicit --provision to create or reuse writable
project object storage. Without --provision, known-project setup must remain
provider-free and leave storage unselected.
Confirm the active identity can manage the project-scoped IAM objects that
secure that storage. Then run npa configure --show,
npa workbench health preflight --json,
npa workbench health preflight --checks nebius --json, and
npa workbench health access --capability paidf,cosmos3 --json.

Do not bypass a failed gate or provision GPU resources yet. If Hugging Face
access is missing, give me the exact model-page links, wait for me to accept the
terms, and rerun the check. Finish only when project storage, credentials, and
the PAIDF/Cosmos 3 model access checks pass, then guide me straight into my first
workflow.
```

For project creation, federation or SSO profiles, non-interactive setup, and
credential recovery, see [configuration](../configuration.md).

## Run PAIDF with Cosmos 3

Use the same agent from input selection through output inspection. For the
Physical AI Data Factory with real source-video-conditioned Cosmos 3, attach a
local H.264 MP4 or set `PAIDF_INPUT_URI` to one private `s3://` MP4, then paste:

```text
Run my first Workbench workflow with me: the PAIDF Cosmos 3 video-conditioning
workflow at npa/workflows/workbench/npa-workflows/paidf-cosmos3.yaml. Follow
docs/workbench/guides/paidf-cosmos3.md and the repository skills. Use the
configured Nebius project, region, writable bucket, and credentials. Use the
attached local H.264 MP4, or PAIDF_INPUT_URI if it is set; if neither is
available, ask me only for the input video before continuing. Keep all input and
artifact locations private.

Never run `env`, `printenv`, `set`, `export -p`, or another command that dumps
the process environment. Inspect only allowlisted variable names and report
present or missing; do not print secret values or read credential files directly.

Re-run the credential and model-access gates for paidf,cosmos3. Validate and
plan the spec with the real bucket and input, starting with one variant and one
supported GPU. Honor configured TF_VAR_* topology and reserved-capacity settings.
Show me the validated plan and explain the required CPU/GPU resources before
provisioning. Bootstrap and verify SkyPilot, provision any required resources
that are absent, and discover the accelerator name the target cluster advertises.
Use that name in the resource overrides and revalidate the plan. Then stage the
input and preflight the selected images. If a selected image fails the
SkyPilot bootstrap contract, build the repository's current compliant image,
push it to an authorized private project registry at an immutable digest, and
repeat preflight. Then submit with --runtime.
Forward only secret names through --secret-env: HF_TOKEN,
NEBIUS_TOKEN_FACTORY_KEY, AWS_ACCESS_KEY_ID, and AWS_SECRET_ACCESS_KEY; never put
secret values in YAML or command arguments.

Stay with the run until it reaches a terminal state. If it fails, diagnose the
recorded stage and resume safely rather than starting an unrelated run. If it
succeeds, show me the generated and curated artifacts and load the final Rerun
recording when an agent viewer is available. A terminal quality rejection after
the workflow's bounded refinement loop is a valid fail-closed result: do not
lower the threshold or force promotion. Show me the generated video, evaluator
report, quality disposition, and Rerun evidence, and explain that labeling and
curation were intentionally skipped.
```

The workflow uses the independent
[`paidf-cosmos3.yaml`](guides/paidf-cosmos3.md) composition; the
original Physical AI Data Factory workflow continues to use Cosmos Transfer
2.5.

The [PAIDF + Cosmos 3 guide](guides/paidf-cosmos3.md) defines the source-video
conditioning, required images, model access, and output locations. Its PAIDF
published composite blends source and generated frames (80% source by default); inspect the raw Cosmos
output separately when evaluating generation quality.

Image preflight can create and delete a temporary probe pod when bootstrap
evidence is absent. Review the [run lifecycle](../run-lifecycle.md) before
running it. Once finished, follow [teardown](../teardown.md) for resources you own.
