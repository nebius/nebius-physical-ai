# Use Workbench with your coding agent

[Workbench docs](README.md)

Workbench is the control plane between your coding agent and physical-AI
workloads on Nebius. Use Codex, Claude Code, or another agent with terminal
access to this checkout; NPA does not require or select a particular reasoning
system. Your agent operates the same checked, auditable `npa` surfaces available
to a human: configure, preflight, plan, provision, submit, monitor, and inspect.

Install `npa` and the Nebius CLI using the [quickstart](../quickstart.md), then
paste one of the prompts below. The [self-hosted browser agent](../agent.md) is
optional and requires a separate deployment.

| Goal | Prompt |
| --- | --- |
| Prepare a project for any supported task | [Set up Workbench](#set-up-workbench) |
| Generate an image with the supported Cosmos 3 workflow | [Run Cosmos 3 generation](#run-cosmos-3-generation) |
| Augment and curate a source video | [Run PAIDF with Cosmos 3](#run-paidf-with-cosmos-3) |

## Choose tools for your task

Start with the [workload guides](guides/README.md) or the
[tool reference](../cli/workbench.md). Ask your agent to inspect the current
`skills/index.yaml`, relevant guides, and command help before choosing a path.
Existing examples are starting points; available tools and their contracts
determine what you can run or combine. If the task needs an unsupported
capability, have the agent explain the gap.

The setup prompt below applies to the task you choose. The two Cosmos prompts
are complete runs: standalone generation is the shorter first workflow, while
PAIDF is a longer source-video augmentation and curation pipeline.

<a id="set-up-workbench"></a>

## Set up Workbench

Supply the project values and any credentials needed by your selected tools
through the agent's **private environment**. Keep token values out of chat.
Supported names include:

```text
NEBIUS_TENANT_ID=<your-tenant-id>
NEBIUS_PROJECT_ID=<your-project-id>
NEBIUS_REGION=<your-project-region>
HF_TOKEN=<your-hugging-face-read-token>
NGC_API_KEY=<your-ngc-api-key>
NEBIUS_TOKEN_FACTORY_KEY=<your-token-factory-key>
```

Credential requirements depend on the selected tools. For gated Hugging Face
assets, the account behind `HF_TOKEN` needs access and a fine-grained token must
include the required repositories. Accept gated terms yourself on Hugging Face.
NGC credentials apply to tools that fetch entitlement-controlled NGC artifacts.
Token Factory credentials apply to hosted inference, including the captioning
and evaluation stages in the PAIDF example below. See
[configuration](../configuration.md#required-credential-key-names) for the
credential names and when each is needed.

First-time storage setup needs **admin permission on the target project**.
`npa configure` creates a project service account, access key, and project-scoped
IAM group with a bucket-scoped `storage.object-editor` permit. Tenant-wide admin
permission and tenant-wide project listing are not required.

Describe the task you want to run, then give your agent this prompt:

```text
Set up Nebius Physical AI Workbench for the task I described. If the intended
result or input is unclear, ask one concise question. Read AGENTS.md,
skills/index.yaml, the relevant skills and workload guide, and current command
help. Choose a supported tool or workflow and state its inputs, artifacts, model
access, infrastructure, and any unsupported part.

Use project values and credentials only through the private environment and
npa's configuration APIs. Never print secret values, dump the environment, read
credential files directly, pass secret values in arguments, or commit live
infrastructure identifiers. Inspect only required variable names as present or
missing. Use NPA_PROJECT_ALIAS when set; otherwise use "workbench".

Install or verify npa, the Nebius CLI, and required host tools. Configure known
values non-interactively with npa configure --save-env-credentials. Before
explicit S3 provisioning, explain the storage and confirm the required project
permissions.

Inspect npa configure --show. Run npa workbench health preflight for the selected
services, including --checks nebius before cloud provisioning, then run health
access for required model assets. Stop on any failed gate and do not provision
GPUs. If access needs human approval, show the official pages and wait; never
accept terms for me. Finish with what passed, blockers, the planned workload
and resources, and the next command.
```

For project creation, federation or SSO profiles, non-interactive setup, and
credential recovery, see [configuration](../configuration.md).

## Run the selected task

Follow the relevant tool or workflow guide with your actual inputs. For a
workflow, validate its specification and inspect the plan before provisioning
compute, staging data, checking images, and submitting. For an individual tool,
use its documented command and runtime requirements. Check the resulting
artifacts against your intended output and follow the applicable recovery and
cleanup instructions.

<a id="run-cosmos-3-generation"></a>

## First workflow: Cosmos 3 generation

This prompt runs the checked-in
[`workflows/testing/cosmos3-generate.yaml`](../../workflows/testing/cosmos3-generate.yaml)
workflow on a compatible Nebius GPU. Its default is Cosmos3-Nano text-to-image
on one H100, producing `vision.jpg` and `generate.json` in S3. The default
guardrails require gated Hugging Face access even though the main checkpoint is
public. Paste this prompt after making project values and `HF_TOKEN` available
to the agent through its private environment:

```text
Run workflows/testing/cosmos3-generate.yaml to a verified result. Follow
AGENTS.md, skills/index.yaml, the relevant first-run, Cosmos 3, and workflow
operation skills, and docs/workbench/cosmos3-generate.md.

Use the configured project, writable bucket, and credentials through npa. Inspect
resource and artifact identifiers locally when required, but redact them from
chat, reports, and commits. Never print secrets, dump the environment, or pass
secret values in arguments or YAML. Inspect only required variable names as
present or missing.

Before provisioning, inspect npa configure --show and run the Nebius,
selected-project S3, and Cosmos 3 access gates. Keep guardrails enabled. If
access needs human approval, show the official pages, wait for me, and rerun the
gate; never accept terms for me or bypass a failure.

Validate and plan the checked-in spec with my actual bucket. State the expected
vision.jpg and generate.json artifacts and requested resources. Preview
provision-if-absent, create only missing resources through npa, bootstrap and
verify SkyPilot, discover the cluster's accelerator name, reconcile it if
needed, then revalidate and preflight the exact images.

Submit with --runtime and forward secrets only by name with --secret-env. Stay
until the run is terminal. On failure, use bounded status, logs, and artifact
inspection and resume when supported. On success, verify non-empty media,
inspect generate.json, and show the result in an available viewer. Report the
outcome, artifacts, and still-running resources, then offer the documented
cancel-before-destroy cleanup path without destroying shared infrastructure.
```

Do not treat a valid plan or accepted submission as completion: the prompt ends
with inspected artifacts or a concrete external blocker. The
[Cosmos 3 generation guide](cosmos3-generate.md) is the command-level source of
truth for the workflow, outputs, access checks, and current limitations.

<a id="run-paidf-with-cosmos-3"></a>

## Advanced workflow: PAIDF with Cosmos 3

Use the same agent from input selection through output inspection. For the
Physical AI Data Factory with real source-video-conditioned Cosmos 3, attach a
local H.264 MP4 or set `PAIDF_INPUT_URI` to one private `s3://` MP4, then paste:

```text
Run workflows/main/paidf-cosmos3.yaml to a terminal result. Follow AGENTS.md,
skills/index.yaml, the relevant repository skills, and
docs/workbench/guides/paidf-cosmos3.md. Use the configured project, region,
writable bucket, and credentials. Use the attached H.264 MP4 or
PAIDF_INPUT_URI; if neither exists, ask only for the input video.

Keep input, artifact, and infrastructure locations private. Never print secrets,
dump the environment, read credential files directly, or pass secret values in
YAML or arguments. Inspect required variable names only as present or missing.

Before provisioning, run npa workbench health preflight for the selected project
with --checks s3,token_factory,nebius, then run npa workbench health access
--capability cosmos3. Validate and plan with the real bucket and input, one
variant, and one supported GPU; honor configured TF_VAR_* topology and reserved
capacity. Show the plan and resources. Through npa, preview and create only
missing resources, bootstrap and verify SkyPilot, discover the advertised
accelerator, apply any required override, and revalidate. Stage the input and
preflight the selected images. If an image fails the bootstrap contract, follow
the guide's compliant immutable build-and-push path and repeat preflight; do not
bypass it.

Submit with --runtime and forward only these secret names with --secret-env:
HF_TOKEN, NEBIUS_TOKEN_FACTORY_KEY, AWS_ACCESS_KEY_ID, and
AWS_SECRET_ACCESS_KEY. Stay until terminal. On failure, diagnose the recorded
stage and resume when safe. On success, show the generated and curated artifacts
and load the final Rerun recording when available. Treat a terminal quality
rejection as valid fail-closed behavior: do not lower thresholds or force
promotion; show the generated video, evaluator report, disposition, and Rerun
evidence, and explain why labeling and curation were skipped.
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
