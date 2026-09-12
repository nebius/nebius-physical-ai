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
result or input is unclear, ask me one concise question before choosing tools.
Read AGENTS.md and skills/index.yaml, then follow the relevant repository skills,
workload guide, and current command help. Select a supported tool or workflow
that directly produces my intended result. State its inputs, output artifacts,
model access, infrastructure, and any unsupported part of my request.

Use project values and the credentials required by the selected tools from
the private process environment. Never print secret values, put them in command
arguments, or write them into the repository.
Never run `env`, `printenv`, `set`, `export -p`, or another command that dumps
the process environment; inspect only allowlisted names and report present or
missing. Do not read credential files except through npa's credential APIs.
Use NPA_PROJECT_ALIAS if it is set; otherwise use "workbench" as the local alias.

Install or verify npa, the Nebius CLI, and the host prerequisites for the
selected runtime. Configure known project values non-interactively; ask only for
required non-secret values you cannot resolve. Persist supported environment
credentials with npa configure --save-env-credentials. If the task needs S3,
explain the required storage before using explicit --provision to create or
reuse it, and first confirm that the active identity has the required project
permissions.

Inspect npa configure --show. Run credential preflight checks for the selected
services, including --checks nebius before cloud provisioning. Use npa workbench
health access for the selected capabilities and their required model assets.

Do not bypass a failed gate or provision GPU resources during setup. If model
access needs human approval, show me the exact official pages and wait; never
accept terms for me. Finish with a concise readiness report: what passed, what
is still blocked, the planned workload and resources, and the next command.
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
Run my first Nebius Physical AI Workbench workflow:
workflows/testing/cosmos3-generate.yaml. Follow AGENTS.md, skills/index.yaml,
the relevant Cosmos 3 and workflow-operation skills, and
docs/workbench/cosmos3-generate.md. Use the current local checkout and its npa
installation. Use the configured project, writable bucket, and credentials.
First install or verify npa, the Nebius CLI, and the host tools required by the
maintained Workbench setup guide.

Never print secret values, put them in command arguments or YAML, or dump the
process environment. Inspect only the allowlisted names needed for this run and
report each as present or missing. Resolve credentials through npa's supported
configuration APIs. Keep project, cluster, bucket, registry, and artifact
identifiers out of chat and committed files.

Inspect npa configure --show. Run the Nebius, selected-project S3, and Cosmos 3
model-access gates before provisioning. If access is pending, show me the exact
official model pages and wait for me to accept the terms; never accept them for
me. Rerun the gate after I finish. Do not disable guardrails or bypass a failed
check.

Validate and plan the checked-in spec using my actual bucket. Explain the
expected vision.jpg and generate.json output plus the requested CPU, memory,
and GPU before provisioning. Preview provision-if-absent, then create only
missing resources through npa. Bootstrap and verify the isolated SkyPilot
runtime, discover the accelerator name exposed by the selected cluster, and
reconcile the workflow resource request if necessary. Revalidate the plan and
preflight the exact workflow images.

Submit the workflow with --runtime and forward secrets only by name with
--secret-env. Stay with the run until it reaches a terminal state. On failure,
use npa workbench workflow status, logs, and artifacts to diagnose the recorded
stage; keep log output bounded and resume safely when supported instead of
launching an unrelated run. On success, verify that the generated media is
non-empty, inspect generate.json, and show me the media in an available agent
viewer. Finish with the run outcome, artifact types, and any still-running
resources. Do not destroy shared infrastructure; offer the documented
cancel-before-destroy cleanup path.
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
Run my first Workbench workflow with me: the PAIDF Cosmos 3 video-conditioning
workflow at workflows/main/paidf-cosmos3.yaml. Follow
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
