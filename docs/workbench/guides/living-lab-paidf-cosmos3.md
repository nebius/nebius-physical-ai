# Living Lab participant guide: PAIDF with Cosmos 3

Run the Physical AI Data Factory (PAIDF) Cosmos 3 workflow on the Nebius GPU
cluster supplied with your Living Lab test project. You can follow this guide
in a terminal or give it to a coding agent that operates your terminal.

The workflow selects a robot video, captions it with a hosted vision-language
model through Token Factory, generates appearance variants with Cosmos3-Nano
on your GPU, and evaluates them with Cosmos Evaluator. Accepted variants pass
through captioning, Cosmos Curator, and FiftyOne Brain before a final report.
Rejected variants produce Rerun quality evidence and stop before curation.

> **Validation scope:** adapted from the Living Lab participant document's
> August 28–September 3, 2026 validation: eight runs on a test project with two
> RTX PRO 6000 Blackwell GPUs and two CPU nodes. The first nine tasks of the
> then-rendered 15-task plan, through the quality-evidence recording, completed
> in three consecutive runs. Those runs rejected the generated data and did
> **not** validate the downstream curation/finalization path. Instructions below
> follow the repository as of September 9, 2026; this documentation conversion
> does not establish a new live validation result. See the
> [historical results](#historical-validation-results).

## Before you start

Your welcome packet or cohort contact should provide:

- A project invitation. Accept it before authenticating; invitations can expire.
- Your tenant ID, project ID, and region.
- The pre-provisioned cluster's ID and name, plus access to its Kubernetes API.
- Token Factory access and any credit instructions for your cohort.

You also need a Hugging Face account and a read token. The cluster must have
capacity for the workflow's GPU task, its CPU tasks, and the SkyPilot controller.
The historical RTX PRO 6000 runs used a 185 GB GPU-node boot disk after smaller
disks caused evictions. That is an observed working size for those images and
weights, not a guarantee for future releases.

This guide adopts your supplied cluster. For a project that does not already
have one, start with [Workbench Kubernetes setup](../kubernetes.md).

## P — Prerequisites

### P1. Install the base tools

On macOS with Homebrew:

```bash
brew install python@3.12 git kubectl awscli ffmpeg socat netcat jq
```

On Ubuntu, install the equivalent packages and Python 3.12 using the
[platform installation guide](../../install.md). Use Python 3.12 for both the
NPA environment and isolated SkyPilot environment in this guide. The supported
SkyPilot runtime requires Python 3.9–3.12; a newer interpreter can fail at submit.
Terraform is not needed to adopt the cluster supplied to participants.

### P2. Install the Nebius CLI

The source guide was validated with version `0.12.254`, which is accepted by
the repository's CLI compatibility check:

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | NEBIUS_CLI_VERSION=0.12.254 bash
```

Use the version requested by NPA if its compatibility check changes. An
unsupported CLI can produce an authentication-looking error; check the version
before recreating a working profile.

### P3. Accept the project invitation and authenticate

Replace each angle-bracket placeholder with the value from your welcome packet.
Keep these non-secret shell variables available for the setup steps:

```bash
export TENANT_ID='<your-tenant-id>'
export PROJECT_ID='<your-project-id>'
export REGION='<your-region>'
export CLUSTER_ID='<your-cluster-id>'
export CLUSTER_NAME='<your-cluster-name>'
export PROJECT_ALIAS=living-lab

nebius profile create living-lab \
  --endpoint api.nebius.cloud \
  --federation-endpoint auth.nebius.com \
  --parent-id "$PROJECT_ID"
```

Authentication opens a browser. If it reports `invalid IAM subject` or
`PermissionDenied`, check the invitation and project permissions in the Nebius
console. Ask your cohort contact to resend an expired invitation.

### P4. Prepare Hugging Face and Token Factory access

Create a Hugging Face read token under **Settings → Access Tokens**. Review the
model terms and request access where the model page requires it:

- [Cosmos3-Nano](https://huggingface.co/nvidia/Cosmos3-Nano)
- [Cosmos-Guardrail1](https://huggingface.co/nvidia/Cosmos-Guardrail1)

The current `health access --capability cosmos3` command checks these repository
entries. Follow any additional dependency or access requirements reported by
your installed version; do not treat a token's presence as proof of model access.
See [Cosmos 3 access preflight](../cosmos3-access-preflight.md).

For Token Factory, sign in to the [Token Factory console](https://tokenfactory.nebius.com/)
and follow your cohort's credit instructions. Under **API keys**, create a key
and save it when displayed; it cannot be reopened later. The
[official authentication instructions](https://docs.tokenfactory.nebius.com/api-reference/introduction#authentication)
describe this process. This key is separate from your Nebius Cloud IAM token.

## S — One-time setup

### S1. Install the repository and NPA CLI

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3.12 -m venv .venv
source .venv/bin/activate
python3.12 -m pip install -e npa/
npa --version
```

Run the remaining commands from this repository root. Participant installation
uses `.venv`; contributor validation tooling uses `npa/.venv` as described in
the [installation guide](../../install.md).

### S2. Configure the project and provision its storage

Use your active Nebius CLI profile to configure NPA and provision the bucket
and storage credentials for the invited project:

```bash
unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN
npa configure --no-interactive --provision \
  --tenant-id "$TENANT_ID" --project-id "$PROJECT_ID" \
  --region "$REGION" --project-alias "$PROJECT_ALIAS"
npa configure --show
```

Note the bucket and storage endpoint it reports. `--provision` here performs
storage setup; cluster adoption is a separate step below.

### S3. Store the service tokens

Merge this `tokens` mapping into `~/.npa/credentials.yaml`, replacing the
placeholders in your local editor. Preserve the project storage records that
`configure` created:

```yaml
tokens:
  HF_TOKEN: <your-hugging-face-read-token>
  NEBIUS_TOKEN_FACTORY_KEY: <your-token-factory-api-key>
```

```bash
chmod 600 ~/.npa/credentials.yaml
```

Storage keys belong to the selected project's record under
`project_credentials.projects`. Use the
[credential-file reference](../../credentials.yaml.example) to inspect the
shape without copying credential values into the repository or shell history.
Submit resolves the named secrets from this store or the process environment.

### S4. Check credentials and model access

```bash
npa workbench health preflight
npa workbench health access --capability cosmos3
```

Resolve failures for S3, Token Factory, and the required Hugging Face assets
before submitting. An NGC warning is not a requirement to add NGC credentials
for this Cosmos 3 path. These checks establish access; they do not prove a
successful GPU run or acceptable generated data.

### S5. Adopt the supplied Kubernetes cluster

Fetching Kubernetes credentials alone does not register the cluster with NPA.
Fetch them, find the exact context name, then record the cluster and bind its
controller:

```bash
nebius mk8s cluster get-credentials --id "$CLUSTER_ID" --external
kubectl config get-contexts
```

```bash
export KUBE_CONTEXT='<context-from-the-previous-command>'
npa cluster kubeconfig \
  --cluster-name "$CLUSTER_NAME" \
  --project "$PROJECT_ALIAS" --context "$KUBE_CONTEXT"
npa skypilot bind-controller \
  --project "$PROJECT_ALIAS" --context "$KUBE_CONTEXT"
```

The Nebius CLI profile supplies cloud authentication, the NPA project alias
selects saved project credentials, and the Kubernetes context selects the
cluster. Use the values belonging to this participant project throughout.

### S6. Bootstrap and verify SkyPilot

```bash
npa skypilot bootstrap --python "$(command -v python3.12)"
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
export KUBECONFIG="$HOME/.npa/clusters/$KUBE_CONTEXT/kubeconfig"
npa skypilot verify --cluster "$KUBE_CONTEXT"
"$NPA_SKYPILOT_BIN" gpus list
```

Verification must report Kubernetes enabled for this context. Copy the exact
GPU spelling from discovery; for example, an RTX PRO 6000 cluster may advertise
`RTXPRO-6000-BLACKWELL-SERVER-EDITION`. Use it with the requested count in the run
section. See [SkyPilot setup](../../orchestration/skypilot-setup.md).

### S7. Configure the AWS profile for direct artifact downloads

NPA submission resolves project storage from its credential store. The `aws`
commands in this guide use a profile named `nebius`. If you use those commands,
merge the selected project's access and secret keys from S2 into your local
`~/.aws/credentials`:

```ini
[nebius]
aws_access_key_id = <your-project-s3-access-key-id>
aws_secret_access_key = <your-project-s3-secret-access-key>
```

Merge the matching region and endpoint into `~/.aws/config`:

```ini
[profile nebius]
region = <your-region>
endpoint_url = https://storage.<your-region>.nebius.cloud
```

Use the exact endpoint reported by `npa configure --show`. Keep `endpoint_url`
at profile level. If the local SkyPilot API server cached an older AWS profile,
restart it after correcting the profile:

```bash
"$NPA_SKYPILOT_BIN" api stop
"$NPA_SKYPILOT_BIN" api start
```

### S8. Source staging is automatic

Current submit stages the NPA package automatically for tasks that need it,
uses a content-addressed source prefix, and persists that reference for recovery.
The old three-command upload of `npa/src`, `npa/workflows`, and `pyproject.toml`
is no longer the setup path. Keep the repository checkout available when you
submit and use the canonical spec at `workflows/main/paidf-cosmos3.yaml`.

## R — Run the workflow

### R1. Select the project, GPU, and caption model

In a new terminal, return to your checkout and activate `.venv`. Restore the
project alias and cluster context selected above, and set the bucket and
accelerator from your setup results:

```bash
source .venv/bin/activate
export PROJECT_ALIAS=living-lab
export KUBE_CONTEXT='<your-adopted-context>'
export BUCKET='<your-configured-bucket-name>'
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
export KUBECONFIG="$HOME/.npa/clusters/$KUBE_CONTEXT/kubeconfig"
export NPA_WORKFLOW_GPU_ACCELERATOR='<discovered-gpu-name>:1'
SPEC=workflows/main/paidf-cosmos3.yaml

npa workbench token-factory models
```

Choose an available vision model for the captioning and evaluator stages:

```bash
export CAPTION_MODEL='<available-vision-model-id>'
```

The workflow currently defaults to `MiniMaxAI/MiniMax-M3`; the model list is
authoritative for availability at run time. The original guide used
`google/gemma-3-27b-it` after its earlier default disappeared. Do not assume a
historically working model is still offered.

Workbench images use the public anonymous GHCR namespace
`ghcr.io/nebius/nebius-physical-ai` by default. No registry credential or pull
secret is needed for that mirror. Use the repository's selected image pins;
the old instruction to patch PR #374's Rerun pin is no longer needed.

### R2. Reserve a run ID, validate, and inspect the plan

```bash
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT_ALIAS")"
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" --assume-decision promote_checkpoint \
  --var bucket="$BUCKET" --var caption_model="$CAPTION_MODEL" --json
npa workbench workflow preflight-images "$SPEC" --project "$PROJECT_ALIAS"
```

Stop and resolve any failed check. The assumed decision lets you inspect a
plan; the runtime submit below follows actual evaluator decisions. Leave image
preflight enabled so incompatible images fail before a stage starts.

### R3. Submit

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket="$BUCKET" \
  --var caption_model="$CAPTION_MODEL" \
  --runtime \
  --assume-decision promote_checkpoint \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN \
  --project "$PROJECT_ALIAS" \
  --durable-s3
```

The first run needs no input preparation: submit fetches and verifies the
pinned real robot starter video, stages it for the run, and passes the source
to the Cosmos 3 workflow. It also stages source code and forwards the four
named secrets without requiring their values in the command.

`--runtime` lets the orchestrator read evaluator decisions and execute real
refinement loops. The default one-shot path flattens the plan using
`--assume-decision` and is not equivalent to runtime retries. The command keeps
the workflow's configured variant count and refinement settings.

For your own video, add **one** of these input options to the submit command:

- `--input-video /absolute/path/source.mp4` for a local H.264 MP4.
- `--input-uri 's3://<your-bucket>/<your-prefix>/source.mp4'` for a stored MP4.

The original participant runs used short robot clips. Check the resulting
duration and motion: basic `video2video` conditioning does not establish
full-episode motion preservation. For LeRobot episode/camera selection and the
generation contract, see the [Cosmos 3 workflow guide](paidf-cosmos3.md).

### R4. Monitor and recover

In another terminal with the same project and run ID:

```bash
npa workbench workflow status "$RUN_ID" --project "$PROJECT_ALIAS" --watch
npa workbench workflow logs "$RUN_ID" --project "$PROJECT_ALIAS" \
  --stage generate-variants --follow
"$NPA_SKYPILOT_BIN" jobs queue
```

The runtime may submit several managed jobs or attempts. Use NPA's stage records
to identify the GPU task and each retry; do not assume one job ID or a fixed
15-row queue describes every run.

For an interrupted launch, follow the recovery command reported by NPA. Replace
`--run-id "$RUN_ID"` with `--resume-run "$RUN_ID"` in the same submit command,
retaining the project, runtime, inputs, configuration, and secret names.
`--run-id` and `--resume-run` are mutually exclusive. Resume reconciles recorded
work; it does not turn a rejected result into an accepted one. Use `prepare-run`
to create a fresh run after changing the experiment. See the
[run lifecycle](../../run-lifecycle.md).

## Troubleshooting

These symptoms came from the original participant validation. Remedies below
use current setup and recovery paths where those have changed.

| Symptom | What to check | Action |
| --- | --- | --- |
| `invalid IAM subject` or `PermissionDenied` | Invitation and project permissions | Accept the pending invite, or ask your cohort contact to resend an expired one; retry P3. |
| `The active Nebius CLI profile cannot authenticate non-interactively` | CLI compatibility as well as authentication | Check the version first; install the compatible CLI from P2 before replacing the profile. |
| `legacy global storage credentials have no unique exact-project ownership` | Credentials left by an older NPA installation | Back up the local credential file, identify which project owns the keys, and reconcile against the [project credential schema](../../credentials.yaml.example). Do not assign ambiguous keys to the new project. |
| Missing bootstrap-contract attestation for `npa-rerun-viewer` | Old checkout or image override | Use current supported pins and rerun `preflight-images`; retain the failing check. The original fix was PR #374. |
| Private-registry `403` when expecting GHCR | Explicit image/registry overrides or an older client | Use the current default public mirror and remove unintended overrides. Current runtime image selection does not inherit `NPA_REGISTRY`. |
| `No NPA cluster identity exists ... refusing controller adoption` | Cluster fetched but not adopted | Complete S5 with the exact cluster name, project alias, and context. |
| `No shared controller owner is bound` | Missing controller binding | Run `npa skypilot bind-controller` as shown in S5. |
| SkyPilot requires Python 3.9–3.12 | Interpreter used by the isolated runtime | Recreate the isolated SkyPilot environment with Python 3.12; see [SkyPilot setup](../../orchestration/skypilot-setup.md). |
| `ClusterOwnerIdentityMismatchError` | Reused local/controller state from another cluster | Verify the project and context, then follow [controller setup](../../orchestration/skypilot-setup.md#managed-jobs-controller). Preserve unrelated projects' state. |
| `StorageBucketGetError` or inaccessible bucket | Selected project's keys, bucket, region, and endpoint | Recheck S2/S7 and restart the local SkyPilot API server if its profile changed. |
| `FAILED_SETUP: Forced include not found` | Incomplete manually staged source | Use current automatic source staging. For a persisted bad source URI, follow the [source-staging recovery guide](physical-ai-data-factory-deploy.md#if-submit-fails). |
| `--run-id` and `--resume-run` are mutually exclusive | Recovery command combines both options | Use only `--resume-run` for the existing run, or `prepare-run` for a fresh experiment. |
| Workflow GPU discovery says `Kubeconfig not found` | Missing NPA-managed kubeconfig | Complete S5/S6 and verify the selected context. |
| Stage repeatedly recreates; `container not found` during setup | Image cannot satisfy SkyPilot bootstrap | Cancel the affected run using the [run lifecycle](../../run-lifecycle.md) and correct its image. Keep image preflight enabled. |
| GPU stage recovers repeatedly; events show `Evicted`, `ephemeral-storage`, or `NodeHasDiskPressure` | GPU-node disk capacity for image layers and runtime weights | Ask your cohort operator to inspect disk pressure. The historical run needed a 185 GB boot disk. |
| Token Factory returns `404` / model does not exist | Hosted model availability | List models again and set `caption_model` to an available vision model. |
| `source_motion_weight must be 0` | Configuration copied from the old guide | Remove the old override or set it to `0.0`; current publication preserves unmodified model output. |

For pod-level and artifact triage, see
[known Workbench issues](../troubleshooting/known-footguns.md).

## Historical validation results

The source document reported the following timings on September 3, 2026, over
three runs whose tasks through quality evidence succeeded. These measurements
describe that hardware, image set, model, and configuration; they are not a
deadline or an estimate established by this Markdown conversion.

| Stage | Reported duration |
| --- | --- |
| `prepare-input` | About 10 seconds |
| `generate-configs` | About 45 seconds |
| `annotate-original` (hosted VLM) | About 1 minute |
| `generate-variants` (RTX PRO 6000; two variants, 24 steps) | About 18 minutes; no restarts |
| `evaluate` (Cosmos Evaluator) | 2–11 minutes |
| Quality gate, disposition, and route | About 45 seconds each |
| `visualize-quality-evidence` (Rerun) | About 2.5 minutes |
| Submit through quality verdict | About 40 minutes |

### Quality rejection in the original runs

The original tests ended at `require-accepted-quality`: their per-variant
checks did not accept Cosmos3-Nano's appearance edits. Curation and finalization
therefore did not run. Reported tuning observations were:

| Historical settings | Reported score | Observation |
| --- | --- | --- |
| `source_motion_weight=0.8` | 0.36–0.44 | Subtle variants; requested attributes could not be verified |
| `source_motion_weight=0.4`, `steps=32` | 0.16 | Stronger changes with poor temporal consistency |

These nonzero motion weights are **historical only**. They blended source and
generated frames, which can produce double exposure when objects move. Current
code requires `source_motion_weight=0.0` and publishes the model's output bytes
unchanged. The old measurements do not prove that every current configuration
will fail or that calibration alone explains every rejection. Inspect the
actual evaluator report and Rerun evidence before accepting generated data.
See [double exposure in older runs](paidf-cosmos3.md#double-exposure-in-older-runs).

### What changed from the earlier participant setup

- Supported images come from the anonymous public GHCR mirror.
- `npa configure --provision` handles project storage setup.
- The Cosmos 3 access check identifies the relevant model repositories.
- Submit prepares a checksum-verified starter video for the first run.
- Submit stages NPA source automatically and persists it for recovery.
- The supplied GPU and CPU nodes avoid provisioning a new cluster during this
  guide; scheduling and disk capacity still need verification.

The participant document attributed its image/bootstrap fixes to
[PR #374](https://github.com/nebius/nebius-physical-ai/pull/374).

## Inspect the outputs

List the actual run artifacts and locate `reports/sim2real.rrd`:

```bash
npa workbench workflow artifacts "$RUN_ID" --project "$PROJECT_ALIAS"
```

To open it locally, use the recording's exact S3 URI from the artifact listing:

```bash
export RRD_URI='s3://<your-bucket>/<your-run-prefix>/reports/sim2real.rrd'
aws s3 cp "$RRD_URI" ./sim2real.rrd --profile nebius
rerun sim2real.rrd
```

The NPA installation includes its pinned Rerun SDK. The recording presents the
available source/generated frames, generation prompts, captions, evaluator
decisions, and pipeline evidence. A rejected run may contain a useful recording
without accepted data or a final curation report. Check the variant MP4s and
the evaluator's individual dispositions as well as its aggregate score.

The original Google document referred to separately shared screenshots, a
sample video, and a Rerun recording. Those media are not bundled with this guide;
the commands above retrieve evidence from your own run. For browser viewing or
sharing, see [Rerun sharing](../rerun-sharing.md).

For help with an invitation or the supplied cluster, contact your Living Lab
cohort channel. For the workflow itself, continue with the
[Cosmos 3 reference guide](paidf-cosmos3.md) and
[workflow lifecycle](../../run-lifecycle.md).
