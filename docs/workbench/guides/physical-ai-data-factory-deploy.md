# Deploy the Physical AI Data Factory (from zero)

A step-by-step, copy-paste runbook that takes you from an empty machine to a
running **NVIDIA Physical AI Data Factory** blueprint on Nebius + SkyPilot, with
results viewable in the NPA agent (Main stages, embedded Rerun, and an explicitly
provenanced dataset view backed by real FiftyOne Brain curation). It complements the conceptual
[Physical AI Data Factory guide](physical-ai-data-factory.md) (blueprint mapping,
stage graph, S3 layout) — read that first if you want the "what/why"; this doc is
the "how to stand it up".

The blueprint is a single `npa.workflow/v0.0.1` spec, promoted to the top of the
workflow tree for prominence:
[`npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml`](../../../npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml).
SkyPilot is the only orchestrator; there is no OSMO and no bespoke "data factory"
tool — every stage is an existing workbench tool or a real `run.shell` step.

> **No IDs are hardcoded here.** Everywhere you see `<...>` substitute your own
> project/tenant/registry/bucket. Credentials live in `~/.npa/credentials.yaml`;
> machine-managed config in `~/.npa/config.yaml`.

---

## Run PAIDF with a coding agent

First [authenticate the Nebius CLI](https://docs.nebius.com/cli/configure).
Create a [Token Factory key](https://docs.tokenfactory.nebius.com/quickstart)
and a [Hugging Face read token](https://huggingface.co/docs/hub/en/security-tokens)
whose account has accepted the
[Cosmos Transfer 2.5 terms](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B).
Save each value alone in a `chmod 600` file outside the repository; give the
agent file paths, never secret values. NGC is not required when using the public
GHCR images in this runbook.

Copy this prompt into your coding agent from the repository root:

```text
Use README.md and its linked PAIDF runbook to complete this task. Do not change
repository files.

The Nebius CLI is already authenticated. Configure NPA, deploy and verify the NPA
agent, provision one CPU node and one on-demand RTX PRO 6000 GPU node (use
preemptible only if needed), then run
`npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml`
end to end with its verified real RoboPro starter input and one augmentation.

Tenant: <tenant-id>
Project: <project-id>
Region: <region>
Token Factory key file: </absolute/path/token-factory-key>
Hugging Face token file: </absolute/path/hf-token>

Never print secrets or copy them into the repository, logs, or shell history.
Use NPA commands, continue autonomously, and report blockers and usability gaps
without fixing code. Report the agent URL, PAIDF run ID, manifest URI, and all
stage results. Leave resources running and show the NPA-only cleanup commands;
do not clean up until I ask.
```

## Quick start (copy-paste)

This is the complete clean-machine path to an input-conditioned demo run. It uses the
exact shipped spec, the public GHCR mirror selected after configured environment
loading, and the endpoint and secrets already stored by `npa
configure`. No stored secret is printed or manually exported. The first run
needs no user dataset: submit fetches and verifies the pinned RoboPro physical
robot capture, derives the exact conditioning clip/frames, caches it, and stages
it under the run prefix before any automatic GPU provisioning.

```bash
set -eu
set -o pipefail
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e npa

# Interactive once per machine/project. This stores project, bucket, endpoint,
# S3 keys, Token Factory key, and optional HF/NGC tokens under ~/.npa/.
npa configure
eval "$(npa configure --show --env)"   # emits non-secret NPA_* assignments only
# Force the public mirror after eval; configure --env may restore a saved
# project registry.
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai

SPEC=npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml
PROJECT="$NPA_PROJECT_ALIAS"
BUCKET="$NPA_BUCKET"
REGISTRY="$NPA_REGISTRY"
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"

npa workbench health preflight
npa workbench health access --capability paidf
npa provision-if-absent --project "$PROJECT" \
  --cpu-nodes 1 --cpu-platform cpu-d3 --cpu-preset 8vcpu-32gb \
  --gpu-nodes 1 --gpu-platform gpu-rtx6000 \
  --gpu-preset 1gpu-24vcpu-218gb --on-demand
npa skypilot bootstrap

# Reload the kube context written by provision-if-absent, discover its actual
# accelerator spelling, and validate/plan with the real bucket.
eval "$(npa configure --show --env)"
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
REGISTRY="$NPA_REGISTRY"
KUBE_CONTEXT="$NPA_KUBE_CONTEXT"
npa workbench workflow gpus --context "$KUBE_CONTEXT" --spec "$SPEC"
npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID" \
  --assume-decision promote_checkpoint \
  --var bucket="$BUCKET" \
  --var n_augmentations=1 --json

# Proves manifest pulls. GHCR is anonymous; for a private Nebius registry,
# submit also refreshes the Kubernetes imagePullSecret before launch.
npa workbench workflow preflight-images "$SPEC" \
  --project "$PROJECT" --registry "$REGISTRY"

# Source staging is automatic and content-addressed. Each secret name resolves
# from the environment or project store; a missing gate fails before GPU launch.
npa workbench workflow submit "$SPEC" \
  --project "$PROJECT" --registry "$REGISTRY" \
  --run-id "$RUN_ID" --runtime --auto-load \
  --var bucket="$BUCKET" \
  --var n_augmentations=1 \
  --assume-decision promote_checkpoint \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN

MANIFEST_URI="s3://$BUCKET/physical-ai-data-factory/$RUN_ID/npa-workflow/manifest.json"
# Normal NPA-only status lookup:
npa workbench workflow status "$RUN_ID" --project "$PROJECT" --watch
# Explicit fallback if this shell cannot resolve the project storage location:
npa workbench workflow status "$RUN_ID" --project "$PROJECT" \
  --workflow-s3-uri "${MANIFEST_URI%/manifest.json}"
npa workbench workflow logs "$MANIFEST_URI" --project "$PROJECT" --stage augment
npa workbench workflow load-artifact "$RUN_ID" --project "$PROJECT" # idempotent retry only
```

The sequence has five fail-fast gates:

| Gate | Success |
| --- | --- |
| Credentials | S3 and Token Factory checks pass |
| Model terms | Cosmos Transfer reports `HF access ok` |
| Cluster | one CPU node fits the controller plus a PAIDF CPU stage; one GPU node fits Transfer |
| Images | every manifest is pullable; private Nebius credentials refresh `npa-nebius-registry` |
| Submit secrets | Token Factory, S3, and `HF_TOKEN` are forwarded without entering YAML |

Stop at the first nonzero command; it prints the remedy. Detailed recovery is
below.

For a pre-authenticated service-account/federation profile with known IDs, the
prompt-free equivalent is:

```bash
npa configure --no-interactive \
  --save-env-credentials \
  --tenant-id "$TENANT_ID" --project-id "$PROJECT_ID" \
  --region "$REGION" --project-alias "$PROJECT_ALIAS"
```

Only non-secret IDs are arguments. Keep all credential material in the active
Nebius profile and `~/.npa/credentials.yaml`, never shell history.

The configured S3 endpoint is selected automatically; `--s3-endpoint` is only an
explicit override. `NPA_REGISTRY` has the same precedence in `preflight-images`
and submit: an explicit `--registry` wins, then `NPA_REGISTRY`, then the selected
project's registry. Set `REGISTRY=ghcr.io/nebius/nebius-physical-ai` explicitly
to choose the public anonymous mirror even when the project has a private one.
The quick start requests one real augmentation variant for a decisive first run;
omit `--var n_augmentations=1` to use the spec's default two-variant multiply, or
raise it together with the requested GPU count for a larger batch.

Observed planning ranges—not SLAs—are 8–25 minutes for first cluster
provisioning/readiness when capacity is available, tens of seconds for config
generation, 1–3 minutes for each Token Factory caption pass, 10–25 minutes for
one warm Cosmos Transfer inference (initial checkpoint/image preparation can add
10–30 minutes), 3–12 minutes for Cosmos Curator plus FiftyOne startup/curation,
and 1–5 minutes for evaluation/visualization/finalization. Controller retries
normally back off for roughly 30 seconds to several minutes. A warm end-to-end
one-variant run is commonly 25–60 minutes; a cold or capacity-constrained run can
take 60–120+ minutes. These ranges vary with input size, image/cache warmth,
capacity, quota, and retry count. The commands do not impose a workflow deadline.

Normal progress is a new exact stage/attempt in the runtime ledger, then a
scheduler observation and (only when the task reports progress) a heartbeat.
Inspect it with `npa workbench workflow status "$RUN_ID" --project "$PROJECT"
--json`; pending reason codes identify accelerator/capacity, image-pull, storage,
init/crash, or controller backoff. `manifest_state: available` means only that
the manifest was read; `stage_ledger_state`, cached log state, and live log state
are separate. During a quiet period use the NPA status command and `npa workbench
workflow logs "$MANIFEST_URI" --stage augment --follow` in another terminal.

`VERIFICATION_UNAVAILABLE` is a failed current observation, not a terminal
workflow outcome: it retains the labeled last-known state and unchanged/stale
heartbeat, advances the verification-attempt time, and exits nonzero. After DNS,
RBAC, authentication, or controller recovery, rerun status; resume work only with
`npa workbench workflow submit "$SPEC" --project "$PROJECT" --resume-run
"$RUN_ID" --runtime ...`. `--cached` is an explicit offline inspection and is
never automation-trustworthy.

The four similarly named identities are distinct: the Nebius CLI profile supplies
provider authentication; the NPA project alias selects a saved project ID and
credentials; the Kubernetes context selects the exact cluster; and the SkyPilot
jobs controller executes managed jobs. In Kubernetes-controller mode, failure to
create an optional SkyPilot Nebius provider profile is reported as a warning only
when the selected Kubernetes path is valid. It remains fatal when no valid
execution path exists.

SkyPilot may temporarily render not-yet-submitted downstream DAG rows with the
first task's CPU summary. Its human-readable queue can also show misleading
relative ages for newly started downstream tasks because those rows inherit the
managed job's submission timestamp. NPA does not rewrite SkyPilot's queue clock;
the durable manifest and stage status timestamps are the authoritative per-stage
times. The rendered augment task and NPA manifest/status retain the exact
submit-time accelerator (for example
`RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`).

With no input selector, submit uses the verified real RoboPro starter described
in the [PAIDF guide](physical-ai-data-factory.md#starter-input-authenticity-licensing-and-replacement).
It prints whether the checksum-verified cache was hit or fetched and stages the
source plus its derived conditioning artifacts at
`s3://$BUCKET/physical-ai-data-factory/$RUN_ID/input/`.

Not sure what is still missing? `submit` checks first and prints everything at
once:

```text
Error: Cannot submit physical-ai-data-factory.yaml: missing prerequisites:
  - SkyPilot CLI is not usable (...)
      fix: run `npa skypilot bootstrap` ...
  - config.bucket is the spec placeholder 'example-bucket'
      fix: pass --var bucket=<your-bucket>
```

Add `--plan-only` to render the SkyPilot YAML without launching. Do not bypass
preflight on a first run: it is what keeps registry/auth/config failures local.

Replace the starter with a local clip or one S3 object by adding exactly one
selector to the complete submit command above:

```bash
# Local H.264 MP4 (NPA verifies and stages it)
--input-video ./my-capture.mp4

# One existing S3 object (not a prefix)
--input-uri s3://source-bucket/captures/my-capture.mp4

# Developers/tests only: geometric synthetic frames, explicitly labeled
--seed-fixture
```

The selectors conflict by design. Local and S3 replacements are labeled
“User-supplied input”; NPA does not claim they are captured or assign a media
license. It validates MP4/H.264, positive dimensions/duration, normalizes the
source to exactly 93 frames, extracts eight caption frames, records all digests
and lineage in `input/provenance.json`, and invokes Cosmos with mandatory
`--condition-on-input`. `--seed-fixture` is never selected silently.

### If submit fails

| Symptom | Cause | Fix |
| --- | --- | --- |
| `staged source verification failed: ...` | a supplied or persisted source URI is missing, inaccessible, or incomplete | Read the preserved provider error, then safely restage with `npa workbench workflow stage-src --bucket <b>` and explicitly retry with `--resume-run <id>`. The command persists the replacement URI; it never stores S3 secrets. |
| `missing prerequisites: ... SkyPilot CLI is not usable` | SkyPilot never bootstrapped, or only exported in a previous shell | `npa skypilot bootstrap` (persists `skypilot.sky_bin`) |
| `missing prerequisites: ... config.bucket is the spec placeholder` | submitting against `example-bucket` | `--var bucket=<your-bucket>` |
| `controller health check failed: ... kubeconfig ... No such file` | a cached `sky-jobs-controller-*` from another setup points at a kubeconfig that is gone | inspect with `npa skypilot status`, then (after all workflows are terminal) run `npa skypilot cleanup-controller --yes`; provision/point at a real cluster (`npa provision-if-absent`), and pass `--infra k8s/<context>` |
| `managed-job launch indeterminate` | the launch may have reached SkyPilot, but exact structured queue reconciliation is unavailable or ambiguous | do not retry with a new ID and do not cancel by name. Restore the selected context/SkyPilot queue access, then use the printed `--resume-run <same-id>` action; NPA adopts an exact existing job or relaunches only after authoritative absence |
| `recovery_deadline_exhausted_verified_absent` | Kubernetes controller creation remained transiently unavailable through the bounded recovery deadline, and structured reconciliation proved no exact job exists | wait for the selected API to stabilize, then `--resume-run <same-id>`; the recorded logical identity prevents a duplicate |
| first-party bootstrap attestation missing/mismatched | the selected digest was not built for the pinned SkyPilot contract | rebuild with the required packages, declared user/sudo and forwarding entrypoint; publish a unique validation tag and submit its resolved digest |
| status reports `EVIDENCE_INCONSISTENT` | exact current-ledger, S3, or immutable-job evidence conflicts | preserve the run and inspect the reported identities; do not force `NOT_SUBMITTED` or rerun succeeded waves |
| `Kube context '<name>' ... is not available` | no cluster for that context: neither your kubeconfig nor `~/.npa/clusters/<name>/` has it | provision one (`npa provision-if-absent --project <alias>`, and read its warnings — it now exits non-zero when it could not) or point `KUBECONFIG` at the cluster you want; `kubectl config get-contexts` lists what is resolvable |
| A cluster is RUNNING in the console but npa has no kubeconfig for it (interrupted provision) | `up` writes the kubeconfig only after apply finishes | `npa cluster kubeconfig --cluster-name <name> --project <alias>` adopts it (writes the kubeconfig + cluster state), or `npa cluster up` again to resume, or `npa cluster down --force` to remove it |
| `blocked` quota rows before apply | one or more exact hard quotas (instance, disk count, `compute.disk.size.network-ssd` bytes, public IP, or GPU) cannot cover the cumulative topology | read each row's exact `required`, `available`, and `shortfall` (disk capacity is also rendered in GiB), reduce the topology, or ask the tenant operator to resolve the named allowance; the default cluster needs 1,151 GiB (128 + 1,023), and the README agent+cluster path needs 1,251 GiB; preemptible nodes consume exactly the same disk bytes |
| `Nebius refused node group ...` mid-apply | the provider changed after the green preflight or rejected a create | NPA rolls back only this operation's newly created Terraform stack. If the journal says `rollback-incomplete`, use its exact recovery command; pre-existing clusters/storage/credentials are preserved. |
| `npa cluster status` reports `DEGRADED` with `provider_state: RUNNING` and a non-ready node group | the control plane is up while that node group was never provisioned | the same quota/capacity fix; the cluster bills while it exists, so tear it down (`npa cluster down --force`) if you cannot get the nodes |
| `npa cluster status` reports `VERIFICATION_UNAVAILABLE` and a DNS/RBAC/auth code | the configured cluster's current provider/API state could not be verified | run the printed NPA retry command after fixing the typed cause; `npa cluster status --cached` is an explicit last-known-only view, not evidence the cluster is healthy |
| `Context <name> not found ... Available contexts: []` (from SkyPilot) | an older npa left `KUBECONFIG` unset for a cluster it had provisioned | upgrade npa: `submit --infra k8s/<context>` now prepends `~/.npa/clusters/<context>/kubeconfig` itself. For `kubectl`/bare `sky`, `export KUBECONFIG=~/.npa/clusters/<context>/kubeconfig` |
| `provision-if-absent` failed on `~/.ssh/id_rsa.pub` | old default node-group key path | upgrade npa: `cluster up` now pins the first key that exists (`NPA_SSH_PUBLIC_KEY`, `id_ed25519.pub`, `id_rsa.pub`, `id_ecdsa.pub`) |
| `offline PAIDF cache miss` | `NPA_PAIDF_OFFLINE=1` forbids the starter fetch and the pinned asset is not cached | unset offline mode for one verified fetch, or populate the printed cache path with the exact pinned bytes |
| `SHA-256 mismatch` | cached/downloaded/staged bytes do not match the committed source | do not bypass integrity; remove a corrupt cache entry or use a new run ID after fixing the source |
| `unsupported video container/codec` | replacement media is not a decodable H.264 MP4 | transcode as the message shows before submitting; validation occurs before automatic provisioning |
| `manifest_state: pending` with `resolution_source: durable_submission_receipt`, `canonical_paidf_s3_prefix`, or `managed_job` | the exact run exists, but artifact publication has not produced its final workflow manifest yet | keep using the NPA status/log/artifact commands; do not resubmit merely to make the manifest appear |
| A stage remains `PENDING` | the exact scheduler/pod/event reason may be accelerator/capacity, image pull, storage, init/crash, or backoff | run `npa workbench workflow status <run> --project <alias> --json`, then its stage `log_command`; if diagnostics are unavailable, fix the reported DNS/RBAC/controller cause rather than guessing |
| `status: VERIFICATION_UNAVAILABLE` | an S3/provider/auth/SkyPilot check failed, so absence cannot be established | fix the reported source; if project storage selection is the problem, retry with `--workflow-s3-uri s3://<bucket>/physical-ai-data-factory/<run>/npa-workflow` |
| `status: NOT_FOUND` with every applicable source listed as checked/absent | no receipt, exact canonical PAIDF object, exact managed job, or ordinary workflow manifest exists for that ID | verify the project alias/run ID; cancellation remains an idempotent no-op for this conclusively absent run |
| cancel reports `NOT_SUBMITTED` | the owner-only durable ledger is still planned/reserved/staged and contains no workflow, stage, controller, or job launch identity | no provider dependency is required; the repeat-safe cancellation no-op exits 0 |
| cancel reports `VERIFICATION_UNAVAILABLE` after local S3/SkyPilot removal | durable evidence says submission began, but no terminal receipt exists and the exact provider dependency is unavailable | restore the receipt-recorded provider context/dependency and retry; missing local tools are never treated as proof of absence |
| provider package does not match lock checksums | the tracked lock lacks/cannot verify this operator package or a registry mirror/cache is inconsistent | upgrade NPA first. Maintainers regenerate with `terraform providers lock` for the recorded Linux/macOS platforms and review the lock diff; never delete the lock or bypass checksums |
| agent setup reaches `access-key list` after configure already reports writable S3 | stale NPA version is redundantly reprovisioning storage credentials | upgrade NPA: setup/preflight now share the deployment credential decision and reuse the health-verified configured key without listing/creating/rotating access keys |

---

## 0. Prerequisites

- **Python 3.10+** and `git` on an operator/dev machine (Linux or macOS). All
  `npa`, `nebius`, and `docker` commands below run on this operator machine — the
  agent VM only receives staged env files.
- A **Nebius** tenant/project with access to: Managed Kubernetes (GPU nodes),
  Container Registry, and Object Storage (S3-compatible).
- A **Token Factory** API key (hosted VLM/LLM inference; zero-GPU stages).
- Optional: a **Hugging Face** token if you stage gated inputs/checkpoints.
- For the GPU augment stage: at least one GPU node. This blueprint is validated
  on **RTX PRO 6000 Blackwell** (`RTXPRO6000`); any SkyPilot-addressable GPU
  works if you remap the accelerator (see §6).

---

## 1. Install `npa`

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai

# A dedicated venv keeps the CLI isolated. Same path as the README.
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa

npa --version
```

`pip install -e npa` installs the `npa` CLI/SDK in editable mode. Re-run it after
pulling changes that touch dependencies.

> Repo-root `.venv` is the path every user-facing doc uses. Contributor tooling
> (`docs/testing/e2e.md`, the agent skills) uses `npa/.venv` for repo validation;
> either works, but stay consistent within a checkout.

---

## 2. Credentials and config

`npa` reads two files under `~/.npa/`. Create them with `npa configure` (or hand
-write them). **Never commit these; they hold secrets.**

### 2a. `~/.npa/credentials.yaml` (secrets)

S3/object-storage keys, the Token Factory key, and optional HF token:

```yaml
storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  endpoint_url: https://storage.<region>.nebius.cloud   # e.g. us-central1
  bucket: s3://<your-artifact-bucket>/
tokens:
  NEBIUS_TOKEN_FACTORY_KEY: <your-token-factory-key>
  HF_TOKEN: <your-hf-token>         # optional
ngc:
  api_key: <your-ngc-api-key>       # optional
```

This is exactly the layout `npa configure` writes — run `npa configure --show`
to print it. The service-shaped aliases `token_factory: {api_key: ...}` and
`huggingface: {token: ...}` are also accepted, but `tokens:` is canonical and
wins when both are present.

Explicit environment variables are optional overrides when scripting. They take
precedence over configured values, but are not required for the quick start:

```bash
export AWS_ACCESS_KEY_ID=<...>
export AWS_SECRET_ACCESS_KEY=<...>
export AWS_ENDPOINT_URL=https://storage.<region>.nebius.cloud
export NEBIUS_TOKEN_FACTORY_KEY=<...>
export HF_TOKEN=<...>                # optional
```

### 2b. `~/.npa/config.yaml` (machine-managed project config)

Project/tenant/registry/cluster metadata (IDs are yours; nothing is baked into
the repo):

```yaml
default_project: <alias>            # e.g. rtxpro
projects:
  <alias>:
    project_id: <nebius-project-id>
    tenant_id: <nebius-tenant-id>
    region: <region>                # e.g. us-central1
    registry_id: <container-registry-id>
    storage:
      checkpoint_bucket: s3://<your-artifact-bucket>/checkpoints/
      endpoint_url: https://storage.<region>.nebius.cloud
    kubernetes:
      kubeconfig: ~/.npa/clusters/<cluster-name>/kubeconfig
      gpu_profile: rtxpro           # optional; profile name your cluster advertises
```

Inspect the merged view any time:

```bash
npa configure --show
```

### 2c. Provision missing infra (optional)

If the bucket / cluster do not exist yet, `provision-if-absent` creates only what
is missing (dry-run first):

```bash
npa provision-if-absent --project <alias> --dry-run --output-format json
npa provision-if-absent --project <alias> \
  --cpu-nodes 1 --cpu-platform cpu-d3 --cpu-preset 8vcpu-32gb \
  --gpu-nodes 1 --gpu-platform gpu-rtx6000 \
  --gpu-preset 1gpu-24vcpu-218gb --on-demand          # real
```

The default cluster it provisions is the small FTUE shape — **1× GPU node
(`gpu-rtx6000`, `1gpu-24vcpu-218gb`) + 1× CPU node (`cpu-d3`, `8vcpu-32gb`),
Shared Filesystem OFF**. The Data Factory passes stage I/O over S3 URIs, so it
needs **no** Shared Filesystem (and no SFS SSD quota). For a training farm or a
shared `/mnt/data`, opt in via `deploy/cluster` tfvars/flags (see
[`deploy/cluster/README.md`](../../../deploy/cluster/README.md): `gpu_nodes_count`
/ multi-GPU `gpu_nodes_preset` / `enable_gpu_cluster`, and `enable_filestore` or
`existing_filestore`).

The documented path uses on-demand nodes. If that capacity is unavailable,
replace `--on-demand` with `--preemptible`; Nebius may reclaim a preemptible GPU
node mid-stage, so rely on PAIDF's durable S3 manifests and resume the run.

The container registry must be reachable for the workbench images. Point
`NPA_REGISTRY` (or the project `registry_id`) at your registry, e.g.
`cr.<region>.nebius.cloud/<registry-id>`.

---

## 3. Deploy the NPA agent

The agent is a public HTTPS workbench VM (basic-auth UI, grounded chat, embedded
Rerun, artifact browser, Dataset & provenance curation surface). See the
[`npa-agent`](../../../skills/tools/npa-agent/SKILL.md) skill for the full
operator reference; the
[`agent-fresh-operate`](../../../skills/workflows/agent-fresh-operate/SKILL.md)
skill covers teardown/fresh-setup loops.

First-time provisioning (Terraform-backed VM + long-lived `npa-agent` service
account when IAM allows). The simplest path — matching the root README — reads
the project ids/region from `~/.npa/config.yaml`, so you only pick a configured
project:

```bash
# Interactive: pick one of the projects `npa configure` saved, then deploy.
npa agent setup --name <agent-name>
```

For scripted / non-interactive deploys, pass the ids explicitly instead:

```bash
npa agent fresh-setup \
  --project <alias> --name <agent-name> \
  --project-id <nebius-project-id> --tenant-id <nebius-tenant-id> \
  --region <region>
```

Both provision the VM and bake the UI/backend. To re-bake later (e.g. after an
agent code change) without reprovisioning:

```bash
# Bake/refresh the UI + backend + nginx on the VM (~1 min; reuses auth/creds).
npa agent bootstrap --project <alias> --name <agent-name>
```

Before IAM or VM creation, agent deploy probes the exact Terraform backend
object. An absent state key is healthy only when a random conditional sibling in
the same prefix can be created, listed, read, deleted, and verified absent; an
existing state must also be readable Terraform JSON. The same project-scoped S3
HMAC credentials are passed by environment to every later Terraform state
command. Nebius CLI authentication is not a substitute for those HMAC keys.
After a partial first install, `npa agent status --project <alias> --name
<agent-name> --json` shows the durable journal, exact created resources, typed
provider verification, and credential-free recovery argv even when no final
agent config exists.

> **Deploying from behind a VPN/firewall?** `setup`/`fresh-setup` finish by
> SSHing into the new VM to wait for cloud-init. If this machine cannot reach the
> VM's public `tcp/22` (corporate VPN / split-tunnel often block it), the deploy
> times out (~4 minutes, with progress lines) and rolls the VM back with a
> one-line SSH-unreachable error. Deploy from a host with direct network access to
> the VM. `npa agent preflight` and the start of `deploy` probe outbound `tcp/22`
> and warn before a VM is created — against a recorded agent IP when you have one,
> otherwise a public host, which a split tunnel can allow while still dropping a
> fresh Nebius IP. Set `NPA_SSH_EGRESS_PROBE=<host>:<port>` to probe a host your
> network allows, or `off` to skip it.

> **Credential exposure on the agent VM.** The S3 access key and secret staged
> onto the VM are written into its cloud-init user data, which anyone with read
> access to the project can retrieve (`nebius compute instance get`). npa keeps
> those values out of the local process table (secret Terraform variables go
> through a 0600 var-file, not `-var`), but treat keys staged onto an agent VM as
> project-readable and rotate them when you destroy it. Tracked in `FIXME.md`.

> The backend and UI are **baked at bootstrap**. Any change to the agent code
> (`cli/agent.py`, `cli/agent_ui.html`, `cli/agent_chat.py`,
> `cli/agent_workflow.py`, `workflows/artifacts.py`, `workflows/data_factory_*.py`)
> requires re-running `npa agent bootstrap` to take effect on the live VM.

Verify the deploy:

```bash
NPA_AGENT_CHAT_LIVE=1 npa agent verify-live --project <alias> --name <agent-name>
```

### 3a. Auth and first connection

- Auth secrets are written to `~/.npa/agents/<alias>/<agent-name>/auth.env`
  (`AGENT_USER`, `AGENT_PASSWORD`) and are stable across redeploys.
- Public URL: `https://<public-ip>/` (self-signed cert on the VM IP).
- **On a phone:** open `https://<public-ip>/healthz` first to accept the
  self-signed certificate, then sign in at `https://<public-ip>/login-help.html`.
- Never use `localhost`/`127.0.0.1`/`:8080` — always use same-origin `/api/...`.

---

## 4. Choose the public mirror or build into a private registry

Three stages pull a workbench image: `augment` needs `npa-cosmos2-transfer`,
`evaluate` needs `npa-cosmos-evaluator`, and `curate` needs `npa-cosmos-curate`.

The public `ghcr.io/nebius/nebius-physical-ai` mirror is anonymously pullable.
A new private project registry starts empty: `npa configure` selects or creates
it, but does not mirror images into it. Pick one path and preflight the same
registry submit will use:

```bash
REGISTRY=ghcr.io/nebius/nebius-physical-ai    # or the configured NPA_REGISTRY
npa workbench workflow preflight-images npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml \
  --project "$PROJECT" --registry "$REGISTRY"
```

That reports each image as `ok` / `not_found` / `forbidden` and prints the exact
build command for anything missing. `npa workbench workflow submit` runs the same
check before it provisions anything, so a missing image costs no GPU time.
For a validation registry containing distinct images, repeat
`--image-override TOOL_REF=IMAGE` on both commands. Each exact override is
resolved and attested independently and takes precedence over `--image`; pass
immutable digest references in recorded/live acceptance commands.

For the public mirror, an attested `ok` needs no login or build. Reachability
alone is insufficient: preflight resolves the tag once, verifies the bootstrap
contract against that immutable digest, and submits that digest. A historical
tag that predates the contract is rejected even when `docker manifest inspect`
succeeds. For a private registry, build and push what preflight reports missing
or incompatible (tags below track
`npa/src/npa/deploy/images.py`, which is what submit pulls):

```bash
REGISTRY="$(npa configure --show 2>/dev/null | grep -o 'cr\.[^ ]*' | head -1)"   # or your NPA_REGISTRY
printf '%s' "$(nebius iam get-access-token)" \
  | docker login "${REGISTRY%%/*}" -u iam --password-stdin

docker buildx create --name npa-cosmos-oss --driver docker-container   # scoped cache
for tool in cosmos-evaluator cosmos-curate; do
  # Tag must match npa/src/npa/deploy/images.py; submit pulls exactly that tag.
  docker buildx build --builder npa-cosmos-oss --push \
    -f "npa/docker/workbench/$tool/Dockerfile" \
    -t "$REGISTRY/npa-$tool:0.1.2-skypilot-v1-20260813T164700Z" npa
done
docker buildx rm npa-cosmos-oss
```

Neither image carries model weights. The evaluator needs none; the curator's GPU
stages fetch theirs at run time with your Hugging Face token:

```bash
docker run --rm -e HF_TOKEN="$HF_TOKEN" -v curator-weights:/config/models \
  "$REGISTRY/npa-cosmos-curate:0.1.2-skypilot-v1-20260813T164700Z" fetch-models --models split-annotate
```

The loop below is only a registry reachability diagnostic. The mandatory
acceptance check is the `preflight-images` command above; it binds all three
results to immutable digests and refuses a missing, stale, or wrong-digest
bootstrap attestation before spending GPU time:

```bash
for ref in npa-cosmos2-transfer:2.5.1-skypilot-ready-20260801T053000Z \
           npa-cosmos-evaluator:0.1.2-skypilot-v1-20260813T164700Z \
           npa-cosmos-curate:0.1.2-skypilot-v1-20260813T164700Z; do
  docker manifest inspect "$REGISTRY/$ref" >/dev/null && echo "OK   $ref" || echo "MISS $ref"
done

npa workbench workflow preflight-images npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml \
  --project "$PROJECT" --registry "$REGISTRY"
```

---

## 5. Submit the Physical AI Data Factory workflow

The blueprint lives at the promoted top-level path. Validate and plan first:

```bash
SPEC=npa/workflows/workbench/npa-workflows/physical-ai-data-factory.yaml

npa workbench workflow validate-spec "$SPEC" --json
npa workbench workflow plan-spec   "$SPEC" \
  --run-id demo --assume-decision promote_checkpoint --json
```

### 5a. Select the starter input

Prepare one project/workflow-scoped fresh `RUN_ID`. With no input selector, submit fetches
the pinned RoboPro Aloha-Agilex physical capture, verifies its SHA-256, caches it,
and stages `source.mp4`, the exact 93-frame `conditioning.mp4`, eight derived
caption frames, and `provenance.json` under the canonical input prefix.

```bash
BUCKET=<your-artifact-bucket>
RUN_ID="$(npa workbench workflow prepare-run "$SPEC" --project "$PROJECT")"
INPUT="s3://$BUCKET/physical-ai-data-factory/$RUN_ID/input"
```

No extra flag is the production starter path. To replace it, add exactly one of:

```bash
--input-video ./my-capture.mp4
--input-uri s3://source-bucket/captures/my-capture.mp4
--seed-fixture  # developers/tests only: explicitly synthetic geometry
```

Local and S3 replacements must be decodable H.264 MP4s. Kubernetes placement is
checked before input or source staging. Conflicts, missing media, unsupported
codec/container/shape, checksum mismatch, or an unavailable object fail with an
actionable error and never fall back to shapes. `NPA_PAIDF_OFFLINE=1` permits
only a verified cache hit. A committed run input is immutable, so a repeated
submit reuses it and refuses a different source rather than overwriting data.

See the [starter provenance and licensing table](physical-ai-data-factory.md#starter-input-authenticity-licensing-and-replacement)
for the immutable source URL, CC BY 4.0 attribution, exact digest/media facts,
and the separate Cosmos source/model/media license boundaries.

### 5b. Submit a real run

Submit with the **same** `RUN_ID` (dynamic gate → pass `--assume-decision`):

```bash
npa workbench workflow submit "$SPEC" \
  --project "$PROJECT" \
  --run-id "$RUN_ID" \
  --var bucket="$BUCKET" \
  --runtime --auto-load \
  --registry "$REGISTRY" \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN \
  --output-format json
```

That command creates a fresh run. After an interruption or controller recovery,
reuse it only by replacing `--run-id "$RUN_ID"` with the explicit
`--resume-run "$RUN_ID"`; non-interactive operation never infers a resume from a
legacy or aged state file.

For a plan-only preflight without launching a GPU job, add `--plan-only`, and read
back the `image_id` lines to confirm the three images are pinned:

```bash
npa workbench workflow submit "$SPEC" --run-id preflight --plan-only \
  --assume-decision promote_checkpoint --registry "$REGISTRY" \
  --var bucket=<your-artifact-bucket> | grep -E 'image_id|accelerators'
```

> **Do not submit with cleared workbench images.** The submit CLI no longer treats
> `NPA_E2E_CLEAR_WORKBENCH_IMAGES` as a global override; it only clears image pins
> when you pass `--image none` explicitly. The plan-only `image_id` check above
> catches accidental unpinned submits before a run silently skips the real Cosmos
> Transfer, evaluator, or curator images.

Also `unset NEBIUS_IAM_TOKEN NPA_NEBIUS_IAM_TOKEN`: the Nebius provider prefers an
ambient (often expired) token over the fresh CLI one.

---

## 6. Multi-GPU and multi-node fan-out (`RTXPRO6000:N`, `--var augment_nodes=N`)

The `augment` stage runs **one Cosmos Transfer 2.5 diffusion per sampled scenario
variant**. Request `N` GPUs and the stage fans the `N` variants across them (one
variant per GPU), so `N` variants complete in roughly one variant's wall-clock
instead of `N x`.

The variants are independent diffusions, so they also scale across **nodes**:
`--var augment_nodes=N` makes SkyPilot gang-schedule `N` identical augment pods
and the stage shards the sampled combos by `SKYPILOT_NODE_RANK`. Concurrent
renders = `augment_nodes` × GPUs per node. See
[section 6b](#6b-multi-node-augment---var-augment_nodesn) for the artifact
contract.

- Set the number of variants with `config.n_augmentations` (via `--var`, default
  `2`).
- Request `N` GPUs for the `gpu` resource profile. Either edit the spec's
  `resources.gpu.accelerators` to `RTXPRO6000:N`, or override at submit time
  without touching the committed blueprint using the
  `NPA_WORKFLOW_GPU_ACCELERATOR` env var (it fully replaces the accelerator
  string, count included):

```bash
# Fan 4 scenario variants across 4 GPUs on one node.
NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO6000:4 \
npa workbench workflow submit "$SPEC" \
  --run-id "$(date -u +paidf-4gpu-%Y%m%dt%H%M%sz)" \
  --var bucket=<your-artifact-bucket> \
  --var n_augmentations=4 \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN
```

When no valid `NPA_SRC_S3_URI` or image override is supplied, real submit
automatically stages the current content-addressed NPA source and reuses that
verified prefix on retry. Add `--stage-src` only to force a provenance-checked
restage.

If your cluster advertises a different product name for the same GPU (e.g.
`RTXPRO-6000-BLACKWELL-SERVER-EDITION`), pass that exact name:
`NPA_WORKFLOW_GPU_ACCELERATOR=<name>:N`. Variant parallelism is auto-detected from
the GPU count; override with `NPA_COSMOS_VARIANT_PARALLELISM`. Every managed
variant is conditioned on the run's `input/` prefix: a supported video is used
directly, or the required PNG/JPEG frames are assembled into a temporary clip
(preserve input geometry/motion, change only appearance). Missing or inaccessible
input fails closed before inference.

### 6b. Multi-node augment (`--var augment_nodes=N`)

One pod is bounded by the GPUs on a single node. `config.augment_nodes` is the
second axis: the `gpu` profile declares `num_nodes: "{{config.augment_nodes}}"`, so
raising it at submit time gang-schedules that many augment pods without editing the
blueprint. `deployIfAbsent` provisions a cluster with at least that many GPU nodes,
and validation requires `augment_nodes <= n_augmentations`. For an existing
cluster, submit reads the explicitly selected Kubernetes context and requires that
many distinct Ready, schedulable, product-compatible nodes after subtracting active
pod GPU, CPU, memory, init-container, and pod-overhead requests and applying the
profile's node selector/required node affinity plus SkyPilot's node allowlist. An
active unbound GPU pod makes shared placement indeterminate and fails closed. This
is an instantaneous preflight snapshot, not a reservation.

```bash
# 16 variants: 4 nodes x 4 GPUs, all rendering at once.
NPA_WORKFLOW_GPU_ACCELERATOR=RTXPRO6000:4 \
npa workbench workflow submit "$SPEC" \
  --run-id "$(date -u +paidf-4x4-%Y%m%dt%H%M%sz)" \
  --var bucket=<your-artifact-bucket> \
  --var n_augmentations=16 \
  --var augment_nodes=4 \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

How the nodes divide the work and rejoin:

- Every pod runs the same augment command and reads the same
  `configs/manifest.json`. Node `k` of `N` renders variants `k, k+N, k+2N, …`
  (striding, so the nodes stay within one variant of each other) and pins each of
  its own concurrent renders to a local GPU starting at 0.
- Variant indices are global, so clip names (`aug-<run-id>-<i>`) stay disjoint.
  Every scheduler-managed wave attempt, including the one-node default, writes only below
  `cosmos_augmented/_attempts/<attempt-id>/`, and every clip dir there is written
  by exactly one node.
- Each node publishes its attempt-private `manifest-rank-<k>.json`; **rank 0**
  waits for all `N` current-attempt shards and conditionally commits the usual
  `cosmos_augmented/manifest.json`, ordered by variant index, with `node_count` and
  a per-rank `shards` block added. A rank that never reports is a hard failure that
  names it — the canonical run manifest remains `publishing` rather than quietly
  understating the fan-out.
- That join waits as long as the slowest sibling needs. It carries no default
  deadline, because a sibling's remaining work is however long its diffusions take,
  and periodically reports elapsed time plus missing and received ranks. Export
  `NPA_COSMOS_SHARD_JOIN_TIMEOUT_S=<seconds>` to give a live-but-hung sibling an
  explicit deterministic deadline; the failure names the missing ranks.
- The rank-0 attempt-id rendezvous likewise has no arbitrary default deadline and
  reports elapsed wait state. `NPA_COSMOS_IDENTITY_TIMEOUT_S=<seconds>` is the
  separate explicit opt-in bound for a missing leader.
- SkyPilot 0.12.2 deliberately preserves its task id across managed recovery and
  exports no globally ordered recovery counter to the workload. The durable NPA
  runtime therefore issues an ordered `(wave sequence, explicit attempt)` fence
  before each launch. Rank 0 may claim only that token and shares its attempt id
  with the exact ordered members. An inner SkyPilot replacement retains the old
  token and cannot supersede an existing same-token claim; if the prior worker
  failed before claiming, the replacement may safely be the first claimant. After
  that job is terminal, an explicitly configured NPA retry receives a higher token. This prevents an
  escaped old rank 0 from taking over by arriving after the replacement. Final
  publication remains compare-and-swap fenced, and late workers can write only to
  their old private prefix.
- The evaluator, Cosmos Curator, FiftyOne curation/finalize, and Rerun viewer follow
  only variants named by an `executed` canonical manifest. They never enumerate
  `_attempts/`, so retained recovery evidence is not counted as a scenario.
- `augment_nodes=1` (the default) writes no shard files, but it uses the same
  scheduler claim, attempt-private clip prefix, and conditional canonical commit.
  A delayed one-node process therefore cannot overwrite a later grade iteration.

Only `augment` is multi-node. Captioning, grading, curation, and visualization are
CPU/Token-Factory stages that stay a single pod.

### 6c. Choose what the augmentation preserves (`--var augment_control=seg`)

Fan-out decides how many variants render at once; the **control modality** decides
what each variant keeps from the input. Edge, visibility blur, and segmentation
may be derived from the staged clip. Depth requires an operator-owned precomputed
weight-free control via `augment_control_asset_uri`:

| `--var augment_control=` | Preserves | Computed by |
| --- | --- | --- |
| `edge` (default) | every intensity edge, including texture detail | Canny |
| `vis` | coarse layout and colour blocks | bilateral blur |
| `depth` | scene geometry | precomputed permissive weight-free control |
| `seg` | class/instance boundaries only | GroundingDINO-base + SAM2 |

`edge` fights a prompt that restyles a surface, because the old material's texture
edges are part of the control. `seg` keeps a region's shape and motion while
letting the prompt change what it is *made of*:

```bash
npa workbench workflow submit "$SPEC" \
  --run-id "$(date -u +paidf-seg-%Y%m%dt%H%M%sz)" \
  --var bucket=<your-artifact-bucket> \
  --var augment_control=seg \
  --var augment_control_prompt="robot arm, conveyor, bin" \
  --var augment_mask_prompt="robot arm" \
  --assume-decision promote_checkpoint \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

- `augment_control_prompt` names what to segment; upstream otherwise defaults it to
  the first 128 words of the appearance prompt.
- `augment_mask_prompt` adds a **region mask**: SAM2 segments that region and the
  control applies only inside it, so the rest of the frame follows the prompt
  freely. `augment_mask_asset_uri` supplies a precomputed binary spatiotemporal
  mask video instead. The two are mutually exclusive.
- `augment_control_asset_uri` substitutes a precomputed control video (e.g. a
  segmentation map from an earlier pipeline) for the on-the-fly one. A named asset
  that is missing fails the stage instead of reverting to on-the-fly. It is
  mandatory for depth. Video Depth Anything Large/Small weights are outside this
  validated path and are neither downloaded nor executed.
- `augment_control_weight` (default `1.0`) trades control fidelity against prompt
  freedom. Upstream accepts `0.0`–`1.0`; anything outside that fails the submit
  rather than the loaded model.
- Each modality is a separate pinned ControlNet checkpoint. Before provisioning,
  submit verifies the caller-owned HF token can access the selected exact
  `nvidia/Cosmos-Transfer2.5-2B` revision/file; token presence is not consent.
- An unsupported modality fails before the GPU is held. NPA previously rewrote
  anything outside `edge`/`vis` to `edge` silently.

What lands in S3, alongside `cosmos_augmented/<clip>/`:

```
cosmos_control/<clip>/control_<modality>.mp4   # the map that conditioned the variant
cosmos_control/<clip>/control_<modality>/*.png
cosmos_control/<clip>/mask_<modality>.mp4      # only when a region mask was used
cosmos_control/<clip>/mask_<modality>/*.png
```

`cosmos_control/` is a **sibling** of `cosmos_augmented/`, never nested inside it:
`cosmos-evaluator` treats every child directory of the augment prefix as a variant
and falls back to the alphabetically first PNG in one, so a nested control dir
would hand the attribute-verify VLM a segmentation map to grade. Rerun logs these
as `control/<clip>/control_<modality>` beside `augmented/<clip>` on the same
timeline, and the augment `manifest.json` records `control`, `control_weight`,
`control_prompt`, `mask_prompt`, and `control_uris`.

`NPA_COSMOS_CONTROL`, `NPA_COSMOS_CONTROL_WEIGHT`, `NPA_COSMOS_CONTROL_PROMPT`,
`NPA_COSMOS_CONTROL_ASSET`, `NPA_COSMOS_MASK_PROMPT`, and `NPA_COSMOS_MASK_ASSET`
set the same knobs from the submit environment when the argv cannot change; the
renderer forwards them into the augment pod.

---

## 7. Real FiftyOne curation (run in the `npa-fiftyone` image)

The `curate` stage runs **real FiftyOne Brain curation** (uniqueness +
duplicate/near-dup detection + a PCA visualization) over the augmented set. It
uses the `workbench.fiftyone.curate_augmented` toolRef and therefore runs in the
`npa-fiftyone` image, which self-hosts `mongod`. The workflow passes
`--require-fiftyone`: a missing Brain/database runtime fails the stage instead of
publishing a report-only summary under a FiftyOne label.

Run real curation standalone against a completed run's augmented output:

```bash
npa workbench fiftyone curate-augmented \
  --augment-uri s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/cosmos_augmented/ \
  --report-uri  s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/curation/report.json \
  --curator-report-uri s3://<your-artifact-bucket>/physical-ai-data-factory/<run-id>/curation/cosmos_curator.json \
  --require-fiftyone \
  --dedup-threshold 0.10
```

This writes an additive `fiftyone` block into `curation/report.json` (schema stays
`npa.fiftyone.curation.v1`; adds `curation_engine`, `curated_kept/dropped`, and a
`fiftyone.{brain,selection,visualization,samples}` block). This happens
automatically in the shipped PAIDF workflow. Standalone callers may omit
`--require-fiftyone` to request a clearly labeled report-only fallback, but the
agent will state that FiftyOne did not run rather than marketing that fallback as
FiftyOne review. See the [FiftyOne](../../../skills/tools/fiftyone/SKILL.md)
skill and the conceptual guide's curation section.

---

## 8. View results in the NPA agent

The agent discovers runs from its artifact bucket. If the agent's base prefix is
`checkpoints`, the run lands at
`checkpoints/physical-ai-data-factory/<run-id>/`. Open `https://<public-ip>/`,
sign in, then:

- **Main stages:** the run's stage graph (config → annotate → augment → grade →
  re-label → curate → visualize → finalize) with per-stage artifacts. Clicking a
  stage shows inline image/video thumbnails.
- **Rerun:** select the run and load it (or click `reports/sim2real.rrd` in the
  artifact browser) to open the embedded Rerun viewer with input + augmented
  frames and captions.
- **Dataset & provenance:** separate Original/input and Synthetic/augmented
  groups, explicit counts and filters, the input source kind/URI, per-sample
  provenance, uniqueness/kept/near-dupe badges, and the FiftyOne Brain PCA
  scatter. If a legacy run contains only report curation, the panel says
  “FiftyOne did not run” and does not present it as a Voxel51 review.

Useful authenticated JSON endpoints for scripted verification:

```bash
GET  /api/artifacts/runs?prefix=physical-ai-data-factory
GET  /api/artifacts/run/<run-id>?prefix=physical-ai-data-factory
GET  /api/artifacts/provenance/<run-id>
GET  /api/fiftyone/dataset/<run-id>
POST /api/sim-viz/load-run        # {"run_id": "<run-id>"}
POST /api/sim-viz/load-artifact   # {"run_id":"<run-id>","key":"reports/sim2real.rrd","prefix":"physical-ai-data-factory"}
```

---

## 8. Tear everything down

Cleanup has three owners — the agent VM, the cluster, and the storage/IAM that
`npa configure` provisioned. Preview the unified, project-scoped teardown first;
add `--yes` only after reviewing the immutable identities and commands:

```bash
npa destroy --project "$PROJECT" --all --json
npa destroy --project "$PROJECT" --all --yes --json
```

The orchestrator journals every phase, continues phases that are independent of
a failure, and leaves blocked phases with exact recovery commands. It never
silently deletes the Nebius project itself. Project retention is the default.
To remove a project too, review the plan and explicitly use
`--delete-project --yes`; NPA accepts only an exact ID with unique durable
NPA-creation proof and a provider-verified empty child inventory. External,
shared, unproven, nonempty, or unreadable projects fail closed. The equivalent
recovery sequence is:

First, see what exists:

```bash
npa agent list                      # agents recorded under ~/.npa/config.yaml
npa cluster list                    # clusters known locally or in the project
npa storage bucket list --project "$PROJECT"
```

Then remove them:

```bash
# 1. Cancel the workflow. Planned/staged runs that never launched are a
#    successful repeat-safe no-op.
npa workflow cancel <run-id> --project "$PROJECT" --json

# 2. Agent VM, its network, local record, and the IAM the deploy created for it.
npa agent destroy --project "$PROJECT" --name <agent-name> --yes

# 3. Remove the shared jobs controller only after every NPA workflow is terminal.
npa skypilot cleanup-controller --project "$PROJECT" --context <context> --yes

# 4. Cluster. `down` owns everything the Terraform path created — cluster, VPC,
#    subnet — and clears ~/.npa/clusters/<context>/. It reads
#    project/tenant/region from ~/.npa/config.yaml when tfvars omit them.
npa cluster down --terraform-dir deploy/cluster --project "$PROJECT" --force

# 5. Object storage. A versioned bucket cannot be deleted immediately, so this
#    schedules the purge, waits for completion, and drops the dead S3 keys.
npa storage bucket delete --project "$PROJECT" --yes --wait

# 6. Storage IAM. Inspect first. If legacy state lacks ownership provenance,
#    reconcile the exact immutable ID before returning to the guarded delete.
npa storage service-account delete --project "$PROJECT" --dry-run
npa storage service-account reconcile --project "$PROJECT" --id <exact-id> --dry-run
npa storage service-account reconcile --project "$PROJECT" --id <exact-id> \
  --reason '<legacy NPA setup evidence>' --attest-npa-created --yes
npa storage service-account delete --project "$PROJECT" --dry-run
npa storage service-account delete --project "$PROJECT" --yes

# 7. If this validation created a private image registry, delete its exact
#    immutable artifact DAG and registry. For an NPA-created disposable project,
#    remove only its unique provider default topology; either command refuses
#    mixed/shared evidence.
npa registry delete --project "$PROJECT" --project-id <project-id> \
  --tenant-id <tenant-id> --id <registry-id> --name <registry-name> --yes
npa network delete-project-default --project "$PROJECT" \
  --project-id <project-id> --tenant-id <tenant-id> --yes

# 8. Optional: delete only a proven NPA-created, provider-empty project.
npa destroy --project "$PROJECT" --all --delete-project --yes --json

# 9. Remove known shared-service credentials, caches, the SkyPilot venv/state,
#    and empty ~/.npa residue. Non-empty/unrelated local data is preserved.
npa cleanup --full --yes --project "$PROJECT"

# 10. Forget the retained/deleted project's local alias after cloud and local
#     cleanup converge.
npa configure --forget-project "$PROJECT"
```

If project configuration was already removed, take the opaque receipt ID printed
before that rewrite and use the same NPA-only recovery surfaces:

```bash
npa agent destroy --receipt <receipt-id> --name <agent-name> --yes
npa skypilot cleanup-controller --receipt <receipt-id> --context <context> --yes
npa cluster down --receipt <receipt-id> --context <context> --force
npa storage service-account delete --receipt <receipt-id> --id <exact-id> --dry-run
npa workflow cancel <run-id> --receipt <receipt-id> --json
# Optional, after the provider proves every managed child absent:
npa destroy --receipt <receipt-id> --all --delete-project --yes --json
```

Exact flags override receipt fields, and receipt fields override live config;
conflicts stop before action. Selectors are opaque IDs below NPA's receipt root,
not arbitrary paths.

Notes:

- **Which cluster verb:** `npa cluster down` is the complete teardown for a
  cluster `npa cluster up` / `npa provision-if-absent` created. `npa cluster
  destroy --name <cluster>` is the API-only path — it deletes the cluster but
  leaves the Terraform-managed `<cluster>-network` and subnet running — so use it
  for a cluster Terraform does not manage, or to clear local state for a cluster
  that is already gone.
- **Default security groups follow the parent network lifecycle.** Nebius does
  not allow direct deletion of a default security group. The agent/cluster
  teardown commands above delete the parent network only when their Terraform
  state proves NPA owns it; an existing/reused/shared network is preserved for
  its owner. Do not substitute a manual security-group delete loop.
- **Agent IAM is removed by default.** `destroy` deletes the project's
  `npa-agent` service account and access keys once no agent is left in the project
  (other agents share it, so it is kept while any remain). A deploy that rolled
  back still created them, which is why this is the default; `--keep-iam` leaves
  them explicitly for a later NPA agent teardown retry.
- **Storage IAM deletion is ownership-gated.** Configure records provenance only
  when NPA's create call made `lerobot-training`. The storage service-account
  command refuses an ID-only legacy record, a mismatched project, or an account
  configure reused. It also refuses to run while bucket credentials remain, which
  is why bucket deletion comes first. Bucket cleanup never removes IAM evidence:
  before pruning secrets it writes a project-scoped tombstone with immutable
  service-account/access-key IDs, ownership markers, and the creation outcome;
  the dedicated `storage_iam` record also remains until the account is deleted
  or confirmed absent, even if agent bootstrap changes the generic
  `nebius.service_account_id`. Legacy recovery uses the NPA reconciliation
  command above: it verifies exact provider scope, records operator-attested
  non-secret provenance, and still requires the guarded NPA delete path.
- **Do not inventory access keys as raw JSON.** The upstream list response may
  contain secret material. NPA cleanup uses CLI-side JSONPath field selection;
  for safe operator inventory use the filtered example in
  [Known footguns](../troubleshooting/known-footguns.md#raw-access-key-list-json-can-disclose-the-secret).
- **Cluster drain preview is non-interactive and eviction-aware.** `cluster down`
  uses NPA's saved kubeconfig for the selected context, disables browser auth in
  a temporary copy, and explains authentication, RBAC, kubeconfig, and API
  failures as preview-only. One inventory covers all nodes, pods, controllers,
  namespaces, and PDBs, so cilium, CoreDNS/autoscaler, metrics-server, and future
  selector matches are evaluated consistently. On a one-node CPU pool it names
  the missing replacement capacity and expected retry/backoff, then requests a
  normal eviction. Only this explicitly confirmed whole-cluster destroy, after
  exact project/context/provider identity verification, may temporarily remove
  the exact kube-system cilium/CoreDNS/autoscaler/metrics-server PDBs. Their
  specs are snapshotted and restored if destroy aborts while the cluster remains.
  Shared clusters, node-pool operations, unverified contexts, and all
  user/application budgets are never weakened or force-deleted.
- **Controller identity and transaction ordering.** Controller teardown has
  shared blast radius and never inherits the current kube context or first
  SkyPilot profile. It uses the explicit/selected NPA project and exact saved
  context, cross-checks stable cluster/project identity, proves remote controller
  absence, writes a durable checkpoint, and only then removes matching local
  metadata. Any auth/RBAC/connectivity/identity uncertainty preserves local state.
  Before the first submit, bind the global controller owner with `npa skypilot
  bind-controller --project "$PROJECT" --context <context>` or use submit's
  `--bind-controller`. A different project is rejected, and explicit `--rebind`
  still requires a terminal managed-job queue.
- **Shared local runtime is not project residue.** The unified project destroy
  preserves the global SkyPilot venv, Terraform provider cache, credentials, and
  unrelated `~/.sky` state. An explicitly broader standalone cleanup remains a
  separate operator choice.
- **Retained audit evidence is not residue.** Managed jobs are audited and
  receipted before SkyPilot state is removed. Versioned non-secret receipts under
  `~/.npa/teardown-receipts/` survive project/config cleanup, so repeat cleanup
  does not turn verified phases back into unknown. Use `npa cleanup
  --list-receipts`; prune only terminal, aged receipts explicitly.
- **NPA itself is separate.** `npa cleanup` never removes the invoking venv.
  `npa uninstall` previews the exact supported repository-local environment;
  actual deferred removal requires `--remove-environment --yes` and never
  includes source, `.git`, credentials, or unrelated caches.

## 9. Where to go next

- Conceptual blueprint, stage-by-stage mapping, and S3 layout:
  [physical-ai-data-factory.md](physical-ai-data-factory.md).
- Authoring/validating `npa.workflow` specs:
  [`author-npa-workflow`](../../../skills/workflows/author-npa-workflow/SKILL.md).
- Operating the agent VM (chat, Rerun, verify-live):
  [`npa-agent`](../../../skills/tools/npa-agent/SKILL.md).
- Staged Sim2Real operations on Kubernetes:
  [`sim2real-operate`](../../../skills/workflows/sim2real-operate/SKILL.md).
- FiftyOne curation deep-dive:
  [`fiftyone`](../../../skills/tools/fiftyone/SKILL.md).
