# npa Quickstart

This is the platform entry point for Nebius Physical AI. Read this first to
install the `npa` CLI/SDK and configure the single user-authored credential
store. After this page is complete, continue with
[Workbench Getting Started](workbench/getting-started.md) for Kubernetes,
SkyPilot, registry, S3, and first workload setup.

## 1. Platform overview

`npa` is the Nebius Physical AI platform CLI/SDK. It provides a common command
surface for physical AI workflows such as simulation, training, inference,
visualization, dataset conversion, and storage handoff.

Workbench is the first solution namespace on the platform. Workbench tools are
containerized services that run on Nebius infrastructure and exchange data
through S3-compatible object storage. This quickstart stops before
workbench-specific setup so new engineers have one platform setup path and one
clear next document.

For a broader architecture map, see the repository [README](../README.md) and
the package overview in [npa/README.md](../npa/README.md).

## 2. Prerequisites

`npa` runs cloud workloads on Nebius, so a Nebius account and the `nebius` CLI
are required.

- Python 3.10 or newer. The package metadata requires `>=3.10`.
- Git, `python3 -m venv`, and `pip`.
- **macOS**, **Linux**, or **Windows via WSL2** (Ubuntu). `npa` cloud workflows
  (S3 / SkyPilot / Kubernetes) assume a POSIX environment — on Windows run
  everything from WSL2 (see [docs/install.md](install.md)).
- **Required:** a Nebius AI Cloud account with billing enabled. Start with the
  Nebius signup guide: <https://docs.nebius.com/signup-billing/sign-up>.
- **Required:** the Nebius AI Cloud CLI binary on `PATH`. Install it from
  <https://docs.nebius.com/cli/install>; `npa configure` creates or reuses a
  local profile for you (no manual `nebius profile create` step).
- Terraform on `PATH` for later managed `deploy` and `--destroy` commands.
- An SSH public key for later managed VM or BYOVM workbench commands. The
  bundled Terraform defaults to `~/.ssh/id_ed25519.pub`; pass
  `--tf-var ssh_public_key_path=<path>.pub` in deploy commands if you use a
  different key.
- Optional: a Hugging Face token for gated/private Cosmos models, private datasets,
  or higher rate limits. Public GR00T N1.7, GEAR-SONIC, Cosmos Reason1/Nano, and
  NuRec PPISP assets work anonymously:
  <https://huggingface.co/settings/tokens>. Use a read-only token unless a
  workflow explicitly needs write access.
- Optional: an NVIDIA NGC API key for paths that actually pull NGC artifacts
  (currently including the NuRec NRE image):
  <https://ngc.nvidia.com/setup/api-key>.

Quick checks:

```bash
python3 --version
git --version
nebius version
nebius profile list
terraform version
```

## 3. Install npa

`npa` works with **Python 3.10+**. It installs editable from a clone (it is not
on PyPI):

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e npa
```

Verify: `npa --version`.

`npa --version` prints `npa <version>`, and `npa --help` prints the command tree
without requiring Nebius, Hugging Face, NGC, Kubernetes, or S3 credentials.

<details>
<summary>Platform notes</summary>

- **Windows:** use **WSL2 Ubuntu**. `npa` cloud workflows (S3 / SkyPilot /
  Kubernetes) assume a POSIX environment; run everything from WSL2.
- **Debian/Ubuntu:** install the venv module first —
  `sudo apt-get install -y python3-venv`.
- **Need Python 3.10+, or a faster installer?** [`uv`](https://docs.astral.sh/uv/)
  can install Python and create the env:
  `uv venv .venv && source .venv/bin/activate && uv pip install -e npa`.
- **Out of scope (needs extra steps):** Alpine/musl, brand-new Python before
  wheels exist, and air-gapped machines.

Full per-platform steps — Nebius CLI, WSL2 setup, operator tools:
[docs/install.md](install.md).
</details>

If you prefer not to activate the venv, call its interpreter directly with
`./.venv/bin/npa`. The rest of this guide assumes the venv is activated.

The base install is fully capable: a plain `pip install -e npa` already
includes every non-GPU workbench dependency (dataframe/reporting, LanceDB, the
Rerun viewer, and the local eval/agent server). There is no separate
`npa[full]` step. Only these extras are opt-in:

```bash
pip install -e "npa[genesis]"   # Genesis + distillation stages (GPU, local)
pip install -e "npa[groot]"     # GR00T SDK (GPU, local)
pip install -e "npa[sonic]"     # SONIC ONNX export/runtime (GPU, local)
pip install -e "npa[dev]"       # tests, lint (pytest, ruff); see Section 6
```

The GPU wheels above are only needed to run those engines **locally** — cloud
jobs execute them inside the Nebius images they launch. To run cloud workloads
(Sections 5+) the base install is enough; you also need the Nebius CLI —
see [docs/install.md § Nebius CLI](install.md#4-nebius-cli-required).

## 4. Configure credentials

For credential setup, `npa` has one user-authored file:

```text
~/.npa/credentials.yaml
```

Do not choose between multiple NPA credential files. Put user-level secrets in
`~/.npa/credentials.yaml` only. Deploy commands may create or update
`~/.npa/config.yaml` for machine-managed project, workbench, endpoint, SSH,
storage, and Terraform state metadata; do not manually populate
`~/.npa/config.yaml` as part of credential setup.

Environment variables can override file values for a single shell. They are
useful for temporary tests, but the canonical repeatable setup is
`~/.npa/credentials.yaml`. The current source does not read
`NPA_CREDENTIALS_PATH`; it resolves credentials from
`Path.home() / ".npa" / "credentials.yaml"`.

Create and secure the credentials file:

```bash
mkdir -p ~/.npa
chmod 700 ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

### 4a. Nebius account authentication

Nebius account authentication is handled by the `nebius` CLI profile, not by a
long-lived `NEBIUS_TOKEN` in `~/.npa/credentials.yaml`.

Before running `npa configure`, sign up for Nebius AI Cloud, note your tenant ID
and exact `project-...` ID, and create a project in the target region. Supplying
the project ID when NPA creates the CLI profile avoids tenant-wide project
discovery, so profiles authorized through a project-scoped IAM group work
without broader tenant permissions. An Object Storage bucket is
optional. Interactive `npa configure` offers storage provisioning by default.
On a fresh setup, pressing Enter at the bucket prompt generates a collision-safe
name containing a UTC timestamp, a short project-identity hash, and a random
suffix. Enter an exact existing bucket name only when you explicitly intend to
reuse it. New buckets default to **standard** storage and can use `enhanced`
storage or a custom size. To reuse your own bucket, create one first; see
[Creating a tenant](https://docs.nebius.com/iam/create-tenants),
[Manage projects](https://docs.nebius.com/iam/manage-projects), and
[Manage buckets](https://docs.nebius.com/object-storage/buckets/manage).

#### Creating a project from the CLI (tenant administrator)

Creating a project is a privileged action outside NPA. A tenant administrator
(or another principal permitted to create projects under the tenant) can use the
pinned CLI's official `iam v2 project` surface instead of the web console. The
list and get commands are read-only and safe verification steps; the console
path linked above remains equivalent.

```bash
TENANT_ID="tenant-id"
PROJECT_NAME="project-name"
REGION=eu-north1
command -v jq >/dev/null

# Optional read-only parent verification before creating anything.
nebius iam v2 project list --parent-id "$TENANT_ID" --all --format json

# Creates the external project and captures its immutable ID from structured
# output (no parsing of a human table). Review tenant/name/region first.
PROJECT_JSON="$(nebius iam v2 project create --parent-id "$TENANT_ID" \
  --name "$PROJECT_NAME" --region "$REGION" --format json)"
PROJECT_ID="$(printf '%s' "$PROJECT_JSON" | jq -er '.metadata.id')"
export PROJECT_ID
test -n "$PROJECT_ID"

# Read-only identity verification.
nebius iam v2 project get --id "$PROJECT_ID" --format json

# Bind the active CLI profile, then continue with NPA under a local alias.
nebius config set tenant-id "$TENANT_ID"
nebius config set parent-id "$PROJECT_ID"
PROJECT_ALIAS="local-npa-alias"
npa configure --no-interactive --no-provision --tenant-id "$TENANT_ID" \
  --project-id "$PROJECT_ID" --region "$REGION" \
  --project-alias "$PROJECT_ALIAS"
```

#### Federation or SSO profiles with many tenants

If your Nebius CLI profile has no `tenant-id` / `parent-id` set — common for
SSO and federation logins — bind it to the project you want **before**
`npa configure`, so discovery targets the right place instead of listing every
tenant:

```bash
nebius config set tenant-id <id>
nebius config set parent-id <project-id>
```

Say **yes** to the object-storage prompt: the agent VM and the Physical AI Data
Factory both need an S3 bucket and access key.

#### Non-interactive setup

If you already know the IDs, skip browser flow and tenant discovery with
`npa configure --no-interactive` (shown above). This saves local project state
without contacting the provider. A valid non-interactive Nebius profile or
service-account credential is required only when you add explicit
`--provision`.

**No secret-value flags are accepted or shown by `npa configure --help`.**
Automation supplies values through protected environment variables and adds the
boolean `--save-env-credentials`; NPA atomically persists only supported
variables in its owner-only credential store. Prompt-free known-project setup is
project-only by default: it does not contact Nebius, Hugging Face, or NGC,
inspect saved storage, or adopt an old bucket. Add `--provision` when storage
mutation is intended. Credential import remains composable with the
machine-readable view: `npa configure --show --env --save-env-credentials`
sends only non-secret shell assignments to stdout and sends import/access
diagnostics to stderr. The provisioning path reuses already-selected S3
credentials only when exact project provenance and a write/read/delete probe
both verify them; otherwise it generates a fresh collision-safe name without
listing or rotating unrelated access keys.

Run interactive setup in a terminal. `npa configure` creates or reuses your
Nebius CLI profile first. When creating one, it asks for the project ID before
browser authentication. It then prompts for your tenant ID, confirms the project
ID and region, and guides you to reuse an explicitly named exact bucket or create
a freshly named bucket (standard storage, configurable size limit in GB). It also
asks for a local **project alias** (default = region; used later as
`-p <alias>`).

`npa configure` is idempotent: re-run it any time to update keys or properties.
On a re-run every prompt is pre-filled with the value already saved in
`~/.npa/config.yaml` / `~/.npa/credentials.yaml`, so pressing Enter through the
flow keeps your current setup unchanged, and typing a new value updates just
that field. When object storage is already configured it defaults to keeping the
existing bucket and S3 key (so a re-run does not mint a new access key); decline
that prompt to re-provision. It then writes `~/.npa/credentials.yaml` and
`~/.npa/config.yaml`. Provisioning setup performs bounded live checks through
the same Hugging Face model/dataset and NGC repository-entitlement paths as
`npa workbench health access`, and prints a one-line `[NOTE]` summary. The note
is advisory so a transient upstream outage does not undo otherwise valid local
setup; use the health command as the access gate — see
[§4e](#4e-accept-and-verify-gated-model-access):

```bash
npa configure
```

Workbench images resolve from the anonymous
`ghcr.io/nebius/nebius-physical-ai` mirror by default, so configure does not ask
for or save a container registry. Existing `container_registry` entries remain
supported as custom overrides; for new private or locally modified images, set
`NPA_REGISTRY` or pass the command's explicit image/registry option.

Storage is committed after its declared write/read capability probe succeeds.
Delete is best-effort probe cleanup and is reported independently. The declared
runtime actions are `GetObject`, `HeadObject`, `PutObject`, `DeleteObject`, and
`ListObjectsV2`; NPA binds `storage.object-editor` to a project-scoped NPA group
at the exact bucket. Initial auto-provisioning therefore requires the active
profile to have `admin` permission on the target project so it can manage that
project's IAM group, membership, and access permit. Tenant-wide project listing
and tenant-wide `admin` permission are not required. A provider-verified existing
`editors` membership is
accepted for older installations. Creating `editors` is an explicit compatibility
fallback only when Nebius reports the narrow role unsupported. Unknown,
unreadable, or insufficient IAM stops before key creation and probing. Newly
Set `NPA_ALLOW_EDITORS_STORAGE_FALLBACK=1` only for that explicit fallback;
otherwise a provider rejection remains terminal. The custom group name is
derived from the exact project ID rather than its mutable alias. Newly
created or changed bindings receive bounded, typed propagation retries against
the same identity, including when an existing active key is reused. If a
provider step or the probe fails, configure prints **Setup incomplete**, keeps
owner-only creation provenance in `~/.npa/credentials.yaml`, and prints the
restart-safe recovery command:

```bash
npa provision-if-absent --project <PROJECT_ALIAS> --skip-k8s
```

That recovery reconciles storage before any cluster work. It rolls back only
resources conclusively created by the failing invocation; reused, shared, or
ownership-unproven resources are preserved.

Credentials are stored under
`project_credentials.projects.<exact-project-id>` with the alias retained only
as metadata. Atomic locked 0600 writes keep two projects on one host isolated.
The legacy top-level `storage`, `storage_iam`, and `nebius` keys are derived
compatibility views for the selected exact project. NPA migrates an older global
record only when its project ownership is exact and unique; otherwise it leaves
the record recoverable and fails closed.

You do not need to run `nebius profile create` manually; the Nebius CLI binary
must still be installed because `npa` invokes it internally.

With the target IDs known, skip browser login, discovery, and tenant selection
entirely:

```bash
npa configure --no-interactive --no-provision \
  --tenant-id "$YOUR_TENANT_ID" --project-id "$YOUR_PROJECT_ID" \
  --region "$NEBIUS_REGION" --project-alias "$PROJECT_ALIAS"
```

This is provider-free project configuration. Add `--provision` to the same
command only when it should also create or reuse verified writable storage.

For a newly created bucket, automation may also select its create-only storage
class and size cap without putting credentials on the command line:

```bash
npa configure --no-interactive \
  --tenant-id "$YOUR_TENANT_ID" --project-id "$YOUR_PROJECT_ID" \
  --region "$NEBIUS_REGION" --project-alias "$PROJECT_ALIAS" \
  --provision \
  --bucket-storage-class enhanced --bucket-size-gb 100
```

The generated name includes a UTC timestamp and random suffix. In the unlikely
event of an exact collision, NPA reports it and stops without adopting or
changing the old bucket; re-run configure to generate another fresh name.
Create-only class and cap options never alter an existing bucket.

These are non-secret identifiers; do not pass IAM, S3, Token Factory, HF, or NGC
secrets on the command line. The command reuses existing storage only after the
same write/read capability probe deployment uses. Without `--provision`, the
prompt-free command saves only project configuration and supported environment
credentials; `--no-provision` makes that provider-free intent explicit and
reports that HF/NGC access probes were skipped.

Gate: after interactive setup, `nebius iam get-access-token` exits successfully.

`npa agent preflight`, `agent setup`, and `agent fresh-setup` share this exact
storage decision. Once configure has health-verified the bucket/key, agent setup
reuses it and provisions only the separately named VM service account; it does
not list, create, or rotate object-storage access keys. If revalidation is ever
needed, provider output stays field-allowlisted and secret-redacted.

Keep these non-secret values handy for later workbench deploys:

- `<YOUR_PROJECT_ID>`: copy it from the Nebius console project selector or list
  projects with the Nebius CLI.
- `<YOUR_TENANT_ID>`: copy it from the Nebius console tenant selector. The
  Nebius docs also show CLI options:
  <https://docs.nebius.com/iam/get-tenants>.
- `<NEBIUS_REGION>`: the region where the project exists, for example
  `eu-north1`.
- `<PROJECT_ALIAS>`: the local profile key `npa configure` prompts for (default
  = region, for example `eu-north1`). It becomes `default_project` in
  `~/.npa/config.yaml`. Pass it as `-p <PROJECT_ALIAS>` to workbench commands
  (or omit `-p` to use the default).

### 4b. Required credential key names

Use these canonical keys in `~/.npa/credentials.yaml`.

| Need | `credentials.yaml` key | Environment override | Required when |
|---|---|---|---|
| Hugging Face token | `tokens.HF_TOKEN` | `HF_TOKEN` | Downloading gated Hugging Face models, datasets, or weights |
| Nebius Token Factory key | `tokens.NEBIUS_TOKEN_FACTORY_KEY` | `NEBIUS_TOKEN_FACTORY_KEY` | Zero-GPU hosted inference (Token Factory / OpenAI-compatible) paths |
| NGC API key | `ngc.api_key` | `NGC_API_KEY` | Pulling an entitlement-controlled NGC image or model, including NuRec NRE |
| NGC organization | `ngc.org` | `NGC_ORG` | Your NGC key is organization-scoped |
| NGC team | `ngc.team` | `NGC_TEAM` | Your NGC key is team-scoped |
| BYOVM SSH host | `ssh.host` | `NPA_BYOVM_HOST`, `NPA_SSH_HOST` | BYOVM commands need a default host |
| BYOVM SSH user | `ssh.user` | `NPA_BYOVM_SSH_USER`, `NPA_SSH_USER` | BYOVM commands need a default SSH user |
| BYOVM SSH private key | `ssh.key_path` | `NPA_BYOVM_SSH_KEY`, `NPA_SSH_KEY` | BYOVM commands need a default private key |
| Object-storage access key | `storage.aws_access_key_id` | `AWS_ACCESS_KEY_ID` | BYOVM, existing storage, or cross-project S3 workflows need explicit storage credentials |
| Object-storage secret key | `storage.aws_secret_access_key` | `AWS_SECRET_ACCESS_KEY` | BYOVM, existing storage, or cross-project S3 workflows need explicit storage credentials |
| Object-storage endpoint | `storage.endpoint_url` | `AWS_ENDPOINT_URL`, `NEBIUS_S3_ENDPOINT`, `NPA_STORAGE_ENDPOINT` | S3-compatible storage is not supplied by managed project config |
| Object-storage bucket | `storage.bucket` | `NPA_CHECKPOINT_BUCKET`, `NEBIUS_S3_BUCKET` | A workflow needs a default checkpoint or artifact bucket |

**How to create each token** (step-by-step, including where to click):

- Hugging Face — [docs/workbench/huggingface-token.md](workbench/huggingface-token.md)
- NVIDIA NGC — [docs/workbench/ngc-api-key.md](workbench/ngc-api-key.md)
- Nebius Token Factory — [docs/workbench/token-factory-key.md](workbench/token-factory-key.md)

`npa configure` links these inline at each prompt, normalizes pasted values
(stripping stray quotes or a `Bearer` prefix), and warns if a Token Factory key
does not look like a `v1.` key.

When `tokens.HF_TOKEN` is loaded, `npa` forwards it to remote services as both
`HF_TOKEN` and `HUGGING_FACE_HUB_TOKEN`.

`NPA_STORAGE_ENDPOINT` is accepted as a convenience alias. For eu-north1
workbench clusters, use:

```bash
export NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud
```

### 4c. Populate `~/.npa/credentials.yaml`

Use this complete template and delete keys you do not need yet:

```yaml
tokens:
  HF_TOKEN: <YOUR_HUGGING_FACE_TOKEN>

ngc:
  api_key: <YOUR_NGC_API_KEY>
  # org: <YOUR_NGC_ORG>
  # team: <YOUR_NGC_TEAM>

# Optional defaults for BYOVM commands only.
ssh:
  host: <BYOVM_HOST_OR_IP>
  user: ubuntu
  key_path: ~/.ssh/id_ed25519

# Optional shared object-storage credentials for BYOVM or existing storage.
storage:
  aws_access_key_id: <YOUR_S3_ACCESS_KEY_ID>
  aws_secret_access_key: <YOUR_S3_SECRET_ACCESS_KEY>
  endpoint_url: https://storage.<NEBIUS_REGION>.nebius.cloud
  bucket: s3://<YOUR_BUCKET>/<PREFIX>/
```

Omit keys you do not have yet. Do not leave placeholder token values in a file
you plan to use for model downloads.

Secure the file after editing:

```bash
chmod 600 ~/.npa/credentials.yaml
```

### 4d. Cross-project storage workflows

If you submit `npa` workloads from an orchestrator that reads resources from one
project and writes outputs to another, pass explicit project aliases for each
side of the S3 boundary:

```python
from npa import demo

demo.stage(
    source_project="project-a-where-source-artifacts-live",
    target_project="project-b-customer-bucket",
    target_bucket="s3://customer-bucket/demo-artifacts/",
)
```

Each project resolves credentials independently. If a scoped principal is
missing access on either side, `ScopedCredentialError` names the specific
project, operation, and bucket that failed.

For development workflows where host credentials are acceptable:

```bash
npa demo stage --source-project project-a --target-project project-b \
  --target-bucket s3://customer-bucket/demo-artifacts/ --allow-host-creds
```

### 4e. Accept and verify gated model access

Some capability selections pull **gated** Hugging Face models or entitlement-controlled
NGC artifacts. Complete access upstream on the owning account. NPA does not invent a
second acceptance flag: the account token plus a successful repository probe is the
automated access preflight, while the artifact's upstream licence still governs use.

When the selected capability needs them, set `HF_TOKEN` or `NGC_API_KEY` and verify
the exact assets it will use. Public Hugging Face assets are checked anonymously:

```bash
npa workbench health access
# or export keys and persist them without putting secret values in argv:
export HF_TOKEN='<your-token>' NGC_API_KEY='<your-key>'
npa workbench health access --save-env-credentials
# scope to one capability, or run offline (presence-only):
npa workbench health access --capability groot
```

A convenience wrapper is also available:

```bash
HF_TOKEN=hf_xxx NGC_API_KEY=nvapi-xxx scripts/accept-model-access.sh
```

The report is PASS/WARN/FAIL per model or repository; it exits non-zero if a
required gated Hugging Face asset or NGC pull is definitively rejected, so it
fits a CI or cold-start preflight. `npa configure` uses these same live access
probes for its bounded advisory summary, but does not turn an optional missing
credential or transient network failure into a setup failure. For the broader
credential preflight (presence/format plus basic HF, S3, and Token Factory
connectivity), use `npa workbench health preflight`.

## 5. First platform checks

These commands should not provision cloud resources:

```bash
npa --help
npa configure --show
```

Gate: both commands render local CLI output without requiring Kubernetes, S3,
NGC, or Hugging Face network access. Note that a bare `npa configure` in a
terminal is interactive and provisions object storage by default (it creates an
S3 bucket and access key); use `npa configure --show` for a read-only view of
the file layout, or `npa configure --no-provision` for provider-free project and
token setup with storage deliberately unselected.

### 5a. Verify the path works: zero-GPU inference (Nebius Token Factory)

Nebius AI Cloud — GPU clusters, managed Kubernetes, and object storage — is the
main substrate `npa` targets; the flagship GPU workload is Cosmos (Section 7).
Before requesting GPUs, though, you can confirm the whole NPA→Nebius path works
with a zero-GPU smoke test: [Token Factory](https://tokenfactory.nebius.com/)
hosted inference is OpenAI-compatible and needs only a `NEBIUS_TOKEN_FACTORY_KEY`
(no cluster, registry, or S3), so it exercises your credentials and connectivity
without spending GPU-hours. Add the key with `npa configure` (or
`export NEBIUS_TOKEN_FACTORY_KEY=v1...`), then confirm it authenticates:

```bash
npa workbench token-factory verify
```

Generate a completion against a hosted model — write a prompt and run:

```bash
printf 'Explain sim-to-real transfer in one sentence.\n' > /tmp/prompts.txt
npa workbench token-factory generate \
  --input-path /tmp/prompts.txt \
  --output-path /tmp/tf-generations.jsonl \
  --output json
```

Gate: `verify` reports `authenticated: true` with a non-zero model count, and
`generate` writes `/tmp/tf-generations.jsonl` with a completion for the prompt.

More Token Factory capabilities (image captioning, physical-scene reasoning) and
the checked-in SkyPilot templates:
[docs/workbench/token-factory.md](workbench/token-factory.md).

### 5b. The same capability, three coherent ways

Every Workbench capability is usable as an `npa` CLI command, a Python SDK call,
and a parameterizable SkyPilot YAML. The three stay coherent; pick whichever
fits your workflow.

**CLI** (shown above):

```bash
npa workbench token-factory generate --input-path /tmp/prompts.txt \
  --output-path /tmp/tf-generations.jsonl --output json
```

**Python SDK:**

```python
from npa.workbench.token_factory import generate_text

result = generate_text(
    input_path="/tmp/prompts.txt",
    output_path="/tmp/tf-generations.jsonl",
)
print(result.generations[0].completion)
```

**Workflow specs:** the checked-in `npa.workflow/v0.0.1` specs run the same tools on the
cluster through SkyPilot — see `npa/workflows/workbench/npa-workflows/` (for example
`tokenfactory-train-triage.yaml`) and [the workflows guide](workbench-yaml-guide.md).

## 6. Developing and testing npa

To work on `npa` itself, install the dev extra into your activated venv and use
the `make` targets from the repo root:

```bash
pip install -e "npa[dev]"   # pytest, pytest-mock, pytest-cov, pytest-timeout, ruff

make test         # fast default: full unit suite, no live/GPU/network
make test-smoke   # quickest: onboarding CLI smoke tests only
make lint         # ruff
make test-e2e     # opt-in: launches real Nebius infrastructure
```

The `make` targets call `python -m pytest`; pass `PYTHON=...` to target a
specific interpreter (for example `make test PYTHON=./.venv/bin/python`). Live,
GPU, and end-to-end tests are marked (`gpu`, `multi_gpu`, `e2e`, `byovm_live`,
`ngc_e2e`, ...) and are deselected from `make test`, so the default suite never
touches real infrastructure even if your shell has Nebius credentials exported.
See [CONTRIBUTING.md](../CONTRIBUTING.md) for the full test layout and PR
conventions (branch → PR → squash, one approval, never self-approve).

## 7. Flagship GPU workload: NVIDIA Cosmos

With your project configured (Section 4), the headline GPU workload is **NVIDIA
Cosmos** — a world-foundation model for synthetic data and world generation.
Cosmos is the recommended first GPU workload because it runs across **multiple
NVIDIA GPU platforms** (for example `gpu-h100-sxm`, `gpu-h200-sxm`,
`gpu-b300-sxm`, `gpu-l40s`) selected with a single `--gpu-type` flag. It does
**not** require RT cores, so you are not locked to one GPU family the way
RT-core tools are.

This step needs Nebius credentials, a `HF_TOKEN` for the gated Cosmos weights
(Section 4), and GPU capacity. Set GPU routing with one flag and keep the same
command shape across platforms:

```bash
# Deploy a Cosmos serving endpoint on the GPU platform of your choice.
npa workbench cosmos -p <your-project-alias> -n cosmos deploy \
  --runtime serverless \
  --gpu-type <gpu-platform> \
  --gpu-preset <gpu-preset> \
  --wait

# Generate from a text prompt; output lands in your bucket.
npa workbench cosmos -p <your-project-alias> -n cosmos infer \
  --prompt "A robot arm stacks colored cubes on a table" \
  --output-path s3://<your-bucket>/cosmos/out/ \
  --output-format json

npa workbench cosmos -p <your-project-alias> -n cosmos teardown --yes
```

Artifact-bearing end-to-end validation (a real serverless GPU job that writes a
`checkpoint.json` to your bucket) is:

```bash
npa workbench cosmos train --runtime serverless --smoke --gpu-type <gpu-platform>
```

This same serverless job is available three coherent ways:

- **CLI:** the `npa workbench cosmos train --runtime serverless` command above.
- **SDK:** Cosmos serverless jobs are submitted programmatically with
  `npa.clients.serverless.ServerlessClient.create_job(...)` plus the
  `npa.serverless_common` env helpers (the `npa.sdk.workbench.cosmos` namespace
  itself currently exposes `check`/`fetch`). See the worked SDK example in
  [docs/sdk/cosmos-serverless.md](sdk/cosmos-serverless.md).
- **GPU-cluster alternative:** the `npa.workflow/v0.0.1` specs under
  `npa/workflows/workbench/npa-workflows/` (for example
  `cosmos3-text-to-image.yaml`) run Cosmos on a GPU cluster through
  `npa workbench workflow submit`. This is a different runtime from Serverless
  AI Jobs (it provisions a cluster and needs network access to the Cosmos
  framework source + gated weights).

Because this launches a real, potentially long GPU job, run it from a durable
launcher (your job queue / SkyPilot-managed job) rather than an interactive
session you might close. See [the Cosmos guide](../skills/tools/cosmos/SKILL.md)
and [the workflows guide](workbench-yaml-guide.md) for routing, backend
selection, and known limits. Isaac Lab is the simulation counterpart but is
RT-core-only (L40S / RTX Pro 6000); see its guide before choosing GPU type.

## 8. Do more with npa

With install → `npa configure` → your first GPU workload on Nebius AI Cloud
(Section 7) done, build outward:

- **Run workbench workloads** — NVIDIA Cosmos (Section 7), vlm-eval, sim2real,
  and more. See [Workbench Getting Started](workbench/getting-started.md) for
  Kubernetes, SkyPilot, registry, and S3 setup, and the
  [robot guides](workbench/guides/README.md).
- **Deploy the self-hosted agent** — `npa agent` is a browser workbench VM. It
  builds on the setup above and additionally needs Terraform, an SSH key pair, a
  Token Factory key, and writable S3. Operator docs:
  [skills/tools/npa-agent/SKILL.md](../skills/tools/npa-agent/SKILL.md).

Reference:

- [CLI and package overview](../npa/README.md): package-level command and
  development notes.
- [Repository overview](../README.md): project map and current workbench list.
- [Source sample config](../npa/src/npa/config/sample_config.yaml):
  machine-managed `~/.npa/config.yaml` shape for reference only.
- [Known onboarding and runtime gotchas](../FIXME.md): active follow-up list.

## 9. Troubleshooting

`npa: command not found`

Activate the virtualenv (or call its interpreter directly):

```bash
source .venv/bin/activate   # or: source ~/.venvs/npa/bin/activate
npa --help
```

`pytest` fails to collect with `ModuleNotFoundError: No module named 'fastapi'`

The test suite needs the dev tooling (which pulls in the server extra). Install
it into your venv:

```bash
pip install -e "npa[dev]"
make test
```

`aws s3 ls` fails with `Could not connect to the endpoint URL`

Nebius object storage is S3-compatible but is not AWS. Older `aws-cli` (v1)
ignores the `AWS_ENDPOINT_URL` environment variable, so it tries the AWS
endpoint and fails. Pass the endpoint explicitly, or use `aws-cli` v2:

```bash
aws s3 ls --endpoint-url https://storage.<your-region>.nebius.cloud
```

`npa` itself does not depend on this: it reads `storage.endpoint_url` from
`~/.npa/credentials.yaml` (or `AWS_ENDPOINT_URL`/`NPA_STORAGE_ENDPOINT`) and
passes it to the S3 client directly.

Jobs land on the wrong cluster, or `kubectl`/`sky` target the wrong place

Pin your Kubernetes context so submissions are unambiguous:

```bash
kubectl config get-contexts
kubectl config use-context <your-workbench-context>
```

`403`/`denied` when pushing or pulling a container image

Supported NPA images pull anonymously from
`ghcr.io/nebius/nebius-physical-ai`. If an explicit BYOF override points at an
operator-owned registry, authenticate to that exact host using the registry's
documented standards-based mechanism and confirm the same immutable reference
can be pulled from the execution environment.

Capacity, quota, or `Not enough resources` errors

These come from the cloud, not from `npa`. Retry in a few minutes, pick a
different GPU type/region, or request a quota increase in the Nebius console.
Run any command with `NPA_DEBUG=1` for a full traceback.

`source repo is not reachable` from `npa workbench cosmos check`

Cosmos checks the framework source repo with `git ls-remote`, injecting
`GITHUB_TOKEN` as an auth header when that variable is set. A stale or invalid
`GITHUB_TOKEN` makes even a public repo return `401`, so the check reports the
source as unreachable. Clear the bad token (the public clone works
anonymously) or export a valid one:

```bash
env -u GITHUB_TOKEN npa workbench cosmos check
```

SkyPilot pods do not inherit your shell's `GITHUB_TOKEN`, so the in-cluster
clone is unaffected.

`PermissionDenied` / `No permission` when submitting a serverless job

Serverless workloads (`cosmos train --runtime serverless`, `cosmos infer`, and
serverless `deploy`) create Nebius AI Jobs. If the principal behind your active
`nebius` profile is a service account that lacks the AI Jobs role, the submit is
rejected with `PermissionDenied: service iam ... appbox-...` even though
authentication, capacity lookup, subnet discovery, and S3 all succeed. This is
an authorization gap, not stale credentials — `nebius iam get-access-token`
still works. Grant that service account a role that permits creating and
managing AI Jobs on the project, or switch to a profile whose principal has it
(`nebius profile activate <profile>`), then retry the same command.

`credentials.yaml` is missing or tokens are not loading

Commands that need no tokens tolerate a missing credentials file, but
token-dependent commands will behave as if no token exists. Create the file
under your home directory and secure it:

```bash
mkdir -p ~/.npa
touch ~/.npa/credentials.yaml
chmod 600 ~/.npa/credentials.yaml
```

`credentials.yaml is readable by other users`

Workbench subcommands warn when group or world permissions are present:

```bash
chmod 600 ~/.npa/credentials.yaml
```

`Warning: HF_TOKEN not found in ~/.npa/credentials.yaml`

Add `tokens.HF_TOKEN` to `~/.npa/credentials.yaml` or export `HF_TOKEN` for the
current shell. Cosmos and GR00T deploy dry-runs fail fast unless that token has
actual upstream access to every gated runtime-fetched asset.

`Error: HF_TOKEN does not have access to <repo>` or `401/403 from Hugging Face`

Use a token from <https://huggingface.co/settings/tokens>, accept the gated
model's terms on Hugging Face, then retry. Environment variable `HF_TOKEN`
overrides the value in `credentials.yaml`.

`NGC API error` or `401 from NGC`

Use a current NGC key from <https://ngc.nvidia.com/setup/api-key>. Put it in
`ngc.api_key` in `credentials.yaml` or export `NGC_API_KEY` for the current
shell. If you use an organization or team-scoped key, also set `ngc.org` and
`ngc.team`, or export `NGC_ORG` and `NGC_TEAM`.

`nebius CLI not found on PATH`

Install the Nebius CLI and restart the shell:
<https://docs.nebius.com/cli/install>.

`Nebius auth failed`

Re-run `npa configure` in a terminal. If a profile exists but the wrong one is
active, switch with `nebius profile activate <profile>` and retry.

`terraform binary not found on PATH`

Install Terraform and verify `terraform version` before running managed deploys.
