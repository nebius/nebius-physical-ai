# Configure credentials and project storage

Start with [installation and first-run setup](quickstart.md). This reference
covers authentication, project storage, credential names, and model access.


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

<a id="4a-nebius-account-authentication"></a>

## Nebius account authentication

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

### Creating a project from the CLI (tenant administrator)

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

### Federation or SSO profiles with many tenants

For an SSO or federation profile without `tenant-id` / `parent-id`, bind the
profile to the project you want **before**
`npa configure`, so discovery targets the right place instead of listing every
tenant:

```bash
nebius config set tenant-id <id>
nebius config set parent-id <project-id>
```

Say **yes** to the object-storage prompt: the agent VM and the Physical AI Data
Factory both need an S3 bucket and access key.

### Non-interactive setup

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
credentials only when exact project provenance and a write/read probe
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
setup. Use the health command to enforce access requirements; see
[§4e](#4e-prepare-and-verify-gated-model-access):

```bash
npa configure
```

Workbench images resolve from the anonymous
`ghcr.io/nebius/nebius-physical-ai` mirror by default, so configure does not ask
for or save a container registry. Existing `container_registry` entries and
`NPA_REGISTRY` remain available to build/BYOF compatibility paths but do not
repoint repository-owned runtime defaults. Pass a complete image or a workflow's
explicit `--registry` when intentionally running private or modified bytes.

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
unreadable, or insufficient IAM stops before key creation and probing.
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

<a id="4b-required-credential-key-names"></a>

## Required credential key names

Use these canonical keys in `~/.npa/credentials.yaml`.

| Need | `credentials.yaml` key | Environment override | Required when |
|---|---|---|---|
| Hugging Face token | `tokens.HF_TOKEN` | `HF_TOKEN` | Downloading gated Hugging Face models, datasets, or weights |
| Nebius Token Factory key | `tokens.NEBIUS_TOKEN_FACTORY_KEY` | `NEBIUS_TOKEN_FACTORY_KEY` | Hosted text generation, captioning, and scene reasoning through Token Factory |
| NGC API key | `ngc.api_key` | `NGC_API_KEY` | Pulling an entitlement-controlled NGC image or model, including NuRec NRE |
| NGC organization | `ngc.org` | `NGC_ORG` | Your NGC key is organization-scoped |
| NGC team | `ngc.team` | `NGC_TEAM` | Your NGC key is team-scoped |
| BYOVM SSH host | `ssh.host` | `NPA_BYOVM_HOST`, `NPA_SSH_HOST` | BYOVM commands need a default host |
| BYOVM SSH user | `ssh.user` | `NPA_BYOVM_SSH_USER`, `NPA_SSH_USER` | BYOVM commands need a default SSH user |
| BYOVM SSH private key | `ssh.key_path` | `NPA_BYOVM_SSH_KEY`, `NPA_SSH_KEY` | BYOVM commands need a default private key |
| Object-storage access key | `storage.aws_access_key_id` | `AWS_ACCESS_KEY_ID` | BYOVM, existing storage, or cross-project S3 workflows need explicit storage credentials |
| Object-storage secret key | `storage.aws_secret_access_key` | `AWS_SECRET_ACCESS_KEY` | BYOVM, existing storage, or cross-project S3 workflows need explicit storage credentials |
| Object-storage endpoint | `storage.endpoint_url` | `AWS_ENDPOINT_URL`, `NEBIUS_S3_ENDPOINT`, `NPA_STORAGE_ENDPOINT` | S3-compatible storage is not supplied by managed project config |
| Object-storage bucket | `storage.bucket` | `NPA_CHECKPOINT_BUCKET`, `NEBIUS_S3_BUCKET` | Default checkpoint/artifact location, supplied as `s3://<bucket>/<prefix>/` |

**How to create each token** (step-by-step, including where to click):

- Hugging Face: [docs/workbench/huggingface-token.md](workbench/huggingface-token.md)
- NVIDIA NGC: [docs/workbench/ngc-api-key.md](workbench/ngc-api-key.md)
- Nebius Token Factory: [docs/workbench/token-factory-key.md](workbench/token-factory-key.md)

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

<a id="4c-populate-npacredentialsyaml"></a>

## Populate `~/.npa/credentials.yaml`

Use this example and omit keys you do not need yet:

```yaml
tokens:
  HF_TOKEN: <YOUR_HUGGING_FACE_TOKEN>
  NEBIUS_TOKEN_FACTORY_KEY: <YOUR_TOKEN_FACTORY_KEY>

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

<a id="4d-cross-project-storage-workflows"></a>

## Cross-project storage workflows

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

<a id="4e-prepare-and-verify-gated-model-access"></a>

## Prepare and verify gated model access

Some capability selections pull **gated** Hugging Face models or entitlement-controlled
NGC artifacts. Complete access upstream on the owning account. NPA does not invent a
second acceptance flag: the account token plus a successful exact-revision payload-byte
authorization probe is the automated access preflight. Repository metadata is public
for many gated assets and is never entitlement proof; the upstream licence still
governs use.

Workbench remains usable without either provider. To explicitly audit the full
current catalog, including newly added gated artifacts, run:

```bash
npa configure --prepare-catalog-access
```

The interactive audit groups missing resources by provider and asks before
opening official pages. NPA never clicks an acceptance control or submits legal
assent. After completing user-bound steps, run the printed resume command to
re-check exact upstream access. `Ready` means only that the provider authorized a
representative payload fetch at the pinned revision; it never means NPA accepted terms
or established legal assent. A previous Ready result is reused only while the
credential, artifact revision, payload probe, and terms evidence are unchanged.

For a selected capability, verify the exact assets it will use:

```bash
npa workbench health access --prepare
# or export keys and persist them without putting secret values in argv:
export HF_TOKEN='<your-token>' NGC_API_KEY='<your-key>'
npa workbench health access --save-env-credentials
# Check access for one capability online:
npa workbench health access --capability groot --prepare
# Check credential presence offline; this does not verify model access:
npa workbench health access --capability groot --offline
```

In JSON or another non-interactive context, `--prepare --json` never prompts or
opens a browser. It returns Ready/Pending/Denied/Unavailable evidence, official
links, and an exact safe resume command. Workflow execution performs the same
selected-toolRef closure check before provisioning or submission; unrelated
capabilities remain available when one dependency is blocked. For the broader
credential preflight (presence/format plus basic HF, S3, and Token Factory
connectivity), use `npa workbench health preflight`.


## Tool-specific credentials

These requirements depend on the selected model and runtime:


- Cosmos: requires `HF_TOKEN` during deploy to download gated Hugging Face Cosmos models.
- GR00T: requires `HF_TOKEN` for gated Hugging Face GR00T models; optional `ngc.api_key` or `NGC_API_KEY` is written to the server env for NGC-backed model paths and readiness displays.
- LeRobot: may need `HF_TOKEN` for gated Hugging Face datasets or models.
- FiftyOne: may need `HF_TOKEN` for gated Hugging Face datasets.
- Isaac Lab and Genesis: no token is required by default.

For compatibility, `NGC_API_KEY`, `NGC_ORG`, and `NGC_TEAM` are also
accepted inside the legacy `tokens:` map. Keep the credentials file private
with `chmod 600 ~/.npa/credentials.yaml`; Workbench warns if other users can
read it. Loaded tokens are forwarded to remote workbench SSH commands as
environment variables.
