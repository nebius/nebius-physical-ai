# Workbench service storage and ingress boundaries

Dataset, Insights, and Terraform-managed VM deployments fail closed unless an
operator configures their authentication, storage, and ingress boundaries.
This is an intentional security migration.

## SSH host identity

Paramiko command execution, SFTP uploads, and Terraform bootstrap require a
verified host key. For newly provisioned VMs, NPA persists a fresh boot challenge
in the deployment's private Terraform directory. Cloud-init emits the challenge
and its Ed25519 public host key to the provider's serial console. Before SSH,
NPA verifies the exact project, instance, address, and boot challenge through
the authenticated provider API, then checks the serial log's provider-supplied
instance identity. This uses existing profile permissions to read serial logs;
it creates no IAM grants or credentials. Missing log delivery is a readiness
condition; identity conflicts and unreadable provider evidence stop bootstrap.

Verified public keys and integrity receipts are stored under the selected
`NPA_CONFIG_DIR` in owner-only files and reused by both Terraform and Python.
The SSH handshake still rejects a changed key, including after serial logs
expire. Terraform uses `StrictHostKeyChecking=yes`; a network `ssh-keyscan`
alone does not establish the VM's identity.

Existing VMs created before this boot challenge and BYOVM deployments require
the operator's independently verified known-hosts entry. Set
`NPA_SSH_KNOWN_HOSTS` to select a separate verified file, or pass `known_hosts=`
to the Python SSH client. An explicitly selected file is the complete trust
source for that connection. Unknown and changed keys fail before authentication
or runtime credential staging. A direct Terraform caller must supply a fresh
`ssh_host_key_nonce` and an NPA-capable Python interpreter through
`NPA_SSH_TRUST_PYTHON`; NPA's provisioner supplies both automatically. Skipping
Terraform's SSH readiness check also skips automatic host-key discovery; obtain
the authenticated pin before a later SSH bootstrap.

SSH uploads create a private staging directory before opening any file. Files
are written with owner-only permissions and published with atomic POSIX rename;
the SFTP server must support the OpenSSH POSIX rename extension. Upload failures
preserve the previous destination. Persistent deployment configuration is
staged privately on the destination filesystem before replacement, and both
success and failure paths remove their staging files. General SFTP uploads now
finish with mode `0600`; deployment installers explicitly set the owner and
mode needed by each service. Credentials travel over SFTP, never in deployment
command arguments.

The agent's nginx password hash is generated with bcrypt at work factor 12 from protected stdin,
then atomically installed as `root:www-data` with mode `0640`. Passwords must
contain 1–72 UTF-8 bytes and no CR, LF, or NUL; longer values are rejected before
deployment to prevent bcrypt truncation. The generated operator password meets
this contract.

Cosmos service environment updates use private staging and atomic replacement,
including on restart. Hugging Face downloads inherit their token from the
environment. LeRobot's Docker launch reads both its existing environment file
and a second private file containing service overrides and shared credentials;
those values no longer appear in process arguments.

VM credential files use literal `KEY=value` data, matching Docker's environment
file contract. Cosmos, FiftyOne, LeRobot, Genesis, and native VM login hooks read
them without shell evaluation. Quotes, dollars, backticks, spaces, and equals
signs remain part of the value. CR, LF, and NUL are rejected by the writers.
Legacy hand-written files containing shell assignments, `export`, or shell
quoting must be rewritten as literal data. New login hooks source only the
root-owned loader program. Terraform does not reapply cloud-init changes to
existing VMs: reprovision a native VM or replace its old profile and bashrc
hooks with the trusted loader before reusing its credentials. Upgrading the
local CLI alone does not replace those older login hooks.

Standalone Lichtblick now binds to loopback by default. Its static server
co-serves the selected MCAP without authentication. Use a verified SSH forward
or authenticated proxy for remote access. Selecting a non-loopback `--host`
explicitly publishes the artifact to clients that can reach that interface.

Native FiftyOne also binds to loopback during its initial Terraform cloud-init,
before SSH configuration runs. Its bootstrap installs the same patched database
and dependency floors as the CLI installer and fails when application readiness
fails. Existing native VMs need the secured installer or fresh provisioning;
changing the template alone does not replace their running application.

## Cosmos3 native Ray serving

The native service requires Ray management authentication and starts its own
local runtime with the dashboard disabled. `RAY_AUTH_MODE=token` is required
independently of `NPA_COSMOS3_RAY_TOKEN`, which authenticates inference requests.
The service creates a fresh owner-only management credential for its local
runtime; the application token is not reused.

Text-only requests require no input storage configuration. For media, prompt
files, and transfer controls, stage exact objects in S3 and configure
`NPA_COSMOS3_RAY_ALLOWED_S3_ROOTS` with the permitted bucket/prefix roots.
The service downloads those inputs to a fresh request directory before invoking
upstream inference. HTTP URLs, server-local paths, sample defaults files, and
caller-controlled output directories are refused. Inline prompts and sampling
options keep their existing contract.

## Dataset and Insights services

The deployed FastAPI applications now default to token authentication. Set the
service token and leave the mode at its secure default:

```bash
export DATASET_TOKEN='<owner-provided-token>'
export INSIGHTS_TOKEN='<owner-provided-token>'
```

`DATASET_AUTH_MODE=none` and `INSIGHTS_AUTH_MODE=none` remain explicit opt-ins
for a deliberately local or isolated test service. They are never the default.

For deployed FastAPI requests, storage access also starts with no authorized
roots. Configure one or more exact S3 bucket/prefix roots as a comma-separated
list and local sandbox roots as a platform path-separated list:

```bash
export DATASET_ALLOWED_S3_ROOTS='s3://<bucket>/<dataset-prefix>'
export INSIGHTS_ALLOWED_S3_ROOTS='s3://<bucket>/<insights-prefix>'
export DATASET_ALLOWED_LOCAL_ROOTS='/srv/npa/dataset-sandbox'
export INSIGHTS_ALLOWED_LOCAL_ROOTS='/srv/npa/insights-sandbox'
```

Every read, write, existence check, and prefix listing is checked in the storage
layer. Foreign buckets/prefixes, unsupported schemes, traversal, and canonical
local paths that escape through a symlink are rejected. A bucket root is allowed
only when the operator explicitly configures `s3://<bucket>`.

The allowlist variables are service-boundary configuration. Default embedded
CLI commands, SDK calls with `service=False` / `mode="local"`, and workflow
toolRef processes execute in the caller's existing filesystem and S3 security
context and do not require these variables. They retain the pre-service path
contract; deploying the FastAPI application is what activates the additional
request-scoped containment boundary.

Existing deployments must set their current roots before upgrade. Prefer the
narrowest prefix each service needs; do not configure a whole bucket when one
dataset or store prefix is sufficient.

## Terraform VM ingress

`ssh_cidr_block` no longer defaults to `0.0.0.0/0`. Application ingress is now a
separate `application_cidr_block`; its rule covers `server_port` and
`extra_ingress_ports`. Empty application ingress creates no public application
rule. An ordinary NPA deploy that needs remote bootstrap must pass an
operator-selected SSH CIDR, for example through its existing repeatable
`--tf-var` option:

```bash
npa <deploy-command> \
  --tf-var 'ssh_cidr_block=<operator-cidr>' \
  --tf-var 'application_cidr_block=<operator-cidr>'
```

There is no repository-supplied live CIDR. Choose the narrow source authorized
for the deployment. For an internal-only application, omit
`application_cidr_block` and use the supported SSH/tunnel access path.

World-open ingress requires a second, conspicuous acknowledgement in addition
to the world-open CIDR:

```text
ssh_cidr_block=0.0.0.0/0
allow_world_open_ssh=true

application_cidr_block=0.0.0.0/0
allow_world_open_application=true
```

The manual repair surfaces follow the same rule. They require `--source`; a
`/0` source additionally requires `--allow-world-open`:

```bash
npa network ensure-ingress --vm <instance-id> --ports <port> \
  --source '<operator-cidr>'
npa workbench fiftyone ensure-ingress -n <alias> \
  --source '<operator-cidr>'
```

Deploy-time ingress reconciliation uses only the explicit
`application_cidr_block`; it never substitutes a world-open source. The simple
agent setup path accepts repeatable `--tf-var` values and fails before VM
creation when either its SSH bootstrap boundary or application boundary is
missing.

This is especially important for FiftyOne, which does not add application-level
authentication. Upgrading an existing VM may change or remove its old shared
SSH/application rule; review the Terraform plan and set the two boundaries
independently before applying.

## Storage credentials during VM bootstrap

Terraform and cloud-init no longer receive S3 access or secret keys. NPA creates
the VM with its attached service-account identity, verifies the exact resulting
VM and SSH channel, then stages the already-scoped runtime storage environment
through owner-only files. The metadata identity supplies rotating Nebius
control-plane authentication; it is not treated as an S3 HMAC credential and no
broader IAM role is created.

If scoped storage credentials or exact VM identity cannot be resolved, runtime
credential staging fails closed. Existing VMs whose old user-data contained S3
keys must be redeployed and those old keys rotated; changing the template cannot
erase provider-retained user-data from an existing instance.

Deployment source bundles use the Git index as their inventory and read the
current bytes of those tracked files. Stage new source files before deploying.
Ignored and untracked local files are excluded; missing files, symlinks, and
special files fail packaging. Deployed bundles carry an inventory of their
archived bytes so an agent can forward its source without Git metadata; every
file must still match its recorded hash. The remote installer stages bundles and
credentials in private directories before atomic installation.

## Verified workflow execution

Unchanged project-credential lookups preserve the exact file bytes used to
verify an isolated SkyPilot runtime and retain owner-only permissions. Adding
an alias, migrating proven legacy ownership, or rotating credentials still
updates the store; a runtime verified against different credentials refuses
reuse until its owned state is reconciled.

GPU capacity checks translate SkyPilot resource requests before comparing them
with Kubernetes inventory. They include GPU CPU/memory defaults and conservative
fractional rounding. The supported SkyPilot Kubernetes template emits memory
with decimal `G`, after normalizing input suffixes; checks use those executor
units. GPU memory ratios and explicit `instance_type` shapes are not proven by
this boundary and fail closed; use explicit absolute `cpus` and `memory`
requests. Native Kubernetes quantities retain their own parsing rules.
