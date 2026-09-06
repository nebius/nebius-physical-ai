# npa Quickstart

This is the platform entry point for Nebius Physical AI. Read this first to
install the `npa` CLI/SDK and configure the single user-authored credential
store. After credential setup, continue with
[Workbench Getting Started](workbench/getting-started.md) for Kubernetes,
SkyPilot, registry, S3, and first workload setup.

## 1. Platform overview

`npa` is the Nebius Physical AI platform CLI/SDK. It provides a common command
surface for physical AI workflows such as simulation, training, inference,
visualization, dataset conversion, and storage handoff.

Workbench is the main solution namespace. Its tools run in containerized jobs
or services on Nebius, or call hosted inference through Token Factory. Pipelines
exchange artifacts through S3-compatible storage. This page covers installation,
credentials, and choosing your first workload; the linked guides cover execution.

For a broader architecture map, see the repository [README](../README.md) and
the package overview in [npa/README.md](../npa/README.md).

## 2. Prerequisites

`npa` runs cloud workloads on Nebius, so a Nebius account and the `nebius` CLI
are required.

- Python 3.10 or newer. The package metadata requires `>=3.10`.
- Git, `python3 -m venv`, and `pip`.
- **macOS**, **Linux**, or **Windows via WSL2** (Ubuntu). `npa` cloud workflows
  (S3 / SkyPilot / Kubernetes) assume a POSIX environment. On Windows, run
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
- **Debian/Ubuntu:** install the venv module first:
  `sudo apt-get install -y python3-venv`.
- **Need Python 3.10+, or a faster installer?** [`uv`](https://docs.astral.sh/uv/)
  can install Python and create the env:
  `uv venv .venv && source .venv/bin/activate && uv pip install -e npa`.
- **Out of scope (needs extra steps):** Alpine/musl, brand-new Python before
  wheels exist, and air-gapped machines.

For the Nebius CLI, WSL2 setup, and operator tools, see:
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

The GPU wheels above are only needed to run those engines **locally**; cloud
jobs execute them inside the Nebius images they launch. To run cloud workloads
(Sections 5+) the base install is enough; you also need the Nebius CLI:
see [docs/install.md § Nebius CLI](install.md#4-nebius-cli-required).

## 4. Configure credentials

Run `npa configure` to select your tenant, project, and region and save
credentials. Interactive setup provisions object storage by default. First-time
storage setup needs admin permission on the target project; tenant-wide admin
permission and tenant-wide project listing are not required.

```bash
npa configure
npa configure --show
```

Keep secret values in the private environment or `~/.npa/credentials.yaml`.
Let NPA manage `~/.npa/config.yaml`. Use `npa configure --no-provision` for
provider-free project and token setup with storage deliberately unselected.

For an existing coding agent, use the [first-run prompts](workbench/agent-first-run.md).
They cover private credentials, project IAM, model access, input preparation,
and the first PAIDF + Cosmos 3 run.

<a id="4a-nebius-account-authentication"></a>
<a id="creating-a-project-from-the-cli-tenant-administrator"></a>
<a id="federation-or-sso-profiles-with-many-tenants"></a>
<a id="non-interactive-setup"></a>
<a id="4b-required-credential-key-names"></a>
<a id="4c-populate-npacredentialsyaml"></a>
<a id="4d-cross-project-storage-workflows"></a>
<a id="4e-prepare-and-verify-gated-model-access"></a>

Detailed setup has moved to [configuration](configuration.md):

- [Nebius authentication](configuration.md#4a-nebius-account-authentication),
  including project creation, SSO profiles, and non-interactive provisioning.
- [Credential names](configuration.md#4b-required-credential-key-names) and
  [credential-file layout](configuration.md#4c-populate-npacredentialsyaml).
- [Cross-project storage](configuration.md#4d-cross-project-storage-workflows).
- [Gated model access](configuration.md#4e-prepare-and-verify-gated-model-access).

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

Before provisioning cloud resources, verify the selected Nebius CLI profile:

```bash
npa workbench health preflight --checks nebius --json
```

This opt-in check verifies authentication. Hosted Token Factory calls do not
require it; resource creation still needs permission on the target project.

<a id="5a-verify-the-path-works-zero-gpu-inference-nebius-token-factory"></a>

### 5a. Hosted text generation with Nebius Token Factory

For standalone GPU image generation, see [Cosmos 3](#standalone-cosmos-3-generation).
For text generation, image captioning, or scene reasoning, use
[Token Factory](https://tokenfactory.nebius.com/). Hosted inference needs its own
`NEBIUS_TOKEN_FACTORY_KEY`; a Nebius IAM token cannot replace it. Local inputs and
outputs need no cluster or S3. Save the key with `npa configure`, then check access:

```bash
npa workbench token-factory verify
npa workbench token-factory models
```

Write a prompt and generate a completion against a hosted model:

```bash
printf 'Explain sim-to-real transfer in one sentence.\n' > /tmp/prompts.txt
npa workbench token-factory generate \
  --input-path /tmp/prompts.txt \
  --output-path /tmp/tf-generations.jsonl \
  --output json
```

Gate: `verify` reports `authenticated: true` with a non-zero model count, and
`generate` writes `/tmp/tf-generations.jsonl` with a nonempty completion. Model
listing proves authentication; successful inference also proves the selected
model is usable. If the default model is unavailable, pass `--model` with an
appropriate text model returned by `models`.

Read the JSONL result before using it downstream. `--dry-run` still calls the
hosted model; it only skips the output write.

More Token Factory capabilities (image captioning, physical-scene reasoning) and
the checked-in SkyPilot templates:
[docs/workbench/token-factory.md](workbench/token-factory.md).

<a id="5b-the-same-capability-three-coherent-ways"></a>

### 5b. Token Factory from Python or a workflow

Token Factory exposes CLI commands, SDK wrappers, and workflow toolRefs for the
same operations. Use the CLI or SDK directly for hosted inference; use a
workflow when it belongs in a larger pipeline.

**CLI** (shown above):

```bash
npa workbench token-factory generate --input-path /tmp/prompts.txt \
  --output-path /tmp/tf-generations.jsonl --output json
```

**Python SDK:**

```python
from npa.sdk.workbench import token_factory

token_factory.generate(
    input_path="/tmp/prompts.txt",
    output_path="/tmp/tf-generations.jsonl",
    output="json",
)
```

The SDK wrapper writes the artifact, prints the result, and returns `None`.
The lower-level `npa.workbench.token_factory.generate_text` returns a dataclass;
call `write_generations` explicitly if you use that API and need a saved result.

**Workflow specs:** `npa.workflow/v0.0.1` runs these tools in CPU tasks through
SkyPilot. This mode needs the configured cluster and S3 runtime described in
[Workbench Getting Started](workbench/getting-started.md). For examples, see
`npa/workflows/workbench/npa-workflows/token-factory-generate.yaml` and
[the workflows guide](workbench-yaml-guide.md).

## 6. Developing and testing npa

Contributors can follow the [package development instructions](../npa/README.md#developing-and-testing-npa).
For your first workload, continue below.

<a id="7-flagship-gpu-workload-nvidia-cosmos"></a>

## 7. Run PAIDF with Cosmos 3

Use [PAIDF + Cosmos 3](workbench/guides/paidf-cosmos3.md) to generate video
variants conditioned on your source clip, evaluate them, and curate accepted
outputs. Provide a local H.264 MP4 or a private S3 MP4 URI. You need writable
project storage, compatible GPU resources, SkyPilot, Hugging Face model access,
and Token Factory credentials for captioning and evaluation.

Follow the [coding-agent workflow prompt](workbench/agent-first-run.md#run-paidf-with-cosmos-3)
to validate and plan with the actual bucket and input, review resources,
prepare input and images, and submit with `--runtime`. The PAIDF guide's
command examples cover validation and planning only.
Image preflight may create and delete a temporary probe pod.

Inspect generated media, evaluator reports, quality disposition, and Rerun
evidence. A quality rejection can be a valid terminal result after bounded
refinement; labeling and curation are then skipped. The published PAIDF composite
blends source and generated frames (80% source by default); use the raw Cosmos output to assess
generation quality. The original
[Physical AI Data Factory](workbench/guides/physical-ai-data-factory-deploy.md)
continues to use Cosmos Transfer 2.5.

### Standalone Cosmos 3 generation

Start with [Cosmos 3 generation](workbench/cosmos3-generate.md). Its
`cosmos3-generate.yaml` workflow runs the containerized model and publishes an
image or video plus `generate.json` to S3. The checked-in default is
`text2image` with public Cosmos3-Nano on one H100, 16 CPU, and 80 GiB host memory.
Choose a compatible model/image/GPU combination; Cosmos versions do not share a
blanket GPU compatibility guarantee.

You need a configured project, writable S3, a GPU Kubernetes cluster, and
SkyPilot. Guardrails are requested by default and require access to their gated
Hugging Face weights even though Cosmos3-Nano itself is public. Follow
[Workbench Getting Started](workbench/getting-started.md) for the ordered access,
planning, cluster, and image checks, then the
[Cosmos 3 guide](workbench/cosmos3-generate.md#workflow) for submission.

After the run succeeds, inspect the generated media and manifest. A completed
job alone does not prove usable output. The Cosmos guide records the known
upstream guardrail limitations; `guardrails: true` records the requested setting,
not proof that every safety model ran.

The older `cosmos deploy --runtime serverless` endpoint has no supported
serverless generated-media export to S3. Likewise,
`cosmos train --runtime serverless --smoke` only checks job execution and writes
a status `checkpoint.json`; it does not train a model or generate media. These
are distinct from the Cosmos 3 generation workflow.

## 8. Do more with npa

With install → `npa configure` → your first GPU workload on Nebius AI Cloud
(Section 7) done, build outward:

- **Run Workbench workloads:** PAIDF + Cosmos 3 (Section 7), VLM evaluation, Sim2Real,
  and more. See [Workbench Getting Started](workbench/getting-started.md) for
  Kubernetes, SkyPilot, registry, and S3 setup, and the
  [robot guides](workbench/guides/README.md).
- **Deploy the self-hosted agent:** `npa agent` is a browser workbench VM. It
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

FastAPI is a base dependency. A missing import indicates an incomplete install
or a different interpreter. Activate this checkout's environment and reinstall
the package with test tooling:

```bash
pip install -e "npa[dev]"
make test PYTHON="$(pwd)/.venv/bin/python"
```

`aws s3 ls` fails with `Could not connect to the endpoint URL`

Nebius object storage is S3-compatible but is not AWS. Older `aws-cli` (v1)
ignores the `AWS_ENDPOINT_URL` environment variable, so it tries the AWS
endpoint and fails. Pass the endpoint explicitly, or use `aws-cli` v2:

```bash
aws s3 ls --endpoint-url https://storage.<your-region>.nebius.cloud
```

Configured workflow commands resolve storage settings and pass the endpoint
to their S3 client. Direct Token Factory S3 calls need `AWS_ENDPOINT_URL` and
AWS credentials in the process environment; see the
[Token Factory guide](workbench/token-factory.md#generate-and-inspect-artifacts).

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
an authorization gap: `nebius iam get-access-token` still works. Grant that service account a role that permits creating and
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
