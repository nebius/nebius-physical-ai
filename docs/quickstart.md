# npa Quickstart

[Docs](README.md)

Install `npa`, connect a Nebius project, and choose a GPU workload. Each linked
workload guide takes you through setup, execution, and inspecting its outputs.
Workbench is designed to be the control plane for a coding agent with terminal
access to this checkout. For guided terminal work, paste the
[setup prompt](workbench/agent-first-run.md#set-up-workbench) or go directly to
the [Cosmos 3 workflow prompt](workbench/agent-first-run.md#run-cosmos-3-generation).
The commands below are the corresponding manual path.

## 1. Platform overview

`npa workbench` provides tools for simulation, policy training, generation,
curation, and evaluation. Workflows run through SkyPilot and exchange artifacts
through S3. [Token Factory](workbench/token-factory.md) provides hosted captioning,
text generation, and reasoning for tools that use those capabilities.

## 2. Prerequisites

- macOS, Linux, or Windows with WSL2 Ubuntu; Python 3.10+ and Git.
- A Nebius account and access to a project in your workload's region.
- The [Nebius CLI](install.md#4-nebius-cli-required) on `PATH`.
- For managed infrastructure: Terraform; for Kubernetes: `kubectl` and the
  [isolated SkyPilot runtime](orchestration/skypilot-setup.md).

Model tokens, GPU requirements, and input formats depend on the workload.
Follow its guide before provisioning. See [platform installation](install.md)
for host tools and WSL2 setup.

## 3. Install npa

Install from the repository; `npa` is not published to PyPI:

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e npa
npa --version
npa workbench --help
```

Both checks work without cloud credentials. In each new shell, return to the
clone and run `source .venv/bin/activate`, or use `.venv/bin/npa` directly.
The base package supports cloud operations; [local engine extras](install.md#3-install-npa-editable-from-the-clone)
are needed only when running those engines in this Python environment.

## 4. Configure credentials

```bash
npa configure
npa configure --show
```

Select the intended tenant, project, and region. Interactive setup provisions
object storage by default; first-time storage setup needs admin permission on
that project. Use `npa configure --no-provision` to save project and token
settings with storage deliberately unselected.

Keep secrets in your private environment or `~/.npa/credentials.yaml` and let
NPA manage `~/.npa/config.yaml`. The project alias is your local name for the
project; it is different from the provider's project ID and region.

<a id="4a-nebius-account-authentication"></a>
<a id="creating-a-project-from-the-cli-tenant-administrator"></a>
<a id="federation-or-sso-profiles-with-many-tenants"></a>
<a id="non-interactive-setup"></a>
<a id="4b-required-credential-key-names"></a>
<a id="4c-populate-npacredentialsyaml"></a>
<a id="4d-cross-project-storage-workflows"></a>
<a id="4e-prepare-and-verify-gated-model-access"></a>

See [configuration](configuration.md) for SSO, project creation, non-interactive
setup, credential names, cross-project storage, and gated-model access.

## 5. First platform checks

Before provisioning, verify the selected Nebius CLI profile:

```bash
npa workbench health preflight --checks nebius --json
```

Expected: the authentication check passes. This does not prove project
permissions, capacity, or model access. [Workbench setup](workbench/getting-started.md)
adds the selected workload's storage, model, cluster, and image checks.

<a id="5a-verify-the-path-works-zero-gpu-inference-nebius-token-factory"></a>

### 5a. Hosted text generation with Nebius Token Factory

For hosted inference, save `NEBIUS_TOKEN_FACTORY_KEY` with `npa configure`, then
run `npa workbench token-factory verify`. A Nebius IAM token cannot replace this
key. Follow [Token Factory](workbench/token-factory.md) to generate and inspect
a real completion or caption artifact. Direct calls with local inputs need no
cluster or S3. Its `--dry-run` still calls the model and only skips output writes.

<a id="5b-the-same-capability-three-coherent-ways"></a>

### 5b. Token Factory from Python or a workflow

Use the [Token Factory CLI and SDK examples](workbench/token-factory.md), or
choose a specification from the [workflow catalog](../workflows/README.md).
Workflow execution needs the SkyPilot and storage setup described below.

## 6. Developing and testing npa

For repository changes, use the separate
[contributor environment and test commands](../npa/README.md#developing-and-testing-npa).
For a first workload, continue below.

<a id="7-flagship-gpu-workload-nvidia-cosmos"></a>

## 7. Choose your workload

| Goal | Guide | Result to inspect |
| --- | --- | --- |
| Generate an image or video | [Cosmos 3](#standalone-cosmos-3-generation) | Generated media and `generate.json` |
| Augment your video | [PAIDF + Cosmos 3](workbench/guides/paidf-cosmos3.md) | Raw generated clips, evaluator reports, and accepted outputs |
| Build a labeled dataset | [Physical AI Data Factory](workbench/guides/physical-ai-data-factory-deploy.md) | Augmented frames, labels, curation reports, and Rerun recording |
| Train or evaluate a robot policy | [Robot guides](workbench/guides/README.md) | The guide's checkpoint, rollout, and evaluation artifacts |
| Reconstruct a captured scene | [Neural reconstruction](workbench/guides/neural-reconstruction.md) | Renderable scene, novel views, and Rerun recording |

### Standalone Cosmos 3 generation

The [Cosmos 3 guide](workbench/cosmos3-generate.md) uses
`workflows/testing/cosmos3-generate.yaml`. Its default requests one H100,
16 CPU, and 80 GiB host memory for public Cosmos3-Nano text-to-image generation.
The requested guardrails additionally need access to gated Hugging Face weights.
Check the guide's model/image/GPU compatibility and guardrail limitations.

Complete [Workbench setup](workbench/getting-started.md), then follow the
[submission example](workbench/cosmos3-generate.md#workflow). Inspect the media
and manifest after success; job completion alone does not establish usable output.

## 8. Do more with npa

- [Workflow guide](workbench/npa-workflow-guide.md): validate, plan, submit,
  monitor, and resume your own pipeline.
- [CLI / SDK walkthrough](workbench/cli-sdk-yaml-walkthrough.md): integrate a
  tool from the shell, Python, or HTTP.
- [Browser workbench](agent.md): deploy the optional agent and embedded viewers.
- [Run lifecycle](run-lifecycle.md) and [teardown](teardown.md): recover runs
  and remove owned resources in order.

## 9. Troubleshooting

| Symptom | Next step |
| --- | --- |
| `npa: command not found` | From the clone root, activate `.venv` or run `.venv/bin/npa --help`. |
| Missing Python module | Reinstall with `pip install -e npa` in the same environment; see [installation](install.md). |
| Nebius authentication or project permission failure | Check the active profile and exact project; see [configuration](configuration.md#4a-nebius-account-authentication). |
| Hugging Face `401` / `403` | Use the correct token and accept the exact model's access terms; see [model access](configuration.md#4e-prepare-and-verify-gated-model-access). |
| NGC `401` | Check the [NGC key](workbench/ngc-api-key.md) and entitlement to the required artifact. |
| S3 endpoint or write failure | Check the selected project's bucket, credentials, and regional endpoint in [configuration](configuration.md); AWS CLI calls need `--endpoint-url`. |
| GPU job remains pending or image pull fails | Follow [runtime troubleshooting](workbench/troubleshooting/known-footguns.md). |
| Credential file is readable by other users | Run `chmod 600 ~/.npa/credentials.yaml`. |

For a failed run, retain its run ID and inspect `workflow status`, `logs`, and
`artifacts` before retrying. [CLI errors](cli-errors.md) explains JSON output,
exit codes, and `NPA_DEBUG=1` tracebacks.
