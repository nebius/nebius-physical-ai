<div align="center">

# Nebius Physical AI

**One CLI, one SDK, one workflow layer for physical-AI workloads on Nebius —
data curation, simulation, synthetic data, policy training, evaluation,
observability, and cluster orchestration.**

<img src="docs/assets/workbench-architecture.png" alt="Nebius Physical AI Workbench architecture" width="820" />

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platforms: macOS · Linux · WSL2](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20WSL2-lightgrey.svg)](docs/install.md)
[![Test](https://github.com/nebius/nebius-physical-ai/actions/workflows/test.yml/badge.svg)](https://github.com/nebius/nebius-physical-ai/actions/workflows/test.yml)
[![PRs welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)

**[Quickstart](docs/quickstart.md)** ·
**[Guides](docs/workbench/guides/README.md)** ·
**[Workbench docs](docs/workbench/)** ·
**[CLI reference](docs/cli/README.md)** ·
**[Cookbooks](docs/workbench/cookbooks/README.md)** ·
**[Contributing](CONTRIBUTING.md)**

</div>

---

## What is `npa`?

`npa` is the CLI and SDK for **Nebius Physical AI**. Workbench is its primary
solution: one command surface that composes data curation, simulation,
synthetic data, policy training, evaluation, export, observability, and
declarative workflows on Nebius object storage, orchestration, vLLM-compatible
serving, managed Kubernetes, and GPU clusters (H100 · H200 · L40S · B300 ·
RTX6000).

> Partners integrate independently. Teams assemble from open blueprints.
> Nebius owns the infrastructure layer and compute substrate.

|                                   |                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| **What you can do**               | Curate datasets · train and evaluate policies · render synthetic data · run sim-to-real loops · serve models |
| **Who it's for**                  | Robotics teams, physical-AI researchers, and partners shipping on Nebius              |
| **Where it runs**                 | Nebius S3, managed Kubernetes, and GPU clusters                                        |
| **How you extend it**             | Declarative `npa.workflow/v0.0.1` YAML specs and reusable Workbench tool refs         |

---

## Setup

Three steps take you from a clone to a real result on Nebius: install `npa`,
configure Nebius, then run your first cloud workload.

### 1. Install npa

`npa` works with **Python 3.10+** and installs editable from the clone (it is
not on PyPI):

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv
source .venv/bin/activate
pip install -e npa
```

Verify: `npa --version`.

> The base install is the core CLI, with no GPU or database wheels. Before
> running cloud workloads, add the remaining workbench dependencies with
> `pip install -e "npa[full]"`.

> **Windows:** use **WSL2 Ubuntu**. Full per-platform steps (venv, Nebius CLI,
> WSL2): [docs/install.md](docs/install.md).

### 2. Configure Nebius

[Sign up](https://docs.nebius.com/signup-billing/sign-up) and create a
[tenant and project](https://docs.nebius.com/iam/manage-projects), install the
Nebius CLI, then run `npa configure` — it creates or reuses your Nebius CLI
profile and writes `~/.npa/credentials.yaml` + `~/.npa/config.yaml`:

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc or ~/.bashrc
npa configure
```

Full account/credential detail: [docs/quickstart.md](docs/quickstart.md).

### 3. Your first cloud result

[Nebius Token Factory](https://tokenfactory.nebius.com/) zero-GPU inference is
the cheapest way to get a real result — it needs only a
`NEBIUS_TOKEN_FACTORY_KEY` (add it during `npa configure`), no cluster or GPU:

```bash
npa workbench token-factory verify        # confirm the key authenticates
printf 'Explain sim-to-real transfer in one sentence.\n' > /tmp/prompts.txt
npa workbench token-factory generate \
  --input-path /tmp/prompts.txt --output-path /tmp/tf-generations.jsonl --output json
```

See [docs/workbench/token-factory.md](docs/workbench/token-factory.md).

---

## Do more with npa

- **Run workbench workloads** — NVIDIA Cosmos, vlm-eval, sim2real, and more.
  Start with the [robot guides](docs/workbench/guides/README.md) and
  [Workbench Getting Started](docs/workbench/getting-started.md); the flagship
  GPU workload is
  [NVIDIA Cosmos](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos).
- **Deploy the self-hosted agent** — `npa agent` is a browser workbench VM. It
  builds on the setup above and additionally needs **Terraform**, an **SSH key
  pair**, a **Token Factory key**, and **writable S3** (~20 min).

### Deploy the self-hosted `npa` agent

`npa agent` is a self-hosted **browser workbench VM**: HTTPS UI with
basic-auth login, grounded chat over Nebius Token Factory
(default `nvidia/Cosmos3-Super-Reasoner`), Sim Assets + Cameras panels, an
embedded [Rerun](https://www.rerun.io) viewer for `.rrd` recordings, and
draft/validate/plan/submit endpoints for `npa.workflow/v0.0.1` specs.

```bash
# Check prerequisites first (terraform, SSH key pair, Nebius auth, Token Factory
# key) — no cloud side effects, so late failures surface before any VM is built:
npa agent preflight

npa agent fresh-setup \
  --project my-agent --name agent \
  --project-id project-... --tenant-id tenant-... --region eu-north1
npa agent status --project my-agent --name agent
NPA_AGENT_CHAT_LIVE=1 npa agent verify-live --project my-agent --name agent
```

`fresh-setup` provisions the VM with Terraform, then `bootstrap` refreshes
the UI/backend/nginx layer without touching infra. Operator docs:
[skills/tools/npa-agent/SKILL.md](skills/tools/npa-agent/SKILL.md) ·
teardown/reproduce loop: [skills/workflows/agent-fresh-operate/SKILL.md](skills/workflows/agent-fresh-operate/SKILL.md).

The agent is tenant-aware for read-only discovery. Its **Agent access** panel
and `GET /api/access` show the running identity's effective access project by
project. Artifact search spans only buckets for which the agent can both
associate the bucket with a visible project and verify S3 object-list access.
Partial access is expected and is reported without hiding accessible projects.
Workflow submission and artifact writes/deletes remain deployment-project
scoped. Arbitrary caller-supplied S3 URIs remain configuration scoped; an exact
artifact selected from a discovered cross-project run can be read without
broadening those mutation boundaries.

This read-only tenant behavior is enforced by the agent application, not by a
structurally read-only IAM credential. Deployments may still attach a service
account with tenant-level editors-group grants; operators must treat that
credential as privileged even though cross-project mutation endpoints are not
exposed. Mutation endpoints continue to target only the configured home project.

`GET /api/artifacts/run/{run_id}` returns at most one native S3 page (up to
1,000 objects), never the entire run. A truncated response includes
`next_cursor`; consumers must repeat the request with that cursor plus the
returned `resolved_prefix` and `bucket` as `resource_bucket` until
`truncated=false`. The bundled UI follows this contract. Older consumers that
assumed a complete array must migrate to cursor following; page-local counts and
`preferred` selection describe only the returned page.

---

## Before you burn GPU-hours — preflight

A short list of things that catch first-time users mid-run. Skim before your
first GPU submit.

- **Run preflight.** `npa workbench health preflight` is a single
  PASS/WARN/FAIL/SKIP check over the credentials nearly every job needs —
  Hugging Face, NVIDIA NGC, Nebius object storage (S3), and Token Factory.
  Add `--offline` to check presence only (no network), or `--json` for
  machine-readable output. See
  [FTUE-AUDIT.md § friction 1](FTUE-AUDIT.md#friction-points-ordered).
- **GPU routing matters.** Isaac Lab needs an **RT-core** GPU (L40S / RTX
  Pro 6000), not an H100. See [docs/workbench/troubleshooting/known-footguns.md § L40S Capacity](docs/workbench/troubleshooting/known-footguns.md#l40s-capacity-is-on-demand-zero).
- **Registry pull secrets expire silently.** A `401` on image pull usually
  means the `npa-nebius-registry` pull secret needs refreshing. See
  [known-footguns.md § Registry Pull Secret](docs/workbench/troubleshooting/known-footguns.md#registry-pull-secret-expires-silently).
- **Prefer `npa workbench workflow submit` for multi-stage jobs.** Pass
  `npa.workflow` specs (or legacy runbooks); avoid hand-editing scheduler
  YAML. See [FTUE-AUDIT.md § friction 4](FTUE-AUDIT.md#friction-points-ordered).
- **Token Factory keys are not Nebius IAM tokens.** They start with `v1.` and
  live under `NEBIUS_TOKEN_FACTORY_KEY`. See
  [docs/workbench/token-factory.md](docs/workbench/token-factory.md).
- **Always pass `-p PROJECT -n NAME` to `<tool> status`.** Bare `status` may
  hit a stale endpoint — see the `[M] <tool> status without -p/-n` entry in
  [FIXME.md](FIXME.md).

For the full known-issues surface: [docs/workbench/troubleshooting/known-footguns.md](docs/workbench/troubleshooting/known-footguns.md)
and the active operational backlog in [FIXME.md](FIXME.md).

---

## Learn by doing — pick a robot

Short copy-paste walkthroughs — pick any robot or simulator. Full index:
[docs/workbench/guides/README.md](docs/workbench/guides/README.md).

| Guide                                                                                             | Robot                | Sim / engine     | Public dataset                     |
| ------------------------------------------------------------------------------------------------- | -------------------- | ---------------- | ---------------------------------- |
| [Pick-and-place with a Franka arm](docs/workbench/guides/franka-pick-and-place-genesis.md)        | Franka Emika Panda   | Genesis          | DROID (Franka)                     |
| [Teach a robot to push a T](docs/workbench/guides/pusht-sim-to-real.md)                           | sim pusher           | sim-to-real loop | `lerobot/pusht`                    |
| [Train a Reachy 2 humanoid policy](docs/workbench/guides/reachy2-lerobot-policy.md)               | Reachy 2             | LeRobot          | Pollen Robotics / LeRobot Hub      |
| [Make a Unitree G1 walk](docs/workbench/guides/g1-humanoid-walk-sonic.md)                         | Unitree G1           | MuJoCo           | NVIDIA GEAR-SONIC                  |
| [Train a quadruped to run](docs/workbench/guides/quadruped-isaac-lab.md)                          | ANYmal / quadruped   | Isaac Lab        | Isaac Lab built-in tasks           |

Longer end-to-end recipes (BDD100K + LanceDB, Isaac-Lab BYOF, LeRobot GPU
benchmarks): [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md).

---

## Workbench at a glance

Workbench is the main product surface. Every tool lives under `npa workbench`
(there is no `solutions` CLI namespace). Highlights:

- **`vlm-eval`** scores rollouts with API or self-hosted vLLM backends —
  see [`vlm-eval-single.yaml`](npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml).
- **`token-factory`** wraps Nebius Token Factory for zero-GPU inference,
  captioning, and reasoning against your own frames.
- **`health preflight`** validates HF / NGC / S3 / Token Factory credentials
  before a deploy or GPU job.
- **`sonic export`** converts locomotion checkpoints to ONNX.
- **`workflow validate-spec` / `plan-spec` / `run-spec` / `submit`** operate on
  customer-facing `npa.workflow/v0.0.1` specs — see
  [Author and submit workflows](#author-and-submit-workflows).
- **`foxglove`** packs run frames/metrics/logs into MCAP and installs the pinned
  [`@foxglove/embed`](https://docs.foxglove.dev/docs/embed/typescript-sdk) assets
  behind the agent's embedded Foxglove viewer — see
  [docs/cli/foxglove.md](docs/cli/foxglove.md).
- **`trigger`** watches S3-compatible prefixes and retriggers workflows.
- **`golden-eval`** runs per-container hello-world reruns as a CI gate.
- Public GHCR releases and exact hardware-specific tags are listed in the
  [Workbench container image catalog](docs/workbench/container-image-catalog.md).
- SONIC image routing is manifest-driven — see
  [sonic-image-catalog.md](docs/workbench/sonic-image-catalog.md).

<details>
<summary><strong>Browse the full command inventory by category</strong></summary>

| Category         | Workbench commands                                                                                                                                                                                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data curation    | `npa workbench fiftyone curate`, `eval`, `load-dataset`, `datasets list`; `npa workbench lancedb deploy`, `create-table`, `import-lerobot`, `import-bdd100k`, `backfill`, `create-mv`, `refresh-mv`, `query-table`, `query`; `npa workbench detection-training train`, `eval`, `status`, `list`         |
| Synthetic data   | `npa workbench cosmos infer`, `train`, `serve`, `status`; `npa workbench cosmos2 transfer`; `npa workbench cosmos3 reason`; `npa workbench genesis generate-demos`; specs such as [`bdd100k-pipeline.yaml`](npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml) |
| Simulation      | `npa workbench isaac-lab train`, `eval`, `export-lerobot`, `export-onnx`; `npa workbench genesis train-teacher`, `generate-demos`, `eval-teacher`, `eval-student`, `diagnose`, `tune`; `npa workbench sonic retargeting run`, `workflow`                                                                    |
| Eval            | `npa workbench vlm-eval run`, `benchmark`, `workflow`, `status`, `list`; `npa workbench mjlab eval`, `workflow`; `npa workbench sonic eval`; `npa workbench fiftyone eval`; `npa workbench isaac-lab eval`; `npa workbench genesis eval-student`; `npa workbench golden-eval run`, `run-all`, `validate` |
| Robot policy    | `npa workbench lerobot train`, `eval`, `serve`, `infer`, `list-checkpoints`, `benchmark`, `profile-train`, `train-student`; `npa workbench groot download`, `finetune`, `eval`, `serve`, `infer`, `convert`; `npa workbench sonic train`, `serve`, `export`, `eval`, `status`, `list`                    |
| World models    | `npa workbench cosmos deploy`, `serve`, `infer`, `train`, `finetune`, `optimize`, `autoscale`, `status`, `system-info`                                                                                                                                                                                   |
| Zero-GPU LLM    | `npa workbench token-factory caption`, `generate`, `reason`, `verify`, `models`, `workflow`, `status`                                                                                                                                                                                                    |
| Workflows       | `npa workbench workflow validate-spec`, `plan-spec`, `run-spec`, `submit`; workbench workflows under [`npa-workflows/`](npa/workflows/workbench/npa-workflows/)                                                                                                                                           |
| Observability   | Tool-level `status`, `list`, and `system-info` commands; `npa workbench workflow status`, `logs`; `npa workbench health preflight`; `npa workbench foxglove convert-run`, `inspect`, `install-sdk`, `config`; `npa rerun host`, `share`, `list-shares`, `revoke`; `npa cluster status`, `list`                                                                                       |
| Platform utils  | `npa configure` / `init`, `npa provision-if-absent`; `npa agent`, `npa skypilot bootstrap/status/verify`, `npa soperator`, `npa burst`, `npa cluster`, `npa network`, `npa adapter convert`, `npa convert lerobot-to-rrd/-mp4`, `npa viz`, `npa demo`                                                    |

</details>

Full CLI reference: [docs/cli/README.md](docs/cli/README.md).

---

## Author and submit workflows

Author pipelines as declarative `npa.workflow/v0.0.1` specs — a state graph of
Workbench `toolRef` steps with S3 handoffs, gates, and loops. The same YAML is
what you validate, plan, and submit to the cluster.

```bash
# Validate and plan (no submit)
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml
npa workbench workflow plan-spec     npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml --run-id demo

# Launch on Nebius (after npa configure)
npa workbench workflow submit npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml \
  --run-id demo --registry cr.eu-north1.nebius.cloud/<your-registry-id>

# Inspect the plan without launching
npa workbench workflow submit npa/workflows/workbench/npa-workflows/token-factory-caption.yaml \
  --plan-only --run-id demo
```

|                        |                                                                 |
| ---------------------- | --------------------------------------------------------------- |
| **Format**             | `apiVersion: npa.workflow/v0.0.1`                               |
| **CLI**                | `validate-spec` · `plan-spec` · `run-spec` · `submit`           |
| **Workbench workflows** | [`npa/workflows/workbench/npa-workflows/`](npa/workflows/workbench/npa-workflows/) |
| **Tool catalog**       | [docs/workbench/npa-workflow-tool-catalog.md](docs/workbench/npa-workflow-tool-catalog.md) |
| **Authoring guide**    | [docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md) |

`submit` plans the graph and launches the run. Prefer these specs for new
pipelines. Parallel fan-out and a few specialized paths remain outside
`v0.0.1` scope — see the catalog README for exceptions.

The **Sim2Real 14-stage engine** is a separate path
([skills/workbench/sim2real-engine/SKILL.md](skills/workbench/sim2real-engine/SKILL.md))
using `sim2real/runbook.yaml` plus Python stage glue.

Architecture context:
[docs/architecture/contributor-context.md](docs/architecture/contributor-context.md).

---

## Container registry

Every Workbench tool ships as a container image in a Nebius container registry —
a primary in `eu-north1` and a mirror in `us-central1`. Resolve the registry
through `npa configure` or `npa.deploy.images`; never hardcode a registry id.
The publicly redistributable subset is also mirrored to GHCR for anonymous
external pulls.

```bash
# Log Docker into the registry (tokens expire; a 401 on pull means refresh)
REGISTRY_HOST=cr.eu-north1.nebius.cloud npa/scripts/nebius_registry_docker_login.sh

# Build and push an image with the canonical tag for its tool
npa/docker/workbench/lerobot/build.sh --registry "$NPA_REGISTRY" --push

# Pull a published image without Nebius registry credentials
export NPA_REGISTRY=ghcr.io/nebius/nebius-physical-ai
docker pull "${NPA_REGISTRY}/npa-retargeting:0.1.1"
```

| Reference | What it tells you |
| --- | --- |
| [Public Workbench image catalog](docs/workbench/container-image-catalog.md) | Exact GHCR image names, published tags, pull command, build dates, and intentional exclusions |
| [Image ↔ GPU compatibility matrix](docs/workbench/image-gpu-compatibility-matrix.md) | Every image against every Nebius GPU platform, and which cells are verified on real hardware |
| [Container packaging contract](docs/workbench/container-packaging.md) | Tiers, non-root users, ports, and redistribution classes each image must satisfy |
| [Container golden evals](docs/security/container-golden-evals.md) | The real capability test each image must pass — not an import probe |
| [Blackwell datacenter compatibility](docs/workbench/blackwell-datacenter-image-compatibility.md) | B200 / B300 build, tag, and validation runbook |
| [SONIC image catalog](docs/workbench/sonic-image-catalog.md) | Manifest-driven SONIC variant routing per GPU |
| [Image reproducibility](docs/security/image-reproducibility.md) | The two-tag strategy (`cuda12`, `cuda13-b300`) and how tags are pinned |

Every image declares a `redistribution` class in the packaging contract, which
decides whether it may leave the owning org. Public images may be mirrored to
GHCR; restricted images remain build-your-own in an operator-owned registry.
`cosmos3-serving` is currently restricted because its pinned base embeds a
runtime under NVIDIA's Deep Learning Container License. Set the class when you
add an image — the packaging-contract test fails a build that bakes a
non-redistributable runtime while claiming `public`.

---

## Validated on Nebius

Eight Workbench tools are validated end-to-end on Nebius today (LanceDB,
FiftyOne, LeRobot, Genesis, Isaac Lab, Cosmos, GR00T, SONIC). Track how each
tool scores across GPU tiers:

| Reference                                                                              | What it tells you                                                                     |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [Image ↔ GPU compatibility matrix](docs/workbench/image-gpu-compatibility-matrix.md)   | Every image × every Nebius GPU platform, with the verified cells called out            |
| [B300 validation matrix](docs/b300-validation-matrix.md)                               | Which tools have passed on B300 vs which are vendor-paced or upstream-blocked         |
| [LeRobot GPU benchmarks](docs/workbench/cookbooks/lerobot-gpu-benchmarks.md)           | Steps/s throughput across H200 · B300 · L40S · RTX Pro 6000 by policy type            |
| [NVIDIA architecture coverage](docs/nvidia-platform-architecture-coverage.md)          | CUDA 12.8 x86_64 vs CUDA 13 aarch64 tool coverage                                     |
| [NPA workflow tool catalog](docs/workbench/npa-workflow-tool-catalog.md)               | Every `toolRef` you can compose in an `npa.workflow/v0.0.1` spec                       |
| [Partner roadmap](docs/architecture/partner-skills-roadmap.md)                         | NVIDIA Omniverse / NuRec / CAD-to-SimReady capabilities on the way — not yet shipped   |

---

## How it runs

Workbench runs on Nebius infrastructure: S3-compatible object storage for
artifacts, managed Kubernetes and GPU runtimes for multi-stage jobs
(H100, H200, L40S, B300, RTX6000 — validated per tool), and vLLM-compatible
endpoints for serving.

User secrets live in `~/.npa/credentials.yaml`; machine-managed config lives
in `~/.npa/config.yaml`. The repo supports multiple top-level solution
namespaces; Workbench is the current primary solution (`npa.workbench` /
`npa workbench`). Future solutions are additive and never rename or nest
Workbench.

See [solutions model](docs/architecture/solutions-model.md) ·
[CLI namespaces](docs/architecture/cli-namespaces.md) ·
[contributor context](docs/architecture/contributor-context.md).

---

## Repository layout

```text
npa/                       # Python package (CLI + SDK); install with `pip install -e npa`
  src/npa/cli/             # Typer entry point and every top-level command
  src/npa/workbench/       # Per-tool implementations (cosmos, lerobot, sonic, ...)
  workflows/workbench/
    npa-workflows/         # Workbench npa.workflow/v0.0.1 specs (author + submit these)
    sim2real/              # Staged 14-stage sim2real runbook
docs/                      # Quickstart, architecture, workbench guides, cookbooks
skills/                    # SKILL.md files for agents and contributors (source of truth)
deploy/                    # Terraform + cluster provisioning (uses Nebius solutions library)
research/                  # LeRobot deploy research (older reference)
workbench/mlflow/          # MLflow tracking-server compose stack
```

More architectural detail: [docs/architecture/contributor-context.md](docs/architecture/contributor-context.md).

---

## Documentation

| Topic                | Where to look                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Install & auth       | [docs/quickstart.md](docs/quickstart.md)                                                               |
| Workbench setup     | [docs/workbench/getting-started.md](docs/workbench/getting-started.md)                                 |
| Beginner robot guides | [docs/workbench/guides/README.md](docs/workbench/guides/README.md)                                    |
| Cookbooks            | [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md) — includes the [BDD100K + LanceDB pipeline](docs/workbench/cookbooks/bdd100k-pipeline.md) and [Isaac-Lab BYOF](docs/workbench/cookbooks/byof-isaac-lab/) |
| Workflow authoring   | [docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md) · [tool catalog](docs/workbench/npa-workflow-tool-catalog.md) |
| Container images     | [Public Workbench image catalog](docs/workbench/container-image-catalog.md) · [packaging contract](docs/workbench/container-packaging.md) |
| `npa agent`          | [skills/tools/npa-agent/SKILL.md](skills/tools/npa-agent/SKILL.md) · [agent operate](skills/workflows/agent-fresh-operate/SKILL.md) |
| Preemptible GPU VMs | [docs/workbench/preemptible-vms.md](docs/workbench/preemptible-vms.md)                                 |
| Troubleshooting      | [docs/workbench/troubleshooting/known-footguns.md](docs/workbench/troubleshooting/known-footguns.md) · [active FIXMEs](FIXME.md) · [FTUE audit](FTUE-AUDIT.md) |
| CLI reference       | [docs/cli/README.md](docs/cli/README.md)                                                               |
| Architecture        | [solutions-model.md](docs/architecture/solutions-model.md) · [cli-namespaces.md](docs/architecture/cli-namespaces.md) · [contributor context](docs/architecture/contributor-context.md) |
| Everything else     | [docs/workbench/](docs/workbench/)                                                                     |

---

## Contributing

We welcome PRs, issues, and workflow contributions.

```bash
pip install -e "npa[dev]"
make test
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the review checklist,
skill-maintenance requirements, and repo hygiene rules. New behavior should
have a matching root `skills/` entry — see [`skills/index.yaml`](skills/index.yaml).
Security disclosures: [SECURITY.md](SECURITY.md). Support and community
happen through GitHub [Issues](https://github.com/nebius/nebius-physical-ai/issues)
and [Pull Requests](https://github.com/nebius/nebius-physical-ai/pulls).

---

## License

Licensed under the [Apache License 2.0](LICENSE). Built by
[Nebius](https://nebius.com) and the physical-AI community.
