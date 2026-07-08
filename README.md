<div align="center">

# Nebius Physical AI

**One CLI, one SDK, one workflow layer for physical-AI workloads on Nebius —
data curation, simulation, synthetic data, policy training, evaluation,
observability, and SkyPilot orchestration.**

<img src="docs/assets/workbench-architecture.png" alt="Nebius Physical AI Workbench architecture" width="820" />

[![License: Apache 2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![Platforms: macOS · Linux · WSL2](https://img.shields.io/badge/platforms-macOS%20%7C%20Linux%20%7C%20WSL2-lightgrey.svg)](docs/quickstart.md#fast-install-by-platform)
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
SkyPilot workflows on Nebius object storage, orchestration, vLLM-compatible
serving, managed Kubernetes, and GPU clusters (H100 · H200 · L40S · B300 ·
RTX6000).

> Partners integrate independently. Teams assemble from open blueprints.
> Nebius owns the infrastructure layer and compute substrate.

|                                   |                                                                                       |
| --------------------------------- | ------------------------------------------------------------------------------------- |
| **What you can do**               | Curate datasets · train and evaluate policies · render synthetic data · run sim-to-real loops · serve models |
| **Who it's for**                  | Robotics teams, physical-AI researchers, and partners shipping on Nebius              |
| **Where it runs**                 | Nebius S3, SkyPilot, managed Kubernetes, GPU clusters; local for stubs and unit tests |
| **How you extend it**             | Declarative `npa.workflow/v0.0.1` YAML specs and reusable Workbench tool refs         |

---

## Pick your path

Three onboarding routes, in ascending order of setup effort. Every one starts
with the same install.

| Route                                              | Time         | Needs                                    | You end up with                                          |
| -------------------------------------------------- | ------------ | ---------------------------------------- | -------------------------------------------------------- |
| **A. [60-second try-it](#a-60-second-try-it)**     | ~60 s        | Python 3.10+                             | A scored VLM benchmark report — no cloud, no GPU, no key |
| **B. [First workload on Nebius](#b-first-workload-on-nebius)** | ~15 min | Nebius account · `nebius` CLI            | A configured project running managed Workbench tools     |
| **C. [Self-hosted `npa agent`](#c-self-hosted-npa-agent)** | ~20 min | Nebius account · Terraform · SSH key    | A browser-based chat workbench VM with embedded Rerun    |

All three share the same one-time install:

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e npa
npa --version
```

> **Windows:** use **WSL2 Ubuntu**. Native PowerShell / `cmd` are not
> supported. Platform-specific install blocks:
> [docs/quickstart.md § Fast install](docs/quickstart.md#fast-install-by-platform).

---

### A. 60-second try-it

Score a shipped sample rollout set with the offline stub backend — no
credentials of any kind:

```bash
npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

You should see a ranked report with `accuracy: 1.0`.

Want to see the **declarative workflow layer** without any cloud either?
Validate and plan a real `npa.workflow/v0.0.1` spec offline:

```bash
npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml
npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/vlm-eval-single.yaml --run-id demo
```

The spec is 33 lines and looks like this — every Workbench tool has a
`toolRef` you can chain, loop, or gate on:

```yaml
apiVersion: npa.workflow/v0.0.1
kind: Workflow
metadata:
  name: vlm-eval-single
config:
  bucket: example-bucket
  prefix: "runs/{{run.id}}/vlm-eval"
  rollouts_uri: "s3://{{config.bucket}}/{{config.prefix}}/rollouts/"
  scores_uri: "s3://{{config.bucket}}/{{config.prefix}}/scores/"
resources:
  gpu:
    cloud: kubernetes
    accelerators: H100:1
initial: score-rollouts
states:
  score-rollouts:
    toolRef: workbench.vlm_eval.run
    resources: gpu
    outputs:
      - uri: "{{config.scores_uri}}report.json"
    terminal: true
```

Author, validate, plan, and run guide: [docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md).
More golden specs: [`npa/workflows/workbench/npa-workflows/`](npa/workflows/workbench/npa-workflows/).

---

### B. First workload on Nebius

1. [Sign up](https://docs.nebius.com/signup-billing/sign-up) and create a
   [tenant and project](https://docs.nebius.com/iam/manage-projects).
2. Install the Nebius CLI (only needed for cloud steps):

   ```bash
   curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
   export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc or ~/.bashrc
   ```

3. Interactive setup — creates or reuses your Nebius CLI profile and prompts
   for tenant, project, region, bucket, and optional API keys:

   ```bash
   npa configure --interactive
   ```

Now you're ready for [docs/workbench/getting-started.md](docs/workbench/getting-started.md).
Full account/credential detail: [docs/quickstart.md](docs/quickstart.md).

**Zero-GPU inference:** [Nebius Token Factory](https://tokenfactory.nebius.com/)
needs only a `NEBIUS_TOKEN_FACTORY_KEY` — see
[docs/workbench/token-factory.md](docs/workbench/token-factory.md). This is
the cheapest way to try large models against your own data.

**Flagship GPU workload:** NVIDIA Cosmos (`npa workbench cosmos deploy/infer`)
— see [docs/quickstart.md § Cosmos](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos).

---

### C. Self-hosted `npa agent`

`npa agent` is a self-hosted **browser workbench VM**: HTTPS UI with
basic-auth login, grounded chat over Nebius Token Factory
(default `nvidia/Cosmos3-Super-Reasoner`), Sim Assets + Cameras panels, an
embedded [Rerun](https://www.rerun.io) viewer for `.rrd` recordings, and
draft/validate/plan/submit endpoints for `npa.workflow/v0.0.1` specs.

```bash
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

---

## Before you burn GPU-hours — preflight

A short list of things that catch first-time users mid-run. Skim before your
first GPU submit.

- **Run preflight.** `npa workbench health sim2real` is a single
  PASS/WARN/FAIL/SKIP check over config, coherence, S3, registry, tokens,
  and cluster. See [FTUE-AUDIT.md § friction 1](FTUE-AUDIT.md#friction-points-ordered).
- **GPU routing matters.** Isaac Lab needs an **RT-core** GPU (L40S / RTX
  Pro 6000), not an H100. See [docs/workbench/troubleshooting/known-footguns.md § L40S Capacity](docs/workbench/troubleshooting/known-footguns.md#l40s-capacity-is-on-demand-zero).
- **Registry pull secrets expire silently.** A `401` on image pull usually
  means the `npa-nebius-registry` pull secret needs refreshing. See
  [known-footguns.md § Registry Pull Secret](docs/workbench/troubleshooting/known-footguns.md#registry-pull-secret-expires-silently).
- **SkyPilot 0.12.2 does not interpolate `${VAR}` inside `envs` / `image_id`.**
  Use `npa workbench workflow submit` (or NPA runners); the materialized-runbook
  or direct-Kubernetes path is documented in
  [FTUE-AUDIT.md § friction 4](FTUE-AUDIT.md#friction-points-ordered).
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

Short copy-paste walkthroughs. Start with the no-GPU guide, then pick any
robot or simulator. Full index:
[docs/workbench/guides/README.md](docs/workbench/guides/README.md).

| Guide                                                                                             | Robot                | Sim / engine     | Public dataset                     |
| ------------------------------------------------------------------------------------------------- | -------------------- | ---------------- | ---------------------------------- |
| [Score a robot in 60 seconds (no GPU)](docs/workbench/guides/score-a-robot-no-gpu.md)             | any                  | offline          | shipped sample rollouts            |
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

- **`vlm-eval`** scores rollouts with stub, API, or self-hosted vLLM backends —
  see [`vlm-eval.yaml`](npa/workflows/workbench/skypilot/vlm-eval.yaml).
- **`token-factory`** wraps Nebius Token Factory for zero-GPU inference,
  captioning, and reasoning against your own frames.
- **`health`** runs preflight checks before a Sim2Real submit.
- **`sonic export`** converts locomotion checkpoints to ONNX.
- **`workflow submit`** runs any SkyPilot YAML on Kubernetes or Nebius with
  `--var KEY=VALUE`; **`workflow validate-spec`** / **`plan-spec`** /
  **`run-spec`** operate on declarative `npa.workflow/v0.0.1` specs.
- **`trigger`** watches S3-compatible prefixes and retriggers workflows
  automatically.
- **`golden-eval`** runs per-container hello-world reruns as a CI gate.
- SONIC image routing is manifest-driven — see
  [sonic-image-catalog.md](docs/workbench/sonic-image-catalog.md).

<details>
<summary><strong>Browse the full command inventory by category</strong></summary>

| Category         | Workbench commands                                                                                                                                                                                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data curation    | `npa workbench fiftyone curate`, `eval`, `load-dataset`, `datasets list`; `npa workbench lancedb deploy`, `create-table`, `import-lerobot`, `import-bdd100k`, `backfill`, `create-mv`, `refresh-mv`, `query-table`, `query`; `npa workbench detection-training train`, `eval`, `status`, `list`         |
| Synthetic data   | `npa workbench cosmos infer`, `train`, `serve`, `status`; `npa workbench cosmos2 transfer`; `npa workbench cosmos3 reason`; `npa workbench genesis generate-demos`; SkyPilot templates such as [`bdd100k-pipeline.yaml`](npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml) and [`curate-augment-train.yaml`](npa/workflows/workbench/templates/curate-augment-train.yaml) |
| Simulation      | `npa workbench isaac-lab train`, `eval`, `export-lerobot`, `export-onnx`; `npa workbench genesis train-teacher`, `generate-demos`, `eval-teacher`, `eval-student`, `diagnose`, `tune`; `npa workbench sonic retargeting run`, `workflow`                                                                    |
| Eval            | `npa workbench vlm-eval run`, `benchmark`, `workflow`, `status`, `list`; `npa workbench mjlab eval`, `workflow`; `npa workbench sonic eval`; `npa workbench fiftyone eval`; `npa workbench isaac-lab eval`; `npa workbench genesis eval-student`; `npa workbench golden-eval run`, `run-all`, `validate` |
| Robot policy    | `npa workbench lerobot train`, `eval`, `serve`, `infer`, `list-checkpoints`, `benchmark`, `profile-train`, `train-student`; `npa workbench groot download`, `finetune`, `eval`, `serve`, `infer`, `convert`; `npa workbench sonic train`, `serve`, `export`, `eval`, `status`, `list`                    |
| World models    | `npa workbench cosmos deploy`, `serve`, `infer`, `train`, `finetune`, `optimize`, `autoscale`, `status`, `system-info`                                                                                                                                                                                   |
| Zero-GPU LLM    | `npa workbench token-factory caption`, `generate`, `reason`, `verify`, `models`, `workflow`, `status`                                                                                                                                                                                                    |
| Blueprints      | `npa workbench workflow submit`, `run-spec`, `validate-spec`, `plan-spec`, `trigger watch`, `status`, `logs`, `artifacts`, `list`, `teardown`, `distill`; checked-in YAML under [`skypilot/`](npa/workflows/workbench/skypilot/), [`npa-workflows/`](npa/workflows/workbench/npa-workflows/), and [`sim2real/`](npa/workflows/workbench/sim2real/) |
| Observability   | Tool-level `status`, `list`, and `system-info` commands; `npa workbench workflow status`, `logs`; `npa workbench health sim2real`; `npa rerun host`, `share`, `list-shares`, `revoke`; `npa cluster status`, `list`                                                                                       |
| Platform utils  | `npa configure` / `init`, `npa provision-if-absent`; `npa agent`, `npa skypilot bootstrap/status/verify`, `npa soperator`, `npa burst`, `npa cluster`, `npa network`, `npa adapter convert`, `npa convert lerobot-to-rrd/-mp4`, `npa viz`, `npa demo`                                                    |

</details>

Full CLI reference: [docs/cli/README.md](docs/cli/README.md).

---

## Validated on Nebius

Eight Workbench tools are validated end-to-end on Nebius today (LanceDB,
FiftyOne, LeRobot, Genesis, Isaac Lab, Cosmos, GR00T, SONIC). Track how each
tool scores across GPU tiers:

| Reference                                                                              | What it tells you                                                                     |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------- |
| [B300 validation matrix](docs/b300-validation-matrix.md)                               | Which tools have passed on B300 vs which are vendor-paced or upstream-blocked         |
| [LeRobot GPU benchmarks](docs/workbench/cookbooks/lerobot-gpu-benchmarks.md)           | Steps/s throughput across H200 · B300 · L40S · RTX Pro 6000 by policy type            |
| [NVIDIA architecture coverage](docs/nvidia-platform-architecture-coverage.md)          | CUDA 12.8 x86_64 vs CUDA 13 aarch64 tool coverage                                     |
| [NPA workflow tool catalog](docs/workbench/npa-workflow-tool-catalog.md)               | Every `toolRef` you can compose in an `npa.workflow/v0.0.1` spec                       |
| [Partner roadmap](docs/architecture/partner-skills-roadmap.md)                         | NVIDIA Omniverse / NuRec / CAD-to-SimReady capabilities on the way — not yet shipped   |

---

## How it runs

Workbench runs on Nebius infrastructure: S3-compatible object storage for
artifacts, SkyPilot for multi-stage jobs, vLLM-compatible endpoints, and GPU
runtimes (H100, H200, L40S, B300, RTX6000 — validated per tool).

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
    skypilot/              # Reference SkyPilot YAMLs
    npa-workflows/         # Golden npa.workflow/v0.0.1 specs
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
