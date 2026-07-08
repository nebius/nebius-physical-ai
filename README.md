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

## Get started in 60 seconds — no cloud, no GPU, no credentials

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai
python3 -m venv .venv && source .venv/bin/activate
pip install --upgrade pip && pip install -e npa

npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

You should see a ranked report with `accuracy: 1.0`. Swap `--backend stub`
for `self-hosted` or `api` once credentials are configured.

---

## Full quick start

Platform-specific install blocks: [docs/quickstart.md § Fast install](docs/quickstart.md#fast-install-by-platform).
Supported hosts: macOS, Linux, and Windows via **WSL2 Ubuntu**. Native Windows
shells (PowerShell, `cmd`) are not supported.

### 1. Install `npa`

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa

npa --version
```

### 2. Install the Nebius CLI (only for cloud steps)

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc or ~/.bashrc
```

### 3. Connect to Nebius

1. [Sign up](https://docs.nebius.com/signup-billing/sign-up) and create a
   [tenant and project](https://docs.nebius.com/iam/manage-projects).
2. Run interactive setup — creates or reuses your Nebius CLI profile and
   prompts for tenant, project, region, bucket, and optional API keys:

   ```bash
   npa configure --interactive
   ```

More: [account and credentials](docs/quickstart.md) ·
[first Workbench workload](docs/workbench/getting-started.md) ·
[preemptible GPU VMs](docs/workbench/preemptible-vms.md).

**Zero-GPU inference:** [Nebius Token Factory](https://tokenfactory.nebius.com/)
needs only a `NEBIUS_TOKEN_FACTORY_KEY` — see
[docs/workbench/token-factory.md](docs/workbench/token-factory.md).

**Flagship GPU workload:** NVIDIA Cosmos (`npa workbench cosmos deploy/infer`)
— see [docs/quickstart.md § Cosmos](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos).

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

---

## Workbench at a glance

Workbench is the main product surface. Every tool lives under `npa workbench`
(there is no `solutions` CLI namespace). Highlights:

- **`vlm-eval`** scores rollouts with stub, API, or self-hosted vLLM backends —
  see [`vlm-eval.yaml`](npa/workflows/workbench/skypilot/vlm-eval.yaml).
- **`sonic export`** converts locomotion checkpoints to ONNX.
- **`npa workbench workflow submit`** runs any SkyPilot YAML on Kubernetes or
  Nebius with `--var KEY=VALUE`.
- **SONIC image routing** is manifest-driven — see
  [sonic-image-catalog.md](docs/workbench/sonic-image-catalog.md).
- **Declarative `npa.workflow/v0.0.1` specs** — tool chains, loops, and S3
  artifact handoff. See the [NPA workflow guide](docs/workbench/npa-workflow-guide.md)
  and golden YAMLs under [`npa/workflows/workbench/npa-workflows/`](npa/workflows/workbench/npa-workflows/).

<details>
<summary><strong>Browse the full command inventory by category</strong></summary>

| Category         | Workbench commands                                                                                                                                                                                                                                                                                     |
| ---------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Data curation    | `npa workbench fiftyone curate`, `eval`, `load-dataset`, `datasets list`; `npa workbench lancedb deploy`, `create-table`, `import-lerobot`, `import-bdd100k`, `backfill`, `create-mv`, `refresh-mv`, `query-table`, `query`; `npa workbench detection-training train`, `eval`, `status`, `list`         |
| Synthetic data   | `npa workbench cosmos infer`, `train`, `serve`, `status`; `npa workbench genesis generate-demos`; SkyPilot templates such as [`bdd100k-pipeline.yaml`](npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml) and [`curate-augment-train.yaml`](npa/workflows/workbench/templates/curate-augment-train.yaml) |
| Simulation      | `npa workbench isaac-lab train`, `eval`, `export-lerobot`; `npa workbench genesis train-teacher`, `generate-demos`, `eval-teacher`, `eval-student`, `diagnose`, `tune`; `npa workbench sonic retargeting run`                                                                                             |
| Eval            | `npa workbench vlm-eval run`, `benchmark`, `workflow`, `status`, `list`; `npa workbench mjlab eval`; `npa workbench sonic eval`; `npa workbench fiftyone eval`; `npa workbench isaac-lab eval`; `npa workbench genesis eval-student`                                                                     |
| Observability   | Tool-level `status`, `list`, and `system-info` commands; `npa workbench workflow status`, `logs`; `npa rerun host`, `share`, `list-shares`, `revoke`; `npa cluster status`, `list`                                                                                                                       |
| Robot policy    | `npa workbench lerobot train`, `eval`, `serve`, `infer`, `list-checkpoints`, `benchmark`, `profile-train`, `train-student`; `npa workbench groot download`, `finetune`, `eval`, `serve`, `infer`, `convert`; `npa workbench sonic train`, `serve`, `export`, `eval`, `status`, `list`                    |
| World models    | `npa workbench cosmos deploy`, `serve`, `infer`, `train`, `status`, `system-info`                                                                                                                                                                                                                       |
| Blueprints      | `npa workbench workflow submit`, `workflow trigger watch`, `status`, `logs`, `teardown`, `distill`; checked-in YAML under [`skypilot/`](npa/workflows/workbench/skypilot/) and [`sim2real/`](npa/workflows/workbench/sim2real/)                                                                          |

</details>

Full CLI reference and end-to-end recipes: [docs/cli/README.md](docs/cli/README.md),
[docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md).

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
[CLI namespaces](docs/architecture/cli-namespaces.md).

---

## Documentation

| Topic                | Where to look                                                                                          |
| -------------------- | ------------------------------------------------------------------------------------------------------ |
| Install & auth       | [docs/quickstart.md](docs/quickstart.md)                                                               |
| Workbench setup     | [docs/workbench/getting-started.md](docs/workbench/getting-started.md)                                 |
| Beginner robot guides | [docs/workbench/guides/README.md](docs/workbench/guides/README.md)                                    |
| Cookbooks            | [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md) — includes the [BDD100K + LanceDB pipeline](docs/workbench/cookbooks/bdd100k-pipeline.md) |
| Workflow authoring   | [docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md) (`npa.workflow/v0.0.1` state machines) |
| Preemptible GPU VMs | [docs/workbench/preemptible-vms.md](docs/workbench/preemptible-vms.md)                                 |
| CLI reference       | [docs/cli/README.md](docs/cli/README.md)                                                               |
| Architecture        | [solutions-model.md](docs/architecture/solutions-model.md) · [cli-namespaces.md](docs/architecture/cli-namespaces.md) |
| Everything else     | [docs/workbench/](docs/workbench/)                                                                     |

---

## Contributing

We welcome PRs, issues, and workflow contributions.

```bash
pip install -e "npa[dev]"
make test
```

Read [CONTRIBUTING.md](CONTRIBUTING.md) for the review checklist,
skill-maintenance requirements, and repo hygiene rules. Security disclosures:
[SECURITY.md](SECURITY.md).

---

## License

Licensed under the [Apache License 2.0](LICENSE). Built by
[Nebius](https://nebius.com) and the physical-AI community.
