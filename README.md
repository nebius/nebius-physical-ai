# Nebius Physical AI

Partners integrate independently. Teams assemble from open blueprints. Nebius
owns the infrastructure layer and compute substrate.

![Nebius Physical AI Workbench](docs/assets/workbench-architecture.png)

`npa` is the CLI and SDK for physical-AI workloads on Nebius. Workbench is the
primary solution: one command surface for data curation, simulation, synthetic
data, policy training, evaluation, export, observability, and SkyPilot
workflows on Nebius object storage, orchestration, vLLM serving, managed
Kubernetes, and GPU clusters.

## Quick Start

**Platforms:** macOS, Linux, and Windows (WSL2 Ubuntu). Native Windows shells
are not supported. Platform-specific install blocks:
[docs/quickstart.md § Fast install](docs/quickstart.md#fast-install-by-platform).

### 1. Install

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai

python3 -m venv .venv
source .venv/bin/activate   # Windows WSL: same; native Windows: not supported
pip install --upgrade pip
pip install -e npa

npa --version
```

Install the [Nebius CLI](https://docs.nebius.com/cli/install) when you connect
to the cloud (skip for the local try-it step below):

```bash
curl -fsSL https://storage.eu-north1.nebius.cloud/cli/install.sh | bash
export PATH="${HOME}/.nebius/bin:${PATH}"   # add to ~/.zshrc or ~/.bashrc
```

### 2. Try it locally (no cloud, no GPU, no credentials)

Score a shipped sample rollout set with the offline stub backend:

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

You should see a ranked report with `accuracy: 1.0`. Swap `--backend stub` for
`self-hosted` or `api` once credentials are configured.

### 3. Connect to Nebius (when you need cloud or GPU)

1. [Sign up](https://docs.nebius.com/signup-billing/sign-up) and create a
   [tenant and project](https://docs.nebius.com/iam/manage-projects).
2. Run interactive setup (creates or reuses your Nebius CLI profile, prompts
   for tenant, project, region, bucket, and optional API keys):

   ```bash
   npa configure --interactive
   ```

Account, bucket, and credential details:
[docs/quickstart.md](docs/quickstart.md). First Workbench workload:
[docs/workbench/getting-started.md](docs/workbench/getting-started.md).

**Zero-GPU inference:** [Nebius Token Factory](https://tokenfactory.nebius.com/)
needs only a `NEBIUS_TOKEN_FACTORY_KEY` — see
[docs/workbench/token-factory.md](docs/workbench/token-factory.md).

**Flagship GPU workload:** NVIDIA Cosmos (`npa workbench cosmos deploy/infer`)
— see [docs/quickstart.md § Cosmos](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos).

**Contributing:** `pip install -e "npa[dev]"` and `make test` — see
[CONTRIBUTING.md](CONTRIBUTING.md).

## Easy Guides

Short copy-paste walkthroughs — robot, simulation, and a public dataset.
Start with the no-GPU guide, then pick a robot:
[docs/workbench/guides/README.md](docs/workbench/guides/README.md).

| Guide | Robot | Sim / engine | Public dataset |
| --- | --- | --- | --- |
| [Score a robot in 60 seconds (no GPU)](docs/workbench/guides/score-a-robot-no-gpu.md) | any | offline | shipped sample rollouts |
| [Pick-and-place with a Franka arm](docs/workbench/guides/franka-pick-and-place-genesis.md) | Franka Emika Panda | Genesis | DROID (Franka) |
| [Teach a robot to push a T](docs/workbench/guides/pusht-sim-to-real.md) | sim pusher | sim-to-real loop | `lerobot/pusht` |
| [Train a Reachy 2 humanoid policy](docs/workbench/guides/reachy2-lerobot-policy.md) | Reachy 2 | LeRobot | Pollen Robotics / LeRobot Hub |
| [Make a Unitree G1 walk](docs/workbench/guides/g1-humanoid-walk-sonic.md) | Unitree G1 | MuJoCo | NVIDIA GEAR-SONIC |
| [Train a quadruped to run](docs/workbench/guides/quadruped-isaac-lab.md) | ANYmal / quadruped | Isaac Lab | Isaac Lab built-in tasks |

## Workbench

Workbench is the main product surface. Tools live under `npa workbench` (there is
no `solutions` CLI namespace).

| Category | Workbench commands |
| --- | --- |
| Data curation | `npa workbench fiftyone curate`, `eval`, `load-dataset`, `datasets list`; `npa workbench lancedb deploy`, `create-table`, `import-lerobot`, `import-bdd100k`, `backfill`, `create-mv`, `refresh-mv`, `query-table`, `query`; `npa workbench detection-training train`, `eval`, `status`, `list` |
| Synthetic data | `npa workbench cosmos infer`, `train`, `serve`, `status`; `npa workbench genesis generate-demos`; SkyPilot templates such as `npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml` and `npa/workflows/workbench/templates/curate-augment-train.yaml` |
| Simulation | `npa workbench isaac-lab train`, `eval`, `export-lerobot`; `npa workbench genesis train-teacher`, `generate-demos`, `eval-teacher`, `eval-student`, `diagnose`, `tune`; `npa workbench sonic retargeting run` |
| Eval | `npa workbench vlm-eval run`, `benchmark`, `workflow`, `status`, `list`; `npa workbench mjlab eval`; `npa workbench sonic eval`; `npa workbench fiftyone eval`; `npa workbench isaac-lab eval`; `npa workbench genesis eval-student` |
| Observability | Tool-level `status`, `list`, and `system-info` commands; `npa workbench workflow status`, `logs`; `npa rerun host`, `share`, `list-shares`, `revoke`; `npa cluster status`, `list` |
| Robot policy | `npa workbench lerobot train`, `eval`, `serve`, `infer`, `list-checkpoints`, `benchmark`, `profile-train`, `train-student`; `npa workbench groot download`, `finetune`, `eval`, `serve`, `infer`, `convert`; `npa workbench sonic train`, `serve`, `export`, `eval`, `status`, `list` |
| World models | `npa workbench cosmos deploy`, `serve`, `infer`, `train`, `status`, `system-info` |
| Blueprints | `npa workbench workflow submit`, `workflow trigger watch`, `status`, `logs`, `teardown`, `distill`; checked-in YAML under `npa/workflows/workbench/skypilot/` and `npa/workflows/workbench/sim2real/` |

**Highlights:** `vlm-eval` scores rollouts with stub, API, or self-hosted vLLM
backends — see `npa/workflows/workbench/skypilot/vlm-eval.yaml`.
`sonic export` converts locomotion checkpoints to ONNX;
`npa workbench workflow submit` runs SkyPilot YAML on Kubernetes or Nebius
with `--var KEY=VALUE`. SONIC image routing is manifest-driven — see
[docs/workbench/sonic-image-catalog.md](docs/workbench/sonic-image-catalog.md).

Declarative **NPA workflow** specs (`apiVersion: npa.workflow/v0.0.1`) — tool
chains, loops, and S3 artifact handoff — see
[docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md)
(golden YAMLs under `npa/workflows/workbench/npa-workflows/`).

CLI reference and cookbooks: [docs/cli/README.md](docs/cli/README.md),
[docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md).

## Solutions Framework

The repository supports multiple top-level solution namespaces. Workbench is the
current primary solution (`npa.workbench` / `npa workbench`). Future solutions
are additive and do not rename or nest Workbench.

See [docs/architecture/solutions-model.md](docs/architecture/solutions-model.md)
and [docs/architecture/cli-namespaces.md](docs/architecture/cli-namespaces.md).

## Nebius Cloud Substrate

Workbench runs on Nebius infrastructure: S3-compatible object storage for
artifacts, SkyPilot for multi-stage jobs, vLLM-compatible endpoints, and GPU
runtimes (H100, H200, L40S, B300, RTX6000 as validated per tool). User secrets
live in `~/.npa/credentials.yaml`; machine-managed config in
`~/.npa/config.yaml`.

## Docs

- [docs/quickstart.md](docs/quickstart.md) — install, Nebius auth, credentials
- [docs/workbench/npa-workflow-guide.md](docs/workbench/npa-workflow-guide.md) —
  authoring NPA workflow YAML (`npa.workflow/v0.0.1` state machines)
- [docs/workbench/getting-started.md](docs/workbench/getting-started.md) —
  Workbench setup and first workload
- [docs/workbench/preemptible-vms.md](docs/workbench/preemptible-vms.md) —
  preemptible (spot) GPU VMs — defaults, flags, and resume tips
- [docs/workbench/guides/README.md](docs/workbench/guides/README.md) —
  beginner-friendly robot guides
- [docs/workbench/](docs/workbench/) — guides, cookbooks, troubleshooting
- [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md) —
  end-to-end cookbooks, including the
  [BDD100K + LanceDB pipeline](docs/workbench/cookbooks/bdd100k-pipeline.md)
- [docs/cli/README.md](docs/cli/README.md) — generated CLI reference
- [docs/architecture/solutions-model.md](docs/architecture/solutions-model.md) —
  solution namespace model
- [docs/architecture/cli-namespaces.md](docs/architecture/cli-namespaces.md) —
  CLI namespace conventions
- [CONTRIBUTING.md](CONTRIBUTING.md) — contribution guidelines
- [LICENSE](LICENSE) — Apache License 2.0
