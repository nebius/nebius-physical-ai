# Nebius Physical AI

Partners integrate independently. Teams assemble from open blueprints. Nebius
owns the infrastructure layer and compute substrate.

![Nebius Physical AI Workbench](docs/assets/workbench-architecture.png)

`npa` is the CLI and SDK for physical-AI workloads on Nebius. Workbench is the
primary solution: it gives developers one command surface for data curation,
simulation, synthetic data, policy training, evaluation, export, observability,
and SkyPilot workflows running on the Nebius substrate of object storage,
orchestration, vLLM serving, managed Kubernetes, and GPU clusters.

## Quick Start

Install the package from the `npa/` Python project:

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai

python3 -m venv npa/.venv
npa/.venv/bin/python -m pip install --upgrade pip
npa/.venv/bin/python -m pip install -e npa
export PATH="$PWD/npa/.venv/bin:$PATH"
```

Authenticate with the Nebius CLI and let `npa` print the local credential
schema for optional Hugging Face, NGC, object-storage, and BYOVM SSH values:

```bash
nebius profile create
nebius iam get-access-token >/dev/null
npa configure
```

Run a first Workbench command without provisioning infrastructure:

```bash
npa workbench vlm-eval list
npa workbench vlm-eval run \
  --input-path ./rollout.json \
  --output-path ./eval.json \
  --backend stub \
  --score 0.9 \
  --dry-run \
  --output json
```

For full cloud setup, continue with [docs/quickstart.md](docs/quickstart.md)
and [docs/workbench/getting-started.md](docs/workbench/getting-started.md).

## Workbench

Workbench is the main product surface in this repository. Current Workbench
tools are mounted directly under `npa workbench`; there is no `solutions` CLI
namespace.

| Category | Workbench commands |
| --- | --- |
| Data curation | `npa workbench data sync`, `npa workbench data status`, `npa workbench data list`; `npa workbench fiftyone curate`, `eval`, `load-dataset`, `datasets list`; `npa workbench lancedb deploy`, `create-table`, `import-lerobot`, `import-bdd100k`, `backfill`, `create-mv`, `refresh-mv`, `query-table`, `query`; `npa workbench detection-training train`, `eval`, `status`, `list` |
| Synthetic data | `npa workbench cosmos infer`, `train`, `serve`, `status`; `npa workbench genesis generate-demos`; SkyPilot templates such as `npa/workflows/workbench/skypilot/bdd100k-pipeline.yaml` and `npa/workflows/workbench/templates/curate-augment-train.yaml` |
| Simulation | `npa workbench isaac-lab train`, `eval`, `export-lerobot`; `npa workbench genesis train-teacher`, `generate-demos`, `eval-teacher`, `eval-student`, `diagnose`, `tune`; `npa workbench retargeting run` |
| Eval | `npa workbench vlm-eval run`, `benchmark`, `workflow`, `status`, `list`; `npa workbench mjlab eval`; `npa workbench sonic eval`; `npa workbench fiftyone eval`; `npa workbench isaac-lab eval`; `npa workbench genesis eval-student` |
| Observability | Tool-level `status`, `list`, and `system-info` commands; `npa workbench workflow status`, `logs`; `npa rerun host`, `share`, `list-shares`, `revoke`; `npa cluster status`, `list` |
| Robot policy | `npa workbench lerobot train`, `eval`, `serve`, `infer`, `list-checkpoints`, `benchmark`, `profile-train`, `train-student`; `npa workbench groot download`, `finetune`, `eval`, `serve`, `infer`, `convert`; `npa workbench sonic train`, `serve`, `export`, `eval`, `status`, `list` |
| World models | `npa workbench cosmos deploy`, `serve`, `infer`, `train`, `status`, `system-info` |
| Blueprints | `npa workbench workflow submit`, `run`, `status`, `logs`, `teardown`, `distill`; checked-in YAML under `npa/workflows/workbench/skypilot/` for Isaac Lab, VLM eval, SONIC export, SONIC eval, SONIC locomotion fine-tuning, retargeting, MJLab eval, sim-to-real, and BDD100K pipelines |

### Eval: VLM Backend

`vlm-eval` is a first-class Eval capability. It scores rollout artifacts with
self-hosted, API, or stub backends and has a checked-in SkyPilot template at
`npa/workflows/workbench/skypilot/vlm-eval.yaml`. The benchmark command sweeps a
labeled rollout set across thresholds, rubrics, and models, then writes a ranked
accuracy report with the best config.

```bash
npa workbench vlm-eval list
npa workbench vlm-eval status
npa workbench vlm-eval workflow
npa workbench vlm-eval run \
  --input-path ./rollout.json \
  --output-path ./eval.json \
  --backend stub \
  --score 0.9 \
  --dry-run
npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct
```

The self-hosted workflow starts an OpenAI-compatible vLLM server and then calls
`npa workbench vlm-eval run`; the benchmark workflow does the same for
`npa workbench vlm-eval benchmark`.

### Robot Policy: GR00T, LeRobot, and SONIC

Robot policy work is split across policy training/serving, humanoid foundation
model operations, whole-body control, and export:

```bash
npa workbench lerobot train --help
npa workbench lerobot serve --help
npa workbench groot finetune --help
npa workbench groot serve --help
npa workbench sonic train --help
npa workbench sonic export --help
npa workbench sonic eval --help
```

`sonic export` is a first-class Robot Policy model-export capability. It
converts a trained SONIC locomotion checkpoint to a deterministic-action ONNX
graph:

```bash
npa workbench sonic export \
  --checkpoint sonic_release/last.pt \
  --output exported/sonic_policy.onnx
```

The matching workflow template is
`npa/workflows/workbench/skypilot/sonic-export.yaml`.
`npa workbench sonic eval` consumes the exported ONNX and sidecar metadata and
can run the built-in reference backend or a configured eval container. The
checked-in SkyPilot template is
`npa/workflows/workbench/skypilot/sonic-eval.yaml`.

### Workflows And Routing

Workbench workflow orchestration lives under the Workbench solution:

```bash
npa workbench workflow submit npa/workflows/workbench/skypilot/vlm-eval.yaml --run-id vlm-eval
npa workbench workflow submit npa/workflows/workbench/skypilot/sonic-export.yaml --run-id sonic-export
npa workbench workflow submit npa/workflows/workbench/skypilot/sonic-eval.yaml --run-id sonic-eval
npa workbench workflow run distill --local
npa workbench workflow status run-1
npa workbench workflow logs run-1 train_student
```

`submit` sends SkyPilot YAML through the NPA controller convention and supports
`--controller-backend kubernetes`, `--controller-backend nebius`, `--run-id`,
and repeated `--var KEY=VALUE` substitutions.

SONIC image routing is manifest-driven:

- `npa workbench sonic train` resolves the first-party image from
  `npa/src/npa/deploy/sonic_image_manifest.json` using `--gpu-type`, with
  `--image` and `--image-variant` available as explicit overrides.
- L40S VM targets use the baked `npa-sonic:0.1.2` image. RTX PRO 6000
  Blackwell Kubernetes targets use the host-mounted `npa-sonic:0.1.2-k8s`
  image. See `docs/workbench/sonic-image-catalog.md`.

### Solution Patterns

Workbench tools share the same platform patterns:

- Object-storage handoff through S3-style `--input-path` and `--output-path`
  values, with `~/.npa/credentials.yaml` as the user-authored credential file.
- SkyPilot workflows checked into `npa/workflows/workbench/` and submitted with
  `npa workbench workflow submit`.
- vLLM-compatible self-hosted serving for VLM eval and model-serving paths where
  the runtime exposes an OpenAI-compatible endpoint.
- Lifecycle commands that keep deploy, status, list, run, train, eval, serve,
  infer, export, and system-info behavior predictable across tools.

## Solutions Framework

The repository supports multiple top-level solution namespaces. Workbench is the
current primary solution and is implemented as the top-level SDK namespace
`npa.workbench` and the CLI namespace `npa workbench`.

Future solutions are additive: a datalake or simfarm solution would sit beside
Workbench as another top-level `npa` namespace. Future solutions should not
rename Workbench, move Workbench under a `solutions` namespace, or require users
to change existing `npa workbench` commands.

## Nebius Cloud Substrate

Workbench runs on Nebius infrastructure rather than hiding it:

- Object storage is the data layer for datasets, checkpoints, rollouts, eval
  JSON, exported models, and Rerun recordings.
- OSMO and SkyPilot provide orchestration patterns for multi-stage jobs and
  managed Kubernetes backed workflows.
- vLLM-compatible endpoints support shared model-serving and Eval backends.
- Managed Kubernetes, VM, BYOVM, container, and serverless runtimes cover GPU
  targets including H100, H200, L40S, B300, and RTX6000 profiles as each tool is
  validated.
- Nebius CLI authentication, IAM, local `~/.npa/credentials.yaml`, and
  machine-managed `~/.npa/config.yaml` keep user secrets separate from project,
  workbench, endpoint, SSH, storage, and Terraform state metadata.

## Docs And Contributing

- [docs/quickstart.md](docs/quickstart.md): install, Nebius auth, and
  credential setup.
- [docs/workbench/getting-started.md](docs/workbench/getting-started.md):
  Workbench setup and first workload path.
- [docs/workbench/](docs/workbench/): Workbench guides, cookbooks, and
  troubleshooting.
- [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md):
  end-to-end cookbooks, including the
  [BDD100K + LanceDB pipeline](docs/workbench/cookbooks/bdd100k-pipeline.md).
- [docs/cli/README.md](docs/cli/README.md): generated CLI reference.
- [docs/architecture/solutions-model.md](docs/architecture/solutions-model.md):
  solution namespace model.
- [docs/architecture/cli-namespaces.md](docs/architecture/cli-namespaces.md):
  CLI namespace conventions.
- [CONTRIBUTING.md](CONTRIBUTING.md): contribution guidelines.
- [LICENSE](LICENSE): Apache License 2.0.
