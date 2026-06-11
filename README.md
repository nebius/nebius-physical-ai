# Nebius Physical AI

**One command surface for robotics, simulation, perception, and synthetic-data
workloads — running on Nebius GPUs.**

`npa` is the CLI and SDK for physical-AI workloads on Nebius. **Workbench is the
primary solution**: data curation, simulation, synthetic data, policy training,
evaluation, export, and observability, all driven by `npa workbench` commands
and reproducible SkyPilot workflows on the Nebius substrate (object storage,
managed Kubernetes, vLLM serving, and GPU clusters).

![Nebius Physical AI Workbench](docs/assets/workbench-architecture.png)

---

## ⚡ Try it in 60 seconds — no cloud, no GPU, no credentials

Install `npa` into a fresh virtual environment (Python 3.10+), then score a
shipped sample rollout set with the offline `stub` backend:

```bash
git clone https://github.com/nebius/nebius-physical-ai.git
cd nebius-physical-ai

python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -e npa

npa workbench vlm-eval benchmark \
  --dataset npa/src/npa/workbench/vlm_eval/fixtures/sample_benchmark/benchmark.json \
  --output /tmp/vlm-eval-benchmark.json \
  --backend stub \
  --thresholds 0.5,0.8,0.9 \
  --rubrics default,strict \
  --models Qwen/Qwen2-VL-7B-Instruct \
  --format json
```

You should see a ranked report with `accuracy: 1.0` over four labeled rollouts.
That is the full local loop — the same command swaps `--backend stub` for a real
`self-hosted` or `api` VLM backend once you add credentials.

---

## 🧭 Find your workflow

**The fastest way to navigate this repo is the workflow catalog:**

### → [Workflow Catalog](npa/workflows/workbench/skypilot/README.md)

It has a single "I want to…" table that maps every goal to the workflow YAML,
the command to run it, the GPU it needs, and its guide. A few common entry
points:

| I want to… | Run | Guide |
| --- | --- | --- |
| Score robot rollouts with a VLM | `npa workbench vlm-eval` | [vlm-eval loop](docs/workbench/cookbooks/vlm-eval-loop-runbook.md) |
| Generate synthetic frames with Cosmos | `npa workbench cosmos` | [quickstart](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos) |
| Curate + train an AV perception model | [`run_bdd100k_pipeline.py`](npa/scripts/run_bdd100k_pipeline.py) | [BDD100K pipeline](docs/workbench/cookbooks/bdd100k-pipeline.md) |
| Run a sim-to-real train → eval loop | [`run_sim_to_real_pipeline.py`](npa/scripts/run_sim_to_real_pipeline.py) | [sim-to-real](docs/workbench/cookbooks/sim-to-real-pipeline.md) |
| Train / fine-tune a robot policy | `npa workbench sonic`, `lerobot`, `groot` | [cookbooks](docs/workbench/cookbooks/README.md) |
| Train an Isaac Lab RL policy | [`run_isaac_lab_rl.py`](npa/scripts/run_isaac_lab_rl.py) | [Isaac Lab BYOF](docs/workbench/cookbooks/byof-isaac-lab/README.md) |

Every workflow YAML lives in
[`npa/workflows/workbench/skypilot/`](npa/workflows/workbench/skypilot/); the
catalog maps each one to the command that runs it and its guide.

---

## 🗺️ Where do I go? (by audience)

| You are a… | Start here |
| --- | --- |
| **Salesperson / evaluator** | [Try it in 60 seconds](#-try-it-in-60-seconds--no-cloud-no-gpu-no-credentials), then the [Workflow Catalog](npa/workflows/workbench/skypilot/README.md) to see what the platform does |
| **Customer running a first workload** | [docs/workbench/getting-started.md](docs/workbench/getting-started.md) |
| **Developer building pipelines** | [Workflow Catalog](npa/workflows/workbench/skypilot/README.md) + [docs/workbench-yaml-guide.md](docs/workbench-yaml-guide.md) |
| **SDK integrator or agent author** | [docs/workbench/cli-sdk-yaml-walkthrough.md](docs/workbench/cli-sdk-yaml-walkthrough.md), [docs/sdk/errors.md](docs/sdk/errors.md) |
| **Contributor to `npa` itself** | [CONTRIBUTING.md](CONTRIBUTING.md) |

---

## 🛠️ Workbench tool surface

Workbench tools are mounted directly under `npa workbench` (there is no
`solutions` CLI namespace). Tools share the same lifecycle verbs — `deploy`,
`status`, `list`, `run`, `train`, `eval`, `serve`, `infer`, `export`,
`system-info` — and hand off data through S3-style `--input-path` /
`--output-path` values.

| Category | Workbench commands |
| --- | --- |
| **Data curation** | `npa workbench data sync/status/list`; `fiftyone curate/eval/load-dataset`; `lancedb deploy/import-bdd100k/create-mv/query`; `detection-training train/eval` |
| **Synthetic data** | `npa workbench cosmos infer/train/serve`; `genesis generate-demos` |
| **Simulation** | `npa workbench isaac-lab train/eval/export-lerobot`; `genesis train-teacher/eval-student`; `retargeting run` |
| **Eval** | `npa workbench vlm-eval run/benchmark`; `mjlab eval`; `sonic eval`; `fiftyone eval`; `isaac-lab eval` |
| **Robot policy** | `npa workbench lerobot train/eval/serve/infer`; `groot finetune/serve/infer`; `sonic train/serve/export/eval` |
| **World models** | `npa workbench cosmos deploy/serve/infer/train/status` |
| **Observability** | Tool-level `status`/`list`/`system-info`; `workflow status/logs`; `rerun host/share`; `cluster status` |
| **Workflows** | `npa workbench workflow submit/run/status/logs/teardown` over the YAMLs in the [catalog](npa/workflows/workbench/skypilot/README.md) |

The generated CLI reference is in [docs/cli/README.md](docs/cli/README.md).

---

## 🚀 Flagship GPU workload: NVIDIA Cosmos

Cosmos is the world-foundation model for synthetic data and world generation. It
runs across multiple NVIDIA GPU platforms via a single `--gpu-type` flag
(`gpu-h100-sxm`, `gpu-h200-sxm`, `gpu-b300-sxm`, `gpu-l40s`) with no RT-core
lock-in:

```bash
npa workbench cosmos -p <your-project-alias> -n cosmos deploy \
  --runtime serverless --gpu-type <gpu-platform> --wait
npa workbench cosmos -p <your-project-alias> -n cosmos infer \
  --prompt "A robot arm stacks colored cubes" \
  --output-path s3://<your-bucket>/cosmos/out/
```

Cosmos needs Nebius credentials, an `HF_TOKEN`, and GPU capacity; see the
flagship walkthrough in
[docs/quickstart.md](docs/quickstart.md#7-flagship-gpu-workload-nvidia-cosmos).

---

## 🪙 Zero-GPU hosted inference: Nebius Token Factory

[Nebius Token Factory](https://tokenfactory.nebius.com/) is an OpenAI-compatible
hosted-inference API for open text and vision models. NPA uses it natively so
several workbench tools run with **no GPU and no server to manage** — you only
need a `NEBIUS_API_KEY`. This includes physical-AI scene reasoning with
`nvidia/Cosmos3-Super-Reasoner` (image/video → scene understanding + plan).

```bash
# 1. Get a key at https://tokenfactory.nebius.com/ -> API keys, then:
npa configure                       # stores NEBIUS_API_KEY in ~/.npa/credentials.yaml
npa workbench token-factory verify  # confirms auth + lists served models

# 2. Use it (zero GPU):
npa workbench token-factory reason   --input-path ./scene  --output-path /tmp/plan      # Cosmos reasoner
npa workbench token-factory caption  --input-path ./frames --output-path /tmp/captions  # vision
npa workbench token-factory generate --input-path ./prompts.jsonl --output-path /tmp/gen # text
npa workbench vlm-eval run --backend api --api-key-env NEBIUS_API_KEY \
  --input-path ./rollout --output-path /tmp/eval                                        # score rollouts
```

Full register-and-use walkthrough, SkyPilot workflows, and a physical-reasoning
hackathon challenge: [docs/workbench/token-factory.md](docs/workbench/token-factory.md).

---

## ☁️ Running on Nebius

To go from the local loop to real GPUs, authenticate with the Nebius CLI and
configure `npa` (full walkthrough in [docs/quickstart.md](docs/quickstart.md)):

```bash
nebius profile create
nebius iam get-access-token >/dev/null
npa configure
```

Workbench runs on Nebius infrastructure rather than hiding it:

- **Object storage** is the data layer for datasets, checkpoints, rollouts, eval
  JSON, exported models, and Rerun recordings.
- **SkyPilot** orchestrates multi-stage jobs and managed-Kubernetes workflows.
- **vLLM-compatible endpoints** back shared model-serving and Eval paths.
- **Managed Kubernetes, VM, BYOVM, container, and serverless** runtimes cover
  H100, H200, L40S, B300, and RTX6000 GPU targets as each tool is validated.
- **Nebius CLI auth + IAM**, with user secrets in `~/.npa/credentials.yaml` kept
  separate from machine-managed config in `~/.npa/config.yaml`.

---

## 🧩 Solutions framework

Workbench is the current primary solution, implemented as the top-level SDK
namespace `npa.workbench` and the CLI namespace `npa workbench`. The repository
supports multiple top-level solution namespaces, and future solutions (a
datalake or simfarm, say) are **additive**: they sit beside Workbench as another
top-level `npa` namespace and must not rename Workbench, move it under a
`solutions` namespace, or change existing `npa workbench` commands. See
[docs/architecture/solutions-model.md](docs/architecture/solutions-model.md).

---

## 🤝 Contributing

To work on `npa` itself, install the dev extra and run the fast suite:

```bash
pip install -e "npa[dev]"
make test
```

See [CONTRIBUTING.md](CONTRIBUTING.md). Repo validation uses
`npa/.venv/bin/python`, never bare `python`.

---

## 📚 Docs

- [Workflow Catalog](npa/workflows/workbench/skypilot/README.md): find any
  SkyPilot workflow by what you want to do.
- [docs/quickstart.md](docs/quickstart.md): install, Nebius auth, credentials.
- [docs/workbench/getting-started.md](docs/workbench/getting-started.md):
  Workbench setup and first workload.
- [docs/workbench/cookbooks/README.md](docs/workbench/cookbooks/README.md):
  end-to-end cookbooks (BDD100K, sim-to-real, VLM-eval loop, SONIC, Isaac Lab).
- [docs/workbench/cli-sdk-yaml-walkthrough.md](docs/workbench/cli-sdk-yaml-walkthrough.md):
  call any tool through the CLI, SDK, and YAML.
- [docs/cli/README.md](docs/cli/README.md): generated CLI reference.
- [docs/README.md](docs/README.md): full documentation index.
- [LICENSE](LICENSE): Apache License 2.0.
