# Workbench SkyPilot Workflow Catalog

**This is the index of every runnable Workbench workflow YAML.** Each YAML is a
SkyPilot pipeline you can submit with `npa workbench workflow submit`. The
narrative walkthroughs (prerequisites, run, verify) live in `docs/`, and thin
submission wrappers live in `npa/scripts/`.

Use the catalog below to find a workflow by what you want to do. As a companion
convention, each YAML should open with a short header comment carrying the same
four pointers, so you can navigate from a file back to its guide and this index:

```text
# What:   one-line description of the pipeline
# Guide:  docs/... walkthrough
# Runner: how to submit it (CLI command or npa/scripts/ wrapper)
# Index:  npa/workflows/workbench/skypilot/README.md  (this file)
```

## Find your workflow

Pick the row that matches what you want to do. (This catalog is the source of
truth; the top-level `README.md` shows only a few common entry points.)

| I want to… | Workflow YAML | Run it with | GPU | Guide |
| --- | --- | --- | --- | --- |
| **Score robot rollouts with a VLM, no GPU/credentials** | [`vlm-eval.yaml`](./vlm-eval.yaml) | `npa workbench vlm-eval` | H100 (or `stub` locally) | [vlm-eval-loop-runbook.md](../../../../docs/workbench/cookbooks/vlm-eval-loop-runbook.md) |
| Sweep eval thresholds / rubrics / models and rank them | [`vlm-eval-benchmark.yaml`](./vlm-eval-benchmark.yaml) | `npa workbench vlm-eval benchmark` | H100 | [vlm-eval-loop-runbook.md](../../../../docs/workbench/cookbooks/vlm-eval-loop-runbook.md) |
| Run the full sim-to-real train → eval pipeline | [`sim-to-real-pipeline.yaml`](./sim-to-real-pipeline.yaml) | [`run_sim_to_real_pipeline.py`](../../../scripts/run_sim_to_real_pipeline.py) | H100 | [sim-to-real-pipeline.md](../../../../docs/workbench/cookbooks/sim-to-real-pipeline.md) |
| Run the sim-to-real VLM-eval feedback loop | [`sim-to-real-loop.yaml`](./sim-to-real-loop.yaml) | [`run_sim_to_real_pipeline.py`](../../../scripts/run_sim_to_real_pipeline.py) | H100 | [vlm-eval-loop-runbook.md](../../../../docs/workbench/cookbooks/vlm-eval-loop-runbook.md) |
| Re-trigger sim-to-real when new data lands in S3 | [`sim-to-real-trigger.yaml`](./sim-to-real-trigger.yaml) | [`run_sim_to_real_pipeline.py`](../../../scripts/run_sim_to_real_pipeline.py) | CPU | [sim-to-real-pipeline.md](../../../../docs/workbench/cookbooks/sim-to-real-pipeline.md) |
| Curate + train an AV perception model (BDD100K) | [`bdd100k-pipeline.yaml`](./bdd100k-pipeline.yaml) | [`run_bdd100k_pipeline.py`](../../../scripts/run_bdd100k_pipeline.py) | H100 | [bdd100k-pipeline.md](../../../../docs/workbench/cookbooks/bdd100k-pipeline.md) |
| Train one Isaac Lab RL policy | [`isaac-lab-rl-train.yaml`](./isaac-lab-rl-train.yaml) | [`run_isaac_lab_rl.py`](../../../scripts/run_isaac_lab_rl.py) | L40S | [byof-isaac-lab/README.md](../../../../docs/workbench/cookbooks/byof-isaac-lab/README.md) |
| Sweep many Isaac Lab RL configs in parallel | [`isaac-lab-rl-sweep.yaml`](./isaac-lab-rl-sweep.yaml) | [`run_isaac_lab_rl.py`](../../../scripts/run_isaac_lab_rl.py) | L40S | [workbench-yaml-guide.md](../../../../docs/workbench-yaml-guide.md#isaac-lab-rl-training) |
| Fine-tune a SONIC G1 locomotion policy + MuJoCo eval | [`sonic-locomotion-finetuning.yaml`](./sonic-locomotion-finetuning.yaml) | `npa workbench sonic` | H100 / L40S | [sonic-locomotion-finetuning.md](../../../../docs/workbench/cookbooks/sonic-locomotion-finetuning.md) |
| Run a SONIC training smoke job | [`sonic-train-standalone.yaml`](./sonic-train-standalone.yaml) | `npa workbench sonic train` | L40S | [sonic-train-runbook.md](../../../../docs/workbench/cookbooks/sonic-train-runbook.md) |
| Export a trained SONIC checkpoint to ONNX | [`sonic-export.yaml`](./sonic-export.yaml) | `npa workbench sonic export` | L40S | [sonic-eval-runbook.md](../../../../docs/workbench/cookbooks/sonic-eval-runbook.md) |
| Export then evaluate a SONIC policy in one run | [`sonic-export-eval.yaml`](./sonic-export-eval.yaml) | `npa workbench sonic` | L40S | [sonic-eval-runbook.md](../../../../docs/workbench/cookbooks/sonic-eval-runbook.md) |
| Evaluate an exported SONIC ONNX policy | [`sonic-eval.yaml`](./sonic-eval.yaml) | `npa workbench sonic eval` | L40S | [sonic-eval-runbook.md](../../../../docs/workbench/cookbooks/sonic-eval-runbook.md) |
| Score a SONIC checkpoint with MJLab metrics | [`mjlab-eval.yaml`](./mjlab-eval.yaml) | `npa workbench mjlab eval` | H100 | [sonic-locomotion-finetuning.md](../../../../docs/workbench/cookbooks/sonic-locomotion-finetuning.md) |
| Retarget source motion to the SONIC embodiment | [`retargeting.yaml`](./retargeting.yaml) | `npa workbench retargeting run` | CPU | [sonic-locomotion-finetuning.md](../../../../docs/workbench/cookbooks/sonic-locomotion-finetuning.md) |
| Generate synthetic frames with Cosmos3 (text → image) | [`cosmos3-text-to-image-inference.yaml`](./cosmos3-text-to-image-inference.yaml) | `npa workbench cosmos` | H100 | [inference SKILL](../../../../.agents/skills/inference/SKILL.md) |
| Stage the Cosmos3 framework + checkpoints | [`cosmos3-ea-fetch.yaml`](./cosmos3-ea-fetch.yaml) | `npa workbench cosmos` | CPU | [cosmos3-setup SKILL](../../../../.agents/skills/cosmos3-setup/SKILL.md) |
| Run Cosmos3 Reason inference | [`cosmos3-reason.yaml`](./cosmos3-reason.yaml) | `npa workbench cosmos` | RTX PRO 6000 | [inference SKILL](../../../../.agents/skills/inference/SKILL.md) |
| Run a Cosmos2 transfer (video-to-world) stage | [`cosmos2-transfer.yaml`](./cosmos2-transfer.yaml) | `npa workbench cosmos` | RTX PRO 6000 | [cosmos SKILL](../../../../.agents/skills/workbench/cosmos/SKILL.md) |
| **Roll out on a k8s GPU, then judge it with hosted Token Factory** | [`tokenfactory-rollout-judge.yaml`](./tokenfactory-rollout-judge.yaml) | `npa workbench workflow submit` | H100 (judge stage CPU) | [tokenfactory-compute-combos.md](../../../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) |
| **Reason over a scene, roll out on a k8s GPU, then judge vs. the plan** | [`tokenfactory-scene-to-rollout-judge.yaml`](./tokenfactory-scene-to-rollout-judge.yaml) | `npa workbench workflow submit` | H100 (reason + judge stages CPU) | [tokenfactory-compute-combos.md](../../../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) |
| Train on a serverless GPU, then triage the run with Token Factory | _(runner script)_ | [`run_tokenfactory_train_triage.py`](../../../scripts/run_tokenfactory_train_triage.py) | serverless GPU (triage CPU) | [tokenfactory-compute-combos.md](../../../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) |
| Design a sweep, run N serverless GPU trains, then rank them with Token Factory | _(runner script)_ | [`run_tokenfactory_sim_sweep.py`](../../../scripts/run_tokenfactory_sim_sweep.py) | serverless GPU ×N (design + rank CPU) | [tokenfactory-compute-combos.md](../../../../docs/workbench/cookbooks/tokenfactory-compute-combos.md) |

To compose your own combo, read
[composing-cloud-and-token-factory.md](../../../../docs/workbench/composing-cloud-and-token-factory.md).

The Cosmos guides are agent skills under `.agents/skills/`; everything else
links to a human-facing guide under `docs/`.

[`sim2real-actions.yaml`](./sim2real-actions.yaml) and
[`sim2real-envgen-split.yaml`](./sim2real-envgen-split.yaml) are step components
of the self-contained Sim2Real runbook at
[`../sim2real/README.md`](../sim2real/README.md) — the one workflow that keeps
its guide and YAML colocated because it is a single CLI/SDK-driven chain rather
than a docs-site cookbook.

## How to submit a workflow

```bash
# 1. Bootstrap SkyPilot once.
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"

# 2a. Submit a YAML directly.
npa workbench workflow submit \
  npa/workflows/workbench/skypilot/vlm-eval.yaml --run-id vlm-eval

# 2b. Or use a thin wrapper when a workflow needs run-scoped S3 paths,
#     secret-env injection, GPU validation, or cleanup.
npa/.venv/bin/python npa/scripts/run_sim_to_real_pipeline.py --help
npa/.venv/bin/python npa/scripts/run_isaac_lab_rl.py --help
npa/.venv/bin/python npa/scripts/run_bdd100k_pipeline.py --help

# 3. Track and tear down.
npa workbench workflow status <run-id>
npa workbench workflow logs <run-id> <task-name>
npa workbench workflow teardown <run-id>
```

`submit` supports `--controller-backend kubernetes`, `--controller-backend
nebius`, `--run-id`, and repeated `--var KEY=VALUE` substitutions.

## Gated Hugging Face models

Several workflows pass an `HF_TOKEN` so a runtime can download model weights or
datasets from Hugging Face. A token alone is **not** enough for *gated* repos:
you must also visit the model page once while signed in with the same account
and accept its license/usage terms, or the download returns `403 Gated`.

<details>
<summary>Per-workflow gated-repo table (operators)</summary>

"None" means the workflow either uses no Hugging Face repo or only public ones
(a token is optional and only helps avoid anonymous rate limits). Gated repos
are marked **(gated — accept license)**.

| Workflow YAML | Hugging Face repos to accept | Notes |
| --- | --- | --- |
| `sonic-train-standalone.yaml` | `nvidia/GEAR-SONIC` **(gated — accept license)** | Default `SONIC_CHECKPOINT=nvidia/GEAR-SONIC:sonic_release/last.pt`. |
| `sonic-locomotion-finetuning.yaml` | `nvidia/GEAR-SONIC` **(gated — accept license)** | Fine-tune stage downloads the released SONIC checkpoint; the MuJoCo-eval stage consumes the S3 checkpoint and needs no HF access. |
| `cosmos3-ea-fetch.yaml` | `nvidia/Cosmos3-Nano` **(gated — early-access, accept license)** | `NPA_COSMOS3_MODEL_ID` default; override for a BYO checkpoint. |
| `cosmos3-text-to-image-inference.yaml` | `nvidia/Cosmos3-Nano` **(gated — early-access, accept license)** | Same `NPA_COSMOS3_MODEL_ID` default; also needs `GITHUB_TOKEN` for the source repo. |
| `cosmos3-reason.yaml` | `nvidia/Cosmos-Reason1-7B` **(gated — accept license)** | `COSMOS3_REASON_MODEL` default. |
| `cosmos2-transfer.yaml` | NVIDIA Cosmos diffusion weights baked into `COSMOS2_TRANSFER_IMAGE` **(gated — accept license)** | The repo depends on the chosen image; the `npa workbench cosmos` default is `nvidia/Cosmos-1.0-Diffusion-7B-Text2World`. |
| `vlm-eval.yaml` | `Qwen/Qwen2-VL-7B-Instruct` (public) | `VLM_MODEL` default. Token optional; not gated. |
| `vlm-eval-benchmark.yaml` | `Qwen/Qwen2-VL-7B-Instruct` (public) | `VLM_MODELS` default. Token optional; not gated. |
| `sim-to-real-loop.yaml` | `Qwen/Qwen2-VL-7B-Instruct` (public) | Self-hosted vLLM default `MODEL`. Token optional; not gated. |
| `sim-to-real-pipeline.yaml` | `lerobot/pusht` (public dataset) | Default `LEROBOT_DATASET_REPO_ID`; the default `VLM_EVAL_BACKEND=stub` pulls no VLM. Token optional. |
| `sim-to-real-trigger.yaml` | `lerobot/pusht` (public dataset) | Watches/retriggers `sim-to-real-pipeline.yaml`; same public dataset. |
| `../sim2real/runbook.yaml` | `nvidia/Cosmos-Reason1-7B` **(gated — accept license)**; `lerobot/pusht` (public dataset) | Default `--vlm-model nvidia/Cosmos-Reason1-7B` and `--trigger-dataset-id lerobot/pusht`. |
| `bdd100k-pipeline.yaml` | None | CLIP embeddings run inside the first-party LanceDB image; no gated HF repo. |
| `isaac-lab-rl-train.yaml`, `isaac-lab-rl-sweep.yaml` | None | Isaac Lab RSL-RL training pulls no HF weights. |
| `sonic-export.yaml`, `sonic-export-eval.yaml`, `sonic-eval.yaml` | None | Operate on already-trained checkpoints staged in S3. |
| `mjlab-eval.yaml`, `retargeting.yaml` | None | Consume S3 artifacts; no HF download. |
| `sim2real-actions.yaml`, `sim2real-envgen-split.yaml` | None | Env generation / action conditioning use BYO container images, not HF repos. |
| `../templates/curate-augment-train.yaml` | None | Placeholder Argo steps; no HF download. |

### Workbench tools driven by the CLI (not YAML)

These are launched through `npa workbench ...` rather than a SkyPilot YAML, but
they also require accepting gated repos before they can download weights:

- GR00T (`npa workbench groot ...`): `nvidia/GR00T-N1.7-3B` and
  `nvidia/Cosmos-Reason2-2B` — both **gated — accept license**.
- Cosmos (`npa workbench cosmos ...`): the configured model, default
  `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` — **gated — accept license**.

</details>

### How to accept a gated repo

1. Sign in to Hugging Face with the account whose token you set as `HF_TOKEN`.
2. Open the model page (for example `https://huggingface.co/nvidia/GEAR-SONIC`)
   and accept the license / "Agree and access repository" prompt. NVIDIA repos
   may also require completing a request form.
3. Confirm the token can reach the repo before launching a long run.
   `npa workbench cosmos check` and `npa workbench groot` validate gated-model
   access for those tools; for other workflows a quick
   `huggingface-cli download <repo> --revision main` smoke check works.

## Why the guides live in `docs/` and not next to each YAML

The guide-to-YAML relationship is many-to-many, so guides are not colocated 1:1:

- One guide often drives several YAMLs (the SONIC locomotion cookbook uses
  `sonic-locomotion-finetuning.yaml`, `retargeting.yaml`, and `mjlab-eval.yaml`).
- One YAML is often referenced by several guides (`bdd100k-pipeline.yaml` is used
  by its cookbook, the demo writeup, and `docs/workbench-yaml-guide.md`).
- Guides are customer-facing product documentation that cross-link to the
  quickstart, getting-started, architecture, and CLI references. They belong in
  the navigable `docs/` tree rooted at `docs/README.md`.

The header comment on each YAML and the catalog above keep both directions easy
to traverse. **Update both ends when you add or rename a workflow.**

## Conventions

- Shared parameter, artifact, naming, GPU, and safety rules:
  [`../schemas/workflow-conventions.md`](../schemas/workflow-conventions.md).
- General YAML structure (label maps, env vars, endpoints, S3 paths):
  [`../../../../docs/workbench-yaml-guide.md`](../../../../docs/workbench-yaml-guide.md).
- Submission, SkyPilot runtime, and cleanup pattern: [`../README.md`](../README.md).
