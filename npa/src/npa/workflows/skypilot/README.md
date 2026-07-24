# Internal SkyPilot task templates

**This directory is not the supported workflow catalog.** The supported,
customer-facing specs are the declarative `npa.workflow` YAMLs
(`apiVersion: npa.workflow/v0.0.1`) under
`npa/workflows/workbench/npa-workflows/`. Author and submit those; SkyPilot is
only the execution engine.

These files are internal, package-owned runtime resources: raw SkyPilot task
YAMLs that the `npa/scripts/run_*.py` wrappers and `npa.workflow` engine render
and launch. They were relocated here (out of `npa/workflows/workbench/`) so the
shown catalog is exclusively `npa.workflow` specs, while SkyPilot-only
capabilities that the engine cannot yet express (parallel sweeps, burst submit,
the trigger watch-loop, and the legacy H100 sim-to-real pipeline/loop) keep a
runnable home.

**Preferred submit path:** `npa workbench workflow submit <npa.workflow.yaml>`
plans the state graph, renders a serial SkyPilot multi-doc YAML, and submits it.
Use the raw YAMLs here only to inspect or operate the underlying SkyPilot task
directly.

**BYOF resource profiles** (the GPU solution-smoke task and the RTX PRO
`imagePullSecrets` global config) that the declarative BYOF specs and the BYOF
runner depend on live under `npa/src/npa/workflows/byof/profiles/`, alongside
this directory.

The supported first path is the Python wrapper or `npa` CLI for each workflow
because wrappers inject secrets, validate image overrides, and clean up owned
clusters.

## Run Pattern

All examples assume SkyPilot 0.12.2.

1. Configure SkyPilot for the target infrastructure and verify the GPU aliases
   used by the YAML are schedulable.

   ```bash
   sky show-gpus --infra kubernetes --all
   ```

2. Copy the YAML to a temporary path and replace only the template values in
   `envs:` and `resources.image_id`. SkyPilot 0.12.2 does not interpolate
   `${VAR}` placeholders inside `envs:`, so do not submit a file that still
   contains placeholders such as `${NPA_S3_BUCKET}` or `docker:${IMAGE}`.
   Avoid blindly running `envsubst` over the whole file because many `run:`
   blocks intentionally contain shell variables.

3. Provide S3-compatible credentials to the pod through SkyPilot secrets,
   Kubernetes secrets referenced by the cluster config, or another supported
   secret mechanism. The YAMLs expect ordinary AWS-compatible variables such as
   `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_ENDPOINT_URL`, and the
   workflow-specific `s3://...` inputs and outputs listed below.

4. Launch the rendered YAML.

   ```bash
   sky launch -y --infra kubernetes/<context-name> -c <cluster-name> /tmp/rendered.yaml
   ```

   YAMLs that declare `cloud: nebius` can be submitted with the corresponding
   Nebius SkyPilot infra target instead of a Kubernetes context.

5. Collect status and logs, then tear down explicitly. Do not rely on
   autodown for these workflows.

   ```bash
   sky queue <cluster-name>
   sky logs <cluster-name>
   sky down -y <cluster-name>

   while sky status --refresh | grep -q "<cluster-name>"; do
     sleep 10
   done
   ```

## Common Inputs

- `NPA_S3_BUCKET`, `S3_BUCKET`, `S3_PREFIX`, `PIPELINE_ROOT_URI`, and
  workflow-specific `*_URI` values select the S3-compatible input and output
  locations. Use a dedicated run prefix for every launch.
- `AWS_ENDPOINT_URL` or workflow-specific endpoint variables select the S3
  endpoint. Keep the endpoint configurable for BYO S3-compatible storage.
- `HF_TOKEN` is needed only for workflows that fetch a gated or private
  Hugging Face repo, or when your organization requires authenticated
  downloads for public repos.
- `NGC_API_KEY` is needed only where the YAML says NGC is required or when you
  rebuild/pull images that depend on NVIDIA NGC entitlement.
- Private registry images require the cluster image-pull secret configured by
  the operator. The raw YAMLs intentionally use placeholder registry IDs where
  the user must supply their own image.

## Per-YAML Reference

| YAML | Description | Target | S3 I/O | HF rights | NGC entitlement |
| --- | --- | --- | --- | --- | --- |
| `bdd100k-pipeline.yaml` | Multi-stage BDD100K ingest, LanceDB backfill, materialized views, detector training, detector eval, and optional FiftyOne app. | Kubernetes CPU plus `H100:1` train/eval stages. | Reads `BDD100K_SOURCE_URI`; writes `PIPELINE_ROOT_URI`, `LANCE_URI`, `TRAIN_OUTPUT_URI`, and `EVAL_OUTPUT_URI` under `s3://<bucket>/bdd100k-pipeline/<run-id>/`. | None. BDD100K dataset access is separate from HF. | None for the workflow; registry access is still required for private images. |
| `cosmos2-transfer.yaml` | Runs a Cosmos2 transfer/augment stage from input assets and scene spec. | Kubernetes `RTXPRO6000:1`. | Reads `NPA_INPUT_URI`, `NPA_ASSETS_URI`, and `NPA_SCENE_SPEC_URI`; writes `NPA_OUTPUT_URI`. | None in the YAML. | Required when pulling or rebuilding Cosmos/NGC-derived images; otherwise registry access for `COSMOS2_TRANSFER_IMAGE`. |
| `cosmos3-ea-fetch.yaml` | Fetches the Cosmos3 framework source and `nvidia/Cosmos3-Nano` checkpoint into node-local cache as an access check. | Kubernetes CPU, `8+` CPUs, `32+` GB memory, large disk. | No durable S3 output by default; cache is node-local. | Required for `nvidia/Cosmos3-Nano` if Hugging Face access approval or authenticated download is needed. Set the token named by `NPA_COSMOS3_HF_TOKEN_ENV`. | Optional by default because `NPA_COSMOS3_REQUIRE_NGC=0`; required when set to `1`. |
| `cosmos3-reason.yaml` | Runs the Cosmos3 reasoning image over a prompt/input bundle. | Kubernetes `RTXPRO6000:1`. | Reads `NPA_INPUT_URI`; writes `NPA_OUTPUT_URI`. | None by default; follow the selected model's HF terms if `COSMOS3_REASON_MODEL` is changed to an HF-hosted model. | Required when pulling or rebuilding Cosmos/NGC-derived images; otherwise registry access for `COSMOS3_REASON_IMAGE`. |
| `cosmos3-text-to-image-inference.yaml` | Clones NVIDIA Cosmos3, downloads `nvidia/Cosmos3-Nano`, and runs a text-to-image smoke/inference command. | Kubernetes `H100:1`, `16+` CPUs, `128+` GB memory, large disk. | Optional upload to `NPA_COSMOS3_OUTPUT_S3_URI`; local output otherwise. | Required for `nvidia/Cosmos3-Nano` if Hugging Face access approval or authenticated download is needed. Set the token named by `NPA_COSMOS3_HF_TOKEN_ENV`. | Optional by default because `NPA_COSMOS3_REQUIRE_NGC=0`; required when set to `1`. |
| `isaac-lab-rl-sweep.yaml` | Runs four Isaac Lab RSL-RL training variants as a learning-rate and entropy sweep. | Kubernetes `L40S:1` per variant. | Writes run logs and summaries to `S3_OUTPUT_PREFIX`. | None. | Required for Isaac Sim/Isaac Lab image entitlement when pulling or rebuilding NGC-derived images. |
| `isaac-lab-rl-train.yaml` | Runs one Isaac Lab RSL-RL training job. | Kubernetes `L40S:1`. | Writes run logs and summaries to `S3_OUTPUT_PREFIX`. | None. | Required for Isaac Sim/Isaac Lab image entitlement when pulling or rebuilding NGC-derived images. |
| `isaac-lab-rl-train-rtxpro.yaml` | Runs one Isaac Lab RSL-RL training job on RTX PRO 6000 Blackwell Kubernetes. | Kubernetes `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`. | Writes run logs and summaries to `S3_OUTPUT_PREFIX`. | None. | Required for Isaac Sim/Isaac Lab image entitlement when pulling or rebuilding NGC-derived images. |
| `isaac-lab-rl-train-rtxpro-smoke.yaml` | Minimal Isaac Lab RSL-RL smoke on RTX PRO (`num_envs=4`, `iterations=1`) for live BYOF onboarding validation. | Kubernetes `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`, `cpus: 4+`, `memory: 16+`. | Writes run logs and summaries to `S3_OUTPUT_PREFIX`. | None. | Required for Isaac Sim/Isaac Lab image entitlement when pulling or rebuilding NGC-derived images. |
| `isaac-lab-cosmos-sdg-burst-smoke.yaml` | Single-task burst smoke: Isaac Lab Cartpole headless training plus a Cosmos SDG transfer-contract manifest. | Nebius `L40S:1` via `npa burst submit-yaml`. | Uses `NPA_OUTPUT_URI` as the run root and emits a Cosmos SDG manifest with Isaac output, assets, and scene-spec URIs. | None. | Required for Isaac Sim/Isaac Lab image entitlement; `npa burst submit-yaml` injects SkyPilot Docker login secrets for private Nebius registry images. |
| `byof-datagen-rtxpro-smoke.yaml` | LeIsaac/BYOF scripted datagen smoke on RTX PRO: runs `scripts/datagen/state_machine/generate.py` with parallel sim envs (no teleop). | Kubernetes `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1`, `cpus: 4+`, `memory: 16+`. | Writes `dataset.hdf5` and `npa_byof_summary.json` to `S3_OUTPUT_PREFIX`. | None. | Requires a BYOF image with LeIsaac cloned under `/opt/byof`; Isaac Sim/Isaac Lab base image entitlement for rebuilds. |
| `byof-container-smoke-rtxpro.yaml` | BYOF container-verify / solution-smoke CPU path: asserts `/opt/byof` clone + metadata, optionally runs `BYOF_SMOKE_COMMAND`, and requires `BYOF_SMOKE_ARTIFACT_NAME` when set. | Kubernetes `cpus: 2+`, `memory: 4+`. | Writes `npa_byof_summary.json`, repo listing, optional smoke logs, and optional solution smoke artifact to `S3_OUTPUT_PREFIX`. | None. | Uses BYOF image built with `--base-profile ubuntu`, explicit `--base-image`, or any OSS repo. |
| `isaac-franka-capture-reason.yaml` | Isaac Lab Franka frame capture on L40S, then Token Factory Cosmos3 reasoner over PNGs. | Kubernetes `L40S:1` + CPU reason stage. | Writes scene PNGs to `SCENE_URI`; reasoning JSON to `PLAN_URI`. | None. | Isaac stage: NGC entitlement for `npa-isaac-lab`; reason stage: `NEBIUS_TOKEN_FACTORY_KEY`. |
| `mjlab-eval.yaml` | Evaluates a SONIC checkpoint through the MJLab evaluation helper. | Kubernetes `H100:1`. | Reads `EVAL_INPUT_URI` and `SONIC_CHECKPOINT_URI`; writes `MJLAB_OUTPUT_URI`. | None. | Image-specific only; required if the selected image depends on NGC content. |
| `retargeting.yaml` | Retargets a source motion to the configured SONIC embodiment. | Kubernetes CPU. | Reads `INPUT_MOTION_URI` and optional `RETARGET_MAP_URI`; writes `RETARGETED_MOTION_URI`. | None. | Image-specific only; required if the selected image depends on NGC content. |
| `sim-to-real-loop.yaml` | Runs the VLM evaluation loop over rollout results using the self-hosted VLM image. | Kubernetes `H100:1`. | Reads rollout input configured in the task payload; writes `OUTPUT_DIR`, commonly a run-local or S3-backed output path. | Optional for public `Qwen/Qwen2-VL-7B-Instruct` downloads; required only for private/gated overrides. | Image-specific only; required if the selected VLM image depends on NGC content. |
| `sim-to-real-pipeline.yaml` | Runs the full Sim2Real pipeline: dataset input, env generation, split, policy training, VLM eval hooks, checkpointing, and reporting. | Kubernetes failover across `H100:1`, `H200:1`, and `L40S:1`. | Reads `INPUT_DATA_URI`/`LEROBOT_DATASET_URI`; writes `PIPELINE_ROOT_URI`, env splits, checkpoints, and visualization artifacts under the configured S3 prefix. | Optional for public `lerobot/pusht`; required for private/gated dataset or model overrides. | Required for any selected Cosmos, SONIC, Isaac, or other NGC-derived image in the pipeline. |
| `sim-to-real-trigger.yaml` | Polls an S3-compatible trigger prefix and submits the Sim2Real pipeline when new input arrives. | Kubernetes CPU. | Reads trigger bucket/prefix and watermark URI; submits pipeline with the configured pipeline S3 bucket, prefix, and input URI. | None. | None. |
| `sim2real-actions.yaml` | Generates action-conditioned rollouts from train environments with a policy image. | Kubernetes `RTXPRO6000:1`. | Reads `NPA_TRAIN_ENVS_URI`; writes `NPA_ACTIONS_URI` and `NPA_OUTPUT_URI`. | None. | Image-specific only; required if `POLICY_IMAGE` depends on NGC content. |
| `sim2real-envgen-split.yaml` | Splits generated Sim2Real environments into train and held-out shards. | Kubernetes `RTXPRO6000:1`. | Reads `NPA_AUGMENTED_FRAMES_URI`; writes `NPA_OUTPUT_URI`. | None. | Image-specific only; required if `ENVGEN_IMAGE` depends on NGC content. |
| `sonic-eval.yaml` | Evaluates an exported SONIC ONNX policy, including containerized render/eval options. | Nebius `L40S:1`. | Reads `SONIC_ONNX` and `SONIC_METADATA`; writes `SONIC_EVAL_OUTPUT`. | None. | Required for SONIC/Isaac image entitlement when pulling or rebuilding NGC-derived images. |
| `sonic-export-eval.yaml` | Exports a SONIC checkpoint to ONNX and evaluates the exported policy. | Nebius `L40S:1`. | Reads `POLICY_CKPT`; writes `OUTPUT_DIR` and eval artifacts. | None unless `POLICY_CKPT` points at an HF checkpoint. | Required for SONIC/Isaac image entitlement when pulling or rebuilding NGC-derived images. |
| `sonic-export.yaml` | Exports a SONIC checkpoint to ONNX metadata and policy artifacts. | Kubernetes `L40S:1`. | Reads `SONIC_CHECKPOINT`; writes `SONIC_OUTPUT` and `SONIC_METADATA`. | None unless `SONIC_CHECKPOINT` points at an HF checkpoint. | Required for SONIC/Isaac image entitlement when pulling or rebuilding NGC-derived images. |
| `sonic-locomotion-finetuning.yaml` | Retargets motion, fine-tunes SONIC, and runs MJLab evaluation. | Kubernetes CPU, `L40S:1` fine-tune, and `H100:1` eval stages. | Reads motion input and optional checkpoint URI; writes retargeted motion, `SONIC_TRAIN_OUTPUT_URI`, and `MJLAB_OUTPUT_URI`. | Required for the default `nvidia/GEAR-SONIC:sonic_release` checkpoint. | Required for SONIC/Isaac image entitlement when pulling or rebuilding NGC-derived images. |
| `sonic-train-standalone.yaml` | Runs a standalone SONIC training smoke using docker-run payload mode. | Nebius `L40S:1`. | Requires `S3_ENDPOINT_URL` and `S3_BUCKET`; writes to `s3://<bucket>/<SONIC_OUTPUT_PREFIX>/`. | Required for the default `nvidia/GEAR-SONIC:sonic_release/last.pt` checkpoint. | Required for SONIC/Isaac image entitlement when pulling or rebuilding NGC-derived images. |
| `token-factory-caption.yaml` | Batch-captions images with a hosted Token Factory vision model. | CPU only (zero-GPU). | Reads `INPUT_URI`; writes `OUTPUT_URI` with `generations.jsonl`. | None. | None. |
| `token-factory-generate.yaml` | Batch text generation from a JSONL prompt file via Token Factory. | CPU only (zero-GPU). | Reads `INPUT_URI`; writes `OUTPUT_URI` with `generations.jsonl`. | None. | None. |
| `token-factory-cosmos-reason.yaml` | Scene reasoning with `nvidia/Cosmos3-Super-Reasoner` over images/video. | CPU only (zero-GPU). | Reads `INPUT_URI`; writes `OUTPUT_URI` with scene reasoning JSON. | None. | None. |
| `tokenfactory-train-triage.yaml` | LeRobot smoke train on k8s GPU, then Token Factory triage report from run artifacts. | Kubernetes GPU + CPU triage stage. | Reads train config; writes artifacts under `ARTIFACTS_URI` and triage under `TRIAGE_URI`. | None. | None. |
| `tokenfactory-rollout-judge.yaml` | LeRobot eval rollout on k8s GPU, then hosted VLM judging via `vlm-eval --backend api`. | Kubernetes GPU + CPU judge stage. | Reads rollout config; writes rollout videos and `VLM_EVAL_OUTPUT_URI`. | None. | None. |
| `tokenfactory-scene-to-rollout-judge.yaml` | Reason over scene images, roll out a policy on k8s GPU, judge against the plan. | CPU reason + k8s GPU rollout + CPU judge. | Reads `SCENE_URI`; writes plan, rollouts, and judge report under configured S3 prefixes. | None. | None. |
| `vlm-eval-token-factory.yaml` | Scores rollout artifacts with a hosted Token Factory VLM (no local vLLM). | CPU only (zero-GPU). | Reads `EVAL_INPUT_URI`; writes `VLM_EVAL_OUTPUT_URI`. | None. | None. |
| `vlm-eval-benchmark.yaml` | Runs a self-hosted VLM benchmark over a benchmark dataset. | Kubernetes `H100:1`. | Reads `VLM_BENCHMARK_DATASET_URI`; writes `VLM_EVAL_BENCHMARK_OUTPUT_URI`. | Optional for public `Qwen/Qwen2-VL-7B-Instruct`; required only for private/gated overrides. | Image-specific only; required if the selected VLM image depends on NGC content. |
| `vlm-eval.yaml` | Runs self-hosted VLM evaluation for one task/input set. | Kubernetes `H100:1`. | Reads `EVAL_INPUT_URI`; writes `VLM_EVAL_OUTPUT_URI`. | Optional for public `Qwen/Qwen2-VL-7B-Instruct`; required only for private/gated overrides. | Image-specific only; required if the selected VLM image depends on NGC content. |

## Standalone Launch Commands

After rendering placeholders into `/tmp/<yaml-name>.yaml`, each YAML can be
launched directly. Use stable, run-specific cluster names so cleanup is
unambiguous.

```bash
sky launch -y --infra kubernetes/<context-name> -c bdd100k-pipeline /tmp/bdd100k-pipeline.yaml
sky launch -y --infra kubernetes/<context-name> -c cosmos2-transfer /tmp/cosmos2-transfer.yaml
sky launch -y --infra kubernetes/<context-name> -c cosmos3-ea-fetch /tmp/cosmos3-ea-fetch.yaml
sky launch -y --infra kubernetes/<context-name> -c cosmos3-reason /tmp/cosmos3-reason.yaml
sky launch -y --infra kubernetes/<context-name> -c cosmos3-t2i /tmp/cosmos3-text-to-image-inference.yaml
sky launch -y --infra kubernetes/<context-name> -c isaac-lab-rl-sweep /tmp/isaac-lab-rl-sweep.yaml
sky launch -y --infra kubernetes/<context-name> -c isaac-lab-rl-train /tmp/isaac-lab-rl-train.yaml
sky launch -y --config npa/src/npa/workflows/byof/profiles/skypilot-kubernetes-rtxpro.yaml --infra kubernetes/<context-name> -c isaac-lab-rl-train-rtxpro /tmp/isaac-lab-rl-train-rtxpro.yaml
sky launch -y --config npa/src/npa/workflows/byof/profiles/skypilot-kubernetes-rtxpro.yaml --infra kubernetes/<context-name> -c isaac-lab-rl-train-rtxpro-smoke /tmp/isaac-lab-rl-train-rtxpro-smoke.yaml
npa burst submit-yaml /tmp/isaac-lab-cosmos-sdg-burst-smoke.yaml --name <run-id>
sky launch -y --config npa/src/npa/workflows/byof/profiles/skypilot-kubernetes-rtxpro.yaml --infra kubernetes/<context-name> -c byof-datagen-rtxpro-smoke /tmp/byof-datagen-rtxpro-smoke.yaml
sky launch -y --infra kubernetes/<context-name> -c mjlab-eval /tmp/mjlab-eval.yaml
sky launch -y --infra kubernetes/<context-name> -c retargeting /tmp/retargeting.yaml
sky launch -y --infra kubernetes/<context-name> -c sim-to-real-loop /tmp/sim-to-real-loop.yaml
sky launch -y --infra kubernetes/<context-name> -c sim-to-real-pipeline /tmp/sim-to-real-pipeline.yaml
sky launch -y --infra kubernetes/<context-name> -c sim-to-real-trigger /tmp/sim-to-real-trigger.yaml
sky launch -y --infra kubernetes/<context-name> -c sim2real-actions /tmp/sim2real-actions.yaml
sky launch -y --infra kubernetes/<context-name> -c sim2real-envgen-split /tmp/sim2real-envgen-split.yaml
sky launch -y --infra nebius -c sonic-eval /tmp/sonic-eval.yaml
sky launch -y --infra nebius -c sonic-export-eval /tmp/sonic-export-eval.yaml
sky launch -y --infra kubernetes/<context-name> -c sonic-export /tmp/sonic-export.yaml
sky launch -y --infra kubernetes/<context-name> -c sonic-locomotion-finetuning /tmp/sonic-locomotion-finetuning.yaml
sky launch -y --infra nebius -c sonic-train-standalone /tmp/sonic-train-standalone.yaml
sky launch -y --infra kubernetes/<context-name> -c vlm-eval-benchmark /tmp/vlm-eval-benchmark.yaml
sky launch -y --infra kubernetes/<context-name> -c vlm-eval /tmp/vlm-eval.yaml
```

## Gated Hugging Face models

Many workflows pass an `HF_TOKEN` so a runtime can download model weights or
datasets from Hugging Face. A token alone is **not** enough for *gated* repos:
you must also open the repo page once while signed in with the same account and
accept its license/usage terms (NVIDIA repos may also require a request form),
or the download fails with `403 Gated`. Public repos need no acceptance and the
token is optional (it only helps avoid anonymous rate limits).

The table below lists, per workflow, the repos you must accept before the run
can fetch weights. Gated repos are marked **(gated — accept license)**.

| Workflow YAML | Hugging Face repos to accept | Notes |
| --- | --- | --- |
| `sonic-train-standalone.yaml` | `nvidia/GEAR-SONIC` **(gated — accept license)** | Default `SONIC_CHECKPOINT=nvidia/GEAR-SONIC:sonic_release/last.pt`. |
| `sonic-locomotion-finetuning.yaml` | `nvidia/GEAR-SONIC` **(gated — accept license)** | Fine-tune stage downloads the released checkpoint; the MuJoCo-eval stage consumes the S3 checkpoint and needs no HF access. |
| `cosmos3-ea-fetch.yaml` | `nvidia/Cosmos3-Nano` **(gated — early-access, accept license)** | `NPA_COSMOS3_MODEL_ID` default; override for a BYO checkpoint. |
| `cosmos3-text-to-image-inference.yaml` | `nvidia/Cosmos3-Nano` **(gated — early-access, accept license)** | Same `NPA_COSMOS3_MODEL_ID` default; also needs `GITHUB_TOKEN` for the source repo. |
| `cosmos3-reason.yaml` | `nvidia/Cosmos-Reason1-7B` **(gated — accept license)** when the reasoning image runs | The YAML only emits a contract manifest recording `COSMOS3_REASON_MODEL` (default `nvidia/Cosmos-Reason1-7B`); the reasoning image that consumes it downloads the gated weights. |
| `cosmos2-transfer.yaml` | NVIDIA Cosmos diffusion weights **(gated — accept license)** when the transfer image runs | The YAML only emits a transfer manifest; `COSMOS2_TRANSFER_IMAGE` downloads the gated weights (the `npa workbench cosmos` default is `nvidia/Cosmos-1.0-Diffusion-7B-Text2World`). |
| `vlm-eval.yaml` | `Qwen/Qwen2-VL-7B-Instruct` (public) | `VLM_MODEL` default. Apache-2.0; token optional, not gated. |
| `vlm-eval-benchmark.yaml` | `Qwen/Qwen2-VL-7B-Instruct` (public) | `VLM_MODELS` default. Token optional, not gated. |
| `sim-to-real-loop.yaml` | `Qwen/Qwen2-VL-7B-Instruct` (public) | Self-hosted vLLM default `MODEL`. Token optional, not gated. |
| `sim-to-real-pipeline.yaml` | `lerobot/pusht` (public dataset) | Default `LEROBOT_DATASET_REPO_ID`; default `VLM_EVAL_BACKEND=stub` pulls no VLM. |
| `sim-to-real-trigger.yaml` | `lerobot/pusht` (public dataset) | Watches/retriggers `sim-to-real-pipeline.yaml`; same public dataset. |
| `bdd100k-pipeline.yaml` | None | CLIP embeddings run inside the first-party LanceDB image; BDD100K dataset access is separate from HF. |
| `isaac-lab-rl-train.yaml`, `isaac-lab-rl-sweep.yaml`, `isaac-lab-rl-train-rtxpro.yaml`, `isaac-lab-rl-train-rtxpro-smoke.yaml`, `byof-datagen-rtxpro-smoke.yaml` | None | Isaac Lab RSL-RL training and LeIsaac scripted datagen pull no HF weights. |
| `sonic-export.yaml`, `sonic-export-eval.yaml`, `sonic-eval.yaml` | None | Operate on already-trained checkpoints staged in S3 (unless an input points at an HF checkpoint). |
| `mjlab-eval.yaml`, `retargeting.yaml` | None | Consume S3 artifacts; no HF download. |
| `sim2real-actions.yaml`, `sim2real-envgen-split.yaml` | None | Env generation / action conditioning use BYO container images, not HF repos. |

The self-contained Sim2Real runbook (`../sim2real/runbook.yaml`) defaults to
dual self-hosted VLM eval: `nvidia/Cosmos-Reason2-8B` and
`nvidia/Cosmos-Reason2-2B`, both **(gated — accept license)**, plus
`nvidia/Cosmos-Transfer2.5-2B` for augment. The public `lerobot/pusht` dataset
needs no HF acceptance. `nvidia/Cosmos3-Super-Reasoner` is **Token Factory only**
(not on Hugging Face); do not use it as `VLM_REASON3_MODEL` for cluster Jobs.

### Gated repos not surfaced by a workflow YAML

Each workbench tool is a containerized service that can be driven by CLI/SDK as
well as the YAMLs above, so the entrypoint does not change which repo is gated.
Two gated repos still aren't visible from the per-workflow table:

- **GR00T has no SkyPilot YAML in this directory** — it is driven only by
  `npa workbench groot` (CLI/SDK). It needs `nvidia/GR00T-N1.7-3B` **and**
  `nvidia/Cosmos-Reason2-2B`, both **gated — accept license**.
- **Driving the Cosmos tool directly** (`npa workbench cosmos ...`) defaults to a
  different repo than the `cosmos3-*` YAMLs above:
  `nvidia/Cosmos-1.0-Diffusion-7B-Text2World` **(gated — accept license)**.

### How to accept a gated repo

1. Sign in to Hugging Face with the account whose token you set as `HF_TOKEN`.
2. Open the repo page (for example `https://huggingface.co/nvidia/GEAR-SONIC`)
   and accept the license / "Agree and access repository" prompt, completing any
   NVIDIA request form.
3. Confirm the token can reach the repo before a long run. `npa workbench cosmos
   check` and `npa workbench groot` validate gated-model access for those tools;
   for other workflows a quick `huggingface-cli download <repo> --revision main`
   smoke check works.

## Cleanup Rules

Raw SkyPilot launches are user-owned. Always keep the cluster name and run
prefix together in your run notes, cancel failed managed jobs explicitly, run
`sky down -y <cluster-name>`, and poll `sky status --refresh` until the cluster
is gone. For Nebius-backed launches, do not rely on autodown.
