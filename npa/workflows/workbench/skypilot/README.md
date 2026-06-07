# Raw SkyPilot Workflows

This directory is the standalone raw-SkyPilot workflow tier for Workbench. These
YAMLs can be copied, materialized, and submitted directly with `sky`; no `npa`
wrapper is required at submit time. The same workflows can also be wrapped by
the NPA CLI when you want rendering, cleanup, and variable materialization:
see [`docs/cli/workflow.md`](../../../../docs/cli/workflow.md) and
[`docs/workbench-yaml-guide.md`](../../../../docs/workbench-yaml-guide.md).

Repository discovery found these raw SkyPilot YAMLs:

- This directory: `bdd100k-pipeline.yaml`, `cosmos2-transfer.yaml`,
  `cosmos3-ea-fetch.yaml`, `cosmos3-reason.yaml`,
  `cosmos3-text-to-image-inference.yaml`, `isaac-lab-rl-sweep.yaml`,
  `isaac-lab-rl-train.yaml`, `mjlab-eval.yaml`, `retargeting.yaml`,
  `sim-to-real-loop.yaml`, `sim-to-real-pipeline.yaml`,
  `sim-to-real-trigger.yaml`, `sim2real-actions.yaml`,
  `sim2real-envgen-split.yaml`, `sonic-eval.yaml`,
  `sonic-export-eval.yaml`, `sonic-export.yaml`,
  `sonic-locomotion-finetuning.yaml`, `sonic-train-standalone.yaml`,
  `vlm-eval-benchmark.yaml`, and `vlm-eval.yaml`.
- Existing co-located runbook outside this directory:
  [`../sim2real/runbook.yaml`](../sim2real/runbook.yaml), documented in
  [`../sim2real/README.md`](../sim2real/README.md).

## Raw SkyPilot Submit Pattern

SkyPilot 0.12.2 does not expand `${VAR}` references inside the YAML `envs:`
block at submit time. Before launching, copy the YAML and replace committed
`envs:` placeholders, `image_id` placeholders, bucket names, run IDs, and S3
endpoint values with literal values. Leave shell expansion inside `run:` blocks
intact.

```bash
export SKY="${NPA_SKYPILOT_BIN:-sky}"
cp npa/workflows/workbench/skypilot/<workflow>.yaml /tmp/<workflow>.yaml
# Edit /tmp/<workflow>.yaml so envs: and image_id values are literals.
"$SKY" jobs launch -n <workflow> /tmp/<workflow>.yaml -y
```

For direct `sky launch` runs, do not rely on `--down` or autodown on Nebius with
SkyPilot 0.12.2. Use an explicit cluster name, then run `sky down` and poll until
the cluster disappears:

```bash
"$SKY" launch -c <cluster-name> /tmp/<workflow>.yaml -y
"$SKY" down <cluster-name> --yes
until ! "$SKY" status --refresh | grep -q "<cluster-name>"; do sleep 10; done
```

All S3 endpoints are bring-your-own. Use your S3-compatible endpoint in
`AWS_ENDPOINT_URL`, `S3_ENDPOINT_URL`, or `NEBIUS_S3_ENDPOINT` as required by
the specific YAML. Do not rely on committed sample endpoint literals.

## Hugging Face And NGC Access

Accept gated Hugging Face repositories on the account whose token is actually
wired into the job as `HF_TOKEN` or `HUGGING_FACE_HUB_TOKEN`. Manual NVIDIA
approval may apply; after approval, regenerate or reuse a token from that same
account. Current HF metadata for the literal repos named below was checked on
2026-06-07.

The only gated HF repositories documented by the sibling raw Sim2Real runbook's
Cosmos augment access path are:

- Required:
  [`nvidia/Cosmos-Transfer2.5-2B`](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
- Required:
  [`nvidia/Cosmos-Predict2.5-2B`](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B)
- Optional when guardrails are enabled:
  [`nvidia/Cosmos-Guardrail1`](https://huggingface.co/nvidia/Cosmos-Guardrail1)

The YAMLs in this directory also reference public HF repos such as
[`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano),
[`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC),
[`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct),
and [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht). These do
not require gated HF rights according to the current HF API metadata, although
some workflows still require a token to run their access check or download step.

NGC access is required only where the selected image or runtime path needs NGC
assets. The committed Cosmos3 YAMLs set `NPA_COSMOS3_REQUIRE_NGC` to `0`; set it
to `1` only when your selected source or image requires `NGC_API_KEY`.

## Workflow Pre-Flight Checklists

### `bdd100k-pipeline.yaml`

- Description: serial BDD100K/LanceDB pipeline for ingest, CPU backfill, CLIP
  backfill, materialized views, H100 detection training/eval, and a FiftyOne app.
- Run: materialize `/tmp/bdd100k-pipeline.yaml`, then
  `sky jobs launch -n bdd100k-pipeline /tmp/bdd100k-pipeline.yaml -y`.
- Target: Kubernetes; CPU stages plus `H100:1` GPU stages for CLIP, training,
  and eval.
- S3: input `raw-bdd100k/subset-demo/`; outputs under
  `bdd100k-pipeline/<run-id>/` for LanceDB, training, and eval artifacts.
- Env/secrets: bucket/run values, service endpoints and optional service tokens,
  S3 credentials, BYO S3 endpoint for eval metric upload.
- HF rights: none.
- NGC entitlement: none.

### `cosmos2-transfer.yaml`

- Description: standalone Cosmos2 transfer contract stage that writes a transfer
  manifest from input, output, asset, and scene-spec URIs.
- Run: materialize `/tmp/cosmos2-transfer.yaml`, then
  `sky jobs launch -n cosmos2-transfer /tmp/cosmos2-transfer.yaml -y`.
- Target: Kubernetes `RTXPRO6000:1`; use a Blackwell-capable image when this
  token resolves to an RTX PRO 6000 Blackwell target.
- S3: input `NPA_INPUT_URI`; output `NPA_OUTPUT_URI`; optional assets and
  scene-spec URIs.
- Env/secrets: `COSMOS2_TRANSFER_IMAGE` plus the URI and run-id envs; S3
  credentials if any URI is `s3://`.
- HF rights: none from this YAML. The YAML does not declare Cosmos model repo
  IDs. Known Cosmos Transfer2.5 augment images require
  [`nvidia/Cosmos-Transfer2.5-2B`](https://huggingface.co/nvidia/Cosmos-Transfer2.5-2B)
  and
  [`nvidia/Cosmos-Predict2.5-2B`](https://huggingface.co/nvidia/Cosmos-Predict2.5-2B);
  they also require
  [`nvidia/Cosmos-Guardrail1`](https://huggingface.co/nvidia/Cosmos-Guardrail1)
  unless launched with guardrails disabled. Treat those as image-specific
  requirements unless the YAML is updated to declare the model IDs directly.
- NGC entitlement: image-dependent; none declared by this YAML.

### `cosmos3-ea-fetch.yaml`

- Description: checks and fetches the Cosmos3 source tree plus checkpoint into
  node-local cache.
- Run: materialize `/tmp/cosmos3-ea-fetch.yaml`, then
  `sky jobs launch -n cosmos3-ea-fetch /tmp/cosmos3-ea-fetch.yaml -y`.
- Target: Kubernetes CPU task with large disk cache; no GPU accelerator is
  requested by the YAML.
- S3: none.
- Env/secrets: `HF_TOKEN` is required by the YAML's access check; `GITHUB_TOKEN`
  is optional for private forks; `NGC_API_KEY` is checked only when
  `NPA_COSMOS3_REQUIRE_NGC=1`.
- HF rights: none. The referenced
  [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) repo is
  public according to current HF metadata.
- NGC entitlement: optional only when `NPA_COSMOS3_REQUIRE_NGC=1`.

### `cosmos3-reason.yaml`

- Description: standalone Cosmos3 reasoning contract stage that records input,
  output, model, prompt, and image selections.
- Run: materialize `/tmp/cosmos3-reason.yaml`, then
  `sky jobs launch -n cosmos3-reason /tmp/cosmos3-reason.yaml -y`.
- Target: Kubernetes `RTXPRO6000:1`; use a Blackwell-capable image when this
  token resolves to an RTX PRO 6000 Blackwell target.
- S3: input `NPA_INPUT_URI`; output `NPA_OUTPUT_URI`.
- Env/secrets: `COSMOS3_REASON_IMAGE`, `COSMOS3_REASON_MODEL`, prompt, run ID,
  and S3 credentials for `s3://` paths.
- HF rights: none. The default model value is `npa-cosmos3-reason`, not a HF
  repo ID; if you override it with a gated HF repo, accept that repo first.
- NGC entitlement: none declared by this YAML.

### `cosmos3-text-to-image-inference.yaml`

- Description: Cosmos3 text-to-image smoke that clones Cosmos source, downloads
  the checkpoint, runs inference, and optionally uploads output image and
  success JSON.
- Run: materialize `/tmp/cosmos3-text-to-image-inference.yaml`, then
  `sky jobs launch -n cosmos3-text-to-image-inference /tmp/cosmos3-text-to-image-inference.yaml -y`.
- Target: Kubernetes `H100:1`, headless, with large CPU, memory, and disk.
- S3: optional output prefix `NPA_COSMOS3_OUTPUT_S3_URI`.
- Env/secrets: `HF_TOKEN` is required by the YAML; `GITHUB_TOKEN` is optional
  for private forks; `NGC_API_KEY` is checked only when
  `NPA_COSMOS3_REQUIRE_NGC=1`; S3 credentials and BYO endpoint are needed only
  when uploading output.
- HF rights: none. The referenced
  [`nvidia/Cosmos3-Nano`](https://huggingface.co/nvidia/Cosmos3-Nano) repo is
  public according to current HF metadata.
- NGC entitlement: optional only when `NPA_COSMOS3_REQUIRE_NGC=1`.

### `isaac-lab-rl-train.yaml`

- Description: single Isaac Lab RSL-RL training job with checkpoint and manifest
  upload.
- Run: materialize `/tmp/isaac-lab-rl-train.yaml`, then
  `sky jobs launch -n isaac-lab-rl-train /tmp/isaac-lab-rl-train.yaml -y`.
- Target: Kubernetes `L40S:1`, 16 CPU, 64 GB memory. L40S is the RT-core target
  used for Isaac Lab rendering.
- S3: writes `isaac-lab-rl/<run-id>/`.
- Env/secrets: Isaac Lab task/run/env knobs, `AWS_ENDPOINT_URL`, S3 credentials.
- HF rights: none.
- NGC entitlement: none declared by this YAML.

### `isaac-lab-rl-sweep.yaml`

- Description: parallel Isaac Lab RSL-RL hyperparameter sweep with four variants.
- Run: materialize `/tmp/isaac-lab-rl-sweep.yaml`, then
  `sky jobs launch -n isaac-lab-rl-sweep /tmp/isaac-lab-rl-sweep.yaml -y`.
- Target: Kubernetes `L40S:1` per variant, 16 CPU, 64 GB memory. L40S is the
  RT-core target used for Isaac Lab rendering.
- S3: writes `isaac-lab-rl/<run-id>/<variant>/` for each variant.
- Env/secrets: Isaac Lab task/run/env knobs, variant overrides,
  `AWS_ENDPOINT_URL`, S3 credentials.
- HF rights: none.
- NGC entitlement: none declared by this YAML.

### `mjlab-eval.yaml`

- Description: MJLab locomotion eval against retargeted motion and a SONIC
  checkpoint.
- Run: materialize `/tmp/mjlab-eval.yaml`, then
  `sky jobs launch -n mjlab-eval /tmp/mjlab-eval.yaml -y`.
- Target: Kubernetes `H100:1`, headless eval.
- S3: reads retargeted motion and checkpoint JSON; writes MJLab output under the
  run prefix.
- Env/secrets: eval URIs, embodiment/suite/episode knobs, BYO S3 endpoint, S3
  credentials.
- HF rights: none.
- NGC entitlement: none.

### `retargeting.yaml`

- Description: CPU retargeting stage from source motion to SONIC-compatible
  retargeted motion.
- Run: materialize `/tmp/retargeting.yaml`, then
  `sky jobs launch -n retargeting /tmp/retargeting.yaml -y`.
- Target: Kubernetes CPU task.
- S3: reads source motion; writes retargeted motion; optional retarget map URI.
- Env/secrets: motion format, embodiment, frame/max-frame knobs, BYO S3 endpoint,
  S3 credentials.
- HF rights: none.
- NGC entitlement: none.

### `sim-to-real-loop.yaml`

- Description: self-hosted VLM eval loop that starts a VLM server, scores
  rollouts, and writes a task-success report.
- Run: materialize `/tmp/sim-to-real-loop.yaml`, then
  `sky jobs launch -n sim-to-real-loop /tmp/sim-to-real-loop.yaml -y`.
- Target: Kubernetes `H100:1`, headless VLM inference.
- S3: reads rollout data; writes `vlm-eval-loop/` output.
- Env/secrets: VLM endpoint/model/backend knobs, BYO S3 endpoint, S3
  credentials.
- HF rights: none. The referenced
  [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
  model is public according to current HF metadata.
- NGC entitlement: none declared by this YAML.

### `sim-to-real-pipeline.yaml`

- Description: generic sim-to-real training and eval pipeline with LeRobot
  dataset staging, Genesis-backed simulation defaults, policy training, eval,
  checkpoint output, and optional VLM feedback seam.
- Run: materialize `/tmp/sim-to-real-pipeline.yaml`, then
  `sky jobs launch -n sim-to-real-pipeline /tmp/sim-to-real-pipeline.yaml -y`.
- Target: Kubernetes GPU candidates `H100:1`, `H200:1`, and `L40S:1`; H100/H200
  are headless training/eval targets.
- S3: reads dataset input; writes raw envs, train/heldout splits, policy
  checkpoint, and Rerun artifact under `sim-to-real/<run-id>/`.
- Env/secrets: run/bucket/prefix, BYO S3 endpoint, S3 credentials, source repo
  and ref, trainer/image/backend knobs.
- HF rights: none. The referenced
  [`lerobot/pusht`](https://huggingface.co/datasets/lerobot/pusht) dataset is
  public according to current HF metadata.
- NGC entitlement: none declared by this YAML.

### `sim-to-real-trigger.yaml`

- Description: CPU polling trigger that watches an S3 dataset prefix and
  launches the sim-to-real pipeline when new data appears.
- Run: materialize `/tmp/sim-to-real-trigger.yaml`, then
  `sky jobs launch -n sim-to-real-trigger /tmp/sim-to-real-trigger.yaml -y`.
- Target: Kubernetes CPU task.
- S3: reads a watched prefix and watermark; writes or triggers pipeline input
  and output prefixes.
- Env/secrets: trigger limits, pipeline YAML path, task cloud, GPU target and
  failover, BYO S3 endpoint, S3 credentials.
- HF rights: none.
- NGC entitlement: none.

### `sim2real-actions.yaml`

- Description: RTX PRO 6000 action-conditioned rollout stage for the Sim2Real
  inner loop.
- Run: materialize `/tmp/sim2real-actions.yaml`, then
  `sky jobs launch -n sim2real-actions /tmp/sim2real-actions.yaml -y`.
- Target: Kubernetes `RTXPRO6000:1`; use sm_120/CUDA 12.8+ capable images when
  this token resolves to an RTX PRO 6000 Blackwell target.
- S3: reads train envs; writes action rollout output.
- Env/secrets: `POLICY_IMAGE`, action/env/run knobs, S3 credentials for URI
  paths.
- HF rights: none.
- NGC entitlement: none declared by this YAML.

### `sim2real-envgen-split.yaml`

- Description: RTX PRO 6000 environment generation and train/heldout split stage.
- Run: materialize `/tmp/sim2real-envgen-split.yaml`, then
  `sky jobs launch -n sim2real-envgen-split /tmp/sim2real-envgen-split.yaml -y`.
- Target: Kubernetes `RTXPRO6000:1`; use sm_120/CUDA 12.8+ capable images when
  this token resolves to an RTX PRO 6000 Blackwell target.
- S3: optional augmented frames input; output env/split URI.
- Env/secrets: `ENVGEN_IMAGE`, env count, shard count, seed, train fraction, S3
  credentials for URI paths.
- HF rights: none.
- NGC entitlement: none declared by this YAML.

### `sonic-train-standalone.yaml`

- Description: standalone SONIC training smoke with direct or nested-Docker
  payload mode.
- Run: materialize `/tmp/sonic-train-standalone.yaml`, then
  `sky jobs launch -n sonic-train-standalone /tmp/sonic-train-standalone.yaml -y`.
- Target: Nebius `L40S:1`, 16 CPU, 64 GB memory; L40S is the RT-core target used
  for the SONIC simulator path.
- S3: writes `sonic-train/<run-id>/` under `S3_BUCKET`.
- Env/secrets: policy image, payload mode, S3 bucket/prefix, BYO S3 endpoint, S3
  credentials, optional Docker registry secret envs.
- HF rights: none. The referenced
  [`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC) checkpoint
  repo is public according to current HF metadata.
- NGC entitlement: none declared by this YAML.

### `sonic-locomotion-finetuning.yaml`

- Description: serial SONIC locomotion workflow: retarget motion, finetune, then
  MJLab eval.
- Run: materialize `/tmp/sonic-locomotion-finetuning.yaml`, then
  `sky jobs launch -n sonic-locomotion-finetuning /tmp/sonic-locomotion-finetuning.yaml -y`.
- Target: CPU retargeting, Kubernetes `L40S:1` finetuning, and `H100:1`
  headless MJLab eval.
- S3: reads source motion; writes retargeted motion, training output, checkpoint
  JSON, and MJLab output under the run prefix.
- Env/secrets: source/retarget/train/eval URI values, BYO S3 endpoint, S3
  credentials, SONIC image and run knobs.
- HF rights: none. The referenced
  [`nvidia/GEAR-SONIC`](https://huggingface.co/nvidia/GEAR-SONIC) checkpoint
  repo is public according to current HF metadata.
- NGC entitlement: none declared by this YAML.

### `sonic-export.yaml`

- Description: exports a SONIC checkpoint to ONNX and sidecar metadata.
- Run: materialize `/tmp/sonic-export.yaml`, then
  `sky jobs launch -n sonic-export /tmp/sonic-export.yaml -y`.
- Target: Kubernetes `L40S:1`.
- S3: reads checkpoint; writes ONNX export and metadata.
- Env/secrets: checkpoint/output URIs, export specs, BYO S3 endpoint, S3
  credentials.
- HF rights: none.
- NGC entitlement: none.

### `sonic-eval.yaml`

- Description: evaluates an exported SONIC ONNX policy through the configured
  eval backend/container path.
- Run: materialize `/tmp/sonic-eval.yaml`, then
  `sky jobs launch -n sonic-eval /tmp/sonic-eval.yaml -y`.
- Target: Nebius `L40S:1` RT-core eval target.
- S3: reads ONNX and metadata; writes eval results JSON.
- Env/secrets: eval container/image/runtime args, BYO S3 endpoint, S3
  credentials.
- HF rights: none.
- NGC entitlement: none.

### `sonic-export-eval.yaml`

- Description: combined SONIC export plus eval job.
- Run: materialize `/tmp/sonic-export-eval.yaml`, then
  `sky jobs launch -n sonic-export-eval /tmp/sonic-export-eval.yaml -y`.
- Target: Nebius `L40S:1` RT-core export/eval target.
- S3: reads checkpoint; writes export artifacts and eval results.
- Env/secrets: policy/output paths, container/image/runtime args, BYO S3
  endpoint, S3 credentials.
- HF rights: none.
- NGC entitlement: none.

### `vlm-eval.yaml`

- Description: self-hosted VLM eval for rollout data.
- Run: materialize `/tmp/vlm-eval.yaml`, then
  `sky jobs launch -n vlm-eval /tmp/vlm-eval.yaml -y`.
- Target: Kubernetes `H100:1`, headless VLM inference.
- S3: reads eval input rollout URI; writes VLM eval output URI.
- Env/secrets: VLM backend/model/endpoint knobs, BYO S3 endpoint, S3
  credentials.
- HF rights: none. The referenced
  [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
  model is public according to current HF metadata.
- NGC entitlement: none declared by this YAML.

### `vlm-eval-benchmark.yaml`

- Description: benchmark sweep for VLM models, rubrics, and thresholds over a
  labeled rollout dataset.
- Run: materialize `/tmp/vlm-eval-benchmark.yaml`, then
  `sky jobs launch -n vlm-eval-benchmark /tmp/vlm-eval-benchmark.yaml -y`.
- Target: Kubernetes `H100:1`, headless VLM inference.
- S3: reads benchmark dataset JSON; writes benchmark results prefix.
- Env/secrets: VLM models/rubrics/thresholds, BYO S3 endpoint, S3 credentials.
- HF rights: none. The referenced
  [`Qwen/Qwen2-VL-7B-Instruct`](https://huggingface.co/Qwen/Qwen2-VL-7B-Instruct)
  model is public according to current HF metadata.
- NGC entitlement: none declared by this YAML.
