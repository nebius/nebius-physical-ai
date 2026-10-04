# Fine-tune FLUX 3 Action for your robot

`npa workbench flux-action finetune` adapts BFL's action-pretrained base using
its standalone full fine-tuning trainer. It accepts a declared robot embodiment
rather than assuming SO-101 controls. The same operation is available as
`npa.sdk.workbench.flux_action.finetune` and workflow step
`workbench.flux_action.finetune`.

**Status:** implemented integration with CPU contract tests. The exact candidate
image has been built, scanned and privately published. Public publication and
automatic GPU rotation remain quarantined. GPU train/export/reload and robot
task performance are not yet qualified.

## Inputs

- `--input-path`: S3 prefix containing a LeRobot dataset's `meta/`, `data/`, and
  `videos/`. Native upstream supports v2.x and v3.x; use video camera features.
- `--recipe-uri`: an S3 JSON object specifying robot semantics and training.
- `--output-path`: a fresh, run-specific S3 prefix. A conditional `run.json`
  claim prevents two invocations from training into the same output.
- `--processes`: local GPU workers, default 8. Match the workflow allocation.
  Multi-node execution is rejected by the tool catalog.
- `--dry-run`: read the recipe and dataset metadata, validate their agreement,
  and print the intended native configuration. It does not validate video bytes
  or prove trainability. It performs no model download or output write.
- `--output-format`: `json` (default) or `text`.

Example recipe for a three-channel arm (replace every channel and unit with
those actually recorded; this is a schema example, not a qualified robot):

```json
{
  "robot": {
    "embodiment": "my_arm",
    "actions": [
      {"name": "shoulder", "unit": "rad"},
      {"name": "elbow", "unit": "rad"},
      {"name": "gripper", "unit": "closed_fraction"}
    ],
    "states": [
      {"name": "shoulder", "unit": "rad"},
      {"name": "elbow", "unit": "rad"},
      {"name": "gripper", "unit": "closed_fraction"}
    ],
    "cameras": {"front": "observation.images.front"},
    "camera_layout": "single",
    "canvas_hw": [512, 512],
    "fps": 20,
    "action_parameterization": "absolute",
    "absolute_action_dims": [],
    "gripper_flip_dims": [],
    "n_action_steps": 8
  },
  "training": {
    "steps": 3000,
    "checkpoint_every": 250,
    "frozen_steps": 200,
    "trunk_warmup_steps": 600,
    "heads_warmup_steps": 200,
    "windows_per_rank": 4,
    "grad_accumulation": 1,
    "num_workers": 1,
    "visits_per_epoch": 1,
    "optimizer_lr": 0.00005,
    "optimizer_lr_heads_multiplier": 5.0,
    "caption_dropout": 0.0,
    "seed": 42
  },
  "export_profile": "ema_0p10"
}
```

Choose your own update count and schedule. The numbers above illustrate the
BFL game schedule and are not a measured optimum or training limit for an arm.
Steps, checkpoint frequency, frozen steps, and both warmups are required.
No checkpoint-retention limit, epoch cap, cooldown, or job deadline is imposed.
`val_episodes` defaults to 1 held-out episode at indexing; set 0 only for an
explicit training-window reload smoke. The ALOHA example holds out 5 episodes.
`steps` must exceed `frozen_steps + 1` so a completed run reaches trunk training.
The other training defaults are batch/accumulation/workers/visits = 1, seed 42,
learning rate `5e-5`, head multiplier 5, and caption dropout 0.

### Embodiment limits and alignment

State and action widths must match in the current upstream model; they need not
be six channels or SO-101 joints. This wrapper does not project unequal vectors.
Declared channel names must match LeRobot metadata when names are present.
Units are operator declarations: the wrapper cannot infer physical correctness
from a float array. Keep the order and meaning identical at deployment.

Camera mappings preserve JSON insertion order. Use `single` for one camera,
`side_by_side` for two, or `grid` for arbitrary camera counts. Canvas dimensions
must be positive multiples of 32. No missing-camera gray-tile fallback is used.
`state_key` defaults to `observation.state`; `action_key` defaults to `action`.
FPS must be a positive integer matching the recording. Changing it does not
resample the data. The fixed target chunk is 32 actions; `n_action_steps`
(default 8, range 1–32) records how many actions to execute per inference plan.

Each indexed episode needs at least 34 frames: native LeRobot windows start at
frame 1 or later. Image/state at `s` condition commands `s:s+32` and future
images `s+1:s+33`. `joint_delta` means **successive command differences**, not
command minus measured state. Under that representation, list channels that
remain absolute in `absolute_action_dims`. Indices are zero-based, nonnegative,
and validated. Gripper inversion defaults to none and must be explicitly set.

The native index computes percentile normalization over all indexed episodes,
including the episodes reserved by `val_episodes`. Those episodes are excluded
from gradient updates, but the reload check is not a statistically independent
evaluation. Keep a separate test corpus outside this input for policy-quality
measurement. Native indexing rejects invalid rows and excludes short episodes;
inspect `index/manifest.json` for counts and exclusions.
A full distributed update needs enough episode visits across ranks, workers,
windows, and accumulation; increase `visits_per_epoch` deliberately for small
corpora. Native training fails if it cannot fill an update.

## Public ALOHA example

A concrete non-SO101 input is
[`lerobot/aloha_sim_transfer_cube_human`](https://huggingface.co/datasets/lerobot/aloha_sim_transfer_cube_human),
pinned to `6a43d500f101255823a9d2b9dc244eeb01a2cd31` (dataset card: MIT).
It contains 50 simulated bimanual demonstrations, 20,000 frames at 50 Hz,
14 state/action channels, and the `observation.images.top` video camera.
The first six channels of each arm are joint positions in radians; each seventh
channel is a normalized gripper opening (recorded commands may overshoot the
nominal 0–1 interval). The example leaves those values unchanged.
The [original simulator](https://github.com/tonyzhaozh/act/blob/main/sim_env.py)
defines their order and normalization.

Download it without model weights:

```bash
uvx --from huggingface-hub hf download lerobot/aloha_sim_transfer_cube_human \
  --repo-type dataset --revision 6a43d500f101255823a9d2b9dc244eeb01a2cd31 \
  --local-dir /tmp/flux-aloha-data
```

Stage that directory to your dataset S3 prefix, and stage
[`flux-action-aloha-smoke.json`](../../npa/workflows/workbench/configs/flux-action-aloha-smoke.json)
as the recipe object. For example, after loading NPA storage credentials:

```python
from npa.clients.credentials import load_credentials
from npa.clients.storage import StorageClient

credentials = load_credentials(export_to_environment=True)
storage = StorageClient.from_environment(
    endpoint_url=credentials.s3_endpoint,
    aws_access_key_id=credentials.s3_access_key_id,
    aws_secret_access_key=credentials.s3_secret_access_key,
)
storage.upload_directory("/tmp/flux-aloha-data", "s3://your-bucket/flux-aloha/dataset",
                         require_empty=True)
storage.upload_file("npa/workflows/workbench/configs/flux-action-aloha-smoke.json",
                    "s3://your-bucket/flux-aloha/recipe.json")
```

The recipe requests the guide's initial four-update qualification run with no
trunk freeze; replace its schedule for actual learning experiments. Those four
updates test native execution and artifacts, not policy quality. On 2026-09-23,
the actual pinned dataset indexed successfully with **50 eligible episodes,
zero rejected rows, and 18,350 valid window starts**. A real decoded sample had
`action=(32,14)`, `state=(14,)`, `images.top=(33,3,360,640)`, and finite actions.
The last shape reflects the native loader's default resize. This CPU result is
not GPU training evidence. All 50 episodes belong to the published training
split; use separate held-out data/seeds for task evaluation.

## Run on Nebius

Build the candidate with `npa/docker/workbench/flux-action/build.sh` after
committing its build inputs. The helper creates a local immutable dev-SHA image;
`--push` is refused until the trusted publication workflow's image checks pass.
Follow the repository secure-image-build process for private validation.

Use [the reference workflow](../../workflows/testing/flux-action-finetune.yaml),
set the dataset/recipe/output URIs, and override the `workbench.flux_action.finetune`
image with the privately built candidate. Its initial allocation is one H200
node with eight GPUs, matching BFL's eight-process launch. This is a starting
configuration, not an NPA-measured capacity claim. Training stores FP32 master
weights, gradients, optimizer state, and EMA profiles; inference memory figures
do not predict training requirements.

Inside that runtime, the direct command is:

```bash
npa workbench flux-action finetune \
  --input-path s3://your-bucket/datasets/robot-training \
  --recipe-uri s3://your-bucket/recipes/flux-action.json \
  --output-path s3://your-bucket/runs/unique-run/flux-action \
  --processes 8
```

S3 credentials use NPA's standard storage environment. `HF_TOKEN`, when supplied, is forwarded
through workflow secret plumbing. The pinned base is currently publicly
readable; a token is optional for direct invocation and grants no additional
model-use rights. The live-submit test matrix requires it to exercise credential
forwarding.
`HF_HOME` selects the runtime cache; without configured shared model-cache
storage it is node-local and disappears with the worker. `NPA_FLUX_ACTION_ROOT`
defaults to `/opt/flux-action`; an alternate runtime must contain the matching
`SOURCE_REVISION` marker and `.venv/bin/python` plus `.venv/bin/hf`.

The container pins source `e2dd1d8dbc5977b54315d61f7548c63c043d6d4f`, Linux
Python 3.12, upstream's locked Torch 2.10/CUDA 12.8 environment, and its documented
NATTEN wheel. It downloads the base and encoders at revision
`eb267865d35e49e4066bde4936237f8d9f15a68c` at runtime. Source is Apache-2.0;
model weights use the FLUX Kommunity License v1.0. Use must satisfy the
[model's exact terms](https://huggingface.co/black-forest-labs/flux-3-action-base/blob/eb267865d35e49e4066bde4936237f8d9f15a68c/LICENSE.md).

The golden eval uses `NPA_FLUX_INPUT_URI`, `NPA_FLUX_RECIPE_URI`,
`NPA_FLUX_OUTPUT_URI`, and optional `NPA_FLUX_PROCESSES` (default 8), invoking
`python -m npa.smoke.test_flux_action_functional`. It requires real training
inputs and records the same verified result; it does not substitute an import
probe for training.

## Outputs and failure behavior

The output prefix contains `recipe.json`, native `training.json`, the index and
statistics, stage logs, `train/metrics.jsonl`, resumable `train/step-N` checkpoints,
and `export/{config.json,model.safetensors,manifest.json}`. The wrapper requests
BF16 export from `model`, `ema_0p10` (default), or `ema_0p05`.

`result.json` is uploaded last, after checking the requested final COMPLETE
checkpoint, finite metrics, fresh-process reload inference, and independently read-back
S3 export hashes. It records source,
base, recipe/index/statistics identities, embodiment, encoder references, and
artifact URIs. Partial S3 uploads have no completion receipt. Exported encoders
remain external pinned Hugging Face references, fetched under the consumer's
own authorization and cache; the export is not a self-contained model bundle.

Checkpoints are uploaded at job completion or a handled failure; sudden worker
loss can discard unuploaded local progress. On failure, the wrapper retains its
local staging directory and attempts to
upload existing logs/checkpoints plus `failure.json`. It never emits a completed
receipt for a failed trainer or export. A retry needs a fresh output prefix;
the wrapper deliberately does not overwrite or automatically resume a claimed
run. To resume manually with BFL's trainer, recover a complete checkpoint and
its index/data, update local paths in `training.json`, and use the upstream
`resume` setting. Preserve the data, batch, schedule, and normalization contract.

The finite job uses CLI/SDK/workflow invocation; it does not run an HTTP training
service or expose robot actuation. Evaluate exports in a separate held-out
closed-loop environment before drawing task-performance conclusions.

Sources: [BFL fine-tuning guide](https://docs.bfl.ai/flux_3/flux3_action_finetuning),
[pinned source and contract research](../research/flux-action-finetuning.md).

## Full-weight execution and completion checks

This command uses the standalone trainer, starts from the pinned action-pretrained
base, initializes the declared embodiment heads, and trains the trunk after its
explicit frozen/warmup schedule. Video/text encoders and unused content streams
retain upstream freezing. No LoRA/PEFT adapters are configured. Native `steps`
counts **optimizer updates**, unlike the SO-101 LeRobot command's microsteps.
The result records `training_mode: full` and `optimizer_updates` explicitly.

Use `npa/workflows/workbench/configs/flux-action-aloha-full.json` for the 3,000-update
public ALOHA example; the neighboring smoke recipe uses four updates. Both retain
the 14-channel, 50 Hz, one-camera contract for the documented transfer-cube dataset.
Stage the chosen recipe in S3 and provide its URI to the linked workflow. The
resource profile starts with eight H200s on one node, 512 GiB host RAM, and
1,024 GB local disk for FP32 training/checkpoints. It is an unbenchmarked starting
configuration; match `flux_processes` to actual GPU allocation and measure it with
the four-update recipe before a full run.

The workflow accepts `flux_image` at an independently scanned immutable private
digest and `image_pull_secret` for its existing registry Secret. Supply
`flux_input_uri`, `flux_recipe_uri`, and a fresh output prefix. The live test
materializer requires `NPA_E2E_FLUX_ACTION_IMAGE`, `NPA_E2E_FLUX_ACTION_PULL_SECRET`,
`NPA_E2E_FLUX_ACTION_INPUT_URI`, and `NPA_E2E_FLUX_ACTION_RECIPE_URI`.

Completion requires positive trunk/head learning rates, finite metrics through
the final optimizer update, a complete native checkpoint, verified BF16 export,
and a **fresh-process** native reload that predicts one real indexed window.
`reload.json` records finite normalized/raw action errors and the declared channel
width. This is a functional reload check; one window does not establish policy
quality. S3 export files are independently stream-hashed before `result.json` is
written last. Failed stages retain logs/checkpoints without a success receipt.

CPU validation exercised the pinned native indexer, real video decoding, native
training configuration, and temporal windows for 3-, 7-, and 14-channel robots.
The exact image also passed SSH startup, CLI help and all three native CPU
channel-contract tests with Torch 2.10/CUDA 12.8. Its payload scan passed; pinned
Trivy reported zero secret findings and zero critical vulnerabilities, with a
331-package SPDX SBOM. Source and immutable image digest are recorded in the
[readiness record](../../workflows/testing/flux-action-finetune.readiness.json).
GPU train/export/reload remains unverified; the completed SO-101 LoRA run does
not qualify this separate runtime.

### Cloud acceptance attempts

The candidate image and pinned ALOHA smoke inputs passed their build, scan,
access and planning checks. Eight H200s and four L40S GPUs returned
`NotEnoughResources`; eight H100s returned `QuotaFailure`.

A later single-H100 attempt passed CUDA vector addition and a 120-second
stability check, indexed the real dataset, fetched the pinned weights, and
entered the full-weight trainer. It exhausted the GPU's 79.18 GiB while creating
FP32 EMA copies, before the first optimizer update. Its failure logs were
preserved in object storage. Trainable parameters numbered 6,909,046,016.
This is a failed training attempt, not end-to-end acceptance.

### Single-GPU qualification recipe

Use [the single-GPU workflow](../../workflows/testing/flux-action-finetune-single-gpu.yaml)
with [its four-update ALOHA recipe](../../npa/workflows/workbench/configs/flux-action-aloha-single-gpu-smoke.json).
The recipe uses native BF16 parameters, disables EMA copies, and exports the
trained `model` profile. The full trunk and robot heads train with positive
learning rates. This precision recipe requires its own GPU qualification;
the default FP32 parameters and two EMA profiles remain the reference settings.

Supply the immutable private image, dataset URI, recipe URI, and an already
Bound disk PVC through `checkpoint_claim`. The PVC holds temporary training
files and the HF cache at `/npa-work`. Preserve the system `/tmp`: mounting a
fresh disk there can prevent apt's unprivileged verification process from
creating temporary files during SkyPilot setup. Keep failed artifacts until
their object-storage copies have been checked.

The first BF16 cloud attempt completed all four optimizer updates, a complete
native checkpoint and the model export. Its fresh evaluator rejected the
export because native training leaves sampler settings unset. The exact job
terminated `FAILED`; its checkpoint, export and failure logs were preserved
in object storage. This partial run does not qualify the workflow.

Recipes now carry explicit `inference` settings into the exported policy.
The ALOHA qualification recipe uses Euler sampling, four denoising steps,
shift 1, guidance 1 on video and actions, and seed 0. These are diagnostic
settings, not a benchmarked ALOHA preset. Override them for policy evaluation.
Native CPU checks validate both training and inference configurations before
GPU submission.

An EMA export requires its corresponding value in `training.ema_sigma_rels`.
Supported values are `0.1` and `0.05`; an empty list requires `export_profile`
to be `model`. `training.param_dtype` accepts `float32` or `bfloat16`.

The single-GPU candidate, eight-GPU reference workflow, and alternate
four-process plan remain unqualified until complete checkpoint, export,
fresh-process reload, remote checksums and native terminal success are verified.
The 3,000-update recipe and closed-loop policy quality are unmeasured.
