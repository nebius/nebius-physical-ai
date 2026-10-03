# Foundation-model training on Soperator

[Training workflow](../../../workflows/testing/policy-training-slurm.yaml) ·
[Slurm contract cookbook](policy-training-slurm.md) ·
[GR00T training](groot-1-7-training.md)

The training pipeline starts with a versioned robot corpus. Its stages are
curation, multi-task training, checkpoint evaluation, task-specific fine-tuning,
and policy evaluation. Data generation is a separate pipeline and is not part
of this graph.

For an implemented public-data path that runs native SmolVLA training and LIBERO
evaluation, start with [the turnkey VLA reference](public-vla-training.md).
It supports a Kubernetes Job or a Slurm/Pyxis allocation and exports real
checkpoint-bound rollouts. The preview below explains the larger architecture.

## See the public-data preview

With `uv` and `ffmpeg` installed, run:

```bash
bash npa/scripts/run-foundation-training-preview.sh /tmp/foundation-training-preview
```

This produces an offline `index.html`, a sixty-second 1920×1080 `demo.mp4`, a
poster, source clips, attribution, the Apache-2.0 license, an evidence manifest
and SHA256 checksums.
Use a new destination. Add `--html-only` to omit MP4 export. Dependencies use the
existing `adapter` and `policy-demo` extras and Playwright Chromium. On minimal
Linux hosts, install Chromium system dependencies with
`npa/.venv/bin/python -m playwright install --with-deps chromium`.

The exporter fetches four existing demonstrations from the Apache-2.0
[`lerobot/libero` dataset](https://huggingface.co/datasets/lerobot/libero/tree/a1aaacb7f6cd6ee5fb43120f673cebb0cfea7dd4).
The immutable revision, original video checksums, episode timestamps, task labels,
and transcoding outputs appear in `evidence.json`. The renderer never reads
customer inputs, private run logs, or cloud configuration. The HTML embeds the
clips, requires no server, and makes no network requests. Public download inputs
are cached under `${XDG_CACHE_HOME:-$HOME/.cache}/npa/foundation-preview`.

This is an **architecture preview**, not a completed foundation-training run.
The animation is schematic and the footage is dataset material, not trained
policy evaluation. There are no invented loss curves, GPU measurements, success
rates, or before/after claims. The analytical reaching demo in the Slurm cookbook
is a separate local contract test; it is not the foundation-model demonstration.

## Training requirements and implementation status

| Requirement | Current implementation and boundary |
| --- | --- |
| Slurm training managed through Kubernetes | Existing batch transport executes `sbatch` through the Soperator login jail. The distributed recipe renderer generates Pyxis/Enroot plus one torchrun launcher per node. |
| Foundation pretraining versus task adaptation | Launch recipes explicitly name `pretrain`, `continued-pretrain`, or `finetune`. A pretrained GR00T fine-tune must not be reported as from-scratch pretraining. The trainer owns the scientific recipe. |
| LeRobot v3 corpus | Curation validates v3 metadata and preserves episode references. The public SmolVLA runner consumes v3 metadata, Parquet and videos through native LeRobot; the architecture preview reads the same public corpus. A model adapter must verify or convert its supported data format at the training boundary. |
| Flagging distinct from selection | Every reviewed episode retains its quality decision and measurements in the curation report. `selection: all` keeps flagged episodes; `quality-pass` is the default. Preview-frame checks are implemented; full-video, trajectory and language quality analysis remain external inputs. |
| Weighted multi-task data | Phase-specific `training_selection` produces a hashed training manifest with normalized dataset probabilities. The trainer must consume those probabilities and exact episode indices. No holdout may be used for training. |
| Task-specific fine-tuning | Explicit `task_ids` filter the training partition. An empty task subset or missing corpus weight fails before batch submission. |
| Different entry events | Corpus events run the full graph. Dataset events can stop after curation or start fine-tuning from an approved checkpoint. Code events start fine-tuning and evaluation without redoing pretraining. |
| Checkpoint-triggered evaluation | A trainer hook hashes a completed checkpoint and its recovery files, atomically publishes a ready event, then submits an evaluation job with a durable receipt. This hook must be wired into the selected trainer's post-save barrier. |
| Shared model server with benchmark clients | Required evaluation topology: benchmark containers exchange observations/actions with one checkpoint-bound GPU policy server. Model-server and benchmark integration are not implemented by the generic batch adapter. |
| Model recovery | The ready-event contract requires model, optimizer, scheduler, RNG and sampler files. The public SmolVLA reference has passed a one-GPU interruption/resume test through its final step. Other trainers and multi-node allocations require their own qualification. |
| Final policy quality | The public SmolVLA reference records actual CUDA updates, 40 native rollout videos and an 8/10 final simulator score. These results do not qualify a different model, benchmark or physical robot. |

## Why Soperator is the default

Soperator preserves Slurm scheduling while managing its cluster on Kubernetes.
It supports shared storage, containerized jobs, gang scheduling, and cluster
health monitoring. These fit a training estate with existing `sbatch` recipes
and long-running multi-node jobs. See the
[official Soperator documentation](https://github.com/nebius/soperator).

`torchrun` launches distributed workers within an allocation; it does not
replace the cluster scheduler, allocate Kubernetes nodes, or automatically save
model state. The training script must save and restore its own state. See
[PyTorch fault-tolerant training](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html).

MK8s with torchrun can be a smaller deployment for an independent single-node
job. For the full workflow, compare gang allocation, rendezvous, failure cleanup,
checkpoint recovery, storage throughput and observability before replacing
Slurm. No performance advantage for either path is claimed without measurements.

Keep active datasets and checkpoints on the shared training filesystem. Archive
versioned artifacts to private object storage. Keep online augmentation and model
preprocessing inside the trainer. Record code/image identities, corpus selection,
mixture weights, checkpoint hashes, task/seed definitions and evaluation results.

## Prepare an event-specific workflow

Write a private event JSON. A code regression uses an existing split index and
an approved `npa.policy.decision.v1` checkpoint record:

```json
{
  "kind": "code-changed",
  "split_uri": "s3://example-bucket/training/splits/index.json",
  "approved_checkpoint_uri": "s3://example-bucket/training/approved.json"
}
```

Other kinds are `corpus-ready` (requires `episodes_uri`) and `dataset-arrived`
(requires `episodes_uri`). Dataset events default to curation only. Set
`benchmark: true` and supply `approved_checkpoint_uri` to follow curation with
task-specific fine-tuning and evaluation. New code/image selection remains in the
private batch settings; an event never rewrites an existing shared environment.

```bash
npa/.venv/bin/python -m npa.workflows.policy_training.foundation event \
  --input-path /tmp/training-event.json \
  --template-path workflows/testing/policy-training-slurm.yaml \
  --output-path /tmp/training-event.yaml
npa/.venv/bin/npa workbench workflow validate-spec /tmp/training-event.yaml --json
```

The module materializes files only. Use the existing generic workflow
`plan-spec` and runtime `submit` commands from the Slurm cookbook after verifying
private inputs, access, storage, images, scripts and the exact runtime target.
It creates no clusters and submits no jobs during preparation.

## Select a training mixture

Add `task_id` to episode records and add `training_selection` to private batch
settings. Each configured training phase requires an explicit policy:

```json
{
  "training_selection": {
    "pretrain": {
      "task_ids": [],
      "dataset_weights": {
        "s3://example-bucket/corpus-v1": 3,
        "s3://example-bucket/corpus-v2": 1
      }
    },
    "finetune": {
      "task_ids": ["place-object"],
      "dataset_weights": {
        "s3://example-bucket/corpus-v1": 1
      }
    }
  }
}
```

An empty pretraining task list means all training tasks. Fine-tuning requires
at least one explicit task. Weights must cover exactly the datasets remaining
after task selection; positive finite weights normalize to probabilities. The
sampling contract is dataset first, then an episode from that dataset. A trainer
that ignores these weights does not implement this contract.

The adapter binds the selected manifest digest into the request and checks it
again on retry. Both development holdouts remain evaluation-only. Repeated
checkpoint selection makes those holdouts development sets; reserve a separate
untouched final test before making a generalization claim. Use immutable source
versions and have the consuming trainer validate the referenced data bytes.

## Render distributed training

A private recipe JSON requires `mode`, `nodes`, `gpus_per_node`, `cpus_per_node`, `memory_gib`, `master_port`,
`image`, `shared_path`, `working_directory`, `output_path`, and `trainer_argv`.
The image must be an immutable registry reference ending in `@sha256:<digest>`.
The working and output directories must be beneath the shared mount. Choose a
rendezvous port appropriate to the allocation. `output_path` creates the shared run directory; also pass it through the
trainer-specific output flag in `trainer_argv`. `trainer_argv` starts with the
trainer's Python script path, followed by literal arguments; do not include
`python` or `torchrun` as its first element.

```bash
npa/.venv/bin/python -m npa.workflows.policy_training.foundation slurm \
  --input-path /tmp/training-recipe.json \
  --output-path /tmp/train.sbatch
bash -n /tmp/train.sbatch
```

Stage the script and code on the Soperator shared jail filesystem. Submit from
the login jail with `sbatch --output=<shared-log-path> /shared/train.sbatch`.
The script requests the configured nodes and GPUs, uses one Slurm task per node,
and gives each torchrun launcher its Slurm node rank. Pyxis mounts the shared
path into the pinned container. Failed Slurm steps terminate sibling launchers.
There is no imposed job-duration, retry-count or spending limit.

This generic launcher does not turn a single-node trainer into a verified
multi-node implementation. In particular, use the existing qualified GR00T
single-node workflow until the selected upstream training configuration passes
multi-node qualification. The intended public model is
[GR00T N1.7 3B](https://github.com/NVIDIA/Isaac-GR00T); its backbone requires exact
Hugging Face payload access before GPU execution:

```bash
npa/.venv/bin/npa workbench health access --capability groot --json
```

The preview records this run as pending. It is not an access test or an attempt
to fetch gated weights. No GPU job is launched by the preview exporter.

## Connect checkpoint evaluation

Call the hook only after the trainer has finished writing every rank's state,
all files are closed, and a distributed save barrier has completed. Supply the
complete file list for each recovery role; do not create placeholder state files.
The sampler role must represent the actual data cursor/order needed by the
trainer, rather than an unrelated empty marker.

```python
from pathlib import Path
from npa.workflows.policy_training.checkpoints import (
    enqueue_evaluation,
    publish_checkpoint,
)

# state_files maps model/optimizer/scheduler/rng/sampler to actual saved files.
event = publish_checkpoint(checkpoint_directory, optimizer_step, state_files, events_directory)
receipt = enqueue_evaluation(event, Path("/shared/jobs/evaluate.sbatch"), receipt_directory)
```

The evaluation script receives `--checkpoint-event <absolute-path>`. It must
load that exact checkpoint into the shared policy server and record benchmark
metrics/videos bound to the checkpoint digest, held-out task set and seeds.
This is a serving architecture, not a pair of actuator perturbations.

Publication hashes actual files and persists the event atomically. Submission
reverifies checkpoint bytes and uses an exclusive pending receipt to avoid
concurrent duplicate jobs. A failed or interrupted submission leaves a pending
receipt: reconcile the deterministic Slurm job name before retrying. A receipt
means submitted, not evaluated or passed. Store all events, receipts and raw
scheduler output privately; none belong in the shareable preview.

## Validate

```bash
npa/.venv/bin/python -m pytest \
  npa/tests/workflows/test_foundation_training.py \
  npa/tests/workflows/test_policy_training.py \
  npa/tests/orchestration/npa_workflow/test_policy_training_spec.py -q
```

These tests cover event routing, training/holdout isolation, mixtures, flagging,
Slurm recipe validation, checkpoint tampering, complete recovery metadata,
idempotent submission and ambiguous-submission refusal. They do not replace
GPU training, restore, model-server or simulator qualification.
