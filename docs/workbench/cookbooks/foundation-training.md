# Policy training design and operator batch adapters

[Single turnkey workflow](../../../workflows/testing/robot-policy-train-and-serve.yaml) ·
[Operator cookbook](robot-policy-train-and-serve.md) ·
[Slurm adapter contracts](policy-training-slurm.md)

Start with the single Workbench workflow for a fully implemented public-data
baseline: pinned LeRobot data, real FiftyOne curation, grouped splits, native
SmolVLA continued training, two promotion gates, checkpoint export, authenticated
GPU serving and an independent GPU-rendered LIBERO client. It produces standalone
HTML/MP4 proof from the actual run through standard catalog and runtime surfaces.
Its qualified deployment scored 9/10 with all 947 served/applied actions
reconciled. This is continued training of a pretrained model's action expert,
not foundation pretraining from scratch. Data generation is a separate pipeline.

The canonical recipe uses managed Kubernetes and torchrun. The following design
contracts also support operator-owned Slurm/Soperator scripts and private
workflows; they are not additional shipped turnkey recipes.

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
| Shared model server with benchmark clients | The public Workbench recipe deploys one checkpoint-bound GPU HTTP server and one independent GPU-rendered LIBERO client. Every returned/applied action is reconciled in the report. Concurrent clients and batching are not qualified by this recipe; the generic batch adapter does not supply a model server. |
| Model recovery | The ready-event contract requires model, optimizer, scheduler, RNG and sampler files. The public SmolVLA reference has passed a one-GPU interruption/resume test through its final step. Other trainers and multi-node allocations require their own qualification. |
| Final policy quality | The canonical workflow records 3727 multi-task updates, 2431 specialist updates, two 7/10 promotion scores and a separate 9/10 deployed simulation result. These results do not qualify a different model, benchmark or physical robot. |

## Scheduling operator-owned batch jobs

Soperator preserves Slurm scheduling while managing its cluster on Kubernetes.
It supports shared storage, containerized jobs, gang scheduling, and cluster
health monitoring. These fit a training estate with existing `sbatch` recipes
and long-running multi-node jobs. See the
[official Soperator documentation](https://github.com/nebius/soperator).

`torchrun` launches distributed workers within an allocation; it does not
replace the cluster scheduler, allocate Kubernetes nodes, or automatically save
model state. The training script must save and restore its own state. See
[PyTorch fault-tolerant training](https://docs.pytorch.org/tutorials/beginner/ddp_series_fault_tolerance.html).

The canonical workflow uses managed Kubernetes plus torchrun for its complete
GPU training and serving path. Use the Slurm adapter when integrating existing
operator batch jobs. Qualify gang allocation, rendezvous, failure cleanup, recovery
and throughput for each different trainer and infrastructure configuration.

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

The event adapter accepts an operator-owned batch workflow template. It
materializes a private spec without creating clusters or submitting jobs.
No batch template is shipped as another pipeline entry point; the JSON fixture
used by contract tests does not supply the required private inputs or scripts.

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

The public turnkey recipe serves SmolVLA; it does not qualify GR00T training or
fetch gated GR00T weights. Access and execution checks are specific to the model.

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
