# Legacy public VLA operator reference

For the complete agent-operated pipeline, use the single
[robot-policy-train-and-serve workflow](robot-policy-train-and-serve.md).
The direct-launch implementation below is retained as an operator reference.

This reference downloads a pretrained SmolVLA policy and public LeRobot v3
demonstrations, performs real gradient updates, adapts the policy to one task,
evaluates actual robot rollouts in LIBERO, and exports a trained checkpoint and
an offline HTML report. All model, demonstration and simulator-asset inputs use
immutable Hugging Face revisions. It does not generate training data.

The starting policy is `HuggingFaceVLA/smolvla_libero`, which was already trained
on LIBERO. This is **continued training and task adaptation**, not from-scratch
foundation pretraining or an unseen-benchmark result. The complete reference is
implemented; there are no operator-written training or evaluation script slots.

This is an **out-of-band operator reference**, launched directly through
Kubernetes or Slurm. It is not an agent-discoverable Workbench `toolRef` or
`npa.workflow`: it has no SkyPilot submission, Workbench run ledger, or durable
interpreter resume. Its argparse `--input-path` and `--output-path` options use
local files, and its console output is human-readable. The canonical
train-and-serve workflow uses catalog stages and the standard Workbench runtime;
its measured qualification covers both promotion gates and independent GPU
serving. Results from this older direct launcher do not qualify that workflow.

## Run on Nebius Managed Kubernetes

Start with Workbench installed with its `adapter` extra, a configured `kubectl` context,
one available NVIDIA GPU, and a default persistent-volume storage class. The
reference requests 16 CPU cores, 96 GiB of host memory, one GPU and a 100 GiB
persistent volume. Its pinned LeRobot 0.6.0 image contains the required CUDA,
SmolVLA, LIBERO and video-decoding dependencies. Public inputs need no HF token.

```bash
bash npa/scripts/run-public-vla.sh \
  --backend kubernetes --output-path ./outputs/public-vla-launch

npa/.venv/bin/python -m npa.workflows.policy_training.public_vla_collect \
  --input-path ./outputs/public-vla-launch/receipt.json \
  --output-path ./outputs/public-vla-results
```

The first command submits the complete job. The second waits for completion or
failure and collects `report/index.html`, native rollout MP4s, measured evidence,
and `exported-policy/`. Launch and collection directories must be new. Pass
`--kube-context`, `--kubeconfig` or `--storage-class` when the runtime needs an
explicit selection. No cluster is created or replaced by this launcher.

Checkpoints, optimizer state and logs remain on the run's persistent volume.
The GPU allocation ends when the Job exits. Collection removes its temporary
CPU reader Pod. Add `--cleanup` to collection to delete the owned namespace and
100 GiB volume after verifying completion, downloaded weights against the saved
checksum, namespace UID and ownership label. This also deletes optimizer state
and private worker logs, so omit it when those are needed for recovery. Cleanup
refuses active/failed jobs, unexpected jobs and old receipts without an ownership
UID. To retry cleanup using an existing collection:

```bash
npa/.venv/bin/python -m npa.workflows.policy_training.public_vla_cleanup \
  --input-path ./outputs/public-vla-launch/receipt.json \
  --output-path ./outputs/public-vla-results
```

Namespace deletion is verified before writing a private `cleanup.json` receipt.
A retained persistent volume with a `Retain` reclaim policy remains an
operator-owned storage resource; namespace deletion does not prove its bytes
were erased.
It does not change project-wide storage settings or publish models to the Hub.

For a single 1920×1080 demonstration film, install the existing
`policy-demo` extra, Playwright Chromium and `ffmpeg`, then run:

```bash
npa/.venv/bin/python -m npa.workflows.policy_training.public_vla_media \
  --input-path ./outputs/public-vla-results/report
```

This creates `demo.mp4` and `poster.png` from the measured dashboard and the
three paired validation examples, prioritizing improved outcomes, regressions,
and a shared success when available. Aggregate scores still cover every evaluated
episode; the film records its explicit example-selection rule. Playback uses the
simulator's 20 Hz control rate: the native evaluator encodes one control-step
frame at 80 fps, so these recordings play at one quarter of their encoded rate.
Each comparison includes complete rollouts and holds completed episodes on
their final frame. The film records source hashes and timing in
`video-provenance.json`; all recorded episodes remain available in HTML.

Headless rendering uses NVIDIA EGL when available. On compute images without
NVIDIA graphics libraries, the runner selects a working Mesa EGL software
device. Simulator rendering is then CPU work; model training and policy
inference still run on CUDA. The report records the rendering backend.

## Run with Slurm / Soperator

Run this command **inside the Soperator login jail**, with Workbench installed
and Pyxis/Enroot available. Choose a new shared directory visible at the same
path to the login node and GPU workers:

```bash
bash npa/scripts/run-public-vla.sh \
  --backend slurm \
  --shared-path /shared/public-vla-run \
  --output-path ./outputs/public-vla-slurm-launch
```

This submits an actual `sbatch` job. Slurm reserves one GPU and `srun` starts the
pinned container through Pyxis. The same runner then invokes `torchrun` for
native LeRobot training. Results are in `/shared/public-vla-run/run/`.
The launcher does not deploy Soperator; use the [Soperator skill](../../../skills/tools/soperator/SKILL.md)
to create a cluster when needed. A Kubernetes Job is the smaller deployment
for this one-GPU reference; Soperator fits an existing Slurm training estate.
Neither path imposes a job-duration or spending limit. Failed Kubernetes Jobs
retain their artifacts and stop for diagnosis instead of retrying automatically.

## Scientific recipe

The default continues training for one pass over the selected multi-task
training frames, then adapts to LIBERO Spatial task 0 for roughly ten passes over that
task's training frames. Optimizer steps are derived from the selected frame
count, epoch count and batch size. This is a reproducible execution recipe,
not a claim of optimal convergence. Model quality is measured separately.
All compared policies execute ten action steps per model call; this execution
horizon is also saved in the exported policy configuration.

Use `--input-path recipe.json` on either launcher to override these fields:

```json
{
  "seed": 42,
  "batch_size": 64,
  "generalist_epochs": 1,
  "specialist_epochs": 10,
  "suite": "libero_spatial",
  "task_id": 0,
  "evaluation_episodes": 10,
  "minimum_success": 0.7,
  "workers": 8
}
```

- **Curation and split:** nonfinite action/state trajectories are flagged, all
  episode decisions are retained, and only eligible rows enter training. Exact
  duplicate trajectory groups stay together. Each task has training, validation
  and test episodes. Normalization is fitted only on the selected training rows.
  This automatic public-data check does not perform human review or run FiftyOne.
- **Generalist training:** the real LeRobot SmolVLA action expert receives
  multi-task observations, instructions and actions. The pretrained vision and
  language backbone stays frozen. Loss, gradient norm, learning rate, update
  duration and CUDA memory are measured at optimizer updates.
- **Task adaptation:** the selected task's training episodes and normalization
  feed a second native fine-tune initialized from the generalist checkpoint.
- **Validation:** the public baseline and both trained candidates run on the
  same fixed simulator starting states. The default records all ten evaluated
  episodes; larger evaluations retain the native evaluator's first ten videos.
  The better trained candidate is selected; ties prefer the specialist.
- **Final test:** the selected candidate runs once on a disjoint range of fixed
  initial states. With ten episodes, validation uses states 0–9 and final test
  uses states 10–19. Different seeds alone do not separate LIBERO initial states.
  This is separation within this run; the pretrained policy has prior LIBERO
  exposure. The held-out demonstration episodes are retained in the manifest
  and are not used to fit or select these policies.
- **Export:** actual weight bytes must differ from the starting checkpoint.
  The inference bundle includes processor state and local tokenizer/config
  dependencies. Load it from the exported bundle directory so relative
  tokenizer and backbone paths resolve. The complete runner prepares LIBERO
  assets and simulator configuration; those assets are not in the model bundle.
  The pipeline reloads the bundle with Hugging Face access disabled, infers a
  finite seven-axis action on CUDA from a public demonstration, and verifies
  changed state-projection tensor values against the starting model.
  Training recovery files remain in the original checkpoints directory.

Changing the recipe requires a new run directory. Restarting the same recipe
reuses pinned downloads and completed stages and restores the latest native
checkpoint when training was interrupted. A failure before the first checkpoint
is retained for diagnosis and requires a fresh run directory. The simulator
reference is single-node; it does not claim multi-node scaling, a shared remote
policy server, or physical-robot deployment qualification.

After an interrupted Kubernetes Job is absent or failed, resume its original
launch receipt and retained volume with:

```bash
bash npa/scripts/run-public-vla.sh --backend kubernetes --resume \
  --output-path ./outputs/public-vla-launch
```

This refuses to replace an active or completed Job. For Slurm, resubmit the
saved `pipeline.sbatch` from its shared run directory after the allocation has
ended. Recovery restores native training state; bitwise replay is not promised.

## Observed GPU qualification

A complete run on one NVIDIA RTX PRO 6000 Blackwell Server Edition used the
pinned LeRobot image with PyTorch 2.11.0+cu130, Kubernetes and one torchrun rank.
The model has 604,934,176 parameters; 97,451,872 were trainable. Curation retained
1,477 training episodes and 108 episodes in each demonstration holdout.

| Policy / evaluation | Successful episodes |
| --- | --- |
| Downloaded policy, validation states 0–9 | 6 / 10 |
| Generalist after 3,727 updates, same validation states | 8 / 10 |
| Specialist after 608 updates, same validation states | 7 / 10 |
| Selected generalist, final test states 10–19 | 8 / 10 |

The configured 70% simulator gate passed. These ten-episode measurements are
execution evidence for this LIBERO-pretrained reference, not broad generalization
or physical-robot qualification. The task-specific update did not beat the
multi-task checkpoint, and the selection logic retained the stronger candidate.

An interruption restored the step-931 native checkpoint, resumed at step 932,
and continued through step 3,727. Recovery files were checksum-verified; bitwise
replay was not established. The exported bundle subsequently loaded with Hub
access disabled and produced a finite seven-axis CUDA action from a public
observation. Its state-projection tensor values changed from the starting model.
All 40 evaluation episodes produced native videos. Rendering used Mesa software
EGL while training and policy inference ran on CUDA. The Slurm launcher was
syntax/contract tested; this live qualification used Kubernetes.

## Read the visual evidence

- The stage strip describes the executed data → train → adapt → evaluate path.
- The three videos compare actual policies on the same validation starting
  state. The episode selector exposes all recorded validation rollouts.
- Learning curves show measured optimizer loss, averaged in windows for
  readability. Loss improvement alone does not establish better robot success.
- The final-test panel shows the measured result and configured quality gate.
  A failed gate stops the runner before checkpoint export and returns an error.
  Failed evaluation evidence remains available for diagnosis. Completion is
  recorded only after both evaluation thresholds and offline export verification
  pass. This final export gate does not implement the two training retry loops
  in the separate Slurm workflow.
  Evaluation requests bind the checkpoint files, task, seed and initial-state
  partition before execution. Cached evaluations with missing or mismatched
  identity cannot approve an export; use a new evaluation directory to rerun.
  `selection.json` records `quality_gate_passed`, the final test score, and the
  selected checkpoint hash for automation. This qualifies the simulator recipe.
- Checkpoint hashes identify the trained bytes, and the runtime line shows the
  actual GPU, training-rank count and rendering backend.

The report is offline and has no network connections. Its exporter uses an
allowlist of measured fields and relative media paths. Raw runtime logs,
credentials, cluster identifiers and customer data do not enter the report or PR.
Model/backbone weights and public demonstrations declare Apache-2.0 licenses;
LIBERO simulator assets retain their upstream component terms and are fetched
privately at runtime, not included in the report or repository.

For the larger foundation-training architecture and generic multi-node Slurm
contracts, see [foundation training](foundation-training.md). That architecture
preview remains explicitly labeled and is separate from this executed reference.

Verify collected live artifacts with:

```bash
NPA_INTEGRATION_E2E=1 NPA_PUBLIC_VLA_RESULTS=./outputs/public-vla-results \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_public_vla_live.py -q
```
