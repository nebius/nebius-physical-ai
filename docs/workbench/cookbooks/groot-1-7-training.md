# GR00T N1.7 operational training pipeline

Use `groot-1-7-finetune.yaml` to validate the complete real-data path from a
GR00T-format LeRobot dataset through distributed optimizer work, an immutable
checkpoint, aligned offline inference, synchronized RRD/MCAP diagnostics, S3
publication, and the deployed NPA agent viewers.

This reference is deliberately a short plumbing validation. It does not claim
statistically meaningful learning, a closed-loop rollout, or physical-robot
performance. Machine-readable output keeps `pipeline_status` separate from
`learning_outcome`; a valid but non-improving checkpoint still reaches artifact
publication and viewer validation, and the operational smoke never promotes a
candidate.

The serial graph is:

```text
access_capacity_preflight -> prepare_deterministic_split ->
baseline_inference_evaluation ->
distributed_training -> trained_checkpoint_resolution ->
post_training_inference_evaluation -> classify_learning_outcome ->
generate_rrd -> generate_mcap -> publish_artifacts_run_summary ->
agent_ui_load_viewer_verification
```

## Validate and plan

Use the repository virtual environment for repository validation:

```bash
SPEC=npa/workflows/workbench/npa-workflows/groot-1-7-finetune.yaml
RUN_ID=groot-n1-7-operational-example

npa/.venv/bin/npa workbench workflow validate-spec "$SPEC"
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID"
```

The default trainer requests two GPUs, one sample per device, no gradient
accumulation, four optimizer steps, logging every step, and one final
`checkpoint-N` resolved from the trainer manifest (`N=4` for the defaults).
`save_steps` must equal `max_steps`, and the CPU preflight rejects an impossible
checkpoint schedule before GPU work. `gpu_count` remains parameterized. Whenever it changes, keep:

```text
global_batch_size = gpu_count * per_device_batch_size * gradient_accumulation_steps
```

Planning tests cover GPU counts 1, 2, 7, 8, and 16. A live validation claim must
name only the GPU count actually run.

## Submit

Supply the bucket, real source dataset, registry image, deployed agent URL, and
runtime secrets at submission time; do not commit tenant or customer values.

```bash
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --var bucket=<bucket> \
  --var source_data_uri=s3://<bucket>/datasets/my-groot-dataset/ \
  --var agent_url=https://<agent-host> \
  --var gpu_count=2 \
  --var per_device_batch_size=1 \
  --var gradient_accumulation_steps=1 \
  --var global_batch_size=2 \
  --registry <registry>/npa-groot:<validated-tag> \
  --secret-env HF_TOKEN \
  --secret-env NPA_AGENT_BASIC_AUTH
```

Before the trainer is scheduled, the CPU preflight checks the GPU/batch/step
contract and the split stage derives real sample coverage. Training evidence
records allocated GPU identities, NCCL collective success, distributed world
size, preflight ranks, per-rank completion of the real vendor trainer, real
optimizer-step losses, effective batch, the single rank-zero publication path,
and the uploaded checkpoint inventory. Resolution downloads the baseline and
trained weight directories, enforces configuration parity, and requires their
weight-only hashes to differ.

Baseline and post-training evaluation use the same deterministic, episode-level,
leakage-free held-out split. Both call the real `Gr00tPolicy.get_action` path.
The report includes trivial predictor floors, repeat-noise evidence,
per-dimension and per-horizon errors, real loss history, checkpoint identity,
and an honest `improved` or `not_improved` classification.
`evaluation_repeats` defaults to five: a duplicated seed proves determinism and
the remaining seeds measure repeat spread. GPU forward cost scales linearly with
this configurable value; one checkpoint/policy instance is reused while every
forward remains explicitly reseeded.

## Outputs and viewers

The default prefix is:

```text
s3://<bucket>/groot-1-7-two-gpu-pipeline/<run-id>/
```

Important outputs include:

```text
reports/access-capacity-preflight.json
reports/split/manifest.json
checkpoints/candidate/checkpoint-<resolved-step>/
reports/trained-checkpoint.json
reports/two-gpu-pipeline-report.json
reports/offline-heldout-comparison.mp4
reports/groot-offline-evaluation.rrd
reports/groot-offline-evaluation.mcap
reports/publish-manifest.json
reports/agent-ui-verification.json
workflow.yaml
```

RRD and MCAP both use the dataset-index timebase. They contain synchronized real
dataset camera frames, expert and baseline/trained predicted actions, loss,
offline metrics, run/stage identity, and checkpoint provenance. Every viewer and
report labels them **OFFLINE EVALUATION / NOT A ROBOT ROLLOUT**.

The terminal stage exercises run discovery, inventory association, Rerun and
Lichtblick loads, and byte-range endpoints through the deployed agent API.
Pixel-level nonblank rendering and “Describe this” remain browser E2E gates.

## Kubernetes image prerequisites

`npa/docker/workbench/groot/Dockerfile.k8s-prereqs` adds the system packages the
SkyPilot Kubernetes bootstrap requires. The canonical image runs as `ubuntu`
and its shared installer supplies system Python, `rsync`, an SSH client, and
passwordless sudo for SkyPilot's in-pod bootstrap. It does not contain
`openssh-server`, its entrypoint is `/bin/bash`, and it does not implement
runtime SSH host-key generation. The derived repair layer adds the SSH server,
generates per-container host keys when SSH starts, removes build-time host keys,
and installs the argument-forwarding entrypoint required by the complete
SkyPilot bootstrap contract. The workflow does not override the pod to uid 0,
so use the repaired image for Kubernetes submission; the canonical image alone
has not passed that complete contract.
