# Reproduce Unitree G1 training with MJLab and RSL-RL

[Guides](README.md) · [MJLab Workbench interfaces](../mjlab.md)

Run the stock `Mjlab-Velocity-Flat-Unitree-G1` task on a Nebius RTX PRO 6000
VM with 4,096 parallel environments. MJLab's native `train` entrypoint
automatically uses RSL-RL PPO; no separate RSL-RL launcher or implementation
change is needed.

This recipe writes local artifacts on an already allocated VM. For S3 outputs,
authenticated services, or declarative training/evaluation workflows, use the
[MJLab Workbench interfaces](../mjlab.md).

## Runtime and resources

Use a Linux VM with Docker, NVIDIA Container Toolkit, working NVIDIA GPU
access, and sufficient host RAM. Complete the
[GPU runtime setup](../getting-started.md) before allocating new compute.

| Resource | Training container |
| --- | --- |
| GPU | One RTX PRO 6000, device 0 |
| CPU | Quota equivalent to 22 CPUs |
| RAM | 160 GiB |

The VM can be larger than these limits. A 24-vCPU / 218-GiB VM was used for
the original comparisons. `--cpus 22` sets a quota; it does not pin execution
to particular vCPUs. Retain the actual host CPU model and driver version when
comparing runs.

The immutable image below contains:

| Component | Version |
| --- | --- |
| MJLab | 1.6.0 |
| RSL-RL (`rsl-rl-lib`) | 5.4.2 |
| PyTorch | 2.13.0+cu130 |

The original training baseline used NVIDIA driver 610.57.04. A different
driver is a changed comparison condition. This is the existing
[public RTX development candidate](../container-image-catalog.md#mjlab-public-development-candidate);
this recipe does not promote it to a supported release or extend its GPU
qualification.

## Run the reproduction

Run the following in Bash on the GPU VM. The numeric user/group option makes
the container write as the invoking account. The image is pulled before
training starts; training itself has networking disabled and uses the
documented TensorBoard logger.

```bash
#!/usr/bin/env bash
set -euo pipefail

mjlab_image='ghcr.io/nebius/nebius-physical-ai/npa-mjlab@sha256:ec8c099dc8ab34c8a390c6076d3cc3c07a88e27818d0eb0b6b6cf819663f242e'
mjlab_out="$PWD/mjlab-g1-repro-$(date -u +%Y%m%d-%H%M%S)"

mkdir -p "$mjlab_out"

# Retain hardware information for comparison.
lscpu > "$mjlab_out/cpu.txt"
nvidia-smi --id=0 \
  --query-gpu=name,driver_version,memory.total \
  --format=csv > "$mjlab_out/gpu.csv"

sudo docker pull "$mjlab_image"

sudo docker run --rm \
  --gpus device=0 \
  --cpus 22 \
  --memory 160g \
  --ipc host \
  --network none \
  --user "$(id -u):$(id -g)" \
  -e HOME=/tmp \
  -v "$mjlab_out:/out" \
  -w /out \
  --entrypoint train \
  "$mjlab_image" \
  Mjlab-Velocity-Flat-Unitree-G1 \
  --env.scene.num-envs 4096 \
  --agent.max-iterations 1500 \
  --agent.seed 42 \
  --agent.logger tensorboard \
  --agent.run-name g1-repro \
  --log-root /out/logs \
  2>&1 | tee "$mjlab_out/train.log"
```

The only task settings overridden here are environment count, training
iterations, seed, logger, run name, and log directory. PPO hyperparameters,
network architecture, rollout length, and precision remain the task's defaults.
The image's CLI uses `--env.scene.num-envs`; inspect that same image's
`train Mjlab-Velocity-Flat-Unitree-G1 --help` before substituting flags from
another MJLab version.

Each update collects 24 steps from each of 4,096 environments. The 1,500-update
run therefore collects **147,456,000 environment transitions**. This reproduces
the comparison horizon; it is not a claim of policy convergence or completion
of the task's longer default training schedule.

## Check and retain the outputs

The Bash pipeline fails if the training command fails. A completed
1,500-update run should have this structure:

```text
mjlab-g1-repro-<utc-timestamp>/
  cpu.txt
  gpu.csv
  train.log
  logs/g1_velocity/<timestamp>_g1-repro/
    events.out.tfevents...
    model_1499.pt
    params/
      agent.yaml
      env.yaml
```

Keep the whole output directory. Confirm that the native training log and
TensorBoard events reach update 1499, and that the final checkpoint and both
parameter files exist. A log file or an exited container alone does not prove
that all requested updates completed. Evaluate the checkpoint separately
through the [native MJLab evaluation path](../mjlab.md#results-and-measurement)
when assessing policy quality.

## Compare performance

Start with an unprofiled run. Preserve the full TensorBoard series for
`Perf/learning_time` and `Perf/collection_time`, as well as the console log.
Compare 50-update windows after the first 10 updates to expose slow/fast
periods, and retain whole-iteration wall times.

The native learning-time scalar is host-observed elapsed time around return
computation and the PPO update. CUDA work is asynchronous, so this scalar is
not an isolated GPU kernel-duration measurement.

Use the same image digest, task configuration, environment count, seed,
resource limits, and logger on each comparison VM. Keep the hardware and
driver files with each run. A different CPU model, driver, affinity setting,
or observer is a separate comparison condition. Do not infer physical turbo
frequency from a static guest `lscpu` MHz field.

## Finish the run

Copy the output directory before stopping or deleting the VM. `--rm` removes
the training container when it exits; it does not release the GPU allocation.
If interrupted, inspect and stop only this run's container. Follow the
[teardown procedure](../../teardown.md) for the compute resources owned by the
run.

## Upstream reference

MJLab documents the task registry, PPO configuration, native training command,
and checkpoint structure in
[Training with RSL-RL](https://github.com/mujocolab/mjlab/blob/main/docs/source/training/rsl_rl.rst).
Use the pinned runtime in this recipe when reproducing these commands.
