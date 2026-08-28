# Isaac Lab 3 workbench

NPA pins the newest released point in the Isaac Lab 3 beta line:
`v3.0.0-beta2.patch1` (`isaaclab==3.0.0b2.post1`) at commit
`ffff603eafc6b74264a5261cc0183d6a65390d78`, paired with Isaac Sim
`6.0.1.0`. Upstream labels this a beta, not a 3.0 GA release. The pin was
selected from the [upstream releases](https://github.com/isaac-sim/IsaacLab/releases)
and checked against the [3.0 beta installation guidance](https://isaac-sim.github.io/IsaacLab/release/3.0.0-beta2/source/setup/installation/index.html).

## Packaging and operation

The public `npa-isaac-lab:3.0.0b2.post1` image contains the Ubuntu 24.04,
Python 3.12, CUDA 12.8, PyTorch 2.11, NPA, and OSS training dependency layers.
Its accepted release digest is
`sha256:bb735577809f9b427493fda78efebc543dcf02e3deac2ec8a36ac019bff8ee46`.
It contains no Isaac Sim, Isaac Lab, Omniverse Client, or other proprietary
NVIDIA runtime payload. On first invocation, `/isaac-sim/python.sh` verifies
every runtime wheel against `isaac3-nvidia-wheels.txt`, then installs it into
the operator's cache under the operator's EULA acceptance. Explicit opt-out
fails before download with exit 78.

Use an RT-core GPU for this PhysX/renderer workbench: L40S or RTX PRO 6000.
B200 is a datacenter compute GPU and is not a substitute for this graphics
path. Managed deployments default to the reproducible container:

```bash
npa workbench isaac-lab deploy \
  --runtime container \
  --gpu-type <discovered-rtx-platform> \
  --gpu-preset <matching-preset>
```

Native `--runtime vm` installation is intentionally unsupported for generation
3 because it cannot preserve the image's snapshot- and hash-pinned packaging
contract. `--runtime byovm` still uses the same container on an existing host.

The reference hardened pipeline is
`npa/workflows/workbench/npa-workflows/isaac-lab-rl-sweep.yaml`. Its four
parallel stages invoke the pinned upstream RSL-RL trainer, require successful
numeric reward and checkpoint evidence, and join at a fail-closed ranking
barrier:

```bash
npa workbench workflow submit \
  npa/workflows/workbench/npa-workflows/isaac-lab-rl-sweep.yaml \
  --run-id <unique-run-id> \
  --runtime \
  --var bucket=<configured-bucket> \
  --image ghcr.io/nebius/nebius-physical-ai/npa-isaac-lab:3.0.0b2.post1 \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY
```

For a reviewable trained-policy rollout, use VM training with trajectory
export. RGB capture is enabled by default for that post-training rollout:

```bash
npa workbench isaac-lab train \
  --task Isaac-Cartpole-v0 \
  --output-dir <output-directory> \
  --export-trajectories
```

The exporter loads the produced RSL-RL checkpoint fail-closed, enables Isaac
Sim cameras only for the rollout, and stores `rgb.npy` beside each episode's
state and action arrays. Frames come from the environment's real
`rgb_array` renderer and share the episode/frame/timestamp timeline with state
and actions. The LeRobot converter recognizes this image-bearing contract,
encodes one video per episode, records checkpoint/runtime/render provenance,
and the Rerun adapter opens the trained-policy environment view as the primary
pane. Metadata-only datasets remain supported and are labeled without a visual
claim. `--no-export-rgb` is an explicit opt-out for scalar-only exports.

## Generation 2 comparison

The reproducible comparison uses the public 2.3.2 baseline and the 3.0 beta
candidate on the same single RTX PRO 6000 role, driver, container runtime,
task, environment count, iteration count, and paired seeds. Each runtime is
bootstrapped before timing, so the measured interval is simulator startup plus
real RSL-RL training and excludes network/cache population. The campaign uses
`Isaac-Cartpole-v0`, 1,024 environments, 50 iterations, and three paired
repetitions. `compare_isaac_lab_benchmarks.py` refuses failed, unequal,
mutable-image, hardware-mismatched, or unpaired records and reports medians.

| Measurement | Isaac Lab 2.3.2 | Isaac Lab 3.0 beta 2 patch 1 | Candidate change |
| --- | ---: | ---: | ---: |
| Median startup + training wall time | 32.654 s | 29.309 s | **10.242% lower (1.114x speedup)** |
| Median reported mean reward | 4.86 | 4.89 | +0.03 (descriptive only) |
| Cold runtime bootstrap | 193.280 s | 340.016 s | 75.919% higher |
| Runtime cache after bootstrap | 10.045 GiB | 17.615 GiB | 75.362% higher |
| Compressed public image | 9.943 GiB | 10.236 GiB | 2.949% higher |

These are paired measurements, not estimates. The candidate reduced the
startup-inclusive training interval for this workload, but it did **not**
improve first-use provisioning: its cold bootstrap and runtime cache were
materially larger. The image itself grew modestly because the proprietary
runtime remains outside both public images.

Wall time is the primary measurement. Mean reward is reported separately and
must not be interpreted as a cross-version policy-quality claim: framework and
training defaults can change across major generations. Results apply only to
the recorded workload and hardware; startup-inclusive wall time is not a
simulator-steps-per-second measurement.
