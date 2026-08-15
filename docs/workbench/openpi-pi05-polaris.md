# OpenPI pi0.5 Polaris serving on B200

`byof-openpi.yaml` packages the immutable upstream
`Physical-Intelligence/openpi@15a9616a00943ada6c20a0f158e3adb39df2ccac`
source and validates the upstream policy, WebSocket server, and WebSocket client
on one datacenter Blackwell B200 (`sm_100`). The image is based on CUDA 12.8,
compiles an `sm_100` runtime probe, and uses upstream's pinned JAX CUDA 12 stack.
A CUDA 13.0 managed-driver MK8s node remains backward compatible with that
userspace stack; the live smoke records the actual driver, XLA platform, JAX,
JAXlib, CUDA plugin, GPU product, and compute capability.

## Policy and checkpoint

- Config: `pi05_droid_jointpos_polaris`
- Checkpoint: `gs://openpi-assets/checkpoints/polaris/pi05_droid_jointpos_polaris`
- Embodiment boundary: Franka Panda, seven arm joints plus one parallel-jaw
  gripper dimension
- Output: a 15-step action chunk with eight dimensions. Dimensions 0–6 are
  **joint-position targets in radians** and dimension 7 is the gripper target.
  They are not joint velocities.

The checkpoint is fetched only after the workload starts, into an ephemeral
operator/project-controlled cache. It is never copied into the distributable
image or uploaded with the smoke artifacts. Evidence includes a SHA-256 over a
canonical manifest of GCS object names, generations, sizes, MD5 values, and
CRC32C values, so the runtime bytes are reproducibly identified without
committing the weights. The public GCS source is opened explicitly in anonymous
mode; the workload does not probe or require Google application credentials.

OpenPI source is Apache-2.0. The CUDA base/runtime retains NVIDIA's upstream
license terms. This BYOF result is stored in the operator's private project
registry and is not classified for public redistribution. Checkpoint weights,
input frames, and robot state are not baked into it. The checkpoint contains
Gemma-derived material: before any image build or checkpoint fetch, the operator
must review the
[Gemma Terms of Use](https://ai.google.dev/gemma/terms) and
[Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy),
then provide the exact, run-scoped gate:

```bash
export NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES
```

NPA forwards that value through SkyPilot's secret channel. It is not a workflow
default, image environment variable, build argument, or persisted credential.
No other agreement is inferred from it.

## Request and response contract

The served client request is:

| Field | Type and shape |
| --- | --- |
| `observation/exterior_image_1_left` | `uint8[224,224,3]` |
| `observation/wrist_image_left` | `uint8[224,224,3]` |
| `observation/joint_position` | `float[7]` |
| `observation/gripper_position` | `float[1]` |
| `prompt` | string, for example `pick up the fork` |

The live transport smoke uses deterministic synthetic frames and a valid
neutral Franka state. This proves model execution and transport, not physical
task success. Both direct `policy.infer` and the upstream WebSocket client/server
round trip inside the scheduled Kubernetes workload must return a finite
`(T, 8)` array with `T >= 5`. The artifact records that the WebSocket client
connects over pod loopback; it does not misrepresent that smoke as an externally
exposed Kubernetes Service. A robot consumer should execute about five position
targets at 15 Hz, observe again, and re-query the policy.

## Validate and run

First validate and render the canonical workflow without mutation:

```bash
npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-openpi.yaml
npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-openpi.yaml \
  --run-id openpi-polaris-plan
```

On an already healthy B200 MK8s context configured for the `neb-phys-ai`
project alias, run the dedicated live E2E. It reads the checked-in workflow and
resource profile unchanged, consumes an immutable digest previously built and
pushed through BYOF, launches through pinned SkyPilot, and verifies the
published result:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_BYOF_OPENPI_LIVE_B200=1 \
NPA_E2E_PROJECT=neb-phys-ai \
NPA_E2E_S3_BUCKET=<existing-project-bucket> \
NPA_BYOF_S3_ENDPOINT=https://storage.<bucket-region>.nebius.cloud \
NPA_BYOF_OPENPI_PROJECT_REGISTRY=cr.us-central1.nebius.cloud/<project-registry> \
NPA_BYOF_OPENPI_REUSE_IMAGE=cr.us-central1.nebius.cloud/<project-registry>/npa-byof-openpi@sha256:<digest> \
NPA_OPENPI_ACCEPT_GEMMA_TERMS=YES \
npa/.venv/bin/python -m pytest -q -s \
  npa/tests/e2e/test_byof_openpi_polaris_live_e2e.py
```

The current milestone is inference and serving. The pinned upstream config
still exposes its RLDS training configuration and evaluation-compatible policy
surface, but NPA does not claim live training or evaluation without a compatible
real dataset and an executed optimization/evaluation step.

## Accepted live baseline

The accepted `byof-openpi-polaris-e2e-20260815T021414Z` run used one B200
(`sm_100`) with JAX/JAXlib/CUDA plugin 0.5.3 and the CUDA 12 userspace stack on a
CUDA 13 managed-driver MK8s node. It fetched 27 checkpoint objects totaling
12,434,530,837 bytes in 24.847 seconds. Direct first-call inference, including
JAX compilation, took 34,976.96 ms; the subsequent upstream WebSocket client
round trip took 55.52 ms (54.006 ms server inference). Both responses were
finite `float64[15,8]` joint-position trajectories. The deterministic frames
remain an inference/transport smoke, not evidence of physical task success.
