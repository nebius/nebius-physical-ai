# SONIC Whole-Body Control

SONIC / GEAR-SONIC is NVIDIA GEAR's humanoid whole-body-control stack. It is a
standalone Workbench tool for low-level motor control, motion tracking,
teleoperation, sim2sim validation, and deployment of full-body humanoid
controllers.

SONIC is a peer component to GR00T. GR00T can emit high-level VLA actions, and
SONIC can decode motion/control targets into full-body joint behavior, but the
first Workbench integration keeps them separate. GR00T+SONIC orchestration is a
future composition workflow.

## Architecture

`npa workbench sonic` is organized around three runtime modes:

- `vm`: long-running SONIC deployment host for C++/TensorRT inference.
- `container`: local or VM-hosted container path for sim, ZMQ, keyboard/gamepad,
  and smoke validation.
- `serverless`: short Isaac Lab training or smoke jobs using Nebius Serverless
  Jobs.

The SONIC container is self-contained. It installs SONIC from
`NVlabs/GR00T-WholeBodyControl` and bundles Isaac Lab as a library dependency
inside the SONIC image. It does not depend on the Workbench Isaac Lab tool image.
The image targets `linux/amd64` for Nebius L40S and normalizes the Isaac Lab
Python package to `isaaclab==2.3.2.post1` during build.

The default image build is focused on training and smoke validation. It includes
the SONIC C++ deploy source and build tools, but leaves `gear_sonic_deploy`
compilation opt-in with `BUILD_SONIC_DEPLOY=1` because TensorRT and ONNX Runtime
discovery are platform-sensitive.

Build and publish the required first-party image from the repo root:

```bash
export NPA_REGISTRY=cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}
npa/docker/workbench/sonic/build.sh --registry "${NPA_REGISTRY}" --push
```

SONIC publishes two first-party image variants. The compatibility source of
truth is `npa/src/npa/deploy/sonic_image_manifest.json`, with the human catalog
in `docs/workbench/sonic-image-catalog.md`. The default L40S VM image is
`${NPA_REGISTRY}/npa-sonic:0.1.2`; the Kubernetes GPU-operator image for
RTX PRO 6000 Blackwell is `${NPA_REGISTRY}/npa-sonic:0.1.2-k8s-runtime`.

Verify the pushed image before launch with:

```bash
docker manifest inspect "${NPA_REGISTRY}/npa-sonic:0.1.2"
docker manifest inspect "${NPA_REGISTRY}/npa-sonic:0.1.2-k8s-runtime"
```

The default embodiment is Unitree G1:

```bash
npa workbench sonic train --runtime serverless --embodiment unitree-g1
```

Internally this maps to the SONIC embodiment tag `UNITREE_G1_SONIC`.

## Quick Start

Plan a container runtime:

```bash
npa workbench sonic -p uk-south1 -n sonic-g1 deploy \
  --runtime container \
  --mode sim \
  --checkpoint-source hf \
  --model-repo nvidia/GEAR-SONIC \
  --dry-run
```

Start the sim/keyboard serving path in smoke mode:

```bash
npa workbench sonic -p uk-south1 -n sonic-g1 serve \
  --runtime container \
  --mode sim \
  --input-type keyboard \
  --headless \
  --smoke
```

Use ZMQ input when an external planner or policy server provides pose/action
messages:

```bash
npa workbench sonic serve \
  --runtime container \
  --mode sim \
  --input-type zmq \
  --zmq-host 127.0.0.1 \
  --zmq-port 5556 \
  --zmq-topic pose
```

Real robot mode is guarded:

```bash
npa workbench sonic serve --mode real --confirm-real
```

Do not use real mode without the robot network, safety procedures, and operator
supervision in place.

## Model Artifacts

The Hugging Face distribution path is `nvidia/GEAR-SONIC`. The deploy smoke
expects these artifacts:

- `model_encoder.onnx`
- `model_decoder.onnx`
- `observation_config.yaml`
- `planner_sonic.onnx`

Training uses `sonic_release/last.pt` by default.

## ONNX Export

`npa workbench sonic export` converts a trained locomotion policy checkpoint to
a deterministic-action ONNX graph:

```bash
npa workbench sonic export \
  --checkpoint sonic_release/last.pt \
  --output exported/sonic_policy.onnx
```

The command exports the mean action path. Defaults are `--opset 17`,
`--axes dynamic`, `--normalize baked`, and `--metadata sidecar`. Use
`--normalize sidecar` when the consumer will apply observation statistics, or
`--normalize none` when the input tensor is already in policy space. Use
`--metadata embedded` to write the same metadata into ONNX `metadata_props`
instead of a sidecar JSON file.

Provide `--config`, `--obs-spec`, and `--action-spec` when the checkpoint does
not carry enough layout information. The metadata records observation/action
ordering, shapes, units when supplied, normalization stats when not baked,
opset, axis mode, and control dt when available.

The matching SkyPilot template is
`npa/workflows/workbench/skypilot/sonic-export.yaml` (`name: sonic-export`) and
uses the same settings through `SONIC_OPSET`, `SONIC_AXES`,
`SONIC_NORMALIZE`, `SONIC_METADATA`, `SONIC_OBS_SPEC`, `SONIC_ACTION_SPEC`, and
`SONIC_CONFIG`.

## Export Then Eval

`npa/workflows/workbench/skypilot/sonic-export-eval.yaml` chains export and
eval in one SkyPilot task. It accepts `POLICY_CKPT`, `OUTPUT_DIR`,
`EVAL_BACKEND`, `EPISODES`, `CONTAINER_IMAGE`, and `GPU` through `envs`.

The default `reference` backend uses `EVAL_ENV=sonic-locomotion-smoke`, which
runs deterministic locomotion rollouts against the exported ONNX policy and
writes `sonic_eval_results.json`.

### BYO External Eval Container

External eval is separate from the required first-party `npa-sonic` runtime.
Set `EVAL_BACKEND=container` plus `CONTAINER_IMAGE` only when you provide a
BYO evaluator image. Workbench stages the ONNX policy and sidecar metadata into
that image through `CONTAINER_POLICY_PATH`, `CONTAINER_METADATA_PATH`, and
`CONTAINER_OUTPUT_PATH`; the external container must write
`sonic_eval_results.json`. No external eval image is shipped as an
`npa-*` image.

## Relationship To GR00T

NVIDIA's workflow describes GR00T PolicyServer output feeding SONIC decoder and
deployment code over ZMQ. Workbench v1 exposes the pieces separately:

- `npa workbench groot` serves or evaluates GR00T policies.
- `npa workbench sonic` trains, serves, and smoke-validates the whole-body
  controller.

Composition is intentionally deferred so each tool keeps a clear operational
boundary.

## Licensing

The SONIC code path is Apache 2.0. Released model weights and checkpoints are
under the NVIDIA Open Model License. Operators should verify downstream customer
usage against both licenses before distributing derived artifacts.
