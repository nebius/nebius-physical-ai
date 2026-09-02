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

The SONIC container installs SONIC from `NVlabs/GR00T-WholeBodyControl`, but the
public image does not bake Isaac Lab or Isaac Sim. The Isaac stack is fetched at
runtime after EULA preflight. It does not depend on the Workbench Isaac Lab tool
image.

The default image build is focused on training and smoke validation. It includes
the SONIC C++ deploy source and build tools, but leaves `gear_sonic_deploy`
compilation opt-in with `BUILD_SONIC_DEPLOY=1` because TensorRT and ONNX Runtime
discovery are platform-sensitive.

SONIC publishes one active first-party runtime-fetch image. The compatibility source of
truth is `npa/src/npa/deploy/sonic_image_manifest.json`, with the human catalog
in `docs/workbench/sonic-image-catalog.md`. Use the exact active tag there for
RTX PRO 6000 Blackwell Kubernetes. Legacy L40S and MuJoCo variants are quarantined.

Verify the pushed image before launch with:

```bash
docker manifest inspect \
  "ghcr.io/nebius/nebius-physical-ai/npa-sonic:<active-runtime-fetch-tag>"
```

The default embodiment is Unitree G1. The serverless command also needs an
independently validated compute-only image because the old L40S/H100/H200 images
are quarantined; the active first-party image is supported only on RTX PRO 6000
Kubernetes with GPU Operator driver mounts:

```bash
npa workbench sonic train --runtime serverless --embodiment unitree-g1 \
  --image <validated-compute-only-image>
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

The matching workflow is the `npa.workflow` spec
`npa/workflows/workbench/npa-workflows/sonic-export.yaml` (`metadata.name:
sonic-export`). It passes `--checkpoint {{config.checkpoint_uri}}` and
`--output {{config.onnx_uri}}` to the same CLI, and both accept `s3://` URIs
directly. The exporter's remaining knobs (`--opset`, `--axes`, `--normalize`,
`--metadata`, `--obs-spec`, `--action-spec`, `--config`) use their CLI defaults;
they are the pinned `spec_gap` for this capability (see
`npa/tests/guardrails/test_three_tier_contract.py`), so set them on the CLI/SDK
until the toolRef argv carries them.

## Export Then Eval

`npa/workflows/workbench/npa-workflows/sonic-export-eval.yaml` chains export and
eval as two stages of one spec. Its `config` block carries `checkpoint_uri`,
`onnx_uri`, `eval_uri`, `episodes` and `env`; override any of them at submit time
with `--var key=value`.

The eval stage's default `reference` backend runs deterministic locomotion
rollouts against the exported ONNX policy and writes `eval.json` (the artifact the
spec declares under `outputs:`) to `config.eval_uri`.

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
