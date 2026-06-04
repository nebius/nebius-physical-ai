# Cosmos 3 Augmentation And Reasoning

This cookbook covers the general-purpose Cosmos surface in Workbench:
controlled-generation augmentation for sim-to-real datasets and structured
reasoning evaluation over frames or video. Both workflows are exposed through
the CLI, SDK, and standalone raw SkyPilot YAML.

Attribution: Built on NVIDIA Cosmos.

## Public Coordinates

The GA Cosmos 3 source of truth is the NVIDIA Cosmos repository and Cosmos
Framework:

- `https://github.com/NVIDIA/cosmos`
- `https://github.com/NVIDIA/cosmos-framework`

The Cosmos 3 model IDs used by Workbench reasoning are:

- `nvidia/Cosmos3-Nano`, checkpoint `Cosmos3-Nano`
- `nvidia/Cosmos3-Super`, checkpoint `Cosmos3-Super`

The public model cards currently describe Nano and Super as 16B and 64B
trainable-parameter models. Workbench accepts `nano`, `super`, and numeric
aliases for model-size selectors, but the Hugging Face coordinates above are
the canonical values.

Cosmos 3 public examples expose generator and reasoner surfaces through
`cosmos_framework.scripts.inference`. The reasoner surface supports image and
video reasoning tasks including physical-plausibility and embodied reasoning.

Cosmos 3 code contains transfer-oriented internals, but the GA public examples
do not yet provide a reliable controlled-generation augmentation recipe with
the same completeness as Transfer 2.5. Workbench therefore uses
Cosmos Transfer 2.5 for `npa workbench cosmos augment` behind the same generic
surface:

- Source: `https://github.com/nvidia-cosmos/cosmos-transfer2.5`
- Model: `nvidia/Cosmos-Transfer2.5-2B`
- Controls: `edge`, visual blur (`vis` / `blur`), `depth`, and segmentation
  (`seg`)
- Robotics-relevant checkpoint family:
  `nvidia/Cosmos-Transfer2.5-2B/robot-multiview-control`

## License And Guardrails

Cosmos workflows require a Hugging Face token whose account has accepted the
relevant model terms. Pass it as `HF_TOKEN` with SkyPilot `--secret HF_TOKEN`.

Guardrails are on by default and there is no customer-facing CLI, SDK, YAML, or
environment switch to disable them. Output metadata records:

```json
{
  "guardrails": "on",
  "attribution": "Built on NVIDIA Cosmos"
}
```

The vendored license and notices live under `third_party/nvidia-cosmos/`.

## CLI: Augmentation

```bash
export NPA_SKYPILOT_BIN=/path/to/sky
export PATH="$(dirname "$NPA_SKYPILOT_BIN"):$PATH"

npa workbench cosmos augment \
  --source s3://example-bucket/cosmos/input/sim-render.mp4 \
  --output-path s3://example-bucket/cosmos/output/augment/ \
  --prompt "A photorealistic robotics workspace preserving motion and layout." \
  --control edge \
  --model-size transfer2.5-2b \
  --variants 1 \
  --replicas 1 \
  --image registry.example/npa-cosmos:3.0.0 \
  --s3-endpoint https://storage.example.invalid \
  --infra kubernetes \
  --accelerator "$NPA_COSMOS_GPU"
```

`--replicas` is the cost and parallelism lever. It maps to `NPA_COSMOS_REPLICAS`
and SkyPilot `--num-nodes`. Use `--control-config` with JSON when a control
modality needs an explicit control input, for example:

```bash
--control depth --control-config '{"control_path":"s3://example-bucket/control/depth.mp4"}'
```

## CLI: Reasoning

```bash
npa workbench cosmos reason \
  --input-path s3://example-bucket/cosmos/input/rollout.mp4 \
  --output-path s3://example-bucket/cosmos/output/reason/ \
  --criteria-prompt "Decide whether the robot completes the task safely." \
  --model-size nano \
  --replicas 1 \
  --image registry.example/npa-cosmos:3.0.0 \
  --s3-endpoint https://storage.example.invalid \
  --infra kubernetes \
  --accelerator "$NPA_COSMOS_GPU"
```

The reasoning workflow asks Cosmos 3 to return structured JSON with
`success`, per-dimension `scores`, and a natural-language `critique`. If the
model returns non-JSON text, Workbench still writes a structured wrapper with
the raw critique.

## SDK

The SDK mirrors the CLI parameters:

```python
from npa.sdk.workbench import cosmos

cosmos.augment(
    source="s3://example-bucket/cosmos/input/sim-render.mp4",
    output_path="s3://example-bucket/cosmos/output/augment/",
    prompt="A photorealistic robotics workspace preserving motion and layout.",
    control="edge",
    variants=1,
    replicas=1,
    image="registry.example/npa-cosmos:3.0.0",
    s3_endpoint="https://storage.example.invalid",
    infra="kubernetes",
    accelerator="GPU_TYPE:1",
)

cosmos.reason(
    input_path="s3://example-bucket/cosmos/input/rollout.mp4",
    output_path="s3://example-bucket/cosmos/output/reason/",
    criteria_prompt="Decide whether the robot completes the task safely.",
    model_size="nano",
    image="registry.example/npa-cosmos:3.0.0",
    s3_endpoint="https://storage.example.invalid",
    infra="kubernetes",
    accelerator="GPU_TYPE:1",
)
```

For programmatic launch planning without invoking Typer, use
`build_cosmos_augment_env`, `build_cosmos_reason_env`, and
`launch_cosmos_sky_workflow` from `npa.workbench.cosmos`.

## Raw SkyPilot YAML

The raw YAML files are standalone. They do not call the NPA CLI or SDK:

- `npa/workflows/workbench/skypilot/cosmos3-augment.yaml`
- `npa/workflows/workbench/skypilot/cosmos3-reason.yaml`

SkyPilot 0.12.2 does not expand shell variables in `envs:` at submit time, so
pass runtime values with `--env` and consume them in `run:`.

```bash
sky launch --infra kubernetes --gpus "$NPA_COSMOS_GPU" \
  --image-id docker:registry.example/npa-cosmos:3.0.0 \
  --env NPA_COSMOS_AUGMENT_SOURCE=s3://example-bucket/cosmos/input/sim-render.mp4 \
  --env NPA_COSMOS_AUGMENT_OUTPUT=s3://example-bucket/cosmos/output/augment/ \
  --env AWS_ENDPOINT_URL=https://storage.example.invalid \
  --secret HF_TOKEN \
  npa/workflows/workbench/skypilot/cosmos3-augment.yaml

sky launch --infra kubernetes --gpus "$NPA_COSMOS_GPU" \
  --image-id docker:registry.example/npa-cosmos:3.0.0 \
  --env NPA_COSMOS_REASON_INPUT=s3://example-bucket/cosmos/input/rollout.mp4 \
  --env NPA_COSMOS_REASON_OUTPUT=s3://example-bucket/cosmos/output/reason/ \
  --env 'NPA_COSMOS_REASON_CRITERIA=Decide whether the robot completes the task safely.' \
  --env AWS_ENDPOINT_URL=https://storage.example.invalid \
  --secret HF_TOKEN \
  npa/workflows/workbench/skypilot/cosmos3-reason.yaml
```

When using S3-compatible storage, set both the S3 URI and the endpoint. Local
paths and HTTP(S) inputs are also supported for source assets.
