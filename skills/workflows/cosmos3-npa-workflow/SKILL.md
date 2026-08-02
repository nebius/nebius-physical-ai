---
name: cosmos3-npa-workflow
description: Use when writing, validating, or submitting an npa.workflow/v0.0.1 YAML that runs Cosmos 3 generation (text2image / image2image / text2video / image2video / video2video) in the npa-cosmos3 container. This is the declarative npa.workflow spec, NOT the SkyPilot YAML.
---

# Cosmos 3 npa.workflow YAML

## When To Use

Use this skill when the user wants a **Cosmos 3 workflow** they can submit with
`npa workbench workflow submit` — authoring the spec, validating it, or running
it. For the raw SkyPilot template, or for guardrail/prompt semantics, use
`skills/workflows/cosmos3-inference/SKILL.md` instead.

## npa.workflow YAML Is Not SkyPilot YAML

Two different files run Cosmos 3 generation. Do not mix them up, and do not
answer a request for one with the other.

| | npa.workflow spec (this skill) | SkyPilot template |
| --- | --- | --- |
| Path | `npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml` | `npa/src/npa/workflows/skypilot/cosmos3-generate.yaml` |
| First line | `apiVersion: npa.workflow/v0.0.1` | `name: cosmos3-generate` |
| Shape | `config` / `resources` / `states` with a `toolRef` | `resources` / `envs` / `run` shell script |
| Submitted with | `npa workbench workflow submit` | `sky launch` |
| Image | resolved automatically from the toolRef | `image_id` must be rendered by the submitter |

The npa.workflow spec is the preferred surface: it resolves the container image,
carries run-scoped S3 output URIs, and records declared outputs.

## Reference Spec

`npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml` is the working
example. Copy its shape:

```yaml
apiVersion: npa.workflow/v0.0.1
kind: Workflow

metadata:
  name: cosmos3-generate
  skypilotTwin: npa/src/npa/workflows/skypilot/cosmos3-generate.yaml

config:
  bucket: example-bucket
  prefix: "runs/{{run.id}}/cosmos3-generate"
  cosmos3_mode: text2image
  cosmos3_checkpoint: Cosmos3-Nano
  prompt: "a robot arm sorting colored blocks on a white workbench"
  output_uri: "s3://{{config.bucket}}/{{config.prefix}}/generated/"

resources:
  gpu:
    cloud: kubernetes
    accelerators: H100:1
    cpus: 16
    memory: 80Gi

initial: generate

states:
  generate:
    toolRef: workbench.cosmos3.generate
    resources: gpu
    outputs:
      - uri: "{{config.output_uri}}generate.json"
        schema: npa.cosmos3.generate.v1
    terminal: true
```

## The toolRef Contract

`workbench.cosmos3.generate` renders to
`npa workbench cosmos3 generate` and reads exactly these config keys:

| Config key | CLI flag | Notes |
| --- | --- | --- |
| `config.cosmos3_mode` | `--mode` | `text2image`, `image2image`, `text2video`, `image2video`, `video2video` |
| `config.prompt` | `--prompt` | required |
| `config.output_uri` | `--output-path` | `s3://` prefix; the artifact and `generate.json` are published there |
| `config.cosmos3_checkpoint` | `--checkpoint` | `Cosmos3-Nano` by default |

Every key in the argv template must exist in `config` or the run fails at
submit. `image2video` and `video2video` additionally need a conditioning asset;
add `--input-path` to the toolRef argv before advertising those modes.

The toolRef resolves to the **`npa-cosmos3`** image (the framework container).
Do not point it at `npa-cosmos3-reason`, which is the Cosmos-Reason VLM image
and has no cosmos-framework in it.

## Author And Validate

```bash
npa workbench workflow validate-spec npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml
npa workbench workflow plan-spec     npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml
```

Then render without submitting to confirm the image and secrets:

```bash
npa workbench workflow submit <spec> --plan-only \
  --infra k8s/<context> --registry <your-registry> --var bucket=<bucket>
```

Check the rendered output for:

- `image_id: docker:<registry>/npa-cosmos3:<tag>` — if it is missing, see the
  `NPA_E2E_CLEAR_WORKBENCH_IMAGES` note below.
- `secret_env_hints: HF_TOKEN`.

## Submit

```bash
npa workbench workflow submit npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml \
  --infra k8s/<context> --registry <your-registry> \
  --var bucket=<bucket> --runtime \
  --secret-env HF_TOKEN --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY
```

## Four Things That Actually Break Submits

These are the failures observed on real runs; check them before debugging
anything else.

1. **Missing `--secret-env HF_TOKEN`.** The plan only *hints* the secret. The
   image bakes no weights, so the stage fails fast with "Cosmos 3 weights are
   not baked into this image". Always pass it explicitly.
2. **`NPA_E2E_CLEAR_WORKBENCH_IMAGES=1` in the environment.** It strips the
   workbench image pin and the stage then runs on a generic SkyPilot image with
   no cosmos-framework. Unset it (or set `0`) for Cosmos 3 runs.
3. **Registry mismatch.** Submit refuses when the task image's registry differs
   from `SKYPILOT_DOCKER_SERVER`. Point the Docker credentials at the same
   registry the image was pushed to.
4. **Accelerator not on the cluster.** The spec's portable default is `H100:1`;
   on a Blackwell cluster substitute
   `RTXPRO-6000-BLACKWELL-SERVER-EDITION:1` before submitting.

## Guardrails And Weights

Guardrails are on unless `--no-guardrails` is passed; the result manifest
records `guardrails` so the posture is auditable. The gated checkpoint, the Wan
VAE, and the guardrail models download at run time under the operator's own
Hugging Face licence acceptance — never bake weights into the image.

Because guardrails pull the gated `nvidia/Cosmos-Guardrail1`, `HF_TOKEN` is
required even when `--checkpoint` points at weights you already staged. The
token check is only skipped when the checkpoint is staged **and**
`--no-guardrails` is set.

## Outputs

The stage publishes `vision.jpg` (or the video artifact) plus `generate.json`
(`npa.cosmos3.generate.v1`) under `config.output_uri`. Useful manifest fields:
`status`, `output_kind`, `output_bytes`, `guardrails`, `weights_baked`,
`hf_auth`, `artifact_uri`.

To view the result, load `artifact_uri` in the agent viewer
(`POST /api/sim-viz/load-artifact` with `{"s3_uri": ...}`); see
`skills/atomic/find-artifacts/SKILL.md`.

## Verified

The reference spec ran green on `npa-rtxpro-mk8s` (RTX PRO 6000 Blackwell,
sm_120) with `Cosmos3-Nano`, producing a non-blank 960x960 JPEG in S3 in about
4 minutes end to end. See `docs/workbench/cosmos3-generate.md`.
