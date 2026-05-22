# Workflow Conventions

## Definition Types

Reference architectures live as Argo `WorkflowTemplate` resources. A submitted `Workflow` is an instance created from one of those templates with `argo submit --from workflowtemplate/<name>`.

Reusable steps also live as `WorkflowTemplate` resources. Reference architecture templates call those steps with `templateRef` so common behavior can be installed once and reused.

## Parameters

Standard workflow parameters:

- `dataset-uri`: S3 URI of the input dataset. The URI may point to any S3 object the Argo artifact credentials can read.
- `output-prefix`: S3 prefix for run outputs. Placeholder workflows default to `argo-artifacts/{{workflow.name}}/`, matching the installed artifact repository key format.
- `gpu-type`: Nebius platform preset name for future GPU steps. Valid values are `gpu-h100-sxm`, `gpu-h200-sxm`, and `gpu-rtx6000`. CPU-only workflows ignore this parameter.

Per-step parameters use `step-<name>-<param>`, for example `step-curate-min-quality`. User-facing aliases are allowed when they are clearer for a reference architecture, but the step-level contract should still map cleanly to this form.

## Artifact Contract

Each step takes one Argo input artifact named `input` and produces one Argo output artifact named `output`.

Argo's controller and wait container handle S3 download and upload through the configured artifact repository. Step containers read and write local paths only; they do not make raw S3 calls.

The installed artifact repository stores workflow artifacts under:

```text
s3://${NPA_S3_BUCKET}/argo-artifacts/<workflow-name>/<pod-name>/
```

The first step of a reference architecture may source its `input` from an arbitrary S3 URI by using Argo's `s3:` artifact source syntax. Later steps pass artifacts by referencing previous step outputs.

## Image Rules

Public images are pinned by sha256 digest and do not need an image pull secret. Placeholder steps use:

```text
python:3.11-slim@sha256:9a7765b36773a37061455b332f18e265e7f58f6fea9c419a550d2a8b0e9db834
```

Nebius registry images use:

```text
cr.eu-north1.nebius.cloud/<your-registry-id>/<tool>:<version>
```

Those private images rely on the `cr-credentials` pull secret installed on the `argo-workflow` service account. Real tool steps land in `W8-tool-pod-runtime`; this bootstrap uses placeholders only.

## Resources

CPU-only steps do not request `nvidia.com/gpu` and do not set GPU node selectors.

Future GPU steps request:

```yaml
resources:
  limits:
    nvidia.com/gpu: 1
```

GPU steps also set the node selector discovered by `W8-h100-node-group-retry`.

## Naming

WorkflowTemplate names are kebab-case and descriptive, for example `curate-augment-train`, `sim-to-real`, and `eval-as-a-service`.

Semantic step names use `step-<verb>`, for example `step-curate`, `step-augment`, and `step-train`. When a step output is referenced by a later step, use a hyphen-free Argo step ID such as `curate`, `augment`, or `train`, and pass the semantic `step-<verb>` value to the step as `step-name`.

## GPU Exclusions

L40S-family GPUs are excluded across all workflows because memory #20 recorded a no-log ERROR pattern. Route GPU workflows to H100 or H200 unless a later validation changes that decision.

B300 is excluded for Genesis steps specifically because memory #23 recorded the Taichi `sm_103` upstream block. B300 is not globally excluded for non-Genesis tools once those tools validate.

When in doubt, default to H100.

## Safety

Every Workflow sets `spec.activeDeadlineSeconds` as runaway protection. Placeholder workflows use `3600` seconds. Real workflows tune this guard based on expected tool runtime; it is not a scheduling budget.
