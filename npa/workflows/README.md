# NPA Workflow Templates

This directory contains Argo WorkflowTemplate definitions for Workbench reference architectures. The current templates use the pod-as-worker execution model: each workflow step is a Kubernetes pod that performs the step work directly.

## Layout

- `templates/`: reference architecture WorkflowTemplates.
- `steps/`: reusable step WorkflowTemplates referenced with `templateRef`.
- `schemas/`: conventions for parameters, artifacts, naming, and runtime constraints.

## Install Templates

Use the target cluster kubeconfig for every command:

```bash
export KUBECONFIG=/tmp/<run-id>/kubeconfig
argo template create -n argo npa/workflows/steps/placeholder-step.yaml
argo template create -n argo npa/workflows/templates/curate-augment-train.yaml
```

For an existing template, use `argo template update -n argo <file>`.

## Submit

Submit a Workflow from a WorkflowTemplate:

```bash
argo submit -n argo \
  --from workflowtemplate/curate-augment-train \
  -p dataset-uri=s3://YOUR_S3_BUCKET/argo-artifacts/fixtures/curate-augment-train-v1/fixture-dataset.txt \
  --watch
```

## Inspect

```bash
argo list -n argo
argo watch -n argo <workflow-name>
argo get -n argo <workflow-name>
argo logs -n argo <workflow-name>
```

The conventions document is the source of truth for parameter names, artifact flow, S3 layout, image rules, and GPU exclusions.
