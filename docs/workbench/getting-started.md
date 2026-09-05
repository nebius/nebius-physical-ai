# Workbench Getting Started

Complete the [platform quickstart](../quickstart.md) through credential setup,
then use this page to prepare a GPU workflow on Nebius Kubernetes. Run commands
from the repository root with your environment activated. Keep project settings
and credentials in the existing NPA stores; no second credential file is needed.

Direct [Token Factory](token-factory.md) inference uses the hosted API. Its local
CLI/SDK calls do not require the Kubernetes setup below.

## Choose the workload first

| Workload | Compute and setup |
| --- | --- |
| [Cosmos 3 generation](cosmos3-generate.md) | The default text-to-image spec requests one H100, 16 CPU, and 80 GiB host memory; gated guardrail access is required. |
| [Compositional Sim2Real](guides/sim2real-workflow.md) | The canonical 14-stage spec requests RTX PRO 6000 GPUs, CPU capacity, and an Isaac runtime cache. Follow its full runbook. |
| [Isaac Lab BYOF](cookbooks/byof-isaac-lab/README.md) | A custom container and RT-core GPU capacity, such as L40S or RTX PRO 6000, as specified by the cookbook. |

H100/H200 do not provide the RT cores needed for Isaac rendering. Read the
chosen workflow's resource profiles before provisioning: model memory, CPU,
driver, and GPU requirements differ between tools.

## Install Workbench Tools

In addition to the [base install](../install.md), Kubernetes runs need `kubectl`
and the isolated SkyPilot runtime installed below. Managed infrastructure also
requires Terraform. Docker is needed for local image builds; AWS CLI v2 is
optional for direct S3 inspection. Public NPA images need no registry login.

```bash
npa --version
npa configure --show
kubectl version --client
terraform version
```

Gate: the CLI and required host tools work, and the intended project and storage
appear in the configuration. The client-version check does not test cluster access.

## Confirm Platform Credentials

Use the selected workload's access checks before provisioning GPUs. For the
default Cosmos 3 generation workflow:

```bash
npa workbench health preflight --checks hf,s3 --json
npa workbench health access --capability cosmos3 --json
```

Gate: both commands exit successfully and the required checks pass. For another
workload, use its documented capability and credential checks. `--offline` checks
presence only. Resolve missing access on the provider's exact model page, then
rerun the check; a token alone does not establish gated-model access.

The S3 health check lists the configured bucket. Submission also checks that it
can write to the bucket resolved by the workflow. If those checks disagree,
reconcile the bucket and endpoint in NPA configuration and environment overrides.

## Plan the workflow

The following example prepares Cosmos 3 text-to-image generation. Replace the
placeholder values with your existing project alias and bucket. Keep private
values outside committed YAML.

```bash
workflow_spec=npa/workflows/workbench/npa-workflows/cosmos3-generate.yaml
project_alias='<your-project-alias>'
bucket_name='<your-bucket>'
cluster_name='<npa-cluster-name>'

npa workbench workflow validate-spec "$workflow_spec"
npa workbench workflow plan-spec "$workflow_spec" --var "bucket=$bucket_name"
```

Gate: validation succeeds and the plan contains the intended tool, resources,
and output prefix. Pass the same `--var` values to image preflight and submit.
These commands do not launch the model or verify live capacity.

## Verify Kubernetes Access

For a new NPA-managed cluster, use the selected workload's sizing instructions
before running the additive provisioning command. Preview the exact plan:

```bash
npa provision-if-absent --project "$project_alias" \
  --cluster-name "$cluster_name" --dry-run --output-format json
```

When its GPU/CPU topology matches the workload, run the same command without
`--dry-run`. See [Kubernetes setup](kubernetes.md) for operational details.

If your operator already provides a cluster, use its existing identity instead.
`provision-if-absent` does not automatically adopt an externally created cluster
and can plan a second one. Register its exact identity and kubeconfig with
`npa cluster kubeconfig`, then bind the controller owner with
`npa skypilot bind-controller`. The [existing-cluster instructions](guides/sim2real-workflow.md#using-a-cluster-provision-if-absent-did-not-create)
show both commands.

Inspect the selected cluster and use the kubeconfig it reports:

```bash
npa cluster status --name "$cluster_name" --project "$project_alias"
export KUBECONFIG="${HOME}/.npa/clusters/${cluster_name}/kubeconfig"
kubectl config current-context
kubectl auth can-i create pods -n default
kubectl get nodes
```

If NPA reports a different kubeconfig path, use that path. Gate: the context
matches the chosen cluster, pod creation is authorized, and suitable nodes are
Ready. SkyPilot tasks use `default`; the `workbench` namespace is needed only
for deployed services.

## Bootstrap SkyPilot

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
npa skypilot verify --cluster "$cluster_name" --output-format json
npa workbench workflow gpus --project "$project_alias" \
  --cluster "$cluster_name" --spec "$workflow_spec" --json
```

Gate: the pinned runtime verifies Kubernetes access against this cluster, and
GPU discovery resolves the spec's accelerator. A bare `sky check` can succeed
with Kubernetes disabled or inspect another context. Each task's requested GPUs
must fit on one node; two one-GPU nodes cannot satisfy a two-GPU task.

The cluster also needs room for the jobs controller and any CPU workflow stages.
See [SkyPilot setup](../orchestration/skypilot-setup.md) and the workload runbook
for their resource requests.

## Verify the Image Channel

```bash
npa workbench workflow preflight-images "$workflow_spec" \
  --project "$project_alias" --infra "k8s/$cluster_name" \
  --var "bucket=$bucket_name" --json
```

Gate: every selected image passes. Supported NPA images pull anonymously from
`ghcr.io/nebius/nebius-physical-ai`. `NPA_REGISTRY` and saved registry settings
do not repoint those defaults. Use a complete image reference or explicit
workflow `--registry` only for intentional custom images, with exact-host
credentials if that registry is private.

## Run and inspect the result

Continue with the chosen workload's submission instructions:

- [Cosmos 3 generation](cosmos3-generate.md#workflow): generated media and
  `generate.json`.
- [Compositional Sim2Real](guides/sim2real-workflow.md): the full 14-stage
  composition, including runtime cache, CPU capacity, immutable images, and resume.
- [Isaac Lab BYOF](cookbooks/byof-isaac-lab/README.md): training checkpoint and
  its manifest.

Use the same project, cluster, bucket, and config overrides throughout. Forward
secrets by name with `--secret-env`; never put their values in workflow YAML.
Monitor the run to a terminal state and inspect its actual media, checkpoint,
or report. A successful plan or job-status response alone is not an artifact check.

Finish with the [teardown guide](../teardown.md): cancel active owned runs before
destroying resources that host them. Keep shared infrastructure intact.

## Common Failures

| Symptom | Next check |
| --- | --- |
| Kubernetes reports an anonymous-user `403` | Verify the exact kubeconfig/context and refresh its authentication. |
| S3 lists successfully but submission cannot write | Compare the workflow bucket/endpoint with NPA configuration and shell overrides; verify write access. |
| `No NPA cluster identity` or `No shared controller owner` | Complete the existing-cluster identity and controller-binding steps above. |
| `ImagePullBackOff` | Rerun image preflight for the exact reference; repair public-tag visibility or the explicit private-registry credentials. |
| GPU scheduling remains pending | Compare the spec with discovered GPU names, per-node capacity, and CPU/memory requests. |

For CLI/SDK examples and workflow authoring, see the
[walkthrough](cli-sdk-yaml-walkthrough.md) and
[workflow guide](../workbench-yaml-guide.md).
