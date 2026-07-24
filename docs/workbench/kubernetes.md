# Workbench On Kubernetes

This guide is the Kubernetes-specific path after
[Workbench Getting Started](getting-started.md). Use it when a Workbench
workflow or service should run on a Nebius Managed Kubernetes cluster through
SkyPilot.

## What Runs Where

| Component | Namespace / location | Purpose |
| --- | --- | --- |
| Workbench services | `workbench` namespace | Long-lived FastAPI tools such as LanceDB, FiftyOne, or detection-training |
| SkyPilot task pods | `default` namespace | Short-lived workflow stages from `npa workbench workflow submit` or runner scripts |
| SkyPilot jobs controller | Kubernetes pod, normally on a CPU node | Tracks managed jobs and starts workflow task pods |
| Artifacts | S3-compatible bucket | Shared data bus for inputs, checkpoints, logs, and reports |

Workbench tools exchange data through S3 URIs. Do not wire tools together by
direct pod-to-pod file paths; use `--input-path` and `--output-path` values such
as `s3://<bucket>/<run-id>/...`.

## User Setup Checklist

Complete the platform quickstart first, then collect these values from your
operator:

```bash
export NEBIUS_PROJECT_ID=<your-project-id>
export NEBIUS_TENANT_ID=<your-tenant-id>
export NPA_S3_BUCKET=<your-bucket>
export NPA_REGISTRY=cr.eu-north1.nebius.cloud/<your-registry-id>
export AWS_ENDPOINT_URL=https://storage.eu-north1.nebius.cloud
export NPA_STORAGE_ENDPOINT=storage.eu-north1.nebius.cloud
```

Secrets belong in `~/.npa/credentials.yaml` or Kubernetes secrets, not in
committed workflow YAML. For S3-backed workflows, the credential file should
contain:

```yaml
storage:
  aws_access_key_id: <your-s3-access-key-id>
  aws_secret_access_key: <your-s3-secret-access-key>
  endpoint_url: https://storage.eu-north1.nebius.cloud
  bucket: s3://<your-bucket>/
```

Verify local access before launching a GPU job:

```bash
nebius iam get-access-token >/dev/null
aws s3 ls "s3://${NPA_S3_BUCKET}/" --endpoint-url "${AWS_ENDPOINT_URL}"
docker login cr.eu-north1.nebius.cloud
```

## Kubernetes Access

Select the managed Kubernetes context provided by your operator:

```bash
kubectl config get-contexts
kubectl config use-context <your-nebius-mk8s-context>
kubectl config current-context
```

SkyPilot must be able to create pods in the namespace it uses, normally
`default`:

```bash
kubectl auth can-i create pods -n default
kubectl auth can-i list pods -n default
kubectl auth can-i list nodes
kubectl get nodes
kubectl get namespace workbench
kubectl get secret npa-nebius-registry -n default
```

Expected result: the `kubectl auth can-i` commands print `yes`, nodes are
listed, the `workbench` namespace exists for services, and the registry pull
secret exists in the SkyPilot namespace. If `sky check` later reports an
anonymous-user `403`, refresh the kube context before debugging workflow YAML.

## SkyPilot Runtime

Use the NPA-managed SkyPilot virtualenv. Do not rely on an unrelated `sky` from
`PATH`.

```bash
npa skypilot bootstrap
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
npa skypilot status
"${NPA_SKYPILOT_BIN}" check
```

The validated SkyPilot version is `0.12.2`. NPA defaults managed jobs to a
Kubernetes controller. The cluster needs a CPU node that can fit the controller
pod, normally 4 vCPU and 16 GiB memory, so it does not compete with GPU
workloads.

## Using Workbench Services

Deploy long-lived services into the Workbench namespace when you want CLI, SDK,
or workflow stages to call the same service endpoint:

```bash
npa workbench detection-training deploy \
  --output-path "s3://${NPA_S3_BUCKET}/detection-training/" \
  --storage-endpoint storage.eu-north1.nebius.cloud \
  --namespace workbench \
  --gpu-type h100
```

Inside Kubernetes, use the cluster-local service endpoint printed by the deploy
command, for example:

```bash
export NPA_DETECTION_TRAINING_ENDPOINT=http://npa-detection-training.workbench.svc.cluster.local:8790
```

From a SkyPilot task, call the service through the CLI, SDK, or HTTP endpoint.
From your laptop, use the endpoint form and access path documented by the tool
or operator.

## Submitting Workflows

Prefer NPA runner scripts or `npa workbench workflow submit` over raw `sky`
commands. They materialize known placeholders, pass secrets, and keep status and
cleanup behavior consistent.

```bash
RUN_ID=workbench-$(date -u +%Y%m%dT%H%M%SZ)

npa workbench workflow submit \
  npa/src/npa/workflows/skypilot/vlm-eval.yaml \
  --run-id "${RUN_ID}" \
  --durable-s3 \
  --workflow-s3-uri "s3://${NPA_S3_BUCKET}/workflows/${RUN_ID}/" \
  --s3-endpoint "https://storage.eu-north1.nebius.cloud" \
  --infra "k8s/<your-nebius-mk8s-context>"
```

Monitor from S3-backed workflow state:

```bash
npa workbench workflow status "s3://${NPA_S3_BUCKET}/workflows/${RUN_ID}/" --watch
npa workbench workflow logs "s3://${NPA_S3_BUCKET}/workflows/${RUN_ID}/" --stage <stage>
npa workbench workflow artifacts "s3://${NPA_S3_BUCKET}/workflows/${RUN_ID}/"
```

Raw YAML launch is useful for debugging, but SkyPilot `0.12.2` does not
interpolate `${VAR}` placeholders inside `envs:`. If you launch raw YAML, render
placeholder values into a temporary file first and avoid committing concrete
project, bucket, registry, or secret values.

## GPU Routing

Pick the GPU by workload, not by availability alone:

| Workload | Typical Kubernetes accelerator | Notes |
| --- | --- | --- |
| General training, VLM eval, CLIP, detection | `H100:1` or `H200:1` | Good for headless compute workloads |
| Isaac Lab / render validation | `L40S:1` or RTX PRO 6000 class | Requires RT cores; H100/H200 are not valid for render paths |
| Cosmos2 transfer / Cosmos3 reason / modern Sim2Real components | `RTXPRO6000:1` or the cluster's RTX PRO 6000 alias | Requires images built for the target GPU stack |
| CPU control stages | No accelerator | Keep trigger and orchestration stages off GPU nodes |

Check the aliases your cluster exposes before submitting raw YAML:

```bash
"${NPA_SKYPILOT_BIN}" show-gpus --infra kubernetes --all
```

## Common Failures

| Symptom | Likely cause | Fix |
| --- | --- | --- |
| `sky check` reports anonymous-user `403` | Expired or wrong kube context | Re-select or refresh the Nebius MK8s context, then rerun `kubectl auth can-i ...` |
| Image pull returns `401 Unauthorized` | Expired Nebius registry token in the pull secret | Recreate `npa-nebius-registry` in the SkyPilot namespace |
| Pod stays pending with no matching GPU | Requested accelerator alias is not exposed by the cluster | Run `show-gpus`, then update the workflow GPU value or ask for the node group |
| S3 upload fails with `NoSuchBucket` | Bucket name, endpoint, or region mismatch | Use `https://storage.eu-north1.nebius.cloud` and verify `NPA_S3_BUCKET` has no `s3://` prefix |
| Literal `${AWS_ENDPOINT_URL}` appears in logs | Submitted raw YAML without materialization | Use an NPA runner or render a temporary YAML before raw SkyPilot launch |

## Next Steps

- [cli-sdk-yaml-walkthrough.md](cli-sdk-yaml-walkthrough.md): how CLI, SDK, and
  workflow YAML call the same Workbench service.
- [../orchestration/skypilot-setup.md](../orchestration/skypilot-setup.md):
  SkyPilot runtime details.
- [../cli/workflow.md](../cli/workflow.md): durable workflow status, logs, and
  artifact commands.
