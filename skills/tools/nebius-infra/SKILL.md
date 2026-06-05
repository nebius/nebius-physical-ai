---
name: nebius-infra
description: Use for Nebius runtime configuration, provision-if-absent setup, cluster, registry, storage, GPU routing, and credential assumptions that affect NPA runs.
---

# Nebius Infrastructure

## When To Use

Use this skill before running NPA workloads that need Nebius runtime settings,
object storage, container registry access, Kubernetes, or GPU routing. Also use
it when reviewing setup changes touching `npa configure`, `npa
provision-if-absent`, `~/.npa/config.yaml`, `~/.npa/credentials.yaml`, or
workflow environment variables.

## Procedure

1. Keep committed files public-repo safe. Never hardcode project IDs, tenant IDs,
   registry IDs, bucket names, VM IPs, private endpoints, or secrets.
2. Capture runtime configuration with `npa configure`. In non-interactive or CI
   contexts, pass explicit flags and placeholders:

   ```bash
   npa configure --non-interactive \
     --project ci \
     --project-id project-ci \
     --tenant-id tenant-ci \
     --region eu-north1 \
     --registry-id registry-ci \
     --s3-bucket s3://ci-bucket/checkpoints/ \
     --aws-access-key-id access \
     --aws-secret-access-key secret
   ```

3. Ensure runtime resources with the additive-only setup command:

   ```bash
   npa provision-if-absent --project ci --dry-run --skip-validate --output-format json
   ```

   Real runs may ensure S3 and Kubernetes. Dry runs only resolve settings and
   print intended actions. The command must not teardown or replace resources.

4. Use `--skip-s3` or `--skip-k8s` when the operator only wants one side
   checked. Use `--sky-smoke` only when live GPU validation is explicitly
   requested.

## Three-Tier Contract

- CLI: `npa configure` writes runtime config and credentials; `npa
  provision-if-absent` ensures missing S3/Kubernetes resources or reports the
  dry-run plan.
- SDK: `npa.sdk.config.resolve_runtime_config`,
  `npa.sdk.config.write_runtime_config`, and
  `npa.sdk.provisioning.provision_if_absent`.
- YAML: workflow YAML reads runtime values through environment variables such as
  `NPA_PROJECT_ID`, `NPA_TENANT_ID`, `NPA_REGION`, `NPA_REGISTRY`,
  `NPA_REGISTRY_ID`, `NPA_S3_BUCKET`, `NPA_STORAGE_ENDPOINT`, and AWS S3 keys.

## GPU Routing

- H100: general training, CLIP embedding, detection, MJLab evaluation, Cosmos
  inference that does not need RT cores, and non-render throughput work.
- L40S: Isaac Lab and SONIC render validation on VM hosts.
- RTX PRO 6000 Blackwell on Kubernetes: Isaac Lab and SONIC render validation
  with NVIDIA GPU Operator mounted drivers.
- H100/H200 do not provide RT cores; do not route Isaac Lab or render validation
  there unless the task explicitly avoids rendering.

## Gotchas

- Use `https://storage.eu-north1.nebius.cloud` for the current primary region.
- Nebius IAM registry tokens expire. If Kubernetes image pulls fail with `401
  Unauthorized`, refresh the registry pull secret in the namespace that owns the
  pod.
- SkyPilot task pods run in `default`; deployed workbench services run in
  `workbench`.
- Cached kubeconfig reuse is a success path for `provision-if-absent`; absence
  of a cached kubeconfig triggers Terraform only outside dry-run mode.

## Verify

Run the CI-backed dry-run example:

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

That test invokes `npa configure --non-interactive` and `npa
provision-if-absent --dry-run --output-format json` against temporary config
paths and asserts the S3/Kubernetes actions are reported without live writes.
