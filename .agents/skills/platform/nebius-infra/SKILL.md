---
name: nebius-infra
description: Use for Nebius cluster, registry, storage, GPU routing, and credential assumptions that affect NPA runs.
---

# Nebius Infrastructure

Primary cluster: `npa-workbench-eu-north1`, Managed Kubernetes, `eu-north1`.

Always use storage endpoint `storage.eu-north1.nebius.cloud`. The CLI default `storage.uk-south1.nebius.cloud` is wrong for this cluster; override it explicitly.

## Resource Identifiers

- Registry prefix: `cr.eu-north1.nebius.cloud/${NPA_REGISTRY_ID}/`. Use `${NPA_REGISTRY_ID}` in committed files, never a concrete registry ID.
- S3 bucket: `${NPA_S3_BUCKET}`. Parameterize bucket names; do not hardcode them in committed files.
- Project and tenant IDs live in `~/.npa/credentials.yaml`; never hardcode them in source.

## GPU Routing

- H100, `1gpu-16vcpu-200gb`: general training, CLIP, detection.
- L40S, `1gpu-40vcpu-160gb`: Isaac Lab and SONIC render validation on compute-only VM hosts; use the baked SONIC image variant.
- RTX PRO 6000 Blackwell on Kubernetes: Isaac Lab and SONIC render validation with NVIDIA GPU Operator mounted drivers; use the host-mounted SONIC image variant.
- Isaac Lab and Cosmos rendering/visual-generation paths require RT cores, such as L40S or RTX Pro 6000. They will not work on H100/H200 for rendering/simulation paths that need RT cores. Standard Cosmos serving/inference only requires a GPU; see the Cosmos skill.
- B300/Blackwell: `sm_103` support is blocked on upstream libraries as of mid-2026; do not prioritize B300 enablement unless the vendor stack has moved.

## Kubernetes Namespaces

- SkyPilot task pods run in `default`.
- Workbench services run in `workbench`.

## Registry Pull Secrets

Nebius IAM tokens expire. If SkyPilot task pods fail image pulls with `401 Unauthorized`, regenerate the token and recreate the `npa-nebius-registry` image pull secret in the `default` namespace.
