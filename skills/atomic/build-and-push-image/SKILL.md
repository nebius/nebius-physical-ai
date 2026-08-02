---
name: build-and-push-image
description: Use when building, tagging, validating, or pushing NPA workbench container images for Nebius registry-backed workflows.
---

# Build And Push Image

## When To Use

Use this skill when a task changes Dockerfiles, image manifests, registry tags,
or workflow image references for NPA workbench tools.

## Procedure

1. Resolve runtime registry settings with `npa configure` or
   `npa.clients.config.resolve_container_registry` or `npa configure`.
2. Build from the checked-in Dockerfile for the tool; do not invent a detached
   image source outside the repo.
3. Tag images with the configured registry prefix and a version that matches the
   tool manifest or release plan.
4. Inspect the image or manifest before pushing.
5. Update image manifests, workflow YAML, and skill guidance together when a
   command starts depending on the new image.

## Three-Tier Contract

- CLI: use `docker buildx build`, tool-specific `deploy --container-image`, and
  command help from the affected workbench tool.
- SDK: resolve registry and project settings through `npa.clients.config` instead of
  hardcoded env reads.
- YAML: workflow `image_id` values should come from variables or manifests,
  especially SONIC's `npa/src/npa/deploy/sonic_image_manifest.json`.

## Packaging Contract

Before changing Dockerfiles, read `docs/workbench/container-packaging.md` and
update `npa/docker/workbench/packaging-contract.yaml` when adding an image or
changing its tier (`service` / `job` / `interactive`).

Security baseline: non-root final USER, no secrets in layers, digest-pinned
bases where possible, Trivy scan coverage. Service images should expose ports
and prefer a `HEALTHCHECK` or K8s probe on `/health`.

## GPU Architecture Coverage

An image only runs on a GPU whose architecture it was compiled for, and there
are two independent knobs:

- A prebuilt torch wheel ships a fixed fat-binary arch set that
  `TORCH_CUDA_ARCH_LIST` cannot change; only the wheel index does. cu128/cu130
  include `sm_100` and `sm_120`, cu124/cu126 stop at `sm_90`. Check with
  `torch.cuda.get_arch_list()`.
- Source-compiled extensions (flash-attn from source, Taichi, natten, custom
  ops) obey `TORCH_CUDA_ARCH_LIST` at build time; omitting an arch fails at
  runtime with `no kernel image is available for execution on the device`.

`npa-base` (`base/cuda13-b300`) builds `8.0 9.0 10.0 10.3 12.0` and asserts the
wheel reports `sm_80 sm_90 sm_100 sm_120`; override with `build.sh --arch-list`
/ `--require-archs`. Validate an image with
`npa/scripts/validate_blackwell_image.sh <image> --target b200|b300 --gpu`.
Use additive tags that name the architectures
(`cuda13-b300-sm80-sm90-sm100-sm103-sm120-<UTC>`) and record the per-image
verdict in `npa/docker/workbench/blackwell-dc-images.json`. Background:
`docs/workbench/blackwell-datacenter-image-compatibility.md`.

## Gotchas

- Do not commit concrete registry IDs or private image digests from a live
  account unless the repo already treats that value as public.
- Nebius registry auth expires; a push or pull failure may require a refreshed
  token rather than an image change.
- For GPU-specific images, verify the target GPU family before changing defaults.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test verifies current deploy command help and the image-manifest file
paths used by the skill.
