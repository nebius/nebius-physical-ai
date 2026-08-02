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

## Redistribution Class

Every image in the packaging contract also declares
`redistribution: public | restricted`, which decides whether it may leave the
owning org:

- `public` — OSS-redistributable, may be mirrored to a public registry. **All 19
  workbench images are currently `public`.**
- `restricted` — bakes a runtime we may not redistribute. Currently **empty**: the four
  Isaac images (`isaac-lab`, `sonic`, `sonic-mujoco`, `groot`) used to be restricted
  because they baked Omniverse Kit, and were re-architected to fetch Isaac Sim / Isaac
  Lab at first run under the operator's own EULA acceptance instead. The class and its
  guards are kept for the next runtime we cannot ship.

When adding an image, set its class. `npa/tests/docker/test_packaging_contract.py` fails
the build if a Dockerfile **bakes** Omniverse Kit (or is built `FROM` a restricted image)
while claiming `public`. Keep `images.OMNIVERSE_RESTRICTED_TOOLS` in sync; it is what
`npa.deploy.publish_public` uses to decide what may be mirrored publicly.

Note the distinction the guard encodes: **baked at build time** vs. **fetched at run
time**. Mentioning `isaacsim` in bootstrap plumbing is fine; installing it in a `RUN`
layer is not. Two of its patterns exist specifically because the runtime-fetch design
created new ways to bake by accident — `RUN isaac-bootstrap ensure` and
`RUN /isaac-sim/python.sh ...` both materialise the whole install into a layer. If you
need a build-time interpreter in an Isaac image, use the image's own venv python (see
`Dockerfile.mujoco`'s `ARG NPA_IMAGE_PYTHON`), never the shim.

### Building an Isaac image

No NGC credentials are needed — nothing credentialed is left to pull:

```bash
npa/docker/workbench/isaac-lab/build.sh --registry cr.<region>.nebius.cloud/<id> --push
npa/docker/workbench/sonic/build.sh --registry cr.<region>.nebius.cloud/<id> --push --variant baked
```

Two practical notes from doing this on the dev VM: an Isaac image build can peak at ~90 GB
of scratch, so `docker builder prune -af` between builds, and prefer `--push` (buildx
streams to the registry) over a local build, which additionally unpacks ~30 GB into the
image store. Verify the result with
`npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py <ref>`.

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
