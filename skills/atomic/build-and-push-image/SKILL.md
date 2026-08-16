---
name: build-and-push-image
description: Use when building, tagging, validating, or publishing NPA workbench container images through private-candidate and public-release GHCR channels.
---

# Build And Push Image

## When To Use

Use this skill when a task changes Dockerfiles, image manifests, registry tags,
or workflow image references for NPA workbench tools.

## Procedure

For SkyPilot workflow images, build/test the versioned bootstrap contract and
record it in OCI config. Preflight consumes the selected digest, not its tag.
Keep live-validation tags immutable, scan built bytes for restricted payloads,
and preserve the private-candidate to public-release digest chain.

1. Resolve execution overrides with `npa configure` or
   `npa.clients.config.resolve_container_registry`. Official publication uses
   `NPA_PRIVATE_REGISTRY` and `NPA_PUBLIC_REGISTRY` independently.
2. Build from the checked-in Dockerfile for the tool; do not invent a detached
   image source outside the repo.
3. Tag private candidates as `dev-<full-git-sha>`. Promote their immutable
   digest to the supported release tag only after validation.
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

- `public` — OSS-redistributable, may enter the official private candidate and
  public release channels. Every
  canonical image in `CONTAINER_IMAGE_NAMES` is currently `public`.
- `restricted` — bakes a runtime we may not redistribute. The four
  Isaac images (`isaac-lab`, `sonic`, `sonic-mujoco`, `groot`) used to be restricted
  because they baked Omniverse Kit, and were re-architected to fetch Isaac Sim / Isaac
  Lab at first run under the operator's own EULA acceptance instead. The separately
  contracted `cosmos3-serving` image is now restricted/build-your-own because its
  pinned vendor base carries derived-distribution conditions that anonymous GHCR does
  not establish.

When adding an image, set its class. `npa/tests/docker/test_packaging_contract.py` fails
the build if a Dockerfile **bakes** Omniverse Kit (or is built `FROM` a restricted image)
while claiming `public`. Keep `images.OMNIVERSE_RESTRICTED_TOOLS` in sync; it is what
`npa.deploy.publish_public` uses to decide what may be released publicly.

`restricted` images, including `cosmos3-serving`, may enter neither official
GHCR channel. Build them only into a generic operator-controlled registry; a
private package or development tag does not change redistribution rights.

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
npa/docker/workbench/isaac-lab/build.sh --registry <your-registry>/<namespace> --push
npa/docker/workbench/sonic/build.sh --registry <your-registry>/<namespace> --push --variant baked
```

Two practical notes from doing this on the dev VM: an Isaac image build can peak at ~90 GB
of scratch, so `docker builder prune -af` between builds, and prefer `--push` (buildx
streams to the registry) over a local build, which additionally unpacks ~30 GB into the
image store. Verify the result with
`npa/.venv/bin/python npa/scripts/scan_image_omniverse_payload.py <ref>`.

## GPU Architecture Coverage

An image only runs on a GPU whose architecture it was compiled for, and there
are two independent knobs:

- A prebuilt torch wheel ships a fixed fat-binary arch set that
  `TORCH_CUDA_ARCH_LIST` cannot change; only the wheel index does. cu128/cu130
  include `sm_100` and `sm_120`, cu124/cu126 stop at `sm_90`. Check with
  `torch._C._cuda_getArchFlags()`. `torch.cuda.get_arch_list()` returns `[]`
  when the build host has no visible GPU and is not valid build-time evidence.
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

The current alias for that contract is
`cuda13-b300-sm80-sm90-sm100-sm103-sm120-v2-latest`. Its `v2` generation is
additive: it was introduced after physical B300 testing proved that the first
five-architecture alias baked an older validator which incorrectly demanded a
literal `sm_103` wheel entry. The v2 image bakes the same-major `sm_100` →
`sm_103` SASS rule. Never repoint either historical alias; introduce a new
truthful generation if another published contract must be superseded. Retain
`cuda13-b300-sm80-sm90-sm120-latest` and the first five-architecture alias only
where historical evidence refers to them.

### Registry-native parent rebases

A registry-native parent rebase is an allowed recovery when the dev VM cannot
materialize a very large, otherwise unchanged child image. Use it only when all
of these are true:

- the child Dockerfile instructions and application payload are unchanged
  except for selecting the replacement parent image;
- the child does not compile CUDA extensions during its build;
- the old child's layer list starts with the exact old parent layer list;
- the rebased child's layer list starts with the exact new parent layer list,
  and its remaining child-layer suffix is byte-for-byte identical;
- provenance labels are updated, the tag is additive, both registry digests
  match, and the exact final tag passes its baked validators plus a real GPU
  capability smoke.

Describe this result as **rebased**, not rebuilt. Never use this shortcut for a
changed Dockerfile, a child that compiles extensions, or as a substitute for
physical GPU execution.

## Gotchas

- Do not commit concrete registry IDs or private image digests from a live
  account unless the repo already treats that value as public.
- Private GHCR auth is explicit and exact-host; a push or pull failure may be a
  package-permission problem rather than an image change.
- For GPU-specific images, verify the target GPU family before changing defaults.

## Verify

```bash
npa/.venv/bin/python -m pytest npa/tests/guardrails/test_skills_index.py -q
```

The smoke test verifies current deploy command help and the image-manifest file
paths used by the skill.
