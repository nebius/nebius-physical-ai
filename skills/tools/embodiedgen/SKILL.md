---
name: embodiedgen
description: Use when running or reviewing the operator-private EmbodiedGen V2 image-to-rigid-object workflow with upstream TRELLIS generation, URDF export, collision validation, and PyBullet evidence.
---

# EmbodiedGen V2 image-to-rigid-object

Use this skill only for the scoped, Tier-1 EmbodiedGen V2 workflow. The stable
user surface is [`workflows/testing/byof-embodiedgen.yaml`](../../../workflows/testing/byof-embodiedgen.yaml)
through generic `npa workbench workflow` commands—not a persistent service,
separate SDK submitter, or an `embodiedgen` command family. Start with the
[operator guide](../../../docs/workbench/embodiedgen.md), then use the
[workflow catalog](../../../workflows/testing/README.md#bring-your-own-framework)
and [OSS capability record](../../../docs/workbench/oss-solution-catalog.md#embodiedgen-v2).

## Capability boundary

The only candidate capability is
`img3d-cli_trellis_image_to_urdf_pybullet`. The real upstream TRELLIS backend
runs through EmbodiedGen `img3d-cli`, emits a URDF and collision meshes,
converts that URDF to MJCF, and validates the exact generated URDF in PyBullet
for gravity contact, settling, and a fully decoded MP4. TRELLIS is selected;
SAM3D and the Tencent cloud backend are not substitutes.

Generated mass, height, and friction are VLM estimates—not calibrated ground
truth. Do not claim articulated-object generation, physics validation in
MuJoCo/Genesis, downstream policy improvement, or model quality. MJCF is a
supported handoff only; PyBullet is the simulator actually exercised.

## Before submission

1. Resolve the project, private registry, storage, and Kubernetes context from
   supported NPA configuration. A host name is not a project alias.
2. Run `npa workbench health preflight --checks nebius,token_factory,s3 --json`
   with the selected project, then `npa workbench health access --capability
   token_factory --json`. TRELLIS is public: do not invent an acceptance flag.
3. Require an operator-private, immutable
   `REGISTRY/npa-embodiedgen@sha256:<64-hex>` image. The tag before resolution
   must be `dev-<full-40-character-NPA-sha>`. Never publish or use a public
   registry image.
4. Pass `NEBIUS_TOKEN_FACTORY_KEY` only with `--secret-env`; it is needed by
   upstream property estimation and must never be written to YAML, reports, or
   shell logs.
5. Use a worker-readable `https://` or `s3://` image URI, never a local path.
   The workflow base64-transports it through the rendered shell as literal data;
   reachability and storage permission remain separate prerequisites.

## Operate through Workbench

Run `validate-spec`, `plan-spec --check-render`, `submit --runtime --plan-only`, target
`preflight-images`, then submit with one matching `base_image` config value and
`--image-override workbench.byof.repo=<same-digest>`. Reuse the same project,
run ID, workflow S3 prefix, input URI, bucket, and image digest for status,
logs, artifacts, cancellation, and any `--resume-run` reconciliation. Do not
use raw SkyPilot commands or create a parallel controller.

Use the same task-scoped `--config-path` for plan, image preflight, submit, and
resume. Its private SkyPilot configuration must select the intended context,
the standard `skypilot-service-account`, and an existing private-registry pull
Secret; see the [operator guide](../../../docs/workbench/embodiedgen.md#private-registry-target-configuration).

The workflow declares and requires these actual S3 outputs in addition to the
summary: `generated_asset.tar.gz` (URDF plus mesh layout), `mjcf_asset.tar.gz`,
`pybullet_view.png`, `pybullet_settle.mp4`, and
`embodiedgen_image_to_rigid_object.json`. Read the primary JSON and its hashes
before unpacking a bundle or handing it downstream. Use generic
`npa workbench workflow artifacts` or `npa studio search` for discovery; the
generic `npa.sdk.workbench.workflow` SDK is read-side only.

For interruption, inspect durable status/logs/artifacts first. The workflow has
no TRELLIS mid-generation checkpoint; a verified retry can rerun the stage but
must not be described as model-level resume. Cancel only the exact run and use
`npa cleanup --json` to audit local residue. Never clean up a shared controller,
cluster, cache, registry, or another owner's resources.

## Delivery and qualification

EmbodiedGen source is Apache-2.0; TRELLIS source and
`microsoft/TRELLIS-image-large@25e0d31ffbebe4b5a97464dd851910efc3002d96` are
MIT and runtime fetched. The CUDA bootstrap's delivery boundary and component
inventory are in `npa/docker/workbench/embodiedgen/REDISTRIBUTION.md` and
`THIRD_PARTY_NOTICES.md`. Runtime fetch does not grant any source, model, input,
output, or service right.

This candidate remains live-qualification pending until one exact private
digest passes the required byte scans and a real RTX PRO 6000 run produces the
declared artifacts. A plan, container build, local test, or historical evidence
does not satisfy that gate.
