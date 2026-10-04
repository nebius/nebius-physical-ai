---
name: embodiedgen
description: Use when running or reviewing the operator-private EmbodiedGen V2 image-to-rigid-object workflow with upstream TRELLIS generation, URDF export, collision validation, and PyBullet evidence.
---

# EmbodiedGen V2 image-to-rigid-object

Use this skill for the scoped EmbodiedGen V2 workflow only. The supported
backend here is upstream TRELLIS, not SAM3D or the Tencent cloud backend.

1. Read `docs/workbench/embodiedgen.md` and inspect
   `workflows/testing/byof-embodiedgen.yaml`.
2. Run `npa workbench health preflight --checks nebius --json`, the general
   health preflight, and an exact public TRELLIS model-access probe before GPU
   submission. Do not invent an acceptance variable: TRELLIS is public; SAM3D
   is not selected because its manual-gated terms are separate.
3. Build only to an operator-private registry with a
   `dev-<full-40-character-source-sha>` tag. Never publish the CUDA development
   derivative anonymously.
4. Supply the Token Factory credential through supported workflow secret
   plumbing. It is used by EmbodiedGen's upstream URDF property-estimation path
   and must not appear in logs, outputs, or manifests.
5. Require `embodiedgen_image_to_rigid_object.json` to show the upstream source,
   TRELLIS source/model revisions, the upstream URDF-to-MJCF handoff, generated
   collision hashes, actual GPU facts, PyBullet contact/settling, and a decoded
   viewable MP4.

The generated URDF's mass, height, and friction are VLM estimates, not
calibrated measurements. Do not claim articulated-object generation, ground
truth physical properties, or policy improvement from this workflow.
