---
name: flux-action
description: Fine-tune FLUX 3 Action on a declared robot embodiment using LeRobot demonstrations and the pinned standalone trainer.
---

# FLUX Action

Use `npa workbench flux-action finetune`, shared with
`npa.sdk.workbench.flux_action.finetune` and
`workbench.flux_action.finetune` in `workflows/testing/flux-action-finetune.yaml`.
Read `docs/workbench/flux-action.md` for the full input and output contract.

1. Obtain an operator-owned LeRobot v2.x/v3.x training split and JSON recipe in
   S3. Define action/state order and units, real video camera mappings, canvas,
   FPS, representation, and gripper inversions explicitly. The current native
   model requires equal state/action widths and 34 or more frames per episode.
2. Run `--dry-run` to check the recipe against dataset metadata. This reads S3;
   it does not decode videos, allocate GPUs, download weights, or train.
3. Before a GPU submission, follow health-preflight, GPU selection, and
   third-party-eula-preflight. Verify access to the exact pinned base/encoders.
   Use operator credentials; do not collect a second NPA terms attestation.
4. Build the candidate privately using the repository secure-image-build flow.
   An earlier candidate was privately built; GPU acceptance and public publication
   remain quarantined. Rebuild changed source before testing it. Do not treat
   its Dockerfile or CPU adapter tests as evidence of GPU training success.
5. Set the workflow's image override to the built image and match the GPU
   resource count to `flux_processes`. This is one node, not multi-node training.
   Training duration and schedule come from the operator's recipe.
6. A successful result needs the final COMPLETE checkpoint, finite metrics,
   positive trunk/head learning rates, fresh-process finite export inference,
   and independently stream-hashed S3 exports. Read `result.json`; absence means incomplete.
   Failed runs retain local files and attempt S3 upload of logs/checkpoints.
7. Keep evaluation data separate from training before indexing. No task-success
   or closed-loop quality claim follows from training loss or export completion.

The container fetches FLUX Kommunity weights at runtime; no data, weights, or
populated caches belong in image layers. The HF cache is ephemeral unless NPA
model-cache storage is explicitly configured. Exports reference pinned external
encoders; they are not self-contained bundles.

Native standalone steps are optimizer updates. Use the ALOHA full/smoke recipes
explicitly and preserve `val_episodes` (default 1; public example 5). Supply an
immutable `flux_image` and `image_pull_secret`; no public accepted image exists.

For one-GPU qualification, use the explicit ALOHA single-GPU smoke recipe and
workflow. Native BF16 parameters with `ema_sigma_rels: []` and `export_profile:
model` avoid the default FP32 EMA copies. All trunk/head weights still train.
Require an already Bound disk PVC, mount it at `/npa-work`, and direct TMPDIR
and HF_HOME there. Preserve system `/tmp` permissions for SkyPilot apt setup.
The observed one-H100 FP32 attempt exhausted 79.18 GiB during EMA construction;
no optimizer update completed. A BF16 attempt completed four updates and export
but failed reload because native training leaves sampler settings unset.
Recipes must carry validated `inference` fields into the exported policy;
validate the native inference config before GPU submission. The Euler/four-step
ALOHA settings are diagnostic, not benchmarked. End-to-end acceptance remains
unverified until the repaired image completes fresh-process reload and durable
artifact verification.
