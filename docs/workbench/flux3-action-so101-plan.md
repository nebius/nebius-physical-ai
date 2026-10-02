# FLUX 3 Action SO-101 task LoRA in Workbench

## Scope

Add SO-101 **task adaptation** using BFL's LeRobot LoRA recipe. This is the
prepared SO-101 checkpoint, not full fine-tuning of the action base or a new
robot embodiment. Keep it in the existing `npa workbench lerobot` surface:
`flux3-so101-finetune` is one new command and
`workbench.lerobot.flux3_so101_finetune` is its workflow catalog entry.
The [one-stage workflow](../../workflows/partners/bfl/flux3-action-so101-finetune.yaml)
uses the registered catalog entry.

The current generic `workbench.lerobot.policy_train` runs `lerobot-train`
with `--policy.type`; it cannot supply BFL's `lora.json`, the prepared
checkpoint and shared encoders, or their saved processors. The published
LeRobot images also do not establish the required Torch 2.11/CUDA 12.8/NATTEN
stack. A distinct reviewed image is needed.

## Input contract

- BFL policy: `black-forest-labs/flux-3-action-so101@c9e13b2aca6a0a472b3ea03fd90cff32d3e85849`.
- Shared encoders: `black-forest-labs/flux-3-action-base@62878e2925e59b7a89ec14463ce89932624c490d`.
- Example data: `LightwheelAI/leisaac-pick-orange@fa6e0625d814352b8e6ee1c6d2482194e4da8ed3`. The public card reports 60 episodes, 36,293 frames, 30 Hz, six action/state channels, two AV1 cameras, and Apache-2.0. Its metadata says LeRobot v2.1; convert to v3, then reopen it through the LeRobot v3 reader. Verify the revision still resolves before execution.
- Map dataset `observation.images.front` to checkpoint `observation.images.scene`; retain `observation.images.wrist`. Inspect a decoded pair to confirm scene is left and wrist is right. Verify all six channel names, units, action alignment, task text, and calibration. The model card describes the prediction as joint deltas plus absolute gripper; the BFL training guide requests raw absolute commanded actions. Let the checkpoint's saved processors perform its representation transform and reject mismatched inputs rather than normalizing or differencing twice.
- Keep model files, saved processors, and normalization statistics together for resume and inference. Fetch exact revisions using the operator's Hugging Face access; do not bake weights or tokens into the image.

## Implementation status

The pinned dataset conversion, calibration, training command, raw and EMA
checkpoint publication, finite-loss check, adapter reload, and S3 readback are
implemented in Workbench. The CLI, SDK, toolRef, image route, packaging
contract, golden eval declaration, and one-stage YAML are wired. Local
contract, access, workflow-render, container startup, CLI, real AV1 decode,
and full pinned dataset conversion checks pass. The Dockerfile restores LeRobot's required PyAV 15.1.0 after the NPA install
and splits CUDA dependencies into registry-friendly layers. The image passes
dataset import and real AV1 decode. The exact private registry manifest, amd64
pull, and target-cluster pull with a task-owned Secret passed. A four-microstep
H100 smoke reached terminal success with finite loss, verified raw and EMA
adapters, and a complete S3 artifact. The full 60,000-step run is pending
operator federation reauthentication; its acceptance is recorded separately
in the workflow readiness record.

## Smallest implementation

1. Pin a LeRobot main commit that contains FLUX 3 support, the matching `examples/flux3/lora.json`, Torch 2.11/CUDA 12.8, NATTEN `0.21.6+torch2110cu128`, FFmpeg with AV1 decode, and the `flux3` extras. Build a dedicated H100-compatible `npa-lerobot-flux3` image. Record package/source pins, classify baked bytes and runtime-fetched weights, scan the built image, run its capability smoke, and publish an immutable digest only after the image gates pass.
2. Add one shared `npa.workbench.lerobot` implementation that downloads the exact model/data revisions, converts and validates LeRobot v3 data, runs the BFL preset with `--policy.path`, `--policy.video_vae_id`, `--policy.text_encoder_id`, the camera rename map, and the requested step count. Keep BFL's rank/alpha 32, batch 2, accumulation 4, BF16, learning rates, gradient checkpointing, and EMA settings unless the pinned preset proves otherwise. Use one GPU; check actual peak memory on the target GPU. A four-microstep smoke tests the runtime with one optimizer update, then the full configured 60,000 microsteps runs under a fresh ID. At accumulation 4, this is 15,000 optimizer updates.
3. Expose that implementation through `npa workbench lerobot flux3-so101-finetune` and the SDK, with `--steps`, `--run-id`, and S3 `--output-path`. Pin the dataset, policy, base, and camera map in the recipe. Make the command exit nonzero on missing revisions, bad dataset contract, nonfinite training, missing checkpoint, upload failure, or failed readback. Refuse a nonempty run prefix.
4. Add `workbench.lerobot.flux3_so101_finetune` to the catalog, route that exact toolRef to the new image, and switch the example YAML from `run.argv` to `toolRef`. Add CLI/SDK, command-argv, workflow-render, image, docs, and skill tests in the existing Workbench gates. Do not add a new top-level `npa` command or duplicate the generic LeRobot runner.
5. Publish numbered raw checkpoints, EMA siblings, train config, source/model/data revision manifest, metrics/logs, and SHA-256 checksums to the run-scoped S3 `artifacts/` prefix. NPA keeps its workflow manifest at the parent prefix. Write `COMPLETE.json` last, after independent Object Storage readback and a successful adapter reload plus finite inference using the original base, encoders, and processors. If resume is added later, bind it to an exact raw checkpoint and its recorded recipe.

## Acceptance

Local schema and render checks prove only that the workflow is well formed.
On Nebius, require the exact managed job to reach terminal success, readable
training logs and finite losses, the declared S3 `COMPLETE.json`, verified
checkpoint hashes, and an adapter reload/inference check. A four-update smoke
does not replace the 60,000-step run. To claim improved pick-orange policy
quality, add paired closed-loop LeIsaac evaluation of the base and trained
adapter on the same held-out starts/seeds, with the same camera mapping,
processors, 30 Hz control, and success definition. Report success counts and
uncertainty; finite loss or a checkpoint alone is not a quality result.

## Execution after implementation

On a Linux operator host with `/proc`, replace the workflow's image
placeholder with the reviewed digest and the bucket placeholder with an
operator-owned writable bucket. Check exact model
access, dataset revision, GPU capacity, image pullability, and S3 permissions.
Then use a fresh run ID:

```bash
SPEC=workflows/partners/bfl/flux3-action-so101-finetune.yaml
RUN_ID=flux3-so101-$(date -u +%Y%m%dT%H%M%SZ)
npa/.venv/bin/npa workbench workflow validate-spec "$SPEC" --json
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID" --var bucket=<your-bucket> --var flux3_image=<reviewed-image-digest> --var image_pull_secret=<existing-pull-secret> --check-render --json
npa/.venv/bin/npa workbench workflow preflight-images "$SPEC" --infra "k8s/<context>" --var bucket=<your-bucket> --var flux3_image=<reviewed-image-digest> --var image_pull_secret=<existing-pull-secret> --image-pull-secret <existing-pull-secret>
npa/.venv/bin/npa workbench workflow submit "$SPEC" --infra "k8s/<context>" --project <project-alias> --run-id "$RUN_ID" --var bucket=<your-bucket> --var flux3_image=<reviewed-image-digest> --var image_pull_secret=<existing-pull-secret> --secret-env HF_TOKEN
```

The workflow uses a real command and an unpublished image. Its [readiness record](../../workflows/partners/bfl/flux3-action-so101-finetune.readiness.json)
tracks the exact stage of live acceptance. Run submission from a Linux operator with `/proc`; macOS supports only validation and planning.

Sources: [BFL SO-101 recipe](https://docs.bfl.ai/flux_3/flux3_action_so101),
[LeRobot FLUX 3 integration](https://huggingface.co/docs/lerobot/main/en/flux3),
[BFL SO-101 model card](https://huggingface.co/black-forest-labs/flux-3-action-so101),
[PickOrange dataset card](https://huggingface.co/datasets/LightwheelAI/leisaac-pick-orange).
