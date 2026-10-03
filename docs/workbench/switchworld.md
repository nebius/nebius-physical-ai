# SwitchWorld enhanced LingBot-World adapters

The [SwitchWorld](https://github.com/yizhiqianbi/SwitchWorld) integration is a
source-pinned, runtime-fetch research workflow for in-stream first-person and
third-person viewpoint switching on LingBot-World. It is not a robot policy,
calibrated-camera system, benchmark result, or real-time service.

Run [switchworld-lingbot-viewpoint-switch.yaml](../../workflows/testing/switchworld-lingbot-viewpoint-switch.yaml)
only after staging a genuine native SwitchWorld case bundle under the selected
run prefix. The bundle has a source image, held-out target MP4, native latent,
condition and prompt-context tensors, and `npa.switchworld.controls.v1`.
Preparation fully decodes the target, verifies that the condition tensor agrees
with its view schedule and switch index, and extracts actual before/after-switch
context frames. It does not invent controls or frames.

The controls sidecar has one entry per native latent frame (15 for this pinned
entry point): `view_ids`, a `switch_frame`, 4×4 `camera_poses`, and four-value
`intrinsics`. Preparation checks these values and the strict LingBot v3 tensor
shapes against the native condition and the case's named prompt context before
any GPU generation is scheduled.

## Provenance and attribution

| Component | Pinned identity | Credit and license | Delivery |
| --- | --- | --- | --- |
| SwitchWorld code | [`yizhiqianbi/SwitchWorld@44d21478f2c15dcccf6423983cb78beee695db1c`](https://github.com/yizhiqianbi/SwitchWorld/tree/44d21478f2c15dcccf6423983cb78beee695db1c) | SwitchWorld contributors / yizhiqianbi, Apache-2.0 [LICENSE](https://github.com/yizhiqianbi/SwitchWorld/blob/44d21478f2c15dcccf6423983cb78beee695db1c/LICENSE), [NOTICE](https://github.com/yizhiqianbi/SwitchWorld/blob/44d21478f2c15dcccf6423983cb78beee695db1c/NOTICE), copyright 2026 SwitchWorld contributors | Runtime checkout |
| Canonical adapters | [`PencilHu/SwitchWorld@5a01361ae1f9c9b1cfe115bef1c4d0377d922c1f`](https://huggingface.co/PencilHu/SwitchWorld/tree/5a01361ae1f9c9b1cfe115bef1c4d0377d922c1f) | PencilHu, Apache-2.0 model card | Runtime fetch, SHA-256 and size checked |
| Verified alias, not executed | [`wangmingxinthu/SwitchWorld@c5b87e32d1f327f040db0d8d1ebf84d6a050ea5e`](https://huggingface.co/wangmingxinthu/SwitchWorld/tree/c5b87e32d1f327f040db0d8d1ebf84d6a050ea5e) | wangmingxinthu, Apache-2.0 model card | Never fetched by this workflow |
| Base model | [`robbyant/lingbot-world-base-cam@6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd`](https://huggingface.co/robbyant/lingbot-world-base-cam/tree/6fc824ffc338d64c97c77e2eb8c0f4cfc24d82bd) | Robbyant Team / LingBot-World, Apache-2.0 card | Existing LingBot runtime fetch |
| LingBot code | [`Robbyant/lingbot-world@a43bec7f8091c83e9b30b16b912f6fc906236fa6`](https://github.com/Robbyant/lingbot-world/tree/a43bec7f8091c83e9b30b16b912f6fc906236fa6) | Robbyant Team, Apache-2.0 retained by the NPA image | Existing immutable image |
| Wan dependency | [`Wan-Video/Wan2.2@42bf4cfaa384bc21833865abc2f9e6c0e67233dc`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc) | Wan team, Apache-2.0; NPA retains notices and TI2V patch record | Existing LingBot runtime |
| Tokenizer | [`google/umt5-xxl@66cb9e7e85526fe440a945569e42c72fb6cbc0ad`](https://huggingface.co/google/umt5-xxl/tree/66cb9e7e85526fe440a945569e42c72fb6cbc0ad) | Google, Apache-2.0 card | Existing LingBot runtime fetch |
| Case data | No released SwitchWorld case payload is selected | Operator owns the rights-cleared native case; the upstream [data note](https://github.com/yizhiqianbi/SwitchWorld/blob/44d21478f2c15dcccf6423983cb78beee695db1c/README.md#data) says raw recordings and cached latents are not committed | Run-scoped input only |

The current SwitchWorld repository commit changes its download link to
`PencilHu/SwitchWorld`, which makes it canonical instead of the older card URL
in `pyproject.toml`. The two HF releases have identical `release_manifest.json`
and all six LFS payloads have matching SHA-256 objects and sizes. The canonical
joint files are `joint_high_rank128.pt` (`a5a88105…1e1b743`, 6,533,976,443
bytes) and `joint_low_rank128.pt` (`8bb55401…0d68aee`, 6,533,975,921 bytes).
Matching bytes do not transfer authorship or provenance from the source-directed
PencilHu release to the alias.

NPA modifications are limited to source/adaptor identity checks, S3 artifact
contracts, actual media metrics, paired MP4/RRD output, and provenance. NPA does
not represent SwitchWorld, LingBot-World, or Wan research as original work.
The inspected source commit and both adapter cards supply neither a
`CITATION.cff` nor a BibTeX citation block, so this integration preserves that
absence rather than inventing a citation. Existing LingBot-World and Wan
attribution stays in the immutable runtime's retained notices; no new image is
built here, so no SwitchWorld OCI-label claim is made.

## Legal delivery decision

The selected SwitchWorld code and adapter cards declare Apache-2.0, but base
models, CUDA/PyTorch runtime packages, case data, inputs, and generated outputs
remain separately governed. The existing `npa-lingbot-world` image contains no
weights, CUDA Python distributions, user media, credentials, or term
acceptance; it fetches runtime dependencies into an operator-owned cache.

No SwitchWorld-specific NPA EULA, `ACCEPT_*` variable, duplicate attestation,
or telemetry consent is added. The inherited Wan runtime keeps only its
documented fetch-bound package terms. The canonical adapters, base model, and
tokenizer were public and ungated at their pinned revisions during inspection.
Technical payload access does not establish a new redistribution or service
right. The workflow never bakes the 39 GB adapter release, model cache, input
case, target media, credentials, or outputs, and adds no public container row.

## Real stages and measurements

1. `prepare-real-case` decodes the target and validates its native controls.
2. `generate-lingbot-baseline` executes upstream
   `infer_lingbot_switchmolora_case.py` without adapters.
3. `generate-adapter-switch` executes the same upstream entry point with the
   two hash-pinned canonical joint adapters.
4. `measure-real-frames` invokes upstream `evaluate_video_pair.py` twice on
   decoded frames, reporting target/adapted and baseline/adapted PSNR, SSIM,
   MAE, plus the declared switch window.
5. `emit-paired-artifacts` decodes all source videos, emits two paired H.264
   MP4s and a Rerun recording of real pixels, then runs `rerun rrd verify`.
   It also inspects the RRD entities after writing; its recording identity is
   bound to the NPA workflow run ID.

The card's `evaluation/metrics.py` lists switch fidelity, cross-view geometry,
identity, action-following, off-screen accuracy, and FVD hooks, but each uses a
mock implementation or injected mock dependency. This workflow excludes them.
The retained PSNR/SSIM/MAE are factual pixel comparisons, not semantic,
geometry, identity, action, benchmark, or robot-success scores.

The upstream entry point selects `cuda:0`, so the workflow requests one B200.
That is distinct from the separately qualified four-B200 LingBot camera
workflow and is not a single-B200 LingBot acceptance claim.

## Validation state

Local YAML/render validation and real H.264 pairing, decoding, RRD writing, and
RRD verification tests have passed. A real GPU workflow remains unverified
until an operator provides a genuine rights-cleared native case bundle and the
selected configuration passes access, storage, pull, GPU-placement, full run,
and artifact readback checks. It must not be called a benchmark, convergence, or
physical-robot result before then.
