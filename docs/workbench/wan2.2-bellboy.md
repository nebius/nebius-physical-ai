# Wan 2.2 TI2V-5B BYOF baseline and Bellboy boundary

This integration packages the official Alibaba Wan 2.2 source as a BYOF
registry candidate and runs a real stock TI2V-5B video generation. It also
defines a conservative episode manifest for Bellboy's real hotel-robot data and
an explicit extension point for Bellboy's private action-conditioned fork.

The public baseline is a generative video model. It is **not** an
action-conditioned robotics simulator, does not train on the episode manifest,
does not predict robot actions, and does not replace evaluation on held-out
real robot episodes.

## Pinned upstream inputs

| Input | Immutable revision | Packaging |
| --- | --- | --- |
| Official source | [`Wan-Video/Wan2.2` `42bf4cf…`](https://github.com/Wan-Video/Wan2.2/tree/42bf4cfaa384bc21833865abc2f9e6c0e67233dc) | cloned into the BYOF image |
| Official TI2V-5B checkpoint | [`Wan-AI/Wan2.2-TI2V-5B` `921dbaf…`](https://huggingface.co/Wan-AI/Wan2.2-TI2V-5B/tree/921dbaf3f1674a56f47e83fb80a34bac8a8f203e) | fetched at run time; never baked |
| UMT5 tokenizer | [`google/umt5-xxl` `66cb9e7…`](https://huggingface.co/google/umt5-xxl/tree/66cb9e7e85526fe440a945569e42c72fb6cbc0ad) | tokenizer files fetched at run time |

The official Wan README documents TI2V-5B as one model for text-to-video and
image-to-video at 1280x704, 24 fps, including a single-GPU offload path. The
hard gate here executes the native `wan.WanTI2V` implementation at that spatial
size with 17 frames and 8 sampling steps. Reducing temporal length and sampling
steps makes this a capability smoke; it does not claim production visual
quality or the official five-second performance result.

The official source exposes no Wan TI2V training entrypoint in the pinned tree.
This integration therefore does not manufacture a training stage from an
unrelated implementation.

## Workflows

- `byof-wan2.2.yaml` is the standalone solution candidate. It builds the pinned
  source, fetches pinned model inputs at run time, generates and decodes an MP4,
  and uploads all smoke outputs through the existing BYOF S3 path.
- `bellboy-wan2.2-e2e.yaml` composes the real dataset-of-record validator,
  Bellboy-specific reference validation, the same Wan BYOF workload, and a
  held-out evaluation-boundary report.

Validate and plan without provisioning infrastructure:

```bash
npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/byof-wan2.2.yaml \
  --run-id wan22-plan

npa/.venv/bin/npa workbench workflow validate-spec \
  npa/workflows/workbench/npa-workflows/bellboy-wan2.2-e2e.yaml
npa/.venv/bin/npa workbench workflow plan-spec \
  npa/workflows/workbench/npa-workflows/bellboy-wan2.2-e2e.yaml \
  --run-id bellboy-wan22-plan
```

For a real run, copy the spec to an operator-owned location or use workflow
variables to replace `bucket`, `manifest_uri`, and `heldout_manifest_uri`. The
`example-bucket` values are documentation templates, not deployed identifiers.
The normal NPA config supplies project, registry, Kubernetes, and object-storage
credentials; do not write those values into the workflow.

The default declaration is T2V. To exercise the optional image input, change
the contract as one atomic override: set `context_image_uri`, set
`capability_name` to `wan2.2_ti2v_5b_image_to_video`, set
`smoke_artifact_name` to `wan2_2_ti2v_5b_image_to_video.json`, and point the
workflow's declared artifact URI (`artifact_uri` in the standalone spec or
`wan_artifact_uri` in the Bellboy spec) at that filename. The smoke compares
its actual input mode to those declarations and fails on a T2V/I2V mismatch.
This keeps the optional real I2V path usable without silently promoting it; it
remains deferred in the catalog until a separately recorded live run passes.

## GPU and runtime contract

The checked-in route requests exactly one Kubernetes `H100` with at least 16
CPU, 128 GiB host memory, and 200 GB scratch disk for the image, pinned
checkpoint cache, and artifacts. The smoke fails closed if a different GPU
product is scheduled. This is the low-risk Hopper path:

- PyTorch 2.7.1 CUDA 12.8 wheels are pinned.
- FlashAttention is not installed. The pinned native Wan attention module has a
  PyTorch scaled-dot-product-attention fallback, which is recorded in the
  evidence artifact.
- No Blackwell/SM120 compatibility claim is made. In particular, the existing
  one-GPU RTX PRO 6000 solution profile is not used.
- The job has no artificial terminal wait deadline. Checkpoint acquisition and
  generation are allowed to reach their natural terminal state.

`WAN22_CACHE_DIR` defaults to `/workspace/model-cache/wan2.2`. Point it at a
persistent mounted cache when repeat runs should avoid another checkpoint
download. `HF_TOKEN` is optional for these public assets and is read only from
the run environment when present.

## Episode input contract

The JSON schema is
[`bellboy-episode-manifest-v1.schema.json`](bellboy-episode-manifest-v1.schema.json).
It stores references, not customer data. Every URI below is an object URI in
Bellboy's S3-compatible layer.

```json
{
  "schema": "npa.bellboy.episode_manifest.v1",
  "dataset_id": "hotel-robot-episodes",
  "version": "2026-08-05",
  "quality_stats": {
    "record_count": 1,
    "modalities": ["gripper_rgb"],
    "events": ["open-door"],
    "locations": ["hotel-room"],
    "mean_completeness": 1.0,
    "corrupt_count": 0,
    "per_modality_counts": {"gripper_rgb": 1}
  },
  "camera": {
    "modality": "rgb",
    "mount": "gripper",
    "projection": "very-wide-angle"
  },
  "action_schema": {
    "uri": "s3://example-bucket/contracts/actions-v1.json",
    "version": "bellboy-actions-v1"
  },
  "episodes": [
    {
      "episode_id": "episode-000001",
      "split": "train",
      "task": "open-door",
      "outcome": "failure",
      "observation": {
        "gripper_rgb_uri": "s3://example-bucket/episodes/000001/gripper.mp4",
        "timestamps_uri": "s3://example-bucket/episodes/000001/rgb-time.jsonl"
      },
      "actions": {
        "uri": "s3://example-bucket/episodes/000001/actions.parquet",
        "timestamps_uri": "s3://example-bucket/episodes/000001/action-time.jsonl"
      },
      "joint_state": {
        "uri": "s3://example-bucket/episodes/000001/joints.parquet",
        "timestamps_uri": "s3://example-bucket/episodes/000001/joint-time.jsonl"
      },
      "timing": {
        "clock": "monotonic-nanoseconds",
        "start_ns": 0,
        "end_ns": 123456789
      },
      "recovery": {
        "parent_episode_id": "episode-000000",
        "attempt": 2,
        "correction": "regrasped the handle after the failed pull"
      }
    }
  ],
  "records": [
    {
      "record_id": "episode-000001-gripper-rgb",
      "modality": "gripper_rgb",
      "uri": "s3://example-bucket/episodes/000001/gripper.mp4"
    }
  ]
}
```

`records` is the canonical projection consumed by
`workbench.dataset.validate`. The Bellboy validator additionally requires the
episode-level task, outcome, timing, retry, RGB, action, and joint-state
references and refuses to proceed unless the generic validation report passed
for that exact manifest URI. It checks reference shape and split isolation; it
deliberately does not download private objects or assert sample-level alignment.
A customer data adapter must verify timestamp tolerances against the exact
action schema before private model training.

Representative tasks may include doors, closets, drawers, lights, towels,
water, bed stripping, room inspection, and semantic navigation. `outcome`
preserves successes, failures, partial runs, and aborted runs, while `retry`
links corrective attempts so failure data remains usable in a later RL loop.

The held-out manifest uses the same schema, but every episode must have
`split: heldout`. It is read only by the boundary stage and must not be used for
model or prompt selection. Conversely, the workflow requires every episode in
`manifest_uri` to have `split: train` and rejects reuse of that URI as the
held-out manifest.

The public interchange contract deliberately standardizes on S3 object URIs,
matching the primary data layer. If an authorized episode originates on
Hugging Face, stage it into the operator-controlled S3 dataset-of-record and
preserve its Hugging Face revision in an additional provenance field; do not
put access tokens or mutable download URLs in the manifest.

## Output and viewer contract

The standalone run publishes under:

```text
s3://<bucket>/oss-solutions/wan2.2/<run-id>/
  npa_byof_summary.json
  wan2_2_ti2v_5b_text_to_video.json
  wan2_2_runtime_inventory.json
  wan2_2_ti2v_5b.mp4
  smoke.log
```

The primary JSON records source/model/tokenizer refs, task, prompt and seed,
requested and observed dimensions/frame count/fps, exact GPU topology, runtime
versions, file size, content-variation statistics, exercised capabilities, and
deferred items. The smoke decodes every MP4 frame with OpenCV and rejects an
unopenable, empty, shape-changing, wrong-sized, wrong-frame-count, invalid-fps,
implausibly small, blank, spatially uniform, or frame-identical result.

`wan2_2_runtime_inventory.json` is collected from inside the pulled image
before model acquisition. It lists baked Python and OS package versions,
available Python license metadata/classifiers, the executing uid and venv
access check, and the result of a fail-closed scan for large checkpoint-shaped
files under `/opt/byof`. It separately records the run-time model/tokenizer
identities and the fact that customer data is not baked.

The BYOF runner uploads every file in `$NPA_SMOKE_OUTPUT_DIR`. The NPA agent
artifact browser already classifies `.mp4` as video, so selecting
`wan2_2_ti2v_5b.mp4` opens the existing video viewer. No Rerun recording is
added because a single video has no synchronized comparison stream that would
benefit from it.

The Bellboy workflow additionally emits:

- `episode-validation.json` (`npa.bellboy.episode_validation.v1`)
- `heldout-boundary.json` (`npa.bellboy.wan_evaluation_boundary.v1`)

The boundary report confirms real Wan artifact validation and held-out manifest
isolation. Its `release_gate.satisfied` remains false until a customer evaluator
produces real action/task metrics.

## Capability and extension boundary

| Capability | Status | Evidence or blocker |
| --- | --- | --- |
| TI2V-5B text-to-video | pending live; local contract accepted | real native generation + decoded MP4 smoke is encoded; no live H100 run is recorded yet |
| decoded MP4 validation | pending live; local contract accepted | every frame is decoded and conservative content checks are hard gates |
| TI2V-5B image-to-video | deferred | real optional S3-image code path exists, but has no separate live input/output evidence |
| T2V-A14B / I2V-A14B | deferred | separate much larger models and GPU contracts; not exercised by the 5B image |
| S2V-14B | deferred | separate speech/audio inputs and checkpoint; not exercised |
| Animate-14B | deferred | separate character-animation inputs and checkpoint; not exercised |
| stock Wan fine-tuning | deferred | the pinned official source has no TI2V training entrypoint |
| stock Wan action prediction | rejected | it is not an upstream Wan 2.2 capability |
| Bellboy private action-conditioned fork | deferred customer extension | private repo entrypoint, immutable ref, checkpoint, exact action schema, authorized data access, and evaluator are not supplied |

The current Cosmos3 image fetches a Wan 2.2 VAE at run time for its own Cosmos
path. That reuse is not a full Wan source/checkpoint integration and is not
evidence for any capability in the table above.

### Customer-owned action extension

Bellboy can plug its private implementation into the same `workbench.byof.repo`
path without changing the public baseline. The customer-owned workflow overlay
must supply, through operator configuration rather than this repository:

1. private `repo_url` plus immutable `repo_ref` and repository credentials;
2. a pinned build command and a real smoke driver/entrypoint;
3. an authorized `npa.bellboy.episode_manifest.v1` URI and exact referenced
   action-schema document;
4. a private `checkpoint_uri` plus immutable checkpoint identity;
5. an action-prediction artifact, for example
   `npa.bellboy.action_predictions.v1`, containing episode id, input time range,
   predicted actions/timestamps, model/checkpoint refs, and uncertainty;
6. a held-out evaluator that aligns predictions to executed actions and reports
   task success, recovery/retry behavior, and the customer-agreed action error
   metrics on real episodes.

Only that extension may claim action-conditioned training, future-observation
and action inference, or action evaluation. Its live gate must execute the
private upstream component and write real predictions; a manifest/import smoke
is insufficient.

## Licensing and publication

- Source: the pinned official repository declares Apache-2.0.
- Model: the pinned official Wan-AI checkpoint card declares Apache-2.0. Model
  files are acquired at run time and are not baked into the image.
- Tokenizer: the pinned official Google UMT5 card declares Apache-2.0; tokenizer
  files are acquired separately at run time and are not baked.
- Baked runtime: Ubuntu, CUDA-compatible PyTorch wheels, FFmpeg, Python wheels,
  and the source checkout are present. The live smoke inventories those actual
  installed packages and fails if a large checkpoint-shaped file is found in
  `/opt/byof`; a build-command review alone is not treated as an audit.
- Data and private checkpoints: customer-owned and never redistributed by this
  candidate.

The BYOF image is therefore **not public/registry-accepted yet**. A trusted live
run must publish the runtime inventory, and redistribution review must classify
its package licenses and scan the built image for unexpected Omniverse,
customer, or other restricted payloads before promotion. Dynamic BYOF
candidates do not belong in the first-class Workbench
`packaging-contract.yaml` until they are promoted to a maintained image.

## Failure modes

- A non-H100 GPU fails before model acquisition; use the reviewed Hopper
  profile rather than silently changing the claim.
- Model/tokenizer revision lookup failure stops the run; mutable fallback refs
  are not used.
- Missing S3 credentials or malformed `context_image_uri` stops I2V input
  materialization.
- A corrupt, empty, wrong-shape, wrong-rate, too-small, blank, or uniform MP4
  fails the capability gate and prevents a successful BYOF summary.
- Missing episode fields, non-S3 objects, duplicate ids, invalid outcomes,
  malformed retry links, or split leakage fail the episode contract.
- The public workflow never changes `heldout-boundary.json` into a passing
  action release gate. That requires Bellboy's real evaluator.

## Tests

```bash
npa/.venv/bin/python -m pytest npa/tests/workflows/test_bellboy_wan.py -q
npa/.venv/bin/python -m pytest npa/tests/workflows/test_byof_solution_smokes.py -q
npa/.venv/bin/python -m pytest npa/tests/e2e/test_byof_wan22_live_e2e.py -q
```

The E2E file always performs local spec/render checks. Its live H100 test is
gated by `NPA_INTEGRATION_E2E=1`, `NPA_BYOF_WAN22_LIVE_GPU=1`, normal NPA
operator configuration, S3 credentials, and an explicitly supplied registry
image/build destination. Until that test succeeds and its run/evidence is
recorded, the catalog status remains pending live.
