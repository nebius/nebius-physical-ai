# Generate twelve appearance videos with preserved padding

[Main workflows](../../../workflows/main/README.md) ·
[Workflow YAML](../../../workflows/main/paidf-cosmos3.yaml)

The [twelve-profile recipe](../examples/paidf-appearance-12.yaml) supplies a
complete, opt-in configuration for `workflows/main/paidf-cosmos3.yaml`. It
generates twelve separate videos from one source using oak, walnut, aluminum,
stainless steel, granite, terracotta, rubber, cork, ceramic, green laminate,
warm side lighting and cool twilight lighting. Existing workflow defaults remain
unchanged.

Both features are on `main`: [#907](https://github.com/nebius/nebius-physical-ai/pull/907)
adds the twelve-profile recipe; [#908](https://github.com/nebius/nebius-physical-ai/pull/908)
adds automatic padding detection, preservation and scene-only scoring.

| What you want | What to do |
| --- | --- |
| Twelve requested appearances | Apply the complete recipe below; increasing `variant_count` alone does not add profiles. |
| Unchanged black borders | Use the current NPA code and a fresh run. Padding handling is automatic; there is no padding YAML switch. |
| Separate videos | Open each `augmented_video.mp4`. No montage step is required. |
| More than twelve variants | Increase `variant_count`; the twelve profiles repeat with new generation seeds. Add profiles to request new looks. |

The profiles edit an **existing stationary work surface** while preserving
robot parts, task objects, geometry, contacts and timing. Select a source where
that surface is visible. These are appearance instructions, not guarantees that
the model preserves every task detail. For a scene without a visible work
surface, define suitable profiles instead of asking the model to invent one.

## Apply the recipe

Complete the [one-time operator setup](../../../workflows/guides/paidf-cosmos3.md#before-you-start)
first: a Linux operator host, verified GPU cluster, project storage, Hugging Face
model access, Token Factory access and the `nebius` AWS profile for downloads.
Run the following steps from the repository root in the same Bash session on
that Linux host. The commands use `uv` and `npa/.venv`; see
[installation](../../install.md) if `uv` is missing. Keep private input paths,
configuration and outputs outside Git.

### 1. Update the code used by the workers

Finish or recover active runs in their original environment before upgrading.
Preserve local edits, then update the checkout and its editable installation:

```bash
git switch main
git pull --ff-only origin main
uv venv --python 3.12 --allow-existing npa/.venv
uv pip install --python npa/.venv/bin/python --upgrade -e ./npa
source npa/.venv/bin/activate
git rev-parse HEAD
npa/.venv/bin/python -c 'import npa; from npa.workflows.video_padding_detection import detect_source_padding; print(npa.__file__)'
unset NPA_SRC_S3_URI NPA_E2E_NPA_SRC_S3_URI NPA_SRC_OVERLAY
```

The import must point into this checkout's `npa/src/npa`. A package version of
`0.1.0` alone does not establish that the update is installed. The canonical
workflow already sets `source_overlay: true`; normal submission stages the
current checkout's NPA code for the workers inside the pinned images. Clearing
old source overrides prevents accidentally selecting earlier code. Keep
automatic source staging enabled; no manual image rebuild or bucket resync is
needed for these NPA changes. Start a fresh run below; resuming an older run
keeps its original source/configuration and does not upgrade its videos.

### 2. Select your MP4 and existing environment

Place the MP4 on the Linux operator host and use its absolute path. Replace
each placeholder with the values from your setup:

```bash
export PROJECT_ALIAS='<your-npa-project-alias>'
export NPA_NEBIUS_PROFILE='<your-verified-nebius-profile>'
export KUBE_CONTEXT='<your-verified-context>'
export KUBECONFIG='<your-verified-kubeconfig-path>'
export BUCKET='<your-configured-bucket-name>'
export NPA_WORKFLOW_GPU_ACCELERATOR='<discovered-gpu-name>:1'
export NPA_SKYPILOT_BIN="$(npa/.venv/bin/npa skypilot status --bin-path)"
INPUT_VIDEO='/absolute/path/source.mp4'

npa/.venv/bin/npa workbench token-factory models
```

Select an available vision model from that list, then check the complete input:

```bash
export CAPTION_MODEL='<available-vision-model-id>'
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate:format=duration \
  -of json "$INPUT_VIDEO"
ffmpeg -v error -xerror -i "$INPUT_VIDEO" -map 0:v:0 -f null -
```

Both media checks must succeed. Use an H.264 MP4 with a visible stationary work
surface for these profiles. Keep source bars in the input: the pipeline detects
top/bottom, left/right and all-around black padding, including embedded bars.
It preserves known normalization padding and only adds detected bands when the
complete clip provides consistent evidence.

### 3. Create a private workflow with all twelve profiles

The recipe file is a **config overlay**, not a standalone workflow. This command
copies the canonical workflow and merges the entire recipe, including its
`appearance_profiles_json`, prompts and quality settings:

```bash
umask 077
PRIVATE_RUN_DIR="$(mktemp -d "${TMPDIR:-/tmp}/paidf-appearance-12.XXXXXX")"
PRIVATE_SPEC="$PRIVATE_RUN_DIR/paidf-cosmos3.yaml"
npa/.venv/bin/python - "$PRIVATE_SPEC" <<'PYTHON'
import json
import sys
from pathlib import Path
import yaml

workflow = yaml.safe_load(Path("workflows/main/paidf-cosmos3.yaml").read_text())
recipe = yaml.safe_load(
    Path("docs/workbench/examples/paidf-appearance-12.yaml").read_text()
)
workflow["config"].update(recipe["config"])
Path(sys.argv[1]).write_text(yaml.safe_dump(workflow, sort_keys=False))
profiles = json.loads(workflow["config"]["appearance_profiles_json"])
print(f"Prepared {len(profiles)} profiles; variant_count={workflow['config']['variant_count']}")
print(f"Private workflow: {sys.argv[1]}")
PYTHON
SPEC="$PRIVATE_SPEC"
```

Expect `Prepared 12 profiles; variant_count=12`. Keep this private spec and its
run records. The recipe uses one GPU sequentially, guidance `5.0`, `35` steps,
RGB conditioning `0.25`, CFG normalization enabled, and quality thresholds
`0.75` / `1.0`. The full YAML is the source of truth; avoid adding old `--var`
overrides that replace its count, profiles or thresholds. To customize a look,
edit the private copy and keep `appearance_profiles_json` as a JSON string with
exactly `lighting`, `background`, `color_grade` and `surface_finish` per profile.

### 4. Reserve, check and submit a fresh run

```bash
RUN_ID="$(npa/.venv/bin/npa workbench workflow prepare-run "$SPEC" --project "$PROJECT_ALIAS")"
export NPA_SKYPILOT_ISOLATED_CONFIG_DIR="$HOME/.npa/workflow-runs/$RUN_ID/skypilot"
RUN_URI="s3://$BUCKET/paidf-cosmos3/$RUN_ID"
npa/.venv/bin/npa workbench workflow validate-spec "$SPEC" --json
npa/.venv/bin/npa workbench workflow plan-spec "$SPEC" \
  --run-id "$RUN_ID" --assume-decision promote_checkpoint --check-render \
  --var bucket="$BUCKET" --var caption_model="$CAPTION_MODEL" --json
npa/.venv/bin/npa workbench workflow preflight-images "$SPEC" \
  --assume-decision promote_checkpoint --project "$PROJECT_ALIAS"
```

Continue only when these checks succeed. The assumed decision is for the plan
and image check; the executing command below follows actual evaluator results:

```bash
npa/.venv/bin/npa workbench workflow submit "$SPEC" \
  --run-id "$RUN_ID" \
  --input-video "$INPUT_VIDEO" \
  --var bucket="$BUCKET" \
  --var caption_model="$CAPTION_MODEL" \
  --runtime --max-wait-seconds 0 \
  --infra "k8s/$KUBE_CONTEXT" \
  --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID \
  --secret-env AWS_SECRET_ACCESS_KEY \
  --secret-env HF_TOKEN \
  --project "$PROJECT_ALIAS" --durable-s3
```

Keep submit running until the workflow is terminal. `--max-wait-seconds 0`
removes the CLI's default one-hour stage deadline; twelve sequential variants
can exceed it. This is a submit option, not a YAML key. In another terminal,
restore this run's environment and check progress:

```bash
npa/.venv/bin/npa workbench workflow status "$RUN_ID" --project "$PROJECT_ALIAS"
npa/.venv/bin/npa workbench workflow logs "$RUN_ID" --project "$PROJECT_ALIAS" \
  --stage generate-variants --no-follow
```

Use the setup guide for [monitoring or recovery](../../../workflows/guides/paidf-cosmos3.md#r4-monitor-and-recover)
and [cleanup of this run's resources](../../../workflows/guides/paidf-cosmos3.md#r7-finish-owned-cleanup).
For a LeRobot input, keep this private spec and use the
[explicit episode/camera submission](../../../workflows/guides/paidf-cosmos3.md#r3a-augment-one-lerobot-episode-and-camera)
in place of `--input-video`.

### 5. Download the videos and verify padding preservation

Wait for the run to become terminal before collecting final evidence; a
refinement pass can replace the latest variants. These commands use the default
run prefix above and the `nebius` AWS profile configured during setup:

```bash
EVIDENCE_DIR="$PRIVATE_RUN_DIR/evidence"
mkdir -p "$EVIDENCE_DIR/cosmos_augmented"
aws s3 cp "$RUN_URI/input/source.mp4" "$EVIDENCE_DIR/source.mp4" --profile nebius
aws s3 cp "$RUN_URI/cosmos_augmented/manifest.json" "$EVIDENCE_DIR/manifest.json" \
  --profile nebius
aws s3 cp "$RUN_URI/cosmos_augmented/" "$EVIDENCE_DIR/cosmos_augmented/" \
  --recursive --exclude '*' --include 'variant-*/augmented_video.mp4' \
  --include 'variant-*/metadata.json' --include 'variant-*/raw_model_video.mp4' \
  --include 'variant-*/raw_model_metadata.json' --profile nebius
aws s3 cp "$RUN_URI/grade/cosmos_evaluator.json" "$EVIDENCE_DIR/cosmos_evaluator.json" \
  --profile nebius
jq '{status, variant_count, published: (.variants | length)}' "$EVIDENCE_DIR/manifest.json"
jq '{clip, profile: .variables, bounds: .source_content_region.bounds,
     detection: .source_content_region.padding_detection.status,
     preservation: (.padding_preservation | if . == null then null else
       {status, scene_pixels_unchanged, padding_matches_source} end)}' \
  "$EVIDENCE_DIR"/cosmos_augmented/variant-*/metadata.json
jq '{status, passed, score, clip_count, passed_clips,
     padding: [.clips[] | {clip_id, padding: .spatial_evidence.padding}]}' \
  "$EVIDENCE_DIR/cosmos_evaluator.json"
```

A completed generation manifest should report twelve published variants. For a
source with verified padding, each metadata record should show preservation `status: verified`,
`scene_pixels_unchanged: true` and `padding_matches_source: true`. The evaluator's
padding diagnostic should show zero RGB error. `detection: detected` identifies
additional embedded bars; `no-additional-padding` can still have preservation
when preparation added bars. For a full-frame source without verified padding,
no preservation receipt or extra raw file is needed.

**Open `augmented_video.mp4`** for the corrected result. `raw_model_video.mp4`
is the retained model output and can still show augmented borders. Missing
`source_content_region` in new output suggests older NPA code ran: recheck step
1, source overrides and the selected run. Existing downloaded videos or HTML
proofs do not change when you update the checkout. A failed/missing generation
or evaluation report requires checking stage logs before treating the evidence
as complete. Padding preservation does not imply an accepted quality gate.

## Count, concurrency and outputs

`variant_count=12` selects all twelve profiles once. They are shuffled using
`augmentation_seed=30`, so output order differs from file order. Read each
variant's `metadata.json` to identify its profile. Raising the count above
twelve repeats profiles with new generation seeds; it does not define new looks.
Twelve unique profiles define requested looks; verify the rendered material in
each output, since the model can miss a material instruction.

`variant_parallelism=1` runs the variants sequentially on one GPU. Increasing
that setting requires a generation job with multiple visible GPUs. It does not
allocate GPUs or spread variants across independent single-GPU nodes.

Each generated video is published separately beneath the run prefix:

```text
cosmos_augmented/variant-0000/augmented_video.mp4
...
cosmos_augmented/variant-0011/augmented_video.mp4
```

The generation manifest and progress report account for completed and failed
variants. Retain videos and evidence from rejected attempts. A final workflow
rejection does not turn generated candidates into accepted dataset entries.

## Review the evidence

The recipe requires timestamp alignment, grade threshold 0.75 and appearance
attribute threshold 1.0. Temporal and appearance-fidelity diagnostics retain the
canonical advisory policy; show their results alongside the gate decision.
Every generated clip must pass the gate. A high average score does not override
a failed profile. Increasing the count adds more clips that must pass.
An accepted gate does not certify motion, contacts or training suitability.

New Cosmos 3 runs record the scene rectangle in `source_content_region` during
reference preparation. The pipeline also inspects every source frame for
persistent paired black bands at opposite edges, including bars embedded in
the source. Detection requires near-black bands, roughly symmetric widths and
a clear brightness step into the scene. It abstains on ambiguous, one-sided,
changing or insufficiently observed borders. Known normalization padding stays
protected even when embedded-bar detection abstains. Interior dark objects are
not padding. No additional YAML setting is needed.

Before publication, the pipeline restores those borders from the source and
encodes a lossless RGB H.264 MP4. Every decoded generated scene pixel and source
border pixel must match exactly, with unchanged frame count and timing. These
MP4s are larger than lossy model outputs. `raw_model_video.mp4` and
`raw_model_metadata.json` preserve the unmodified model evidence; the published
`augmented_video.mp4`, extracted frames and alignment hashes describe the
corrected output. `metadata.json.padding_preservation` records both video
hashes, scene/border pixel hashes and the verified policy. This fixes border
artifacts; it does not repair defects within the generated scene.

The evaluator verifies the source hash and complete video alignment, then uses lossless scene crops for all four checks. Artificial
letterbox or pillarbox pixels cannot lower the scene score or satisfy a
requested appearance change. Real black objects remain in the scene. Explicit
metric regions keep their full-canvas coordinates and are intersected with the
scene; a padding-only region is an error.

Each clip's `spatial_evidence` reports the full-video and crop hashes, rectangle,
excluded pixel fraction and a separate padding diagnostic. The diagnostic
reports mean RGB error and the fraction of padding pixels differing from the
source by more than 8 RGB levels in any channel. It is advisory, has no quality
pass threshold, and does not change the scene score. Successfully preserved
borders have zero RGB error against the reference. The raw output remains
available to inspect what the model originally generated.

Older variants without region provenance retain full-frame scoring. Start a
fresh run to detect and preserve borders automatically; existing stored videos
are not rewritten. Invalid or stale provenance fails instead of silently
selecting another crop.

Compare source/output frames at contacts, generation-window joins and the final
state. Check that the requested material or lighting is visibly present inside
the original camera area, and inspect added padding separately. Preserve raw
outputs when making any presentation crop. Use the
[realism review procedure](paidf-realistic-augmentation.md#review-and-tune-the-actual-outputs)
and [public LeRobot comparisons](paidf-lerobot-realism.md) to assess limitations.

Validate the unchanged recipe on independent sources before making broader
claims. Source-specific prompts, output media and exact infrastructure evidence
must remain in private storage for private recordings.

The retained-run audit checks workflow completion, all twelve profiles,
complete video decoding, source/output alignment, native control receipts and
evaluator accounting. It verifies that the evaluator's video hashes match the
retained outputs, so an earlier pass cannot stand in for the final results.
Create a private JSON array with `run_uri` and `expected_frames` for each proof,
then run:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_PROJECT="$PROJECT_ALIAS" \
NPA_PAIDF_APPEARANCE_CASES="$PRIVATE_RUN_DIR/proof-cases.json" \
  npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_paidf_appearance_recipe_live.py -q
```

This audit makes read-only storage calls and downloads actual videos. It does
not generate more candidates or reinterpret a rejection as acceptance.
The canonical workflow deliberately finishes a rejected batch with a failed
`reject-quality` terminal stage after preserving review evidence. The audit
requires that exact rejection path, a valid rejected disposition and successful
preceding stages; other workflow failures do not qualify.

To audit padding-aware scoring on retained outputs, use the same private case
file with `NPA_PAIDF_PADDING_CASES`, set `NPA_PAIDF_PADDING_EVIDENCE_DIR` to a
private local directory, and run
`npa/tests/e2e/test_paidf_padding_evaluation_live.py`. This audit downloads the
videos and makes real Token Factory calls. It requires the same hallucination
engine as the retained grades; for production Cosmos Evaluator runs, use its
pinned NVIDIA checkout and dependencies (`NPA_COSMOS_EVALUATOR_SRC`). Legacy
rectangles are reconstructed only from the retained normalization recipe and
the hash-verified original and prepared videos. Reports are written locally;
retained videos and original grades are not overwritten. The audit also measures
a fresh full-frame hallucination baseline with the same engine to isolate the
effect of padding removal on that metric.

To test source detection and border preservation on retained real outputs, use
`NPA_PAIDF_PADDING_CASES` with the same private case file and set
`NPA_PAIDF_PADDING_PRESERVATION_DIR` to a new private local directory, then run
`npa/tests/e2e/test_paidf_padding_preservation_live.py` with the integration and
project variables above. This makes read-only storage calls, detects borders
without relying on old preparation metadata, and checks every decoded scene
and border pixel in each corrected video. It neither generates new model
outputs nor reruns the VLM.

## Public reference measurement

The unchanged recipe was run on episode 0, camera
`observation.images.cam_high`, seconds 0–8 of
[`lerobot/aloha_static_cups_open`](https://huggingface.co/datasets/lerobot/aloha_static_cups_open/tree/d793c969cf716001dcca18a0842c3d7e9de9e41b).
The prepared reference and each output contain 192 frames at 24 fps. Both
generation passes produced all twelve separate videos, with full decoding and
timestamp alignment verified. Results recorded on October 7–8, 2026 using the
original full-frame evaluator:

| Generation pass | Guidance / steps | Mean score | Clips passing gate | Temporal advisory passes | Appearance advisory passes | Batch decision |
|---|---|---|---|---|---|---|
| Initial | 5.0 / 35 | 0.722930 | 2/12 | 0/12 | 3/12 | Rejected |
| Canonical refinement | 4.5 / 39 | 0.708905 | 2/12 | 0/12 | 3/12 | Rejected |

This demonstrates twelve-profile execution, not an accepted training dataset.
Refinement did not improve this batch's measured result. Matched-frame review
found visible lighting and material variation, but also incomplete surface
coverage, artificial material boundaries, color spill onto robot parts and
changes to small task details. More profiles increase requested diversity;
they do not guarantee realism, preservation or a higher acceptance rate.
