# Realistic manipulation augmentation with PAIDF Cosmos 3

Use this procedure with [PAIDF Cosmos 3](../../../workflows/guides/paidf-cosmos3.md)
to vary the appearance of a recorded manipulation while preserving the task.
Run the workflow on a supported Nebius GPU through SkyPilot. Inspect the actual
generated video against the complete source before using it as training data.
The starter's exploratory acceptance thresholds establish neither contact
correctness nor task success.

The [LeRobot comparison](paidf-lerobot-realism.md) applies the review to pinned
cup-opening, coffee-preparation and simulated cube-lift episodes, with a matched
native sampling experiment for each task.

## Define the edit before generation

Inspect the selected episode and camera. Name the objects, small task features,
camera motion, contacts and outcome that must remain unchanged in `prompt` and
`augment_subject`. Choose appearance changes that the view can actually show:
room lighting for an overhead camera, or visible work-surface finish for a
wrist camera. A wrist view may contain no background wall to replace.

Set `appearance_profiles_json` to a JSON array of coherent profiles. Each
profile requires exactly four nonempty string fields: `lighting`, `background`,
`color_grade`, and `surface_finish`. Keep the profile realistic as a whole;
avoid combining dim lighting with bright sun or metallic reflections with a
matte finish. Shared values across profiles deliberately hold other attributes
constant. Do not introduce appearance changes that remove task information,
such as recoloring a color-sorting target or hiding battery polarity markings.

An empty string (the default) retains the existing starter sampler. Invalid
profiles fail during validation/planning and before the sampler reads or writes
artifacts. Profiles are shuffled without replacement per cycle, using
`augmentation_seed`. More variants than profiles repeat appearances with
distinct inference seeds; that is additional stochastic sampling, not additional
appearance diversity. Custom profiles cannot be combined with a quality anchor.

The config manifest records the profiles, selected combinations, prompts and
evaluator option table. Evaluation includes the custom attribute values and
the default alternatives. Generation labels source captions as observations,
samples them across the complete episode, and puts the requested edit after
those observations. The source video remains authoritative for motion and
geometry. These instructions improve the specification; they do not enforce
physical correctness in the model output.

The frame extractor samples evenly spaced decoded frame indices across the
complete video, including its first and last frame. It produces up to eight
distinct frames by default, including for short clips, and uses the same method
for source and generated-video captions. Earlier extraction sampled one frame
per second and stopped after eight frames, omitting the ending of longer clips.
Re-extract and re-caption older inputs before testing this change; reusing their
existing frame directory retains that earlier coverage limitation.

Source captioning receives `augment_subject` as task context through
`caption_instruction`. Its default asks the captioner to distinguish visible
evidence from uncertain identities and avoid claiming motion or completion from
a still frame. Override `caption_instruction` for camera-specific terminology;
inspect the captions before treating them as reliable observations. In the
initial battery run, generic captions confused white gripper parts with lights
and missed the insertion task. The shared caption tool keeps its existing
instruction when an older workflow omits this optional configuration.

## Battery insertion example

The [pinned `lerobot/aloha_static_battery` dataset](https://huggingface.co/datasets/lerobot/aloha_static_battery/tree/06dc3da83c4fd3d1889b00f1dfd3780da8421f64)
declares Apache-2.0. Its task is to place a battery into a remote controller's
slot. It has 49 episodes, four cameras, and 50 fps source video. Episode `0`
occupies seconds 0–12 of each camera's shared MP4. Start with
`observation.images.cam_high` to inspect both grippers and the battery/slot;
inspect `observation.images.cam_left_wrist` separately to check close-up behavior.
Keep the source README and metadata with your staged input.

After the setup guide's S1–S8 and R1–R2, download this pinned selection into a new
private directory. The workflow consumes the video and metadata; it does not
require the action parquet for augmentation.

```bash
DATASET_REVISION=06dc3da83c4fd3d1889b00f1dfd3780da8421f64
DATASET_BASE="https://huggingface.co/datasets/lerobot/aloha_static_battery/resolve/$DATASET_REVISION"
LEROBOT_CAMERA=observation.images.cam_high
LEROBOT_DIR="$(mktemp -d "${TMPDIR:-/tmp}/paidf-battery.XXXXXX")"
for file in README.md meta/info.json meta/episodes/chunk-000/file-000.parquet \
  "videos/$LEROBOT_CAMERA/chunk-000/file-000.mp4"; do
  mkdir -p "$(dirname "$LEROBOT_DIR/$file")"
  curl --fail --location "$DATASET_BASE/$file" --output "$LEROBOT_DIR/$file" || exit 1
done
LEROBOT_URI="s3://$BUCKET/paidf-cosmos3/$RUN_ID/datasets/battery"
aws --profile nebius s3 sync "$LEROBOT_DIR/" "$LEROBOT_URI/" --only-show-errors
```

Record downloaded file hashes and the dataset revision in private run evidence.
The following reproduces the earlier **rejected subtle-lighting baseline**. Its
appearance changes were too weak; use it for comparison, not as the recipe to
fan out. The visible-surface experiment below separates material changes from
first-frame conditioning. Both still require task-quality review.

```bash
APPEARANCE_PROFILES='[
  {"lighting":"soft warm indoor illumination",
   "background":"neutral gray work surface with the existing layout",
   "color_grade":"neutral balanced color palette",
   "surface_finish":"matte low-gloss work surface finish"},
  {"lighting":"soft cool diffuse indoor illumination",
   "background":"neutral gray work surface with the existing layout",
   "color_grade":"neutral balanced color palette",
   "surface_finish":"matte low-gloss work surface finish"}
]'
TASK_PROMPT='Preserve both robot arms and grippers with white plastic plates and dark charcoal-gray mechanical parts. Preserve the black cylindrical battery with metallic gold-colored ends, its existing markings and polarity, and the white remote controller with its open battery slot. Keep these object identity colors exactly as in the source. Keep the exact grasp and insertion contacts, camera viewpoint, trajectories, timing, occlusions and final outcome. Apply only the requested subtle lighting variation to existing surfaces.'

QUALITY_ARGS=(
  --var "appearance_profiles_json=$APPEARANCE_PROFILES"
  --var 'augment_subject=two white-and-dark-gray robotic grippers inserting a black cylindrical battery with metallic gold-colored ends into a white remote controller'
  --var "prompt=$TASK_PROMPT"
  --var 'negative_prompt=red battery, orange battery, recolored battery terminals, changed object colors, extra or missing objects, warped grippers, altered battery or slot, floating objects, interpenetration, motion retiming, flicker, ghosting, unsafe content'
  --var augmentation_seed=30 --var seed=17
  --var transfer_edge_threshold=low
  --var transfer_rgb_weight=0.5
  --var control_guidance=1.0 --var guidance=3.0 --var steps=35
  --var grade_threshold=0.75 --var attribute_threshold=1.0
  --var temporal_consistency_mode=required
  --var temporal_consistency_threshold=0.8
)
npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID" \
  --assume-decision promote_checkpoint --var "bucket=$BUCKET" \
  --var input_kind=lerobot --var "lerobot_dataset_uri=$LEROBOT_URI/" \
  --var "input_camera=$LEROBOT_CAMERA" --var input_episode=0 \
  "${QUALITY_ARGS[@]}" --json
npa workbench workflow submit "$SPEC" --run-id "$RUN_ID" \
  --project "$PROJECT_ALIAS" --infra "k8s/$KUBE_CONTEXT" \
  --var "bucket=$BUCKET" \
  --lerobot-uri "$LEROBOT_URI/" --lerobot-camera "$LEROBOT_CAMERA" \
  --lerobot-episode 0 --require-explicit-lerobot-selection \
  --var "caption_model=$CAPTION_MODEL" "${QUALITY_ARGS[@]}" \
  --secret-env HF_TOKEN --secret-env NEBIUS_TOKEN_FACTORY_KEY \
  --secret-env AWS_ACCESS_KEY_ID --secret-env AWS_SECRET_ACCESS_KEY \
  --runtime --durable-s3
```

These gentler sampling settings reduced excessive contrast in the battery
experiment. Adding RGB conditioning improved the measured source-relative
temporal score, but both candidates still failed the unchanged quality gates.
These settings remain an **unqualified starting experiment**, not a
training-quality recommendation. Before the explicit identity-color wording shown
here, the cool candidate still recolored the battery and the warm candidate's
intended appearance change was difficult to verify.
A real trial with
`control_guidance=3.0`, `guidance=3.5`, and `steps=32` produced harsh contrast and
distorted gripper/battery details and was rejected. Higher control guidance did
not establish task preservation. Compare changes on the same episode, appearance
profiles and seeds in separate fresh run IDs, with a separate SkyPilot API
directory per run. When testing different source revisions, also give each run
its own `NPA_CONFIG_DIR`: source staging updates that configuration, and a live
API correctly rejects changes to its bound configuration. Use the same strict
evaluation settings for each. Adjust one
generation control at a time after observing the failure; lowering thresholds
does not improve pixels.

## Review and tune the actual outputs

Keep `structural_control=edge`, `alignment_mode=required`, guardrails enabled,
and `source_motion_weight=0.0`. The source is normalized to 832×480 at 24 fps
without changing duration. This example becomes 288 frames over 12 seconds.
Frame alignment proves coverage and timing, not correct grasp geometry.

Inspect the source control images as well as RGB frames. Set
`transfer_edge_threshold` to a native Canny preset: `very_low` (20/50), `low`
(50/100), `medium` (100/200, unchanged default), `high` (200/300), or `very_high`
(300/400). Lower thresholds retain weaker edges, including dark mechanism and
small-object details; they can also admit noise. The setting reaches the actual
control preprocessor, and `transfer.json` records the preset, algorithm and
decoded-control hash verified by the native loader. It requires
`structural_control=edge`; invalid presets fail before planning or model work.
Compare a lower preset with all other settings fixed before adopting it. On the
battery source, the medium preset omits substantial internal gripper detail;
this observation alone does not qualify a lower preset's generated output.

Edges omit RGB color and surface cues. The optional `transfer_rgb_weight` adds
the complete source RGB video as a second native conditioning input, weighted
relative to edge weight 1. Zero, the default, leaves edge-only behavior unchanged.
A positive value uses the framework's `blur` hint with its `none` preset. The
adapter writes lossless RGB controls and verifies every decoded control pixel
and frame through the native loader; `transfer.json` records this evidence.
This is model conditioning, not source/output pixel blending. Compare it on the
same source, captions, profiles and seeds, checking both fine task details and
the intended appearance change: stronger source conditioning may suppress edits.

`transfer_cfg_normalization=enabled` enables the pinned framework's native
classifier-free guidance normalization during diffusion. The compatibility
default is `disabled`; both are literal strings so workflow argv can carry the
choice without an ambiguous boolean flag. The CLI equivalent is
`--transfer-cfg-normalization enabled` on `cosmos3 generate-variants`. It requires
edge transfer and records the actual `normalize_cfg` boolean in `transfer.json`.
This changes the model's guided prediction; it does not postprocess or blend the
published video. Compare it with normalization disabled using the same source,
captions, profile, seed, guidance, steps and source controls. Inspect contrast,
material appearance and small task features separately; normalization is not a
guarantee of realistic contacts or preserved identity.

The source's **first RGB frame is a separate appearance anchor**. Earlier adapter
versions always supplied it to the first native generation window, even with
`transfer_rgb_weight=0`. Set `transfer_first_chunk_conditional_frames=0` to release
that anchor when testing material or color changes from the beginning of the
clip. The compatibility default is 1; only 0 and 1 are supported by this adapter.
The pinned native framework's transfer default is 0. Full-video edge conditioning
remains active, and later windows still use five generated overlap frames for
continuity. This is a model input setting, not output blending. Releasing the
anchor may also change foreground identity, so compare matched sources, profiles
and seeds and inspect task details. The native transfer receipt records both the
first-window conditioning count and subsequent overlap count.

| Observed failure | Next controlled experiment |
| --- | --- |
| Small battery, gripper or slot geometry drifts | Reject the output. Check whether source edges resolve the feature and whether captions misidentify it. Simplify the appearance edit, then compare one control-guidance change at a time; larger values can still distort details. |
| Appearance hardly changes | Confirm the selected profile is present in `metadata.json`'s effective prompt. Check the raw output against the source. Try one more visible but plausible lighting change; stronger structural control can suppress edits. |
| Lighting changes but object identity colors change too | Name the visible base colors and markings in the task specification, keeping them separate from the requested illumination. Re-caption the source with that context and retain uncertainty about unreadable markings. Review actual pixels; more detailed wording is not an enforcement mechanism. |
| Flicker or discontinuity near generation joins | Inspect `transfer.json` for actual native chunk count and review frames around each join. Compare a larger supported `transfer_chunk_frames` value with the same seeds; it must be `4k+1`, between 9 and 297. Larger windows need more memory and are not guaranteed to improve quality. |
| Evaluation accepts a visibly incorrect insertion | Reject that candidate in the review record. Increase task-specific inspection coverage; attribute checks and image motion diagnostics cannot certify physics. |
| Global appearance-fidelity check rejects intended relighting | Keep it advisory for the relit scene. Require fidelity only for declared invariant regions, with tolerances established against reviewed positive and negative examples. |

Review source/output pairs at the same timestamps, including pre-grasp, pickup,
alignment, contact, insertion, release, final state, and every generation join.
Use slow playback and crops for the battery and grippers; a contact sheet alone
can miss a brief failure. Require unchanged object count, geometry and markings,
source-matched occlusion and contacts, and a clearly visible but plausible
appearance change. Inspect complete clips for temporal stability.

Retain all attempts, rejected candidates and evaluator reports. Record each
variant's effective prompt, seed, guidance, steps, control settings, video hash,
automatic verdict and visual-review findings. Do not select a lucky successful
seed and describe the whole batch as realistic. Verify the selected recipe on
other episodes and a wrist view before scaling; keep evaluation episodes apart
from tuning episodes. Each camera is generated independently, so equal seeds do
not establish synchronized multi-view consistency.

The outputs remain augmented videos, captions, curation artifacts and Rerun
recordings. They are not a newly aligned LeRobot action dataset. Training use
requires separate action/timestamp alignment, multi-view checks where applicable,
and task validation.

## Tune visible changes before fan-out

The runtime has no battery-dataset branch, fixed battery appearance profile, or
special case for episode zero. Input selection, task description, appearance
profiles, sampling settings and variant count come from configuration. The
battery values above are a worked experiment. For another task, replace the
input selection, `augment_subject`, `prompt`, `negative_prompt` and profiles
together; do not retain battery-specific restrictions in an unrelated task.
The adapter still enforces the pinned model's supported frame size and timing
contracts. Those are model constraints, not dataset-specific tuning.

The measured battery recipe produced very similar scenes: it requested only
subtle lighting changes, held the work surface and palette nearly constant,
and added source RGB conditioning. Increasing its variant count repeats that
weak edit. Before scaling, choose a visible change that the selected camera can
show and specify how it should differ from the source. For example, compare
diffuse frontal lighting with directional side lighting and coherent shadows,
or compare two visibly different finishes on an existing work surface. Keep
task-bearing geometry and identity cues fixed. These are experiment designs,
not generation-qualified recipes.

| Control | How to tune it |
| --- | --- |
| `appearance_profiles_json` | Define visibly distinct, coherent profiles. Change one appearance axis during diagnosis, then combine qualified edits. A neutral color grade can counteract a requested warm/cool look; keep the four fields consistent. |
| `prompt`, `negative_prompt` | State task invariants and the allowed appearance changes separately. Remove a blanket “only subtle lighting” restriction when testing a more visible edit. Preserve object identity without demanding unchanged illumination on every surface. |
| `transfer_rgb_weight` | Compare 0, 0.25 and 0.5 as illustrative experiment points while holding source, captions, profiles and seeds fixed. Smaller values reduce RGB conditioning; inspect whether edit strength improves and whether task details deteriorate. These values are not qualified defaults. |
| `transfer_first_chunk_conditional_frames` | Compare 1 with 0 to test whether the original first-frame appearance suppresses the requested edit. This is independent of `transfer_rgb_weight`; zero RGB hint weight alone does not release the first frame. Keep structural controls and later generated overlap unchanged. |
| `guidance` | With the RGB setting fixed, compare text-guidance values around the current recipe. Stronger guidance may increase the edit or introduce artifacts; inspect both. |
| `control_guidance`, `transfer_edge_threshold` | Tune source structure independently of text guidance. Inspect edge controls when small or dark features disappear. Higher control guidance did not reliably preserve the battery task. |
| `steps`, `transfer_chunk_frames` | Use these for sampling quality or visible temporal discontinuities after the edit is specified. More steps or longer windows do not establish greater appearance diversity. |

For each comparison, retain the source hash, resolved model revision, profiles,
actual per-variant seeds, effective prompts and control receipts. Change one
generation control at a time and use fresh run IDs. Keep the same quality
criteria across comparisons. Review two separate outcomes: whether task details
survived, and whether the intended appearance change is visibly present relative
to the original. Reject a near-copy when the objective is appearance diversity.

The existing attribute verifier examines the generated video; it does not prove
that an attribute changed relative to the source. An already cool-lit source can
match a cool-light description with almost no augmentation. Source-relative
appearance checks measure preservation, not a minimum edit strength. There is
currently no automatic minimum-diversity gate: record paired visual review
separately, including examples with no effective edit. Do not substitute a pixel
difference threshold for this review; gripper deformation also changes pixels.

### Visible surface change experiment

For a work-surface edit, replace the corresponding fields in a private copy of
the workflow's `config` with this fragment. Keep the input selection, task
description, negative prompt and strict evaluation settings from your task.
The material names are configuration values, not special cases in the adapter.
Do not retain this prompt's battery identities for an unrelated task.
If reusing the earlier plan/submit commands, remove matching baseline overrides
from `QUALITY_ARGS` or update them to these values: command-line `--var` values
take precedence over the private workflow's `config`. Inspect the effective
planned configuration before submission.

```yaml
appearance_profiles_json: >-
  [
    {"lighting":"soft neutral diffuse indoor illumination",
     "background":"the existing flat tabletop rendered as light natural oak with fine realistic wood grain, retaining its exact boundary and layout",
     "color_grade":"neutral white balance with light tan wood confined to the tabletop",
     "surface_finish":"matte low-gloss natural oak work surface"},
    {"lighting":"soft neutral diffuse indoor illumination",
     "background":"the existing flat tabletop rendered as a clearly muted blue work surface, retaining its exact boundary and layout",
     "color_grade":"neutral white balance with muted blue confined to the tabletop",
     "surface_finish":"matte low-gloss blue work surface with fine realistic texture"}
  ]
prompt: >-
  Change the existing gray tabletop to the material and color specified in the
  requested appearance edit. Make that tabletop change clearly visible and
  spatially consistent throughout the video. Preserve its flat geometry,
  boundaries and contact shadows; do not add a mat, seam, props or objects.
  Preserve both robot arms and grippers with their white plastic plates and
  dark charcoal-gray mechanical parts. Preserve the black cylindrical battery
  with metallic gold-colored ends, its markings and polarity, and the white
  remote controller with its open battery slot. Retain these foreground identity
  colors. Keep exact grasp and insertion contacts, camera viewpoint, trajectories,
  timing, occlusions and final outcome.
transfer_edge_threshold: low
transfer_rgb_weight: 0.0
transfer_first_chunk_conditional_frames: "0"
control_guidance: "1.0"
guidance: "5.0"
steps: "35"
augmentation_seed: "30"
seed: "17"
variant_count: "2"
variant_parallelism: "1"
```

Use a fresh run with `transfer_first_chunk_conditional_frames=1` for a matched
baseline. Keep all other generation settings fixed. Inspect the first frame,
the whole timeline and the padded image boundaries: gradual gray-to-color
transitions or newly rendered material in letterbox padding do not establish a
consistent edit to the intended surface. For a subsequent RGB-control comparison,
hold the first-frame setting fixed and change only `transfer_rgb_weight`.

## Fan out a reviewed recipe

Separate the number of desired outputs from their execution concurrency:

| Setting | What it expands |
| --- | --- |
| Number of profiles | Requested appearance diversity. Profiles are shuffled without replacement within each cycle. |
| `variant_count` | Generated candidates per episode/camera and generation pass. With P profiles and R samples per profile, choose P × R candidates for R complete cycles. More candidates than profiles repeat appearances with distinct generation seeds. |
| `augmentation_seed` | Profile ordering; hold it fixed for matched comparisons. |
| `seed` | Cosmos 3 sampling. Actual seeds are `seed + attempt * retry_seed_stride + variant_index`; read each output's metadata as the authority. When increasing variant count, choose a retry stride at least as large as that count if seed ranges must not overlap across retries. |
| `variant_parallelism` | Requested generation concurrency inside one job. Effective concurrency is the minimum of variant count, this setting and visible GPUs. It does not allocate GPUs or create jobs on other nodes. |
| Episode/camera selection | Source coverage. Each selection requires its own workflow run and source lineage; equal seeds across cameras do not enforce multi-view consistency. |

Keep `variant_parallelism=1` when expanding output count on a single GPU. This
executes the additional candidates serially. Increasing GPU concurrency requires
an appropriately allocated generation job and separate live qualification of
that setting; the battery runs in this guide do not qualify multi-GPU fan-out.
Use the setup guide's discovered accelerator override or the private spec's GPU
resource profile, then verify the rendered resource request and the resulting
manifest's actual `variant_parallelism`. `parallelism_preset` is a model runtime
setting, not an episode or variant fan-out count.

For a concrete count expansion, copy the `QUALITY_ARGS` definition above into
your private run configuration and replace its task/profile values with the
reviewed recipe. Set `VARIANT_COUNT` to the desired candidate count, then add
these arguments once to both its plan and submit commands. Set
`LEROBOT_EPISODE` and `LEROBOT_CAMERA` from the selected source recording:

```bash
FANOUT_ARGS=(
  --var "variant_count=$VARIANT_COUNT"
  --var variant_parallelism=1
)
npa workbench workflow plan-spec "$SPEC" --run-id "$RUN_ID" \
  --assume-decision promote_checkpoint --var "bucket=$BUCKET" \
  --var input_kind=lerobot --var "lerobot_dataset_uri=$LEROBOT_URI/" \
  --var "input_camera=$LEROBOT_CAMERA" --var "input_episode=$LEROBOT_EPISODE" \
  "${QUALITY_ARGS[@]}" "${FANOUT_ARGS[@]}" --json
```

Use the same `QUALITY_ARGS` and `FANOUT_ARGS` for submission, selecting
`--lerobot-episode "$LEROBOT_EPISODE"` and the corresponding camera in the
submit command above. Review both accepted and rejected plans before launching;
`--assume-decision` is for planning and does not override live evaluation.
Variant count is per generation pass: the existing `refinement_iterations` and
retry controls may produce additional candidates. Keep them explicit in the
campaign record instead of treating the first-pass count as the campaign total.

For dataset expansion, keep a private selection manifest containing dataset
revision, episode, camera, recipe revision and a unique run ID for every selected
combination. Stage the video and metadata for every selected camera. Reuse the
same source and seeds when comparing recipes; use held-out episodes when
qualifying the chosen recipe. Each concurrently submitted workflow needs its own
artifact prefix, isolated SkyPilot API directory and `NPA_CONFIG_DIR`; follow the
setup and cleanup procedures for each run. Independent workflows can be placed
on separately allocated GPUs after their resource requests are reviewed.

Check each output manifest against the requested count and profile coverage,
retaining rejected candidates and any incomplete run. Confirm that requested
appearance differences survived generation before reporting diversity. For
training, perform the action/timestamp and multi-view validation described above.
The rejected battery recipe is not a qualified campaign to scale unchanged.

## Make a synchronized comparison

Download `input/source.mp4` and the selected variant's `augmented_video.mp4`
from the same run. Use the prepared source: the shared dataset MP4 contains
other episodes and runs at a different frame rate. Preserve the original
downloads and record their hashes alongside the variant metadata and verdict.

From the repository root, set the two local paths and verify the complete
decoded timelines before rendering:

```bash
ORIGINAL='<downloaded input/source.mp4>'
AUGMENTED='<downloaded cosmos_augmented/variant-0000/augmented_video.mp4>'
npa/.venv/bin/python - "$ORIGINAL" "$AUGMENTED" <<'PY'
import json
from pathlib import Path
import sys
from npa.workflows.paidf_cosmos3_media import verify_pair

print(json.dumps(verify_pair(Path(sys.argv[1]), Path(sys.argv[2])), indent=2))
PY
```

Continue only if verification succeeds. Put the original on the left and the
model output on the right, retaining every frame:

```bash
ffmpeg -i "$ORIGINAL" -i "$AUGMENTED" \
  -filter_complex '[0:v][1:v]hstack=inputs=2[comparison]' \
  -map '[comparison]' -an -c:v libx264 -crf 18 -pix_fmt yuv420p \
  -movflags +faststart original-vs-augmented.mp4
```

Do not trim to the shorter clip, retime either video, or blend the source into
the generated frames. A rejected candidate is still useful for comparison;
label its quality verdict separately from successful decoding and alignment.

## Validation status

The pinned high and left-wrist episode-zero inputs were downloaded and passed
the real local LeRobot v3 selector and full-video normalization: each retained
12 seconds and decoded to 288 frames at 832×480 and 24 fps. This validates input
selection and preparation. Custom-profile forwarding, evaluator options and
episode-spanning caption selection have automated regression coverage.
On September 17, 2026, real Cosmos 3 generation on an RTX PRO 6000 Blackwell
produced a complete episode-zero high-camera cool-lighting candidate with the
high-control settings described above. Its four native windows produced 288
frames; the synchronized source/output comparison had zero timestamp error.
Native text checks and configured video postprocessing completed. Visual review
rejected the candidate for harsh contrast and distorted gripper/battery details.
The warm sibling was cancelled before completion, so this is not a completed
two-variant batch or a full accepted PAIDF run. The source, native output,
receipts and comparison remain in private run evidence.

Task-aware captioning was separately exercised against eight original frames.
It removed the initial false lighting descriptions but retained uncertainty and
some object/contact errors. Caption context is useful input, not task validation.
Earlier starter live evidence does not establish this dataset's augmentation
quality. No recipe is yet qualified for training or multi-view consistency.

A second real GPU trial generated both warm and cool variants on B200 using
task-aware captions and the starter sampling controls (1.5 control guidance,
5.0 text guidance, 24 steps). Both fully decode to the same 288-frame timeline
with zero timestamp error. Contact crops still show distorted, excessively
contrasted grippers, so both were visually rejected. The changed captions and
sampling controls make this a recipe comparison, not an isolated causal test.
The real evaluator completed with zero accepted clips and an aggregate score
of 0.071303 against 0.75. Both failed required temporal checks (scores 0.071894
and 0.070713 against 0.8) and strict attribute checks (2/4 and 1/4 correct).
An advisory appearance-fidelity pass did not establish correct task geometry.

A third trial changed only the edge preset from `medium` to `low`, retaining
the second trial's source, captions, profiles, seeds and sampling controls.
Both variants completed on B200. The native loader verified the actual
50/100 Canny controls, and both 288-frame outputs again had zero timestamp
error. More internal gripper detail survived, but excessive contrast and altered
battery/gripper features remained. Visual review rejected both; the evaluator
also accepted zero clips, with aggregate score 0.073388, temporal scores
0.074397 and 0.072379, and 2/4 attributes correct for each candidate. This
qualifies execution of the lower preset, not its output for training.

A fourth B200 pair retained low edges, profiles, source captions and seeds, but
used control guidance 1.0, text guidance 3.0 and 35 steps. This is a complete
sampling-recipe comparison, not an isolated test of one parameter. Both outputs
retained the 288-frame timeline with zero timestamp error. Contrast and gripper
appearance improved visibly. The cool candidate still turned the battery's
gold end red; the warm candidate retained its gold coloring in reviewed frames
but changed fine gripper details. An advisory paired-frame VLM review rejected
the cool recoloring and found insufficient evidence of the warm appearance
edit. It does not replace human inspection or establish continuous dynamics.
The unchanged evaluator rejected both: aggregate score 0.146111, temporal scores
0.142263 and 0.149958, and 2/4 attributes correct for each candidate.

A fifth pair kept the fourth trial's sampling controls and added explicit
object identity colors to the task specification, with fresh task-aware source
captions. Both videos and the real evaluator report completed. The gold battery
ends were retained in reviewed frames, but fine gripper details still differed.
The unchanged evaluator rejected both: aggregate score 0.149534, temporal scores
0.148944 and 0.150124, and attribute scores 2/4 and 1/4. The workflow did not
complete its terminal disposition stage: its isolated API refused a source
configuration changed by another trial. These measurements come from the retained
evaluator report, not a successful end-to-end workflow receipt.

A sixth pair re-extracted and re-captioned the complete episode using the same
identity-color specification and sampling controls. The prepared video hash
remained unchanged. The eight saved caption frames matched source indices 0,
41, 82, 123, 164, 205, 246 and 287, including release and the final state. Both
generated videos retained all 288 frames with zero timestamp error. The actual
outputs still changed fine gripper details; the evaluator rejected both with
aggregate score 0.150257, temporal scores 0.147766 and 0.152749, and attribute
scores 2/4 and 1/4. Complete caption coverage fixes missing observations; it
does not establish faithful generation.

A seventh pair added only `transfer_rgb_weight=0.5` to the sixth trial's
generation controls, reusing its prepared source, complete-clip captions,
profiles, prompts and seeds. The resolved model revision was unchanged. Both
native-loader receipts verified all 288 lossless RGB control frames, alongside
the same low-threshold edge pixels. Text and video guardrails remained enabled;
no source pixels were blended into the generated outputs. Both complete outputs
again had zero timestamp error, and the workflow reached its terminal rejection
disposition. The aggregate score improved from 0.150257 to 0.178542, with
temporal scores 0.177784 and 0.179300 and attribute scores 3/4 and 1/4. Both
still failed the unchanged aggregate threshold of 0.75, required temporal
threshold of 0.8, and requirement for all four attributes to match. Gold battery
ends remained visible in reviewed frames, but fine gripper geometry still
differed and the intended appearance edits were weak. RGB conditioning is a
verified general-purpose control, not a qualified battery augmentation recipe.

A subsequent material experiment replaced the subtle-lighting profiles with
muted blue and light oak tabletop appearances, removed the blanket subtle-edit
instruction and RGB hint, and used low edges, control guidance 1.0, text guidance
5.0 and 35 steps. With the original first-frame anchor retained, the blue surface
emerged partway through the clip; the oak candidate largely retained the gray
surface and generated wood in side padding. Both were rejected. The evaluator
reported aggregate score 0.133584, temporal scores 0.138742 and 0.128425, and
attribute checks 4/4 and 3/4. The blue attribute pass did not detect the initial
gray-to-blue transition; the oak background pass did not prove the intended
surface changed.

The matched follow-up changed only first-window source conditioning from 1 to
0. Both videos now show their requested blue or oak work-surface appearance from
the first frame through the reviewed ending. Native receipts verify zero source
frames anchoring the first window, five generated overlap frames for later
windows, four native windows and complete edge coverage. Both outputs retain
288 aligned frames. Mean absolute decoded channel differences in the source's
image area were 32.69/255 for blue and 41.13/255 for oak, compared with roughly
4/255 for the earlier weak RGB-conditioned lighting edits. These describe edit
magnitude, not realism. Sampled contact review still found altered gripper and
controller details; the oak render also changed padding and contact shadows.
Both matched all four requested appearance attributes, while the unchanged
required temporal check rejected both (0.127409 and 0.091969; aggregate 0.109689).
This establishes a visible-edit control, not a training-qualified recipe.

A final matched pair retained zero first-window source frames and changed only
`transfer_rgb_weight` from 0 to 0.25. Both native receipts verified all 288 RGB
and edge control frames; text/video guardrails remained enabled. The blue
surface remained visibly changed in reviewed first, contact, release and final
frames, while fine gripper/controller details still differed. Oak remained
visible on the lower work surface but also appeared in side padding, with a
gray upper surface and altered shadows. Mean absolute decoded channel
differences in the source image area were 29.30/255 and 40.03/255. Both outputs
retained the complete aligned timeline and matched all four appearance
attributes. The unchanged evaluator still rejected both: aggregate score
0.117891, temporal scores 0.140340 and 0.095442. The slight score improvement
over zero RGB weight does not qualify either candidate for training.

These temporal scores come from NPA's source-relative motion-residual check,
which accompanies the upstream Cosmos Evaluator result. They do not certify
contact physics. Across these ten trials, nineteen complete generated clips
were reviewed; the nine completed two-variant evaluator batches accepted none.
Generation covered episode zero's high camera only. The separately prepared
wrist input and the remaining episodes have not been generation-validated.
