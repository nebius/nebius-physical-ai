# Realistic manipulation augmentation with PAIDF Cosmos 3

Use this procedure with [PAIDF Cosmos 3](../../../workflows/guides/paidf-cosmos3.md)
to vary the appearance of a recorded manipulation while preserving the task.
Run the workflow on a supported Nebius GPU through SkyPilot. Inspect the actual
generated video against the complete source before using it as training data.
The starter's exploratory acceptance thresholds establish neither contact
correctness nor task success.

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
Set a coherent appearance experiment, keeping task identity colors unchanged:

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
experiment; they remain an **unqualified starting experiment**, not a
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
