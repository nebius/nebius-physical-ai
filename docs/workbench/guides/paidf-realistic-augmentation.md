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
TASK_PROMPT='Preserve both robot arms and grippers, the same battery, its polarity markings, the remote controller, its open slot, the exact grasp and insertion contacts, camera viewpoint, trajectories, timing, occlusions and final outcome. Apply only the requested appearance variation to existing surfaces.'

QUALITY_ARGS=(
  --var "appearance_profiles_json=$APPEARANCE_PROFILES"
  --var 'augment_subject=two robot arms placing a battery into a remote controller slot'
  --var "prompt=$TASK_PROMPT"
  --var 'negative_prompt=extra or missing objects, warped grippers, altered battery or slot, floating objects, interpenetration, motion retiming, flicker, ghosting, unsafe content'
  --var augmentation_seed=30 --var seed=17
  --var control_guidance=3.0 --var guidance=3.5 --var steps=32
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

These settings are an **unqualified starting experiment**, not a measured quality
recommendation. Compare the same episode, appearance profiles and seeds using
the starter sampling controls (`control_guidance=1.5`, `guidance=5.0`, `steps=24`)
and the proposed controls above in separate fresh run IDs. Use the same strict
evaluation settings for both. Do not compare a strict candidate against a
baseline admitted only by exploratory thresholds. Adjust one generation control
at a time after observing the failure; lowering thresholds does not improve pixels.

## Review and tune the actual outputs

Keep `structural_control=edge`, `alignment_mode=required`, guardrails enabled,
and `source_motion_weight=0.0`. The source is normalized to 832×480 at 24 fps
without changing duration. This example becomes 288 frames over 12 seconds.
Frame alignment proves coverage and timing, not correct grasp geometry.

| Observed failure | Next controlled experiment |
| --- | --- |
| Small battery, gripper or slot geometry drifts | Increase `control_guidance` relative to the baseline, reduce conflicting prompt guidance, and simplify the appearance change. Recheck whether the source edges actually resolve the small feature. |
| Appearance hardly changes | Confirm the selected profile is present in `metadata.json`'s effective prompt. Check the raw output against the source. Try one more visible but plausible lighting change; stronger structural control can suppress edits. |
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
The proposed sampling controls have not yet been qualified by GPU generation
or visual review of augmented battery data. Earlier starter live evidence does
not validate these new prompts or this dataset's augmentation quality.
