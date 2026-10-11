# Reviewable IAA and EVG proof

The standalone HTML should let a reviewer assess the task from actual inputs,
generated media, and annotations. Scope it to the two workflows in this package.
Keep an integration-completion verdict separate from the media-quality verdict.

## Choose seeds for the task

| Workflow | Upstream starting point | What representative review should establish |
| --- | --- | --- |
| IAA | Person images grouped by person ID; NVIDIA suggests RSTPReid or a similar dataset | The requested attributes changed while identity, pose, and composition remain suitable for the intended person-image task |
| EVG | User-provided scene images; the guide illustrates warehouse entrances and loading docks | The requested event visibly occurs in the seed scene and its annotations describe the generated video |

See the pinned [IAA](https://github.com/NVIDIA/paidf-orchestration/blob/f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9/docs/image-attribute-augmentation/getting-started.md)
and [EVG](https://github.com/NVIDIA/paidf-orchestration/blob/f7ecd8c5d7aeec28b2d476b9e71b53a48ba8c0f9/docs/event-video-generation/getting-started.md)
input guides. The last column gives NPA review criteria, not upstream guarantees.
There is no automatic NPA seed download. Record the source, selected subset,
person/scene identity, and original image hash before preparing worker input.

The retained silhouette and cameraman runs exercise the native component
protocol. They remain historical integration fixtures. In the retained EVG
example, all 21 anomaly answers are negative; answer coverage alone does not
verify the requested falling event. Preserve that result when showing the video.

For a recognizable Physical AI demonstration, start IAA with a real person in
a human–robot workcell, and EVG with a real workcell or loading-dock scene.
Show the source next to the actual generated result. Name the consumer: clothing
robustness for human perception, or event candidates for workcell safety
perception. A realistic input is not evidence of successful generation. Complete
the native labeling stages and review the actual edits/events before reporting
task success. Keep source attribution, source-view limitations, sampled
environment mismatches, failed attempts, and rejected samples visible.

For IAA, prefer a full person view with the requested clothing visible. A cropped
source can make the model invent a new pose or lower body to satisfy footwear
instructions. Review person identity, pose, robot geometry, tool arrangement,
and background separately from attribute verification; that verifier cannot
establish scene preservation. For EVG, match the sampled environment to the
source scene where possible and check the complete video for event occurrence,
robot continuity, and changes at the outer borders.

The fixed native sampler can be exercised reproducibly without adding another
augmentation backend. For the first variant of one input, IAA seed `0` samples
a red vest, blue jeans, and white sneakers; EVG seed `6` samples
`fire_or_smoke` / `warehouse`. These are documented evaluation conditions,
not configurable distributions or safety-equipment certification. Confirm the
actual sampled values in `configs/manifest.json`, since adding inputs or variants
advances the sampler. Photograph-derived results do not establish policy benefit.

An ALOHA or other robot-dataset frame can supply an EVG scene. Record dataset,
episode, camera, and frame identity, and assess the requested generated event.
The IAA/EVG graphs do not consume robot actions or evaluate a policy; results
cannot establish action alignment, contact fidelity, or robot-policy improvement.

For robotics utility, identify the consumer: safety perception, person tracking,
inspection, or manipulation-policy training. Show its task-specific evaluation
when claiming an improvement. For robot demonstrations, reuse the existing
[main PAIDF](../main/paidf-cosmos3.yaml) and LeRobot/Sim2Real paths; validate source
action/timestamp alignment and visual fidelity before comparing original-only
and augmented-data policy training on the same held-out evaluation.

## Show the evidence together

For IAA, display the original and prepared person image, sampled attribute
values, raw generated edit, postprocessed dataset image, and verification/search
outputs. Include checked, skipped, and failed attributes, plus rejected samples.
Inspect the actual identity and composition alongside the requested clothing.

For EVG, display the original and prepared seed, sampled event/environment,
complete playable generated video, and its detection/tracking, captions,
anomaly/person QA, and attribute-search outputs. Report answered, empty, missing,
and negative answers separately. Inspect event occurrence and motion continuity
through the full video; a terminal protocol pass is not a positive event verdict.

For both, bind the page to the spec and upstream/model revisions, producer
revision, media SHA256 hashes, final dataset manifest, and terminal validation.
Label execution dates and identify historical results separately from current
local planning checks. Retain failed attempts and rejection accounting.

## Make it standalone

Embed CSS, scripts, seed images, generated images/video, and displayed annotation
summaries in one HTML file. Use actual output bytes; if a browser-compatible
view copy is necessary, identify its conversion and retain the raw file/hash.
Optional source links can remain external. Rendering and playback must work
with networking disabled. Keep exact cloud identifiers, credentials, private
endpoints, and raw operational receipts in access-controlled evidence outside Git.

Verify offline desktop and mobile layouts, decoded images, full video playback,
embedded-asset hashes, and absence of script errors or external asset requests.
Save the generated HTML outside Git with a terminal-friendly filename, retain
its SHA256/verification receipt, and hand off its absolute path.

## Build the local report

The [template](proof-template.html) presents exactly IAA and EVG, their existing
state graphs, embedded native media, robotics task fit, and an interactive Bash
command preview. Its form requires an input prefix and non-placeholder private
image digests before enabling copy. Live access and image preflights remain
required; the form cannot establish their result. It makes no API calls or
stores settings. Actual robot-source examples are labeled separately from CDF
generation results.

Prepare reviewed evidence JSON outside Git. Supply only sanitized display
metadata, never credentials, concrete cloud identifiers, or raw operational
receipts. The top-level fields are `built_at`, `evidence_scope`,
`upstream_revision`, `current_source_revision`, `workflows`, `assets`, and
`robot_assets`, plus `claim_summary`, `recorded_checks`, and
`fresh_gpu_execution`. Each workflow row needs `id` (`iaa` or `evg`), `title`,
`spec_filename`, `input_guidance`, `resources`, `final_manifest`, `capabilities`,
`queries`, `sampled_variables`, `producer_revisions`, `dataset_sha256`, and
`terminal_status`, and an `evidence` object. The evidence object contains
`title`, `status_label`, `protocol_passed`, `media_ids`, `metrics` (each with
`value` and `label`), `quality_verdict`, `variables_label`, `notes`, and `source`
(`summary`, optional `url`, and optional `media_ids` for prepared input images).
Keep predicted sampling labeled as planned
until the real configuration exists, and keep missing outputs pending rather
than reusing unrelated successful metrics. Optional `preferred_seed` sets the
selected workflow's command-form seed; without it the form uses `42`.
Each asset needs `id`, `path`, `mime_type`, `sha256`, `alt`,
and `caption`; paths are resolved relative to the evidence JSON. The template
renders the exact `media_ids` supplied for each workflow. `robot_assets` lists
optional source-reference asset IDs.

Use the packer from the repository root:

```bash
npa/.venv/bin/python workflows/cosmos-data-factory/build-proof.py \
  /absolute/private/evidence.json \
  /absolute/private/cosmos-data-factory-turnkey.html
```

The packer verifies every original media hash, embeds JPEG/PNG/MP4 bytes, removes
asset paths from the page, and derives stage/resource/model details and spec
hashes from the two adjacent YAML files. It prints the delivered HTML's byte
count and SHA256. Revalidate and visually inspect each generated report with
networking disabled before handoff. The template takes execution metrics,
annotation samples, quality verdicts, and source disclosures from the reviewed
evidence. Update those fields from the real run when replacing media; do not
carry successful fixture verdicts into a pending or failed representative run.
