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

An ALOHA or other robot-dataset frame can supply an EVG scene. Record dataset,
episode, camera, and frame identity, and assess the requested generated event.
The IAA/EVG graphs do not consume robot actions or evaluate a policy; results
cannot establish action alignment, contact fidelity, or robot-policy improvement.

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
