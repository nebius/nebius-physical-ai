# Camera views and spatial evidence

## Check the requested capability

Discover the authenticated models and OpenAPI before submitting. Single-source
augmentation and structural transfer do not establish synchronized multiview
generation. If the supplied service does not expose a qualified multiview model,
profile and request, report that gap. Do not add native Framework fields to a
strict hosted request or relabel independent single-view jobs as synchronized
multiview output.

The eleven combinations of edge, depth, segmentation and blur combine two,
three or four structural controls. They are not a camera count.

NVIDIA's separate [Cosmos3-Nano-Transfer-Auto checkpoint](https://huggingface.co/nvidia/Cosmos3-Nano-Transfer-Auto)
documents up to eleven standard/fisheye camera views. These are learned physical
camera identities, not arbitrary new angles obtained from one ordinary video.
The [native inference contract](https://nvidia.github.io/cosmos-framework/inference_auto/)
requires matched camera-local WSM controls and captions. Image-conditioned use
also requires aligned initial RGB for every selected camera. Native availability
does not make this checkpoint an exposed base Nano/Super API profile.

For a claim of at least ten seconds, check every decoded generated video. The
published seven-camera example has 300-frame, 30-FPS inputs but requests only
297 output frames, or 9.9 seconds. Preserve source sampling, model conditioning
FPS and output playback FPS separately; changing a header does not resample a
scene. Align control coverage and caption intervals to the actual horizon.

## Form a standalone spatial proof

A demo claiming spatial accuracy must contain actual model outputs. A support
report or a form with empty output slots is preparation, not a successful demo.
Keep exact infrastructure identities, credentials and private media outside Git.
Do not publish or share the proof beyond the customer's authorized audience.

Build one self-contained HTML artifact with the following evidence:

| View or control | Required evidence |
| --- | --- |
| Camera rig | A diagram with physical camera key, learned identity, position, orientation, lens model and effective field of view; identify calibration source and units. |
| Per-camera comparison | Matching source RGB when available, WSM/control input, and real generated output for that same camera and interval. Label WSM-only conditioning explicitly; do not invent reference RGB. |
| Playback | One shared timeline, play/pause, seek and camera/case selection. Show decoded frame count, FPS, duration and source timestamp mapping. Each generated view must meet the requested duration. |
| Spatial inspection | Time-aligned overlays and crops for projected objects, lane boundaries and overlapping views. Retain object/landmark identities and show start, middle, end and worst-error frames. |
| Measurements | State the reference geometry, correspondence method, units, sample count, visibility/occlusion rules and thresholds before evaluation. Report errors and failures per camera and across overlapping views. |
| Reproduction | Exact model/checkpoint/runtime revision, sanitized request, effective settings, source/control/output hashes, durable run identity, technical validation and any missing evidence. |

Distinguish the claims supported by each measurement. Pixel overlap, edge
distance, object-center displacement and box IoU measure 2D alignment against
their stated reference. They do not prove 3D spatial accuracy. Calibrated 3D or
cross-camera claims require verified intrinsics/extrinsics, corresponding
landmarks or objects and measured reprojection, epipolar, triangulation or
metric-position errors appropriate to the lens models. State how correspondences
were obtained; guessed matches or a plausible camera diagram are not calibration.
Report insufficient overlap or correspondences as unmeasured, not zero error.

Test smaller valid camera subsets and the intended full rig with real matching
assets, and test rejection beyond supported identities. Preserve physical IDs
when selecting or reordering views. Do not duplicate one camera's media under
several identities to reach eleven. Retain failed cases and uncertainty instead
of showing only the best frames.

Embed all required media, data and viewer code. Verify offline playback,
synchronized seeking, downloads and hashes in a real browser, with no unexpected
network requests or JavaScript errors. Full decode and successful delivery remain
separate from spatial fidelity and suitability for reusing source training labels.
