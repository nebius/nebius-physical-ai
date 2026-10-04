# NCore: native photometry compensation, complete fitted-frame evidence

One frozen photometry-only experiment completed all four standard workflow
stages at source `10b3c34dd6033a1de8798d0013b062848c73a3a0` on an RTX PRO 6000
Blackwell Server Edition. The fitted-frame numerical gates passed. Independent
review of this new complete output population and its diagnostic visual protocol
is pending; this is not a claim of held-out quality, physical safety, or public
image-release authorization.

## Frozen experiment and all 518 results

The exposure-bracketed input retains all 518 photographs, three cameras and
163,453 SfM points. Geometry, seed 42, one epoch, 30,000 samples, Gaussian
schedule, losses, render offsets and all numerical gates remain unchanged from
the [failed geometry-only experiment](https://github.com/nebius/nebius-physical-ai/blob/7b10dc8eb311715108761e64dc5b263bc7b068c9/docs/testing/evidence/ncore-geometry-f037-20261004/README.md).
The sole scientific recipe change is the supported native PPISP subtree:
`model/post_processing@model.post_processing.b=ppisp`. This was a pre-frozen
causal experiment, not a threshold adjustment or a retry for a favorable score.

| Fitted-population metric | Native result | Unchanged gate | Result |
| --- | ---: | ---: | --- |
| PSNR | 23.8717880249 | at least 15 | PASS |
| SSIM | 0.6742462516 | at least 0.5 | PASS |
| LPIPS | 0.4939148724 | at most 0.5 | PASS |

[All 518 rows](metrics-518.json) and [all paired comparisons](paired-metrics-518.json)
are retained. Each of the 518 frames improved on all three metrics relative to
the geometry-only run. That is a descriptive, same-population comparison, not
held-out generalization, statistical significance or proof of a sole cause.
Stochastic training does not establish that every frame was sampled during
training. Camera populations are 267, 60 and 191 frames.

The native PPISP controller was absent. Novel views use frame index -1 and omit
per-frame exposure/color adjustments; per-camera vignetting and response-curve
processing remain. Therefore these fitted scores do not establish novel-view
quality. Actual retained optimizer logs show zero PPISP learning rates through
step 249 and nonzero rates at step 299; they do not identify the exact first
parameter mutation.

## Complete RGB media and fixed-index gallery

These six unmodified native videos include every reconstruction and novel RGB
frame. They sequence independent photographic viewpoints, not synchronized
capture times or a continuous physical trajectory. Original compressed bytes,
encoder dimensions and padding are preserved.

| Camera | Reconstruction RGB | Novel RGB |
| --- | --- | --- |
| 1 | [267 frames](media/reconstruction-camera1.mp4) | [267 frames](media/novel-camera1.mp4) |
| 2 | [60 frames](media/reconstruction-camera2.mp4) | [60 frames](media/novel-camera2.mp4) |
| 3 | [191 frames](media/reconstruction-camera3.mp4) | [191 frames](media/novel-camera3.mp4) |

The [18-image gallery](gallery.md) uses the same fixed first/integer-middle/last
selection as the failed run, with no retouching, re-encoding or score-based
selection. Owner inspection of four separate frozen diagnostic frames found
recognizable sculpture, water and buildings, but soft edges, smeared reflections
and local surface distortion remain. This is neither independent acceptance nor
whole-population visual inspection. No new hosted-provider call has been made
for these pixels.

[Population hashes](population-hashes.json) cover all 2,072 native PNGs,
12 videos and 2,072 decoded video frames, including privately retained
distance/opacity channels. Complete machine decoding is not every-pixel human
review. SHA-256 values in that inventory use documented uppercase hexadecimal;
all digest bits are unchanged from the private original. A hash-shaped scanner
finding and the independent exact-frame recomputation are retained privately;
no scanner rule was suppressed. [Media manifest](media-manifest.json) binds the
published files to the unchanged original bytes.

## Source and artifact provenance

- The source is NVIDIA's CC-BY-4.0 [PhysicalAI-NuRec-PPISP dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP/tree/2521064a3af6ab1c1caa2ba1b01ddde7eecded69), `struktur28`. ZIP SHA-256: `cf7ab7f100da66b2bf05b178ebcfa3a950e1bf2b1d7ff64a6c7a1e1f682afa8d`. These media are NRE-generated derivatives, not original photographs; attribution is to NVIDIA and the dataset contributors.
- Canonical converter source: `9d31474a4cff893f81b0831177129f0513de638b`. Conversion receipt SHA-256: `6ce24c423c24f97fc4a2fbc76fdc53a5b8eab510e1265b0aa4de19c0d0f1892c`. All ten canonical input members remain byte-identical, with an explicit reuse receipt rather than a fictitious new conversion. All 52 image-build inputs remain unchanged through the current consumer successor.
- Native NRE image: `sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284`. The actual parsed recipe SHA-256 is `6140d85e7185032569bfda0d5fab08174ba946e16be8f8bedf97c00a020be010`.
- Complete terminal readback covers 2,192 objects / 4,828,795,521 bytes. Mandatory source, conversion, rig and native render receipts are present. The original native and cloud metric bytes match. USD dependency traversal found no unresolved dependencies; this is not an Isaac rendered workload.
- The closed 1,626,065,960-byte native event journal contains 23,166 records with every header and payload CRC32C verified. All 600 main training points, 43 scalar tags and 1,554 image events are retained. This is distinct from the older failed run's truncated journal.
- The original four-stage workflow succeeded, but its initial qualification consumer rejected the genuine standard final report, then a staged repair exposed a workflow-ID versus recording-prefix-ID mismatch. Both refusals remain retained. Signed consumer successor `aa3958d9d63de78a127e55ad1759311dd9848374` validates literal producer fields and exact prefix binding; its unchanged-byte CPU replay passes, including 322 Rerun rows and 72 exact novel-image rows. The model/workflow execution remains the original `10b3c34d`, not this later source SHA.

## Checks, failures and remaining limits

The narrow consumer successor has 396 affected test passes and 270 prechecks.
Its current-head protected CI is being observed separately. Earlier exact-tree
`10b3c34d` CI was independently accepted: 42,656 passes, 926 summary skips
(884 emitted skipped nodes plus 42 non-item reports), one non-strict XPASS and
77.860233% coverage. Those are not relabeled as later-head test executions.

All earlier failed runs remain failures. In particular, the original run failed
all three numeric gates; the geometry-only run failed PSNR and LPIPS and its
workflow failed on missing lineage. The historical hosted score of 0.8 disagreed
with independent severe-blur observations and does not review these new pixels.
The current 20-frame diagnostic protocol has zero provider calls and awaits the
existing independent lane's actual-pixel and request-reuse disposition.

Owned GPU pods and the task controller are absent; four unrelated controllers
were preserved. Standard cleanup remains `degraded_local_metadata` because it
refused to clear another cluster's ownership record. That record and the local
API were left intact; cloud absence is not complete local cleanup.

Source readiness, private image qualification and public image release remain
separate. The local whole-byte image policy used 129 actually observed exact
infrastructure literals, not a complete customer inventory or equivalence to
private CI regex policy. Earlier raw `valid=false` scans and adjudications remain
unchanged. No private image, model or vendor payload is published here. Distinct
AI reviews are not independent human approval. PR 646 remains draft while its
current-head CI, independent output review and diagnostic visual conditions are
unresolved.
