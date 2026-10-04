# NCore: corrected-geometry NRE execution, quality still fails

This records the actual `f03737a9b5283e5e8f1b4ae17a82c54ac36e04bb`
execution. Reconstruction and novel rendering succeeded on an RTX PRO 6000
Blackwell Server Edition. The original workflow subsequently **failed** during
visualization. Neither source CI nor this artifact collection establishes
model-quality acceptance, physical safety, or image-release authorization.

## Complete numerical population

All 518 photographs, three cameras and 163,453 SfM points were retained. The
native recipe used seed 42, one epoch and 30,000 samples, with unchanged Gaussian
schedule and no exposure compensation. The only scientific change from the
earlier failed run was the corrected independent-camera geometry. The exact
canonical input and native consumer were independently checked before training.

| Whole-population metric | Native result | Frozen gate | Result |
| --- | ---: | ---: | --- |
| PSNR | 14.8628644943 | at least 15 | FAIL |
| SSIM | 0.5613004565 | at least 0.5 | PASS |
| LPIPS | 0.6297209263 | at most 0.5 | FAIL |

[All 518 metric rows](metrics-518.json) retain the actual values. Independent
recomputation agrees within numerical precision. These are fitted-frame
reconstruction metrics, not held-out generalization or calibrated visual
usefulness. Camera populations are 267, 60 and 191 frames.

## Actual media, not a favorable selection

The six unmodified native RGB videos below contain every reconstruction and
novel-view RGB frame. They sequence independent photographic views; they do not
establish capture-time synchronization, continuous motion or temporal quality.
The native encoder's dimensions/padding and compressed pixels are preserved.

| Camera | All reconstruction RGB frames | All novel RGB frames |
| --- | --- | --- |
| 1 | [267 frames](media/reconstruction-camera1.mp4) | [267 frames](media/novel-camera1.mp4) |
| 2 | [60 frames](media/reconstruction-camera2.mp4) | [60 frames](media/novel-camera2.mp4) |
| 3 | [191 frames](media/reconstruction-camera3.mp4) | [191 frames](media/novel-camera3.mp4) |

[Fixed first/middle/last PNG gallery](gallery.md) contains the exact 18 RGB
images opened by the independent reviewer, with no retouching or re-encoding.
The reviewer observed recognizable sculpture, water and facade, but soft/blurred
surfaces, smeared edges, floaters and exposure differences; the middle camera-2
reconstruction has broad washed-out haze. **No visual-quality approval followed.**

[Population hashes](population-hashes.json) cover all 2,072 original PNGs and
all 12 native videos / 2,072 decoded video frames, including distance/opacity
outputs retained privately. Independent machine decoding/hash comparison covered
that entire population; direct visual inspection covered the 18 listed images,
not every pixel or an independently calibrated temporal review. The complete
2,096-object raw run remains privately retained.

## Provenance and failures that remain

- The input is NVIDIA's CC-BY-4.0 [PhysicalAI-NuRec-PPISP dataset](https://huggingface.co/datasets/nvidia/PhysicalAI-NuRec-PPISP/tree/2521064a3af6ab1c1caa2ba1b01ddde7eecded69), `struktur28`, at the pinned revision shown in that link. The source ZIP SHA-256 is `cf7ab7f100da66b2bf05b178ebcfa3a950e1bf2b1d7ff64a6c7a1e1f682afa8d`. These media are NRE-generated derivatives, not original photographs; attribution is to NVIDIA and the dataset's contributors.
- The converter/image producer is `9d31474a4cff893f81b0831177129f0513de638b`. Its canonical conversion SHA-256 is `6ce24c423c24f97fc4a2fbc76fdc53a5b8eab510e1265b0aa4de19c0d0f1892c`. The 52 image-build input files remain byte-identical through source `5779ecfadd9fdd558c15d24940f91ff6137fabbf`; this is a source bridge, not a new model execution.
- Native NRE image digest: `sha256:97f43e7130c5636ce3e80ea3184d97f56a87fdd989b05cce42230881dbdea284`. The self-contained USDZ SHA-256 is `c38d80d6e612613063b437675b2d4dff83fe93dc6aaf7f8cdacc62dcce26722e`. USD dependency traversal resolved all dependencies; it is not an Isaac rendered workload.
- The original run root lacked required source/conversion/rig sidecars; visualization correctly failed closed. The native render receipt was also absent. Exact-image CPU diagnosis found the video decoder dependency missing; a narrowly reviewed setup fix installs it before work. Neither that fix nor a separate post-hoc CPU Rerun derivative rewrites the failed original workflow or creates its missing receipt.
- The retained journal snapshot lacks the final record's four-byte CRC. All 600 points for each of 32 training scalar tags precede that tail and independently match. No complete-journal or CRC32C-validation claim is made.
- Exact owned GPU pods and controller are absent; four unrelated controllers were preserved. Standard cleanup remains `degraded_local_metadata`: it refused to clear a global ownership record for a different cluster. The record and local API were left intact.
- The earlier run's three failed quality gates and absent conversion receipt remain historical failures. Its hosted judge's final score of 0.8 disagreed with independent severe-blur observations. Those calls were diagnostic, not independently calibrated model-quality evidence, and do not review these newer pixels.

## Source checks and limits

Source `300f2da2cc94752678a032b6dbbf65e064223bfe` has independently accepted
exact-tree CI evidence: 42,507 passes, 924 summary skips (882 emitted skipped
nodes plus 42 non-item reports), one non-strict XPASS, zero failures/errors and
77.847% coverage. Thirty-one jobs succeeded and three workflow jobs skipped;
required contexts used GitHub Actions app 15368. This is not CI for a future SHA.

The decoder successor has 2,141 affected passes, 117 explicit optional-dependency
skips, eight new decoder controls and 270 prechecks. The signed current-main
integration has 1,580 affected passes, two live-GPU opt-in skips, 270 prechecks
and a passing documentation-drift check; fresh protected CI is separately
observed. Independent Codex review is not human approval or a new Claude pass.

Private image qualification and source readiness remain distinct from public
image release. The local full-byte policy used 129 actually observed exact
infrastructure literals; it is not an assertion of equivalence to a private CI
regex policy or a complete customer inventory. Earlier raw `valid=false` scans
and their adjudications remain unchanged, not relabeled as automated acceptance.

[Summary and immutable receipt hashes](summary.json) give the exact evidence
boundaries. This proof intentionally records failures; PR 646 remains draft
until the remaining scientific, workflow and current-head gates are resolved.
