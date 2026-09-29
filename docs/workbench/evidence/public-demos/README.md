# Public demo qualification

The full `real-to-sim`, `synthetic-data`, and `nurec` demos are qualified from
source `39a662651f6b46db0ce57dbff5fb9828c9d22643` on NVIDIA RTX PRO 6000 Blackwell
GPUs. Scan qualification completed on September 29, 2026; SDG and NuRec completed
on September 28. The receipts bind measured outputs to source, workflow, image,
staged-source, and artifact hashes without disclosing runtime infrastructure or
storage locations.

The current retention change updates the shared navigation runtime. The scan
receipt below remains qualification of its recorded source; a fresh full scan
run must qualify the new source. SDG and NuRec workload code, selected inputs,
image selections and rendered tasks remain unchanged. Their recorded measurements retain
their original scope.

| Demo | Independently verified result | Receipt |
| --- | --- | --- |
| Scan-to-policy navigation | 4,000 robots in one measured scene, 500 training updates, and 3,415/4,000 successful held-out routes (85.375%, above the unchanged 80% gate) | [Scan qualification](real-to-sim.json) |
| Industrial sensor generation | 265 poses, four cameras, 1,060 views, 4,505 capture artifacts, and 976,895,672 valid depth pixels and fused colored points | [SDG qualification](synthetic-data.json) |
| NuRec | Checkpoint at 30,000 steps, 38 rig-offset novel views, PSNR 31.025, SSIM 0.832, and LPIPS 0.267 | [NuRec qualification](nurec.json) |

The scan check independently recomputed every held-out trajectory and verified
all 500 PPO updates, 16,000,000 training transitions, changed actor and critic
parameters, and native optimizer state. Four fresh-process physical controls
passed separately for training and evaluation. All 4,000 robots share one
measured collision scene and use static range observations that exclude peers.
The check read back nine publication bundles containing 5,809 files
(2,731,814,370 bytes); these are outputs from eight executable workflow states.

The [scan visual review](real-to-sim-visual-review.json) verifies the original
native report and byte-identical `demo view` download. The HTML shows the scored
focal episode through step 8; the full 301-frame video also contains later
observer frames. Frame 0 shows the reconstructed room without a visible robot;
the original is retained, and the robot appears from step 1 at 0.2 seconds.
This proves held-out goals within the reconstructed public TUM office. It does
not establish unseen-building transfer, a physical robot, or an operator
camera-input policy. Robot and actuator hashes observed after load do not
establish a complete vendor asset closure pinned before load.

The SDG check downloaded all 6,562 published objects (70,987,686,095 bytes),
verified artifact hashes, decoded every view, and recomputed calibration,
synchronization, metric-depth backprojection, and fused-cloud geometry. The
four 1280×720 cameras follow a supplied 66 m route through NVIDIA's public
warehouse. All 2,034 prepared input files were hash-verified. The
[attribution receipt](synthetic-data-attribution.json) binds the actual captured
input manifest to its public source and license reference. This is sensor-only
capture: the colored points derive from depth images, and the run does not
establish autonomous navigation, collision-free travel, or a physical LiDAR
beam model.

The NuRec check inspected checkpoint counters without executing pickle code,
compared embedded and published training configuration, decoded all 114
validation images and Rerun entities, and verified the 0.25 m rig offset.
All 216 published objects (580,340,666 bytes) were downloaded. The pinned
NVIDIA PPISP capture is attributed under CC BY 4.0, with reconstruction and
display modifications identified in the report. Native metrics describe NRE
validation; appearance reconstruction does not establish collision geometry or
navigation readiness.

Both actual `demo view` commands downloaded bytes identical to the original
native reports. No presentation repair was needed. Offline browser checks at
widths 1440 and 390 exercised each report's timeline with no JavaScript errors,
network requests, or horizontal overflow. The [SDG visual review](synthetic-data-visual-review.json)
and [NuRec visual review](nurec-visual-review.json) record image decoding,
metadata checks, screenshot hashes, and public sample attribution.

GPU observations matched the executing process's run identity and stage
interval. SDG capture had 80 observations, including four with nonzero
utilization; observed peaks were 4,948 MiB and 91%. NuRec reconstruction had
47 observations, including 28 with nonzero utilization; observed peaks were
3,849 MiB and 78%. Sampled peaks are not average utilization. The render stage
had memory observations but no sampled nonzero utilization.

[Earlier qualification](history/2eb77537d/README.md) remains preserved with its
original source, measurements, and NuRec presentation-correction provenance.
The [subsequent RL replay result](rl-improvement-rejection.json) completed full
baseline and candidate training, then improved development success from 82.475%
to 93.0%. It nevertheless failed 371 metric checks across 257 cases, including
warehouse retention. Selection retained the baseline and never evaluated final
outcomes. Its original HTML and independently verified rejection remain evidence
of a failed quality gate. Required native bundles were read back and checked;
this receipt does not claim an inventory of every object or continuous GPU
observation across every phase.

The new baseline-retention recipe still requires its own qualifying final policy
comparison. Execution and physics checks alone do not establish improvement
over a baseline.
Every operator must also pass storage, vendor-access, and selected-runtime
preflight.
