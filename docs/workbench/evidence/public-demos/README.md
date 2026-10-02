# Public demo qualification

The full scan-to-policy demo is qualified from source
`de63c25f4009aa09b6ec68710170c462bedbd20c` on NVIDIA RTX PRO 6000 Blackwell GPUs.
The fresh run qualifies the updated shared navigation runtime. SDG and NuRec
retain their full qualification from source
`39a662651f6b46db0ce57dbff5fb9828c9d22643`: their workload code, selected inputs,
image selections and rendered tasks remain unchanged. Scan qualification
completed on September 29, 2026; SDG and NuRec completed on September 28.

All four workflow implementations are present. Three demos have passed their
scientific quality checks. The RL workflow has verified full training, comparison
and rejection behavior; its candidate did not qualify for promotion. These
measurements support reviewing the workflow implementation separately from the
unproven claim of policy improvement.

The receipts bind measured outputs to recorded source, image, workflow and
artifact evidence without exposing runtime infrastructure or storage locations.
The [earlier scan qualification](history/39a662651/README.md) remains preserved
with its original source and report hashes.

| Demo | Independently verified result | Receipt |
| --- | --- | --- |
| Scan-to-policy navigation | 4,000 robots in one measured scene, 500 training updates, and 3,386/4,000 successful held-out routes (84.65%, above the unchanged 80% gate) | [Scan qualification](real-to-sim.json) |
| Industrial sensor generation | 265 poses, four cameras, 1,060 views, 4,505 capture artifacts, and 976,895,672 valid depth pixels and fused colored points | [SDG qualification](synthetic-data.json) |
| NuRec | Checkpoint at 30,000 steps, 38 rig-offset novel views, PSNR 31.025, SSIM 0.832, and LPIPS 0.267 | [NuRec qualification](nurec.json) |
| RL comparison and rejection | Full 1,500 baseline + 1,500 new candidate updates; development success 82.475% → 82.05%; 699 violations across 399 cases; baseline retained, final not evaluated | [Current rejection evidence](rl-improvement-rejection.json) |

The scan check independently recomputed every held-out trajectory and verified
all 500 PPO updates, 16,000,000 training transitions, changed actor and critic
parameters, and native optimizer state. Four fresh-process physical controls
passed separately for training and evaluation. All 4,000 robots share one
measured collision scene and use static range observations that exclude peers.
The check read back nine publication bundles containing 5,529 files
(2,234,165,143 bytes); these are outputs from eight executable workflow states.

The [scan visual review](real-to-sim-visual-review.json) verifies the original
native report and byte-identical `demo view` download. The HTML shows the scored
focal episode through step 7 (eight frames); the full 21-frame video spans
4.2 seconds and also contains later observer frames. All 4,000 episodes reached
their terminal events by native step 20 within the unchanged 300-step maximum.
Frame 0 shows the reconstructed room without a visible robot;
the original is retained, and the robot appears from step 1 at 0.2 seconds.
This proves held-out goals within the reconstructed public TUM office. It does
not establish unseen-building transfer, a physical robot, or an operator
camera-input policy. Robot and actuator hashes observed after load do not
establish a complete vendor asset closure pinned before load.

The scan GPU check binds 557 training observations and four evaluation
observations to the exact owned pod/container while each sealed native main
process was active. This is sampled device activity, not continuous coverage,
a host/container PID map or per-policy allocation accounting. The native driver
completed successfully. Two original external policy-auditor invocations failed
to import the frozen source; the same auditors passed after their CPU environment
explicitly selected that source payload. The failed invocations and original
watcher failure remain preserved. No native source, budget or quality gate changed.

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
The [earlier full RL replay rejection](history/6da3c2aaa/README.md) remains
preserved with its original 82.475% → 93.0% result and 371 violations across
257 cases. The current frozen-baseline penalty experiment is recorded separately.

The [current RL rejection](rl-improvement-rejection.json) completed full baseline
and candidate training: 1,500 updates and 48 million transitions each, with
4,000 robots. It observed 1,488 simulation failures among 4,000 office training
routes, admitted the predeclared public capture, reconstructed its exact frozen
geometry, and continued from the baseline model and optimizer. The original
teacher, fixed coefficient 10.0 and all 1,500 KL records were verified. Twenty
fresh controls covered candidate learning and training-exposed diagnostics;
eight additional controls covered development. Training diagnostics do not
select or promote a policy.

On 4,000 development routes per policy, successes declined from 3,299 to 3,282
(82.475% → 82.05%, −0.425 percentage points). Both absolute 80% checks passed,
but the improvement, regional retention and per-case guards failed: 699 metric
violations across 399 cases. The workflow published its rejection report,
retained the baseline and exited unsuccessfully at selection. The unchanged
final-access guard rejected this candidate before any final evaluation.
No policy was deployed. The penalty has not established effective promotion.

The [RL report review](rl-improvement-visual-review.json) records the unchanged
native HTML, actual byte-identical `demo view` result and offline browser checks.
All 23 scored JPEGs were decoded; public-source attribution and display
modifications are present. One raw infrastructure-pattern match occurred only
inside compressed JPEG base64 and was resolved by exact-byte structure,
metadata and surrounding-text checks. Two initial independent report
regeneration attempts changed JSON key ordering; both failures were preserved,
and the original constructors reproduced the original HTML/result exactly.
No product source, cohort, training budget or quality threshold changed.

Sampled GPU evidence binds all nine native training/evaluation roles to the
recorded main process and owned container/image: 328 baseline-training, 453
candidate-training and 59 evaluation device samples. All nine roles have
observed activity. Five after-load public asset samples do not establish a
complete asset closure or pre-load pinning. These samples do not prove
continuous activity, host/container PID equivalence or in-process imports.

Required native archives and publication records were checked. Their receipts
state the exact inventories; they do not claim every object under the run prefix
or continuous GPU observation. Every operator must still pass storage,
vendor-access and selected-runtime preflight.
