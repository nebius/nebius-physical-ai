# Public demo qualification

The full `synthetic-data` and `nurec` demo commands completed on September 28,
2026 from source `39a662651f6b46db0ce57dbff5fb9828c9d22643`. Both used NVIDIA RTX
PRO 6000 Blackwell GPUs. The receipts bind measured outputs to source, workflow,
image, staged-source, and artifact hashes without disclosing runtime
infrastructure or storage locations.

| Demo | Independently verified result | Receipt |
| --- | --- | --- |
| Industrial sensor generation | 265 poses, four cameras, 1,060 views, 4,505 capture artifacts, and 976,895,672 valid depth pixels and fused colored points | [SDG qualification](synthetic-data.json) |
| NuRec | Checkpoint at 30,000 steps, 38 rig-offset novel views, PSNR 31.025, SSIM 0.832, and LPIPS 0.267 | [NuRec qualification](nurec.json) |

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
The scan-to-policy and RL improvement demos still require qualifying policy
results. Execution and physics checks alone do not establish useful navigation
or improvement over a baseline. Every operator must also pass storage,
vendor-access, and selected-runtime preflight.
