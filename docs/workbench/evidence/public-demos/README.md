# Public demo qualification

The full `synthetic-data` and `nurec` demo commands completed on September 28,
2026 from source `2eb77537d3b09374b8c31457c634d8c1ab036c12`. Both used NVIDIA RTX
PRO 6000 Blackwell GPUs. The receipts below bind measured outputs to source,
workflow, image, staged-source, and artifact hashes without disclosing runtime
infrastructure or storage locations.

| Demo | Independently verified result | Receipt |
| --- | --- | --- |
| Industrial sensor generation | 265 poses, four cameras, 1,060 views, 4,505 capture artifacts, and 976,895,672 valid depth pixels and fused colored points | [SDG qualification](synthetic-data.json) |
| NuRec | Checkpoint at 30,000 steps, 38 rig-offset novel views, PSNR 31.012, SSIM 0.832, and LPIPS 0.268 | [NuRec qualification](nurec.json) |

The SDG check downloaded the complete published input and output trees, verified
artifact hashes, decoded every view, and recomputed calibration, synchronization,
metric-depth backprojection, and fused-cloud geometry. Preparation, capture, and
validation all ran through the common demo command at the recorded revision.
Sensor capture followed a scripted route; it does not establish autonomous
navigation.

The NuRec check inspected checkpoint counters without executing pickle code,
compared embedded and published training configuration, decoded images and Rerun
entities, and verified the 0.25 m rig offset. Native quality metrics describe NRE
validation. They do not establish collision geometry or navigation readiness.

Both actual `demo view` commands exited successfully and downloaded bytes
identical to their canonical reports. Offline browser checks at widths 1440 and
390 exercised each report's timeline with no JavaScript errors or network
requests. GPU observations matched the executing process's run identity and
recorded stage interval. Sampled peaks are observations, not average utilization.

NuRec's original report omitted available metrics because its reader did not
recognize NRE's aggregated metric schema. The corrected report was generated on
CPU from 169 unchanged native input files. The receipt records original and
derived HTML hashes and the correction's provenance hash. The original stored
report remains unchanged; this presentation correction is not another GPU run.

The scan-to-policy and RL improvement demos still require qualifying policy
results. Their current full attempts stopped before learning; runtime completion
and policy improvement must be established independently. Every future operator
must also pass storage, vendor-access, and selected-runtime preflight.
