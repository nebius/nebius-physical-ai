# Recorded cuRobo paths and visual-review failures

These are factual plots and Rerun recordings derived from real B200 and RTX PRO 6000 runs. They show declared goals, recorded tool paths and planner status. They do not display robot geometry or obstacles and do not establish collision freedom, torque validity or physical robot safety.

![Recorded B200 path](b200-golden-feasible-v3.png)

Open an interactive viewer, with no login or download required:

- [b200-golden](https://app.rerun.io/version/0.31.4/index.html?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnebius%2Fnebius-physical-ai%2F2e8b31ac60c0d4614ee22008b502d8270571a379%2Fdocs%2Ftesting%2Fevidence%2Fcurobo-recorded-paths-0b0accdd%2Fb200-golden.rrd&url=https%3A%2F%2Fraw.githubusercontent.com%2Fnebius%2Fnebius-physical-ai%2F3ec222b7f08652a6082a1a56ecb5fbf7dc7f083f%2Fdocs%2Ftesting%2Fevidence%2Fcurobo-recorded-paths-0b0accdd%2Fcurobo-review-fit.rbl&persist=false)
- [rtx-golden](https://app.rerun.io/version/0.31.4/index.html?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnebius%2Fnebius-physical-ai%2F2e8b31ac60c0d4614ee22008b502d8270571a379%2Fdocs%2Ftesting%2Fevidence%2Fcurobo-recorded-paths-0b0accdd%2Frtx-golden.rrd&url=https%3A%2F%2Fraw.githubusercontent.com%2Fnebius%2Fnebius-physical-ai%2F3ec222b7f08652a6082a1a56ecb5fbf7dc7f083f%2Fdocs%2Ftesting%2Fevidence%2Fcurobo-recorded-paths-0b0accdd%2Fcurobo-review-fit.rbl&persist=false)
- [b200-benchmark-selected](https://app.rerun.io/version/0.31.4/index.html?url=https%3A%2F%2Fraw.githubusercontent.com%2Fnebius%2Fnebius-physical-ai%2F2e8b31ac60c0d4614ee22008b502d8270571a379%2Fdocs%2Ftesting%2Fevidence%2Fcurobo-recorded-paths-0b0accdd%2Fb200-benchmark-selected.rrd&url=https%3A%2F%2Fraw.githubusercontent.com%2Fnebius%2Fnebius-physical-ai%2F3ec222b7f08652a6082a1a56ecb5fbf7dc7f083f%2Fdocs%2Ftesting%2Fevidence%2Fcurobo-recorded-paths-0b0accdd%2Fcurobo-review-fit.rbl&persist=false)

The viewer may start on the last failed case. To inspect the first successful case, double-click the problem-index number at the bottom right, select all, enter **0**, then press Enter. The large view shows the tool path and declared goal; the right-hand tabs show status and provenance. The camera includes every recorded path and goal. Use the timeline to inspect the retained failures too.

All three links were opened in fresh anonymous Chrome contexts. Recording and blueprint fetches returned HTTP 200 with exact expected hashes, and the index-zero instructions were exercised through the actual UI. [Browser receipts](browser-verification.json) and screenshots: [B200 control](b200-golden-viewer-index-zero.png), [RTX control](rtx-golden-viewer-index-zero.png), [B200 benchmark sample](b200-benchmark-selected-viewer-index-zero.png). These are access/presentation checks, not a VLM acceptance pass.

Download the unchanged recordings: [B200 controls](b200-golden.rrd), [RTX PRO 6000 controls](rtx-golden.rrd), [eight selected B200 benchmark records](b200-benchmark-selected.rrd). [Camera bounds](camera-bounds.json) derive from actual recorded data; the separate [blueprint](curobo-review-fit.rbl) contains presentation settings only.

The benchmark sample uses first success/first failure in pinned input order for each dataset/mode cell. Where no dynamics MotionBenchMaker failure occurred, the last success is explicitly labelled as the fallback. These eight records are a selected B200 sample; the full RTX benchmark is not visually represented here. All failures and invalid inputs remain in the [full numerical matrix report](../curobo-full-matrices-49fed65d/README.md).

| Hardware | Full input count | Successful | Failed | Invalid |
| --- | --- | --- | --- | --- |
| B200 | 5,200 | 5,162 | 18 | 20 |
| RTX PRO 6000 | 5,200 | 5,166 | 14 | 20 |

Both numerical matrices passed the frozen benchmark gates. Both original managed workflows failed during visualization. The repaired consumer subsequently decoded the complete retained journals on CPU; this preserves the original failures and does not claim a successful rerun of those workflows. Final source `0b0accdd35406189a82bc3e2595ea3cfdef04195` passed 24,243 local tests, with 154 skips and one non-strict xpass. That suite excludes live/GPU/e2e tests; the GPU matrix evidence is separate. The exact new private image also passed full retained-journal CPU replay and recording decode, runtime/dependency checks, the fixable-CRITICAL vulnerability gate, an all-severity final-filesystem secret scan and normalized SPDX validation. Complete image-byte policy acceptance remains pending. GPU inference has not run on the new image; its unchanged numerical runner is bound to the original GPU producer.

**The visual acceptance gate failed.** There were 60 real hosted-model calls and 56,535 returned tokens across calibration and prospective holdout trials. No judge passed calibration plus a fresh sealed holdout; consequently there were zero final acceptance calls. MiniMax false-passed an acknowledged kinematic/dynamics mismatch and a clipped 44/81-sample path. MiniCPM and Gemma failed calibration. A revised explicit-observation protocol still produced malformed JSON and false negatives. These results are retained as failures, not counted as accepted visual quality.

The plots use the same camera orientation and per-capture equal-axis bounds over the complete path and goal. Earlier global bounds obscured a short valid path and were superseded; that failed trial remains retained. The [machine-readable review](review.json) records exact artifact hashes, selection, counts and limitations. All recording values and metadata were decoded for the confidentiality review, and all PNG metadata was checked before publication.

## All recorded captures

- [B200 / golden controls / kinematic / feasible](b200-golden-feasible-v3.png)
- [B200 / golden controls / kinematic / blocked](b200-golden-blocked-v3.png)
- [RTX_PRO_6000 / golden controls / kinematic / feasible](rtx_pro_6000-golden-feasible-v3.png)
- [RTX_PRO_6000 / golden controls / kinematic / blocked](rtx_pro_6000-golden-blocked-v3.png)
- [B200 / motion_benchmaker / kinematic / success](b200-motion-benchmaker-kinematic-success-v3.png)
- [B200 / motion_benchmaker / kinematic / failed](b200-motion-benchmaker-kinematic-failed-v3.png)
- [B200 / motion_benchmaker / dynamics / success](b200-motion-benchmaker-dynamics-success-v3.png)
- [B200 / motion_benchmaker / dynamics / last_success_no_failure_observed](b200-motion-benchmaker-dynamics-last-success-no-failure-observed-v3.png)
- [B200 / mpinets / kinematic / success](b200-mpinets-kinematic-success-v3.png)
- [B200 / mpinets / kinematic / failed](b200-mpinets-kinematic-failed-v3.png)
- [B200 / mpinets / dynamics / success](b200-mpinets-dynamics-success-v3.png)
- [B200 / mpinets / dynamics / failed](b200-mpinets-dynamics-failed-v3.png)

[SHA-256 checksums](SHA256SUMS) bind the downloadable artifacts. No customer credential, operational resource ID or private storage URL is included.
