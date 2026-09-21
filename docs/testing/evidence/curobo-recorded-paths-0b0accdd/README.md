# Recorded cuRobo paths and visual-review failures

These are factual plots and Rerun recordings derived from real B200 and RTX PRO 6000 runs. They show declared goals, recorded tool paths and planner status. They do not display robot geometry or obstacles and do not establish collision freedom, torque validity or physical robot safety.

![Recorded B200 path](b200-golden-feasible-v3.png)

Open or download the recordings: [B200 feasible/blocked controls](b200-golden.rrd), [RTX PRO 6000 feasible/blocked controls](rtx-golden.rrd), [eight selected B200 benchmark records](b200-benchmark-selected.rrd). Rerun 0.31.4 decodes all three. Browser access is checked separately after publication.

The benchmark sample uses first success/first failure in pinned input order for each dataset/mode cell. Where no dynamics MotionBenchMaker failure occurred, the last success is explicitly labelled as the fallback. These eight records are a selected B200 sample; the full RTX benchmark is not visually represented here. All failures and invalid inputs remain in the [full numerical matrix report](../curobo-full-matrices-49fed65d/README.md).

| Hardware | Full input count | Successful | Failed | Invalid |
| --- | --- | --- | --- | --- |
| B200 | 5,200 | 5,162 | 18 | 20 |
| RTX PRO 6000 | 5,200 | 5,166 | 14 | 20 |

Both numerical matrices passed the frozen benchmark gates. Both original managed workflows failed during visualization. The repaired consumer subsequently decoded the complete retained journals on CPU; this preserves the original failures and does not claim a successful rerun of those workflows. Final source `0b0accdd35406189a82bc3e2595ea3cfdef04195` passed 24,243 local tests, with 154 skips and one non-strict xpass. That suite excludes live/GPU/e2e tests; the GPU matrix evidence is separate. Qualification of the newly built private image remains pending.

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
