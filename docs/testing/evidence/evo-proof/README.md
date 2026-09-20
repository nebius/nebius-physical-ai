# PR #584: retained execution proof

Independent evidence audit for candidate
`c0e53c67f99075f8fc75fd24997ca4b6ce849353`, checked on 2026-09-20 UTC.
This proves the advertised CPU trajectory-evaluation workload through a direct
SkyPilot BYOF launch on Kubernetes. A GPU is not part of this workload. It does
not prove execution through an outer `npa.workflow` launch, whose BYOF entry is
classified as plan-only because nested SkyPilot launch is unsupported.

The native `evo_ape`, `evo_rpe`, and `evo_traj` commands executed successfully.
The malformed-input negative control exited 1 and produced no result archive.
All ten result archives were independently decoded; their finite error arrays
reproduce the reported root-mean-square errors. All 80 files in the frozen
evidence inventory match their recorded SHA-256 values.

| Input | APE errors | APE RMSE (m) | RPE errors | RPE RMSE (m) | Result |
| --- | ---: | ---: | ---: | ---: | --- |
| Exact synthetic reference replay | 120 | 7.175037e-16 | 110 | 7.454746e-16 | Accepted |
| Bounded synthetic perturbation | 120 | 0.003545434 | 110 | 0.005949814 | Accepted |
| Nonlinear synthetic drift | 120 | 0.443811606 | 110 | 0.142876707 | Rejected |
| Malformed trajectory | — | — | — | — | Rejected |
| Pinned upstream KITTI 00 ORB | 4,541 | 1.303449715 | 4,458 | 1.250926005 | Observed compatibility result |
| Pinned upstream KITTI 00 S-PTAM | 4,541 | 3.738487908 | 4,458 | 2.720131609 | Observed compatibility result |

APE and RPE measure translation after SE(3) Umeyama alignment. Synthetic RPE
uses a 10-frame delta; KITTI RPE uses a 100-metre delta, both with all eligible
pairs. Their RPE numbers therefore describe different evaluation settings.
The frozen synthetic acceptance thresholds were APE RMSE ≤ 0.05 m, RPE RMSE
≤ 0.02 m, and at least 100 matched errors. Controls yield TP=2, TN=2, FP=0,
FN=0. These thresholds were not tuned to the KITTI outcomes.

The attached control plot uses wholly generated trajectories: a 5-metre circular
path with sinusoidal vertical motion, a bounded lateral perturbation, and a
quadratic lateral drift. All four control input files were independently
regenerated from those formulas and match the live-run bytes exactly. No KITTI
poses or customer sensor data enter this plot. Its PNG metadata contains only
Matplotlib/software and image-density fields; no private infrastructure labels,
credentials, or appended payloads were found.

KITTI-derived graphics and trajectory arrays are excluded from this public
package. The publisher specifies CC BY-NC-SA 3.0 and an academic-use statement;
the audit did not establish a compatible basis for republishing those derived
graphics here. The table retains measured evaluation facts with source
attribution: Andreas Geiger, Philip Lenz, and Raquel Urtasun, *Are we ready for
Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012.
[KITTI copyright and citation](https://www.cvlibs.net/datasets/kitti/).

The privately retained hosted visual review used `MiniMaxAI/MiniMax-M3`. Its claim is
**plot identity and readability**, not trajectory quality or robot performance.
An initial unlabeled cross-swap test false-passed twice; that failure remains
retained. After adding source-linked identity banners and an identify-first
rubric, both cross-swaps were rejected at 0.0 and the final labeled plots scored
0.95 each against an unchanged 0.8 threshold. Provider request IDs and usage
were unavailable in the retained normalized client responses; this audit did
not verify the calls against provider records or issue new model calls.

The producing NPA revision was `2f872775be4b7f116c0b0b8195c45ef2c4dbcca0`.
The workflow bytes are identical at that revision and the reviewed candidate:

| Identity | SHA-256 or revision |
| --- | --- |
| Upstream evo v1.35.1 source | `8dd6cfe0ec1747f9e1b5b569edd82c54d1a3f422` |
| Candidate workflow | `1c993736c780e98075abae3a1de3b039d92b7f176da72907e04d9872e0c262e8` |
| Executed smoke command | `502b7e482ce94ce8309a80b7e8ce0e229ffbff6ef0f05ad00b5d621de9403f7d` |
| OCI image index | `sha256:f21e645b49039864bd5ab9beec85f97e5b3a6d80230c011aecff58972596e948` |
| Executed linux/amd64 image manifest | `sha256:5383f081fc34f4deb259f647ee4ee88bb349c95e1d9bdf642968e5f880637274` |
| Frozen evidence manifest | `de0896669739c3f8e7401dfdd41a74a3b41bc6b34bdbffbcdab68833f0c5f121` |
| Objective evaluation plan | `986f4caa6c5211ee1f5c106e51a7317307edca34f7fe2c0f3b72b8b5d5a8d347` |

The registry index was independently inspected and binds that platform manifest
to the executed image index. Its complete-byte restricted-payload scan reports
22,826 entries, a complete scan, and no payload/history/weight-shaped findings.
The image remains an operator-built, non-public candidate; the scan is not a
general security or redistribution approval.

A separate reviewer approved source identity, objective measurements, controls,
and evidence integrity at this candidate. This additional audit recomputed the
raw arrays and hashes independently. Merge readiness still requires completed
CI, the full applicable test suite, and the remaining architecture/acceptance
review. Evidence approval alone is not merge approval.

This demonstrates a useful trajectory gate on the stated controls. It does not
establish real-world SLAM quality, navigation success, sensor accuracy, robot
safety, generalization to other formats, or an improvement to the upstream
trajectory estimates themselves.

## Synthetic control plot

![Wholly generated reference, bounded perturbation, and nonlinear drift controls.](plots/synthetic_controls_trajectories.png)
