# Cursor workbench PR readiness

Snapshot: **2026-09-21T04:23:43.973863+00:00**. **53 tracked open PRs; 15 ready to enter the required merge queue.** Required check identity was independently refreshed at 2026-09-21T04:24:15.628677+00:00. No PR was merged by this audit.

These counts include adopted work. They do not mean 53 new PRs in this turn, 53 accepted improvements, or 53 GPU-tested features. Closed superseded #611 and closed experimental #656 do not count as delivered work.

The 15 queue candidates are [#579](https://github.com/nebius/nebius-physical-ai/pull/579), [#581](https://github.com/nebius/nebius-physical-ai/pull/581), [#584](https://github.com/nebius/nebius-physical-ai/pull/584), [#594](https://github.com/nebius/nebius-physical-ai/pull/594), [#596](https://github.com/nebius/nebius-physical-ai/pull/596), [#607](https://github.com/nebius/nebius-physical-ai/pull/607), [#609](https://github.com/nebius/nebius-physical-ai/pull/609), [#613](https://github.com/nebius/nebius-physical-ai/pull/613), [#614](https://github.com/nebius/nebius-physical-ai/pull/614), [#616](https://github.com/nebius/nebius-physical-ai/pull/616), [#621](https://github.com/nebius/nebius-physical-ai/pull/621), [#628](https://github.com/nebius/nebius-physical-ai/pull/628), [#629](https://github.com/nebius/nebius-physical-ai/pull/629), [#640](https://github.com/nebius/nebius-physical-ai/pull/640), [#650](https://github.com/nebius/nebius-physical-ai/pull/650). They are non-draft, target main, have accepted scoped reviews and applicable proof, and their current required checks passed from the required GitHub Actions app. The repository merge queue must still run its integration checks. New commits invalidate this snapshot.

| Workload | Current result | Remaining acceptance |
| --- | --- | --- |
| SeedVR2 | Exact H100 image completed both four-stage workflows; original generated videos and 60 artifact hashes verified. | Frozen quality metrics and real hosted annotation failed; controlled 7B remediation is in progress. |
| RoboCasa | Fresh-container registration failure invalidated the previous image; negative evidence retained. | Correct source/image, six matched pairs, 12 videos, native task metrics and qualified visual review. |
| cuRobo | Real B200 planning ran; strict serialized-trajectory metric audit exposed a numeric bug. Reviewed fix and follow-up diagnostic are in progress. | Accepted rebuilt image, full B200/RTX benchmark coverage, controls and visual review. |
| Open3D | Earlier six-stage CPU workflow completed. Uneven sampling exposed misleading geometry claims; new source/viewer fixes are under review. | Exact repaired image/runtime, native counterexample, and clean viewer proof. |
| AprilTag | CPU detection workload accepted; disclosed visual calibration failed. | Reviewed request/harness repairs and qualified calibration before opening the holdout. |
| Agent metadata | CPU deployment proof accepted; test-only repair keeps production source identical. | Fresh CI; documented mutation-test coverage gaps remain nonblocking. |

GPU tests apply to GPU workloads. CPU concurrency, storage, credential isolation and orchestration repairs use relevant regression, adversarial, process-fault or live CPU evidence. Hosted VLM request tests prove their stated mechanism; failed calibration does not qualify a judge or prove physical correctness.

Open proof: [SeedVR generated videos, objective metrics and failed hosted review](https://github.com/nebius/nebius-physical-ai/blob/0db7db260bbc526061a26c3a16999f94e92b705f/docs/testing/evidence/seedvr-h100-exact-b8d9c9c5/README.md), [agent repair source equivalence and CPU deployment proof](https://github.com/nebius/nebius-physical-ai/blob/b83b862e1469d0bfa22ae7c8bffae11db6b1c9b3/docs/testing/evidence/agent-metadata-test-repair-09d0fa9f/README.md), [Open3D earlier workflow proof and limitations](https://github.com/nebius/nebius-physical-ai/pull/602#issuecomment-5754870500). Prior proof retains its original source and workload scope. Earlier Open3D claims that sample-distance filtering guarantees fabricated geometry are superseded: the retained uneven-coverage counterexample shows that correct geometry can be classified as distant from sparse observations.

The [machine-readable inventory](readiness.json) includes complete source hashes, check summaries, stack dependencies and available review/evidence links.

| PR | Source | Status | Depends on |
| --- | --- | --- | --- |
| [#579 — Bound Workbench I/O concurrency and preserve downloads on failure](https://github.com/nebius/nebius-physical-ai/pull/579) | `dcc530d9a` | Ready for required merge queue | — |
| [#581 — Report a group-writable checkout once instead of 428 scan failures](https://github.com/nebius/nebius-physical-ai/pull/581) | `f0e2ccfe8` | Ready for required merge queue | — |
| [#584 — Add evo trajectory evaluation BYOF workflow](https://github.com/nebius/nebius-physical-ai/pull/584) | `8c6bb6b41` | Ready for required merge queue | — |
| [#593 — feat(workbench): add SeedVR2 video restoration](https://github.com/nebius/nebius-physical-ai/pull/593) | `b8d9c9c56` | Acceptance incomplete | — |
| [#594 — Fail closed on corrupt workflow resume ledgers](https://github.com/nebius/nebius-physical-ai/pull/594) | `93d37b494` | Ready for required merge queue | — |
| [#595 — Fix RoboCasa policy evidence alignment](https://github.com/nebius/nebius-physical-ai/pull/595) | `2a79cbd6e` | Acceptance incomplete | — |
| [#596 — Retain verifiable provenance for VLM evaluations](https://github.com/nebius/nebius-physical-ai/pull/596) | `93e92802e` | Ready for required merge queue | — |
| [#597 — Expose VLM provider and score-gate contradictions](https://github.com/nebius/nebius-physical-ai/pull/597) | `2995156d4` | Accepted stack; ancestors must land first | #596 |
| [#602 — Add Open3D registration and support-distance surface filtering](https://github.com/nebius/nebius-physical-ai/pull/602) | `16012b108` | Acceptance incomplete | — |
| [#607 — Let build provenance through the payload scan instead of refusing the image](https://github.com/nebius/nebius-physical-ai/pull/607) | `c0a125720` | Ready for required merge queue | — |
| [#608 — Guard against module-level GPU imports in tests](https://github.com/nebius/nebius-physical-ai/pull/608) | `0d6b5bb08` | Current CI / review closure | — |
| [#609 — Recover stage log attribution from durable waves](https://github.com/nebius/nebius-physical-ai/pull/609) | `e962e6fbd` | Ready for required merge queue | — |
| [#610 — Add AprilTag fiducial detection BYOF workflow](https://github.com/nebius/nebius-physical-ai/pull/610) | `c725a44af` | Acceptance incomplete | — |
| [#611 — Preserve active SkyPilot jobs during cleanup](https://github.com/nebius/nebius-physical-ai/pull/611) | `87df770d1` | Closed: superseded by #614 | — |
| [#612 — Retain source-frame sampling provenance for VLM requests](https://github.com/nebius/nebius-physical-ai/pull/612) | `d116cbbcb` | Accepted stack; ancestors must land first | #597 |
| [#613 — Keep declarative workflow outputs visible](https://github.com/nebius/nebius-physical-ai/pull/613) | `54d61f256` | Ready for required merge queue | — |
| [#614 — Fail closed on contradictory cancellation state](https://github.com/nebius/nebius-physical-ai/pull/614) | `b1123f912` | Ready for required merge queue | — |
| [#615 — Fail closed on corrupt submission receipts](https://github.com/nebius/nebius-physical-ai/pull/615) | `eea74da2e` | Current CI / review closure | — |
| [#616 — test: stop the unit suite writing into the operator's real ~/.npa](https://github.com/nebius/nebius-physical-ai/pull/616) | `d5798ab09` | Ready for required merge queue | — |
| [#617 — Use backend-neutral VLM result artifact names](https://github.com/nebius/nebius-physical-ai/pull/617) | `e232f713f` | Accepted stack; ancestors must land first | #612 |
| [#619 — Unify managed-job terminality checks](https://github.com/nebius/nebius-physical-ai/pull/619) | `2e1805c5a` | Accepted stack; ancestors must land first | #614 |
| [#620 — Guard workflow teardown allowance invariants](https://github.com/nebius/nebius-physical-ai/pull/620) | `788726b34` | Accepted stack; ancestors must land first | #619 |
| [#621 — Complete workflow image preflight across decision paths](https://github.com/nebius/nebius-physical-ai/pull/621) | `4369e175c` | Ready for required merge queue | — |
| [#622 — Preserve paired VLM judge disagreements](https://github.com/nebius/nebius-physical-ai/pull/622) | `f4f4cb7aa` | Accepted stack; ancestors must land first | #617 |
| [#623 — Sanitize workflow CLI failure output](https://github.com/nebius/nebius-physical-ai/pull/623) | `525007e2a` | Accepted stack; ancestors must land first | #615 |
| [#624 — feat: expose VLM benchmark confusion failures](https://github.com/nebius/nebius-physical-ai/pull/624) | `b88de92b0` | Accepted stack; ancestors must land first | #622 |
| [#625 — feat(curobo): qualify auditable motion-planning image](https://github.com/nebius/nebius-physical-ai/pull/625) | `d0a606fc9` | Acceptance incomplete | — |
| [#626 — Report workflow image preflight planning failures](https://github.com/nebius/nebius-physical-ai/pull/626) | `d15aa2439` | Accepted stack; ancestors must land first | #615, #621, #623 |
| [#627 — Add blinded, order-balanced VLM preference audits](https://github.com/nebius/nebius-physical-ai/pull/627) | `46f50ad77` | Accepted stack; ancestors must land first | #624 |
| [#628 — Fail closed on unreadable workflow stage status](https://github.com/nebius/nebius-physical-ai/pull/628) | `96b9147b9` | Ready for required merge queue | — |
| [#629 — Scan every page for workflow prefix outputs](https://github.com/nebius/nebius-physical-ai/pull/629) | `803115985` | Ready for required merge queue | — |
| [#630 — Reuse complete outputs before recovery limits](https://github.com/nebius/nebius-physical-ai/pull/630) | `48220e837` | Accepted stack; ancestors must land first | #629 |
| [#631 — Keep pending workflow artifact lookups exact](https://github.com/nebius/nebius-physical-ai/pull/631) | `fa621c005` | Accepted stack; ancestors must land first | #613 |
| [#632 — Block workflow output reuse on identity drift](https://github.com/nebius/nebius-physical-ai/pull/632) | `281fd30bf` | Stack needs dependency / current-check review | #630 |
| [#633 — fix(workflow): preserve provider status during output reuse](https://github.com/nebius/nebius-physical-ai/pull/633) | `16601ddbc` | Accepted stack; ancestors must land first | #632 |
| [#634 — Require visible terminal evidence in VLM rollout gates](https://github.com/nebius/nebius-physical-ai/pull/634) | `52cd330b3` | Accepted stack; ancestors must land first | #627 |
| [#635 — fix(workflow): reconcile blocked waves before resume](https://github.com/nebius/nebius-physical-ai/pull/635) | `edfe23a7a` | Accepted stack; ancestors must land first | #633 |
| [#637 — fix(workflow): resume verified output reuse after crash](https://github.com/nebius/nebius-physical-ai/pull/637) | `c3e91b816` | Accepted stack; ancestors must land first | #635 |
| [#638 — fix(skypilot): contain listener exit race](https://github.com/nebius/nebius-physical-ai/pull/638) | `f6fbad609` | Accepted stack; ancestors must land first | #635 |
| [#639 — fix(workflow): fence completed replay identity](https://github.com/nebius/nebius-physical-ai/pull/639) | `abbe30736` | Accepted stack; ancestors must land first | #637 |
| [#640 — fix(skypilot): link default ~/.kube/config into isolated HOME when KUBECONFIG is unset](https://github.com/nebius/nebius-physical-ai/pull/640) | `ca6a515d2` | Ready for required merge queue | — |
| [#641 — fix(workflow): bind image identity to references](https://github.com/nebius/nebius-physical-ai/pull/641) | `d28b270d4` | Accepted stack; ancestors must land first | #639 |
| [#643 — fix(workflow): bind image selection identity](https://github.com/nebius/nebius-physical-ai/pull/643) | `7a6454add` | Accepted stack; ancestors must land first | #641 |
| [#644 — fix(workflow): reject unmatched image overrides](https://github.com/nebius/nebius-physical-ai/pull/644) | `204e126ae` | Accepted stack; ancestors must land first | #643 |
| [#646 — feat(ncore): qualify full COLMAP conversion for NRE](https://github.com/nebius/nebius-physical-ai/pull/646) | `fe85f6d75` | Acceptance incomplete | — |
| [#647 — Add audit-only rich visual VLM reviews](https://github.com/nebius/nebius-physical-ai/pull/647) | `c958ac2a2` | Accepted stack; ancestors must land first | #634 |
| [#649 — Studio: expose illustrative film-review cues](https://github.com/nebius/nebius-physical-ai/pull/649) | `788b4e632` | Accepted stack; ancestors must land first | #558 |
| [#650 — fix(sim2real): gate submit preflight on all sim2real gated HF repos](https://github.com/nebius/nebius-physical-ai/pull/650) | `47f7fea38` | Ready for required merge queue | — |
| [#651 — docs(sim2real): record Token Factory evaluator decision (NVIDIA TF models are text-only)](https://github.com/nebius/nebius-physical-ai/pull/651) | `fd4e9cd3f` | Current CI / review closure | — |
| [#652 — fix(skypilot): retry ambiguous Kubernetes readiness evidence instead of failing the launch](https://github.com/nebius/nebius-physical-ai/pull/652) | `cd196a85a` | Current CI / review closure | — |
| [#653 — fix(workflow): redact durable diagnostics](https://github.com/nebius/nebius-physical-ai/pull/653) | `c3f53eaa7` | Accepted stack; ancestors must land first | #623 |
| [#654 — Harden agent metadata credential request paths](https://github.com/nebius/nebius-physical-ai/pull/654) | `09d0fa9f3` | Acceptance incomplete | — |
| [#659 — fix(skypilot): retain Nebius storage for Kubernetes jobs](https://github.com/nebius/nebius-physical-ai/pull/659) | `f1751358c` | Acceptance incomplete | — |
| [#678 — Fix Kimi-K3 hosted VLM completion contract](https://github.com/nebius/nebius-physical-ai/pull/678) | `a26a868e3` | Acceptance incomplete | — |
