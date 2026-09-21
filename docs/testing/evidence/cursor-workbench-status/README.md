# Cursor workbench PR readiness

GitHub snapshot: **2026-09-21T02:07:59.226812+00:00**; targeted state refresh: **2026-09-21T02:13:01.884549+00:00**.

**49 historically tracked PRs; 48 open; 13 candidates for the required merge queue.** Closed [#611](https://github.com/nebius/nebius-physical-ai/pull/611#issuecomment-5754494671) is wholly superseded by #614 and adds no delivered improvement. This audit performed no merge.

Queue candidates: #581, #584, #594, #596, #607, #609, #613, #614, #616, #621, #628, #629, #640. They have accepted exact-source records and applicable checks green. The required merge queue still runs integration checks; new commits need renewed checks and review.

Fresh independent proof: [#640 filesystem-link behavior and 151 passing tests](independent-cpu-reviews/pr640/README.md), and [#644 selector mutation controls and 71 passing tests](independent-cpu-reviews/pr644/README.md). #640 joins the queue candidates. #644 remains dependent on #643 and is not directly ready for main.

#579 at `dcc530d9adea4fbec2f656af6f64bddcc57d507c` and #615 at `eea74da2e0e084293e3facb4f2ed735b8c35b6d7` have exact independent acceptance and no observed conflict; current CI remains incomplete. #615 acceptance now matches its published successor, rather than its older conflicting head.

GPU acceptance remains incomplete for SeedVR2, RoboCasa and cuRobo. Open3D still needs the standard workflow runtime; AprilTag has failed visual calibration. CPU workflow and test-isolation changes use relevant fault-injection, storage and regression evidence. These CPU results make no new GPU, visual-quality or robot-safety claim.

The [machine-readable record](readiness.json) includes full source hashes, dependency PRs and available public proof/review links. Failed evidence remains failed. This inventory reconciles retained independent reviews; it does not claim every earlier experiment was repeated.

| PR | Exact source | Status | Depends on |
| --- | --- | --- | --- |
| [#579 — Bound Workbench I/O concurrency and preserve downloads on failure](https://github.com/nebius/nebius-physical-ai/pull/579) | `dcc530d9a` | Exact review accepted; current CI incomplete | — |
| [#581 — Report a group-writable checkout once instead of 428 scan failures](https://github.com/nebius/nebius-physical-ai/pull/581) | `f0e2ccfe8` | Ready for required merge queue | — |
| [#584 — Add evo trajectory evaluation BYOF workflow](https://github.com/nebius/nebius-physical-ai/pull/584) | `8c6bb6b41` | Ready for required merge queue | — |
| [#593 — feat(workbench): add SeedVR2 video restoration](https://github.com/nebius/nebius-physical-ai/pull/593) | `b8d9c9c56` | Full workload evidence incomplete | — |
| [#594 — Fail closed on corrupt workflow resume ledgers](https://github.com/nebius/nebius-physical-ai/pull/594) | `93d37b494` | Ready for required merge queue | — |
| [#595 — Fix RoboCasa policy evidence alignment](https://github.com/nebius/nebius-physical-ai/pull/595) | `2a79cbd6e` | Full workload evidence incomplete | — |
| [#596 — Retain verifiable provenance for VLM evaluations](https://github.com/nebius/nebius-physical-ai/pull/596) | `93e92802e` | Ready for required merge queue | — |
| [#597 — Expose VLM provider and score-gate contradictions](https://github.com/nebius/nebius-physical-ai/pull/597) | `2995156d4` | Accepted; dependency must land first | #596 |
| [#602 — Add Open3D registration and reject reconstructed surface the scan never supported](https://github.com/nebius/nebius-physical-ai/pull/602) | `aa3c9e602` | Full workload evidence incomplete | — |
| [#607 — Let build provenance through the payload scan instead of refusing the image](https://github.com/nebius/nebius-physical-ai/pull/607) | `c0a125720` | Ready for required merge queue | — |
| [#608 — Guard against module-level GPU imports in tests](https://github.com/nebius/nebius-physical-ai/pull/608) | `0d6b5bb08` | Current review/CI closure | — |
| [#609 — Recover stage log attribution from durable waves](https://github.com/nebius/nebius-physical-ai/pull/609) | `e962e6fbd` | Ready for required merge queue | — |
| [#610 — Add AprilTag fiducial detection BYOF workflow](https://github.com/nebius/nebius-physical-ai/pull/610) | `c725a44af` | Full workload evidence incomplete | — |
| [#611 — Preserve active SkyPilot jobs during cleanup](https://github.com/nebius/nebius-physical-ai/pull/611) | `87df770d1` | Closed; superseded by #614; not counted as delivered | — |
| [#612 — Retain source-frame sampling provenance for VLM requests](https://github.com/nebius/nebius-physical-ai/pull/612) | `d116cbbcb` | Accepted; dependency must land first | #597 |
| [#613 — Keep declarative workflow outputs visible](https://github.com/nebius/nebius-physical-ai/pull/613) | `54d61f256` | Ready for required merge queue | — |
| [#614 — Fail closed on contradictory cancellation state](https://github.com/nebius/nebius-physical-ai/pull/614) | `b1123f912` | Ready for required merge queue | — |
| [#615 — Fail closed on corrupt submission receipts](https://github.com/nebius/nebius-physical-ai/pull/615) | `eea74da2e` | Exact review accepted; current CI incomplete | — |
| [#616 — test: stop the unit suite writing into the operator's real ~/.npa](https://github.com/nebius/nebius-physical-ai/pull/616) | `d5798ab09` | Ready for required merge queue | — |
| [#617 — Use backend-neutral VLM result artifact names](https://github.com/nebius/nebius-physical-ai/pull/617) | `e232f713f` | Accepted; dependency must land first | #612 |
| [#619 — Unify managed-job terminality checks](https://github.com/nebius/nebius-physical-ai/pull/619) | `2e1805c5a` | Accepted; dependency must land first | #614 |
| [#620 — Guard workflow teardown allowance invariants](https://github.com/nebius/nebius-physical-ai/pull/620) | `788726b34` | Accepted; dependency must land first | #619 |
| [#621 — Complete workflow image preflight across decision paths](https://github.com/nebius/nebius-physical-ai/pull/621) | `4369e175c` | Ready for required merge queue | — |
| [#622 — Preserve paired VLM judge disagreements](https://github.com/nebius/nebius-physical-ai/pull/622) | `f4f4cb7aa` | Accepted; dependency must land first | #617 |
| [#623 — Sanitize workflow CLI failure output](https://github.com/nebius/nebius-physical-ai/pull/623) | `525007e2a` | Accepted; dependency must land first | #615 |
| [#624 — feat: expose VLM benchmark confusion failures](https://github.com/nebius/nebius-physical-ai/pull/624) | `b88de92b0` | Accepted; dependency must land first | #622 |
| [#625 — feat(curobo): qualify auditable motion-planning image](https://github.com/nebius/nebius-physical-ai/pull/625) | `b12c10051` | Full workload evidence incomplete | — |
| [#626 — Report workflow image preflight planning failures](https://github.com/nebius/nebius-physical-ai/pull/626) | `d15aa2439` | Accepted; dependency must land first | #621, #615, #623 |
| [#627 — Add blinded, order-balanced VLM preference audits](https://github.com/nebius/nebius-physical-ai/pull/627) | `2bf755b31` | Accepted; dependency must land first | #624 |
| [#628 — Fail closed on unreadable workflow stage status](https://github.com/nebius/nebius-physical-ai/pull/628) | `96b9147b9` | Ready for required merge queue | — |
| [#629 — Scan every page for workflow prefix outputs](https://github.com/nebius/nebius-physical-ai/pull/629) | `803115985` | Ready for required merge queue | — |
| [#630 — Reuse complete outputs before recovery limits](https://github.com/nebius/nebius-physical-ai/pull/630) | `48220e837` | Accepted; dependency must land first | #629 |
| [#631 — Keep pending workflow artifact lookups exact](https://github.com/nebius/nebius-physical-ai/pull/631) | `fa621c005` | Accepted; dependency must land first | #613 |
| [#632 — Block workflow output reuse on identity drift](https://github.com/nebius/nebius-physical-ai/pull/632) | `281fd30bf` | Accepted; dependency must land first | #630 |
| [#633 — fix(workflow): preserve provider status during output reuse](https://github.com/nebius/nebius-physical-ai/pull/633) | `16601ddbc` | Accepted; dependency must land first | #632 |
| [#634 — Require visible terminal evidence in VLM rollout gates](https://github.com/nebius/nebius-physical-ai/pull/634) | `52cd330b3` | Accepted; dependency must land first | #627 |
| [#635 — fix(workflow): reconcile blocked waves before resume](https://github.com/nebius/nebius-physical-ai/pull/635) | `edfe23a7a` | Accepted; dependency must land first | #633 |
| [#637 — fix(workflow): resume verified output reuse after crash](https://github.com/nebius/nebius-physical-ai/pull/637) | `c3e91b816` | Accepted; dependency must land first | #635 |
| [#638 — fix(skypilot): contain listener exit race](https://github.com/nebius/nebius-physical-ai/pull/638) | `f6fbad609` | Accepted; dependency must land first | #635 |
| [#639 — fix(workflow): fence completed replay identity](https://github.com/nebius/nebius-physical-ai/pull/639) | `abbe30736` | Accepted; dependency must land first | #637 |
| [#640 — fix(skypilot): link default ~/.kube/config into isolated HOME when KUBECONFIG is unset](https://github.com/nebius/nebius-physical-ai/pull/640) | `ca6a515d2` | Ready for required merge queue | — |
| [#641 — fix(workflow): bind image identity to references](https://github.com/nebius/nebius-physical-ai/pull/641) | `d28b270d4` | Accepted; dependency must land first | #639 |
| [#643 — fix(workflow): bind image selection identity](https://github.com/nebius/nebius-physical-ai/pull/643) | `7a6454add` | Accepted; dependency must land first | #641 |
| [#644 — fix(workflow): reject unmatched image overrides](https://github.com/nebius/nebius-physical-ai/pull/644) | `204e126ae` | Accepted; dependency must land first | #643 |
| [#646 — feat(ncore): qualify full COLMAP conversion for NRE](https://github.com/nebius/nebius-physical-ai/pull/646) | `fe85f6d75` | Changes and validation required | — |
| [#647 — Add audit-only rich visual VLM reviews](https://github.com/nebius/nebius-physical-ai/pull/647) | `c958ac2a2` | Accepted; dependency must land first | #634 |
| [#649 — Studio: expose illustrative film-review cues](https://github.com/nebius/nebius-physical-ai/pull/649) | `788b4e632` | Accepted; dependency must land first | #558 |
| [#653 — fix(workflow): redact durable diagnostics](https://github.com/nebius/nebius-physical-ai/pull/653) | `7e33da910` | Accepted; dependency must land first | #623 |
| [#654 — Harden agent metadata credential request paths](https://github.com/nebius/nebius-physical-ai/pull/654) | `0666db15f` | Full workload evidence incomplete | — |

Public evidence examples: [evo raw-error recomputation](https://github.com/nebius/nebius-physical-ai/blob/e53a7103b52a4a4ce0cc7c3c8b87b5d2bede6e4c/docs/testing/evidence/evo-recompute/README.md), [workflow resume fault injection](https://github.com/nebius/nebius-physical-ai/pull/594#issuecomment-5747050271), and [credential-isolation independent review](https://github.com/nebius/nebius-physical-ai/pull/616#issuecomment-5751833498).
