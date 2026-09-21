# Cursor workbench PR readiness

Base GitHub snapshot: **2026-09-21T02:07:59.226812+00:00**. Targeted refresh of #602, #608, #615, #654 and browser #650: **2026-09-21T03:02:07.842946+00:00**. Other rows retain their stated earlier snapshots and acceptance records.

**49 historically tracked program PRs; 48 open; 14 program candidates for the required merge queue.** Closed #611 is wholly superseded by #614 and adds no delivered improvement. The separately verified browser Cursor cohort adds **#650**, for **15 confirmed queue candidates across the two cohorts**; it is not added to the historical49. No PR was merged by this audit.

Program candidates: #579, #581, #584, #594, #596, #607, #609, #613, #614, #616, #621, #628, #629, #640. Browser #650 at `47f7fea38fa5e7d6b2e162bab9ab4e032abc1e03` is non-draft and CLEAN with 14 successful checks,4 skipped, the three required checks bound to GitHub Actions,37 independent tests and2 mutation kills. It covers CPU browser/API wiring, not GPU execution. All candidates still require the repository merge queue and integration checks; any changed head needs renewed review and CI.

[Open3D's real six-stage standard workflow completed](../open3d-standard-runtime-256c26bc/README.md), with independently hashed artifacts, verified recording cameras and a new same-camera CPU comparison. Final viewer capture, exact-head review/CI and quality acceptance remain separate; durable-mount resume was not demonstrated. The public pack preserves non-manifold geometry and historical held-out regressions.

[#654's exact-source review and real CPU deployment proof are accepted](../agent-metadata-cpu-0666db15/README.md):74+132 tests passed, plus unmocked process failure controls. Its current CI and draft promotion remain pending. #608 and #615 also have accepted exact-source reviews and incomplete current CI. They are not added to queue candidates yet.

[SeedVR H100 diagnostic media and measurements](../seedvr-h100-diagnostic/README.md) remain a **failure**: LPIPS and temporal gates failed; completed task annotation also failed (candidate1/9 versus bicubic2/9 task tuples; new critical contradictions at frames76/96). The visual score is INCONCLUSIVE due frozen runner identity mismatch. The predecessor-image-plus-adapter result does not qualify the exact built image or shipped workflow. RoboCasa and cuRobo workload acceptance remain incomplete; AprilTag calibration failed. No new physical-correctness or robot-safety claim is made.

[Exact hashes, dependencies and proof links](readiness.json) make the scope reviewable. Earlier CPU reviews: [#640 filesystem behavior and151 tests](independent-cpu-reviews/pr640/README.md); [#644 selector mutation controls and71 tests](independent-cpu-reviews/pr644/README.md). #644 still depends on #643.

#627 at `46f50ad77777ca469d6782fe1101896d9999ec2f` is now ready for review **within parent stack #624**, with20 successful checks and2 skipped. Its [actual six-request hosted evidence](https://github.com/nebius/nebius-physical-ai/blob/46f50ad77777ca469d6782fe1101896d9999ec2f/docs/workbench/evidence/vlm-blinded-preference-review.md) retains two low-confidence pair escalations, three scale misreadings and the limits of three views from one scene. This does not qualify the judge or physical geometry and does not increase the main-queue count.

| PR | Exact source | Status | Depends on |
| --- | --- | --- | --- |
| [#579 — Bound Workbench I/O concurrency and preserve downloads on failure](https://github.com/nebius/nebius-physical-ai/pull/579) | `dcc530d9a` | Ready for required merge queue | — |
| [#581 — Report a group-writable checkout once instead of 428 scan failures](https://github.com/nebius/nebius-physical-ai/pull/581) | `f0e2ccfe8` | Ready for required merge queue | — |
| [#584 — Add evo trajectory evaluation BYOF workflow](https://github.com/nebius/nebius-physical-ai/pull/584) | `8c6bb6b41` | Ready for required merge queue | — |
| [#593 — feat(workbench): add SeedVR2 video restoration](https://github.com/nebius/nebius-physical-ai/pull/593) | `b8d9c9c56` | H100 diagnostic ran; objective failed; full acceptance incomplete | — |
| [#594 — Fail closed on corrupt workflow resume ledgers](https://github.com/nebius/nebius-physical-ai/pull/594) | `93d37b494` | Ready for required merge queue | — |
| [#595 — Fix RoboCasa policy evidence alignment](https://github.com/nebius/nebius-physical-ai/pull/595) | `2a79cbd6e` | Full workload evidence incomplete | — |
| [#596 — Retain verifiable provenance for VLM evaluations](https://github.com/nebius/nebius-physical-ai/pull/596) | `93e92802e` | Ready for required merge queue | — |
| [#597 — Expose VLM provider and score-gate contradictions](https://github.com/nebius/nebius-physical-ai/pull/597) | `2995156d4` | Accepted; dependency must land first | #596 |
| [#602 — Add Open3D registration and reject reconstructed surface the scan never supported](https://github.com/nebius/nebius-physical-ai/pull/602) | `81fae3d51` | Six-stage CPU runtime verified; final review, viewer proof and current CI incomplete | — |
| [#607 — Let build provenance through the payload scan instead of refusing the image](https://github.com/nebius/nebius-physical-ai/pull/607) | `c0a125720` | Ready for required merge queue | — |
| [#608 — Guard against module-level GPU imports in tests](https://github.com/nebius/nebius-physical-ai/pull/608) | `0d6b5bb08` | Exact review accepted; current CI and draft promotion pending | — |
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
| [#627 — Add blinded, order-balanced VLM preference audits](https://github.com/nebius/nebius-physical-ai/pull/627) | `46f50ad77` | Accepted and ready for review; dependency #624 must land first | #624 |
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
| [#654 — Harden agent metadata credential request paths](https://github.com/nebius/nebius-physical-ai/pull/654) | `0666db15f` | Exact review and live CPU proof accepted; current CI and draft promotion pending | — |

Public evidence examples: [evo raw-error recomputation](https://github.com/nebius/nebius-physical-ai/blob/e53a7103b52a4a4ce0cc7c3c8b87b5d2bede6e4c/docs/testing/evidence/evo-recompute/README.md), [workflow resume fault injection](https://github.com/nebius/nebius-physical-ai/pull/594#issuecomment-5747050271), and [credential-isolation independent review](https://github.com/nebius/nebius-physical-ai/pull/616#issuecomment-5751833498).
