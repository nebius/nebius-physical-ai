# Cursor PR inventory

GitHub checked 2026-09-22T17:42:41.702390+00:00; additional Cursor branches checked 2026-09-22T17:47:32.037020+00:00.

**65 open Cursor-branch PRs: 32 ready for the required merge queue, 25 waiting on predecessor PRs, three blocked by GPU qualification, and five needing other work or review.** Four additional Codex follow-up PRs are ready, bringing the open workstream to 69 PRs and the ready count to 36.

The recent workstream contains 74 PRs including six Codex follow-ups: 69 open, one merged and four closed unmerged. The earlier 68-PR tracker omitted six recent Cursor branches; this inventory includes them.

All-time Cursor-branch history is 164 PRs: 65 open, 78 merged and 21 closed without merge. Historical PRs are separate from the current 100-PR aspiration.

[Download every Cursor PR (CSV)](all-cursor-prs.csv) · [Download all open workstream PRs (CSV)](current-open-prs.csv) · [Complete status data](inventory.json) · [Hashes](SHA256SUMS)

Ready means accepted scope/evidence plus current passing checks and no conflicts; the required merge queue must still validate integration. No merge or enqueue was performed by this handoff. Current fixes are by Codex because Cursor has reached its monthly per-user limit.

## Ready Cursor PRs — 32

| PR | Change | Current state |
| --- | --- | --- |
| [581](https://github.com/nebius/nebius-physical-ai/pull/581) | Report a group-writable checkout once instead of 428 scan failures | Checks pass; conflict-free; required merge queue remains. |
| [584](https://github.com/nebius/nebius-physical-ai/pull/584) | Add evo trajectory evaluation BYOF workflow | Checks pass; conflict-free; required merge queue remains. |
| [594](https://github.com/nebius/nebius-physical-ai/pull/594) | Fail closed on corrupt workflow resume ledgers | Checks pass; conflict-free; required merge queue remains. |
| [596](https://github.com/nebius/nebius-physical-ai/pull/596) | Retain verifiable provenance for VLM evaluations | Checks pass; conflict-free; required merge queue remains. |
| [602](https://github.com/nebius/nebius-physical-ai/pull/602) | Add Open3D registration and support-distance surface filtering | Checks pass; conflict-free; required merge queue remains. |
| [607](https://github.com/nebius/nebius-physical-ai/pull/607) | Let build provenance through the payload scan instead of refusing the image | Checks pass; conflict-free; required merge queue remains. |
| [608](https://github.com/nebius/nebius-physical-ai/pull/608) | Guard against module-level GPU imports in tests | Checks pass; conflict-free; required merge queue remains. |
| [609](https://github.com/nebius/nebius-physical-ai/pull/609) | Recover stage log attribution from durable waves | Checks pass; conflict-free; required merge queue remains. |
| [610](https://github.com/nebius/nebius-physical-ai/pull/610) | Add AprilTag fiducial detection BYOF workflow | Checks pass; conflict-free; required merge queue remains. |
| [613](https://github.com/nebius/nebius-physical-ai/pull/613) | Keep declarative workflow outputs visible | Checks pass; conflict-free; required merge queue remains. |
| [614](https://github.com/nebius/nebius-physical-ai/pull/614) | Fail closed on contradictory cancellation state | Checks pass; conflict-free; required merge queue remains. |
| [615](https://github.com/nebius/nebius-physical-ai/pull/615) | Fail closed on corrupt submission receipts | Checks pass; conflict-free; required merge queue remains. |
| [616](https://github.com/nebius/nebius-physical-ai/pull/616) | test: stop the unit suite writing into the operator's real ~/.npa | Checks pass; conflict-free; required merge queue remains. |
| [621](https://github.com/nebius/nebius-physical-ai/pull/621) | Complete workflow image preflight across decision paths | Checks pass; conflict-free; required merge queue remains. |
| [625](https://github.com/nebius/nebius-physical-ai/pull/625) | feat(curobo): qualify auditable motion-planning image | Checks pass; conflict-free; required merge queue remains. |
| [628](https://github.com/nebius/nebius-physical-ai/pull/628) | Fail closed on unreadable workflow stage status | Checks pass; conflict-free; required merge queue remains. |
| [629](https://github.com/nebius/nebius-physical-ai/pull/629) | Scan every page for workflow prefix outputs | Checks pass; conflict-free; required merge queue remains. |
| [640](https://github.com/nebius/nebius-physical-ai/pull/640) | fix(skypilot): link default ~/.kube/config into isolated HOME when KUBECONFIG is unset | Checks pass; conflict-free; required merge queue remains. |
| [650](https://github.com/nebius/nebius-physical-ai/pull/650) | fix(sim2real): gate submit preflight on all sim2real gated HF repos | Checks pass; conflict-free; required merge queue remains. |
| [651](https://github.com/nebius/nebius-physical-ai/pull/651) | docs(sim2real): document hosted evaluator capability checks | Checks pass; conflict-free; required merge queue remains. |
| [652](https://github.com/nebius/nebius-physical-ai/pull/652) | fix(skypilot): retry ambiguous Kubernetes readiness evidence instead of failing the launch | Checks pass; conflict-free; required merge queue remains. |
| [654](https://github.com/nebius/nebius-physical-ai/pull/654) | Harden agent metadata credential request paths | Checks pass; conflict-free; required merge queue remains. |
| [659](https://github.com/nebius/nebius-physical-ai/pull/659) | fix(skypilot): retain Nebius storage for Kubernetes jobs | Checks pass; conflict-free; required merge queue remains. |
| [678](https://github.com/nebius/nebius-physical-ai/pull/678) | Fix Kimi-K3 hosted VLM completion contract | Checks pass; conflict-free; required merge queue remains. |
| [679](https://github.com/nebius/nebius-physical-ai/pull/679) | fix(workflow): preserve executable recovery commands | Checks pass; conflict-free; required merge queue remains. |
| [681](https://github.com/nebius/nebius-physical-ai/pull/681) | Scan complete image bytes concurrently without changing the ledger | Checks pass; conflict-free; required merge queue remains. |
| [682](https://github.com/nebius/nebius-physical-ai/pull/682) | fix(workflow): reject missing explicit SkyPilot config | Checks pass; conflict-free; required merge queue remains. |
| [683](https://github.com/nebius/nebius-physical-ai/pull/683) | Verify Kubernetes image-pull authority after admission | Checks pass; conflict-free; required merge queue remains. |
| [684](https://github.com/nebius/nebius-physical-ai/pull/684) | fix(skypilot): report preserved cleanup metadata | Checks pass; conflict-free; required merge queue remains. |
| [686](https://github.com/nebius/nebius-physical-ai/pull/686) | Fix BYOF SSH host keys retained in image layers | Checks pass; conflict-free; required merge queue remains. |
| [687](https://github.com/nebius/nebius-physical-ai/pull/687) | Reject image secrets at every severity before public copy | Checks pass; conflict-free; required merge queue remains. |
| [688](https://github.com/nebius/nebius-physical-ai/pull/688) | fix(scan): detect private keys in the payload scanners that claim to | Checks pass; conflict-free; required merge queue remains. |

## Ready Codex follow-ups — 4

| PR | Change | Current state |
| --- | --- | --- |
| [692](https://github.com/nebius/nebius-physical-ai/pull/692) | fix(mk8s): align provider RPC deadlines with apply budget | Checks pass; conflict-free; required merge queue remains. |
| [709](https://github.com/nebius/nebius-physical-ai/pull/709) | Safely finalize owned MK8s operations after verified absence | Checks pass; conflict-free; required merge queue remains. |
| [716](https://github.com/nebius/nebius-physical-ai/pull/716) | fix(workflow): reconcile absent SkyPilot resources without losing failure evidence | Checks pass; conflict-free; required merge queue remains. |
| [723](https://github.com/nebius/nebius-physical-ai/pull/723) | Handle SkyPilot daemon exit during listener ownership checks | Checks pass; conflict-free; required merge queue remains. |

## Waiting on predecessor PRs — 25

| PR | Change | Current state |
| --- | --- | --- |
| [597](https://github.com/nebius/nebius-physical-ai/pull/597) | Expose VLM provider and score-gate contradictions | Await #596; draft |
| [612](https://github.com/nebius/nebius-physical-ai/pull/612) | Retain source-frame sampling provenance for VLM requests | Await #597; draft |
| [617](https://github.com/nebius/nebius-physical-ai/pull/617) | Use backend-neutral VLM result artifact names | Await #612; draft |
| [619](https://github.com/nebius/nebius-physical-ai/pull/619) | Unify managed-job terminality checks | Await #614; open |
| [620](https://github.com/nebius/nebius-physical-ai/pull/620) | Guard workflow teardown allowance invariants | Await #619; open |
| [622](https://github.com/nebius/nebius-physical-ai/pull/622) | Preserve paired VLM judge disagreements | Await #617; draft |
| [623](https://github.com/nebius/nebius-physical-ai/pull/623) | Sanitize workflow CLI failure output | Await #615; open |
| [624](https://github.com/nebius/nebius-physical-ai/pull/624) | feat: expose VLM benchmark confusion failures | Await #622; draft |
| [626](https://github.com/nebius/nebius-physical-ai/pull/626) | Report workflow image preflight planning failures | Await #621; open |
| [627](https://github.com/nebius/nebius-physical-ai/pull/627) | Add blinded, order-balanced VLM preference audits | Await #624; open |
| [630](https://github.com/nebius/nebius-physical-ai/pull/630) | Reuse complete outputs before recovery limits | Await #629; open |
| [631](https://github.com/nebius/nebius-physical-ai/pull/631) | Keep pending workflow artifact lookups exact | Await #613; open |
| [632](https://github.com/nebius/nebius-physical-ai/pull/632) | Block workflow output reuse on identity drift | Await #630; open |
| [633](https://github.com/nebius/nebius-physical-ai/pull/633) | fix(workflow): preserve provider status during output reuse | Await #632; open |
| [634](https://github.com/nebius/nebius-physical-ai/pull/634) | Require visible terminal evidence in VLM rollout gates | Await #627; draft |
| [635](https://github.com/nebius/nebius-physical-ai/pull/635) | fix(workflow): reconcile blocked waves before resume | Await #633; open |
| [637](https://github.com/nebius/nebius-physical-ai/pull/637) | fix(workflow): resume verified output reuse after crash | Await #635; open |
| [638](https://github.com/nebius/nebius-physical-ai/pull/638) | fix(skypilot): contain listener exit race | Await #635; open |
| [639](https://github.com/nebius/nebius-physical-ai/pull/639) | fix(workflow): fence completed replay identity | Await #637; open |
| [641](https://github.com/nebius/nebius-physical-ai/pull/641) | fix(workflow): bind image identity to references | Await #639; open |
| [643](https://github.com/nebius/nebius-physical-ai/pull/643) | fix(workflow): bind image selection identity | Await #641; open |
| [644](https://github.com/nebius/nebius-physical-ai/pull/644) | fix(workflow): reject unmatched image overrides | Await #643; open |
| [647](https://github.com/nebius/nebius-physical-ai/pull/647) | Add audit-only rich visual VLM reviews | Await #634; draft |
| [649](https://github.com/nebius/nebius-physical-ai/pull/649) | Studio: expose illustrative film-review cues | Await #558; draft |
| [653](https://github.com/nebius/nebius-physical-ai/pull/653) | fix(workflow): redact durable diagnostics | Await #623; open |

## GPU qualification blockers — 3

| PR | Change | Current state |
| --- | --- | --- |
| [593](https://github.com/nebius/nebius-physical-ai/pull/593) | feat(workbench): add SeedVR2 video restoration | Draft; actual B200 output failed fixed LPIPS, temporal, codec and light-input quality gates. CI passes. No heldout/VLM acceptance. |
| [595](https://github.com/nebius/nebius-physical-ai/pull/595) | Fix RoboCasa policy evidence alignment | Draft; actual RTX scene reset reached environment reset, then failed immutable artifact publication on S3 metadata-key casing. CI passes; source fix/new image and full GPU evaluation needed. |
| [646](https://github.com/nebius/nebius-physical-ai/pull/646) | feat(ncore): qualify full COLMAP conversion for NRE | Draft; CPU conversion passed; native RTX reconstruction failed timestamp interpolation. CI passes; real-SDK fix is private and unqualified. |

## Additional open Cursor PRs — 5

| PR | Change | Current state |
| --- | --- | --- |
| [561](https://github.com/nebius/nebius-physical-ai/pull/561) | Close the two offline FIXMEs and root-cause the third | Draft; merge conflict and no current-head checks. Needs main integration and validation. |
| [636](https://github.com/nebius/nebius-physical-ai/pull/636) | fix(execution-preflight): surface Nebius CLI compat error instead of masking it as an identity mismatch | Draft; CI passes. Final merge-readiness review outstanding. |
| [642](https://github.com/nebius/nebius-physical-ai/pull/642) | fix(skypilot): scope gang pending-GPU-pod contention to the requested accelerator | Draft; lint and security-regression checks fail. |
| [645](https://github.com/nebius/nebius-physical-ai/pull/645) | fix(skypilot): write lowercase skypilot.co/accelerator label for B200 | Draft; CI passes. Final merge-readiness review outstanding. |
| [718](https://github.com/nebius/nebius-physical-ai/pull/718) | fix(sim2real): reject managed-driver RTX nodes before Isaac renders | Draft; CI passes. Final merge-readiness review and applicable live validation outstanding. |

## Recent merged or closed PRs

| PR | Change | State |
| --- | --- | --- |
| [579](https://github.com/nebius/nebius-physical-ai/pull/579) | Bound Workbench I/O concurrency and preserve downloads on failure | Merged |
| [611](https://github.com/nebius/nebius-physical-ai/pull/611) | Preserve active SkyPilot jobs during cleanup | Closed without merge |
| [656](https://github.com/nebius/nebius-physical-ai/pull/656) | Preserve controller minimum resource strings | Closed without merge |
| [685](https://github.com/nebius/nebius-physical-ai/pull/685) | fix(byof): remove build-generated SSH host keys in the layer that creates them | Closed without merge |
| [693](https://github.com/nebius/nebius-physical-ai/pull/693) | fix(ci): isolate base-image scans and reclaim owned disk | Closed without merge |

[SeedVR actual GPU comparison, failed quality gate and downloadable media](https://github.com/nebius/nebius-physical-ai/blob/8887b040a0564d4c07156dace6ea2c5a2de32e05/docs/testing/evidence/seedvr3b-b200-quality-f1de6727/README.md). GPU qualification is not inferred from CPU or smoke-test success. No customer credentials, private infrastructure identifiers, or operational logs are included.
