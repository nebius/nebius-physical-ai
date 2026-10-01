# Ten Cursor PRs: final review evidence

**Ready for the merge queue at the recorded exact-head CI snapshot.**

The [finding-by-finding resolution](review-findings.md) maps each supplied review concern to its fix and evidence.

Combined source `c47d1aec50042c74b8b8edb076374d1636df2f52` on base `5a84f305b95ee29933f49eb4b94a40f68be9eca8`.
The [head manifest](pr-heads.json) binds all ten PRs. Merge #692, then #709, then
#716; the table gives the complete batch order. Required merge-queue validation
still applies. This packet does not merge anything or claim human approval.

| PR | Exact reviewed head |
| --- | --- |
| [#652](https://github.com/nebius/nebius-physical-ai/pull/652) | `3c96e8dd068c9b9994f966794460be960e7a3b9d` |
| [#653](https://github.com/nebius/nebius-physical-ai/pull/653) | `37c1f939533dc81f2e53a0b9206203f14fabd00c` |
| [#654](https://github.com/nebius/nebius-physical-ai/pull/654) | `d3401036395b8094b7a374fa5ba8984a0a8d9d8a` |
| [#659](https://github.com/nebius/nebius-physical-ai/pull/659) | `f82673f72323c797cce0f5ff4fafafc4b2ac279f` |
| [#681](https://github.com/nebius/nebius-physical-ai/pull/681) | `441c11ddaca8e5990624313c175dd49e8185b2c9` |
| [#683](https://github.com/nebius/nebius-physical-ai/pull/683) | `e5cfb4e40f75e9c0d5966d0f760585b53ba9e06b` |
| [#686](https://github.com/nebius/nebius-physical-ai/pull/686) | `49dc46dd51b5971dd0fd5d2ad477baf529874e94` |
| [#692](https://github.com/nebius/nebius-physical-ai/pull/692) | `222c90aade662f97eb20c0812239cd2e96d450de` |
| [#709](https://github.com/nebius/nebius-physical-ai/pull/709) | `db247b50baa2e63992a7a5648cf0f486d14d1bcf` |
| [#716](https://github.com/nebius/nebius-physical-ai/pull/716) | `b8fc146f2e8f932798dc3bb593e2361f570fd5fd` |

**Validation.** The [final Linux suite](validation/linux-v10-summary.json) passed
34,979 tests, with 157 skipped and
1 non-strict xpasses, zero failures/errors. All eight local
gates passed. The [security comparison](validation/security-v10-summary.json)
reported 681 baseline and 681
candidate findings, zero regressions and zero blockers; existing findings are
not a claim of zero vulnerabilities. All ten PRs were open, nondraft, clean and mergeable at 2026-10-01T22:12:07.692931+00:00; 254 check runs completed successfully or with applicability skips/neutral results.
The [CI snapshot](validation/hosted-ci.json) and [AI review closure](validation/independent-review-closure.json)
retain their exact source scopes. Claude/Codex reviews are agent reviews, not a
second human account's GitHub approval.

The [earlier v7 full run](validation/earlier-v7-result.json) had 34,869 passes and
one failed capability-audit fixture. Its [tests-only repair](agent/audit-successor-binding.json)
passed ten audit tests and 266 precheck controls while retaining all live-tested
runtime bytes. Three earlier collection attempts failed on task-checkout
permissions; the [independent diagnosis](validation/runner-diagnosis.json)
reproduces xdist masking that refusal. Those attempts remain failures.
The [earlier passing v9 suite](validation/earlier-v9-result.json) retains its
original source attribution; it is separate from the current v10 execution.

**Live and native proof.**

- [Agent identity/inventory](agent/agent-inventory-live.json): 437 Linux tests and
  a fresh verified-TLS CPU lifecycle passed in 375.68 seconds. Six installed
  modules and the rendered backend matched; authenticated requests returned 200,
  four anonymous routes returned 401, and owned infrastructure was removed.
  Failed inventory stays unknown, unverified GPU drafts cannot claim readiness,
  and a reaped leader alone does not release an uncertain descendant breaker.
- [Private image pull](registry/private-pull.json): the fifth attempt verified
  the immutable image in 590.28 seconds. Both live cases passed:
  an unscheduled one-second timeout control and a real pull without an NPA/Pod
  deadline. Probe deletion and namespace absence were verified. This success used
  [one owned node's runtime adaptation](registry/node-watchdog.json): watchdog
  `0s` and local image-pull mode. The 14.7 GB cold failure, its retry, the 8.3 GB
  failure, and the interrupted zero-timeout transfer-mode stall remain recorded.
  Partial cached layers may have been reused; this is neither stock-default
  success nor an independently cold download. Reported image size is not measured
  transferred bytes. This proves CPU container delivery, not vendor workload execution.
- [Final owned-provider cleanup](absence/provider-final-cleanup.public.json)
  verifies infrastructure absence. Node-to-instance and boot-disk ownership was
  bound before teardown; it does not label an unreachable destroyed Kubernetes
  API as a Node-UID 404. Earlier lifecycle/adjustment receipts retain their
  then-pending cleanup status and are superseded by this final receipt.
- [Scanner proof](scanner/summary.json): 807 Python tests, native/race/vet and
  one-CPU controls passed; three complete archive ledgers matched the baseline.
  Under a 384 MiB cgroup the baseline was OOM-killed, while the candidate completed
  all 12 records/96 MiB with the same four findings as complete larger-cap runs.
  Its Go limit is soft, not an RSS ceiling or OOM guarantee. Advisory-skip cases
  have unit/mutation coverage. [Integration binding](scanner/integration-v10-binding.json)
  preserves actual producer attribution.
- [Provider/recovery proof](absence/readme.md): actual provider read and typed
  deadline controls, 23 Linux and 23 macOS process controls, and 191 related
  tests have separate producer bindings. The [independent ownership review](absence/independent-review.json)
  retained the external-reap negative controls. [Three native SkyPilot guards](absence/native-sky-contract-final.public.json)
  check all nine installed-wheel sources. Arbitrary competing external reapers
  remain outside the supported synchronization model; recovery never implies
  historical workload success.
- [OAuth compatibility](oauth/summary.json): twelve real loopback HTTPS controls
  passed for OAuthLib 4. These do not certify vendor-specific consent/extensions.
  The used HCL parser dependency is retained only in #692/#709/#716.

No fresh GPU workload or VLM result is claimed. The [historical September 29 packet](https://github.com/nebius/nebius-physical-ai/blob/f8a53a2e162245b3bf5023ac010b9b68cf9e6b78/docs/testing/evidence/cursor-batch-six-20260929/README.md)
links the retained #659 and #686 receipts; its earlier heads and readiness do not
apply to this follow-up. #659's retained B200 proof shows
admission/storage but its later cuRobo numeric audit failed. #686 retains earlier
layer-level SSH-key absence and fresh startup-key observations; matching the
install fragment does not constitute a new exact-current image build. Every
retained producer and failure must be judged within its recorded scope.

[Source bindings](packet-source-bindings.json), [readiness record](packet-validation.json),
and the [complete SHA256 manifest](sha256sums.txt) accompany this local packet.
Raw credentials, provider identifiers, TLS keys and private logs are excluded.
Anonymous public readback is recorded separately by the publication step.
