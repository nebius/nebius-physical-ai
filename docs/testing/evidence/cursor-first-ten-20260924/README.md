# First ten Cursor-workstream PRs: merge in two batches

Codex checked these exact heads at **2026-09-24T02:52:43.656536+00:00**, without using Cursor. Main: `fad4a0f7197b34c6326a17825baf581941c5a342`.

All ten are open, non-draft, conflict-free against that main, and green on current GitHub checks. All ahead-base commits have verified signatures; there are no unresolved review threads. Required checks are bound to each head and GitHub Actions app 15368.

Use the required merge queue in the order below. Let batch one finish before submitting batch two and refresh checks after main advances. The queue merges one PR per entry; no PR was merged or enqueued by this validation.

## Batch 1

| Order | PR | Improvement |
| --- | --- | --- |
| 1 | [#594](https://github.com/nebius/nebius-physical-ai/pull/594) | Preserve corrupt or mismatched resume ledgers instead of overwriting history. |
| 2 | [#613](https://github.com/nebius/nebius-physical-ai/pull/613) | Show declarative outputs under their exact run root. |
| 3 | [#614](https://github.com/nebius/nebius-physical-ai/pull/614) | Prevent cancellation and teardown from claiming success on contradictory or still-live job state. |
| 4 | [#615](https://github.com/nebius/nebius-physical-ai/pull/615) | Block duplicate submission on damaged receipts and redact failure diagnostics. |
| 5 | [#621](https://github.com/nebius/nebius-physical-ai/pull/621) | Check branch-only images and preserve exact image/pull-authority attribution. |

## Batch 2

| Order | PR | Improvement |
| --- | --- | --- |
| 6 | [#609](https://github.com/nebius/nebius-physical-ai/pull/609) | Recover stage log identity only from unambiguous durable waves. |
| 7 | [#640](https://github.com/nebius/nebius-physical-ai/pull/640) | Keep the default Kubernetes configuration available to isolated SkyPilot submits. |
| 8 | [#679](https://github.com/nebius/nebius-physical-ai/pull/679) | Record executable recovery commands with the original run and non-secret options. |
| 9 | [#687](https://github.com/nebius/nebius-physical-ai/pull/687) | Reject image secrets at every severity before public copy. |
| 10 | [#650](https://github.com/nebius/nebius-physical-ai/pull/650) | Check all catalogued gated Sim2Real dependencies before provisioning or submitting. |

## Validation

- First five together on main `c2a162278ea2`: **960 affected tests passed**.
- All ten together: **26969 passed, 133 skipped, 1 xpassed, 445 warnings in 222.59s (0:03:42)**.
- Combined lint passed. The preceding main integration passed 114 standalone smoke tests and 4,422 standalone guardrails; the final full suite includes those tests and the newly merged readiness checks. Confidentiality and Gitleaks passed.
- Real native security scanners, using current-main policy: **603 baseline / 603 candidate findings; zero regressions or blocking findings**. Real hostile Python, Actions and dependency controls passed.
- Three merge-precheck failures reproduced on unchanged main with the old host Git. A task-local Git 2.55.0 corrected the environment; all five focused checks passed before the fresh full rerun. No test was removed or weakened.
- Main advanced during review with #740. The ordered integration was rebuilt and verified to differ only by that exact six-file main delta; fresh focused readiness tests and the full suite passed on the new combined tree.
- Both tested source trees remained clean and unchanged. The full local CPU suite ran in parallel and excludes live/GPU and end-to-end markers; this does not replace the required hosted merge-queue checks.

The source hashes, required-check links, combined tree hashes, original-log hashes, scope limitations and retained setup failures are in [report.json](report.json). Check the files against [SHA256SUMS](SHA256SUMS).

## Workload evidence and scope

These changes run on the host CPU. GPU runs and VLM scoring do not exercise their changed boundaries. The PRs retain fault-injection and review evidence; #594, #613 and #609 also retain actual object-storage checks. #609 does not claim a real controller-log tail.

- [#687: actual Trivy severity comparison](https://github.com/nebius/nebius-physical-ai/blob/e0b17231183fb11188cb49a5922a7fad26e874e3/docs/testing/evidence/image-secret-severity-control/README.md).

The linked bundle was opened anonymously. The publication scanner source remains byte-identical between PR #687 and the combined tree. Retained proof is not relabeled as a new workload run. Raw logs, private image bytes, generated keys, credentials and infrastructure details remain private.

## Ordering and compatibility

#628 and #629 were removed after ordered integration found conflicts. #688 was returned to draft after a new native control found that a parse-valid padded private key evades detection. Those changes need separate repair and validation before merge.

#621 tightens submit preflight for unqualified Docker Hub base references used by eight existing BYOF/partner specs; its PR documents the compatibility impact. #640 retains live symlinks to provider configuration, so isolated SkyPilot state does not make provider files independent.

The held #688 finding has a [measured control report](held-pr688-control.json) and [reproducer](reproduce_pr688.py). Run from PR #688 at its stated head with the repository virtualenv, cryptography and OpenSSL installed. The script checks the exact helper hash, generates disposable key material only in memory, and prints hashes/verdicts without private-key bytes.
