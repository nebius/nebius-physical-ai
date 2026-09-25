# Cursor batch four: merge-readiness proof

Verified 2026-09-25T16:21:03.187917+00:00. Codex refreshed these Cursor-originated PRs without using Cursor.

Merge in this order: **#629 → #630 → #631 → #632 → #633**. All five are open, non-draft, conflict-free against the observed main, and have green required GitHub checks. Nothing was merged or enqueued by this work.

Corrected the GitHub noreply account associated with the existing registered signing key. Every source tree is byte-identical to the fully tested tree; only commit metadata and dependent parent IDs changed. GitHub CI reran on all corrected heads.
The report distinguishes the locally tested commit from its corrected published commit and binds both to the same Git tree. All published commit signatures are now verified by GitHub.

| Order | PR | Head | Scoped diff | Full CPU tests passed | Hostile-input tests passed |
|---|---|---|---|---:|---:|
| 1 | [#629](https://github.com/nebius/nebius-physical-ai/pull/629) | 79a115a17822 | [Review diff](https://github.com/nebius/nebius-physical-ai/compare/bf4788a0a94ce2c2e842f54c5dc39932795d6cc5...79a115a17822ebbb707ddd59f5e16eb972cf9d7f) | 27,592 | 859 |
| 2 | [#630](https://github.com/nebius/nebius-physical-ai/pull/630) | dd9e5d5f49e1 | [Review diff](https://github.com/nebius/nebius-physical-ai/compare/79a115a17822ebbb707ddd59f5e16eb972cf9d7f...dd9e5d5f49e15726ec891a630d036f18f0182983) | 27,609 | 859 |
| 3 | [#631](https://github.com/nebius/nebius-physical-ai/pull/631) | 858564664955 | [Review diff](https://github.com/nebius/nebius-physical-ai/compare/bf4788a0a94ce2c2e842f54c5dc39932795d6cc5...858564664955980bcd0e68dbabba7e980068f8a2) | 27,563 | 859 |
| 4 | [#632](https://github.com/nebius/nebius-physical-ai/pull/632) | 1abe9cd59fc8 | [Review diff](https://github.com/nebius/nebius-physical-ai/compare/dd9e5d5f49e15726ec891a630d036f18f0182983...1abe9cd59fc80bdc87f889895aae827889d2bc18) | 27,624 | 859 |
| 5 | [#633](https://github.com/nebius/nebius-physical-ai/pull/633) | f710be1e3efa | [Review diff](https://github.com/nebius/nebius-physical-ai/compare/1abe9cd59fc80bdc87f889895aae827889d2bc18...f710be1e3efa490a03aec958af4a61433aa8cef7) | 27,625 | 859 |

Each PR passed focused workflow tests, lint and merge prechecks, smoke, 4,527 guardrails, the full CPU suite, docs drift, hostile-input security tests, confidentiality and PR-range Gitleaks. Real Bandit, Zizmor and Trivy comparisons found **zero new regressions and zero blocking findings**; pre-existing baseline findings remain recorded. Hosted checks also cover browser tests, coverage and image-security policy.

The five-PR squash integration passed 27,632 tests. The newer-main integration at 12e2584176a5 passed 27,659 tests. Its tree is b280cd52a99dd3cdaa4b8d9233b9856189a8ae44.

## Behavioral evidence

- **14 real boto3 HTTP/XML cases** exercise both production output-checking entry points: 1,000 markers before a real page-two output, empty prefixes, inconsistent pagination, repeated tokens and access denial. [Results](behavior-proof.json) · [Executed harness](behavior-proof.py).
- **337,920 decision comparisons** preserve the original accepted identity-gate behavior after separating its succeeded-attempt check from the transient-recovery guard.
- **9/9 targeted reversal mutations were detected** by behavioral tests: first-page truncation, marker acceptance, missing pagination flags, malformed empty records, recovery-limit precedence, skipped cancellation, namespace widening, identity bypass and provider-status overwrite. [Results](mutation-proof.json) · [Executed harness](mutation-proof.py).

The first simultaneous host runs for #629 and #632 reported occupied loopback listeners (two Cosmos cases and one API setup case). The exact failures are retained in the report. The first isolated retry then exposed a packaging-dependency download in the local bootstrap test. Prefetching its public wheels allowed the real installer and unchanged complete suites to pass offline. No assertions were removed or weakened.

## Stack and scope

The repository squash-merges. Provider-status implementation prerequisites were brought into #630. #633 retains reserved-adoption regression coverage and documentation. The ordered squash sequence was simulated and its combined code tested.

CPU orchestration only. No model execution, rendering, changed workflow spec or tool image; GPU and VLM testing do not apply. HTTP fault-server evidence uses real boto3 on loopback, not a live cloud workload.

The JSON report records source SHAs, trees, check URLs, test counts and retained-log hashes. Integration commits were created locally to simulate squash merges; their tree hashes can be reproduced from the listed public PR heads and bases. It distinguishes each exact PR head from the combined integration. The public harnesses use synthetic fixtures; their retained validation layout has the tested checkout at combined/source and a task-local tool directory. They require the repository development environment.

Readiness is a snapshot. The repository merge queue validates the final merge candidate as main advances. [Full report](report.json) · [SHA-256 manifest](SHA256SUMS).
