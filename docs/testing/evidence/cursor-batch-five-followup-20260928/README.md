# Cursor batch five: review follow-up

Merge order: **#635 → #637 → #638 → #639 → #641 → #643 → #644**.

This follow-up addresses the supplied review with Codex; Cursor was not used. All seven branches include current main and apply cleanly in a true ordered squash simulation. No PR is merged or enqueued by this work.

## Review changes

- **#635:** release `block_relaunch` only with verified cancellation and terminal provider status; unknown decisions remain blocked. Keep the exact integer launch-sequence check. The conflict with #551 deliberately preserves these semantics.
- **#637:** place phase definitions beside the supervisor producer; record sanitized exception class in durable blocked-state evidence. Recovery checks current rendered image identity before reusing artifacts.
- **#638:** clarify that its incremental change is listener race coverage after #635 lands.
- **#639:** reject empty, missing, changed, or malformed completed-replay identities before artifact/provider calls.
- **#641:** ship core regression coverage with the production change. Bind the complete selection to resolved per-state images, preserving current-main provenance and rejecting swapped bindings even when the overall image set is unchanged.
- **#643:** retain supplemental image precedence and resource-identity coverage.
- **#644:** document unmatched selector rejection and validate once per rendered/resource-profile batch while keeping direct-call validation. Preserve both selector validation and the current-main dedicated live-gate refusal.

A test-only upstream repair uses one patch manager for the execution preflight. Mixing two managers leaked a mock after teardown. The same three failures reproduced on base and candidate; the corrected two-module ordered run passed 251 tests.

A second test-only fixture correction supplies the copied Python executable's matching standard library inside that one test. The validation runner does not export `PYTHONHOME`, so unrelated system-Python and isolated-interpreter tests keep their own runtime. The rejected global-setting experiment and its failures are retained in [diagnostics](diagnostics.json).

A third test-only correction explicitly sets the unsafe-directory fixture to mode 0755. This preserves the rejection assertion even when a preceding test or runner leaves an owner-only umask. The full module passes 66 tests under umask077.

## Review and limitations

[Claude Code review](review.md) is AI review, not a second human GitHub approval. [Triage](review-triage.json) records the reviewed revision, final source equivalence, follow-up notes, and fixture verification. The reviewer found no concrete safety blockers. Author-posted evidence comments are identified as such.

Before deploying the new image identity protocol, finish old runs or retain their original controller, source, and image for resume. Do not rewrite recorded identities. The new identity fences cover completed replay, durable output reuse, and supervised recovery. Ordinary in-flight exact-provider adoption remains a pre-existing separate path; its behavior was reproduced unchanged on base and candidate.

## Real GPU evidence

[Open the previously published B200 proof](https://github.com/nebius/nebius-physical-ai/blob/6c045f21503387f49d2181a40f9532f640fdf0c3/docs/testing/evidence/cursor-batch-five-seven-20260925/README.md).

That run used source `4602cc392b7725513cf963c96b05b63d817d8cb7`: 32 actual SGD steps, 1,024 training and 1,024 held-out samples, and a CPU scalar oracle for every loss, gradient, update, final weight, and momentum value. Initial loss was 3.253974364401865; final training loss was 0.0016543993155160276 and held-out loss 0.001557499957530664. Both injected crash boundaries recovered without duplicate GPU launch; cleanup evidence is included.

This is historical GPU evidence, not a GPU rerun of these revised control-plane heads. It does not claim robot-policy quality, VLM evaluation, distributed training, or mid-training optimizer resume. No new GPU resources were provisioned for this follow-up.

## Fresh validation

The final ordered integration passed all eight local gates. Full Linux suite: **33823 passed, 156 skipped, 1 xpassed, 456 warnings in 886.66s (0:14:46)**. Affected workflow/CLI/listener tests and hostile-input security tests also passed separately. See [validation and sanitized logs](validation.json). [Current-head GitHub checks](github-checks.json) are green for all seven PRs. This local full-suite result applies to the integrated tree; each PR has its own GitHub checks.

[Machine-readable report](report.json). Public evidence includes only sanitized results; raw operator locations and credentials are withheld.
