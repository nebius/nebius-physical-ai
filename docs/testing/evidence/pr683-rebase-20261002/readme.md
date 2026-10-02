# PR683 rebase verification

PR [#683](https://github.com/nebius/nebius-physical-ai/pull/683) is clean and ready for merge-queue validation at 2026-10-02T02:28:18.169303+00:00.
Head `a4dcc4e8ea54bf0280bacef22a256a8e13413743` was rebased onto main `0580ca7a2ea71f9b8ceaaf04335981283e44ac17`. The previous head was `e5cfb4e40f75e9c0d5966d0f760585b53ba9e06b`.

Both conflicting invocations in `npa/tests/cli/test_workflow_submit_preflight.py` retain `--infra k8s/unit-context` and `--registry` with the test's operator registry. [Source comparison](source-binding.json) proves the final tree equals the normal merge result except for those two resolutions. Dependency files are byte-identical to main; no production conflict resolution was required.

- [Focused and precheck validation](focused-validation.json): 324 focused tests and 270 precheck tests passed, zero failed. The rewritten PR history passed Gitleaks.
- [Fresh isolated Linux suite](linux-validation.json): 34,991 passed, zero failed, 156 skipped and 1 non-strict XPASS. Documentation and confidentiality checks passed. The owned validation namespace reaped all descendants.
- [Hosted checks](hosted-ci.json): 20 successful and 4 applicability skips; all required checks passed. GitHub reports clean and mergeable.

The [earlier live proof](https://github.com/nebius/nebius-physical-ai/blob/5c46e7d65a786143dccf50bfb7f2b49043d82d11/docs/testing/evidence/cursor-batch-six-20261001/readme.md) keeps its original producer attribution, including the successful private-image pull, required node-runtime adaptation, failed attempts and verified cleanup. Both `registry_preflight.py` and `image_bootstrap_contract.py` are byte-identical to the earlier PR head, as recorded in the source binding. This rebase does not claim a new live pull, GPU execution or VLM qualification.

Raw logs, credentials and infrastructure identifiers are excluded. Merge-queue validation remains required; no merge was performed.
