# PR #644: exact CPU mutation review proof

[PR #644](https://github.com/nebius/nebius-physical-ai/pull/644) at `204e126aefe63d0a363bda9d6be9e1c35b0d919a` was independently accepted on 2026-09-21. This final delta adds truncated-selector regression tests over accepted `c3a0bde5ddf722ca2a7289160915ae71ec9062ca`; production code is unchanged.

**71 render-module tests passed.** The reviewer then changed only the loaded function in memory:

```python
# Candidate family matching:
tool_ref.startswith(str(raw_selector) + ".")
# Intentional mutation:
tool_ref.startswith(str(raw_selector))
```

| Boundary | New truncated selector | Existing transposed selector |
| --- | --- | --- |
| Plan images | Detects mutation: expected rejection is missing | Still passes |
| Pull secrets | Detects mutation: expected rejection is missing | Still passes |
| Render | Detects mutation: expected rejection is missing | Still passes |

All three newly added `workbench.vlm_eval.ru` controls fail with `DID NOT RAISE` under the mutation. The three older `workbench.vlm_eval.rnu` controls still pass, showing why the added cases matter. These are three intentional mutation detections; the unmodified candidate passes its full render-module suite. No source file was changed by the review.

Evidence: [independent verdict](review.json), [test output](pytest.log), [sanitized command receipt](pytest-receipt.json), [mutation inputs and six outcomes](mutation.json), [readable mutation summary](mutation-summary.log). [The manifest](../manifest.json) binds the published bytes.

This is CPU selector-validation proof. No provider, GPU, or VLM work is applicable to this test-only delta, and it makes no new workload or physical-AI-quality claim. Acceptance is against the current [PR #643](https://github.com/nebius/nebius-physical-ai/pull/643) feature base. The stack must land in order; rebase or retarget requires refreshed checks and review. It is **not directly ready to merge into main**. No merge was performed.
