# PR #640: exact CPU review proof

[PR #640](https://github.com/nebius/nebius-physical-ai/pull/640) at `ca6a515d2562008d96570029ed5cbcc7b447fd71` was independently accepted on 2026-09-21. This final delta changes one documentation paragraph; the implementation was previously accepted at `1f2434e2e8f18be44af97c13fdc25f6ba14f45b2`.

**151 scoped workflow tests passed.** The reviewer also exercised actual filesystem links using synthetic files in three cases:

| Case | Observed result |
| --- | --- |
| Default configuration | Selected kubeconfig and provider-directory links resolve to their sources; writes reach those sources and retain `0600` permissions. |
| Explicit task configuration | Writes reach the selected task-owned kubeconfig, while the shared default remains unchanged. |
| Missing configuration | No spurious kubeconfig link appears. |

The documentation accurately warns that these are live links, not copied credential files. The installed Kubernetes client defaults to `persist_config=True`; this does not claim that every authentication call rewrites configuration.

The Linux-only runtime boundary remained enabled and rejected the macOS process after it created the links. That expected refusal was retained, not bypassed. This proves synthetic local file behavior and documentation accuracy. It does **not** prove a live provider refresh, SkyPilot runtime start, deployment, GPU workload, visual quality, or robot safety. No real provider configuration or credentials were used.

Evidence: [independent verdict](review.json), [test output](pytest.log), [sanitized command receipt](pytest-receipt.json), [filesystem observations](synthetic-file-probe.json). Original source-verdict and receipt hashes are retained in the sanitized records; [the manifest](../manifest.json) binds the published bytes.

At publication verification, this exact head remained main-targeted, non-draft, mergeable, with required checks green. It is a candidate for the repository's required merge queue; queue integration checks still apply. No merge was performed.
