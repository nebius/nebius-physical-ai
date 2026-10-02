# Live storage tests

Run the Insights storage tests against an explicitly selected project's own
configured credentials:

```bash
NPA_INTEGRATION_E2E=1 NPA_E2E_PROJECT=<project-alias> \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_insights_storage_live_e2e.py -q
```

Both variables are required; tests skip without them. No infrastructure is
provisioned. Fixtures write and delete only a unique test-owned prefix and
verify removal of current objects, object versions, and delete markers.
Set `NPA_CONFIG_DIR` to use an isolated operator configuration.

Set `NPA_E2E_INSIGHTS_BUCKET_ROOT=1` to also verify JSONL discovery from both
bucket-root URI forms. This additional opt-in has no default and enumerates
key metadata across the named bucket. It does not read unrelated object
bodies, and its writes and deletions remain inside the unique fixture prefix.
See [the test module](test_insights_storage_live_e2e.py) for the full contract.

## Soperator user namespaces

After deploying a dedicated Soperator cluster with the default unconfined
worker profile, verify its actual host setting and non-root user-namespace
creation on every worker:

```bash
NPA_INTEGRATION_E2E=1 \
NPA_E2E_SOPERATOR_CONTEXT=<dedicated-kube-context> \
NPA_E2E_SOPERATOR_WORKER_COUNT=<exact-worker-count> \
  npa/.venv/bin/python -m pytest npa/tests/e2e/test_soperator_userns_live.py -q
```

All three variables are required; the test skips without them. The test creates
no cloud resources or GPU jobs. It runs a short namespace probe as the existing
non-root `soperatorchecks` identity in each worker container. It does not change
host settings and does not replace the deployment's CUDA creation checks.
`NPA_E2E_SOPERATOR_NAMESPACE` defaults to `soperator`.
`NPA_E2E_SOPERATOR_EVIDENCE_DIR` optionally names a private output directory for
a new `userns-live.json` receipt; an existing receipt is never overwritten.
