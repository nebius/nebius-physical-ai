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
