# Smoke Tests

Smoke tests exercise CLI subprocess paths and may use local fakes or external
services depending on the tool.

The standard non-e2e suite collects smoke tests, so heavyweight smoke checks must
skip unless their required environment is explicitly enabled.

## Cosmos Serverless

The Cosmos serverless smoke file is skip-by-default in the standard suite. To
maintain or debug that local-fake smoke path, opt it in explicitly:

```bash
NPA_COSMOS_SERVERLESS_SMOKE=1 npa/.venv/bin/python -m pytest npa/tests/smoke/test_cosmos_serverless_smoke.py -v
```

By default these tests skip because their fixture uses the placeholder project
`project-smoke` and should not be treated as live infrastructure. Live Cosmos
Serverless AI Endpoint coverage is in the e2e suite:

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest npa/tests/e2e/test_cosmos_serverless_e2e.py -v
```
