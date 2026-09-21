# npa/tests

## Runtime status diagnostics live check

To check failure diagnostics against an already-run Serverless job, set
`NPA_E2E_RUNTIME_DIAGNOSTICS_CONFIG` to an owner-only JSON file and run:

```bash
NPA_INTEGRATION_E2E=1 npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_serverless_status_diagnostics_readonly_e2e.py -q
```

The variable has no default; the live tests skip when it is absent. The
[test module](e2e/test_serverless_status_diagnostics_readonly_e2e.py)
documents the required job, artifact-prefix, expected-status, and log-marker
fields. Cosmos and SONIC status commands read that exact job; supplying both
`lerobot_project` and `lerobot_workbench` also checks the saved LeRobot alias.
Set `NPA_CONFIG_DIR` when using an isolated operator configuration. These checks
only read existing resources and do not establish model quality or checkpoint
recovery; retain separate workload evidence for those claims.

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for the full test layout and PR
conventions (branch → PR → squash, one approval, never self-approve).
