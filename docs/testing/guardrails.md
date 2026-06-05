# Repo-Resident Guardrails

This repository has additive no-GPU guardrails for PRs:

- `confidentiality scan`: scans the PR diff and tracked tree with regexes sourced from the required `CUSTOMER_DENYLIST` GitHub Actions secret and, when configured, the supplemental `INFRA_DENYLIST` secret. Local operator runs may provide the same regexes through `${PATTERN_ENV}_FILE` or `~/.config/npa/<lowercase-pattern-env>.regex`. The workflow prints only redacted file locations and fails closed when the required customer source is absent.
- `harness guardrails`: runs static three-tier workbench contracts, pytest collection protection, GPU-skip lint, and SkyPilot teardown lint.

Recommended branch-protection policy is an admin decision. The confidentiality scan and harness guardrails are designed to be merge-blocking checks; the local registry check is informational until it can run from an environment with registry reachability.

Operator-private denylist files must stay outside the repository. For example,
`CUSTOMER_DENYLIST` falls back to `~/.config/npa/customer-denylist.regex` for
local scans when the environment variable is not set.

## Local Registry Image Check

Registry image reachability is environment-dependent, so image existence checks stay out of GitHub CI and can be run from a host with access to the container registry:

```bash
npa/.venv/bin/python npa/scripts/check_workflow_images.py --registry-id "$NPA_REGISTRY_ID"
```

Current workflow image tags still include operator placeholders, so the script reports `SEAM` until those placeholders are rendered for a specific run or registry.

## Proven-Run Drift

The optional YAML-to-proven-run drift guard needs a committed canonical proven-run manifest before it can compare drift. Until that manifest exists, this remains a documented `SEAM` rather than a GitHub CI gate.
