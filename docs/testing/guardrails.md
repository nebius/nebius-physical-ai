# Repo-Resident Guardrails

This repository has additive no-GPU guardrails for PRs:

- `confidentiality scan`: scans the PR diff and tracked tree with regexes sourced from the `CUSTOMER_DENYLIST` and `INFRA_DENYLIST` GitHub Actions secrets. The workflow prints only redacted file locations.
- `harness guardrails`: runs static three-tier workbench contracts, pytest collection protection, GPU-skip lint, and SkyPilot teardown lint.

Recommended branch-protection policy is an admin decision. The confidentiality scan and harness guardrails are designed to be merge-blocking checks; the local registry check is informational until it can run from an environment with registry reachability.

## Local Registry Image Check

Registry image reachability is environment-dependent, so image existence checks stay out of GitHub CI and can be run from a host with access to the container registry:

```bash
npa/.venv/bin/python npa/scripts/check_workflow_images.py --registry-id "$NPA_REGISTRY_ID"
```

Current workflow image tags still include operator placeholders, so the script reports `SEAM` until those placeholders are rendered for a specific run or registry.

## Proven-Run Drift

The optional YAML-to-proven-run drift guard needs a committed canonical proven-run manifest before it can compare drift. Until that manifest exists, this remains a documented `SEAM` rather than a GitHub CI gate.
