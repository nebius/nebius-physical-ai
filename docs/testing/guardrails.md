# Repo-Resident Guardrails

This repository has additive no-GPU guardrails for PRs:

- `confidentiality scan`: always scans added PR lines with the public Nebius infrastructure patterns, then scans the diff and tracked tree with regexes sourced from the required `CUSTOMER_DENYLIST` GitHub Actions secret and, when configured, the supplemental `INFRA_DENYLIST` secret. Fork and Dependabot PRs cannot read Actions secrets, so their private scan is skipped only after the public scan passes; ordinary same-repository PRs still fail closed when the required customer source is absent. Local operator runs may provide the private regexes through `${PATTERN_ENV}_FILE` or `~/.config/npa/<lowercase-pattern-env>.regex`. The workflow prints only redacted file locations.
- `harness guardrails`: runs static three-tier workbench contracts, pytest collection protection, GPU-skip lint, and SkyPilot teardown lint.

Recommended branch-protection policy is an admin decision. The confidentiality scan and harness guardrails are designed to be merge-blocking checks; the local registry check is informational until it can run from an environment with registry reachability.

Operator-private denylist files must stay outside the repository. For example,
`CUSTOMER_DENYLIST` falls back to `~/.config/npa/customer-denylist.regex` for
local scans when the environment variable is not set.

## Public Image Reachability Check

Registry reachability is environment-dependent, so anonymous image existence
checks stay out of GitHub CI and can be run from a host with internet access:

```bash
npa/.venv/bin/python -m npa.deploy.publish_public \
  --target ghcr.io/nebius/nebius-physical-ai --verify-accepted-releases
```

BYOF workflow placeholders are intentionally excluded from the supported public
release inventory and require an explicitly selected operator registry.

## Proven-Run Drift

The optional YAML-to-proven-run drift guard needs a committed canonical proven-run manifest before it can compare drift. Until that manifest exists, this remains a documented `SEAM` rather than a GitHub CI gate.
