# Script Placement Checklist

- Generic infrastructure setup: `infra/bootstrap/`.
- Generic one-shot provisioning wrappers: `scripts/`.
- Demo-specific data prep, launch, or verification: `demos/<demo>/`.
- Workbench SkyPilot templates: `npa/workflows/workbench/skypilot/`.
- Operator-facing runbooks: `docs/workbench/cookbooks/`.

## Current Blocker

The actual local-script refactor is BLOCKED until those local scripts are pushed
to a branch. This repository now has the target layout and placement checklist,
but unseen local scripts cannot be moved or rewritten safely.
