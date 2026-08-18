---
name: antioch
description: Use when deploying, operating, debugging, or composing the Antioch Workbench integration and its offline policy-data contract.
---

# Antioch Workbench

Use the supported structured Antioch CLI only. Never call undocumented Rome HTTP
endpoints, print identity/config/environment data, or inspect unrelated runs.

## Before spending

1. Run `npa workbench health preflight --checks s3 --json`.
2. Run `npa workbench antioch health --output json`; do not start interactive login.
3. Confirm inputs are public or synthetic and output is a unique run-scoped S3 prefix.
4. Use the pinned runtime-fetch adapter documented in `docs/workbench/antioch.md`.

## Operate

- Use `run` for blocking workflow composition; `submit` plus `status` for manual control.
- Always pass `--input-path`, `--output-path`, `--workflow-run`, and `--state-id`.
- Reuse the same identities after a retry. `reconcile` closes a submission crash window.
- Treat 429/5xx as retryable. Treat auth, malformed output, id conflicts, checksums,
  and episode schema errors as terminal.
- Gate downstream work on `_SUCCESS.json`, then consume `<output>/dataset`.
- The dataset is for offline imitation learning only. Use the LeRobot policy trainer;
  do not describe it as online PPO or RSL-RL.

## Cleanup and evidence

Cancel the exact test run before releasing its exact project machine. Record only
run ids, states, check names, schemas, checksums, artifact basenames, and sanitized
links. Never record tokens, signed URLs, config contents, organization/customer
identifiers, unrelated run metadata, or internal infrastructure coordinates.

See `docs/workbench/antioch.md` for authentication, deployment, schemas, licensing,
recovery, console access, and the current personal-OAuth limitation.
