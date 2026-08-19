---
name: antioch
description: Operate and debug supported Antioch CLI projects, GPU machines, services, scenarios, suites, tunnels, offline policy-data collection, and the private RTX Isaac-to-B200 OpenPI Franka bridge. Use for safe authentication/profile selection, ambiguous-run reconciliation, exact cleanup, hosted-camera readiness, runtime-fetch packaging, or Antioch image release validation.
---

# Antioch Workbench

Use only supported Antioch CLI/SDK behavior. Never inspect an auth file, browser
session, cookie, token, SSH identity, or undocumented API. Keep raw CLI JSON and
detached logs outside the repository; emit only sanitized assertions.

## Choose the operating mode

- For authentication, project initialization, machine assignment, scenario or
  suite lifecycle, and exact cleanup, read
  [authentication and lifecycle](references/authentication-and-lifecycle.md).
- For the hosted RTX Isaac/Franka camera bridge to a private B200 OpenPI server,
  read [OpenPI Franka operation](references/openpi-franka-operation.md) and
  `docs/workbench/antioch-openpi-franka.md`.
- For image classification, interrupted-copy recovery, digest scans, public
  publication, and sanitized evidence, read
  [release and recovery](references/release-and-recovery.md).
- Treat [machine-readable contracts](references/contracts.yaml) as the minimum
  security, readiness, and control invariants. Do not weaken them to pass a
  smoke.

## Start with proof, not process existence

1. Run `npa workbench health preflight --checks s3 --json` when artifacts use S3.
2. Prove the supported Antioch session and harmless API access as described in
   the lifecycle reference. Select the intended organization with `auth switch`;
   never infer it from files.
3. Run `npa workbench antioch health --output json` for the NPA adapter path.
4. Resolve every runtime image to a registry digest. Keep live infrastructure
   values and secret names out of committed examples and evidence.
5. Require supported API state plus health and useful workload evidence. A PID,
   open TCP socket, tunnel process, container, or GPU allocation alone is not
   readiness.

## Operate the NPA offline adapter

- Use `run` for blocking composition; use `submit`, `status`, `reconcile`,
  `resume`, and `cancel` for explicit lifecycle control.
- Always pass `--input-path`, `--output-path`, `--workflow-run`, and `--state-id`.
  Reuse those identities after interruption; never guess a second submission.
- Treat 429/5xx as retryable. Treat auth, conflicting identity, malformed JSON,
  invalid checksum/schema, and credential discovery as terminal.
- Gate consumers on `_SUCCESS.json`, then consume `<output>/dataset`. This is
  offline imitation data, not online PPO or RSL-RL.
- Follow `docs/workbench/antioch.md` for the immutable project and episode
  contracts.

## Preserve the two-GPU security boundary

- Keep OpenPI as a private ClusterIP service on port 8000 with bridge-only
  ingress. Place B200 policy and RTX simulator workloads independently.
- Give the model warmer alone the model entitlement and writable cache. Give
  the policy server a verified read-only cache. Give only the simulator side
  Antioch/Isaac runtime secrets. Never cross those secret scopes.
- Treat continuous soft-real-time streaming as production and the finite
  one-observation path as smoke only. Keep camera, policy, and control cadences
  distinct; neither console video nor a connected tunnel proves readiness.
- Treat exact finite `[15,8]` actions, Franka joint/gripper limits, bounded
  per-step motion, age/deadline checks, one in-flight request, reconnect epoch,
  and rate limiting as one fail-closed boundary.
- On failure, apply no new policy target and enter the configured safe hold/no-
  action state. Never reinterpret malformed or late output as a hold command or
  best-effort action, and never claim hard-real-time guarantees.

## Finish safely

Cancel the exact run before stopping its exact services or releasing its exact
machine. Delete only named, run-scoped Kubernetes objects after capturing
sanitized evidence. Preserve requested registry artifacts. Never use broad
cleanup selectors when ownership is uncertain.
