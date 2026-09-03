---
name: antioch
description: Use when deploying, operating, debugging, or composing the Antioch Workbench integration and its offline policy-data contract.
---

# Antioch Workbench

Use the supported structured Antioch CLI only. Never call undocumented Rome HTTP
endpoints, print identity/config/environment data, or inspect unrelated runs.

## Before spending

1. Run `npa workbench health preflight --checks s3 --json`.
2. Explicitly set `NPA_ANTIOCH_ACCEPT_TERMS=YES`, then run
   `npa workbench antioch terms-preflight --output json` for the pinned scope.
3. Run `npa workbench antioch health --output json`; do not start interactive login.
4. Confirm inputs are public or synthetic and output is a unique run-scoped S3 prefix.
5. Use the pinned runtime-fetch adapter documented in `docs/workbench/antioch.md`.

## Operate

- Use `run` for blocking workflow composition; `submit` plus `status` for manual control.
- Always pass `--input-path`, `--output-path`, `--workflow-run`, and `--state-id`.
- Always pass explicit `--robot-type` and `--task`; collection has no fallback labels.
- Reuse the same identities after a retry. `reconcile` closes a submission crash window.
- Treat 429/5xx as retryable. Treat auth, malformed output, id conflicts, checksums,
  and episode schema errors as terminal.
- Gate downstream work on `_SUCCESS.json`, then consume `<output>/dataset`.
- The dataset is for offline imitation learning only. Use the LeRobot policy trainer;
  do not describe it as online PPO or RSL-RL.

### Continuing live viewport + external policy

- Use only `antioch services build|up|exec|cp|down` and `antioch scenario run
  --stream --verbose`. Do not call Rome or infer a console URL.
- Start the sim service before copying a run-scoped CA/API-key/endpoint bundle into
  a 0700 service directory. Never use `--set`, environment dumps, tmux command
  arguments, or project source for policy credentials.
- The supported steady-state path is the MK8s adapter Deployment: run the Antioch
  service tunnel and bounded relay in one pod network namespace and target a
  ClusterIP policy Service. The operator VM may deploy/status it but must not
  carry frames or actions. The legacy tmux path is recovery-only and must not be
  retained after a cluster-native cutover.
- An accepted interactive run can outlive a detached foreground CLI. Before a
  renewal, reconcile the exact project-scoped scenario through supported
  `scenario list` and `machine status` JSON. Adopt only the matching stream owner,
  wait for terminal state, and fail closed on absent or ambiguous ownership;
  never dispatch another run merely because the local CLI reported the occupied
  lease.
- A scenario dispatch or renewal may recreate the sim container. While the
  foreground run lives, verify every required bundle file through `services exec` and
  re-stage missing files with `services cp`; never bake them into the sim image.
- Treat the built service image as machine-local too. If a recycled assignment
  makes `services up` report that its image is absent, run the supported
  `services build --service sim` before `services up`. Do not submit a scenario
  until the service, reviewed source hashes, and every private bundle file verify.
  Stage a full bundle generation and atomically switch one symlink so a renewing
  bridge cannot observe a certificate/key or token generation mix.
- When direct policy egress is unavailable, use a declared Antioch service port
  published only on adapter-pod localhost. When the streamed scenario is not the
  declared-port owner, terminate authenticated WSS in the persistent sim service,
  preferably as the detached service entrypoint waiting for the runtime-staged
  bundle. Use short `services exec` health probes rather than a long exec-held
  daemon. Let the scenario connect outbound with a distinct authenticated role, and
  run the same-pod relay that bridges the operator role to the policy's independently
  authenticated, CA-verified ClusterIP WSS port 443. Bound messages, queues, connections,
  requests, timeouts, and reconnect backoff; never substitute an unauthenticated
  public proxy, disabled TLS verification, a token in a URL, or an undocumented
  Antioch endpoint.
- Treat livestream state `ready` as published but awaiting an authenticated Mission
  Control viewer. Do not claim an actively viewed frame until a supported viewer
  connection advances the first render, and never inspect browser auth storage.
- Describe renewal honestly: each boundary resets the simulated episode and briefly
  interrupts the viewport; it does not create one infinitely lived simulator process.
- A live policy loop must log current camera frames and decision counters from the
  executing scenario. Reject stale, malformed, non-finite, wrong-shaped, or unsafe
  actions and hold position while reconnecting. Do not claim hard real-time control.
- Give live telemetry one explicit logger root and resolve every Rerun blueprint
  origin through that same root; unprefixed view origins do not select entities
  emitted by a named Antioch logger.
- `npa/examples/antioch-openpi-live` is the public-source reference. Its checked-in
  project identity is intentionally unusable and is replaced only in private runtime
  state. The OpenPI gateway/controller lives in
  `npa.workflows.byof.openpi_live`.
- Use `npa workbench antioch live-k8s-deploy|live-k8s-status|live-k8s-stop`
  with one mode-0600 runtime config. Keep exact Kubernetes, Antioch, and secret
  coordinates in that file, not argv or ordinary output. Finalize removal of an
  exact owned public rollback Service only after sustained acceptance.

## Cleanup and evidence

Cancel the exact test run before releasing its exact project machine. A requested
retained live demo is the exception: leave its exact sim service and policy
Deployment running, and provide exact supported stop commands privately. Record only
run ids, states, check names, schemas, checksums, artifact basenames, and sanitized
links. Never record tokens, signed URLs, config contents, organization/customer
identifiers, unrelated run metadata, or internal infrastructure coordinates.

See `docs/workbench/antioch.md` for authentication, deployment, schemas, licensing,
recovery, console access, and the current personal-OAuth limitation.
