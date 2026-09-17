---
name: antioch
description: Use when deploying, operating, debugging, or composing the Antioch Workbench integration and its offline policy-data contract.
---

# Antioch Workbench

Use the supported structured Antioch CLI only. Never call undocumented HTTP
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

- Use only current structured commands: `antioch project build`, `antioch session
  new|status|release`, `antioch service exec|cp|ps|ports`, and `antioch scenario
  list|run|cancel`. Run the live scenario with `--stream --verbose`. Do not call
  undocumented HTTP endpoints or infer an API origin from a console URL.
- Resolve the exact deployment through the CLI's supported `ANTIOCH_ENV` profile
  and preserve it explicitly in the runtime. Discover organization/project
  topology with supported JSON commands; never hardcode a discovered endpoint,
  project, organization, or session as a product default.
- Create the run-scoped config directory from the current CLI login and include
  only its owner-readable `auth.json` and optional `workspace.json`. Never carry
  forward legacy machine, SSH, tunnel, lock, or cached endpoint state.
- Start the exact project session before copying a run-scoped
  CA/API-key/endpoint bundle. `service cp` accepts only paths under
  `/workspace/project`; upload there, then use `service exec` to install one
  owner-only generation under `/tmp`. Never use `--set`, environment dumps,
  command arguments, or project source for policy credentials.
- The supported steady-state path is the MK8s adapter Deployment: run the Antioch
  named-route client and bounded relay in one pod network namespace and target a
  ClusterIP policy Service. The operator VM may deploy/status it but must not
  carry frames or actions. The legacy tmux path is recovery-only and must not be
  retained after a cluster-native cutover.
- Reconcile the exact project-scoped scenario through supported `scenario list`,
  `session status`, and `service ps` JSON. The one live scenario's `session_id`
  must equal the current project session and the simulator process/session must
  be ready. Fail closed on absent, mismatched, or ambiguous ownership.
- A scenario dispatch or failure recovery may recreate the sim container. While the
  foreground run lives, verify every required bundle file through `service exec`
  and re-stage missing files with `service cp`; never bake them into the sim image.
  Stage a full generation and atomically switch one symlink so a recovering bridge
  cannot observe a certificate/key or token generation mix.
- Build an immutable revision with `project build`, start it with `session new
  --revision`, and do not force-replace unrelated active work. On a lost session,
  cancel and prove the exact scenario absent before creating one successor.
  Retry only structured retryable connectivity, 429, or 5xx failures.
- When direct policy egress is unavailable, use a declared Antioch service port
  bound only on adapter-pod localhost. Directly own `service ports --bind
  sim.policy-relay=127.0.0.1:18444 --serve sim` as a foreground child of the pod
  controller. Terminate authenticated WSS in the sim service and let the scenario
  connect outbound with a distinct authenticated role. The same-pod relay bridges
  the client role to the policy's independently authenticated, CA-verified
  ClusterIP WSS port 443. Bound messages, queues, connections, requests, timeouts,
  and reconnect backoff; never substitute an unauthenticated public proxy,
  disabled TLS verification, a token in a URL, or an undocumented endpoint.
- Treat livestream state `ready` as published but awaiting an authenticated Mission
  Control viewer. Do not claim an actively viewed frame until a supported viewer
  connection advances the first render, and never inspect browser auth storage.
- For the OpenPI PoC, require a finite `completed/passed` record with named checks
  for advancing nonblack exterior/wrist observations and finite `[15,8]` responses.
  Retire compute after verifying that durable record; never use timeout/renewal as
  the successful outcome.
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
- Require the runtime config's explicit supported Antioch deployment profile and
  forward it to the controller only as `ANTIOCH_ENV`; never let a cluster-native
  run inherit the vendor CLI's default deployment implicitly.

## Cleanup and evidence

Cancel the exact test run before releasing its exact project session. Prove the
current structured session identity still matches the controller-owned session
before invoking project-scoped release; fail closed on replacement. A requested
retained live demo is the exception: leave its exact session and policy
Deployment running, and provide exact supported stop commands privately. Record only
run ids, states, check names, schemas, checksums, artifact basenames, and sanitized
links. Never record tokens, signed URLs, config contents, organization/customer
identifiers, unrelated run metadata, or internal infrastructure coordinates.

See `docs/workbench/antioch.md` for authentication, deployment, schemas, licensing,
recovery, console access, and the current personal-OAuth limitation.
