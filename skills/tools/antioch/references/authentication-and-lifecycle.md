# Authentication and lifecycle

## Establish host authority before operating

Classify each available host as either read-only operator/developer access or an
explicitly authorized mutable build/runtime environment. A usable CLI session,
SSH access, Docker socket, disk space, or cached source does not grant authority
to build, pull, install, authenticate, relay, start, stop, or clean up there.
Before any mutation, require explicit operator authorization for that host and
purpose.

When a host is read-only, leave it untouched, including prior task state: do not
"clean up" files, processes, images, caches, services, credentials, or history.
Prefer no access. If inspection is necessary, use supported read-only commands,
suppress identities and sensitive output, and avoid commands that refresh caches
or metadata. Route builds through the trusted private registry and authorized
Kubernetes-native builders/workloads; route GPU execution through authorized
managed Kubernetes or Antioch resources. Never use a read-only host as a relay
or staging hop.

## Authentication decision tree

Use a private temporary file with restrictive permissions for structured CLI
responses. Parse only the fields required for a boolean assertion; do not print
identity, organization, project, machine, endpoint, or environment values.

1. Run `antioch auth whoami --json`. Failure here is missing, expired, or
   unusable authentication—not a machine or tunnel problem.
2. Run `antioch project list --json` as the harmless authenticated API probe.
   If `whoami` passes but this fails, diagnose account/API reachability or
   authorization before allocating a machine.
3. Select the operator-named organization with `antioch auth switch`. Prefer the
   supported terminal chooser. Use `--org` only when the operator already
   supplied that non-secret identifier; never recover it from auth storage.
4. Repeat both checks after switching. Do not call `auth login`, inspect config,
   or extract browser state unless the operator explicitly requested a new
   interactive login.

Classify later failures separately:

- Empty `machine list` or `services ps` reporting no allocation means no machine.
- `machine status` failing for an exact known assignment means machine state or
  selection is stale.
- API state healthy but the declared local tunnel refusing connections means
  services/port convergence or tunnel state, not authentication.
- A listening PID with failed service health remains unready.

## Project and machine flow

1. Initialize only the intended directory with `antioch init --json`, or inspect
   an existing checkout with `antioch project show <project> --json`. Run
   `scenario collect --json` and `suite collect --json` before spending GPU time.
2. Discover assignments with `machine list --project ... --json`. Use
   `machine checkout --json` to make an exact assignment current; `--none`
   clears a stale choice without releasing anything.
3. Use `machine status --json` to prove machine, process, stream, and service
   state. Do not treat an SSH session or machine assignment as application
   readiness.
4. `services up --json` allocates when required, builds changes, starts detached
   services, and waits for declared health. Confirm again with
   `services ps --json`; process presence is only one input.

Keep engine/SDK runtime selection in the Antioch project. Keep the public
adapter image version and hosted engine/runtime version as separate identities;
changing one does not prove or change the other.

## Submit, monitor, and reconcile

- Prefer `scenario run --queue --json` or `suite run NAME --queue --json` when a
  durable remote identity is required. Use interactive execution only when live
  viewport streaming or attached debugging is the evidence under test.
- After a client interruption or ambiguous response, do not resubmit. Query
  `scenario list` or `suite list` using the exact project, authored name,
  dispatch mode, caller, and narrow creation window. If available, use the exact
  invocation or suite-run identity. Continue only when one unambiguous run is
  found; otherwise stop and preserve evidence.
- Inspect an exact run with `scenario show RUN --json` or
  `suite show RUN --follow --json`. A terminal success requires passed checks and
  useful result fields, not only a completed phase.
- Treat 429/5xx and transient disconnects with bounded exponential backoff.
  Authentication, malformed state, identity collisions, and incompatible
  project/runtime inputs are terminal.

For the NPA queued adapter, reuse the same workflow/run state and call
`reconcile`; its conditional state claim closes the submit-to-record crash
window. Never create a fresh output prefix to hide an ambiguous submission.

## Exact cancellation and release

1. Cancel the exact standalone run with `scenario cancel RUN --json` or the exact
   suite with `suite cancel RUN --json`.
2. Poll the exact supported show/status API to a terminal state.
3. Run `services down --json` only in the exact project whose services this task
   started.
4. Release only the verified task-owned assignment with
   `machine release --machine MACHINE --project PROJECT --yes --json`. `--yes`
   acknowledges the exact release non-interactively; it does not broaden scope.

Do not delete completed records as cleanup unless deletion is itself requested.
Never use `--all`, broad time ranges, process killing, or host-wide container
cleanup as a substitute for exact lifecycle commands.
