# SkyPilot Kubernetes controller launch transaction

Status: accepted for the NPA SkyPilot 0.12.2 execution boundary.

## Adjacent convergence contracts

The controller boundary is one of three reconciled lifecycle boundaries. Agent
setup persists exact VM/endpoint/service/credential/health phases and adopts a
matching healthy marker after client transport loss. Workflow images resolve to
immutable digests and satisfy a versioned bootstrap contract before controller
launch. Workflow status gives submitted/running/terminal evidence precedence
over stale planning, while exact conflicts are typed instead of coerced to
`NOT_SUBMITTED`. Across all three, loss of a local response never proves absence
and never authorizes duplicate mutation.

## Context

`sky jobs launch` is not idempotent. SkyPilot 0.12.2 sends a launch request and
job names are labels, not uniqueness keys. On a fresh Kubernetes cluster, a
point-in-time GPU/context preflight can pass and the API can still refuse the
later request that creates the managed-jobs controller. Retrying by name can
duplicate work; cancelling by name can target the wrong job.

NPA therefore treats every managed-job launch as a transaction around SkyPilot.
The implementation is provider-independent above the `kubectl /readyz` adapter
and is shared by one-shot and runtime-wave submission.

## Invariants

1. The exact selected context and the `KUBECONFIG` environment passed to
   SkyPilot are also used for readiness. NPA records only redacted command
   evidence, never kubeconfig contents or credential values.
2. A Kubernetes API is ready only after three consecutive successful `/readyz`
   observations spanning at least 10 seconds. A transient sample resets the
   streak. Authentication, RBAC, context, identity, certificate, and config
   failures stop immediately.
3. `UP` and `STOPPED` are usable existing-controller states. `ABSENT` is not
   controller-health success: stable Kubernetes API readiness is required before
   creation. Other controller states remain unhealthy.
4. One owner-only advisory lock serializes callers for the logical identity
   derived from project, run, wave, attempt, and exact task contract. `flock`
   releases after a process crash; the durable ledger remains the recovery truth.
5. Before launch and after every failed or uncertain launch, NPA reads
   `sky jobs queue --all --output json` and matches the exact job name. Zero
   matches is authoritative absence only when that command succeeds with valid
   structured output; one match adopts its immutable ID; multiple matches or an
   unreadable queue are indeterminate.
6. A retry is permitted only when exact absence is proven and the central
   phase-aware taxonomy classifies the launch failure as Kubernetes transport,
   API 429, or appropriate API 5xx. Backoff is capped exponential jitter behind
   a 180-second recovery deadline. Unknown, workload, capacity, schema, auth,
   config, and identity failures never enter this retry path.
7. Indeterminate existence fails closed: NPA neither duplicates the launch nor
   cancels by name. Resume with the same ID is the recovery action after queue
   access returns.
8. Cancellation uses an authoritative immutable job ID only. The ledger records
   `requested`, `verified`, `failed`, or `not_applicable`; requesting cancellation
   does not rewrite scheduler truth to `CANCELLED`. Reconciliation/cancellation
   errors are additive and never replace the primary launch failure.

## Durable and API contract

Runtime schema `npa.workflow.runtime.v1` remains backward compatible and gains
additive fields: `logical_launch_id`, `launch_sequence`, `error_category`,
`readiness`, `reconciliation`, `recovery_decision`, `operator_remedy`,
`primary_error`, `reconciliation_error`, and `cancellation`. One-shot JSON adds
`launch_transaction` with schema `npa.skypilot.launch-transaction.v1`; readiness
records use `npa.skypilot.kubernetes-readiness.v1`. Its stable `state`,
`existence`, `category`, and `controller` fields distinguish submitted/adopted,
verified absent, transient API failure, terminal failure, and indeterminate
evidence without interpreting prose.

Legacy successful waves replay as before. A legacy/current incomplete wave is
reconciled before action. An exact existing job is adopted, an authoritatively
absent transient launch may be relaunched, indeterminate evidence blocks, and a
terminal workload failure remains terminal.

## SkyPilot integration decision

The pinned source confirms the safest public CLI boundary is structured
`jobs queue --all --output json`. NPA does not depend on an undocumented request
ID, internal SkyPilot database, or job-name idempotency. This keeps the adapter
compatible with the isolated pinned runtime while making duplicate prevention
and recovery decisions explicit in NPA.

## Live verification

An authorized operator can run
`npa/tests/e2e/test_controller_launch_transaction_live.py` with
`NPA_LIVE_CONTROLLER_LAUNCH_TRANSACTION=1` plus exact project, context,
`KUBECONFIG`, SkyPilot binary, and unique run ID selectors. It creates one tiny
managed job and intentionally performs no automatic/fuzzy cleanup so evidence
can be inspected before exact-ID cancellation.

```bash
export NPA_LIVE_CONTROLLER_LAUNCH_TRANSACTION=1
export NPA_E2E_PROJECT=<exact-project-alias>
export NPA_E2E_CLUSTER_CONTEXT=<exact-fresh-context>
export NPA_E2E_CONTROLLER_TRANSACTION_RUN_ID=<new-unique-run-id>
export KUBECONFIG=<one-exact-kubeconfig-path>
export NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"
npa/.venv/bin/python -m pytest \
  npa/tests/e2e/test_controller_launch_transaction_live.py -q -s
```

The test does not create credentials, embed selectors, cancel by name, or tear
down the shared controller. After inspecting the JSON evidence, cancel the exact
recorded run through `npa workbench workflow cancel <run-id> --project
<exact-project-alias> --json`; remove the shared controller only under the
separate owner-verified cleanup runbook after all jobs are terminal.
