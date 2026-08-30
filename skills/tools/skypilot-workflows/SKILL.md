---
name: skypilot-workflows
description: "Use when running or debugging how the engine renders and submits SkyPilot from an npa.workflow spec: invocation, SkyPilot limits, JobGroups, multi-node tasks, runner scripts, and cleanup. For authoring, use author-npa-workflow."
---

# SkyPilot Workflows

For teardown after project/config removal, use the existing immutable teardown
receipt (`npa skypilot cleanup-controller --receipt <id> --context <context>
--yes`). A terminal receipt or exact provider-verified missing cluster is a no-op
without bootstrapping SkyPilot. Otherwise cleanup stays bound to the exact
recorded project/context and preserves local controller state until remote
absence is independently verified. Never fall back to the current kube context
or an unrelated SkyPilot profile.

SkyPilot is the workflow **execution engine** in this repo. Argo is deprecated;
do not add or revive Argo workflows.

SkyPilot is not the repository authoring surface. Pipelines are
`npa.workflow/v0.0.1` specs, and the engine plans those specs and renders
SkyPilot tasks. Use `skills/workflows/author-npa-workflow/SKILL.md` to author a
pipeline. The retired raw-task catalog must not return; raw SkyPilot input is
still accepted for customer-provided tasks and guarded tool-specific examples.

## Invocation

SkyPilot lives in an isolated virtualenv outside NPA's main Python environment. Invoke it through `NPA_SKYPILOT_BIN`; never rely on `sky` from `PATH`.

Use `npa skypilot bootstrap` to create or reuse the pinned SkyPilot `0.12.2`
venv, then set `NPA_SKYPILOT_BIN="$(npa skypilot status --bin-path)"`.

The Kubernetes controller is the default path (`W9-skypilot-k8s-controller`). The VM controller exists only as a fallback.

## Known SkyPilot 0.12.2 Limits

- Raw SkyPilot `envs` does not support self-referencing variable interpolation.
  Repository specs use resolved `config` tokens instead.
- `sky jobs launch` has no dry-run flag. For a spec, use `workflow submit
  --plan-only` to inspect the rendered plan without launching.
- Mixed serial/parallel task groups are expressed in the spec. The runtime
  driver emits a SkyPilot JobGroup for each `parallel:` wave and submits the
  following state as its barrier.
- Managed-job Python API `Dag` support is effectively single-task for this repo's burst path. Use `npa burst submit-yaml` only for rendered single-task SkyPilot YAMLs; use `npa workbench workflow submit` for multi-stage workbench YAMLs.
- Direct Nebius burst jobs pull `resources.image_id` before YAML `setup` runs.
  Official public GHCR development and release tags need no registry secret.
  Operator-controlled private registries require explicit exact-host SkyPilot
  Docker credentials; NPA forwards them but never mints a provider token.

## What the Renderer Emits

`npa/src/npa/orchestration/npa_workflow/skypilot_render.py` turns a planned spec
into SkyPilot documents:

- A `parallel:` wave becomes a SkyPilot JobGroup; `maxConcurrency` splits a
  larger group into batches.
- `num_nodes` is emitted at the SkyPilot task level. SkyPilot gang-schedules the
  pods and exports `SKYPILOT_NODE_RANK` and `SKYPILOT_NODE_IPS`.
- Package extras, third-party requirements, source staging, and vendor
  interpreters are selected from the toolRef.
- A self-hosted service that must survive into the task command belongs in the
  run preamble, not in `setup`.
- Isaac-routed stages carry `ACCEPT_EULA=Y` by default. An explicit
  `--no-accept-eula` renders and forwards an empty value, including through
  Kubernetes Sim2Real paths, and fails before download. Generic BYOF routes are
  gated only when `config.base_profile` or `config.base_image` selects Isaac;
  GR00T routes are gated only for Isaac simulation.

## Live Debugging Traps

- A vendor image may put a stale npa tree on `PYTHONPATH`; the renderer stages
  the selected source first and runs the recorded interpreter.
- Operator wrappers must set `NPA_LIVE_WT` to an existing, durable worktree and
  fail before submission when it is missing. Never fall back to an ambient
  checkout: a vanished `/tmp` worktree once made the harness test another
  contributor's tree and report a misleading skip.
- The `NPA_SRC_S3_URI` overlay has no embedded provenance. Re-stage it after a
  source change before diagnosing a renamed flag in a live pod.
- An unresolved placeholder in rendered setup is rejected before submission.

## Reference Pattern

- Canonical spec: `npa/workflows/workbench/npa-workflows/bdd100k-pipeline.yaml`.
- Runner script pattern: `npa/scripts/run_bdd100k_pipeline.py`, a thin wrapper around `npa.orchestration.skypilot.submit_workflow`.
- Isaac Lab runners follow the same shape through `npa/scripts/run_isaac_lab_rl.py`.

## Commit And Cleanup

Acquire `/tmp/npa-commit-lock/workflows-skypilot` before committing workflow files in parallel-run contexts.

Cleanup is best-effort and must not raise. `also_teardown_controller=False` is
the safe default; only opt into controller teardown when no other run can be
using it. Explicit controller teardown requires a single NPA project and its
exact saved Kubernetes context, cross-checks stable provider identity, performs
remote deletion against a clone of SkyPilot state, independently proves remote
absence, checkpoints that evidence, and only then converges the matching local
metadata. Authentication/RBAC/connectivity/identity uncertainty preserves local
state; never fall back to an ambient context or unrelated SkyPilot profile.

Runtime-orchestrated workflows persist one immutable managed-job identity per
wave and attempt. Status and cancellation use that identity for only the wave's
encoded stage members; retries remain historical records and the final attempt
is selected deterministically. A discovered/root job ID must never be broadcast
across runtime stages. Conflicting or missing history is reported as ambiguous
or unknown. Root-ID fan-out is compatible only with the legacy single-managed-
job manifest contract.

SkyPilot 0.12.2 job names are not idempotency keys. NPA wraps the non-idempotent
launch POST in an owner-only logical-identity lock. Production submit uses the
asynchronous API surface, then performs structured exact-name/ID queue
reconciliation before returning control to the runtime supervisor. The durable wave records readiness samples, launch sequence,
failure category, reconciliation/adoption, recovery decision, and cancellation
verification. `UP` and `STOPPED` controllers are usable; controller absence is a
distinct state that requires stable Kubernetes API readiness before creation.
Unknown/ambiguous queue evidence blocks both relaunch and fuzzy cancellation.

The standard runtime's lightweight supervisor stays outside payload pods and
persists content-addressed attempt decisions in S3. It classifies actionable
configuration, transient infrastructure, payload, and unknown evidence. Only a
typed transient with matching immutable workflow/source/image identity, verified
declared-output state, passing launch preflights, and exact cancellation may
advance to a new attempt. The expected identities are recomputed independently
from current runtime inputs. `--max-infrastructure-recoveries` is finite and
separate from payload `--retries`; exhaustion is durable and terminal.
Cancellation is polled by exact provider ID until terminal under a finite
verification policy; a merely requested cancellation blocks relaunch.
Completed-wave reuse validates declared S3 outputs;
mid-stage resume additionally requires a real compatible tool checkpoint loader.

The shared supervisor is also active in Genesis' existing production Serverless
Jobs command. This does not route individual `npa.workflow/v0.0.1` stages to
Serverless; runtime workflow waves remain SkyPilot/Kubernetes.
