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

For workflows that write S3 state or artifacts, supply `config.bucket` and a
run-scoped `config.prefix`, even when their declared output URIs are absolute.
`NPA_CONFIG_DIR` selects NPA configuration; set `KUBECONFIG` separately to the
verified file containing the selected cluster context. An isolated configuration
must also resolve the authorized S3 endpoint and credentials through supported
private sources or the child process environment. Keep credential values out of
arguments and logs. A successful structural render or `--plan-only` does not
prove these live submission prerequisites.
Preserve a failed submission intent and inspect the exact run state before
recovering from a preflight failure; do not delete the intent or assume a failed
CLI exit means no launch occurred.

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
- The standard Kubernetes template derives the provider namespace from the
  selected kubeconfig context. In 0.12.2, native pod creation overwrites
  `pod_config.metadata.namespace` with that provider namespace. Before using
  existing PVCs or Secrets, configure a workload-specific context with their
  namespace and verify the resulting pod namespace after submission.

## What the Renderer Emits

`npa/src/npa/orchestration/npa_workflow/skypilot_render.py` turns a planned spec
into SkyPilot documents:

- A `parallel:` wave becomes a SkyPilot JobGroup; `maxConcurrency` splits a
  larger group into batches.
- `num_nodes` is emitted at the SkyPilot task level. SkyPilot gang-schedules the
  pods and exports `SKYPILOT_NODE_RANK` and `SKYPILOT_NODE_IPS` to `run`
  commands. Do not assume rank is available during `setup`.
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

### Native Ray development

Do not infer SkyPilot compatibility from image pullability or local Docker
success. The non-root LanceDB image lacks the `sudo` needed by SkyPilot's SSH
bootstrap. The guarded CLIP development example therefore reuses an immutable
public PyTorch image and prepares a separately pinned Ray application environment.
The target policy must permit that image's root user inside an unprivileged pod;
this does not grant host-root or privileged-container access.
Record the image digest and prepared dependency freeze separately: the digest
does not attest later pip installations. Its Python 3.12 has pip but no ensurepip;
use the tested `venv --system-site-packages --without-pip` preparation. This
qualifies the canonical Workbench CLIP UDF and LanceDB library, not the published
LanceDB service image or HTTP backfill API.

The tool-specific development cluster is
`npa/workflows/workbench/ray-clip-development/cluster.yaml`; the complete customer
journey is [Run and edit a GPU Ray application](../../../docs/testing/fast-source-iteration.md).
This is a customer development task, not a second workflow catalog. SkyPilot
owns the named cluster and its application service task. Ray Jobs owns application
submission, logs, status and cancellation, using its public CLI/SDK. Ray
`runtime_env.working_dir` transfers ordinary application Python and the exact
Workbench UDF; no NPA source overlay or custom submit/finalize protocol is used.
Persist and verify application outputs before cancelling the exact service task
and removing the named development cluster with SkyPilot. Shared SkyPilot API
services and the underlying configured Kubernetes cluster remain operator-owned.
The guide includes a separately owned upstream API Compose contract for hosts
without a suitable API: it mounts explicit backend credentials read-only, keeps
its own state volume and fixed namespace, and verifies a completed dry run.
Do not copy another API's state, patch its backend or stop its processes to make
a development submit work. Remove an owned API only after all its development
clusters are gone; preserve a platform supplied by another operator.

Keep application Ray in its own environment with explicit addresses and separate
ports. Never connect through ambient management-Ray discovery or use `ray stop`.
Jobs/Dashboard binds to loopback behind an authenticated SSH tunnel. The startup
script runs `ray start --block`; SkyPilot owns that task's process lifetime.
SkyPilot 0.12.2 uses management Ray 2.9.3 with a head worker range of
11002–65535. Keep application service and worker ports below 11002, avoid fixed
management ports, and verify the OS ephemeral range starts above the application's
highest port. The example worker range is 10010–10999. Keep Jobs drivers on the
head when application checkpoints and aggregation outputs use its local disk.
Verify Jobs readiness and the expected GPU/node resources before submission.

New Ray source submissions create new actors and reload model weights; this is
source redeployment, not hot reload. Python edits within the prepared dependency
boundary require no image rebuild. Native/ABI changes require a compatible image.
Report cold environment preparation and model loading separately from source
iteration. For distributed checks match the GPU actor count to available GPUs
and report actual node/GPU placement; several GPUs on one node are not multi-node
validation. Review imported source hashes, persisted vectors and retrieval
results after each source edit.

### Existing NPA runtime

- A vendor image may put a stale npa tree on `PYTHONPATH`; the renderer stages
  the selected source first and runs the recorded interpreter.
- Operator wrappers must set `NPA_LIVE_WT` to an existing, durable worktree and
  fail before submission when it is missing. Never fall back to an ambient
  checkout: a vanished `/tmp` worktree once made the harness test another
  contributor's tree and report a misleading skip.
- The `NPA_SRC_S3_URI` overlay has no embedded provenance. Re-stage it after a
  source change before diagnosing a renamed flag in a live pod.
- An unresolved placeholder in rendered setup is rejected before submission.
- With a remote SkyPilot API server, inspect its effective kubeconfig and selected
  context namespace. A correct client-side context or rendered pod metadata does
  not establish the server's namespace. Keep API health, namespace selection and
  actual pod placement as separate checks; see the upstream
  [API server configuration guide](https://docs.skypilot.ai/en/stable/reference/api-server/api-server-admin-deploy.html).

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
