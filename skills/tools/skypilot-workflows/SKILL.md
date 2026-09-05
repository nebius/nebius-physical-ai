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

### Customer application image runtime

For raw `run.argv` / `run.shell` stages whose application owns its interpreter
and source delivery, set `config.runtime_setup: image`. Every executable stage
must use a registry-qualified immutable image digest. The mode rejects `toolRef`
and conflicting `require_baked_npa`; it skips the NPA installer, source-URI
injection and interpreter/PYTHONPATH shims. It retains the standard workflow,
SkyPilot lifecycle, credential forwarding and model-cache handling. The image
still needs SkyPilot's worker bootstrap prerequisites. Image mode makes no claim
about a baked NPA source revision.

Do not infer SkyPilot compatibility from image pullability or local Docker
success. A live attempt with the published non-root LanceDB image failed its SSH
bootstrap because `sudo` was absent. The Ray development recipe therefore uses
an existing official root PyTorch runtime image, prepares pinned application
dependencies once per session, and sends the actual Workbench UDF as the fourth
reviewed file through standard Ray `working_dir` via required `--udf-source`.
No image build or NPA source overlay is involved. Root inside its isolated pod
does not imply privileged/host-root access, but the target policy must permit
that runtime user. Record the base digest and prepared dependency freeze
separately; the base digest does not attest later pip installations.
The selected PyTorch image's interpreter is `/usr/bin/python3.12`; pip is present
but ensurepip is absent. Use its tested `venv --system-site-packages --without-pip`
plus base `pip --python <venv-python>` preparation. The recipe qualifies the
canonical Workbench UDF and LanceDB library, not the published service image or
HTTP backfill API. Keep startup failures separate from source-iteration timing.

The scoped example is
`npa/workflows/workbench/npa-workflows/ray-clip-development-session.yaml`, with
the standard Ray Jobs client under
`npa/workflows/workbench/ray-clip-development/`. Keep application Ray in its own
environment with explicit addresses and separate ports; never use ambient
management-Ray discovery or `ray stop`. Keep Jobs/Dashboard on loopback behind
authenticated forwarding. Customer source travels through Ray `runtime_env`;
do not stage it through the NPA overlay. Verify worker source hashes and real
outputs after an edit. This recipe's new source jobs create new actors and reload models;
native/ABI dependency changes need a compatible image. Finish exact jobs, persist
artifacts, then complete the session's scoped finish protocol before teardown.
Match `config.gpus_per_node` to the accelerator request. Use `config.nodes`
for multiple nodes or several GPUs on one node; report the actual node/GPU scope
and do not claim multi-node validation from a one-node run.
Put the run ID in the application prefix and end the durable workflow URI in
`/<run-id>/workflow`; explicit status/cancel locators must resolve the same run.
Prepared rank receipts precede Ray readiness. Verify the application Jobs API
and expected live GPU/node resources before submitting the source sequence.
SkyPilot's Ray 2.9.3 management head worker range is 11002–65535, not 19999.
Keep all application service/worker ports below 11002 while avoiding fixed
management ports, and verify the OS ephemeral range starts above the application's
highest port for non-head management workers. The canonical worker range is
10010–10999. Keep Jobs drivers on the head for this recipe's local checkpoint,
baseline comparison and finish-upload contract.
An isolated config directory does not change SkyPilot's host-wide default API
endpoint. In SkyPilot 0.12.2, controller identity belongs to the API server;
per-request user IDs do not select independent controllers on a shared server.
Use an explicitly owned API endpoint and exact Kubernetes context for an isolated
session. Never stop another run's API daemon to make a development submit work.
The pinned API queue also binds fixed port 50011, so changing only the HTTP port
does not isolate two host processes. Use the upstream Docker API deployment with
bridge networking and loopback-only HTTP publication; mount only run-owned state
and the exact provider/kubeconfig inputs. Verify this path before GPU submission.
For the non-root API recipe, preserve the installed rsync helper's bytes while
making only that helper owned by the API UID, then verify non-root chmod works;
SkyPilot 0.12.2 performs that chmod before Kubernetes source syncing. Follow the
exact-file copy/hash procedure in the reproduction; do not change package code,
grant broad package ownership, or add capabilities to bypass it.
Nebius storage additionally requires `[nebius]` in that API home's
`.aws/credentials` and `[profile nebius]` in `.aws/config`. Materialize the
existing selected S3 credentials/region/endpoint in owner-only files there;
AWS environment variables alone do not enable storage. Require `sky check nebius`
inside the container to report both compute and storage enabled. A listable
bucket from NPA health does not replace the exact workflow write preflight.

This interactive session is render-only in the shared submit matrix because its
separate Jobs client and finish marker must be coordinated. Run the dedicated
session recipe for real GPU qualification; a planned session is not workload
evidence.

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
