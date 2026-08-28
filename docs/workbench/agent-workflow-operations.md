# Agent-operated Workbench workflows

An automation agent can operate any Workbench workflow through NPA without
knowing how NPA schedules or executes it. The caller owns its reasoning system
and provider configuration. Workflow operations do not select, configure, or
require a language model.

The declarative contract is an `apiVersion: npa.workflow/v0.0.1` YAML file.
Agents author and revise that contract; NPA validates it, expands it into a plan,
prepares the configured runtime, and executes it.

## Operation boundary

Use the existing NPA agent API to discover the live tool catalog and maintain a
draft:

| Purpose | Agent API |
| --- | --- |
| Discover available `toolRef` values | `GET /api/tools` |
| Read the current draft | `GET /api/workflows/draft` |
| Save and check a draft | `POST /api/workflows/draft` |
| Validate or plan without mutation | `POST /api/workflows/validate`, `POST /api/workflows/plan` |

The API accepts YAML text and returns typed validation and plan results. A caller
using subprocesses must invoke only the fixed `npa ...` operations below. It
must not invoke an execution backend, a cluster client, a terminal multiplexer,
or an arbitrary shell command.

## Read-only preparation

Run these checks in order. Keep the same YAML file and config overrides for the
plan and eventual submission.

```console
npa workbench workflow validate-spec <workflow.yaml> --json
npa workbench workflow plan-spec <workflow.yaml> --run-id <run-id> --waves --json
npa workbench workflow run-spec <workflow.yaml> --run-id <run-id> --plan-only --scheduler-plan --json
npa workbench health preflight --checks hf,ngc,s3 --json
npa provision-if-absent --project <project> --dry-run --output-format json
npa workbench workflow preflight-images <workflow.yaml> --project <project> --json
npa workbench workflow submit <workflow.yaml> --project <project> --run-id <run-id> --runtime --plan-only --output-format json
```

Choose the credential checks required by the workflow. Hosted inference
credentials are not part of the generic workflow contract and should be checked
only when a selected `toolRef` actually uses them.

The plan-only submission is the final read-only gate. It exercises NPA's real
translation path without creating a workflow job. Provisioning and submission
are separate mutations and require the caller's explicit authorization.

## Submit, resume, and observe

After authorization, prepare configured infrastructure and submit through NPA:

```console
npa provision-if-absent --project <project> --output-format json
npa workbench workflow submit <workflow.yaml> --project <project> --run-id <run-id> --runtime --output-format json
```

For the canonical compositional Sim2Real workflow, use
`npa/workflows/workbench/npa-workflows/sim2real.yaml` and always retain
`--runtime`. If the controller is interrupted, resume the exact run rather than
creating a replacement:

```console
npa workbench workflow submit <workflow.yaml> --project <project> --resume-run <run-id> --runtime --output-format json
```

Observe with non-following, machine-readable calls. Status contains typed
diagnostics and retry guidance. Logs are credential-redacted and hard-bounded;
when truncated, NPA returns the diagnostic tail plus character counts.

```console
npa workbench workflow status <run-id> --project <project> --no-watch --json
npa workbench workflow logs <run-id> --project <project> --stage <stage> --cached --max-output-chars 32768 --json
npa workbench workflow artifacts <run-id> --project <project> --json
```

Diagnose a failed run from the status result first, then request the named failed
stage's bounded log tail and artifact inventory. Preserve the run ID for resume;
do not guess backend job identities or inspect backend state.

## Stop and clean up safely

Cancel the workflow before considering infrastructure teardown. Cancellation is
repeat-safe for a never-launched or already-terminal run.

```console
npa workbench workflow cancel <run-id> --project <project> --json
```

Infrastructure is not automatically destroyed when a workflow stops because it
may serve other runs. After the operator separately authorizes teardown of the
exact configured target, use NPA again:

```console
npa cluster down --project <project> --context <cluster> --force --json
```

Do not infer permission to provision, submit, cancel, or destroy from a request
to validate, plan, inspect, or diagnose.
