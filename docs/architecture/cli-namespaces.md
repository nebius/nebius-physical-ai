# CLI Namespace Conventions

The `npa` CLI is the platform entry point. Product behavior lives behind
solution namespaces such as `npa workbench`, while shared platform utilities
remain at top level only when they are genuinely cross-solution.

## `npa workbench <tool> <action>`

For deployable Nebius services. Each tool has a lifecycle: provision,
configure, run, observe, and tear down. Tools consume Nebius compute and
storage resources.

Current lifecycle-bearing workbench tools include:

- `lerobot` - LeRobot policy training, evaluation, serving, and inference
- `cosmos` - NVIDIA Cosmos foundation model server
- `groot` - NVIDIA GR00T humanoid policy model server
- `fiftyone` - Voxel51 FiftyOne curation server
- `genesis` - Genesis simulation, training, and evaluation workbench
- `isaac-lab` - NVIDIA Isaac Lab simulation workbench

A command belongs under `workbench` when it:

- Provisions or terminates Nebius infrastructure
- Manages the lifecycle of a long-running service
- Has a corresponding Nebius VM, container, or managed endpoint

## `npa workbench workflow <command>`

For Workbench-owned multi-stage workflow orchestration. Workflows compose
Workbench tools and infrastructure, so they live under the Workbench solution
namespace rather than as first-class platform commands.

Current workflow commands include:

- `run` - Run a named workflow on existing infrastructure
- `status` - Check workflow run status
- `logs` - Show workflow stage logs
- `teardown` - Destroy distill workflow VMs
- `distill` - Turnkey expert distillation

The legacy `npa workflow ...` shim remains callable for compatibility, prints a
visible deprecation warning, and is hidden from top-level help.

## `npa <verb> <subcommand>`

For stateless actions that operate on data or infrastructure. No lifecycle;
invoked, performs action, exits. These verbs must be platform-level utilities,
not Workbench internals.

Current verbs include:

- `convert` - Format conversion (`lerobot-to-rrd`, `lerobot-to-mp4`)
- `rerun` - Stateless Rerun sharing (`host`, `share`, `list-shares`, `revoke`)
- `demo` - Demo orchestration (`stage`, `verify`)
A command belongs at top level when it:

- Has no Nebius service lifecycle to manage
- Operates on files, S3 objects, or existing infrastructure
- Provides cross-solution utility behavior

## Current Top-Level Surface

| Command | Type | Status |
| --- | --- | --- |
| `workbench` | Tool group | Lifecycle-bearing workbench tools: `lerobot`, `cosmos`, `groot`, `fiftyone`, `genesis`, `isaac-lab` |
| `convert` | Verb | `lerobot-to-rrd`, `lerobot-to-mp4` |
| `rerun` | Verb | `host`, `share`, `list-shares`, `revoke` |
| `demo` | Verb | `stage`, `verify` |
| `viz` | Deprecated namespace | `lerobot`; use `npa convert lerobot-to-mp4` |
| `adapter` | Transitional one-entry namespace | `convert`; consolidation tracked by `ADAPTER_NAMESPACE_CONSOLIDATION` |
| `network` | Transitional one-entry namespace | `ensure-ingress`; consolidation tracked by `NETWORK_NAMESPACE_CONSOLIDATION` |
| `configure`, `init` | Bare commands | Setup guidance commands; `init` is currently an alias for `configure` |

## Boundary Case: `npa rerun` At Top Level

`npa rerun host`, `share`, `list-shares`, and `revoke` are stateless actions for
ad-hoc recording sharing via `app.rerun.io` and presigned URLs. They are at top
level because there is no Nebius service to manage: the viewer runs in the
user's browser, and the recording sits in S3.

A future Tier 2 capability, such as a managed Rerun service on a Nebius VM,
would correctly live under `npa workbench rerun deploy/status/serve/...`.
That command would have lifecycle; the current top-level command does not.

## Naming Smells To Avoid

One-entry namespaces: a top-level verb with only one subcommand is either
premature or misclassified. Example: `npa viz lerobot` was a one-entry
namespace. LeRobot rendering is adapter-layer work that belongs under
`npa convert`, so it moved to `npa convert lerobot-to-mp4`, and `viz` was
deprecated.

Current acknowledged exceptions are `adapter` and `network`. They are
transitional one-entry namespaces, not patterns to copy for new command
families. Consolidation is tracked by `ADAPTER_NAMESPACE_CONSOLIDATION` and
`NETWORK_NAMESPACE_CONSOLIDATION`.

Category names that overpromise: `viz` implied coverage of all visualization
paths, including FiftyOne, Rerun, and matplotlib. The command only did one
specific thing. Prefer verbs that describe the action over categories that
describe a conceptual area.

Tool-as-namespace at top level: top-level names should be verbs or
noun-namespaces with multiple subcommands. A specific tool name at top level
works when there are multiple stateless actions for that tool. A single-action
tool name is usually misclassified: it is either lifecycle-bearing and belongs
under `workbench`, or it is a one-entry namespace that should be reconsidered.

## Cross-Project Credential Model

Commands that operate across project boundaries use `--source-project` and
`--target-project` flags. When unset, they preserve the current default project
behavior. Credentials are resolved per project at the point of the S3 operation.

This is necessary because SDK consumers building orchestrators commonly need to
read inputs from one project and write outputs to another, with different scoped
principals for each. When a scoped principal is missing access, the
`ScopedCredentialError` names the failed project, operation, and bucket, and
points at host-credential fallback or IAM grants as remediation.

## Python API (SDK)

The SDK mirrors the supported CLI namespaces with Python identifiers. Hyphenated
CLI names become underscores: `list-shares` is `list_shares`, `isaac-lab` is
`isaac_lab`, and `--target-project` is `target_project`.

```python
from npa import convert, demo, rerun, workbench
from npa.errors import ScopedCredentialError

# Render a LeRobot dataset to MP4.
convert.lerobot_to_mp4(input_path="dataset", output_path="out.mp4")

# Stage demo artifacts to a target bucket.
demo.stage(target_bucket="my-bucket", target_project="eu-north1")

# Share a Rerun recording through the hosted web viewer.
result = rerun.host("recording.rrd", target_bucket="my-bucket")
print(result.share_url)

# Deploy a Cosmos workbench.
workbench.cosmos.deploy(project_id="project-id", tenant_id="tenant-id")
```

Public top-level SDK namespaces are:

- `npa.convert` - `lerobot_to_mp4`, `lerobot_to_rrd`
- `npa.demo` - `stage`, `verify`
- `npa.rerun` - `host`, `share`, `list_shares`, `revoke`
- `npa.network` - `ensure_ingress`
- `npa.workflow` - `run`, `status`, `logs`, `teardown`, `distill`
- `npa.workbench` - `cosmos`, `fiftyone`, `genesis`, `groot`, `isaac_lab`,
  and `lerobot` submodules
- `npa.errors` - `NpaError`, `ScopedCredentialError`

The SDK is currently v0 and unstable. Signatures may change before v1, so
customers should pin the exact `npa` version for integrations. For maximum
stability today, the CLI remains the operator contract.

Lower-level adapter imports remain available for advanced use cases, but they
are implementation APIs rather than the preferred SDK surface:

```python
from npa.adapter.lerobot.render import render_lerobot_to_mp4_result
```
