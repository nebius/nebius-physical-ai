# CLI Namespace Conventions

The `npa` CLI organizes commands into two top-level shapes. New commands
should match one of these shapes; commands that do not fit cleanly are usually
misclassified and should be reconsidered.

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

## `npa <verb> <subcommand>`

For stateless actions that operate on data or infrastructure. No lifecycle;
invoked, performs action, exits.

Current verbs include:

- `convert` - Format conversion (`lerobot-to-rrd`, `lerobot-to-mp4`)
- `network` - Network and ingress configuration
- `rerun` - Stateless Rerun sharing (`host`, `share`, `list-shares`, `revoke`)
- `demo` - Demo orchestration (`stage`, `verify`)

A command belongs at top level when it:

- Has no Nebius service lifecycle to manage
- Operates on files, S3 objects, or existing infrastructure
- Composes outputs of workbench tools without being a service

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

Category names that overpromise: `viz` implied coverage of all visualization
paths, including FiftyOne, Rerun, and matplotlib. The command only did one
specific thing. Prefer verbs that describe the action over categories that
describe a conceptual area.

Tool-as-namespace at top level: top-level names should be verbs or
noun-namespaces with multiple subcommands. A specific tool name at top level
works when there are multiple stateless actions for that tool. A single-action
tool name is usually misclassified: it is either lifecycle-bearing and belongs
under `workbench`, or it is a one-entry namespace that should be reconsidered.

## SDK Shape

The CLI surface is also the SDK surface. Any command should be invokable from
Python as well:

```python
from npa import convert, demo, rerun

convert.lerobot_to_mp4(input_path=..., output_path=...)
rerun.host(path=...)
demo.stage(target_bucket=...)
```

For developers building orchestrators on top of `npa`, rather than operators
using the CLI directly, the SDK is the primary surface. The CLI is the thin
wrapper.
