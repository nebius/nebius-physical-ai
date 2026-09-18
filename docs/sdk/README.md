# Python SDK Surface

[Docs](../README.md)

`npa` ships a Python SDK alongside the CLI. This page is the map of what is
public. For a task-oriented tour that runs a pipeline through all three tiers,
read the [CLI/SDK/YAML walkthrough](../workbench/cli-sdk-yaml-walkthrough.md).

## Stability

The SDK is v0. Names listed here are supported; anything reached through a
module not listed here is internal and can move in any release. Pin the `npa`
version for integrations until the public API reaches v1.

Every namespace declares its own `__all__`, and `dir()` reports exactly that
surface, so the package is self-describing:

```python
import npa

dir(npa)
dir(npa.sdk.workbench)
```

## Namespaces

`npa.sdk` is the namespaced entrypoint. The remaining top-level names are
shortcuts for surfaces that have no per-tool client.

| Namespace | Holds |
| --- | --- |
| `npa.sdk.workbench.<tool>` | Per-tool clients — `train`, `run`, `deploy`, `status`, and tool-specific verbs |
| `npa.sdk.fleet` | Multi-cluster fleet deployment from an `npa.fleet/v0.0.1` spec |
| `npa.sdk.provisioning` | Project, cluster, and storage provisioning |
| `npa.sdk.soperator` | Slurm-on-Kubernetes deployment |
| `npa.workbench.<tool>` | The implementation layer the clients call; importable, and the only surface for tools without an SDK client |
| `npa.workflow` | `submit`, `run`, `status`, `logs`, `teardown`, `distill` |
| `npa.convert` | `lerobot_to_mp4`, `lerobot_to_rrd` |
| `npa.rerun` | `configure_browser_cors`, `host`, `share`, `list_shares`, `revoke` |
| `npa.demo` | `stage`, `verify` |
| `npa.network` | `ensure_ingress` |
| `npa.errors` | `NpaError`, `ScopedCredentialError` — see [SDK errors](errors.md) |

`npa.__version__` is the installed package version.

## Importing

Both forms work and reach the same module:

```python
from npa.sdk.workbench import detection_training

import npa
npa.sdk.workbench.detection_training
```

Submodules are imported on first access, so importing a namespace costs
nothing and pulls in no tool dependencies. This is what lets the minimal
workbench images — which ship a partial dependency set — import `npa` and
`npa.sdk` at all. Import the client you need and the cost arrives with it.

## Credentials And Configuration

SDK functions read the same configuration as the CLI: machine-managed settings
from `~/.npa/config.yaml` and credentials from environment variables, falling
back to `~/.npa/credentials.yaml`. Environment variables win. See
[Configuration](../configuration.md); there is nothing SDK-specific to set up
beyond what the CLI already needs.

## Adding To The Surface

A new client goes in `npa/src/npa/sdk/workbench/<tool>.py` and must be added to
that package's `__all__` — `npa/tests/test_sdk_surface.py` fails on a public
module that is not exported, because an unexported module is unreachable by
attribute access and invisible to `dir()`. Name genuinely internal helpers with
a leading underscore. The full checklist for a new tool is in
[Adding a workbench tool](../architecture/contributor-context.md).
