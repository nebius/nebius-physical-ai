# `npa workbench openarm`

## Command Tree

```text
Usage: npa workbench openarm [OPTIONS] COMMAND [ARGS]...

Enactic OpenArm simulation with real MuJoCo and Isaac Sim/Isaac Lab.

Options
--help  Show this message and exit.
Commands
run  Run a real OpenArm control rollout or upstream Isaac Lab training.
qualify  Validate every MuJoCo and Isaac artifact and write a qualification report.
status  Read one service-run state.
list  List service runs.
system-info  Show packaged upstream and simulator identities without fetching Isaac.
serve  Start the OpenArm service in the current container.
deploy  Ensure an authenticated single-RTX OpenArm service is deployed.
delete  Delete only the named OpenArm service resources; retain cache PVCs.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `run` | Run a real OpenArm control rollout or upstream Isaac Lab training. |
| `qualify` | Validate every MuJoCo and Isaac artifact and write a qualification report. |
| `status` | Read one service-run state. |
| `list` | List service runs. |
| `system-info` | Show packaged upstream and simulator identities without fetching Isaac. |
| `serve` | Start the OpenArm service in the current container. |
| `deploy` | Ensure an authenticated single-RTX OpenArm service is deployed. |
| `delete` | Delete only the named OpenArm service resources; retain cache PVCs. |

## Examples

```bash
npa workbench openarm --help
npa workbench openarm run --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `openarm`.
