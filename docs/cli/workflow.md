# `npa workbench workflow`

## Command Tree

```text
Usage: npa workbench workflow [OPTIONS] COMMAND [ARGS]...

Multi-stage training workflow orchestration.

Options
--help  Show this message and exit.
Commands
submit  Submit a SkyPilot workflow YAML through the NPA controller convention.
run  Run a named workflow end-to-end.
status  Check the status of a workflow run.
logs  Show logs for a specific stage of a workflow run.
teardown  Destroy both VMs from a distill workflow run.
distill  Run expert distillation: L40S (Genesis) + H100 (LeRobot).
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `submit` | Submit a SkyPilot workflow YAML through the NPA controller convention. |
| `run` | Run a named workflow end-to-end. |
| `status` | Check the status of a workflow run. |
| `logs` | Show logs for a specific stage of a workflow run. |
| `teardown` | Destroy both VMs from a distill workflow run. |
| `distill` | Run expert distillation: L40S (Genesis) + H100 (LeRobot). |

## Examples

```bash
npa workbench workflow --help
npa workbench workflow submit --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `workflow`.
