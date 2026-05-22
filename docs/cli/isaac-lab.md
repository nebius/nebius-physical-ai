# `npa workbench isaac-lab`

## Command Tree

```text
Usage: npa workbench isaac-lab [OPTIONS] COMMAND [ARGS]...

Isaac Lab simulation workbench deployment, training, and evaluation.

Options
--project  -p  TEXT  Project alias from ~/.npa/config.yaml.
--name  -n  TEXT  Workbench instance name within the project.
--help  Show this message and exit.
Commands
list  List configured Isaac Lab workbenches.
cleanup-partial  Clean up orphaned Terraform resources from an interrupted Isaac Lab deploy.
deploy  Deploy or destroy an Isaac Lab workbench.
status  Check Isaac Lab VM status via SSH.
system-info  Collect and display system hardware information from the Isaac Lab VM.
train  Run Isaac Lab training on the VM via SSH.
eval  Run Isaac Lab evaluation on the VM via SSH.
export-lerobot  Generate Isaac Lab G1 rollouts and export them as a standard LeRobotDataset.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  TEXT  Project alias from ~/.npa/config.yaml. |
| `--name` | -n  TEXT  Workbench instance name within the project. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `list` | List configured Isaac Lab workbenches. |
| `cleanup-partial` | Clean up orphaned Terraform resources from an interrupted Isaac Lab deploy. |
| `deploy` | Deploy or destroy an Isaac Lab workbench. |
| `status` | Check Isaac Lab VM status via SSH. |
| `system-info` | Collect and display system hardware information from the Isaac Lab VM. |
| `train` | Run Isaac Lab training on the VM via SSH. |
| `eval` | Run Isaac Lab evaluation on the VM via SSH. |
| `export-lerobot` | Generate Isaac Lab G1 rollouts and export them as a standard LeRobotDataset. |

## Examples

```bash
npa workbench isaac-lab --help
npa workbench isaac-lab list --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `isaac-lab`.
