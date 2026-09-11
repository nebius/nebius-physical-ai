# `npa workbench robocasa`

## Command Tree

```text
Usage: npa workbench robocasa [OPTIONS] COMMAND [ARGS]...

RoboCasa kitchen-task simulation workbench.

Options
--help  Show this message and exit.
Commands
deploy  Deploy the RoboCasa service to an NPA Workbench Kubernetes cluster.
run  Run a RoboCasa capability (task registration, asset check, EGL reset, or random rollout).
status  Fetch RoboCasa run status.
system-info  Show RoboCasa runtime information.
list  List service-managed runs or Kubernetes resources.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `deploy` | Deploy the RoboCasa service to an NPA Workbench Kubernetes cluster. |
| `run` | Run a RoboCasa capability (task registration, asset check, EGL reset, or random rollout). |
| `status` | Fetch RoboCasa run status. |
| `system-info` | Show RoboCasa runtime information. |
| `list` | List service-managed runs or Kubernetes resources. |

## Examples

```bash
npa workbench robocasa --help
npa workbench robocasa deploy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `robocasa`.
