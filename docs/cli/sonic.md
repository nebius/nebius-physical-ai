# `npa workbench sonic`

## Command Tree

```text
Usage: npa workbench sonic [OPTIONS] COMMAND [ARGS]...

NVIDIA GEAR-SONIC whole-body-control workbench.

Options
--project  -p  TEXT  Project alias from ~/.npa/config.yaml.
--name  -n  TEXT  Workbench instance name within the project.
--help  Show this message and exit.
Commands
deploy  Prepare or plan a SONIC runtime.
train  Run SONIC Isaac Lab training or smoke validation.
serve  Launch or describe a SONIC serving path.
status  Inspect SONIC runtime state.
list  List configured SONIC workbenches and default model artifacts.
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
| `deploy` | Prepare or plan a SONIC runtime. |
| `train` | Run SONIC Isaac Lab training or smoke validation. |
| `serve` | Launch or describe a SONIC serving path. |
| `status` | Inspect SONIC runtime state. |
| `list` | List configured SONIC workbenches and default model artifacts. |

## Examples

```bash
npa workbench sonic --help
npa workbench sonic deploy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `sonic`.
