# `npa workbench sonic`

## Command Tree

```text
Usage: npa workbench sonic [OPTIONS] COMMAND [ARGS]...

NVIDIA GEAR-SONIC whole-body-control workbench.

Options
--project  -p  <str>  Project alias from ~/.npa/config.yaml.
--name  -n  <str>  Workbench instance name within the project.
--help  Show this message and exit.
Commands
deploy  Prepare or plan a SONIC runtime.
train  Run SONIC Isaac Lab training or smoke validation.
export  Export a SONIC locomotion policy to deterministic-action ONNX.
eval  Evaluate an exported SONIC ONNX locomotion policy.
serve  Launch or describe a SONIC serving path.
status  Inspect SONIC runtime state.
list  List configured SONIC workbenches and default model artifacts.
retargeting  Motion retargeting for SONIC locomotion workflows.
```

## Options

| Option | Description |
| --- | --- |
| `--project` | -p  <str>  Project alias from ~/.npa/config.yaml. |
| `--name` | -n  <str>  Workbench instance name within the project. |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `deploy` | Prepare or plan a SONIC runtime. |
| `train` | Run SONIC Isaac Lab training or smoke validation. |
| `export` | Export a SONIC locomotion policy to deterministic-action ONNX. |
| `eval` | Evaluate an exported SONIC ONNX locomotion policy. |
| `serve` | Launch or describe a SONIC serving path. |
| `status` | Inspect SONIC runtime state. |
| `list` | List configured SONIC workbenches and default model artifacts. |
| `retargeting` | Motion retargeting for SONIC locomotion workflows. |

## Examples

```bash
npa workbench sonic --help
npa workbench sonic deploy --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `sonic`.
