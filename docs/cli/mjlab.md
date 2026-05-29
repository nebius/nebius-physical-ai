# `npa workbench mjlab`

## Command Tree

```text
Usage: npa workbench mjlab [OPTIONS] COMMAND [ARGS]...

MJLab locomotion policy evaluation for SONIC workflows.

Options
--help  Show this message and exit.
Commands
eval  Evaluate a SONIC locomotion checkpoint against MJLab metrics.
workflow  Show the SkyPilot YAML template for MJLab evaluation.
status  Show MJLab tool status.
list  List supported MJLab evaluation suites.
```

## Options

| Option | Description |
| --- | --- |
| `--help` | Show this message and exit. |

## Subcommands

| Command | Description |
| --- | --- |
| `eval` | Evaluate a SONIC locomotion checkpoint against MJLab metrics. |
| `workflow` | Show the SkyPilot YAML template for MJLab evaluation. |
| `status` | Show MJLab tool status. |
| `list` | List supported MJLab evaluation suites. |

## Examples

```bash
npa workbench mjlab --help
npa workbench mjlab eval --help
```

Regenerate this page with `bash scripts/build_docs.sh` after changing `mjlab`.
